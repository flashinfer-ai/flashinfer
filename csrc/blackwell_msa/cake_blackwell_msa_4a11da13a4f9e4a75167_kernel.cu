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
#define SMEM_SEG_SUM_STAGE_BYTES 1024
#define SMEM_SEG_SUM_STRIDE 1024
#define SMEM_STATE_OFF 66560
#define SMEM_STATE_STAGE_BYTES 512
#define SMEM_STATE_STRIDE 512
#define SMEM_CAND_OFF 67072
#define SMEM_CAND_STAGE_BYTES 8192
#define SMEM_CAND_STRIDE 8192
#define SMEM_CAND_CNT_OFF 75264
#define SMEM_CAND_CNT_STAGE_BYTES 64
#define SMEM_CAND_CNT_STRIDE 64
#define SMEM_FLAGS_OFF 75456
#define SMEM_FLAGS_STAGE_BYTES 64
#define SMEM_FLAGS_STRIDE 64
#define SMEM_TOTAL 75520
#define THREADS 256


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

__global__ __launch_bounds__(THREADS) void
kernel_cake_blackwell_msa_4a11da13a4f9e4a75167(float* __restrict__ max_score, int* __restrict__ output, unsigned long long* __restrict__ partials, unsigned int* __restrict__ counters, unsigned int* __restrict__ block_max, int num_heads, int max_k_tiles, int total_q, int num_valid_pages, int force_begin_blocks, int force_end_blocks, int num_qtiles, int num_slices, int slice_len, int index_bits)
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
    int q = qtile * 16 + q_local;
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
    int _min_1 = ((q) < (total_q - 1) ? (q) : (total_q - 1));
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
    if (tid_0 < 16) {
        cand_cnt[tid_0] = 0;
        unsigned int mode_t = mode0;
        if (qtile * 16 + tid_0 >= total_q) {
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
    int blks[16];
    unsigned int subs[4];
    unsigned int pre[2];
    unsigned int res[3];
    int nres[1];
    for (int p = 0; p < 11; p++) {
        if (any_open != 0) {
            int n_hist = 0;
            int n_stage = 0;
            unsigned int tmin = 1024;
            #pragma unroll
            for (int qq = 0; qq < 16; qq++) {
                unsigned int mq = state[qq * 8];
                n_hist = n_hist + (int)(mq == 0);
                n_stage = n_stage + (int)(mq == 1);
                unsigned int tq = state[qq * 8 + 5];
                unsigned int tcand = ((mq == 1) ? tq : 1024);
                unsigned int _min_2 = ((tmin) < (tcand) ? (tmin) : (tcand));
                tmin = _min_2;
            }
            unsigned int mode_me = state[q_local * 8];
            unsigned int rshift_me = state[q_local * 8 + 1];
            unsigned int rp_lo = state[q_local * 8 + 2];
            unsigned int rp_hi = state[q_local * 8 + 3];
            unsigned int ngt_me = state[q_local * 8 + 4];
            unsigned int tbin_me = state[q_local * 8 + 5];
            unsigned long long rprefix_me = (unsigned long long)rp_hi << 32 | (unsigned long long)rp_lo;
            int is_hist = (int)(mode_me == 0);
            int is_stage = (int)(mode_me == 1);
            int is_pass0 = (int)(rshift_me >= 64);
            unsigned int _min_3 = ((10) < (rshift_me) ? (10) : (rshift_me));
            unsigned int w_me = _min_3;
            if (rshift_me > 22) {
                unsigned int _min_4 = ((10) < (rshift_me - 22) ? (10) : (rshift_me - 22));
                w_me = _min_4;
            }
            unsigned long long dsh_me = (unsigned long long)(rshift_me - w_me);
            unsigned int dmask_me = (one_u << w_me) - one_u;
            unsigned int _min_5 = ((rshift_me) < (63) ? (rshift_me) : (63));
            unsigned long long rsh_m = (unsigned long long)_min_5;
            unsigned int key_lo = 0;
            if (tbin_me == 511) {
                key_lo = 1207435264;
            } else if (tbin_me == 512) {
                key_lo = 2147483648u;
            } else {
                if (tbin_me == 1023) {
                    key_lo = 3354918912u;
                }
            }
            unsigned long long rthr = rprefix_me << rsh_m;
            unsigned long long klo = (unsigned long long)key_lo;
            unsigned int ilo = ~idx_mask;
            if (rshift_me < 54) {
                klo = rthr >> 22 & 4294967295;
                ilo = (unsigned int)(rthr & 4194303) | ~idx_mask;
            }
            unsigned long long cthr_me = klo << 32 | (unsigned long long)ilo;
            if (n_stage == 0 && p == 0) {
                #pragma unroll 1
                for (int i0 = 0; i0 < iters; i0 += 16) {
                    int i1 = i0 + 8;
                    unsigned long long off = off0 + (unsigned long long)i0 * stride16;
                    int blk = lo_s + i0 * 16 + r_sub;
                    blks[0] = blk;
                    int act = (int)(blk < hi_s) & qvalid;
                    xs[0] = __uint_as_float(2143289344);
                    if (act != 0) {
                        xs[0] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_0 = lo_s + (i0 + 1) * 16 + r_sub;
                    blks[1] = blk_0;
                    int act_1 = (int)(blk_0 < hi_s) & qvalid;
                    xs[1] = __uint_as_float(2143289344);
                    if (act_1 != 0) {
                        xs[1] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_2 = lo_s + (i0 + 2) * 16 + r_sub;
                    blks[2] = blk_2;
                    int act_3 = (int)(blk_2 < hi_s) & qvalid;
                    xs[2] = __uint_as_float(2143289344);
                    if (act_3 != 0) {
                        xs[2] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_4 = lo_s + (i0 + 3) * 16 + r_sub;
                    blks[3] = blk_4;
                    int act_5 = (int)(blk_4 < hi_s) & qvalid;
                    xs[3] = __uint_as_float(2143289344);
                    if (act_5 != 0) {
                        xs[3] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_6 = lo_s + (i0 + 4) * 16 + r_sub;
                    blks[4] = blk_6;
                    int act_7 = (int)(blk_6 < hi_s) & qvalid;
                    xs[4] = __uint_as_float(2143289344);
                    if (act_7 != 0) {
                        xs[4] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_8 = lo_s + (i0 + 5) * 16 + r_sub;
                    blks[5] = blk_8;
                    int act_9 = (int)(blk_8 < hi_s) & qvalid;
                    xs[5] = __uint_as_float(2143289344);
                    if (act_9 != 0) {
                        xs[5] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_10 = lo_s + (i0 + 6) * 16 + r_sub;
                    blks[6] = blk_10;
                    int act_11 = (int)(blk_10 < hi_s) & qvalid;
                    xs[6] = __uint_as_float(2143289344);
                    if (act_11 != 0) {
                        xs[6] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_12 = lo_s + (i0 + 7) * 16 + r_sub;
                    blks[7] = blk_12;
                    int act_13 = (int)(blk_12 < hi_s) & qvalid;
                    xs[7] = __uint_as_float(2143289344);
                    if (act_13 != 0) {
                        xs[7] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_14 = lo_s + (i0 + 8) * 16 + r_sub;
                    blks[8] = blk_14;
                    int act_15 = (int)(blk_14 < hi_s) & qvalid;
                    xs[8] = __uint_as_float(2143289344);
                    if (act_15 != 0) {
                        xs[8] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_16 = lo_s + (i0 + 9) * 16 + r_sub;
                    blks[9] = blk_16;
                    int act_17 = (int)(blk_16 < hi_s) & qvalid;
                    xs[9] = __uint_as_float(2143289344);
                    if (act_17 != 0) {
                        xs[9] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_18 = lo_s + (i0 + 10) * 16 + r_sub;
                    blks[10] = blk_18;
                    int act_19 = (int)(blk_18 < hi_s) & qvalid;
                    xs[10] = __uint_as_float(2143289344);
                    if (act_19 != 0) {
                        xs[10] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_20 = lo_s + (i0 + 11) * 16 + r_sub;
                    blks[11] = blk_20;
                    int act_21 = (int)(blk_20 < hi_s) & qvalid;
                    xs[11] = __uint_as_float(2143289344);
                    if (act_21 != 0) {
                        xs[11] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_22 = lo_s + (i0 + 12) * 16 + r_sub;
                    blks[12] = blk_22;
                    int act_23 = (int)(blk_22 < hi_s) & qvalid;
                    xs[12] = __uint_as_float(2143289344);
                    if (act_23 != 0) {
                        xs[12] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_24 = lo_s + (i0 + 13) * 16 + r_sub;
                    blks[13] = blk_24;
                    int act_25 = (int)(blk_24 < hi_s) & qvalid;
                    xs[13] = __uint_as_float(2143289344);
                    if (act_25 != 0) {
                        xs[13] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_26 = lo_s + (i0 + 14) * 16 + r_sub;
                    blks[14] = blk_26;
                    int act_27 = (int)(blk_26 < hi_s) & qvalid;
                    xs[14] = __uint_as_float(2143289344);
                    if (act_27 != 0) {
                        xs[14] = max_score[off];
                    }
                    off = off + stride16;
                    int blk_28 = lo_s + (i0 + 15) * 16 + r_sub;
                    blks[15] = blk_28;
                    int act_29 = (int)(blk_28 < hi_s) & qvalid;
                    xs[15] = __uint_as_float(2143289344);
                    if (act_29 != 0) {
                        xs[15] = max_score[off];
                    }
                    off = off + stride16;
                    int chunk0 = i0 >> 3;
                    unsigned long long bm_idx = bm_base + (unsigned long long)((chunk0 * 16 + r_sub) * 4);
                    int m[8];
                    unsigned int raw = __as_u32(xs[0]);
                    unsigned int mag = raw & 2147483647;
                    unsigned int nan = (unsigned int)(mag > 2139095040);
                    unsigned int bits = ((mag != 0) ? raw : 0);
                    int em = (int)(bits >> 19 & 4095);
                    int _max_2 = ((em - 1792) > (0) ? (em - 1792) : (0));
                    int _min_6 = ((_max_2) < (511) ? (_max_2) : (511));
                    int raw_30 = _min_6;
                    int neg = (int)(bits >> 31);
                    int h10 = ((neg != 0) ? 511 - raw_30 : 512 + raw_30);
                    int hb = ((nan != 0) ? 1024 : h10);
                    atomicAdd(&hist_ext[hb * 16 + q_local], 1);
                    m[0] = ((nan != 0) ? 0 : h10);
                    unsigned int raw_31 = __as_u32(xs[1]);
                    unsigned int mag_32 = raw_31 & 2147483647;
                    unsigned int nan_33 = (unsigned int)(mag_32 > 2139095040);
                    unsigned int bits_34 = ((mag_32 != 0) ? raw_31 : 0);
                    int em_35 = (int)(bits_34 >> 19 & 4095);
                    int _max_19 = ((em_35 - 1792) > (0) ? (em_35 - 1792) : (0));
                    int _min_7 = ((_max_19) < (511) ? (_max_19) : (511));
                    int raw_36 = _min_7;
                    int neg_37 = (int)(bits_34 >> 31);
                    int h10_38 = ((neg_37 != 0) ? 511 - raw_36 : 512 + raw_36);
                    int hb_39 = ((nan_33 != 0) ? 1024 : h10_38);
                    atomicAdd(&hist_ext[hb_39 * 16 + q_local], 1);
                    m[1] = ((nan_33 != 0) ? 0 : h10_38);
                    unsigned int raw_40 = __as_u32(xs[2]);
                    unsigned int mag_41 = raw_40 & 2147483647;
                    unsigned int nan_42 = (unsigned int)(mag_41 > 2139095040);
                    unsigned int bits_43 = ((mag_41 != 0) ? raw_40 : 0);
                    int em_44 = (int)(bits_43 >> 19 & 4095);
                    int _max_36 = ((em_44 - 1792) > (0) ? (em_44 - 1792) : (0));
                    int _min_8 = ((_max_36) < (511) ? (_max_36) : (511));
                    int raw_45 = _min_8;
                    int neg_46 = (int)(bits_43 >> 31);
                    int h10_47 = ((neg_46 != 0) ? 511 - raw_45 : 512 + raw_45);
                    int hb_48 = ((nan_42 != 0) ? 1024 : h10_47);
                    atomicAdd(&hist_ext[hb_48 * 16 + q_local], 1);
                    m[2] = ((nan_42 != 0) ? 0 : h10_47);
                    unsigned int raw_49 = __as_u32(xs[3]);
                    unsigned int mag_50 = raw_49 & 2147483647;
                    unsigned int nan_51 = (unsigned int)(mag_50 > 2139095040);
                    unsigned int bits_52 = ((mag_50 != 0) ? raw_49 : 0);
                    int em_53 = (int)(bits_52 >> 19 & 4095);
                    int _max_53 = ((em_53 - 1792) > (0) ? (em_53 - 1792) : (0));
                    int _min_9 = ((_max_53) < (511) ? (_max_53) : (511));
                    int raw_54 = _min_9;
                    int neg_55 = (int)(bits_52 >> 31);
                    int h10_56 = ((neg_55 != 0) ? 511 - raw_54 : 512 + raw_54);
                    int hb_57 = ((nan_51 != 0) ? 1024 : h10_56);
                    atomicAdd(&hist_ext[hb_57 * 16 + q_local], 1);
                    m[3] = ((nan_51 != 0) ? 0 : h10_56);
                    unsigned int raw_58 = __as_u32(xs[4]);
                    unsigned int mag_59 = raw_58 & 2147483647;
                    unsigned int nan_60 = (unsigned int)(mag_59 > 2139095040);
                    unsigned int bits_61 = ((mag_59 != 0) ? raw_58 : 0);
                    int em_62 = (int)(bits_61 >> 19 & 4095);
                    int _max_70 = ((em_62 - 1792) > (0) ? (em_62 - 1792) : (0));
                    int _min_10 = ((_max_70) < (511) ? (_max_70) : (511));
                    int raw_63 = _min_10;
                    int neg_64 = (int)(bits_61 >> 31);
                    int h10_65 = ((neg_64 != 0) ? 511 - raw_63 : 512 + raw_63);
                    int hb_66 = ((nan_60 != 0) ? 1024 : h10_65);
                    atomicAdd(&hist_ext[hb_66 * 16 + q_local], 1);
                    m[4] = ((nan_60 != 0) ? 0 : h10_65);
                    unsigned int raw_67 = __as_u32(xs[5]);
                    unsigned int mag_68 = raw_67 & 2147483647;
                    unsigned int nan_69 = (unsigned int)(mag_68 > 2139095040);
                    unsigned int bits_70 = ((mag_68 != 0) ? raw_67 : 0);
                    int em_71 = (int)(bits_70 >> 19 & 4095);
                    int _max_87 = ((em_71 - 1792) > (0) ? (em_71 - 1792) : (0));
                    int _min_11 = ((_max_87) < (511) ? (_max_87) : (511));
                    int raw_72 = _min_11;
                    int neg_73 = (int)(bits_70 >> 31);
                    int h10_74 = ((neg_73 != 0) ? 511 - raw_72 : 512 + raw_72);
                    int hb_75 = ((nan_69 != 0) ? 1024 : h10_74);
                    atomicAdd(&hist_ext[hb_75 * 16 + q_local], 1);
                    m[5] = ((nan_69 != 0) ? 0 : h10_74);
                    unsigned int raw_76 = __as_u32(xs[6]);
                    unsigned int mag_77 = raw_76 & 2147483647;
                    unsigned int nan_78 = (unsigned int)(mag_77 > 2139095040);
                    unsigned int bits_79 = ((mag_77 != 0) ? raw_76 : 0);
                    int em_80 = (int)(bits_79 >> 19 & 4095);
                    int _max_104 = ((em_80 - 1792) > (0) ? (em_80 - 1792) : (0));
                    int _min_12 = ((_max_104) < (511) ? (_max_104) : (511));
                    int raw_81 = _min_12;
                    int neg_82 = (int)(bits_79 >> 31);
                    int h10_83 = ((neg_82 != 0) ? 511 - raw_81 : 512 + raw_81);
                    int hb_84 = ((nan_78 != 0) ? 1024 : h10_83);
                    atomicAdd(&hist_ext[hb_84 * 16 + q_local], 1);
                    m[6] = ((nan_78 != 0) ? 0 : h10_83);
                    unsigned int raw_85 = __as_u32(xs[7]);
                    unsigned int mag_86 = raw_85 & 2147483647;
                    unsigned int nan_87 = (unsigned int)(mag_86 > 2139095040);
                    unsigned int bits_88 = ((mag_86 != 0) ? raw_85 : 0);
                    int em_89 = (int)(bits_88 >> 19 & 4095);
                    int _max_121 = ((em_89 - 1792) > (0) ? (em_89 - 1792) : (0));
                    int _min_13 = ((_max_121) < (511) ? (_max_121) : (511));
                    int raw_90 = _min_13;
                    int neg_91 = (int)(bits_88 >> 31);
                    int h10_92 = ((neg_91 != 0) ? 511 - raw_90 : 512 + raw_90);
                    int hb_93 = ((nan_87 != 0) ? 1024 : h10_92);
                    atomicAdd(&hist_ext[hb_93 * 16 + q_local], 1);
                    m[7] = ((nan_87 != 0) ? 0 : h10_92);
                    {
                        int b3 = lane_1 & 8;
                        int b2 = lane_1 & 4;
                        int b1 = lane_1 & 2;
                        int a[4];
                        int send = ((b3 != 0) ? m[0] : m[4]);
                        int keep = ((b3 != 0) ? m[4] : m[0]);
                        int _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, send, 8);
                        int recv = _shfl_xor_126;
                        int _max_122 = ((keep) > (recv) ? (keep) : (recv));
                        a[0] = _max_122;
                        int send_0 = ((b3 != 0) ? m[1] : m[5]);
                        int keep_1 = ((b3 != 0) ? m[5] : m[1]);
                        int _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, send_0, 8);
                        int recv_2 = _shfl_xor_127;
                        int _max_123 = ((keep_1) > (recv_2) ? (keep_1) : (recv_2));
                        a[1] = _max_123;
                        int send_3 = ((b3 != 0) ? m[2] : m[6]);
                        int keep_4 = ((b3 != 0) ? m[6] : m[2]);
                        int _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, send_3, 8);
                        int recv_5 = _shfl_xor_128;
                        int _max_124 = ((keep_4) > (recv_5) ? (keep_4) : (recv_5));
                        a[2] = _max_124;
                        int send_6 = ((b3 != 0) ? m[3] : m[7]);
                        int keep_7 = ((b3 != 0) ? m[7] : m[3]);
                        int _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, send_6, 8);
                        int recv_8 = _shfl_xor_129;
                        int _max_125 = ((keep_7) > (recv_8) ? (keep_7) : (recv_8));
                        a[3] = _max_125;
                        int b[2];
                        int send2 = ((b2 != 0) ? a[0] : a[2]);
                        int keep2 = ((b2 != 0) ? a[2] : a[0]);
                        int _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, send2, 4);
                        int recv2 = _shfl_xor_130;
                        int _max_126 = ((keep2) > (recv2) ? (keep2) : (recv2));
                        b[0] = _max_126;
                        int send2_9 = ((b2 != 0) ? a[1] : a[3]);
                        int keep2_10 = ((b2 != 0) ? a[3] : a[1]);
                        int _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, send2_9, 4);
                        int recv2_11 = _shfl_xor_131;
                        int _max_127 = ((keep2_10) > (recv2_11) ? (keep2_10) : (recv2_11));
                        b[1] = _max_127;
                        int send3 = ((b1 != 0) ? b[0] : b[1]);
                        int keep3 = ((b1 != 0) ? b[1] : b[0]);
                        int _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, send3, 2);
                        int recv3 = _shfl_xor_132;
                        int _max_128 = ((keep3) > (recv3) ? (keep3) : (recv3));
                        int c = _max_128;
                        int _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
                        int recv4 = _shfl_xor_133;
                        int _max_129 = ((c) > (recv4) ? (c) : (recv4));
                        int d = _max_129;
                        int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, d, 2);
                        int other = _shfl_xor_134;
                        int word = ((b1 == 0) ? d | other << 16 : other | d << 16);
                        block_max[bm_idx + (unsigned long long)(lane_1 >> 2 & 3)] = (unsigned int)word;
                    }
                    if (i1 < iters) {
                        int m_0[8];
                        unsigned int raw_1 = __as_u32(xs[8]);
                        unsigned int mag_2 = raw_1 & 2147483647;
                        unsigned int nan_3 = (unsigned int)(mag_2 > 2139095040);
                        unsigned int bits_4 = ((mag_2 != 0) ? raw_1 : 0);
                        int em_5 = (int)(bits_4 >> 19 & 4095);
                        int _max_138 = ((em_5 - 1792) > (0) ? (em_5 - 1792) : (0));
                        int _min_14 = ((_max_138) < (511) ? (_max_138) : (511));
                        int raw_6 = _min_14;
                        int neg_7 = (int)(bits_4 >> 31);
                        int h10_8 = ((neg_7 != 0) ? 511 - raw_6 : 512 + raw_6);
                        int hb_9 = ((nan_3 != 0) ? 1024 : h10_8);
                        atomicAdd(&hist_ext[hb_9 * 16 + q_local], 1);
                        m_0[0] = ((nan_3 != 0) ? 0 : h10_8);
                        unsigned int raw_10 = __as_u32(xs[9]);
                        unsigned int mag_11 = raw_10 & 2147483647;
                        unsigned int nan_12 = (unsigned int)(mag_11 > 2139095040);
                        unsigned int bits_13 = ((mag_11 != 0) ? raw_10 : 0);
                        int em_14 = (int)(bits_13 >> 19 & 4095);
                        int _max_155 = ((em_14 - 1792) > (0) ? (em_14 - 1792) : (0));
                        int _min_15 = ((_max_155) < (511) ? (_max_155) : (511));
                        int raw_15 = _min_15;
                        int neg_16 = (int)(bits_13 >> 31);
                        int h10_17 = ((neg_16 != 0) ? 511 - raw_15 : 512 + raw_15);
                        int hb_18 = ((nan_12 != 0) ? 1024 : h10_17);
                        atomicAdd(&hist_ext[hb_18 * 16 + q_local], 1);
                        m_0[1] = ((nan_12 != 0) ? 0 : h10_17);
                        unsigned int raw_19 = __as_u32(xs[10]);
                        unsigned int mag_20 = raw_19 & 2147483647;
                        unsigned int nan_21 = (unsigned int)(mag_20 > 2139095040);
                        unsigned int bits_22 = ((mag_20 != 0) ? raw_19 : 0);
                        int em_23 = (int)(bits_22 >> 19 & 4095);
                        int _max_172 = ((em_23 - 1792) > (0) ? (em_23 - 1792) : (0));
                        int _min_16 = ((_max_172) < (511) ? (_max_172) : (511));
                        int raw_24 = _min_16;
                        int neg_25 = (int)(bits_22 >> 31);
                        int h10_26 = ((neg_25 != 0) ? 511 - raw_24 : 512 + raw_24);
                        int hb_27 = ((nan_21 != 0) ? 1024 : h10_26);
                        atomicAdd(&hist_ext[hb_27 * 16 + q_local], 1);
                        m_0[2] = ((nan_21 != 0) ? 0 : h10_26);
                        unsigned int raw_28 = __as_u32(xs[11]);
                        unsigned int mag_29 = raw_28 & 2147483647;
                        unsigned int nan_30 = (unsigned int)(mag_29 > 2139095040);
                        unsigned int bits_31 = ((mag_29 != 0) ? raw_28 : 0);
                        int em_32 = (int)(bits_31 >> 19 & 4095);
                        int _max_189 = ((em_32 - 1792) > (0) ? (em_32 - 1792) : (0));
                        int _min_17 = ((_max_189) < (511) ? (_max_189) : (511));
                        int raw_33 = _min_17;
                        int neg_34 = (int)(bits_31 >> 31);
                        int h10_35 = ((neg_34 != 0) ? 511 - raw_33 : 512 + raw_33);
                        int hb_36 = ((nan_30 != 0) ? 1024 : h10_35);
                        atomicAdd(&hist_ext[hb_36 * 16 + q_local], 1);
                        m_0[3] = ((nan_30 != 0) ? 0 : h10_35);
                        unsigned int raw_37 = __as_u32(xs[12]);
                        unsigned int mag_38 = raw_37 & 2147483647;
                        unsigned int nan_39 = (unsigned int)(mag_38 > 2139095040);
                        unsigned int bits_40 = ((mag_38 != 0) ? raw_37 : 0);
                        int em_41 = (int)(bits_40 >> 19 & 4095);
                        int _max_206 = ((em_41 - 1792) > (0) ? (em_41 - 1792) : (0));
                        int _min_18 = ((_max_206) < (511) ? (_max_206) : (511));
                        int raw_42 = _min_18;
                        int neg_43 = (int)(bits_40 >> 31);
                        int h10_44 = ((neg_43 != 0) ? 511 - raw_42 : 512 + raw_42);
                        int hb_45 = ((nan_39 != 0) ? 1024 : h10_44);
                        atomicAdd(&hist_ext[hb_45 * 16 + q_local], 1);
                        m_0[4] = ((nan_39 != 0) ? 0 : h10_44);
                        unsigned int raw_46 = __as_u32(xs[13]);
                        unsigned int mag_47 = raw_46 & 2147483647;
                        unsigned int nan_48 = (unsigned int)(mag_47 > 2139095040);
                        unsigned int bits_49 = ((mag_47 != 0) ? raw_46 : 0);
                        int em_50 = (int)(bits_49 >> 19 & 4095);
                        int _max_223 = ((em_50 - 1792) > (0) ? (em_50 - 1792) : (0));
                        int _min_19 = ((_max_223) < (511) ? (_max_223) : (511));
                        int raw_51 = _min_19;
                        int neg_52 = (int)(bits_49 >> 31);
                        int h10_53 = ((neg_52 != 0) ? 511 - raw_51 : 512 + raw_51);
                        int hb_54 = ((nan_48 != 0) ? 1024 : h10_53);
                        atomicAdd(&hist_ext[hb_54 * 16 + q_local], 1);
                        m_0[5] = ((nan_48 != 0) ? 0 : h10_53);
                        unsigned int raw_55 = __as_u32(xs[14]);
                        unsigned int mag_56 = raw_55 & 2147483647;
                        unsigned int nan_57 = (unsigned int)(mag_56 > 2139095040);
                        unsigned int bits_58 = ((mag_56 != 0) ? raw_55 : 0);
                        int em_59 = (int)(bits_58 >> 19 & 4095);
                        int _max_240 = ((em_59 - 1792) > (0) ? (em_59 - 1792) : (0));
                        int _min_20 = ((_max_240) < (511) ? (_max_240) : (511));
                        int raw_60 = _min_20;
                        int neg_61 = (int)(bits_58 >> 31);
                        int h10_62 = ((neg_61 != 0) ? 511 - raw_60 : 512 + raw_60);
                        int hb_63 = ((nan_57 != 0) ? 1024 : h10_62);
                        atomicAdd(&hist_ext[hb_63 * 16 + q_local], 1);
                        m_0[6] = ((nan_57 != 0) ? 0 : h10_62);
                        unsigned int raw_64 = __as_u32(xs[15]);
                        unsigned int mag_65 = raw_64 & 2147483647;
                        unsigned int nan_66 = (unsigned int)(mag_65 > 2139095040);
                        unsigned int bits_67 = ((mag_65 != 0) ? raw_64 : 0);
                        int em_68 = (int)(bits_67 >> 19 & 4095);
                        int _max_257 = ((em_68 - 1792) > (0) ? (em_68 - 1792) : (0));
                        int _min_21 = ((_max_257) < (511) ? (_max_257) : (511));
                        int raw_69 = _min_21;
                        int neg_70 = (int)(bits_67 >> 31);
                        int h10_71 = ((neg_70 != 0) ? 511 - raw_69 : 512 + raw_69);
                        int hb_72 = ((nan_66 != 0) ? 1024 : h10_71);
                        atomicAdd(&hist_ext[hb_72 * 16 + q_local], 1);
                        m_0[7] = ((nan_66 != 0) ? 0 : h10_71);
                        if (nchunks_nom > chunk0 + 1) {
                            int b3_1 = lane_1 & 8;
                            int b2_1 = lane_1 & 4;
                            int b1_1 = lane_1 & 2;
                            int a_1[4];
                            int send_1 = ((b3_1 != 0) ? m_0[0] : m_0[4]);
                            int keep_2 = ((b3_1 != 0) ? m_0[4] : m_0[0]);
                            int _shfl_xor_279 = __shfl_xor_sync(0xFFFFFFFF, send_1, 8);
                            int recv_1 = _shfl_xor_279;
                            int _max_266 = ((keep_2) > (recv_1) ? (keep_2) : (recv_1));
                            a_1[0] = _max_266;
                            int send_0_1 = ((b3_1 != 0) ? m_0[1] : m_0[5]);
                            int keep_1_1 = ((b3_1 != 0) ? m_0[5] : m_0[1]);
                            int _shfl_xor_280 = __shfl_xor_sync(0xFFFFFFFF, send_0_1, 8);
                            int recv_2_1 = _shfl_xor_280;
                            int _max_267 = ((keep_1_1) > (recv_2_1) ? (keep_1_1) : (recv_2_1));
                            a_1[1] = _max_267;
                            int send_3_1 = ((b3_1 != 0) ? m_0[2] : m_0[6]);
                            int keep_4_1 = ((b3_1 != 0) ? m_0[6] : m_0[2]);
                            int _shfl_xor_281 = __shfl_xor_sync(0xFFFFFFFF, send_3_1, 8);
                            int recv_5_1 = _shfl_xor_281;
                            int _max_268 = ((keep_4_1) > (recv_5_1) ? (keep_4_1) : (recv_5_1));
                            a_1[2] = _max_268;
                            int send_6_1 = ((b3_1 != 0) ? m_0[3] : m_0[7]);
                            int keep_7_1 = ((b3_1 != 0) ? m_0[7] : m_0[3]);
                            int _shfl_xor_282 = __shfl_xor_sync(0xFFFFFFFF, send_6_1, 8);
                            int recv_8_1 = _shfl_xor_282;
                            int _max_269 = ((keep_7_1) > (recv_8_1) ? (keep_7_1) : (recv_8_1));
                            a_1[3] = _max_269;
                            int b_1[2];
                            int send2_1 = ((b2_1 != 0) ? a_1[0] : a_1[2]);
                            int keep2_1 = ((b2_1 != 0) ? a_1[2] : a_1[0]);
                            int _shfl_xor_283 = __shfl_xor_sync(0xFFFFFFFF, send2_1, 4);
                            int recv2_1 = _shfl_xor_283;
                            int _max_270 = ((keep2_1) > (recv2_1) ? (keep2_1) : (recv2_1));
                            b_1[0] = _max_270;
                            int send2_9_1 = ((b2_1 != 0) ? a_1[1] : a_1[3]);
                            int keep2_10_1 = ((b2_1 != 0) ? a_1[3] : a_1[1]);
                            int _shfl_xor_284 = __shfl_xor_sync(0xFFFFFFFF, send2_9_1, 4);
                            int recv2_11_1 = _shfl_xor_284;
                            int _max_271 = ((keep2_10_1) > (recv2_11_1) ? (keep2_10_1) : (recv2_11_1));
                            b_1[1] = _max_271;
                            int send3_1 = ((b1_1 != 0) ? b_1[0] : b_1[1]);
                            int keep3_1 = ((b1_1 != 0) ? b_1[1] : b_1[0]);
                            int _shfl_xor_285 = __shfl_xor_sync(0xFFFFFFFF, send3_1, 2);
                            int recv3_1 = _shfl_xor_285;
                            int _max_272 = ((keep3_1) > (recv3_1) ? (keep3_1) : (recv3_1));
                            int c_1 = _max_272;
                            int _shfl_xor_286 = __shfl_xor_sync(0xFFFFFFFF, c_1, 1);
                            int recv4_1 = _shfl_xor_286;
                            int _max_273 = ((c_1) > (recv4_1) ? (c_1) : (recv4_1));
                            int d_1 = _max_273;
                            int _shfl_xor_287 = __shfl_xor_sync(0xFFFFFFFF, d_1, 2);
                            int other_1 = _shfl_xor_287;
                            int word_1 = ((b1_1 == 0) ? d_1 | other_1 << 16 : other_1 | d_1 << 16);
                            block_max[bm_idx + 64 + (unsigned long long)(lane_1 >> 2 & 3)] = (unsigned int)word_1;
                        }
                    }
                }
            } else if (n_hist == 0) {
                if (p > 0 && slen > 224) {
                    int nchunks = iters + 8 - 1 >> 3;
                    for (int w0 = 0; w0 < slen; w0 += 8192) {
                        int c_lo = w0 >> 7;
                        int _min_22 = ((nchunks) < (w0 + 8192 >> 7) ? (nchunks) : (w0 + 8192 >> 7));
                        int c_hi = _min_22;
                        int n_ent = (c_hi - c_lo) * 16;
                        int cnt = 0;
                        int masks[4];
                        int js[4];
                        int e = tid_0;
                        int c_2 = c_lo + (e >> 4);
                        int r_0 = e & 15;
                        js[0] = c_2 * 128 + r_0;
                        masks[0] = 0;
                        if (e < n_ent) {
                            unsigned int _vec_load_0[4];
                            {
                                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_2 * 16 + r_0) * 4));
                                _vec_load_0[0 + 0] = _uv4_0.x;
                                _vec_load_0[0 + 1] = _uv4_0.y;
                                _vec_load_0[0 + 2] = _uv4_0.z;
                                _vec_load_0[0 + 3] = _uv4_0.w;
                            }
                            int mk = 0;
                            unsigned int mu = _vec_load_0[0] & 65535;
                            int j_0 = js[0];
                            int keep_3 = (int)(mu >= tmin && j_0 < slen);
                            mk = mk | keep_3;
                            unsigned int mu_1 = _vec_load_0[0] >> 16 & 65535;
                            int j_2 = js[0] + 16;
                            int keep_3_1 = (int)(mu_1 >= tmin && j_2 < slen);
                            mk = mk | keep_3_1 << 1;
                            unsigned int mu_4 = _vec_load_0[1] & 65535;
                            int j_5 = js[0] + 32;
                            int keep_6 = (int)(mu_4 >= tmin && j_5 < slen);
                            mk = mk | keep_6 << 2;
                            unsigned int mu_7 = _vec_load_0[1] >> 16 & 65535;
                            int j_8 = js[0] + 48;
                            int keep_9 = (int)(mu_7 >= tmin && j_8 < slen);
                            mk = mk | keep_9 << 3;
                            unsigned int mu_10 = _vec_load_0[2] & 65535;
                            int j_11 = js[0] + 64;
                            int keep_12 = (int)(mu_10 >= tmin && j_11 < slen);
                            mk = mk | keep_12 << 4;
                            unsigned int mu_13 = _vec_load_0[2] >> 16 & 65535;
                            int j_14 = js[0] + 80;
                            int keep_15 = (int)(mu_13 >= tmin && j_14 < slen);
                            mk = mk | keep_15 << 5;
                            unsigned int mu_16 = _vec_load_0[3] & 65535;
                            int j_17 = js[0] + 96;
                            int keep_18 = (int)(mu_16 >= tmin && j_17 < slen);
                            mk = mk | keep_18 << 6;
                            unsigned int mu_19 = _vec_load_0[3] >> 16 & 65535;
                            int j_20 = js[0] + 112;
                            int keep_21 = (int)(mu_19 >= tmin && j_20 < slen);
                            mk = mk | keep_21 << 7;
                            masks[0] = mk;
                            int _popc_0 = __popc(mk);
                            cnt = cnt + _popc_0;
                        }
                        int e_1 = tid_0 + 256;
                        int c_2_1 = c_lo + (e_1 >> 4);
                        int r_3 = e_1 & 15;
                        js[1] = c_2_1 * 128 + r_3;
                        masks[1] = 0;
                        if (e_1 < n_ent) {
                            unsigned int _vec_load_1[4];
                            {
                                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_2_1 * 16 + r_3) * 4));
                                _vec_load_1[0 + 0] = _uv4_1.x;
                                _vec_load_1[0 + 1] = _uv4_1.y;
                                _vec_load_1[0 + 2] = _uv4_1.z;
                                _vec_load_1[0 + 3] = _uv4_1.w;
                            }
                            int mk_1 = 0;
                            unsigned int mu_2 = _vec_load_1[0] & 65535;
                            int j_0_1 = js[1];
                            int keep_5 = (int)(mu_2 >= tmin && j_0_1 < slen);
                            mk_1 = mk_1 | keep_5;
                            unsigned int mu_1_1 = _vec_load_1[0] >> 16 & 65535;
                            int j_2_1 = js[1] + 16;
                            int keep_3_2 = (int)(mu_1_1 >= tmin && j_2_1 < slen);
                            mk_1 = mk_1 | keep_3_2 << 1;
                            unsigned int mu_4_1 = _vec_load_1[1] & 65535;
                            int j_5_1 = js[1] + 32;
                            int keep_6_1 = (int)(mu_4_1 >= tmin && j_5_1 < slen);
                            mk_1 = mk_1 | keep_6_1 << 2;
                            unsigned int mu_7_1 = _vec_load_1[1] >> 16 & 65535;
                            int j_8_1 = js[1] + 48;
                            int keep_9_1 = (int)(mu_7_1 >= tmin && j_8_1 < slen);
                            mk_1 = mk_1 | keep_9_1 << 3;
                            unsigned int mu_10_1 = _vec_load_1[2] & 65535;
                            int j_11_1 = js[1] + 64;
                            int keep_12_1 = (int)(mu_10_1 >= tmin && j_11_1 < slen);
                            mk_1 = mk_1 | keep_12_1 << 4;
                            unsigned int mu_13_1 = _vec_load_1[2] >> 16 & 65535;
                            int j_14_1 = js[1] + 80;
                            int keep_15_1 = (int)(mu_13_1 >= tmin && j_14_1 < slen);
                            mk_1 = mk_1 | keep_15_1 << 5;
                            unsigned int mu_16_1 = _vec_load_1[3] & 65535;
                            int j_17_1 = js[1] + 96;
                            int keep_18_1 = (int)(mu_16_1 >= tmin && j_17_1 < slen);
                            mk_1 = mk_1 | keep_18_1 << 6;
                            unsigned int mu_19_1 = _vec_load_1[3] >> 16 & 65535;
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
                            unsigned int _vec_load_2[4];
                            {
                                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_5 * 16 + r_6) * 4));
                                _vec_load_2[0 + 0] = _uv4_2.x;
                                _vec_load_2[0 + 1] = _uv4_2.y;
                                _vec_load_2[0 + 2] = _uv4_2.z;
                                _vec_load_2[0 + 3] = _uv4_2.w;
                            }
                            int mk_2 = 0;
                            unsigned int mu_3 = _vec_load_2[0] & 65535;
                            int j_0_2 = js[2];
                            int keep_8 = (int)(mu_3 >= tmin && j_0_2 < slen);
                            mk_2 = mk_2 | keep_8;
                            unsigned int mu_1_2 = _vec_load_2[0] >> 16 & 65535;
                            int j_2_2 = js[2] + 16;
                            int keep_3_3 = (int)(mu_1_2 >= tmin && j_2_2 < slen);
                            mk_2 = mk_2 | keep_3_3 << 1;
                            unsigned int mu_4_2 = _vec_load_2[1] & 65535;
                            int j_5_2 = js[2] + 32;
                            int keep_6_2 = (int)(mu_4_2 >= tmin && j_5_2 < slen);
                            mk_2 = mk_2 | keep_6_2 << 2;
                            unsigned int mu_7_2 = _vec_load_2[1] >> 16 & 65535;
                            int j_8_2 = js[2] + 48;
                            int keep_9_2 = (int)(mu_7_2 >= tmin && j_8_2 < slen);
                            mk_2 = mk_2 | keep_9_2 << 3;
                            unsigned int mu_10_2 = _vec_load_2[2] & 65535;
                            int j_11_2 = js[2] + 64;
                            int keep_12_2 = (int)(mu_10_2 >= tmin && j_11_2 < slen);
                            mk_2 = mk_2 | keep_12_2 << 4;
                            unsigned int mu_13_2 = _vec_load_2[2] >> 16 & 65535;
                            int j_14_2 = js[2] + 80;
                            int keep_15_2 = (int)(mu_13_2 >= tmin && j_14_2 < slen);
                            mk_2 = mk_2 | keep_15_2 << 5;
                            unsigned int mu_16_2 = _vec_load_2[3] & 65535;
                            int j_17_2 = js[2] + 96;
                            int keep_18_2 = (int)(mu_16_2 >= tmin && j_17_2 < slen);
                            mk_2 = mk_2 | keep_18_2 << 6;
                            unsigned int mu_19_2 = _vec_load_2[3] >> 16 & 65535;
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
                            unsigned int _vec_load_3[4];
                            {
                                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_8 * 16 + r_9) * 4));
                                _vec_load_3[0 + 0] = _uv4_3.x;
                                _vec_load_3[0 + 1] = _uv4_3.y;
                                _vec_load_3[0 + 2] = _uv4_3.z;
                                _vec_load_3[0 + 3] = _uv4_3.w;
                            }
                            int mk_3 = 0;
                            unsigned int mu_5 = _vec_load_3[0] & 65535;
                            int j_0_3 = js[3];
                            int keep_10 = (int)(mu_5 >= tmin && j_0_3 < slen);
                            mk_3 = mk_3 | keep_10;
                            unsigned int mu_1_3 = _vec_load_3[0] >> 16 & 65535;
                            int j_2_3 = js[3] + 16;
                            int keep_3_4 = (int)(mu_1_3 >= tmin && j_2_3 < slen);
                            mk_3 = mk_3 | keep_3_4 << 1;
                            unsigned int mu_4_3 = _vec_load_3[1] & 65535;
                            int j_5_3 = js[3] + 32;
                            int keep_6_3 = (int)(mu_4_3 >= tmin && j_5_3 < slen);
                            mk_3 = mk_3 | keep_6_3 << 2;
                            unsigned int mu_7_3 = _vec_load_3[1] >> 16 & 65535;
                            int j_8_3 = js[3] + 48;
                            int keep_9_3 = (int)(mu_7_3 >= tmin && j_8_3 < slen);
                            mk_3 = mk_3 | keep_9_3 << 3;
                            unsigned int mu_10_3 = _vec_load_3[2] & 65535;
                            int j_11_3 = js[3] + 64;
                            int keep_12_3 = (int)(mu_10_3 >= tmin && j_11_3 < slen);
                            mk_3 = mk_3 | keep_12_3 << 4;
                            unsigned int mu_13_3 = _vec_load_3[2] >> 16 & 65535;
                            int j_14_3 = js[3] + 80;
                            int keep_15_3 = (int)(mu_13_3 >= tmin && j_14_3 < slen);
                            mk_3 = mk_3 | keep_15_3 << 5;
                            unsigned int mu_16_3 = _vec_load_3[3] & 65535;
                            int j_17_3 = js[3] + 96;
                            int keep_18_3 = (int)(mu_16_3 >= tmin && j_17_3 < slen);
                            mk_3 = mk_3 | keep_18_3 << 6;
                            unsigned int mu_19_3 = _vec_load_3[3] >> 16 & 65535;
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
                        int dummy = 768 + tid_0;
                        int keep2_2 = masks[0] & 1;
                        int idx = ((keep2_2 != 0) ? 1024 + base : dummy);
                        hist[idx] = (unsigned int)(lo_s + js[0]);
                        base = base + keep2_2;
                        int keep2_21 = masks[0] >> 1 & 1;
                        int idx_22 = ((keep2_21 != 0) ? 1024 + base : dummy);
                        hist[idx_22] = (unsigned int)(lo_s + js[0] + 16);
                        base = base + keep2_21;
                        int keep2_23 = masks[0] >> 2 & 1;
                        int idx_24 = ((keep2_23 != 0) ? 1024 + base : dummy);
                        hist[idx_24] = (unsigned int)(lo_s + js[0] + 32);
                        base = base + keep2_23;
                        int keep2_25 = masks[0] >> 3 & 1;
                        int idx_26 = ((keep2_25 != 0) ? 1024 + base : dummy);
                        hist[idx_26] = (unsigned int)(lo_s + js[0] + 48);
                        base = base + keep2_25;
                        int keep2_27 = masks[0] >> 4 & 1;
                        int idx_28 = ((keep2_27 != 0) ? 1024 + base : dummy);
                        hist[idx_28] = (unsigned int)(lo_s + js[0] + 64);
                        base = base + keep2_27;
                        int keep2_29 = masks[0] >> 5 & 1;
                        int idx_30 = ((keep2_29 != 0) ? 1024 + base : dummy);
                        hist[idx_30] = (unsigned int)(lo_s + js[0] + 80);
                        base = base + keep2_29;
                        int keep2_31 = masks[0] >> 6 & 1;
                        int idx_32 = ((keep2_31 != 0) ? 1024 + base : dummy);
                        hist[idx_32] = (unsigned int)(lo_s + js[0] + 96);
                        base = base + keep2_31;
                        int keep2_33 = masks[0] >> 7 & 1;
                        int idx_34 = ((keep2_33 != 0) ? 1024 + base : dummy);
                        hist[idx_34] = (unsigned int)(lo_s + js[0] + 112);
                        base = base + keep2_33;
                        int keep2_35 = masks[1] & 1;
                        int idx_36 = ((keep2_35 != 0) ? 1024 + base : dummy);
                        hist[idx_36] = (unsigned int)(lo_s + js[1]);
                        base = base + keep2_35;
                        int keep2_37 = masks[1] >> 1 & 1;
                        int idx_38 = ((keep2_37 != 0) ? 1024 + base : dummy);
                        hist[idx_38] = (unsigned int)(lo_s + js[1] + 16);
                        base = base + keep2_37;
                        int keep2_39 = masks[1] >> 2 & 1;
                        int idx_40 = ((keep2_39 != 0) ? 1024 + base : dummy);
                        hist[idx_40] = (unsigned int)(lo_s + js[1] + 32);
                        base = base + keep2_39;
                        int keep2_41 = masks[1] >> 3 & 1;
                        int idx_42 = ((keep2_41 != 0) ? 1024 + base : dummy);
                        hist[idx_42] = (unsigned int)(lo_s + js[1] + 48);
                        base = base + keep2_41;
                        int keep2_43 = masks[1] >> 4 & 1;
                        int idx_44 = ((keep2_43 != 0) ? 1024 + base : dummy);
                        hist[idx_44] = (unsigned int)(lo_s + js[1] + 64);
                        base = base + keep2_43;
                        int keep2_45 = masks[1] >> 5 & 1;
                        int idx_46 = ((keep2_45 != 0) ? 1024 + base : dummy);
                        hist[idx_46] = (unsigned int)(lo_s + js[1] + 80);
                        base = base + keep2_45;
                        int keep2_47 = masks[1] >> 6 & 1;
                        int idx_48 = ((keep2_47 != 0) ? 1024 + base : dummy);
                        hist[idx_48] = (unsigned int)(lo_s + js[1] + 96);
                        base = base + keep2_47;
                        int keep2_49 = masks[1] >> 7 & 1;
                        int idx_50 = ((keep2_49 != 0) ? 1024 + base : dummy);
                        hist[idx_50] = (unsigned int)(lo_s + js[1] + 112);
                        base = base + keep2_49;
                        int keep2_51 = masks[2] & 1;
                        int idx_52 = ((keep2_51 != 0) ? 1024 + base : dummy);
                        hist[idx_52] = (unsigned int)(lo_s + js[2]);
                        base = base + keep2_51;
                        int keep2_53 = masks[2] >> 1 & 1;
                        int idx_54 = ((keep2_53 != 0) ? 1024 + base : dummy);
                        hist[idx_54] = (unsigned int)(lo_s + js[2] + 16);
                        base = base + keep2_53;
                        int keep2_55 = masks[2] >> 2 & 1;
                        int idx_56 = ((keep2_55 != 0) ? 1024 + base : dummy);
                        hist[idx_56] = (unsigned int)(lo_s + js[2] + 32);
                        base = base + keep2_55;
                        int keep2_57 = masks[2] >> 3 & 1;
                        int idx_58 = ((keep2_57 != 0) ? 1024 + base : dummy);
                        hist[idx_58] = (unsigned int)(lo_s + js[2] + 48);
                        base = base + keep2_57;
                        int keep2_59 = masks[2] >> 4 & 1;
                        int idx_60 = ((keep2_59 != 0) ? 1024 + base : dummy);
                        hist[idx_60] = (unsigned int)(lo_s + js[2] + 64);
                        base = base + keep2_59;
                        int keep2_61 = masks[2] >> 5 & 1;
                        int idx_62 = ((keep2_61 != 0) ? 1024 + base : dummy);
                        hist[idx_62] = (unsigned int)(lo_s + js[2] + 80);
                        base = base + keep2_61;
                        int keep2_63 = masks[2] >> 6 & 1;
                        int idx_64 = ((keep2_63 != 0) ? 1024 + base : dummy);
                        hist[idx_64] = (unsigned int)(lo_s + js[2] + 96);
                        base = base + keep2_63;
                        int keep2_65 = masks[2] >> 7 & 1;
                        int idx_66 = ((keep2_65 != 0) ? 1024 + base : dummy);
                        hist[idx_66] = (unsigned int)(lo_s + js[2] + 112);
                        base = base + keep2_65;
                        int keep2_67 = masks[3] & 1;
                        int idx_68 = ((keep2_67 != 0) ? 1024 + base : dummy);
                        hist[idx_68] = (unsigned int)(lo_s + js[3]);
                        base = base + keep2_67;
                        int keep2_69 = masks[3] >> 1 & 1;
                        int idx_70 = ((keep2_69 != 0) ? 1024 + base : dummy);
                        hist[idx_70] = (unsigned int)(lo_s + js[3] + 16);
                        base = base + keep2_69;
                        int keep2_71 = masks[3] >> 2 & 1;
                        int idx_72 = ((keep2_71 != 0) ? 1024 + base : dummy);
                        hist[idx_72] = (unsigned int)(lo_s + js[3] + 32);
                        base = base + keep2_71;
                        int keep2_73 = masks[3] >> 3 & 1;
                        int idx_74 = ((keep2_73 != 0) ? 1024 + base : dummy);
                        hist[idx_74] = (unsigned int)(lo_s + js[3] + 48);
                        base = base + keep2_73;
                        int keep2_75 = masks[3] >> 4 & 1;
                        int idx_76 = ((keep2_75 != 0) ? 1024 + base : dummy);
                        hist[idx_76] = (unsigned int)(lo_s + js[3] + 64);
                        base = base + keep2_75;
                        int keep2_77 = masks[3] >> 5 & 1;
                        int idx_78 = ((keep2_77 != 0) ? 1024 + base : dummy);
                        hist[idx_78] = (unsigned int)(lo_s + js[3] + 80);
                        base = base + keep2_77;
                        int keep2_79 = masks[3] >> 6 & 1;
                        int idx_80 = ((keep2_79 != 0) ? 1024 + base : dummy);
                        hist[idx_80] = (unsigned int)(lo_s + js[3] + 96);
                        base = base + keep2_79;
                        int keep2_81 = masks[3] >> 7 & 1;
                        int idx_82 = ((keep2_81 != 0) ? 1024 + base : dummy);
                        hist[idx_82] = (unsigned int)(lo_s + js[3] + 112);
                        base = base + keep2_81;
                        nres[0] = total;
                        __syncthreads();
                        int n_list = nres[0];
                        int n_it = n_list + 15 >> 4;
                        #pragma unroll 1
                        for (int i0_1 = 0; i0_1 < n_it; i0_1 += 16) {
                            int i1_1 = i0_1 + 8;
                            int _max_274 = ((n_list - 1) > (0) ? (n_list - 1) : (0));
                            int last = _max_274;
                            int e_0 = i0_1 * 16 + r_sub;
                            int act_2 = (int)(e_0 < n_list) & qvalid;
                            int _min_23 = ((e_0) < (last) ? (e_0) : (last));
                            int blk_1 = (int)hist[1024 + _min_23];
                            blks[0] = blk_1;
                            unsigned long long off_1 = (row_base64 + (unsigned long long)blk_1) * total_q64 + q_addr64;
                            xs[0] = __uint_as_float(2143289344);
                            if (act_2 != 0) {
                                xs[0] = max_score[off_1];
                            }
                            int e_2 = (i0_1 + 1) * 16 + r_sub;
                            int act_3_1 = (int)(e_2 < n_list) & qvalid;
                            int _min_24 = ((e_2) < (last) ? (e_2) : (last));
                            int blk_4_1 = (int)hist[1024 + _min_24];
                            blks[1] = blk_4_1;
                            unsigned long long off_5 = (row_base64 + (unsigned long long)blk_4_1) * total_q64 + q_addr64;
                            xs[1] = __uint_as_float(2143289344);
                            if (act_3_1 != 0) {
                                xs[1] = max_score[off_5];
                            }
                            int e_6 = (i0_1 + 2) * 16 + r_sub;
                            int act_7_1 = (int)(e_6 < n_list) & qvalid;
                            int _min_25 = ((e_6) < (last) ? (e_6) : (last));
                            int blk_8_1 = (int)hist[1024 + _min_25];
                            blks[2] = blk_8_1;
                            unsigned long long off_9 = (row_base64 + (unsigned long long)blk_8_1) * total_q64 + q_addr64;
                            xs[2] = __uint_as_float(2143289344);
                            if (act_7_1 != 0) {
                                xs[2] = max_score[off_9];
                            }
                            int e_10 = (i0_1 + 3) * 16 + r_sub;
                            int act_11_1 = (int)(e_10 < n_list) & qvalid;
                            int _min_26 = ((e_10) < (last) ? (e_10) : (last));
                            int blk_12_1 = (int)hist[1024 + _min_26];
                            blks[3] = blk_12_1;
                            unsigned long long off_13 = (row_base64 + (unsigned long long)blk_12_1) * total_q64 + q_addr64;
                            xs[3] = __uint_as_float(2143289344);
                            if (act_11_1 != 0) {
                                xs[3] = max_score[off_13];
                            }
                            int e_14 = (i0_1 + 4) * 16 + r_sub;
                            int act_15_1 = (int)(e_14 < n_list) & qvalid;
                            int _min_27 = ((e_14) < (last) ? (e_14) : (last));
                            int blk_16_1 = (int)hist[1024 + _min_27];
                            blks[4] = blk_16_1;
                            unsigned long long off_17 = (row_base64 + (unsigned long long)blk_16_1) * total_q64 + q_addr64;
                            xs[4] = __uint_as_float(2143289344);
                            if (act_15_1 != 0) {
                                xs[4] = max_score[off_17];
                            }
                            int e_18 = (i0_1 + 5) * 16 + r_sub;
                            int act_19_1 = (int)(e_18 < n_list) & qvalid;
                            int _min_28 = ((e_18) < (last) ? (e_18) : (last));
                            int blk_20_1 = (int)hist[1024 + _min_28];
                            blks[5] = blk_20_1;
                            unsigned long long off_21 = (row_base64 + (unsigned long long)blk_20_1) * total_q64 + q_addr64;
                            xs[5] = __uint_as_float(2143289344);
                            if (act_19_1 != 0) {
                                xs[5] = max_score[off_21];
                            }
                            int e_22 = (i0_1 + 6) * 16 + r_sub;
                            int act_23_1 = (int)(e_22 < n_list) & qvalid;
                            int _min_29 = ((e_22) < (last) ? (e_22) : (last));
                            int blk_24_1 = (int)hist[1024 + _min_29];
                            blks[6] = blk_24_1;
                            unsigned long long off_25 = (row_base64 + (unsigned long long)blk_24_1) * total_q64 + q_addr64;
                            xs[6] = __uint_as_float(2143289344);
                            if (act_23_1 != 0) {
                                xs[6] = max_score[off_25];
                            }
                            int e_26 = (i0_1 + 7) * 16 + r_sub;
                            int act_27_1 = (int)(e_26 < n_list) & qvalid;
                            int _min_30 = ((e_26) < (last) ? (e_26) : (last));
                            int blk_28_1 = (int)hist[1024 + _min_30];
                            blks[7] = blk_28_1;
                            unsigned long long off_29 = (row_base64 + (unsigned long long)blk_28_1) * total_q64 + q_addr64;
                            xs[7] = __uint_as_float(2143289344);
                            if (act_27_1 != 0) {
                                xs[7] = max_score[off_29];
                            }
                            int e_30 = (i0_1 + 8) * 16 + r_sub;
                            int act_31 = (int)(e_30 < n_list) & qvalid;
                            int _min_31 = ((e_30) < (last) ? (e_30) : (last));
                            int blk_32 = (int)hist[1024 + _min_31];
                            blks[8] = blk_32;
                            unsigned long long off_33 = (row_base64 + (unsigned long long)blk_32) * total_q64 + q_addr64;
                            xs[8] = __uint_as_float(2143289344);
                            if (act_31 != 0) {
                                xs[8] = max_score[off_33];
                            }
                            int e_34 = (i0_1 + 9) * 16 + r_sub;
                            int act_35 = (int)(e_34 < n_list) & qvalid;
                            int _min_32 = ((e_34) < (last) ? (e_34) : (last));
                            int blk_36 = (int)hist[1024 + _min_32];
                            blks[9] = blk_36;
                            unsigned long long off_37 = (row_base64 + (unsigned long long)blk_36) * total_q64 + q_addr64;
                            xs[9] = __uint_as_float(2143289344);
                            if (act_35 != 0) {
                                xs[9] = max_score[off_37];
                            }
                            int e_38 = (i0_1 + 10) * 16 + r_sub;
                            int act_39 = (int)(e_38 < n_list) & qvalid;
                            int _min_33 = ((e_38) < (last) ? (e_38) : (last));
                            int blk_40 = (int)hist[1024 + _min_33];
                            blks[10] = blk_40;
                            unsigned long long off_41 = (row_base64 + (unsigned long long)blk_40) * total_q64 + q_addr64;
                            xs[10] = __uint_as_float(2143289344);
                            if (act_39 != 0) {
                                xs[10] = max_score[off_41];
                            }
                            int e_42 = (i0_1 + 11) * 16 + r_sub;
                            int act_43 = (int)(e_42 < n_list) & qvalid;
                            int _min_34 = ((e_42) < (last) ? (e_42) : (last));
                            int blk_44 = (int)hist[1024 + _min_34];
                            blks[11] = blk_44;
                            unsigned long long off_45 = (row_base64 + (unsigned long long)blk_44) * total_q64 + q_addr64;
                            xs[11] = __uint_as_float(2143289344);
                            if (act_43 != 0) {
                                xs[11] = max_score[off_45];
                            }
                            int e_46 = (i0_1 + 12) * 16 + r_sub;
                            int act_47 = (int)(e_46 < n_list) & qvalid;
                            int _min_35 = ((e_46) < (last) ? (e_46) : (last));
                            int blk_48 = (int)hist[1024 + _min_35];
                            blks[12] = blk_48;
                            unsigned long long off_49 = (row_base64 + (unsigned long long)blk_48) * total_q64 + q_addr64;
                            xs[12] = __uint_as_float(2143289344);
                            if (act_47 != 0) {
                                xs[12] = max_score[off_49];
                            }
                            int e_50 = (i0_1 + 13) * 16 + r_sub;
                            int act_51 = (int)(e_50 < n_list) & qvalid;
                            int _min_36 = ((e_50) < (last) ? (e_50) : (last));
                            int blk_52 = (int)hist[1024 + _min_36];
                            blks[13] = blk_52;
                            unsigned long long off_53 = (row_base64 + (unsigned long long)blk_52) * total_q64 + q_addr64;
                            xs[13] = __uint_as_float(2143289344);
                            if (act_51 != 0) {
                                xs[13] = max_score[off_53];
                            }
                            int e_54 = (i0_1 + 14) * 16 + r_sub;
                            int act_55 = (int)(e_54 < n_list) & qvalid;
                            int _min_37 = ((e_54) < (last) ? (e_54) : (last));
                            int blk_56 = (int)hist[1024 + _min_37];
                            blks[14] = blk_56;
                            unsigned long long off_57 = (row_base64 + (unsigned long long)blk_56) * total_q64 + q_addr64;
                            xs[14] = __uint_as_float(2143289344);
                            if (act_55 != 0) {
                                xs[14] = max_score[off_57];
                            }
                            int e_58 = (i0_1 + 15) * 16 + r_sub;
                            int act_59 = (int)(e_58 < n_list) & qvalid;
                            int _min_38 = ((e_58) < (last) ? (e_58) : (last));
                            int blk_60 = (int)hist[1024 + _min_38];
                            blks[15] = blk_60;
                            unsigned long long off_61 = (row_base64 + (unsigned long long)blk_60) * total_q64 + q_addr64;
                            xs[15] = __uint_as_float(2143289344);
                            if (act_59 != 0) {
                                xs[15] = max_score[off_61];
                            }
                            unsigned int raw_2 = __as_u32(xs[0]);
                            unsigned int mag_1 = raw_2 & 2147483647;
                            unsigned int nan_1 = (unsigned int)(mag_1 > 2139095040);
                            unsigned int bits_1 = ((mag_1 != 0) ? raw_2 : 0);
                            unsigned int neg_mask = -(bits_1 >> 31);
                            unsigned int key = bits_1 ^ (neg_mask | 2147483648u);
                            unsigned int inv = ~(unsigned int)blks[0];
                            unsigned long long comp = (unsigned long long)key << 32 | (unsigned long long)inv;
                            if (is_stage != 0 && nan_1 == 0 && comp >= cthr_me) {
                                unsigned int _atomic_old_0 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot = _atomic_old_0;
                                if (slot < 64) {
                                    cand[q_local * 64 + (int)slot] = comp;
                                }
                            }
                            unsigned int raw_62 = __as_u32(xs[1]);
                            unsigned int mag_63 = raw_62 & 2147483647;
                            unsigned int nan_64 = (unsigned int)(mag_63 > 2139095040);
                            unsigned int bits_65 = ((mag_63 != 0) ? raw_62 : 0);
                            unsigned int neg_mask_66 = -(bits_65 >> 31);
                            unsigned int key_67 = bits_65 ^ (neg_mask_66 | 2147483648u);
                            unsigned int inv_68 = ~(unsigned int)blks[1];
                            unsigned long long comp_69 = (unsigned long long)key_67 << 32 | (unsigned long long)inv_68;
                            if (is_stage != 0 && nan_64 == 0 && comp_69 >= cthr_me) {
                                unsigned int _atomic_old_1 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_1 = _atomic_old_1;
                                if (slot_1 < 64) {
                                    cand[q_local * 64 + (int)slot_1] = comp_69;
                                }
                            }
                            unsigned int raw_70 = __as_u32(xs[2]);
                            unsigned int mag_71 = raw_70 & 2147483647;
                            unsigned int nan_72 = (unsigned int)(mag_71 > 2139095040);
                            unsigned int bits_73 = ((mag_71 != 0) ? raw_70 : 0);
                            unsigned int neg_mask_74 = -(bits_73 >> 31);
                            unsigned int key_75 = bits_73 ^ (neg_mask_74 | 2147483648u);
                            unsigned int inv_76 = ~(unsigned int)blks[2];
                            unsigned long long comp_77 = (unsigned long long)key_75 << 32 | (unsigned long long)inv_76;
                            if (is_stage != 0 && nan_72 == 0 && comp_77 >= cthr_me) {
                                unsigned int _atomic_old_2 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_2 = _atomic_old_2;
                                if (slot_2 < 64) {
                                    cand[q_local * 64 + (int)slot_2] = comp_77;
                                }
                            }
                            unsigned int raw_78 = __as_u32(xs[3]);
                            unsigned int mag_79 = raw_78 & 2147483647;
                            unsigned int nan_80 = (unsigned int)(mag_79 > 2139095040);
                            unsigned int bits_81 = ((mag_79 != 0) ? raw_78 : 0);
                            unsigned int neg_mask_82 = -(bits_81 >> 31);
                            unsigned int key_83 = bits_81 ^ (neg_mask_82 | 2147483648u);
                            unsigned int inv_84 = ~(unsigned int)blks[3];
                            unsigned long long comp_85 = (unsigned long long)key_83 << 32 | (unsigned long long)inv_84;
                            if (is_stage != 0 && nan_80 == 0 && comp_85 >= cthr_me) {
                                unsigned int _atomic_old_3 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_3 = _atomic_old_3;
                                if (slot_3 < 64) {
                                    cand[q_local * 64 + (int)slot_3] = comp_85;
                                }
                            }
                            unsigned int raw_86 = __as_u32(xs[4]);
                            unsigned int mag_87 = raw_86 & 2147483647;
                            unsigned int nan_88 = (unsigned int)(mag_87 > 2139095040);
                            unsigned int bits_89 = ((mag_87 != 0) ? raw_86 : 0);
                            unsigned int neg_mask_90 = -(bits_89 >> 31);
                            unsigned int key_91 = bits_89 ^ (neg_mask_90 | 2147483648u);
                            unsigned int inv_92 = ~(unsigned int)blks[4];
                            unsigned long long comp_93 = (unsigned long long)key_91 << 32 | (unsigned long long)inv_92;
                            if (is_stage != 0 && nan_88 == 0 && comp_93 >= cthr_me) {
                                unsigned int _atomic_old_4 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_4 = _atomic_old_4;
                                if (slot_4 < 64) {
                                    cand[q_local * 64 + (int)slot_4] = comp_93;
                                }
                            }
                            unsigned int raw_94 = __as_u32(xs[5]);
                            unsigned int mag_95 = raw_94 & 2147483647;
                            unsigned int nan_96 = (unsigned int)(mag_95 > 2139095040);
                            unsigned int bits_97 = ((mag_95 != 0) ? raw_94 : 0);
                            unsigned int neg_mask_98 = -(bits_97 >> 31);
                            unsigned int key_99 = bits_97 ^ (neg_mask_98 | 2147483648u);
                            unsigned int inv_100 = ~(unsigned int)blks[5];
                            unsigned long long comp_101 = (unsigned long long)key_99 << 32 | (unsigned long long)inv_100;
                            if (is_stage != 0 && nan_96 == 0 && comp_101 >= cthr_me) {
                                unsigned int _atomic_old_5 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_5 = _atomic_old_5;
                                if (slot_5 < 64) {
                                    cand[q_local * 64 + (int)slot_5] = comp_101;
                                }
                            }
                            unsigned int raw_102 = __as_u32(xs[6]);
                            unsigned int mag_103 = raw_102 & 2147483647;
                            unsigned int nan_104 = (unsigned int)(mag_103 > 2139095040);
                            unsigned int bits_105 = ((mag_103 != 0) ? raw_102 : 0);
                            unsigned int neg_mask_106 = -(bits_105 >> 31);
                            unsigned int key_107 = bits_105 ^ (neg_mask_106 | 2147483648u);
                            unsigned int inv_108 = ~(unsigned int)blks[6];
                            unsigned long long comp_109 = (unsigned long long)key_107 << 32 | (unsigned long long)inv_108;
                            if (is_stage != 0 && nan_104 == 0 && comp_109 >= cthr_me) {
                                unsigned int _atomic_old_6 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_6 = _atomic_old_6;
                                if (slot_6 < 64) {
                                    cand[q_local * 64 + (int)slot_6] = comp_109;
                                }
                            }
                            unsigned int raw_110 = __as_u32(xs[7]);
                            unsigned int mag_111 = raw_110 & 2147483647;
                            unsigned int nan_112 = (unsigned int)(mag_111 > 2139095040);
                            unsigned int bits_113 = ((mag_111 != 0) ? raw_110 : 0);
                            unsigned int neg_mask_114 = -(bits_113 >> 31);
                            unsigned int key_115 = bits_113 ^ (neg_mask_114 | 2147483648u);
                            unsigned int inv_116 = ~(unsigned int)blks[7];
                            unsigned long long comp_117 = (unsigned long long)key_115 << 32 | (unsigned long long)inv_116;
                            if (is_stage != 0 && nan_112 == 0 && comp_117 >= cthr_me) {
                                unsigned int _atomic_old_7 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_7 = _atomic_old_7;
                                if (slot_7 < 64) {
                                    cand[q_local * 64 + (int)slot_7] = comp_117;
                                }
                            }
                            if (i1_1 < n_it) {
                                unsigned int raw_0 = __as_u32(xs[8]);
                                unsigned int mag_1_1 = raw_0 & 2147483647;
                                unsigned int nan_2 = (unsigned int)(mag_1_1 > 2139095040);
                                unsigned int bits_3 = ((mag_1_1 != 0) ? raw_0 : 0);
                                unsigned int neg_mask_4 = -(bits_3 >> 31);
                                unsigned int key_5 = bits_3 ^ (neg_mask_4 | 2147483648u);
                                unsigned int inv_6 = ~(unsigned int)blks[8];
                                unsigned long long comp_7 = (unsigned long long)key_5 << 32 | (unsigned long long)inv_6;
                                if (is_stage != 0 && nan_2 == 0 && comp_7 >= cthr_me) {
                                    unsigned int _atomic_old_8 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_8 = _atomic_old_8;
                                    if (slot_8 < 64) {
                                        cand[q_local * 64 + (int)slot_8] = comp_7;
                                    }
                                }
                                unsigned int raw_8 = __as_u32(xs[9]);
                                unsigned int mag_9 = raw_8 & 2147483647;
                                unsigned int nan_10 = (unsigned int)(mag_9 > 2139095040);
                                unsigned int bits_11 = ((mag_9 != 0) ? raw_8 : 0);
                                unsigned int neg_mask_12 = -(bits_11 >> 31);
                                unsigned int key_13 = bits_11 ^ (neg_mask_12 | 2147483648u);
                                unsigned int inv_14 = ~(unsigned int)blks[9];
                                unsigned long long comp_15 = (unsigned long long)key_13 << 32 | (unsigned long long)inv_14;
                                if (is_stage != 0 && nan_10 == 0 && comp_15 >= cthr_me) {
                                    unsigned int _atomic_old_9 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_9 = _atomic_old_9;
                                    if (slot_9 < 64) {
                                        cand[q_local * 64 + (int)slot_9] = comp_15;
                                    }
                                }
                                unsigned int raw_16 = __as_u32(xs[10]);
                                unsigned int mag_17 = raw_16 & 2147483647;
                                unsigned int nan_18 = (unsigned int)(mag_17 > 2139095040);
                                unsigned int bits_19 = ((mag_17 != 0) ? raw_16 : 0);
                                unsigned int neg_mask_20 = -(bits_19 >> 31);
                                unsigned int key_21 = bits_19 ^ (neg_mask_20 | 2147483648u);
                                unsigned int inv_22 = ~(unsigned int)blks[10];
                                unsigned long long comp_23 = (unsigned long long)key_21 << 32 | (unsigned long long)inv_22;
                                if (is_stage != 0 && nan_18 == 0 && comp_23 >= cthr_me) {
                                    unsigned int _atomic_old_10 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_10 = _atomic_old_10;
                                    if (slot_10 < 64) {
                                        cand[q_local * 64 + (int)slot_10] = comp_23;
                                    }
                                }
                                unsigned int raw_24_1 = __as_u32(xs[11]);
                                unsigned int mag_25 = raw_24_1 & 2147483647;
                                unsigned int nan_26 = (unsigned int)(mag_25 > 2139095040);
                                unsigned int bits_27 = ((mag_25 != 0) ? raw_24_1 : 0);
                                unsigned int neg_mask_28 = -(bits_27 >> 31);
                                unsigned int key_29 = bits_27 ^ (neg_mask_28 | 2147483648u);
                                unsigned int inv_30 = ~(unsigned int)blks[11];
                                unsigned long long comp_31 = (unsigned long long)key_29 << 32 | (unsigned long long)inv_30;
                                if (is_stage != 0 && nan_26 == 0 && comp_31 >= cthr_me) {
                                    unsigned int _atomic_old_11 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_11 = _atomic_old_11;
                                    if (slot_11 < 64) {
                                        cand[q_local * 64 + (int)slot_11] = comp_31;
                                    }
                                }
                                unsigned int raw_32 = __as_u32(xs[12]);
                                unsigned int mag_33 = raw_32 & 2147483647;
                                unsigned int nan_34 = (unsigned int)(mag_33 > 2139095040);
                                unsigned int bits_35 = ((mag_33 != 0) ? raw_32 : 0);
                                unsigned int neg_mask_36 = -(bits_35 >> 31);
                                unsigned int key_37 = bits_35 ^ (neg_mask_36 | 2147483648u);
                                unsigned int inv_38 = ~(unsigned int)blks[12];
                                unsigned long long comp_39 = (unsigned long long)key_37 << 32 | (unsigned long long)inv_38;
                                if (is_stage != 0 && nan_34 == 0 && comp_39 >= cthr_me) {
                                    unsigned int _atomic_old_12 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_12 = _atomic_old_12;
                                    if (slot_12 < 64) {
                                        cand[q_local * 64 + (int)slot_12] = comp_39;
                                    }
                                }
                                unsigned int raw_40_1 = __as_u32(xs[13]);
                                unsigned int mag_41_1 = raw_40_1 & 2147483647;
                                unsigned int nan_42_1 = (unsigned int)(mag_41_1 > 2139095040);
                                unsigned int bits_43_1 = ((mag_41_1 != 0) ? raw_40_1 : 0);
                                unsigned int neg_mask_44 = -(bits_43_1 >> 31);
                                unsigned int key_45 = bits_43_1 ^ (neg_mask_44 | 2147483648u);
                                unsigned int inv_46 = ~(unsigned int)blks[13];
                                unsigned long long comp_47 = (unsigned long long)key_45 << 32 | (unsigned long long)inv_46;
                                if (is_stage != 0 && nan_42_1 == 0 && comp_47 >= cthr_me) {
                                    unsigned int _atomic_old_13 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_13 = _atomic_old_13;
                                    if (slot_13 < 64) {
                                        cand[q_local * 64 + (int)slot_13] = comp_47;
                                    }
                                }
                                unsigned int raw_48 = __as_u32(xs[14]);
                                unsigned int mag_49 = raw_48 & 2147483647;
                                unsigned int nan_50 = (unsigned int)(mag_49 > 2139095040);
                                unsigned int bits_51 = ((mag_49 != 0) ? raw_48 : 0);
                                unsigned int neg_mask_52 = -(bits_51 >> 31);
                                unsigned int key_53 = bits_51 ^ (neg_mask_52 | 2147483648u);
                                unsigned int inv_54 = ~(unsigned int)blks[14];
                                unsigned long long comp_55 = (unsigned long long)key_53 << 32 | (unsigned long long)inv_54;
                                if (is_stage != 0 && nan_50 == 0 && comp_55 >= cthr_me) {
                                    unsigned int _atomic_old_14 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_14 = _atomic_old_14;
                                    if (slot_14 < 64) {
                                        cand[q_local * 64 + (int)slot_14] = comp_55;
                                    }
                                }
                                unsigned int raw_56 = __as_u32(xs[15]);
                                unsigned int mag_57 = raw_56 & 2147483647;
                                unsigned int nan_58 = (unsigned int)(mag_57 > 2139095040);
                                unsigned int bits_59 = ((mag_57 != 0) ? raw_56 : 0);
                                unsigned int neg_mask_60 = -(bits_59 >> 31);
                                unsigned int key_61 = bits_59 ^ (neg_mask_60 | 2147483648u);
                                unsigned int inv_62 = ~(unsigned int)blks[15];
                                unsigned long long comp_63 = (unsigned long long)key_61 << 32 | (unsigned long long)inv_62;
                                if (is_stage != 0 && nan_58 == 0 && comp_63 >= cthr_me) {
                                    unsigned int _atomic_old_15 = atomicAdd(&cand_cnt[q_local], 1);
                                    unsigned int slot_15 = _atomic_old_15;
                                    if (slot_15 < 64) {
                                        cand[q_local * 64 + (int)slot_15] = comp_63;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll 1
                    for (int i0_2 = 0; i0_2 < iters; i0_2 += 16) {
                        int i1_2 = i0_2 + 8;
                        unsigned long long off_2 = off0 + (unsigned long long)i0_2 * stride16;
                        int blk_3 = lo_s + i0_2 * 16 + r_sub;
                        blks[0] = blk_3;
                        int act_4 = (int)(blk_3 < hi_s) & qvalid;
                        xs[0] = __uint_as_float(2143289344);
                        if (act_4 != 0) {
                            xs[0] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_0_1 = lo_s + (i0_2 + 1) * 16 + r_sub;
                        blks[1] = blk_0_1;
                        int act_1_1 = (int)(blk_0_1 < hi_s) & qvalid;
                        xs[1] = __uint_as_float(2143289344);
                        if (act_1_1 != 0) {
                            xs[1] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_2_1 = lo_s + (i0_2 + 2) * 16 + r_sub;
                        blks[2] = blk_2_1;
                        int act_3_2 = (int)(blk_2_1 < hi_s) & qvalid;
                        xs[2] = __uint_as_float(2143289344);
                        if (act_3_2 != 0) {
                            xs[2] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_4_2 = lo_s + (i0_2 + 3) * 16 + r_sub;
                        blks[3] = blk_4_2;
                        int act_5_1 = (int)(blk_4_2 < hi_s) & qvalid;
                        xs[3] = __uint_as_float(2143289344);
                        if (act_5_1 != 0) {
                            xs[3] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_6_1 = lo_s + (i0_2 + 4) * 16 + r_sub;
                        blks[4] = blk_6_1;
                        int act_7_2 = (int)(blk_6_1 < hi_s) & qvalid;
                        xs[4] = __uint_as_float(2143289344);
                        if (act_7_2 != 0) {
                            xs[4] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_8_2 = lo_s + (i0_2 + 5) * 16 + r_sub;
                        blks[5] = blk_8_2;
                        int act_9_1 = (int)(blk_8_2 < hi_s) & qvalid;
                        xs[5] = __uint_as_float(2143289344);
                        if (act_9_1 != 0) {
                            xs[5] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_10_1 = lo_s + (i0_2 + 6) * 16 + r_sub;
                        blks[6] = blk_10_1;
                        int act_11_2 = (int)(blk_10_1 < hi_s) & qvalid;
                        xs[6] = __uint_as_float(2143289344);
                        if (act_11_2 != 0) {
                            xs[6] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_12_2 = lo_s + (i0_2 + 7) * 16 + r_sub;
                        blks[7] = blk_12_2;
                        int act_13_1 = (int)(blk_12_2 < hi_s) & qvalid;
                        xs[7] = __uint_as_float(2143289344);
                        if (act_13_1 != 0) {
                            xs[7] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_14_1 = lo_s + (i0_2 + 8) * 16 + r_sub;
                        blks[8] = blk_14_1;
                        int act_15_2 = (int)(blk_14_1 < hi_s) & qvalid;
                        xs[8] = __uint_as_float(2143289344);
                        if (act_15_2 != 0) {
                            xs[8] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_16_2 = lo_s + (i0_2 + 9) * 16 + r_sub;
                        blks[9] = blk_16_2;
                        int act_17_1 = (int)(blk_16_2 < hi_s) & qvalid;
                        xs[9] = __uint_as_float(2143289344);
                        if (act_17_1 != 0) {
                            xs[9] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_18_1 = lo_s + (i0_2 + 10) * 16 + r_sub;
                        blks[10] = blk_18_1;
                        int act_19_2 = (int)(blk_18_1 < hi_s) & qvalid;
                        xs[10] = __uint_as_float(2143289344);
                        if (act_19_2 != 0) {
                            xs[10] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_20_2 = lo_s + (i0_2 + 11) * 16 + r_sub;
                        blks[11] = blk_20_2;
                        int act_21_1 = (int)(blk_20_2 < hi_s) & qvalid;
                        xs[11] = __uint_as_float(2143289344);
                        if (act_21_1 != 0) {
                            xs[11] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_22_1 = lo_s + (i0_2 + 12) * 16 + r_sub;
                        blks[12] = blk_22_1;
                        int act_23_2 = (int)(blk_22_1 < hi_s) & qvalid;
                        xs[12] = __uint_as_float(2143289344);
                        if (act_23_2 != 0) {
                            xs[12] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_24_2 = lo_s + (i0_2 + 13) * 16 + r_sub;
                        blks[13] = blk_24_2;
                        int act_25_1 = (int)(blk_24_2 < hi_s) & qvalid;
                        xs[13] = __uint_as_float(2143289344);
                        if (act_25_1 != 0) {
                            xs[13] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_26_1 = lo_s + (i0_2 + 14) * 16 + r_sub;
                        blks[14] = blk_26_1;
                        int act_27_2 = (int)(blk_26_1 < hi_s) & qvalid;
                        xs[14] = __uint_as_float(2143289344);
                        if (act_27_2 != 0) {
                            xs[14] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        int blk_28_2 = lo_s + (i0_2 + 15) * 16 + r_sub;
                        blks[15] = blk_28_2;
                        int act_29_1 = (int)(blk_28_2 < hi_s) & qvalid;
                        xs[15] = __uint_as_float(2143289344);
                        if (act_29_1 != 0) {
                            xs[15] = max_score[off_2];
                        }
                        off_2 = off_2 + stride16;
                        unsigned int raw_3 = __as_u32(xs[0]);
                        unsigned int mag_3 = raw_3 & 2147483647;
                        unsigned int nan_4 = (unsigned int)(mag_3 > 2139095040);
                        unsigned int bits_2 = ((mag_3 != 0) ? raw_3 : 0);
                        unsigned int neg_mask_1 = -(bits_2 >> 31);
                        unsigned int key_1 = bits_2 ^ (neg_mask_1 | 2147483648u);
                        unsigned int inv_1 = ~(unsigned int)blks[0];
                        unsigned long long comp_1 = (unsigned long long)key_1 << 32 | (unsigned long long)inv_1;
                        if (is_stage != 0 && nan_4 == 0 && comp_1 >= cthr_me) {
                            unsigned int _atomic_old_16 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_16 = _atomic_old_16;
                            if (slot_16 < 64) {
                                cand[q_local * 64 + (int)slot_16] = comp_1;
                            }
                        }
                        unsigned int raw_30_1 = __as_u32(xs[1]);
                        unsigned int mag_31 = raw_30_1 & 2147483647;
                        unsigned int nan_32 = (unsigned int)(mag_31 > 2139095040);
                        unsigned int bits_33 = ((mag_31 != 0) ? raw_30_1 : 0);
                        unsigned int neg_mask_34 = -(bits_33 >> 31);
                        unsigned int key_35 = bits_33 ^ (neg_mask_34 | 2147483648u);
                        unsigned int inv_36 = ~(unsigned int)blks[1];
                        unsigned long long comp_37 = (unsigned long long)key_35 << 32 | (unsigned long long)inv_36;
                        if (is_stage != 0 && nan_32 == 0 && comp_37 >= cthr_me) {
                            unsigned int _atomic_old_17 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_17 = _atomic_old_17;
                            if (slot_17 < 64) {
                                cand[q_local * 64 + (int)slot_17] = comp_37;
                            }
                        }
                        unsigned int raw_38 = __as_u32(xs[2]);
                        unsigned int mag_39 = raw_38 & 2147483647;
                        unsigned int nan_40 = (unsigned int)(mag_39 > 2139095040);
                        unsigned int bits_41 = ((mag_39 != 0) ? raw_38 : 0);
                        unsigned int neg_mask_42 = -(bits_41 >> 31);
                        unsigned int key_43 = bits_41 ^ (neg_mask_42 | 2147483648u);
                        unsigned int inv_44 = ~(unsigned int)blks[2];
                        unsigned long long comp_45 = (unsigned long long)key_43 << 32 | (unsigned long long)inv_44;
                        if (is_stage != 0 && nan_40 == 0 && comp_45 >= cthr_me) {
                            unsigned int _atomic_old_18 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_18 = _atomic_old_18;
                            if (slot_18 < 64) {
                                cand[q_local * 64 + (int)slot_18] = comp_45;
                            }
                        }
                        unsigned int raw_46_1 = __as_u32(xs[3]);
                        unsigned int mag_47_1 = raw_46_1 & 2147483647;
                        unsigned int nan_48_1 = (unsigned int)(mag_47_1 > 2139095040);
                        unsigned int bits_49_1 = ((mag_47_1 != 0) ? raw_46_1 : 0);
                        unsigned int neg_mask_50 = -(bits_49_1 >> 31);
                        unsigned int key_51 = bits_49_1 ^ (neg_mask_50 | 2147483648u);
                        unsigned int inv_52 = ~(unsigned int)blks[3];
                        unsigned long long comp_53 = (unsigned long long)key_51 << 32 | (unsigned long long)inv_52;
                        if (is_stage != 0 && nan_48_1 == 0 && comp_53 >= cthr_me) {
                            unsigned int _atomic_old_19 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_19 = _atomic_old_19;
                            if (slot_19 < 64) {
                                cand[q_local * 64 + (int)slot_19] = comp_53;
                            }
                        }
                        unsigned int raw_54_1 = __as_u32(xs[4]);
                        unsigned int mag_55 = raw_54_1 & 2147483647;
                        unsigned int nan_56 = (unsigned int)(mag_55 > 2139095040);
                        unsigned int bits_57 = ((mag_55 != 0) ? raw_54_1 : 0);
                        unsigned int neg_mask_58 = -(bits_57 >> 31);
                        unsigned int key_59 = bits_57 ^ (neg_mask_58 | 2147483648u);
                        unsigned int inv_60 = ~(unsigned int)blks[4];
                        unsigned long long comp_61 = (unsigned long long)key_59 << 32 | (unsigned long long)inv_60;
                        if (is_stage != 0 && nan_56 == 0 && comp_61 >= cthr_me) {
                            unsigned int _atomic_old_20 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_20 = _atomic_old_20;
                            if (slot_20 < 64) {
                                cand[q_local * 64 + (int)slot_20] = comp_61;
                            }
                        }
                        unsigned int raw_62_1 = __as_u32(xs[5]);
                        unsigned int mag_63_1 = raw_62_1 & 2147483647;
                        unsigned int nan_64_1 = (unsigned int)(mag_63_1 > 2139095040);
                        unsigned int bits_65_1 = ((mag_63_1 != 0) ? raw_62_1 : 0);
                        unsigned int neg_mask_66_1 = -(bits_65_1 >> 31);
                        unsigned int key_67_1 = bits_65_1 ^ (neg_mask_66_1 | 2147483648u);
                        unsigned int inv_68_1 = ~(unsigned int)blks[5];
                        unsigned long long comp_69_1 = (unsigned long long)key_67_1 << 32 | (unsigned long long)inv_68_1;
                        if (is_stage != 0 && nan_64_1 == 0 && comp_69_1 >= cthr_me) {
                            unsigned int _atomic_old_21 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_21 = _atomic_old_21;
                            if (slot_21 < 64) {
                                cand[q_local * 64 + (int)slot_21] = comp_69_1;
                            }
                        }
                        unsigned int raw_70_1 = __as_u32(xs[6]);
                        unsigned int mag_71_1 = raw_70_1 & 2147483647;
                        unsigned int nan_72_1 = (unsigned int)(mag_71_1 > 2139095040);
                        unsigned int bits_73_1 = ((mag_71_1 != 0) ? raw_70_1 : 0);
                        unsigned int neg_mask_74_1 = -(bits_73_1 >> 31);
                        unsigned int key_75_1 = bits_73_1 ^ (neg_mask_74_1 | 2147483648u);
                        unsigned int inv_76_1 = ~(unsigned int)blks[6];
                        unsigned long long comp_77_1 = (unsigned long long)key_75_1 << 32 | (unsigned long long)inv_76_1;
                        if (is_stage != 0 && nan_72_1 == 0 && comp_77_1 >= cthr_me) {
                            unsigned int _atomic_old_22 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_22 = _atomic_old_22;
                            if (slot_22 < 64) {
                                cand[q_local * 64 + (int)slot_22] = comp_77_1;
                            }
                        }
                        unsigned int raw_78_1 = __as_u32(xs[7]);
                        unsigned int mag_79_1 = raw_78_1 & 2147483647;
                        unsigned int nan_80_1 = (unsigned int)(mag_79_1 > 2139095040);
                        unsigned int bits_81_1 = ((mag_79_1 != 0) ? raw_78_1 : 0);
                        unsigned int neg_mask_82_1 = -(bits_81_1 >> 31);
                        unsigned int key_83_1 = bits_81_1 ^ (neg_mask_82_1 | 2147483648u);
                        unsigned int inv_84_1 = ~(unsigned int)blks[7];
                        unsigned long long comp_85_1 = (unsigned long long)key_83_1 << 32 | (unsigned long long)inv_84_1;
                        if (is_stage != 0 && nan_80_1 == 0 && comp_85_1 >= cthr_me) {
                            unsigned int _atomic_old_23 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_23 = _atomic_old_23;
                            if (slot_23 < 64) {
                                cand[q_local * 64 + (int)slot_23] = comp_85_1;
                            }
                        }
                        if (i1_2 < iters) {
                            unsigned int raw_0_1 = __as_u32(xs[8]);
                            unsigned int mag_1_2 = raw_0_1 & 2147483647;
                            unsigned int nan_2_1 = (unsigned int)(mag_1_2 > 2139095040);
                            unsigned int bits_3_1 = ((mag_1_2 != 0) ? raw_0_1 : 0);
                            unsigned int neg_mask_4_1 = -(bits_3_1 >> 31);
                            unsigned int key_5_1 = bits_3_1 ^ (neg_mask_4_1 | 2147483648u);
                            unsigned int inv_6_1 = ~(unsigned int)blks[8];
                            unsigned long long comp_7_1 = (unsigned long long)key_5_1 << 32 | (unsigned long long)inv_6_1;
                            if (is_stage != 0 && nan_2_1 == 0 && comp_7_1 >= cthr_me) {
                                unsigned int _atomic_old_24 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_24 = _atomic_old_24;
                                if (slot_24 < 64) {
                                    cand[q_local * 64 + (int)slot_24] = comp_7_1;
                                }
                            }
                            unsigned int raw_8_1 = __as_u32(xs[9]);
                            unsigned int mag_9_1 = raw_8_1 & 2147483647;
                            unsigned int nan_10_1 = (unsigned int)(mag_9_1 > 2139095040);
                            unsigned int bits_11_1 = ((mag_9_1 != 0) ? raw_8_1 : 0);
                            unsigned int neg_mask_12_1 = -(bits_11_1 >> 31);
                            unsigned int key_13_1 = bits_11_1 ^ (neg_mask_12_1 | 2147483648u);
                            unsigned int inv_14_1 = ~(unsigned int)blks[9];
                            unsigned long long comp_15_1 = (unsigned long long)key_13_1 << 32 | (unsigned long long)inv_14_1;
                            if (is_stage != 0 && nan_10_1 == 0 && comp_15_1 >= cthr_me) {
                                unsigned int _atomic_old_25 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_25 = _atomic_old_25;
                                if (slot_25 < 64) {
                                    cand[q_local * 64 + (int)slot_25] = comp_15_1;
                                }
                            }
                            unsigned int raw_16_1 = __as_u32(xs[10]);
                            unsigned int mag_17_1 = raw_16_1 & 2147483647;
                            unsigned int nan_18_1 = (unsigned int)(mag_17_1 > 2139095040);
                            unsigned int bits_19_1 = ((mag_17_1 != 0) ? raw_16_1 : 0);
                            unsigned int neg_mask_20_1 = -(bits_19_1 >> 31);
                            unsigned int key_21_1 = bits_19_1 ^ (neg_mask_20_1 | 2147483648u);
                            unsigned int inv_22_1 = ~(unsigned int)blks[10];
                            unsigned long long comp_23_1 = (unsigned long long)key_21_1 << 32 | (unsigned long long)inv_22_1;
                            if (is_stage != 0 && nan_18_1 == 0 && comp_23_1 >= cthr_me) {
                                unsigned int _atomic_old_26 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_26 = _atomic_old_26;
                                if (slot_26 < 64) {
                                    cand[q_local * 64 + (int)slot_26] = comp_23_1;
                                }
                            }
                            unsigned int raw_24_2 = __as_u32(xs[11]);
                            unsigned int mag_25_1 = raw_24_2 & 2147483647;
                            unsigned int nan_26_1 = (unsigned int)(mag_25_1 > 2139095040);
                            unsigned int bits_27_1 = ((mag_25_1 != 0) ? raw_24_2 : 0);
                            unsigned int neg_mask_28_1 = -(bits_27_1 >> 31);
                            unsigned int key_29_1 = bits_27_1 ^ (neg_mask_28_1 | 2147483648u);
                            unsigned int inv_30_1 = ~(unsigned int)blks[11];
                            unsigned long long comp_31_1 = (unsigned long long)key_29_1 << 32 | (unsigned long long)inv_30_1;
                            if (is_stage != 0 && nan_26_1 == 0 && comp_31_1 >= cthr_me) {
                                unsigned int _atomic_old_27 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_27 = _atomic_old_27;
                                if (slot_27 < 64) {
                                    cand[q_local * 64 + (int)slot_27] = comp_31_1;
                                }
                            }
                            unsigned int raw_32_1 = __as_u32(xs[12]);
                            unsigned int mag_33_1 = raw_32_1 & 2147483647;
                            unsigned int nan_34_1 = (unsigned int)(mag_33_1 > 2139095040);
                            unsigned int bits_35_1 = ((mag_33_1 != 0) ? raw_32_1 : 0);
                            unsigned int neg_mask_36_1 = -(bits_35_1 >> 31);
                            unsigned int key_37_1 = bits_35_1 ^ (neg_mask_36_1 | 2147483648u);
                            unsigned int inv_38_1 = ~(unsigned int)blks[12];
                            unsigned long long comp_39_1 = (unsigned long long)key_37_1 << 32 | (unsigned long long)inv_38_1;
                            if (is_stage != 0 && nan_34_1 == 0 && comp_39_1 >= cthr_me) {
                                unsigned int _atomic_old_28 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_28 = _atomic_old_28;
                                if (slot_28 < 64) {
                                    cand[q_local * 64 + (int)slot_28] = comp_39_1;
                                }
                            }
                            unsigned int raw_40_2 = __as_u32(xs[13]);
                            unsigned int mag_41_2 = raw_40_2 & 2147483647;
                            unsigned int nan_42_2 = (unsigned int)(mag_41_2 > 2139095040);
                            unsigned int bits_43_2 = ((mag_41_2 != 0) ? raw_40_2 : 0);
                            unsigned int neg_mask_44_1 = -(bits_43_2 >> 31);
                            unsigned int key_45_1 = bits_43_2 ^ (neg_mask_44_1 | 2147483648u);
                            unsigned int inv_46_1 = ~(unsigned int)blks[13];
                            unsigned long long comp_47_1 = (unsigned long long)key_45_1 << 32 | (unsigned long long)inv_46_1;
                            if (is_stage != 0 && nan_42_2 == 0 && comp_47_1 >= cthr_me) {
                                unsigned int _atomic_old_29 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_29 = _atomic_old_29;
                                if (slot_29 < 64) {
                                    cand[q_local * 64 + (int)slot_29] = comp_47_1;
                                }
                            }
                            unsigned int raw_48_1 = __as_u32(xs[14]);
                            unsigned int mag_49_1 = raw_48_1 & 2147483647;
                            unsigned int nan_50_1 = (unsigned int)(mag_49_1 > 2139095040);
                            unsigned int bits_51_1 = ((mag_49_1 != 0) ? raw_48_1 : 0);
                            unsigned int neg_mask_52_1 = -(bits_51_1 >> 31);
                            unsigned int key_53_1 = bits_51_1 ^ (neg_mask_52_1 | 2147483648u);
                            unsigned int inv_54_1 = ~(unsigned int)blks[14];
                            unsigned long long comp_55_1 = (unsigned long long)key_53_1 << 32 | (unsigned long long)inv_54_1;
                            if (is_stage != 0 && nan_50_1 == 0 && comp_55_1 >= cthr_me) {
                                unsigned int _atomic_old_30 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_30 = _atomic_old_30;
                                if (slot_30 < 64) {
                                    cand[q_local * 64 + (int)slot_30] = comp_55_1;
                                }
                            }
                            unsigned int raw_56_1 = __as_u32(xs[15]);
                            unsigned int mag_57_1 = raw_56_1 & 2147483647;
                            unsigned int nan_58_1 = (unsigned int)(mag_57_1 > 2139095040);
                            unsigned int bits_59_1 = ((mag_57_1 != 0) ? raw_56_1 : 0);
                            unsigned int neg_mask_60_1 = -(bits_59_1 >> 31);
                            unsigned int key_61_1 = bits_59_1 ^ (neg_mask_60_1 | 2147483648u);
                            unsigned int inv_62_1 = ~(unsigned int)blks[15];
                            unsigned long long comp_63_1 = (unsigned long long)key_61_1 << 32 | (unsigned long long)inv_62_1;
                            if (is_stage != 0 && nan_58_1 == 0 && comp_63_1 >= cthr_me) {
                                unsigned int _atomic_old_31 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_31 = _atomic_old_31;
                                if (slot_31 < 64) {
                                    cand[q_local * 64 + (int)slot_31] = comp_63_1;
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int i0_3 = 0; i0_3 < iters; i0_3 += 16) {
                    int i1_3 = i0_3 + 8;
                    unsigned long long off_3 = off0 + (unsigned long long)i0_3 * stride16;
                    int blk_5 = lo_s + i0_3 * 16 + r_sub;
                    blks[0] = blk_5;
                    int act_6 = (int)(blk_5 < hi_s) & qvalid;
                    xs[0] = __uint_as_float(2143289344);
                    if (act_6 != 0) {
                        xs[0] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_0_2 = lo_s + (i0_3 + 1) * 16 + r_sub;
                    blks[1] = blk_0_2;
                    int act_1_2 = (int)(blk_0_2 < hi_s) & qvalid;
                    xs[1] = __uint_as_float(2143289344);
                    if (act_1_2 != 0) {
                        xs[1] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_2_2 = lo_s + (i0_3 + 2) * 16 + r_sub;
                    blks[2] = blk_2_2;
                    int act_3_3 = (int)(blk_2_2 < hi_s) & qvalid;
                    xs[2] = __uint_as_float(2143289344);
                    if (act_3_3 != 0) {
                        xs[2] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_4_3 = lo_s + (i0_3 + 3) * 16 + r_sub;
                    blks[3] = blk_4_3;
                    int act_5_2 = (int)(blk_4_3 < hi_s) & qvalid;
                    xs[3] = __uint_as_float(2143289344);
                    if (act_5_2 != 0) {
                        xs[3] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_6_2 = lo_s + (i0_3 + 4) * 16 + r_sub;
                    blks[4] = blk_6_2;
                    int act_7_3 = (int)(blk_6_2 < hi_s) & qvalid;
                    xs[4] = __uint_as_float(2143289344);
                    if (act_7_3 != 0) {
                        xs[4] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_8_3 = lo_s + (i0_3 + 5) * 16 + r_sub;
                    blks[5] = blk_8_3;
                    int act_9_2 = (int)(blk_8_3 < hi_s) & qvalid;
                    xs[5] = __uint_as_float(2143289344);
                    if (act_9_2 != 0) {
                        xs[5] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_10_2 = lo_s + (i0_3 + 6) * 16 + r_sub;
                    blks[6] = blk_10_2;
                    int act_11_3 = (int)(blk_10_2 < hi_s) & qvalid;
                    xs[6] = __uint_as_float(2143289344);
                    if (act_11_3 != 0) {
                        xs[6] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_12_3 = lo_s + (i0_3 + 7) * 16 + r_sub;
                    blks[7] = blk_12_3;
                    int act_13_2 = (int)(blk_12_3 < hi_s) & qvalid;
                    xs[7] = __uint_as_float(2143289344);
                    if (act_13_2 != 0) {
                        xs[7] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_14_2 = lo_s + (i0_3 + 8) * 16 + r_sub;
                    blks[8] = blk_14_2;
                    int act_15_3 = (int)(blk_14_2 < hi_s) & qvalid;
                    xs[8] = __uint_as_float(2143289344);
                    if (act_15_3 != 0) {
                        xs[8] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_16_3 = lo_s + (i0_3 + 9) * 16 + r_sub;
                    blks[9] = blk_16_3;
                    int act_17_2 = (int)(blk_16_3 < hi_s) & qvalid;
                    xs[9] = __uint_as_float(2143289344);
                    if (act_17_2 != 0) {
                        xs[9] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_18_2 = lo_s + (i0_3 + 10) * 16 + r_sub;
                    blks[10] = blk_18_2;
                    int act_19_3 = (int)(blk_18_2 < hi_s) & qvalid;
                    xs[10] = __uint_as_float(2143289344);
                    if (act_19_3 != 0) {
                        xs[10] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_20_3 = lo_s + (i0_3 + 11) * 16 + r_sub;
                    blks[11] = blk_20_3;
                    int act_21_2 = (int)(blk_20_3 < hi_s) & qvalid;
                    xs[11] = __uint_as_float(2143289344);
                    if (act_21_2 != 0) {
                        xs[11] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_22_2 = lo_s + (i0_3 + 12) * 16 + r_sub;
                    blks[12] = blk_22_2;
                    int act_23_3 = (int)(blk_22_2 < hi_s) & qvalid;
                    xs[12] = __uint_as_float(2143289344);
                    if (act_23_3 != 0) {
                        xs[12] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_24_3 = lo_s + (i0_3 + 13) * 16 + r_sub;
                    blks[13] = blk_24_3;
                    int act_25_2 = (int)(blk_24_3 < hi_s) & qvalid;
                    xs[13] = __uint_as_float(2143289344);
                    if (act_25_2 != 0) {
                        xs[13] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_26_2 = lo_s + (i0_3 + 14) * 16 + r_sub;
                    blks[14] = blk_26_2;
                    int act_27_3 = (int)(blk_26_2 < hi_s) & qvalid;
                    xs[14] = __uint_as_float(2143289344);
                    if (act_27_3 != 0) {
                        xs[14] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    int blk_28_3 = lo_s + (i0_3 + 15) * 16 + r_sub;
                    blks[15] = blk_28_3;
                    int act_29_2 = (int)(blk_28_3 < hi_s) & qvalid;
                    xs[15] = __uint_as_float(2143289344);
                    if (act_29_2 != 0) {
                        xs[15] = max_score[off_3];
                    }
                    off_3 = off_3 + stride16;
                    unsigned int raw_4 = __as_u32(xs[0]);
                    unsigned int mag_4 = raw_4 & 2147483647;
                    unsigned int nan_5 = (unsigned int)(mag_4 > 2139095040);
                    unsigned int bits_5 = ((mag_4 != 0) ? raw_4 : 0);
                    int em_1 = (int)(bits_5 >> 19 & 4095);
                    int _max_275 = ((em_1 - 1792) > (0) ? (em_1 - 1792) : (0));
                    int _min_39 = ((_max_275) < (511) ? (_max_275) : (511));
                    int raw_30_2 = _min_39;
                    int neg_1 = (int)(bits_5 >> 31);
                    int h10_1 = ((neg_1 != 0) ? 511 - raw_30_2 : 512 + raw_30_2);
                    unsigned int neg_mask_2 = -(bits_5 >> 31);
                    unsigned int key_2 = bits_5 ^ (neg_mask_2 | 2147483648u);
                    int blk_31 = blks[0];
                    unsigned int invm = ~(unsigned int)blk_31 & idx_mask;
                    unsigned long long rk = (unsigned long long)h10_1 << 54 | (unsigned long long)key_2 << 22 | (unsigned long long)invm;
                    int in_window = is_pass0 | (int)(rk >> rsh_m == rprefix_me);
                    int d_2 = (int)((unsigned int)(rk >> dsh_me) & dmask_me);
                    int counted = is_hist & (int)(nan_5 == 0) & in_window;
                    int dd = ((counted != 0) ? d_2 : 1024);
                    atomicAdd(&hist_ext[dd * 16 + q_local], 1);
                    unsigned int inv_2 = ~(unsigned int)blk_31;
                    unsigned long long comp_2 = (unsigned long long)key_2 << 32 | (unsigned long long)inv_2;
                    if (is_stage != 0 && nan_5 == 0 && comp_2 >= cthr_me) {
                        unsigned int _atomic_old_32 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_32 = _atomic_old_32;
                        if (slot_32 < 64) {
                            cand[q_local * 64 + (int)slot_32] = comp_2;
                        }
                    }
                    unsigned int raw_32_2 = __as_u32(xs[1]);
                    unsigned int mag_33_2 = raw_32_2 & 2147483647;
                    unsigned int nan_34_2 = (unsigned int)(mag_33_2 > 2139095040);
                    unsigned int bits_35_2 = ((mag_33_2 != 0) ? raw_32_2 : 0);
                    int em_36 = (int)(bits_35_2 >> 19 & 4095);
                    int _max_276 = ((em_36 - 1792) > (0) ? (em_36 - 1792) : (0));
                    int _min_40 = ((_max_276) < (511) ? (_max_276) : (511));
                    int raw_37_1 = _min_40;
                    int neg_38 = (int)(bits_35_2 >> 31);
                    int h10_39 = ((neg_38 != 0) ? 511 - raw_37_1 : 512 + raw_37_1);
                    unsigned int neg_mask_40 = -(bits_35_2 >> 31);
                    unsigned int key_41 = bits_35_2 ^ (neg_mask_40 | 2147483648u);
                    int blk_42 = blks[1];
                    unsigned int invm_43 = ~(unsigned int)blk_42 & idx_mask;
                    unsigned long long rk_44 = (unsigned long long)h10_39 << 54 | (unsigned long long)key_41 << 22 | (unsigned long long)invm_43;
                    int in_window_45 = is_pass0 | (int)(rk_44 >> rsh_m == rprefix_me);
                    int d_46 = (int)((unsigned int)(rk_44 >> dsh_me) & dmask_me);
                    int counted_47 = is_hist & (int)(nan_34_2 == 0) & in_window_45;
                    int dd_48 = ((counted_47 != 0) ? d_46 : 1024);
                    atomicAdd(&hist_ext[dd_48 * 16 + q_local], 1);
                    unsigned int inv_49 = ~(unsigned int)blk_42;
                    unsigned long long comp_50 = (unsigned long long)key_41 << 32 | (unsigned long long)inv_49;
                    if (is_stage != 0 && nan_34_2 == 0 && comp_50 >= cthr_me) {
                        unsigned int _atomic_old_33 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_33 = _atomic_old_33;
                        if (slot_33 < 64) {
                            cand[q_local * 64 + (int)slot_33] = comp_50;
                        }
                    }
                    unsigned int raw_51_1 = __as_u32(xs[2]);
                    unsigned int mag_52 = raw_51_1 & 2147483647;
                    unsigned int nan_53 = (unsigned int)(mag_52 > 2139095040);
                    unsigned int bits_54 = ((mag_52 != 0) ? raw_51_1 : 0);
                    int em_55 = (int)(bits_54 >> 19 & 4095);
                    int _max_277 = ((em_55 - 1792) > (0) ? (em_55 - 1792) : (0));
                    int _min_41 = ((_max_277) < (511) ? (_max_277) : (511));
                    int raw_56_2 = _min_41;
                    int neg_57 = (int)(bits_54 >> 31);
                    int h10_58 = ((neg_57 != 0) ? 511 - raw_56_2 : 512 + raw_56_2);
                    unsigned int neg_mask_59 = -(bits_54 >> 31);
                    unsigned int key_60 = bits_54 ^ (neg_mask_59 | 2147483648u);
                    int blk_61 = blks[2];
                    unsigned int invm_62 = ~(unsigned int)blk_61 & idx_mask;
                    unsigned long long rk_63 = (unsigned long long)h10_58 << 54 | (unsigned long long)key_60 << 22 | (unsigned long long)invm_62;
                    int in_window_64 = is_pass0 | (int)(rk_63 >> rsh_m == rprefix_me);
                    int d_65 = (int)((unsigned int)(rk_63 >> dsh_me) & dmask_me);
                    int counted_66 = is_hist & (int)(nan_53 == 0) & in_window_64;
                    int dd_67 = ((counted_66 != 0) ? d_65 : 1024);
                    atomicAdd(&hist_ext[dd_67 * 16 + q_local], 1);
                    unsigned int inv_68_2 = ~(unsigned int)blk_61;
                    unsigned long long comp_69_2 = (unsigned long long)key_60 << 32 | (unsigned long long)inv_68_2;
                    if (is_stage != 0 && nan_53 == 0 && comp_69_2 >= cthr_me) {
                        unsigned int _atomic_old_34 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_34 = _atomic_old_34;
                        if (slot_34 < 64) {
                            cand[q_local * 64 + (int)slot_34] = comp_69_2;
                        }
                    }
                    unsigned int raw_70_2 = __as_u32(xs[3]);
                    unsigned int mag_71_2 = raw_70_2 & 2147483647;
                    unsigned int nan_72_2 = (unsigned int)(mag_71_2 > 2139095040);
                    unsigned int bits_73_2 = ((mag_71_2 != 0) ? raw_70_2 : 0);
                    int em_74 = (int)(bits_73_2 >> 19 & 4095);
                    int _max_278 = ((em_74 - 1792) > (0) ? (em_74 - 1792) : (0));
                    int _min_42 = ((_max_278) < (511) ? (_max_278) : (511));
                    int raw_75 = _min_42;
                    int neg_76 = (int)(bits_73_2 >> 31);
                    int h10_77 = ((neg_76 != 0) ? 511 - raw_75 : 512 + raw_75);
                    unsigned int neg_mask_78 = -(bits_73_2 >> 31);
                    unsigned int key_79 = bits_73_2 ^ (neg_mask_78 | 2147483648u);
                    int blk_80 = blks[3];
                    unsigned int invm_81 = ~(unsigned int)blk_80 & idx_mask;
                    unsigned long long rk_82 = (unsigned long long)h10_77 << 54 | (unsigned long long)key_79 << 22 | (unsigned long long)invm_81;
                    int in_window_83 = is_pass0 | (int)(rk_82 >> rsh_m == rprefix_me);
                    int d_84 = (int)((unsigned int)(rk_82 >> dsh_me) & dmask_me);
                    int counted_85 = is_hist & (int)(nan_72_2 == 0) & in_window_83;
                    int dd_86 = ((counted_85 != 0) ? d_84 : 1024);
                    atomicAdd(&hist_ext[dd_86 * 16 + q_local], 1);
                    unsigned int inv_87 = ~(unsigned int)blk_80;
                    unsigned long long comp_88 = (unsigned long long)key_79 << 32 | (unsigned long long)inv_87;
                    if (is_stage != 0 && nan_72_2 == 0 && comp_88 >= cthr_me) {
                        unsigned int _atomic_old_35 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_35 = _atomic_old_35;
                        if (slot_35 < 64) {
                            cand[q_local * 64 + (int)slot_35] = comp_88;
                        }
                    }
                    unsigned int raw_89 = __as_u32(xs[4]);
                    unsigned int mag_90 = raw_89 & 2147483647;
                    unsigned int nan_91 = (unsigned int)(mag_90 > 2139095040);
                    unsigned int bits_92 = ((mag_90 != 0) ? raw_89 : 0);
                    int em_93 = (int)(bits_92 >> 19 & 4095);
                    int _max_279 = ((em_93 - 1792) > (0) ? (em_93 - 1792) : (0));
                    int _min_43 = ((_max_279) < (511) ? (_max_279) : (511));
                    int raw_94_1 = _min_43;
                    int neg_95 = (int)(bits_92 >> 31);
                    int h10_96 = ((neg_95 != 0) ? 511 - raw_94_1 : 512 + raw_94_1);
                    unsigned int neg_mask_97 = -(bits_92 >> 31);
                    unsigned int key_98 = bits_92 ^ (neg_mask_97 | 2147483648u);
                    int blk_99 = blks[4];
                    unsigned int invm_100 = ~(unsigned int)blk_99 & idx_mask;
                    unsigned long long rk_101 = (unsigned long long)h10_96 << 54 | (unsigned long long)key_98 << 22 | (unsigned long long)invm_100;
                    int in_window_102 = is_pass0 | (int)(rk_101 >> rsh_m == rprefix_me);
                    int d_103 = (int)((unsigned int)(rk_101 >> dsh_me) & dmask_me);
                    int counted_104 = is_hist & (int)(nan_91 == 0) & in_window_102;
                    int dd_105 = ((counted_104 != 0) ? d_103 : 1024);
                    atomicAdd(&hist_ext[dd_105 * 16 + q_local], 1);
                    unsigned int inv_106 = ~(unsigned int)blk_99;
                    unsigned long long comp_107 = (unsigned long long)key_98 << 32 | (unsigned long long)inv_106;
                    if (is_stage != 0 && nan_91 == 0 && comp_107 >= cthr_me) {
                        unsigned int _atomic_old_36 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_36 = _atomic_old_36;
                        if (slot_36 < 64) {
                            cand[q_local * 64 + (int)slot_36] = comp_107;
                        }
                    }
                    unsigned int raw_108 = __as_u32(xs[5]);
                    unsigned int mag_109 = raw_108 & 2147483647;
                    unsigned int nan_110 = (unsigned int)(mag_109 > 2139095040);
                    unsigned int bits_111 = ((mag_109 != 0) ? raw_108 : 0);
                    int em_112 = (int)(bits_111 >> 19 & 4095);
                    int _max_280 = ((em_112 - 1792) > (0) ? (em_112 - 1792) : (0));
                    int _min_44 = ((_max_280) < (511) ? (_max_280) : (511));
                    int raw_113 = _min_44;
                    int neg_114 = (int)(bits_111 >> 31);
                    int h10_115 = ((neg_114 != 0) ? 511 - raw_113 : 512 + raw_113);
                    unsigned int neg_mask_116 = -(bits_111 >> 31);
                    unsigned int key_117 = bits_111 ^ (neg_mask_116 | 2147483648u);
                    int blk_118 = blks[5];
                    unsigned int invm_119 = ~(unsigned int)blk_118 & idx_mask;
                    unsigned long long rk_120 = (unsigned long long)h10_115 << 54 | (unsigned long long)key_117 << 22 | (unsigned long long)invm_119;
                    int in_window_121 = is_pass0 | (int)(rk_120 >> rsh_m == rprefix_me);
                    int d_122 = (int)((unsigned int)(rk_120 >> dsh_me) & dmask_me);
                    int counted_123 = is_hist & (int)(nan_110 == 0) & in_window_121;
                    int dd_124 = ((counted_123 != 0) ? d_122 : 1024);
                    atomicAdd(&hist_ext[dd_124 * 16 + q_local], 1);
                    unsigned int inv_125 = ~(unsigned int)blk_118;
                    unsigned long long comp_126 = (unsigned long long)key_117 << 32 | (unsigned long long)inv_125;
                    if (is_stage != 0 && nan_110 == 0 && comp_126 >= cthr_me) {
                        unsigned int _atomic_old_37 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_37 = _atomic_old_37;
                        if (slot_37 < 64) {
                            cand[q_local * 64 + (int)slot_37] = comp_126;
                        }
                    }
                    unsigned int raw_127 = __as_u32(xs[6]);
                    unsigned int mag_128 = raw_127 & 2147483647;
                    unsigned int nan_129 = (unsigned int)(mag_128 > 2139095040);
                    unsigned int bits_130 = ((mag_128 != 0) ? raw_127 : 0);
                    int em_131 = (int)(bits_130 >> 19 & 4095);
                    int _max_281 = ((em_131 - 1792) > (0) ? (em_131 - 1792) : (0));
                    int _min_45 = ((_max_281) < (511) ? (_max_281) : (511));
                    int raw_132 = _min_45;
                    int neg_133 = (int)(bits_130 >> 31);
                    int h10_134 = ((neg_133 != 0) ? 511 - raw_132 : 512 + raw_132);
                    unsigned int neg_mask_135 = -(bits_130 >> 31);
                    unsigned int key_136 = bits_130 ^ (neg_mask_135 | 2147483648u);
                    int blk_137 = blks[6];
                    unsigned int invm_138 = ~(unsigned int)blk_137 & idx_mask;
                    unsigned long long rk_139 = (unsigned long long)h10_134 << 54 | (unsigned long long)key_136 << 22 | (unsigned long long)invm_138;
                    int in_window_140 = is_pass0 | (int)(rk_139 >> rsh_m == rprefix_me);
                    int d_141 = (int)((unsigned int)(rk_139 >> dsh_me) & dmask_me);
                    int counted_142 = is_hist & (int)(nan_129 == 0) & in_window_140;
                    int dd_143 = ((counted_142 != 0) ? d_141 : 1024);
                    atomicAdd(&hist_ext[dd_143 * 16 + q_local], 1);
                    unsigned int inv_144 = ~(unsigned int)blk_137;
                    unsigned long long comp_145 = (unsigned long long)key_136 << 32 | (unsigned long long)inv_144;
                    if (is_stage != 0 && nan_129 == 0 && comp_145 >= cthr_me) {
                        unsigned int _atomic_old_38 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_38 = _atomic_old_38;
                        if (slot_38 < 64) {
                            cand[q_local * 64 + (int)slot_38] = comp_145;
                        }
                    }
                    unsigned int raw_146 = __as_u32(xs[7]);
                    unsigned int mag_147 = raw_146 & 2147483647;
                    unsigned int nan_148 = (unsigned int)(mag_147 > 2139095040);
                    unsigned int bits_149 = ((mag_147 != 0) ? raw_146 : 0);
                    int em_150 = (int)(bits_149 >> 19 & 4095);
                    int _max_282 = ((em_150 - 1792) > (0) ? (em_150 - 1792) : (0));
                    int _min_46 = ((_max_282) < (511) ? (_max_282) : (511));
                    int raw_151 = _min_46;
                    int neg_152 = (int)(bits_149 >> 31);
                    int h10_153 = ((neg_152 != 0) ? 511 - raw_151 : 512 + raw_151);
                    unsigned int neg_mask_154 = -(bits_149 >> 31);
                    unsigned int key_155 = bits_149 ^ (neg_mask_154 | 2147483648u);
                    int blk_156 = blks[7];
                    unsigned int invm_157 = ~(unsigned int)blk_156 & idx_mask;
                    unsigned long long rk_158 = (unsigned long long)h10_153 << 54 | (unsigned long long)key_155 << 22 | (unsigned long long)invm_157;
                    int in_window_159 = is_pass0 | (int)(rk_158 >> rsh_m == rprefix_me);
                    int d_160 = (int)((unsigned int)(rk_158 >> dsh_me) & dmask_me);
                    int counted_161 = is_hist & (int)(nan_148 == 0) & in_window_159;
                    int dd_162 = ((counted_161 != 0) ? d_160 : 1024);
                    atomicAdd(&hist_ext[dd_162 * 16 + q_local], 1);
                    unsigned int inv_163 = ~(unsigned int)blk_156;
                    unsigned long long comp_164 = (unsigned long long)key_155 << 32 | (unsigned long long)inv_163;
                    if (is_stage != 0 && nan_148 == 0 && comp_164 >= cthr_me) {
                        unsigned int _atomic_old_39 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_39 = _atomic_old_39;
                        if (slot_39 < 64) {
                            cand[q_local * 64 + (int)slot_39] = comp_164;
                        }
                    }
                    if (i1_3 < iters) {
                        unsigned int raw_0_2 = __as_u32(xs[8]);
                        unsigned int mag_1_3 = raw_0_2 & 2147483647;
                        unsigned int nan_2_2 = (unsigned int)(mag_1_3 > 2139095040);
                        unsigned int bits_3_2 = ((mag_1_3 != 0) ? raw_0_2 : 0);
                        int em_4 = (int)(bits_3_2 >> 19 & 4095);
                        int _max_283 = ((em_4 - 1792) > (0) ? (em_4 - 1792) : (0));
                        int _min_47 = ((_max_283) < (511) ? (_max_283) : (511));
                        int raw_5 = _min_47;
                        int neg_6 = (int)(bits_3_2 >> 31);
                        int h10_7 = ((neg_6 != 0) ? 511 - raw_5 : 512 + raw_5);
                        unsigned int neg_mask_8 = -(bits_3_2 >> 31);
                        unsigned int key_9 = bits_3_2 ^ (neg_mask_8 | 2147483648u);
                        int blk_11 = blks[8];
                        unsigned int invm_12 = ~(unsigned int)blk_11 & idx_mask;
                        unsigned long long rk_13 = (unsigned long long)h10_7 << 54 | (unsigned long long)key_9 << 22 | (unsigned long long)invm_12;
                        int in_window_14 = is_pass0 | (int)(rk_13 >> rsh_m == rprefix_me);
                        int d_15 = (int)((unsigned int)(rk_13 >> dsh_me) & dmask_me);
                        int counted_16 = is_hist & (int)(nan_2_2 == 0) & in_window_14;
                        int dd_17 = ((counted_16 != 0) ? d_15 : 1024);
                        atomicAdd(&hist_ext[dd_17 * 16 + q_local], 1);
                        unsigned int inv_18 = ~(unsigned int)blk_11;
                        unsigned long long comp_19 = (unsigned long long)key_9 << 32 | (unsigned long long)inv_18;
                        if (is_stage != 0 && nan_2_2 == 0 && comp_19 >= cthr_me) {
                            unsigned int _atomic_old_40 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_40 = _atomic_old_40;
                            if (slot_40 < 64) {
                                cand[q_local * 64 + (int)slot_40] = comp_19;
                            }
                        }
                        unsigned int raw_20 = __as_u32(xs[9]);
                        unsigned int mag_21 = raw_20 & 2147483647;
                        unsigned int nan_22 = (unsigned int)(mag_21 > 2139095040);
                        unsigned int bits_23 = ((mag_21 != 0) ? raw_20 : 0);
                        int em_24 = (int)(bits_23 >> 19 & 4095);
                        int _max_284 = ((em_24 - 1792) > (0) ? (em_24 - 1792) : (0));
                        int _min_48 = ((_max_284) < (511) ? (_max_284) : (511));
                        int raw_25 = _min_48;
                        int neg_26 = (int)(bits_23 >> 31);
                        int h10_27 = ((neg_26 != 0) ? 511 - raw_25 : 512 + raw_25);
                        unsigned int neg_mask_28_2 = -(bits_23 >> 31);
                        unsigned int key_29_2 = bits_23 ^ (neg_mask_28_2 | 2147483648u);
                        int blk_30 = blks[9];
                        unsigned int invm_31 = ~(unsigned int)blk_30 & idx_mask;
                        unsigned long long rk_32 = (unsigned long long)h10_27 << 54 | (unsigned long long)key_29_2 << 22 | (unsigned long long)invm_31;
                        int in_window_33 = is_pass0 | (int)(rk_32 >> rsh_m == rprefix_me);
                        int d_34 = (int)((unsigned int)(rk_32 >> dsh_me) & dmask_me);
                        int counted_35 = is_hist & (int)(nan_22 == 0) & in_window_33;
                        int dd_36 = ((counted_35 != 0) ? d_34 : 1024);
                        atomicAdd(&hist_ext[dd_36 * 16 + q_local], 1);
                        unsigned int inv_37 = ~(unsigned int)blk_30;
                        unsigned long long comp_38 = (unsigned long long)key_29_2 << 32 | (unsigned long long)inv_37;
                        if (is_stage != 0 && nan_22 == 0 && comp_38 >= cthr_me) {
                            unsigned int _atomic_old_41 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_41 = _atomic_old_41;
                            if (slot_41 < 64) {
                                cand[q_local * 64 + (int)slot_41] = comp_38;
                            }
                        }
                        unsigned int raw_39 = __as_u32(xs[10]);
                        unsigned int mag_40 = raw_39 & 2147483647;
                        unsigned int nan_41 = (unsigned int)(mag_40 > 2139095040);
                        unsigned int bits_42 = ((mag_40 != 0) ? raw_39 : 0);
                        int em_43 = (int)(bits_42 >> 19 & 4095);
                        int _max_285 = ((em_43 - 1792) > (0) ? (em_43 - 1792) : (0));
                        int _min_49 = ((_max_285) < (511) ? (_max_285) : (511));
                        int raw_44 = _min_49;
                        int neg_45 = (int)(bits_42 >> 31);
                        int h10_46 = ((neg_45 != 0) ? 511 - raw_44 : 512 + raw_44);
                        unsigned int neg_mask_47 = -(bits_42 >> 31);
                        unsigned int key_48 = bits_42 ^ (neg_mask_47 | 2147483648u);
                        int blk_49 = blks[10];
                        unsigned int invm_50 = ~(unsigned int)blk_49 & idx_mask;
                        unsigned long long rk_51 = (unsigned long long)h10_46 << 54 | (unsigned long long)key_48 << 22 | (unsigned long long)invm_50;
                        int in_window_52 = is_pass0 | (int)(rk_51 >> rsh_m == rprefix_me);
                        int d_53 = (int)((unsigned int)(rk_51 >> dsh_me) & dmask_me);
                        int counted_54 = is_hist & (int)(nan_41 == 0) & in_window_52;
                        int dd_55 = ((counted_54 != 0) ? d_53 : 1024);
                        atomicAdd(&hist_ext[dd_55 * 16 + q_local], 1);
                        unsigned int inv_56 = ~(unsigned int)blk_49;
                        unsigned long long comp_57 = (unsigned long long)key_48 << 32 | (unsigned long long)inv_56;
                        if (is_stage != 0 && nan_41 == 0 && comp_57 >= cthr_me) {
                            unsigned int _atomic_old_42 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_42 = _atomic_old_42;
                            if (slot_42 < 64) {
                                cand[q_local * 64 + (int)slot_42] = comp_57;
                            }
                        }
                        unsigned int raw_58_1 = __as_u32(xs[11]);
                        unsigned int mag_59_1 = raw_58_1 & 2147483647;
                        unsigned int nan_60_1 = (unsigned int)(mag_59_1 > 2139095040);
                        unsigned int bits_61_1 = ((mag_59_1 != 0) ? raw_58_1 : 0);
                        int em_62_1 = (int)(bits_61_1 >> 19 & 4095);
                        int _max_286 = ((em_62_1 - 1792) > (0) ? (em_62_1 - 1792) : (0));
                        int _min_50 = ((_max_286) < (511) ? (_max_286) : (511));
                        int raw_63_1 = _min_50;
                        int neg_64_1 = (int)(bits_61_1 >> 31);
                        int h10_65_1 = ((neg_64_1 != 0) ? 511 - raw_63_1 : 512 + raw_63_1);
                        unsigned int neg_mask_66_2 = -(bits_61_1 >> 31);
                        unsigned int key_67_2 = bits_61_1 ^ (neg_mask_66_2 | 2147483648u);
                        int blk_68 = blks[11];
                        unsigned int invm_69 = ~(unsigned int)blk_68 & idx_mask;
                        unsigned long long rk_70 = (unsigned long long)h10_65_1 << 54 | (unsigned long long)key_67_2 << 22 | (unsigned long long)invm_69;
                        int in_window_71 = is_pass0 | (int)(rk_70 >> rsh_m == rprefix_me);
                        int d_72 = (int)((unsigned int)(rk_70 >> dsh_me) & dmask_me);
                        int counted_73 = is_hist & (int)(nan_60_1 == 0) & in_window_71;
                        int dd_74 = ((counted_73 != 0) ? d_72 : 1024);
                        atomicAdd(&hist_ext[dd_74 * 16 + q_local], 1);
                        unsigned int inv_75 = ~(unsigned int)blk_68;
                        unsigned long long comp_76 = (unsigned long long)key_67_2 << 32 | (unsigned long long)inv_75;
                        if (is_stage != 0 && nan_60_1 == 0 && comp_76 >= cthr_me) {
                            unsigned int _atomic_old_43 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_43 = _atomic_old_43;
                            if (slot_43 < 64) {
                                cand[q_local * 64 + (int)slot_43] = comp_76;
                            }
                        }
                        unsigned int raw_77 = __as_u32(xs[12]);
                        unsigned int mag_78 = raw_77 & 2147483647;
                        unsigned int nan_79 = (unsigned int)(mag_78 > 2139095040);
                        unsigned int bits_80 = ((mag_78 != 0) ? raw_77 : 0);
                        int em_81 = (int)(bits_80 >> 19 & 4095);
                        int _max_287 = ((em_81 - 1792) > (0) ? (em_81 - 1792) : (0));
                        int _min_51 = ((_max_287) < (511) ? (_max_287) : (511));
                        int raw_82 = _min_51;
                        int neg_83 = (int)(bits_80 >> 31);
                        int h10_84 = ((neg_83 != 0) ? 511 - raw_82 : 512 + raw_82);
                        unsigned int neg_mask_85 = -(bits_80 >> 31);
                        unsigned int key_86 = bits_80 ^ (neg_mask_85 | 2147483648u);
                        int blk_87 = blks[12];
                        unsigned int invm_88 = ~(unsigned int)blk_87 & idx_mask;
                        unsigned long long rk_89 = (unsigned long long)h10_84 << 54 | (unsigned long long)key_86 << 22 | (unsigned long long)invm_88;
                        int in_window_90 = is_pass0 | (int)(rk_89 >> rsh_m == rprefix_me);
                        int d_91 = (int)((unsigned int)(rk_89 >> dsh_me) & dmask_me);
                        int counted_92 = is_hist & (int)(nan_79 == 0) & in_window_90;
                        int dd_93 = ((counted_92 != 0) ? d_91 : 1024);
                        atomicAdd(&hist_ext[dd_93 * 16 + q_local], 1);
                        unsigned int inv_94 = ~(unsigned int)blk_87;
                        unsigned long long comp_95 = (unsigned long long)key_86 << 32 | (unsigned long long)inv_94;
                        if (is_stage != 0 && nan_79 == 0 && comp_95 >= cthr_me) {
                            unsigned int _atomic_old_44 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_44 = _atomic_old_44;
                            if (slot_44 < 64) {
                                cand[q_local * 64 + (int)slot_44] = comp_95;
                            }
                        }
                        unsigned int raw_96 = __as_u32(xs[13]);
                        unsigned int mag_97 = raw_96 & 2147483647;
                        unsigned int nan_98 = (unsigned int)(mag_97 > 2139095040);
                        unsigned int bits_99 = ((mag_97 != 0) ? raw_96 : 0);
                        int em_100 = (int)(bits_99 >> 19 & 4095);
                        int _max_288 = ((em_100 - 1792) > (0) ? (em_100 - 1792) : (0));
                        int _min_52 = ((_max_288) < (511) ? (_max_288) : (511));
                        int raw_101 = _min_52;
                        int neg_102 = (int)(bits_99 >> 31);
                        int h10_103 = ((neg_102 != 0) ? 511 - raw_101 : 512 + raw_101);
                        unsigned int neg_mask_104 = -(bits_99 >> 31);
                        unsigned int key_105 = bits_99 ^ (neg_mask_104 | 2147483648u);
                        int blk_106 = blks[13];
                        unsigned int invm_107 = ~(unsigned int)blk_106 & idx_mask;
                        unsigned long long rk_108 = (unsigned long long)h10_103 << 54 | (unsigned long long)key_105 << 22 | (unsigned long long)invm_107;
                        int in_window_109 = is_pass0 | (int)(rk_108 >> rsh_m == rprefix_me);
                        int d_110 = (int)((unsigned int)(rk_108 >> dsh_me) & dmask_me);
                        int counted_111 = is_hist & (int)(nan_98 == 0) & in_window_109;
                        int dd_112 = ((counted_111 != 0) ? d_110 : 1024);
                        atomicAdd(&hist_ext[dd_112 * 16 + q_local], 1);
                        unsigned int inv_113 = ~(unsigned int)blk_106;
                        unsigned long long comp_114 = (unsigned long long)key_105 << 32 | (unsigned long long)inv_113;
                        if (is_stage != 0 && nan_98 == 0 && comp_114 >= cthr_me) {
                            unsigned int _atomic_old_45 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_45 = _atomic_old_45;
                            if (slot_45 < 64) {
                                cand[q_local * 64 + (int)slot_45] = comp_114;
                            }
                        }
                        unsigned int raw_115 = __as_u32(xs[14]);
                        unsigned int mag_116 = raw_115 & 2147483647;
                        unsigned int nan_117 = (unsigned int)(mag_116 > 2139095040);
                        unsigned int bits_118 = ((mag_116 != 0) ? raw_115 : 0);
                        int em_119 = (int)(bits_118 >> 19 & 4095);
                        int _max_289 = ((em_119 - 1792) > (0) ? (em_119 - 1792) : (0));
                        int _min_53 = ((_max_289) < (511) ? (_max_289) : (511));
                        int raw_120 = _min_53;
                        int neg_121 = (int)(bits_118 >> 31);
                        int h10_122 = ((neg_121 != 0) ? 511 - raw_120 : 512 + raw_120);
                        unsigned int neg_mask_123 = -(bits_118 >> 31);
                        unsigned int key_124 = bits_118 ^ (neg_mask_123 | 2147483648u);
                        int blk_125 = blks[14];
                        unsigned int invm_126 = ~(unsigned int)blk_125 & idx_mask;
                        unsigned long long rk_127 = (unsigned long long)h10_122 << 54 | (unsigned long long)key_124 << 22 | (unsigned long long)invm_126;
                        int in_window_128 = is_pass0 | (int)(rk_127 >> rsh_m == rprefix_me);
                        int d_129 = (int)((unsigned int)(rk_127 >> dsh_me) & dmask_me);
                        int counted_130 = is_hist & (int)(nan_117 == 0) & in_window_128;
                        int dd_131 = ((counted_130 != 0) ? d_129 : 1024);
                        atomicAdd(&hist_ext[dd_131 * 16 + q_local], 1);
                        unsigned int inv_132 = ~(unsigned int)blk_125;
                        unsigned long long comp_133 = (unsigned long long)key_124 << 32 | (unsigned long long)inv_132;
                        if (is_stage != 0 && nan_117 == 0 && comp_133 >= cthr_me) {
                            unsigned int _atomic_old_46 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_46 = _atomic_old_46;
                            if (slot_46 < 64) {
                                cand[q_local * 64 + (int)slot_46] = comp_133;
                            }
                        }
                        unsigned int raw_134 = __as_u32(xs[15]);
                        unsigned int mag_135 = raw_134 & 2147483647;
                        unsigned int nan_136 = (unsigned int)(mag_135 > 2139095040);
                        unsigned int bits_137 = ((mag_135 != 0) ? raw_134 : 0);
                        int em_138 = (int)(bits_137 >> 19 & 4095);
                        int _max_290 = ((em_138 - 1792) > (0) ? (em_138 - 1792) : (0));
                        int _min_54 = ((_max_290) < (511) ? (_max_290) : (511));
                        int raw_139 = _min_54;
                        int neg_140 = (int)(bits_137 >> 31);
                        int h10_141 = ((neg_140 != 0) ? 511 - raw_139 : 512 + raw_139);
                        unsigned int neg_mask_142 = -(bits_137 >> 31);
                        unsigned int key_143 = bits_137 ^ (neg_mask_142 | 2147483648u);
                        int blk_144 = blks[15];
                        unsigned int invm_145 = ~(unsigned int)blk_144 & idx_mask;
                        unsigned long long rk_146 = (unsigned long long)h10_141 << 54 | (unsigned long long)key_143 << 22 | (unsigned long long)invm_145;
                        int in_window_147 = is_pass0 | (int)(rk_146 >> rsh_m == rprefix_me);
                        int d_148 = (int)((unsigned int)(rk_146 >> dsh_me) & dmask_me);
                        int counted_149 = is_hist & (int)(nan_136 == 0) & in_window_147;
                        int dd_150 = ((counted_149 != 0) ? d_148 : 1024);
                        atomicAdd(&hist_ext[dd_150 * 16 + q_local], 1);
                        unsigned int inv_151 = ~(unsigned int)blk_144;
                        unsigned long long comp_152 = (unsigned long long)key_143 << 32 | (unsigned long long)inv_151;
                        if (is_stage != 0 && nan_136 == 0 && comp_152 >= cthr_me) {
                            unsigned int _atomic_old_47 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_47 = _atomic_old_47;
                            if (slot_47 < 64) {
                                cand[q_local * 64 + (int)slot_47] = comp_152;
                            }
                        }
                    }
                }
            }
            __syncthreads();
            if (is_hist != 0) {
                int base_1 = seg * 1024 + q_local;
                subs[0] = 0;
                subs[1] = 0;
                subs[2] = 0;
                subs[3] = 0;
                unsigned int c_3 = hist[base_1];
                subs[0] = subs[0] + c_3;
                unsigned int c_0 = hist[base_1 + 16];
                subs[0] = subs[0] + c_0;
                unsigned int c_1_1 = hist[base_1 + 32];
                subs[0] = subs[0] + c_1_1;
                unsigned int c_2_2 = hist[base_1 + 48];
                subs[0] = subs[0] + c_2_2;
                unsigned int c_3_1 = hist[base_1 + 64];
                subs[0] = subs[0] + c_3_1;
                unsigned int c_4 = hist[base_1 + 80];
                subs[0] = subs[0] + c_4;
                unsigned int c_5_1 = hist[base_1 + 96];
                subs[0] = subs[0] + c_5_1;
                unsigned int c_6 = hist[base_1 + 112];
                subs[0] = subs[0] + c_6;
                unsigned int c_7 = hist[base_1 + 128];
                subs[0] = subs[0] + c_7;
                unsigned int c_8_1 = hist[base_1 + 144];
                subs[0] = subs[0] + c_8_1;
                unsigned int c_9 = hist[base_1 + 160];
                subs[0] = subs[0] + c_9;
                unsigned int c_10 = hist[base_1 + 176];
                subs[0] = subs[0] + c_10;
                unsigned int c_11 = hist[base_1 + 192];
                subs[0] = subs[0] + c_11;
                unsigned int c_12 = hist[base_1 + 208];
                subs[0] = subs[0] + c_12;
                unsigned int c_13 = hist[base_1 + 224];
                subs[0] = subs[0] + c_13;
                unsigned int c_14 = hist[base_1 + 240];
                subs[0] = subs[0] + c_14;
                unsigned int c_15 = hist[base_1 + 256];
                subs[1] = subs[1] + c_15;
                unsigned int c_16 = hist[base_1 + 272];
                subs[1] = subs[1] + c_16;
                unsigned int c_17 = hist[base_1 + 288];
                subs[1] = subs[1] + c_17;
                unsigned int c_18 = hist[base_1 + 304];
                subs[1] = subs[1] + c_18;
                unsigned int c_19 = hist[base_1 + 320];
                subs[1] = subs[1] + c_19;
                unsigned int c_20 = hist[base_1 + 336];
                subs[1] = subs[1] + c_20;
                unsigned int c_21 = hist[base_1 + 352];
                subs[1] = subs[1] + c_21;
                unsigned int c_22 = hist[base_1 + 368];
                subs[1] = subs[1] + c_22;
                unsigned int c_23 = hist[base_1 + 384];
                subs[1] = subs[1] + c_23;
                unsigned int c_24 = hist[base_1 + 400];
                subs[1] = subs[1] + c_24;
                unsigned int c_25 = hist[base_1 + 416];
                subs[1] = subs[1] + c_25;
                unsigned int c_26 = hist[base_1 + 432];
                subs[1] = subs[1] + c_26;
                unsigned int c_27 = hist[base_1 + 448];
                subs[1] = subs[1] + c_27;
                unsigned int c_28 = hist[base_1 + 464];
                subs[1] = subs[1] + c_28;
                unsigned int c_29 = hist[base_1 + 480];
                subs[1] = subs[1] + c_29;
                unsigned int c_30 = hist[base_1 + 496];
                subs[1] = subs[1] + c_30;
                unsigned int c_31 = hist[base_1 + 512];
                subs[2] = subs[2] + c_31;
                unsigned int c_32 = hist[base_1 + 528];
                subs[2] = subs[2] + c_32;
                unsigned int c_33 = hist[base_1 + 544];
                subs[2] = subs[2] + c_33;
                unsigned int c_34 = hist[base_1 + 560];
                subs[2] = subs[2] + c_34;
                unsigned int c_35 = hist[base_1 + 576];
                subs[2] = subs[2] + c_35;
                unsigned int c_36 = hist[base_1 + 592];
                subs[2] = subs[2] + c_36;
                unsigned int c_37 = hist[base_1 + 608];
                subs[2] = subs[2] + c_37;
                unsigned int c_38 = hist[base_1 + 624];
                subs[2] = subs[2] + c_38;
                unsigned int c_39 = hist[base_1 + 640];
                subs[2] = subs[2] + c_39;
                unsigned int c_40 = hist[base_1 + 656];
                subs[2] = subs[2] + c_40;
                unsigned int c_41 = hist[base_1 + 672];
                subs[2] = subs[2] + c_41;
                unsigned int c_42 = hist[base_1 + 688];
                subs[2] = subs[2] + c_42;
                unsigned int c_43 = hist[base_1 + 704];
                subs[2] = subs[2] + c_43;
                unsigned int c_44 = hist[base_1 + 720];
                subs[2] = subs[2] + c_44;
                unsigned int c_45 = hist[base_1 + 736];
                subs[2] = subs[2] + c_45;
                unsigned int c_46 = hist[base_1 + 752];
                subs[2] = subs[2] + c_46;
                unsigned int c_47 = hist[base_1 + 768];
                subs[3] = subs[3] + c_47;
                unsigned int c_48 = hist[base_1 + 784];
                subs[3] = subs[3] + c_48;
                unsigned int c_49 = hist[base_1 + 800];
                subs[3] = subs[3] + c_49;
                unsigned int c_50 = hist[base_1 + 816];
                subs[3] = subs[3] + c_50;
                unsigned int c_51 = hist[base_1 + 832];
                subs[3] = subs[3] + c_51;
                unsigned int c_52 = hist[base_1 + 848];
                subs[3] = subs[3] + c_52;
                unsigned int c_53 = hist[base_1 + 864];
                subs[3] = subs[3] + c_53;
                unsigned int c_54 = hist[base_1 + 880];
                subs[3] = subs[3] + c_54;
                unsigned int c_55 = hist[base_1 + 896];
                subs[3] = subs[3] + c_55;
                unsigned int c_56 = hist[base_1 + 912];
                subs[3] = subs[3] + c_56;
                unsigned int c_57 = hist[base_1 + 928];
                subs[3] = subs[3] + c_57;
                unsigned int c_58 = hist[base_1 + 944];
                subs[3] = subs[3] + c_58;
                unsigned int c_59 = hist[base_1 + 960];
                subs[3] = subs[3] + c_59;
                unsigned int c_60 = hist[base_1 + 976];
                subs[3] = subs[3] + c_60;
                unsigned int c_61 = hist[base_1 + 992];
                subs[3] = subs[3] + c_61;
                unsigned int c_62 = hist[base_1 + 1008];
                subs[3] = subs[3] + c_62;
                seg_sum[seg * 16 + q_local] = subs[0] + subs[1] + subs[2] + subs[3];
            }
            __syncthreads();
            unsigned int contrib = 0;
            if (is_hist != 0) {
                unsigned int above = 0;
                unsigned int total_1 = 0;
                unsigned int v = seg_sum[q_local];
                total_1 = total_1 + v;
                above = above + ((seg < 0) ? v : 0);
                unsigned int v_0 = seg_sum[16 + q_local];
                total_1 = total_1 + v_0;
                above = above + ((seg < 1) ? v_0 : 0);
                unsigned int v_1 = seg_sum[32 + q_local];
                total_1 = total_1 + v_1;
                above = above + ((seg < 2) ? v_1 : 0);
                unsigned int v_2 = seg_sum[48 + q_local];
                total_1 = total_1 + v_2;
                above = above + ((seg < 3) ? v_2 : 0);
                unsigned int v_3 = seg_sum[64 + q_local];
                total_1 = total_1 + v_3;
                above = above + ((seg < 4) ? v_3 : 0);
                unsigned int v_4 = seg_sum[80 + q_local];
                total_1 = total_1 + v_4;
                above = above + ((seg < 5) ? v_4 : 0);
                unsigned int v_5 = seg_sum[96 + q_local];
                total_1 = total_1 + v_5;
                above = above + ((seg < 6) ? v_5 : 0);
                unsigned int v_6 = seg_sum[112 + q_local];
                total_1 = total_1 + v_6;
                above = above + ((seg < 7) ? v_6 : 0);
                unsigned int v_7 = seg_sum[128 + q_local];
                total_1 = total_1 + v_7;
                above = above + ((seg < 8) ? v_7 : 0);
                unsigned int v_8 = seg_sum[144 + q_local];
                total_1 = total_1 + v_8;
                above = above + ((seg < 9) ? v_8 : 0);
                unsigned int v_9 = seg_sum[160 + q_local];
                total_1 = total_1 + v_9;
                above = above + ((seg < 10) ? v_9 : 0);
                unsigned int v_10 = seg_sum[176 + q_local];
                total_1 = total_1 + v_10;
                above = above + ((seg < 11) ? v_10 : 0);
                unsigned int v_11 = seg_sum[192 + q_local];
                total_1 = total_1 + v_11;
                above = above + ((seg < 12) ? v_11 : 0);
                unsigned int v_12 = seg_sum[208 + q_local];
                total_1 = total_1 + v_12;
                above = above + ((seg < 13) ? v_12 : 0);
                unsigned int v_13 = seg_sum[224 + q_local];
                total_1 = total_1 + v_13;
                above = above + ((seg < 14) ? v_13 : 0);
                unsigned int v_14 = seg_sum[240 + q_local];
                total_1 = total_1 + v_14;
                above = above + ((seg < 15) ? v_14 : 0);
                pre[0] = above;
                pre[1] = total_1;
                unsigned int above_15 = pre[0];
                unsigned int total_16 = pre[1];
                unsigned int mine = subs[0] + subs[1] + subs[2] + subs[3];
                unsigned int r_rem = (unsigned int)r - ngt_me;
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
                        unsigned int c_63 = hist[base_2 + 240];
                        unsigned int take = (unsigned int)(found == 0 && r_rem <= cum + c_63);
                        tsel_0 = ((take != 0) ? (unsigned int)(bin0 + 15) : tsel_0);
                        ct_1 = ((take != 0) ? c_63 : ct_1);
                        nadd_2 = ((take != 0) ? cum : nadd_2);
                        found = found | take;
                        cum = cum + c_63;
                        unsigned int c_3_2 = hist[base_2 + 224];
                        unsigned int take_4 = (unsigned int)(found == 0 && r_rem <= cum + c_3_2);
                        tsel_0 = ((take_4 != 0) ? (unsigned int)(bin0 + 14) : tsel_0);
                        ct_1 = ((take_4 != 0) ? c_3_2 : ct_1);
                        nadd_2 = ((take_4 != 0) ? cum : nadd_2);
                        found = found | take_4;
                        cum = cum + c_3_2;
                        unsigned int c_5_2 = hist[base_2 + 208];
                        unsigned int take_6 = (unsigned int)(found == 0 && r_rem <= cum + c_5_2);
                        tsel_0 = ((take_6 != 0) ? (unsigned int)(bin0 + 13) : tsel_0);
                        ct_1 = ((take_6 != 0) ? c_5_2 : ct_1);
                        nadd_2 = ((take_6 != 0) ? cum : nadd_2);
                        found = found | take_6;
                        cum = cum + c_5_2;
                        unsigned int c_7_1 = hist[base_2 + 192];
                        unsigned int take_8 = (unsigned int)(found == 0 && r_rem <= cum + c_7_1);
                        tsel_0 = ((take_8 != 0) ? (unsigned int)(bin0 + 12) : tsel_0);
                        ct_1 = ((take_8 != 0) ? c_7_1 : ct_1);
                        nadd_2 = ((take_8 != 0) ? cum : nadd_2);
                        found = found | take_8;
                        cum = cum + c_7_1;
                        unsigned int c_9_1 = hist[base_2 + 176];
                        unsigned int take_10 = (unsigned int)(found == 0 && r_rem <= cum + c_9_1);
                        tsel_0 = ((take_10 != 0) ? (unsigned int)(bin0 + 11) : tsel_0);
                        ct_1 = ((take_10 != 0) ? c_9_1 : ct_1);
                        nadd_2 = ((take_10 != 0) ? cum : nadd_2);
                        found = found | take_10;
                        cum = cum + c_9_1;
                        unsigned int c_11_1 = hist[base_2 + 160];
                        unsigned int take_12 = (unsigned int)(found == 0 && r_rem <= cum + c_11_1);
                        tsel_0 = ((take_12 != 0) ? (unsigned int)(bin0 + 10) : tsel_0);
                        ct_1 = ((take_12 != 0) ? c_11_1 : ct_1);
                        nadd_2 = ((take_12 != 0) ? cum : nadd_2);
                        found = found | take_12;
                        cum = cum + c_11_1;
                        unsigned int c_13_1 = hist[base_2 + 144];
                        unsigned int take_14 = (unsigned int)(found == 0 && r_rem <= cum + c_13_1);
                        tsel_0 = ((take_14 != 0) ? (unsigned int)(bin0 + 9) : tsel_0);
                        ct_1 = ((take_14 != 0) ? c_13_1 : ct_1);
                        nadd_2 = ((take_14 != 0) ? cum : nadd_2);
                        found = found | take_14;
                        cum = cum + c_13_1;
                        unsigned int c_15_1 = hist[base_2 + 128];
                        unsigned int take_16 = (unsigned int)(found == 0 && r_rem <= cum + c_15_1);
                        tsel_0 = ((take_16 != 0) ? (unsigned int)(bin0 + 8) : tsel_0);
                        ct_1 = ((take_16 != 0) ? c_15_1 : ct_1);
                        nadd_2 = ((take_16 != 0) ? cum : nadd_2);
                        found = found | take_16;
                        cum = cum + c_15_1;
                        unsigned int c_17_1 = hist[base_2 + 112];
                        unsigned int take_18 = (unsigned int)(found == 0 && r_rem <= cum + c_17_1);
                        tsel_0 = ((take_18 != 0) ? (unsigned int)(bin0 + 7) : tsel_0);
                        ct_1 = ((take_18 != 0) ? c_17_1 : ct_1);
                        nadd_2 = ((take_18 != 0) ? cum : nadd_2);
                        found = found | take_18;
                        cum = cum + c_17_1;
                        unsigned int c_19_1 = hist[base_2 + 96];
                        unsigned int take_20 = (unsigned int)(found == 0 && r_rem <= cum + c_19_1);
                        tsel_0 = ((take_20 != 0) ? (unsigned int)(bin0 + 6) : tsel_0);
                        ct_1 = ((take_20 != 0) ? c_19_1 : ct_1);
                        nadd_2 = ((take_20 != 0) ? cum : nadd_2);
                        found = found | take_20;
                        cum = cum + c_19_1;
                        unsigned int c_21_1 = hist[base_2 + 80];
                        unsigned int take_22 = (unsigned int)(found == 0 && r_rem <= cum + c_21_1);
                        tsel_0 = ((take_22 != 0) ? (unsigned int)(bin0 + 5) : tsel_0);
                        ct_1 = ((take_22 != 0) ? c_21_1 : ct_1);
                        nadd_2 = ((take_22 != 0) ? cum : nadd_2);
                        found = found | take_22;
                        cum = cum + c_21_1;
                        unsigned int c_23_1 = hist[base_2 + 64];
                        unsigned int take_24 = (unsigned int)(found == 0 && r_rem <= cum + c_23_1);
                        tsel_0 = ((take_24 != 0) ? (unsigned int)(bin0 + 4) : tsel_0);
                        ct_1 = ((take_24 != 0) ? c_23_1 : ct_1);
                        nadd_2 = ((take_24 != 0) ? cum : nadd_2);
                        found = found | take_24;
                        cum = cum + c_23_1;
                        unsigned int c_25_1 = hist[base_2 + 48];
                        unsigned int take_26 = (unsigned int)(found == 0 && r_rem <= cum + c_25_1);
                        tsel_0 = ((take_26 != 0) ? (unsigned int)(bin0 + 3) : tsel_0);
                        ct_1 = ((take_26 != 0) ? c_25_1 : ct_1);
                        nadd_2 = ((take_26 != 0) ? cum : nadd_2);
                        found = found | take_26;
                        cum = cum + c_25_1;
                        unsigned int c_27_1 = hist[base_2 + 32];
                        unsigned int take_28 = (unsigned int)(found == 0 && r_rem <= cum + c_27_1);
                        tsel_0 = ((take_28 != 0) ? (unsigned int)(bin0 + 2) : tsel_0);
                        ct_1 = ((take_28 != 0) ? c_27_1 : ct_1);
                        nadd_2 = ((take_28 != 0) ? cum : nadd_2);
                        found = found | take_28;
                        cum = cum + c_27_1;
                        unsigned int c_29_1 = hist[base_2 + 16];
                        unsigned int take_30 = (unsigned int)(found == 0 && r_rem <= cum + c_29_1);
                        tsel_0 = ((take_30 != 0) ? (unsigned int)(bin0 + 1) : tsel_0);
                        ct_1 = ((take_30 != 0) ? c_29_1 : ct_1);
                        nadd_2 = ((take_30 != 0) ? cum : nadd_2);
                        found = found | take_30;
                        cum = cum + c_29_1;
                        unsigned int c_31_1 = hist[base_2];
                        unsigned int take_32 = (unsigned int)(found == 0 && r_rem <= cum + c_31_1);
                        tsel_0 = ((take_32 != 0) ? (unsigned int)bin0 : tsel_0);
                        ct_1 = ((take_32 != 0) ? c_31_1 : ct_1);
                        nadd_2 = ((take_32 != 0) ? cum : nadd_2);
                        found = found | take_32;
                        cum = cum + c_31_1;
                        res[0] = tsel_0;
                        res[1] = ct_1;
                        res[2] = nadd_2;
                        tsel = res[0];
                        ct = res[1];
                        nadd = res[2];
                    } else {
                        ct = hist[q_local];
                        nadd = total_16 - ct;
                    }
                    unsigned int ngt_new = ngt_me + nadd;
                    int over = (int)(ngt_new + ct > 64);
                    unsigned int rsh_new = 0;
                    unsigned long long rp_new = 0;
                    if (is_pass0 != 0) {
                        int sat = (int)(tsel == 0 || tsel == 511 || tsel == 512 || tsel == 1023);
                        if (sat != 0) {
                            rsh_new = 54;
                            rp_new = (unsigned long long)tsel;
                        } else {
                            rsh_new = 41;
                            unsigned int p13 = tsel + ((tsel >= 512) ? 5376 : 1792);
                            rp_new = (unsigned long long)tsel << 13 | (unsigned long long)p13;
                        }
                        state[q_local * 8 + 5] = tsel;
                    } else {
                        rp_new = rprefix_me << (unsigned long long)w_me | (unsigned long long)tsel;
                        rsh_new = rshift_me - w_me;
                        if (rsh_new == 22 && over != 0) {
                            rp_new = rp_new << (unsigned long long)(22 - index_bits);
                            rsh_new = (unsigned int)index_bits;
                        }
                    }
                    unsigned int mode_new = ((over != 0) ? 0 : 1);
                    state[q_local * 8] = mode_new;
                    state[q_local * 8 + 1] = rsh_new;
                    state[q_local * 8 + 2] = (unsigned int)rp_new;
                    state[q_local * 8 + 3] = (unsigned int)(rp_new >> 32);
                    state[q_local * 8 + 4] = ngt_new;
                    contrib = 1;
                }
                int base_3 = seg * 1024 + q_local;
                hist[base_3] = 0;
                hist[base_3 + 16] = 0;
                hist[base_3 + 32] = 0;
                hist[base_3 + 48] = 0;
                hist[base_3 + 64] = 0;
                hist[base_3 + 80] = 0;
                hist[base_3 + 96] = 0;
                hist[base_3 + 112] = 0;
                hist[base_3 + 128] = 0;
                hist[base_3 + 144] = 0;
                hist[base_3 + 160] = 0;
                hist[base_3 + 176] = 0;
                hist[base_3 + 192] = 0;
                hist[base_3 + 208] = 0;
                hist[base_3 + 224] = 0;
                hist[base_3 + 240] = 0;
                hist[base_3 + 256] = 0;
                hist[base_3 + 272] = 0;
                hist[base_3 + 288] = 0;
                hist[base_3 + 304] = 0;
                hist[base_3 + 320] = 0;
                hist[base_3 + 336] = 0;
                hist[base_3 + 352] = 0;
                hist[base_3 + 368] = 0;
                hist[base_3 + 384] = 0;
                hist[base_3 + 400] = 0;
                hist[base_3 + 416] = 0;
                hist[base_3 + 432] = 0;
                hist[base_3 + 448] = 0;
                hist[base_3 + 464] = 0;
                hist[base_3 + 480] = 0;
                hist[base_3 + 496] = 0;
                hist[base_3 + 512] = 0;
                hist[base_3 + 528] = 0;
                hist[base_3 + 544] = 0;
                hist[base_3 + 560] = 0;
                hist[base_3 + 576] = 0;
                hist[base_3 + 592] = 0;
                hist[base_3 + 608] = 0;
                hist[base_3 + 624] = 0;
                hist[base_3 + 640] = 0;
                hist[base_3 + 656] = 0;
                hist[base_3 + 672] = 0;
                hist[base_3 + 688] = 0;
                hist[base_3 + 704] = 0;
                hist[base_3 + 720] = 0;
                hist[base_3 + 736] = 0;
                hist[base_3 + 752] = 0;
                hist[base_3 + 768] = 0;
                hist[base_3 + 784] = 0;
                hist[base_3 + 800] = 0;
                hist[base_3 + 816] = 0;
                hist[base_3 + 832] = 0;
                hist[base_3 + 848] = 0;
                hist[base_3 + 864] = 0;
                hist[base_3 + 880] = 0;
                hist[base_3 + 896] = 0;
                hist[base_3 + 912] = 0;
                hist[base_3 + 928] = 0;
                hist[base_3 + 944] = 0;
                hist[base_3 + 960] = 0;
                hist[base_3 + 976] = 0;
                hist[base_3 + 992] = 0;
                hist[base_3 + 1008] = 0;
            } else if (is_stage != 0) {
                if (seg == 0) {
                    state[q_local * 8] = 2;
                }
            }
            uint32_t _cta_or_0 = __syncthreads_or(contrib != 0);
            any_open = _cta_or_0;
        }
    }
    unsigned int cnt_me = cand_cnt[q_local];
    unsigned int _min_55 = ((cnt_me) < (64) ? (cnt_me) : (64));
    int n_me = (int)_min_55;
    unsigned long long ovf_c = -1;
    unsigned long long cs[4];
    int ranks[4];
    int i = j;
    cs[0] = 0;
    if (i < n_me) {
        cs[0] = cand[q_local * 64 + i];
    }
    ranks[0] = 0;
    int i_4 = j + 16;
    cs[1] = 0;
    if (i_4 < n_me) {
        cs[1] = cand[q_local * 64 + i_4];
    }
    ranks[1] = 0;
    int i_5 = j + 32;
    cs[2] = 0;
    if (i_5 < n_me) {
        cs[2] = cand[q_local * 64 + i_5];
    }
    ranks[2] = 0;
    int i_6 = j + 48;
    cs[3] = 0;
    if (i_6 < n_me) {
        cs[3] = cand[q_local * 64 + i_6];
    }
    ranks[3] = 0;
    for (int m_1 = 0; m_1 < n_me; m_1++) {
        unsigned long long cm = cand[q_local * 64 + m_1];
        ranks[0] = ranks[0] + (int)(cm > cs[0]);
        ranks[1] = ranks[1] + (int)(cm > cs[1]);
        ranks[2] = ranks[2] + (int)(cm > cs[2]);
        ranks[3] = ranks[3] + (int)(cm > cs[3]);
    }
    int i2 = j;
    int live = (int)(i2 < n_me) & (int)(ranks[0] < r);
    if (live != 0) {
        hist64[q_local * 16 + ranks[0]] = cs[0];
    }
    int i2_7 = j + 16;
    int live_8 = (int)(i2_7 < n_me) & (int)(ranks[1] < r);
    if (live_8 != 0) {
        hist64[q_local * 16 + ranks[1]] = cs[1];
    }
    int i2_9 = j + 32;
    int live_10 = (int)(i2_9 < n_me) & (int)(ranks[2] < r);
    if (live_10 != 0) {
        hist64[q_local * 16 + ranks[2]] = cs[2];
    }
    int i2_11 = j + 48;
    int live_12 = (int)(i2_11 < n_me) & (int)(ranks[3] < r);
    if (live_12 != 0) {
        hist64[q_local * 16 + ranks[3]] = cs[3];
    }
    if (j >= n_me && j < r) {
        hist64[q_local * 16 + j] = 0;
    }
    __syncthreads();
    int is_last = 0;
    if (num_slices > 1) {
        int qm = warp_2 * 2 + (lane_1 >> 4);
        int jm = lane_1 & 15;
        unsigned long long pv = 0;
        if (jm < r) {
            pv = hist64[qm * 16 + jm];
        }
        unsigned int cnt_m = cand_cnt[qm];
        unsigned int mode_m = state[qm * 8];
        int bad_m = (int)(cnt_m > 64) | (int)(mode_m != 2);
        if (jm == 0 && bad_m != 0) {
            pv = ovf_c;
        }
        partials[((group * num_slices + s) * 16 + qm) * 16 + jm] = pv;
        __syncthreads();
        if (tid_0 == 0) {
            uint32_t _atomic_inc_old_0;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_inc_old_0) : "l"(&counters[group]), "r"(static_cast<uint32_t>(num_slices - 1)) : "memory");
            unsigned int old = _atomic_inc_old_0;
            flags[0] = (unsigned int)(old == (unsigned int)(num_slices - 1));
        }
        __syncthreads();
        unsigned int last_flag = flags[0];
        is_last = (int)(last_flag != 0);
        if (is_last != 0) {
            unsigned long long _vec_load_4[1];
            {
                uint64_t _scalar_bits_4;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_4) : "l"((const void*)(partials + ((group * num_slices * 16 + qm) * 16 + jm))) : "memory");
                _vec_load_4[0] = (unsigned long long)_scalar_bits_4;
            }
            unsigned long long vm = _vec_load_4[0];
            for (int sl = 1; sl < num_slices; sl++) {
                unsigned long long _vec_load_5[1];
                {
                    uint64_t _scalar_bits_5;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_5) : "l"((const void*)(partials + (((group * num_slices + sl) * 16 + qm) * 16 + (15 - jm)))) : "memory");
                    _vec_load_5[0] = (unsigned long long)_scalar_bits_5;
                }
                unsigned long long bv = _vec_load_5[0];
                vm = ((vm > bv) ? vm : bv);
                unsigned long long cur = vm;
                unsigned long long _shfl_xor_288 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
                unsigned long long vp = _shfl_xor_288;
                int take_max = (int)((jm & 8) == 0);
                int gt = (int)(cur > vp);
                cur = ((gt == take_max) ? cur : vp);
                unsigned long long _shfl_xor_289 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
                unsigned long long vp_0 = _shfl_xor_289;
                int take_max_1 = (int)((jm & 4) == 0);
                int gt_2 = (int)(cur > vp_0);
                cur = ((gt_2 == take_max_1) ? cur : vp_0);
                unsigned long long _shfl_xor_290 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
                unsigned long long vp_3 = _shfl_xor_290;
                int take_max_4 = (int)((jm & 2) == 0);
                int gt_5 = (int)(cur > vp_3);
                cur = ((gt_5 == take_max_4) ? cur : vp_3);
                unsigned long long _shfl_xor_291 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
                unsigned long long vp_6 = _shfl_xor_291;
                int take_max_7 = (int)((jm & 1) == 0);
                int gt_8 = (int)(cur > vp_6);
                cur = ((gt_8 == take_max_7) ? cur : vp_6);
                vm = cur;
            }
            hist64[qm * 16 + jm] = vm;
            __syncthreads();
        }
    }
    int do_out = (int)(num_slices == 1) | is_last;
    if (do_out != 0) {
        int k_1 = lane_1 & 15;
        int qo_local = warp_2 * 2 + (lane_1 >> 4);
        int qo = qtile * 16 + qo_local;
        unsigned int cnt_o = cand_cnt[qo_local];
        unsigned int mode_o = state[qo_local * 8];
        unsigned long long top_o = hist64[qo_local * 16];
        int ovf_o = (int)(r > 0) & ((int)(cnt_o > 64) | (int)(mode_o != 2) | (int)(top_o == ovf_c));
        int value = max_k_tiles;
        if (k_1 < fb) {
            value = k_1;
        } else if (k_1 < fb + fe) {
            value = num_valid_pages - fe + (k_1 - fb);
        } else {
            unsigned long long comp_o = hist64[qo_local * 16 + (k_1 - fb - fe)];
            if (comp_o != 0) {
                value = (int)~(unsigned int)comp_o;
            }
        }
        int _shfl_xor_292 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_56 = ((value) < (_shfl_xor_292) ? (value) : (_shfl_xor_292));
        int lo = _min_56;
        int _max_291 = ((value) > (_shfl_xor_292) ? (value) : (_shfl_xor_292));
        int hi = _max_291;
        int ascending = (int)((k_1 & 2) == 0);
        int lower_lane = (int)((k_1 & 1) == 0);
        int take_low = (int)(ascending == lower_lane);
        value = ((take_low == 1) ? lo : hi);
        int _shfl_xor_293 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_57 = ((value) < (_shfl_xor_293) ? (value) : (_shfl_xor_293));
        int lo_0 = _min_57;
        int _max_292 = ((value) > (_shfl_xor_293) ? (value) : (_shfl_xor_293));
        int hi_1 = _max_292;
        int ascending_2 = (int)((k_1 & 4) == 0);
        int lower_lane_3 = (int)((k_1 & 2) == 0);
        int take_low_4 = (int)(ascending_2 == lower_lane_3);
        value = ((take_low_4 == 1) ? lo_0 : hi_1);
        int _shfl_xor_294 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_58 = ((value) < (_shfl_xor_294) ? (value) : (_shfl_xor_294));
        int lo_5 = _min_58;
        int _max_293 = ((value) > (_shfl_xor_294) ? (value) : (_shfl_xor_294));
        int hi_6 = _max_293;
        int ascending_7 = (int)((k_1 & 4) == 0);
        int lower_lane_8 = (int)((k_1 & 1) == 0);
        int take_low_9 = (int)(ascending_7 == lower_lane_8);
        value = ((take_low_9 == 1) ? lo_5 : hi_6);
        int _shfl_xor_295 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_59 = ((value) < (_shfl_xor_295) ? (value) : (_shfl_xor_295));
        int lo_10 = _min_59;
        int _max_294 = ((value) > (_shfl_xor_295) ? (value) : (_shfl_xor_295));
        int hi_11 = _max_294;
        int ascending_12 = (int)((k_1 & 8) == 0);
        int lower_lane_13 = (int)((k_1 & 4) == 0);
        int take_low_14 = (int)(ascending_12 == lower_lane_13);
        value = ((take_low_14 == 1) ? lo_10 : hi_11);
        int _shfl_xor_296 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_60 = ((value) < (_shfl_xor_296) ? (value) : (_shfl_xor_296));
        int lo_15 = _min_60;
        int _max_295 = ((value) > (_shfl_xor_296) ? (value) : (_shfl_xor_296));
        int hi_16 = _max_295;
        int ascending_17 = (int)((k_1 & 8) == 0);
        int lower_lane_18 = (int)((k_1 & 2) == 0);
        int take_low_19 = (int)(ascending_17 == lower_lane_18);
        value = ((take_low_19 == 1) ? lo_15 : hi_16);
        int _shfl_xor_297 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_61 = ((value) < (_shfl_xor_297) ? (value) : (_shfl_xor_297));
        int lo_20 = _min_61;
        int _max_296 = ((value) > (_shfl_xor_297) ? (value) : (_shfl_xor_297));
        int hi_21 = _max_296;
        int ascending_22 = (int)((k_1 & 8) == 0);
        int lower_lane_23 = (int)((k_1 & 1) == 0);
        int take_low_24 = (int)(ascending_22 == lower_lane_23);
        value = ((take_low_24 == 1) ? lo_20 : hi_21);
        int _shfl_xor_298 = __shfl_xor_sync(0xFFFFFFFF, value, 8);
        int _min_62 = ((value) < (_shfl_xor_298) ? (value) : (_shfl_xor_298));
        int lo_25 = _min_62;
        int _max_297 = ((value) > (_shfl_xor_298) ? (value) : (_shfl_xor_298));
        int hi_26 = _max_297;
        int ascending_27 = (int)((k_1 & 16) == 0);
        int lower_lane_28 = (int)((k_1 & 8) == 0);
        int take_low_29 = (int)(ascending_27 == lower_lane_28);
        value = ((take_low_29 == 1) ? lo_25 : hi_26);
        int _shfl_xor_299 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_63 = ((value) < (_shfl_xor_299) ? (value) : (_shfl_xor_299));
        int lo_30 = _min_63;
        int _max_298 = ((value) > (_shfl_xor_299) ? (value) : (_shfl_xor_299));
        int hi_31 = _max_298;
        int ascending_32 = (int)((k_1 & 16) == 0);
        int lower_lane_33 = (int)((k_1 & 4) == 0);
        int take_low_34 = (int)(ascending_32 == lower_lane_33);
        value = ((take_low_34 == 1) ? lo_30 : hi_31);
        int _shfl_xor_300 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_64 = ((value) < (_shfl_xor_300) ? (value) : (_shfl_xor_300));
        int lo_35 = _min_64;
        int _max_299 = ((value) > (_shfl_xor_300) ? (value) : (_shfl_xor_300));
        int hi_36 = _max_299;
        int ascending_37 = (int)((k_1 & 16) == 0);
        int lower_lane_38 = (int)((k_1 & 2) == 0);
        int take_low_39 = (int)(ascending_37 == lower_lane_38);
        value = ((take_low_39 == 1) ? lo_35 : hi_36);
        int _shfl_xor_301 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_65 = ((value) < (_shfl_xor_301) ? (value) : (_shfl_xor_301));
        int lo_40 = _min_65;
        int _max_300 = ((value) > (_shfl_xor_301) ? (value) : (_shfl_xor_301));
        int hi_41 = _max_300;
        int ascending_42 = (int)((k_1 & 16) == 0);
        int lower_lane_43 = (int)((k_1 & 1) == 0);
        int take_low_44 = (int)(ascending_42 == lower_lane_43);
        value = ((take_low_44 == 1) ? lo_40 : hi_41);
        if (qo < total_q) {
            int stored = value;
            if (value >= num_valid_pages) {
                stored = -1;
            }
            if (ovf_o != 0) {
                int dbg = -2;
                if (k_1 == 1) {
                    dbg = (int)cnt_o;
                } else if (k_1 == 2) {
                    dbg = (int)state[qo_local * 8 + 1];
                } else {
                    if (k_1 == 3) {
                        dbg = (int)state[qo_local * 8 + 2];
                    } else if (k_1 == 4) {
                        dbg = (int)state[qo_local * 8 + 3];
                    } else {
                        if (k_1 == 5) {
                            dbg = (int)state[qo_local * 8 + 4];
                        } else if (k_1 == 6) {
                            dbg = (int)state[qo_local * 8 + 5];
                        } else {
                            if (k_1 == 7) {
                                dbg = (int)state[qo_local * 8];
                            }
                        }
                    }
                }
                stored = dbg;
            }
            output[(qo * num_heads + h) * 16 + k_1] = stored;
        }
    }
}

} // extern "C"
