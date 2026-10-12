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
kernel_cake_blackwell_msa_1c83041e9e73cbe650cf(float* __restrict__ max_score, int* __restrict__ output, unsigned long long* __restrict__ partials, unsigned int* __restrict__ counters, unsigned int* __restrict__ block_max, int num_heads, int max_k_tiles, int total_q, int num_valid_pages, int force_begin_blocks, int force_end_blocks, int num_qtiles, int num_slices, int slice_len, int index_bits)
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
    int bm_stride = (iters_nom + 8 - 1 >> 3) * 64;
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
    float xs[8];
    int blks[8];
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
                for (int i0 = 0; i0 < iters; i0 += 8) {
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
                    unsigned long long bm_idx = bm_base + (unsigned long long)(((i0 >> 3) * 16 + r_sub) * 4);
                    int m[8];
                    unsigned int raw = __as_u32(xs[0]);
                    unsigned int mag = raw & 2147483647;
                    unsigned int nan = (unsigned int)(mag > 2139095040);
                    unsigned int bits = ((mag != 0) ? raw : 0);
                    int em = (int)(bits >> 19 & 4095);
                    int _max_2 = ((em - 1792) > (0) ? (em - 1792) : (0));
                    int _min_6 = ((_max_2) < (511) ? (_max_2) : (511));
                    int raw_14 = _min_6;
                    int neg = (int)(bits >> 31);
                    int h10 = ((neg != 0) ? 511 - raw_14 : 512 + raw_14);
                    int hb = ((nan != 0) ? 1024 : h10);
                    atomicAdd(&hist_ext[hb * 16 + q_local], 1);
                    m[0] = ((nan != 0) ? 0 : h10);
                    unsigned int raw_15 = __as_u32(xs[1]);
                    unsigned int mag_16 = raw_15 & 2147483647;
                    unsigned int nan_17 = (unsigned int)(mag_16 > 2139095040);
                    unsigned int bits_18 = ((mag_16 != 0) ? raw_15 : 0);
                    int em_19 = (int)(bits_18 >> 19 & 4095);
                    int _max_3 = ((em_19 - 1792) > (0) ? (em_19 - 1792) : (0));
                    int _min_7 = ((_max_3) < (511) ? (_max_3) : (511));
                    int raw_20 = _min_7;
                    int neg_21 = (int)(bits_18 >> 31);
                    int h10_22 = ((neg_21 != 0) ? 511 - raw_20 : 512 + raw_20);
                    int hb_23 = ((nan_17 != 0) ? 1024 : h10_22);
                    atomicAdd(&hist_ext[hb_23 * 16 + q_local], 1);
                    m[1] = ((nan_17 != 0) ? 0 : h10_22);
                    unsigned int raw_24 = __as_u32(xs[2]);
                    unsigned int mag_25 = raw_24 & 2147483647;
                    unsigned int nan_26 = (unsigned int)(mag_25 > 2139095040);
                    unsigned int bits_27 = ((mag_25 != 0) ? raw_24 : 0);
                    int em_28 = (int)(bits_27 >> 19 & 4095);
                    int _max_4 = ((em_28 - 1792) > (0) ? (em_28 - 1792) : (0));
                    int _min_8 = ((_max_4) < (511) ? (_max_4) : (511));
                    int raw_29 = _min_8;
                    int neg_30 = (int)(bits_27 >> 31);
                    int h10_31 = ((neg_30 != 0) ? 511 - raw_29 : 512 + raw_29);
                    int hb_32 = ((nan_26 != 0) ? 1024 : h10_31);
                    atomicAdd(&hist_ext[hb_32 * 16 + q_local], 1);
                    m[2] = ((nan_26 != 0) ? 0 : h10_31);
                    unsigned int raw_33 = __as_u32(xs[3]);
                    unsigned int mag_34 = raw_33 & 2147483647;
                    unsigned int nan_35 = (unsigned int)(mag_34 > 2139095040);
                    unsigned int bits_36 = ((mag_34 != 0) ? raw_33 : 0);
                    int em_37 = (int)(bits_36 >> 19 & 4095);
                    int _max_5 = ((em_37 - 1792) > (0) ? (em_37 - 1792) : (0));
                    int _min_9 = ((_max_5) < (511) ? (_max_5) : (511));
                    int raw_38 = _min_9;
                    int neg_39 = (int)(bits_36 >> 31);
                    int h10_40 = ((neg_39 != 0) ? 511 - raw_38 : 512 + raw_38);
                    int hb_41 = ((nan_35 != 0) ? 1024 : h10_40);
                    atomicAdd(&hist_ext[hb_41 * 16 + q_local], 1);
                    m[3] = ((nan_35 != 0) ? 0 : h10_40);
                    unsigned int raw_42 = __as_u32(xs[4]);
                    unsigned int mag_43 = raw_42 & 2147483647;
                    unsigned int nan_44 = (unsigned int)(mag_43 > 2139095040);
                    unsigned int bits_45 = ((mag_43 != 0) ? raw_42 : 0);
                    int em_46 = (int)(bits_45 >> 19 & 4095);
                    int _max_6 = ((em_46 - 1792) > (0) ? (em_46 - 1792) : (0));
                    int _min_10 = ((_max_6) < (511) ? (_max_6) : (511));
                    int raw_47 = _min_10;
                    int neg_48 = (int)(bits_45 >> 31);
                    int h10_49 = ((neg_48 != 0) ? 511 - raw_47 : 512 + raw_47);
                    int hb_50 = ((nan_44 != 0) ? 1024 : h10_49);
                    atomicAdd(&hist_ext[hb_50 * 16 + q_local], 1);
                    m[4] = ((nan_44 != 0) ? 0 : h10_49);
                    unsigned int raw_51 = __as_u32(xs[5]);
                    unsigned int mag_52 = raw_51 & 2147483647;
                    unsigned int nan_53 = (unsigned int)(mag_52 > 2139095040);
                    unsigned int bits_54 = ((mag_52 != 0) ? raw_51 : 0);
                    int em_55 = (int)(bits_54 >> 19 & 4095);
                    int _max_7 = ((em_55 - 1792) > (0) ? (em_55 - 1792) : (0));
                    int _min_11 = ((_max_7) < (511) ? (_max_7) : (511));
                    int raw_56 = _min_11;
                    int neg_57 = (int)(bits_54 >> 31);
                    int h10_58 = ((neg_57 != 0) ? 511 - raw_56 : 512 + raw_56);
                    int hb_59 = ((nan_53 != 0) ? 1024 : h10_58);
                    atomicAdd(&hist_ext[hb_59 * 16 + q_local], 1);
                    m[5] = ((nan_53 != 0) ? 0 : h10_58);
                    unsigned int raw_60 = __as_u32(xs[6]);
                    unsigned int mag_61 = raw_60 & 2147483647;
                    unsigned int nan_62 = (unsigned int)(mag_61 > 2139095040);
                    unsigned int bits_63 = ((mag_61 != 0) ? raw_60 : 0);
                    int em_64 = (int)(bits_63 >> 19 & 4095);
                    int _max_8 = ((em_64 - 1792) > (0) ? (em_64 - 1792) : (0));
                    int _min_12 = ((_max_8) < (511) ? (_max_8) : (511));
                    int raw_65 = _min_12;
                    int neg_66 = (int)(bits_63 >> 31);
                    int h10_67 = ((neg_66 != 0) ? 511 - raw_65 : 512 + raw_65);
                    int hb_68 = ((nan_62 != 0) ? 1024 : h10_67);
                    atomicAdd(&hist_ext[hb_68 * 16 + q_local], 1);
                    m[6] = ((nan_62 != 0) ? 0 : h10_67);
                    unsigned int raw_69 = __as_u32(xs[7]);
                    unsigned int mag_70 = raw_69 & 2147483647;
                    unsigned int nan_71 = (unsigned int)(mag_70 > 2139095040);
                    unsigned int bits_72 = ((mag_70 != 0) ? raw_69 : 0);
                    int em_73 = (int)(bits_72 >> 19 & 4095);
                    int _max_9 = ((em_73 - 1792) > (0) ? (em_73 - 1792) : (0));
                    int _min_13 = ((_max_9) < (511) ? (_max_9) : (511));
                    int raw_74 = _min_13;
                    int neg_75 = (int)(bits_72 >> 31);
                    int h10_76 = ((neg_75 != 0) ? 511 - raw_74 : 512 + raw_74);
                    int hb_77 = ((nan_71 != 0) ? 1024 : h10_76);
                    atomicAdd(&hist_ext[hb_77 * 16 + q_local], 1);
                    m[7] = ((nan_71 != 0) ? 0 : h10_76);
                    int b3 = lane_1 & 8;
                    int b2 = lane_1 & 4;
                    int b1 = lane_1 & 2;
                    int a[4];
                    int send = ((b3 != 0) ? m[0] : m[4]);
                    int keep = ((b3 != 0) ? m[4] : m[0]);
                    int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 8);
                    int recv = _shfl_xor_0;
                    int _max_10 = ((keep) > (recv) ? (keep) : (recv));
                    a[0] = _max_10;
                    int send_78 = ((b3 != 0) ? m[1] : m[5]);
                    int keep_79 = ((b3 != 0) ? m[5] : m[1]);
                    int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_78, 8);
                    int recv_80 = _shfl_xor_1;
                    int _max_11 = ((keep_79) > (recv_80) ? (keep_79) : (recv_80));
                    a[1] = _max_11;
                    int send_81 = ((b3 != 0) ? m[2] : m[6]);
                    int keep_82 = ((b3 != 0) ? m[6] : m[2]);
                    int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_81, 8);
                    int recv_83 = _shfl_xor_2;
                    int _max_12 = ((keep_82) > (recv_83) ? (keep_82) : (recv_83));
                    a[2] = _max_12;
                    int send_84 = ((b3 != 0) ? m[3] : m[7]);
                    int keep_85 = ((b3 != 0) ? m[7] : m[3]);
                    int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_84, 8);
                    int recv_86 = _shfl_xor_3;
                    int _max_13 = ((keep_85) > (recv_86) ? (keep_85) : (recv_86));
                    a[3] = _max_13;
                    int b[2];
                    int send2 = ((b2 != 0) ? a[0] : a[2]);
                    int keep2 = ((b2 != 0) ? a[2] : a[0]);
                    int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send2, 4);
                    int recv2 = _shfl_xor_4;
                    int _max_14 = ((keep2) > (recv2) ? (keep2) : (recv2));
                    b[0] = _max_14;
                    int send2_87 = ((b2 != 0) ? a[1] : a[3]);
                    int keep2_88 = ((b2 != 0) ? a[3] : a[1]);
                    int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send2_87, 4);
                    int recv2_89 = _shfl_xor_5;
                    int _max_15 = ((keep2_88) > (recv2_89) ? (keep2_88) : (recv2_89));
                    b[1] = _max_15;
                    int send3 = ((b1 != 0) ? b[0] : b[1]);
                    int keep3 = ((b1 != 0) ? b[1] : b[0]);
                    int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send3, 2);
                    int recv3 = _shfl_xor_6;
                    int _max_16 = ((keep3) > (recv3) ? (keep3) : (recv3));
                    int c = _max_16;
                    int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, c, 1);
                    int recv4 = _shfl_xor_7;
                    int _max_17 = ((c) > (recv4) ? (c) : (recv4));
                    int d = _max_17;
                    int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, d, 2);
                    int other = _shfl_xor_8;
                    int word = ((b1 == 0) ? d | other << 16 : other | d << 16);
                    block_max[bm_idx + (unsigned long long)(lane_1 >> 2 & 3)] = (unsigned int)word;
                }
            } else if (n_hist == 0) {
                if (p > 0) {
                    int nchunks = iters + 8 - 1 >> 3;
                    for (int w0 = 0; w0 < slen; w0 += 8192) {
                        int c_lo = w0 >> 7;
                        int _min_14 = ((nchunks) < (w0 + 8192 >> 7) ? (nchunks) : (w0 + 8192 >> 7));
                        int c_hi = _min_14;
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
                            unsigned int _vec_load_0[4];
                            {
                                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_1 * 16 + r_0) * 4));
                                _vec_load_0[0 + 0] = _uv4_0.x;
                                _vec_load_0[0 + 1] = _uv4_0.y;
                                _vec_load_0[0 + 2] = _uv4_0.z;
                                _vec_load_0[0 + 3] = _uv4_0.w;
                            }
                            int mk = 0;
                            unsigned int mu = _vec_load_0[0] & 65535;
                            int j_0 = js[0];
                            int keep_1 = (int)(mu >= tmin && j_0 < slen);
                            mk = mk | keep_1;
                            unsigned int mu_1 = _vec_load_0[0] >> 16 & 65535;
                            int j_2 = js[0] + 16;
                            int keep_3 = (int)(mu_1 >= tmin && j_2 < slen);
                            mk = mk | keep_3 << 1;
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
                        int c_2 = c_lo + (e_1 >> 4);
                        int r_3 = e_1 & 15;
                        js[1] = c_2 * 128 + r_3;
                        masks[1] = 0;
                        if (e_1 < n_ent) {
                            unsigned int _vec_load_1[4];
                            {
                                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(block_max + bm_base + (unsigned long long)((c_2 * 16 + r_3) * 4));
                                _vec_load_1[0 + 0] = _uv4_1.x;
                                _vec_load_1[0 + 1] = _uv4_1.y;
                                _vec_load_1[0 + 2] = _uv4_1.z;
                                _vec_load_1[0 + 3] = _uv4_1.w;
                            }
                            int mk_1 = 0;
                            unsigned int mu_2 = _vec_load_1[0] & 65535;
                            int j_0_1 = js[1];
                            int keep_2 = (int)(mu_2 >= tmin && j_0_1 < slen);
                            mk_1 = mk_1 | keep_2;
                            unsigned int mu_1_1 = _vec_load_1[0] >> 16 & 65535;
                            int j_2_1 = js[1] + 16;
                            int keep_3_1 = (int)(mu_1_1 >= tmin && j_2_1 < slen);
                            mk_1 = mk_1 | keep_3_1 << 1;
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
                            int keep_4 = (int)(mu_3 >= tmin && j_0_2 < slen);
                            mk_2 = mk_2 | keep_4;
                            unsigned int mu_1_2 = _vec_load_2[0] >> 16 & 65535;
                            int j_2_2 = js[2] + 16;
                            int keep_3_2 = (int)(mu_1_2 >= tmin && j_2_2 < slen);
                            mk_2 = mk_2 | keep_3_2 << 1;
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
                            int keep_5 = (int)(mu_5 >= tmin && j_0_3 < slen);
                            mk_3 = mk_3 | keep_5;
                            unsigned int mu_1_3 = _vec_load_3[0] >> 16 & 65535;
                            int j_2_3 = js[3] + 16;
                            int keep_3_3 = (int)(mu_1_3 >= tmin && j_2_3 < slen);
                            mk_3 = mk_3 | keep_3_3 << 1;
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
                        int keep2_1 = masks[0] & 1;
                        int idx = ((keep2_1 != 0) ? 1024 + base : dummy);
                        hist[idx] = (unsigned int)(lo_s + js[0]);
                        base = base + keep2_1;
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
                        for (int i0_1 = 0; i0_1 < n_it; i0_1 += 8) {
                            int _max_18 = ((n_list - 1) > (0) ? (n_list - 1) : (0));
                            int last = _max_18;
                            int e_0 = i0_1 * 16 + r_sub;
                            int act_2 = (int)(e_0 < n_list) & qvalid;
                            int _min_15 = ((e_0) < (last) ? (e_0) : (last));
                            int blk_1 = (int)hist[1024 + _min_15];
                            blks[0] = blk_1;
                            unsigned long long off_1 = (row_base64 + (unsigned long long)blk_1) * total_q64 + q_addr64;
                            xs[0] = __uint_as_float(2143289344);
                            if (act_2 != 0) {
                                xs[0] = max_score[off_1];
                            }
                            int e_2 = (i0_1 + 1) * 16 + r_sub;
                            int act_3_1 = (int)(e_2 < n_list) & qvalid;
                            int _min_16 = ((e_2) < (last) ? (e_2) : (last));
                            int blk_4_1 = (int)hist[1024 + _min_16];
                            blks[1] = blk_4_1;
                            unsigned long long off_5 = (row_base64 + (unsigned long long)blk_4_1) * total_q64 + q_addr64;
                            xs[1] = __uint_as_float(2143289344);
                            if (act_3_1 != 0) {
                                xs[1] = max_score[off_5];
                            }
                            int e_6 = (i0_1 + 2) * 16 + r_sub;
                            int act_7_1 = (int)(e_6 < n_list) & qvalid;
                            int _min_17 = ((e_6) < (last) ? (e_6) : (last));
                            int blk_8_1 = (int)hist[1024 + _min_17];
                            blks[2] = blk_8_1;
                            unsigned long long off_9 = (row_base64 + (unsigned long long)blk_8_1) * total_q64 + q_addr64;
                            xs[2] = __uint_as_float(2143289344);
                            if (act_7_1 != 0) {
                                xs[2] = max_score[off_9];
                            }
                            int e_10 = (i0_1 + 3) * 16 + r_sub;
                            int act_11_1 = (int)(e_10 < n_list) & qvalid;
                            int _min_18 = ((e_10) < (last) ? (e_10) : (last));
                            int blk_12_1 = (int)hist[1024 + _min_18];
                            blks[3] = blk_12_1;
                            unsigned long long off_13 = (row_base64 + (unsigned long long)blk_12_1) * total_q64 + q_addr64;
                            xs[3] = __uint_as_float(2143289344);
                            if (act_11_1 != 0) {
                                xs[3] = max_score[off_13];
                            }
                            int e_14 = (i0_1 + 4) * 16 + r_sub;
                            int act_15 = (int)(e_14 < n_list) & qvalid;
                            int _min_19 = ((e_14) < (last) ? (e_14) : (last));
                            int blk_16 = (int)hist[1024 + _min_19];
                            blks[4] = blk_16;
                            unsigned long long off_17 = (row_base64 + (unsigned long long)blk_16) * total_q64 + q_addr64;
                            xs[4] = __uint_as_float(2143289344);
                            if (act_15 != 0) {
                                xs[4] = max_score[off_17];
                            }
                            int e_18 = (i0_1 + 5) * 16 + r_sub;
                            int act_19 = (int)(e_18 < n_list) & qvalid;
                            int _min_20 = ((e_18) < (last) ? (e_18) : (last));
                            int blk_20 = (int)hist[1024 + _min_20];
                            blks[5] = blk_20;
                            unsigned long long off_21 = (row_base64 + (unsigned long long)blk_20) * total_q64 + q_addr64;
                            xs[5] = __uint_as_float(2143289344);
                            if (act_19 != 0) {
                                xs[5] = max_score[off_21];
                            }
                            int e_22 = (i0_1 + 6) * 16 + r_sub;
                            int act_23 = (int)(e_22 < n_list) & qvalid;
                            int _min_21 = ((e_22) < (last) ? (e_22) : (last));
                            int blk_24 = (int)hist[1024 + _min_21];
                            blks[6] = blk_24;
                            unsigned long long off_25 = (row_base64 + (unsigned long long)blk_24) * total_q64 + q_addr64;
                            xs[6] = __uint_as_float(2143289344);
                            if (act_23 != 0) {
                                xs[6] = max_score[off_25];
                            }
                            int e_26 = (i0_1 + 7) * 16 + r_sub;
                            int act_27 = (int)(e_26 < n_list) & qvalid;
                            int _min_22 = ((e_26) < (last) ? (e_26) : (last));
                            int blk_28 = (int)hist[1024 + _min_22];
                            blks[7] = blk_28;
                            unsigned long long off_29 = (row_base64 + (unsigned long long)blk_28) * total_q64 + q_addr64;
                            xs[7] = __uint_as_float(2143289344);
                            if (act_27 != 0) {
                                xs[7] = max_score[off_29];
                            }
                            unsigned int raw_1 = __as_u32(xs[0]);
                            unsigned int mag_1 = raw_1 & 2147483647;
                            unsigned int nan_1 = (unsigned int)(mag_1 > 2139095040);
                            unsigned int bits_1 = ((mag_1 != 0) ? raw_1 : 0);
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
                            unsigned int raw_30 = __as_u32(xs[1]);
                            unsigned int mag_31 = raw_30 & 2147483647;
                            unsigned int nan_32 = (unsigned int)(mag_31 > 2139095040);
                            unsigned int bits_33 = ((mag_31 != 0) ? raw_30 : 0);
                            unsigned int neg_mask_34 = -(bits_33 >> 31);
                            unsigned int key_35 = bits_33 ^ (neg_mask_34 | 2147483648u);
                            unsigned int inv_36 = ~(unsigned int)blks[1];
                            unsigned long long comp_37 = (unsigned long long)key_35 << 32 | (unsigned long long)inv_36;
                            if (is_stage != 0 && nan_32 == 0 && comp_37 >= cthr_me) {
                                unsigned int _atomic_old_1 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_1 = _atomic_old_1;
                                if (slot_1 < 64) {
                                    cand[q_local * 64 + (int)slot_1] = comp_37;
                                }
                            }
                            unsigned int raw_38_1 = __as_u32(xs[2]);
                            unsigned int mag_39 = raw_38_1 & 2147483647;
                            unsigned int nan_40 = (unsigned int)(mag_39 > 2139095040);
                            unsigned int bits_41 = ((mag_39 != 0) ? raw_38_1 : 0);
                            unsigned int neg_mask_42 = -(bits_41 >> 31);
                            unsigned int key_43 = bits_41 ^ (neg_mask_42 | 2147483648u);
                            unsigned int inv_44 = ~(unsigned int)blks[2];
                            unsigned long long comp_45 = (unsigned long long)key_43 << 32 | (unsigned long long)inv_44;
                            if (is_stage != 0 && nan_40 == 0 && comp_45 >= cthr_me) {
                                unsigned int _atomic_old_2 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_2 = _atomic_old_2;
                                if (slot_2 < 64) {
                                    cand[q_local * 64 + (int)slot_2] = comp_45;
                                }
                            }
                            unsigned int raw_46 = __as_u32(xs[3]);
                            unsigned int mag_47 = raw_46 & 2147483647;
                            unsigned int nan_48 = (unsigned int)(mag_47 > 2139095040);
                            unsigned int bits_49 = ((mag_47 != 0) ? raw_46 : 0);
                            unsigned int neg_mask_50 = -(bits_49 >> 31);
                            unsigned int key_51 = bits_49 ^ (neg_mask_50 | 2147483648u);
                            unsigned int inv_52 = ~(unsigned int)blks[3];
                            unsigned long long comp_53 = (unsigned long long)key_51 << 32 | (unsigned long long)inv_52;
                            if (is_stage != 0 && nan_48 == 0 && comp_53 >= cthr_me) {
                                unsigned int _atomic_old_3 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_3 = _atomic_old_3;
                                if (slot_3 < 64) {
                                    cand[q_local * 64 + (int)slot_3] = comp_53;
                                }
                            }
                            unsigned int raw_54 = __as_u32(xs[4]);
                            unsigned int mag_55 = raw_54 & 2147483647;
                            unsigned int nan_56 = (unsigned int)(mag_55 > 2139095040);
                            unsigned int bits_57 = ((mag_55 != 0) ? raw_54 : 0);
                            unsigned int neg_mask_58 = -(bits_57 >> 31);
                            unsigned int key_59 = bits_57 ^ (neg_mask_58 | 2147483648u);
                            unsigned int inv_60 = ~(unsigned int)blks[4];
                            unsigned long long comp_61 = (unsigned long long)key_59 << 32 | (unsigned long long)inv_60;
                            if (is_stage != 0 && nan_56 == 0 && comp_61 >= cthr_me) {
                                unsigned int _atomic_old_4 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_4 = _atomic_old_4;
                                if (slot_4 < 64) {
                                    cand[q_local * 64 + (int)slot_4] = comp_61;
                                }
                            }
                            unsigned int raw_62 = __as_u32(xs[5]);
                            unsigned int mag_63 = raw_62 & 2147483647;
                            unsigned int nan_64 = (unsigned int)(mag_63 > 2139095040);
                            unsigned int bits_65 = ((mag_63 != 0) ? raw_62 : 0);
                            unsigned int neg_mask_66 = -(bits_65 >> 31);
                            unsigned int key_67 = bits_65 ^ (neg_mask_66 | 2147483648u);
                            unsigned int inv_68 = ~(unsigned int)blks[5];
                            unsigned long long comp_69 = (unsigned long long)key_67 << 32 | (unsigned long long)inv_68;
                            if (is_stage != 0 && nan_64 == 0 && comp_69 >= cthr_me) {
                                unsigned int _atomic_old_5 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_5 = _atomic_old_5;
                                if (slot_5 < 64) {
                                    cand[q_local * 64 + (int)slot_5] = comp_69;
                                }
                            }
                            unsigned int raw_70 = __as_u32(xs[6]);
                            unsigned int mag_71 = raw_70 & 2147483647;
                            unsigned int nan_72 = (unsigned int)(mag_71 > 2139095040);
                            unsigned int bits_73 = ((mag_71 != 0) ? raw_70 : 0);
                            unsigned int neg_mask_74 = -(bits_73 >> 31);
                            unsigned int key_75 = bits_73 ^ (neg_mask_74 | 2147483648u);
                            unsigned int inv_76 = ~(unsigned int)blks[6];
                            unsigned long long comp_77 = (unsigned long long)key_75 << 32 | (unsigned long long)inv_76;
                            if (is_stage != 0 && nan_72 == 0 && comp_77 >= cthr_me) {
                                unsigned int _atomic_old_6 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_6 = _atomic_old_6;
                                if (slot_6 < 64) {
                                    cand[q_local * 64 + (int)slot_6] = comp_77;
                                }
                            }
                            unsigned int raw_78 = __as_u32(xs[7]);
                            unsigned int mag_79 = raw_78 & 2147483647;
                            unsigned int nan_80 = (unsigned int)(mag_79 > 2139095040);
                            unsigned int bits_81 = ((mag_79 != 0) ? raw_78 : 0);
                            unsigned int neg_mask_82 = -(bits_81 >> 31);
                            unsigned int key_83 = bits_81 ^ (neg_mask_82 | 2147483648u);
                            unsigned int inv_84 = ~(unsigned int)blks[7];
                            unsigned long long comp_85 = (unsigned long long)key_83 << 32 | (unsigned long long)inv_84;
                            if (is_stage != 0 && nan_80 == 0 && comp_85 >= cthr_me) {
                                unsigned int _atomic_old_7 = atomicAdd(&cand_cnt[q_local], 1);
                                unsigned int slot_7 = _atomic_old_7;
                                if (slot_7 < 64) {
                                    cand[q_local * 64 + (int)slot_7] = comp_85;
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
                        unsigned int raw_2 = __as_u32(xs[0]);
                        unsigned int mag_2 = raw_2 & 2147483647;
                        unsigned int nan_2 = (unsigned int)(mag_2 > 2139095040);
                        unsigned int bits_2 = ((mag_2 != 0) ? raw_2 : 0);
                        unsigned int neg_mask_1 = -(bits_2 >> 31);
                        unsigned int key_1 = bits_2 ^ (neg_mask_1 | 2147483648u);
                        unsigned int inv_1 = ~(unsigned int)blks[0];
                        unsigned long long comp_1 = (unsigned long long)key_1 << 32 | (unsigned long long)inv_1;
                        if (is_stage != 0 && nan_2 == 0 && comp_1 >= cthr_me) {
                            unsigned int _atomic_old_8 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_8 = _atomic_old_8;
                            if (slot_8 < 64) {
                                cand[q_local * 64 + (int)slot_8] = comp_1;
                            }
                        }
                        unsigned int raw_14_1 = __as_u32(xs[1]);
                        unsigned int mag_15 = raw_14_1 & 2147483647;
                        unsigned int nan_16 = (unsigned int)(mag_15 > 2139095040);
                        unsigned int bits_17 = ((mag_15 != 0) ? raw_14_1 : 0);
                        unsigned int neg_mask_18 = -(bits_17 >> 31);
                        unsigned int key_19 = bits_17 ^ (neg_mask_18 | 2147483648u);
                        unsigned int inv_20 = ~(unsigned int)blks[1];
                        unsigned long long comp_21 = (unsigned long long)key_19 << 32 | (unsigned long long)inv_20;
                        if (is_stage != 0 && nan_16 == 0 && comp_21 >= cthr_me) {
                            unsigned int _atomic_old_9 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_9 = _atomic_old_9;
                            if (slot_9 < 64) {
                                cand[q_local * 64 + (int)slot_9] = comp_21;
                            }
                        }
                        unsigned int raw_22 = __as_u32(xs[2]);
                        unsigned int mag_23 = raw_22 & 2147483647;
                        unsigned int nan_24 = (unsigned int)(mag_23 > 2139095040);
                        unsigned int bits_25 = ((mag_23 != 0) ? raw_22 : 0);
                        unsigned int neg_mask_26 = -(bits_25 >> 31);
                        unsigned int key_27 = bits_25 ^ (neg_mask_26 | 2147483648u);
                        unsigned int inv_28 = ~(unsigned int)blks[2];
                        unsigned long long comp_29 = (unsigned long long)key_27 << 32 | (unsigned long long)inv_28;
                        if (is_stage != 0 && nan_24 == 0 && comp_29 >= cthr_me) {
                            unsigned int _atomic_old_10 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_10 = _atomic_old_10;
                            if (slot_10 < 64) {
                                cand[q_local * 64 + (int)slot_10] = comp_29;
                            }
                        }
                        unsigned int raw_30_1 = __as_u32(xs[3]);
                        unsigned int mag_31_1 = raw_30_1 & 2147483647;
                        unsigned int nan_32_1 = (unsigned int)(mag_31_1 > 2139095040);
                        unsigned int bits_33_1 = ((mag_31_1 != 0) ? raw_30_1 : 0);
                        unsigned int neg_mask_34_1 = -(bits_33_1 >> 31);
                        unsigned int key_35_1 = bits_33_1 ^ (neg_mask_34_1 | 2147483648u);
                        unsigned int inv_36_1 = ~(unsigned int)blks[3];
                        unsigned long long comp_37_1 = (unsigned long long)key_35_1 << 32 | (unsigned long long)inv_36_1;
                        if (is_stage != 0 && nan_32_1 == 0 && comp_37_1 >= cthr_me) {
                            unsigned int _atomic_old_11 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_11 = _atomic_old_11;
                            if (slot_11 < 64) {
                                cand[q_local * 64 + (int)slot_11] = comp_37_1;
                            }
                        }
                        unsigned int raw_38_2 = __as_u32(xs[4]);
                        unsigned int mag_39_1 = raw_38_2 & 2147483647;
                        unsigned int nan_40_1 = (unsigned int)(mag_39_1 > 2139095040);
                        unsigned int bits_41_1 = ((mag_39_1 != 0) ? raw_38_2 : 0);
                        unsigned int neg_mask_42_1 = -(bits_41_1 >> 31);
                        unsigned int key_43_1 = bits_41_1 ^ (neg_mask_42_1 | 2147483648u);
                        unsigned int inv_44_1 = ~(unsigned int)blks[4];
                        unsigned long long comp_45_1 = (unsigned long long)key_43_1 << 32 | (unsigned long long)inv_44_1;
                        if (is_stage != 0 && nan_40_1 == 0 && comp_45_1 >= cthr_me) {
                            unsigned int _atomic_old_12 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_12 = _atomic_old_12;
                            if (slot_12 < 64) {
                                cand[q_local * 64 + (int)slot_12] = comp_45_1;
                            }
                        }
                        unsigned int raw_46_1 = __as_u32(xs[5]);
                        unsigned int mag_47_1 = raw_46_1 & 2147483647;
                        unsigned int nan_48_1 = (unsigned int)(mag_47_1 > 2139095040);
                        unsigned int bits_49_1 = ((mag_47_1 != 0) ? raw_46_1 : 0);
                        unsigned int neg_mask_50_1 = -(bits_49_1 >> 31);
                        unsigned int key_51_1 = bits_49_1 ^ (neg_mask_50_1 | 2147483648u);
                        unsigned int inv_52_1 = ~(unsigned int)blks[5];
                        unsigned long long comp_53_1 = (unsigned long long)key_51_1 << 32 | (unsigned long long)inv_52_1;
                        if (is_stage != 0 && nan_48_1 == 0 && comp_53_1 >= cthr_me) {
                            unsigned int _atomic_old_13 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_13 = _atomic_old_13;
                            if (slot_13 < 64) {
                                cand[q_local * 64 + (int)slot_13] = comp_53_1;
                            }
                        }
                        unsigned int raw_54_1 = __as_u32(xs[6]);
                        unsigned int mag_55_1 = raw_54_1 & 2147483647;
                        unsigned int nan_56_1 = (unsigned int)(mag_55_1 > 2139095040);
                        unsigned int bits_57_1 = ((mag_55_1 != 0) ? raw_54_1 : 0);
                        unsigned int neg_mask_58_1 = -(bits_57_1 >> 31);
                        unsigned int key_59_1 = bits_57_1 ^ (neg_mask_58_1 | 2147483648u);
                        unsigned int inv_60_1 = ~(unsigned int)blks[6];
                        unsigned long long comp_61_1 = (unsigned long long)key_59_1 << 32 | (unsigned long long)inv_60_1;
                        if (is_stage != 0 && nan_56_1 == 0 && comp_61_1 >= cthr_me) {
                            unsigned int _atomic_old_14 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_14 = _atomic_old_14;
                            if (slot_14 < 64) {
                                cand[q_local * 64 + (int)slot_14] = comp_61_1;
                            }
                        }
                        unsigned int raw_62_1 = __as_u32(xs[7]);
                        unsigned int mag_63_1 = raw_62_1 & 2147483647;
                        unsigned int nan_64_1 = (unsigned int)(mag_63_1 > 2139095040);
                        unsigned int bits_65_1 = ((mag_63_1 != 0) ? raw_62_1 : 0);
                        unsigned int neg_mask_66_1 = -(bits_65_1 >> 31);
                        unsigned int key_67_1 = bits_65_1 ^ (neg_mask_66_1 | 2147483648u);
                        unsigned int inv_68_1 = ~(unsigned int)blks[7];
                        unsigned long long comp_69_1 = (unsigned long long)key_67_1 << 32 | (unsigned long long)inv_68_1;
                        if (is_stage != 0 && nan_64_1 == 0 && comp_69_1 >= cthr_me) {
                            unsigned int _atomic_old_15 = atomicAdd(&cand_cnt[q_local], 1);
                            unsigned int slot_15 = _atomic_old_15;
                            if (slot_15 < 64) {
                                cand[q_local * 64 + (int)slot_15] = comp_69_1;
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
                    unsigned int raw_3 = __as_u32(xs[0]);
                    unsigned int mag_3 = raw_3 & 2147483647;
                    unsigned int nan_3 = (unsigned int)(mag_3 > 2139095040);
                    unsigned int bits_3 = ((mag_3 != 0) ? raw_3 : 0);
                    int em_1 = (int)(bits_3 >> 19 & 4095);
                    int _max_19 = ((em_1 - 1792) > (0) ? (em_1 - 1792) : (0));
                    int _min_23 = ((_max_19) < (511) ? (_max_19) : (511));
                    int raw_14_2 = _min_23;
                    int neg_1 = (int)(bits_3 >> 31);
                    int h10_1 = ((neg_1 != 0) ? 511 - raw_14_2 : 512 + raw_14_2);
                    unsigned int neg_mask_2 = -(bits_3 >> 31);
                    unsigned int key_2 = bits_3 ^ (neg_mask_2 | 2147483648u);
                    int blk_15 = blks[0];
                    unsigned int invm = ~(unsigned int)blk_15 & idx_mask;
                    unsigned long long rk = (unsigned long long)h10_1 << 54 | (unsigned long long)key_2 << 22 | (unsigned long long)invm;
                    int in_window = is_pass0 | (int)(rk >> rsh_m == rprefix_me);
                    int d_1 = (int)((unsigned int)(rk >> dsh_me) & dmask_me);
                    int counted = is_hist & (int)(nan_3 == 0) & in_window;
                    int dd = ((counted != 0) ? d_1 : 1024);
                    atomicAdd(&hist_ext[dd * 16 + q_local], 1);
                    unsigned int inv_2 = ~(unsigned int)blk_15;
                    unsigned long long comp_2 = (unsigned long long)key_2 << 32 | (unsigned long long)inv_2;
                    if (is_stage != 0 && nan_3 == 0 && comp_2 >= cthr_me) {
                        unsigned int _atomic_old_16 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_16 = _atomic_old_16;
                        if (slot_16 < 64) {
                            cand[q_local * 64 + (int)slot_16] = comp_2;
                        }
                    }
                    unsigned int raw_16 = __as_u32(xs[1]);
                    unsigned int mag_17 = raw_16 & 2147483647;
                    unsigned int nan_18 = (unsigned int)(mag_17 > 2139095040);
                    unsigned int bits_19 = ((mag_17 != 0) ? raw_16 : 0);
                    int em_20 = (int)(bits_19 >> 19 & 4095);
                    int _max_20 = ((em_20 - 1792) > (0) ? (em_20 - 1792) : (0));
                    int _min_24 = ((_max_20) < (511) ? (_max_20) : (511));
                    int raw_21 = _min_24;
                    int neg_22 = (int)(bits_19 >> 31);
                    int h10_23 = ((neg_22 != 0) ? 511 - raw_21 : 512 + raw_21);
                    unsigned int neg_mask_24 = -(bits_19 >> 31);
                    unsigned int key_25 = bits_19 ^ (neg_mask_24 | 2147483648u);
                    int blk_26 = blks[1];
                    unsigned int invm_27 = ~(unsigned int)blk_26 & idx_mask;
                    unsigned long long rk_28 = (unsigned long long)h10_23 << 54 | (unsigned long long)key_25 << 22 | (unsigned long long)invm_27;
                    int in_window_29 = is_pass0 | (int)(rk_28 >> rsh_m == rprefix_me);
                    int d_30 = (int)((unsigned int)(rk_28 >> dsh_me) & dmask_me);
                    int counted_31 = is_hist & (int)(nan_18 == 0) & in_window_29;
                    int dd_32 = ((counted_31 != 0) ? d_30 : 1024);
                    atomicAdd(&hist_ext[dd_32 * 16 + q_local], 1);
                    unsigned int inv_33 = ~(unsigned int)blk_26;
                    unsigned long long comp_34 = (unsigned long long)key_25 << 32 | (unsigned long long)inv_33;
                    if (is_stage != 0 && nan_18 == 0 && comp_34 >= cthr_me) {
                        unsigned int _atomic_old_17 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_17 = _atomic_old_17;
                        if (slot_17 < 64) {
                            cand[q_local * 64 + (int)slot_17] = comp_34;
                        }
                    }
                    unsigned int raw_35 = __as_u32(xs[2]);
                    unsigned int mag_36 = raw_35 & 2147483647;
                    unsigned int nan_37 = (unsigned int)(mag_36 > 2139095040);
                    unsigned int bits_38 = ((mag_36 != 0) ? raw_35 : 0);
                    int em_39 = (int)(bits_38 >> 19 & 4095);
                    int _max_21 = ((em_39 - 1792) > (0) ? (em_39 - 1792) : (0));
                    int _min_25 = ((_max_21) < (511) ? (_max_21) : (511));
                    int raw_40 = _min_25;
                    int neg_41 = (int)(bits_38 >> 31);
                    int h10_42 = ((neg_41 != 0) ? 511 - raw_40 : 512 + raw_40);
                    unsigned int neg_mask_43 = -(bits_38 >> 31);
                    unsigned int key_44 = bits_38 ^ (neg_mask_43 | 2147483648u);
                    int blk_45 = blks[2];
                    unsigned int invm_46 = ~(unsigned int)blk_45 & idx_mask;
                    unsigned long long rk_47 = (unsigned long long)h10_42 << 54 | (unsigned long long)key_44 << 22 | (unsigned long long)invm_46;
                    int in_window_48 = is_pass0 | (int)(rk_47 >> rsh_m == rprefix_me);
                    int d_49 = (int)((unsigned int)(rk_47 >> dsh_me) & dmask_me);
                    int counted_50 = is_hist & (int)(nan_37 == 0) & in_window_48;
                    int dd_51 = ((counted_50 != 0) ? d_49 : 1024);
                    atomicAdd(&hist_ext[dd_51 * 16 + q_local], 1);
                    unsigned int inv_52_2 = ~(unsigned int)blk_45;
                    unsigned long long comp_53_2 = (unsigned long long)key_44 << 32 | (unsigned long long)inv_52_2;
                    if (is_stage != 0 && nan_37 == 0 && comp_53_2 >= cthr_me) {
                        unsigned int _atomic_old_18 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_18 = _atomic_old_18;
                        if (slot_18 < 64) {
                            cand[q_local * 64 + (int)slot_18] = comp_53_2;
                        }
                    }
                    unsigned int raw_54_2 = __as_u32(xs[3]);
                    unsigned int mag_55_2 = raw_54_2 & 2147483647;
                    unsigned int nan_56_2 = (unsigned int)(mag_55_2 > 2139095040);
                    unsigned int bits_57_2 = ((mag_55_2 != 0) ? raw_54_2 : 0);
                    int em_58 = (int)(bits_57_2 >> 19 & 4095);
                    int _max_22 = ((em_58 - 1792) > (0) ? (em_58 - 1792) : (0));
                    int _min_26 = ((_max_22) < (511) ? (_max_22) : (511));
                    int raw_59 = _min_26;
                    int neg_60 = (int)(bits_57_2 >> 31);
                    int h10_61 = ((neg_60 != 0) ? 511 - raw_59 : 512 + raw_59);
                    unsigned int neg_mask_62 = -(bits_57_2 >> 31);
                    unsigned int key_63 = bits_57_2 ^ (neg_mask_62 | 2147483648u);
                    int blk_64 = blks[3];
                    unsigned int invm_65 = ~(unsigned int)blk_64 & idx_mask;
                    unsigned long long rk_66 = (unsigned long long)h10_61 << 54 | (unsigned long long)key_63 << 22 | (unsigned long long)invm_65;
                    int in_window_67 = is_pass0 | (int)(rk_66 >> rsh_m == rprefix_me);
                    int d_68 = (int)((unsigned int)(rk_66 >> dsh_me) & dmask_me);
                    int counted_69 = is_hist & (int)(nan_56_2 == 0) & in_window_67;
                    int dd_70 = ((counted_69 != 0) ? d_68 : 1024);
                    atomicAdd(&hist_ext[dd_70 * 16 + q_local], 1);
                    unsigned int inv_71 = ~(unsigned int)blk_64;
                    unsigned long long comp_72 = (unsigned long long)key_63 << 32 | (unsigned long long)inv_71;
                    if (is_stage != 0 && nan_56_2 == 0 && comp_72 >= cthr_me) {
                        unsigned int _atomic_old_19 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_19 = _atomic_old_19;
                        if (slot_19 < 64) {
                            cand[q_local * 64 + (int)slot_19] = comp_72;
                        }
                    }
                    unsigned int raw_73 = __as_u32(xs[4]);
                    unsigned int mag_74 = raw_73 & 2147483647;
                    unsigned int nan_75 = (unsigned int)(mag_74 > 2139095040);
                    unsigned int bits_76 = ((mag_74 != 0) ? raw_73 : 0);
                    int em_77 = (int)(bits_76 >> 19 & 4095);
                    int _max_23 = ((em_77 - 1792) > (0) ? (em_77 - 1792) : (0));
                    int _min_27 = ((_max_23) < (511) ? (_max_23) : (511));
                    int raw_78_1 = _min_27;
                    int neg_79 = (int)(bits_76 >> 31);
                    int h10_80 = ((neg_79 != 0) ? 511 - raw_78_1 : 512 + raw_78_1);
                    unsigned int neg_mask_81 = -(bits_76 >> 31);
                    unsigned int key_82 = bits_76 ^ (neg_mask_81 | 2147483648u);
                    int blk_83 = blks[4];
                    unsigned int invm_84 = ~(unsigned int)blk_83 & idx_mask;
                    unsigned long long rk_85 = (unsigned long long)h10_80 << 54 | (unsigned long long)key_82 << 22 | (unsigned long long)invm_84;
                    int in_window_86 = is_pass0 | (int)(rk_85 >> rsh_m == rprefix_me);
                    int d_87 = (int)((unsigned int)(rk_85 >> dsh_me) & dmask_me);
                    int counted_88 = is_hist & (int)(nan_75 == 0) & in_window_86;
                    int dd_89 = ((counted_88 != 0) ? d_87 : 1024);
                    atomicAdd(&hist_ext[dd_89 * 16 + q_local], 1);
                    unsigned int inv_90 = ~(unsigned int)blk_83;
                    unsigned long long comp_91 = (unsigned long long)key_82 << 32 | (unsigned long long)inv_90;
                    if (is_stage != 0 && nan_75 == 0 && comp_91 >= cthr_me) {
                        unsigned int _atomic_old_20 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_20 = _atomic_old_20;
                        if (slot_20 < 64) {
                            cand[q_local * 64 + (int)slot_20] = comp_91;
                        }
                    }
                    unsigned int raw_92 = __as_u32(xs[5]);
                    unsigned int mag_93 = raw_92 & 2147483647;
                    unsigned int nan_94 = (unsigned int)(mag_93 > 2139095040);
                    unsigned int bits_95 = ((mag_93 != 0) ? raw_92 : 0);
                    int em_96 = (int)(bits_95 >> 19 & 4095);
                    int _max_24 = ((em_96 - 1792) > (0) ? (em_96 - 1792) : (0));
                    int _min_28 = ((_max_24) < (511) ? (_max_24) : (511));
                    int raw_97 = _min_28;
                    int neg_98 = (int)(bits_95 >> 31);
                    int h10_99 = ((neg_98 != 0) ? 511 - raw_97 : 512 + raw_97);
                    unsigned int neg_mask_100 = -(bits_95 >> 31);
                    unsigned int key_101 = bits_95 ^ (neg_mask_100 | 2147483648u);
                    int blk_102 = blks[5];
                    unsigned int invm_103 = ~(unsigned int)blk_102 & idx_mask;
                    unsigned long long rk_104 = (unsigned long long)h10_99 << 54 | (unsigned long long)key_101 << 22 | (unsigned long long)invm_103;
                    int in_window_105 = is_pass0 | (int)(rk_104 >> rsh_m == rprefix_me);
                    int d_106 = (int)((unsigned int)(rk_104 >> dsh_me) & dmask_me);
                    int counted_107 = is_hist & (int)(nan_94 == 0) & in_window_105;
                    int dd_108 = ((counted_107 != 0) ? d_106 : 1024);
                    atomicAdd(&hist_ext[dd_108 * 16 + q_local], 1);
                    unsigned int inv_109 = ~(unsigned int)blk_102;
                    unsigned long long comp_110 = (unsigned long long)key_101 << 32 | (unsigned long long)inv_109;
                    if (is_stage != 0 && nan_94 == 0 && comp_110 >= cthr_me) {
                        unsigned int _atomic_old_21 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_21 = _atomic_old_21;
                        if (slot_21 < 64) {
                            cand[q_local * 64 + (int)slot_21] = comp_110;
                        }
                    }
                    unsigned int raw_111 = __as_u32(xs[6]);
                    unsigned int mag_112 = raw_111 & 2147483647;
                    unsigned int nan_113 = (unsigned int)(mag_112 > 2139095040);
                    unsigned int bits_114 = ((mag_112 != 0) ? raw_111 : 0);
                    int em_115 = (int)(bits_114 >> 19 & 4095);
                    int _max_25 = ((em_115 - 1792) > (0) ? (em_115 - 1792) : (0));
                    int _min_29 = ((_max_25) < (511) ? (_max_25) : (511));
                    int raw_116 = _min_29;
                    int neg_117 = (int)(bits_114 >> 31);
                    int h10_118 = ((neg_117 != 0) ? 511 - raw_116 : 512 + raw_116);
                    unsigned int neg_mask_119 = -(bits_114 >> 31);
                    unsigned int key_120 = bits_114 ^ (neg_mask_119 | 2147483648u);
                    int blk_121 = blks[6];
                    unsigned int invm_122 = ~(unsigned int)blk_121 & idx_mask;
                    unsigned long long rk_123 = (unsigned long long)h10_118 << 54 | (unsigned long long)key_120 << 22 | (unsigned long long)invm_122;
                    int in_window_124 = is_pass0 | (int)(rk_123 >> rsh_m == rprefix_me);
                    int d_125 = (int)((unsigned int)(rk_123 >> dsh_me) & dmask_me);
                    int counted_126 = is_hist & (int)(nan_113 == 0) & in_window_124;
                    int dd_127 = ((counted_126 != 0) ? d_125 : 1024);
                    atomicAdd(&hist_ext[dd_127 * 16 + q_local], 1);
                    unsigned int inv_128 = ~(unsigned int)blk_121;
                    unsigned long long comp_129 = (unsigned long long)key_120 << 32 | (unsigned long long)inv_128;
                    if (is_stage != 0 && nan_113 == 0 && comp_129 >= cthr_me) {
                        unsigned int _atomic_old_22 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_22 = _atomic_old_22;
                        if (slot_22 < 64) {
                            cand[q_local * 64 + (int)slot_22] = comp_129;
                        }
                    }
                    unsigned int raw_130 = __as_u32(xs[7]);
                    unsigned int mag_131 = raw_130 & 2147483647;
                    unsigned int nan_132 = (unsigned int)(mag_131 > 2139095040);
                    unsigned int bits_133 = ((mag_131 != 0) ? raw_130 : 0);
                    int em_134 = (int)(bits_133 >> 19 & 4095);
                    int _max_26 = ((em_134 - 1792) > (0) ? (em_134 - 1792) : (0));
                    int _min_30 = ((_max_26) < (511) ? (_max_26) : (511));
                    int raw_135 = _min_30;
                    int neg_136 = (int)(bits_133 >> 31);
                    int h10_137 = ((neg_136 != 0) ? 511 - raw_135 : 512 + raw_135);
                    unsigned int neg_mask_138 = -(bits_133 >> 31);
                    unsigned int key_139 = bits_133 ^ (neg_mask_138 | 2147483648u);
                    int blk_140 = blks[7];
                    unsigned int invm_141 = ~(unsigned int)blk_140 & idx_mask;
                    unsigned long long rk_142 = (unsigned long long)h10_137 << 54 | (unsigned long long)key_139 << 22 | (unsigned long long)invm_141;
                    int in_window_143 = is_pass0 | (int)(rk_142 >> rsh_m == rprefix_me);
                    int d_144 = (int)((unsigned int)(rk_142 >> dsh_me) & dmask_me);
                    int counted_145 = is_hist & (int)(nan_132 == 0) & in_window_143;
                    int dd_146 = ((counted_145 != 0) ? d_144 : 1024);
                    atomicAdd(&hist_ext[dd_146 * 16 + q_local], 1);
                    unsigned int inv_147 = ~(unsigned int)blk_140;
                    unsigned long long comp_148 = (unsigned long long)key_139 << 32 | (unsigned long long)inv_147;
                    if (is_stage != 0 && nan_132 == 0 && comp_148 >= cthr_me) {
                        unsigned int _atomic_old_23 = atomicAdd(&cand_cnt[q_local], 1);
                        unsigned int slot_23 = _atomic_old_23;
                        if (slot_23 < 64) {
                            cand[q_local * 64 + (int)slot_23] = comp_148;
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
                unsigned int c_2_1 = hist[base_1 + 48];
                subs[0] = subs[0] + c_2_1;
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
    unsigned int _min_31 = ((cnt_me) < (64) ? (cnt_me) : (64));
    int n_me = (int)_min_31;
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
                unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
                unsigned long long vp = _shfl_xor_9;
                int take_max = (int)((jm & 8) == 0);
                int gt = (int)(cur > vp);
                cur = ((gt == take_max) ? cur : vp);
                unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
                unsigned long long vp_0 = _shfl_xor_10;
                int take_max_1 = (int)((jm & 4) == 0);
                int gt_2 = (int)(cur > vp_0);
                cur = ((gt_2 == take_max_1) ? cur : vp_0);
                unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
                unsigned long long vp_3 = _shfl_xor_11;
                int take_max_4 = (int)((jm & 2) == 0);
                int gt_5 = (int)(cur > vp_3);
                cur = ((gt_5 == take_max_4) ? cur : vp_3);
                unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
                unsigned long long vp_6 = _shfl_xor_12;
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
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_32 = ((value) < (_shfl_xor_13) ? (value) : (_shfl_xor_13));
        int lo = _min_32;
        int _max_27 = ((value) > (_shfl_xor_13) ? (value) : (_shfl_xor_13));
        int hi = _max_27;
        int ascending = (int)((k_1 & 2) == 0);
        int lower_lane = (int)((k_1 & 1) == 0);
        int take_low = (int)(ascending == lower_lane);
        value = ((take_low == 1) ? lo : hi);
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_33 = ((value) < (_shfl_xor_14) ? (value) : (_shfl_xor_14));
        int lo_0 = _min_33;
        int _max_28 = ((value) > (_shfl_xor_14) ? (value) : (_shfl_xor_14));
        int hi_1 = _max_28;
        int ascending_2 = (int)((k_1 & 4) == 0);
        int lower_lane_3 = (int)((k_1 & 2) == 0);
        int take_low_4 = (int)(ascending_2 == lower_lane_3);
        value = ((take_low_4 == 1) ? lo_0 : hi_1);
        int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_34 = ((value) < (_shfl_xor_15) ? (value) : (_shfl_xor_15));
        int lo_5 = _min_34;
        int _max_29 = ((value) > (_shfl_xor_15) ? (value) : (_shfl_xor_15));
        int hi_6 = _max_29;
        int ascending_7 = (int)((k_1 & 4) == 0);
        int lower_lane_8 = (int)((k_1 & 1) == 0);
        int take_low_9 = (int)(ascending_7 == lower_lane_8);
        value = ((take_low_9 == 1) ? lo_5 : hi_6);
        int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_35 = ((value) < (_shfl_xor_16) ? (value) : (_shfl_xor_16));
        int lo_10 = _min_35;
        int _max_30 = ((value) > (_shfl_xor_16) ? (value) : (_shfl_xor_16));
        int hi_11 = _max_30;
        int ascending_12 = (int)((k_1 & 8) == 0);
        int lower_lane_13 = (int)((k_1 & 4) == 0);
        int take_low_14 = (int)(ascending_12 == lower_lane_13);
        value = ((take_low_14 == 1) ? lo_10 : hi_11);
        int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_36 = ((value) < (_shfl_xor_17) ? (value) : (_shfl_xor_17));
        int lo_15 = _min_36;
        int _max_31 = ((value) > (_shfl_xor_17) ? (value) : (_shfl_xor_17));
        int hi_16 = _max_31;
        int ascending_17 = (int)((k_1 & 8) == 0);
        int lower_lane_18 = (int)((k_1 & 2) == 0);
        int take_low_19 = (int)(ascending_17 == lower_lane_18);
        value = ((take_low_19 == 1) ? lo_15 : hi_16);
        int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_37 = ((value) < (_shfl_xor_18) ? (value) : (_shfl_xor_18));
        int lo_20 = _min_37;
        int _max_32 = ((value) > (_shfl_xor_18) ? (value) : (_shfl_xor_18));
        int hi_21 = _max_32;
        int ascending_22 = (int)((k_1 & 8) == 0);
        int lower_lane_23 = (int)((k_1 & 1) == 0);
        int take_low_24 = (int)(ascending_22 == lower_lane_23);
        value = ((take_low_24 == 1) ? lo_20 : hi_21);
        int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, value, 8);
        int _min_38 = ((value) < (_shfl_xor_19) ? (value) : (_shfl_xor_19));
        int lo_25 = _min_38;
        int _max_33 = ((value) > (_shfl_xor_19) ? (value) : (_shfl_xor_19));
        int hi_26 = _max_33;
        int ascending_27 = (int)((k_1 & 16) == 0);
        int lower_lane_28 = (int)((k_1 & 8) == 0);
        int take_low_29 = (int)(ascending_27 == lower_lane_28);
        value = ((take_low_29 == 1) ? lo_25 : hi_26);
        int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, value, 4);
        int _min_39 = ((value) < (_shfl_xor_20) ? (value) : (_shfl_xor_20));
        int lo_30 = _min_39;
        int _max_34 = ((value) > (_shfl_xor_20) ? (value) : (_shfl_xor_20));
        int hi_31 = _max_34;
        int ascending_32 = (int)((k_1 & 16) == 0);
        int lower_lane_33 = (int)((k_1 & 4) == 0);
        int take_low_34 = (int)(ascending_32 == lower_lane_33);
        value = ((take_low_34 == 1) ? lo_30 : hi_31);
        int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, value, 2);
        int _min_40 = ((value) < (_shfl_xor_21) ? (value) : (_shfl_xor_21));
        int lo_35 = _min_40;
        int _max_35 = ((value) > (_shfl_xor_21) ? (value) : (_shfl_xor_21));
        int hi_36 = _max_35;
        int ascending_37 = (int)((k_1 & 16) == 0);
        int lower_lane_38 = (int)((k_1 & 2) == 0);
        int take_low_39 = (int)(ascending_37 == lower_lane_38);
        value = ((take_low_39 == 1) ? lo_35 : hi_36);
        int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, value, 1);
        int _min_41 = ((value) < (_shfl_xor_22) ? (value) : (_shfl_xor_22));
        int lo_40 = _min_41;
        int _max_36 = ((value) > (_shfl_xor_22) ? (value) : (_shfl_xor_22));
        int hi_41 = _max_36;
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
