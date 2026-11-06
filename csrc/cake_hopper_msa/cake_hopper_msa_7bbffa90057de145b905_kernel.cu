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

#define CAKE_WARP_UNIFORM_INT 1
#include "cake_hopper_msa_device.h"

#define NUM_MAIN_1_STAGES 1
#define SMEM_SCR_OFF 1024
#define SMEM_SCR_STAGE_BYTES 8704
#define SMEM_SCR_STRIDE 8704
#define SMEM_NIB_OFF 9728
#define SMEM_NIB_STAGE_BYTES 1024
#define SMEM_NIB_STRIDE 1024
#define SMEM_Q2_OFF 0
#define SMEM_Q2_STAGE_BYTES 1024
#define SMEM_Q2_STRIDE 1024
#define SMEM_TOTAL 10752
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_hopper_msa_7bbffa90057de145b905(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* scr = reinterpret_cast<float*>(smem_raw + 1024);
    const int scr_addr = smem + 1024;
    unsigned int* nib = reinterpret_cast<unsigned int*>(smem_raw + 9728);
    const int nib_addr = smem + 9728;
    float* q2 = reinterpret_cast<float*>(smem_raw + 0);
    const int q2_addr = smem + 0;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1;
    int c = tid_1 - w;
    int whi = w / 32;
    int col = blockIdx.x + c;
    int head = blockIdx.y;
    int lim = tiles;
    if (col >= total_q) {
        lim = 0;
    }
    if (col < total_q) {
        int nv = nvp[col];
        if (nv < lim) {
            lim = nv;
        }
        if (lim < 0) {
            lim = 0;
        }
    }
    long long nq64 = (long long)total_q;
    long long cbase = (long long)head * (long long)tiles * nq64 + (long long)col;
    int dense = 0;
    if (total_q * 4 < 128) {
        dense = 1;
    }
    int ts_r = 1;
    int wm_r = 16;
    if (dense != 0) {
        ts_r = 128;
        wm_r = 1;
    }
    long long sts64 = (long long)ts_r * nq64;
    float neg_inf = -CUDART_INF_F;
    int ldlim = tiles;
    if (col >= total_q) {
        ldlim = 0;
    }
    unsigned int tq32 = (unsigned int)total_q;
    unsigned int cbase32 = (unsigned int)head * (unsigned int)tiles * tq32 + (unsigned int)col;
    int lim_fe = lim - fe;
    int lnx = tid_1 & 15;
    int gx = tid_1 >> 4;
    int cgx = gx & 0;
    int sgx = gx;
    int lnrx = 15 - lnx;
    int upx[4];
    upx[0] = 0;
    if ((lnx & 8) == 0) {
        upx[0] = 1;
    }
    upx[1] = 0;
    if ((lnx & 4) == 0) {
        upx[1] = 1;
    }
    upx[2] = 0;
    if ((lnx & 2) == 0) {
        upx[2] = 1;
    }
    upx[3] = 0;
    if ((lnx & 1) == 0) {
        upx[3] = 1;
    }
    float a[16];
    a[0] = neg_inf;
    a[1] = neg_inf;
    a[2] = neg_inf;
    a[3] = neg_inf;
    a[4] = neg_inf;
    a[5] = neg_inf;
    a[6] = neg_inf;
    a[7] = neg_inf;
    a[8] = neg_inf;
    a[9] = neg_inf;
    a[10] = neg_inf;
    a[11] = neg_inf;
    a[12] = neg_inf;
    a[13] = neg_inf;
    a[14] = neg_inf;
    a[15] = neg_inf;
    float rej = neg_inf;
    int t0 = w * wm_r;
    unsigned int p32 = cbase32 + (unsigned int)t0 * tq32;
    unsigned int sts32 = (unsigned int)ts_r * tq32;
    unsigned int raw[16];
    raw[0] = 4286578688u;
    if (ldlim > t0) {
        raw[0] = S[p32];
    }
    raw[1] = 4286578688u;
    if (ldlim > t0 + ts_r) {
        raw[1] = S[p32 + sts32];
    }
    raw[2] = 4286578688u;
    if (ldlim > t0 + 2 * ts_r) {
        raw[2] = S[p32 + 2 * sts32];
    }
    raw[3] = 4286578688u;
    if (ldlim > t0 + 3 * ts_r) {
        raw[3] = S[p32 + 3 * sts32];
    }
    raw[4] = 4286578688u;
    if (ldlim > t0 + 4 * ts_r) {
        raw[4] = S[p32 + 4 * sts32];
    }
    raw[5] = 4286578688u;
    if (ldlim > t0 + 5 * ts_r) {
        raw[5] = S[p32 + 5 * sts32];
    }
    raw[6] = 4286578688u;
    if (ldlim > t0 + 6 * ts_r) {
        raw[6] = S[p32 + 6 * sts32];
    }
    raw[7] = 4286578688u;
    if (ldlim > t0 + 7 * ts_r) {
        raw[7] = S[p32 + 7 * sts32];
    }
    raw[8] = 4286578688u;
    if (ldlim > t0 + 8 * ts_r) {
        raw[8] = S[p32 + 8 * sts32];
    }
    raw[9] = 4286578688u;
    if (ldlim > t0 + 9 * ts_r) {
        raw[9] = S[p32 + 9 * sts32];
    }
    raw[10] = 4286578688u;
    if (ldlim > t0 + 10 * ts_r) {
        raw[10] = S[p32 + 10 * sts32];
    }
    raw[11] = 4286578688u;
    if (ldlim > t0 + 11 * ts_r) {
        raw[11] = S[p32 + 11 * sts32];
    }
    raw[12] = 4286578688u;
    if (ldlim > t0 + 12 * ts_r) {
        raw[12] = S[p32 + 12 * sts32];
    }
    raw[13] = 4286578688u;
    if (ldlim > t0 + 13 * ts_r) {
        raw[13] = S[p32 + 13 * sts32];
    }
    raw[14] = 4286578688u;
    if (ldlim > t0 + 14 * ts_r) {
        raw[14] = S[p32 + 14 * sts32];
    }
    raw[15] = 4286578688u;
    if (ldlim > t0 + 15 * ts_r) {
        raw[15] = S[p32 + 15 * sts32];
    }
    float kb[16];
    unsigned int kot = raw[0] | raw[1] | (raw[2] | raw[3]) | (raw[4] | raw[5] | (raw[6] | raw[7])) | (raw[8] | raw[9] | (raw[10] | raw[11])) | (raw[12] | raw[13] | (raw[14] | raw[15]));
    unsigned int kow = (kot | (unsigned int)lim) & (unsigned int)(tiles >> 31);
    float konb = __uint_as_float(4278190080u | kow);
    int td = t0;
    float sc = __uint_as_float(raw[0]);
    sc = fmaxf(sc, konb);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int sb = __as_u32(sc);
    if (sb == 2147483648u) {
        sb = 0;
    }
    unsigned int vw = sb;
    unsigned int kw = sb & 4294967280u | 0;
    if (td >= lim || sb == 4278190080u) {
        vw = 4278190080u;
        kw = 4278190080u;
    }
    if ((td < fb || td >= lim_fe) && td < lim) {
        unsigned int fcls = 2139095024 - ((unsigned int)td << 4);
        vw = fcls;
        kw = fcls | 0;
    }
    scr[tid_1 * 17] = __uint_as_float(vw);
    kb[0] = __uint_as_float(kw);
    int td_0 = t0 + ts_r;
    float sc_1 = __uint_as_float(raw[1]);
    sc_1 = fmaxf(sc_1, konb);
    sc_1 = fminf(sc_1, 1.7014118e+38f);
    unsigned int sb_2 = __as_u32(sc_1);
    if (sb_2 == 2147483648u) {
        sb_2 = 0;
    }
    unsigned int vw_3 = sb_2;
    unsigned int kw_4 = sb_2 & 4294967280u | 1;
    if (td_0 >= lim || sb_2 == 4278190080u) {
        vw_3 = 4278190080u;
        kw_4 = 4278190097u;
    }
    if ((td_0 < fb || td_0 >= lim_fe) && td_0 < lim) {
        unsigned int fcls_1 = 2139095024 - ((unsigned int)td_0 << 4);
        vw_3 = fcls_1;
        kw_4 = fcls_1 | 1;
    }
    scr[tid_1 * 17 + 1] = __uint_as_float(vw_3);
    kb[1] = __uint_as_float(kw_4);
    int td_5 = t0 + 2 * ts_r;
    float sc_6 = __uint_as_float(raw[2]);
    sc_6 = fmaxf(sc_6, konb);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int sb_7 = __as_u32(sc_6);
    if (sb_7 == 2147483648u) {
        sb_7 = 0;
    }
    unsigned int vw_8 = sb_7;
    unsigned int kw_9 = sb_7 & 4294967280u | 2;
    if (td_5 >= lim || sb_7 == 4278190080u) {
        vw_8 = 4278190080u;
        kw_9 = 4278190114u;
    }
    if ((td_5 < fb || td_5 >= lim_fe) && td_5 < lim) {
        unsigned int fcls_2 = 2139095024 - ((unsigned int)td_5 << 4);
        vw_8 = fcls_2;
        kw_9 = fcls_2 | 2;
    }
    scr[tid_1 * 17 + 2] = __uint_as_float(vw_8);
    kb[2] = __uint_as_float(kw_9);
    int td_10 = t0 + 3 * ts_r;
    float sc_11 = __uint_as_float(raw[3]);
    sc_11 = fmaxf(sc_11, konb);
    sc_11 = fminf(sc_11, 1.7014118e+38f);
    unsigned int sb_12 = __as_u32(sc_11);
    if (sb_12 == 2147483648u) {
        sb_12 = 0;
    }
    unsigned int vw_13 = sb_12;
    unsigned int kw_14 = sb_12 & 4294967280u | 3;
    if (td_10 >= lim || sb_12 == 4278190080u) {
        vw_13 = 4278190080u;
        kw_14 = 4278190131u;
    }
    if ((td_10 < fb || td_10 >= lim_fe) && td_10 < lim) {
        unsigned int fcls_3 = 2139095024 - ((unsigned int)td_10 << 4);
        vw_13 = fcls_3;
        kw_14 = fcls_3 | 3;
    }
    scr[tid_1 * 17 + 3] = __uint_as_float(vw_13);
    kb[3] = __uint_as_float(kw_14);
    int td_15 = t0 + 4 * ts_r;
    float sc_16 = __uint_as_float(raw[4]);
    sc_16 = fmaxf(sc_16, konb);
    sc_16 = fminf(sc_16, 1.7014118e+38f);
    unsigned int sb_17 = __as_u32(sc_16);
    if (sb_17 == 2147483648u) {
        sb_17 = 0;
    }
    unsigned int vw_18 = sb_17;
    unsigned int kw_19 = sb_17 & 4294967280u | 4;
    if (td_15 >= lim || sb_17 == 4278190080u) {
        vw_18 = 4278190080u;
        kw_19 = 4278190148u;
    }
    if ((td_15 < fb || td_15 >= lim_fe) && td_15 < lim) {
        unsigned int fcls_4 = 2139095024 - ((unsigned int)td_15 << 4);
        vw_18 = fcls_4;
        kw_19 = fcls_4 | 4;
    }
    scr[tid_1 * 17 + 4] = __uint_as_float(vw_18);
    kb[4] = __uint_as_float(kw_19);
    int td_20 = t0 + 5 * ts_r;
    float sc_21 = __uint_as_float(raw[5]);
    sc_21 = fmaxf(sc_21, konb);
    sc_21 = fminf(sc_21, 1.7014118e+38f);
    unsigned int sb_22 = __as_u32(sc_21);
    if (sb_22 == 2147483648u) {
        sb_22 = 0;
    }
    unsigned int vw_23 = sb_22;
    unsigned int kw_24 = sb_22 & 4294967280u | 5;
    if (td_20 >= lim || sb_22 == 4278190080u) {
        vw_23 = 4278190080u;
        kw_24 = 4278190165u;
    }
    if ((td_20 < fb || td_20 >= lim_fe) && td_20 < lim) {
        unsigned int fcls_5 = 2139095024 - ((unsigned int)td_20 << 4);
        vw_23 = fcls_5;
        kw_24 = fcls_5 | 5;
    }
    scr[tid_1 * 17 + 5] = __uint_as_float(vw_23);
    kb[5] = __uint_as_float(kw_24);
    int td_25 = t0 + 6 * ts_r;
    float sc_26 = __uint_as_float(raw[6]);
    sc_26 = fmaxf(sc_26, konb);
    sc_26 = fminf(sc_26, 1.7014118e+38f);
    unsigned int sb_27 = __as_u32(sc_26);
    if (sb_27 == 2147483648u) {
        sb_27 = 0;
    }
    unsigned int vw_28 = sb_27;
    unsigned int kw_29 = sb_27 & 4294967280u | 6;
    if (td_25 >= lim || sb_27 == 4278190080u) {
        vw_28 = 4278190080u;
        kw_29 = 4278190182u;
    }
    if ((td_25 < fb || td_25 >= lim_fe) && td_25 < lim) {
        unsigned int fcls_6 = 2139095024 - ((unsigned int)td_25 << 4);
        vw_28 = fcls_6;
        kw_29 = fcls_6 | 6;
    }
    scr[tid_1 * 17 + 6] = __uint_as_float(vw_28);
    kb[6] = __uint_as_float(kw_29);
    int td_30 = t0 + 7 * ts_r;
    float sc_31 = __uint_as_float(raw[7]);
    sc_31 = fmaxf(sc_31, konb);
    sc_31 = fminf(sc_31, 1.7014118e+38f);
    unsigned int sb_32 = __as_u32(sc_31);
    if (sb_32 == 2147483648u) {
        sb_32 = 0;
    }
    unsigned int vw_33 = sb_32;
    unsigned int kw_34 = sb_32 & 4294967280u | 7;
    if (td_30 >= lim || sb_32 == 4278190080u) {
        vw_33 = 4278190080u;
        kw_34 = 4278190199u;
    }
    if ((td_30 < fb || td_30 >= lim_fe) && td_30 < lim) {
        unsigned int fcls_7 = 2139095024 - ((unsigned int)td_30 << 4);
        vw_33 = fcls_7;
        kw_34 = fcls_7 | 7;
    }
    scr[tid_1 * 17 + 7] = __uint_as_float(vw_33);
    kb[7] = __uint_as_float(kw_34);
    int td_35 = t0 + 8 * ts_r;
    float sc_36 = __uint_as_float(raw[8]);
    sc_36 = fmaxf(sc_36, konb);
    sc_36 = fminf(sc_36, 1.7014118e+38f);
    unsigned int sb_37 = __as_u32(sc_36);
    if (sb_37 == 2147483648u) {
        sb_37 = 0;
    }
    unsigned int vw_38 = sb_37;
    unsigned int kw_39 = sb_37 & 4294967280u | 8;
    if (td_35 >= lim || sb_37 == 4278190080u) {
        vw_38 = 4278190080u;
        kw_39 = 4278190216u;
    }
    if ((td_35 < fb || td_35 >= lim_fe) && td_35 < lim) {
        unsigned int fcls_8 = 2139095024 - ((unsigned int)td_35 << 4);
        vw_38 = fcls_8;
        kw_39 = fcls_8 | 8;
    }
    scr[tid_1 * 17 + 8] = __uint_as_float(vw_38);
    kb[8] = __uint_as_float(kw_39);
    int td_40 = t0 + 9 * ts_r;
    float sc_41 = __uint_as_float(raw[9]);
    sc_41 = fmaxf(sc_41, konb);
    sc_41 = fminf(sc_41, 1.7014118e+38f);
    unsigned int sb_42 = __as_u32(sc_41);
    if (sb_42 == 2147483648u) {
        sb_42 = 0;
    }
    unsigned int vw_43 = sb_42;
    unsigned int kw_44 = sb_42 & 4294967280u | 9;
    if (td_40 >= lim || sb_42 == 4278190080u) {
        vw_43 = 4278190080u;
        kw_44 = 4278190233u;
    }
    if ((td_40 < fb || td_40 >= lim_fe) && td_40 < lim) {
        unsigned int fcls_9 = 2139095024 - ((unsigned int)td_40 << 4);
        vw_43 = fcls_9;
        kw_44 = fcls_9 | 9;
    }
    scr[tid_1 * 17 + 9] = __uint_as_float(vw_43);
    kb[9] = __uint_as_float(kw_44);
    int td_45 = t0 + 10 * ts_r;
    float sc_46 = __uint_as_float(raw[10]);
    sc_46 = fmaxf(sc_46, konb);
    sc_46 = fminf(sc_46, 1.7014118e+38f);
    unsigned int sb_47 = __as_u32(sc_46);
    if (sb_47 == 2147483648u) {
        sb_47 = 0;
    }
    unsigned int vw_48 = sb_47;
    unsigned int kw_49 = sb_47 & 4294967280u | 10;
    if (td_45 >= lim || sb_47 == 4278190080u) {
        vw_48 = 4278190080u;
        kw_49 = 4278190250u;
    }
    if ((td_45 < fb || td_45 >= lim_fe) && td_45 < lim) {
        unsigned int fcls_10 = 2139095024 - ((unsigned int)td_45 << 4);
        vw_48 = fcls_10;
        kw_49 = fcls_10 | 10;
    }
    scr[tid_1 * 17 + 10] = __uint_as_float(vw_48);
    kb[10] = __uint_as_float(kw_49);
    int td_50 = t0 + 11 * ts_r;
    float sc_51 = __uint_as_float(raw[11]);
    sc_51 = fmaxf(sc_51, konb);
    sc_51 = fminf(sc_51, 1.7014118e+38f);
    unsigned int sb_52 = __as_u32(sc_51);
    if (sb_52 == 2147483648u) {
        sb_52 = 0;
    }
    unsigned int vw_53 = sb_52;
    unsigned int kw_54 = sb_52 & 4294967280u | 11;
    if (td_50 >= lim || sb_52 == 4278190080u) {
        vw_53 = 4278190080u;
        kw_54 = 4278190267u;
    }
    if ((td_50 < fb || td_50 >= lim_fe) && td_50 < lim) {
        unsigned int fcls_11 = 2139095024 - ((unsigned int)td_50 << 4);
        vw_53 = fcls_11;
        kw_54 = fcls_11 | 11;
    }
    scr[tid_1 * 17 + 11] = __uint_as_float(vw_53);
    kb[11] = __uint_as_float(kw_54);
    int td_55 = t0 + 12 * ts_r;
    float sc_56 = __uint_as_float(raw[12]);
    sc_56 = fmaxf(sc_56, konb);
    sc_56 = fminf(sc_56, 1.7014118e+38f);
    unsigned int sb_57 = __as_u32(sc_56);
    if (sb_57 == 2147483648u) {
        sb_57 = 0;
    }
    unsigned int vw_58 = sb_57;
    unsigned int kw_59 = sb_57 & 4294967280u | 12;
    if (td_55 >= lim || sb_57 == 4278190080u) {
        vw_58 = 4278190080u;
        kw_59 = 4278190284u;
    }
    if ((td_55 < fb || td_55 >= lim_fe) && td_55 < lim) {
        unsigned int fcls_12 = 2139095024 - ((unsigned int)td_55 << 4);
        vw_58 = fcls_12;
        kw_59 = fcls_12 | 12;
    }
    scr[tid_1 * 17 + 12] = __uint_as_float(vw_58);
    kb[12] = __uint_as_float(kw_59);
    int td_60 = t0 + 13 * ts_r;
    float sc_61 = __uint_as_float(raw[13]);
    sc_61 = fmaxf(sc_61, konb);
    sc_61 = fminf(sc_61, 1.7014118e+38f);
    unsigned int sb_62 = __as_u32(sc_61);
    if (sb_62 == 2147483648u) {
        sb_62 = 0;
    }
    unsigned int vw_63 = sb_62;
    unsigned int kw_64 = sb_62 & 4294967280u | 13;
    if (td_60 >= lim || sb_62 == 4278190080u) {
        vw_63 = 4278190080u;
        kw_64 = 4278190301u;
    }
    if ((td_60 < fb || td_60 >= lim_fe) && td_60 < lim) {
        unsigned int fcls_13 = 2139095024 - ((unsigned int)td_60 << 4);
        vw_63 = fcls_13;
        kw_64 = fcls_13 | 13;
    }
    scr[tid_1 * 17 + 13] = __uint_as_float(vw_63);
    kb[13] = __uint_as_float(kw_64);
    int td_65 = t0 + 14 * ts_r;
    float sc_66 = __uint_as_float(raw[14]);
    sc_66 = fmaxf(sc_66, konb);
    sc_66 = fminf(sc_66, 1.7014118e+38f);
    unsigned int sb_67 = __as_u32(sc_66);
    if (sb_67 == 2147483648u) {
        sb_67 = 0;
    }
    unsigned int vw_68 = sb_67;
    unsigned int kw_69 = sb_67 & 4294967280u | 14;
    if (td_65 >= lim || sb_67 == 4278190080u) {
        vw_68 = 4278190080u;
        kw_69 = 4278190318u;
    }
    if ((td_65 < fb || td_65 >= lim_fe) && td_65 < lim) {
        unsigned int fcls_14 = 2139095024 - ((unsigned int)td_65 << 4);
        vw_68 = fcls_14;
        kw_69 = fcls_14 | 14;
    }
    scr[tid_1 * 17 + 14] = __uint_as_float(vw_68);
    kb[14] = __uint_as_float(kw_69);
    int td_70 = t0 + 15 * ts_r;
    float sc_71 = __uint_as_float(raw[15]);
    sc_71 = fmaxf(sc_71, konb);
    sc_71 = fminf(sc_71, 1.7014118e+38f);
    unsigned int sb_72 = __as_u32(sc_71);
    if (sb_72 == 2147483648u) {
        sb_72 = 0;
    }
    unsigned int vw_73 = sb_72;
    unsigned int kw_74 = sb_72 & 4294967280u | 15;
    if (td_70 >= lim || sb_72 == 4278190080u) {
        vw_73 = 4278190080u;
        kw_74 = 4278190335u;
    }
    if ((td_70 < fb || td_70 >= lim_fe) && td_70 < lim) {
        unsigned int fcls_15 = 2139095024 - ((unsigned int)td_70 << 4);
        vw_73 = fcls_15;
        kw_74 = fcls_15 | 15;
    }
    scr[tid_1 * 17 + 15] = __uint_as_float(vw_73);
    kb[15] = __uint_as_float(kw_74);
    float hi = fmaxf(kb[0], kb[13]);
    float lo = fminf(kb[0], kb[13]);
    kb[0] = hi;
    kb[13] = lo;
    float hi_75 = fmaxf(kb[1], kb[12]);
    float lo_76 = fminf(kb[1], kb[12]);
    kb[1] = hi_75;
    kb[12] = lo_76;
    float hi_77 = fmaxf(kb[2], kb[15]);
    float lo_78 = fminf(kb[2], kb[15]);
    kb[2] = hi_77;
    kb[15] = lo_78;
    float hi_79 = fmaxf(kb[3], kb[14]);
    float lo_80 = fminf(kb[3], kb[14]);
    kb[3] = hi_79;
    kb[14] = lo_80;
    float hi_81 = fmaxf(kb[4], kb[8]);
    float lo_82 = fminf(kb[4], kb[8]);
    kb[4] = hi_81;
    kb[8] = lo_82;
    float hi_83 = fmaxf(kb[5], kb[6]);
    float lo_84 = fminf(kb[5], kb[6]);
    kb[5] = hi_83;
    kb[6] = lo_84;
    float hi_85 = fmaxf(kb[7], kb[11]);
    float lo_86 = fminf(kb[7], kb[11]);
    kb[7] = hi_85;
    kb[11] = lo_86;
    float hi_87 = fmaxf(kb[9], kb[10]);
    float lo_88 = fminf(kb[9], kb[10]);
    kb[9] = hi_87;
    kb[10] = lo_88;
    float hi_89 = fmaxf(kb[0], kb[5]);
    float lo_90 = fminf(kb[0], kb[5]);
    kb[0] = hi_89;
    kb[5] = lo_90;
    float hi_91 = fmaxf(kb[1], kb[7]);
    float lo_92 = fminf(kb[1], kb[7]);
    kb[1] = hi_91;
    kb[7] = lo_92;
    float hi_93 = fmaxf(kb[2], kb[9]);
    float lo_94 = fminf(kb[2], kb[9]);
    kb[2] = hi_93;
    kb[9] = lo_94;
    float hi_95 = fmaxf(kb[3], kb[4]);
    float lo_96 = fminf(kb[3], kb[4]);
    kb[3] = hi_95;
    kb[4] = lo_96;
    float hi_97 = fmaxf(kb[6], kb[13]);
    float lo_98 = fminf(kb[6], kb[13]);
    kb[6] = hi_97;
    kb[13] = lo_98;
    float hi_99 = fmaxf(kb[8], kb[14]);
    float lo_100 = fminf(kb[8], kb[14]);
    kb[8] = hi_99;
    kb[14] = lo_100;
    float hi_101 = fmaxf(kb[10], kb[15]);
    float lo_102 = fminf(kb[10], kb[15]);
    kb[10] = hi_101;
    kb[15] = lo_102;
    float hi_103 = fmaxf(kb[11], kb[12]);
    float lo_104 = fminf(kb[11], kb[12]);
    kb[11] = hi_103;
    kb[12] = lo_104;
    float hi_105 = fmaxf(kb[0], kb[1]);
    float lo_106 = fminf(kb[0], kb[1]);
    kb[0] = hi_105;
    kb[1] = lo_106;
    float hi_107 = fmaxf(kb[2], kb[3]);
    float lo_108 = fminf(kb[2], kb[3]);
    kb[2] = hi_107;
    kb[3] = lo_108;
    float hi_109 = fmaxf(kb[4], kb[5]);
    float lo_110 = fminf(kb[4], kb[5]);
    kb[4] = hi_109;
    kb[5] = lo_110;
    float hi_111 = fmaxf(kb[6], kb[8]);
    float lo_112 = fminf(kb[6], kb[8]);
    kb[6] = hi_111;
    kb[8] = lo_112;
    float hi_113 = fmaxf(kb[7], kb[9]);
    float lo_114 = fminf(kb[7], kb[9]);
    kb[7] = hi_113;
    kb[9] = lo_114;
    float hi_115 = fmaxf(kb[10], kb[11]);
    float lo_116 = fminf(kb[10], kb[11]);
    kb[10] = hi_115;
    kb[11] = lo_116;
    float hi_117 = fmaxf(kb[12], kb[13]);
    float lo_118 = fminf(kb[12], kb[13]);
    kb[12] = hi_117;
    kb[13] = lo_118;
    float hi_119 = fmaxf(kb[14], kb[15]);
    float lo_120 = fminf(kb[14], kb[15]);
    kb[14] = hi_119;
    kb[15] = lo_120;
    float hi_121 = fmaxf(kb[0], kb[2]);
    float lo_122 = fminf(kb[0], kb[2]);
    kb[0] = hi_121;
    kb[2] = lo_122;
    float hi_123 = fmaxf(kb[1], kb[3]);
    float lo_124 = fminf(kb[1], kb[3]);
    kb[1] = hi_123;
    kb[3] = lo_124;
    float hi_125 = fmaxf(kb[4], kb[10]);
    float lo_126 = fminf(kb[4], kb[10]);
    kb[4] = hi_125;
    kb[10] = lo_126;
    float hi_127 = fmaxf(kb[5], kb[11]);
    float lo_128 = fminf(kb[5], kb[11]);
    kb[5] = hi_127;
    kb[11] = lo_128;
    float hi_129 = fmaxf(kb[6], kb[7]);
    float lo_130 = fminf(kb[6], kb[7]);
    kb[6] = hi_129;
    kb[7] = lo_130;
    float hi_131 = fmaxf(kb[8], kb[9]);
    float lo_132 = fminf(kb[8], kb[9]);
    kb[8] = hi_131;
    kb[9] = lo_132;
    float hi_133 = fmaxf(kb[12], kb[14]);
    float lo_134 = fminf(kb[12], kb[14]);
    kb[12] = hi_133;
    kb[14] = lo_134;
    float hi_135 = fmaxf(kb[13], kb[15]);
    float lo_136 = fminf(kb[13], kb[15]);
    kb[13] = hi_135;
    kb[15] = lo_136;
    float hi_137 = fmaxf(kb[1], kb[2]);
    float lo_138 = fminf(kb[1], kb[2]);
    kb[1] = hi_137;
    kb[2] = lo_138;
    float hi_139 = fmaxf(kb[3], kb[12]);
    float lo_140 = fminf(kb[3], kb[12]);
    kb[3] = hi_139;
    kb[12] = lo_140;
    float hi_141 = fmaxf(kb[4], kb[6]);
    float lo_142 = fminf(kb[4], kb[6]);
    kb[4] = hi_141;
    kb[6] = lo_142;
    float hi_143 = fmaxf(kb[5], kb[7]);
    float lo_144 = fminf(kb[5], kb[7]);
    kb[5] = hi_143;
    kb[7] = lo_144;
    float hi_145 = fmaxf(kb[8], kb[10]);
    float lo_146 = fminf(kb[8], kb[10]);
    kb[8] = hi_145;
    kb[10] = lo_146;
    float hi_147 = fmaxf(kb[9], kb[11]);
    float lo_148 = fminf(kb[9], kb[11]);
    kb[9] = hi_147;
    kb[11] = lo_148;
    float hi_149 = fmaxf(kb[13], kb[14]);
    float lo_150 = fminf(kb[13], kb[14]);
    kb[13] = hi_149;
    kb[14] = lo_150;
    float hi_151 = fmaxf(kb[1], kb[4]);
    float lo_152 = fminf(kb[1], kb[4]);
    kb[1] = hi_151;
    kb[4] = lo_152;
    float hi_153 = fmaxf(kb[2], kb[6]);
    float lo_154 = fminf(kb[2], kb[6]);
    kb[2] = hi_153;
    kb[6] = lo_154;
    float hi_155 = fmaxf(kb[5], kb[8]);
    float lo_156 = fminf(kb[5], kb[8]);
    kb[5] = hi_155;
    kb[8] = lo_156;
    float hi_157 = fmaxf(kb[7], kb[10]);
    float lo_158 = fminf(kb[7], kb[10]);
    kb[7] = hi_157;
    kb[10] = lo_158;
    float hi_159 = fmaxf(kb[9], kb[13]);
    float lo_160 = fminf(kb[9], kb[13]);
    kb[9] = hi_159;
    kb[13] = lo_160;
    float hi_161 = fmaxf(kb[11], kb[14]);
    float lo_162 = fminf(kb[11], kb[14]);
    kb[11] = hi_161;
    kb[14] = lo_162;
    float hi_163 = fmaxf(kb[2], kb[4]);
    float lo_164 = fminf(kb[2], kb[4]);
    kb[2] = hi_163;
    kb[4] = lo_164;
    float hi_165 = fmaxf(kb[3], kb[6]);
    float lo_166 = fminf(kb[3], kb[6]);
    kb[3] = hi_165;
    kb[6] = lo_166;
    float hi_167 = fmaxf(kb[9], kb[12]);
    float lo_168 = fminf(kb[9], kb[12]);
    kb[9] = hi_167;
    kb[12] = lo_168;
    float hi_169 = fmaxf(kb[11], kb[13]);
    float lo_170 = fminf(kb[11], kb[13]);
    kb[11] = hi_169;
    kb[13] = lo_170;
    float hi_171 = fmaxf(kb[3], kb[5]);
    float lo_172 = fminf(kb[3], kb[5]);
    kb[3] = hi_171;
    kb[5] = lo_172;
    float hi_173 = fmaxf(kb[6], kb[8]);
    float lo_174 = fminf(kb[6], kb[8]);
    kb[6] = hi_173;
    kb[8] = lo_174;
    float hi_175 = fmaxf(kb[7], kb[9]);
    float lo_176 = fminf(kb[7], kb[9]);
    kb[7] = hi_175;
    kb[9] = lo_176;
    float hi_177 = fmaxf(kb[10], kb[12]);
    float lo_178 = fminf(kb[10], kb[12]);
    kb[10] = hi_177;
    kb[12] = lo_178;
    float hi_179 = fmaxf(kb[3], kb[4]);
    float lo_180 = fminf(kb[3], kb[4]);
    kb[3] = hi_179;
    kb[4] = lo_180;
    float hi_181 = fmaxf(kb[5], kb[6]);
    float lo_182 = fminf(kb[5], kb[6]);
    kb[5] = hi_181;
    kb[6] = lo_182;
    float hi_183 = fmaxf(kb[7], kb[8]);
    float lo_184 = fminf(kb[7], kb[8]);
    kb[7] = hi_183;
    kb[8] = lo_184;
    float hi_185 = fmaxf(kb[9], kb[10]);
    float lo_186 = fminf(kb[9], kb[10]);
    kb[9] = hi_185;
    kb[10] = lo_186;
    float hi_187 = fmaxf(kb[11], kb[12]);
    float lo_188 = fminf(kb[11], kb[12]);
    kb[11] = hi_187;
    kb[12] = lo_188;
    float hi_189 = fmaxf(kb[6], kb[7]);
    float lo_190 = fminf(kb[6], kb[7]);
    kb[6] = hi_189;
    kb[7] = lo_190;
    float hi_191 = fmaxf(kb[8], kb[9]);
    float lo_192 = fminf(kb[8], kb[9]);
    kb[8] = hi_191;
    kb[9] = lo_192;
    float v[16];
    unsigned int x[16];
    x[0] = __as_u32(kb[0]) & 15;
    x[1] = __as_u32(kb[1]) & 15;
    x[2] = __as_u32(kb[2]) & 15;
    x[3] = __as_u32(kb[3]) & 15;
    x[4] = __as_u32(kb[4]) & 15;
    x[5] = __as_u32(kb[5]) & 15;
    x[6] = __as_u32(kb[6]) & 15;
    x[7] = __as_u32(kb[7]) & 15;
    x[8] = __as_u32(kb[8]) & 15;
    x[9] = __as_u32(kb[9]) & 15;
    x[10] = __as_u32(kb[10]) & 15;
    x[11] = __as_u32(kb[11]) & 15;
    x[12] = __as_u32(kb[12]) & 15;
    x[13] = __as_u32(kb[13]) & 15;
    x[14] = __as_u32(kb[14]) & 15;
    x[15] = __as_u32(kb[15]) & 15;
    int run = 0;
    if ((__as_u32(kb[0]) ^ __as_u32(kb[1])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[1]) ^ __as_u32(kb[2])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[2]) ^ __as_u32(kb[3])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[3]) ^ __as_u32(kb[4])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[4]) ^ __as_u32(kb[5])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[5]) ^ __as_u32(kb[6])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[6]) ^ __as_u32(kb[7])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[7]) ^ __as_u32(kb[8])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[8]) ^ __as_u32(kb[9])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[9]) ^ __as_u32(kb[10])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[10]) ^ __as_u32(kb[11])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[11]) ^ __as_u32(kb[12])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[12]) ^ __as_u32(kb[13])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[13]) ^ __as_u32(kb[14])) < 16) {
        run = 1;
    }
    if ((__as_u32(kb[14]) ^ __as_u32(kb[15])) < 16) {
        run = 1;
    }
    int more = __any_sync(0xFFFFFFFF, run != 0);
    if (more != 0) {
        v[0] = scr[tid_1 * 17 + (int)x[0]];
        v[1] = scr[tid_1 * 17 + (int)x[1]];
        v[2] = scr[tid_1 * 17 + (int)x[2]];
        v[3] = scr[tid_1 * 17 + (int)x[3]];
        v[4] = scr[tid_1 * 17 + (int)x[4]];
        v[5] = scr[tid_1 * 17 + (int)x[5]];
        v[6] = scr[tid_1 * 17 + (int)x[6]];
        v[7] = scr[tid_1 * 17 + (int)x[7]];
        v[8] = scr[tid_1 * 17 + (int)x[8]];
        v[9] = scr[tid_1 * 17 + (int)x[9]];
        v[10] = scr[tid_1 * 17 + (int)x[10]];
        v[11] = scr[tid_1 * 17 + (int)x[11]];
        v[12] = scr[tid_1 * 17 + (int)x[12]];
        v[13] = scr[tid_1 * 17 + (int)x[13]];
        v[14] = scr[tid_1 * 17 + (int)x[14]];
        v[15] = scr[tid_1 * 17 + (int)x[15]];
    }
    if (more != 0) {
        int sw = 0;
        int p = 0;
        if (v[0] >= v[1]) {
            p = 1;
        }
        float hi_0 = fmaxf(v[0], v[1]);
        float lo_1 = fminf(v[0], v[1]);
        unsigned int xi = x[0];
        unsigned int xj = x[1];
        v[0] = hi_0;
        v[1] = lo_1;
        x[0] = ((p != 0) ? xi : xj);
        x[1] = ((p != 0) ? xj : xi);
        if (p == 0) {
            sw = 1;
        }
        int p_2 = 0;
        if (v[2] >= v[3]) {
            p_2 = 1;
        }
        float hi_3 = fmaxf(v[2], v[3]);
        float lo_4 = fminf(v[2], v[3]);
        unsigned int xi_5 = x[2];
        unsigned int xj_6 = x[3];
        v[2] = hi_3;
        v[3] = lo_4;
        x[2] = ((p_2 != 0) ? xi_5 : xj_6);
        x[3] = ((p_2 != 0) ? xj_6 : xi_5);
        if (p_2 == 0) {
            sw = 1;
        }
        int p_7 = 0;
        if (v[4] >= v[5]) {
            p_7 = 1;
        }
        float hi_8 = fmaxf(v[4], v[5]);
        float lo_9 = fminf(v[4], v[5]);
        unsigned int xi_10 = x[4];
        unsigned int xj_11 = x[5];
        v[4] = hi_8;
        v[5] = lo_9;
        x[4] = ((p_7 != 0) ? xi_10 : xj_11);
        x[5] = ((p_7 != 0) ? xj_11 : xi_10);
        if (p_7 == 0) {
            sw = 1;
        }
        int p_12 = 0;
        if (v[6] >= v[7]) {
            p_12 = 1;
        }
        float hi_13 = fmaxf(v[6], v[7]);
        float lo_14 = fminf(v[6], v[7]);
        unsigned int xi_15 = x[6];
        unsigned int xj_16 = x[7];
        v[6] = hi_13;
        v[7] = lo_14;
        x[6] = ((p_12 != 0) ? xi_15 : xj_16);
        x[7] = ((p_12 != 0) ? xj_16 : xi_15);
        if (p_12 == 0) {
            sw = 1;
        }
        int p_17 = 0;
        if (v[8] >= v[9]) {
            p_17 = 1;
        }
        float hi_18 = fmaxf(v[8], v[9]);
        float lo_19 = fminf(v[8], v[9]);
        unsigned int xi_20 = x[8];
        unsigned int xj_21 = x[9];
        v[8] = hi_18;
        v[9] = lo_19;
        x[8] = ((p_17 != 0) ? xi_20 : xj_21);
        x[9] = ((p_17 != 0) ? xj_21 : xi_20);
        if (p_17 == 0) {
            sw = 1;
        }
        int p_22 = 0;
        if (v[10] >= v[11]) {
            p_22 = 1;
        }
        float hi_23 = fmaxf(v[10], v[11]);
        float lo_24 = fminf(v[10], v[11]);
        unsigned int xi_25 = x[10];
        unsigned int xj_26 = x[11];
        v[10] = hi_23;
        v[11] = lo_24;
        x[10] = ((p_22 != 0) ? xi_25 : xj_26);
        x[11] = ((p_22 != 0) ? xj_26 : xi_25);
        if (p_22 == 0) {
            sw = 1;
        }
        int p_27 = 0;
        if (v[12] >= v[13]) {
            p_27 = 1;
        }
        float hi_28 = fmaxf(v[12], v[13]);
        float lo_29 = fminf(v[12], v[13]);
        unsigned int xi_30 = x[12];
        unsigned int xj_31 = x[13];
        v[12] = hi_28;
        v[13] = lo_29;
        x[12] = ((p_27 != 0) ? xi_30 : xj_31);
        x[13] = ((p_27 != 0) ? xj_31 : xi_30);
        if (p_27 == 0) {
            sw = 1;
        }
        int p_32 = 0;
        if (v[14] >= v[15]) {
            p_32 = 1;
        }
        float hi_33 = fmaxf(v[14], v[15]);
        float lo_34 = fminf(v[14], v[15]);
        unsigned int xi_35 = x[14];
        unsigned int xj_36 = x[15];
        v[14] = hi_33;
        v[15] = lo_34;
        x[14] = ((p_32 != 0) ? xi_35 : xj_36);
        x[15] = ((p_32 != 0) ? xj_36 : xi_35);
        if (p_32 == 0) {
            sw = 1;
        }
        int p_37 = 0;
        if (v[1] >= v[2]) {
            p_37 = 1;
        }
        float hi_38 = fmaxf(v[1], v[2]);
        float lo_39 = fminf(v[1], v[2]);
        unsigned int xi_40 = x[1];
        unsigned int xj_41 = x[2];
        v[1] = hi_38;
        v[2] = lo_39;
        x[1] = ((p_37 != 0) ? xi_40 : xj_41);
        x[2] = ((p_37 != 0) ? xj_41 : xi_40);
        if (p_37 == 0) {
            sw = 1;
        }
        int p_42 = 0;
        if (v[3] >= v[4]) {
            p_42 = 1;
        }
        float hi_43 = fmaxf(v[3], v[4]);
        float lo_44 = fminf(v[3], v[4]);
        unsigned int xi_45 = x[3];
        unsigned int xj_46 = x[4];
        v[3] = hi_43;
        v[4] = lo_44;
        x[3] = ((p_42 != 0) ? xi_45 : xj_46);
        x[4] = ((p_42 != 0) ? xj_46 : xi_45);
        if (p_42 == 0) {
            sw = 1;
        }
        int p_47 = 0;
        if (v[5] >= v[6]) {
            p_47 = 1;
        }
        float hi_48 = fmaxf(v[5], v[6]);
        float lo_49 = fminf(v[5], v[6]);
        unsigned int xi_50 = x[5];
        unsigned int xj_51 = x[6];
        v[5] = hi_48;
        v[6] = lo_49;
        x[5] = ((p_47 != 0) ? xi_50 : xj_51);
        x[6] = ((p_47 != 0) ? xj_51 : xi_50);
        if (p_47 == 0) {
            sw = 1;
        }
        int p_52 = 0;
        if (v[7] >= v[8]) {
            p_52 = 1;
        }
        float hi_53 = fmaxf(v[7], v[8]);
        float lo_54 = fminf(v[7], v[8]);
        unsigned int xi_55 = x[7];
        unsigned int xj_56 = x[8];
        v[7] = hi_53;
        v[8] = lo_54;
        x[7] = ((p_52 != 0) ? xi_55 : xj_56);
        x[8] = ((p_52 != 0) ? xj_56 : xi_55);
        if (p_52 == 0) {
            sw = 1;
        }
        int p_57 = 0;
        if (v[9] >= v[10]) {
            p_57 = 1;
        }
        float hi_58 = fmaxf(v[9], v[10]);
        float lo_59 = fminf(v[9], v[10]);
        unsigned int xi_60 = x[9];
        unsigned int xj_61 = x[10];
        v[9] = hi_58;
        v[10] = lo_59;
        x[9] = ((p_57 != 0) ? xi_60 : xj_61);
        x[10] = ((p_57 != 0) ? xj_61 : xi_60);
        if (p_57 == 0) {
            sw = 1;
        }
        int p_62 = 0;
        if (v[11] >= v[12]) {
            p_62 = 1;
        }
        float hi_63 = fmaxf(v[11], v[12]);
        float lo_64 = fminf(v[11], v[12]);
        unsigned int xi_65 = x[11];
        unsigned int xj_66 = x[12];
        v[11] = hi_63;
        v[12] = lo_64;
        x[11] = ((p_62 != 0) ? xi_65 : xj_66);
        x[12] = ((p_62 != 0) ? xj_66 : xi_65);
        if (p_62 == 0) {
            sw = 1;
        }
        int p_67 = 0;
        if (v[13] >= v[14]) {
            p_67 = 1;
        }
        float hi_68 = fmaxf(v[13], v[14]);
        float lo_69 = fminf(v[13], v[14]);
        unsigned int xi_70 = x[13];
        unsigned int xj_71 = x[14];
        v[13] = hi_68;
        v[14] = lo_69;
        x[13] = ((p_67 != 0) ? xi_70 : xj_71);
        x[14] = ((p_67 != 0) ? xj_71 : xi_70);
        if (p_67 == 0) {
            sw = 1;
        }
        int _warp_vote_0 = __any_sync(0xFFFFFFFF, sw != 0);
        more = _warp_vote_0;
    }
    if (more != 0) {
        int sw_1 = 0;
        int p_1 = 0;
        if (v[0] >= v[1]) {
            p_1 = 1;
        }
        float hi_0_1 = fmaxf(v[0], v[1]);
        float lo_1_1 = fminf(v[0], v[1]);
        unsigned int xi_1 = x[0];
        unsigned int xj_1 = x[1];
        v[0] = hi_0_1;
        v[1] = lo_1_1;
        x[0] = ((p_1 != 0) ? xi_1 : xj_1);
        x[1] = ((p_1 != 0) ? xj_1 : xi_1);
        if (p_1 == 0) {
            sw_1 = 1;
        }
        int p_2_1 = 0;
        if (v[2] >= v[3]) {
            p_2_1 = 1;
        }
        float hi_3_1 = fmaxf(v[2], v[3]);
        float lo_4_1 = fminf(v[2], v[3]);
        unsigned int xi_5_1 = x[2];
        unsigned int xj_6_1 = x[3];
        v[2] = hi_3_1;
        v[3] = lo_4_1;
        x[2] = ((p_2_1 != 0) ? xi_5_1 : xj_6_1);
        x[3] = ((p_2_1 != 0) ? xj_6_1 : xi_5_1);
        if (p_2_1 == 0) {
            sw_1 = 1;
        }
        int p_7_1 = 0;
        if (v[4] >= v[5]) {
            p_7_1 = 1;
        }
        float hi_8_1 = fmaxf(v[4], v[5]);
        float lo_9_1 = fminf(v[4], v[5]);
        unsigned int xi_10_1 = x[4];
        unsigned int xj_11_1 = x[5];
        v[4] = hi_8_1;
        v[5] = lo_9_1;
        x[4] = ((p_7_1 != 0) ? xi_10_1 : xj_11_1);
        x[5] = ((p_7_1 != 0) ? xj_11_1 : xi_10_1);
        if (p_7_1 == 0) {
            sw_1 = 1;
        }
        int p_12_1 = 0;
        if (v[6] >= v[7]) {
            p_12_1 = 1;
        }
        float hi_13_1 = fmaxf(v[6], v[7]);
        float lo_14_1 = fminf(v[6], v[7]);
        unsigned int xi_15_1 = x[6];
        unsigned int xj_16_1 = x[7];
        v[6] = hi_13_1;
        v[7] = lo_14_1;
        x[6] = ((p_12_1 != 0) ? xi_15_1 : xj_16_1);
        x[7] = ((p_12_1 != 0) ? xj_16_1 : xi_15_1);
        if (p_12_1 == 0) {
            sw_1 = 1;
        }
        int p_17_1 = 0;
        if (v[8] >= v[9]) {
            p_17_1 = 1;
        }
        float hi_18_1 = fmaxf(v[8], v[9]);
        float lo_19_1 = fminf(v[8], v[9]);
        unsigned int xi_20_1 = x[8];
        unsigned int xj_21_1 = x[9];
        v[8] = hi_18_1;
        v[9] = lo_19_1;
        x[8] = ((p_17_1 != 0) ? xi_20_1 : xj_21_1);
        x[9] = ((p_17_1 != 0) ? xj_21_1 : xi_20_1);
        if (p_17_1 == 0) {
            sw_1 = 1;
        }
        int p_22_1 = 0;
        if (v[10] >= v[11]) {
            p_22_1 = 1;
        }
        float hi_23_1 = fmaxf(v[10], v[11]);
        float lo_24_1 = fminf(v[10], v[11]);
        unsigned int xi_25_1 = x[10];
        unsigned int xj_26_1 = x[11];
        v[10] = hi_23_1;
        v[11] = lo_24_1;
        x[10] = ((p_22_1 != 0) ? xi_25_1 : xj_26_1);
        x[11] = ((p_22_1 != 0) ? xj_26_1 : xi_25_1);
        if (p_22_1 == 0) {
            sw_1 = 1;
        }
        int p_27_1 = 0;
        if (v[12] >= v[13]) {
            p_27_1 = 1;
        }
        float hi_28_1 = fmaxf(v[12], v[13]);
        float lo_29_1 = fminf(v[12], v[13]);
        unsigned int xi_30_1 = x[12];
        unsigned int xj_31_1 = x[13];
        v[12] = hi_28_1;
        v[13] = lo_29_1;
        x[12] = ((p_27_1 != 0) ? xi_30_1 : xj_31_1);
        x[13] = ((p_27_1 != 0) ? xj_31_1 : xi_30_1);
        if (p_27_1 == 0) {
            sw_1 = 1;
        }
        int p_32_1 = 0;
        if (v[14] >= v[15]) {
            p_32_1 = 1;
        }
        float hi_33_1 = fmaxf(v[14], v[15]);
        float lo_34_1 = fminf(v[14], v[15]);
        unsigned int xi_35_1 = x[14];
        unsigned int xj_36_1 = x[15];
        v[14] = hi_33_1;
        v[15] = lo_34_1;
        x[14] = ((p_32_1 != 0) ? xi_35_1 : xj_36_1);
        x[15] = ((p_32_1 != 0) ? xj_36_1 : xi_35_1);
        if (p_32_1 == 0) {
            sw_1 = 1;
        }
        int p_37_1 = 0;
        if (v[1] >= v[2]) {
            p_37_1 = 1;
        }
        float hi_38_1 = fmaxf(v[1], v[2]);
        float lo_39_1 = fminf(v[1], v[2]);
        unsigned int xi_40_1 = x[1];
        unsigned int xj_41_1 = x[2];
        v[1] = hi_38_1;
        v[2] = lo_39_1;
        x[1] = ((p_37_1 != 0) ? xi_40_1 : xj_41_1);
        x[2] = ((p_37_1 != 0) ? xj_41_1 : xi_40_1);
        if (p_37_1 == 0) {
            sw_1 = 1;
        }
        int p_42_1 = 0;
        if (v[3] >= v[4]) {
            p_42_1 = 1;
        }
        float hi_43_1 = fmaxf(v[3], v[4]);
        float lo_44_1 = fminf(v[3], v[4]);
        unsigned int xi_45_1 = x[3];
        unsigned int xj_46_1 = x[4];
        v[3] = hi_43_1;
        v[4] = lo_44_1;
        x[3] = ((p_42_1 != 0) ? xi_45_1 : xj_46_1);
        x[4] = ((p_42_1 != 0) ? xj_46_1 : xi_45_1);
        if (p_42_1 == 0) {
            sw_1 = 1;
        }
        int p_47_1 = 0;
        if (v[5] >= v[6]) {
            p_47_1 = 1;
        }
        float hi_48_1 = fmaxf(v[5], v[6]);
        float lo_49_1 = fminf(v[5], v[6]);
        unsigned int xi_50_1 = x[5];
        unsigned int xj_51_1 = x[6];
        v[5] = hi_48_1;
        v[6] = lo_49_1;
        x[5] = ((p_47_1 != 0) ? xi_50_1 : xj_51_1);
        x[6] = ((p_47_1 != 0) ? xj_51_1 : xi_50_1);
        if (p_47_1 == 0) {
            sw_1 = 1;
        }
        int p_52_1 = 0;
        if (v[7] >= v[8]) {
            p_52_1 = 1;
        }
        float hi_53_1 = fmaxf(v[7], v[8]);
        float lo_54_1 = fminf(v[7], v[8]);
        unsigned int xi_55_1 = x[7];
        unsigned int xj_56_1 = x[8];
        v[7] = hi_53_1;
        v[8] = lo_54_1;
        x[7] = ((p_52_1 != 0) ? xi_55_1 : xj_56_1);
        x[8] = ((p_52_1 != 0) ? xj_56_1 : xi_55_1);
        if (p_52_1 == 0) {
            sw_1 = 1;
        }
        int p_57_1 = 0;
        if (v[9] >= v[10]) {
            p_57_1 = 1;
        }
        float hi_58_1 = fmaxf(v[9], v[10]);
        float lo_59_1 = fminf(v[9], v[10]);
        unsigned int xi_60_1 = x[9];
        unsigned int xj_61_1 = x[10];
        v[9] = hi_58_1;
        v[10] = lo_59_1;
        x[9] = ((p_57_1 != 0) ? xi_60_1 : xj_61_1);
        x[10] = ((p_57_1 != 0) ? xj_61_1 : xi_60_1);
        if (p_57_1 == 0) {
            sw_1 = 1;
        }
        int p_62_1 = 0;
        if (v[11] >= v[12]) {
            p_62_1 = 1;
        }
        float hi_63_1 = fmaxf(v[11], v[12]);
        float lo_64_1 = fminf(v[11], v[12]);
        unsigned int xi_65_1 = x[11];
        unsigned int xj_66_1 = x[12];
        v[11] = hi_63_1;
        v[12] = lo_64_1;
        x[11] = ((p_62_1 != 0) ? xi_65_1 : xj_66_1);
        x[12] = ((p_62_1 != 0) ? xj_66_1 : xi_65_1);
        if (p_62_1 == 0) {
            sw_1 = 1;
        }
        int p_67_1 = 0;
        if (v[13] >= v[14]) {
            p_67_1 = 1;
        }
        float hi_68_1 = fmaxf(v[13], v[14]);
        float lo_69_1 = fminf(v[13], v[14]);
        unsigned int xi_70_1 = x[13];
        unsigned int xj_71_1 = x[14];
        v[13] = hi_68_1;
        v[14] = lo_69_1;
        x[13] = ((p_67_1 != 0) ? xi_70_1 : xj_71_1);
        x[14] = ((p_67_1 != 0) ? xj_71_1 : xi_70_1);
        if (p_67_1 == 0) {
            sw_1 = 1;
        }
        int _warp_vote_1 = __any_sync(0xFFFFFFFF, sw_1 != 0);
        more = _warp_vote_1;
    }
    if (more != 0) {
        int sw_2 = 0;
        int p_3 = 0;
        if (v[0] >= v[1]) {
            p_3 = 1;
        }
        float hi_0_2 = fmaxf(v[0], v[1]);
        float lo_1_2 = fminf(v[0], v[1]);
        unsigned int xi_2 = x[0];
        unsigned int xj_2 = x[1];
        v[0] = hi_0_2;
        v[1] = lo_1_2;
        x[0] = ((p_3 != 0) ? xi_2 : xj_2);
        x[1] = ((p_3 != 0) ? xj_2 : xi_2);
        if (p_3 == 0) {
            sw_2 = 1;
        }
        int p_2_2 = 0;
        if (v[2] >= v[3]) {
            p_2_2 = 1;
        }
        float hi_3_2 = fmaxf(v[2], v[3]);
        float lo_4_2 = fminf(v[2], v[3]);
        unsigned int xi_5_2 = x[2];
        unsigned int xj_6_2 = x[3];
        v[2] = hi_3_2;
        v[3] = lo_4_2;
        x[2] = ((p_2_2 != 0) ? xi_5_2 : xj_6_2);
        x[3] = ((p_2_2 != 0) ? xj_6_2 : xi_5_2);
        if (p_2_2 == 0) {
            sw_2 = 1;
        }
        int p_7_2 = 0;
        if (v[4] >= v[5]) {
            p_7_2 = 1;
        }
        float hi_8_2 = fmaxf(v[4], v[5]);
        float lo_9_2 = fminf(v[4], v[5]);
        unsigned int xi_10_2 = x[4];
        unsigned int xj_11_2 = x[5];
        v[4] = hi_8_2;
        v[5] = lo_9_2;
        x[4] = ((p_7_2 != 0) ? xi_10_2 : xj_11_2);
        x[5] = ((p_7_2 != 0) ? xj_11_2 : xi_10_2);
        if (p_7_2 == 0) {
            sw_2 = 1;
        }
        int p_12_2 = 0;
        if (v[6] >= v[7]) {
            p_12_2 = 1;
        }
        float hi_13_2 = fmaxf(v[6], v[7]);
        float lo_14_2 = fminf(v[6], v[7]);
        unsigned int xi_15_2 = x[6];
        unsigned int xj_16_2 = x[7];
        v[6] = hi_13_2;
        v[7] = lo_14_2;
        x[6] = ((p_12_2 != 0) ? xi_15_2 : xj_16_2);
        x[7] = ((p_12_2 != 0) ? xj_16_2 : xi_15_2);
        if (p_12_2 == 0) {
            sw_2 = 1;
        }
        int p_17_2 = 0;
        if (v[8] >= v[9]) {
            p_17_2 = 1;
        }
        float hi_18_2 = fmaxf(v[8], v[9]);
        float lo_19_2 = fminf(v[8], v[9]);
        unsigned int xi_20_2 = x[8];
        unsigned int xj_21_2 = x[9];
        v[8] = hi_18_2;
        v[9] = lo_19_2;
        x[8] = ((p_17_2 != 0) ? xi_20_2 : xj_21_2);
        x[9] = ((p_17_2 != 0) ? xj_21_2 : xi_20_2);
        if (p_17_2 == 0) {
            sw_2 = 1;
        }
        int p_22_2 = 0;
        if (v[10] >= v[11]) {
            p_22_2 = 1;
        }
        float hi_23_2 = fmaxf(v[10], v[11]);
        float lo_24_2 = fminf(v[10], v[11]);
        unsigned int xi_25_2 = x[10];
        unsigned int xj_26_2 = x[11];
        v[10] = hi_23_2;
        v[11] = lo_24_2;
        x[10] = ((p_22_2 != 0) ? xi_25_2 : xj_26_2);
        x[11] = ((p_22_2 != 0) ? xj_26_2 : xi_25_2);
        if (p_22_2 == 0) {
            sw_2 = 1;
        }
        int p_27_2 = 0;
        if (v[12] >= v[13]) {
            p_27_2 = 1;
        }
        float hi_28_2 = fmaxf(v[12], v[13]);
        float lo_29_2 = fminf(v[12], v[13]);
        unsigned int xi_30_2 = x[12];
        unsigned int xj_31_2 = x[13];
        v[12] = hi_28_2;
        v[13] = lo_29_2;
        x[12] = ((p_27_2 != 0) ? xi_30_2 : xj_31_2);
        x[13] = ((p_27_2 != 0) ? xj_31_2 : xi_30_2);
        if (p_27_2 == 0) {
            sw_2 = 1;
        }
        int p_32_2 = 0;
        if (v[14] >= v[15]) {
            p_32_2 = 1;
        }
        float hi_33_2 = fmaxf(v[14], v[15]);
        float lo_34_2 = fminf(v[14], v[15]);
        unsigned int xi_35_2 = x[14];
        unsigned int xj_36_2 = x[15];
        v[14] = hi_33_2;
        v[15] = lo_34_2;
        x[14] = ((p_32_2 != 0) ? xi_35_2 : xj_36_2);
        x[15] = ((p_32_2 != 0) ? xj_36_2 : xi_35_2);
        if (p_32_2 == 0) {
            sw_2 = 1;
        }
        int p_37_2 = 0;
        if (v[1] >= v[2]) {
            p_37_2 = 1;
        }
        float hi_38_2 = fmaxf(v[1], v[2]);
        float lo_39_2 = fminf(v[1], v[2]);
        unsigned int xi_40_2 = x[1];
        unsigned int xj_41_2 = x[2];
        v[1] = hi_38_2;
        v[2] = lo_39_2;
        x[1] = ((p_37_2 != 0) ? xi_40_2 : xj_41_2);
        x[2] = ((p_37_2 != 0) ? xj_41_2 : xi_40_2);
        if (p_37_2 == 0) {
            sw_2 = 1;
        }
        int p_42_2 = 0;
        if (v[3] >= v[4]) {
            p_42_2 = 1;
        }
        float hi_43_2 = fmaxf(v[3], v[4]);
        float lo_44_2 = fminf(v[3], v[4]);
        unsigned int xi_45_2 = x[3];
        unsigned int xj_46_2 = x[4];
        v[3] = hi_43_2;
        v[4] = lo_44_2;
        x[3] = ((p_42_2 != 0) ? xi_45_2 : xj_46_2);
        x[4] = ((p_42_2 != 0) ? xj_46_2 : xi_45_2);
        if (p_42_2 == 0) {
            sw_2 = 1;
        }
        int p_47_2 = 0;
        if (v[5] >= v[6]) {
            p_47_2 = 1;
        }
        float hi_48_2 = fmaxf(v[5], v[6]);
        float lo_49_2 = fminf(v[5], v[6]);
        unsigned int xi_50_2 = x[5];
        unsigned int xj_51_2 = x[6];
        v[5] = hi_48_2;
        v[6] = lo_49_2;
        x[5] = ((p_47_2 != 0) ? xi_50_2 : xj_51_2);
        x[6] = ((p_47_2 != 0) ? xj_51_2 : xi_50_2);
        if (p_47_2 == 0) {
            sw_2 = 1;
        }
        int p_52_2 = 0;
        if (v[7] >= v[8]) {
            p_52_2 = 1;
        }
        float hi_53_2 = fmaxf(v[7], v[8]);
        float lo_54_2 = fminf(v[7], v[8]);
        unsigned int xi_55_2 = x[7];
        unsigned int xj_56_2 = x[8];
        v[7] = hi_53_2;
        v[8] = lo_54_2;
        x[7] = ((p_52_2 != 0) ? xi_55_2 : xj_56_2);
        x[8] = ((p_52_2 != 0) ? xj_56_2 : xi_55_2);
        if (p_52_2 == 0) {
            sw_2 = 1;
        }
        int p_57_2 = 0;
        if (v[9] >= v[10]) {
            p_57_2 = 1;
        }
        float hi_58_2 = fmaxf(v[9], v[10]);
        float lo_59_2 = fminf(v[9], v[10]);
        unsigned int xi_60_2 = x[9];
        unsigned int xj_61_2 = x[10];
        v[9] = hi_58_2;
        v[10] = lo_59_2;
        x[9] = ((p_57_2 != 0) ? xi_60_2 : xj_61_2);
        x[10] = ((p_57_2 != 0) ? xj_61_2 : xi_60_2);
        if (p_57_2 == 0) {
            sw_2 = 1;
        }
        int p_62_2 = 0;
        if (v[11] >= v[12]) {
            p_62_2 = 1;
        }
        float hi_63_2 = fmaxf(v[11], v[12]);
        float lo_64_2 = fminf(v[11], v[12]);
        unsigned int xi_65_2 = x[11];
        unsigned int xj_66_2 = x[12];
        v[11] = hi_63_2;
        v[12] = lo_64_2;
        x[11] = ((p_62_2 != 0) ? xi_65_2 : xj_66_2);
        x[12] = ((p_62_2 != 0) ? xj_66_2 : xi_65_2);
        if (p_62_2 == 0) {
            sw_2 = 1;
        }
        int p_67_2 = 0;
        if (v[13] >= v[14]) {
            p_67_2 = 1;
        }
        float hi_68_2 = fmaxf(v[13], v[14]);
        float lo_69_2 = fminf(v[13], v[14]);
        unsigned int xi_70_2 = x[13];
        unsigned int xj_71_2 = x[14];
        v[13] = hi_68_2;
        v[14] = lo_69_2;
        x[13] = ((p_67_2 != 0) ? xi_70_2 : xj_71_2);
        x[14] = ((p_67_2 != 0) ? xj_71_2 : xi_70_2);
        if (p_67_2 == 0) {
            sw_2 = 1;
        }
        int _warp_vote_2 = __any_sync(0xFFFFFFFF, sw_2 != 0);
        more = _warp_vote_2;
    }
    if (more != 0) {
        int p_4 = 0;
        if (v[0] >= v[13]) {
            p_4 = 1;
        }
        float hi_0_3 = fmaxf(v[0], v[13]);
        float lo_1_3 = fminf(v[0], v[13]);
        unsigned int xi_3 = x[0];
        unsigned int xj_3 = x[13];
        v[0] = hi_0_3;
        v[13] = lo_1_3;
        x[0] = ((p_4 != 0) ? xi_3 : xj_3);
        x[13] = ((p_4 != 0) ? xj_3 : xi_3);
        int p_2_3 = 0;
        if (v[1] >= v[12]) {
            p_2_3 = 1;
        }
        float hi_3_3 = fmaxf(v[1], v[12]);
        float lo_4_3 = fminf(v[1], v[12]);
        unsigned int xi_5_3 = x[1];
        unsigned int xj_6_3 = x[12];
        v[1] = hi_3_3;
        v[12] = lo_4_3;
        x[1] = ((p_2_3 != 0) ? xi_5_3 : xj_6_3);
        x[12] = ((p_2_3 != 0) ? xj_6_3 : xi_5_3);
        int p_7_3 = 0;
        if (v[2] >= v[15]) {
            p_7_3 = 1;
        }
        float hi_8_3 = fmaxf(v[2], v[15]);
        float lo_9_3 = fminf(v[2], v[15]);
        unsigned int xi_10_3 = x[2];
        unsigned int xj_11_3 = x[15];
        v[2] = hi_8_3;
        v[15] = lo_9_3;
        x[2] = ((p_7_3 != 0) ? xi_10_3 : xj_11_3);
        x[15] = ((p_7_3 != 0) ? xj_11_3 : xi_10_3);
        int p_12_3 = 0;
        if (v[3] >= v[14]) {
            p_12_3 = 1;
        }
        float hi_13_3 = fmaxf(v[3], v[14]);
        float lo_14_3 = fminf(v[3], v[14]);
        unsigned int xi_15_3 = x[3];
        unsigned int xj_16_3 = x[14];
        v[3] = hi_13_3;
        v[14] = lo_14_3;
        x[3] = ((p_12_3 != 0) ? xi_15_3 : xj_16_3);
        x[14] = ((p_12_3 != 0) ? xj_16_3 : xi_15_3);
        int p_17_3 = 0;
        if (v[4] >= v[8]) {
            p_17_3 = 1;
        }
        float hi_18_3 = fmaxf(v[4], v[8]);
        float lo_19_3 = fminf(v[4], v[8]);
        unsigned int xi_20_3 = x[4];
        unsigned int xj_21_3 = x[8];
        v[4] = hi_18_3;
        v[8] = lo_19_3;
        x[4] = ((p_17_3 != 0) ? xi_20_3 : xj_21_3);
        x[8] = ((p_17_3 != 0) ? xj_21_3 : xi_20_3);
        int p_22_3 = 0;
        if (v[5] >= v[6]) {
            p_22_3 = 1;
        }
        float hi_23_3 = fmaxf(v[5], v[6]);
        float lo_24_3 = fminf(v[5], v[6]);
        unsigned int xi_25_3 = x[5];
        unsigned int xj_26_3 = x[6];
        v[5] = hi_23_3;
        v[6] = lo_24_3;
        x[5] = ((p_22_3 != 0) ? xi_25_3 : xj_26_3);
        x[6] = ((p_22_3 != 0) ? xj_26_3 : xi_25_3);
        int p_27_3 = 0;
        if (v[7] >= v[11]) {
            p_27_3 = 1;
        }
        float hi_28_3 = fmaxf(v[7], v[11]);
        float lo_29_3 = fminf(v[7], v[11]);
        unsigned int xi_30_3 = x[7];
        unsigned int xj_31_3 = x[11];
        v[7] = hi_28_3;
        v[11] = lo_29_3;
        x[7] = ((p_27_3 != 0) ? xi_30_3 : xj_31_3);
        x[11] = ((p_27_3 != 0) ? xj_31_3 : xi_30_3);
        int p_32_3 = 0;
        if (v[9] >= v[10]) {
            p_32_3 = 1;
        }
        float hi_33_3 = fmaxf(v[9], v[10]);
        float lo_34_3 = fminf(v[9], v[10]);
        unsigned int xi_35_3 = x[9];
        unsigned int xj_36_3 = x[10];
        v[9] = hi_33_3;
        v[10] = lo_34_3;
        x[9] = ((p_32_3 != 0) ? xi_35_3 : xj_36_3);
        x[10] = ((p_32_3 != 0) ? xj_36_3 : xi_35_3);
        int p_37_3 = 0;
        if (v[0] >= v[5]) {
            p_37_3 = 1;
        }
        float hi_38_3 = fmaxf(v[0], v[5]);
        float lo_39_3 = fminf(v[0], v[5]);
        unsigned int xi_40_3 = x[0];
        unsigned int xj_41_3 = x[5];
        v[0] = hi_38_3;
        v[5] = lo_39_3;
        x[0] = ((p_37_3 != 0) ? xi_40_3 : xj_41_3);
        x[5] = ((p_37_3 != 0) ? xj_41_3 : xi_40_3);
        int p_42_3 = 0;
        if (v[1] >= v[7]) {
            p_42_3 = 1;
        }
        float hi_43_3 = fmaxf(v[1], v[7]);
        float lo_44_3 = fminf(v[1], v[7]);
        unsigned int xi_45_3 = x[1];
        unsigned int xj_46_3 = x[7];
        v[1] = hi_43_3;
        v[7] = lo_44_3;
        x[1] = ((p_42_3 != 0) ? xi_45_3 : xj_46_3);
        x[7] = ((p_42_3 != 0) ? xj_46_3 : xi_45_3);
        int p_47_3 = 0;
        if (v[2] >= v[9]) {
            p_47_3 = 1;
        }
        float hi_48_3 = fmaxf(v[2], v[9]);
        float lo_49_3 = fminf(v[2], v[9]);
        unsigned int xi_50_3 = x[2];
        unsigned int xj_51_3 = x[9];
        v[2] = hi_48_3;
        v[9] = lo_49_3;
        x[2] = ((p_47_3 != 0) ? xi_50_3 : xj_51_3);
        x[9] = ((p_47_3 != 0) ? xj_51_3 : xi_50_3);
        int p_52_3 = 0;
        if (v[3] >= v[4]) {
            p_52_3 = 1;
        }
        float hi_53_3 = fmaxf(v[3], v[4]);
        float lo_54_3 = fminf(v[3], v[4]);
        unsigned int xi_55_3 = x[3];
        unsigned int xj_56_3 = x[4];
        v[3] = hi_53_3;
        v[4] = lo_54_3;
        x[3] = ((p_52_3 != 0) ? xi_55_3 : xj_56_3);
        x[4] = ((p_52_3 != 0) ? xj_56_3 : xi_55_3);
        int p_57_3 = 0;
        if (v[6] >= v[13]) {
            p_57_3 = 1;
        }
        float hi_58_3 = fmaxf(v[6], v[13]);
        float lo_59_3 = fminf(v[6], v[13]);
        unsigned int xi_60_3 = x[6];
        unsigned int xj_61_3 = x[13];
        v[6] = hi_58_3;
        v[13] = lo_59_3;
        x[6] = ((p_57_3 != 0) ? xi_60_3 : xj_61_3);
        x[13] = ((p_57_3 != 0) ? xj_61_3 : xi_60_3);
        int p_62_3 = 0;
        if (v[8] >= v[14]) {
            p_62_3 = 1;
        }
        float hi_63_3 = fmaxf(v[8], v[14]);
        float lo_64_3 = fminf(v[8], v[14]);
        unsigned int xi_65_3 = x[8];
        unsigned int xj_66_3 = x[14];
        v[8] = hi_63_3;
        v[14] = lo_64_3;
        x[8] = ((p_62_3 != 0) ? xi_65_3 : xj_66_3);
        x[14] = ((p_62_3 != 0) ? xj_66_3 : xi_65_3);
        int p_67_3 = 0;
        if (v[10] >= v[15]) {
            p_67_3 = 1;
        }
        float hi_68_3 = fmaxf(v[10], v[15]);
        float lo_69_3 = fminf(v[10], v[15]);
        unsigned int xi_70_3 = x[10];
        unsigned int xj_71_3 = x[15];
        v[10] = hi_68_3;
        v[15] = lo_69_3;
        x[10] = ((p_67_3 != 0) ? xi_70_3 : xj_71_3);
        x[15] = ((p_67_3 != 0) ? xj_71_3 : xi_70_3);
        int p_72 = 0;
        if (v[11] >= v[12]) {
            p_72 = 1;
        }
        float hi_73 = fmaxf(v[11], v[12]);
        float lo_74 = fminf(v[11], v[12]);
        unsigned int xi_75 = x[11];
        unsigned int xj_76 = x[12];
        v[11] = hi_73;
        v[12] = lo_74;
        x[11] = ((p_72 != 0) ? xi_75 : xj_76);
        x[12] = ((p_72 != 0) ? xj_76 : xi_75);
        int p_77 = 0;
        if (v[0] >= v[1]) {
            p_77 = 1;
        }
        float hi_78 = fmaxf(v[0], v[1]);
        float lo_79 = fminf(v[0], v[1]);
        unsigned int xi_80 = x[0];
        unsigned int xj_81 = x[1];
        v[0] = hi_78;
        v[1] = lo_79;
        x[0] = ((p_77 != 0) ? xi_80 : xj_81);
        x[1] = ((p_77 != 0) ? xj_81 : xi_80);
        int p_82 = 0;
        if (v[2] >= v[3]) {
            p_82 = 1;
        }
        float hi_84 = fmaxf(v[2], v[3]);
        float lo_85 = fminf(v[2], v[3]);
        unsigned int xi_86 = x[2];
        unsigned int xj_87 = x[3];
        v[2] = hi_84;
        v[3] = lo_85;
        x[2] = ((p_82 != 0) ? xi_86 : xj_87);
        x[3] = ((p_82 != 0) ? xj_87 : xi_86);
        int p_88 = 0;
        if (v[4] >= v[5]) {
            p_88 = 1;
        }
        float hi_90 = fmaxf(v[4], v[5]);
        float lo_91 = fminf(v[4], v[5]);
        unsigned int xi_92 = x[4];
        unsigned int xj_93 = x[5];
        v[4] = hi_90;
        v[5] = lo_91;
        x[4] = ((p_88 != 0) ? xi_92 : xj_93);
        x[5] = ((p_88 != 0) ? xj_93 : xi_92);
        int p_94 = 0;
        if (v[6] >= v[8]) {
            p_94 = 1;
        }
        float hi_96 = fmaxf(v[6], v[8]);
        float lo_97 = fminf(v[6], v[8]);
        unsigned int xi_98 = x[6];
        unsigned int xj_99 = x[8];
        v[6] = hi_96;
        v[8] = lo_97;
        x[6] = ((p_94 != 0) ? xi_98 : xj_99);
        x[8] = ((p_94 != 0) ? xj_99 : xi_98);
        int p_100 = 0;
        if (v[7] >= v[9]) {
            p_100 = 1;
        }
        float hi_102 = fmaxf(v[7], v[9]);
        float lo_103 = fminf(v[7], v[9]);
        unsigned int xi_104 = x[7];
        unsigned int xj_105 = x[9];
        v[7] = hi_102;
        v[9] = lo_103;
        x[7] = ((p_100 != 0) ? xi_104 : xj_105);
        x[9] = ((p_100 != 0) ? xj_105 : xi_104);
        int p_106 = 0;
        if (v[10] >= v[11]) {
            p_106 = 1;
        }
        float hi_108 = fmaxf(v[10], v[11]);
        float lo_109 = fminf(v[10], v[11]);
        unsigned int xi_110 = x[10];
        unsigned int xj_111 = x[11];
        v[10] = hi_108;
        v[11] = lo_109;
        x[10] = ((p_106 != 0) ? xi_110 : xj_111);
        x[11] = ((p_106 != 0) ? xj_111 : xi_110);
        int p_112 = 0;
        if (v[12] >= v[13]) {
            p_112 = 1;
        }
        float hi_114 = fmaxf(v[12], v[13]);
        float lo_115 = fminf(v[12], v[13]);
        unsigned int xi_116 = x[12];
        unsigned int xj_117 = x[13];
        v[12] = hi_114;
        v[13] = lo_115;
        x[12] = ((p_112 != 0) ? xi_116 : xj_117);
        x[13] = ((p_112 != 0) ? xj_117 : xi_116);
        int p_118 = 0;
        if (v[14] >= v[15]) {
            p_118 = 1;
        }
        float hi_120 = fmaxf(v[14], v[15]);
        float lo_121 = fminf(v[14], v[15]);
        unsigned int xi_122 = x[14];
        unsigned int xj_123 = x[15];
        v[14] = hi_120;
        v[15] = lo_121;
        x[14] = ((p_118 != 0) ? xi_122 : xj_123);
        x[15] = ((p_118 != 0) ? xj_123 : xi_122);
        int p_124 = 0;
        if (v[0] >= v[2]) {
            p_124 = 1;
        }
        float hi_126 = fmaxf(v[0], v[2]);
        float lo_127 = fminf(v[0], v[2]);
        unsigned int xi_128 = x[0];
        unsigned int xj_129 = x[2];
        v[0] = hi_126;
        v[2] = lo_127;
        x[0] = ((p_124 != 0) ? xi_128 : xj_129);
        x[2] = ((p_124 != 0) ? xj_129 : xi_128);
        int p_130 = 0;
        if (v[1] >= v[3]) {
            p_130 = 1;
        }
        float hi_132 = fmaxf(v[1], v[3]);
        float lo_133 = fminf(v[1], v[3]);
        unsigned int xi_134 = x[1];
        unsigned int xj_135 = x[3];
        v[1] = hi_132;
        v[3] = lo_133;
        x[1] = ((p_130 != 0) ? xi_134 : xj_135);
        x[3] = ((p_130 != 0) ? xj_135 : xi_134);
        int p_136 = 0;
        if (v[4] >= v[10]) {
            p_136 = 1;
        }
        float hi_138 = fmaxf(v[4], v[10]);
        float lo_139 = fminf(v[4], v[10]);
        unsigned int xi_140 = x[4];
        unsigned int xj_141 = x[10];
        v[4] = hi_138;
        v[10] = lo_139;
        x[4] = ((p_136 != 0) ? xi_140 : xj_141);
        x[10] = ((p_136 != 0) ? xj_141 : xi_140);
        int p_142 = 0;
        if (v[5] >= v[11]) {
            p_142 = 1;
        }
        float hi_144 = fmaxf(v[5], v[11]);
        float lo_145 = fminf(v[5], v[11]);
        unsigned int xi_146 = x[5];
        unsigned int xj_147 = x[11];
        v[5] = hi_144;
        v[11] = lo_145;
        x[5] = ((p_142 != 0) ? xi_146 : xj_147);
        x[11] = ((p_142 != 0) ? xj_147 : xi_146);
        int p_148 = 0;
        if (v[6] >= v[7]) {
            p_148 = 1;
        }
        float hi_150 = fmaxf(v[6], v[7]);
        float lo_151 = fminf(v[6], v[7]);
        unsigned int xi_152 = x[6];
        unsigned int xj_153 = x[7];
        v[6] = hi_150;
        v[7] = lo_151;
        x[6] = ((p_148 != 0) ? xi_152 : xj_153);
        x[7] = ((p_148 != 0) ? xj_153 : xi_152);
        int p_154 = 0;
        if (v[8] >= v[9]) {
            p_154 = 1;
        }
        float hi_156 = fmaxf(v[8], v[9]);
        float lo_157 = fminf(v[8], v[9]);
        unsigned int xi_158 = x[8];
        unsigned int xj_159 = x[9];
        v[8] = hi_156;
        v[9] = lo_157;
        x[8] = ((p_154 != 0) ? xi_158 : xj_159);
        x[9] = ((p_154 != 0) ? xj_159 : xi_158);
        int p_160 = 0;
        if (v[12] >= v[14]) {
            p_160 = 1;
        }
        float hi_162 = fmaxf(v[12], v[14]);
        float lo_163 = fminf(v[12], v[14]);
        unsigned int xi_164 = x[12];
        unsigned int xj_165 = x[14];
        v[12] = hi_162;
        v[14] = lo_163;
        x[12] = ((p_160 != 0) ? xi_164 : xj_165);
        x[14] = ((p_160 != 0) ? xj_165 : xi_164);
        int p_166 = 0;
        if (v[13] >= v[15]) {
            p_166 = 1;
        }
        float hi_168 = fmaxf(v[13], v[15]);
        float lo_169 = fminf(v[13], v[15]);
        unsigned int xi_170 = x[13];
        unsigned int xj_171 = x[15];
        v[13] = hi_168;
        v[15] = lo_169;
        x[13] = ((p_166 != 0) ? xi_170 : xj_171);
        x[15] = ((p_166 != 0) ? xj_171 : xi_170);
        int p_172 = 0;
        if (v[1] >= v[2]) {
            p_172 = 1;
        }
        float hi_174 = fmaxf(v[1], v[2]);
        float lo_175 = fminf(v[1], v[2]);
        unsigned int xi_176 = x[1];
        unsigned int xj_177 = x[2];
        v[1] = hi_174;
        v[2] = lo_175;
        x[1] = ((p_172 != 0) ? xi_176 : xj_177);
        x[2] = ((p_172 != 0) ? xj_177 : xi_176);
        int p_178 = 0;
        if (v[3] >= v[12]) {
            p_178 = 1;
        }
        float hi_180 = fmaxf(v[3], v[12]);
        float lo_181 = fminf(v[3], v[12]);
        unsigned int xi_182 = x[3];
        unsigned int xj_183 = x[12];
        v[3] = hi_180;
        v[12] = lo_181;
        x[3] = ((p_178 != 0) ? xi_182 : xj_183);
        x[12] = ((p_178 != 0) ? xj_183 : xi_182);
        int p_184 = 0;
        if (v[4] >= v[6]) {
            p_184 = 1;
        }
        float hi_186 = fmaxf(v[4], v[6]);
        float lo_187 = fminf(v[4], v[6]);
        unsigned int xi_188 = x[4];
        unsigned int xj_189 = x[6];
        v[4] = hi_186;
        v[6] = lo_187;
        x[4] = ((p_184 != 0) ? xi_188 : xj_189);
        x[6] = ((p_184 != 0) ? xj_189 : xi_188);
        int p_190 = 0;
        if (v[5] >= v[7]) {
            p_190 = 1;
        }
        float hi_192 = fmaxf(v[5], v[7]);
        float lo_193 = fminf(v[5], v[7]);
        unsigned int xi_194 = x[5];
        unsigned int xj_195 = x[7];
        v[5] = hi_192;
        v[7] = lo_193;
        x[5] = ((p_190 != 0) ? xi_194 : xj_195);
        x[7] = ((p_190 != 0) ? xj_195 : xi_194);
        int p_196 = 0;
        if (v[8] >= v[10]) {
            p_196 = 1;
        }
        float hi_197 = fmaxf(v[8], v[10]);
        float lo_198 = fminf(v[8], v[10]);
        unsigned int xi_199 = x[8];
        unsigned int xj_200 = x[10];
        v[8] = hi_197;
        v[10] = lo_198;
        x[8] = ((p_196 != 0) ? xi_199 : xj_200);
        x[10] = ((p_196 != 0) ? xj_200 : xi_199);
        int p_201 = 0;
        if (v[9] >= v[11]) {
            p_201 = 1;
        }
        float hi_202 = fmaxf(v[9], v[11]);
        float lo_203 = fminf(v[9], v[11]);
        unsigned int xi_204 = x[9];
        unsigned int xj_205 = x[11];
        v[9] = hi_202;
        v[11] = lo_203;
        x[9] = ((p_201 != 0) ? xi_204 : xj_205);
        x[11] = ((p_201 != 0) ? xj_205 : xi_204);
        int p_206 = 0;
        if (v[13] >= v[14]) {
            p_206 = 1;
        }
        float hi_207 = fmaxf(v[13], v[14]);
        float lo_208 = fminf(v[13], v[14]);
        unsigned int xi_209 = x[13];
        unsigned int xj_210 = x[14];
        v[13] = hi_207;
        v[14] = lo_208;
        x[13] = ((p_206 != 0) ? xi_209 : xj_210);
        x[14] = ((p_206 != 0) ? xj_210 : xi_209);
        int p_211 = 0;
        if (v[1] >= v[4]) {
            p_211 = 1;
        }
        float hi_212 = fmaxf(v[1], v[4]);
        float lo_213 = fminf(v[1], v[4]);
        unsigned int xi_214 = x[1];
        unsigned int xj_215 = x[4];
        v[1] = hi_212;
        v[4] = lo_213;
        x[1] = ((p_211 != 0) ? xi_214 : xj_215);
        x[4] = ((p_211 != 0) ? xj_215 : xi_214);
        int p_216 = 0;
        if (v[2] >= v[6]) {
            p_216 = 1;
        }
        float hi_217 = fmaxf(v[2], v[6]);
        float lo_218 = fminf(v[2], v[6]);
        unsigned int xi_219 = x[2];
        unsigned int xj_220 = x[6];
        v[2] = hi_217;
        v[6] = lo_218;
        x[2] = ((p_216 != 0) ? xi_219 : xj_220);
        x[6] = ((p_216 != 0) ? xj_220 : xi_219);
        int p_221 = 0;
        if (v[5] >= v[8]) {
            p_221 = 1;
        }
        float hi_222 = fmaxf(v[5], v[8]);
        float lo_223 = fminf(v[5], v[8]);
        unsigned int xi_224 = x[5];
        unsigned int xj_225 = x[8];
        v[5] = hi_222;
        v[8] = lo_223;
        x[5] = ((p_221 != 0) ? xi_224 : xj_225);
        x[8] = ((p_221 != 0) ? xj_225 : xi_224);
        int p_226 = 0;
        if (v[7] >= v[10]) {
            p_226 = 1;
        }
        float hi_227 = fmaxf(v[7], v[10]);
        float lo_228 = fminf(v[7], v[10]);
        unsigned int xi_229 = x[7];
        unsigned int xj_230 = x[10];
        v[7] = hi_227;
        v[10] = lo_228;
        x[7] = ((p_226 != 0) ? xi_229 : xj_230);
        x[10] = ((p_226 != 0) ? xj_230 : xi_229);
        int p_231 = 0;
        if (v[9] >= v[13]) {
            p_231 = 1;
        }
        float hi_232 = fmaxf(v[9], v[13]);
        float lo_233 = fminf(v[9], v[13]);
        unsigned int xi_234 = x[9];
        unsigned int xj_235 = x[13];
        v[9] = hi_232;
        v[13] = lo_233;
        x[9] = ((p_231 != 0) ? xi_234 : xj_235);
        x[13] = ((p_231 != 0) ? xj_235 : xi_234);
        int p_236 = 0;
        if (v[11] >= v[14]) {
            p_236 = 1;
        }
        float hi_237 = fmaxf(v[11], v[14]);
        float lo_238 = fminf(v[11], v[14]);
        unsigned int xi_239 = x[11];
        unsigned int xj_240 = x[14];
        v[11] = hi_237;
        v[14] = lo_238;
        x[11] = ((p_236 != 0) ? xi_239 : xj_240);
        x[14] = ((p_236 != 0) ? xj_240 : xi_239);
        int p_241 = 0;
        if (v[2] >= v[4]) {
            p_241 = 1;
        }
        float hi_242 = fmaxf(v[2], v[4]);
        float lo_243 = fminf(v[2], v[4]);
        unsigned int xi_244 = x[2];
        unsigned int xj_245 = x[4];
        v[2] = hi_242;
        v[4] = lo_243;
        x[2] = ((p_241 != 0) ? xi_244 : xj_245);
        x[4] = ((p_241 != 0) ? xj_245 : xi_244);
        int p_246 = 0;
        if (v[3] >= v[6]) {
            p_246 = 1;
        }
        float hi_247 = fmaxf(v[3], v[6]);
        float lo_248 = fminf(v[3], v[6]);
        unsigned int xi_249 = x[3];
        unsigned int xj_250 = x[6];
        v[3] = hi_247;
        v[6] = lo_248;
        x[3] = ((p_246 != 0) ? xi_249 : xj_250);
        x[6] = ((p_246 != 0) ? xj_250 : xi_249);
        int p_251 = 0;
        if (v[9] >= v[12]) {
            p_251 = 1;
        }
        float hi_252 = fmaxf(v[9], v[12]);
        float lo_253 = fminf(v[9], v[12]);
        unsigned int xi_254 = x[9];
        unsigned int xj_255 = x[12];
        v[9] = hi_252;
        v[12] = lo_253;
        x[9] = ((p_251 != 0) ? xi_254 : xj_255);
        x[12] = ((p_251 != 0) ? xj_255 : xi_254);
        int p_256 = 0;
        if (v[11] >= v[13]) {
            p_256 = 1;
        }
        float hi_257 = fmaxf(v[11], v[13]);
        float lo_258 = fminf(v[11], v[13]);
        unsigned int xi_259 = x[11];
        unsigned int xj_260 = x[13];
        v[11] = hi_257;
        v[13] = lo_258;
        x[11] = ((p_256 != 0) ? xi_259 : xj_260);
        x[13] = ((p_256 != 0) ? xj_260 : xi_259);
        int p_261 = 0;
        if (v[3] >= v[5]) {
            p_261 = 1;
        }
        float hi_262 = fmaxf(v[3], v[5]);
        float lo_263 = fminf(v[3], v[5]);
        unsigned int xi_264 = x[3];
        unsigned int xj_265 = x[5];
        v[3] = hi_262;
        v[5] = lo_263;
        x[3] = ((p_261 != 0) ? xi_264 : xj_265);
        x[5] = ((p_261 != 0) ? xj_265 : xi_264);
        int p_266 = 0;
        if (v[6] >= v[8]) {
            p_266 = 1;
        }
        float hi_267 = fmaxf(v[6], v[8]);
        float lo_268 = fminf(v[6], v[8]);
        unsigned int xi_269 = x[6];
        unsigned int xj_270 = x[8];
        v[6] = hi_267;
        v[8] = lo_268;
        x[6] = ((p_266 != 0) ? xi_269 : xj_270);
        x[8] = ((p_266 != 0) ? xj_270 : xi_269);
        int p_271 = 0;
        if (v[7] >= v[9]) {
            p_271 = 1;
        }
        float hi_272 = fmaxf(v[7], v[9]);
        float lo_273 = fminf(v[7], v[9]);
        unsigned int xi_274 = x[7];
        unsigned int xj_275 = x[9];
        v[7] = hi_272;
        v[9] = lo_273;
        x[7] = ((p_271 != 0) ? xi_274 : xj_275);
        x[9] = ((p_271 != 0) ? xj_275 : xi_274);
        int p_276 = 0;
        if (v[10] >= v[12]) {
            p_276 = 1;
        }
        float hi_277 = fmaxf(v[10], v[12]);
        float lo_278 = fminf(v[10], v[12]);
        unsigned int xi_279 = x[10];
        unsigned int xj_280 = x[12];
        v[10] = hi_277;
        v[12] = lo_278;
        x[10] = ((p_276 != 0) ? xi_279 : xj_280);
        x[12] = ((p_276 != 0) ? xj_280 : xi_279);
        int p_281 = 0;
        if (v[3] >= v[4]) {
            p_281 = 1;
        }
        float hi_282 = fmaxf(v[3], v[4]);
        float lo_283 = fminf(v[3], v[4]);
        unsigned int xi_284 = x[3];
        unsigned int xj_285 = x[4];
        v[3] = hi_282;
        v[4] = lo_283;
        x[3] = ((p_281 != 0) ? xi_284 : xj_285);
        x[4] = ((p_281 != 0) ? xj_285 : xi_284);
        int p_286 = 0;
        if (v[5] >= v[6]) {
            p_286 = 1;
        }
        float hi_287 = fmaxf(v[5], v[6]);
        float lo_288 = fminf(v[5], v[6]);
        unsigned int xi_289 = x[5];
        unsigned int xj_290 = x[6];
        v[5] = hi_287;
        v[6] = lo_288;
        x[5] = ((p_286 != 0) ? xi_289 : xj_290);
        x[6] = ((p_286 != 0) ? xj_290 : xi_289);
        int p_291 = 0;
        if (v[7] >= v[8]) {
            p_291 = 1;
        }
        float hi_292 = fmaxf(v[7], v[8]);
        float lo_293 = fminf(v[7], v[8]);
        unsigned int xi_294 = x[7];
        unsigned int xj_295 = x[8];
        v[7] = hi_292;
        v[8] = lo_293;
        x[7] = ((p_291 != 0) ? xi_294 : xj_295);
        x[8] = ((p_291 != 0) ? xj_295 : xi_294);
        int p_296 = 0;
        if (v[9] >= v[10]) {
            p_296 = 1;
        }
        float hi_297 = fmaxf(v[9], v[10]);
        float lo_298 = fminf(v[9], v[10]);
        unsigned int xi_299 = x[9];
        unsigned int xj_300 = x[10];
        v[9] = hi_297;
        v[10] = lo_298;
        x[9] = ((p_296 != 0) ? xi_299 : xj_300);
        x[10] = ((p_296 != 0) ? xj_300 : xi_299);
        int p_301 = 0;
        if (v[11] >= v[12]) {
            p_301 = 1;
        }
        float hi_302 = fmaxf(v[11], v[12]);
        float lo_303 = fminf(v[11], v[12]);
        unsigned int xi_304 = x[11];
        unsigned int xj_305 = x[12];
        v[11] = hi_302;
        v[12] = lo_303;
        x[11] = ((p_301 != 0) ? xi_304 : xj_305);
        x[12] = ((p_301 != 0) ? xj_305 : xi_304);
        int p_306 = 0;
        if (v[6] >= v[7]) {
            p_306 = 1;
        }
        float hi_307 = fmaxf(v[6], v[7]);
        float lo_308 = fminf(v[6], v[7]);
        unsigned int xi_309 = x[6];
        unsigned int xj_310 = x[7];
        v[6] = hi_307;
        v[7] = lo_308;
        x[6] = ((p_306 != 0) ? xi_309 : xj_310);
        x[7] = ((p_306 != 0) ? xj_310 : xi_309);
        int p_311 = 0;
        if (v[8] >= v[9]) {
            p_311 = 1;
        }
        float hi_312 = fmaxf(v[8], v[9]);
        float lo_313 = fminf(v[8], v[9]);
        unsigned int xi_314 = x[8];
        unsigned int xj_315 = x[9];
        v[8] = hi_312;
        v[9] = lo_313;
        x[8] = ((p_311 != 0) ? xi_314 : xj_315);
        x[9] = ((p_311 != 0) ? xj_315 : xi_314);
    }
    unsigned int nw0 = x[0];
    nw0 = nw0 | x[1] << 4;
    nw0 = nw0 | x[2] << 8;
    nw0 = nw0 | x[3] << 12;
    nw0 = nw0 | x[4] << 16;
    nw0 = nw0 | x[5] << 20;
    nw0 = nw0 | x[6] << 24;
    nw0 = nw0 | x[7] << 28;
    unsigned int nw1 = x[8];
    nw1 = nw1 | x[9] << 4;
    nw1 = nw1 | x[10] << 8;
    nw1 = nw1 | x[11] << 12;
    nw1 = nw1 | x[12] << 16;
    nw1 = nw1 | x[13] << 20;
    nw1 = nw1 | x[14] << 24;
    nw1 = nw1 | x[15] << 28;
    nib[tid_1 * 2] = nw0;
    nib[tid_1 * 2 + 1] = nw1;
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    float V[8];
    unsigned int X[8];
    float PV[8];
    unsigned int PX[8];
    int lv = sgx * 16;
    unsigned int nw = nib[lv * 2 + (lnx >> 3)];
    unsigned int sh = (unsigned int)((lnx & 7) << 2);
    int pos = (int)(nw >> sh & 15);
    V[0] = scr[lv * 17 + pos];
    X[0] = (unsigned int)(lv * wm_r + pos * ts_r);
    int lv_193 = sgx * 16 + 8;
    unsigned int nw_194 = nib[lv_193 * 2 + (lnrx >> 3)];
    unsigned int sh_195 = (unsigned int)((lnrx & 7) << 2);
    int pos_196 = (int)(nw_194 >> sh_195 & 15);
    PV[0] = scr[lv_193 * 17 + pos_196];
    PX[0] = (unsigned int)(lv_193 * wm_r + pos_196 * ts_r);
    int p_5 = 0;
    if (V[0] >= PV[0]) {
        p_5 = 1;
    }
    V[0] = fmaxf(V[0], PV[0]);
    X[0] = ((p_5 != 0) ? X[0] : PX[0]);
    float wv = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_197_1 = fmaxf(V[0], wv);
    float lo_198_1 = fminf(V[0], wv);
    float vn = ((upx[0] != 0) ? hi_197_1 : lo_198_1);
    X[0] = ((V[0] == vn) ? X[0] : wx);
    V[0] = vn;
    float wv_199 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_200 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_201 = fmaxf(V[0], wv_199);
    float lo_202 = fminf(V[0], wv_199);
    float vn_203 = ((upx[1] != 0) ? hi_201 : lo_202);
    X[0] = ((V[0] == vn_203) ? X[0] : wx_200);
    V[0] = vn_203;
    float wv_204 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_205 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_206 = fmaxf(V[0], wv_204);
    float lo_207 = fminf(V[0], wv_204);
    float vn_208 = ((upx[2] != 0) ? hi_206 : lo_207);
    X[0] = ((V[0] == vn_208) ? X[0] : wx_205);
    V[0] = vn_208;
    float wv_209 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_210 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_211 = fmaxf(V[0], wv_209);
    float lo_212 = fminf(V[0], wv_209);
    float vn_213 = ((upx[3] != 0) ? hi_211 : lo_212);
    X[0] = ((V[0] == vn_213) ? X[0] : wx_210);
    V[0] = vn_213;
    int lv_214 = sgx * 16 + 1;
    unsigned int nw_215 = nib[lv_214 * 2 + (lnx >> 3)];
    unsigned int sh_216 = (unsigned int)((lnx & 7) << 2);
    int pos_217 = (int)(nw_215 >> sh_216 & 15);
    V[1] = scr[lv_214 * 17 + pos_217];
    X[1] = (unsigned int)(lv_214 * wm_r + pos_217 * ts_r);
    int lv_218 = sgx * 16 + 1 + 8;
    unsigned int nw_219 = nib[lv_218 * 2 + (lnrx >> 3)];
    unsigned int sh_220 = (unsigned int)((lnrx & 7) << 2);
    int pos_221 = (int)(nw_219 >> sh_220 & 15);
    PV[1] = scr[lv_218 * 17 + pos_221];
    PX[1] = (unsigned int)(lv_218 * wm_r + pos_221 * ts_r);
    int p_222 = 0;
    if (V[1] >= PV[1]) {
        p_222 = 1;
    }
    V[1] = fmaxf(V[1], PV[1]);
    X[1] = ((p_222 != 0) ? X[1] : PX[1]);
    float wv_223 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_224 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_225 = fmaxf(V[1], wv_223);
    float lo_226 = fminf(V[1], wv_223);
    float vn_227 = ((upx[0] != 0) ? hi_225 : lo_226);
    X[1] = ((V[1] == vn_227) ? X[1] : wx_224);
    V[1] = vn_227;
    float wv_228 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_229 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_230 = fmaxf(V[1], wv_228);
    float lo_231 = fminf(V[1], wv_228);
    float vn_232 = ((upx[1] != 0) ? hi_230 : lo_231);
    X[1] = ((V[1] == vn_232) ? X[1] : wx_229);
    V[1] = vn_232;
    float wv_233 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_234 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_235 = fmaxf(V[1], wv_233);
    float lo_236 = fminf(V[1], wv_233);
    float vn_237 = ((upx[2] != 0) ? hi_235 : lo_236);
    X[1] = ((V[1] == vn_237) ? X[1] : wx_234);
    V[1] = vn_237;
    float wv_238 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_239 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_240 = fmaxf(V[1], wv_238);
    float lo_241 = fminf(V[1], wv_238);
    float vn_242 = ((upx[3] != 0) ? hi_240 : lo_241);
    X[1] = ((V[1] == vn_242) ? X[1] : wx_239);
    V[1] = vn_242;
    int lv_243 = sgx * 16 + 2;
    unsigned int nw_244 = nib[lv_243 * 2 + (lnx >> 3)];
    unsigned int sh_245 = (unsigned int)((lnx & 7) << 2);
    int pos_246 = (int)(nw_244 >> sh_245 & 15);
    V[2] = scr[lv_243 * 17 + pos_246];
    X[2] = (unsigned int)(lv_243 * wm_r + pos_246 * ts_r);
    int lv_247 = sgx * 16 + 2 + 8;
    unsigned int nw_248 = nib[lv_247 * 2 + (lnrx >> 3)];
    unsigned int sh_249 = (unsigned int)((lnrx & 7) << 2);
    int pos_250 = (int)(nw_248 >> sh_249 & 15);
    PV[2] = scr[lv_247 * 17 + pos_250];
    PX[2] = (unsigned int)(lv_247 * wm_r + pos_250 * ts_r);
    int p_251_1 = 0;
    if (V[2] >= PV[2]) {
        p_251_1 = 1;
    }
    V[2] = fmaxf(V[2], PV[2]);
    X[2] = ((p_251_1 != 0) ? X[2] : PX[2]);
    float wv_252 = __shfl_xor_sync(0xFFFFFFFF, V[2], 8);
    unsigned int wx_253 = __shfl_xor_sync(0xFFFFFFFF, X[2], 8);
    float hi_254 = fmaxf(V[2], wv_252);
    float lo_255 = fminf(V[2], wv_252);
    float vn_256 = ((upx[0] != 0) ? hi_254 : lo_255);
    X[2] = ((V[2] == vn_256) ? X[2] : wx_253);
    V[2] = vn_256;
    float wv_257 = __shfl_xor_sync(0xFFFFFFFF, V[2], 4);
    unsigned int wx_258 = __shfl_xor_sync(0xFFFFFFFF, X[2], 4);
    float hi_259 = fmaxf(V[2], wv_257);
    float lo_260 = fminf(V[2], wv_257);
    float vn_261 = ((upx[1] != 0) ? hi_259 : lo_260);
    X[2] = ((V[2] == vn_261) ? X[2] : wx_258);
    V[2] = vn_261;
    float wv_262 = __shfl_xor_sync(0xFFFFFFFF, V[2], 2);
    unsigned int wx_263 = __shfl_xor_sync(0xFFFFFFFF, X[2], 2);
    float hi_264 = fmaxf(V[2], wv_262);
    float lo_265 = fminf(V[2], wv_262);
    float vn_266 = ((upx[2] != 0) ? hi_264 : lo_265);
    X[2] = ((V[2] == vn_266) ? X[2] : wx_263);
    V[2] = vn_266;
    float wv_267 = __shfl_xor_sync(0xFFFFFFFF, V[2], 1);
    unsigned int wx_268 = __shfl_xor_sync(0xFFFFFFFF, X[2], 1);
    float hi_269 = fmaxf(V[2], wv_267);
    float lo_270 = fminf(V[2], wv_267);
    float vn_271 = ((upx[3] != 0) ? hi_269 : lo_270);
    X[2] = ((V[2] == vn_271) ? X[2] : wx_268);
    V[2] = vn_271;
    int lv_272 = sgx * 16 + 3;
    unsigned int nw_273 = nib[lv_272 * 2 + (lnx >> 3)];
    unsigned int sh_274 = (unsigned int)((lnx & 7) << 2);
    int pos_275 = (int)(nw_273 >> sh_274 & 15);
    V[3] = scr[lv_272 * 17 + pos_275];
    X[3] = (unsigned int)(lv_272 * wm_r + pos_275 * ts_r);
    int lv_276 = sgx * 16 + 3 + 8;
    unsigned int nw_277 = nib[lv_276 * 2 + (lnrx >> 3)];
    unsigned int sh_278 = (unsigned int)((lnrx & 7) << 2);
    int pos_279 = (int)(nw_277 >> sh_278 & 15);
    PV[3] = scr[lv_276 * 17 + pos_279];
    PX[3] = (unsigned int)(lv_276 * wm_r + pos_279 * ts_r);
    int p_280 = 0;
    if (V[3] >= PV[3]) {
        p_280 = 1;
    }
    V[3] = fmaxf(V[3], PV[3]);
    X[3] = ((p_280 != 0) ? X[3] : PX[3]);
    float wv_281 = __shfl_xor_sync(0xFFFFFFFF, V[3], 8);
    unsigned int wx_282 = __shfl_xor_sync(0xFFFFFFFF, X[3], 8);
    float hi_283 = fmaxf(V[3], wv_281);
    float lo_284 = fminf(V[3], wv_281);
    float vn_285 = ((upx[0] != 0) ? hi_283 : lo_284);
    X[3] = ((V[3] == vn_285) ? X[3] : wx_282);
    V[3] = vn_285;
    float wv_286 = __shfl_xor_sync(0xFFFFFFFF, V[3], 4);
    unsigned int wx_287 = __shfl_xor_sync(0xFFFFFFFF, X[3], 4);
    float hi_288 = fmaxf(V[3], wv_286);
    float lo_289 = fminf(V[3], wv_286);
    float vn_290 = ((upx[1] != 0) ? hi_288 : lo_289);
    X[3] = ((V[3] == vn_290) ? X[3] : wx_287);
    V[3] = vn_290;
    float wv_291 = __shfl_xor_sync(0xFFFFFFFF, V[3], 2);
    unsigned int wx_292 = __shfl_xor_sync(0xFFFFFFFF, X[3], 2);
    float hi_293 = fmaxf(V[3], wv_291);
    float lo_294 = fminf(V[3], wv_291);
    float vn_295 = ((upx[2] != 0) ? hi_293 : lo_294);
    X[3] = ((V[3] == vn_295) ? X[3] : wx_292);
    V[3] = vn_295;
    float wv_296 = __shfl_xor_sync(0xFFFFFFFF, V[3], 1);
    unsigned int wx_297 = __shfl_xor_sync(0xFFFFFFFF, X[3], 1);
    float hi_298 = fmaxf(V[3], wv_296);
    float lo_299 = fminf(V[3], wv_296);
    float vn_300 = ((upx[3] != 0) ? hi_298 : lo_299);
    X[3] = ((V[3] == vn_300) ? X[3] : wx_297);
    V[3] = vn_300;
    int lv_301 = sgx * 16 + 4;
    unsigned int nw_302 = nib[lv_301 * 2 + (lnx >> 3)];
    unsigned int sh_303 = (unsigned int)((lnx & 7) << 2);
    int pos_304 = (int)(nw_302 >> sh_303 & 15);
    V[4] = scr[lv_301 * 17 + pos_304];
    X[4] = (unsigned int)(lv_301 * wm_r + pos_304 * ts_r);
    int lv_305 = sgx * 16 + 4 + 8;
    unsigned int nw_306 = nib[lv_305 * 2 + (lnrx >> 3)];
    unsigned int sh_307 = (unsigned int)((lnrx & 7) << 2);
    int pos_308 = (int)(nw_306 >> sh_307 & 15);
    PV[4] = scr[lv_305 * 17 + pos_308];
    PX[4] = (unsigned int)(lv_305 * wm_r + pos_308 * ts_r);
    int p_309 = 0;
    if (V[4] >= PV[4]) {
        p_309 = 1;
    }
    V[4] = fmaxf(V[4], PV[4]);
    X[4] = ((p_309 != 0) ? X[4] : PX[4]);
    float wv_310 = __shfl_xor_sync(0xFFFFFFFF, V[4], 8);
    unsigned int wx_311 = __shfl_xor_sync(0xFFFFFFFF, X[4], 8);
    float hi_312_1 = fmaxf(V[4], wv_310);
    float lo_313_1 = fminf(V[4], wv_310);
    float vn_314 = ((upx[0] != 0) ? hi_312_1 : lo_313_1);
    X[4] = ((V[4] == vn_314) ? X[4] : wx_311);
    V[4] = vn_314;
    float wv_315 = __shfl_xor_sync(0xFFFFFFFF, V[4], 4);
    unsigned int wx_316 = __shfl_xor_sync(0xFFFFFFFF, X[4], 4);
    float hi_317 = fmaxf(V[4], wv_315);
    float lo_318 = fminf(V[4], wv_315);
    float vn_319 = ((upx[1] != 0) ? hi_317 : lo_318);
    X[4] = ((V[4] == vn_319) ? X[4] : wx_316);
    V[4] = vn_319;
    float wv_320 = __shfl_xor_sync(0xFFFFFFFF, V[4], 2);
    unsigned int wx_321 = __shfl_xor_sync(0xFFFFFFFF, X[4], 2);
    float hi_322 = fmaxf(V[4], wv_320);
    float lo_323 = fminf(V[4], wv_320);
    float vn_324 = ((upx[2] != 0) ? hi_322 : lo_323);
    X[4] = ((V[4] == vn_324) ? X[4] : wx_321);
    V[4] = vn_324;
    float wv_325 = __shfl_xor_sync(0xFFFFFFFF, V[4], 1);
    unsigned int wx_326 = __shfl_xor_sync(0xFFFFFFFF, X[4], 1);
    float hi_327 = fmaxf(V[4], wv_325);
    float lo_328 = fminf(V[4], wv_325);
    float vn_329 = ((upx[3] != 0) ? hi_327 : lo_328);
    X[4] = ((V[4] == vn_329) ? X[4] : wx_326);
    V[4] = vn_329;
    int lv_330 = sgx * 16 + 5;
    unsigned int nw_331 = nib[lv_330 * 2 + (lnx >> 3)];
    unsigned int sh_332 = (unsigned int)((lnx & 7) << 2);
    int pos_333 = (int)(nw_331 >> sh_332 & 15);
    V[5] = scr[lv_330 * 17 + pos_333];
    X[5] = (unsigned int)(lv_330 * wm_r + pos_333 * ts_r);
    int lv_334 = sgx * 16 + 5 + 8;
    unsigned int nw_335 = nib[lv_334 * 2 + (lnrx >> 3)];
    unsigned int sh_336 = (unsigned int)((lnrx & 7) << 2);
    int pos_337 = (int)(nw_335 >> sh_336 & 15);
    PV[5] = scr[lv_334 * 17 + pos_337];
    PX[5] = (unsigned int)(lv_334 * wm_r + pos_337 * ts_r);
    int p_338 = 0;
    if (V[5] >= PV[5]) {
        p_338 = 1;
    }
    V[5] = fmaxf(V[5], PV[5]);
    X[5] = ((p_338 != 0) ? X[5] : PX[5]);
    float wv_339 = __shfl_xor_sync(0xFFFFFFFF, V[5], 8);
    unsigned int wx_340 = __shfl_xor_sync(0xFFFFFFFF, X[5], 8);
    float hi_341 = fmaxf(V[5], wv_339);
    float lo_342 = fminf(V[5], wv_339);
    float vn_343 = ((upx[0] != 0) ? hi_341 : lo_342);
    X[5] = ((V[5] == vn_343) ? X[5] : wx_340);
    V[5] = vn_343;
    float wv_344 = __shfl_xor_sync(0xFFFFFFFF, V[5], 4);
    unsigned int wx_345 = __shfl_xor_sync(0xFFFFFFFF, X[5], 4);
    float hi_346 = fmaxf(V[5], wv_344);
    float lo_347 = fminf(V[5], wv_344);
    float vn_348 = ((upx[1] != 0) ? hi_346 : lo_347);
    X[5] = ((V[5] == vn_348) ? X[5] : wx_345);
    V[5] = vn_348;
    float wv_349 = __shfl_xor_sync(0xFFFFFFFF, V[5], 2);
    unsigned int wx_350 = __shfl_xor_sync(0xFFFFFFFF, X[5], 2);
    float hi_351 = fmaxf(V[5], wv_349);
    float lo_352 = fminf(V[5], wv_349);
    float vn_353 = ((upx[2] != 0) ? hi_351 : lo_352);
    X[5] = ((V[5] == vn_353) ? X[5] : wx_350);
    V[5] = vn_353;
    float wv_354 = __shfl_xor_sync(0xFFFFFFFF, V[5], 1);
    unsigned int wx_355 = __shfl_xor_sync(0xFFFFFFFF, X[5], 1);
    float hi_356 = fmaxf(V[5], wv_354);
    float lo_357 = fminf(V[5], wv_354);
    float vn_358 = ((upx[3] != 0) ? hi_356 : lo_357);
    X[5] = ((V[5] == vn_358) ? X[5] : wx_355);
    V[5] = vn_358;
    int lv_359 = sgx * 16 + 6;
    unsigned int nw_360 = nib[lv_359 * 2 + (lnx >> 3)];
    unsigned int sh_361 = (unsigned int)((lnx & 7) << 2);
    int pos_362 = (int)(nw_360 >> sh_361 & 15);
    V[6] = scr[lv_359 * 17 + pos_362];
    X[6] = (unsigned int)(lv_359 * wm_r + pos_362 * ts_r);
    int lv_363 = sgx * 16 + 6 + 8;
    unsigned int nw_364 = nib[lv_363 * 2 + (lnrx >> 3)];
    unsigned int sh_365 = (unsigned int)((lnrx & 7) << 2);
    int pos_366 = (int)(nw_364 >> sh_365 & 15);
    PV[6] = scr[lv_363 * 17 + pos_366];
    PX[6] = (unsigned int)(lv_363 * wm_r + pos_366 * ts_r);
    int p_367 = 0;
    if (V[6] >= PV[6]) {
        p_367 = 1;
    }
    V[6] = fmaxf(V[6], PV[6]);
    X[6] = ((p_367 != 0) ? X[6] : PX[6]);
    float wv_368 = __shfl_xor_sync(0xFFFFFFFF, V[6], 8);
    unsigned int wx_369 = __shfl_xor_sync(0xFFFFFFFF, X[6], 8);
    float hi_370 = fmaxf(V[6], wv_368);
    float lo_371 = fminf(V[6], wv_368);
    float vn_372 = ((upx[0] != 0) ? hi_370 : lo_371);
    X[6] = ((V[6] == vn_372) ? X[6] : wx_369);
    V[6] = vn_372;
    float wv_373 = __shfl_xor_sync(0xFFFFFFFF, V[6], 4);
    unsigned int wx_374 = __shfl_xor_sync(0xFFFFFFFF, X[6], 4);
    float hi_375 = fmaxf(V[6], wv_373);
    float lo_376 = fminf(V[6], wv_373);
    float vn_377 = ((upx[1] != 0) ? hi_375 : lo_376);
    X[6] = ((V[6] == vn_377) ? X[6] : wx_374);
    V[6] = vn_377;
    float wv_378 = __shfl_xor_sync(0xFFFFFFFF, V[6], 2);
    unsigned int wx_379 = __shfl_xor_sync(0xFFFFFFFF, X[6], 2);
    float hi_380 = fmaxf(V[6], wv_378);
    float lo_381 = fminf(V[6], wv_378);
    float vn_382 = ((upx[2] != 0) ? hi_380 : lo_381);
    X[6] = ((V[6] == vn_382) ? X[6] : wx_379);
    V[6] = vn_382;
    float wv_383 = __shfl_xor_sync(0xFFFFFFFF, V[6], 1);
    unsigned int wx_384 = __shfl_xor_sync(0xFFFFFFFF, X[6], 1);
    float hi_385 = fmaxf(V[6], wv_383);
    float lo_386 = fminf(V[6], wv_383);
    float vn_387 = ((upx[3] != 0) ? hi_385 : lo_386);
    X[6] = ((V[6] == vn_387) ? X[6] : wx_384);
    V[6] = vn_387;
    int lv_388 = sgx * 16 + 7;
    unsigned int nw_389 = nib[lv_388 * 2 + (lnx >> 3)];
    unsigned int sh_390 = (unsigned int)((lnx & 7) << 2);
    int pos_391 = (int)(nw_389 >> sh_390 & 15);
    V[7] = scr[lv_388 * 17 + pos_391];
    X[7] = (unsigned int)(lv_388 * wm_r + pos_391 * ts_r);
    int lv_392 = sgx * 16 + 7 + 8;
    unsigned int nw_393 = nib[lv_392 * 2 + (lnrx >> 3)];
    unsigned int sh_394 = (unsigned int)((lnrx & 7) << 2);
    int pos_395 = (int)(nw_393 >> sh_394 & 15);
    PV[7] = scr[lv_392 * 17 + pos_395];
    PX[7] = (unsigned int)(lv_392 * wm_r + pos_395 * ts_r);
    int p_396 = 0;
    if (V[7] >= PV[7]) {
        p_396 = 1;
    }
    V[7] = fmaxf(V[7], PV[7]);
    X[7] = ((p_396 != 0) ? X[7] : PX[7]);
    float wv_397 = __shfl_xor_sync(0xFFFFFFFF, V[7], 8);
    unsigned int wx_398 = __shfl_xor_sync(0xFFFFFFFF, X[7], 8);
    float hi_399 = fmaxf(V[7], wv_397);
    float lo_400 = fminf(V[7], wv_397);
    float vn_401 = ((upx[0] != 0) ? hi_399 : lo_400);
    X[7] = ((V[7] == vn_401) ? X[7] : wx_398);
    V[7] = vn_401;
    float wv_402 = __shfl_xor_sync(0xFFFFFFFF, V[7], 4);
    unsigned int wx_403 = __shfl_xor_sync(0xFFFFFFFF, X[7], 4);
    float hi_404 = fmaxf(V[7], wv_402);
    float lo_405 = fminf(V[7], wv_402);
    float vn_406 = ((upx[1] != 0) ? hi_404 : lo_405);
    X[7] = ((V[7] == vn_406) ? X[7] : wx_403);
    V[7] = vn_406;
    float wv_407 = __shfl_xor_sync(0xFFFFFFFF, V[7], 2);
    unsigned int wx_408 = __shfl_xor_sync(0xFFFFFFFF, X[7], 2);
    float hi_409 = fmaxf(V[7], wv_407);
    float lo_410 = fminf(V[7], wv_407);
    float vn_411 = ((upx[2] != 0) ? hi_409 : lo_410);
    X[7] = ((V[7] == vn_411) ? X[7] : wx_408);
    V[7] = vn_411;
    float wv_412 = __shfl_xor_sync(0xFFFFFFFF, V[7], 1);
    unsigned int wx_413 = __shfl_xor_sync(0xFFFFFFFF, X[7], 1);
    float hi_414 = fmaxf(V[7], wv_412);
    float lo_415 = fminf(V[7], wv_412);
    float vn_416 = ((upx[3] != 0) ? hi_414 : lo_415);
    X[7] = ((V[7] == vn_416) ? X[7] : wx_413);
    V[7] = vn_416;
    float wv1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    unsigned int wx1 = __shfl_xor_sync(0xFFFFFFFF, X[4], 15);
    int p_417 = 0;
    if (V[0] >= wv1) {
        p_417 = 1;
    }
    V[0] = fmaxf(V[0], wv1);
    X[0] = ((p_417 != 0) ? X[0] : wx1);
    float wv_418 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_419 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_420 = fmaxf(V[0], wv_418);
    float lo_421 = fminf(V[0], wv_418);
    float vn_422 = ((upx[0] != 0) ? hi_420 : lo_421);
    X[0] = ((V[0] == vn_422) ? X[0] : wx_419);
    V[0] = vn_422;
    float wv_423 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_424 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_425 = fmaxf(V[0], wv_423);
    float lo_426 = fminf(V[0], wv_423);
    float vn_427 = ((upx[1] != 0) ? hi_425 : lo_426);
    X[0] = ((V[0] == vn_427) ? X[0] : wx_424);
    V[0] = vn_427;
    float wv_428 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_429 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_430 = fmaxf(V[0], wv_428);
    float lo_431 = fminf(V[0], wv_428);
    float vn_432 = ((upx[2] != 0) ? hi_430 : lo_431);
    X[0] = ((V[0] == vn_432) ? X[0] : wx_429);
    V[0] = vn_432;
    float wv_433 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_434 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_435 = fmaxf(V[0], wv_433);
    float lo_436 = fminf(V[0], wv_433);
    float vn_437 = ((upx[3] != 0) ? hi_435 : lo_436);
    X[0] = ((V[0] == vn_437) ? X[0] : wx_434);
    V[0] = vn_437;
    float wv1_438 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    unsigned int wx1_439 = __shfl_xor_sync(0xFFFFFFFF, X[5], 15);
    int p_440 = 0;
    if (V[1] >= wv1_438) {
        p_440 = 1;
    }
    V[1] = fmaxf(V[1], wv1_438);
    X[1] = ((p_440 != 0) ? X[1] : wx1_439);
    float wv_441 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_442 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_443 = fmaxf(V[1], wv_441);
    float lo_444 = fminf(V[1], wv_441);
    float vn_445 = ((upx[0] != 0) ? hi_443 : lo_444);
    X[1] = ((V[1] == vn_445) ? X[1] : wx_442);
    V[1] = vn_445;
    float wv_446 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_447 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_448 = fmaxf(V[1], wv_446);
    float lo_449 = fminf(V[1], wv_446);
    float vn_450 = ((upx[1] != 0) ? hi_448 : lo_449);
    X[1] = ((V[1] == vn_450) ? X[1] : wx_447);
    V[1] = vn_450;
    float wv_451 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_452 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_453 = fmaxf(V[1], wv_451);
    float lo_454 = fminf(V[1], wv_451);
    float vn_455 = ((upx[2] != 0) ? hi_453 : lo_454);
    X[1] = ((V[1] == vn_455) ? X[1] : wx_452);
    V[1] = vn_455;
    float wv_456 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_457 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_458 = fmaxf(V[1], wv_456);
    float lo_459 = fminf(V[1], wv_456);
    float vn_460 = ((upx[3] != 0) ? hi_458 : lo_459);
    X[1] = ((V[1] == vn_460) ? X[1] : wx_457);
    V[1] = vn_460;
    float wv1_461 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    unsigned int wx1_462 = __shfl_xor_sync(0xFFFFFFFF, X[6], 15);
    int p_463 = 0;
    if (V[2] >= wv1_461) {
        p_463 = 1;
    }
    V[2] = fmaxf(V[2], wv1_461);
    X[2] = ((p_463 != 0) ? X[2] : wx1_462);
    float wv_464 = __shfl_xor_sync(0xFFFFFFFF, V[2], 8);
    unsigned int wx_465 = __shfl_xor_sync(0xFFFFFFFF, X[2], 8);
    float hi_466 = fmaxf(V[2], wv_464);
    float lo_467 = fminf(V[2], wv_464);
    float vn_468 = ((upx[0] != 0) ? hi_466 : lo_467);
    X[2] = ((V[2] == vn_468) ? X[2] : wx_465);
    V[2] = vn_468;
    float wv_469 = __shfl_xor_sync(0xFFFFFFFF, V[2], 4);
    unsigned int wx_470 = __shfl_xor_sync(0xFFFFFFFF, X[2], 4);
    float hi_471 = fmaxf(V[2], wv_469);
    float lo_472 = fminf(V[2], wv_469);
    float vn_473 = ((upx[1] != 0) ? hi_471 : lo_472);
    X[2] = ((V[2] == vn_473) ? X[2] : wx_470);
    V[2] = vn_473;
    float wv_474 = __shfl_xor_sync(0xFFFFFFFF, V[2], 2);
    unsigned int wx_475 = __shfl_xor_sync(0xFFFFFFFF, X[2], 2);
    float hi_476 = fmaxf(V[2], wv_474);
    float lo_477 = fminf(V[2], wv_474);
    float vn_478 = ((upx[2] != 0) ? hi_476 : lo_477);
    X[2] = ((V[2] == vn_478) ? X[2] : wx_475);
    V[2] = vn_478;
    float wv_479 = __shfl_xor_sync(0xFFFFFFFF, V[2], 1);
    unsigned int wx_480 = __shfl_xor_sync(0xFFFFFFFF, X[2], 1);
    float hi_481 = fmaxf(V[2], wv_479);
    float lo_482 = fminf(V[2], wv_479);
    float vn_483 = ((upx[3] != 0) ? hi_481 : lo_482);
    X[2] = ((V[2] == vn_483) ? X[2] : wx_480);
    V[2] = vn_483;
    float wv1_484 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    unsigned int wx1_485 = __shfl_xor_sync(0xFFFFFFFF, X[7], 15);
    int p_486 = 0;
    if (V[3] >= wv1_484) {
        p_486 = 1;
    }
    V[3] = fmaxf(V[3], wv1_484);
    X[3] = ((p_486 != 0) ? X[3] : wx1_485);
    float wv_487 = __shfl_xor_sync(0xFFFFFFFF, V[3], 8);
    unsigned int wx_488 = __shfl_xor_sync(0xFFFFFFFF, X[3], 8);
    float hi_489 = fmaxf(V[3], wv_487);
    float lo_490 = fminf(V[3], wv_487);
    float vn_491 = ((upx[0] != 0) ? hi_489 : lo_490);
    X[3] = ((V[3] == vn_491) ? X[3] : wx_488);
    V[3] = vn_491;
    float wv_492 = __shfl_xor_sync(0xFFFFFFFF, V[3], 4);
    unsigned int wx_493 = __shfl_xor_sync(0xFFFFFFFF, X[3], 4);
    float hi_494 = fmaxf(V[3], wv_492);
    float lo_495 = fminf(V[3], wv_492);
    float vn_496 = ((upx[1] != 0) ? hi_494 : lo_495);
    X[3] = ((V[3] == vn_496) ? X[3] : wx_493);
    V[3] = vn_496;
    float wv_497 = __shfl_xor_sync(0xFFFFFFFF, V[3], 2);
    unsigned int wx_498 = __shfl_xor_sync(0xFFFFFFFF, X[3], 2);
    float hi_499 = fmaxf(V[3], wv_497);
    float lo_500 = fminf(V[3], wv_497);
    float vn_501 = ((upx[2] != 0) ? hi_499 : lo_500);
    X[3] = ((V[3] == vn_501) ? X[3] : wx_498);
    V[3] = vn_501;
    float wv_502 = __shfl_xor_sync(0xFFFFFFFF, V[3], 1);
    unsigned int wx_503 = __shfl_xor_sync(0xFFFFFFFF, X[3], 1);
    float hi_504 = fmaxf(V[3], wv_502);
    float lo_505 = fminf(V[3], wv_502);
    float vn_506 = ((upx[3] != 0) ? hi_504 : lo_505);
    X[3] = ((V[3] == vn_506) ? X[3] : wx_503);
    V[3] = vn_506;
    float wv1_507 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    unsigned int wx1_508 = __shfl_xor_sync(0xFFFFFFFF, X[2], 15);
    int p_509 = 0;
    if (V[0] >= wv1_507) {
        p_509 = 1;
    }
    V[0] = fmaxf(V[0], wv1_507);
    X[0] = ((p_509 != 0) ? X[0] : wx1_508);
    float wv_510 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_511 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_512 = fmaxf(V[0], wv_510);
    float lo_513 = fminf(V[0], wv_510);
    float vn_514 = ((upx[0] != 0) ? hi_512 : lo_513);
    X[0] = ((V[0] == vn_514) ? X[0] : wx_511);
    V[0] = vn_514;
    float wv_515 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_516 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_517 = fmaxf(V[0], wv_515);
    float lo_518 = fminf(V[0], wv_515);
    float vn_519 = ((upx[1] != 0) ? hi_517 : lo_518);
    X[0] = ((V[0] == vn_519) ? X[0] : wx_516);
    V[0] = vn_519;
    float wv_520 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_521 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_522 = fmaxf(V[0], wv_520);
    float lo_523 = fminf(V[0], wv_520);
    float vn_524 = ((upx[2] != 0) ? hi_522 : lo_523);
    X[0] = ((V[0] == vn_524) ? X[0] : wx_521);
    V[0] = vn_524;
    float wv_525 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_526 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_527 = fmaxf(V[0], wv_525);
    float lo_528 = fminf(V[0], wv_525);
    float vn_529 = ((upx[3] != 0) ? hi_527 : lo_528);
    X[0] = ((V[0] == vn_529) ? X[0] : wx_526);
    V[0] = vn_529;
    float wv1_530 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    unsigned int wx1_531 = __shfl_xor_sync(0xFFFFFFFF, X[3], 15);
    int p_532 = 0;
    if (V[1] >= wv1_530) {
        p_532 = 1;
    }
    V[1] = fmaxf(V[1], wv1_530);
    X[1] = ((p_532 != 0) ? X[1] : wx1_531);
    float wv_533 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_534 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_535 = fmaxf(V[1], wv_533);
    float lo_536 = fminf(V[1], wv_533);
    float vn_537 = ((upx[0] != 0) ? hi_535 : lo_536);
    X[1] = ((V[1] == vn_537) ? X[1] : wx_534);
    V[1] = vn_537;
    float wv_538 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_539 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_540 = fmaxf(V[1], wv_538);
    float lo_541 = fminf(V[1], wv_538);
    float vn_542 = ((upx[1] != 0) ? hi_540 : lo_541);
    X[1] = ((V[1] == vn_542) ? X[1] : wx_539);
    V[1] = vn_542;
    float wv_543 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_544 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_545 = fmaxf(V[1], wv_543);
    float lo_546 = fminf(V[1], wv_543);
    float vn_547 = ((upx[2] != 0) ? hi_545 : lo_546);
    X[1] = ((V[1] == vn_547) ? X[1] : wx_544);
    V[1] = vn_547;
    float wv_548 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_549 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_550 = fmaxf(V[1], wv_548);
    float lo_551 = fminf(V[1], wv_548);
    float vn_552 = ((upx[3] != 0) ? hi_550 : lo_551);
    X[1] = ((V[1] == vn_552) ? X[1] : wx_549);
    V[1] = vn_552;
    float wvl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    unsigned int wxl = __shfl_xor_sync(0xFFFFFFFF, X[1], 15);
    int p_553 = 0;
    if (V[0] >= wvl) {
        p_553 = 1;
    }
    V[0] = fmaxf(V[0], wvl);
    X[0] = ((p_553 != 0) ? X[0] : wxl);
    float wv_554 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_555 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_556 = fmaxf(V[0], wv_554);
    float lo_557 = fminf(V[0], wv_554);
    float vn_558 = ((upx[0] != 0) ? hi_556 : lo_557);
    X[0] = ((V[0] == vn_558) ? X[0] : wx_555);
    V[0] = vn_558;
    float wv_559 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_560 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_561 = fmaxf(V[0], wv_559);
    float lo_562 = fminf(V[0], wv_559);
    float vn_563 = ((upx[1] != 0) ? hi_561 : lo_562);
    X[0] = ((V[0] == vn_563) ? X[0] : wx_560);
    V[0] = vn_563;
    float wv_564 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_565 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_566 = fmaxf(V[0], wv_564);
    float lo_567 = fminf(V[0], wv_564);
    float vn_568 = ((upx[2] != 0) ? hi_566 : lo_567);
    X[0] = ((V[0] == vn_568) ? X[0] : wx_565);
    V[0] = vn_568;
    float wv_569 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_570 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_571 = fmaxf(V[0], wv_569);
    float lo_572 = fminf(V[0], wv_569);
    float vn_573 = ((upx[3] != 0) ? hi_571 : lo_572);
    X[0] = ((V[0] == vn_573) ? X[0] : wx_570);
    V[0] = vn_573;
    float KV = V[0];
    unsigned int KX = X[0];
    int qb = (sgx + cgx) * 32;
    q2[qb + lnx] = KV;
    q2[qb + 16 + lnx] = __uint_as_float(KX);
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    if (tid_1 < 32) {
        float V2[4];
        unsigned int X2[4];
        V2[0] = q2[cgx * 32 + lnx];
        X2[0] = __as_u32(q2[cgx * 32 + 16 + lnx]);
        float bv2 = q2[(4 + cgx) * 32 + lnrx];
        unsigned int bx2 = __as_u32(q2[(4 + cgx) * 32 + 16 + lnrx]);
        int p_0 = 0;
        if (V2[0] >= bv2) {
            p_0 = 1;
        }
        V2[0] = fmaxf(V2[0], bv2);
        X2[0] = ((p_0 != 0) ? X2[0] : bx2);
        float wv_1 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 8);
        unsigned int wx_2 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 8);
        float hi_3_4 = fmaxf(V2[0], wv_1);
        float lo_4_4 = fminf(V2[0], wv_1);
        float vn_5 = ((upx[0] != 0) ? hi_3_4 : lo_4_4);
        X2[0] = ((V2[0] == vn_5) ? X2[0] : wx_2);
        V2[0] = vn_5;
        float wv_6 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 4);
        unsigned int wx_7 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 4);
        float hi_8_4 = fmaxf(V2[0], wv_6);
        float lo_9_4 = fminf(V2[0], wv_6);
        float vn_10 = ((upx[1] != 0) ? hi_8_4 : lo_9_4);
        X2[0] = ((V2[0] == vn_10) ? X2[0] : wx_7);
        V2[0] = vn_10;
        float wv_11 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 2);
        unsigned int wx_12 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 2);
        float hi_13_4 = fmaxf(V2[0], wv_11);
        float lo_14_4 = fminf(V2[0], wv_11);
        float vn_15 = ((upx[2] != 0) ? hi_13_4 : lo_14_4);
        X2[0] = ((V2[0] == vn_15) ? X2[0] : wx_12);
        V2[0] = vn_15;
        float wv_16 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 1);
        unsigned int wx_17 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 1);
        float hi_18_4 = fmaxf(V2[0], wv_16);
        float lo_19_4 = fminf(V2[0], wv_16);
        float vn_20 = ((upx[3] != 0) ? hi_18_4 : lo_19_4);
        X2[0] = ((V2[0] == vn_20) ? X2[0] : wx_17);
        V2[0] = vn_20;
        V2[1] = q2[(1 + cgx) * 32 + lnx];
        X2[1] = __as_u32(q2[(1 + cgx) * 32 + 16 + lnx]);
        float bv2_21 = q2[(5 + cgx) * 32 + lnrx];
        unsigned int bx2_22 = __as_u32(q2[(5 + cgx) * 32 + 16 + lnrx]);
        int p_23 = 0;
        if (V2[1] >= bv2_21) {
            p_23 = 1;
        }
        V2[1] = fmaxf(V2[1], bv2_21);
        X2[1] = ((p_23 != 0) ? X2[1] : bx2_22);
        float wv_24 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 8);
        unsigned int wx_25 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 8);
        float hi_26 = fmaxf(V2[1], wv_24);
        float lo_27 = fminf(V2[1], wv_24);
        float vn_28 = ((upx[0] != 0) ? hi_26 : lo_27);
        X2[1] = ((V2[1] == vn_28) ? X2[1] : wx_25);
        V2[1] = vn_28;
        float wv_29 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 4);
        unsigned int wx_30 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 4);
        float hi_31 = fmaxf(V2[1], wv_29);
        float lo_32 = fminf(V2[1], wv_29);
        float vn_33 = ((upx[1] != 0) ? hi_31 : lo_32);
        X2[1] = ((V2[1] == vn_33) ? X2[1] : wx_30);
        V2[1] = vn_33;
        float wv_34 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 2);
        unsigned int wx_35 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 2);
        float hi_36 = fmaxf(V2[1], wv_34);
        float lo_37 = fminf(V2[1], wv_34);
        float vn_38 = ((upx[2] != 0) ? hi_36 : lo_37);
        X2[1] = ((V2[1] == vn_38) ? X2[1] : wx_35);
        V2[1] = vn_38;
        float wv_39 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 1);
        unsigned int wx_40 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 1);
        float hi_41 = fmaxf(V2[1], wv_39);
        float lo_42 = fminf(V2[1], wv_39);
        float vn_43 = ((upx[3] != 0) ? hi_41 : lo_42);
        X2[1] = ((V2[1] == vn_43) ? X2[1] : wx_40);
        V2[1] = vn_43;
        V2[2] = q2[(2 + cgx) * 32 + lnx];
        X2[2] = __as_u32(q2[(2 + cgx) * 32 + 16 + lnx]);
        float bv2_44 = q2[(6 + cgx) * 32 + lnrx];
        unsigned int bx2_45 = __as_u32(q2[(6 + cgx) * 32 + 16 + lnrx]);
        int p_46 = 0;
        if (V2[2] >= bv2_44) {
            p_46 = 1;
        }
        V2[2] = fmaxf(V2[2], bv2_44);
        X2[2] = ((p_46 != 0) ? X2[2] : bx2_45);
        float wv_47 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 8);
        unsigned int wx_48 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 8);
        float hi_49 = fmaxf(V2[2], wv_47);
        float lo_50 = fminf(V2[2], wv_47);
        float vn_51 = ((upx[0] != 0) ? hi_49 : lo_50);
        X2[2] = ((V2[2] == vn_51) ? X2[2] : wx_48);
        V2[2] = vn_51;
        float wv_52 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 4);
        unsigned int wx_53 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 4);
        float hi_54 = fmaxf(V2[2], wv_52);
        float lo_55 = fminf(V2[2], wv_52);
        float vn_56 = ((upx[1] != 0) ? hi_54 : lo_55);
        X2[2] = ((V2[2] == vn_56) ? X2[2] : wx_53);
        V2[2] = vn_56;
        float wv_57 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 2);
        unsigned int wx_58 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 2);
        float hi_59 = fmaxf(V2[2], wv_57);
        float lo_60 = fminf(V2[2], wv_57);
        float vn_61 = ((upx[2] != 0) ? hi_59 : lo_60);
        X2[2] = ((V2[2] == vn_61) ? X2[2] : wx_58);
        V2[2] = vn_61;
        float wv_62 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 1);
        unsigned int wx_63 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 1);
        float hi_64 = fmaxf(V2[2], wv_62);
        float lo_65 = fminf(V2[2], wv_62);
        float vn_66 = ((upx[3] != 0) ? hi_64 : lo_65);
        X2[2] = ((V2[2] == vn_66) ? X2[2] : wx_63);
        V2[2] = vn_66;
        V2[3] = q2[(3 + cgx) * 32 + lnx];
        X2[3] = __as_u32(q2[(3 + cgx) * 32 + 16 + lnx]);
        float bv2_67 = q2[(7 + cgx) * 32 + lnrx];
        unsigned int bx2_68 = __as_u32(q2[(7 + cgx) * 32 + 16 + lnrx]);
        int p_69 = 0;
        if (V2[3] >= bv2_67) {
            p_69 = 1;
        }
        V2[3] = fmaxf(V2[3], bv2_67);
        X2[3] = ((p_69 != 0) ? X2[3] : bx2_68);
        float wv_70 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 8);
        unsigned int wx_71 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 8);
        float hi_72 = fmaxf(V2[3], wv_70);
        float lo_73 = fminf(V2[3], wv_70);
        float vn_74 = ((upx[0] != 0) ? hi_72 : lo_73);
        X2[3] = ((V2[3] == vn_74) ? X2[3] : wx_71);
        V2[3] = vn_74;
        float wv_75 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 4);
        unsigned int wx_76 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 4);
        float hi_78_1 = fmaxf(V2[3], wv_75);
        float lo_79_1 = fminf(V2[3], wv_75);
        float vn_80 = ((upx[1] != 0) ? hi_78_1 : lo_79_1);
        X2[3] = ((V2[3] == vn_80) ? X2[3] : wx_76);
        V2[3] = vn_80;
        float wv_81 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 2);
        unsigned int wx_82 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 2);
        float hi_84_1 = fmaxf(V2[3], wv_81);
        float lo_85_1 = fminf(V2[3], wv_81);
        float vn_86 = ((upx[2] != 0) ? hi_84_1 : lo_85_1);
        X2[3] = ((V2[3] == vn_86) ? X2[3] : wx_82);
        V2[3] = vn_86;
        float wv_87 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 1);
        unsigned int wx_88 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 1);
        float hi_90_1 = fmaxf(V2[3], wv_87);
        float lo_91_1 = fminf(V2[3], wv_87);
        float vn_92 = ((upx[3] != 0) ? hi_90_1 : lo_91_1);
        X2[3] = ((V2[3] == vn_92) ? X2[3] : wx_88);
        V2[3] = vn_92;
        float wv3 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 15);
        unsigned int wx3 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 15);
        int p_93 = 0;
        if (V2[0] >= wv3) {
            p_93 = 1;
        }
        V2[0] = fmaxf(V2[0], wv3);
        X2[0] = ((p_93 != 0) ? X2[0] : wx3);
        float wv_94 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 8);
        unsigned int wx_95 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 8);
        float hi_96_1 = fmaxf(V2[0], wv_94);
        float lo_97_1 = fminf(V2[0], wv_94);
        float vn_98 = ((upx[0] != 0) ? hi_96_1 : lo_97_1);
        X2[0] = ((V2[0] == vn_98) ? X2[0] : wx_95);
        V2[0] = vn_98;
        float wv_99 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 4);
        unsigned int wx_100 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 4);
        float hi_102_1 = fmaxf(V2[0], wv_99);
        float lo_103_1 = fminf(V2[0], wv_99);
        float vn_104 = ((upx[1] != 0) ? hi_102_1 : lo_103_1);
        X2[0] = ((V2[0] == vn_104) ? X2[0] : wx_100);
        V2[0] = vn_104;
        float wv_105 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 2);
        unsigned int wx_106 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 2);
        float hi_108_1 = fmaxf(V2[0], wv_105);
        float lo_109_1 = fminf(V2[0], wv_105);
        float vn_110 = ((upx[2] != 0) ? hi_108_1 : lo_109_1);
        X2[0] = ((V2[0] == vn_110) ? X2[0] : wx_106);
        V2[0] = vn_110;
        float wv_111 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 1);
        unsigned int wx_112 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 1);
        float hi_114_1 = fmaxf(V2[0], wv_111);
        float lo_115_1 = fminf(V2[0], wv_111);
        float vn_116 = ((upx[3] != 0) ? hi_114_1 : lo_115_1);
        X2[0] = ((V2[0] == vn_116) ? X2[0] : wx_112);
        V2[0] = vn_116;
        float wv3_117 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 15);
        unsigned int wx3_118 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 15);
        int p_119 = 0;
        if (V2[1] >= wv3_117) {
            p_119 = 1;
        }
        V2[1] = fmaxf(V2[1], wv3_117);
        X2[1] = ((p_119 != 0) ? X2[1] : wx3_118);
        float wv_120 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 8);
        unsigned int wx_121 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 8);
        float hi_122 = fmaxf(V2[1], wv_120);
        float lo_123 = fminf(V2[1], wv_120);
        float vn_124 = ((upx[0] != 0) ? hi_122 : lo_123);
        X2[1] = ((V2[1] == vn_124) ? X2[1] : wx_121);
        V2[1] = vn_124;
        float wv_125 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 4);
        unsigned int wx_126 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 4);
        float hi_128 = fmaxf(V2[1], wv_125);
        float lo_129 = fminf(V2[1], wv_125);
        float vn_130 = ((upx[1] != 0) ? hi_128 : lo_129);
        X2[1] = ((V2[1] == vn_130) ? X2[1] : wx_126);
        V2[1] = vn_130;
        float wv_131 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 2);
        unsigned int wx_132 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 2);
        float hi_134 = fmaxf(V2[1], wv_131);
        float lo_135 = fminf(V2[1], wv_131);
        float vn_136 = ((upx[2] != 0) ? hi_134 : lo_135);
        X2[1] = ((V2[1] == vn_136) ? X2[1] : wx_132);
        V2[1] = vn_136;
        float wv_137 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 1);
        unsigned int wx_138 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 1);
        float hi_140 = fmaxf(V2[1], wv_137);
        float lo_141 = fminf(V2[1], wv_137);
        float vn_142 = ((upx[3] != 0) ? hi_140 : lo_141);
        X2[1] = ((V2[1] == vn_142) ? X2[1] : wx_138);
        V2[1] = vn_142;
        float wvl2 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 15);
        unsigned int wxl2 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 15);
        int p_143 = 0;
        if (V2[0] >= wvl2) {
            p_143 = 1;
        }
        V2[0] = fmaxf(V2[0], wvl2);
        X2[0] = ((p_143 != 0) ? X2[0] : wxl2);
        KV = V2[0];
        KX = X2[0];
        int ucol = blockIdx.x + cgx;
        int commit = 0;
        if (gx < 1 && ucol < total_q) {
            commit = 1;
        }
        unsigned int kk = __as_u32(KV);
        unsigned int idx = 2048;
        if (kk < 4278190080u) {
            idx = KX;
        }
        unsigned int rkey = idx << 4 | (unsigned int)lnx;
        int rank = 0;
        unsigned int ox = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_144 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        if (ox_144 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_145 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        if (ox_145 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_146 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        if (ox_146 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_147 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        if (ox_147 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_148 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        if (ox_148 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_149 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        if (ox_149 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_150 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        if (ox_150 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_151 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        if (ox_151 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_152 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        if (ox_152 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_153 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        if (ox_153 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_154 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        if (ox_154 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_155 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        if (ox_155 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_156 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        if (ox_156 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_157 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        if (ox_157 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 2047) {
            val = (int)idx;
        }
        if (commit != 0) {
            unsigned int obase32 = ((unsigned int)ucol * (unsigned int)num_heads + (unsigned int)head) * 16;
            out[obase32 + (unsigned int)rank] = val;
        }
    }
}

} // extern "C"
