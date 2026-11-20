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

    // Prologue
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
    int td_1_0 = t0;
    float sc = __uint_as_float(raw[0]);
    sc = fmaxf(sc, konb);
    sc = fminf(sc, 1.7014118e+38f);
    unsigned int sb_1_0 = __as_u32(sc);
    if (sb_1_0 == 2147483648u) {
        sb_1_0 = 0;
    }
    unsigned int vw_1_0 = sb_1_0;
    unsigned int kw_1_0 = sb_1_0 & 4294967280u | 0;
    if (sb_1_0 == 4278190080u) {
        kw_1_0 = 4278190080u;
    }
    if (td_1_0 >= lim || raw[0] == 4286578688u) {
        vw_1_0 = 4282384384u;
        kw_1_0 = 4282384384u;
    }
    if ((td_1_0 < fb || td_1_0 >= lim_fe) && td_1_0 < lim) {
        unsigned int fcls_1_0 = 2139095024 - ((unsigned int)td_1_0 << 4);
        vw_1_0 = fcls_1_0;
        kw_1_0 = fcls_1_0 | 0;
    }
    scr[tid_1 * 17] = __uint_as_float(vw_1_0);
    kb[0] = __uint_as_float(kw_1_0);
    int td_1_1 = t0 + ts_r;
    float sc_0 = __uint_as_float(raw[1]);
    sc_0 = fmaxf(sc_0, konb);
    sc_0 = fminf(sc_0, 1.7014118e+38f);
    unsigned int sb_1_1 = __as_u32(sc_0);
    if (sb_1_1 == 2147483648u) {
        sb_1_1 = 0;
    }
    unsigned int vw_1_1 = sb_1_1;
    unsigned int kw_1_1 = sb_1_1 & 4294967280u | 1;
    if (sb_1_1 == 4278190080u) {
        kw_1_1 = 4278190097u;
    }
    if (td_1_1 >= lim || raw[1] == 4286578688u) {
        vw_1_1 = 4282384384u;
        kw_1_1 = 4282384401u;
    }
    if ((td_1_1 < fb || td_1_1 >= lim_fe) && td_1_1 < lim) {
        unsigned int fcls_1_1 = 2139095024 - ((unsigned int)td_1_1 << 4);
        vw_1_1 = fcls_1_1;
        kw_1_1 = fcls_1_1 | 1;
    }
    scr[tid_1 * 17 + 1] = __uint_as_float(vw_1_1);
    kb[1] = __uint_as_float(kw_1_1);
    int td_1_2 = t0 + 2 * ts_r;
    float sc_1 = __uint_as_float(raw[2]);
    sc_1 = fmaxf(sc_1, konb);
    sc_1 = fminf(sc_1, 1.7014118e+38f);
    unsigned int sb_1_2 = __as_u32(sc_1);
    if (sb_1_2 == 2147483648u) {
        sb_1_2 = 0;
    }
    unsigned int vw_1_2 = sb_1_2;
    unsigned int kw_1_2 = sb_1_2 & 4294967280u | 2;
    if (sb_1_2 == 4278190080u) {
        kw_1_2 = 4278190114u;
    }
    if (td_1_2 >= lim || raw[2] == 4286578688u) {
        vw_1_2 = 4282384384u;
        kw_1_2 = 4282384418u;
    }
    if ((td_1_2 < fb || td_1_2 >= lim_fe) && td_1_2 < lim) {
        unsigned int fcls_1_2 = 2139095024 - ((unsigned int)td_1_2 << 4);
        vw_1_2 = fcls_1_2;
        kw_1_2 = fcls_1_2 | 2;
    }
    scr[tid_1 * 17 + 2] = __uint_as_float(vw_1_2);
    kb[2] = __uint_as_float(kw_1_2);
    int td_1_3 = t0 + 3 * ts_r;
    float sc_2 = __uint_as_float(raw[3]);
    sc_2 = fmaxf(sc_2, konb);
    sc_2 = fminf(sc_2, 1.7014118e+38f);
    unsigned int sb_1_3 = __as_u32(sc_2);
    if (sb_1_3 == 2147483648u) {
        sb_1_3 = 0;
    }
    unsigned int vw_1_3 = sb_1_3;
    unsigned int kw_1_3 = sb_1_3 & 4294967280u | 3;
    if (sb_1_3 == 4278190080u) {
        kw_1_3 = 4278190131u;
    }
    if (td_1_3 >= lim || raw[3] == 4286578688u) {
        vw_1_3 = 4282384384u;
        kw_1_3 = 4282384435u;
    }
    if ((td_1_3 < fb || td_1_3 >= lim_fe) && td_1_3 < lim) {
        unsigned int fcls_1_3 = 2139095024 - ((unsigned int)td_1_3 << 4);
        vw_1_3 = fcls_1_3;
        kw_1_3 = fcls_1_3 | 3;
    }
    scr[tid_1 * 17 + 3] = __uint_as_float(vw_1_3);
    kb[3] = __uint_as_float(kw_1_3);
    int td_1_4 = t0 + 4 * ts_r;
    float sc_3 = __uint_as_float(raw[4]);
    sc_3 = fmaxf(sc_3, konb);
    sc_3 = fminf(sc_3, 1.7014118e+38f);
    unsigned int sb_1_4 = __as_u32(sc_3);
    if (sb_1_4 == 2147483648u) {
        sb_1_4 = 0;
    }
    unsigned int vw_1_4 = sb_1_4;
    unsigned int kw_1_4 = sb_1_4 & 4294967280u | 4;
    if (sb_1_4 == 4278190080u) {
        kw_1_4 = 4278190148u;
    }
    if (td_1_4 >= lim || raw[4] == 4286578688u) {
        vw_1_4 = 4282384384u;
        kw_1_4 = 4282384452u;
    }
    if ((td_1_4 < fb || td_1_4 >= lim_fe) && td_1_4 < lim) {
        unsigned int fcls_1_4 = 2139095024 - ((unsigned int)td_1_4 << 4);
        vw_1_4 = fcls_1_4;
        kw_1_4 = fcls_1_4 | 4;
    }
    scr[tid_1 * 17 + 4] = __uint_as_float(vw_1_4);
    kb[4] = __uint_as_float(kw_1_4);
    int td_1_5 = t0 + 5 * ts_r;
    float sc_4 = __uint_as_float(raw[5]);
    sc_4 = fmaxf(sc_4, konb);
    sc_4 = fminf(sc_4, 1.7014118e+38f);
    unsigned int sb_1_5 = __as_u32(sc_4);
    if (sb_1_5 == 2147483648u) {
        sb_1_5 = 0;
    }
    unsigned int vw_1_5 = sb_1_5;
    unsigned int kw_1_5 = sb_1_5 & 4294967280u | 5;
    if (sb_1_5 == 4278190080u) {
        kw_1_5 = 4278190165u;
    }
    if (td_1_5 >= lim || raw[5] == 4286578688u) {
        vw_1_5 = 4282384384u;
        kw_1_5 = 4282384469u;
    }
    if ((td_1_5 < fb || td_1_5 >= lim_fe) && td_1_5 < lim) {
        unsigned int fcls_1_5 = 2139095024 - ((unsigned int)td_1_5 << 4);
        vw_1_5 = fcls_1_5;
        kw_1_5 = fcls_1_5 | 5;
    }
    scr[tid_1 * 17 + 5] = __uint_as_float(vw_1_5);
    kb[5] = __uint_as_float(kw_1_5);
    int td_1_6 = t0 + 6 * ts_r;
    float sc_5 = __uint_as_float(raw[6]);
    sc_5 = fmaxf(sc_5, konb);
    sc_5 = fminf(sc_5, 1.7014118e+38f);
    unsigned int sb_1_6 = __as_u32(sc_5);
    if (sb_1_6 == 2147483648u) {
        sb_1_6 = 0;
    }
    unsigned int vw_1_6 = sb_1_6;
    unsigned int kw_1_6 = sb_1_6 & 4294967280u | 6;
    if (sb_1_6 == 4278190080u) {
        kw_1_6 = 4278190182u;
    }
    if (td_1_6 >= lim || raw[6] == 4286578688u) {
        vw_1_6 = 4282384384u;
        kw_1_6 = 4282384486u;
    }
    if ((td_1_6 < fb || td_1_6 >= lim_fe) && td_1_6 < lim) {
        unsigned int fcls_1_6 = 2139095024 - ((unsigned int)td_1_6 << 4);
        vw_1_6 = fcls_1_6;
        kw_1_6 = fcls_1_6 | 6;
    }
    scr[tid_1 * 17 + 6] = __uint_as_float(vw_1_6);
    kb[6] = __uint_as_float(kw_1_6);
    int td_1_7 = t0 + 7 * ts_r;
    float sc_6 = __uint_as_float(raw[7]);
    sc_6 = fmaxf(sc_6, konb);
    sc_6 = fminf(sc_6, 1.7014118e+38f);
    unsigned int sb_1_7 = __as_u32(sc_6);
    if (sb_1_7 == 2147483648u) {
        sb_1_7 = 0;
    }
    unsigned int vw_1_7 = sb_1_7;
    unsigned int kw_1_7 = sb_1_7 & 4294967280u | 7;
    if (sb_1_7 == 4278190080u) {
        kw_1_7 = 4278190199u;
    }
    if (td_1_7 >= lim || raw[7] == 4286578688u) {
        vw_1_7 = 4282384384u;
        kw_1_7 = 4282384503u;
    }
    if ((td_1_7 < fb || td_1_7 >= lim_fe) && td_1_7 < lim) {
        unsigned int fcls_1_7 = 2139095024 - ((unsigned int)td_1_7 << 4);
        vw_1_7 = fcls_1_7;
        kw_1_7 = fcls_1_7 | 7;
    }
    scr[tid_1 * 17 + 7] = __uint_as_float(vw_1_7);
    kb[7] = __uint_as_float(kw_1_7);
    int td_1_8 = t0 + 8 * ts_r;
    float sc_7 = __uint_as_float(raw[8]);
    sc_7 = fmaxf(sc_7, konb);
    sc_7 = fminf(sc_7, 1.7014118e+38f);
    unsigned int sb_1_8 = __as_u32(sc_7);
    if (sb_1_8 == 2147483648u) {
        sb_1_8 = 0;
    }
    unsigned int vw_1_8 = sb_1_8;
    unsigned int kw_1_8 = sb_1_8 & 4294967280u | 8;
    if (sb_1_8 == 4278190080u) {
        kw_1_8 = 4278190216u;
    }
    if (td_1_8 >= lim || raw[8] == 4286578688u) {
        vw_1_8 = 4282384384u;
        kw_1_8 = 4282384520u;
    }
    if ((td_1_8 < fb || td_1_8 >= lim_fe) && td_1_8 < lim) {
        unsigned int fcls_1_8 = 2139095024 - ((unsigned int)td_1_8 << 4);
        vw_1_8 = fcls_1_8;
        kw_1_8 = fcls_1_8 | 8;
    }
    scr[tid_1 * 17 + 8] = __uint_as_float(vw_1_8);
    kb[8] = __uint_as_float(kw_1_8);
    int td_1_9 = t0 + 9 * ts_r;
    float sc_8 = __uint_as_float(raw[9]);
    sc_8 = fmaxf(sc_8, konb);
    sc_8 = fminf(sc_8, 1.7014118e+38f);
    unsigned int sb_1_9 = __as_u32(sc_8);
    if (sb_1_9 == 2147483648u) {
        sb_1_9 = 0;
    }
    unsigned int vw_1_9 = sb_1_9;
    unsigned int kw_1_9 = sb_1_9 & 4294967280u | 9;
    if (sb_1_9 == 4278190080u) {
        kw_1_9 = 4278190233u;
    }
    if (td_1_9 >= lim || raw[9] == 4286578688u) {
        vw_1_9 = 4282384384u;
        kw_1_9 = 4282384537u;
    }
    if ((td_1_9 < fb || td_1_9 >= lim_fe) && td_1_9 < lim) {
        unsigned int fcls_1_9 = 2139095024 - ((unsigned int)td_1_9 << 4);
        vw_1_9 = fcls_1_9;
        kw_1_9 = fcls_1_9 | 9;
    }
    scr[tid_1 * 17 + 9] = __uint_as_float(vw_1_9);
    kb[9] = __uint_as_float(kw_1_9);
    int td_1_10 = t0 + 10 * ts_r;
    float sc_9 = __uint_as_float(raw[10]);
    sc_9 = fmaxf(sc_9, konb);
    sc_9 = fminf(sc_9, 1.7014118e+38f);
    unsigned int sb_1_10 = __as_u32(sc_9);
    if (sb_1_10 == 2147483648u) {
        sb_1_10 = 0;
    }
    unsigned int vw_1_10 = sb_1_10;
    unsigned int kw_1_10 = sb_1_10 & 4294967280u | 10;
    if (sb_1_10 == 4278190080u) {
        kw_1_10 = 4278190250u;
    }
    if (td_1_10 >= lim || raw[10] == 4286578688u) {
        vw_1_10 = 4282384384u;
        kw_1_10 = 4282384554u;
    }
    if ((td_1_10 < fb || td_1_10 >= lim_fe) && td_1_10 < lim) {
        unsigned int fcls_1_10 = 2139095024 - ((unsigned int)td_1_10 << 4);
        vw_1_10 = fcls_1_10;
        kw_1_10 = fcls_1_10 | 10;
    }
    scr[tid_1 * 17 + 10] = __uint_as_float(vw_1_10);
    kb[10] = __uint_as_float(kw_1_10);
    int td_1_11 = t0 + 11 * ts_r;
    float sc_10 = __uint_as_float(raw[11]);
    sc_10 = fmaxf(sc_10, konb);
    sc_10 = fminf(sc_10, 1.7014118e+38f);
    unsigned int sb_1_11 = __as_u32(sc_10);
    if (sb_1_11 == 2147483648u) {
        sb_1_11 = 0;
    }
    unsigned int vw_1_11 = sb_1_11;
    unsigned int kw_1_11 = sb_1_11 & 4294967280u | 11;
    if (sb_1_11 == 4278190080u) {
        kw_1_11 = 4278190267u;
    }
    if (td_1_11 >= lim || raw[11] == 4286578688u) {
        vw_1_11 = 4282384384u;
        kw_1_11 = 4282384571u;
    }
    if ((td_1_11 < fb || td_1_11 >= lim_fe) && td_1_11 < lim) {
        unsigned int fcls_1_11 = 2139095024 - ((unsigned int)td_1_11 << 4);
        vw_1_11 = fcls_1_11;
        kw_1_11 = fcls_1_11 | 11;
    }
    scr[tid_1 * 17 + 11] = __uint_as_float(vw_1_11);
    kb[11] = __uint_as_float(kw_1_11);
    int td_1_12 = t0 + 12 * ts_r;
    float sc_11 = __uint_as_float(raw[12]);
    sc_11 = fmaxf(sc_11, konb);
    sc_11 = fminf(sc_11, 1.7014118e+38f);
    unsigned int sb_1_12 = __as_u32(sc_11);
    if (sb_1_12 == 2147483648u) {
        sb_1_12 = 0;
    }
    unsigned int vw_1_12 = sb_1_12;
    unsigned int kw_1_12 = sb_1_12 & 4294967280u | 12;
    if (sb_1_12 == 4278190080u) {
        kw_1_12 = 4278190284u;
    }
    if (td_1_12 >= lim || raw[12] == 4286578688u) {
        vw_1_12 = 4282384384u;
        kw_1_12 = 4282384588u;
    }
    if ((td_1_12 < fb || td_1_12 >= lim_fe) && td_1_12 < lim) {
        unsigned int fcls_1_12 = 2139095024 - ((unsigned int)td_1_12 << 4);
        vw_1_12 = fcls_1_12;
        kw_1_12 = fcls_1_12 | 12;
    }
    scr[tid_1 * 17 + 12] = __uint_as_float(vw_1_12);
    kb[12] = __uint_as_float(kw_1_12);
    int td_1_13 = t0 + 13 * ts_r;
    float sc_12 = __uint_as_float(raw[13]);
    sc_12 = fmaxf(sc_12, konb);
    sc_12 = fminf(sc_12, 1.7014118e+38f);
    unsigned int sb_1_13 = __as_u32(sc_12);
    if (sb_1_13 == 2147483648u) {
        sb_1_13 = 0;
    }
    unsigned int vw_1_13 = sb_1_13;
    unsigned int kw_1_13 = sb_1_13 & 4294967280u | 13;
    if (sb_1_13 == 4278190080u) {
        kw_1_13 = 4278190301u;
    }
    if (td_1_13 >= lim || raw[13] == 4286578688u) {
        vw_1_13 = 4282384384u;
        kw_1_13 = 4282384605u;
    }
    if ((td_1_13 < fb || td_1_13 >= lim_fe) && td_1_13 < lim) {
        unsigned int fcls_1_13 = 2139095024 - ((unsigned int)td_1_13 << 4);
        vw_1_13 = fcls_1_13;
        kw_1_13 = fcls_1_13 | 13;
    }
    scr[tid_1 * 17 + 13] = __uint_as_float(vw_1_13);
    kb[13] = __uint_as_float(kw_1_13);
    int td_1_14 = t0 + 14 * ts_r;
    float sc_13 = __uint_as_float(raw[14]);
    sc_13 = fmaxf(sc_13, konb);
    sc_13 = fminf(sc_13, 1.7014118e+38f);
    unsigned int sb_1_14 = __as_u32(sc_13);
    if (sb_1_14 == 2147483648u) {
        sb_1_14 = 0;
    }
    unsigned int vw_1_14 = sb_1_14;
    unsigned int kw_1_14 = sb_1_14 & 4294967280u | 14;
    if (sb_1_14 == 4278190080u) {
        kw_1_14 = 4278190318u;
    }
    if (td_1_14 >= lim || raw[14] == 4286578688u) {
        vw_1_14 = 4282384384u;
        kw_1_14 = 4282384622u;
    }
    if ((td_1_14 < fb || td_1_14 >= lim_fe) && td_1_14 < lim) {
        unsigned int fcls_1_14 = 2139095024 - ((unsigned int)td_1_14 << 4);
        vw_1_14 = fcls_1_14;
        kw_1_14 = fcls_1_14 | 14;
    }
    scr[tid_1 * 17 + 14] = __uint_as_float(vw_1_14);
    kb[14] = __uint_as_float(kw_1_14);
    int td_1_15 = t0 + 15 * ts_r;
    float sc_14 = __uint_as_float(raw[15]);
    sc_14 = fmaxf(sc_14, konb);
    sc_14 = fminf(sc_14, 1.7014118e+38f);
    unsigned int sb_1_15 = __as_u32(sc_14);
    if (sb_1_15 == 2147483648u) {
        sb_1_15 = 0;
    }
    unsigned int vw_1_15 = sb_1_15;
    unsigned int kw_1_15 = sb_1_15 & 4294967280u | 15;
    if (sb_1_15 == 4278190080u) {
        kw_1_15 = 4278190335u;
    }
    if (td_1_15 >= lim || raw[15] == 4286578688u) {
        vw_1_15 = 4282384384u;
        kw_1_15 = 4282384639u;
    }
    if ((td_1_15 < fb || td_1_15 >= lim_fe) && td_1_15 < lim) {
        unsigned int fcls_1_15 = 2139095024 - ((unsigned int)td_1_15 << 4);
        vw_1_15 = fcls_1_15;
        kw_1_15 = fcls_1_15 | 15;
    }
    scr[tid_1 * 17 + 15] = __uint_as_float(vw_1_15);
    kb[15] = __uint_as_float(kw_1_15);
    float hi_1_0 = fmaxf(kb[0], kb[13]);
    float lo_1_0 = fminf(kb[0], kb[13]);
    kb[0] = hi_1_0;
    kb[13] = lo_1_0;
    float hi_1_1 = fmaxf(kb[1], kb[12]);
    float lo_1_1 = fminf(kb[1], kb[12]);
    kb[1] = hi_1_1;
    kb[12] = lo_1_1;
    float hi_1_2 = fmaxf(kb[2], kb[15]);
    float lo_1_2 = fminf(kb[2], kb[15]);
    kb[2] = hi_1_2;
    kb[15] = lo_1_2;
    float hi_1_3 = fmaxf(kb[3], kb[14]);
    float lo_1_3 = fminf(kb[3], kb[14]);
    kb[3] = hi_1_3;
    kb[14] = lo_1_3;
    float hi_1_4 = fmaxf(kb[4], kb[8]);
    float lo_1_4 = fminf(kb[4], kb[8]);
    kb[4] = hi_1_4;
    kb[8] = lo_1_4;
    float hi_1_5 = fmaxf(kb[5], kb[6]);
    float lo_1_5 = fminf(kb[5], kb[6]);
    kb[5] = hi_1_5;
    kb[6] = lo_1_5;
    float hi_1_6 = fmaxf(kb[7], kb[11]);
    float lo_1_6 = fminf(kb[7], kb[11]);
    kb[7] = hi_1_6;
    kb[11] = lo_1_6;
    float hi_1_7 = fmaxf(kb[9], kb[10]);
    float lo_1_7 = fminf(kb[9], kb[10]);
    kb[9] = hi_1_7;
    kb[10] = lo_1_7;
    float hi_1_8 = fmaxf(kb[0], kb[5]);
    float lo_1_8 = fminf(kb[0], kb[5]);
    kb[0] = hi_1_8;
    kb[5] = lo_1_8;
    float hi_1_9 = fmaxf(kb[1], kb[7]);
    float lo_1_9 = fminf(kb[1], kb[7]);
    kb[1] = hi_1_9;
    kb[7] = lo_1_9;
    float hi_1_10 = fmaxf(kb[2], kb[9]);
    float lo_1_10 = fminf(kb[2], kb[9]);
    kb[2] = hi_1_10;
    kb[9] = lo_1_10;
    float hi_1_11 = fmaxf(kb[3], kb[4]);
    float lo_1_11 = fminf(kb[3], kb[4]);
    kb[3] = hi_1_11;
    kb[4] = lo_1_11;
    float hi_1_12 = fmaxf(kb[6], kb[13]);
    float lo_1_12 = fminf(kb[6], kb[13]);
    kb[6] = hi_1_12;
    kb[13] = lo_1_12;
    float hi_1_13 = fmaxf(kb[8], kb[14]);
    float lo_1_13 = fminf(kb[8], kb[14]);
    kb[8] = hi_1_13;
    kb[14] = lo_1_13;
    float hi_1_14 = fmaxf(kb[10], kb[15]);
    float lo_1_14 = fminf(kb[10], kb[15]);
    kb[10] = hi_1_14;
    kb[15] = lo_1_14;
    float hi_1_15 = fmaxf(kb[11], kb[12]);
    float lo_1_15 = fminf(kb[11], kb[12]);
    kb[11] = hi_1_15;
    kb[12] = lo_1_15;
    float hi_1_16 = fmaxf(kb[0], kb[1]);
    float lo_1_16 = fminf(kb[0], kb[1]);
    kb[0] = hi_1_16;
    kb[1] = lo_1_16;
    float hi_1_17 = fmaxf(kb[2], kb[3]);
    float lo_1_17 = fminf(kb[2], kb[3]);
    kb[2] = hi_1_17;
    kb[3] = lo_1_17;
    float hi_1_18 = fmaxf(kb[4], kb[5]);
    float lo_1_18 = fminf(kb[4], kb[5]);
    kb[4] = hi_1_18;
    kb[5] = lo_1_18;
    float hi_1_19 = fmaxf(kb[6], kb[8]);
    float lo_1_19 = fminf(kb[6], kb[8]);
    kb[6] = hi_1_19;
    kb[8] = lo_1_19;
    float hi_1_20 = fmaxf(kb[7], kb[9]);
    float lo_1_20 = fminf(kb[7], kb[9]);
    kb[7] = hi_1_20;
    kb[9] = lo_1_20;
    float hi_1_21 = fmaxf(kb[10], kb[11]);
    float lo_1_21 = fminf(kb[10], kb[11]);
    kb[10] = hi_1_21;
    kb[11] = lo_1_21;
    float hi_1_22 = fmaxf(kb[12], kb[13]);
    float lo_1_22 = fminf(kb[12], kb[13]);
    kb[12] = hi_1_22;
    kb[13] = lo_1_22;
    float hi_1_23 = fmaxf(kb[14], kb[15]);
    float lo_1_23 = fminf(kb[14], kb[15]);
    kb[14] = hi_1_23;
    kb[15] = lo_1_23;
    float hi_1_24 = fmaxf(kb[0], kb[2]);
    float lo_1_24 = fminf(kb[0], kb[2]);
    kb[0] = hi_1_24;
    kb[2] = lo_1_24;
    float hi_1_25 = fmaxf(kb[1], kb[3]);
    float lo_1_25 = fminf(kb[1], kb[3]);
    kb[1] = hi_1_25;
    kb[3] = lo_1_25;
    float hi_1_26 = fmaxf(kb[4], kb[10]);
    float lo_1_26 = fminf(kb[4], kb[10]);
    kb[4] = hi_1_26;
    kb[10] = lo_1_26;
    float hi_1_27 = fmaxf(kb[5], kb[11]);
    float lo_1_27 = fminf(kb[5], kb[11]);
    kb[5] = hi_1_27;
    kb[11] = lo_1_27;
    float hi_1_28 = fmaxf(kb[6], kb[7]);
    float lo_1_28 = fminf(kb[6], kb[7]);
    kb[6] = hi_1_28;
    kb[7] = lo_1_28;
    float hi_1_29 = fmaxf(kb[8], kb[9]);
    float lo_1_29 = fminf(kb[8], kb[9]);
    kb[8] = hi_1_29;
    kb[9] = lo_1_29;
    float hi_1_30 = fmaxf(kb[12], kb[14]);
    float lo_1_30 = fminf(kb[12], kb[14]);
    kb[12] = hi_1_30;
    kb[14] = lo_1_30;
    float hi_1_31 = fmaxf(kb[13], kb[15]);
    float lo_1_31 = fminf(kb[13], kb[15]);
    kb[13] = hi_1_31;
    kb[15] = lo_1_31;
    float hi_1_32 = fmaxf(kb[1], kb[2]);
    float lo_1_32 = fminf(kb[1], kb[2]);
    kb[1] = hi_1_32;
    kb[2] = lo_1_32;
    float hi_1_33 = fmaxf(kb[3], kb[12]);
    float lo_1_33 = fminf(kb[3], kb[12]);
    kb[3] = hi_1_33;
    kb[12] = lo_1_33;
    float hi_1_34 = fmaxf(kb[4], kb[6]);
    float lo_1_34 = fminf(kb[4], kb[6]);
    kb[4] = hi_1_34;
    kb[6] = lo_1_34;
    float hi_1_35 = fmaxf(kb[5], kb[7]);
    float lo_1_35 = fminf(kb[5], kb[7]);
    kb[5] = hi_1_35;
    kb[7] = lo_1_35;
    float hi_1_36 = fmaxf(kb[8], kb[10]);
    float lo_1_36 = fminf(kb[8], kb[10]);
    kb[8] = hi_1_36;
    kb[10] = lo_1_36;
    float hi_1_37 = fmaxf(kb[9], kb[11]);
    float lo_1_37 = fminf(kb[9], kb[11]);
    kb[9] = hi_1_37;
    kb[11] = lo_1_37;
    float hi_1_38 = fmaxf(kb[13], kb[14]);
    float lo_1_38 = fminf(kb[13], kb[14]);
    kb[13] = hi_1_38;
    kb[14] = lo_1_38;
    float hi_1_39 = fmaxf(kb[1], kb[4]);
    float lo_1_39 = fminf(kb[1], kb[4]);
    kb[1] = hi_1_39;
    kb[4] = lo_1_39;
    float hi_1_40 = fmaxf(kb[2], kb[6]);
    float lo_1_40 = fminf(kb[2], kb[6]);
    kb[2] = hi_1_40;
    kb[6] = lo_1_40;
    float hi_1_41 = fmaxf(kb[5], kb[8]);
    float lo_1_41 = fminf(kb[5], kb[8]);
    kb[5] = hi_1_41;
    kb[8] = lo_1_41;
    float hi_1_42 = fmaxf(kb[7], kb[10]);
    float lo_1_42 = fminf(kb[7], kb[10]);
    kb[7] = hi_1_42;
    kb[10] = lo_1_42;
    float hi_1_43 = fmaxf(kb[9], kb[13]);
    float lo_1_43 = fminf(kb[9], kb[13]);
    kb[9] = hi_1_43;
    kb[13] = lo_1_43;
    float hi_1_44 = fmaxf(kb[11], kb[14]);
    float lo_1_44 = fminf(kb[11], kb[14]);
    kb[11] = hi_1_44;
    kb[14] = lo_1_44;
    float hi_1_45 = fmaxf(kb[2], kb[4]);
    float lo_1_45 = fminf(kb[2], kb[4]);
    kb[2] = hi_1_45;
    kb[4] = lo_1_45;
    float hi_1_46 = fmaxf(kb[3], kb[6]);
    float lo_1_46 = fminf(kb[3], kb[6]);
    kb[3] = hi_1_46;
    kb[6] = lo_1_46;
    float hi_1_47 = fmaxf(kb[9], kb[12]);
    float lo_1_47 = fminf(kb[9], kb[12]);
    kb[9] = hi_1_47;
    kb[12] = lo_1_47;
    float hi_1_48 = fmaxf(kb[11], kb[13]);
    float lo_1_48 = fminf(kb[11], kb[13]);
    kb[11] = hi_1_48;
    kb[13] = lo_1_48;
    float hi_1_49 = fmaxf(kb[3], kb[5]);
    float lo_1_49 = fminf(kb[3], kb[5]);
    kb[3] = hi_1_49;
    kb[5] = lo_1_49;
    float hi_1_50 = fmaxf(kb[6], kb[8]);
    float lo_1_50 = fminf(kb[6], kb[8]);
    kb[6] = hi_1_50;
    kb[8] = lo_1_50;
    float hi_1_51 = fmaxf(kb[7], kb[9]);
    float lo_1_51 = fminf(kb[7], kb[9]);
    kb[7] = hi_1_51;
    kb[9] = lo_1_51;
    float hi_1_52 = fmaxf(kb[10], kb[12]);
    float lo_1_52 = fminf(kb[10], kb[12]);
    kb[10] = hi_1_52;
    kb[12] = lo_1_52;
    float hi_1_53 = fmaxf(kb[3], kb[4]);
    float lo_1_53 = fminf(kb[3], kb[4]);
    kb[3] = hi_1_53;
    kb[4] = lo_1_53;
    float hi_1_54 = fmaxf(kb[5], kb[6]);
    float lo_1_54 = fminf(kb[5], kb[6]);
    kb[5] = hi_1_54;
    kb[6] = lo_1_54;
    float hi_1_55 = fmaxf(kb[7], kb[8]);
    float lo_1_55 = fminf(kb[7], kb[8]);
    kb[7] = hi_1_55;
    kb[8] = lo_1_55;
    float hi_1_56 = fmaxf(kb[9], kb[10]);
    float lo_1_56 = fminf(kb[9], kb[10]);
    kb[9] = hi_1_56;
    kb[10] = lo_1_56;
    float hi_1_57 = fmaxf(kb[11], kb[12]);
    float lo_1_57 = fminf(kb[11], kb[12]);
    kb[11] = hi_1_57;
    kb[12] = lo_1_57;
    float hi_1_58 = fmaxf(kb[6], kb[7]);
    float lo_1_58 = fminf(kb[6], kb[7]);
    kb[6] = hi_1_58;
    kb[7] = lo_1_58;
    float hi_1_59 = fmaxf(kb[8], kb[9]);
    float lo_1_59 = fminf(kb[8], kb[9]);
    kb[8] = hi_1_59;
    kb[9] = lo_1_59;
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
        float hi = fmaxf(v[0], v[1]);
        float lo = fminf(v[0], v[1]);
        unsigned int xi = x[0];
        unsigned int xj = x[1];
        v[0] = hi;
        v[1] = lo;
        x[0] = ((p != 0) ? xi : xj);
        x[1] = ((p != 0) ? xj : xi);
        if (p == 0) {
            sw = 1;
        }
        int p_0 = 0;
        if (v[2] >= v[3]) {
            p_0 = 1;
        }
        float hi_0 = fmaxf(v[2], v[3]);
        float lo_0 = fminf(v[2], v[3]);
        unsigned int xi_0 = x[2];
        unsigned int xj_0 = x[3];
        v[2] = hi_0;
        v[3] = lo_0;
        x[2] = ((p_0 != 0) ? xi_0 : xj_0);
        x[3] = ((p_0 != 0) ? xj_0 : xi_0);
        if (p_0 == 0) {
            sw = 1;
        }
        int p_1 = 0;
        if (v[4] >= v[5]) {
            p_1 = 1;
        }
        float hi_1 = fmaxf(v[4], v[5]);
        float lo_1 = fminf(v[4], v[5]);
        unsigned int xi_1 = x[4];
        unsigned int xj_1 = x[5];
        v[4] = hi_1;
        v[5] = lo_1;
        x[4] = ((p_1 != 0) ? xi_1 : xj_1);
        x[5] = ((p_1 != 0) ? xj_1 : xi_1);
        if (p_1 == 0) {
            sw = 1;
        }
        int p_2 = 0;
        if (v[6] >= v[7]) {
            p_2 = 1;
        }
        float hi_2 = fmaxf(v[6], v[7]);
        float lo_2 = fminf(v[6], v[7]);
        unsigned int xi_2 = x[6];
        unsigned int xj_2 = x[7];
        v[6] = hi_2;
        v[7] = lo_2;
        x[6] = ((p_2 != 0) ? xi_2 : xj_2);
        x[7] = ((p_2 != 0) ? xj_2 : xi_2);
        if (p_2 == 0) {
            sw = 1;
        }
        int p_3 = 0;
        if (v[8] >= v[9]) {
            p_3 = 1;
        }
        float hi_3 = fmaxf(v[8], v[9]);
        float lo_3 = fminf(v[8], v[9]);
        unsigned int xi_3 = x[8];
        unsigned int xj_3 = x[9];
        v[8] = hi_3;
        v[9] = lo_3;
        x[8] = ((p_3 != 0) ? xi_3 : xj_3);
        x[9] = ((p_3 != 0) ? xj_3 : xi_3);
        if (p_3 == 0) {
            sw = 1;
        }
        int p_4 = 0;
        if (v[10] >= v[11]) {
            p_4 = 1;
        }
        float hi_4 = fmaxf(v[10], v[11]);
        float lo_4 = fminf(v[10], v[11]);
        unsigned int xi_4 = x[10];
        unsigned int xj_4 = x[11];
        v[10] = hi_4;
        v[11] = lo_4;
        x[10] = ((p_4 != 0) ? xi_4 : xj_4);
        x[11] = ((p_4 != 0) ? xj_4 : xi_4);
        if (p_4 == 0) {
            sw = 1;
        }
        int p_5 = 0;
        if (v[12] >= v[13]) {
            p_5 = 1;
        }
        float hi_5 = fmaxf(v[12], v[13]);
        float lo_5 = fminf(v[12], v[13]);
        unsigned int xi_5 = x[12];
        unsigned int xj_5 = x[13];
        v[12] = hi_5;
        v[13] = lo_5;
        x[12] = ((p_5 != 0) ? xi_5 : xj_5);
        x[13] = ((p_5 != 0) ? xj_5 : xi_5);
        if (p_5 == 0) {
            sw = 1;
        }
        int p_6 = 0;
        if (v[14] >= v[15]) {
            p_6 = 1;
        }
        float hi_6 = fmaxf(v[14], v[15]);
        float lo_6 = fminf(v[14], v[15]);
        unsigned int xi_6 = x[14];
        unsigned int xj_6 = x[15];
        v[14] = hi_6;
        v[15] = lo_6;
        x[14] = ((p_6 != 0) ? xi_6 : xj_6);
        x[15] = ((p_6 != 0) ? xj_6 : xi_6);
        if (p_6 == 0) {
            sw = 1;
        }
        int p_7 = 0;
        if (v[1] >= v[2]) {
            p_7 = 1;
        }
        float hi_7 = fmaxf(v[1], v[2]);
        float lo_7 = fminf(v[1], v[2]);
        unsigned int xi_7 = x[1];
        unsigned int xj_7 = x[2];
        v[1] = hi_7;
        v[2] = lo_7;
        x[1] = ((p_7 != 0) ? xi_7 : xj_7);
        x[2] = ((p_7 != 0) ? xj_7 : xi_7);
        if (p_7 == 0) {
            sw = 1;
        }
        int p_8 = 0;
        if (v[3] >= v[4]) {
            p_8 = 1;
        }
        float hi_8 = fmaxf(v[3], v[4]);
        float lo_8 = fminf(v[3], v[4]);
        unsigned int xi_8 = x[3];
        unsigned int xj_8 = x[4];
        v[3] = hi_8;
        v[4] = lo_8;
        x[3] = ((p_8 != 0) ? xi_8 : xj_8);
        x[4] = ((p_8 != 0) ? xj_8 : xi_8);
        if (p_8 == 0) {
            sw = 1;
        }
        int p_9 = 0;
        if (v[5] >= v[6]) {
            p_9 = 1;
        }
        float hi_9 = fmaxf(v[5], v[6]);
        float lo_9 = fminf(v[5], v[6]);
        unsigned int xi_9 = x[5];
        unsigned int xj_9 = x[6];
        v[5] = hi_9;
        v[6] = lo_9;
        x[5] = ((p_9 != 0) ? xi_9 : xj_9);
        x[6] = ((p_9 != 0) ? xj_9 : xi_9);
        if (p_9 == 0) {
            sw = 1;
        }
        int p_10 = 0;
        if (v[7] >= v[8]) {
            p_10 = 1;
        }
        float hi_10 = fmaxf(v[7], v[8]);
        float lo_10 = fminf(v[7], v[8]);
        unsigned int xi_10 = x[7];
        unsigned int xj_10 = x[8];
        v[7] = hi_10;
        v[8] = lo_10;
        x[7] = ((p_10 != 0) ? xi_10 : xj_10);
        x[8] = ((p_10 != 0) ? xj_10 : xi_10);
        if (p_10 == 0) {
            sw = 1;
        }
        int p_11 = 0;
        if (v[9] >= v[10]) {
            p_11 = 1;
        }
        float hi_11 = fmaxf(v[9], v[10]);
        float lo_11 = fminf(v[9], v[10]);
        unsigned int xi_11 = x[9];
        unsigned int xj_11 = x[10];
        v[9] = hi_11;
        v[10] = lo_11;
        x[9] = ((p_11 != 0) ? xi_11 : xj_11);
        x[10] = ((p_11 != 0) ? xj_11 : xi_11);
        if (p_11 == 0) {
            sw = 1;
        }
        int p_12 = 0;
        if (v[11] >= v[12]) {
            p_12 = 1;
        }
        float hi_12 = fmaxf(v[11], v[12]);
        float lo_12 = fminf(v[11], v[12]);
        unsigned int xi_12 = x[11];
        unsigned int xj_12 = x[12];
        v[11] = hi_12;
        v[12] = lo_12;
        x[11] = ((p_12 != 0) ? xi_12 : xj_12);
        x[12] = ((p_12 != 0) ? xj_12 : xi_12);
        if (p_12 == 0) {
            sw = 1;
        }
        int p_13 = 0;
        if (v[13] >= v[14]) {
            p_13 = 1;
        }
        float hi_13 = fmaxf(v[13], v[14]);
        float lo_13 = fminf(v[13], v[14]);
        unsigned int xi_13 = x[13];
        unsigned int xj_13 = x[14];
        v[13] = hi_13;
        v[14] = lo_13;
        x[13] = ((p_13 != 0) ? xi_13 : xj_13);
        x[14] = ((p_13 != 0) ? xj_13 : xi_13);
        if (p_13 == 0) {
            sw = 1;
        }
        int _warp_vote_0 = __any_sync(0xFFFFFFFF, sw != 0);
        more = _warp_vote_0;
    }
    if (more != 0) {
        int sw_1 = 0;
        int p_14 = 0;
        if (v[0] >= v[1]) {
            p_14 = 1;
        }
        float hi_14 = fmaxf(v[0], v[1]);
        float lo_14 = fminf(v[0], v[1]);
        unsigned int xi_14 = x[0];
        unsigned int xj_14 = x[1];
        v[0] = hi_14;
        v[1] = lo_14;
        x[0] = ((p_14 != 0) ? xi_14 : xj_14);
        x[1] = ((p_14 != 0) ? xj_14 : xi_14);
        if (p_14 == 0) {
            sw_1 = 1;
        }
        int p_14_1 = 0;
        if (v[2] >= v[3]) {
            p_14_1 = 1;
        }
        float hi_14_1 = fmaxf(v[2], v[3]);
        float lo_14_1 = fminf(v[2], v[3]);
        unsigned int xi_14_1 = x[2];
        unsigned int xj_14_1 = x[3];
        v[2] = hi_14_1;
        v[3] = lo_14_1;
        x[2] = ((p_14_1 != 0) ? xi_14_1 : xj_14_1);
        x[3] = ((p_14_1 != 0) ? xj_14_1 : xi_14_1);
        if (p_14_1 == 0) {
            sw_1 = 1;
        }
        int p_15 = 0;
        if (v[4] >= v[5]) {
            p_15 = 1;
        }
        float hi_15 = fmaxf(v[4], v[5]);
        float lo_15 = fminf(v[4], v[5]);
        unsigned int xi_15 = x[4];
        unsigned int xj_15 = x[5];
        v[4] = hi_15;
        v[5] = lo_15;
        x[4] = ((p_15 != 0) ? xi_15 : xj_15);
        x[5] = ((p_15 != 0) ? xj_15 : xi_15);
        if (p_15 == 0) {
            sw_1 = 1;
        }
        int p_16 = 0;
        if (v[6] >= v[7]) {
            p_16 = 1;
        }
        float hi_16 = fmaxf(v[6], v[7]);
        float lo_16 = fminf(v[6], v[7]);
        unsigned int xi_16 = x[6];
        unsigned int xj_16 = x[7];
        v[6] = hi_16;
        v[7] = lo_16;
        x[6] = ((p_16 != 0) ? xi_16 : xj_16);
        x[7] = ((p_16 != 0) ? xj_16 : xi_16);
        if (p_16 == 0) {
            sw_1 = 1;
        }
        int p_17 = 0;
        if (v[8] >= v[9]) {
            p_17 = 1;
        }
        float hi_17 = fmaxf(v[8], v[9]);
        float lo_17 = fminf(v[8], v[9]);
        unsigned int xi_17 = x[8];
        unsigned int xj_17 = x[9];
        v[8] = hi_17;
        v[9] = lo_17;
        x[8] = ((p_17 != 0) ? xi_17 : xj_17);
        x[9] = ((p_17 != 0) ? xj_17 : xi_17);
        if (p_17 == 0) {
            sw_1 = 1;
        }
        int p_18 = 0;
        if (v[10] >= v[11]) {
            p_18 = 1;
        }
        float hi_18 = fmaxf(v[10], v[11]);
        float lo_18 = fminf(v[10], v[11]);
        unsigned int xi_18 = x[10];
        unsigned int xj_18 = x[11];
        v[10] = hi_18;
        v[11] = lo_18;
        x[10] = ((p_18 != 0) ? xi_18 : xj_18);
        x[11] = ((p_18 != 0) ? xj_18 : xi_18);
        if (p_18 == 0) {
            sw_1 = 1;
        }
        int p_19 = 0;
        if (v[12] >= v[13]) {
            p_19 = 1;
        }
        float hi_19 = fmaxf(v[12], v[13]);
        float lo_19 = fminf(v[12], v[13]);
        unsigned int xi_19 = x[12];
        unsigned int xj_19 = x[13];
        v[12] = hi_19;
        v[13] = lo_19;
        x[12] = ((p_19 != 0) ? xi_19 : xj_19);
        x[13] = ((p_19 != 0) ? xj_19 : xi_19);
        if (p_19 == 0) {
            sw_1 = 1;
        }
        int p_20 = 0;
        if (v[14] >= v[15]) {
            p_20 = 1;
        }
        float hi_20 = fmaxf(v[14], v[15]);
        float lo_20 = fminf(v[14], v[15]);
        unsigned int xi_20 = x[14];
        unsigned int xj_20 = x[15];
        v[14] = hi_20;
        v[15] = lo_20;
        x[14] = ((p_20 != 0) ? xi_20 : xj_20);
        x[15] = ((p_20 != 0) ? xj_20 : xi_20);
        if (p_20 == 0) {
            sw_1 = 1;
        }
        int p_21 = 0;
        if (v[1] >= v[2]) {
            p_21 = 1;
        }
        float hi_21 = fmaxf(v[1], v[2]);
        float lo_21 = fminf(v[1], v[2]);
        unsigned int xi_21 = x[1];
        unsigned int xj_21 = x[2];
        v[1] = hi_21;
        v[2] = lo_21;
        x[1] = ((p_21 != 0) ? xi_21 : xj_21);
        x[2] = ((p_21 != 0) ? xj_21 : xi_21);
        if (p_21 == 0) {
            sw_1 = 1;
        }
        int p_22 = 0;
        if (v[3] >= v[4]) {
            p_22 = 1;
        }
        float hi_22 = fmaxf(v[3], v[4]);
        float lo_22 = fminf(v[3], v[4]);
        unsigned int xi_22 = x[3];
        unsigned int xj_22 = x[4];
        v[3] = hi_22;
        v[4] = lo_22;
        x[3] = ((p_22 != 0) ? xi_22 : xj_22);
        x[4] = ((p_22 != 0) ? xj_22 : xi_22);
        if (p_22 == 0) {
            sw_1 = 1;
        }
        int p_23 = 0;
        if (v[5] >= v[6]) {
            p_23 = 1;
        }
        float hi_23 = fmaxf(v[5], v[6]);
        float lo_23 = fminf(v[5], v[6]);
        unsigned int xi_23 = x[5];
        unsigned int xj_23 = x[6];
        v[5] = hi_23;
        v[6] = lo_23;
        x[5] = ((p_23 != 0) ? xi_23 : xj_23);
        x[6] = ((p_23 != 0) ? xj_23 : xi_23);
        if (p_23 == 0) {
            sw_1 = 1;
        }
        int p_24 = 0;
        if (v[7] >= v[8]) {
            p_24 = 1;
        }
        float hi_24 = fmaxf(v[7], v[8]);
        float lo_24 = fminf(v[7], v[8]);
        unsigned int xi_24 = x[7];
        unsigned int xj_24 = x[8];
        v[7] = hi_24;
        v[8] = lo_24;
        x[7] = ((p_24 != 0) ? xi_24 : xj_24);
        x[8] = ((p_24 != 0) ? xj_24 : xi_24);
        if (p_24 == 0) {
            sw_1 = 1;
        }
        int p_25 = 0;
        if (v[9] >= v[10]) {
            p_25 = 1;
        }
        float hi_25 = fmaxf(v[9], v[10]);
        float lo_25 = fminf(v[9], v[10]);
        unsigned int xi_25 = x[9];
        unsigned int xj_25 = x[10];
        v[9] = hi_25;
        v[10] = lo_25;
        x[9] = ((p_25 != 0) ? xi_25 : xj_25);
        x[10] = ((p_25 != 0) ? xj_25 : xi_25);
        if (p_25 == 0) {
            sw_1 = 1;
        }
        int p_26 = 0;
        if (v[11] >= v[12]) {
            p_26 = 1;
        }
        float hi_26 = fmaxf(v[11], v[12]);
        float lo_26 = fminf(v[11], v[12]);
        unsigned int xi_26 = x[11];
        unsigned int xj_26 = x[12];
        v[11] = hi_26;
        v[12] = lo_26;
        x[11] = ((p_26 != 0) ? xi_26 : xj_26);
        x[12] = ((p_26 != 0) ? xj_26 : xi_26);
        if (p_26 == 0) {
            sw_1 = 1;
        }
        int p_27 = 0;
        if (v[13] >= v[14]) {
            p_27 = 1;
        }
        float hi_27 = fmaxf(v[13], v[14]);
        float lo_27 = fminf(v[13], v[14]);
        unsigned int xi_27 = x[13];
        unsigned int xj_27 = x[14];
        v[13] = hi_27;
        v[14] = lo_27;
        x[13] = ((p_27 != 0) ? xi_27 : xj_27);
        x[14] = ((p_27 != 0) ? xj_27 : xi_27);
        if (p_27 == 0) {
            sw_1 = 1;
        }
        int _warp_vote_1 = __any_sync(0xFFFFFFFF, sw_1 != 0);
        more = _warp_vote_1;
    }
    if (more != 0) {
        int sw_2 = 0;
        int p_28 = 0;
        if (v[0] >= v[1]) {
            p_28 = 1;
        }
        float hi_28 = fmaxf(v[0], v[1]);
        float lo_28 = fminf(v[0], v[1]);
        unsigned int xi_28 = x[0];
        unsigned int xj_28 = x[1];
        v[0] = hi_28;
        v[1] = lo_28;
        x[0] = ((p_28 != 0) ? xi_28 : xj_28);
        x[1] = ((p_28 != 0) ? xj_28 : xi_28);
        if (p_28 == 0) {
            sw_2 = 1;
        }
        int p_28_1 = 0;
        if (v[2] >= v[3]) {
            p_28_1 = 1;
        }
        float hi_28_1 = fmaxf(v[2], v[3]);
        float lo_28_1 = fminf(v[2], v[3]);
        unsigned int xi_28_1 = x[2];
        unsigned int xj_28_1 = x[3];
        v[2] = hi_28_1;
        v[3] = lo_28_1;
        x[2] = ((p_28_1 != 0) ? xi_28_1 : xj_28_1);
        x[3] = ((p_28_1 != 0) ? xj_28_1 : xi_28_1);
        if (p_28_1 == 0) {
            sw_2 = 1;
        }
        int p_29 = 0;
        if (v[4] >= v[5]) {
            p_29 = 1;
        }
        float hi_29 = fmaxf(v[4], v[5]);
        float lo_29 = fminf(v[4], v[5]);
        unsigned int xi_29 = x[4];
        unsigned int xj_29 = x[5];
        v[4] = hi_29;
        v[5] = lo_29;
        x[4] = ((p_29 != 0) ? xi_29 : xj_29);
        x[5] = ((p_29 != 0) ? xj_29 : xi_29);
        if (p_29 == 0) {
            sw_2 = 1;
        }
        int p_30 = 0;
        if (v[6] >= v[7]) {
            p_30 = 1;
        }
        float hi_30 = fmaxf(v[6], v[7]);
        float lo_30 = fminf(v[6], v[7]);
        unsigned int xi_30 = x[6];
        unsigned int xj_30 = x[7];
        v[6] = hi_30;
        v[7] = lo_30;
        x[6] = ((p_30 != 0) ? xi_30 : xj_30);
        x[7] = ((p_30 != 0) ? xj_30 : xi_30);
        if (p_30 == 0) {
            sw_2 = 1;
        }
        int p_31 = 0;
        if (v[8] >= v[9]) {
            p_31 = 1;
        }
        float hi_31 = fmaxf(v[8], v[9]);
        float lo_31 = fminf(v[8], v[9]);
        unsigned int xi_31 = x[8];
        unsigned int xj_31 = x[9];
        v[8] = hi_31;
        v[9] = lo_31;
        x[8] = ((p_31 != 0) ? xi_31 : xj_31);
        x[9] = ((p_31 != 0) ? xj_31 : xi_31);
        if (p_31 == 0) {
            sw_2 = 1;
        }
        int p_32 = 0;
        if (v[10] >= v[11]) {
            p_32 = 1;
        }
        float hi_32 = fmaxf(v[10], v[11]);
        float lo_32 = fminf(v[10], v[11]);
        unsigned int xi_32 = x[10];
        unsigned int xj_32 = x[11];
        v[10] = hi_32;
        v[11] = lo_32;
        x[10] = ((p_32 != 0) ? xi_32 : xj_32);
        x[11] = ((p_32 != 0) ? xj_32 : xi_32);
        if (p_32 == 0) {
            sw_2 = 1;
        }
        int p_33 = 0;
        if (v[12] >= v[13]) {
            p_33 = 1;
        }
        float hi_33 = fmaxf(v[12], v[13]);
        float lo_33 = fminf(v[12], v[13]);
        unsigned int xi_33 = x[12];
        unsigned int xj_33 = x[13];
        v[12] = hi_33;
        v[13] = lo_33;
        x[12] = ((p_33 != 0) ? xi_33 : xj_33);
        x[13] = ((p_33 != 0) ? xj_33 : xi_33);
        if (p_33 == 0) {
            sw_2 = 1;
        }
        int p_34 = 0;
        if (v[14] >= v[15]) {
            p_34 = 1;
        }
        float hi_34 = fmaxf(v[14], v[15]);
        float lo_34 = fminf(v[14], v[15]);
        unsigned int xi_34 = x[14];
        unsigned int xj_34 = x[15];
        v[14] = hi_34;
        v[15] = lo_34;
        x[14] = ((p_34 != 0) ? xi_34 : xj_34);
        x[15] = ((p_34 != 0) ? xj_34 : xi_34);
        if (p_34 == 0) {
            sw_2 = 1;
        }
        int p_35 = 0;
        if (v[1] >= v[2]) {
            p_35 = 1;
        }
        float hi_35 = fmaxf(v[1], v[2]);
        float lo_35 = fminf(v[1], v[2]);
        unsigned int xi_35 = x[1];
        unsigned int xj_35 = x[2];
        v[1] = hi_35;
        v[2] = lo_35;
        x[1] = ((p_35 != 0) ? xi_35 : xj_35);
        x[2] = ((p_35 != 0) ? xj_35 : xi_35);
        if (p_35 == 0) {
            sw_2 = 1;
        }
        int p_36 = 0;
        if (v[3] >= v[4]) {
            p_36 = 1;
        }
        float hi_36 = fmaxf(v[3], v[4]);
        float lo_36 = fminf(v[3], v[4]);
        unsigned int xi_36 = x[3];
        unsigned int xj_36 = x[4];
        v[3] = hi_36;
        v[4] = lo_36;
        x[3] = ((p_36 != 0) ? xi_36 : xj_36);
        x[4] = ((p_36 != 0) ? xj_36 : xi_36);
        if (p_36 == 0) {
            sw_2 = 1;
        }
        int p_37 = 0;
        if (v[5] >= v[6]) {
            p_37 = 1;
        }
        float hi_37 = fmaxf(v[5], v[6]);
        float lo_37 = fminf(v[5], v[6]);
        unsigned int xi_37 = x[5];
        unsigned int xj_37 = x[6];
        v[5] = hi_37;
        v[6] = lo_37;
        x[5] = ((p_37 != 0) ? xi_37 : xj_37);
        x[6] = ((p_37 != 0) ? xj_37 : xi_37);
        if (p_37 == 0) {
            sw_2 = 1;
        }
        int p_38 = 0;
        if (v[7] >= v[8]) {
            p_38 = 1;
        }
        float hi_38 = fmaxf(v[7], v[8]);
        float lo_38 = fminf(v[7], v[8]);
        unsigned int xi_38 = x[7];
        unsigned int xj_38 = x[8];
        v[7] = hi_38;
        v[8] = lo_38;
        x[7] = ((p_38 != 0) ? xi_38 : xj_38);
        x[8] = ((p_38 != 0) ? xj_38 : xi_38);
        if (p_38 == 0) {
            sw_2 = 1;
        }
        int p_39 = 0;
        if (v[9] >= v[10]) {
            p_39 = 1;
        }
        float hi_39 = fmaxf(v[9], v[10]);
        float lo_39 = fminf(v[9], v[10]);
        unsigned int xi_39 = x[9];
        unsigned int xj_39 = x[10];
        v[9] = hi_39;
        v[10] = lo_39;
        x[9] = ((p_39 != 0) ? xi_39 : xj_39);
        x[10] = ((p_39 != 0) ? xj_39 : xi_39);
        if (p_39 == 0) {
            sw_2 = 1;
        }
        int p_40 = 0;
        if (v[11] >= v[12]) {
            p_40 = 1;
        }
        float hi_40 = fmaxf(v[11], v[12]);
        float lo_40 = fminf(v[11], v[12]);
        unsigned int xi_40 = x[11];
        unsigned int xj_40 = x[12];
        v[11] = hi_40;
        v[12] = lo_40;
        x[11] = ((p_40 != 0) ? xi_40 : xj_40);
        x[12] = ((p_40 != 0) ? xj_40 : xi_40);
        if (p_40 == 0) {
            sw_2 = 1;
        }
        int p_41 = 0;
        if (v[13] >= v[14]) {
            p_41 = 1;
        }
        float hi_41 = fmaxf(v[13], v[14]);
        float lo_41 = fminf(v[13], v[14]);
        unsigned int xi_41 = x[13];
        unsigned int xj_41 = x[14];
        v[13] = hi_41;
        v[14] = lo_41;
        x[13] = ((p_41 != 0) ? xi_41 : xj_41);
        x[14] = ((p_41 != 0) ? xj_41 : xi_41);
        if (p_41 == 0) {
            sw_2 = 1;
        }
        int _warp_vote_2 = __any_sync(0xFFFFFFFF, sw_2 != 0);
        more = _warp_vote_2;
    }
    if (more != 0) {
        int p_42 = 0;
        if (v[0] >= v[13]) {
            p_42 = 1;
        }
        float hi_42 = fmaxf(v[0], v[13]);
        float lo_42 = fminf(v[0], v[13]);
        unsigned int xi_42 = x[0];
        unsigned int xj_42 = x[13];
        v[0] = hi_42;
        v[13] = lo_42;
        x[0] = ((p_42 != 0) ? xi_42 : xj_42);
        x[13] = ((p_42 != 0) ? xj_42 : xi_42);
        int p_42_1 = 0;
        if (v[1] >= v[12]) {
            p_42_1 = 1;
        }
        float hi_42_1 = fmaxf(v[1], v[12]);
        float lo_42_1 = fminf(v[1], v[12]);
        unsigned int xi_42_1 = x[1];
        unsigned int xj_42_1 = x[12];
        v[1] = hi_42_1;
        v[12] = lo_42_1;
        x[1] = ((p_42_1 != 0) ? xi_42_1 : xj_42_1);
        x[12] = ((p_42_1 != 0) ? xj_42_1 : xi_42_1);
        int p_43 = 0;
        if (v[2] >= v[15]) {
            p_43 = 1;
        }
        float hi_43 = fmaxf(v[2], v[15]);
        float lo_43 = fminf(v[2], v[15]);
        unsigned int xi_43 = x[2];
        unsigned int xj_43 = x[15];
        v[2] = hi_43;
        v[15] = lo_43;
        x[2] = ((p_43 != 0) ? xi_43 : xj_43);
        x[15] = ((p_43 != 0) ? xj_43 : xi_43);
        int p_44 = 0;
        if (v[3] >= v[14]) {
            p_44 = 1;
        }
        float hi_44 = fmaxf(v[3], v[14]);
        float lo_44 = fminf(v[3], v[14]);
        unsigned int xi_44 = x[3];
        unsigned int xj_44 = x[14];
        v[3] = hi_44;
        v[14] = lo_44;
        x[3] = ((p_44 != 0) ? xi_44 : xj_44);
        x[14] = ((p_44 != 0) ? xj_44 : xi_44);
        int p_45 = 0;
        if (v[4] >= v[8]) {
            p_45 = 1;
        }
        float hi_45 = fmaxf(v[4], v[8]);
        float lo_45 = fminf(v[4], v[8]);
        unsigned int xi_45 = x[4];
        unsigned int xj_45 = x[8];
        v[4] = hi_45;
        v[8] = lo_45;
        x[4] = ((p_45 != 0) ? xi_45 : xj_45);
        x[8] = ((p_45 != 0) ? xj_45 : xi_45);
        int p_46 = 0;
        if (v[5] >= v[6]) {
            p_46 = 1;
        }
        float hi_46 = fmaxf(v[5], v[6]);
        float lo_46 = fminf(v[5], v[6]);
        unsigned int xi_46 = x[5];
        unsigned int xj_46 = x[6];
        v[5] = hi_46;
        v[6] = lo_46;
        x[5] = ((p_46 != 0) ? xi_46 : xj_46);
        x[6] = ((p_46 != 0) ? xj_46 : xi_46);
        int p_47 = 0;
        if (v[7] >= v[11]) {
            p_47 = 1;
        }
        float hi_47 = fmaxf(v[7], v[11]);
        float lo_47 = fminf(v[7], v[11]);
        unsigned int xi_47 = x[7];
        unsigned int xj_47 = x[11];
        v[7] = hi_47;
        v[11] = lo_47;
        x[7] = ((p_47 != 0) ? xi_47 : xj_47);
        x[11] = ((p_47 != 0) ? xj_47 : xi_47);
        int p_48 = 0;
        if (v[9] >= v[10]) {
            p_48 = 1;
        }
        float hi_48 = fmaxf(v[9], v[10]);
        float lo_48 = fminf(v[9], v[10]);
        unsigned int xi_48 = x[9];
        unsigned int xj_48 = x[10];
        v[9] = hi_48;
        v[10] = lo_48;
        x[9] = ((p_48 != 0) ? xi_48 : xj_48);
        x[10] = ((p_48 != 0) ? xj_48 : xi_48);
        int p_49 = 0;
        if (v[0] >= v[5]) {
            p_49 = 1;
        }
        float hi_49 = fmaxf(v[0], v[5]);
        float lo_49 = fminf(v[0], v[5]);
        unsigned int xi_49 = x[0];
        unsigned int xj_49 = x[5];
        v[0] = hi_49;
        v[5] = lo_49;
        x[0] = ((p_49 != 0) ? xi_49 : xj_49);
        x[5] = ((p_49 != 0) ? xj_49 : xi_49);
        int p_50 = 0;
        if (v[1] >= v[7]) {
            p_50 = 1;
        }
        float hi_50 = fmaxf(v[1], v[7]);
        float lo_50 = fminf(v[1], v[7]);
        unsigned int xi_50 = x[1];
        unsigned int xj_50 = x[7];
        v[1] = hi_50;
        v[7] = lo_50;
        x[1] = ((p_50 != 0) ? xi_50 : xj_50);
        x[7] = ((p_50 != 0) ? xj_50 : xi_50);
        int p_51 = 0;
        if (v[2] >= v[9]) {
            p_51 = 1;
        }
        float hi_51 = fmaxf(v[2], v[9]);
        float lo_51 = fminf(v[2], v[9]);
        unsigned int xi_51 = x[2];
        unsigned int xj_51 = x[9];
        v[2] = hi_51;
        v[9] = lo_51;
        x[2] = ((p_51 != 0) ? xi_51 : xj_51);
        x[9] = ((p_51 != 0) ? xj_51 : xi_51);
        int p_52 = 0;
        if (v[3] >= v[4]) {
            p_52 = 1;
        }
        float hi_52 = fmaxf(v[3], v[4]);
        float lo_52 = fminf(v[3], v[4]);
        unsigned int xi_52 = x[3];
        unsigned int xj_52 = x[4];
        v[3] = hi_52;
        v[4] = lo_52;
        x[3] = ((p_52 != 0) ? xi_52 : xj_52);
        x[4] = ((p_52 != 0) ? xj_52 : xi_52);
        int p_53 = 0;
        if (v[6] >= v[13]) {
            p_53 = 1;
        }
        float hi_53 = fmaxf(v[6], v[13]);
        float lo_53 = fminf(v[6], v[13]);
        unsigned int xi_53 = x[6];
        unsigned int xj_53 = x[13];
        v[6] = hi_53;
        v[13] = lo_53;
        x[6] = ((p_53 != 0) ? xi_53 : xj_53);
        x[13] = ((p_53 != 0) ? xj_53 : xi_53);
        int p_54 = 0;
        if (v[8] >= v[14]) {
            p_54 = 1;
        }
        float hi_54 = fmaxf(v[8], v[14]);
        float lo_54 = fminf(v[8], v[14]);
        unsigned int xi_54 = x[8];
        unsigned int xj_54 = x[14];
        v[8] = hi_54;
        v[14] = lo_54;
        x[8] = ((p_54 != 0) ? xi_54 : xj_54);
        x[14] = ((p_54 != 0) ? xj_54 : xi_54);
        int p_55 = 0;
        if (v[10] >= v[15]) {
            p_55 = 1;
        }
        float hi_55 = fmaxf(v[10], v[15]);
        float lo_55 = fminf(v[10], v[15]);
        unsigned int xi_55 = x[10];
        unsigned int xj_55 = x[15];
        v[10] = hi_55;
        v[15] = lo_55;
        x[10] = ((p_55 != 0) ? xi_55 : xj_55);
        x[15] = ((p_55 != 0) ? xj_55 : xi_55);
        int p_56 = 0;
        if (v[11] >= v[12]) {
            p_56 = 1;
        }
        float hi_56 = fmaxf(v[11], v[12]);
        float lo_56 = fminf(v[11], v[12]);
        unsigned int xi_56 = x[11];
        unsigned int xj_56 = x[12];
        v[11] = hi_56;
        v[12] = lo_56;
        x[11] = ((p_56 != 0) ? xi_56 : xj_56);
        x[12] = ((p_56 != 0) ? xj_56 : xi_56);
        int p_57 = 0;
        if (v[0] >= v[1]) {
            p_57 = 1;
        }
        float hi_57 = fmaxf(v[0], v[1]);
        float lo_57 = fminf(v[0], v[1]);
        unsigned int xi_57 = x[0];
        unsigned int xj_57 = x[1];
        v[0] = hi_57;
        v[1] = lo_57;
        x[0] = ((p_57 != 0) ? xi_57 : xj_57);
        x[1] = ((p_57 != 0) ? xj_57 : xi_57);
        int p_58 = 0;
        if (v[2] >= v[3]) {
            p_58 = 1;
        }
        float hi_58 = fmaxf(v[2], v[3]);
        float lo_58 = fminf(v[2], v[3]);
        unsigned int xi_58 = x[2];
        unsigned int xj_58 = x[3];
        v[2] = hi_58;
        v[3] = lo_58;
        x[2] = ((p_58 != 0) ? xi_58 : xj_58);
        x[3] = ((p_58 != 0) ? xj_58 : xi_58);
        int p_59 = 0;
        if (v[4] >= v[5]) {
            p_59 = 1;
        }
        float hi_59 = fmaxf(v[4], v[5]);
        float lo_59 = fminf(v[4], v[5]);
        unsigned int xi_59 = x[4];
        unsigned int xj_59 = x[5];
        v[4] = hi_59;
        v[5] = lo_59;
        x[4] = ((p_59 != 0) ? xi_59 : xj_59);
        x[5] = ((p_59 != 0) ? xj_59 : xi_59);
        int p_60 = 0;
        if (v[6] >= v[8]) {
            p_60 = 1;
        }
        float hi_60 = fmaxf(v[6], v[8]);
        float lo_60 = fminf(v[6], v[8]);
        unsigned int xi_60 = x[6];
        unsigned int xj_60 = x[8];
        v[6] = hi_60;
        v[8] = lo_60;
        x[6] = ((p_60 != 0) ? xi_60 : xj_60);
        x[8] = ((p_60 != 0) ? xj_60 : xi_60);
        int p_61 = 0;
        if (v[7] >= v[9]) {
            p_61 = 1;
        }
        float hi_61 = fmaxf(v[7], v[9]);
        float lo_61 = fminf(v[7], v[9]);
        unsigned int xi_61 = x[7];
        unsigned int xj_61 = x[9];
        v[7] = hi_61;
        v[9] = lo_61;
        x[7] = ((p_61 != 0) ? xi_61 : xj_61);
        x[9] = ((p_61 != 0) ? xj_61 : xi_61);
        int p_62 = 0;
        if (v[10] >= v[11]) {
            p_62 = 1;
        }
        float hi_62 = fmaxf(v[10], v[11]);
        float lo_62 = fminf(v[10], v[11]);
        unsigned int xi_62 = x[10];
        unsigned int xj_62 = x[11];
        v[10] = hi_62;
        v[11] = lo_62;
        x[10] = ((p_62 != 0) ? xi_62 : xj_62);
        x[11] = ((p_62 != 0) ? xj_62 : xi_62);
        int p_63 = 0;
        if (v[12] >= v[13]) {
            p_63 = 1;
        }
        float hi_63 = fmaxf(v[12], v[13]);
        float lo_63 = fminf(v[12], v[13]);
        unsigned int xi_63 = x[12];
        unsigned int xj_63 = x[13];
        v[12] = hi_63;
        v[13] = lo_63;
        x[12] = ((p_63 != 0) ? xi_63 : xj_63);
        x[13] = ((p_63 != 0) ? xj_63 : xi_63);
        int p_64 = 0;
        if (v[14] >= v[15]) {
            p_64 = 1;
        }
        float hi_64 = fmaxf(v[14], v[15]);
        float lo_64 = fminf(v[14], v[15]);
        unsigned int xi_64 = x[14];
        unsigned int xj_64 = x[15];
        v[14] = hi_64;
        v[15] = lo_64;
        x[14] = ((p_64 != 0) ? xi_64 : xj_64);
        x[15] = ((p_64 != 0) ? xj_64 : xi_64);
        int p_65 = 0;
        if (v[0] >= v[2]) {
            p_65 = 1;
        }
        float hi_65 = fmaxf(v[0], v[2]);
        float lo_65 = fminf(v[0], v[2]);
        unsigned int xi_65 = x[0];
        unsigned int xj_65 = x[2];
        v[0] = hi_65;
        v[2] = lo_65;
        x[0] = ((p_65 != 0) ? xi_65 : xj_65);
        x[2] = ((p_65 != 0) ? xj_65 : xi_65);
        int p_66 = 0;
        if (v[1] >= v[3]) {
            p_66 = 1;
        }
        float hi_66 = fmaxf(v[1], v[3]);
        float lo_66 = fminf(v[1], v[3]);
        unsigned int xi_66 = x[1];
        unsigned int xj_66 = x[3];
        v[1] = hi_66;
        v[3] = lo_66;
        x[1] = ((p_66 != 0) ? xi_66 : xj_66);
        x[3] = ((p_66 != 0) ? xj_66 : xi_66);
        int p_67 = 0;
        if (v[4] >= v[10]) {
            p_67 = 1;
        }
        float hi_67 = fmaxf(v[4], v[10]);
        float lo_67 = fminf(v[4], v[10]);
        unsigned int xi_67 = x[4];
        unsigned int xj_67 = x[10];
        v[4] = hi_67;
        v[10] = lo_67;
        x[4] = ((p_67 != 0) ? xi_67 : xj_67);
        x[10] = ((p_67 != 0) ? xj_67 : xi_67);
        int p_68 = 0;
        if (v[5] >= v[11]) {
            p_68 = 1;
        }
        float hi_68 = fmaxf(v[5], v[11]);
        float lo_68 = fminf(v[5], v[11]);
        unsigned int xi_68 = x[5];
        unsigned int xj_68 = x[11];
        v[5] = hi_68;
        v[11] = lo_68;
        x[5] = ((p_68 != 0) ? xi_68 : xj_68);
        x[11] = ((p_68 != 0) ? xj_68 : xi_68);
        int p_69 = 0;
        if (v[6] >= v[7]) {
            p_69 = 1;
        }
        float hi_69 = fmaxf(v[6], v[7]);
        float lo_69 = fminf(v[6], v[7]);
        unsigned int xi_69 = x[6];
        unsigned int xj_69 = x[7];
        v[6] = hi_69;
        v[7] = lo_69;
        x[6] = ((p_69 != 0) ? xi_69 : xj_69);
        x[7] = ((p_69 != 0) ? xj_69 : xi_69);
        int p_70 = 0;
        if (v[8] >= v[9]) {
            p_70 = 1;
        }
        float hi_70 = fmaxf(v[8], v[9]);
        float lo_70 = fminf(v[8], v[9]);
        unsigned int xi_70 = x[8];
        unsigned int xj_70 = x[9];
        v[8] = hi_70;
        v[9] = lo_70;
        x[8] = ((p_70 != 0) ? xi_70 : xj_70);
        x[9] = ((p_70 != 0) ? xj_70 : xi_70);
        int p_71 = 0;
        if (v[12] >= v[14]) {
            p_71 = 1;
        }
        float hi_71 = fmaxf(v[12], v[14]);
        float lo_71 = fminf(v[12], v[14]);
        unsigned int xi_71 = x[12];
        unsigned int xj_71 = x[14];
        v[12] = hi_71;
        v[14] = lo_71;
        x[12] = ((p_71 != 0) ? xi_71 : xj_71);
        x[14] = ((p_71 != 0) ? xj_71 : xi_71);
        int p_72 = 0;
        if (v[13] >= v[15]) {
            p_72 = 1;
        }
        float hi_72 = fmaxf(v[13], v[15]);
        float lo_72 = fminf(v[13], v[15]);
        unsigned int xi_72 = x[13];
        unsigned int xj_72 = x[15];
        v[13] = hi_72;
        v[15] = lo_72;
        x[13] = ((p_72 != 0) ? xi_72 : xj_72);
        x[15] = ((p_72 != 0) ? xj_72 : xi_72);
        int p_73 = 0;
        if (v[1] >= v[2]) {
            p_73 = 1;
        }
        float hi_73 = fmaxf(v[1], v[2]);
        float lo_73 = fminf(v[1], v[2]);
        unsigned int xi_73 = x[1];
        unsigned int xj_73 = x[2];
        v[1] = hi_73;
        v[2] = lo_73;
        x[1] = ((p_73 != 0) ? xi_73 : xj_73);
        x[2] = ((p_73 != 0) ? xj_73 : xi_73);
        int p_74 = 0;
        if (v[3] >= v[12]) {
            p_74 = 1;
        }
        float hi_74 = fmaxf(v[3], v[12]);
        float lo_74 = fminf(v[3], v[12]);
        unsigned int xi_74 = x[3];
        unsigned int xj_74 = x[12];
        v[3] = hi_74;
        v[12] = lo_74;
        x[3] = ((p_74 != 0) ? xi_74 : xj_74);
        x[12] = ((p_74 != 0) ? xj_74 : xi_74);
        int p_75 = 0;
        if (v[4] >= v[6]) {
            p_75 = 1;
        }
        float hi_75 = fmaxf(v[4], v[6]);
        float lo_75 = fminf(v[4], v[6]);
        unsigned int xi_75 = x[4];
        unsigned int xj_75 = x[6];
        v[4] = hi_75;
        v[6] = lo_75;
        x[4] = ((p_75 != 0) ? xi_75 : xj_75);
        x[6] = ((p_75 != 0) ? xj_75 : xi_75);
        int p_76 = 0;
        if (v[5] >= v[7]) {
            p_76 = 1;
        }
        float hi_76 = fmaxf(v[5], v[7]);
        float lo_76 = fminf(v[5], v[7]);
        unsigned int xi_76 = x[5];
        unsigned int xj_76 = x[7];
        v[5] = hi_76;
        v[7] = lo_76;
        x[5] = ((p_76 != 0) ? xi_76 : xj_76);
        x[7] = ((p_76 != 0) ? xj_76 : xi_76);
        int p_77 = 0;
        if (v[8] >= v[10]) {
            p_77 = 1;
        }
        float hi_77 = fmaxf(v[8], v[10]);
        float lo_77 = fminf(v[8], v[10]);
        unsigned int xi_77 = x[8];
        unsigned int xj_77 = x[10];
        v[8] = hi_77;
        v[10] = lo_77;
        x[8] = ((p_77 != 0) ? xi_77 : xj_77);
        x[10] = ((p_77 != 0) ? xj_77 : xi_77);
        int p_78 = 0;
        if (v[9] >= v[11]) {
            p_78 = 1;
        }
        float hi_78 = fmaxf(v[9], v[11]);
        float lo_78 = fminf(v[9], v[11]);
        unsigned int xi_78 = x[9];
        unsigned int xj_78 = x[11];
        v[9] = hi_78;
        v[11] = lo_78;
        x[9] = ((p_78 != 0) ? xi_78 : xj_78);
        x[11] = ((p_78 != 0) ? xj_78 : xi_78);
        int p_79 = 0;
        if (v[13] >= v[14]) {
            p_79 = 1;
        }
        float hi_79 = fmaxf(v[13], v[14]);
        float lo_79 = fminf(v[13], v[14]);
        unsigned int xi_79 = x[13];
        unsigned int xj_79 = x[14];
        v[13] = hi_79;
        v[14] = lo_79;
        x[13] = ((p_79 != 0) ? xi_79 : xj_79);
        x[14] = ((p_79 != 0) ? xj_79 : xi_79);
        int p_80 = 0;
        if (v[1] >= v[4]) {
            p_80 = 1;
        }
        float hi_80 = fmaxf(v[1], v[4]);
        float lo_80 = fminf(v[1], v[4]);
        unsigned int xi_80 = x[1];
        unsigned int xj_80 = x[4];
        v[1] = hi_80;
        v[4] = lo_80;
        x[1] = ((p_80 != 0) ? xi_80 : xj_80);
        x[4] = ((p_80 != 0) ? xj_80 : xi_80);
        int p_81 = 0;
        if (v[2] >= v[6]) {
            p_81 = 1;
        }
        float hi_81 = fmaxf(v[2], v[6]);
        float lo_81 = fminf(v[2], v[6]);
        unsigned int xi_81 = x[2];
        unsigned int xj_81 = x[6];
        v[2] = hi_81;
        v[6] = lo_81;
        x[2] = ((p_81 != 0) ? xi_81 : xj_81);
        x[6] = ((p_81 != 0) ? xj_81 : xi_81);
        int p_82 = 0;
        if (v[5] >= v[8]) {
            p_82 = 1;
        }
        float hi_82 = fmaxf(v[5], v[8]);
        float lo_82 = fminf(v[5], v[8]);
        unsigned int xi_82 = x[5];
        unsigned int xj_82 = x[8];
        v[5] = hi_82;
        v[8] = lo_82;
        x[5] = ((p_82 != 0) ? xi_82 : xj_82);
        x[8] = ((p_82 != 0) ? xj_82 : xi_82);
        int p_83 = 0;
        if (v[7] >= v[10]) {
            p_83 = 1;
        }
        float hi_83 = fmaxf(v[7], v[10]);
        float lo_83 = fminf(v[7], v[10]);
        unsigned int xi_83 = x[7];
        unsigned int xj_83 = x[10];
        v[7] = hi_83;
        v[10] = lo_83;
        x[7] = ((p_83 != 0) ? xi_83 : xj_83);
        x[10] = ((p_83 != 0) ? xj_83 : xi_83);
        int p_84 = 0;
        if (v[9] >= v[13]) {
            p_84 = 1;
        }
        float hi_84 = fmaxf(v[9], v[13]);
        float lo_84 = fminf(v[9], v[13]);
        unsigned int xi_84 = x[9];
        unsigned int xj_84 = x[13];
        v[9] = hi_84;
        v[13] = lo_84;
        x[9] = ((p_84 != 0) ? xi_84 : xj_84);
        x[13] = ((p_84 != 0) ? xj_84 : xi_84);
        int p_85 = 0;
        if (v[11] >= v[14]) {
            p_85 = 1;
        }
        float hi_85 = fmaxf(v[11], v[14]);
        float lo_85 = fminf(v[11], v[14]);
        unsigned int xi_85 = x[11];
        unsigned int xj_85 = x[14];
        v[11] = hi_85;
        v[14] = lo_85;
        x[11] = ((p_85 != 0) ? xi_85 : xj_85);
        x[14] = ((p_85 != 0) ? xj_85 : xi_85);
        int p_86 = 0;
        if (v[2] >= v[4]) {
            p_86 = 1;
        }
        float hi_86 = fmaxf(v[2], v[4]);
        float lo_86 = fminf(v[2], v[4]);
        unsigned int xi_86 = x[2];
        unsigned int xj_86 = x[4];
        v[2] = hi_86;
        v[4] = lo_86;
        x[2] = ((p_86 != 0) ? xi_86 : xj_86);
        x[4] = ((p_86 != 0) ? xj_86 : xi_86);
        int p_87 = 0;
        if (v[3] >= v[6]) {
            p_87 = 1;
        }
        float hi_87 = fmaxf(v[3], v[6]);
        float lo_87 = fminf(v[3], v[6]);
        unsigned int xi_87 = x[3];
        unsigned int xj_87 = x[6];
        v[3] = hi_87;
        v[6] = lo_87;
        x[3] = ((p_87 != 0) ? xi_87 : xj_87);
        x[6] = ((p_87 != 0) ? xj_87 : xi_87);
        int p_88 = 0;
        if (v[9] >= v[12]) {
            p_88 = 1;
        }
        float hi_88 = fmaxf(v[9], v[12]);
        float lo_88 = fminf(v[9], v[12]);
        unsigned int xi_88 = x[9];
        unsigned int xj_88 = x[12];
        v[9] = hi_88;
        v[12] = lo_88;
        x[9] = ((p_88 != 0) ? xi_88 : xj_88);
        x[12] = ((p_88 != 0) ? xj_88 : xi_88);
        int p_89 = 0;
        if (v[11] >= v[13]) {
            p_89 = 1;
        }
        float hi_89 = fmaxf(v[11], v[13]);
        float lo_89 = fminf(v[11], v[13]);
        unsigned int xi_89 = x[11];
        unsigned int xj_89 = x[13];
        v[11] = hi_89;
        v[13] = lo_89;
        x[11] = ((p_89 != 0) ? xi_89 : xj_89);
        x[13] = ((p_89 != 0) ? xj_89 : xi_89);
        int p_90 = 0;
        if (v[3] >= v[5]) {
            p_90 = 1;
        }
        float hi_90 = fmaxf(v[3], v[5]);
        float lo_90 = fminf(v[3], v[5]);
        unsigned int xi_90 = x[3];
        unsigned int xj_90 = x[5];
        v[3] = hi_90;
        v[5] = lo_90;
        x[3] = ((p_90 != 0) ? xi_90 : xj_90);
        x[5] = ((p_90 != 0) ? xj_90 : xi_90);
        int p_91 = 0;
        if (v[6] >= v[8]) {
            p_91 = 1;
        }
        float hi_91 = fmaxf(v[6], v[8]);
        float lo_91 = fminf(v[6], v[8]);
        unsigned int xi_91 = x[6];
        unsigned int xj_91 = x[8];
        v[6] = hi_91;
        v[8] = lo_91;
        x[6] = ((p_91 != 0) ? xi_91 : xj_91);
        x[8] = ((p_91 != 0) ? xj_91 : xi_91);
        int p_92 = 0;
        if (v[7] >= v[9]) {
            p_92 = 1;
        }
        float hi_92 = fmaxf(v[7], v[9]);
        float lo_92 = fminf(v[7], v[9]);
        unsigned int xi_92 = x[7];
        unsigned int xj_92 = x[9];
        v[7] = hi_92;
        v[9] = lo_92;
        x[7] = ((p_92 != 0) ? xi_92 : xj_92);
        x[9] = ((p_92 != 0) ? xj_92 : xi_92);
        int p_93 = 0;
        if (v[10] >= v[12]) {
            p_93 = 1;
        }
        float hi_93 = fmaxf(v[10], v[12]);
        float lo_93 = fminf(v[10], v[12]);
        unsigned int xi_93 = x[10];
        unsigned int xj_93 = x[12];
        v[10] = hi_93;
        v[12] = lo_93;
        x[10] = ((p_93 != 0) ? xi_93 : xj_93);
        x[12] = ((p_93 != 0) ? xj_93 : xi_93);
        int p_94 = 0;
        if (v[3] >= v[4]) {
            p_94 = 1;
        }
        float hi_94 = fmaxf(v[3], v[4]);
        float lo_94 = fminf(v[3], v[4]);
        unsigned int xi_94 = x[3];
        unsigned int xj_94 = x[4];
        v[3] = hi_94;
        v[4] = lo_94;
        x[3] = ((p_94 != 0) ? xi_94 : xj_94);
        x[4] = ((p_94 != 0) ? xj_94 : xi_94);
        int p_95 = 0;
        if (v[5] >= v[6]) {
            p_95 = 1;
        }
        float hi_95 = fmaxf(v[5], v[6]);
        float lo_95 = fminf(v[5], v[6]);
        unsigned int xi_95 = x[5];
        unsigned int xj_95 = x[6];
        v[5] = hi_95;
        v[6] = lo_95;
        x[5] = ((p_95 != 0) ? xi_95 : xj_95);
        x[6] = ((p_95 != 0) ? xj_95 : xi_95);
        int p_96 = 0;
        if (v[7] >= v[8]) {
            p_96 = 1;
        }
        float hi_96 = fmaxf(v[7], v[8]);
        float lo_96 = fminf(v[7], v[8]);
        unsigned int xi_96 = x[7];
        unsigned int xj_96 = x[8];
        v[7] = hi_96;
        v[8] = lo_96;
        x[7] = ((p_96 != 0) ? xi_96 : xj_96);
        x[8] = ((p_96 != 0) ? xj_96 : xi_96);
        int p_97 = 0;
        if (v[9] >= v[10]) {
            p_97 = 1;
        }
        float hi_97 = fmaxf(v[9], v[10]);
        float lo_97 = fminf(v[9], v[10]);
        unsigned int xi_97 = x[9];
        unsigned int xj_97 = x[10];
        v[9] = hi_97;
        v[10] = lo_97;
        x[9] = ((p_97 != 0) ? xi_97 : xj_97);
        x[10] = ((p_97 != 0) ? xj_97 : xi_97);
        int p_98 = 0;
        if (v[11] >= v[12]) {
            p_98 = 1;
        }
        float hi_98 = fmaxf(v[11], v[12]);
        float lo_98 = fminf(v[11], v[12]);
        unsigned int xi_98 = x[11];
        unsigned int xj_98 = x[12];
        v[11] = hi_98;
        v[12] = lo_98;
        x[11] = ((p_98 != 0) ? xi_98 : xj_98);
        x[12] = ((p_98 != 0) ? xj_98 : xi_98);
        int p_99 = 0;
        if (v[6] >= v[7]) {
            p_99 = 1;
        }
        float hi_99 = fmaxf(v[6], v[7]);
        float lo_99 = fminf(v[6], v[7]);
        unsigned int xi_99 = x[6];
        unsigned int xj_99 = x[7];
        v[6] = hi_99;
        v[7] = lo_99;
        x[6] = ((p_99 != 0) ? xi_99 : xj_99);
        x[7] = ((p_99 != 0) ? xj_99 : xi_99);
        int p_100 = 0;
        if (v[8] >= v[9]) {
            p_100 = 1;
        }
        float hi_100 = fmaxf(v[8], v[9]);
        float lo_100 = fminf(v[8], v[9]);
        unsigned int xi_100 = x[8];
        unsigned int xj_100 = x[9];
        v[8] = hi_100;
        v[9] = lo_100;
        x[8] = ((p_100 != 0) ? xi_100 : xj_100);
        x[9] = ((p_100 != 0) ? xj_100 : xi_100);
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
    int lv_0 = sgx * 16 + 8;
    unsigned int nw_0 = nib[lv_0 * 2 + (lnrx >> 3)];
    unsigned int sh_0 = (unsigned int)((lnrx & 7) << 2);
    int pos_0 = (int)(nw_0 >> sh_0 & 15);
    PV[0] = scr[lv_0 * 17 + pos_0];
    PX[0] = (unsigned int)(lv_0 * wm_r + pos_0 * ts_r);
    int p_101 = 0;
    if (V[0] >= PV[0]) {
        p_101 = 1;
    }
    V[0] = fmaxf(V[0], PV[0]);
    X[0] = ((p_101 != 0) ? X[0] : PX[0]);
    float wv_1_0 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_1_0 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_101 = fmaxf(V[0], wv_1_0);
    float lo_101 = fminf(V[0], wv_1_0);
    float vn_1_0 = ((upx[0] != 0) ? hi_101 : lo_101);
    X[0] = ((V[0] == vn_1_0) ? X[0] : wx_1_0);
    V[0] = vn_1_0;
    float wv_1_1 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_1_1 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_102 = fmaxf(V[0], wv_1_1);
    float lo_102 = fminf(V[0], wv_1_1);
    float vn_1_1 = ((upx[1] != 0) ? hi_102 : lo_102);
    X[0] = ((V[0] == vn_1_1) ? X[0] : wx_1_1);
    V[0] = vn_1_1;
    float wv_1_2 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_1_2 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_103 = fmaxf(V[0], wv_1_2);
    float lo_103 = fminf(V[0], wv_1_2);
    float vn_1_2 = ((upx[2] != 0) ? hi_103 : lo_103);
    X[0] = ((V[0] == vn_1_2) ? X[0] : wx_1_2);
    V[0] = vn_1_2;
    float wv_1_3 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_1_3 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_104 = fmaxf(V[0], wv_1_3);
    float lo_104 = fminf(V[0], wv_1_3);
    float vn_1_3 = ((upx[3] != 0) ? hi_104 : lo_104);
    X[0] = ((V[0] == vn_1_3) ? X[0] : wx_1_3);
    V[0] = vn_1_3;
    int lv_1 = sgx * 16 + 1;
    unsigned int nw_1 = nib[lv_1 * 2 + (lnx >> 3)];
    unsigned int sh_1 = (unsigned int)((lnx & 7) << 2);
    int pos_1 = (int)(nw_1 >> sh_1 & 15);
    V[1] = scr[lv_1 * 17 + pos_1];
    X[1] = (unsigned int)(lv_1 * wm_r + pos_1 * ts_r);
    int lv_2 = sgx * 16 + 1 + 8;
    unsigned int nw_2 = nib[lv_2 * 2 + (lnrx >> 3)];
    unsigned int sh_2 = (unsigned int)((lnrx & 7) << 2);
    int pos_2 = (int)(nw_2 >> sh_2 & 15);
    PV[1] = scr[lv_2 * 17 + pos_2];
    PX[1] = (unsigned int)(lv_2 * wm_r + pos_2 * ts_r);
    int p_101_1 = 0;
    if (V[1] >= PV[1]) {
        p_101_1 = 1;
    }
    V[1] = fmaxf(V[1], PV[1]);
    X[1] = ((p_101_1 != 0) ? X[1] : PX[1]);
    float wv_0 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_0 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_105 = fmaxf(V[1], wv_0);
    float lo_105 = fminf(V[1], wv_0);
    float vn_0 = ((upx[0] != 0) ? hi_105 : lo_105);
    X[1] = ((V[1] == vn_0) ? X[1] : wx_0);
    V[1] = vn_0;
    float wv_1 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_1 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_106 = fmaxf(V[1], wv_1);
    float lo_106 = fminf(V[1], wv_1);
    float vn_1 = ((upx[1] != 0) ? hi_106 : lo_106);
    X[1] = ((V[1] == vn_1) ? X[1] : wx_1);
    V[1] = vn_1;
    float wv_2 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_2 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_107 = fmaxf(V[1], wv_2);
    float lo_107 = fminf(V[1], wv_2);
    float vn_2 = ((upx[2] != 0) ? hi_107 : lo_107);
    X[1] = ((V[1] == vn_2) ? X[1] : wx_2);
    V[1] = vn_2;
    float wv_3 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_3 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_108 = fmaxf(V[1], wv_3);
    float lo_108 = fminf(V[1], wv_3);
    float vn_3 = ((upx[3] != 0) ? hi_108 : lo_108);
    X[1] = ((V[1] == vn_3) ? X[1] : wx_3);
    V[1] = vn_3;
    int lv_3 = sgx * 16 + 2;
    unsigned int nw_3 = nib[lv_3 * 2 + (lnx >> 3)];
    unsigned int sh_3 = (unsigned int)((lnx & 7) << 2);
    int pos_3 = (int)(nw_3 >> sh_3 & 15);
    V[2] = scr[lv_3 * 17 + pos_3];
    X[2] = (unsigned int)(lv_3 * wm_r + pos_3 * ts_r);
    int lv_4 = sgx * 16 + 2 + 8;
    unsigned int nw_4 = nib[lv_4 * 2 + (lnrx >> 3)];
    unsigned int sh_4 = (unsigned int)((lnrx & 7) << 2);
    int pos_4 = (int)(nw_4 >> sh_4 & 15);
    PV[2] = scr[lv_4 * 17 + pos_4];
    PX[2] = (unsigned int)(lv_4 * wm_r + pos_4 * ts_r);
    int p_102 = 0;
    if (V[2] >= PV[2]) {
        p_102 = 1;
    }
    V[2] = fmaxf(V[2], PV[2]);
    X[2] = ((p_102 != 0) ? X[2] : PX[2]);
    float wv_4 = __shfl_xor_sync(0xFFFFFFFF, V[2], 8);
    unsigned int wx_4 = __shfl_xor_sync(0xFFFFFFFF, X[2], 8);
    float hi_109 = fmaxf(V[2], wv_4);
    float lo_109 = fminf(V[2], wv_4);
    float vn_4 = ((upx[0] != 0) ? hi_109 : lo_109);
    X[2] = ((V[2] == vn_4) ? X[2] : wx_4);
    V[2] = vn_4;
    float wv_5 = __shfl_xor_sync(0xFFFFFFFF, V[2], 4);
    unsigned int wx_5 = __shfl_xor_sync(0xFFFFFFFF, X[2], 4);
    float hi_110 = fmaxf(V[2], wv_5);
    float lo_110 = fminf(V[2], wv_5);
    float vn_5 = ((upx[1] != 0) ? hi_110 : lo_110);
    X[2] = ((V[2] == vn_5) ? X[2] : wx_5);
    V[2] = vn_5;
    float wv_6 = __shfl_xor_sync(0xFFFFFFFF, V[2], 2);
    unsigned int wx_6 = __shfl_xor_sync(0xFFFFFFFF, X[2], 2);
    float hi_111 = fmaxf(V[2], wv_6);
    float lo_111 = fminf(V[2], wv_6);
    float vn_6 = ((upx[2] != 0) ? hi_111 : lo_111);
    X[2] = ((V[2] == vn_6) ? X[2] : wx_6);
    V[2] = vn_6;
    float wv_7 = __shfl_xor_sync(0xFFFFFFFF, V[2], 1);
    unsigned int wx_7 = __shfl_xor_sync(0xFFFFFFFF, X[2], 1);
    float hi_112 = fmaxf(V[2], wv_7);
    float lo_112 = fminf(V[2], wv_7);
    float vn_7 = ((upx[3] != 0) ? hi_112 : lo_112);
    X[2] = ((V[2] == vn_7) ? X[2] : wx_7);
    V[2] = vn_7;
    int lv_5 = sgx * 16 + 3;
    unsigned int nw_5 = nib[lv_5 * 2 + (lnx >> 3)];
    unsigned int sh_5 = (unsigned int)((lnx & 7) << 2);
    int pos_5 = (int)(nw_5 >> sh_5 & 15);
    V[3] = scr[lv_5 * 17 + pos_5];
    X[3] = (unsigned int)(lv_5 * wm_r + pos_5 * ts_r);
    int lv_6 = sgx * 16 + 3 + 8;
    unsigned int nw_6 = nib[lv_6 * 2 + (lnrx >> 3)];
    unsigned int sh_6 = (unsigned int)((lnrx & 7) << 2);
    int pos_6 = (int)(nw_6 >> sh_6 & 15);
    PV[3] = scr[lv_6 * 17 + pos_6];
    PX[3] = (unsigned int)(lv_6 * wm_r + pos_6 * ts_r);
    int p_103 = 0;
    if (V[3] >= PV[3]) {
        p_103 = 1;
    }
    V[3] = fmaxf(V[3], PV[3]);
    X[3] = ((p_103 != 0) ? X[3] : PX[3]);
    float wv_8 = __shfl_xor_sync(0xFFFFFFFF, V[3], 8);
    unsigned int wx_8 = __shfl_xor_sync(0xFFFFFFFF, X[3], 8);
    float hi_113 = fmaxf(V[3], wv_8);
    float lo_113 = fminf(V[3], wv_8);
    float vn_8 = ((upx[0] != 0) ? hi_113 : lo_113);
    X[3] = ((V[3] == vn_8) ? X[3] : wx_8);
    V[3] = vn_8;
    float wv_9 = __shfl_xor_sync(0xFFFFFFFF, V[3], 4);
    unsigned int wx_9 = __shfl_xor_sync(0xFFFFFFFF, X[3], 4);
    float hi_114 = fmaxf(V[3], wv_9);
    float lo_114 = fminf(V[3], wv_9);
    float vn_9 = ((upx[1] != 0) ? hi_114 : lo_114);
    X[3] = ((V[3] == vn_9) ? X[3] : wx_9);
    V[3] = vn_9;
    float wv_10 = __shfl_xor_sync(0xFFFFFFFF, V[3], 2);
    unsigned int wx_10 = __shfl_xor_sync(0xFFFFFFFF, X[3], 2);
    float hi_115 = fmaxf(V[3], wv_10);
    float lo_115 = fminf(V[3], wv_10);
    float vn_10 = ((upx[2] != 0) ? hi_115 : lo_115);
    X[3] = ((V[3] == vn_10) ? X[3] : wx_10);
    V[3] = vn_10;
    float wv_11 = __shfl_xor_sync(0xFFFFFFFF, V[3], 1);
    unsigned int wx_11 = __shfl_xor_sync(0xFFFFFFFF, X[3], 1);
    float hi_116 = fmaxf(V[3], wv_11);
    float lo_116 = fminf(V[3], wv_11);
    float vn_11 = ((upx[3] != 0) ? hi_116 : lo_116);
    X[3] = ((V[3] == vn_11) ? X[3] : wx_11);
    V[3] = vn_11;
    int lv_7 = sgx * 16 + 4;
    unsigned int nw_7 = nib[lv_7 * 2 + (lnx >> 3)];
    unsigned int sh_7 = (unsigned int)((lnx & 7) << 2);
    int pos_7 = (int)(nw_7 >> sh_7 & 15);
    V[4] = scr[lv_7 * 17 + pos_7];
    X[4] = (unsigned int)(lv_7 * wm_r + pos_7 * ts_r);
    int lv_8 = sgx * 16 + 4 + 8;
    unsigned int nw_8 = nib[lv_8 * 2 + (lnrx >> 3)];
    unsigned int sh_8 = (unsigned int)((lnrx & 7) << 2);
    int pos_8 = (int)(nw_8 >> sh_8 & 15);
    PV[4] = scr[lv_8 * 17 + pos_8];
    PX[4] = (unsigned int)(lv_8 * wm_r + pos_8 * ts_r);
    int p_104 = 0;
    if (V[4] >= PV[4]) {
        p_104 = 1;
    }
    V[4] = fmaxf(V[4], PV[4]);
    X[4] = ((p_104 != 0) ? X[4] : PX[4]);
    float wv_12 = __shfl_xor_sync(0xFFFFFFFF, V[4], 8);
    unsigned int wx_12 = __shfl_xor_sync(0xFFFFFFFF, X[4], 8);
    float hi_117 = fmaxf(V[4], wv_12);
    float lo_117 = fminf(V[4], wv_12);
    float vn_12 = ((upx[0] != 0) ? hi_117 : lo_117);
    X[4] = ((V[4] == vn_12) ? X[4] : wx_12);
    V[4] = vn_12;
    float wv_13 = __shfl_xor_sync(0xFFFFFFFF, V[4], 4);
    unsigned int wx_13 = __shfl_xor_sync(0xFFFFFFFF, X[4], 4);
    float hi_118 = fmaxf(V[4], wv_13);
    float lo_118 = fminf(V[4], wv_13);
    float vn_13 = ((upx[1] != 0) ? hi_118 : lo_118);
    X[4] = ((V[4] == vn_13) ? X[4] : wx_13);
    V[4] = vn_13;
    float wv_14 = __shfl_xor_sync(0xFFFFFFFF, V[4], 2);
    unsigned int wx_14 = __shfl_xor_sync(0xFFFFFFFF, X[4], 2);
    float hi_119 = fmaxf(V[4], wv_14);
    float lo_119 = fminf(V[4], wv_14);
    float vn_14 = ((upx[2] != 0) ? hi_119 : lo_119);
    X[4] = ((V[4] == vn_14) ? X[4] : wx_14);
    V[4] = vn_14;
    float wv_15 = __shfl_xor_sync(0xFFFFFFFF, V[4], 1);
    unsigned int wx_15 = __shfl_xor_sync(0xFFFFFFFF, X[4], 1);
    float hi_120 = fmaxf(V[4], wv_15);
    float lo_120 = fminf(V[4], wv_15);
    float vn_15 = ((upx[3] != 0) ? hi_120 : lo_120);
    X[4] = ((V[4] == vn_15) ? X[4] : wx_15);
    V[4] = vn_15;
    int lv_9 = sgx * 16 + 5;
    unsigned int nw_9 = nib[lv_9 * 2 + (lnx >> 3)];
    unsigned int sh_9 = (unsigned int)((lnx & 7) << 2);
    int pos_9 = (int)(nw_9 >> sh_9 & 15);
    V[5] = scr[lv_9 * 17 + pos_9];
    X[5] = (unsigned int)(lv_9 * wm_r + pos_9 * ts_r);
    int lv_10 = sgx * 16 + 5 + 8;
    unsigned int nw_10 = nib[lv_10 * 2 + (lnrx >> 3)];
    unsigned int sh_10 = (unsigned int)((lnrx & 7) << 2);
    int pos_10 = (int)(nw_10 >> sh_10 & 15);
    PV[5] = scr[lv_10 * 17 + pos_10];
    PX[5] = (unsigned int)(lv_10 * wm_r + pos_10 * ts_r);
    int p_105 = 0;
    if (V[5] >= PV[5]) {
        p_105 = 1;
    }
    V[5] = fmaxf(V[5], PV[5]);
    X[5] = ((p_105 != 0) ? X[5] : PX[5]);
    float wv_16 = __shfl_xor_sync(0xFFFFFFFF, V[5], 8);
    unsigned int wx_16 = __shfl_xor_sync(0xFFFFFFFF, X[5], 8);
    float hi_121 = fmaxf(V[5], wv_16);
    float lo_121 = fminf(V[5], wv_16);
    float vn_16 = ((upx[0] != 0) ? hi_121 : lo_121);
    X[5] = ((V[5] == vn_16) ? X[5] : wx_16);
    V[5] = vn_16;
    float wv_17 = __shfl_xor_sync(0xFFFFFFFF, V[5], 4);
    unsigned int wx_17 = __shfl_xor_sync(0xFFFFFFFF, X[5], 4);
    float hi_122 = fmaxf(V[5], wv_17);
    float lo_122 = fminf(V[5], wv_17);
    float vn_17 = ((upx[1] != 0) ? hi_122 : lo_122);
    X[5] = ((V[5] == vn_17) ? X[5] : wx_17);
    V[5] = vn_17;
    float wv_18 = __shfl_xor_sync(0xFFFFFFFF, V[5], 2);
    unsigned int wx_18 = __shfl_xor_sync(0xFFFFFFFF, X[5], 2);
    float hi_123 = fmaxf(V[5], wv_18);
    float lo_123 = fminf(V[5], wv_18);
    float vn_18 = ((upx[2] != 0) ? hi_123 : lo_123);
    X[5] = ((V[5] == vn_18) ? X[5] : wx_18);
    V[5] = vn_18;
    float wv_19 = __shfl_xor_sync(0xFFFFFFFF, V[5], 1);
    unsigned int wx_19 = __shfl_xor_sync(0xFFFFFFFF, X[5], 1);
    float hi_124 = fmaxf(V[5], wv_19);
    float lo_124 = fminf(V[5], wv_19);
    float vn_19 = ((upx[3] != 0) ? hi_124 : lo_124);
    X[5] = ((V[5] == vn_19) ? X[5] : wx_19);
    V[5] = vn_19;
    int lv_11 = sgx * 16 + 6;
    unsigned int nw_11 = nib[lv_11 * 2 + (lnx >> 3)];
    unsigned int sh_11 = (unsigned int)((lnx & 7) << 2);
    int pos_11 = (int)(nw_11 >> sh_11 & 15);
    V[6] = scr[lv_11 * 17 + pos_11];
    X[6] = (unsigned int)(lv_11 * wm_r + pos_11 * ts_r);
    int lv_12 = sgx * 16 + 6 + 8;
    unsigned int nw_12 = nib[lv_12 * 2 + (lnrx >> 3)];
    unsigned int sh_12 = (unsigned int)((lnrx & 7) << 2);
    int pos_12 = (int)(nw_12 >> sh_12 & 15);
    PV[6] = scr[lv_12 * 17 + pos_12];
    PX[6] = (unsigned int)(lv_12 * wm_r + pos_12 * ts_r);
    int p_106 = 0;
    if (V[6] >= PV[6]) {
        p_106 = 1;
    }
    V[6] = fmaxf(V[6], PV[6]);
    X[6] = ((p_106 != 0) ? X[6] : PX[6]);
    float wv_20 = __shfl_xor_sync(0xFFFFFFFF, V[6], 8);
    unsigned int wx_20 = __shfl_xor_sync(0xFFFFFFFF, X[6], 8);
    float hi_125 = fmaxf(V[6], wv_20);
    float lo_125 = fminf(V[6], wv_20);
    float vn_20 = ((upx[0] != 0) ? hi_125 : lo_125);
    X[6] = ((V[6] == vn_20) ? X[6] : wx_20);
    V[6] = vn_20;
    float wv_21 = __shfl_xor_sync(0xFFFFFFFF, V[6], 4);
    unsigned int wx_21 = __shfl_xor_sync(0xFFFFFFFF, X[6], 4);
    float hi_126 = fmaxf(V[6], wv_21);
    float lo_126 = fminf(V[6], wv_21);
    float vn_21 = ((upx[1] != 0) ? hi_126 : lo_126);
    X[6] = ((V[6] == vn_21) ? X[6] : wx_21);
    V[6] = vn_21;
    float wv_22 = __shfl_xor_sync(0xFFFFFFFF, V[6], 2);
    unsigned int wx_22 = __shfl_xor_sync(0xFFFFFFFF, X[6], 2);
    float hi_127 = fmaxf(V[6], wv_22);
    float lo_127 = fminf(V[6], wv_22);
    float vn_22 = ((upx[2] != 0) ? hi_127 : lo_127);
    X[6] = ((V[6] == vn_22) ? X[6] : wx_22);
    V[6] = vn_22;
    float wv_23 = __shfl_xor_sync(0xFFFFFFFF, V[6], 1);
    unsigned int wx_23 = __shfl_xor_sync(0xFFFFFFFF, X[6], 1);
    float hi_128 = fmaxf(V[6], wv_23);
    float lo_128 = fminf(V[6], wv_23);
    float vn_23 = ((upx[3] != 0) ? hi_128 : lo_128);
    X[6] = ((V[6] == vn_23) ? X[6] : wx_23);
    V[6] = vn_23;
    int lv_13 = sgx * 16 + 7;
    unsigned int nw_13 = nib[lv_13 * 2 + (lnx >> 3)];
    unsigned int sh_13 = (unsigned int)((lnx & 7) << 2);
    int pos_13 = (int)(nw_13 >> sh_13 & 15);
    V[7] = scr[lv_13 * 17 + pos_13];
    X[7] = (unsigned int)(lv_13 * wm_r + pos_13 * ts_r);
    int lv_14 = sgx * 16 + 7 + 8;
    unsigned int nw_14 = nib[lv_14 * 2 + (lnrx >> 3)];
    unsigned int sh_14 = (unsigned int)((lnrx & 7) << 2);
    int pos_14 = (int)(nw_14 >> sh_14 & 15);
    PV[7] = scr[lv_14 * 17 + pos_14];
    PX[7] = (unsigned int)(lv_14 * wm_r + pos_14 * ts_r);
    int p_107 = 0;
    if (V[7] >= PV[7]) {
        p_107 = 1;
    }
    V[7] = fmaxf(V[7], PV[7]);
    X[7] = ((p_107 != 0) ? X[7] : PX[7]);
    float wv_24 = __shfl_xor_sync(0xFFFFFFFF, V[7], 8);
    unsigned int wx_24 = __shfl_xor_sync(0xFFFFFFFF, X[7], 8);
    float hi_129 = fmaxf(V[7], wv_24);
    float lo_129 = fminf(V[7], wv_24);
    float vn_24 = ((upx[0] != 0) ? hi_129 : lo_129);
    X[7] = ((V[7] == vn_24) ? X[7] : wx_24);
    V[7] = vn_24;
    float wv_25 = __shfl_xor_sync(0xFFFFFFFF, V[7], 4);
    unsigned int wx_25 = __shfl_xor_sync(0xFFFFFFFF, X[7], 4);
    float hi_130 = fmaxf(V[7], wv_25);
    float lo_130 = fminf(V[7], wv_25);
    float vn_25 = ((upx[1] != 0) ? hi_130 : lo_130);
    X[7] = ((V[7] == vn_25) ? X[7] : wx_25);
    V[7] = vn_25;
    float wv_26 = __shfl_xor_sync(0xFFFFFFFF, V[7], 2);
    unsigned int wx_26 = __shfl_xor_sync(0xFFFFFFFF, X[7], 2);
    float hi_131 = fmaxf(V[7], wv_26);
    float lo_131 = fminf(V[7], wv_26);
    float vn_26 = ((upx[2] != 0) ? hi_131 : lo_131);
    X[7] = ((V[7] == vn_26) ? X[7] : wx_26);
    V[7] = vn_26;
    float wv_27 = __shfl_xor_sync(0xFFFFFFFF, V[7], 1);
    unsigned int wx_27 = __shfl_xor_sync(0xFFFFFFFF, X[7], 1);
    float hi_132 = fmaxf(V[7], wv_27);
    float lo_132 = fminf(V[7], wv_27);
    float vn_27 = ((upx[3] != 0) ? hi_132 : lo_132);
    X[7] = ((V[7] == vn_27) ? X[7] : wx_27);
    V[7] = vn_27;
    float wv1 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    unsigned int wx1 = __shfl_xor_sync(0xFFFFFFFF, X[4], 15);
    int p_108 = 0;
    if (V[0] >= wv1) {
        p_108 = 1;
    }
    V[0] = fmaxf(V[0], wv1);
    X[0] = ((p_108 != 0) ? X[0] : wx1);
    float wv_28 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_28 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_133 = fmaxf(V[0], wv_28);
    float lo_133 = fminf(V[0], wv_28);
    float vn_28 = ((upx[0] != 0) ? hi_133 : lo_133);
    X[0] = ((V[0] == vn_28) ? X[0] : wx_28);
    V[0] = vn_28;
    float wv_29 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_29 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_134 = fmaxf(V[0], wv_29);
    float lo_134 = fminf(V[0], wv_29);
    float vn_29 = ((upx[1] != 0) ? hi_134 : lo_134);
    X[0] = ((V[0] == vn_29) ? X[0] : wx_29);
    V[0] = vn_29;
    float wv_30 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_30 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_135 = fmaxf(V[0], wv_30);
    float lo_135 = fminf(V[0], wv_30);
    float vn_30 = ((upx[2] != 0) ? hi_135 : lo_135);
    X[0] = ((V[0] == vn_30) ? X[0] : wx_30);
    V[0] = vn_30;
    float wv_31 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_31 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_136 = fmaxf(V[0], wv_31);
    float lo_136 = fminf(V[0], wv_31);
    float vn_31 = ((upx[3] != 0) ? hi_136 : lo_136);
    X[0] = ((V[0] == vn_31) ? X[0] : wx_31);
    V[0] = vn_31;
    float wv1_438 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    unsigned int wx1_439 = __shfl_xor_sync(0xFFFFFFFF, X[5], 15);
    int p_109 = 0;
    if (V[1] >= wv1_438) {
        p_109 = 1;
    }
    V[1] = fmaxf(V[1], wv1_438);
    X[1] = ((p_109 != 0) ? X[1] : wx1_439);
    float wv_32 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_32 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_137 = fmaxf(V[1], wv_32);
    float lo_137 = fminf(V[1], wv_32);
    float vn_32 = ((upx[0] != 0) ? hi_137 : lo_137);
    X[1] = ((V[1] == vn_32) ? X[1] : wx_32);
    V[1] = vn_32;
    float wv_33 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_33 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_138 = fmaxf(V[1], wv_33);
    float lo_138 = fminf(V[1], wv_33);
    float vn_33 = ((upx[1] != 0) ? hi_138 : lo_138);
    X[1] = ((V[1] == vn_33) ? X[1] : wx_33);
    V[1] = vn_33;
    float wv_34 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_34 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_139 = fmaxf(V[1], wv_34);
    float lo_139 = fminf(V[1], wv_34);
    float vn_34 = ((upx[2] != 0) ? hi_139 : lo_139);
    X[1] = ((V[1] == vn_34) ? X[1] : wx_34);
    V[1] = vn_34;
    float wv_35 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_35 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_140 = fmaxf(V[1], wv_35);
    float lo_140 = fminf(V[1], wv_35);
    float vn_35 = ((upx[3] != 0) ? hi_140 : lo_140);
    X[1] = ((V[1] == vn_35) ? X[1] : wx_35);
    V[1] = vn_35;
    float wv1_461 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    unsigned int wx1_462 = __shfl_xor_sync(0xFFFFFFFF, X[6], 15);
    int p_110 = 0;
    if (V[2] >= wv1_461) {
        p_110 = 1;
    }
    V[2] = fmaxf(V[2], wv1_461);
    X[2] = ((p_110 != 0) ? X[2] : wx1_462);
    float wv_36 = __shfl_xor_sync(0xFFFFFFFF, V[2], 8);
    unsigned int wx_36 = __shfl_xor_sync(0xFFFFFFFF, X[2], 8);
    float hi_141 = fmaxf(V[2], wv_36);
    float lo_141 = fminf(V[2], wv_36);
    float vn_36 = ((upx[0] != 0) ? hi_141 : lo_141);
    X[2] = ((V[2] == vn_36) ? X[2] : wx_36);
    V[2] = vn_36;
    float wv_37 = __shfl_xor_sync(0xFFFFFFFF, V[2], 4);
    unsigned int wx_37 = __shfl_xor_sync(0xFFFFFFFF, X[2], 4);
    float hi_142 = fmaxf(V[2], wv_37);
    float lo_142 = fminf(V[2], wv_37);
    float vn_37 = ((upx[1] != 0) ? hi_142 : lo_142);
    X[2] = ((V[2] == vn_37) ? X[2] : wx_37);
    V[2] = vn_37;
    float wv_38 = __shfl_xor_sync(0xFFFFFFFF, V[2], 2);
    unsigned int wx_38 = __shfl_xor_sync(0xFFFFFFFF, X[2], 2);
    float hi_143 = fmaxf(V[2], wv_38);
    float lo_143 = fminf(V[2], wv_38);
    float vn_38 = ((upx[2] != 0) ? hi_143 : lo_143);
    X[2] = ((V[2] == vn_38) ? X[2] : wx_38);
    V[2] = vn_38;
    float wv_39 = __shfl_xor_sync(0xFFFFFFFF, V[2], 1);
    unsigned int wx_39 = __shfl_xor_sync(0xFFFFFFFF, X[2], 1);
    float hi_144 = fmaxf(V[2], wv_39);
    float lo_144 = fminf(V[2], wv_39);
    float vn_39 = ((upx[3] != 0) ? hi_144 : lo_144);
    X[2] = ((V[2] == vn_39) ? X[2] : wx_39);
    V[2] = vn_39;
    float wv1_484 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    unsigned int wx1_485 = __shfl_xor_sync(0xFFFFFFFF, X[7], 15);
    int p_111 = 0;
    if (V[3] >= wv1_484) {
        p_111 = 1;
    }
    V[3] = fmaxf(V[3], wv1_484);
    X[3] = ((p_111 != 0) ? X[3] : wx1_485);
    float wv_40 = __shfl_xor_sync(0xFFFFFFFF, V[3], 8);
    unsigned int wx_40 = __shfl_xor_sync(0xFFFFFFFF, X[3], 8);
    float hi_145 = fmaxf(V[3], wv_40);
    float lo_145 = fminf(V[3], wv_40);
    float vn_40 = ((upx[0] != 0) ? hi_145 : lo_145);
    X[3] = ((V[3] == vn_40) ? X[3] : wx_40);
    V[3] = vn_40;
    float wv_41 = __shfl_xor_sync(0xFFFFFFFF, V[3], 4);
    unsigned int wx_41 = __shfl_xor_sync(0xFFFFFFFF, X[3], 4);
    float hi_146 = fmaxf(V[3], wv_41);
    float lo_146 = fminf(V[3], wv_41);
    float vn_41 = ((upx[1] != 0) ? hi_146 : lo_146);
    X[3] = ((V[3] == vn_41) ? X[3] : wx_41);
    V[3] = vn_41;
    float wv_42 = __shfl_xor_sync(0xFFFFFFFF, V[3], 2);
    unsigned int wx_42 = __shfl_xor_sync(0xFFFFFFFF, X[3], 2);
    float hi_147 = fmaxf(V[3], wv_42);
    float lo_147 = fminf(V[3], wv_42);
    float vn_42 = ((upx[2] != 0) ? hi_147 : lo_147);
    X[3] = ((V[3] == vn_42) ? X[3] : wx_42);
    V[3] = vn_42;
    float wv_43 = __shfl_xor_sync(0xFFFFFFFF, V[3], 1);
    unsigned int wx_43 = __shfl_xor_sync(0xFFFFFFFF, X[3], 1);
    float hi_148 = fmaxf(V[3], wv_43);
    float lo_148 = fminf(V[3], wv_43);
    float vn_43 = ((upx[3] != 0) ? hi_148 : lo_148);
    X[3] = ((V[3] == vn_43) ? X[3] : wx_43);
    V[3] = vn_43;
    float wv1_507 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    unsigned int wx1_508 = __shfl_xor_sync(0xFFFFFFFF, X[2], 15);
    int p_112 = 0;
    if (V[0] >= wv1_507) {
        p_112 = 1;
    }
    V[0] = fmaxf(V[0], wv1_507);
    X[0] = ((p_112 != 0) ? X[0] : wx1_508);
    float wv_44 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_44 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_149 = fmaxf(V[0], wv_44);
    float lo_149 = fminf(V[0], wv_44);
    float vn_44 = ((upx[0] != 0) ? hi_149 : lo_149);
    X[0] = ((V[0] == vn_44) ? X[0] : wx_44);
    V[0] = vn_44;
    float wv_45 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_45 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_150 = fmaxf(V[0], wv_45);
    float lo_150 = fminf(V[0], wv_45);
    float vn_45 = ((upx[1] != 0) ? hi_150 : lo_150);
    X[0] = ((V[0] == vn_45) ? X[0] : wx_45);
    V[0] = vn_45;
    float wv_46 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_46 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_151 = fmaxf(V[0], wv_46);
    float lo_151 = fminf(V[0], wv_46);
    float vn_46 = ((upx[2] != 0) ? hi_151 : lo_151);
    X[0] = ((V[0] == vn_46) ? X[0] : wx_46);
    V[0] = vn_46;
    float wv_47 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_47 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_152 = fmaxf(V[0], wv_47);
    float lo_152 = fminf(V[0], wv_47);
    float vn_47 = ((upx[3] != 0) ? hi_152 : lo_152);
    X[0] = ((V[0] == vn_47) ? X[0] : wx_47);
    V[0] = vn_47;
    float wv1_530 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    unsigned int wx1_531 = __shfl_xor_sync(0xFFFFFFFF, X[3], 15);
    int p_113 = 0;
    if (V[1] >= wv1_530) {
        p_113 = 1;
    }
    V[1] = fmaxf(V[1], wv1_530);
    X[1] = ((p_113 != 0) ? X[1] : wx1_531);
    float wv_48 = __shfl_xor_sync(0xFFFFFFFF, V[1], 8);
    unsigned int wx_48 = __shfl_xor_sync(0xFFFFFFFF, X[1], 8);
    float hi_153 = fmaxf(V[1], wv_48);
    float lo_153 = fminf(V[1], wv_48);
    float vn_48 = ((upx[0] != 0) ? hi_153 : lo_153);
    X[1] = ((V[1] == vn_48) ? X[1] : wx_48);
    V[1] = vn_48;
    float wv_49 = __shfl_xor_sync(0xFFFFFFFF, V[1], 4);
    unsigned int wx_49 = __shfl_xor_sync(0xFFFFFFFF, X[1], 4);
    float hi_154 = fmaxf(V[1], wv_49);
    float lo_154 = fminf(V[1], wv_49);
    float vn_49 = ((upx[1] != 0) ? hi_154 : lo_154);
    X[1] = ((V[1] == vn_49) ? X[1] : wx_49);
    V[1] = vn_49;
    float wv_50 = __shfl_xor_sync(0xFFFFFFFF, V[1], 2);
    unsigned int wx_50 = __shfl_xor_sync(0xFFFFFFFF, X[1], 2);
    float hi_155 = fmaxf(V[1], wv_50);
    float lo_155 = fminf(V[1], wv_50);
    float vn_50 = ((upx[2] != 0) ? hi_155 : lo_155);
    X[1] = ((V[1] == vn_50) ? X[1] : wx_50);
    V[1] = vn_50;
    float wv_51 = __shfl_xor_sync(0xFFFFFFFF, V[1], 1);
    unsigned int wx_51 = __shfl_xor_sync(0xFFFFFFFF, X[1], 1);
    float hi_156 = fmaxf(V[1], wv_51);
    float lo_156 = fminf(V[1], wv_51);
    float vn_51 = ((upx[3] != 0) ? hi_156 : lo_156);
    X[1] = ((V[1] == vn_51) ? X[1] : wx_51);
    V[1] = vn_51;
    float wvl = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    unsigned int wxl = __shfl_xor_sync(0xFFFFFFFF, X[1], 15);
    int p_114 = 0;
    if (V[0] >= wvl) {
        p_114 = 1;
    }
    V[0] = fmaxf(V[0], wvl);
    X[0] = ((p_114 != 0) ? X[0] : wxl);
    float wv_52 = __shfl_xor_sync(0xFFFFFFFF, V[0], 8);
    unsigned int wx_52 = __shfl_xor_sync(0xFFFFFFFF, X[0], 8);
    float hi_157 = fmaxf(V[0], wv_52);
    float lo_157 = fminf(V[0], wv_52);
    float vn_52 = ((upx[0] != 0) ? hi_157 : lo_157);
    X[0] = ((V[0] == vn_52) ? X[0] : wx_52);
    V[0] = vn_52;
    float wv_53 = __shfl_xor_sync(0xFFFFFFFF, V[0], 4);
    unsigned int wx_53 = __shfl_xor_sync(0xFFFFFFFF, X[0], 4);
    float hi_158 = fmaxf(V[0], wv_53);
    float lo_158 = fminf(V[0], wv_53);
    float vn_53 = ((upx[1] != 0) ? hi_158 : lo_158);
    X[0] = ((V[0] == vn_53) ? X[0] : wx_53);
    V[0] = vn_53;
    float wv_54 = __shfl_xor_sync(0xFFFFFFFF, V[0], 2);
    unsigned int wx_54 = __shfl_xor_sync(0xFFFFFFFF, X[0], 2);
    float hi_159 = fmaxf(V[0], wv_54);
    float lo_159 = fminf(V[0], wv_54);
    float vn_54 = ((upx[2] != 0) ? hi_159 : lo_159);
    X[0] = ((V[0] == vn_54) ? X[0] : wx_54);
    V[0] = vn_54;
    float wv_55 = __shfl_xor_sync(0xFFFFFFFF, V[0], 1);
    unsigned int wx_55 = __shfl_xor_sync(0xFFFFFFFF, X[0], 1);
    float hi_160 = fmaxf(V[0], wv_55);
    float lo_160 = fminf(V[0], wv_55);
    float vn_55 = ((upx[3] != 0) ? hi_160 : lo_160);
    X[0] = ((V[0] == vn_55) ? X[0] : wx_55);
    V[0] = vn_55;
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
        int p_115 = 0;
        if (V2[0] >= bv2) {
            p_115 = 1;
        }
        V2[0] = fmaxf(V2[0], bv2);
        X2[0] = ((p_115 != 0) ? X2[0] : bx2);
        float wv_56 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 8);
        unsigned int wx_56 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 8);
        float hi_161 = fmaxf(V2[0], wv_56);
        float lo_161 = fminf(V2[0], wv_56);
        float vn_56 = ((upx[0] != 0) ? hi_161 : lo_161);
        X2[0] = ((V2[0] == vn_56) ? X2[0] : wx_56);
        V2[0] = vn_56;
        float wv_57 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 4);
        unsigned int wx_57 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 4);
        float hi_162 = fmaxf(V2[0], wv_57);
        float lo_162 = fminf(V2[0], wv_57);
        float vn_57 = ((upx[1] != 0) ? hi_162 : lo_162);
        X2[0] = ((V2[0] == vn_57) ? X2[0] : wx_57);
        V2[0] = vn_57;
        float wv_58 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 2);
        unsigned int wx_58 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 2);
        float hi_163 = fmaxf(V2[0], wv_58);
        float lo_163 = fminf(V2[0], wv_58);
        float vn_58 = ((upx[2] != 0) ? hi_163 : lo_163);
        X2[0] = ((V2[0] == vn_58) ? X2[0] : wx_58);
        V2[0] = vn_58;
        float wv_59 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 1);
        unsigned int wx_59 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 1);
        float hi_164 = fmaxf(V2[0], wv_59);
        float lo_164 = fminf(V2[0], wv_59);
        float vn_59 = ((upx[3] != 0) ? hi_164 : lo_164);
        X2[0] = ((V2[0] == vn_59) ? X2[0] : wx_59);
        V2[0] = vn_59;
        V2[1] = q2[(1 + cgx) * 32 + lnx];
        X2[1] = __as_u32(q2[(1 + cgx) * 32 + 16 + lnx]);
        float bv2_21 = q2[(5 + cgx) * 32 + lnrx];
        unsigned int bx2_22 = __as_u32(q2[(5 + cgx) * 32 + 16 + lnrx]);
        int p_116 = 0;
        if (V2[1] >= bv2_21) {
            p_116 = 1;
        }
        V2[1] = fmaxf(V2[1], bv2_21);
        X2[1] = ((p_116 != 0) ? X2[1] : bx2_22);
        float wv_60 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 8);
        unsigned int wx_60 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 8);
        float hi_165 = fmaxf(V2[1], wv_60);
        float lo_165 = fminf(V2[1], wv_60);
        float vn_60 = ((upx[0] != 0) ? hi_165 : lo_165);
        X2[1] = ((V2[1] == vn_60) ? X2[1] : wx_60);
        V2[1] = vn_60;
        float wv_61 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 4);
        unsigned int wx_61 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 4);
        float hi_166 = fmaxf(V2[1], wv_61);
        float lo_166 = fminf(V2[1], wv_61);
        float vn_61 = ((upx[1] != 0) ? hi_166 : lo_166);
        X2[1] = ((V2[1] == vn_61) ? X2[1] : wx_61);
        V2[1] = vn_61;
        float wv_62 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 2);
        unsigned int wx_62 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 2);
        float hi_167 = fmaxf(V2[1], wv_62);
        float lo_167 = fminf(V2[1], wv_62);
        float vn_62 = ((upx[2] != 0) ? hi_167 : lo_167);
        X2[1] = ((V2[1] == vn_62) ? X2[1] : wx_62);
        V2[1] = vn_62;
        float wv_63 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 1);
        unsigned int wx_63 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 1);
        float hi_168 = fmaxf(V2[1], wv_63);
        float lo_168 = fminf(V2[1], wv_63);
        float vn_63 = ((upx[3] != 0) ? hi_168 : lo_168);
        X2[1] = ((V2[1] == vn_63) ? X2[1] : wx_63);
        V2[1] = vn_63;
        V2[2] = q2[(2 + cgx) * 32 + lnx];
        X2[2] = __as_u32(q2[(2 + cgx) * 32 + 16 + lnx]);
        float bv2_44 = q2[(6 + cgx) * 32 + lnrx];
        unsigned int bx2_45 = __as_u32(q2[(6 + cgx) * 32 + 16 + lnrx]);
        int p_117 = 0;
        if (V2[2] >= bv2_44) {
            p_117 = 1;
        }
        V2[2] = fmaxf(V2[2], bv2_44);
        X2[2] = ((p_117 != 0) ? X2[2] : bx2_45);
        float wv_64 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 8);
        unsigned int wx_64 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 8);
        float hi_169 = fmaxf(V2[2], wv_64);
        float lo_169 = fminf(V2[2], wv_64);
        float vn_64 = ((upx[0] != 0) ? hi_169 : lo_169);
        X2[2] = ((V2[2] == vn_64) ? X2[2] : wx_64);
        V2[2] = vn_64;
        float wv_65 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 4);
        unsigned int wx_65 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 4);
        float hi_170 = fmaxf(V2[2], wv_65);
        float lo_170 = fminf(V2[2], wv_65);
        float vn_65 = ((upx[1] != 0) ? hi_170 : lo_170);
        X2[2] = ((V2[2] == vn_65) ? X2[2] : wx_65);
        V2[2] = vn_65;
        float wv_66 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 2);
        unsigned int wx_66 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 2);
        float hi_171 = fmaxf(V2[2], wv_66);
        float lo_171 = fminf(V2[2], wv_66);
        float vn_66 = ((upx[2] != 0) ? hi_171 : lo_171);
        X2[2] = ((V2[2] == vn_66) ? X2[2] : wx_66);
        V2[2] = vn_66;
        float wv_67 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 1);
        unsigned int wx_67 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 1);
        float hi_172 = fmaxf(V2[2], wv_67);
        float lo_172 = fminf(V2[2], wv_67);
        float vn_67 = ((upx[3] != 0) ? hi_172 : lo_172);
        X2[2] = ((V2[2] == vn_67) ? X2[2] : wx_67);
        V2[2] = vn_67;
        V2[3] = q2[(3 + cgx) * 32 + lnx];
        X2[3] = __as_u32(q2[(3 + cgx) * 32 + 16 + lnx]);
        float bv2_67 = q2[(7 + cgx) * 32 + lnrx];
        unsigned int bx2_68 = __as_u32(q2[(7 + cgx) * 32 + 16 + lnrx]);
        int p_118 = 0;
        if (V2[3] >= bv2_67) {
            p_118 = 1;
        }
        V2[3] = fmaxf(V2[3], bv2_67);
        X2[3] = ((p_118 != 0) ? X2[3] : bx2_68);
        float wv_68 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 8);
        unsigned int wx_68 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 8);
        float hi_173 = fmaxf(V2[3], wv_68);
        float lo_173 = fminf(V2[3], wv_68);
        float vn_68 = ((upx[0] != 0) ? hi_173 : lo_173);
        X2[3] = ((V2[3] == vn_68) ? X2[3] : wx_68);
        V2[3] = vn_68;
        float wv_69 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 4);
        unsigned int wx_69 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 4);
        float hi_174 = fmaxf(V2[3], wv_69);
        float lo_174 = fminf(V2[3], wv_69);
        float vn_69 = ((upx[1] != 0) ? hi_174 : lo_174);
        X2[3] = ((V2[3] == vn_69) ? X2[3] : wx_69);
        V2[3] = vn_69;
        float wv_70 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 2);
        unsigned int wx_70 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 2);
        float hi_175 = fmaxf(V2[3], wv_70);
        float lo_175 = fminf(V2[3], wv_70);
        float vn_70 = ((upx[2] != 0) ? hi_175 : lo_175);
        X2[3] = ((V2[3] == vn_70) ? X2[3] : wx_70);
        V2[3] = vn_70;
        float wv_71 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 1);
        unsigned int wx_71 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 1);
        float hi_176 = fmaxf(V2[3], wv_71);
        float lo_176 = fminf(V2[3], wv_71);
        float vn_71 = ((upx[3] != 0) ? hi_176 : lo_176);
        X2[3] = ((V2[3] == vn_71) ? X2[3] : wx_71);
        V2[3] = vn_71;
        float wv3 = __shfl_xor_sync(0xFFFFFFFF, V2[2], 15);
        unsigned int wx3 = __shfl_xor_sync(0xFFFFFFFF, X2[2], 15);
        int p_119 = 0;
        if (V2[0] >= wv3) {
            p_119 = 1;
        }
        V2[0] = fmaxf(V2[0], wv3);
        X2[0] = ((p_119 != 0) ? X2[0] : wx3);
        float wv_72 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 8);
        unsigned int wx_72 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 8);
        float hi_177 = fmaxf(V2[0], wv_72);
        float lo_177 = fminf(V2[0], wv_72);
        float vn_72 = ((upx[0] != 0) ? hi_177 : lo_177);
        X2[0] = ((V2[0] == vn_72) ? X2[0] : wx_72);
        V2[0] = vn_72;
        float wv_73 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 4);
        unsigned int wx_73 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 4);
        float hi_178 = fmaxf(V2[0], wv_73);
        float lo_178 = fminf(V2[0], wv_73);
        float vn_73 = ((upx[1] != 0) ? hi_178 : lo_178);
        X2[0] = ((V2[0] == vn_73) ? X2[0] : wx_73);
        V2[0] = vn_73;
        float wv_74 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 2);
        unsigned int wx_74 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 2);
        float hi_179 = fmaxf(V2[0], wv_74);
        float lo_179 = fminf(V2[0], wv_74);
        float vn_74 = ((upx[2] != 0) ? hi_179 : lo_179);
        X2[0] = ((V2[0] == vn_74) ? X2[0] : wx_74);
        V2[0] = vn_74;
        float wv_75 = __shfl_xor_sync(0xFFFFFFFF, V2[0], 1);
        unsigned int wx_75 = __shfl_xor_sync(0xFFFFFFFF, X2[0], 1);
        float hi_180 = fmaxf(V2[0], wv_75);
        float lo_180 = fminf(V2[0], wv_75);
        float vn_75 = ((upx[3] != 0) ? hi_180 : lo_180);
        X2[0] = ((V2[0] == vn_75) ? X2[0] : wx_75);
        V2[0] = vn_75;
        float wv3_117 = __shfl_xor_sync(0xFFFFFFFF, V2[3], 15);
        unsigned int wx3_118 = __shfl_xor_sync(0xFFFFFFFF, X2[3], 15);
        int p_120 = 0;
        if (V2[1] >= wv3_117) {
            p_120 = 1;
        }
        V2[1] = fmaxf(V2[1], wv3_117);
        X2[1] = ((p_120 != 0) ? X2[1] : wx3_118);
        float wv_76 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 8);
        unsigned int wx_76 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 8);
        float hi_181 = fmaxf(V2[1], wv_76);
        float lo_181 = fminf(V2[1], wv_76);
        float vn_76 = ((upx[0] != 0) ? hi_181 : lo_181);
        X2[1] = ((V2[1] == vn_76) ? X2[1] : wx_76);
        V2[1] = vn_76;
        float wv_77 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 4);
        unsigned int wx_77 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 4);
        float hi_182 = fmaxf(V2[1], wv_77);
        float lo_182 = fminf(V2[1], wv_77);
        float vn_77 = ((upx[1] != 0) ? hi_182 : lo_182);
        X2[1] = ((V2[1] == vn_77) ? X2[1] : wx_77);
        V2[1] = vn_77;
        float wv_78 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 2);
        unsigned int wx_78 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 2);
        float hi_183 = fmaxf(V2[1], wv_78);
        float lo_183 = fminf(V2[1], wv_78);
        float vn_78 = ((upx[2] != 0) ? hi_183 : lo_183);
        X2[1] = ((V2[1] == vn_78) ? X2[1] : wx_78);
        V2[1] = vn_78;
        float wv_79 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 1);
        unsigned int wx_79 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 1);
        float hi_184 = fmaxf(V2[1], wv_79);
        float lo_184 = fminf(V2[1], wv_79);
        float vn_79 = ((upx[3] != 0) ? hi_184 : lo_184);
        X2[1] = ((V2[1] == vn_79) ? X2[1] : wx_79);
        V2[1] = vn_79;
        float wvl2 = __shfl_xor_sync(0xFFFFFFFF, V2[1], 15);
        unsigned int wxl2 = __shfl_xor_sync(0xFFFFFFFF, X2[1], 15);
        int p_121 = 0;
        if (V2[0] >= wvl2) {
            p_121 = 1;
        }
        V2[0] = fmaxf(V2[0], wvl2);
        X2[0] = ((p_121 != 0) ? X2[0] : wxl2);
        KV = V2[0];
        KX = X2[0];
        int ucol = blockIdx.x + cgx;
        int commit = 0;
        if (gx < 1 && ucol < total_q) {
            commit = 1;
        }
        unsigned int kk = __as_u32(KV);
        unsigned int idx = 2048;
        if (kk < 4282384384u) {
            idx = KX;
        }
        unsigned int rkey = idx << 4 | (unsigned int)lnx;
        int rank = 0;
        unsigned int ox_1_0 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        if (ox_1_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_1 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        if (ox_1_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_2 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        if (ox_1_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_3 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        if (ox_1_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_4 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        if (ox_1_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_5 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        if (ox_1_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_6 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        if (ox_1_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_7 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        if (ox_1_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_8 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        if (ox_1_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_9 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        if (ox_1_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_10 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        if (ox_1_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_11 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        if (ox_1_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_12 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        if (ox_1_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_13 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        if (ox_1_13 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_1_14 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        if (ox_1_14 < rkey) {
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
