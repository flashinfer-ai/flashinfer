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

__global__ __launch_bounds__(512) void
kernel_cake_hopper_msa_eadeab9fa70f479ad877(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int qcol_addr = smem + 0;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 128);
    const int flagw_addr = smem + 128;
    unsigned int* ccnt = reinterpret_cast<unsigned int*>(smem_raw + 34960);
    const int ccnt_addr = smem + 34960;
    int* limc = reinterpret_cast<int*>(smem_raw + 35088);
    const int limc_addr = smem + 35088;
    unsigned int* cbuf = reinterpret_cast<unsigned int*>(smem_raw + 35216);
    const int cbuf_addr = smem + 35216;
    float* pub = reinterpret_cast<float*>(smem_raw + 144);
    const int pub_addr = smem + 144;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 32;
    int c = tid_1 - w * 32;
    int whi = w;
    int col = blockIdx.x * 32 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
        flagw[2] = 0;
    }
    if (tid_1 == 0) {
        flagw[1] = 0;
    }
    if (tid_1 < 32) {
        ccnt[tid_1] = 0;
    }
    int lim = tiles;
    if (col >= total_q) {
        lim = 0;
    }
    int lim_full = lim;
    if (col < total_q) {
        int nv = nvp[col];
        if (nv < lim) {
            lim = nv;
        }
        if (lim < 0) {
            lim = 0;
        }
    }
    if (w == 0) {
        limc[c] = lim;
    }
    long long nq64 = (long long)total_q;
    long long cbase = (long long)head * (long long)tiles * nq64 + (long long)col;
    float neg_inf = -CUDART_INF_F;
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
    unsigned int cb[16];
    float kb[16];
    int t0 = w * 16;
    long long p = cbase + (long long)t0 * nq64;
    cb[0] = 4286578688u;
    if (lim_full > t0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688u;
    if (lim_full > t0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688u;
    if (lim_full > t0 + 2) {
        cb[2] = S[p + 2ll * nq64];
    }
    cb[3] = 4286578688u;
    if (lim_full > t0 + 3) {
        cb[3] = S[p + 3ll * nq64];
    }
    cb[4] = 4286578688u;
    if (lim_full > t0 + 4) {
        cb[4] = S[p + 4ll * nq64];
    }
    cb[5] = 4286578688u;
    if (lim_full > t0 + 5) {
        cb[5] = S[p + 5ll * nq64];
    }
    cb[6] = 4286578688u;
    if (lim_full > t0 + 6) {
        cb[6] = S[p + 6ll * nq64];
    }
    cb[7] = 4286578688u;
    if (lim_full > t0 + 7) {
        cb[7] = S[p + 7ll * nq64];
    }
    cb[8] = 4286578688u;
    if (lim_full > t0 + 8) {
        cb[8] = S[p + 8ll * nq64];
    }
    cb[9] = 4286578688u;
    if (lim_full > t0 + 9) {
        cb[9] = S[p + 9ll * nq64];
    }
    cb[10] = 4286578688u;
    if (lim_full > t0 + 10) {
        cb[10] = S[p + 10ll * nq64];
    }
    cb[11] = 4286578688u;
    if (lim_full > t0 + 11) {
        cb[11] = S[p + 11ll * nq64];
    }
    cb[12] = 4286578688u;
    if (lim_full > t0 + 12) {
        cb[12] = S[p + 12ll * nq64];
    }
    cb[13] = 4286578688u;
    if (lim_full > t0 + 13) {
        cb[13] = S[p + 13ll * nq64];
    }
    cb[14] = 4286578688u;
    if (lim_full > t0 + 14) {
        cb[14] = S[p + 14ll * nq64];
    }
    cb[15] = 4286578688u;
    if (lim_full > t0 + 15) {
        cb[15] = S[p + 15ll * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_1 = w * 16;
    if (lim <= t0_1) {
        cb[0] = 4286578688u;
    }
    if (lim <= t0_1 + 1) {
        cb[1] = 4286578688u;
    }
    if (lim <= t0_1 + 2) {
        cb[2] = 4286578688u;
    }
    if (lim <= t0_1 + 3) {
        cb[3] = 4286578688u;
    }
    if (lim <= t0_1 + 4) {
        cb[4] = 4286578688u;
    }
    if (lim <= t0_1 + 5) {
        cb[5] = 4286578688u;
    }
    if (lim <= t0_1 + 6) {
        cb[6] = 4286578688u;
    }
    if (lim <= t0_1 + 7) {
        cb[7] = 4286578688u;
    }
    if (lim <= t0_1 + 8) {
        cb[8] = 4286578688u;
    }
    if (lim <= t0_1 + 9) {
        cb[9] = 4286578688u;
    }
    if (lim <= t0_1 + 10) {
        cb[10] = 4286578688u;
    }
    if (lim <= t0_1 + 11) {
        cb[11] = 4286578688u;
    }
    if (lim <= t0_1 + 12) {
        cb[12] = 4286578688u;
    }
    if (lim <= t0_1 + 13) {
        cb[13] = 4286578688u;
    }
    if (lim <= t0_1 + 14) {
        cb[14] = 4286578688u;
    }
    if (lim <= t0_1 + 15) {
        cb[15] = 4286578688u;
    }
    unsigned int nb[16];
    #pragma unroll 1
    for (int j = 0; j < num_chunks; j++) {
        int t0_2 = ((j + 1) * 16 + w) * 16;
        long long p_0 = cbase + (long long)t0_2 * nq64;
        nb[0] = 4286578688u;
        if (lim > t0_2) {
            nb[0] = S[p_0];
        }
        nb[1] = 4286578688u;
        if (lim > t0_2 + 1) {
            nb[1] = S[p_0 + nq64];
        }
        nb[2] = 4286578688u;
        if (lim > t0_2 + 2) {
            nb[2] = S[p_0 + 2ll * nq64];
        }
        nb[3] = 4286578688u;
        if (lim > t0_2 + 3) {
            nb[3] = S[p_0 + 3ll * nq64];
        }
        nb[4] = 4286578688u;
        if (lim > t0_2 + 4) {
            nb[4] = S[p_0 + 4ll * nq64];
        }
        nb[5] = 4286578688u;
        if (lim > t0_2 + 5) {
            nb[5] = S[p_0 + 5ll * nq64];
        }
        nb[6] = 4286578688u;
        if (lim > t0_2 + 6) {
            nb[6] = S[p_0 + 6ll * nq64];
        }
        nb[7] = 4286578688u;
        if (lim > t0_2 + 7) {
            nb[7] = S[p_0 + 7ll * nq64];
        }
        nb[8] = 4286578688u;
        if (lim > t0_2 + 8) {
            nb[8] = S[p_0 + 8ll * nq64];
        }
        nb[9] = 4286578688u;
        if (lim > t0_2 + 9) {
            nb[9] = S[p_0 + 9ll * nq64];
        }
        nb[10] = 4286578688u;
        if (lim > t0_2 + 10) {
            nb[10] = S[p_0 + 10ll * nq64];
        }
        nb[11] = 4286578688u;
        if (lim > t0_2 + 11) {
            nb[11] = S[p_0 + 11ll * nq64];
        }
        nb[12] = 4286578688u;
        if (lim > t0_2 + 12) {
            nb[12] = S[p_0 + 12ll * nq64];
        }
        nb[13] = 4286578688u;
        if (lim > t0_2 + 13) {
            nb[13] = S[p_0 + 13ll * nq64];
        }
        nb[14] = 4286578688u;
        if (lim > t0_2 + 14) {
            nb[14] = S[p_0 + 14ll * nq64];
        }
        nb[15] = 4286578688u;
        if (lim > t0_2 + 15) {
            nb[15] = S[p_0 + 15ll * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_3 = (j * 16 + w) * 16;
        float sc = __uint_as_float(cb[0]);
        sc = fmaxf(sc, -1.7014118e+38f);
        sc = fminf(sc, 1.7014118e+38f);
        unsigned int key_1_0 = __as_u32(sc) & 4294965248u | (unsigned int)t0_3;
        kb[0] = __uint_as_float(key_1_0);
        float sc_0 = __uint_as_float(cb[1]);
        sc_0 = fmaxf(sc_0, -1.7014118e+38f);
        sc_0 = fminf(sc_0, 1.7014118e+38f);
        unsigned int key_1_1 = __as_u32(sc_0) & 4294965248u | (unsigned int)(t0_3 + 1);
        kb[1] = __uint_as_float(key_1_1);
        float sc_1 = __uint_as_float(cb[2]);
        sc_1 = fmaxf(sc_1, -1.7014118e+38f);
        sc_1 = fminf(sc_1, 1.7014118e+38f);
        unsigned int key_1_2 = __as_u32(sc_1) & 4294965248u | (unsigned int)(t0_3 + 2);
        kb[2] = __uint_as_float(key_1_2);
        float sc_2 = __uint_as_float(cb[3]);
        sc_2 = fmaxf(sc_2, -1.7014118e+38f);
        sc_2 = fminf(sc_2, 1.7014118e+38f);
        unsigned int key_1_3 = __as_u32(sc_2) & 4294965248u | (unsigned int)(t0_3 + 3);
        kb[3] = __uint_as_float(key_1_3);
        float sc_3 = __uint_as_float(cb[4]);
        sc_3 = fmaxf(sc_3, -1.7014118e+38f);
        sc_3 = fminf(sc_3, 1.7014118e+38f);
        unsigned int key_1_4 = __as_u32(sc_3) & 4294965248u | (unsigned int)(t0_3 + 4);
        kb[4] = __uint_as_float(key_1_4);
        float sc_4 = __uint_as_float(cb[5]);
        sc_4 = fmaxf(sc_4, -1.7014118e+38f);
        sc_4 = fminf(sc_4, 1.7014118e+38f);
        unsigned int key_1_5 = __as_u32(sc_4) & 4294965248u | (unsigned int)(t0_3 + 5);
        kb[5] = __uint_as_float(key_1_5);
        float sc_5 = __uint_as_float(cb[6]);
        sc_5 = fmaxf(sc_5, -1.7014118e+38f);
        sc_5 = fminf(sc_5, 1.7014118e+38f);
        unsigned int key_1_6 = __as_u32(sc_5) & 4294965248u | (unsigned int)(t0_3 + 6);
        kb[6] = __uint_as_float(key_1_6);
        float sc_6 = __uint_as_float(cb[7]);
        sc_6 = fmaxf(sc_6, -1.7014118e+38f);
        sc_6 = fminf(sc_6, 1.7014118e+38f);
        unsigned int key_1_7 = __as_u32(sc_6) & 4294965248u | (unsigned int)(t0_3 + 7);
        kb[7] = __uint_as_float(key_1_7);
        float sc_7 = __uint_as_float(cb[8]);
        sc_7 = fmaxf(sc_7, -1.7014118e+38f);
        sc_7 = fminf(sc_7, 1.7014118e+38f);
        unsigned int key_1_8 = __as_u32(sc_7) & 4294965248u | (unsigned int)(t0_3 + 8);
        kb[8] = __uint_as_float(key_1_8);
        float sc_8 = __uint_as_float(cb[9]);
        sc_8 = fmaxf(sc_8, -1.7014118e+38f);
        sc_8 = fminf(sc_8, 1.7014118e+38f);
        unsigned int key_1_9 = __as_u32(sc_8) & 4294965248u | (unsigned int)(t0_3 + 9);
        kb[9] = __uint_as_float(key_1_9);
        float sc_9 = __uint_as_float(cb[10]);
        sc_9 = fmaxf(sc_9, -1.7014118e+38f);
        sc_9 = fminf(sc_9, 1.7014118e+38f);
        unsigned int key_1_10 = __as_u32(sc_9) & 4294965248u | (unsigned int)(t0_3 + 10);
        kb[10] = __uint_as_float(key_1_10);
        float sc_10 = __uint_as_float(cb[11]);
        sc_10 = fmaxf(sc_10, -1.7014118e+38f);
        sc_10 = fminf(sc_10, 1.7014118e+38f);
        unsigned int key_1_11 = __as_u32(sc_10) & 4294965248u | (unsigned int)(t0_3 + 11);
        kb[11] = __uint_as_float(key_1_11);
        float sc_11 = __uint_as_float(cb[12]);
        sc_11 = fmaxf(sc_11, -1.7014118e+38f);
        sc_11 = fminf(sc_11, 1.7014118e+38f);
        unsigned int key_1_12 = __as_u32(sc_11) & 4294965248u | (unsigned int)(t0_3 + 12);
        kb[12] = __uint_as_float(key_1_12);
        float sc_12 = __uint_as_float(cb[13]);
        sc_12 = fmaxf(sc_12, -1.7014118e+38f);
        sc_12 = fminf(sc_12, 1.7014118e+38f);
        unsigned int key_1_13 = __as_u32(sc_12) & 4294965248u | (unsigned int)(t0_3 + 13);
        kb[13] = __uint_as_float(key_1_13);
        float sc_13 = __uint_as_float(cb[14]);
        sc_13 = fmaxf(sc_13, -1.7014118e+38f);
        sc_13 = fminf(sc_13, 1.7014118e+38f);
        unsigned int key_1_14 = __as_u32(sc_13) & 4294965248u | (unsigned int)(t0_3 + 14);
        kb[14] = __uint_as_float(key_1_14);
        float sc_14 = __uint_as_float(cb[15]);
        sc_14 = fmaxf(sc_14, -1.7014118e+38f);
        sc_14 = fminf(sc_14, 1.7014118e+38f);
        unsigned int key_1_15 = __as_u32(sc_14) & 4294965248u | (unsigned int)(t0_3 + 15);
        kb[15] = __uint_as_float(key_1_15);
        int f = 0;
        if ((t0_3 < fb || t0_3 + 16 > lim - fe) && t0_3 < lim) {
            f = 1;
        }
        if (f != 0) {
            int f_0 = 0;
            if ((t0_3 < fb || t0_3 >= lim - fe) && lim > t0_3) {
                f_0 = 1;
            }
            if (f_0 != 0) {
                kb[0] = __uint_as_float(2139092992 | (unsigned int)t0_3);
            }
            int f_1 = 0;
            if ((t0_3 + 1 < fb || t0_3 + 1 >= lim - fe) && lim > t0_3 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                kb[1] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 1));
            }
            int f_2 = 0;
            if ((t0_3 + 2 < fb || t0_3 + 2 >= lim - fe) && lim > t0_3 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                kb[2] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 2));
            }
            int f_3 = 0;
            if ((t0_3 + 3 < fb || t0_3 + 3 >= lim - fe) && lim > t0_3 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                kb[3] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 3));
            }
            int f_4 = 0;
            if ((t0_3 + 4 < fb || t0_3 + 4 >= lim - fe) && lim > t0_3 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                kb[4] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 4));
            }
            int f_5 = 0;
            if ((t0_3 + 5 < fb || t0_3 + 5 >= lim - fe) && lim > t0_3 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                kb[5] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 5));
            }
            int f_6 = 0;
            if ((t0_3 + 6 < fb || t0_3 + 6 >= lim - fe) && lim > t0_3 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                kb[6] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 6));
            }
            int f_7 = 0;
            if ((t0_3 + 7 < fb || t0_3 + 7 >= lim - fe) && lim > t0_3 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                kb[7] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 7));
            }
            int f_8 = 0;
            if ((t0_3 + 8 < fb || t0_3 + 8 >= lim - fe) && lim > t0_3 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                kb[8] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 8));
            }
            int f_9 = 0;
            if ((t0_3 + 9 < fb || t0_3 + 9 >= lim - fe) && lim > t0_3 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                kb[9] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 9));
            }
            int f_10 = 0;
            if ((t0_3 + 10 < fb || t0_3 + 10 >= lim - fe) && lim > t0_3 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                kb[10] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 10));
            }
            int f_11 = 0;
            if ((t0_3 + 11 < fb || t0_3 + 11 >= lim - fe) && lim > t0_3 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                kb[11] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 11));
            }
            int f_12 = 0;
            if ((t0_3 + 12 < fb || t0_3 + 12 >= lim - fe) && lim > t0_3 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                kb[12] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 12));
            }
            int f_13 = 0;
            if ((t0_3 + 13 < fb || t0_3 + 13 >= lim - fe) && lim > t0_3 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                kb[13] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 13));
            }
            int f_14 = 0;
            if ((t0_3 + 14 < fb || t0_3 + 14 >= lim - fe) && lim > t0_3 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                kb[14] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 14));
            }
            int f_15 = 0;
            if ((t0_3 + 15 < fb || t0_3 + 15 >= lim - fe) && lim > t0_3 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                kb[15] = __uint_as_float(2139092992 | (unsigned int)(t0_3 + 15));
            }
        }
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
        float r = rej;
        float hi_0 = fmaxf(a[0], kb[15]);
        float lo_0 = fminf(a[0], kb[15]);
        a[0] = hi_0;
        r = fmaxf(r, lo_0);
        float hi_1 = fmaxf(a[1], kb[14]);
        float lo_1 = fminf(a[1], kb[14]);
        a[1] = hi_1;
        r = fmaxf(r, lo_1);
        float hi_2 = fmaxf(a[2], kb[13]);
        float lo_2 = fminf(a[2], kb[13]);
        a[2] = hi_2;
        r = fmaxf(r, lo_2);
        float hi_3 = fmaxf(a[3], kb[12]);
        float lo_3 = fminf(a[3], kb[12]);
        a[3] = hi_3;
        r = fmaxf(r, lo_3);
        float hi_4 = fmaxf(a[4], kb[11]);
        float lo_4 = fminf(a[4], kb[11]);
        a[4] = hi_4;
        r = fmaxf(r, lo_4);
        float hi_5 = fmaxf(a[5], kb[10]);
        float lo_5 = fminf(a[5], kb[10]);
        a[5] = hi_5;
        r = fmaxf(r, lo_5);
        float hi_6 = fmaxf(a[6], kb[9]);
        float lo_6 = fminf(a[6], kb[9]);
        a[6] = hi_6;
        r = fmaxf(r, lo_6);
        float hi_7 = fmaxf(a[7], kb[8]);
        float lo_7 = fminf(a[7], kb[8]);
        a[7] = hi_7;
        r = fmaxf(r, lo_7);
        float hi_8 = fmaxf(a[8], kb[7]);
        float lo_8 = fminf(a[8], kb[7]);
        a[8] = hi_8;
        r = fmaxf(r, lo_8);
        float hi_9 = fmaxf(a[9], kb[6]);
        float lo_9 = fminf(a[9], kb[6]);
        a[9] = hi_9;
        r = fmaxf(r, lo_9);
        float hi_10 = fmaxf(a[10], kb[5]);
        float lo_10 = fminf(a[10], kb[5]);
        a[10] = hi_10;
        r = fmaxf(r, lo_10);
        float hi_11 = fmaxf(a[11], kb[4]);
        float lo_11 = fminf(a[11], kb[4]);
        a[11] = hi_11;
        r = fmaxf(r, lo_11);
        float hi_12 = fmaxf(a[12], kb[3]);
        float lo_12 = fminf(a[12], kb[3]);
        a[12] = hi_12;
        r = fmaxf(r, lo_12);
        float hi_13 = fmaxf(a[13], kb[2]);
        float lo_13 = fminf(a[13], kb[2]);
        a[13] = hi_13;
        r = fmaxf(r, lo_13);
        float hi_14 = fmaxf(a[14], kb[1]);
        float lo_14 = fminf(a[14], kb[1]);
        a[14] = hi_14;
        r = fmaxf(r, lo_14);
        float hi_15 = fmaxf(a[15], kb[0]);
        float lo_15 = fminf(a[15], kb[0]);
        a[15] = hi_15;
        r = fmaxf(r, lo_15);
        float hi_16 = fmaxf(a[0], a[8]);
        float lo_16 = fminf(a[0], a[8]);
        a[0] = hi_16;
        a[8] = lo_16;
        float hi_17 = fmaxf(a[1], a[9]);
        float lo_17 = fminf(a[1], a[9]);
        a[1] = hi_17;
        a[9] = lo_17;
        float hi_18 = fmaxf(a[2], a[10]);
        float lo_18 = fminf(a[2], a[10]);
        a[2] = hi_18;
        a[10] = lo_18;
        float hi_19 = fmaxf(a[3], a[11]);
        float lo_19 = fminf(a[3], a[11]);
        a[3] = hi_19;
        a[11] = lo_19;
        float hi_20 = fmaxf(a[4], a[12]);
        float lo_20 = fminf(a[4], a[12]);
        a[4] = hi_20;
        a[12] = lo_20;
        float hi_21 = fmaxf(a[5], a[13]);
        float lo_21 = fminf(a[5], a[13]);
        a[5] = hi_21;
        a[13] = lo_21;
        float hi_22 = fmaxf(a[6], a[14]);
        float lo_22 = fminf(a[6], a[14]);
        a[6] = hi_22;
        a[14] = lo_22;
        float hi_23 = fmaxf(a[7], a[15]);
        float lo_23 = fminf(a[7], a[15]);
        a[7] = hi_23;
        a[15] = lo_23;
        float hi_24 = fmaxf(a[0], a[4]);
        float lo_24 = fminf(a[0], a[4]);
        a[0] = hi_24;
        a[4] = lo_24;
        float hi_25 = fmaxf(a[1], a[5]);
        float lo_25 = fminf(a[1], a[5]);
        a[1] = hi_25;
        a[5] = lo_25;
        float hi_26 = fmaxf(a[2], a[6]);
        float lo_26 = fminf(a[2], a[6]);
        a[2] = hi_26;
        a[6] = lo_26;
        float hi_27 = fmaxf(a[3], a[7]);
        float lo_27 = fminf(a[3], a[7]);
        a[3] = hi_27;
        a[7] = lo_27;
        float hi_28 = fmaxf(a[8], a[12]);
        float lo_28 = fminf(a[8], a[12]);
        a[8] = hi_28;
        a[12] = lo_28;
        float hi_29 = fmaxf(a[9], a[13]);
        float lo_29 = fminf(a[9], a[13]);
        a[9] = hi_29;
        a[13] = lo_29;
        float hi_30 = fmaxf(a[10], a[14]);
        float lo_30 = fminf(a[10], a[14]);
        a[10] = hi_30;
        a[14] = lo_30;
        float hi_31 = fmaxf(a[11], a[15]);
        float lo_31 = fminf(a[11], a[15]);
        a[11] = hi_31;
        a[15] = lo_31;
        float hi_32 = fmaxf(a[0], a[2]);
        float lo_32 = fminf(a[0], a[2]);
        a[0] = hi_32;
        a[2] = lo_32;
        float hi_33 = fmaxf(a[1], a[3]);
        float lo_33 = fminf(a[1], a[3]);
        a[1] = hi_33;
        a[3] = lo_33;
        float hi_34 = fmaxf(a[4], a[6]);
        float lo_34 = fminf(a[4], a[6]);
        a[4] = hi_34;
        a[6] = lo_34;
        float hi_35 = fmaxf(a[5], a[7]);
        float lo_35 = fminf(a[5], a[7]);
        a[5] = hi_35;
        a[7] = lo_35;
        float hi_36 = fmaxf(a[8], a[10]);
        float lo_36 = fminf(a[8], a[10]);
        a[8] = hi_36;
        a[10] = lo_36;
        float hi_37 = fmaxf(a[9], a[11]);
        float lo_37 = fminf(a[9], a[11]);
        a[9] = hi_37;
        a[11] = lo_37;
        float hi_38 = fmaxf(a[12], a[14]);
        float lo_38 = fminf(a[12], a[14]);
        a[12] = hi_38;
        a[14] = lo_38;
        float hi_39 = fmaxf(a[13], a[15]);
        float lo_39 = fminf(a[13], a[15]);
        a[13] = hi_39;
        a[15] = lo_39;
        float hi_40 = fmaxf(a[0], a[1]);
        float lo_40 = fminf(a[0], a[1]);
        a[0] = hi_40;
        a[1] = lo_40;
        float hi_41 = fmaxf(a[2], a[3]);
        float lo_41 = fminf(a[2], a[3]);
        a[2] = hi_41;
        a[3] = lo_41;
        float hi_42 = fmaxf(a[4], a[5]);
        float lo_42 = fminf(a[4], a[5]);
        a[4] = hi_42;
        a[5] = lo_42;
        float hi_43 = fmaxf(a[6], a[7]);
        float lo_43 = fminf(a[6], a[7]);
        a[6] = hi_43;
        a[7] = lo_43;
        float hi_44 = fmaxf(a[8], a[9]);
        float lo_44 = fminf(a[8], a[9]);
        a[8] = hi_44;
        a[9] = lo_44;
        float hi_45 = fmaxf(a[10], a[11]);
        float lo_45 = fminf(a[10], a[11]);
        a[10] = hi_45;
        a[11] = lo_45;
        float hi_46 = fmaxf(a[12], a[13]);
        float lo_46 = fminf(a[12], a[13]);
        a[12] = hi_46;
        a[13] = lo_46;
        float hi_47 = fmaxf(a[14], a[15]);
        float lo_47 = fminf(a[14], a[15]);
        a[14] = hi_47;
        a[15] = lo_47;
        rej = r;
        cb[0] = nb[0];
        cb[1] = nb[1];
        cb[2] = nb[2];
        cb[3] = nb[3];
        cb[4] = nb[4];
        cb[5] = nb[5];
        cb[6] = nb[6];
        cb[7] = nb[7];
        cb[8] = nb[8];
        cb[9] = nb[9];
        cb[10] = nb[10];
        cb[11] = nb[11];
        cb[12] = nb[12];
        cb[13] = nb[13];
        cb[14] = nb[14];
        cb[15] = nb[15];
    }
    int ln = tid_1 & 15;
    int g = tid_1 >> 4;
    int cg = g & 31;
    int sg = g >> 5;
    int lnr = 15 - ln;
    int up[4];
    up[0] = 0;
    if ((ln & 8) == 0) {
        up[0] = 1;
    }
    up[1] = 0;
    if ((ln & 4) == 0) {
        up[1] = 1;
    }
    up[2] = 0;
    if ((ln & 2) == 0) {
        up[2] = 1;
    }
    up[3] = 0;
    if ((ln & 1) == 0) {
        up[3] = 1;
    }
    float rr1[1];
    int pb = tid_1 * 17;
    pub[pb] = a[0];
    pub[pb + 1] = a[1];
    pub[pb + 2] = a[2];
    pub[pb + 3] = a[3];
    pub[pb + 4] = a[4];
    pub[pb + 5] = a[5];
    pub[pb + 6] = a[6];
    pub[pb + 7] = a[7];
    pub[pb + 8] = a[8];
    pub[pb + 9] = a[9];
    pub[pb + 10] = a[10];
    pub[pb + 11] = a[11];
    pub[pb + 12] = a[12];
    pub[pb + 13] = a[13];
    pub[pb + 14] = a[14];
    pub[pb + 15] = a[15];
    pub[pb + 16] = rej;
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    float r_1 = neg_inf;
    float V[8];
    int s0_0 = (sg * 16 * 32 + cg) * 17;
    int s1_0 = ((sg * 16 + 8) * 32 + cg) * 17;
    float x0_0 = pub[s0_0 + ln];
    float y0_0 = pub[s1_0 + lnr];
    float lo0_0 = fminf(x0_0, y0_0);
    r_1 = fmaxf(r_1, lo0_0);
    float hi0_0 = fmaxf(x0_0, y0_0);
    float cur = hi0_0;
    float pv_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float hi_48 = fmaxf(cur, pv_0);
    float lo_48 = fminf(cur, pv_0);
    cur = ((up[0] != 0) ? hi_48 : lo_48);
    float pv_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float hi_49 = fmaxf(cur, pv_1);
    float lo_49 = fminf(cur, pv_1);
    cur = ((up[1] != 0) ? hi_49 : lo_49);
    float pv_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float hi_50 = fmaxf(cur, pv_2);
    float lo_50 = fminf(cur, pv_2);
    cur = ((up[2] != 0) ? hi_50 : lo_50);
    float pv_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float hi_51 = fmaxf(cur, pv_3);
    float lo_51 = fminf(cur, pv_3);
    cur = ((up[3] != 0) ? hi_51 : lo_51);
    V[0] = cur;
    int s0_1 = ((sg * 16 + 1) * 32 + cg) * 17;
    int s1_1 = ((sg * 16 + 1 + 8) * 32 + cg) * 17;
    float x0_1 = pub[s0_1 + ln];
    float y0_1 = pub[s1_1 + lnr];
    float lo0_1 = fminf(x0_1, y0_1);
    r_1 = fmaxf(r_1, lo0_1);
    float hi0_1 = fmaxf(x0_1, y0_1);
    float cur_0 = hi0_1;
    float pv_4 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 8);
    float hi_52 = fmaxf(cur_0, pv_4);
    float lo_52 = fminf(cur_0, pv_4);
    cur_0 = ((up[0] != 0) ? hi_52 : lo_52);
    float pv_5 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 4);
    float hi_53 = fmaxf(cur_0, pv_5);
    float lo_53 = fminf(cur_0, pv_5);
    cur_0 = ((up[1] != 0) ? hi_53 : lo_53);
    float pv_6 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 2);
    float hi_54 = fmaxf(cur_0, pv_6);
    float lo_54 = fminf(cur_0, pv_6);
    cur_0 = ((up[2] != 0) ? hi_54 : lo_54);
    float pv_7 = __shfl_xor_sync(0xFFFFFFFF, cur_0, 1);
    float hi_55 = fmaxf(cur_0, pv_7);
    float lo_55 = fminf(cur_0, pv_7);
    cur_0 = ((up[3] != 0) ? hi_55 : lo_55);
    V[1] = cur_0;
    int s0_2 = ((sg * 16 + 2) * 32 + cg) * 17;
    int s1_2 = ((sg * 16 + 2 + 8) * 32 + cg) * 17;
    float x0_2 = pub[s0_2 + ln];
    float y0_2 = pub[s1_2 + lnr];
    float lo0_2 = fminf(x0_2, y0_2);
    r_1 = fmaxf(r_1, lo0_2);
    float hi0_2 = fmaxf(x0_2, y0_2);
    float cur_1 = hi0_2;
    float pv_8 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 8);
    float hi_56 = fmaxf(cur_1, pv_8);
    float lo_56 = fminf(cur_1, pv_8);
    cur_1 = ((up[0] != 0) ? hi_56 : lo_56);
    float pv_9 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 4);
    float hi_57 = fmaxf(cur_1, pv_9);
    float lo_57 = fminf(cur_1, pv_9);
    cur_1 = ((up[1] != 0) ? hi_57 : lo_57);
    float pv_10 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 2);
    float hi_58 = fmaxf(cur_1, pv_10);
    float lo_58 = fminf(cur_1, pv_10);
    cur_1 = ((up[2] != 0) ? hi_58 : lo_58);
    float pv_11 = __shfl_xor_sync(0xFFFFFFFF, cur_1, 1);
    float hi_59 = fmaxf(cur_1, pv_11);
    float lo_59 = fminf(cur_1, pv_11);
    cur_1 = ((up[3] != 0) ? hi_59 : lo_59);
    V[2] = cur_1;
    int s0_3 = ((sg * 16 + 3) * 32 + cg) * 17;
    int s1_3 = ((sg * 16 + 3 + 8) * 32 + cg) * 17;
    float x0_3 = pub[s0_3 + ln];
    float y0_3 = pub[s1_3 + lnr];
    float lo0_3 = fminf(x0_3, y0_3);
    r_1 = fmaxf(r_1, lo0_3);
    float hi0_3 = fmaxf(x0_3, y0_3);
    float cur_2 = hi0_3;
    float pv_12 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 8);
    float hi_60 = fmaxf(cur_2, pv_12);
    float lo_60 = fminf(cur_2, pv_12);
    cur_2 = ((up[0] != 0) ? hi_60 : lo_60);
    float pv_13 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 4);
    float hi_61 = fmaxf(cur_2, pv_13);
    float lo_61 = fminf(cur_2, pv_13);
    cur_2 = ((up[1] != 0) ? hi_61 : lo_61);
    float pv_14 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 2);
    float hi_62 = fmaxf(cur_2, pv_14);
    float lo_62 = fminf(cur_2, pv_14);
    cur_2 = ((up[2] != 0) ? hi_62 : lo_62);
    float pv_15 = __shfl_xor_sync(0xFFFFFFFF, cur_2, 1);
    float hi_63 = fmaxf(cur_2, pv_15);
    float lo_63 = fminf(cur_2, pv_15);
    cur_2 = ((up[3] != 0) ? hi_63 : lo_63);
    V[3] = cur_2;
    int s0_4 = ((sg * 16 + 4) * 32 + cg) * 17;
    int s1_4 = ((sg * 16 + 4 + 8) * 32 + cg) * 17;
    float x0_4 = pub[s0_4 + ln];
    float y0_4 = pub[s1_4 + lnr];
    float lo0_4 = fminf(x0_4, y0_4);
    r_1 = fmaxf(r_1, lo0_4);
    float hi0_4 = fmaxf(x0_4, y0_4);
    float cur_3 = hi0_4;
    float pv_16 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 8);
    float hi_64 = fmaxf(cur_3, pv_16);
    float lo_64 = fminf(cur_3, pv_16);
    cur_3 = ((up[0] != 0) ? hi_64 : lo_64);
    float pv_17 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 4);
    float hi_65 = fmaxf(cur_3, pv_17);
    float lo_65 = fminf(cur_3, pv_17);
    cur_3 = ((up[1] != 0) ? hi_65 : lo_65);
    float pv_18 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 2);
    float hi_66 = fmaxf(cur_3, pv_18);
    float lo_66 = fminf(cur_3, pv_18);
    cur_3 = ((up[2] != 0) ? hi_66 : lo_66);
    float pv_19 = __shfl_xor_sync(0xFFFFFFFF, cur_3, 1);
    float hi_67 = fmaxf(cur_3, pv_19);
    float lo_67 = fminf(cur_3, pv_19);
    cur_3 = ((up[3] != 0) ? hi_67 : lo_67);
    V[4] = cur_3;
    int s0_5 = ((sg * 16 + 5) * 32 + cg) * 17;
    int s1_5 = ((sg * 16 + 5 + 8) * 32 + cg) * 17;
    float x0_5 = pub[s0_5 + ln];
    float y0_5 = pub[s1_5 + lnr];
    float lo0_5 = fminf(x0_5, y0_5);
    r_1 = fmaxf(r_1, lo0_5);
    float hi0_5 = fmaxf(x0_5, y0_5);
    float cur_4 = hi0_5;
    float pv_20 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 8);
    float hi_68 = fmaxf(cur_4, pv_20);
    float lo_68 = fminf(cur_4, pv_20);
    cur_4 = ((up[0] != 0) ? hi_68 : lo_68);
    float pv_21 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 4);
    float hi_69 = fmaxf(cur_4, pv_21);
    float lo_69 = fminf(cur_4, pv_21);
    cur_4 = ((up[1] != 0) ? hi_69 : lo_69);
    float pv_22 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 2);
    float hi_70 = fmaxf(cur_4, pv_22);
    float lo_70 = fminf(cur_4, pv_22);
    cur_4 = ((up[2] != 0) ? hi_70 : lo_70);
    float pv_23 = __shfl_xor_sync(0xFFFFFFFF, cur_4, 1);
    float hi_71 = fmaxf(cur_4, pv_23);
    float lo_71 = fminf(cur_4, pv_23);
    cur_4 = ((up[3] != 0) ? hi_71 : lo_71);
    V[5] = cur_4;
    int s0_6 = ((sg * 16 + 6) * 32 + cg) * 17;
    int s1_6 = ((sg * 16 + 6 + 8) * 32 + cg) * 17;
    float x0_6 = pub[s0_6 + ln];
    float y0_6 = pub[s1_6 + lnr];
    float lo0_6 = fminf(x0_6, y0_6);
    r_1 = fmaxf(r_1, lo0_6);
    float hi0_6 = fmaxf(x0_6, y0_6);
    float cur_5 = hi0_6;
    float pv_24 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 8);
    float hi_72 = fmaxf(cur_5, pv_24);
    float lo_72 = fminf(cur_5, pv_24);
    cur_5 = ((up[0] != 0) ? hi_72 : lo_72);
    float pv_25 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 4);
    float hi_73 = fmaxf(cur_5, pv_25);
    float lo_73 = fminf(cur_5, pv_25);
    cur_5 = ((up[1] != 0) ? hi_73 : lo_73);
    float pv_26 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 2);
    float hi_74 = fmaxf(cur_5, pv_26);
    float lo_74 = fminf(cur_5, pv_26);
    cur_5 = ((up[2] != 0) ? hi_74 : lo_74);
    float pv_27 = __shfl_xor_sync(0xFFFFFFFF, cur_5, 1);
    float hi_75 = fmaxf(cur_5, pv_27);
    float lo_75 = fminf(cur_5, pv_27);
    cur_5 = ((up[3] != 0) ? hi_75 : lo_75);
    V[6] = cur_5;
    int s0_7 = ((sg * 16 + 7) * 32 + cg) * 17;
    int s1_7 = ((sg * 16 + 7 + 8) * 32 + cg) * 17;
    float x0_7 = pub[s0_7 + ln];
    float y0_7 = pub[s1_7 + lnr];
    float lo0_7 = fminf(x0_7, y0_7);
    r_1 = fmaxf(r_1, lo0_7);
    float hi0_7 = fmaxf(x0_7, y0_7);
    float cur_6 = hi0_7;
    float pv_28 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 8);
    float hi_76 = fmaxf(cur_6, pv_28);
    float lo_76 = fminf(cur_6, pv_28);
    cur_6 = ((up[0] != 0) ? hi_76 : lo_76);
    float pv_29 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 4);
    float hi_77 = fmaxf(cur_6, pv_29);
    float lo_77 = fminf(cur_6, pv_29);
    cur_6 = ((up[1] != 0) ? hi_77 : lo_77);
    float pv_30 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 2);
    float hi_78 = fmaxf(cur_6, pv_30);
    float lo_78 = fminf(cur_6, pv_30);
    cur_6 = ((up[2] != 0) ? hi_78 : lo_78);
    float pv_31 = __shfl_xor_sync(0xFFFFFFFF, cur_6, 1);
    float hi_79 = fmaxf(cur_6, pv_31);
    float lo_79 = fminf(cur_6, pv_31);
    cur_6 = ((up[3] != 0) ? hi_79 : lo_79);
    V[7] = cur_6;
    float rs = pub[((sg * 16 + ln) * 32 + cg) * 17 + 16];
    r_1 = fmaxf(r_1, rs);
    float y1_0 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float lo1_0 = fminf(V[0], y1_0);
    r_1 = fmaxf(r_1, lo1_0);
    float hi1_0 = fmaxf(V[0], y1_0);
    float cur_7 = hi1_0;
    float pv_32 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 8);
    float hi_80 = fmaxf(cur_7, pv_32);
    float lo_80 = fminf(cur_7, pv_32);
    cur_7 = ((up[0] != 0) ? hi_80 : lo_80);
    float pv_33 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 4);
    float hi_81 = fmaxf(cur_7, pv_33);
    float lo_81 = fminf(cur_7, pv_33);
    cur_7 = ((up[1] != 0) ? hi_81 : lo_81);
    float pv_34 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 2);
    float hi_82 = fmaxf(cur_7, pv_34);
    float lo_82 = fminf(cur_7, pv_34);
    cur_7 = ((up[2] != 0) ? hi_82 : lo_82);
    float pv_35 = __shfl_xor_sync(0xFFFFFFFF, cur_7, 1);
    float hi_83 = fmaxf(cur_7, pv_35);
    float lo_83 = fminf(cur_7, pv_35);
    cur_7 = ((up[3] != 0) ? hi_83 : lo_83);
    V[0] = cur_7;
    float y1_1 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float lo1_1 = fminf(V[1], y1_1);
    r_1 = fmaxf(r_1, lo1_1);
    float hi1_1 = fmaxf(V[1], y1_1);
    float cur_8 = hi1_1;
    float pv_36 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 8);
    float hi_84 = fmaxf(cur_8, pv_36);
    float lo_84 = fminf(cur_8, pv_36);
    cur_8 = ((up[0] != 0) ? hi_84 : lo_84);
    float pv_37 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 4);
    float hi_85 = fmaxf(cur_8, pv_37);
    float lo_85 = fminf(cur_8, pv_37);
    cur_8 = ((up[1] != 0) ? hi_85 : lo_85);
    float pv_38 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 2);
    float hi_86 = fmaxf(cur_8, pv_38);
    float lo_86 = fminf(cur_8, pv_38);
    cur_8 = ((up[2] != 0) ? hi_86 : lo_86);
    float pv_39 = __shfl_xor_sync(0xFFFFFFFF, cur_8, 1);
    float hi_87 = fmaxf(cur_8, pv_39);
    float lo_87 = fminf(cur_8, pv_39);
    cur_8 = ((up[3] != 0) ? hi_87 : lo_87);
    V[1] = cur_8;
    float y1_2 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float lo1_2 = fminf(V[2], y1_2);
    r_1 = fmaxf(r_1, lo1_2);
    float hi1_2 = fmaxf(V[2], y1_2);
    float cur_9 = hi1_2;
    float pv_40 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 8);
    float hi_88 = fmaxf(cur_9, pv_40);
    float lo_88 = fminf(cur_9, pv_40);
    cur_9 = ((up[0] != 0) ? hi_88 : lo_88);
    float pv_41 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 4);
    float hi_89 = fmaxf(cur_9, pv_41);
    float lo_89 = fminf(cur_9, pv_41);
    cur_9 = ((up[1] != 0) ? hi_89 : lo_89);
    float pv_42 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 2);
    float hi_90 = fmaxf(cur_9, pv_42);
    float lo_90 = fminf(cur_9, pv_42);
    cur_9 = ((up[2] != 0) ? hi_90 : lo_90);
    float pv_43 = __shfl_xor_sync(0xFFFFFFFF, cur_9, 1);
    float hi_91 = fmaxf(cur_9, pv_43);
    float lo_91 = fminf(cur_9, pv_43);
    cur_9 = ((up[3] != 0) ? hi_91 : lo_91);
    V[2] = cur_9;
    float y1_3 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float lo1_3 = fminf(V[3], y1_3);
    r_1 = fmaxf(r_1, lo1_3);
    float hi1_3 = fmaxf(V[3], y1_3);
    float cur_10 = hi1_3;
    float pv_44 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 8);
    float hi_92 = fmaxf(cur_10, pv_44);
    float lo_92 = fminf(cur_10, pv_44);
    cur_10 = ((up[0] != 0) ? hi_92 : lo_92);
    float pv_45 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 4);
    float hi_93 = fmaxf(cur_10, pv_45);
    float lo_93 = fminf(cur_10, pv_45);
    cur_10 = ((up[1] != 0) ? hi_93 : lo_93);
    float pv_46 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 2);
    float hi_94 = fmaxf(cur_10, pv_46);
    float lo_94 = fminf(cur_10, pv_46);
    cur_10 = ((up[2] != 0) ? hi_94 : lo_94);
    float pv_47 = __shfl_xor_sync(0xFFFFFFFF, cur_10, 1);
    float hi_95 = fmaxf(cur_10, pv_47);
    float lo_95 = fminf(cur_10, pv_47);
    cur_10 = ((up[3] != 0) ? hi_95 : lo_95);
    V[3] = cur_10;
    float y1_4 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float lo1_4 = fminf(V[0], y1_4);
    r_1 = fmaxf(r_1, lo1_4);
    float hi1_4 = fmaxf(V[0], y1_4);
    float cur_11 = hi1_4;
    float pv_48 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 8);
    float hi_96 = fmaxf(cur_11, pv_48);
    float lo_96 = fminf(cur_11, pv_48);
    cur_11 = ((up[0] != 0) ? hi_96 : lo_96);
    float pv_49 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 4);
    float hi_97 = fmaxf(cur_11, pv_49);
    float lo_97 = fminf(cur_11, pv_49);
    cur_11 = ((up[1] != 0) ? hi_97 : lo_97);
    float pv_50 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 2);
    float hi_98 = fmaxf(cur_11, pv_50);
    float lo_98 = fminf(cur_11, pv_50);
    cur_11 = ((up[2] != 0) ? hi_98 : lo_98);
    float pv_51 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 1);
    float hi_99 = fmaxf(cur_11, pv_51);
    float lo_99 = fminf(cur_11, pv_51);
    cur_11 = ((up[3] != 0) ? hi_99 : lo_99);
    V[0] = cur_11;
    float y1_5 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float lo1_5 = fminf(V[1], y1_5);
    r_1 = fmaxf(r_1, lo1_5);
    float hi1_5 = fmaxf(V[1], y1_5);
    float cur_12 = hi1_5;
    float pv_52 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 8);
    float hi_100 = fmaxf(cur_12, pv_52);
    float lo_100 = fminf(cur_12, pv_52);
    cur_12 = ((up[0] != 0) ? hi_100 : lo_100);
    float pv_53 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 4);
    float hi_101 = fmaxf(cur_12, pv_53);
    float lo_101 = fminf(cur_12, pv_53);
    cur_12 = ((up[1] != 0) ? hi_101 : lo_101);
    float pv_54 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 2);
    float hi_102 = fmaxf(cur_12, pv_54);
    float lo_102 = fminf(cur_12, pv_54);
    cur_12 = ((up[2] != 0) ? hi_102 : lo_102);
    float pv_55 = __shfl_xor_sync(0xFFFFFFFF, cur_12, 1);
    float hi_103 = fmaxf(cur_12, pv_55);
    float lo_103 = fminf(cur_12, pv_55);
    cur_12 = ((up[3] != 0) ? hi_103 : lo_103);
    V[1] = cur_12;
    float y1_6 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float lo1_6 = fminf(V[0], y1_6);
    r_1 = fmaxf(r_1, lo1_6);
    float hi1_6 = fmaxf(V[0], y1_6);
    V[0] = hi1_6;
    float K = V[0];
    rr1[0] = r_1;
    int ucol = blockIdx.x * 32 + cg;
    int commit = 0;
    if (g < 32 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 512) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        u16 = fminf(u16, _shfl_xor_sync_0);
        float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        cr = fmaxf(cr, _shfl_xor_sync_1);
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        u16 = fminf(u16, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        cr = fmaxf(cr, _shfl_xor_sync_3);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        u16 = fminf(u16, _shfl_xor_sync_4);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        cr = fmaxf(cr, _shfl_xor_sync_5);
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        u16 = fminf(u16, _shfl_xor_sync_6);
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        cr = fmaxf(cr, _shfl_xor_sync_7);
        unsigned int u16b = __as_u32(u16);
        unsigned int c16 = u16b & 4294965248u;
        unsigned int crb = __as_u32(cr) & 4294965248u;
        unsigned int q = 4294967295u;
        if (c16 == crb && u16b < 4278190080u && c16 != 2139092992 && commit != 0) {
            q = c16;
            if (ln == 0) {
                flagw[0] = 1;
            }
        }
        if (ln == 0 && g < 32) {
            qcol[cg] = q;
        }
        if (u16b >= 4278190080u && commit != 0 && ln == 0) {
            flagw[2] = 1;
        }
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    unsigned int need2 = flagw[0];
    unsigned int qc = qcol[c];
    unsigned int fz = flagw[2];
    if (fz != 0) {
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
        rej = neg_inf;
        int t0_4 = w * 16;
        long long p_1 = cbase + (long long)t0_4 * nq64;
        cb[0] = 4286578688u;
        if (lim_full > t0_4) {
            cb[0] = S[p_1];
        }
        cb[1] = 4286578688u;
        if (lim_full > t0_4 + 1) {
            cb[1] = S[p_1 + nq64];
        }
        cb[2] = 4286578688u;
        if (lim_full > t0_4 + 2) {
            cb[2] = S[p_1 + 2ll * nq64];
        }
        cb[3] = 4286578688u;
        if (lim_full > t0_4 + 3) {
            cb[3] = S[p_1 + 3ll * nq64];
        }
        cb[4] = 4286578688u;
        if (lim_full > t0_4 + 4) {
            cb[4] = S[p_1 + 4ll * nq64];
        }
        cb[5] = 4286578688u;
        if (lim_full > t0_4 + 5) {
            cb[5] = S[p_1 + 5ll * nq64];
        }
        cb[6] = 4286578688u;
        if (lim_full > t0_4 + 6) {
            cb[6] = S[p_1 + 6ll * nq64];
        }
        cb[7] = 4286578688u;
        if (lim_full > t0_4 + 7) {
            cb[7] = S[p_1 + 7ll * nq64];
        }
        cb[8] = 4286578688u;
        if (lim_full > t0_4 + 8) {
            cb[8] = S[p_1 + 8ll * nq64];
        }
        cb[9] = 4286578688u;
        if (lim_full > t0_4 + 9) {
            cb[9] = S[p_1 + 9ll * nq64];
        }
        cb[10] = 4286578688u;
        if (lim_full > t0_4 + 10) {
            cb[10] = S[p_1 + 10ll * nq64];
        }
        cb[11] = 4286578688u;
        if (lim_full > t0_4 + 11) {
            cb[11] = S[p_1 + 11ll * nq64];
        }
        cb[12] = 4286578688u;
        if (lim_full > t0_4 + 12) {
            cb[12] = S[p_1 + 12ll * nq64];
        }
        cb[13] = 4286578688u;
        if (lim_full > t0_4 + 13) {
            cb[13] = S[p_1 + 13ll * nq64];
        }
        cb[14] = 4286578688u;
        if (lim_full > t0_4 + 14) {
            cb[14] = S[p_1 + 14ll * nq64];
        }
        cb[15] = 4286578688u;
        if (lim_full > t0_4 + 15) {
            cb[15] = S[p_1 + 15ll * nq64];
        }
        asm volatile("" ::: "memory");
        if (lim <= t0_1) {
            cb[0] = 4286578688u;
        }
        if (lim <= t0_1 + 1) {
            cb[1] = 4286578688u;
        }
        if (lim <= t0_1 + 2) {
            cb[2] = 4286578688u;
        }
        if (lim <= t0_1 + 3) {
            cb[3] = 4286578688u;
        }
        if (lim <= t0_1 + 4) {
            cb[4] = 4286578688u;
        }
        if (lim <= t0_1 + 5) {
            cb[5] = 4286578688u;
        }
        if (lim <= t0_1 + 6) {
            cb[6] = 4286578688u;
        }
        if (lim <= t0_1 + 7) {
            cb[7] = 4286578688u;
        }
        if (lim <= t0_1 + 8) {
            cb[8] = 4286578688u;
        }
        if (lim <= t0_1 + 9) {
            cb[9] = 4286578688u;
        }
        if (lim <= t0_1 + 10) {
            cb[10] = 4286578688u;
        }
        if (lim <= t0_1 + 11) {
            cb[11] = 4286578688u;
        }
        if (lim <= t0_1 + 12) {
            cb[12] = 4286578688u;
        }
        if (lim <= t0_1 + 13) {
            cb[13] = 4286578688u;
        }
        if (lim <= t0_1 + 14) {
            cb[14] = 4286578688u;
        }
        if (lim <= t0_1 + 15) {
            cb[15] = 4286578688u;
        }
        #pragma unroll 1
        for (int jz = 0; jz < num_chunks; jz++) {
            int t0_5 = ((jz + 1) * 16 + w) * 16;
            long long p_2 = cbase + (long long)t0_5 * nq64;
            nb[0] = 4286578688u;
            if (lim > t0_5) {
                nb[0] = S[p_2];
            }
            nb[1] = 4286578688u;
            if (lim > t0_5 + 1) {
                nb[1] = S[p_2 + nq64];
            }
            nb[2] = 4286578688u;
            if (lim > t0_5 + 2) {
                nb[2] = S[p_2 + 2ll * nq64];
            }
            nb[3] = 4286578688u;
            if (lim > t0_5 + 3) {
                nb[3] = S[p_2 + 3ll * nq64];
            }
            nb[4] = 4286578688u;
            if (lim > t0_5 + 4) {
                nb[4] = S[p_2 + 4ll * nq64];
            }
            nb[5] = 4286578688u;
            if (lim > t0_5 + 5) {
                nb[5] = S[p_2 + 5ll * nq64];
            }
            nb[6] = 4286578688u;
            if (lim > t0_5 + 6) {
                nb[6] = S[p_2 + 6ll * nq64];
            }
            nb[7] = 4286578688u;
            if (lim > t0_5 + 7) {
                nb[7] = S[p_2 + 7ll * nq64];
            }
            nb[8] = 4286578688u;
            if (lim > t0_5 + 8) {
                nb[8] = S[p_2 + 8ll * nq64];
            }
            nb[9] = 4286578688u;
            if (lim > t0_5 + 9) {
                nb[9] = S[p_2 + 9ll * nq64];
            }
            nb[10] = 4286578688u;
            if (lim > t0_5 + 10) {
                nb[10] = S[p_2 + 10ll * nq64];
            }
            nb[11] = 4286578688u;
            if (lim > t0_5 + 11) {
                nb[11] = S[p_2 + 11ll * nq64];
            }
            nb[12] = 4286578688u;
            if (lim > t0_5 + 12) {
                nb[12] = S[p_2 + 12ll * nq64];
            }
            nb[13] = 4286578688u;
            if (lim > t0_5 + 13) {
                nb[13] = S[p_2 + 13ll * nq64];
            }
            nb[14] = 4286578688u;
            if (lim > t0_5 + 14) {
                nb[14] = S[p_2 + 14ll * nq64];
            }
            nb[15] = 4286578688u;
            if (lim > t0_5 + 15) {
                nb[15] = S[p_2 + 15ll * nq64];
            }
            asm volatile("" ::: "memory");
            int t0_6 = (jz * 16 + w) * 16;
            float sc_15 = __uint_as_float(cb[0]);
            sc_15 = fmaxf(sc_15, -1.7014118e+38f);
            sc_15 = fminf(sc_15, 1.7014118e+38f);
            unsigned int kbits = __as_u32(sc_15);
            if (cb[0] == 4286578688u) {
                kbits = 4282384384u;
            }
            unsigned int key_1_16 = kbits & 4294965248u | (unsigned int)t0_6;
            kb[0] = __uint_as_float(key_1_16);
            float sc_15_1 = __uint_as_float(cb[1]);
            sc_15_1 = fmaxf(sc_15_1, -1.7014118e+38f);
            sc_15_1 = fminf(sc_15_1, 1.7014118e+38f);
            unsigned int kbits_0 = __as_u32(sc_15_1);
            if (cb[1] == 4286578688u) {
                kbits_0 = 4282384384u;
            }
            unsigned int key_1_17 = kbits_0 & 4294965248u | (unsigned int)(t0_6 + 1);
            kb[1] = __uint_as_float(key_1_17);
            float sc_16 = __uint_as_float(cb[2]);
            sc_16 = fmaxf(sc_16, -1.7014118e+38f);
            sc_16 = fminf(sc_16, 1.7014118e+38f);
            unsigned int kbits_1 = __as_u32(sc_16);
            if (cb[2] == 4286578688u) {
                kbits_1 = 4282384384u;
            }
            unsigned int key_1_18 = kbits_1 & 4294965248u | (unsigned int)(t0_6 + 2);
            kb[2] = __uint_as_float(key_1_18);
            float sc_17 = __uint_as_float(cb[3]);
            sc_17 = fmaxf(sc_17, -1.7014118e+38f);
            sc_17 = fminf(sc_17, 1.7014118e+38f);
            unsigned int kbits_2 = __as_u32(sc_17);
            if (cb[3] == 4286578688u) {
                kbits_2 = 4282384384u;
            }
            unsigned int key_1_19 = kbits_2 & 4294965248u | (unsigned int)(t0_6 + 3);
            kb[3] = __uint_as_float(key_1_19);
            float sc_18 = __uint_as_float(cb[4]);
            sc_18 = fmaxf(sc_18, -1.7014118e+38f);
            sc_18 = fminf(sc_18, 1.7014118e+38f);
            unsigned int kbits_3 = __as_u32(sc_18);
            if (cb[4] == 4286578688u) {
                kbits_3 = 4282384384u;
            }
            unsigned int key_1_20 = kbits_3 & 4294965248u | (unsigned int)(t0_6 + 4);
            kb[4] = __uint_as_float(key_1_20);
            float sc_19 = __uint_as_float(cb[5]);
            sc_19 = fmaxf(sc_19, -1.7014118e+38f);
            sc_19 = fminf(sc_19, 1.7014118e+38f);
            unsigned int kbits_4 = __as_u32(sc_19);
            if (cb[5] == 4286578688u) {
                kbits_4 = 4282384384u;
            }
            unsigned int key_1_21 = kbits_4 & 4294965248u | (unsigned int)(t0_6 + 5);
            kb[5] = __uint_as_float(key_1_21);
            float sc_20 = __uint_as_float(cb[6]);
            sc_20 = fmaxf(sc_20, -1.7014118e+38f);
            sc_20 = fminf(sc_20, 1.7014118e+38f);
            unsigned int kbits_5 = __as_u32(sc_20);
            if (cb[6] == 4286578688u) {
                kbits_5 = 4282384384u;
            }
            unsigned int key_1_22 = kbits_5 & 4294965248u | (unsigned int)(t0_6 + 6);
            kb[6] = __uint_as_float(key_1_22);
            float sc_21 = __uint_as_float(cb[7]);
            sc_21 = fmaxf(sc_21, -1.7014118e+38f);
            sc_21 = fminf(sc_21, 1.7014118e+38f);
            unsigned int kbits_6 = __as_u32(sc_21);
            if (cb[7] == 4286578688u) {
                kbits_6 = 4282384384u;
            }
            unsigned int key_1_23 = kbits_6 & 4294965248u | (unsigned int)(t0_6 + 7);
            kb[7] = __uint_as_float(key_1_23);
            float sc_22 = __uint_as_float(cb[8]);
            sc_22 = fmaxf(sc_22, -1.7014118e+38f);
            sc_22 = fminf(sc_22, 1.7014118e+38f);
            unsigned int kbits_7 = __as_u32(sc_22);
            if (cb[8] == 4286578688u) {
                kbits_7 = 4282384384u;
            }
            unsigned int key_1_24 = kbits_7 & 4294965248u | (unsigned int)(t0_6 + 8);
            kb[8] = __uint_as_float(key_1_24);
            float sc_23 = __uint_as_float(cb[9]);
            sc_23 = fmaxf(sc_23, -1.7014118e+38f);
            sc_23 = fminf(sc_23, 1.7014118e+38f);
            unsigned int kbits_8 = __as_u32(sc_23);
            if (cb[9] == 4286578688u) {
                kbits_8 = 4282384384u;
            }
            unsigned int key_1_25 = kbits_8 & 4294965248u | (unsigned int)(t0_6 + 9);
            kb[9] = __uint_as_float(key_1_25);
            float sc_24 = __uint_as_float(cb[10]);
            sc_24 = fmaxf(sc_24, -1.7014118e+38f);
            sc_24 = fminf(sc_24, 1.7014118e+38f);
            unsigned int kbits_9 = __as_u32(sc_24);
            if (cb[10] == 4286578688u) {
                kbits_9 = 4282384384u;
            }
            unsigned int key_1_26 = kbits_9 & 4294965248u | (unsigned int)(t0_6 + 10);
            kb[10] = __uint_as_float(key_1_26);
            float sc_25 = __uint_as_float(cb[11]);
            sc_25 = fmaxf(sc_25, -1.7014118e+38f);
            sc_25 = fminf(sc_25, 1.7014118e+38f);
            unsigned int kbits_10 = __as_u32(sc_25);
            if (cb[11] == 4286578688u) {
                kbits_10 = 4282384384u;
            }
            unsigned int key_1_27 = kbits_10 & 4294965248u | (unsigned int)(t0_6 + 11);
            kb[11] = __uint_as_float(key_1_27);
            float sc_26 = __uint_as_float(cb[12]);
            sc_26 = fmaxf(sc_26, -1.7014118e+38f);
            sc_26 = fminf(sc_26, 1.7014118e+38f);
            unsigned int kbits_11 = __as_u32(sc_26);
            if (cb[12] == 4286578688u) {
                kbits_11 = 4282384384u;
            }
            unsigned int key_1_28 = kbits_11 & 4294965248u | (unsigned int)(t0_6 + 12);
            kb[12] = __uint_as_float(key_1_28);
            float sc_27 = __uint_as_float(cb[13]);
            sc_27 = fmaxf(sc_27, -1.7014118e+38f);
            sc_27 = fminf(sc_27, 1.7014118e+38f);
            unsigned int kbits_12 = __as_u32(sc_27);
            if (cb[13] == 4286578688u) {
                kbits_12 = 4282384384u;
            }
            unsigned int key_1_29 = kbits_12 & 4294965248u | (unsigned int)(t0_6 + 13);
            kb[13] = __uint_as_float(key_1_29);
            float sc_28 = __uint_as_float(cb[14]);
            sc_28 = fmaxf(sc_28, -1.7014118e+38f);
            sc_28 = fminf(sc_28, 1.7014118e+38f);
            unsigned int kbits_13 = __as_u32(sc_28);
            if (cb[14] == 4286578688u) {
                kbits_13 = 4282384384u;
            }
            unsigned int key_1_30 = kbits_13 & 4294965248u | (unsigned int)(t0_6 + 14);
            kb[14] = __uint_as_float(key_1_30);
            float sc_29 = __uint_as_float(cb[15]);
            sc_29 = fmaxf(sc_29, -1.7014118e+38f);
            sc_29 = fminf(sc_29, 1.7014118e+38f);
            unsigned int kbits_14 = __as_u32(sc_29);
            if (cb[15] == 4286578688u) {
                kbits_14 = 4282384384u;
            }
            unsigned int key_1_31 = kbits_14 & 4294965248u | (unsigned int)(t0_6 + 15);
            kb[15] = __uint_as_float(key_1_31);
            int f_16 = 0;
            if ((t0_6 < fb || t0_6 + 16 > lim - fe) && t0_6 < lim) {
                f_16 = 1;
            }
            if (f_16 != 0) {
                int f_16_1 = 0;
                if ((t0_6 < fb || t0_6 >= lim - fe) && lim > t0_6) {
                    f_16_1 = 1;
                }
                if (f_16_1 != 0) {
                    kb[0] = __uint_as_float(2139092992 | (unsigned int)t0_6);
                }
                int f_17 = 0;
                if ((t0_6 + 1 < fb || t0_6 + 1 >= lim - fe) && lim > t0_6 + 1) {
                    f_17 = 1;
                }
                if (f_17 != 0) {
                    kb[1] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 1));
                }
                int f_18 = 0;
                if ((t0_6 + 2 < fb || t0_6 + 2 >= lim - fe) && lim > t0_6 + 2) {
                    f_18 = 1;
                }
                if (f_18 != 0) {
                    kb[2] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 2));
                }
                int f_19 = 0;
                if ((t0_6 + 3 < fb || t0_6 + 3 >= lim - fe) && lim > t0_6 + 3) {
                    f_19 = 1;
                }
                if (f_19 != 0) {
                    kb[3] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 3));
                }
                int f_20 = 0;
                if ((t0_6 + 4 < fb || t0_6 + 4 >= lim - fe) && lim > t0_6 + 4) {
                    f_20 = 1;
                }
                if (f_20 != 0) {
                    kb[4] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 4));
                }
                int f_21 = 0;
                if ((t0_6 + 5 < fb || t0_6 + 5 >= lim - fe) && lim > t0_6 + 5) {
                    f_21 = 1;
                }
                if (f_21 != 0) {
                    kb[5] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 5));
                }
                int f_22 = 0;
                if ((t0_6 + 6 < fb || t0_6 + 6 >= lim - fe) && lim > t0_6 + 6) {
                    f_22 = 1;
                }
                if (f_22 != 0) {
                    kb[6] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 6));
                }
                int f_23 = 0;
                if ((t0_6 + 7 < fb || t0_6 + 7 >= lim - fe) && lim > t0_6 + 7) {
                    f_23 = 1;
                }
                if (f_23 != 0) {
                    kb[7] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 7));
                }
                int f_24 = 0;
                if ((t0_6 + 8 < fb || t0_6 + 8 >= lim - fe) && lim > t0_6 + 8) {
                    f_24 = 1;
                }
                if (f_24 != 0) {
                    kb[8] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 8));
                }
                int f_25 = 0;
                if ((t0_6 + 9 < fb || t0_6 + 9 >= lim - fe) && lim > t0_6 + 9) {
                    f_25 = 1;
                }
                if (f_25 != 0) {
                    kb[9] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 9));
                }
                int f_26 = 0;
                if ((t0_6 + 10 < fb || t0_6 + 10 >= lim - fe) && lim > t0_6 + 10) {
                    f_26 = 1;
                }
                if (f_26 != 0) {
                    kb[10] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 10));
                }
                int f_27 = 0;
                if ((t0_6 + 11 < fb || t0_6 + 11 >= lim - fe) && lim > t0_6 + 11) {
                    f_27 = 1;
                }
                if (f_27 != 0) {
                    kb[11] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 11));
                }
                int f_28 = 0;
                if ((t0_6 + 12 < fb || t0_6 + 12 >= lim - fe) && lim > t0_6 + 12) {
                    f_28 = 1;
                }
                if (f_28 != 0) {
                    kb[12] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 12));
                }
                int f_29 = 0;
                if ((t0_6 + 13 < fb || t0_6 + 13 >= lim - fe) && lim > t0_6 + 13) {
                    f_29 = 1;
                }
                if (f_29 != 0) {
                    kb[13] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 13));
                }
                int f_30 = 0;
                if ((t0_6 + 14 < fb || t0_6 + 14 >= lim - fe) && lim > t0_6 + 14) {
                    f_30 = 1;
                }
                if (f_30 != 0) {
                    kb[14] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 14));
                }
                int f_31 = 0;
                if ((t0_6 + 15 < fb || t0_6 + 15 >= lim - fe) && lim > t0_6 + 15) {
                    f_31 = 1;
                }
                if (f_31 != 0) {
                    kb[15] = __uint_as_float(2139092992 | (unsigned int)(t0_6 + 15));
                }
            }
            float hi_1_60 = fmaxf(kb[0], kb[13]);
            float lo_1_60 = fminf(kb[0], kb[13]);
            kb[0] = hi_1_60;
            kb[13] = lo_1_60;
            float hi_1_61 = fmaxf(kb[1], kb[12]);
            float lo_1_61 = fminf(kb[1], kb[12]);
            kb[1] = hi_1_61;
            kb[12] = lo_1_61;
            float hi_1_62 = fmaxf(kb[2], kb[15]);
            float lo_1_62 = fminf(kb[2], kb[15]);
            kb[2] = hi_1_62;
            kb[15] = lo_1_62;
            float hi_1_63 = fmaxf(kb[3], kb[14]);
            float lo_1_63 = fminf(kb[3], kb[14]);
            kb[3] = hi_1_63;
            kb[14] = lo_1_63;
            float hi_1_64 = fmaxf(kb[4], kb[8]);
            float lo_1_64 = fminf(kb[4], kb[8]);
            kb[4] = hi_1_64;
            kb[8] = lo_1_64;
            float hi_1_65 = fmaxf(kb[5], kb[6]);
            float lo_1_65 = fminf(kb[5], kb[6]);
            kb[5] = hi_1_65;
            kb[6] = lo_1_65;
            float hi_1_66 = fmaxf(kb[7], kb[11]);
            float lo_1_66 = fminf(kb[7], kb[11]);
            kb[7] = hi_1_66;
            kb[11] = lo_1_66;
            float hi_1_67 = fmaxf(kb[9], kb[10]);
            float lo_1_67 = fminf(kb[9], kb[10]);
            kb[9] = hi_1_67;
            kb[10] = lo_1_67;
            float hi_1_68 = fmaxf(kb[0], kb[5]);
            float lo_1_68 = fminf(kb[0], kb[5]);
            kb[0] = hi_1_68;
            kb[5] = lo_1_68;
            float hi_1_69 = fmaxf(kb[1], kb[7]);
            float lo_1_69 = fminf(kb[1], kb[7]);
            kb[1] = hi_1_69;
            kb[7] = lo_1_69;
            float hi_1_70 = fmaxf(kb[2], kb[9]);
            float lo_1_70 = fminf(kb[2], kb[9]);
            kb[2] = hi_1_70;
            kb[9] = lo_1_70;
            float hi_1_71 = fmaxf(kb[3], kb[4]);
            float lo_1_71 = fminf(kb[3], kb[4]);
            kb[3] = hi_1_71;
            kb[4] = lo_1_71;
            float hi_1_72 = fmaxf(kb[6], kb[13]);
            float lo_1_72 = fminf(kb[6], kb[13]);
            kb[6] = hi_1_72;
            kb[13] = lo_1_72;
            float hi_1_73 = fmaxf(kb[8], kb[14]);
            float lo_1_73 = fminf(kb[8], kb[14]);
            kb[8] = hi_1_73;
            kb[14] = lo_1_73;
            float hi_1_74 = fmaxf(kb[10], kb[15]);
            float lo_1_74 = fminf(kb[10], kb[15]);
            kb[10] = hi_1_74;
            kb[15] = lo_1_74;
            float hi_1_75 = fmaxf(kb[11], kb[12]);
            float lo_1_75 = fminf(kb[11], kb[12]);
            kb[11] = hi_1_75;
            kb[12] = lo_1_75;
            float hi_1_76 = fmaxf(kb[0], kb[1]);
            float lo_1_76 = fminf(kb[0], kb[1]);
            kb[0] = hi_1_76;
            kb[1] = lo_1_76;
            float hi_1_77 = fmaxf(kb[2], kb[3]);
            float lo_1_77 = fminf(kb[2], kb[3]);
            kb[2] = hi_1_77;
            kb[3] = lo_1_77;
            float hi_1_78 = fmaxf(kb[4], kb[5]);
            float lo_1_78 = fminf(kb[4], kb[5]);
            kb[4] = hi_1_78;
            kb[5] = lo_1_78;
            float hi_1_79 = fmaxf(kb[6], kb[8]);
            float lo_1_79 = fminf(kb[6], kb[8]);
            kb[6] = hi_1_79;
            kb[8] = lo_1_79;
            float hi_1_80 = fmaxf(kb[7], kb[9]);
            float lo_1_80 = fminf(kb[7], kb[9]);
            kb[7] = hi_1_80;
            kb[9] = lo_1_80;
            float hi_1_81 = fmaxf(kb[10], kb[11]);
            float lo_1_81 = fminf(kb[10], kb[11]);
            kb[10] = hi_1_81;
            kb[11] = lo_1_81;
            float hi_1_82 = fmaxf(kb[12], kb[13]);
            float lo_1_82 = fminf(kb[12], kb[13]);
            kb[12] = hi_1_82;
            kb[13] = lo_1_82;
            float hi_1_83 = fmaxf(kb[14], kb[15]);
            float lo_1_83 = fminf(kb[14], kb[15]);
            kb[14] = hi_1_83;
            kb[15] = lo_1_83;
            float hi_1_84 = fmaxf(kb[0], kb[2]);
            float lo_1_84 = fminf(kb[0], kb[2]);
            kb[0] = hi_1_84;
            kb[2] = lo_1_84;
            float hi_1_85 = fmaxf(kb[1], kb[3]);
            float lo_1_85 = fminf(kb[1], kb[3]);
            kb[1] = hi_1_85;
            kb[3] = lo_1_85;
            float hi_1_86 = fmaxf(kb[4], kb[10]);
            float lo_1_86 = fminf(kb[4], kb[10]);
            kb[4] = hi_1_86;
            kb[10] = lo_1_86;
            float hi_1_87 = fmaxf(kb[5], kb[11]);
            float lo_1_87 = fminf(kb[5], kb[11]);
            kb[5] = hi_1_87;
            kb[11] = lo_1_87;
            float hi_1_88 = fmaxf(kb[6], kb[7]);
            float lo_1_88 = fminf(kb[6], kb[7]);
            kb[6] = hi_1_88;
            kb[7] = lo_1_88;
            float hi_1_89 = fmaxf(kb[8], kb[9]);
            float lo_1_89 = fminf(kb[8], kb[9]);
            kb[8] = hi_1_89;
            kb[9] = lo_1_89;
            float hi_1_90 = fmaxf(kb[12], kb[14]);
            float lo_1_90 = fminf(kb[12], kb[14]);
            kb[12] = hi_1_90;
            kb[14] = lo_1_90;
            float hi_1_91 = fmaxf(kb[13], kb[15]);
            float lo_1_91 = fminf(kb[13], kb[15]);
            kb[13] = hi_1_91;
            kb[15] = lo_1_91;
            float hi_1_92 = fmaxf(kb[1], kb[2]);
            float lo_1_92 = fminf(kb[1], kb[2]);
            kb[1] = hi_1_92;
            kb[2] = lo_1_92;
            float hi_1_93 = fmaxf(kb[3], kb[12]);
            float lo_1_93 = fminf(kb[3], kb[12]);
            kb[3] = hi_1_93;
            kb[12] = lo_1_93;
            float hi_1_94 = fmaxf(kb[4], kb[6]);
            float lo_1_94 = fminf(kb[4], kb[6]);
            kb[4] = hi_1_94;
            kb[6] = lo_1_94;
            float hi_1_95 = fmaxf(kb[5], kb[7]);
            float lo_1_95 = fminf(kb[5], kb[7]);
            kb[5] = hi_1_95;
            kb[7] = lo_1_95;
            float hi_1_96 = fmaxf(kb[8], kb[10]);
            float lo_1_96 = fminf(kb[8], kb[10]);
            kb[8] = hi_1_96;
            kb[10] = lo_1_96;
            float hi_1_97 = fmaxf(kb[9], kb[11]);
            float lo_1_97 = fminf(kb[9], kb[11]);
            kb[9] = hi_1_97;
            kb[11] = lo_1_97;
            float hi_1_98 = fmaxf(kb[13], kb[14]);
            float lo_1_98 = fminf(kb[13], kb[14]);
            kb[13] = hi_1_98;
            kb[14] = lo_1_98;
            float hi_1_99 = fmaxf(kb[1], kb[4]);
            float lo_1_99 = fminf(kb[1], kb[4]);
            kb[1] = hi_1_99;
            kb[4] = lo_1_99;
            float hi_1_100 = fmaxf(kb[2], kb[6]);
            float lo_1_100 = fminf(kb[2], kb[6]);
            kb[2] = hi_1_100;
            kb[6] = lo_1_100;
            float hi_1_101 = fmaxf(kb[5], kb[8]);
            float lo_1_101 = fminf(kb[5], kb[8]);
            kb[5] = hi_1_101;
            kb[8] = lo_1_101;
            float hi_1_102 = fmaxf(kb[7], kb[10]);
            float lo_1_102 = fminf(kb[7], kb[10]);
            kb[7] = hi_1_102;
            kb[10] = lo_1_102;
            float hi_1_103 = fmaxf(kb[9], kb[13]);
            float lo_1_103 = fminf(kb[9], kb[13]);
            kb[9] = hi_1_103;
            kb[13] = lo_1_103;
            float hi_1_104 = fmaxf(kb[11], kb[14]);
            float lo_1_104 = fminf(kb[11], kb[14]);
            kb[11] = hi_1_104;
            kb[14] = lo_1_104;
            float hi_1_105 = fmaxf(kb[2], kb[4]);
            float lo_1_105 = fminf(kb[2], kb[4]);
            kb[2] = hi_1_105;
            kb[4] = lo_1_105;
            float hi_1_106 = fmaxf(kb[3], kb[6]);
            float lo_1_106 = fminf(kb[3], kb[6]);
            kb[3] = hi_1_106;
            kb[6] = lo_1_106;
            float hi_1_107 = fmaxf(kb[9], kb[12]);
            float lo_1_107 = fminf(kb[9], kb[12]);
            kb[9] = hi_1_107;
            kb[12] = lo_1_107;
            float hi_1_108 = fmaxf(kb[11], kb[13]);
            float lo_1_108 = fminf(kb[11], kb[13]);
            kb[11] = hi_1_108;
            kb[13] = lo_1_108;
            float hi_1_109 = fmaxf(kb[3], kb[5]);
            float lo_1_109 = fminf(kb[3], kb[5]);
            kb[3] = hi_1_109;
            kb[5] = lo_1_109;
            float hi_1_110 = fmaxf(kb[6], kb[8]);
            float lo_1_110 = fminf(kb[6], kb[8]);
            kb[6] = hi_1_110;
            kb[8] = lo_1_110;
            float hi_1_111 = fmaxf(kb[7], kb[9]);
            float lo_1_111 = fminf(kb[7], kb[9]);
            kb[7] = hi_1_111;
            kb[9] = lo_1_111;
            float hi_1_112 = fmaxf(kb[10], kb[12]);
            float lo_1_112 = fminf(kb[10], kb[12]);
            kb[10] = hi_1_112;
            kb[12] = lo_1_112;
            float hi_1_113 = fmaxf(kb[3], kb[4]);
            float lo_1_113 = fminf(kb[3], kb[4]);
            kb[3] = hi_1_113;
            kb[4] = lo_1_113;
            float hi_1_114 = fmaxf(kb[5], kb[6]);
            float lo_1_114 = fminf(kb[5], kb[6]);
            kb[5] = hi_1_114;
            kb[6] = lo_1_114;
            float hi_1_115 = fmaxf(kb[7], kb[8]);
            float lo_1_115 = fminf(kb[7], kb[8]);
            kb[7] = hi_1_115;
            kb[8] = lo_1_115;
            float hi_1_116 = fmaxf(kb[9], kb[10]);
            float lo_1_116 = fminf(kb[9], kb[10]);
            kb[9] = hi_1_116;
            kb[10] = lo_1_116;
            float hi_1_117 = fmaxf(kb[11], kb[12]);
            float lo_1_117 = fminf(kb[11], kb[12]);
            kb[11] = hi_1_117;
            kb[12] = lo_1_117;
            float hi_1_118 = fmaxf(kb[6], kb[7]);
            float lo_1_118 = fminf(kb[6], kb[7]);
            kb[6] = hi_1_118;
            kb[7] = lo_1_118;
            float hi_1_119 = fmaxf(kb[8], kb[9]);
            float lo_1_119 = fminf(kb[8], kb[9]);
            kb[8] = hi_1_119;
            kb[9] = lo_1_119;
            float r_0 = rej;
            float hi_104 = fmaxf(a[0], kb[15]);
            float lo_104 = fminf(a[0], kb[15]);
            a[0] = hi_104;
            r_0 = fmaxf(r_0, lo_104);
            float hi_105 = fmaxf(a[1], kb[14]);
            float lo_105 = fminf(a[1], kb[14]);
            a[1] = hi_105;
            r_0 = fmaxf(r_0, lo_105);
            float hi_106 = fmaxf(a[2], kb[13]);
            float lo_106 = fminf(a[2], kb[13]);
            a[2] = hi_106;
            r_0 = fmaxf(r_0, lo_106);
            float hi_107 = fmaxf(a[3], kb[12]);
            float lo_107 = fminf(a[3], kb[12]);
            a[3] = hi_107;
            r_0 = fmaxf(r_0, lo_107);
            float hi_108 = fmaxf(a[4], kb[11]);
            float lo_108 = fminf(a[4], kb[11]);
            a[4] = hi_108;
            r_0 = fmaxf(r_0, lo_108);
            float hi_109 = fmaxf(a[5], kb[10]);
            float lo_109 = fminf(a[5], kb[10]);
            a[5] = hi_109;
            r_0 = fmaxf(r_0, lo_109);
            float hi_110 = fmaxf(a[6], kb[9]);
            float lo_110 = fminf(a[6], kb[9]);
            a[6] = hi_110;
            r_0 = fmaxf(r_0, lo_110);
            float hi_111 = fmaxf(a[7], kb[8]);
            float lo_111 = fminf(a[7], kb[8]);
            a[7] = hi_111;
            r_0 = fmaxf(r_0, lo_111);
            float hi_112 = fmaxf(a[8], kb[7]);
            float lo_112 = fminf(a[8], kb[7]);
            a[8] = hi_112;
            r_0 = fmaxf(r_0, lo_112);
            float hi_113 = fmaxf(a[9], kb[6]);
            float lo_113 = fminf(a[9], kb[6]);
            a[9] = hi_113;
            r_0 = fmaxf(r_0, lo_113);
            float hi_114 = fmaxf(a[10], kb[5]);
            float lo_114 = fminf(a[10], kb[5]);
            a[10] = hi_114;
            r_0 = fmaxf(r_0, lo_114);
            float hi_115 = fmaxf(a[11], kb[4]);
            float lo_115 = fminf(a[11], kb[4]);
            a[11] = hi_115;
            r_0 = fmaxf(r_0, lo_115);
            float hi_116 = fmaxf(a[12], kb[3]);
            float lo_116 = fminf(a[12], kb[3]);
            a[12] = hi_116;
            r_0 = fmaxf(r_0, lo_116);
            float hi_117 = fmaxf(a[13], kb[2]);
            float lo_117 = fminf(a[13], kb[2]);
            a[13] = hi_117;
            r_0 = fmaxf(r_0, lo_117);
            float hi_118 = fmaxf(a[14], kb[1]);
            float lo_118 = fminf(a[14], kb[1]);
            a[14] = hi_118;
            r_0 = fmaxf(r_0, lo_118);
            float hi_119 = fmaxf(a[15], kb[0]);
            float lo_119 = fminf(a[15], kb[0]);
            a[15] = hi_119;
            r_0 = fmaxf(r_0, lo_119);
            float hi_120 = fmaxf(a[0], a[8]);
            float lo_120 = fminf(a[0], a[8]);
            a[0] = hi_120;
            a[8] = lo_120;
            float hi_121 = fmaxf(a[1], a[9]);
            float lo_121 = fminf(a[1], a[9]);
            a[1] = hi_121;
            a[9] = lo_121;
            float hi_122 = fmaxf(a[2], a[10]);
            float lo_122 = fminf(a[2], a[10]);
            a[2] = hi_122;
            a[10] = lo_122;
            float hi_123 = fmaxf(a[3], a[11]);
            float lo_123 = fminf(a[3], a[11]);
            a[3] = hi_123;
            a[11] = lo_123;
            float hi_124 = fmaxf(a[4], a[12]);
            float lo_124 = fminf(a[4], a[12]);
            a[4] = hi_124;
            a[12] = lo_124;
            float hi_125 = fmaxf(a[5], a[13]);
            float lo_125 = fminf(a[5], a[13]);
            a[5] = hi_125;
            a[13] = lo_125;
            float hi_126 = fmaxf(a[6], a[14]);
            float lo_126 = fminf(a[6], a[14]);
            a[6] = hi_126;
            a[14] = lo_126;
            float hi_127 = fmaxf(a[7], a[15]);
            float lo_127 = fminf(a[7], a[15]);
            a[7] = hi_127;
            a[15] = lo_127;
            float hi_128 = fmaxf(a[0], a[4]);
            float lo_128 = fminf(a[0], a[4]);
            a[0] = hi_128;
            a[4] = lo_128;
            float hi_129 = fmaxf(a[1], a[5]);
            float lo_129 = fminf(a[1], a[5]);
            a[1] = hi_129;
            a[5] = lo_129;
            float hi_130 = fmaxf(a[2], a[6]);
            float lo_130 = fminf(a[2], a[6]);
            a[2] = hi_130;
            a[6] = lo_130;
            float hi_131 = fmaxf(a[3], a[7]);
            float lo_131 = fminf(a[3], a[7]);
            a[3] = hi_131;
            a[7] = lo_131;
            float hi_132 = fmaxf(a[8], a[12]);
            float lo_132 = fminf(a[8], a[12]);
            a[8] = hi_132;
            a[12] = lo_132;
            float hi_133 = fmaxf(a[9], a[13]);
            float lo_133 = fminf(a[9], a[13]);
            a[9] = hi_133;
            a[13] = lo_133;
            float hi_134 = fmaxf(a[10], a[14]);
            float lo_134 = fminf(a[10], a[14]);
            a[10] = hi_134;
            a[14] = lo_134;
            float hi_135 = fmaxf(a[11], a[15]);
            float lo_135 = fminf(a[11], a[15]);
            a[11] = hi_135;
            a[15] = lo_135;
            float hi_136 = fmaxf(a[0], a[2]);
            float lo_136 = fminf(a[0], a[2]);
            a[0] = hi_136;
            a[2] = lo_136;
            float hi_137 = fmaxf(a[1], a[3]);
            float lo_137 = fminf(a[1], a[3]);
            a[1] = hi_137;
            a[3] = lo_137;
            float hi_138 = fmaxf(a[4], a[6]);
            float lo_138 = fminf(a[4], a[6]);
            a[4] = hi_138;
            a[6] = lo_138;
            float hi_139 = fmaxf(a[5], a[7]);
            float lo_139 = fminf(a[5], a[7]);
            a[5] = hi_139;
            a[7] = lo_139;
            float hi_140 = fmaxf(a[8], a[10]);
            float lo_140 = fminf(a[8], a[10]);
            a[8] = hi_140;
            a[10] = lo_140;
            float hi_141 = fmaxf(a[9], a[11]);
            float lo_141 = fminf(a[9], a[11]);
            a[9] = hi_141;
            a[11] = lo_141;
            float hi_142 = fmaxf(a[12], a[14]);
            float lo_142 = fminf(a[12], a[14]);
            a[12] = hi_142;
            a[14] = lo_142;
            float hi_143 = fmaxf(a[13], a[15]);
            float lo_143 = fminf(a[13], a[15]);
            a[13] = hi_143;
            a[15] = lo_143;
            float hi_144 = fmaxf(a[0], a[1]);
            float lo_144 = fminf(a[0], a[1]);
            a[0] = hi_144;
            a[1] = lo_144;
            float hi_145 = fmaxf(a[2], a[3]);
            float lo_145 = fminf(a[2], a[3]);
            a[2] = hi_145;
            a[3] = lo_145;
            float hi_146 = fmaxf(a[4], a[5]);
            float lo_146 = fminf(a[4], a[5]);
            a[4] = hi_146;
            a[5] = lo_146;
            float hi_147 = fmaxf(a[6], a[7]);
            float lo_147 = fminf(a[6], a[7]);
            a[6] = hi_147;
            a[7] = lo_147;
            float hi_148 = fmaxf(a[8], a[9]);
            float lo_148 = fminf(a[8], a[9]);
            a[8] = hi_148;
            a[9] = lo_148;
            float hi_149 = fmaxf(a[10], a[11]);
            float lo_149 = fminf(a[10], a[11]);
            a[10] = hi_149;
            a[11] = lo_149;
            float hi_150 = fmaxf(a[12], a[13]);
            float lo_150 = fminf(a[12], a[13]);
            a[12] = hi_150;
            a[13] = lo_150;
            float hi_151 = fmaxf(a[14], a[15]);
            float lo_151 = fminf(a[14], a[15]);
            a[14] = hi_151;
            a[15] = lo_151;
            rej = r_0;
            cb[0] = nb[0];
            cb[1] = nb[1];
            cb[2] = nb[2];
            cb[3] = nb[3];
            cb[4] = nb[4];
            cb[5] = nb[5];
            cb[6] = nb[6];
            cb[7] = nb[7];
            cb[8] = nb[8];
            cb[9] = nb[9];
            cb[10] = nb[10];
            cb[11] = nb[11];
            cb[12] = nb[12];
            cb[13] = nb[13];
            cb[14] = nb[14];
            cb[15] = nb[15];
        }
        int pb_0 = tid_1 * 17;
        pub[pb_0] = a[0];
        pub[pb_0 + 1] = a[1];
        pub[pb_0 + 2] = a[2];
        pub[pb_0 + 3] = a[3];
        pub[pb_0 + 4] = a[4];
        pub[pb_0 + 5] = a[5];
        pub[pb_0 + 6] = a[6];
        pub[pb_0 + 7] = a[7];
        pub[pb_0 + 8] = a[8];
        pub[pb_0 + 9] = a[9];
        pub[pb_0 + 10] = a[10];
        pub[pb_0 + 11] = a[11];
        pub[pb_0 + 12] = a[12];
        pub[pb_0 + 13] = a[13];
        pub[pb_0 + 14] = a[14];
        pub[pb_0 + 15] = a[15];
        pub[pb_0 + 16] = rej;
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        float r_1_1 = neg_inf;
        float V_0[8];
        int s0_8 = (sg * 16 * 32 + cg) * 17;
        int s1_8 = ((sg * 16 + 8) * 32 + cg) * 17;
        float x0_8 = pub[s0_8 + ln];
        float y0_8 = pub[s1_8 + lnr];
        float lo0_8 = fminf(x0_8, y0_8);
        r_1_1 = fmaxf(r_1_1, lo0_8);
        float hi0_8 = fmaxf(x0_8, y0_8);
        float cur_13 = hi0_8;
        float pv_56 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 8);
        float hi_152 = fmaxf(cur_13, pv_56);
        float lo_152 = fminf(cur_13, pv_56);
        cur_13 = ((up[0] != 0) ? hi_152 : lo_152);
        float pv_57 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 4);
        float hi_153 = fmaxf(cur_13, pv_57);
        float lo_153 = fminf(cur_13, pv_57);
        cur_13 = ((up[1] != 0) ? hi_153 : lo_153);
        float pv_58 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 2);
        float hi_154 = fmaxf(cur_13, pv_58);
        float lo_154 = fminf(cur_13, pv_58);
        cur_13 = ((up[2] != 0) ? hi_154 : lo_154);
        float pv_59 = __shfl_xor_sync(0xFFFFFFFF, cur_13, 1);
        float hi_155 = fmaxf(cur_13, pv_59);
        float lo_155 = fminf(cur_13, pv_59);
        cur_13 = ((up[3] != 0) ? hi_155 : lo_155);
        V_0[0] = cur_13;
        int s0_9 = ((sg * 16 + 1) * 32 + cg) * 17;
        int s1_9 = ((sg * 16 + 1 + 8) * 32 + cg) * 17;
        float x0_9 = pub[s0_9 + ln];
        float y0_9 = pub[s1_9 + lnr];
        float lo0_9 = fminf(x0_9, y0_9);
        r_1_1 = fmaxf(r_1_1, lo0_9);
        float hi0_9 = fmaxf(x0_9, y0_9);
        float cur_14 = hi0_9;
        float pv_60 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 8);
        float hi_156 = fmaxf(cur_14, pv_60);
        float lo_156 = fminf(cur_14, pv_60);
        cur_14 = ((up[0] != 0) ? hi_156 : lo_156);
        float pv_61 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 4);
        float hi_157 = fmaxf(cur_14, pv_61);
        float lo_157 = fminf(cur_14, pv_61);
        cur_14 = ((up[1] != 0) ? hi_157 : lo_157);
        float pv_62 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 2);
        float hi_158 = fmaxf(cur_14, pv_62);
        float lo_158 = fminf(cur_14, pv_62);
        cur_14 = ((up[2] != 0) ? hi_158 : lo_158);
        float pv_63 = __shfl_xor_sync(0xFFFFFFFF, cur_14, 1);
        float hi_159 = fmaxf(cur_14, pv_63);
        float lo_159 = fminf(cur_14, pv_63);
        cur_14 = ((up[3] != 0) ? hi_159 : lo_159);
        V_0[1] = cur_14;
        int s0_10 = ((sg * 16 + 2) * 32 + cg) * 17;
        int s1_10 = ((sg * 16 + 2 + 8) * 32 + cg) * 17;
        float x0_10 = pub[s0_10 + ln];
        float y0_10 = pub[s1_10 + lnr];
        float lo0_10 = fminf(x0_10, y0_10);
        r_1_1 = fmaxf(r_1_1, lo0_10);
        float hi0_10 = fmaxf(x0_10, y0_10);
        float cur_15 = hi0_10;
        float pv_64 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
        float hi_160 = fmaxf(cur_15, pv_64);
        float lo_160 = fminf(cur_15, pv_64);
        cur_15 = ((up[0] != 0) ? hi_160 : lo_160);
        float pv_65 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
        float hi_161 = fmaxf(cur_15, pv_65);
        float lo_161 = fminf(cur_15, pv_65);
        cur_15 = ((up[1] != 0) ? hi_161 : lo_161);
        float pv_66 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
        float hi_162 = fmaxf(cur_15, pv_66);
        float lo_162 = fminf(cur_15, pv_66);
        cur_15 = ((up[2] != 0) ? hi_162 : lo_162);
        float pv_67 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
        float hi_163 = fmaxf(cur_15, pv_67);
        float lo_163 = fminf(cur_15, pv_67);
        cur_15 = ((up[3] != 0) ? hi_163 : lo_163);
        V_0[2] = cur_15;
        int s0_11 = ((sg * 16 + 3) * 32 + cg) * 17;
        int s1_11 = ((sg * 16 + 3 + 8) * 32 + cg) * 17;
        float x0_11 = pub[s0_11 + ln];
        float y0_11 = pub[s1_11 + lnr];
        float lo0_11 = fminf(x0_11, y0_11);
        r_1_1 = fmaxf(r_1_1, lo0_11);
        float hi0_11 = fmaxf(x0_11, y0_11);
        float cur_16 = hi0_11;
        float pv_68 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 8);
        float hi_164 = fmaxf(cur_16, pv_68);
        float lo_164 = fminf(cur_16, pv_68);
        cur_16 = ((up[0] != 0) ? hi_164 : lo_164);
        float pv_69 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 4);
        float hi_165 = fmaxf(cur_16, pv_69);
        float lo_165 = fminf(cur_16, pv_69);
        cur_16 = ((up[1] != 0) ? hi_165 : lo_165);
        float pv_70 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 2);
        float hi_166 = fmaxf(cur_16, pv_70);
        float lo_166 = fminf(cur_16, pv_70);
        cur_16 = ((up[2] != 0) ? hi_166 : lo_166);
        float pv_71 = __shfl_xor_sync(0xFFFFFFFF, cur_16, 1);
        float hi_167 = fmaxf(cur_16, pv_71);
        float lo_167 = fminf(cur_16, pv_71);
        cur_16 = ((up[3] != 0) ? hi_167 : lo_167);
        V_0[3] = cur_16;
        int s0_12 = ((sg * 16 + 4) * 32 + cg) * 17;
        int s1_12 = ((sg * 16 + 4 + 8) * 32 + cg) * 17;
        float x0_12 = pub[s0_12 + ln];
        float y0_12 = pub[s1_12 + lnr];
        float lo0_12 = fminf(x0_12, y0_12);
        r_1_1 = fmaxf(r_1_1, lo0_12);
        float hi0_12 = fmaxf(x0_12, y0_12);
        float cur_17 = hi0_12;
        float pv_72 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 8);
        float hi_168 = fmaxf(cur_17, pv_72);
        float lo_168 = fminf(cur_17, pv_72);
        cur_17 = ((up[0] != 0) ? hi_168 : lo_168);
        float pv_73 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 4);
        float hi_169 = fmaxf(cur_17, pv_73);
        float lo_169 = fminf(cur_17, pv_73);
        cur_17 = ((up[1] != 0) ? hi_169 : lo_169);
        float pv_74 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 2);
        float hi_170 = fmaxf(cur_17, pv_74);
        float lo_170 = fminf(cur_17, pv_74);
        cur_17 = ((up[2] != 0) ? hi_170 : lo_170);
        float pv_75 = __shfl_xor_sync(0xFFFFFFFF, cur_17, 1);
        float hi_171 = fmaxf(cur_17, pv_75);
        float lo_171 = fminf(cur_17, pv_75);
        cur_17 = ((up[3] != 0) ? hi_171 : lo_171);
        V_0[4] = cur_17;
        int s0_13 = ((sg * 16 + 5) * 32 + cg) * 17;
        int s1_13 = ((sg * 16 + 5 + 8) * 32 + cg) * 17;
        float x0_13 = pub[s0_13 + ln];
        float y0_13 = pub[s1_13 + lnr];
        float lo0_13 = fminf(x0_13, y0_13);
        r_1_1 = fmaxf(r_1_1, lo0_13);
        float hi0_13 = fmaxf(x0_13, y0_13);
        float cur_18 = hi0_13;
        float pv_76 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 8);
        float hi_172 = fmaxf(cur_18, pv_76);
        float lo_172 = fminf(cur_18, pv_76);
        cur_18 = ((up[0] != 0) ? hi_172 : lo_172);
        float pv_77 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 4);
        float hi_173 = fmaxf(cur_18, pv_77);
        float lo_173 = fminf(cur_18, pv_77);
        cur_18 = ((up[1] != 0) ? hi_173 : lo_173);
        float pv_78 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 2);
        float hi_174 = fmaxf(cur_18, pv_78);
        float lo_174 = fminf(cur_18, pv_78);
        cur_18 = ((up[2] != 0) ? hi_174 : lo_174);
        float pv_79 = __shfl_xor_sync(0xFFFFFFFF, cur_18, 1);
        float hi_175 = fmaxf(cur_18, pv_79);
        float lo_175 = fminf(cur_18, pv_79);
        cur_18 = ((up[3] != 0) ? hi_175 : lo_175);
        V_0[5] = cur_18;
        int s0_14 = ((sg * 16 + 6) * 32 + cg) * 17;
        int s1_14 = ((sg * 16 + 6 + 8) * 32 + cg) * 17;
        float x0_14 = pub[s0_14 + ln];
        float y0_14 = pub[s1_14 + lnr];
        float lo0_14 = fminf(x0_14, y0_14);
        r_1_1 = fmaxf(r_1_1, lo0_14);
        float hi0_14 = fmaxf(x0_14, y0_14);
        float cur_19 = hi0_14;
        float pv_80 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 8);
        float hi_176 = fmaxf(cur_19, pv_80);
        float lo_176 = fminf(cur_19, pv_80);
        cur_19 = ((up[0] != 0) ? hi_176 : lo_176);
        float pv_81 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 4);
        float hi_177 = fmaxf(cur_19, pv_81);
        float lo_177 = fminf(cur_19, pv_81);
        cur_19 = ((up[1] != 0) ? hi_177 : lo_177);
        float pv_82 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 2);
        float hi_178 = fmaxf(cur_19, pv_82);
        float lo_178 = fminf(cur_19, pv_82);
        cur_19 = ((up[2] != 0) ? hi_178 : lo_178);
        float pv_83 = __shfl_xor_sync(0xFFFFFFFF, cur_19, 1);
        float hi_179 = fmaxf(cur_19, pv_83);
        float lo_179 = fminf(cur_19, pv_83);
        cur_19 = ((up[3] != 0) ? hi_179 : lo_179);
        V_0[6] = cur_19;
        int s0_15 = ((sg * 16 + 7) * 32 + cg) * 17;
        int s1_15 = ((sg * 16 + 7 + 8) * 32 + cg) * 17;
        float x0_15 = pub[s0_15 + ln];
        float y0_15 = pub[s1_15 + lnr];
        float lo0_15 = fminf(x0_15, y0_15);
        r_1_1 = fmaxf(r_1_1, lo0_15);
        float hi0_15 = fmaxf(x0_15, y0_15);
        float cur_20 = hi0_15;
        float pv_84 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 8);
        float hi_180 = fmaxf(cur_20, pv_84);
        float lo_180 = fminf(cur_20, pv_84);
        cur_20 = ((up[0] != 0) ? hi_180 : lo_180);
        float pv_85 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 4);
        float hi_181 = fmaxf(cur_20, pv_85);
        float lo_181 = fminf(cur_20, pv_85);
        cur_20 = ((up[1] != 0) ? hi_181 : lo_181);
        float pv_86 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 2);
        float hi_182 = fmaxf(cur_20, pv_86);
        float lo_182 = fminf(cur_20, pv_86);
        cur_20 = ((up[2] != 0) ? hi_182 : lo_182);
        float pv_87 = __shfl_xor_sync(0xFFFFFFFF, cur_20, 1);
        float hi_183 = fmaxf(cur_20, pv_87);
        float lo_183 = fminf(cur_20, pv_87);
        cur_20 = ((up[3] != 0) ? hi_183 : lo_183);
        V_0[7] = cur_20;
        float rs_0 = pub[((sg * 16 + ln) * 32 + cg) * 17 + 16];
        r_1_1 = fmaxf(r_1_1, rs_0);
        float y1_7 = __shfl_xor_sync(0xFFFFFFFF, V_0[4], 15);
        float lo1_7 = fminf(V_0[0], y1_7);
        r_1_1 = fmaxf(r_1_1, lo1_7);
        float hi1_7 = fmaxf(V_0[0], y1_7);
        float cur_21 = hi1_7;
        float pv_88 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 8);
        float hi_184 = fmaxf(cur_21, pv_88);
        float lo_184 = fminf(cur_21, pv_88);
        cur_21 = ((up[0] != 0) ? hi_184 : lo_184);
        float pv_89 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 4);
        float hi_185 = fmaxf(cur_21, pv_89);
        float lo_185 = fminf(cur_21, pv_89);
        cur_21 = ((up[1] != 0) ? hi_185 : lo_185);
        float pv_90 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 2);
        float hi_186 = fmaxf(cur_21, pv_90);
        float lo_186 = fminf(cur_21, pv_90);
        cur_21 = ((up[2] != 0) ? hi_186 : lo_186);
        float pv_91 = __shfl_xor_sync(0xFFFFFFFF, cur_21, 1);
        float hi_187 = fmaxf(cur_21, pv_91);
        float lo_187 = fminf(cur_21, pv_91);
        cur_21 = ((up[3] != 0) ? hi_187 : lo_187);
        V_0[0] = cur_21;
        float y1_8 = __shfl_xor_sync(0xFFFFFFFF, V_0[5], 15);
        float lo1_8 = fminf(V_0[1], y1_8);
        r_1_1 = fmaxf(r_1_1, lo1_8);
        float hi1_8 = fmaxf(V_0[1], y1_8);
        float cur_22 = hi1_8;
        float pv_92 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 8);
        float hi_188 = fmaxf(cur_22, pv_92);
        float lo_188 = fminf(cur_22, pv_92);
        cur_22 = ((up[0] != 0) ? hi_188 : lo_188);
        float pv_93 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 4);
        float hi_189 = fmaxf(cur_22, pv_93);
        float lo_189 = fminf(cur_22, pv_93);
        cur_22 = ((up[1] != 0) ? hi_189 : lo_189);
        float pv_94 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 2);
        float hi_190 = fmaxf(cur_22, pv_94);
        float lo_190 = fminf(cur_22, pv_94);
        cur_22 = ((up[2] != 0) ? hi_190 : lo_190);
        float pv_95 = __shfl_xor_sync(0xFFFFFFFF, cur_22, 1);
        float hi_191 = fmaxf(cur_22, pv_95);
        float lo_191 = fminf(cur_22, pv_95);
        cur_22 = ((up[3] != 0) ? hi_191 : lo_191);
        V_0[1] = cur_22;
        float y1_9 = __shfl_xor_sync(0xFFFFFFFF, V_0[6], 15);
        float lo1_9 = fminf(V_0[2], y1_9);
        r_1_1 = fmaxf(r_1_1, lo1_9);
        float hi1_9 = fmaxf(V_0[2], y1_9);
        float cur_23 = hi1_9;
        float pv_96 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 8);
        float hi_192 = fmaxf(cur_23, pv_96);
        float lo_192 = fminf(cur_23, pv_96);
        cur_23 = ((up[0] != 0) ? hi_192 : lo_192);
        float pv_97 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 4);
        float hi_193 = fmaxf(cur_23, pv_97);
        float lo_193 = fminf(cur_23, pv_97);
        cur_23 = ((up[1] != 0) ? hi_193 : lo_193);
        float pv_98 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 2);
        float hi_194 = fmaxf(cur_23, pv_98);
        float lo_194 = fminf(cur_23, pv_98);
        cur_23 = ((up[2] != 0) ? hi_194 : lo_194);
        float pv_99 = __shfl_xor_sync(0xFFFFFFFF, cur_23, 1);
        float hi_195 = fmaxf(cur_23, pv_99);
        float lo_195 = fminf(cur_23, pv_99);
        cur_23 = ((up[3] != 0) ? hi_195 : lo_195);
        V_0[2] = cur_23;
        float y1_10 = __shfl_xor_sync(0xFFFFFFFF, V_0[7], 15);
        float lo1_10 = fminf(V_0[3], y1_10);
        r_1_1 = fmaxf(r_1_1, lo1_10);
        float hi1_10 = fmaxf(V_0[3], y1_10);
        float cur_24 = hi1_10;
        float pv_100 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 8);
        float hi_196 = fmaxf(cur_24, pv_100);
        float lo_196 = fminf(cur_24, pv_100);
        cur_24 = ((up[0] != 0) ? hi_196 : lo_196);
        float pv_101 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 4);
        float hi_197 = fmaxf(cur_24, pv_101);
        float lo_197 = fminf(cur_24, pv_101);
        cur_24 = ((up[1] != 0) ? hi_197 : lo_197);
        float pv_102 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 2);
        float hi_198 = fmaxf(cur_24, pv_102);
        float lo_198 = fminf(cur_24, pv_102);
        cur_24 = ((up[2] != 0) ? hi_198 : lo_198);
        float pv_103 = __shfl_xor_sync(0xFFFFFFFF, cur_24, 1);
        float hi_199 = fmaxf(cur_24, pv_103);
        float lo_199 = fminf(cur_24, pv_103);
        cur_24 = ((up[3] != 0) ? hi_199 : lo_199);
        V_0[3] = cur_24;
        float y1_11 = __shfl_xor_sync(0xFFFFFFFF, V_0[2], 15);
        float lo1_11 = fminf(V_0[0], y1_11);
        r_1_1 = fmaxf(r_1_1, lo1_11);
        float hi1_11 = fmaxf(V_0[0], y1_11);
        float cur_25 = hi1_11;
        float pv_104 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 8);
        float hi_200 = fmaxf(cur_25, pv_104);
        float lo_200 = fminf(cur_25, pv_104);
        cur_25 = ((up[0] != 0) ? hi_200 : lo_200);
        float pv_105 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 4);
        float hi_201 = fmaxf(cur_25, pv_105);
        float lo_201 = fminf(cur_25, pv_105);
        cur_25 = ((up[1] != 0) ? hi_201 : lo_201);
        float pv_106 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 2);
        float hi_202 = fmaxf(cur_25, pv_106);
        float lo_202 = fminf(cur_25, pv_106);
        cur_25 = ((up[2] != 0) ? hi_202 : lo_202);
        float pv_107 = __shfl_xor_sync(0xFFFFFFFF, cur_25, 1);
        float hi_203 = fmaxf(cur_25, pv_107);
        float lo_203 = fminf(cur_25, pv_107);
        cur_25 = ((up[3] != 0) ? hi_203 : lo_203);
        V_0[0] = cur_25;
        float y1_12 = __shfl_xor_sync(0xFFFFFFFF, V_0[3], 15);
        float lo1_12 = fminf(V_0[1], y1_12);
        r_1_1 = fmaxf(r_1_1, lo1_12);
        float hi1_12 = fmaxf(V_0[1], y1_12);
        float cur_26 = hi1_12;
        float pv_108 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 8);
        float hi_204 = fmaxf(cur_26, pv_108);
        float lo_204 = fminf(cur_26, pv_108);
        cur_26 = ((up[0] != 0) ? hi_204 : lo_204);
        float pv_109 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 4);
        float hi_205 = fmaxf(cur_26, pv_109);
        float lo_205 = fminf(cur_26, pv_109);
        cur_26 = ((up[1] != 0) ? hi_205 : lo_205);
        float pv_110 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 2);
        float hi_206 = fmaxf(cur_26, pv_110);
        float lo_206 = fminf(cur_26, pv_110);
        cur_26 = ((up[2] != 0) ? hi_206 : lo_206);
        float pv_111 = __shfl_xor_sync(0xFFFFFFFF, cur_26, 1);
        float hi_207 = fmaxf(cur_26, pv_111);
        float lo_207 = fminf(cur_26, pv_111);
        cur_26 = ((up[3] != 0) ? hi_207 : lo_207);
        V_0[1] = cur_26;
        float y1_13 = __shfl_xor_sync(0xFFFFFFFF, V_0[1], 15);
        float lo1_13 = fminf(V_0[0], y1_13);
        r_1_1 = fmaxf(r_1_1, lo1_13);
        float hi1_13 = fmaxf(V_0[0], y1_13);
        V_0[0] = hi1_13;
        float K_0 = V_0[0];
        rr1[0] = r_1_1;
        K = K_0;
    }
    int flagged = 0;
    if (qc != 4294967295u) {
        flagged = 1;
    }
    unsigned int qg = qcol[cg];
    int gflag = 0;
    if (qg != 4294967295u) {
        gflag = 1;
    }
    float K2 = 0.0f;
    if (need2 != 0) {
        unsigned int qv7 = 4294967295u;
        if ((tid_1 & 31) < 32) {
            qv7 = qcol[tid_1 & 31];
        }
        unsigned int fm7 = __ballot_sync(0xFFFFFFFF, qv7 != 4294967295u);
        int nf7 = (int)(unsigned int)__popc(fm7);
        #pragma unroll 1
        for (int it7 = 0; it7 < nf7; it7++) {
            int _find_first_set_8 = __ffs(fm7);
            int fc7 = _find_first_set_8 - 1;
            fm7 = fm7 & fm7 - 1;
            unsigned int qf7 = qcol[fc7];
            int limv = limc[fc7];
            long long cbase2 = (long long)head * (long long)tiles * nq64 + (long long)(blockIdx.x * 32 + fc7);
            int lane_1 = tid_1 & 31;
            unsigned int bb[4];
            bb[0] = 4286578688u;
            int s5 = lane_1;
            int j5 = s5 >> 4 & 7;
            int v5 = 0;
            int s4 = lane_1;
            int i4 = s4 & 15;
            int j4 = s4 >> 4 & 7;
            int g4 = whi + (s4 >> 7);
            int t4 = (j4 * 16 + g4) * 16 + i4;
            if (j5 < num_chunks && t4 < limv) {
                v5 = 1;
            }
            if (v5 != 0) {
                int s4_0 = lane_1;
                int i4_0 = s4_0 & 15;
                int j4_0 = s4_0 >> 4 & 7;
                int g4_0 = whi + (s4_0 >> 7);
                int t4_0 = (j4_0 * 16 + g4_0) * 16 + i4_0;
                bb[0] = S[cbase2 + (long long)t4_0 * nq64];
            }
            bb[1] = 4286578688u;
            int s5_0 = lane_1 + 32;
            int j5_0 = s5_0 >> 4 & 7;
            int v5_0 = 0;
            int s4_1 = lane_1 + 32;
            int i4_1 = s4_1 & 15;
            int j4_1 = s4_1 >> 4 & 7;
            int g4_1 = whi + (s4_1 >> 7);
            int t4_1 = (j4_1 * 16 + g4_1) * 16 + i4_1;
            if (j5_0 < num_chunks && t4_1 < limv) {
                v5_0 = 1;
            }
            if (v5_0 != 0) {
                int s4_2 = lane_1 + 32;
                int i4_2 = s4_2 & 15;
                int j4_2 = s4_2 >> 4 & 7;
                int g4_2 = whi + (s4_2 >> 7);
                int t4_2 = (j4_2 * 16 + g4_2) * 16 + i4_2;
                bb[1] = S[cbase2 + (long long)t4_2 * nq64];
            }
            bb[2] = 4286578688u;
            int s5_1 = lane_1 + 64;
            int j5_1 = s5_1 >> 4 & 7;
            int v5_1 = 0;
            int s4_3 = lane_1 + 64;
            int i4_3 = s4_3 & 15;
            int j4_3 = s4_3 >> 4 & 7;
            int g4_3 = whi + (s4_3 >> 7);
            int t4_3 = (j4_3 * 16 + g4_3) * 16 + i4_3;
            if (j5_1 < num_chunks && t4_3 < limv) {
                v5_1 = 1;
            }
            if (v5_1 != 0) {
                int s4_4 = lane_1 + 64;
                int i4_4 = s4_4 & 15;
                int j4_4 = s4_4 >> 4 & 7;
                int g4_4 = whi + (s4_4 >> 7);
                int t4_4 = (j4_4 * 16 + g4_4) * 16 + i4_4;
                bb[2] = S[cbase2 + (long long)t4_4 * nq64];
            }
            bb[3] = 4286578688u;
            int s5_2 = lane_1 + 96;
            int j5_2 = s5_2 >> 4 & 7;
            int v5_2 = 0;
            int s4_5 = lane_1 + 96;
            int i4_5 = s4_5 & 15;
            int j4_5 = s4_5 >> 4 & 7;
            int g4_5 = whi + (s4_5 >> 7);
            int t4_5 = (j4_5 * 16 + g4_5) * 16 + i4_5;
            if (j5_2 < num_chunks && t4_5 < limv) {
                v5_2 = 1;
            }
            if (v5_2 != 0) {
                int s4_6 = lane_1 + 96;
                int i4_6 = s4_6 & 15;
                int j4_6 = s4_6 >> 4 & 7;
                int g4_6 = whi + (s4_6 >> 7);
                int t4_6 = (j4_6 * 16 + g4_6) * 16 + i4_6;
                bb[3] = S[cbase2 + (long long)t4_6 * nq64];
            }
            asm volatile("" ::: "memory");
            unsigned int kk2[4];
            kk2[0] = 0;
            int s5_3 = lane_1;
            int j5_3 = s5_3 >> 4 & 7;
            int v5_3 = 0;
            int s4_7 = lane_1;
            int i4_7 = s4_7 & 15;
            int j4_7 = s4_7 >> 4 & 7;
            int g4_7 = whi + (s4_7 >> 7);
            int t4_7 = (j4_7 * 16 + g4_7) * 16 + i4_7;
            if (j5_3 < num_chunks && t4_7 < limv) {
                v5_3 = 1;
            }
            if (v5_3 != 0) {
                int s4_0_1 = lane_1;
                int i4_1_0 = s4_0_1 & 15;
                int j4_2_1 = s4_0_1 >> 4 & 7;
                int g4_3_0 = whi + (s4_0_1 >> 7);
                int t4_4_1 = (j4_2_1 * 16 + g4_3_0) * 16 + i4_1_0;
                float sc_30 = __uint_as_float(bb[0]);
                sc_30 = fmaxf(sc_30, -1.7014118e+38f);
                sc_30 = fminf(sc_30, 1.7014118e+38f);
                unsigned int u8 = __as_u32(sc_30);
                unsigned int key8 = 0;
                if ((u8 & 4294965248u) == qf7) {
                    unsigned int lowb8 = (u8 ^ (unsigned int)((int)u8 >> 31) & 2047) & 2047;
                    key8 = 536870912 | lowb8 << 11 | (unsigned int)t4_4_1;
                }
                int f2 = 0;
                if ((t4_4_1 < fb || t4_4_1 >= limv - fe) && t4_4_1 < limv) {
                    f2 = 1;
                }
                if (f2 != 0) {
                    key8 = 0;
                }
                kk2[0] = key8;
            }
            kk2[1] = 0;
            int s5_4 = lane_1 + 32;
            int j5_4 = s5_4 >> 4 & 7;
            int v5_4 = 0;
            int s4_8 = lane_1 + 32;
            int i4_8 = s4_8 & 15;
            int j4_8 = s4_8 >> 4 & 7;
            int g4_8 = whi + (s4_8 >> 7);
            int t4_8 = (j4_8 * 16 + g4_8) * 16 + i4_8;
            if (j5_4 < num_chunks && t4_8 < limv) {
                v5_4 = 1;
            }
            if (v5_4 != 0) {
                int s4_0_2 = lane_1 + 32;
                int i4_1_1 = s4_0_2 & 15;
                int j4_2_2 = s4_0_2 >> 4 & 7;
                int g4_3_1 = whi + (s4_0_2 >> 7);
                int t4_4_2 = (j4_2_2 * 16 + g4_3_1) * 16 + i4_1_1;
                float sc_31 = __uint_as_float(bb[1]);
                sc_31 = fmaxf(sc_31, -1.7014118e+38f);
                sc_31 = fminf(sc_31, 1.7014118e+38f);
                unsigned int u8_1 = __as_u32(sc_31);
                unsigned int key8_1 = 0;
                if ((u8_1 & 4294965248u) == qf7) {
                    unsigned int lowb8_1 = (u8_1 ^ (unsigned int)((int)u8_1 >> 31) & 2047) & 2047;
                    key8_1 = 536870912 | lowb8_1 << 11 | (unsigned int)t4_4_2;
                }
                int f2_1 = 0;
                if ((t4_4_2 < fb || t4_4_2 >= limv - fe) && t4_4_2 < limv) {
                    f2_1 = 1;
                }
                if (f2_1 != 0) {
                    key8_1 = 0;
                }
                kk2[1] = key8_1;
            }
            kk2[2] = 0;
            int s5_5 = lane_1 + 64;
            int j5_5 = s5_5 >> 4 & 7;
            int v5_5 = 0;
            int s4_9 = lane_1 + 64;
            int i4_9 = s4_9 & 15;
            int j4_9 = s4_9 >> 4 & 7;
            int g4_9 = whi + (s4_9 >> 7);
            int t4_9 = (j4_9 * 16 + g4_9) * 16 + i4_9;
            if (j5_5 < num_chunks && t4_9 < limv) {
                v5_5 = 1;
            }
            if (v5_5 != 0) {
                int s4_0_3 = lane_1 + 64;
                int i4_1_2 = s4_0_3 & 15;
                int j4_2_3 = s4_0_3 >> 4 & 7;
                int g4_3_2 = whi + (s4_0_3 >> 7);
                int t4_4_3 = (j4_2_3 * 16 + g4_3_2) * 16 + i4_1_2;
                float sc_32 = __uint_as_float(bb[2]);
                sc_32 = fmaxf(sc_32, -1.7014118e+38f);
                sc_32 = fminf(sc_32, 1.7014118e+38f);
                unsigned int u8_2 = __as_u32(sc_32);
                unsigned int key8_2 = 0;
                if ((u8_2 & 4294965248u) == qf7) {
                    unsigned int lowb8_2 = (u8_2 ^ (unsigned int)((int)u8_2 >> 31) & 2047) & 2047;
                    key8_2 = 536870912 | lowb8_2 << 11 | (unsigned int)t4_4_3;
                }
                int f2_2 = 0;
                if ((t4_4_3 < fb || t4_4_3 >= limv - fe) && t4_4_3 < limv) {
                    f2_2 = 1;
                }
                if (f2_2 != 0) {
                    key8_2 = 0;
                }
                kk2[2] = key8_2;
            }
            kk2[3] = 0;
            int s5_6 = lane_1 + 96;
            int j5_6 = s5_6 >> 4 & 7;
            int v5_6 = 0;
            int s4_10 = lane_1 + 96;
            int i4_10 = s4_10 & 15;
            int j4_10 = s4_10 >> 4 & 7;
            int g4_10 = whi + (s4_10 >> 7);
            int t4_10 = (j4_10 * 16 + g4_10) * 16 + i4_10;
            if (j5_6 < num_chunks && t4_10 < limv) {
                v5_6 = 1;
            }
            if (v5_6 != 0) {
                int s4_0_4 = lane_1 + 96;
                int i4_1_3 = s4_0_4 & 15;
                int j4_2_4 = s4_0_4 >> 4 & 7;
                int g4_3_3 = whi + (s4_0_4 >> 7);
                int t4_4_4 = (j4_2_4 * 16 + g4_3_3) * 16 + i4_1_3;
                float sc_33 = __uint_as_float(bb[3]);
                sc_33 = fmaxf(sc_33, -1.7014118e+38f);
                sc_33 = fminf(sc_33, 1.7014118e+38f);
                unsigned int u8_3 = __as_u32(sc_33);
                unsigned int key8_3 = 0;
                if ((u8_3 & 4294965248u) == qf7) {
                    unsigned int lowb8_3 = (u8_3 ^ (unsigned int)((int)u8_3 >> 31) & 2047) & 2047;
                    key8_3 = 536870912 | lowb8_3 << 11 | (unsigned int)t4_4_4;
                }
                int f2_3 = 0;
                if ((t4_4_4 < fb || t4_4_4 >= limv - fe) && t4_4_4 < limv) {
                    f2_3 = 1;
                }
                if (f2_3 != 0) {
                    key8_3 = 0;
                }
                kk2[3] = key8_3;
            }
            int n3 = 0;
            if (kk2[0] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[1] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[2] != 0) {
                n3 = n3 + 1;
            }
            if (kk2[3] != 0) {
                n3 = n3 + 1;
            }
            if (n3 > 0) {
                unsigned int base3 = atomicAdd(&ccnt[fc7], (unsigned int)n3);
                if (base3 + (unsigned int)n3 > 16) {
                    flagw[1] = 1;
                } else {
                    int pos3 = fc7 * 16 + (int)base3;
                    if (kk2[0] != 0) {
                        cbuf[pos3] = kk2[0];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[1] != 0) {
                        cbuf[pos3] = kk2[1];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[2] != 0) {
                        cbuf[pos3] = kk2[2];
                        pos3 = pos3 + 1;
                    }
                    if (kk2[3] != 0) {
                        cbuf[pos3] = kk2[3];
                        pos3 = pos3 + 1;
                    }
                }
            }
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        unsigned int need_fullx = flagw[1];
        unsigned int cntx = ccnt[c];
        int ovfx = 0;
        if (flagged != 0 && cntx > 16) {
            ovfx = 1;
        }
        if (need_fullx != 0) {
            int lim2x = 0;
            if (ovfx != 0) {
                lim2x = lim;
            }
            float a2x[16];
            a2x[0] = 0.0f;
            a2x[1] = 0.0f;
            a2x[2] = 0.0f;
            a2x[3] = 0.0f;
            a2x[4] = 0.0f;
            a2x[5] = 0.0f;
            a2x[6] = 0.0f;
            a2x[7] = 0.0f;
            a2x[8] = 0.0f;
            a2x[9] = 0.0f;
            a2x[10] = 0.0f;
            a2x[11] = 0.0f;
            a2x[12] = 0.0f;
            a2x[13] = 0.0f;
            a2x[14] = 0.0f;
            a2x[15] = 0.0f;
            int _warp_vote_9 = __any_sync(0xFFFFFFFF, ovfx != 0);
            if (_warp_vote_9 != 0) {
                unsigned int cb2[16];
                unsigned int nb2[16];
                float kb2[16];
                int t0_7 = w * 16;
                long long p_3 = cbase + (long long)t0_7 * nq64;
                cb2[0] = 4286578688u;
                if (lim2x > t0_7) {
                    cb2[0] = S[p_3];
                }
                cb2[1] = 4286578688u;
                if (lim2x > t0_7 + 1) {
                    cb2[1] = S[p_3 + nq64];
                }
                cb2[2] = 4286578688u;
                if (lim2x > t0_7 + 2) {
                    cb2[2] = S[p_3 + 2ll * nq64];
                }
                cb2[3] = 4286578688u;
                if (lim2x > t0_7 + 3) {
                    cb2[3] = S[p_3 + 3ll * nq64];
                }
                cb2[4] = 4286578688u;
                if (lim2x > t0_7 + 4) {
                    cb2[4] = S[p_3 + 4ll * nq64];
                }
                cb2[5] = 4286578688u;
                if (lim2x > t0_7 + 5) {
                    cb2[5] = S[p_3 + 5ll * nq64];
                }
                cb2[6] = 4286578688u;
                if (lim2x > t0_7 + 6) {
                    cb2[6] = S[p_3 + 6ll * nq64];
                }
                cb2[7] = 4286578688u;
                if (lim2x > t0_7 + 7) {
                    cb2[7] = S[p_3 + 7ll * nq64];
                }
                cb2[8] = 4286578688u;
                if (lim2x > t0_7 + 8) {
                    cb2[8] = S[p_3 + 8ll * nq64];
                }
                cb2[9] = 4286578688u;
                if (lim2x > t0_7 + 9) {
                    cb2[9] = S[p_3 + 9ll * nq64];
                }
                cb2[10] = 4286578688u;
                if (lim2x > t0_7 + 10) {
                    cb2[10] = S[p_3 + 10ll * nq64];
                }
                cb2[11] = 4286578688u;
                if (lim2x > t0_7 + 11) {
                    cb2[11] = S[p_3 + 11ll * nq64];
                }
                cb2[12] = 4286578688u;
                if (lim2x > t0_7 + 12) {
                    cb2[12] = S[p_3 + 12ll * nq64];
                }
                cb2[13] = 4286578688u;
                if (lim2x > t0_7 + 13) {
                    cb2[13] = S[p_3 + 13ll * nq64];
                }
                cb2[14] = 4286578688u;
                if (lim2x > t0_7 + 14) {
                    cb2[14] = S[p_3 + 14ll * nq64];
                }
                cb2[15] = 4286578688u;
                if (lim2x > t0_7 + 15) {
                    cb2[15] = S[p_3 + 15ll * nq64];
                }
                asm volatile("" ::: "memory");
                #pragma unroll 1
                for (int j_1 = 0; j_1 < num_chunks; j_1++) {
                    int t0_8 = ((j_1 + 1) * 16 + w) * 16;
                    long long p_4 = cbase + (long long)t0_8 * nq64;
                    nb2[0] = 4286578688u;
                    if (lim2x > t0_8) {
                        nb2[0] = S[p_4];
                    }
                    nb2[1] = 4286578688u;
                    if (lim2x > t0_8 + 1) {
                        nb2[1] = S[p_4 + nq64];
                    }
                    nb2[2] = 4286578688u;
                    if (lim2x > t0_8 + 2) {
                        nb2[2] = S[p_4 + 2ll * nq64];
                    }
                    nb2[3] = 4286578688u;
                    if (lim2x > t0_8 + 3) {
                        nb2[3] = S[p_4 + 3ll * nq64];
                    }
                    nb2[4] = 4286578688u;
                    if (lim2x > t0_8 + 4) {
                        nb2[4] = S[p_4 + 4ll * nq64];
                    }
                    nb2[5] = 4286578688u;
                    if (lim2x > t0_8 + 5) {
                        nb2[5] = S[p_4 + 5ll * nq64];
                    }
                    nb2[6] = 4286578688u;
                    if (lim2x > t0_8 + 6) {
                        nb2[6] = S[p_4 + 6ll * nq64];
                    }
                    nb2[7] = 4286578688u;
                    if (lim2x > t0_8 + 7) {
                        nb2[7] = S[p_4 + 7ll * nq64];
                    }
                    nb2[8] = 4286578688u;
                    if (lim2x > t0_8 + 8) {
                        nb2[8] = S[p_4 + 8ll * nq64];
                    }
                    nb2[9] = 4286578688u;
                    if (lim2x > t0_8 + 9) {
                        nb2[9] = S[p_4 + 9ll * nq64];
                    }
                    nb2[10] = 4286578688u;
                    if (lim2x > t0_8 + 10) {
                        nb2[10] = S[p_4 + 10ll * nq64];
                    }
                    nb2[11] = 4286578688u;
                    if (lim2x > t0_8 + 11) {
                        nb2[11] = S[p_4 + 11ll * nq64];
                    }
                    nb2[12] = 4286578688u;
                    if (lim2x > t0_8 + 12) {
                        nb2[12] = S[p_4 + 12ll * nq64];
                    }
                    nb2[13] = 4286578688u;
                    if (lim2x > t0_8 + 13) {
                        nb2[13] = S[p_4 + 13ll * nq64];
                    }
                    nb2[14] = 4286578688u;
                    if (lim2x > t0_8 + 14) {
                        nb2[14] = S[p_4 + 14ll * nq64];
                    }
                    nb2[15] = 4286578688u;
                    if (lim2x > t0_8 + 15) {
                        nb2[15] = S[p_4 + 15ll * nq64];
                    }
                    asm volatile("" ::: "memory");
                    int t0_9 = (j_1 * 16 + w) * 16;
                    float qf = __uint_as_float(qc);
                    float sc_34 = __uint_as_float(cb2[0]);
                    sc_34 = fmaxf(sc_34, -1.7014118e+38f);
                    sc_34 = fminf(sc_34, 1.7014118e+38f);
                    unsigned int u_1_0 = __as_u32(sc_34);
                    unsigned int cls_1_0 = u_1_0 & 4294965248u;
                    int f_32 = 0;
                    if ((t0_9 < fb || t0_9 >= lim - fe) && lim > t0_9) {
                        f_32 = 1;
                    }
                    if (f_32 != 0) {
                        cls_1_0 = 2139092992;
                    }
                    unsigned int key = 0;
                    if ((qf < __uint_as_float(cls_1_0) || __uint_as_float(cls_1_0) == qf && cls_1_0 < qc) && cls_1_0 < 4278190080u) {
                        key = 1073741824 | (unsigned int)t0_9;
                    }
                    if (cls_1_0 == qc) {
                        unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 2047) & 2047;
                        key = 536870912 | lowb << 11 | (unsigned int)t0_9;
                    }
                    kb2[0] = __uint_as_float(key);
                    float sc_30_1 = __uint_as_float(cb2[1]);
                    sc_30_1 = fmaxf(sc_30_1, -1.7014118e+38f);
                    sc_30_1 = fminf(sc_30_1, 1.7014118e+38f);
                    unsigned int u_1_1 = __as_u32(sc_30_1);
                    unsigned int cls_1_1 = u_1_1 & 4294965248u;
                    int f_32_1 = 0;
                    if ((t0_9 + 1 < fb || t0_9 + 1 >= lim - fe) && lim > t0_9 + 1) {
                        f_32_1 = 1;
                    }
                    if (f_32_1 != 0) {
                        cls_1_1 = 2139092992;
                    }
                    unsigned int key_0 = 0;
                    if ((qf < __uint_as_float(cls_1_1) || __uint_as_float(cls_1_1) == qf && cls_1_1 < qc) && cls_1_1 < 4278190080u) {
                        key_0 = 1073741824 | (unsigned int)(t0_9 + 1);
                    }
                    if (cls_1_1 == qc) {
                        unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 2047) & 2047;
                        key_0 = 536870912 | lowb_1 << 11 | (unsigned int)(t0_9 + 1);
                    }
                    kb2[1] = __uint_as_float(key_0);
                    float sc_31_1 = __uint_as_float(cb2[2]);
                    sc_31_1 = fmaxf(sc_31_1, -1.7014118e+38f);
                    sc_31_1 = fminf(sc_31_1, 1.7014118e+38f);
                    unsigned int u_1_2 = __as_u32(sc_31_1);
                    unsigned int cls_1_2 = u_1_2 & 4294965248u;
                    int f_33 = 0;
                    if ((t0_9 + 2 < fb || t0_9 + 2 >= lim - fe) && lim > t0_9 + 2) {
                        f_33 = 1;
                    }
                    if (f_33 != 0) {
                        cls_1_2 = 2139092992;
                    }
                    unsigned int key_1 = 0;
                    if ((qf < __uint_as_float(cls_1_2) || __uint_as_float(cls_1_2) == qf && cls_1_2 < qc) && cls_1_2 < 4278190080u) {
                        key_1 = 1073741824 | (unsigned int)(t0_9 + 2);
                    }
                    if (cls_1_2 == qc) {
                        unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 2047) & 2047;
                        key_1 = 536870912 | lowb_2 << 11 | (unsigned int)(t0_9 + 2);
                    }
                    kb2[2] = __uint_as_float(key_1);
                    float sc_32_1 = __uint_as_float(cb2[3]);
                    sc_32_1 = fmaxf(sc_32_1, -1.7014118e+38f);
                    sc_32_1 = fminf(sc_32_1, 1.7014118e+38f);
                    unsigned int u_1_3 = __as_u32(sc_32_1);
                    unsigned int cls_1_3 = u_1_3 & 4294965248u;
                    int f_34 = 0;
                    if ((t0_9 + 3 < fb || t0_9 + 3 >= lim - fe) && lim > t0_9 + 3) {
                        f_34 = 1;
                    }
                    if (f_34 != 0) {
                        cls_1_3 = 2139092992;
                    }
                    unsigned int key_2 = 0;
                    if ((qf < __uint_as_float(cls_1_3) || __uint_as_float(cls_1_3) == qf && cls_1_3 < qc) && cls_1_3 < 4278190080u) {
                        key_2 = 1073741824 | (unsigned int)(t0_9 + 3);
                    }
                    if (cls_1_3 == qc) {
                        unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 2047) & 2047;
                        key_2 = 536870912 | lowb_3 << 11 | (unsigned int)(t0_9 + 3);
                    }
                    kb2[3] = __uint_as_float(key_2);
                    float sc_33_1 = __uint_as_float(cb2[4]);
                    sc_33_1 = fmaxf(sc_33_1, -1.7014118e+38f);
                    sc_33_1 = fminf(sc_33_1, 1.7014118e+38f);
                    unsigned int u_1_4 = __as_u32(sc_33_1);
                    unsigned int cls_1_4 = u_1_4 & 4294965248u;
                    int f_35 = 0;
                    if ((t0_9 + 4 < fb || t0_9 + 4 >= lim - fe) && lim > t0_9 + 4) {
                        f_35 = 1;
                    }
                    if (f_35 != 0) {
                        cls_1_4 = 2139092992;
                    }
                    unsigned int key_3 = 0;
                    if ((qf < __uint_as_float(cls_1_4) || __uint_as_float(cls_1_4) == qf && cls_1_4 < qc) && cls_1_4 < 4278190080u) {
                        key_3 = 1073741824 | (unsigned int)(t0_9 + 4);
                    }
                    if (cls_1_4 == qc) {
                        unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 2047) & 2047;
                        key_3 = 536870912 | lowb_4 << 11 | (unsigned int)(t0_9 + 4);
                    }
                    kb2[4] = __uint_as_float(key_3);
                    float sc_34_1 = __uint_as_float(cb2[5]);
                    sc_34_1 = fmaxf(sc_34_1, -1.7014118e+38f);
                    sc_34_1 = fminf(sc_34_1, 1.7014118e+38f);
                    unsigned int u_1_5 = __as_u32(sc_34_1);
                    unsigned int cls_1_5 = u_1_5 & 4294965248u;
                    int f_36 = 0;
                    if ((t0_9 + 5 < fb || t0_9 + 5 >= lim - fe) && lim > t0_9 + 5) {
                        f_36 = 1;
                    }
                    if (f_36 != 0) {
                        cls_1_5 = 2139092992;
                    }
                    unsigned int key_4 = 0;
                    if ((qf < __uint_as_float(cls_1_5) || __uint_as_float(cls_1_5) == qf && cls_1_5 < qc) && cls_1_5 < 4278190080u) {
                        key_4 = 1073741824 | (unsigned int)(t0_9 + 5);
                    }
                    if (cls_1_5 == qc) {
                        unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 2047) & 2047;
                        key_4 = 536870912 | lowb_5 << 11 | (unsigned int)(t0_9 + 5);
                    }
                    kb2[5] = __uint_as_float(key_4);
                    float sc_35 = __uint_as_float(cb2[6]);
                    sc_35 = fmaxf(sc_35, -1.7014118e+38f);
                    sc_35 = fminf(sc_35, 1.7014118e+38f);
                    unsigned int u_1_6 = __as_u32(sc_35);
                    unsigned int cls_1_6 = u_1_6 & 4294965248u;
                    int f_37 = 0;
                    if ((t0_9 + 6 < fb || t0_9 + 6 >= lim - fe) && lim > t0_9 + 6) {
                        f_37 = 1;
                    }
                    if (f_37 != 0) {
                        cls_1_6 = 2139092992;
                    }
                    unsigned int key_5 = 0;
                    if ((qf < __uint_as_float(cls_1_6) || __uint_as_float(cls_1_6) == qf && cls_1_6 < qc) && cls_1_6 < 4278190080u) {
                        key_5 = 1073741824 | (unsigned int)(t0_9 + 6);
                    }
                    if (cls_1_6 == qc) {
                        unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 2047) & 2047;
                        key_5 = 536870912 | lowb_6 << 11 | (unsigned int)(t0_9 + 6);
                    }
                    kb2[6] = __uint_as_float(key_5);
                    float sc_36 = __uint_as_float(cb2[7]);
                    sc_36 = fmaxf(sc_36, -1.7014118e+38f);
                    sc_36 = fminf(sc_36, 1.7014118e+38f);
                    unsigned int u_1_7 = __as_u32(sc_36);
                    unsigned int cls_1_7 = u_1_7 & 4294965248u;
                    int f_38 = 0;
                    if ((t0_9 + 7 < fb || t0_9 + 7 >= lim - fe) && lim > t0_9 + 7) {
                        f_38 = 1;
                    }
                    if (f_38 != 0) {
                        cls_1_7 = 2139092992;
                    }
                    unsigned int key_6 = 0;
                    if ((qf < __uint_as_float(cls_1_7) || __uint_as_float(cls_1_7) == qf && cls_1_7 < qc) && cls_1_7 < 4278190080u) {
                        key_6 = 1073741824 | (unsigned int)(t0_9 + 7);
                    }
                    if (cls_1_7 == qc) {
                        unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 2047) & 2047;
                        key_6 = 536870912 | lowb_7 << 11 | (unsigned int)(t0_9 + 7);
                    }
                    kb2[7] = __uint_as_float(key_6);
                    float sc_37 = __uint_as_float(cb2[8]);
                    sc_37 = fmaxf(sc_37, -1.7014118e+38f);
                    sc_37 = fminf(sc_37, 1.7014118e+38f);
                    unsigned int u_1_8 = __as_u32(sc_37);
                    unsigned int cls_1_8 = u_1_8 & 4294965248u;
                    int f_39 = 0;
                    if ((t0_9 + 8 < fb || t0_9 + 8 >= lim - fe) && lim > t0_9 + 8) {
                        f_39 = 1;
                    }
                    if (f_39 != 0) {
                        cls_1_8 = 2139092992;
                    }
                    unsigned int key_7 = 0;
                    if ((qf < __uint_as_float(cls_1_8) || __uint_as_float(cls_1_8) == qf && cls_1_8 < qc) && cls_1_8 < 4278190080u) {
                        key_7 = 1073741824 | (unsigned int)(t0_9 + 8);
                    }
                    if (cls_1_8 == qc) {
                        unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 2047) & 2047;
                        key_7 = 536870912 | lowb_8 << 11 | (unsigned int)(t0_9 + 8);
                    }
                    kb2[8] = __uint_as_float(key_7);
                    float sc_38 = __uint_as_float(cb2[9]);
                    sc_38 = fmaxf(sc_38, -1.7014118e+38f);
                    sc_38 = fminf(sc_38, 1.7014118e+38f);
                    unsigned int u_1_9 = __as_u32(sc_38);
                    unsigned int cls_1_9 = u_1_9 & 4294965248u;
                    int f_40 = 0;
                    if ((t0_9 + 9 < fb || t0_9 + 9 >= lim - fe) && lim > t0_9 + 9) {
                        f_40 = 1;
                    }
                    if (f_40 != 0) {
                        cls_1_9 = 2139092992;
                    }
                    unsigned int key_8 = 0;
                    if ((qf < __uint_as_float(cls_1_9) || __uint_as_float(cls_1_9) == qf && cls_1_9 < qc) && cls_1_9 < 4278190080u) {
                        key_8 = 1073741824 | (unsigned int)(t0_9 + 9);
                    }
                    if (cls_1_9 == qc) {
                        unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 2047) & 2047;
                        key_8 = 536870912 | lowb_9 << 11 | (unsigned int)(t0_9 + 9);
                    }
                    kb2[9] = __uint_as_float(key_8);
                    float sc_39 = __uint_as_float(cb2[10]);
                    sc_39 = fmaxf(sc_39, -1.7014118e+38f);
                    sc_39 = fminf(sc_39, 1.7014118e+38f);
                    unsigned int u_1_10 = __as_u32(sc_39);
                    unsigned int cls_1_10 = u_1_10 & 4294965248u;
                    int f_41 = 0;
                    if ((t0_9 + 10 < fb || t0_9 + 10 >= lim - fe) && lim > t0_9 + 10) {
                        f_41 = 1;
                    }
                    if (f_41 != 0) {
                        cls_1_10 = 2139092992;
                    }
                    unsigned int key_9 = 0;
                    if ((qf < __uint_as_float(cls_1_10) || __uint_as_float(cls_1_10) == qf && cls_1_10 < qc) && cls_1_10 < 4278190080u) {
                        key_9 = 1073741824 | (unsigned int)(t0_9 + 10);
                    }
                    if (cls_1_10 == qc) {
                        unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 2047) & 2047;
                        key_9 = 536870912 | lowb_10 << 11 | (unsigned int)(t0_9 + 10);
                    }
                    kb2[10] = __uint_as_float(key_9);
                    float sc_40 = __uint_as_float(cb2[11]);
                    sc_40 = fmaxf(sc_40, -1.7014118e+38f);
                    sc_40 = fminf(sc_40, 1.7014118e+38f);
                    unsigned int u_1_11 = __as_u32(sc_40);
                    unsigned int cls_1_11 = u_1_11 & 4294965248u;
                    int f_42 = 0;
                    if ((t0_9 + 11 < fb || t0_9 + 11 >= lim - fe) && lim > t0_9 + 11) {
                        f_42 = 1;
                    }
                    if (f_42 != 0) {
                        cls_1_11 = 2139092992;
                    }
                    unsigned int key_10 = 0;
                    if ((qf < __uint_as_float(cls_1_11) || __uint_as_float(cls_1_11) == qf && cls_1_11 < qc) && cls_1_11 < 4278190080u) {
                        key_10 = 1073741824 | (unsigned int)(t0_9 + 11);
                    }
                    if (cls_1_11 == qc) {
                        unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 2047) & 2047;
                        key_10 = 536870912 | lowb_11 << 11 | (unsigned int)(t0_9 + 11);
                    }
                    kb2[11] = __uint_as_float(key_10);
                    float sc_41 = __uint_as_float(cb2[12]);
                    sc_41 = fmaxf(sc_41, -1.7014118e+38f);
                    sc_41 = fminf(sc_41, 1.7014118e+38f);
                    unsigned int u_1_12 = __as_u32(sc_41);
                    unsigned int cls_1_12 = u_1_12 & 4294965248u;
                    int f_43 = 0;
                    if ((t0_9 + 12 < fb || t0_9 + 12 >= lim - fe) && lim > t0_9 + 12) {
                        f_43 = 1;
                    }
                    if (f_43 != 0) {
                        cls_1_12 = 2139092992;
                    }
                    unsigned int key_11 = 0;
                    if ((qf < __uint_as_float(cls_1_12) || __uint_as_float(cls_1_12) == qf && cls_1_12 < qc) && cls_1_12 < 4278190080u) {
                        key_11 = 1073741824 | (unsigned int)(t0_9 + 12);
                    }
                    if (cls_1_12 == qc) {
                        unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 2047) & 2047;
                        key_11 = 536870912 | lowb_12 << 11 | (unsigned int)(t0_9 + 12);
                    }
                    kb2[12] = __uint_as_float(key_11);
                    float sc_42 = __uint_as_float(cb2[13]);
                    sc_42 = fmaxf(sc_42, -1.7014118e+38f);
                    sc_42 = fminf(sc_42, 1.7014118e+38f);
                    unsigned int u_1_13 = __as_u32(sc_42);
                    unsigned int cls_1_13 = u_1_13 & 4294965248u;
                    int f_44 = 0;
                    if ((t0_9 + 13 < fb || t0_9 + 13 >= lim - fe) && lim > t0_9 + 13) {
                        f_44 = 1;
                    }
                    if (f_44 != 0) {
                        cls_1_13 = 2139092992;
                    }
                    unsigned int key_12 = 0;
                    if ((qf < __uint_as_float(cls_1_13) || __uint_as_float(cls_1_13) == qf && cls_1_13 < qc) && cls_1_13 < 4278190080u) {
                        key_12 = 1073741824 | (unsigned int)(t0_9 + 13);
                    }
                    if (cls_1_13 == qc) {
                        unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 2047) & 2047;
                        key_12 = 536870912 | lowb_13 << 11 | (unsigned int)(t0_9 + 13);
                    }
                    kb2[13] = __uint_as_float(key_12);
                    float sc_43 = __uint_as_float(cb2[14]);
                    sc_43 = fmaxf(sc_43, -1.7014118e+38f);
                    sc_43 = fminf(sc_43, 1.7014118e+38f);
                    unsigned int u_1_14 = __as_u32(sc_43);
                    unsigned int cls_1_14 = u_1_14 & 4294965248u;
                    int f_45 = 0;
                    if ((t0_9 + 14 < fb || t0_9 + 14 >= lim - fe) && lim > t0_9 + 14) {
                        f_45 = 1;
                    }
                    if (f_45 != 0) {
                        cls_1_14 = 2139092992;
                    }
                    unsigned int key_13 = 0;
                    if ((qf < __uint_as_float(cls_1_14) || __uint_as_float(cls_1_14) == qf && cls_1_14 < qc) && cls_1_14 < 4278190080u) {
                        key_13 = 1073741824 | (unsigned int)(t0_9 + 14);
                    }
                    if (cls_1_14 == qc) {
                        unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 2047) & 2047;
                        key_13 = 536870912 | lowb_14 << 11 | (unsigned int)(t0_9 + 14);
                    }
                    kb2[14] = __uint_as_float(key_13);
                    float sc_44 = __uint_as_float(cb2[15]);
                    sc_44 = fmaxf(sc_44, -1.7014118e+38f);
                    sc_44 = fminf(sc_44, 1.7014118e+38f);
                    unsigned int u_1_15 = __as_u32(sc_44);
                    unsigned int cls_1_15 = u_1_15 & 4294965248u;
                    int f_46 = 0;
                    if ((t0_9 + 15 < fb || t0_9 + 15 >= lim - fe) && lim > t0_9 + 15) {
                        f_46 = 1;
                    }
                    if (f_46 != 0) {
                        cls_1_15 = 2139092992;
                    }
                    unsigned int key_14 = 0;
                    if ((qf < __uint_as_float(cls_1_15) || __uint_as_float(cls_1_15) == qf && cls_1_15 < qc) && cls_1_15 < 4278190080u) {
                        key_14 = 1073741824 | (unsigned int)(t0_9 + 15);
                    }
                    if (cls_1_15 == qc) {
                        unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 2047) & 2047;
                        key_14 = 536870912 | lowb_15 << 11 | (unsigned int)(t0_9 + 15);
                    }
                    kb2[15] = __uint_as_float(key_14);
                    unsigned int nzc = 0;
                    nzc = nzc | __as_u32(kb2[0]);
                    nzc = nzc | __as_u32(kb2[1]);
                    nzc = nzc | __as_u32(kb2[2]);
                    nzc = nzc | __as_u32(kb2[3]);
                    nzc = nzc | __as_u32(kb2[4]);
                    nzc = nzc | __as_u32(kb2[5]);
                    nzc = nzc | __as_u32(kb2[6]);
                    nzc = nzc | __as_u32(kb2[7]);
                    nzc = nzc | __as_u32(kb2[8]);
                    nzc = nzc | __as_u32(kb2[9]);
                    nzc = nzc | __as_u32(kb2[10]);
                    nzc = nzc | __as_u32(kb2[11]);
                    nzc = nzc | __as_u32(kb2[12]);
                    nzc = nzc | __as_u32(kb2[13]);
                    nzc = nzc | __as_u32(kb2[14]);
                    nzc = nzc | __as_u32(kb2[15]);
                    int _warp_vote_10 = __any_sync(0xFFFFFFFF, nzc != 0);
                    if (_warp_vote_10 != 0) {
                        float hi_1_120 = fmaxf(kb2[0], kb2[13]);
                        float lo_1_120 = fminf(kb2[0], kb2[13]);
                        kb2[0] = hi_1_120;
                        kb2[13] = lo_1_120;
                        float hi_1_121 = fmaxf(kb2[1], kb2[12]);
                        float lo_1_121 = fminf(kb2[1], kb2[12]);
                        kb2[1] = hi_1_121;
                        kb2[12] = lo_1_121;
                        float hi_1_122 = fmaxf(kb2[2], kb2[15]);
                        float lo_1_122 = fminf(kb2[2], kb2[15]);
                        kb2[2] = hi_1_122;
                        kb2[15] = lo_1_122;
                        float hi_1_123 = fmaxf(kb2[3], kb2[14]);
                        float lo_1_123 = fminf(kb2[3], kb2[14]);
                        kb2[3] = hi_1_123;
                        kb2[14] = lo_1_123;
                        float hi_1_124 = fmaxf(kb2[4], kb2[8]);
                        float lo_1_124 = fminf(kb2[4], kb2[8]);
                        kb2[4] = hi_1_124;
                        kb2[8] = lo_1_124;
                        float hi_1_125 = fmaxf(kb2[5], kb2[6]);
                        float lo_1_125 = fminf(kb2[5], kb2[6]);
                        kb2[5] = hi_1_125;
                        kb2[6] = lo_1_125;
                        float hi_1_126 = fmaxf(kb2[7], kb2[11]);
                        float lo_1_126 = fminf(kb2[7], kb2[11]);
                        kb2[7] = hi_1_126;
                        kb2[11] = lo_1_126;
                        float hi_1_127 = fmaxf(kb2[9], kb2[10]);
                        float lo_1_127 = fminf(kb2[9], kb2[10]);
                        kb2[9] = hi_1_127;
                        kb2[10] = lo_1_127;
                        float hi_1_128 = fmaxf(kb2[0], kb2[5]);
                        float lo_1_128 = fminf(kb2[0], kb2[5]);
                        kb2[0] = hi_1_128;
                        kb2[5] = lo_1_128;
                        float hi_1_129 = fmaxf(kb2[1], kb2[7]);
                        float lo_1_129 = fminf(kb2[1], kb2[7]);
                        kb2[1] = hi_1_129;
                        kb2[7] = lo_1_129;
                        float hi_1_130 = fmaxf(kb2[2], kb2[9]);
                        float lo_1_130 = fminf(kb2[2], kb2[9]);
                        kb2[2] = hi_1_130;
                        kb2[9] = lo_1_130;
                        float hi_1_131 = fmaxf(kb2[3], kb2[4]);
                        float lo_1_131 = fminf(kb2[3], kb2[4]);
                        kb2[3] = hi_1_131;
                        kb2[4] = lo_1_131;
                        float hi_1_132 = fmaxf(kb2[6], kb2[13]);
                        float lo_1_132 = fminf(kb2[6], kb2[13]);
                        kb2[6] = hi_1_132;
                        kb2[13] = lo_1_132;
                        float hi_1_133 = fmaxf(kb2[8], kb2[14]);
                        float lo_1_133 = fminf(kb2[8], kb2[14]);
                        kb2[8] = hi_1_133;
                        kb2[14] = lo_1_133;
                        float hi_1_134 = fmaxf(kb2[10], kb2[15]);
                        float lo_1_134 = fminf(kb2[10], kb2[15]);
                        kb2[10] = hi_1_134;
                        kb2[15] = lo_1_134;
                        float hi_1_135 = fmaxf(kb2[11], kb2[12]);
                        float lo_1_135 = fminf(kb2[11], kb2[12]);
                        kb2[11] = hi_1_135;
                        kb2[12] = lo_1_135;
                        float hi_1_136 = fmaxf(kb2[0], kb2[1]);
                        float lo_1_136 = fminf(kb2[0], kb2[1]);
                        kb2[0] = hi_1_136;
                        kb2[1] = lo_1_136;
                        float hi_1_137 = fmaxf(kb2[2], kb2[3]);
                        float lo_1_137 = fminf(kb2[2], kb2[3]);
                        kb2[2] = hi_1_137;
                        kb2[3] = lo_1_137;
                        float hi_1_138 = fmaxf(kb2[4], kb2[5]);
                        float lo_1_138 = fminf(kb2[4], kb2[5]);
                        kb2[4] = hi_1_138;
                        kb2[5] = lo_1_138;
                        float hi_1_139 = fmaxf(kb2[6], kb2[8]);
                        float lo_1_139 = fminf(kb2[6], kb2[8]);
                        kb2[6] = hi_1_139;
                        kb2[8] = lo_1_139;
                        float hi_1_140 = fmaxf(kb2[7], kb2[9]);
                        float lo_1_140 = fminf(kb2[7], kb2[9]);
                        kb2[7] = hi_1_140;
                        kb2[9] = lo_1_140;
                        float hi_1_141 = fmaxf(kb2[10], kb2[11]);
                        float lo_1_141 = fminf(kb2[10], kb2[11]);
                        kb2[10] = hi_1_141;
                        kb2[11] = lo_1_141;
                        float hi_1_142 = fmaxf(kb2[12], kb2[13]);
                        float lo_1_142 = fminf(kb2[12], kb2[13]);
                        kb2[12] = hi_1_142;
                        kb2[13] = lo_1_142;
                        float hi_1_143 = fmaxf(kb2[14], kb2[15]);
                        float lo_1_143 = fminf(kb2[14], kb2[15]);
                        kb2[14] = hi_1_143;
                        kb2[15] = lo_1_143;
                        float hi_1_144 = fmaxf(kb2[0], kb2[2]);
                        float lo_1_144 = fminf(kb2[0], kb2[2]);
                        kb2[0] = hi_1_144;
                        kb2[2] = lo_1_144;
                        float hi_1_145 = fmaxf(kb2[1], kb2[3]);
                        float lo_1_145 = fminf(kb2[1], kb2[3]);
                        kb2[1] = hi_1_145;
                        kb2[3] = lo_1_145;
                        float hi_1_146 = fmaxf(kb2[4], kb2[10]);
                        float lo_1_146 = fminf(kb2[4], kb2[10]);
                        kb2[4] = hi_1_146;
                        kb2[10] = lo_1_146;
                        float hi_1_147 = fmaxf(kb2[5], kb2[11]);
                        float lo_1_147 = fminf(kb2[5], kb2[11]);
                        kb2[5] = hi_1_147;
                        kb2[11] = lo_1_147;
                        float hi_1_148 = fmaxf(kb2[6], kb2[7]);
                        float lo_1_148 = fminf(kb2[6], kb2[7]);
                        kb2[6] = hi_1_148;
                        kb2[7] = lo_1_148;
                        float hi_1_149 = fmaxf(kb2[8], kb2[9]);
                        float lo_1_149 = fminf(kb2[8], kb2[9]);
                        kb2[8] = hi_1_149;
                        kb2[9] = lo_1_149;
                        float hi_1_150 = fmaxf(kb2[12], kb2[14]);
                        float lo_1_150 = fminf(kb2[12], kb2[14]);
                        kb2[12] = hi_1_150;
                        kb2[14] = lo_1_150;
                        float hi_1_151 = fmaxf(kb2[13], kb2[15]);
                        float lo_1_151 = fminf(kb2[13], kb2[15]);
                        kb2[13] = hi_1_151;
                        kb2[15] = lo_1_151;
                        float hi_1_152 = fmaxf(kb2[1], kb2[2]);
                        float lo_1_152 = fminf(kb2[1], kb2[2]);
                        kb2[1] = hi_1_152;
                        kb2[2] = lo_1_152;
                        float hi_1_153 = fmaxf(kb2[3], kb2[12]);
                        float lo_1_153 = fminf(kb2[3], kb2[12]);
                        kb2[3] = hi_1_153;
                        kb2[12] = lo_1_153;
                        float hi_1_154 = fmaxf(kb2[4], kb2[6]);
                        float lo_1_154 = fminf(kb2[4], kb2[6]);
                        kb2[4] = hi_1_154;
                        kb2[6] = lo_1_154;
                        float hi_1_155 = fmaxf(kb2[5], kb2[7]);
                        float lo_1_155 = fminf(kb2[5], kb2[7]);
                        kb2[5] = hi_1_155;
                        kb2[7] = lo_1_155;
                        float hi_1_156 = fmaxf(kb2[8], kb2[10]);
                        float lo_1_156 = fminf(kb2[8], kb2[10]);
                        kb2[8] = hi_1_156;
                        kb2[10] = lo_1_156;
                        float hi_1_157 = fmaxf(kb2[9], kb2[11]);
                        float lo_1_157 = fminf(kb2[9], kb2[11]);
                        kb2[9] = hi_1_157;
                        kb2[11] = lo_1_157;
                        float hi_1_158 = fmaxf(kb2[13], kb2[14]);
                        float lo_1_158 = fminf(kb2[13], kb2[14]);
                        kb2[13] = hi_1_158;
                        kb2[14] = lo_1_158;
                        float hi_1_159 = fmaxf(kb2[1], kb2[4]);
                        float lo_1_159 = fminf(kb2[1], kb2[4]);
                        kb2[1] = hi_1_159;
                        kb2[4] = lo_1_159;
                        float hi_1_160 = fmaxf(kb2[2], kb2[6]);
                        float lo_1_160 = fminf(kb2[2], kb2[6]);
                        kb2[2] = hi_1_160;
                        kb2[6] = lo_1_160;
                        float hi_1_161 = fmaxf(kb2[5], kb2[8]);
                        float lo_1_161 = fminf(kb2[5], kb2[8]);
                        kb2[5] = hi_1_161;
                        kb2[8] = lo_1_161;
                        float hi_1_162 = fmaxf(kb2[7], kb2[10]);
                        float lo_1_162 = fminf(kb2[7], kb2[10]);
                        kb2[7] = hi_1_162;
                        kb2[10] = lo_1_162;
                        float hi_1_163 = fmaxf(kb2[9], kb2[13]);
                        float lo_1_163 = fminf(kb2[9], kb2[13]);
                        kb2[9] = hi_1_163;
                        kb2[13] = lo_1_163;
                        float hi_1_164 = fmaxf(kb2[11], kb2[14]);
                        float lo_1_164 = fminf(kb2[11], kb2[14]);
                        kb2[11] = hi_1_164;
                        kb2[14] = lo_1_164;
                        float hi_1_165 = fmaxf(kb2[2], kb2[4]);
                        float lo_1_165 = fminf(kb2[2], kb2[4]);
                        kb2[2] = hi_1_165;
                        kb2[4] = lo_1_165;
                        float hi_1_166 = fmaxf(kb2[3], kb2[6]);
                        float lo_1_166 = fminf(kb2[3], kb2[6]);
                        kb2[3] = hi_1_166;
                        kb2[6] = lo_1_166;
                        float hi_1_167 = fmaxf(kb2[9], kb2[12]);
                        float lo_1_167 = fminf(kb2[9], kb2[12]);
                        kb2[9] = hi_1_167;
                        kb2[12] = lo_1_167;
                        float hi_1_168 = fmaxf(kb2[11], kb2[13]);
                        float lo_1_168 = fminf(kb2[11], kb2[13]);
                        kb2[11] = hi_1_168;
                        kb2[13] = lo_1_168;
                        float hi_1_169 = fmaxf(kb2[3], kb2[5]);
                        float lo_1_169 = fminf(kb2[3], kb2[5]);
                        kb2[3] = hi_1_169;
                        kb2[5] = lo_1_169;
                        float hi_1_170 = fmaxf(kb2[6], kb2[8]);
                        float lo_1_170 = fminf(kb2[6], kb2[8]);
                        kb2[6] = hi_1_170;
                        kb2[8] = lo_1_170;
                        float hi_1_171 = fmaxf(kb2[7], kb2[9]);
                        float lo_1_171 = fminf(kb2[7], kb2[9]);
                        kb2[7] = hi_1_171;
                        kb2[9] = lo_1_171;
                        float hi_1_172 = fmaxf(kb2[10], kb2[12]);
                        float lo_1_172 = fminf(kb2[10], kb2[12]);
                        kb2[10] = hi_1_172;
                        kb2[12] = lo_1_172;
                        float hi_1_173 = fmaxf(kb2[3], kb2[4]);
                        float lo_1_173 = fminf(kb2[3], kb2[4]);
                        kb2[3] = hi_1_173;
                        kb2[4] = lo_1_173;
                        float hi_1_174 = fmaxf(kb2[5], kb2[6]);
                        float lo_1_174 = fminf(kb2[5], kb2[6]);
                        kb2[5] = hi_1_174;
                        kb2[6] = lo_1_174;
                        float hi_1_175 = fmaxf(kb2[7], kb2[8]);
                        float lo_1_175 = fminf(kb2[7], kb2[8]);
                        kb2[7] = hi_1_175;
                        kb2[8] = lo_1_175;
                        float hi_1_176 = fmaxf(kb2[9], kb2[10]);
                        float lo_1_176 = fminf(kb2[9], kb2[10]);
                        kb2[9] = hi_1_176;
                        kb2[10] = lo_1_176;
                        float hi_1_177 = fmaxf(kb2[11], kb2[12]);
                        float lo_1_177 = fminf(kb2[11], kb2[12]);
                        kb2[11] = hi_1_177;
                        kb2[12] = lo_1_177;
                        float hi_1_178 = fmaxf(kb2[6], kb2[7]);
                        float lo_1_178 = fminf(kb2[6], kb2[7]);
                        kb2[6] = hi_1_178;
                        kb2[7] = lo_1_178;
                        float hi_1_179 = fmaxf(kb2[8], kb2[9]);
                        float lo_1_179 = fminf(kb2[8], kb2[9]);
                        kb2[8] = hi_1_179;
                        kb2[9] = lo_1_179;
                        float hi_208 = fmaxf(a2x[0], kb2[15]);
                        a2x[0] = hi_208;
                        float hi_209 = fmaxf(a2x[1], kb2[14]);
                        a2x[1] = hi_209;
                        float hi_210 = fmaxf(a2x[2], kb2[13]);
                        a2x[2] = hi_210;
                        float hi_211 = fmaxf(a2x[3], kb2[12]);
                        a2x[3] = hi_211;
                        float hi_212 = fmaxf(a2x[4], kb2[11]);
                        a2x[4] = hi_212;
                        float hi_213 = fmaxf(a2x[5], kb2[10]);
                        a2x[5] = hi_213;
                        float hi_214 = fmaxf(a2x[6], kb2[9]);
                        a2x[6] = hi_214;
                        float hi_215 = fmaxf(a2x[7], kb2[8]);
                        a2x[7] = hi_215;
                        float hi_216 = fmaxf(a2x[8], kb2[7]);
                        a2x[8] = hi_216;
                        float hi_217 = fmaxf(a2x[9], kb2[6]);
                        a2x[9] = hi_217;
                        float hi_218 = fmaxf(a2x[10], kb2[5]);
                        a2x[10] = hi_218;
                        float hi_219 = fmaxf(a2x[11], kb2[4]);
                        a2x[11] = hi_219;
                        float hi_220 = fmaxf(a2x[12], kb2[3]);
                        a2x[12] = hi_220;
                        float hi_221 = fmaxf(a2x[13], kb2[2]);
                        a2x[13] = hi_221;
                        float hi_222 = fmaxf(a2x[14], kb2[1]);
                        a2x[14] = hi_222;
                        float hi_223 = fmaxf(a2x[15], kb2[0]);
                        a2x[15] = hi_223;
                        float hi_224 = fmaxf(a2x[0], a2x[8]);
                        float lo_208 = fminf(a2x[0], a2x[8]);
                        a2x[0] = hi_224;
                        a2x[8] = lo_208;
                        float hi_225 = fmaxf(a2x[1], a2x[9]);
                        float lo_209 = fminf(a2x[1], a2x[9]);
                        a2x[1] = hi_225;
                        a2x[9] = lo_209;
                        float hi_226 = fmaxf(a2x[2], a2x[10]);
                        float lo_210 = fminf(a2x[2], a2x[10]);
                        a2x[2] = hi_226;
                        a2x[10] = lo_210;
                        float hi_227 = fmaxf(a2x[3], a2x[11]);
                        float lo_211 = fminf(a2x[3], a2x[11]);
                        a2x[3] = hi_227;
                        a2x[11] = lo_211;
                        float hi_228 = fmaxf(a2x[4], a2x[12]);
                        float lo_212 = fminf(a2x[4], a2x[12]);
                        a2x[4] = hi_228;
                        a2x[12] = lo_212;
                        float hi_229 = fmaxf(a2x[5], a2x[13]);
                        float lo_213 = fminf(a2x[5], a2x[13]);
                        a2x[5] = hi_229;
                        a2x[13] = lo_213;
                        float hi_230 = fmaxf(a2x[6], a2x[14]);
                        float lo_214 = fminf(a2x[6], a2x[14]);
                        a2x[6] = hi_230;
                        a2x[14] = lo_214;
                        float hi_231 = fmaxf(a2x[7], a2x[15]);
                        float lo_215 = fminf(a2x[7], a2x[15]);
                        a2x[7] = hi_231;
                        a2x[15] = lo_215;
                        float hi_232 = fmaxf(a2x[0], a2x[4]);
                        float lo_216 = fminf(a2x[0], a2x[4]);
                        a2x[0] = hi_232;
                        a2x[4] = lo_216;
                        float hi_233 = fmaxf(a2x[1], a2x[5]);
                        float lo_217 = fminf(a2x[1], a2x[5]);
                        a2x[1] = hi_233;
                        a2x[5] = lo_217;
                        float hi_234 = fmaxf(a2x[2], a2x[6]);
                        float lo_218 = fminf(a2x[2], a2x[6]);
                        a2x[2] = hi_234;
                        a2x[6] = lo_218;
                        float hi_235 = fmaxf(a2x[3], a2x[7]);
                        float lo_219 = fminf(a2x[3], a2x[7]);
                        a2x[3] = hi_235;
                        a2x[7] = lo_219;
                        float hi_236 = fmaxf(a2x[8], a2x[12]);
                        float lo_220 = fminf(a2x[8], a2x[12]);
                        a2x[8] = hi_236;
                        a2x[12] = lo_220;
                        float hi_237 = fmaxf(a2x[9], a2x[13]);
                        float lo_221 = fminf(a2x[9], a2x[13]);
                        a2x[9] = hi_237;
                        a2x[13] = lo_221;
                        float hi_238 = fmaxf(a2x[10], a2x[14]);
                        float lo_222 = fminf(a2x[10], a2x[14]);
                        a2x[10] = hi_238;
                        a2x[14] = lo_222;
                        float hi_239 = fmaxf(a2x[11], a2x[15]);
                        float lo_223 = fminf(a2x[11], a2x[15]);
                        a2x[11] = hi_239;
                        a2x[15] = lo_223;
                        float hi_240 = fmaxf(a2x[0], a2x[2]);
                        float lo_224 = fminf(a2x[0], a2x[2]);
                        a2x[0] = hi_240;
                        a2x[2] = lo_224;
                        float hi_241 = fmaxf(a2x[1], a2x[3]);
                        float lo_225 = fminf(a2x[1], a2x[3]);
                        a2x[1] = hi_241;
                        a2x[3] = lo_225;
                        float hi_242 = fmaxf(a2x[4], a2x[6]);
                        float lo_226 = fminf(a2x[4], a2x[6]);
                        a2x[4] = hi_242;
                        a2x[6] = lo_226;
                        float hi_243 = fmaxf(a2x[5], a2x[7]);
                        float lo_227 = fminf(a2x[5], a2x[7]);
                        a2x[5] = hi_243;
                        a2x[7] = lo_227;
                        float hi_244 = fmaxf(a2x[8], a2x[10]);
                        float lo_228 = fminf(a2x[8], a2x[10]);
                        a2x[8] = hi_244;
                        a2x[10] = lo_228;
                        float hi_245 = fmaxf(a2x[9], a2x[11]);
                        float lo_229 = fminf(a2x[9], a2x[11]);
                        a2x[9] = hi_245;
                        a2x[11] = lo_229;
                        float hi_246 = fmaxf(a2x[12], a2x[14]);
                        float lo_230 = fminf(a2x[12], a2x[14]);
                        a2x[12] = hi_246;
                        a2x[14] = lo_230;
                        float hi_247 = fmaxf(a2x[13], a2x[15]);
                        float lo_231 = fminf(a2x[13], a2x[15]);
                        a2x[13] = hi_247;
                        a2x[15] = lo_231;
                        float hi_248 = fmaxf(a2x[0], a2x[1]);
                        float lo_232 = fminf(a2x[0], a2x[1]);
                        a2x[0] = hi_248;
                        a2x[1] = lo_232;
                        float hi_249 = fmaxf(a2x[2], a2x[3]);
                        float lo_233 = fminf(a2x[2], a2x[3]);
                        a2x[2] = hi_249;
                        a2x[3] = lo_233;
                        float hi_250 = fmaxf(a2x[4], a2x[5]);
                        float lo_234 = fminf(a2x[4], a2x[5]);
                        a2x[4] = hi_250;
                        a2x[5] = lo_234;
                        float hi_251 = fmaxf(a2x[6], a2x[7]);
                        float lo_235 = fminf(a2x[6], a2x[7]);
                        a2x[6] = hi_251;
                        a2x[7] = lo_235;
                        float hi_252 = fmaxf(a2x[8], a2x[9]);
                        float lo_236 = fminf(a2x[8], a2x[9]);
                        a2x[8] = hi_252;
                        a2x[9] = lo_236;
                        float hi_253 = fmaxf(a2x[10], a2x[11]);
                        float lo_237 = fminf(a2x[10], a2x[11]);
                        a2x[10] = hi_253;
                        a2x[11] = lo_237;
                        float hi_254 = fmaxf(a2x[12], a2x[13]);
                        float lo_238 = fminf(a2x[12], a2x[13]);
                        a2x[12] = hi_254;
                        a2x[13] = lo_238;
                        float hi_255 = fmaxf(a2x[14], a2x[15]);
                        float lo_239 = fminf(a2x[14], a2x[15]);
                        a2x[14] = hi_255;
                        a2x[15] = lo_239;
                    }
                    cb2[0] = nb2[0];
                    cb2[1] = nb2[1];
                    cb2[2] = nb2[2];
                    cb2[3] = nb2[3];
                    cb2[4] = nb2[4];
                    cb2[5] = nb2[5];
                    cb2[6] = nb2[6];
                    cb2[7] = nb2[7];
                    cb2[8] = nb2[8];
                    cb2[9] = nb2[9];
                    cb2[10] = nb2[10];
                    cb2[11] = nb2[11];
                    cb2[12] = nb2[12];
                    cb2[13] = nb2[13];
                    cb2[14] = nb2[14];
                    cb2[15] = nb2[15];
                }
            }
            float rr2x[1];
            int pb_1 = tid_1 * 17;
            pub[pb_1] = a2x[0];
            pub[pb_1 + 1] = a2x[1];
            pub[pb_1 + 2] = a2x[2];
            pub[pb_1 + 3] = a2x[3];
            pub[pb_1 + 4] = a2x[4];
            pub[pb_1 + 5] = a2x[5];
            pub[pb_1 + 6] = a2x[6];
            pub[pb_1 + 7] = a2x[7];
            pub[pb_1 + 8] = a2x[8];
            pub[pb_1 + 9] = a2x[9];
            pub[pb_1 + 10] = a2x[10];
            pub[pb_1 + 11] = a2x[11];
            pub[pb_1 + 12] = a2x[12];
            pub[pb_1 + 13] = a2x[13];
            pub[pb_1 + 14] = a2x[14];
            pub[pb_1 + 15] = a2x[15];
            pub[pb_1 + 16] = neg_inf;
            asm volatile("barrier.sync 1, 512;" ::: "memory");
            float r_2 = neg_inf;
            float V_1[8];
            int s0_16 = (sg * 16 * 32 + cg) * 17;
            int s1_16 = ((sg * 16 + 8) * 32 + cg) * 17;
            float x0_16 = pub[s0_16 + ln];
            float y0_16 = pub[s1_16 + lnr];
            float lo0_16 = fminf(x0_16, y0_16);
            r_2 = fmaxf(r_2, lo0_16);
            float hi0_16 = fmaxf(x0_16, y0_16);
            float cur_27 = hi0_16;
            float pv_112 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 8);
            float hi_256 = fmaxf(cur_27, pv_112);
            float lo_240 = fminf(cur_27, pv_112);
            cur_27 = ((up[0] != 0) ? hi_256 : lo_240);
            float pv_113 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 4);
            float hi_257 = fmaxf(cur_27, pv_113);
            float lo_241 = fminf(cur_27, pv_113);
            cur_27 = ((up[1] != 0) ? hi_257 : lo_241);
            float pv_114 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 2);
            float hi_258 = fmaxf(cur_27, pv_114);
            float lo_242 = fminf(cur_27, pv_114);
            cur_27 = ((up[2] != 0) ? hi_258 : lo_242);
            float pv_115 = __shfl_xor_sync(0xFFFFFFFF, cur_27, 1);
            float hi_259 = fmaxf(cur_27, pv_115);
            float lo_243 = fminf(cur_27, pv_115);
            cur_27 = ((up[3] != 0) ? hi_259 : lo_243);
            V_1[0] = cur_27;
            int s0_17 = ((sg * 16 + 1) * 32 + cg) * 17;
            int s1_17 = ((sg * 16 + 1 + 8) * 32 + cg) * 17;
            float x0_17 = pub[s0_17 + ln];
            float y0_17 = pub[s1_17 + lnr];
            float lo0_17 = fminf(x0_17, y0_17);
            r_2 = fmaxf(r_2, lo0_17);
            float hi0_17 = fmaxf(x0_17, y0_17);
            float cur_28 = hi0_17;
            float pv_116 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
            float hi_260 = fmaxf(cur_28, pv_116);
            float lo_244 = fminf(cur_28, pv_116);
            cur_28 = ((up[0] != 0) ? hi_260 : lo_244);
            float pv_117 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
            float hi_261 = fmaxf(cur_28, pv_117);
            float lo_245 = fminf(cur_28, pv_117);
            cur_28 = ((up[1] != 0) ? hi_261 : lo_245);
            float pv_118 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
            float hi_262 = fmaxf(cur_28, pv_118);
            float lo_246 = fminf(cur_28, pv_118);
            cur_28 = ((up[2] != 0) ? hi_262 : lo_246);
            float pv_119 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
            float hi_263 = fmaxf(cur_28, pv_119);
            float lo_247 = fminf(cur_28, pv_119);
            cur_28 = ((up[3] != 0) ? hi_263 : lo_247);
            V_1[1] = cur_28;
            int s0_18 = ((sg * 16 + 2) * 32 + cg) * 17;
            int s1_18 = ((sg * 16 + 2 + 8) * 32 + cg) * 17;
            float x0_18 = pub[s0_18 + ln];
            float y0_18 = pub[s1_18 + lnr];
            float lo0_18 = fminf(x0_18, y0_18);
            r_2 = fmaxf(r_2, lo0_18);
            float hi0_18 = fmaxf(x0_18, y0_18);
            float cur_29 = hi0_18;
            float pv_120 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 8);
            float hi_264 = fmaxf(cur_29, pv_120);
            float lo_248 = fminf(cur_29, pv_120);
            cur_29 = ((up[0] != 0) ? hi_264 : lo_248);
            float pv_121 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 4);
            float hi_265 = fmaxf(cur_29, pv_121);
            float lo_249 = fminf(cur_29, pv_121);
            cur_29 = ((up[1] != 0) ? hi_265 : lo_249);
            float pv_122 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 2);
            float hi_266 = fmaxf(cur_29, pv_122);
            float lo_250 = fminf(cur_29, pv_122);
            cur_29 = ((up[2] != 0) ? hi_266 : lo_250);
            float pv_123 = __shfl_xor_sync(0xFFFFFFFF, cur_29, 1);
            float hi_267 = fmaxf(cur_29, pv_123);
            float lo_251 = fminf(cur_29, pv_123);
            cur_29 = ((up[3] != 0) ? hi_267 : lo_251);
            V_1[2] = cur_29;
            int s0_19 = ((sg * 16 + 3) * 32 + cg) * 17;
            int s1_19 = ((sg * 16 + 3 + 8) * 32 + cg) * 17;
            float x0_19 = pub[s0_19 + ln];
            float y0_19 = pub[s1_19 + lnr];
            float lo0_19 = fminf(x0_19, y0_19);
            r_2 = fmaxf(r_2, lo0_19);
            float hi0_19 = fmaxf(x0_19, y0_19);
            float cur_30 = hi0_19;
            float pv_124 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 8);
            float hi_268 = fmaxf(cur_30, pv_124);
            float lo_252 = fminf(cur_30, pv_124);
            cur_30 = ((up[0] != 0) ? hi_268 : lo_252);
            float pv_125 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 4);
            float hi_269 = fmaxf(cur_30, pv_125);
            float lo_253 = fminf(cur_30, pv_125);
            cur_30 = ((up[1] != 0) ? hi_269 : lo_253);
            float pv_126 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 2);
            float hi_270 = fmaxf(cur_30, pv_126);
            float lo_254 = fminf(cur_30, pv_126);
            cur_30 = ((up[2] != 0) ? hi_270 : lo_254);
            float pv_127 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 1);
            float hi_271 = fmaxf(cur_30, pv_127);
            float lo_255 = fminf(cur_30, pv_127);
            cur_30 = ((up[3] != 0) ? hi_271 : lo_255);
            V_1[3] = cur_30;
            int s0_20 = ((sg * 16 + 4) * 32 + cg) * 17;
            int s1_20 = ((sg * 16 + 4 + 8) * 32 + cg) * 17;
            float x0_20 = pub[s0_20 + ln];
            float y0_20 = pub[s1_20 + lnr];
            float lo0_20 = fminf(x0_20, y0_20);
            r_2 = fmaxf(r_2, lo0_20);
            float hi0_20 = fmaxf(x0_20, y0_20);
            float cur_31 = hi0_20;
            float pv_128 = __shfl_xor_sync(0xFFFFFFFF, cur_31, 8);
            float hi_272 = fmaxf(cur_31, pv_128);
            float lo_256 = fminf(cur_31, pv_128);
            cur_31 = ((up[0] != 0) ? hi_272 : lo_256);
            float pv_129 = __shfl_xor_sync(0xFFFFFFFF, cur_31, 4);
            float hi_273 = fmaxf(cur_31, pv_129);
            float lo_257 = fminf(cur_31, pv_129);
            cur_31 = ((up[1] != 0) ? hi_273 : lo_257);
            float pv_130 = __shfl_xor_sync(0xFFFFFFFF, cur_31, 2);
            float hi_274 = fmaxf(cur_31, pv_130);
            float lo_258 = fminf(cur_31, pv_130);
            cur_31 = ((up[2] != 0) ? hi_274 : lo_258);
            float pv_131 = __shfl_xor_sync(0xFFFFFFFF, cur_31, 1);
            float hi_275 = fmaxf(cur_31, pv_131);
            float lo_259 = fminf(cur_31, pv_131);
            cur_31 = ((up[3] != 0) ? hi_275 : lo_259);
            V_1[4] = cur_31;
            int s0_21 = ((sg * 16 + 5) * 32 + cg) * 17;
            int s1_21 = ((sg * 16 + 5 + 8) * 32 + cg) * 17;
            float x0_21 = pub[s0_21 + ln];
            float y0_21 = pub[s1_21 + lnr];
            float lo0_21 = fminf(x0_21, y0_21);
            r_2 = fmaxf(r_2, lo0_21);
            float hi0_21 = fmaxf(x0_21, y0_21);
            float cur_32 = hi0_21;
            float pv_132 = __shfl_xor_sync(0xFFFFFFFF, cur_32, 8);
            float hi_276 = fmaxf(cur_32, pv_132);
            float lo_260 = fminf(cur_32, pv_132);
            cur_32 = ((up[0] != 0) ? hi_276 : lo_260);
            float pv_133 = __shfl_xor_sync(0xFFFFFFFF, cur_32, 4);
            float hi_277 = fmaxf(cur_32, pv_133);
            float lo_261 = fminf(cur_32, pv_133);
            cur_32 = ((up[1] != 0) ? hi_277 : lo_261);
            float pv_134 = __shfl_xor_sync(0xFFFFFFFF, cur_32, 2);
            float hi_278 = fmaxf(cur_32, pv_134);
            float lo_262 = fminf(cur_32, pv_134);
            cur_32 = ((up[2] != 0) ? hi_278 : lo_262);
            float pv_135 = __shfl_xor_sync(0xFFFFFFFF, cur_32, 1);
            float hi_279 = fmaxf(cur_32, pv_135);
            float lo_263 = fminf(cur_32, pv_135);
            cur_32 = ((up[3] != 0) ? hi_279 : lo_263);
            V_1[5] = cur_32;
            int s0_22 = ((sg * 16 + 6) * 32 + cg) * 17;
            int s1_22 = ((sg * 16 + 6 + 8) * 32 + cg) * 17;
            float x0_22 = pub[s0_22 + ln];
            float y0_22 = pub[s1_22 + lnr];
            float lo0_22 = fminf(x0_22, y0_22);
            r_2 = fmaxf(r_2, lo0_22);
            float hi0_22 = fmaxf(x0_22, y0_22);
            float cur_33 = hi0_22;
            float pv_136 = __shfl_xor_sync(0xFFFFFFFF, cur_33, 8);
            float hi_280 = fmaxf(cur_33, pv_136);
            float lo_264 = fminf(cur_33, pv_136);
            cur_33 = ((up[0] != 0) ? hi_280 : lo_264);
            float pv_137 = __shfl_xor_sync(0xFFFFFFFF, cur_33, 4);
            float hi_281 = fmaxf(cur_33, pv_137);
            float lo_265 = fminf(cur_33, pv_137);
            cur_33 = ((up[1] != 0) ? hi_281 : lo_265);
            float pv_138 = __shfl_xor_sync(0xFFFFFFFF, cur_33, 2);
            float hi_282 = fmaxf(cur_33, pv_138);
            float lo_266 = fminf(cur_33, pv_138);
            cur_33 = ((up[2] != 0) ? hi_282 : lo_266);
            float pv_139 = __shfl_xor_sync(0xFFFFFFFF, cur_33, 1);
            float hi_283 = fmaxf(cur_33, pv_139);
            float lo_267 = fminf(cur_33, pv_139);
            cur_33 = ((up[3] != 0) ? hi_283 : lo_267);
            V_1[6] = cur_33;
            int s0_23 = ((sg * 16 + 7) * 32 + cg) * 17;
            int s1_23 = ((sg * 16 + 7 + 8) * 32 + cg) * 17;
            float x0_23 = pub[s0_23 + ln];
            float y0_23 = pub[s1_23 + lnr];
            float lo0_23 = fminf(x0_23, y0_23);
            r_2 = fmaxf(r_2, lo0_23);
            float hi0_23 = fmaxf(x0_23, y0_23);
            float cur_34 = hi0_23;
            float pv_140 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 8);
            float hi_284 = fmaxf(cur_34, pv_140);
            float lo_268 = fminf(cur_34, pv_140);
            cur_34 = ((up[0] != 0) ? hi_284 : lo_268);
            float pv_141 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 4);
            float hi_285 = fmaxf(cur_34, pv_141);
            float lo_269 = fminf(cur_34, pv_141);
            cur_34 = ((up[1] != 0) ? hi_285 : lo_269);
            float pv_142 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 2);
            float hi_286 = fmaxf(cur_34, pv_142);
            float lo_270 = fminf(cur_34, pv_142);
            cur_34 = ((up[2] != 0) ? hi_286 : lo_270);
            float pv_143 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 1);
            float hi_287 = fmaxf(cur_34, pv_143);
            float lo_271 = fminf(cur_34, pv_143);
            cur_34 = ((up[3] != 0) ? hi_287 : lo_271);
            V_1[7] = cur_34;
            float rs_1 = pub[((sg * 16 + ln) * 32 + cg) * 17 + 16];
            r_2 = fmaxf(r_2, rs_1);
            float y1_14 = __shfl_xor_sync(0xFFFFFFFF, V_1[4], 15);
            float lo1_14 = fminf(V_1[0], y1_14);
            r_2 = fmaxf(r_2, lo1_14);
            float hi1_14 = fmaxf(V_1[0], y1_14);
            float cur_35 = hi1_14;
            float pv_144 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 8);
            float hi_288 = fmaxf(cur_35, pv_144);
            float lo_272 = fminf(cur_35, pv_144);
            cur_35 = ((up[0] != 0) ? hi_288 : lo_272);
            float pv_145 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 4);
            float hi_289 = fmaxf(cur_35, pv_145);
            float lo_273 = fminf(cur_35, pv_145);
            cur_35 = ((up[1] != 0) ? hi_289 : lo_273);
            float pv_146 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 2);
            float hi_290 = fmaxf(cur_35, pv_146);
            float lo_274 = fminf(cur_35, pv_146);
            cur_35 = ((up[2] != 0) ? hi_290 : lo_274);
            float pv_147 = __shfl_xor_sync(0xFFFFFFFF, cur_35, 1);
            float hi_291 = fmaxf(cur_35, pv_147);
            float lo_275 = fminf(cur_35, pv_147);
            cur_35 = ((up[3] != 0) ? hi_291 : lo_275);
            V_1[0] = cur_35;
            float y1_15 = __shfl_xor_sync(0xFFFFFFFF, V_1[5], 15);
            float lo1_15 = fminf(V_1[1], y1_15);
            r_2 = fmaxf(r_2, lo1_15);
            float hi1_15 = fmaxf(V_1[1], y1_15);
            float cur_36 = hi1_15;
            float pv_148 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 8);
            float hi_292 = fmaxf(cur_36, pv_148);
            float lo_276 = fminf(cur_36, pv_148);
            cur_36 = ((up[0] != 0) ? hi_292 : lo_276);
            float pv_149 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 4);
            float hi_293 = fmaxf(cur_36, pv_149);
            float lo_277 = fminf(cur_36, pv_149);
            cur_36 = ((up[1] != 0) ? hi_293 : lo_277);
            float pv_150 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 2);
            float hi_294 = fmaxf(cur_36, pv_150);
            float lo_278 = fminf(cur_36, pv_150);
            cur_36 = ((up[2] != 0) ? hi_294 : lo_278);
            float pv_151 = __shfl_xor_sync(0xFFFFFFFF, cur_36, 1);
            float hi_295 = fmaxf(cur_36, pv_151);
            float lo_279 = fminf(cur_36, pv_151);
            cur_36 = ((up[3] != 0) ? hi_295 : lo_279);
            V_1[1] = cur_36;
            float y1_16 = __shfl_xor_sync(0xFFFFFFFF, V_1[6], 15);
            float lo1_16 = fminf(V_1[2], y1_16);
            r_2 = fmaxf(r_2, lo1_16);
            float hi1_16 = fmaxf(V_1[2], y1_16);
            float cur_37 = hi1_16;
            float pv_152 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 8);
            float hi_296 = fmaxf(cur_37, pv_152);
            float lo_280 = fminf(cur_37, pv_152);
            cur_37 = ((up[0] != 0) ? hi_296 : lo_280);
            float pv_153 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 4);
            float hi_297 = fmaxf(cur_37, pv_153);
            float lo_281 = fminf(cur_37, pv_153);
            cur_37 = ((up[1] != 0) ? hi_297 : lo_281);
            float pv_154 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 2);
            float hi_298 = fmaxf(cur_37, pv_154);
            float lo_282 = fminf(cur_37, pv_154);
            cur_37 = ((up[2] != 0) ? hi_298 : lo_282);
            float pv_155 = __shfl_xor_sync(0xFFFFFFFF, cur_37, 1);
            float hi_299 = fmaxf(cur_37, pv_155);
            float lo_283 = fminf(cur_37, pv_155);
            cur_37 = ((up[3] != 0) ? hi_299 : lo_283);
            V_1[2] = cur_37;
            float y1_17 = __shfl_xor_sync(0xFFFFFFFF, V_1[7], 15);
            float lo1_17 = fminf(V_1[3], y1_17);
            r_2 = fmaxf(r_2, lo1_17);
            float hi1_17 = fmaxf(V_1[3], y1_17);
            float cur_38 = hi1_17;
            float pv_156 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 8);
            float hi_300 = fmaxf(cur_38, pv_156);
            float lo_284 = fminf(cur_38, pv_156);
            cur_38 = ((up[0] != 0) ? hi_300 : lo_284);
            float pv_157 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 4);
            float hi_301 = fmaxf(cur_38, pv_157);
            float lo_285 = fminf(cur_38, pv_157);
            cur_38 = ((up[1] != 0) ? hi_301 : lo_285);
            float pv_158 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 2);
            float hi_302 = fmaxf(cur_38, pv_158);
            float lo_286 = fminf(cur_38, pv_158);
            cur_38 = ((up[2] != 0) ? hi_302 : lo_286);
            float pv_159 = __shfl_xor_sync(0xFFFFFFFF, cur_38, 1);
            float hi_303 = fmaxf(cur_38, pv_159);
            float lo_287 = fminf(cur_38, pv_159);
            cur_38 = ((up[3] != 0) ? hi_303 : lo_287);
            V_1[3] = cur_38;
            float y1_18 = __shfl_xor_sync(0xFFFFFFFF, V_1[2], 15);
            float lo1_18 = fminf(V_1[0], y1_18);
            r_2 = fmaxf(r_2, lo1_18);
            float hi1_18 = fmaxf(V_1[0], y1_18);
            float cur_39 = hi1_18;
            float pv_160 = __shfl_xor_sync(0xFFFFFFFF, cur_39, 8);
            float hi_304 = fmaxf(cur_39, pv_160);
            float lo_288 = fminf(cur_39, pv_160);
            cur_39 = ((up[0] != 0) ? hi_304 : lo_288);
            float pv_161 = __shfl_xor_sync(0xFFFFFFFF, cur_39, 4);
            float hi_305 = fmaxf(cur_39, pv_161);
            float lo_289 = fminf(cur_39, pv_161);
            cur_39 = ((up[1] != 0) ? hi_305 : lo_289);
            float pv_162 = __shfl_xor_sync(0xFFFFFFFF, cur_39, 2);
            float hi_306 = fmaxf(cur_39, pv_162);
            float lo_290 = fminf(cur_39, pv_162);
            cur_39 = ((up[2] != 0) ? hi_306 : lo_290);
            float pv_163 = __shfl_xor_sync(0xFFFFFFFF, cur_39, 1);
            float hi_307 = fmaxf(cur_39, pv_163);
            float lo_291 = fminf(cur_39, pv_163);
            cur_39 = ((up[3] != 0) ? hi_307 : lo_291);
            V_1[0] = cur_39;
            float y1_19 = __shfl_xor_sync(0xFFFFFFFF, V_1[3], 15);
            float lo1_19 = fminf(V_1[1], y1_19);
            r_2 = fmaxf(r_2, lo1_19);
            float hi1_19 = fmaxf(V_1[1], y1_19);
            float cur_40 = hi1_19;
            float pv_164 = __shfl_xor_sync(0xFFFFFFFF, cur_40, 8);
            float hi_308 = fmaxf(cur_40, pv_164);
            float lo_292 = fminf(cur_40, pv_164);
            cur_40 = ((up[0] != 0) ? hi_308 : lo_292);
            float pv_165 = __shfl_xor_sync(0xFFFFFFFF, cur_40, 4);
            float hi_309 = fmaxf(cur_40, pv_165);
            float lo_293 = fminf(cur_40, pv_165);
            cur_40 = ((up[1] != 0) ? hi_309 : lo_293);
            float pv_166 = __shfl_xor_sync(0xFFFFFFFF, cur_40, 2);
            float hi_310 = fmaxf(cur_40, pv_166);
            float lo_294 = fminf(cur_40, pv_166);
            cur_40 = ((up[2] != 0) ? hi_310 : lo_294);
            float pv_167 = __shfl_xor_sync(0xFFFFFFFF, cur_40, 1);
            float hi_311 = fmaxf(cur_40, pv_167);
            float lo_295 = fminf(cur_40, pv_167);
            cur_40 = ((up[3] != 0) ? hi_311 : lo_295);
            V_1[1] = cur_40;
            float y1_20 = __shfl_xor_sync(0xFFFFFFFF, V_1[1], 15);
            float lo1_20 = fminf(V_1[0], y1_20);
            r_2 = fmaxf(r_2, lo1_20);
            float hi1_20 = fmaxf(V_1[0], y1_20);
            V_1[0] = hi1_20;
            float K_1 = V_1[0];
            rr2x[0] = r_2;
            K2 = K_1;
        }
        if (tid_1 < 512) {
            unsigned int cgx = ccnt[cg];
            unsigned int k1x = __as_u32(K);
            int isq = 0;
            if (gflag != 0 && (k1x & 4294965248u) == qg) {
                isq = 1;
            }
            unsigned int mqx = __ballot_sync(0xFFFFFFFF, isq != 0);
            unsigned int gmx = mqx & 65535;
            if ((tid_1 & 16) != 0) {
                gmx = mqx >> 16 & 65535;
            }
            unsigned int lowx = (1 << (unsigned int)ln) - 1;
            int rx = (int)(unsigned int)__popc(gmx & lowx);
            if (gflag != 0 && cgx <= 16) {
                K2 = __uint_as_float(1073741824 | k1x & 2047);
                if (isq != 0) {
                    float xs[16];
                    xs[0] = 0.0f;
                    if (cgx > 0) {
                        xs[0] = __uint_as_float(cbuf[cg * 16]);
                    }
                    xs[1] = 0.0f;
                    if (cgx > 1) {
                        xs[1] = __uint_as_float(cbuf[cg * 16 + 1]);
                    }
                    xs[2] = 0.0f;
                    if (cgx > 2) {
                        xs[2] = __uint_as_float(cbuf[cg * 16 + 2]);
                    }
                    xs[3] = 0.0f;
                    if (cgx > 3) {
                        xs[3] = __uint_as_float(cbuf[cg * 16 + 3]);
                    }
                    xs[4] = 0.0f;
                    if (cgx > 4) {
                        xs[4] = __uint_as_float(cbuf[cg * 16 + 4]);
                    }
                    xs[5] = 0.0f;
                    if (cgx > 5) {
                        xs[5] = __uint_as_float(cbuf[cg * 16 + 5]);
                    }
                    xs[6] = 0.0f;
                    if (cgx > 6) {
                        xs[6] = __uint_as_float(cbuf[cg * 16 + 6]);
                    }
                    xs[7] = 0.0f;
                    if (cgx > 7) {
                        xs[7] = __uint_as_float(cbuf[cg * 16 + 7]);
                    }
                    xs[8] = 0.0f;
                    if (cgx > 8) {
                        xs[8] = __uint_as_float(cbuf[cg * 16 + 8]);
                    }
                    xs[9] = 0.0f;
                    if (cgx > 9) {
                        xs[9] = __uint_as_float(cbuf[cg * 16 + 9]);
                    }
                    xs[10] = 0.0f;
                    if (cgx > 10) {
                        xs[10] = __uint_as_float(cbuf[cg * 16 + 10]);
                    }
                    xs[11] = 0.0f;
                    if (cgx > 11) {
                        xs[11] = __uint_as_float(cbuf[cg * 16 + 11]);
                    }
                    xs[12] = 0.0f;
                    if (cgx > 12) {
                        xs[12] = __uint_as_float(cbuf[cg * 16 + 12]);
                    }
                    xs[13] = 0.0f;
                    if (cgx > 13) {
                        xs[13] = __uint_as_float(cbuf[cg * 16 + 13]);
                    }
                    xs[14] = 0.0f;
                    if (cgx > 14) {
                        xs[14] = __uint_as_float(cbuf[cg * 16 + 14]);
                    }
                    xs[15] = 0.0f;
                    if (cgx > 15) {
                        xs[15] = __uint_as_float(cbuf[cg * 16 + 15]);
                    }
                    float hi_1_180 = fmaxf(xs[0], xs[13]);
                    float lo_1_180 = fminf(xs[0], xs[13]);
                    xs[0] = hi_1_180;
                    xs[13] = lo_1_180;
                    float hi_1_181 = fmaxf(xs[1], xs[12]);
                    float lo_1_181 = fminf(xs[1], xs[12]);
                    xs[1] = hi_1_181;
                    xs[12] = lo_1_181;
                    float hi_1_182 = fmaxf(xs[2], xs[15]);
                    float lo_1_182 = fminf(xs[2], xs[15]);
                    xs[2] = hi_1_182;
                    xs[15] = lo_1_182;
                    float hi_1_183 = fmaxf(xs[3], xs[14]);
                    float lo_1_183 = fminf(xs[3], xs[14]);
                    xs[3] = hi_1_183;
                    xs[14] = lo_1_183;
                    float hi_1_184 = fmaxf(xs[4], xs[8]);
                    float lo_1_184 = fminf(xs[4], xs[8]);
                    xs[4] = hi_1_184;
                    xs[8] = lo_1_184;
                    float hi_1_185 = fmaxf(xs[5], xs[6]);
                    float lo_1_185 = fminf(xs[5], xs[6]);
                    xs[5] = hi_1_185;
                    xs[6] = lo_1_185;
                    float hi_1_186 = fmaxf(xs[7], xs[11]);
                    float lo_1_186 = fminf(xs[7], xs[11]);
                    xs[7] = hi_1_186;
                    xs[11] = lo_1_186;
                    float hi_1_187 = fmaxf(xs[9], xs[10]);
                    float lo_1_187 = fminf(xs[9], xs[10]);
                    xs[9] = hi_1_187;
                    xs[10] = lo_1_187;
                    float hi_1_188 = fmaxf(xs[0], xs[5]);
                    float lo_1_188 = fminf(xs[0], xs[5]);
                    xs[0] = hi_1_188;
                    xs[5] = lo_1_188;
                    float hi_1_189 = fmaxf(xs[1], xs[7]);
                    float lo_1_189 = fminf(xs[1], xs[7]);
                    xs[1] = hi_1_189;
                    xs[7] = lo_1_189;
                    float hi_1_190 = fmaxf(xs[2], xs[9]);
                    float lo_1_190 = fminf(xs[2], xs[9]);
                    xs[2] = hi_1_190;
                    xs[9] = lo_1_190;
                    float hi_1_191 = fmaxf(xs[3], xs[4]);
                    float lo_1_191 = fminf(xs[3], xs[4]);
                    xs[3] = hi_1_191;
                    xs[4] = lo_1_191;
                    float hi_1_192 = fmaxf(xs[6], xs[13]);
                    float lo_1_192 = fminf(xs[6], xs[13]);
                    xs[6] = hi_1_192;
                    xs[13] = lo_1_192;
                    float hi_1_193 = fmaxf(xs[8], xs[14]);
                    float lo_1_193 = fminf(xs[8], xs[14]);
                    xs[8] = hi_1_193;
                    xs[14] = lo_1_193;
                    float hi_1_194 = fmaxf(xs[10], xs[15]);
                    float lo_1_194 = fminf(xs[10], xs[15]);
                    xs[10] = hi_1_194;
                    xs[15] = lo_1_194;
                    float hi_1_195 = fmaxf(xs[11], xs[12]);
                    float lo_1_195 = fminf(xs[11], xs[12]);
                    xs[11] = hi_1_195;
                    xs[12] = lo_1_195;
                    float hi_1_196 = fmaxf(xs[0], xs[1]);
                    float lo_1_196 = fminf(xs[0], xs[1]);
                    xs[0] = hi_1_196;
                    xs[1] = lo_1_196;
                    float hi_1_197 = fmaxf(xs[2], xs[3]);
                    float lo_1_197 = fminf(xs[2], xs[3]);
                    xs[2] = hi_1_197;
                    xs[3] = lo_1_197;
                    float hi_1_198 = fmaxf(xs[4], xs[5]);
                    float lo_1_198 = fminf(xs[4], xs[5]);
                    xs[4] = hi_1_198;
                    xs[5] = lo_1_198;
                    float hi_1_199 = fmaxf(xs[6], xs[8]);
                    float lo_1_199 = fminf(xs[6], xs[8]);
                    xs[6] = hi_1_199;
                    xs[8] = lo_1_199;
                    float hi_1_200 = fmaxf(xs[7], xs[9]);
                    float lo_1_200 = fminf(xs[7], xs[9]);
                    xs[7] = hi_1_200;
                    xs[9] = lo_1_200;
                    float hi_1_201 = fmaxf(xs[10], xs[11]);
                    float lo_1_201 = fminf(xs[10], xs[11]);
                    xs[10] = hi_1_201;
                    xs[11] = lo_1_201;
                    float hi_1_202 = fmaxf(xs[12], xs[13]);
                    float lo_1_202 = fminf(xs[12], xs[13]);
                    xs[12] = hi_1_202;
                    xs[13] = lo_1_202;
                    float hi_1_203 = fmaxf(xs[14], xs[15]);
                    float lo_1_203 = fminf(xs[14], xs[15]);
                    xs[14] = hi_1_203;
                    xs[15] = lo_1_203;
                    float hi_1_204 = fmaxf(xs[0], xs[2]);
                    float lo_1_204 = fminf(xs[0], xs[2]);
                    xs[0] = hi_1_204;
                    xs[2] = lo_1_204;
                    float hi_1_205 = fmaxf(xs[1], xs[3]);
                    float lo_1_205 = fminf(xs[1], xs[3]);
                    xs[1] = hi_1_205;
                    xs[3] = lo_1_205;
                    float hi_1_206 = fmaxf(xs[4], xs[10]);
                    float lo_1_206 = fminf(xs[4], xs[10]);
                    xs[4] = hi_1_206;
                    xs[10] = lo_1_206;
                    float hi_1_207 = fmaxf(xs[5], xs[11]);
                    float lo_1_207 = fminf(xs[5], xs[11]);
                    xs[5] = hi_1_207;
                    xs[11] = lo_1_207;
                    float hi_1_208 = fmaxf(xs[6], xs[7]);
                    float lo_1_208 = fminf(xs[6], xs[7]);
                    xs[6] = hi_1_208;
                    xs[7] = lo_1_208;
                    float hi_1_209 = fmaxf(xs[8], xs[9]);
                    float lo_1_209 = fminf(xs[8], xs[9]);
                    xs[8] = hi_1_209;
                    xs[9] = lo_1_209;
                    float hi_1_210 = fmaxf(xs[12], xs[14]);
                    float lo_1_210 = fminf(xs[12], xs[14]);
                    xs[12] = hi_1_210;
                    xs[14] = lo_1_210;
                    float hi_1_211 = fmaxf(xs[13], xs[15]);
                    float lo_1_211 = fminf(xs[13], xs[15]);
                    xs[13] = hi_1_211;
                    xs[15] = lo_1_211;
                    float hi_1_212 = fmaxf(xs[1], xs[2]);
                    float lo_1_212 = fminf(xs[1], xs[2]);
                    xs[1] = hi_1_212;
                    xs[2] = lo_1_212;
                    float hi_1_213 = fmaxf(xs[3], xs[12]);
                    float lo_1_213 = fminf(xs[3], xs[12]);
                    xs[3] = hi_1_213;
                    xs[12] = lo_1_213;
                    float hi_1_214 = fmaxf(xs[4], xs[6]);
                    float lo_1_214 = fminf(xs[4], xs[6]);
                    xs[4] = hi_1_214;
                    xs[6] = lo_1_214;
                    float hi_1_215 = fmaxf(xs[5], xs[7]);
                    float lo_1_215 = fminf(xs[5], xs[7]);
                    xs[5] = hi_1_215;
                    xs[7] = lo_1_215;
                    float hi_1_216 = fmaxf(xs[8], xs[10]);
                    float lo_1_216 = fminf(xs[8], xs[10]);
                    xs[8] = hi_1_216;
                    xs[10] = lo_1_216;
                    float hi_1_217 = fmaxf(xs[9], xs[11]);
                    float lo_1_217 = fminf(xs[9], xs[11]);
                    xs[9] = hi_1_217;
                    xs[11] = lo_1_217;
                    float hi_1_218 = fmaxf(xs[13], xs[14]);
                    float lo_1_218 = fminf(xs[13], xs[14]);
                    xs[13] = hi_1_218;
                    xs[14] = lo_1_218;
                    float hi_1_219 = fmaxf(xs[1], xs[4]);
                    float lo_1_219 = fminf(xs[1], xs[4]);
                    xs[1] = hi_1_219;
                    xs[4] = lo_1_219;
                    float hi_1_220 = fmaxf(xs[2], xs[6]);
                    float lo_1_220 = fminf(xs[2], xs[6]);
                    xs[2] = hi_1_220;
                    xs[6] = lo_1_220;
                    float hi_1_221 = fmaxf(xs[5], xs[8]);
                    float lo_1_221 = fminf(xs[5], xs[8]);
                    xs[5] = hi_1_221;
                    xs[8] = lo_1_221;
                    float hi_1_222 = fmaxf(xs[7], xs[10]);
                    float lo_1_222 = fminf(xs[7], xs[10]);
                    xs[7] = hi_1_222;
                    xs[10] = lo_1_222;
                    float hi_1_223 = fmaxf(xs[9], xs[13]);
                    float lo_1_223 = fminf(xs[9], xs[13]);
                    xs[9] = hi_1_223;
                    xs[13] = lo_1_223;
                    float hi_1_224 = fmaxf(xs[11], xs[14]);
                    float lo_1_224 = fminf(xs[11], xs[14]);
                    xs[11] = hi_1_224;
                    xs[14] = lo_1_224;
                    float hi_1_225 = fmaxf(xs[2], xs[4]);
                    float lo_1_225 = fminf(xs[2], xs[4]);
                    xs[2] = hi_1_225;
                    xs[4] = lo_1_225;
                    float hi_1_226 = fmaxf(xs[3], xs[6]);
                    float lo_1_226 = fminf(xs[3], xs[6]);
                    xs[3] = hi_1_226;
                    xs[6] = lo_1_226;
                    float hi_1_227 = fmaxf(xs[9], xs[12]);
                    float lo_1_227 = fminf(xs[9], xs[12]);
                    xs[9] = hi_1_227;
                    xs[12] = lo_1_227;
                    float hi_1_228 = fmaxf(xs[11], xs[13]);
                    float lo_1_228 = fminf(xs[11], xs[13]);
                    xs[11] = hi_1_228;
                    xs[13] = lo_1_228;
                    float hi_1_229 = fmaxf(xs[3], xs[5]);
                    float lo_1_229 = fminf(xs[3], xs[5]);
                    xs[3] = hi_1_229;
                    xs[5] = lo_1_229;
                    float hi_1_230 = fmaxf(xs[6], xs[8]);
                    float lo_1_230 = fminf(xs[6], xs[8]);
                    xs[6] = hi_1_230;
                    xs[8] = lo_1_230;
                    float hi_1_231 = fmaxf(xs[7], xs[9]);
                    float lo_1_231 = fminf(xs[7], xs[9]);
                    xs[7] = hi_1_231;
                    xs[9] = lo_1_231;
                    float hi_1_232 = fmaxf(xs[10], xs[12]);
                    float lo_1_232 = fminf(xs[10], xs[12]);
                    xs[10] = hi_1_232;
                    xs[12] = lo_1_232;
                    float hi_1_233 = fmaxf(xs[3], xs[4]);
                    float lo_1_233 = fminf(xs[3], xs[4]);
                    xs[3] = hi_1_233;
                    xs[4] = lo_1_233;
                    float hi_1_234 = fmaxf(xs[5], xs[6]);
                    float lo_1_234 = fminf(xs[5], xs[6]);
                    xs[5] = hi_1_234;
                    xs[6] = lo_1_234;
                    float hi_1_235 = fmaxf(xs[7], xs[8]);
                    float lo_1_235 = fminf(xs[7], xs[8]);
                    xs[7] = hi_1_235;
                    xs[8] = lo_1_235;
                    float hi_1_236 = fmaxf(xs[9], xs[10]);
                    float lo_1_236 = fminf(xs[9], xs[10]);
                    xs[9] = hi_1_236;
                    xs[10] = lo_1_236;
                    float hi_1_237 = fmaxf(xs[11], xs[12]);
                    float lo_1_237 = fminf(xs[11], xs[12]);
                    xs[11] = hi_1_237;
                    xs[12] = lo_1_237;
                    float hi_1_238 = fmaxf(xs[6], xs[7]);
                    float lo_1_238 = fminf(xs[6], xs[7]);
                    xs[6] = hi_1_238;
                    xs[7] = lo_1_238;
                    float hi_1_239 = fmaxf(xs[8], xs[9]);
                    float lo_1_239 = fminf(xs[8], xs[9]);
                    xs[8] = hi_1_239;
                    xs[9] = lo_1_239;
                    K2 = xs[0];
                    if (rx == 1) {
                        K2 = xs[1];
                    }
                    if (rx == 2) {
                        K2 = xs[2];
                    }
                    if (rx == 3) {
                        K2 = xs[3];
                    }
                    if (rx == 4) {
                        K2 = xs[4];
                    }
                    if (rx == 5) {
                        K2 = xs[5];
                    }
                    if (rx == 6) {
                        K2 = xs[6];
                    }
                    if (rx == 7) {
                        K2 = xs[7];
                    }
                    if (rx == 8) {
                        K2 = xs[8];
                    }
                    if (rx == 9) {
                        K2 = xs[9];
                    }
                    if (rx == 10) {
                        K2 = xs[10];
                    }
                    if (rx == 11) {
                        K2 = xs[11];
                    }
                    if (rx == 12) {
                        K2 = xs[12];
                    }
                    if (rx == 13) {
                        K2 = xs[13];
                    }
                    if (rx == 14) {
                        K2 = xs[14];
                    }
                    if (rx == 15) {
                        K2 = xs[15];
                    }
                }
            }
        }
    }
    if (tid_1 < 512) {
        unsigned int kk = __as_u32(K);
        unsigned int idx = 2048;
        if (kk < 4282384384u) {
            idx = kk & 2047;
        }
        if (gflag != 0) {
            unsigned int k2 = __as_u32(K2);
            idx = 2048;
            if (k2 != 0) {
                idx = k2 & 2047;
            }
        }
        unsigned int rkey = idx << 4 | (unsigned int)ln;
        int rank = 0;
        unsigned int ox_14_0 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        if (ox_14_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_1 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        if (ox_14_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_2 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        if (ox_14_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_3 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        if (ox_14_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_4 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        if (ox_14_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_5 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        if (ox_14_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_6 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        if (ox_14_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_7 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        if (ox_14_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_8 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        if (ox_14_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_9 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        if (ox_14_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_10 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        if (ox_14_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_11 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        if (ox_14_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_12 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        if (ox_14_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_13 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        if (ox_14_13 < rkey) {
            rank = rank + 1;
        }
        unsigned int ox_14_14 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        if (ox_14_14 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 2047) {
            val = (int)idx;
        }
        if (commit != 0) {
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16ll;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
