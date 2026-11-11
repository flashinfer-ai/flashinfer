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
kernel_cake_hopper_msa_26311d226b72e91edb96(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    float* tree = reinterpret_cast<float*>(smem_raw + 0);
    const int tree_addr = smem + 0;
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 34816);
    const int qcol_addr = smem + 34816;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 35072);
    const int flagw_addr = smem + 35072;

    // === Task calls (declaration order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 64;
    int c = tid_1 - w * 64;
    int whi = w;
    int col = blockIdx.x * 64 + c;
    int head = blockIdx.y;
    if (tid_1 == 0) {
        flagw[0] = 0;
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
        int t0_2 = ((j + 1) * 8 + w) * 16;
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
        int t0_3 = (j * 8 + w) * 16;
        float sc = __uint_as_float(cb[0]);
        sc = fmaxf(sc, -1.7014118e+38f);
        sc = fminf(sc, 1.7014118e+38f);
        unsigned int kbits = __as_u32(sc);
        if (cb[0] == 4286578688u) {
            kbits = 4282384384u;
        }
        unsigned int key_1_0 = kbits & 4294966272u | (unsigned int)t0_3;
        kb[0] = __uint_as_float(key_1_0);
        float sc_0 = __uint_as_float(cb[1]);
        sc_0 = fmaxf(sc_0, -1.7014118e+38f);
        sc_0 = fminf(sc_0, 1.7014118e+38f);
        unsigned int kbits_0 = __as_u32(sc_0);
        if (cb[1] == 4286578688u) {
            kbits_0 = 4282384384u;
        }
        unsigned int key_1_1 = kbits_0 & 4294966272u | (unsigned int)(t0_3 + 1);
        kb[1] = __uint_as_float(key_1_1);
        float sc_1 = __uint_as_float(cb[2]);
        sc_1 = fmaxf(sc_1, -1.7014118e+38f);
        sc_1 = fminf(sc_1, 1.7014118e+38f);
        unsigned int kbits_1 = __as_u32(sc_1);
        if (cb[2] == 4286578688u) {
            kbits_1 = 4282384384u;
        }
        unsigned int key_1_2 = kbits_1 & 4294966272u | (unsigned int)(t0_3 + 2);
        kb[2] = __uint_as_float(key_1_2);
        float sc_2 = __uint_as_float(cb[3]);
        sc_2 = fmaxf(sc_2, -1.7014118e+38f);
        sc_2 = fminf(sc_2, 1.7014118e+38f);
        unsigned int kbits_2 = __as_u32(sc_2);
        if (cb[3] == 4286578688u) {
            kbits_2 = 4282384384u;
        }
        unsigned int key_1_3 = kbits_2 & 4294966272u | (unsigned int)(t0_3 + 3);
        kb[3] = __uint_as_float(key_1_3);
        float sc_3 = __uint_as_float(cb[4]);
        sc_3 = fmaxf(sc_3, -1.7014118e+38f);
        sc_3 = fminf(sc_3, 1.7014118e+38f);
        unsigned int kbits_3 = __as_u32(sc_3);
        if (cb[4] == 4286578688u) {
            kbits_3 = 4282384384u;
        }
        unsigned int key_1_4 = kbits_3 & 4294966272u | (unsigned int)(t0_3 + 4);
        kb[4] = __uint_as_float(key_1_4);
        float sc_4 = __uint_as_float(cb[5]);
        sc_4 = fmaxf(sc_4, -1.7014118e+38f);
        sc_4 = fminf(sc_4, 1.7014118e+38f);
        unsigned int kbits_4 = __as_u32(sc_4);
        if (cb[5] == 4286578688u) {
            kbits_4 = 4282384384u;
        }
        unsigned int key_1_5 = kbits_4 & 4294966272u | (unsigned int)(t0_3 + 5);
        kb[5] = __uint_as_float(key_1_5);
        float sc_5 = __uint_as_float(cb[6]);
        sc_5 = fmaxf(sc_5, -1.7014118e+38f);
        sc_5 = fminf(sc_5, 1.7014118e+38f);
        unsigned int kbits_5 = __as_u32(sc_5);
        if (cb[6] == 4286578688u) {
            kbits_5 = 4282384384u;
        }
        unsigned int key_1_6 = kbits_5 & 4294966272u | (unsigned int)(t0_3 + 6);
        kb[6] = __uint_as_float(key_1_6);
        float sc_6 = __uint_as_float(cb[7]);
        sc_6 = fmaxf(sc_6, -1.7014118e+38f);
        sc_6 = fminf(sc_6, 1.7014118e+38f);
        unsigned int kbits_6 = __as_u32(sc_6);
        if (cb[7] == 4286578688u) {
            kbits_6 = 4282384384u;
        }
        unsigned int key_1_7 = kbits_6 & 4294966272u | (unsigned int)(t0_3 + 7);
        kb[7] = __uint_as_float(key_1_7);
        float sc_7 = __uint_as_float(cb[8]);
        sc_7 = fmaxf(sc_7, -1.7014118e+38f);
        sc_7 = fminf(sc_7, 1.7014118e+38f);
        unsigned int kbits_7 = __as_u32(sc_7);
        if (cb[8] == 4286578688u) {
            kbits_7 = 4282384384u;
        }
        unsigned int key_1_8 = kbits_7 & 4294966272u | (unsigned int)(t0_3 + 8);
        kb[8] = __uint_as_float(key_1_8);
        float sc_8 = __uint_as_float(cb[9]);
        sc_8 = fmaxf(sc_8, -1.7014118e+38f);
        sc_8 = fminf(sc_8, 1.7014118e+38f);
        unsigned int kbits_8 = __as_u32(sc_8);
        if (cb[9] == 4286578688u) {
            kbits_8 = 4282384384u;
        }
        unsigned int key_1_9 = kbits_8 & 4294966272u | (unsigned int)(t0_3 + 9);
        kb[9] = __uint_as_float(key_1_9);
        float sc_9 = __uint_as_float(cb[10]);
        sc_9 = fmaxf(sc_9, -1.7014118e+38f);
        sc_9 = fminf(sc_9, 1.7014118e+38f);
        unsigned int kbits_9 = __as_u32(sc_9);
        if (cb[10] == 4286578688u) {
            kbits_9 = 4282384384u;
        }
        unsigned int key_1_10 = kbits_9 & 4294966272u | (unsigned int)(t0_3 + 10);
        kb[10] = __uint_as_float(key_1_10);
        float sc_10 = __uint_as_float(cb[11]);
        sc_10 = fmaxf(sc_10, -1.7014118e+38f);
        sc_10 = fminf(sc_10, 1.7014118e+38f);
        unsigned int kbits_10 = __as_u32(sc_10);
        if (cb[11] == 4286578688u) {
            kbits_10 = 4282384384u;
        }
        unsigned int key_1_11 = kbits_10 & 4294966272u | (unsigned int)(t0_3 + 11);
        kb[11] = __uint_as_float(key_1_11);
        float sc_11 = __uint_as_float(cb[12]);
        sc_11 = fmaxf(sc_11, -1.7014118e+38f);
        sc_11 = fminf(sc_11, 1.7014118e+38f);
        unsigned int kbits_11 = __as_u32(sc_11);
        if (cb[12] == 4286578688u) {
            kbits_11 = 4282384384u;
        }
        unsigned int key_1_12 = kbits_11 & 4294966272u | (unsigned int)(t0_3 + 12);
        kb[12] = __uint_as_float(key_1_12);
        float sc_12 = __uint_as_float(cb[13]);
        sc_12 = fmaxf(sc_12, -1.7014118e+38f);
        sc_12 = fminf(sc_12, 1.7014118e+38f);
        unsigned int kbits_12 = __as_u32(sc_12);
        if (cb[13] == 4286578688u) {
            kbits_12 = 4282384384u;
        }
        unsigned int key_1_13 = kbits_12 & 4294966272u | (unsigned int)(t0_3 + 13);
        kb[13] = __uint_as_float(key_1_13);
        float sc_13 = __uint_as_float(cb[14]);
        sc_13 = fmaxf(sc_13, -1.7014118e+38f);
        sc_13 = fminf(sc_13, 1.7014118e+38f);
        unsigned int kbits_13 = __as_u32(sc_13);
        if (cb[14] == 4286578688u) {
            kbits_13 = 4282384384u;
        }
        unsigned int key_1_14 = kbits_13 & 4294966272u | (unsigned int)(t0_3 + 14);
        kb[14] = __uint_as_float(key_1_14);
        float sc_14 = __uint_as_float(cb[15]);
        sc_14 = fmaxf(sc_14, -1.7014118e+38f);
        sc_14 = fminf(sc_14, 1.7014118e+38f);
        unsigned int kbits_14 = __as_u32(sc_14);
        if (cb[15] == 4286578688u) {
            kbits_14 = 4282384384u;
        }
        unsigned int key_1_15 = kbits_14 & 4294966272u | (unsigned int)(t0_3 + 15);
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
                kb[0] = __uint_as_float(2139094016 | (unsigned int)t0_3);
            }
            int f_1 = 0;
            if ((t0_3 + 1 < fb || t0_3 + 1 >= lim - fe) && lim > t0_3 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                kb[1] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 1));
            }
            int f_2 = 0;
            if ((t0_3 + 2 < fb || t0_3 + 2 >= lim - fe) && lim > t0_3 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                kb[2] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 2));
            }
            int f_3 = 0;
            if ((t0_3 + 3 < fb || t0_3 + 3 >= lim - fe) && lim > t0_3 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                kb[3] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 3));
            }
            int f_4 = 0;
            if ((t0_3 + 4 < fb || t0_3 + 4 >= lim - fe) && lim > t0_3 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                kb[4] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 4));
            }
            int f_5 = 0;
            if ((t0_3 + 5 < fb || t0_3 + 5 >= lim - fe) && lim > t0_3 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                kb[5] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 5));
            }
            int f_6 = 0;
            if ((t0_3 + 6 < fb || t0_3 + 6 >= lim - fe) && lim > t0_3 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                kb[6] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 6));
            }
            int f_7 = 0;
            if ((t0_3 + 7 < fb || t0_3 + 7 >= lim - fe) && lim > t0_3 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                kb[7] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 7));
            }
            int f_8 = 0;
            if ((t0_3 + 8 < fb || t0_3 + 8 >= lim - fe) && lim > t0_3 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                kb[8] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 8));
            }
            int f_9 = 0;
            if ((t0_3 + 9 < fb || t0_3 + 9 >= lim - fe) && lim > t0_3 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                kb[9] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 9));
            }
            int f_10 = 0;
            if ((t0_3 + 10 < fb || t0_3 + 10 >= lim - fe) && lim > t0_3 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                kb[10] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 10));
            }
            int f_11 = 0;
            if ((t0_3 + 11 < fb || t0_3 + 11 >= lim - fe) && lim > t0_3 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                kb[11] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 11));
            }
            int f_12 = 0;
            if ((t0_3 + 12 < fb || t0_3 + 12 >= lim - fe) && lim > t0_3 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                kb[12] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 12));
            }
            int f_13 = 0;
            if ((t0_3 + 13 < fb || t0_3 + 13 >= lim - fe) && lim > t0_3 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                kb[13] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 13));
            }
            int f_14 = 0;
            if ((t0_3 + 14 < fb || t0_3 + 14 >= lim - fe) && lim > t0_3 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                kb[14] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 14));
            }
            int f_15 = 0;
            if ((t0_3 + 15 < fb || t0_3 + 15 >= lim - fe) && lim > t0_3 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                kb[15] = __uint_as_float(2139094016 | (unsigned int)(t0_3 + 15));
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
    #pragma unroll 1
    for (int k = 0; k < 3; k++) {
        int bit = 1 << k;
        int low = whi & (bit << 1) - 1;
        if (low == bit) {
            int base = whi * 1088 + c;
            tree[base] = a[0];
            tree[base + 64] = a[1];
            tree[base + 128] = a[2];
            tree[base + 192] = a[3];
            tree[base + 256] = a[4];
            tree[base + 320] = a[5];
            tree[base + 384] = a[6];
            tree[base + 448] = a[7];
            tree[base + 512] = a[8];
            tree[base + 576] = a[9];
            tree[base + 640] = a[10];
            tree[base + 704] = a[11];
            tree[base + 768] = a[12];
            tree[base + 832] = a[13];
            tree[base + 896] = a[14];
            tree[base + 960] = a[15];
            tree[base + 1024] = rej;
        }
        asm volatile("barrier.sync 1, 512;" ::: "memory");
        if (low == 0) {
            float sb[16];
            int rbase = (whi + bit) * 1088 + c;
            sb[0] = tree[rbase];
            sb[1] = tree[rbase + 64];
            sb[2] = tree[rbase + 128];
            sb[3] = tree[rbase + 192];
            sb[4] = tree[rbase + 256];
            sb[5] = tree[rbase + 320];
            sb[6] = tree[rbase + 384];
            sb[7] = tree[rbase + 448];
            sb[8] = tree[rbase + 512];
            sb[9] = tree[rbase + 576];
            sb[10] = tree[rbase + 640];
            sb[11] = tree[rbase + 704];
            sb[12] = tree[rbase + 768];
            sb[13] = tree[rbase + 832];
            sb[14] = tree[rbase + 896];
            sb[15] = tree[rbase + 960];
            float srej = tree[rbase + 1024];
            rej = fmaxf(rej, srej);
            float r_1 = rej;
            float hi_1_60 = fmaxf(a[0], sb[15]);
            float lo_2_0 = fminf(a[0], sb[15]);
            a[0] = hi_1_60;
            r_1 = fmaxf(r_1, lo_2_0);
            float hi_1_61 = fmaxf(a[1], sb[14]);
            float lo_2_1 = fminf(a[1], sb[14]);
            a[1] = hi_1_61;
            r_1 = fmaxf(r_1, lo_2_1);
            float hi_1_62 = fmaxf(a[2], sb[13]);
            float lo_2_2 = fminf(a[2], sb[13]);
            a[2] = hi_1_62;
            r_1 = fmaxf(r_1, lo_2_2);
            float hi_1_63 = fmaxf(a[3], sb[12]);
            float lo_2_3 = fminf(a[3], sb[12]);
            a[3] = hi_1_63;
            r_1 = fmaxf(r_1, lo_2_3);
            float hi_1_64 = fmaxf(a[4], sb[11]);
            float lo_2_4 = fminf(a[4], sb[11]);
            a[4] = hi_1_64;
            r_1 = fmaxf(r_1, lo_2_4);
            float hi_1_65 = fmaxf(a[5], sb[10]);
            float lo_2_5 = fminf(a[5], sb[10]);
            a[5] = hi_1_65;
            r_1 = fmaxf(r_1, lo_2_5);
            float hi_1_66 = fmaxf(a[6], sb[9]);
            float lo_2_6 = fminf(a[6], sb[9]);
            a[6] = hi_1_66;
            r_1 = fmaxf(r_1, lo_2_6);
            float hi_1_67 = fmaxf(a[7], sb[8]);
            float lo_2_7 = fminf(a[7], sb[8]);
            a[7] = hi_1_67;
            r_1 = fmaxf(r_1, lo_2_7);
            float hi_1_68 = fmaxf(a[8], sb[7]);
            float lo_2_8 = fminf(a[8], sb[7]);
            a[8] = hi_1_68;
            r_1 = fmaxf(r_1, lo_2_8);
            float hi_1_69 = fmaxf(a[9], sb[6]);
            float lo_2_9 = fminf(a[9], sb[6]);
            a[9] = hi_1_69;
            r_1 = fmaxf(r_1, lo_2_9);
            float hi_1_70 = fmaxf(a[10], sb[5]);
            float lo_2_10 = fminf(a[10], sb[5]);
            a[10] = hi_1_70;
            r_1 = fmaxf(r_1, lo_2_10);
            float hi_1_71 = fmaxf(a[11], sb[4]);
            float lo_2_11 = fminf(a[11], sb[4]);
            a[11] = hi_1_71;
            r_1 = fmaxf(r_1, lo_2_11);
            float hi_1_72 = fmaxf(a[12], sb[3]);
            float lo_2_12 = fminf(a[12], sb[3]);
            a[12] = hi_1_72;
            r_1 = fmaxf(r_1, lo_2_12);
            float hi_1_73 = fmaxf(a[13], sb[2]);
            float lo_2_13 = fminf(a[13], sb[2]);
            a[13] = hi_1_73;
            r_1 = fmaxf(r_1, lo_2_13);
            float hi_1_74 = fmaxf(a[14], sb[1]);
            float lo_2_14 = fminf(a[14], sb[1]);
            a[14] = hi_1_74;
            r_1 = fmaxf(r_1, lo_2_14);
            float hi_1_75 = fmaxf(a[15], sb[0]);
            float lo_2_15 = fminf(a[15], sb[0]);
            a[15] = hi_1_75;
            r_1 = fmaxf(r_1, lo_2_15);
            float hi_48 = fmaxf(a[0], a[8]);
            float lo_48 = fminf(a[0], a[8]);
            a[0] = hi_48;
            a[8] = lo_48;
            float hi_49 = fmaxf(a[1], a[9]);
            float lo_49 = fminf(a[1], a[9]);
            a[1] = hi_49;
            a[9] = lo_49;
            float hi_50 = fmaxf(a[2], a[10]);
            float lo_50 = fminf(a[2], a[10]);
            a[2] = hi_50;
            a[10] = lo_50;
            float hi_51 = fmaxf(a[3], a[11]);
            float lo_51 = fminf(a[3], a[11]);
            a[3] = hi_51;
            a[11] = lo_51;
            float hi_52 = fmaxf(a[4], a[12]);
            float lo_52 = fminf(a[4], a[12]);
            a[4] = hi_52;
            a[12] = lo_52;
            float hi_53 = fmaxf(a[5], a[13]);
            float lo_53 = fminf(a[5], a[13]);
            a[5] = hi_53;
            a[13] = lo_53;
            float hi_54 = fmaxf(a[6], a[14]);
            float lo_54 = fminf(a[6], a[14]);
            a[6] = hi_54;
            a[14] = lo_54;
            float hi_55 = fmaxf(a[7], a[15]);
            float lo_55 = fminf(a[7], a[15]);
            a[7] = hi_55;
            a[15] = lo_55;
            float hi_56 = fmaxf(a[0], a[4]);
            float lo_56 = fminf(a[0], a[4]);
            a[0] = hi_56;
            a[4] = lo_56;
            float hi_57 = fmaxf(a[1], a[5]);
            float lo_57 = fminf(a[1], a[5]);
            a[1] = hi_57;
            a[5] = lo_57;
            float hi_58 = fmaxf(a[2], a[6]);
            float lo_58 = fminf(a[2], a[6]);
            a[2] = hi_58;
            a[6] = lo_58;
            float hi_59 = fmaxf(a[3], a[7]);
            float lo_59 = fminf(a[3], a[7]);
            a[3] = hi_59;
            a[7] = lo_59;
            float hi_60 = fmaxf(a[8], a[12]);
            float lo_60 = fminf(a[8], a[12]);
            a[8] = hi_60;
            a[12] = lo_60;
            float hi_61 = fmaxf(a[9], a[13]);
            float lo_61 = fminf(a[9], a[13]);
            a[9] = hi_61;
            a[13] = lo_61;
            float hi_62 = fmaxf(a[10], a[14]);
            float lo_62 = fminf(a[10], a[14]);
            a[10] = hi_62;
            a[14] = lo_62;
            float hi_63 = fmaxf(a[11], a[15]);
            float lo_63 = fminf(a[11], a[15]);
            a[11] = hi_63;
            a[15] = lo_63;
            float hi_64 = fmaxf(a[0], a[2]);
            float lo_64 = fminf(a[0], a[2]);
            a[0] = hi_64;
            a[2] = lo_64;
            float hi_65 = fmaxf(a[1], a[3]);
            float lo_65 = fminf(a[1], a[3]);
            a[1] = hi_65;
            a[3] = lo_65;
            float hi_66 = fmaxf(a[4], a[6]);
            float lo_66 = fminf(a[4], a[6]);
            a[4] = hi_66;
            a[6] = lo_66;
            float hi_67 = fmaxf(a[5], a[7]);
            float lo_67 = fminf(a[5], a[7]);
            a[5] = hi_67;
            a[7] = lo_67;
            float hi_68 = fmaxf(a[8], a[10]);
            float lo_68 = fminf(a[8], a[10]);
            a[8] = hi_68;
            a[10] = lo_68;
            float hi_69 = fmaxf(a[9], a[11]);
            float lo_69 = fminf(a[9], a[11]);
            a[9] = hi_69;
            a[11] = lo_69;
            float hi_70 = fmaxf(a[12], a[14]);
            float lo_70 = fminf(a[12], a[14]);
            a[12] = hi_70;
            a[14] = lo_70;
            float hi_71 = fmaxf(a[13], a[15]);
            float lo_71 = fminf(a[13], a[15]);
            a[13] = hi_71;
            a[15] = lo_71;
            float hi_72 = fmaxf(a[0], a[1]);
            float lo_72 = fminf(a[0], a[1]);
            a[0] = hi_72;
            a[1] = lo_72;
            float hi_73 = fmaxf(a[2], a[3]);
            float lo_73 = fminf(a[2], a[3]);
            a[2] = hi_73;
            a[3] = lo_73;
            float hi_74 = fmaxf(a[4], a[5]);
            float lo_74 = fminf(a[4], a[5]);
            a[4] = hi_74;
            a[5] = lo_74;
            float hi_75 = fmaxf(a[6], a[7]);
            float lo_75 = fminf(a[6], a[7]);
            a[6] = hi_75;
            a[7] = lo_75;
            float hi_76 = fmaxf(a[8], a[9]);
            float lo_76 = fminf(a[8], a[9]);
            a[8] = hi_76;
            a[9] = lo_76;
            float hi_77 = fmaxf(a[10], a[11]);
            float lo_77 = fminf(a[10], a[11]);
            a[10] = hi_77;
            a[11] = lo_77;
            float hi_78 = fmaxf(a[12], a[13]);
            float lo_78 = fminf(a[12], a[13]);
            a[12] = hi_78;
            a[13] = lo_78;
            float hi_79 = fmaxf(a[14], a[15]);
            float lo_79 = fminf(a[14], a[15]);
            a[14] = hi_79;
            a[15] = lo_79;
            rej = r_1;
        }
    }
    if (w == 0) {
        unsigned int u16 = __as_u32(a[15]);
        unsigned int c16 = u16 & 4294966272u;
        unsigned int cr = __as_u32(rej) & 4294966272u;
        unsigned int q = 4294967295u;
        if (c16 == cr && u16 < 4278190080u && c16 != 2139094016 && col < total_q) {
            q = c16;
            flagw[0] = 1;
        }
        qcol[c] = q;
    }
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    unsigned int need2 = flagw[0];
    unsigned int qc = qcol[c];
    int flagged = 0;
    if (qc != 4294967295u) {
        flagged = 1;
    }
    float a2[16];
    a2[0] = 0.0f;
    a2[1] = 0.0f;
    a2[2] = 0.0f;
    a2[3] = 0.0f;
    a2[4] = 0.0f;
    a2[5] = 0.0f;
    a2[6] = 0.0f;
    a2[7] = 0.0f;
    a2[8] = 0.0f;
    a2[9] = 0.0f;
    a2[10] = 0.0f;
    a2[11] = 0.0f;
    a2[12] = 0.0f;
    a2[13] = 0.0f;
    a2[14] = 0.0f;
    a2[15] = 0.0f;
    if (need2 != 0) {
        int lim2 = 0;
        if (flagged != 0) {
            lim2 = lim;
        }
        int _warp_vote_0 = __any_sync(0xFFFFFFFF, flagged != 0);
        if (_warp_vote_0 != 0) {
            unsigned int cb2[16];
            unsigned int nb2[16];
            float kb2[16];
            int t0_4 = w * 16;
            long long p_1 = cbase + (long long)t0_4 * nq64;
            cb2[0] = 4286578688u;
            if (lim2 > t0_4) {
                cb2[0] = S[p_1];
            }
            cb2[1] = 4286578688u;
            if (lim2 > t0_4 + 1) {
                cb2[1] = S[p_1 + nq64];
            }
            cb2[2] = 4286578688u;
            if (lim2 > t0_4 + 2) {
                cb2[2] = S[p_1 + 2ll * nq64];
            }
            cb2[3] = 4286578688u;
            if (lim2 > t0_4 + 3) {
                cb2[3] = S[p_1 + 3ll * nq64];
            }
            cb2[4] = 4286578688u;
            if (lim2 > t0_4 + 4) {
                cb2[4] = S[p_1 + 4ll * nq64];
            }
            cb2[5] = 4286578688u;
            if (lim2 > t0_4 + 5) {
                cb2[5] = S[p_1 + 5ll * nq64];
            }
            cb2[6] = 4286578688u;
            if (lim2 > t0_4 + 6) {
                cb2[6] = S[p_1 + 6ll * nq64];
            }
            cb2[7] = 4286578688u;
            if (lim2 > t0_4 + 7) {
                cb2[7] = S[p_1 + 7ll * nq64];
            }
            cb2[8] = 4286578688u;
            if (lim2 > t0_4 + 8) {
                cb2[8] = S[p_1 + 8ll * nq64];
            }
            cb2[9] = 4286578688u;
            if (lim2 > t0_4 + 9) {
                cb2[9] = S[p_1 + 9ll * nq64];
            }
            cb2[10] = 4286578688u;
            if (lim2 > t0_4 + 10) {
                cb2[10] = S[p_1 + 10ll * nq64];
            }
            cb2[11] = 4286578688u;
            if (lim2 > t0_4 + 11) {
                cb2[11] = S[p_1 + 11ll * nq64];
            }
            cb2[12] = 4286578688u;
            if (lim2 > t0_4 + 12) {
                cb2[12] = S[p_1 + 12ll * nq64];
            }
            cb2[13] = 4286578688u;
            if (lim2 > t0_4 + 13) {
                cb2[13] = S[p_1 + 13ll * nq64];
            }
            cb2[14] = 4286578688u;
            if (lim2 > t0_4 + 14) {
                cb2[14] = S[p_1 + 14ll * nq64];
            }
            cb2[15] = 4286578688u;
            if (lim2 > t0_4 + 15) {
                cb2[15] = S[p_1 + 15ll * nq64];
            }
            asm volatile("" ::: "memory");
            #pragma unroll 1
            for (int j_1 = 0; j_1 < num_chunks; j_1++) {
                int t0_5 = ((j_1 + 1) * 8 + w) * 16;
                long long p_2 = cbase + (long long)t0_5 * nq64;
                nb2[0] = 4286578688u;
                if (lim2 > t0_5) {
                    nb2[0] = S[p_2];
                }
                nb2[1] = 4286578688u;
                if (lim2 > t0_5 + 1) {
                    nb2[1] = S[p_2 + nq64];
                }
                nb2[2] = 4286578688u;
                if (lim2 > t0_5 + 2) {
                    nb2[2] = S[p_2 + 2ll * nq64];
                }
                nb2[3] = 4286578688u;
                if (lim2 > t0_5 + 3) {
                    nb2[3] = S[p_2 + 3ll * nq64];
                }
                nb2[4] = 4286578688u;
                if (lim2 > t0_5 + 4) {
                    nb2[4] = S[p_2 + 4ll * nq64];
                }
                nb2[5] = 4286578688u;
                if (lim2 > t0_5 + 5) {
                    nb2[5] = S[p_2 + 5ll * nq64];
                }
                nb2[6] = 4286578688u;
                if (lim2 > t0_5 + 6) {
                    nb2[6] = S[p_2 + 6ll * nq64];
                }
                nb2[7] = 4286578688u;
                if (lim2 > t0_5 + 7) {
                    nb2[7] = S[p_2 + 7ll * nq64];
                }
                nb2[8] = 4286578688u;
                if (lim2 > t0_5 + 8) {
                    nb2[8] = S[p_2 + 8ll * nq64];
                }
                nb2[9] = 4286578688u;
                if (lim2 > t0_5 + 9) {
                    nb2[9] = S[p_2 + 9ll * nq64];
                }
                nb2[10] = 4286578688u;
                if (lim2 > t0_5 + 10) {
                    nb2[10] = S[p_2 + 10ll * nq64];
                }
                nb2[11] = 4286578688u;
                if (lim2 > t0_5 + 11) {
                    nb2[11] = S[p_2 + 11ll * nq64];
                }
                nb2[12] = 4286578688u;
                if (lim2 > t0_5 + 12) {
                    nb2[12] = S[p_2 + 12ll * nq64];
                }
                nb2[13] = 4286578688u;
                if (lim2 > t0_5 + 13) {
                    nb2[13] = S[p_2 + 13ll * nq64];
                }
                nb2[14] = 4286578688u;
                if (lim2 > t0_5 + 14) {
                    nb2[14] = S[p_2 + 14ll * nq64];
                }
                nb2[15] = 4286578688u;
                if (lim2 > t0_5 + 15) {
                    nb2[15] = S[p_2 + 15ll * nq64];
                }
                asm volatile("" ::: "memory");
                int t0_6 = (j_1 * 8 + w) * 16;
                int thr = 2147483647;
                if (qc != 4294967295u) {
                    unsigned int img = qc ^ (unsigned int)((int)qc >> 31) & 2147483647;
                    thr = (int)img;
                }
                float sc_15 = __uint_as_float(cb2[0]);
                sc_15 = fmaxf(sc_15, -1.7014118e+38f);
                sc_15 = fminf(sc_15, 1.7014118e+38f);
                unsigned int u_1_0 = __as_u32(sc_15);
                unsigned int cls_1_0 = u_1_0 & 4294966272u;
                int f_16 = 0;
                if ((t0_6 < fb || t0_6 >= lim - fe) && lim > t0_6) {
                    f_16 = 1;
                }
                if (f_16 != 0) {
                    cls_1_0 = 2139094016;
                }
                unsigned int key = 0;
                unsigned int img_1 = cls_1_0 ^ (unsigned int)((int)cls_1_0 >> 31) & 2147483647;
                if (thr < (int)img_1) {
                    key = 1073741824 | (unsigned int)t0_6;
                }
                if (cls_1_0 == qc) {
                    unsigned int lowb = (u_1_0 ^ (unsigned int)((int)u_1_0 >> 31) & 1023) & 1023;
                    key = 536870912 | lowb << 10 | (unsigned int)t0_6;
                }
                kb2[0] = __uint_as_float(key);
                float sc_15_1 = __uint_as_float(cb2[1]);
                sc_15_1 = fmaxf(sc_15_1, -1.7014118e+38f);
                sc_15_1 = fminf(sc_15_1, 1.7014118e+38f);
                unsigned int u_1_1 = __as_u32(sc_15_1);
                unsigned int cls_1_1 = u_1_1 & 4294966272u;
                int f_16_1 = 0;
                if ((t0_6 + 1 < fb || t0_6 + 1 >= lim - fe) && lim > t0_6 + 1) {
                    f_16_1 = 1;
                }
                if (f_16_1 != 0) {
                    cls_1_1 = 2139094016;
                }
                unsigned int key_0 = 0;
                unsigned int img_0 = cls_1_1 ^ (unsigned int)((int)cls_1_1 >> 31) & 2147483647;
                if (thr < (int)img_0) {
                    key_0 = 1073741824 | (unsigned int)(t0_6 + 1);
                }
                if (cls_1_1 == qc) {
                    unsigned int lowb_1 = (u_1_1 ^ (unsigned int)((int)u_1_1 >> 31) & 1023) & 1023;
                    key_0 = 536870912 | lowb_1 << 10 | (unsigned int)(t0_6 + 1);
                }
                kb2[1] = __uint_as_float(key_0);
                float sc_16 = __uint_as_float(cb2[2]);
                sc_16 = fmaxf(sc_16, -1.7014118e+38f);
                sc_16 = fminf(sc_16, 1.7014118e+38f);
                unsigned int u_1_2 = __as_u32(sc_16);
                unsigned int cls_1_2 = u_1_2 & 4294966272u;
                int f_17 = 0;
                if ((t0_6 + 2 < fb || t0_6 + 2 >= lim - fe) && lim > t0_6 + 2) {
                    f_17 = 1;
                }
                if (f_17 != 0) {
                    cls_1_2 = 2139094016;
                }
                unsigned int key_1 = 0;
                unsigned int img_1_1 = cls_1_2 ^ (unsigned int)((int)cls_1_2 >> 31) & 2147483647;
                if (thr < (int)img_1_1) {
                    key_1 = 1073741824 | (unsigned int)(t0_6 + 2);
                }
                if (cls_1_2 == qc) {
                    unsigned int lowb_2 = (u_1_2 ^ (unsigned int)((int)u_1_2 >> 31) & 1023) & 1023;
                    key_1 = 536870912 | lowb_2 << 10 | (unsigned int)(t0_6 + 2);
                }
                kb2[2] = __uint_as_float(key_1);
                float sc_17 = __uint_as_float(cb2[3]);
                sc_17 = fmaxf(sc_17, -1.7014118e+38f);
                sc_17 = fminf(sc_17, 1.7014118e+38f);
                unsigned int u_1_3 = __as_u32(sc_17);
                unsigned int cls_1_3 = u_1_3 & 4294966272u;
                int f_18 = 0;
                if ((t0_6 + 3 < fb || t0_6 + 3 >= lim - fe) && lim > t0_6 + 3) {
                    f_18 = 1;
                }
                if (f_18 != 0) {
                    cls_1_3 = 2139094016;
                }
                unsigned int key_2 = 0;
                unsigned int img_2 = cls_1_3 ^ (unsigned int)((int)cls_1_3 >> 31) & 2147483647;
                if (thr < (int)img_2) {
                    key_2 = 1073741824 | (unsigned int)(t0_6 + 3);
                }
                if (cls_1_3 == qc) {
                    unsigned int lowb_3 = (u_1_3 ^ (unsigned int)((int)u_1_3 >> 31) & 1023) & 1023;
                    key_2 = 536870912 | lowb_3 << 10 | (unsigned int)(t0_6 + 3);
                }
                kb2[3] = __uint_as_float(key_2);
                float sc_18 = __uint_as_float(cb2[4]);
                sc_18 = fmaxf(sc_18, -1.7014118e+38f);
                sc_18 = fminf(sc_18, 1.7014118e+38f);
                unsigned int u_1_4 = __as_u32(sc_18);
                unsigned int cls_1_4 = u_1_4 & 4294966272u;
                int f_19 = 0;
                if ((t0_6 + 4 < fb || t0_6 + 4 >= lim - fe) && lim > t0_6 + 4) {
                    f_19 = 1;
                }
                if (f_19 != 0) {
                    cls_1_4 = 2139094016;
                }
                unsigned int key_3 = 0;
                unsigned int img_3 = cls_1_4 ^ (unsigned int)((int)cls_1_4 >> 31) & 2147483647;
                if (thr < (int)img_3) {
                    key_3 = 1073741824 | (unsigned int)(t0_6 + 4);
                }
                if (cls_1_4 == qc) {
                    unsigned int lowb_4 = (u_1_4 ^ (unsigned int)((int)u_1_4 >> 31) & 1023) & 1023;
                    key_3 = 536870912 | lowb_4 << 10 | (unsigned int)(t0_6 + 4);
                }
                kb2[4] = __uint_as_float(key_3);
                float sc_19 = __uint_as_float(cb2[5]);
                sc_19 = fmaxf(sc_19, -1.7014118e+38f);
                sc_19 = fminf(sc_19, 1.7014118e+38f);
                unsigned int u_1_5 = __as_u32(sc_19);
                unsigned int cls_1_5 = u_1_5 & 4294966272u;
                int f_20 = 0;
                if ((t0_6 + 5 < fb || t0_6 + 5 >= lim - fe) && lim > t0_6 + 5) {
                    f_20 = 1;
                }
                if (f_20 != 0) {
                    cls_1_5 = 2139094016;
                }
                unsigned int key_4 = 0;
                unsigned int img_4 = cls_1_5 ^ (unsigned int)((int)cls_1_5 >> 31) & 2147483647;
                if (thr < (int)img_4) {
                    key_4 = 1073741824 | (unsigned int)(t0_6 + 5);
                }
                if (cls_1_5 == qc) {
                    unsigned int lowb_5 = (u_1_5 ^ (unsigned int)((int)u_1_5 >> 31) & 1023) & 1023;
                    key_4 = 536870912 | lowb_5 << 10 | (unsigned int)(t0_6 + 5);
                }
                kb2[5] = __uint_as_float(key_4);
                float sc_20 = __uint_as_float(cb2[6]);
                sc_20 = fmaxf(sc_20, -1.7014118e+38f);
                sc_20 = fminf(sc_20, 1.7014118e+38f);
                unsigned int u_1_6 = __as_u32(sc_20);
                unsigned int cls_1_6 = u_1_6 & 4294966272u;
                int f_21 = 0;
                if ((t0_6 + 6 < fb || t0_6 + 6 >= lim - fe) && lim > t0_6 + 6) {
                    f_21 = 1;
                }
                if (f_21 != 0) {
                    cls_1_6 = 2139094016;
                }
                unsigned int key_5 = 0;
                unsigned int img_5 = cls_1_6 ^ (unsigned int)((int)cls_1_6 >> 31) & 2147483647;
                if (thr < (int)img_5) {
                    key_5 = 1073741824 | (unsigned int)(t0_6 + 6);
                }
                if (cls_1_6 == qc) {
                    unsigned int lowb_6 = (u_1_6 ^ (unsigned int)((int)u_1_6 >> 31) & 1023) & 1023;
                    key_5 = 536870912 | lowb_6 << 10 | (unsigned int)(t0_6 + 6);
                }
                kb2[6] = __uint_as_float(key_5);
                float sc_21 = __uint_as_float(cb2[7]);
                sc_21 = fmaxf(sc_21, -1.7014118e+38f);
                sc_21 = fminf(sc_21, 1.7014118e+38f);
                unsigned int u_1_7 = __as_u32(sc_21);
                unsigned int cls_1_7 = u_1_7 & 4294966272u;
                int f_22 = 0;
                if ((t0_6 + 7 < fb || t0_6 + 7 >= lim - fe) && lim > t0_6 + 7) {
                    f_22 = 1;
                }
                if (f_22 != 0) {
                    cls_1_7 = 2139094016;
                }
                unsigned int key_6 = 0;
                unsigned int img_6 = cls_1_7 ^ (unsigned int)((int)cls_1_7 >> 31) & 2147483647;
                if (thr < (int)img_6) {
                    key_6 = 1073741824 | (unsigned int)(t0_6 + 7);
                }
                if (cls_1_7 == qc) {
                    unsigned int lowb_7 = (u_1_7 ^ (unsigned int)((int)u_1_7 >> 31) & 1023) & 1023;
                    key_6 = 536870912 | lowb_7 << 10 | (unsigned int)(t0_6 + 7);
                }
                kb2[7] = __uint_as_float(key_6);
                float sc_22 = __uint_as_float(cb2[8]);
                sc_22 = fmaxf(sc_22, -1.7014118e+38f);
                sc_22 = fminf(sc_22, 1.7014118e+38f);
                unsigned int u_1_8 = __as_u32(sc_22);
                unsigned int cls_1_8 = u_1_8 & 4294966272u;
                int f_23 = 0;
                if ((t0_6 + 8 < fb || t0_6 + 8 >= lim - fe) && lim > t0_6 + 8) {
                    f_23 = 1;
                }
                if (f_23 != 0) {
                    cls_1_8 = 2139094016;
                }
                unsigned int key_7 = 0;
                unsigned int img_7 = cls_1_8 ^ (unsigned int)((int)cls_1_8 >> 31) & 2147483647;
                if (thr < (int)img_7) {
                    key_7 = 1073741824 | (unsigned int)(t0_6 + 8);
                }
                if (cls_1_8 == qc) {
                    unsigned int lowb_8 = (u_1_8 ^ (unsigned int)((int)u_1_8 >> 31) & 1023) & 1023;
                    key_7 = 536870912 | lowb_8 << 10 | (unsigned int)(t0_6 + 8);
                }
                kb2[8] = __uint_as_float(key_7);
                float sc_23 = __uint_as_float(cb2[9]);
                sc_23 = fmaxf(sc_23, -1.7014118e+38f);
                sc_23 = fminf(sc_23, 1.7014118e+38f);
                unsigned int u_1_9 = __as_u32(sc_23);
                unsigned int cls_1_9 = u_1_9 & 4294966272u;
                int f_24 = 0;
                if ((t0_6 + 9 < fb || t0_6 + 9 >= lim - fe) && lim > t0_6 + 9) {
                    f_24 = 1;
                }
                if (f_24 != 0) {
                    cls_1_9 = 2139094016;
                }
                unsigned int key_8 = 0;
                unsigned int img_8 = cls_1_9 ^ (unsigned int)((int)cls_1_9 >> 31) & 2147483647;
                if (thr < (int)img_8) {
                    key_8 = 1073741824 | (unsigned int)(t0_6 + 9);
                }
                if (cls_1_9 == qc) {
                    unsigned int lowb_9 = (u_1_9 ^ (unsigned int)((int)u_1_9 >> 31) & 1023) & 1023;
                    key_8 = 536870912 | lowb_9 << 10 | (unsigned int)(t0_6 + 9);
                }
                kb2[9] = __uint_as_float(key_8);
                float sc_24 = __uint_as_float(cb2[10]);
                sc_24 = fmaxf(sc_24, -1.7014118e+38f);
                sc_24 = fminf(sc_24, 1.7014118e+38f);
                unsigned int u_1_10 = __as_u32(sc_24);
                unsigned int cls_1_10 = u_1_10 & 4294966272u;
                int f_25 = 0;
                if ((t0_6 + 10 < fb || t0_6 + 10 >= lim - fe) && lim > t0_6 + 10) {
                    f_25 = 1;
                }
                if (f_25 != 0) {
                    cls_1_10 = 2139094016;
                }
                unsigned int key_9 = 0;
                unsigned int img_9 = cls_1_10 ^ (unsigned int)((int)cls_1_10 >> 31) & 2147483647;
                if (thr < (int)img_9) {
                    key_9 = 1073741824 | (unsigned int)(t0_6 + 10);
                }
                if (cls_1_10 == qc) {
                    unsigned int lowb_10 = (u_1_10 ^ (unsigned int)((int)u_1_10 >> 31) & 1023) & 1023;
                    key_9 = 536870912 | lowb_10 << 10 | (unsigned int)(t0_6 + 10);
                }
                kb2[10] = __uint_as_float(key_9);
                float sc_25 = __uint_as_float(cb2[11]);
                sc_25 = fmaxf(sc_25, -1.7014118e+38f);
                sc_25 = fminf(sc_25, 1.7014118e+38f);
                unsigned int u_1_11 = __as_u32(sc_25);
                unsigned int cls_1_11 = u_1_11 & 4294966272u;
                int f_26 = 0;
                if ((t0_6 + 11 < fb || t0_6 + 11 >= lim - fe) && lim > t0_6 + 11) {
                    f_26 = 1;
                }
                if (f_26 != 0) {
                    cls_1_11 = 2139094016;
                }
                unsigned int key_10 = 0;
                unsigned int img_10 = cls_1_11 ^ (unsigned int)((int)cls_1_11 >> 31) & 2147483647;
                if (thr < (int)img_10) {
                    key_10 = 1073741824 | (unsigned int)(t0_6 + 11);
                }
                if (cls_1_11 == qc) {
                    unsigned int lowb_11 = (u_1_11 ^ (unsigned int)((int)u_1_11 >> 31) & 1023) & 1023;
                    key_10 = 536870912 | lowb_11 << 10 | (unsigned int)(t0_6 + 11);
                }
                kb2[11] = __uint_as_float(key_10);
                float sc_26 = __uint_as_float(cb2[12]);
                sc_26 = fmaxf(sc_26, -1.7014118e+38f);
                sc_26 = fminf(sc_26, 1.7014118e+38f);
                unsigned int u_1_12 = __as_u32(sc_26);
                unsigned int cls_1_12 = u_1_12 & 4294966272u;
                int f_27 = 0;
                if ((t0_6 + 12 < fb || t0_6 + 12 >= lim - fe) && lim > t0_6 + 12) {
                    f_27 = 1;
                }
                if (f_27 != 0) {
                    cls_1_12 = 2139094016;
                }
                unsigned int key_11 = 0;
                unsigned int img_11 = cls_1_12 ^ (unsigned int)((int)cls_1_12 >> 31) & 2147483647;
                if (thr < (int)img_11) {
                    key_11 = 1073741824 | (unsigned int)(t0_6 + 12);
                }
                if (cls_1_12 == qc) {
                    unsigned int lowb_12 = (u_1_12 ^ (unsigned int)((int)u_1_12 >> 31) & 1023) & 1023;
                    key_11 = 536870912 | lowb_12 << 10 | (unsigned int)(t0_6 + 12);
                }
                kb2[12] = __uint_as_float(key_11);
                float sc_27 = __uint_as_float(cb2[13]);
                sc_27 = fmaxf(sc_27, -1.7014118e+38f);
                sc_27 = fminf(sc_27, 1.7014118e+38f);
                unsigned int u_1_13 = __as_u32(sc_27);
                unsigned int cls_1_13 = u_1_13 & 4294966272u;
                int f_28 = 0;
                if ((t0_6 + 13 < fb || t0_6 + 13 >= lim - fe) && lim > t0_6 + 13) {
                    f_28 = 1;
                }
                if (f_28 != 0) {
                    cls_1_13 = 2139094016;
                }
                unsigned int key_12 = 0;
                unsigned int img_12 = cls_1_13 ^ (unsigned int)((int)cls_1_13 >> 31) & 2147483647;
                if (thr < (int)img_12) {
                    key_12 = 1073741824 | (unsigned int)(t0_6 + 13);
                }
                if (cls_1_13 == qc) {
                    unsigned int lowb_13 = (u_1_13 ^ (unsigned int)((int)u_1_13 >> 31) & 1023) & 1023;
                    key_12 = 536870912 | lowb_13 << 10 | (unsigned int)(t0_6 + 13);
                }
                kb2[13] = __uint_as_float(key_12);
                float sc_28 = __uint_as_float(cb2[14]);
                sc_28 = fmaxf(sc_28, -1.7014118e+38f);
                sc_28 = fminf(sc_28, 1.7014118e+38f);
                unsigned int u_1_14 = __as_u32(sc_28);
                unsigned int cls_1_14 = u_1_14 & 4294966272u;
                int f_29 = 0;
                if ((t0_6 + 14 < fb || t0_6 + 14 >= lim - fe) && lim > t0_6 + 14) {
                    f_29 = 1;
                }
                if (f_29 != 0) {
                    cls_1_14 = 2139094016;
                }
                unsigned int key_13 = 0;
                unsigned int img_13 = cls_1_14 ^ (unsigned int)((int)cls_1_14 >> 31) & 2147483647;
                if (thr < (int)img_13) {
                    key_13 = 1073741824 | (unsigned int)(t0_6 + 14);
                }
                if (cls_1_14 == qc) {
                    unsigned int lowb_14 = (u_1_14 ^ (unsigned int)((int)u_1_14 >> 31) & 1023) & 1023;
                    key_13 = 536870912 | lowb_14 << 10 | (unsigned int)(t0_6 + 14);
                }
                kb2[14] = __uint_as_float(key_13);
                float sc_29 = __uint_as_float(cb2[15]);
                sc_29 = fmaxf(sc_29, -1.7014118e+38f);
                sc_29 = fminf(sc_29, 1.7014118e+38f);
                unsigned int u_1_15 = __as_u32(sc_29);
                unsigned int cls_1_15 = u_1_15 & 4294966272u;
                int f_30 = 0;
                if ((t0_6 + 15 < fb || t0_6 + 15 >= lim - fe) && lim > t0_6 + 15) {
                    f_30 = 1;
                }
                if (f_30 != 0) {
                    cls_1_15 = 2139094016;
                }
                unsigned int key_14 = 0;
                unsigned int img_14 = cls_1_15 ^ (unsigned int)((int)cls_1_15 >> 31) & 2147483647;
                if (thr < (int)img_14) {
                    key_14 = 1073741824 | (unsigned int)(t0_6 + 15);
                }
                if (cls_1_15 == qc) {
                    unsigned int lowb_15 = (u_1_15 ^ (unsigned int)((int)u_1_15 >> 31) & 1023) & 1023;
                    key_14 = 536870912 | lowb_15 << 10 | (unsigned int)(t0_6 + 15);
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
                int _warp_vote_1 = __any_sync(0xFFFFFFFF, nzc != 0);
                if (_warp_vote_1 != 0) {
                    float hi_1_76 = fmaxf(kb2[0], kb2[13]);
                    float lo_1_60 = fminf(kb2[0], kb2[13]);
                    kb2[0] = hi_1_76;
                    kb2[13] = lo_1_60;
                    float hi_1_77 = fmaxf(kb2[1], kb2[12]);
                    float lo_1_61 = fminf(kb2[1], kb2[12]);
                    kb2[1] = hi_1_77;
                    kb2[12] = lo_1_61;
                    float hi_1_78 = fmaxf(kb2[2], kb2[15]);
                    float lo_1_62 = fminf(kb2[2], kb2[15]);
                    kb2[2] = hi_1_78;
                    kb2[15] = lo_1_62;
                    float hi_1_79 = fmaxf(kb2[3], kb2[14]);
                    float lo_1_63 = fminf(kb2[3], kb2[14]);
                    kb2[3] = hi_1_79;
                    kb2[14] = lo_1_63;
                    float hi_1_80 = fmaxf(kb2[4], kb2[8]);
                    float lo_1_64 = fminf(kb2[4], kb2[8]);
                    kb2[4] = hi_1_80;
                    kb2[8] = lo_1_64;
                    float hi_1_81 = fmaxf(kb2[5], kb2[6]);
                    float lo_1_65 = fminf(kb2[5], kb2[6]);
                    kb2[5] = hi_1_81;
                    kb2[6] = lo_1_65;
                    float hi_1_82 = fmaxf(kb2[7], kb2[11]);
                    float lo_1_66 = fminf(kb2[7], kb2[11]);
                    kb2[7] = hi_1_82;
                    kb2[11] = lo_1_66;
                    float hi_1_83 = fmaxf(kb2[9], kb2[10]);
                    float lo_1_67 = fminf(kb2[9], kb2[10]);
                    kb2[9] = hi_1_83;
                    kb2[10] = lo_1_67;
                    float hi_1_84 = fmaxf(kb2[0], kb2[5]);
                    float lo_1_68 = fminf(kb2[0], kb2[5]);
                    kb2[0] = hi_1_84;
                    kb2[5] = lo_1_68;
                    float hi_1_85 = fmaxf(kb2[1], kb2[7]);
                    float lo_1_69 = fminf(kb2[1], kb2[7]);
                    kb2[1] = hi_1_85;
                    kb2[7] = lo_1_69;
                    float hi_1_86 = fmaxf(kb2[2], kb2[9]);
                    float lo_1_70 = fminf(kb2[2], kb2[9]);
                    kb2[2] = hi_1_86;
                    kb2[9] = lo_1_70;
                    float hi_1_87 = fmaxf(kb2[3], kb2[4]);
                    float lo_1_71 = fminf(kb2[3], kb2[4]);
                    kb2[3] = hi_1_87;
                    kb2[4] = lo_1_71;
                    float hi_1_88 = fmaxf(kb2[6], kb2[13]);
                    float lo_1_72 = fminf(kb2[6], kb2[13]);
                    kb2[6] = hi_1_88;
                    kb2[13] = lo_1_72;
                    float hi_1_89 = fmaxf(kb2[8], kb2[14]);
                    float lo_1_73 = fminf(kb2[8], kb2[14]);
                    kb2[8] = hi_1_89;
                    kb2[14] = lo_1_73;
                    float hi_1_90 = fmaxf(kb2[10], kb2[15]);
                    float lo_1_74 = fminf(kb2[10], kb2[15]);
                    kb2[10] = hi_1_90;
                    kb2[15] = lo_1_74;
                    float hi_1_91 = fmaxf(kb2[11], kb2[12]);
                    float lo_1_75 = fminf(kb2[11], kb2[12]);
                    kb2[11] = hi_1_91;
                    kb2[12] = lo_1_75;
                    float hi_1_92 = fmaxf(kb2[0], kb2[1]);
                    float lo_1_76 = fminf(kb2[0], kb2[1]);
                    kb2[0] = hi_1_92;
                    kb2[1] = lo_1_76;
                    float hi_1_93 = fmaxf(kb2[2], kb2[3]);
                    float lo_1_77 = fminf(kb2[2], kb2[3]);
                    kb2[2] = hi_1_93;
                    kb2[3] = lo_1_77;
                    float hi_1_94 = fmaxf(kb2[4], kb2[5]);
                    float lo_1_78 = fminf(kb2[4], kb2[5]);
                    kb2[4] = hi_1_94;
                    kb2[5] = lo_1_78;
                    float hi_1_95 = fmaxf(kb2[6], kb2[8]);
                    float lo_1_79 = fminf(kb2[6], kb2[8]);
                    kb2[6] = hi_1_95;
                    kb2[8] = lo_1_79;
                    float hi_1_96 = fmaxf(kb2[7], kb2[9]);
                    float lo_1_80 = fminf(kb2[7], kb2[9]);
                    kb2[7] = hi_1_96;
                    kb2[9] = lo_1_80;
                    float hi_1_97 = fmaxf(kb2[10], kb2[11]);
                    float lo_1_81 = fminf(kb2[10], kb2[11]);
                    kb2[10] = hi_1_97;
                    kb2[11] = lo_1_81;
                    float hi_1_98 = fmaxf(kb2[12], kb2[13]);
                    float lo_1_82 = fminf(kb2[12], kb2[13]);
                    kb2[12] = hi_1_98;
                    kb2[13] = lo_1_82;
                    float hi_1_99 = fmaxf(kb2[14], kb2[15]);
                    float lo_1_83 = fminf(kb2[14], kb2[15]);
                    kb2[14] = hi_1_99;
                    kb2[15] = lo_1_83;
                    float hi_1_100 = fmaxf(kb2[0], kb2[2]);
                    float lo_1_84 = fminf(kb2[0], kb2[2]);
                    kb2[0] = hi_1_100;
                    kb2[2] = lo_1_84;
                    float hi_1_101 = fmaxf(kb2[1], kb2[3]);
                    float lo_1_85 = fminf(kb2[1], kb2[3]);
                    kb2[1] = hi_1_101;
                    kb2[3] = lo_1_85;
                    float hi_1_102 = fmaxf(kb2[4], kb2[10]);
                    float lo_1_86 = fminf(kb2[4], kb2[10]);
                    kb2[4] = hi_1_102;
                    kb2[10] = lo_1_86;
                    float hi_1_103 = fmaxf(kb2[5], kb2[11]);
                    float lo_1_87 = fminf(kb2[5], kb2[11]);
                    kb2[5] = hi_1_103;
                    kb2[11] = lo_1_87;
                    float hi_1_104 = fmaxf(kb2[6], kb2[7]);
                    float lo_1_88 = fminf(kb2[6], kb2[7]);
                    kb2[6] = hi_1_104;
                    kb2[7] = lo_1_88;
                    float hi_1_105 = fmaxf(kb2[8], kb2[9]);
                    float lo_1_89 = fminf(kb2[8], kb2[9]);
                    kb2[8] = hi_1_105;
                    kb2[9] = lo_1_89;
                    float hi_1_106 = fmaxf(kb2[12], kb2[14]);
                    float lo_1_90 = fminf(kb2[12], kb2[14]);
                    kb2[12] = hi_1_106;
                    kb2[14] = lo_1_90;
                    float hi_1_107 = fmaxf(kb2[13], kb2[15]);
                    float lo_1_91 = fminf(kb2[13], kb2[15]);
                    kb2[13] = hi_1_107;
                    kb2[15] = lo_1_91;
                    float hi_1_108 = fmaxf(kb2[1], kb2[2]);
                    float lo_1_92 = fminf(kb2[1], kb2[2]);
                    kb2[1] = hi_1_108;
                    kb2[2] = lo_1_92;
                    float hi_1_109 = fmaxf(kb2[3], kb2[12]);
                    float lo_1_93 = fminf(kb2[3], kb2[12]);
                    kb2[3] = hi_1_109;
                    kb2[12] = lo_1_93;
                    float hi_1_110 = fmaxf(kb2[4], kb2[6]);
                    float lo_1_94 = fminf(kb2[4], kb2[6]);
                    kb2[4] = hi_1_110;
                    kb2[6] = lo_1_94;
                    float hi_1_111 = fmaxf(kb2[5], kb2[7]);
                    float lo_1_95 = fminf(kb2[5], kb2[7]);
                    kb2[5] = hi_1_111;
                    kb2[7] = lo_1_95;
                    float hi_1_112 = fmaxf(kb2[8], kb2[10]);
                    float lo_1_96 = fminf(kb2[8], kb2[10]);
                    kb2[8] = hi_1_112;
                    kb2[10] = lo_1_96;
                    float hi_1_113 = fmaxf(kb2[9], kb2[11]);
                    float lo_1_97 = fminf(kb2[9], kb2[11]);
                    kb2[9] = hi_1_113;
                    kb2[11] = lo_1_97;
                    float hi_1_114 = fmaxf(kb2[13], kb2[14]);
                    float lo_1_98 = fminf(kb2[13], kb2[14]);
                    kb2[13] = hi_1_114;
                    kb2[14] = lo_1_98;
                    float hi_1_115 = fmaxf(kb2[1], kb2[4]);
                    float lo_1_99 = fminf(kb2[1], kb2[4]);
                    kb2[1] = hi_1_115;
                    kb2[4] = lo_1_99;
                    float hi_1_116 = fmaxf(kb2[2], kb2[6]);
                    float lo_1_100 = fminf(kb2[2], kb2[6]);
                    kb2[2] = hi_1_116;
                    kb2[6] = lo_1_100;
                    float hi_1_117 = fmaxf(kb2[5], kb2[8]);
                    float lo_1_101 = fminf(kb2[5], kb2[8]);
                    kb2[5] = hi_1_117;
                    kb2[8] = lo_1_101;
                    float hi_1_118 = fmaxf(kb2[7], kb2[10]);
                    float lo_1_102 = fminf(kb2[7], kb2[10]);
                    kb2[7] = hi_1_118;
                    kb2[10] = lo_1_102;
                    float hi_1_119 = fmaxf(kb2[9], kb2[13]);
                    float lo_1_103 = fminf(kb2[9], kb2[13]);
                    kb2[9] = hi_1_119;
                    kb2[13] = lo_1_103;
                    float hi_1_120 = fmaxf(kb2[11], kb2[14]);
                    float lo_1_104 = fminf(kb2[11], kb2[14]);
                    kb2[11] = hi_1_120;
                    kb2[14] = lo_1_104;
                    float hi_1_121 = fmaxf(kb2[2], kb2[4]);
                    float lo_1_105 = fminf(kb2[2], kb2[4]);
                    kb2[2] = hi_1_121;
                    kb2[4] = lo_1_105;
                    float hi_1_122 = fmaxf(kb2[3], kb2[6]);
                    float lo_1_106 = fminf(kb2[3], kb2[6]);
                    kb2[3] = hi_1_122;
                    kb2[6] = lo_1_106;
                    float hi_1_123 = fmaxf(kb2[9], kb2[12]);
                    float lo_1_107 = fminf(kb2[9], kb2[12]);
                    kb2[9] = hi_1_123;
                    kb2[12] = lo_1_107;
                    float hi_1_124 = fmaxf(kb2[11], kb2[13]);
                    float lo_1_108 = fminf(kb2[11], kb2[13]);
                    kb2[11] = hi_1_124;
                    kb2[13] = lo_1_108;
                    float hi_1_125 = fmaxf(kb2[3], kb2[5]);
                    float lo_1_109 = fminf(kb2[3], kb2[5]);
                    kb2[3] = hi_1_125;
                    kb2[5] = lo_1_109;
                    float hi_1_126 = fmaxf(kb2[6], kb2[8]);
                    float lo_1_110 = fminf(kb2[6], kb2[8]);
                    kb2[6] = hi_1_126;
                    kb2[8] = lo_1_110;
                    float hi_1_127 = fmaxf(kb2[7], kb2[9]);
                    float lo_1_111 = fminf(kb2[7], kb2[9]);
                    kb2[7] = hi_1_127;
                    kb2[9] = lo_1_111;
                    float hi_1_128 = fmaxf(kb2[10], kb2[12]);
                    float lo_1_112 = fminf(kb2[10], kb2[12]);
                    kb2[10] = hi_1_128;
                    kb2[12] = lo_1_112;
                    float hi_1_129 = fmaxf(kb2[3], kb2[4]);
                    float lo_1_113 = fminf(kb2[3], kb2[4]);
                    kb2[3] = hi_1_129;
                    kb2[4] = lo_1_113;
                    float hi_1_130 = fmaxf(kb2[5], kb2[6]);
                    float lo_1_114 = fminf(kb2[5], kb2[6]);
                    kb2[5] = hi_1_130;
                    kb2[6] = lo_1_114;
                    float hi_1_131 = fmaxf(kb2[7], kb2[8]);
                    float lo_1_115 = fminf(kb2[7], kb2[8]);
                    kb2[7] = hi_1_131;
                    kb2[8] = lo_1_115;
                    float hi_1_132 = fmaxf(kb2[9], kb2[10]);
                    float lo_1_116 = fminf(kb2[9], kb2[10]);
                    kb2[9] = hi_1_132;
                    kb2[10] = lo_1_116;
                    float hi_1_133 = fmaxf(kb2[11], kb2[12]);
                    float lo_1_117 = fminf(kb2[11], kb2[12]);
                    kb2[11] = hi_1_133;
                    kb2[12] = lo_1_117;
                    float hi_1_134 = fmaxf(kb2[6], kb2[7]);
                    float lo_1_118 = fminf(kb2[6], kb2[7]);
                    kb2[6] = hi_1_134;
                    kb2[7] = lo_1_118;
                    float hi_1_135 = fmaxf(kb2[8], kb2[9]);
                    float lo_1_119 = fminf(kb2[8], kb2[9]);
                    kb2[8] = hi_1_135;
                    kb2[9] = lo_1_119;
                    float hi_80 = fmaxf(a2[0], kb2[15]);
                    a2[0] = hi_80;
                    float hi_81 = fmaxf(a2[1], kb2[14]);
                    a2[1] = hi_81;
                    float hi_82 = fmaxf(a2[2], kb2[13]);
                    a2[2] = hi_82;
                    float hi_83 = fmaxf(a2[3], kb2[12]);
                    a2[3] = hi_83;
                    float hi_84 = fmaxf(a2[4], kb2[11]);
                    a2[4] = hi_84;
                    float hi_85 = fmaxf(a2[5], kb2[10]);
                    a2[5] = hi_85;
                    float hi_86 = fmaxf(a2[6], kb2[9]);
                    a2[6] = hi_86;
                    float hi_87 = fmaxf(a2[7], kb2[8]);
                    a2[7] = hi_87;
                    float hi_88 = fmaxf(a2[8], kb2[7]);
                    a2[8] = hi_88;
                    float hi_89 = fmaxf(a2[9], kb2[6]);
                    a2[9] = hi_89;
                    float hi_90 = fmaxf(a2[10], kb2[5]);
                    a2[10] = hi_90;
                    float hi_91 = fmaxf(a2[11], kb2[4]);
                    a2[11] = hi_91;
                    float hi_92 = fmaxf(a2[12], kb2[3]);
                    a2[12] = hi_92;
                    float hi_93 = fmaxf(a2[13], kb2[2]);
                    a2[13] = hi_93;
                    float hi_94 = fmaxf(a2[14], kb2[1]);
                    a2[14] = hi_94;
                    float hi_95 = fmaxf(a2[15], kb2[0]);
                    a2[15] = hi_95;
                    float hi_96 = fmaxf(a2[0], a2[8]);
                    float lo_80 = fminf(a2[0], a2[8]);
                    a2[0] = hi_96;
                    a2[8] = lo_80;
                    float hi_97 = fmaxf(a2[1], a2[9]);
                    float lo_81 = fminf(a2[1], a2[9]);
                    a2[1] = hi_97;
                    a2[9] = lo_81;
                    float hi_98 = fmaxf(a2[2], a2[10]);
                    float lo_82 = fminf(a2[2], a2[10]);
                    a2[2] = hi_98;
                    a2[10] = lo_82;
                    float hi_99 = fmaxf(a2[3], a2[11]);
                    float lo_83 = fminf(a2[3], a2[11]);
                    a2[3] = hi_99;
                    a2[11] = lo_83;
                    float hi_100 = fmaxf(a2[4], a2[12]);
                    float lo_84 = fminf(a2[4], a2[12]);
                    a2[4] = hi_100;
                    a2[12] = lo_84;
                    float hi_101 = fmaxf(a2[5], a2[13]);
                    float lo_85 = fminf(a2[5], a2[13]);
                    a2[5] = hi_101;
                    a2[13] = lo_85;
                    float hi_102 = fmaxf(a2[6], a2[14]);
                    float lo_86 = fminf(a2[6], a2[14]);
                    a2[6] = hi_102;
                    a2[14] = lo_86;
                    float hi_103 = fmaxf(a2[7], a2[15]);
                    float lo_87 = fminf(a2[7], a2[15]);
                    a2[7] = hi_103;
                    a2[15] = lo_87;
                    float hi_104 = fmaxf(a2[0], a2[4]);
                    float lo_88 = fminf(a2[0], a2[4]);
                    a2[0] = hi_104;
                    a2[4] = lo_88;
                    float hi_105 = fmaxf(a2[1], a2[5]);
                    float lo_89 = fminf(a2[1], a2[5]);
                    a2[1] = hi_105;
                    a2[5] = lo_89;
                    float hi_106 = fmaxf(a2[2], a2[6]);
                    float lo_90 = fminf(a2[2], a2[6]);
                    a2[2] = hi_106;
                    a2[6] = lo_90;
                    float hi_107 = fmaxf(a2[3], a2[7]);
                    float lo_91 = fminf(a2[3], a2[7]);
                    a2[3] = hi_107;
                    a2[7] = lo_91;
                    float hi_108 = fmaxf(a2[8], a2[12]);
                    float lo_92 = fminf(a2[8], a2[12]);
                    a2[8] = hi_108;
                    a2[12] = lo_92;
                    float hi_109 = fmaxf(a2[9], a2[13]);
                    float lo_93 = fminf(a2[9], a2[13]);
                    a2[9] = hi_109;
                    a2[13] = lo_93;
                    float hi_110 = fmaxf(a2[10], a2[14]);
                    float lo_94 = fminf(a2[10], a2[14]);
                    a2[10] = hi_110;
                    a2[14] = lo_94;
                    float hi_111 = fmaxf(a2[11], a2[15]);
                    float lo_95 = fminf(a2[11], a2[15]);
                    a2[11] = hi_111;
                    a2[15] = lo_95;
                    float hi_112 = fmaxf(a2[0], a2[2]);
                    float lo_96 = fminf(a2[0], a2[2]);
                    a2[0] = hi_112;
                    a2[2] = lo_96;
                    float hi_113 = fmaxf(a2[1], a2[3]);
                    float lo_97 = fminf(a2[1], a2[3]);
                    a2[1] = hi_113;
                    a2[3] = lo_97;
                    float hi_114 = fmaxf(a2[4], a2[6]);
                    float lo_98 = fminf(a2[4], a2[6]);
                    a2[4] = hi_114;
                    a2[6] = lo_98;
                    float hi_115 = fmaxf(a2[5], a2[7]);
                    float lo_99 = fminf(a2[5], a2[7]);
                    a2[5] = hi_115;
                    a2[7] = lo_99;
                    float hi_116 = fmaxf(a2[8], a2[10]);
                    float lo_100 = fminf(a2[8], a2[10]);
                    a2[8] = hi_116;
                    a2[10] = lo_100;
                    float hi_117 = fmaxf(a2[9], a2[11]);
                    float lo_101 = fminf(a2[9], a2[11]);
                    a2[9] = hi_117;
                    a2[11] = lo_101;
                    float hi_118 = fmaxf(a2[12], a2[14]);
                    float lo_102 = fminf(a2[12], a2[14]);
                    a2[12] = hi_118;
                    a2[14] = lo_102;
                    float hi_119 = fmaxf(a2[13], a2[15]);
                    float lo_103 = fminf(a2[13], a2[15]);
                    a2[13] = hi_119;
                    a2[15] = lo_103;
                    float hi_120 = fmaxf(a2[0], a2[1]);
                    float lo_104 = fminf(a2[0], a2[1]);
                    a2[0] = hi_120;
                    a2[1] = lo_104;
                    float hi_121 = fmaxf(a2[2], a2[3]);
                    float lo_105 = fminf(a2[2], a2[3]);
                    a2[2] = hi_121;
                    a2[3] = lo_105;
                    float hi_122 = fmaxf(a2[4], a2[5]);
                    float lo_106 = fminf(a2[4], a2[5]);
                    a2[4] = hi_122;
                    a2[5] = lo_106;
                    float hi_123 = fmaxf(a2[6], a2[7]);
                    float lo_107 = fminf(a2[6], a2[7]);
                    a2[6] = hi_123;
                    a2[7] = lo_107;
                    float hi_124 = fmaxf(a2[8], a2[9]);
                    float lo_108 = fminf(a2[8], a2[9]);
                    a2[8] = hi_124;
                    a2[9] = lo_108;
                    float hi_125 = fmaxf(a2[10], a2[11]);
                    float lo_109 = fminf(a2[10], a2[11]);
                    a2[10] = hi_125;
                    a2[11] = lo_109;
                    float hi_126 = fmaxf(a2[12], a2[13]);
                    float lo_110 = fminf(a2[12], a2[13]);
                    a2[12] = hi_126;
                    a2[13] = lo_110;
                    float hi_127 = fmaxf(a2[14], a2[15]);
                    float lo_111 = fminf(a2[14], a2[15]);
                    a2[14] = hi_127;
                    a2[15] = lo_111;
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
        #pragma unroll 1
        for (int k_1 = 0; k_1 < 3; k_1++) {
            int bit2 = 1 << k_1;
            int low2 = whi & (bit2 << 1) - 1;
            if (low2 == bit2) {
                int base_1 = whi * 1088 + c;
                tree[base_1] = a2[0];
                tree[base_1 + 64] = a2[1];
                tree[base_1 + 128] = a2[2];
                tree[base_1 + 192] = a2[3];
                tree[base_1 + 256] = a2[4];
                tree[base_1 + 320] = a2[5];
                tree[base_1 + 384] = a2[6];
                tree[base_1 + 448] = a2[7];
                tree[base_1 + 512] = a2[8];
                tree[base_1 + 576] = a2[9];
                tree[base_1 + 640] = a2[10];
                tree[base_1 + 704] = a2[11];
                tree[base_1 + 768] = a2[12];
                tree[base_1 + 832] = a2[13];
                tree[base_1 + 896] = a2[14];
                tree[base_1 + 960] = a2[15];
            }
            asm volatile("barrier.sync 1, 512;" ::: "memory");
            if (low2 == 0) {
                float sb2[16];
                int rbase2 = (whi + bit2) * 1088 + c;
                sb2[0] = tree[rbase2];
                sb2[1] = tree[rbase2 + 64];
                sb2[2] = tree[rbase2 + 128];
                sb2[3] = tree[rbase2 + 192];
                sb2[4] = tree[rbase2 + 256];
                sb2[5] = tree[rbase2 + 320];
                sb2[6] = tree[rbase2 + 384];
                sb2[7] = tree[rbase2 + 448];
                sb2[8] = tree[rbase2 + 512];
                sb2[9] = tree[rbase2 + 576];
                sb2[10] = tree[rbase2 + 640];
                sb2[11] = tree[rbase2 + 704];
                sb2[12] = tree[rbase2 + 768];
                sb2[13] = tree[rbase2 + 832];
                sb2[14] = tree[rbase2 + 896];
                sb2[15] = tree[rbase2 + 960];
                float hi_128 = fmaxf(a2[0], sb2[15]);
                a2[0] = hi_128;
                float hi_129 = fmaxf(a2[1], sb2[14]);
                a2[1] = hi_129;
                float hi_130 = fmaxf(a2[2], sb2[13]);
                a2[2] = hi_130;
                float hi_131 = fmaxf(a2[3], sb2[12]);
                a2[3] = hi_131;
                float hi_132 = fmaxf(a2[4], sb2[11]);
                a2[4] = hi_132;
                float hi_133 = fmaxf(a2[5], sb2[10]);
                a2[5] = hi_133;
                float hi_134 = fmaxf(a2[6], sb2[9]);
                a2[6] = hi_134;
                float hi_135 = fmaxf(a2[7], sb2[8]);
                a2[7] = hi_135;
                float hi_136 = fmaxf(a2[8], sb2[7]);
                a2[8] = hi_136;
                float hi_137 = fmaxf(a2[9], sb2[6]);
                a2[9] = hi_137;
                float hi_138 = fmaxf(a2[10], sb2[5]);
                a2[10] = hi_138;
                float hi_139 = fmaxf(a2[11], sb2[4]);
                a2[11] = hi_139;
                float hi_140 = fmaxf(a2[12], sb2[3]);
                a2[12] = hi_140;
                float hi_141 = fmaxf(a2[13], sb2[2]);
                a2[13] = hi_141;
                float hi_142 = fmaxf(a2[14], sb2[1]);
                a2[14] = hi_142;
                float hi_143 = fmaxf(a2[15], sb2[0]);
                a2[15] = hi_143;
                float hi_144 = fmaxf(a2[0], a2[8]);
                float lo_112 = fminf(a2[0], a2[8]);
                a2[0] = hi_144;
                a2[8] = lo_112;
                float hi_145 = fmaxf(a2[1], a2[9]);
                float lo_113 = fminf(a2[1], a2[9]);
                a2[1] = hi_145;
                a2[9] = lo_113;
                float hi_146 = fmaxf(a2[2], a2[10]);
                float lo_114 = fminf(a2[2], a2[10]);
                a2[2] = hi_146;
                a2[10] = lo_114;
                float hi_147 = fmaxf(a2[3], a2[11]);
                float lo_115 = fminf(a2[3], a2[11]);
                a2[3] = hi_147;
                a2[11] = lo_115;
                float hi_148 = fmaxf(a2[4], a2[12]);
                float lo_116 = fminf(a2[4], a2[12]);
                a2[4] = hi_148;
                a2[12] = lo_116;
                float hi_149 = fmaxf(a2[5], a2[13]);
                float lo_117 = fminf(a2[5], a2[13]);
                a2[5] = hi_149;
                a2[13] = lo_117;
                float hi_150 = fmaxf(a2[6], a2[14]);
                float lo_118 = fminf(a2[6], a2[14]);
                a2[6] = hi_150;
                a2[14] = lo_118;
                float hi_151 = fmaxf(a2[7], a2[15]);
                float lo_119 = fminf(a2[7], a2[15]);
                a2[7] = hi_151;
                a2[15] = lo_119;
                float hi_152 = fmaxf(a2[0], a2[4]);
                float lo_120 = fminf(a2[0], a2[4]);
                a2[0] = hi_152;
                a2[4] = lo_120;
                float hi_153 = fmaxf(a2[1], a2[5]);
                float lo_121 = fminf(a2[1], a2[5]);
                a2[1] = hi_153;
                a2[5] = lo_121;
                float hi_154 = fmaxf(a2[2], a2[6]);
                float lo_122 = fminf(a2[2], a2[6]);
                a2[2] = hi_154;
                a2[6] = lo_122;
                float hi_155 = fmaxf(a2[3], a2[7]);
                float lo_123 = fminf(a2[3], a2[7]);
                a2[3] = hi_155;
                a2[7] = lo_123;
                float hi_156 = fmaxf(a2[8], a2[12]);
                float lo_124 = fminf(a2[8], a2[12]);
                a2[8] = hi_156;
                a2[12] = lo_124;
                float hi_157 = fmaxf(a2[9], a2[13]);
                float lo_125 = fminf(a2[9], a2[13]);
                a2[9] = hi_157;
                a2[13] = lo_125;
                float hi_158 = fmaxf(a2[10], a2[14]);
                float lo_126 = fminf(a2[10], a2[14]);
                a2[10] = hi_158;
                a2[14] = lo_126;
                float hi_159 = fmaxf(a2[11], a2[15]);
                float lo_127 = fminf(a2[11], a2[15]);
                a2[11] = hi_159;
                a2[15] = lo_127;
                float hi_160 = fmaxf(a2[0], a2[2]);
                float lo_128 = fminf(a2[0], a2[2]);
                a2[0] = hi_160;
                a2[2] = lo_128;
                float hi_161 = fmaxf(a2[1], a2[3]);
                float lo_129 = fminf(a2[1], a2[3]);
                a2[1] = hi_161;
                a2[3] = lo_129;
                float hi_162 = fmaxf(a2[4], a2[6]);
                float lo_130 = fminf(a2[4], a2[6]);
                a2[4] = hi_162;
                a2[6] = lo_130;
                float hi_163 = fmaxf(a2[5], a2[7]);
                float lo_131 = fminf(a2[5], a2[7]);
                a2[5] = hi_163;
                a2[7] = lo_131;
                float hi_164 = fmaxf(a2[8], a2[10]);
                float lo_132 = fminf(a2[8], a2[10]);
                a2[8] = hi_164;
                a2[10] = lo_132;
                float hi_165 = fmaxf(a2[9], a2[11]);
                float lo_133 = fminf(a2[9], a2[11]);
                a2[9] = hi_165;
                a2[11] = lo_133;
                float hi_166 = fmaxf(a2[12], a2[14]);
                float lo_134 = fminf(a2[12], a2[14]);
                a2[12] = hi_166;
                a2[14] = lo_134;
                float hi_167 = fmaxf(a2[13], a2[15]);
                float lo_135 = fminf(a2[13], a2[15]);
                a2[13] = hi_167;
                a2[15] = lo_135;
                float hi_168 = fmaxf(a2[0], a2[1]);
                float lo_136 = fminf(a2[0], a2[1]);
                a2[0] = hi_168;
                a2[1] = lo_136;
                float hi_169 = fmaxf(a2[2], a2[3]);
                float lo_137 = fminf(a2[2], a2[3]);
                a2[2] = hi_169;
                a2[3] = lo_137;
                float hi_170 = fmaxf(a2[4], a2[5]);
                float lo_138 = fminf(a2[4], a2[5]);
                a2[4] = hi_170;
                a2[5] = lo_138;
                float hi_171 = fmaxf(a2[6], a2[7]);
                float lo_139 = fminf(a2[6], a2[7]);
                a2[6] = hi_171;
                a2[7] = lo_139;
                float hi_172 = fmaxf(a2[8], a2[9]);
                float lo_140 = fminf(a2[8], a2[9]);
                a2[8] = hi_172;
                a2[9] = lo_140;
                float hi_173 = fmaxf(a2[10], a2[11]);
                float lo_141 = fminf(a2[10], a2[11]);
                a2[10] = hi_173;
                a2[11] = lo_141;
                float hi_174 = fmaxf(a2[12], a2[13]);
                float lo_142 = fminf(a2[12], a2[13]);
                a2[12] = hi_174;
                a2[13] = lo_142;
                float hi_175 = fmaxf(a2[14], a2[15]);
                float lo_143 = fminf(a2[14], a2[15]);
                a2[14] = hi_175;
                a2[15] = lo_143;
            }
        }
    }
    if (w == 0 && col < total_q) {
        unsigned int o[16];
        unsigned int k1_1_0 = __as_u32(a[0]);
        unsigned int idx_2_0 = 4294967295u;
        if (k1_1_0 < 4282384384u) {
            idx_2_0 = k1_1_0 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_0 = __as_u32(a2[0]);
            idx_2_0 = 4294967295u;
            if (k2_1_0 != 0) {
                idx_2_0 = k2_1_0 & 1023;
            }
        }
        o[0] = idx_2_0;
        unsigned int k1_1_1 = __as_u32(a[1]);
        unsigned int idx_2_1 = 4294967295u;
        if (k1_1_1 < 4282384384u) {
            idx_2_1 = k1_1_1 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_1 = __as_u32(a2[1]);
            idx_2_1 = 4294967295u;
            if (k2_1_1 != 0) {
                idx_2_1 = k2_1_1 & 1023;
            }
        }
        o[1] = idx_2_1;
        unsigned int k1_1_2 = __as_u32(a[2]);
        unsigned int idx_2_2 = 4294967295u;
        if (k1_1_2 < 4282384384u) {
            idx_2_2 = k1_1_2 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_2 = __as_u32(a2[2]);
            idx_2_2 = 4294967295u;
            if (k2_1_2 != 0) {
                idx_2_2 = k2_1_2 & 1023;
            }
        }
        o[2] = idx_2_2;
        unsigned int k1_1_3 = __as_u32(a[3]);
        unsigned int idx_2_3 = 4294967295u;
        if (k1_1_3 < 4282384384u) {
            idx_2_3 = k1_1_3 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_3 = __as_u32(a2[3]);
            idx_2_3 = 4294967295u;
            if (k2_1_3 != 0) {
                idx_2_3 = k2_1_3 & 1023;
            }
        }
        o[3] = idx_2_3;
        unsigned int k1_1_4 = __as_u32(a[4]);
        unsigned int idx_2_4 = 4294967295u;
        if (k1_1_4 < 4282384384u) {
            idx_2_4 = k1_1_4 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_4 = __as_u32(a2[4]);
            idx_2_4 = 4294967295u;
            if (k2_1_4 != 0) {
                idx_2_4 = k2_1_4 & 1023;
            }
        }
        o[4] = idx_2_4;
        unsigned int k1_1_5 = __as_u32(a[5]);
        unsigned int idx_2_5 = 4294967295u;
        if (k1_1_5 < 4282384384u) {
            idx_2_5 = k1_1_5 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_5 = __as_u32(a2[5]);
            idx_2_5 = 4294967295u;
            if (k2_1_5 != 0) {
                idx_2_5 = k2_1_5 & 1023;
            }
        }
        o[5] = idx_2_5;
        unsigned int k1_1_6 = __as_u32(a[6]);
        unsigned int idx_2_6 = 4294967295u;
        if (k1_1_6 < 4282384384u) {
            idx_2_6 = k1_1_6 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_6 = __as_u32(a2[6]);
            idx_2_6 = 4294967295u;
            if (k2_1_6 != 0) {
                idx_2_6 = k2_1_6 & 1023;
            }
        }
        o[6] = idx_2_6;
        unsigned int k1_1_7 = __as_u32(a[7]);
        unsigned int idx_2_7 = 4294967295u;
        if (k1_1_7 < 4282384384u) {
            idx_2_7 = k1_1_7 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_7 = __as_u32(a2[7]);
            idx_2_7 = 4294967295u;
            if (k2_1_7 != 0) {
                idx_2_7 = k2_1_7 & 1023;
            }
        }
        o[7] = idx_2_7;
        unsigned int k1_1_8 = __as_u32(a[8]);
        unsigned int idx_2_8 = 4294967295u;
        if (k1_1_8 < 4282384384u) {
            idx_2_8 = k1_1_8 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_8 = __as_u32(a2[8]);
            idx_2_8 = 4294967295u;
            if (k2_1_8 != 0) {
                idx_2_8 = k2_1_8 & 1023;
            }
        }
        o[8] = idx_2_8;
        unsigned int k1_1_9 = __as_u32(a[9]);
        unsigned int idx_2_9 = 4294967295u;
        if (k1_1_9 < 4282384384u) {
            idx_2_9 = k1_1_9 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_9 = __as_u32(a2[9]);
            idx_2_9 = 4294967295u;
            if (k2_1_9 != 0) {
                idx_2_9 = k2_1_9 & 1023;
            }
        }
        o[9] = idx_2_9;
        unsigned int k1_1_10 = __as_u32(a[10]);
        unsigned int idx_2_10 = 4294967295u;
        if (k1_1_10 < 4282384384u) {
            idx_2_10 = k1_1_10 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_10 = __as_u32(a2[10]);
            idx_2_10 = 4294967295u;
            if (k2_1_10 != 0) {
                idx_2_10 = k2_1_10 & 1023;
            }
        }
        o[10] = idx_2_10;
        unsigned int k1_1_11 = __as_u32(a[11]);
        unsigned int idx_2_11 = 4294967295u;
        if (k1_1_11 < 4282384384u) {
            idx_2_11 = k1_1_11 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_11 = __as_u32(a2[11]);
            idx_2_11 = 4294967295u;
            if (k2_1_11 != 0) {
                idx_2_11 = k2_1_11 & 1023;
            }
        }
        o[11] = idx_2_11;
        unsigned int k1_1_12 = __as_u32(a[12]);
        unsigned int idx_2_12 = 4294967295u;
        if (k1_1_12 < 4282384384u) {
            idx_2_12 = k1_1_12 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_12 = __as_u32(a2[12]);
            idx_2_12 = 4294967295u;
            if (k2_1_12 != 0) {
                idx_2_12 = k2_1_12 & 1023;
            }
        }
        o[12] = idx_2_12;
        unsigned int k1_1_13 = __as_u32(a[13]);
        unsigned int idx_2_13 = 4294967295u;
        if (k1_1_13 < 4282384384u) {
            idx_2_13 = k1_1_13 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_13 = __as_u32(a2[13]);
            idx_2_13 = 4294967295u;
            if (k2_1_13 != 0) {
                idx_2_13 = k2_1_13 & 1023;
            }
        }
        o[13] = idx_2_13;
        unsigned int k1_1_14 = __as_u32(a[14]);
        unsigned int idx_2_14 = 4294967295u;
        if (k1_1_14 < 4282384384u) {
            idx_2_14 = k1_1_14 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_14 = __as_u32(a2[14]);
            idx_2_14 = 4294967295u;
            if (k2_1_14 != 0) {
                idx_2_14 = k2_1_14 & 1023;
            }
        }
        o[14] = idx_2_14;
        unsigned int k1_1_15 = __as_u32(a[15]);
        unsigned int idx_2_15 = 4294967295u;
        if (k1_1_15 < 4282384384u) {
            idx_2_15 = k1_1_15 & 1023;
        }
        if (flagged != 0) {
            unsigned int k2_1_15 = __as_u32(a2[15]);
            idx_2_15 = 4294967295u;
            if (k2_1_15 != 0) {
                idx_2_15 = k2_1_15 & 1023;
            }
        }
        o[15] = idx_2_15;
        unsigned int hi_176 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
        unsigned int lo_144 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
        o[0] = lo_144;
        o[13] = hi_176;
        unsigned int hi_177 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
        unsigned int lo_145 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
        o[1] = lo_145;
        o[12] = hi_177;
        unsigned int hi_178 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
        unsigned int lo_146 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
        o[2] = lo_146;
        o[15] = hi_178;
        unsigned int hi_179 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
        unsigned int lo_147 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
        o[3] = lo_147;
        o[14] = hi_179;
        unsigned int hi_180 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
        unsigned int lo_148 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
        o[4] = lo_148;
        o[8] = hi_180;
        unsigned int hi_181 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_149 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        o[5] = lo_149;
        o[6] = hi_181;
        unsigned int hi_182 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
        unsigned int lo_150 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
        o[7] = lo_150;
        o[11] = hi_182;
        unsigned int hi_183 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_151 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        o[9] = lo_151;
        o[10] = hi_183;
        unsigned int hi_184 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
        unsigned int lo_152 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
        o[0] = lo_152;
        o[5] = hi_184;
        unsigned int hi_185 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
        unsigned int lo_153 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
        o[1] = lo_153;
        o[7] = hi_185;
        unsigned int hi_186 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
        unsigned int lo_154 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
        o[2] = lo_154;
        o[9] = hi_186;
        unsigned int hi_187 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_155 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        o[3] = lo_155;
        o[4] = hi_187;
        unsigned int hi_188 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
        unsigned int lo_156 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
        o[6] = lo_156;
        o[13] = hi_188;
        unsigned int hi_189 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
        unsigned int lo_157 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
        o[8] = lo_157;
        o[14] = hi_189;
        unsigned int hi_190 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
        unsigned int lo_158 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
        o[10] = lo_158;
        o[15] = hi_190;
        unsigned int hi_191 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_159 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        o[11] = lo_159;
        o[12] = hi_191;
        unsigned int hi_192 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
        unsigned int lo_160 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
        o[0] = lo_160;
        o[1] = hi_192;
        unsigned int hi_193 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
        unsigned int lo_161 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
        o[2] = lo_161;
        o[3] = hi_193;
        unsigned int hi_194 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
        unsigned int lo_162 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
        o[4] = lo_162;
        o[5] = hi_194;
        unsigned int hi_195 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_163 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        o[6] = lo_163;
        o[8] = hi_195;
        unsigned int hi_196 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_164 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        o[7] = lo_164;
        o[9] = hi_196;
        unsigned int hi_197 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
        unsigned int lo_165 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
        o[10] = lo_165;
        o[11] = hi_197;
        unsigned int hi_198 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
        unsigned int lo_166 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
        o[12] = lo_166;
        o[13] = hi_198;
        unsigned int hi_199 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
        unsigned int lo_167 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
        o[14] = lo_167;
        o[15] = hi_199;
        unsigned int hi_200 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
        unsigned int lo_168 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
        o[0] = lo_168;
        o[2] = hi_200;
        unsigned int hi_201 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
        unsigned int lo_169 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
        o[1] = lo_169;
        o[3] = hi_201;
        unsigned int hi_202 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
        unsigned int lo_170 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
        o[4] = lo_170;
        o[10] = hi_202;
        unsigned int hi_203 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
        unsigned int lo_171 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
        o[5] = lo_171;
        o[11] = hi_203;
        unsigned int hi_204 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_172 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        o[6] = lo_172;
        o[7] = hi_204;
        unsigned int hi_205 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_173 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        o[8] = lo_173;
        o[9] = hi_205;
        unsigned int hi_206 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
        unsigned int lo_174 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
        o[12] = lo_174;
        o[14] = hi_206;
        unsigned int hi_207 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
        unsigned int lo_175 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
        o[13] = lo_175;
        o[15] = hi_207;
        unsigned int hi_208 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
        unsigned int lo_176 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
        o[1] = lo_176;
        o[2] = hi_208;
        unsigned int hi_209 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
        unsigned int lo_177 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
        o[3] = lo_177;
        o[12] = hi_209;
        unsigned int hi_210 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
        unsigned int lo_178 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
        o[4] = lo_178;
        o[6] = hi_210;
        unsigned int hi_211 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
        unsigned int lo_179 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
        o[5] = lo_179;
        o[7] = hi_211;
        unsigned int hi_212 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
        unsigned int lo_180 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
        o[8] = lo_180;
        o[10] = hi_212;
        unsigned int hi_213 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
        unsigned int lo_181 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
        o[9] = lo_181;
        o[11] = hi_213;
        unsigned int hi_214 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
        unsigned int lo_182 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
        o[13] = lo_182;
        o[14] = hi_214;
        unsigned int hi_215 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
        unsigned int lo_183 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
        o[1] = lo_183;
        o[4] = hi_215;
        unsigned int hi_216 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
        unsigned int lo_184 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
        o[2] = lo_184;
        o[6] = hi_216;
        unsigned int hi_217 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
        unsigned int lo_185 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
        o[5] = lo_185;
        o[8] = hi_217;
        unsigned int hi_218 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
        unsigned int lo_186 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
        o[7] = lo_186;
        o[10] = hi_218;
        unsigned int hi_219 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
        unsigned int lo_187 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
        o[9] = lo_187;
        o[13] = hi_219;
        unsigned int hi_220 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
        unsigned int lo_188 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
        o[11] = lo_188;
        o[14] = hi_220;
        unsigned int hi_221 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
        unsigned int lo_189 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
        o[2] = lo_189;
        o[4] = hi_221;
        unsigned int hi_222 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
        unsigned int lo_190 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
        o[3] = lo_190;
        o[6] = hi_222;
        unsigned int hi_223 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
        unsigned int lo_191 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
        o[9] = lo_191;
        o[12] = hi_223;
        unsigned int hi_224 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
        unsigned int lo_192 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
        o[11] = lo_192;
        o[13] = hi_224;
        unsigned int hi_225 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
        unsigned int lo_193 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
        o[3] = lo_193;
        o[5] = hi_225;
        unsigned int hi_226 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_194 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        o[6] = lo_194;
        o[8] = hi_226;
        unsigned int hi_227 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_195 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        o[7] = lo_195;
        o[9] = hi_227;
        unsigned int hi_228 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
        unsigned int lo_196 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
        o[10] = lo_196;
        o[12] = hi_228;
        unsigned int hi_229 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_197 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        o[3] = lo_197;
        o[4] = hi_229;
        unsigned int hi_230 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_198 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        o[5] = lo_198;
        o[6] = hi_230;
        unsigned int hi_231 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
        unsigned int lo_199 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
        o[7] = lo_199;
        o[8] = hi_231;
        unsigned int hi_232 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_200 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        o[9] = lo_200;
        o[10] = hi_232;
        unsigned int hi_233 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_201 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        o[11] = lo_201;
        o[12] = hi_233;
        unsigned int hi_234 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_202 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        o[6] = lo_202;
        o[7] = hi_234;
        unsigned int hi_235 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_203 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        o[8] = lo_203;
        o[9] = hi_235;
        int oi[16];
        oi[0] = (int)o[0];
        oi[1] = (int)o[1];
        oi[2] = (int)o[2];
        oi[3] = (int)o[3];
        oi[4] = (int)o[4];
        oi[5] = (int)o[5];
        oi[6] = (int)o[6];
        oi[7] = (int)o[7];
        oi[8] = (int)o[8];
        oi[9] = (int)o[9];
        oi[10] = (int)o[10];
        oi[11] = (int)o[11];
        oi[12] = (int)o[12];
        oi[13] = (int)o[13];
        oi[14] = (int)o[14];
        oi[15] = (int)o[15];
        long long obase = ((long long)col * (long long)num_heads + (long long)head) * 16ll;
        {
            int4 _iv4 = make_int4(oi[0 + 0], oi[0 + 1], oi[0 + 2], oi[0 + 3]);
            *reinterpret_cast<int4*>(out + obase) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[4 + 0], oi[4 + 1], oi[4 + 2], oi[4 + 3]);
            *reinterpret_cast<int4*>(out + obase + 4ll) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[8 + 0], oi[8 + 1], oi[8 + 2], oi[8 + 3]);
            *reinterpret_cast<int4*>(out + obase + 8ll) = _iv4;
        }
        {
            int4 _iv4 = make_int4(oi[12 + 0], oi[12 + 1], oi[12 + 2], oi[12 + 3]);
            *reinterpret_cast<int4*>(out + obase + 12ll) = _iv4;
        }
    }
}

} // extern "C"
