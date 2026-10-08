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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_QCOL_OFF 0
#define SMEM_QCOL_STAGE_BYTES 32
#define SMEM_QCOL_STRIDE 32
#define SMEM_FLAGW_OFF 32
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_PUB_OFF 48
#define SMEM_PUB_STAGE_BYTES 17408
#define SMEM_PUB_STRIDE 17408
#define SMEM_Q2_OFF 17456
#define SMEM_Q2_STAGE_BYTES 2048
#define SMEM_Q2_STRIDE 2048
#define SMEM_TOTAL 19584
#define THREADS 256

#include <math_constants.h>


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

__global__ __launch_bounds__(256) void
kernel_cake_hopper_msa_a8624e8d93fe530d3b35(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int qcol_addr = smem + 0;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 32);
    const int flagw_addr = smem + 32;
    float* pub = reinterpret_cast<float*>(smem_raw + 48);
    const int pub_addr = smem + 48;
    float* q2 = reinterpret_cast<float*>(smem_raw + 17456);
    const int q2_addr = smem + 17456;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 8;
    int c = tid_1 - w * 8;
    int whi = w / 4;
    int col = blockIdx.x * 8 + c;
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
    float neg_inf = -CAKE_INF;
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
    unsigned int nb[16];
    float kb[16];
    int t0 = w * 16;
    long long p = cbase + (long long)t0 * nq64;
    cb[0] = 4286578688;
    if (lim_full > t0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688;
    if (lim_full > t0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688;
    if (lim_full > t0 + 2) {
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688;
    if (lim_full > t0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688;
    if (lim_full > t0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688;
    if (lim_full > t0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688;
    if (lim_full > t0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688;
    if (lim_full > t0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688;
    if (lim_full > t0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688;
    if (lim_full > t0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688;
    if (lim_full > t0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688;
    if (lim_full > t0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688;
    if (lim_full > t0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688;
    if (lim_full > t0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688;
    if (lim_full > t0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688;
    if (lim_full > t0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t00 = w * 16;
    if (lim <= t00) {
        cb[0] = 4286578688;
    }
    if (lim <= t00 + 1) {
        cb[1] = 4286578688;
    }
    if (lim <= t00 + 2) {
        cb[2] = 4286578688;
    }
    if (lim <= t00 + 3) {
        cb[3] = 4286578688;
    }
    if (lim <= t00 + 4) {
        cb[4] = 4286578688;
    }
    if (lim <= t00 + 5) {
        cb[5] = 4286578688;
    }
    if (lim <= t00 + 6) {
        cb[6] = 4286578688;
    }
    if (lim <= t00 + 7) {
        cb[7] = 4286578688;
    }
    if (lim <= t00 + 8) {
        cb[8] = 4286578688;
    }
    if (lim <= t00 + 9) {
        cb[9] = 4286578688;
    }
    if (lim <= t00 + 10) {
        cb[10] = 4286578688;
    }
    if (lim <= t00 + 11) {
        cb[11] = 4286578688;
    }
    if (lim <= t00 + 12) {
        cb[12] = 4286578688;
    }
    if (lim <= t00 + 13) {
        cb[13] = 4286578688;
    }
    if (lim <= t00 + 14) {
        cb[14] = 4286578688;
    }
    if (lim <= t00 + 15) {
        cb[15] = 4286578688;
    }
    #pragma unroll 1
    for (int j = 0; j < num_chunks; j++) {
        int t0_0 = ((j + 1) * 32 + w) * 16;
        long long p_1 = cbase + (long long)t0_0 * nq64;
        nb[0] = 4286578688;
        if (lim > t0_0) {
            nb[0] = S[p_1];
        }
        nb[1] = 4286578688;
        if (lim > t0_0 + 1) {
            nb[1] = S[p_1 + nq64];
        }
        nb[2] = 4286578688;
        if (lim > t0_0 + 2) {
            nb[2] = S[p_1 + 2 * nq64];
        }
        nb[3] = 4286578688;
        if (lim > t0_0 + 3) {
            nb[3] = S[p_1 + 3 * nq64];
        }
        nb[4] = 4286578688;
        if (lim > t0_0 + 4) {
            nb[4] = S[p_1 + 4 * nq64];
        }
        nb[5] = 4286578688;
        if (lim > t0_0 + 5) {
            nb[5] = S[p_1 + 5 * nq64];
        }
        nb[6] = 4286578688;
        if (lim > t0_0 + 6) {
            nb[6] = S[p_1 + 6 * nq64];
        }
        nb[7] = 4286578688;
        if (lim > t0_0 + 7) {
            nb[7] = S[p_1 + 7 * nq64];
        }
        nb[8] = 4286578688;
        if (lim > t0_0 + 8) {
            nb[8] = S[p_1 + 8 * nq64];
        }
        nb[9] = 4286578688;
        if (lim > t0_0 + 9) {
            nb[9] = S[p_1 + 9 * nq64];
        }
        nb[10] = 4286578688;
        if (lim > t0_0 + 10) {
            nb[10] = S[p_1 + 10 * nq64];
        }
        nb[11] = 4286578688;
        if (lim > t0_0 + 11) {
            nb[11] = S[p_1 + 11 * nq64];
        }
        nb[12] = 4286578688;
        if (lim > t0_0 + 12) {
            nb[12] = S[p_1 + 12 * nq64];
        }
        nb[13] = 4286578688;
        if (lim > t0_0 + 13) {
            nb[13] = S[p_1 + 13 * nq64];
        }
        nb[14] = 4286578688;
        if (lim > t0_0 + 14) {
            nb[14] = S[p_1 + 14 * nq64];
        }
        nb[15] = 4286578688;
        if (lim > t0_0 + 15) {
            nb[15] = S[p_1 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_2 = (j * 32 + w) * 16;
        float sc = __uint_as_float(cb[0]);
        float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
        sc = _fmax_0;
        float _min_0 = fminf(sc, 1.7014118346046923e+38f);
        sc = _min_0;
        sc = sc;
        float sc_3 = sc;
        unsigned int key = __as_u32(sc_3) & 4294966784u | (unsigned int)t0_2;
        kb[0] = __uint_as_float(key);
        float sc_4 = __uint_as_float(cb[1]);
        float _fmax_1 = fmaxf(sc_4, -1.7014118346046923e+38f);
        sc_4 = _fmax_1;
        float _min_1 = fminf(sc_4, 1.7014118346046923e+38f);
        sc_4 = _min_1;
        sc_4 = sc_4;
        float sc_5 = sc_4;
        unsigned int key_6 = __as_u32(sc_5) & 4294966784u | (unsigned int)(t0_2 + 1);
        kb[1] = __uint_as_float(key_6);
        float sc_7 = __uint_as_float(cb[2]);
        float _fmax_2 = fmaxf(sc_7, -1.7014118346046923e+38f);
        sc_7 = _fmax_2;
        float _min_2 = fminf(sc_7, 1.7014118346046923e+38f);
        sc_7 = _min_2;
        sc_7 = sc_7;
        float sc_8 = sc_7;
        unsigned int key_9 = __as_u32(sc_8) & 4294966784u | (unsigned int)(t0_2 + 2);
        kb[2] = __uint_as_float(key_9);
        float sc_10 = __uint_as_float(cb[3]);
        float _fmax_3 = fmaxf(sc_10, -1.7014118346046923e+38f);
        sc_10 = _fmax_3;
        float _min_3 = fminf(sc_10, 1.7014118346046923e+38f);
        sc_10 = _min_3;
        sc_10 = sc_10;
        float sc_11 = sc_10;
        unsigned int key_12 = __as_u32(sc_11) & 4294966784u | (unsigned int)(t0_2 + 3);
        kb[3] = __uint_as_float(key_12);
        float sc_13 = __uint_as_float(cb[4]);
        float _fmax_4 = fmaxf(sc_13, -1.7014118346046923e+38f);
        sc_13 = _fmax_4;
        float _min_4 = fminf(sc_13, 1.7014118346046923e+38f);
        sc_13 = _min_4;
        sc_13 = sc_13;
        float sc_14 = sc_13;
        unsigned int key_15 = __as_u32(sc_14) & 4294966784u | (unsigned int)(t0_2 + 4);
        kb[4] = __uint_as_float(key_15);
        float sc_16 = __uint_as_float(cb[5]);
        float _fmax_5 = fmaxf(sc_16, -1.7014118346046923e+38f);
        sc_16 = _fmax_5;
        float _min_5 = fminf(sc_16, 1.7014118346046923e+38f);
        sc_16 = _min_5;
        sc_16 = sc_16;
        float sc_17 = sc_16;
        unsigned int key_18 = __as_u32(sc_17) & 4294966784u | (unsigned int)(t0_2 + 5);
        kb[5] = __uint_as_float(key_18);
        float sc_19 = __uint_as_float(cb[6]);
        float _fmax_6 = fmaxf(sc_19, -1.7014118346046923e+38f);
        sc_19 = _fmax_6;
        float _min_6 = fminf(sc_19, 1.7014118346046923e+38f);
        sc_19 = _min_6;
        sc_19 = sc_19;
        float sc_20 = sc_19;
        unsigned int key_21 = __as_u32(sc_20) & 4294966784u | (unsigned int)(t0_2 + 6);
        kb[6] = __uint_as_float(key_21);
        float sc_22 = __uint_as_float(cb[7]);
        float _fmax_7 = fmaxf(sc_22, -1.7014118346046923e+38f);
        sc_22 = _fmax_7;
        float _min_7 = fminf(sc_22, 1.7014118346046923e+38f);
        sc_22 = _min_7;
        sc_22 = sc_22;
        float sc_23 = sc_22;
        unsigned int key_24 = __as_u32(sc_23) & 4294966784u | (unsigned int)(t0_2 + 7);
        kb[7] = __uint_as_float(key_24);
        float sc_25 = __uint_as_float(cb[8]);
        float _fmax_8 = fmaxf(sc_25, -1.7014118346046923e+38f);
        sc_25 = _fmax_8;
        float _min_8 = fminf(sc_25, 1.7014118346046923e+38f);
        sc_25 = _min_8;
        sc_25 = sc_25;
        float sc_26 = sc_25;
        unsigned int key_27 = __as_u32(sc_26) & 4294966784u | (unsigned int)(t0_2 + 8);
        kb[8] = __uint_as_float(key_27);
        float sc_28 = __uint_as_float(cb[9]);
        float _fmax_9 = fmaxf(sc_28, -1.7014118346046923e+38f);
        sc_28 = _fmax_9;
        float _min_9 = fminf(sc_28, 1.7014118346046923e+38f);
        sc_28 = _min_9;
        sc_28 = sc_28;
        float sc_29 = sc_28;
        unsigned int key_30 = __as_u32(sc_29) & 4294966784u | (unsigned int)(t0_2 + 9);
        kb[9] = __uint_as_float(key_30);
        float sc_31 = __uint_as_float(cb[10]);
        float _fmax_10 = fmaxf(sc_31, -1.7014118346046923e+38f);
        sc_31 = _fmax_10;
        float _min_10 = fminf(sc_31, 1.7014118346046923e+38f);
        sc_31 = _min_10;
        sc_31 = sc_31;
        float sc_32 = sc_31;
        unsigned int key_33 = __as_u32(sc_32) & 4294966784u | (unsigned int)(t0_2 + 10);
        kb[10] = __uint_as_float(key_33);
        float sc_34 = __uint_as_float(cb[11]);
        float _fmax_11 = fmaxf(sc_34, -1.7014118346046923e+38f);
        sc_34 = _fmax_11;
        float _min_11 = fminf(sc_34, 1.7014118346046923e+38f);
        sc_34 = _min_11;
        sc_34 = sc_34;
        float sc_35 = sc_34;
        unsigned int key_36 = __as_u32(sc_35) & 4294966784u | (unsigned int)(t0_2 + 11);
        kb[11] = __uint_as_float(key_36);
        float sc_37 = __uint_as_float(cb[12]);
        float _fmax_12 = fmaxf(sc_37, -1.7014118346046923e+38f);
        sc_37 = _fmax_12;
        float _min_12 = fminf(sc_37, 1.7014118346046923e+38f);
        sc_37 = _min_12;
        sc_37 = sc_37;
        float sc_38 = sc_37;
        unsigned int key_39 = __as_u32(sc_38) & 4294966784u | (unsigned int)(t0_2 + 12);
        kb[12] = __uint_as_float(key_39);
        float sc_40 = __uint_as_float(cb[13]);
        float _fmax_13 = fmaxf(sc_40, -1.7014118346046923e+38f);
        sc_40 = _fmax_13;
        float _min_13 = fminf(sc_40, 1.7014118346046923e+38f);
        sc_40 = _min_13;
        sc_40 = sc_40;
        float sc_41 = sc_40;
        unsigned int key_42 = __as_u32(sc_41) & 4294966784u | (unsigned int)(t0_2 + 13);
        kb[13] = __uint_as_float(key_42);
        float sc_43 = __uint_as_float(cb[14]);
        float _fmax_14 = fmaxf(sc_43, -1.7014118346046923e+38f);
        sc_43 = _fmax_14;
        float _min_14 = fminf(sc_43, 1.7014118346046923e+38f);
        sc_43 = _min_14;
        sc_43 = sc_43;
        float sc_44 = sc_43;
        unsigned int key_45 = __as_u32(sc_44) & 4294966784u | (unsigned int)(t0_2 + 14);
        kb[14] = __uint_as_float(key_45);
        float sc_46 = __uint_as_float(cb[15]);
        float _fmax_15 = fmaxf(sc_46, -1.7014118346046923e+38f);
        sc_46 = _fmax_15;
        float _min_15 = fminf(sc_46, 1.7014118346046923e+38f);
        sc_46 = _min_15;
        sc_46 = sc_46;
        float sc_47 = sc_46;
        unsigned int key_48 = __as_u32(sc_47) & 4294966784u | (unsigned int)(t0_2 + 15);
        kb[15] = __uint_as_float(key_48);
        int f = 0;
        if (t0_2 < fb || t0_2 + 16 > lim - fe && t0_2 < lim) {
            f = 1;
        }
        int fch1 = f;
        if (fch1 != 0) {
            int f_0 = 0;
            if (t0_2 < fb || t0_2 >= lim - fe && lim > t0_2) {
                f_0 = 1;
            }
            if (f_0 != 0) {
                kb[0] = __uint_as_float(2139094528 | (unsigned int)t0_2);
            }
            int f_1 = 0;
            if (t0_2 + 1 < fb || t0_2 + 1 >= lim - fe && lim > t0_2 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                kb[1] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 1));
            }
            int f_2 = 0;
            if (t0_2 + 2 < fb || t0_2 + 2 >= lim - fe && lim > t0_2 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                kb[2] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 2));
            }
            int f_3 = 0;
            if (t0_2 + 3 < fb || t0_2 + 3 >= lim - fe && lim > t0_2 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                kb[3] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 3));
            }
            int f_4 = 0;
            if (t0_2 + 4 < fb || t0_2 + 4 >= lim - fe && lim > t0_2 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                kb[4] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 4));
            }
            int f_5 = 0;
            if (t0_2 + 5 < fb || t0_2 + 5 >= lim - fe && lim > t0_2 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                kb[5] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 5));
            }
            int f_6 = 0;
            if (t0_2 + 6 < fb || t0_2 + 6 >= lim - fe && lim > t0_2 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                kb[6] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 6));
            }
            int f_7 = 0;
            if (t0_2 + 7 < fb || t0_2 + 7 >= lim - fe && lim > t0_2 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                kb[7] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 7));
            }
            int f_8 = 0;
            if (t0_2 + 8 < fb || t0_2 + 8 >= lim - fe && lim > t0_2 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                kb[8] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 8));
            }
            int f_9 = 0;
            if (t0_2 + 9 < fb || t0_2 + 9 >= lim - fe && lim > t0_2 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                kb[9] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 9));
            }
            int f_10 = 0;
            if (t0_2 + 10 < fb || t0_2 + 10 >= lim - fe && lim > t0_2 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                kb[10] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 10));
            }
            int f_11 = 0;
            if (t0_2 + 11 < fb || t0_2 + 11 >= lim - fe && lim > t0_2 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                kb[11] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 11));
            }
            int f_12 = 0;
            if (t0_2 + 12 < fb || t0_2 + 12 >= lim - fe && lim > t0_2 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                kb[12] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 12));
            }
            int f_13 = 0;
            if (t0_2 + 13 < fb || t0_2 + 13 >= lim - fe && lim > t0_2 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                kb[13] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 13));
            }
            int f_14 = 0;
            if (t0_2 + 14 < fb || t0_2 + 14 >= lim - fe && lim > t0_2 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                kb[14] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 14));
            }
            int f_15 = 0;
            if (t0_2 + 15 < fb || t0_2 + 15 >= lim - fe && lim > t0_2 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                kb[15] = __uint_as_float(2139094528 | (unsigned int)(t0_2 + 15));
            }
        }
        float _fmax_16 = fmaxf(kb[0], kb[13]);
        float hi = _fmax_16;
        float _min_16 = fminf(kb[0], kb[13]);
        float lo = _min_16;
        kb[0] = hi;
        kb[13] = lo;
        float _fmax_17 = fmaxf(kb[1], kb[12]);
        float hi_49 = _fmax_17;
        float _min_17 = fminf(kb[1], kb[12]);
        float lo_50 = _min_17;
        kb[1] = hi_49;
        kb[12] = lo_50;
        float _fmax_18 = fmaxf(kb[2], kb[15]);
        float hi_51 = _fmax_18;
        float _min_18 = fminf(kb[2], kb[15]);
        float lo_52 = _min_18;
        kb[2] = hi_51;
        kb[15] = lo_52;
        float _fmax_19 = fmaxf(kb[3], kb[14]);
        float hi_53 = _fmax_19;
        float _min_19 = fminf(kb[3], kb[14]);
        float lo_54 = _min_19;
        kb[3] = hi_53;
        kb[14] = lo_54;
        float _fmax_20 = fmaxf(kb[4], kb[8]);
        float hi_55 = _fmax_20;
        float _min_20 = fminf(kb[4], kb[8]);
        float lo_56 = _min_20;
        kb[4] = hi_55;
        kb[8] = lo_56;
        float _fmax_21 = fmaxf(kb[5], kb[6]);
        float hi_57 = _fmax_21;
        float _min_21 = fminf(kb[5], kb[6]);
        float lo_58 = _min_21;
        kb[5] = hi_57;
        kb[6] = lo_58;
        float _fmax_22 = fmaxf(kb[7], kb[11]);
        float hi_59 = _fmax_22;
        float _min_22 = fminf(kb[7], kb[11]);
        float lo_60 = _min_22;
        kb[7] = hi_59;
        kb[11] = lo_60;
        float _fmax_23 = fmaxf(kb[9], kb[10]);
        float hi_61 = _fmax_23;
        float _min_23 = fminf(kb[9], kb[10]);
        float lo_62 = _min_23;
        kb[9] = hi_61;
        kb[10] = lo_62;
        float _fmax_24 = fmaxf(kb[0], kb[5]);
        float hi_63 = _fmax_24;
        float _min_24 = fminf(kb[0], kb[5]);
        float lo_64 = _min_24;
        kb[0] = hi_63;
        kb[5] = lo_64;
        float _fmax_25 = fmaxf(kb[1], kb[7]);
        float hi_65 = _fmax_25;
        float _min_25 = fminf(kb[1], kb[7]);
        float lo_66 = _min_25;
        kb[1] = hi_65;
        kb[7] = lo_66;
        float _fmax_26 = fmaxf(kb[2], kb[9]);
        float hi_67 = _fmax_26;
        float _min_26 = fminf(kb[2], kb[9]);
        float lo_68 = _min_26;
        kb[2] = hi_67;
        kb[9] = lo_68;
        float _fmax_27 = fmaxf(kb[3], kb[4]);
        float hi_69 = _fmax_27;
        float _min_27 = fminf(kb[3], kb[4]);
        float lo_70 = _min_27;
        kb[3] = hi_69;
        kb[4] = lo_70;
        float _fmax_28 = fmaxf(kb[6], kb[13]);
        float hi_71 = _fmax_28;
        float _min_28 = fminf(kb[6], kb[13]);
        float lo_72 = _min_28;
        kb[6] = hi_71;
        kb[13] = lo_72;
        float _fmax_29 = fmaxf(kb[8], kb[14]);
        float hi_73 = _fmax_29;
        float _min_29 = fminf(kb[8], kb[14]);
        float lo_74 = _min_29;
        kb[8] = hi_73;
        kb[14] = lo_74;
        float _fmax_30 = fmaxf(kb[10], kb[15]);
        float hi_75 = _fmax_30;
        float _min_30 = fminf(kb[10], kb[15]);
        float lo_76 = _min_30;
        kb[10] = hi_75;
        kb[15] = lo_76;
        float _fmax_31 = fmaxf(kb[11], kb[12]);
        float hi_77 = _fmax_31;
        float _min_31 = fminf(kb[11], kb[12]);
        float lo_78 = _min_31;
        kb[11] = hi_77;
        kb[12] = lo_78;
        float _fmax_32 = fmaxf(kb[0], kb[1]);
        float hi_79 = _fmax_32;
        float _min_32 = fminf(kb[0], kb[1]);
        float lo_80 = _min_32;
        kb[0] = hi_79;
        kb[1] = lo_80;
        float _fmax_33 = fmaxf(kb[2], kb[3]);
        float hi_81 = _fmax_33;
        float _min_33 = fminf(kb[2], kb[3]);
        float lo_82 = _min_33;
        kb[2] = hi_81;
        kb[3] = lo_82;
        float _fmax_34 = fmaxf(kb[4], kb[5]);
        float hi_83 = _fmax_34;
        float _min_34 = fminf(kb[4], kb[5]);
        float lo_84 = _min_34;
        kb[4] = hi_83;
        kb[5] = lo_84;
        float _fmax_35 = fmaxf(kb[6], kb[8]);
        float hi_85 = _fmax_35;
        float _min_35 = fminf(kb[6], kb[8]);
        float lo_86 = _min_35;
        kb[6] = hi_85;
        kb[8] = lo_86;
        float _fmax_36 = fmaxf(kb[7], kb[9]);
        float hi_87 = _fmax_36;
        float _min_36 = fminf(kb[7], kb[9]);
        float lo_88 = _min_36;
        kb[7] = hi_87;
        kb[9] = lo_88;
        float _fmax_37 = fmaxf(kb[10], kb[11]);
        float hi_89 = _fmax_37;
        float _min_37 = fminf(kb[10], kb[11]);
        float lo_90 = _min_37;
        kb[10] = hi_89;
        kb[11] = lo_90;
        float _fmax_38 = fmaxf(kb[12], kb[13]);
        float hi_91 = _fmax_38;
        float _min_38 = fminf(kb[12], kb[13]);
        float lo_92 = _min_38;
        kb[12] = hi_91;
        kb[13] = lo_92;
        float _fmax_39 = fmaxf(kb[14], kb[15]);
        float hi_93 = _fmax_39;
        float _min_39 = fminf(kb[14], kb[15]);
        float lo_94 = _min_39;
        kb[14] = hi_93;
        kb[15] = lo_94;
        float _fmax_40 = fmaxf(kb[0], kb[2]);
        float hi_95 = _fmax_40;
        float _min_40 = fminf(kb[0], kb[2]);
        float lo_96 = _min_40;
        kb[0] = hi_95;
        kb[2] = lo_96;
        float _fmax_41 = fmaxf(kb[1], kb[3]);
        float hi_97 = _fmax_41;
        float _min_41 = fminf(kb[1], kb[3]);
        float lo_98 = _min_41;
        kb[1] = hi_97;
        kb[3] = lo_98;
        float _fmax_42 = fmaxf(kb[4], kb[10]);
        float hi_99 = _fmax_42;
        float _min_42 = fminf(kb[4], kb[10]);
        float lo_100 = _min_42;
        kb[4] = hi_99;
        kb[10] = lo_100;
        float _fmax_43 = fmaxf(kb[5], kb[11]);
        float hi_101 = _fmax_43;
        float _min_43 = fminf(kb[5], kb[11]);
        float lo_102 = _min_43;
        kb[5] = hi_101;
        kb[11] = lo_102;
        float _fmax_44 = fmaxf(kb[6], kb[7]);
        float hi_103 = _fmax_44;
        float _min_44 = fminf(kb[6], kb[7]);
        float lo_104 = _min_44;
        kb[6] = hi_103;
        kb[7] = lo_104;
        float _fmax_45 = fmaxf(kb[8], kb[9]);
        float hi_105 = _fmax_45;
        float _min_45 = fminf(kb[8], kb[9]);
        float lo_106 = _min_45;
        kb[8] = hi_105;
        kb[9] = lo_106;
        float _fmax_46 = fmaxf(kb[12], kb[14]);
        float hi_107 = _fmax_46;
        float _min_46 = fminf(kb[12], kb[14]);
        float lo_108 = _min_46;
        kb[12] = hi_107;
        kb[14] = lo_108;
        float _fmax_47 = fmaxf(kb[13], kb[15]);
        float hi_109 = _fmax_47;
        float _min_47 = fminf(kb[13], kb[15]);
        float lo_110 = _min_47;
        kb[13] = hi_109;
        kb[15] = lo_110;
        float _fmax_48 = fmaxf(kb[1], kb[2]);
        float hi_111 = _fmax_48;
        float _min_48 = fminf(kb[1], kb[2]);
        float lo_112 = _min_48;
        kb[1] = hi_111;
        kb[2] = lo_112;
        float _fmax_49 = fmaxf(kb[3], kb[12]);
        float hi_113 = _fmax_49;
        float _min_49 = fminf(kb[3], kb[12]);
        float lo_114 = _min_49;
        kb[3] = hi_113;
        kb[12] = lo_114;
        float _fmax_50 = fmaxf(kb[4], kb[6]);
        float hi_115 = _fmax_50;
        float _min_50 = fminf(kb[4], kb[6]);
        float lo_116 = _min_50;
        kb[4] = hi_115;
        kb[6] = lo_116;
        float _fmax_51 = fmaxf(kb[5], kb[7]);
        float hi_117 = _fmax_51;
        float _min_51 = fminf(kb[5], kb[7]);
        float lo_118 = _min_51;
        kb[5] = hi_117;
        kb[7] = lo_118;
        float _fmax_52 = fmaxf(kb[8], kb[10]);
        float hi_119 = _fmax_52;
        float _min_52 = fminf(kb[8], kb[10]);
        float lo_120 = _min_52;
        kb[8] = hi_119;
        kb[10] = lo_120;
        float _fmax_53 = fmaxf(kb[9], kb[11]);
        float hi_121 = _fmax_53;
        float _min_53 = fminf(kb[9], kb[11]);
        float lo_122 = _min_53;
        kb[9] = hi_121;
        kb[11] = lo_122;
        float _fmax_54 = fmaxf(kb[13], kb[14]);
        float hi_123 = _fmax_54;
        float _min_54 = fminf(kb[13], kb[14]);
        float lo_124 = _min_54;
        kb[13] = hi_123;
        kb[14] = lo_124;
        float _fmax_55 = fmaxf(kb[1], kb[4]);
        float hi_125 = _fmax_55;
        float _min_55 = fminf(kb[1], kb[4]);
        float lo_126 = _min_55;
        kb[1] = hi_125;
        kb[4] = lo_126;
        float _fmax_56 = fmaxf(kb[2], kb[6]);
        float hi_127 = _fmax_56;
        float _min_56 = fminf(kb[2], kb[6]);
        float lo_128 = _min_56;
        kb[2] = hi_127;
        kb[6] = lo_128;
        float _fmax_57 = fmaxf(kb[5], kb[8]);
        float hi_129 = _fmax_57;
        float _min_57 = fminf(kb[5], kb[8]);
        float lo_130 = _min_57;
        kb[5] = hi_129;
        kb[8] = lo_130;
        float _fmax_58 = fmaxf(kb[7], kb[10]);
        float hi_131 = _fmax_58;
        float _min_58 = fminf(kb[7], kb[10]);
        float lo_132 = _min_58;
        kb[7] = hi_131;
        kb[10] = lo_132;
        float _fmax_59 = fmaxf(kb[9], kb[13]);
        float hi_133 = _fmax_59;
        float _min_59 = fminf(kb[9], kb[13]);
        float lo_134 = _min_59;
        kb[9] = hi_133;
        kb[13] = lo_134;
        float _fmax_60 = fmaxf(kb[11], kb[14]);
        float hi_135 = _fmax_60;
        float _min_60 = fminf(kb[11], kb[14]);
        float lo_136 = _min_60;
        kb[11] = hi_135;
        kb[14] = lo_136;
        float _fmax_61 = fmaxf(kb[2], kb[4]);
        float hi_137 = _fmax_61;
        float _min_61 = fminf(kb[2], kb[4]);
        float lo_138 = _min_61;
        kb[2] = hi_137;
        kb[4] = lo_138;
        float _fmax_62 = fmaxf(kb[3], kb[6]);
        float hi_139 = _fmax_62;
        float _min_62 = fminf(kb[3], kb[6]);
        float lo_140 = _min_62;
        kb[3] = hi_139;
        kb[6] = lo_140;
        float _fmax_63 = fmaxf(kb[9], kb[12]);
        float hi_141 = _fmax_63;
        float _min_63 = fminf(kb[9], kb[12]);
        float lo_142 = _min_63;
        kb[9] = hi_141;
        kb[12] = lo_142;
        float _fmax_64 = fmaxf(kb[11], kb[13]);
        float hi_143 = _fmax_64;
        float _min_64 = fminf(kb[11], kb[13]);
        float lo_144 = _min_64;
        kb[11] = hi_143;
        kb[13] = lo_144;
        float _fmax_65 = fmaxf(kb[3], kb[5]);
        float hi_145 = _fmax_65;
        float _min_65 = fminf(kb[3], kb[5]);
        float lo_146 = _min_65;
        kb[3] = hi_145;
        kb[5] = lo_146;
        float _fmax_66 = fmaxf(kb[6], kb[8]);
        float hi_147 = _fmax_66;
        float _min_66 = fminf(kb[6], kb[8]);
        float lo_148 = _min_66;
        kb[6] = hi_147;
        kb[8] = lo_148;
        float _fmax_67 = fmaxf(kb[7], kb[9]);
        float hi_149 = _fmax_67;
        float _min_67 = fminf(kb[7], kb[9]);
        float lo_150 = _min_67;
        kb[7] = hi_149;
        kb[9] = lo_150;
        float _fmax_68 = fmaxf(kb[10], kb[12]);
        float hi_151 = _fmax_68;
        float _min_68 = fminf(kb[10], kb[12]);
        float lo_152 = _min_68;
        kb[10] = hi_151;
        kb[12] = lo_152;
        float _fmax_69 = fmaxf(kb[3], kb[4]);
        float hi_153 = _fmax_69;
        float _min_69 = fminf(kb[3], kb[4]);
        float lo_154 = _min_69;
        kb[3] = hi_153;
        kb[4] = lo_154;
        float _fmax_70 = fmaxf(kb[5], kb[6]);
        float hi_155 = _fmax_70;
        float _min_70 = fminf(kb[5], kb[6]);
        float lo_156 = _min_70;
        kb[5] = hi_155;
        kb[6] = lo_156;
        float _fmax_71 = fmaxf(kb[7], kb[8]);
        float hi_157 = _fmax_71;
        float _min_71 = fminf(kb[7], kb[8]);
        float lo_158 = _min_71;
        kb[7] = hi_157;
        kb[8] = lo_158;
        float _fmax_72 = fmaxf(kb[9], kb[10]);
        float hi_159 = _fmax_72;
        float _min_72 = fminf(kb[9], kb[10]);
        float lo_160 = _min_72;
        kb[9] = hi_159;
        kb[10] = lo_160;
        float _fmax_73 = fmaxf(kb[11], kb[12]);
        float hi_161 = _fmax_73;
        float _min_73 = fminf(kb[11], kb[12]);
        float lo_162 = _min_73;
        kb[11] = hi_161;
        kb[12] = lo_162;
        float _fmax_74 = fmaxf(kb[6], kb[7]);
        float hi_163 = _fmax_74;
        float _min_74 = fminf(kb[6], kb[7]);
        float lo_164 = _min_74;
        kb[6] = hi_163;
        kb[7] = lo_164;
        float _fmax_75 = fmaxf(kb[8], kb[9]);
        float hi_165 = _fmax_75;
        float _min_75 = fminf(kb[8], kb[9]);
        float lo_166 = _min_75;
        kb[8] = hi_165;
        kb[9] = lo_166;
        float r = rej;
        float _fmax_76 = fmaxf(a[0], kb[15]);
        float hi_167 = _fmax_76;
        float _min_76 = fminf(a[0], kb[15]);
        float lo_168 = _min_76;
        a[0] = hi_167;
        float _fmax_77 = fmaxf(r, lo_168);
        r = _fmax_77;
        float _fmax_78 = fmaxf(a[1], kb[14]);
        float hi_169 = _fmax_78;
        float _min_77 = fminf(a[1], kb[14]);
        float lo_170 = _min_77;
        a[1] = hi_169;
        float _fmax_79 = fmaxf(r, lo_170);
        r = _fmax_79;
        float _fmax_80 = fmaxf(a[2], kb[13]);
        float hi_171 = _fmax_80;
        float _min_78 = fminf(a[2], kb[13]);
        float lo_172 = _min_78;
        a[2] = hi_171;
        float _fmax_81 = fmaxf(r, lo_172);
        r = _fmax_81;
        float _fmax_82 = fmaxf(a[3], kb[12]);
        float hi_173 = _fmax_82;
        float _min_79 = fminf(a[3], kb[12]);
        float lo_174 = _min_79;
        a[3] = hi_173;
        float _fmax_83 = fmaxf(r, lo_174);
        r = _fmax_83;
        float _fmax_84 = fmaxf(a[4], kb[11]);
        float hi_175 = _fmax_84;
        float _min_80 = fminf(a[4], kb[11]);
        float lo_176 = _min_80;
        a[4] = hi_175;
        float _fmax_85 = fmaxf(r, lo_176);
        r = _fmax_85;
        float _fmax_86 = fmaxf(a[5], kb[10]);
        float hi_177 = _fmax_86;
        float _min_81 = fminf(a[5], kb[10]);
        float lo_178 = _min_81;
        a[5] = hi_177;
        float _fmax_87 = fmaxf(r, lo_178);
        r = _fmax_87;
        float _fmax_88 = fmaxf(a[6], kb[9]);
        float hi_179 = _fmax_88;
        float _min_82 = fminf(a[6], kb[9]);
        float lo_180 = _min_82;
        a[6] = hi_179;
        float _fmax_89 = fmaxf(r, lo_180);
        r = _fmax_89;
        float _fmax_90 = fmaxf(a[7], kb[8]);
        float hi_181 = _fmax_90;
        float _min_83 = fminf(a[7], kb[8]);
        float lo_182 = _min_83;
        a[7] = hi_181;
        float _fmax_91 = fmaxf(r, lo_182);
        r = _fmax_91;
        float _fmax_92 = fmaxf(a[8], kb[7]);
        float hi_183 = _fmax_92;
        float _min_84 = fminf(a[8], kb[7]);
        float lo_184 = _min_84;
        a[8] = hi_183;
        float _fmax_93 = fmaxf(r, lo_184);
        r = _fmax_93;
        float _fmax_94 = fmaxf(a[9], kb[6]);
        float hi_185 = _fmax_94;
        float _min_85 = fminf(a[9], kb[6]);
        float lo_186 = _min_85;
        a[9] = hi_185;
        float _fmax_95 = fmaxf(r, lo_186);
        r = _fmax_95;
        float _fmax_96 = fmaxf(a[10], kb[5]);
        float hi_187 = _fmax_96;
        float _min_86 = fminf(a[10], kb[5]);
        float lo_188 = _min_86;
        a[10] = hi_187;
        float _fmax_97 = fmaxf(r, lo_188);
        r = _fmax_97;
        float _fmax_98 = fmaxf(a[11], kb[4]);
        float hi_189 = _fmax_98;
        float _min_87 = fminf(a[11], kb[4]);
        float lo_190 = _min_87;
        a[11] = hi_189;
        float _fmax_99 = fmaxf(r, lo_190);
        r = _fmax_99;
        float _fmax_100 = fmaxf(a[12], kb[3]);
        float hi_191 = _fmax_100;
        float _min_88 = fminf(a[12], kb[3]);
        float lo_192 = _min_88;
        a[12] = hi_191;
        float _fmax_101 = fmaxf(r, lo_192);
        r = _fmax_101;
        float _fmax_102 = fmaxf(a[13], kb[2]);
        float hi_193 = _fmax_102;
        float _min_89 = fminf(a[13], kb[2]);
        float lo_194 = _min_89;
        a[13] = hi_193;
        float _fmax_103 = fmaxf(r, lo_194);
        r = _fmax_103;
        float _fmax_104 = fmaxf(a[14], kb[1]);
        float hi_195 = _fmax_104;
        float _min_90 = fminf(a[14], kb[1]);
        float lo_196 = _min_90;
        a[14] = hi_195;
        float _fmax_105 = fmaxf(r, lo_196);
        r = _fmax_105;
        float _fmax_106 = fmaxf(a[15], kb[0]);
        float hi_197 = _fmax_106;
        float _min_91 = fminf(a[15], kb[0]);
        float lo_198 = _min_91;
        a[15] = hi_197;
        float _fmax_107 = fmaxf(r, lo_198);
        r = _fmax_107;
        float _fmax_108 = fmaxf(a[0], a[8]);
        float hi_199 = _fmax_108;
        float _min_92 = fminf(a[0], a[8]);
        float lo_200 = _min_92;
        a[0] = hi_199;
        a[8] = lo_200;
        float _fmax_109 = fmaxf(a[1], a[9]);
        float hi_201 = _fmax_109;
        float _min_93 = fminf(a[1], a[9]);
        float lo_202 = _min_93;
        a[1] = hi_201;
        a[9] = lo_202;
        float _fmax_110 = fmaxf(a[2], a[10]);
        float hi_203 = _fmax_110;
        float _min_94 = fminf(a[2], a[10]);
        float lo_204 = _min_94;
        a[2] = hi_203;
        a[10] = lo_204;
        float _fmax_111 = fmaxf(a[3], a[11]);
        float hi_205 = _fmax_111;
        float _min_95 = fminf(a[3], a[11]);
        float lo_206 = _min_95;
        a[3] = hi_205;
        a[11] = lo_206;
        float _fmax_112 = fmaxf(a[4], a[12]);
        float hi_207 = _fmax_112;
        float _min_96 = fminf(a[4], a[12]);
        float lo_208 = _min_96;
        a[4] = hi_207;
        a[12] = lo_208;
        float _fmax_113 = fmaxf(a[5], a[13]);
        float hi_209 = _fmax_113;
        float _min_97 = fminf(a[5], a[13]);
        float lo_210 = _min_97;
        a[5] = hi_209;
        a[13] = lo_210;
        float _fmax_114 = fmaxf(a[6], a[14]);
        float hi_211 = _fmax_114;
        float _min_98 = fminf(a[6], a[14]);
        float lo_212 = _min_98;
        a[6] = hi_211;
        a[14] = lo_212;
        float _fmax_115 = fmaxf(a[7], a[15]);
        float hi_213 = _fmax_115;
        float _min_99 = fminf(a[7], a[15]);
        float lo_214 = _min_99;
        a[7] = hi_213;
        a[15] = lo_214;
        float _fmax_116 = fmaxf(a[0], a[4]);
        float hi_215 = _fmax_116;
        float _min_100 = fminf(a[0], a[4]);
        float lo_216 = _min_100;
        a[0] = hi_215;
        a[4] = lo_216;
        float _fmax_117 = fmaxf(a[1], a[5]);
        float hi_217 = _fmax_117;
        float _min_101 = fminf(a[1], a[5]);
        float lo_218 = _min_101;
        a[1] = hi_217;
        a[5] = lo_218;
        float _fmax_118 = fmaxf(a[2], a[6]);
        float hi_219 = _fmax_118;
        float _min_102 = fminf(a[2], a[6]);
        float lo_220 = _min_102;
        a[2] = hi_219;
        a[6] = lo_220;
        float _fmax_119 = fmaxf(a[3], a[7]);
        float hi_221 = _fmax_119;
        float _min_103 = fminf(a[3], a[7]);
        float lo_222 = _min_103;
        a[3] = hi_221;
        a[7] = lo_222;
        float _fmax_120 = fmaxf(a[8], a[12]);
        float hi_223 = _fmax_120;
        float _min_104 = fminf(a[8], a[12]);
        float lo_224 = _min_104;
        a[8] = hi_223;
        a[12] = lo_224;
        float _fmax_121 = fmaxf(a[9], a[13]);
        float hi_225 = _fmax_121;
        float _min_105 = fminf(a[9], a[13]);
        float lo_226 = _min_105;
        a[9] = hi_225;
        a[13] = lo_226;
        float _fmax_122 = fmaxf(a[10], a[14]);
        float hi_227 = _fmax_122;
        float _min_106 = fminf(a[10], a[14]);
        float lo_228 = _min_106;
        a[10] = hi_227;
        a[14] = lo_228;
        float _fmax_123 = fmaxf(a[11], a[15]);
        float hi_229 = _fmax_123;
        float _min_107 = fminf(a[11], a[15]);
        float lo_230 = _min_107;
        a[11] = hi_229;
        a[15] = lo_230;
        float _fmax_124 = fmaxf(a[0], a[2]);
        float hi_231 = _fmax_124;
        float _min_108 = fminf(a[0], a[2]);
        float lo_232 = _min_108;
        a[0] = hi_231;
        a[2] = lo_232;
        float _fmax_125 = fmaxf(a[1], a[3]);
        float hi_233 = _fmax_125;
        float _min_109 = fminf(a[1], a[3]);
        float lo_234 = _min_109;
        a[1] = hi_233;
        a[3] = lo_234;
        float _fmax_126 = fmaxf(a[4], a[6]);
        float hi_235 = _fmax_126;
        float _min_110 = fminf(a[4], a[6]);
        float lo_236 = _min_110;
        a[4] = hi_235;
        a[6] = lo_236;
        float _fmax_127 = fmaxf(a[5], a[7]);
        float hi_237 = _fmax_127;
        float _min_111 = fminf(a[5], a[7]);
        float lo_238 = _min_111;
        a[5] = hi_237;
        a[7] = lo_238;
        float _fmax_128 = fmaxf(a[8], a[10]);
        float hi_239 = _fmax_128;
        float _min_112 = fminf(a[8], a[10]);
        float lo_240 = _min_112;
        a[8] = hi_239;
        a[10] = lo_240;
        float _fmax_129 = fmaxf(a[9], a[11]);
        float hi_241 = _fmax_129;
        float _min_113 = fminf(a[9], a[11]);
        float lo_242 = _min_113;
        a[9] = hi_241;
        a[11] = lo_242;
        float _fmax_130 = fmaxf(a[12], a[14]);
        float hi_243 = _fmax_130;
        float _min_114 = fminf(a[12], a[14]);
        float lo_244 = _min_114;
        a[12] = hi_243;
        a[14] = lo_244;
        float _fmax_131 = fmaxf(a[13], a[15]);
        float hi_245 = _fmax_131;
        float _min_115 = fminf(a[13], a[15]);
        float lo_246 = _min_115;
        a[13] = hi_245;
        a[15] = lo_246;
        float _fmax_132 = fmaxf(a[0], a[1]);
        float hi_247 = _fmax_132;
        float _min_116 = fminf(a[0], a[1]);
        float lo_248 = _min_116;
        a[0] = hi_247;
        a[1] = lo_248;
        float _fmax_133 = fmaxf(a[2], a[3]);
        float hi_249 = _fmax_133;
        float _min_117 = fminf(a[2], a[3]);
        float lo_250 = _min_117;
        a[2] = hi_249;
        a[3] = lo_250;
        float _fmax_134 = fmaxf(a[4], a[5]);
        float hi_251 = _fmax_134;
        float _min_118 = fminf(a[4], a[5]);
        float lo_252 = _min_118;
        a[4] = hi_251;
        a[5] = lo_252;
        float _fmax_135 = fmaxf(a[6], a[7]);
        float hi_253 = _fmax_135;
        float _min_119 = fminf(a[6], a[7]);
        float lo_254 = _min_119;
        a[6] = hi_253;
        a[7] = lo_254;
        float _fmax_136 = fmaxf(a[8], a[9]);
        float hi_255 = _fmax_136;
        float _min_120 = fminf(a[8], a[9]);
        float lo_256 = _min_120;
        a[8] = hi_255;
        a[9] = lo_256;
        float _fmax_137 = fmaxf(a[10], a[11]);
        float hi_257 = _fmax_137;
        float _min_121 = fminf(a[10], a[11]);
        float lo_258 = _min_121;
        a[10] = hi_257;
        a[11] = lo_258;
        float _fmax_138 = fmaxf(a[12], a[13]);
        float hi_259 = _fmax_138;
        float _min_122 = fminf(a[12], a[13]);
        float lo_260 = _min_122;
        a[12] = hi_259;
        a[13] = lo_260;
        float _fmax_139 = fmaxf(a[14], a[15]);
        float hi_261 = _fmax_139;
        float _min_123 = fminf(a[14], a[15]);
        float lo_262 = _min_123;
        a[14] = hi_261;
        a[15] = lo_262;
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
    int cg = g & 7;
    int sg = g >> 3;
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
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    float r_1 = neg_inf;
    float V[8];
    int s0 = (sg * 16 * 8 + cg) * 17;
    int s1 = ((sg * 16 + 8) * 8 + cg) * 17;
    float x0 = pub[s0 + ln];
    float y0 = pub[s1 + lnr];
    float _min_124 = fminf(x0, y0);
    float lo0 = _min_124;
    float _fmax_140 = fmaxf(r_1, lo0);
    r_1 = _fmax_140;
    float _fmax_141 = fmaxf(x0, y0);
    float hi0 = _fmax_141;
    float cur = hi0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float pv = _shfl_xor_0;
    float _fmax_142 = fmaxf(cur, pv);
    float hi_1 = _fmax_142;
    float _min_125 = fminf(cur, pv);
    float lo_1 = _min_125;
    cur = ((up[0] != 0) ? hi_1 : lo_1);
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_0 = _shfl_xor_1;
    float _fmax_143 = fmaxf(cur, pv_0);
    float hi_1_1 = _fmax_143;
    float _min_126 = fminf(cur, pv_0);
    float lo_2 = _min_126;
    cur = ((up[1] != 0) ? hi_1_1 : lo_2);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_3 = _shfl_xor_2;
    float _fmax_144 = fmaxf(cur, pv_3);
    float hi_4 = _fmax_144;
    float _min_127 = fminf(cur, pv_3);
    float lo_5 = _min_127;
    cur = ((up[2] != 0) ? hi_4 : lo_5);
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_6 = _shfl_xor_3;
    float _fmax_145 = fmaxf(cur, pv_6);
    float hi_7 = _fmax_145;
    float _min_128 = fminf(cur, pv_6);
    float lo_8 = _min_128;
    cur = ((up[3] != 0) ? hi_7 : lo_8);
    V[0] = cur;
    int s0_9 = ((sg * 16 + 1) * 8 + cg) * 17;
    int s1_10 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
    float x0_11 = pub[s0_9 + ln];
    float y0_12 = pub[s1_10 + lnr];
    float _min_129 = fminf(x0_11, y0_12);
    float lo0_13 = _min_129;
    float _fmax_146 = fmaxf(r_1, lo0_13);
    r_1 = _fmax_146;
    float _fmax_147 = fmaxf(x0_11, y0_12);
    float hi0_14 = _fmax_147;
    float cur_15 = hi0_14;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 8);
    float pv_16 = _shfl_xor_4;
    float _fmax_148 = fmaxf(cur_15, pv_16);
    float hi_17 = _fmax_148;
    float _min_130 = fminf(cur_15, pv_16);
    float lo_18 = _min_130;
    cur_15 = ((up[0] != 0) ? hi_17 : lo_18);
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 4);
    float pv_19 = _shfl_xor_5;
    float _fmax_149 = fmaxf(cur_15, pv_19);
    float hi_20 = _fmax_149;
    float _min_131 = fminf(cur_15, pv_19);
    float lo_21 = _min_131;
    cur_15 = ((up[1] != 0) ? hi_20 : lo_21);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 2);
    float pv_22 = _shfl_xor_6;
    float _fmax_150 = fmaxf(cur_15, pv_22);
    float hi_23 = _fmax_150;
    float _min_132 = fminf(cur_15, pv_22);
    float lo_24 = _min_132;
    cur_15 = ((up[2] != 0) ? hi_23 : lo_24);
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cur_15, 1);
    float pv_25 = _shfl_xor_7;
    float _fmax_151 = fmaxf(cur_15, pv_25);
    float hi_26 = _fmax_151;
    float _min_133 = fminf(cur_15, pv_25);
    float lo_27 = _min_133;
    cur_15 = ((up[3] != 0) ? hi_26 : lo_27);
    V[1] = cur_15;
    int s0_28 = ((sg * 16 + 2) * 8 + cg) * 17;
    int s1_29 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
    float x0_30 = pub[s0_28 + ln];
    float y0_31 = pub[s1_29 + lnr];
    float _min_134 = fminf(x0_30, y0_31);
    float lo0_32 = _min_134;
    float _fmax_152 = fmaxf(r_1, lo0_32);
    r_1 = _fmax_152;
    float _fmax_153 = fmaxf(x0_30, y0_31);
    float hi0_33 = _fmax_153;
    float cur_34 = hi0_33;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 8);
    float pv_35 = _shfl_xor_8;
    float _fmax_154 = fmaxf(cur_34, pv_35);
    float hi_36 = _fmax_154;
    float _min_135 = fminf(cur_34, pv_35);
    float lo_37 = _min_135;
    cur_34 = ((up[0] != 0) ? hi_36 : lo_37);
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 4);
    float pv_38 = _shfl_xor_9;
    float _fmax_155 = fmaxf(cur_34, pv_38);
    float hi_39 = _fmax_155;
    float _min_136 = fminf(cur_34, pv_38);
    float lo_40 = _min_136;
    cur_34 = ((up[1] != 0) ? hi_39 : lo_40);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 2);
    float pv_41 = _shfl_xor_10;
    float _fmax_156 = fmaxf(cur_34, pv_41);
    float hi_42 = _fmax_156;
    float _min_137 = fminf(cur_34, pv_41);
    float lo_43 = _min_137;
    cur_34 = ((up[2] != 0) ? hi_42 : lo_43);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cur_34, 1);
    float pv_44 = _shfl_xor_11;
    float _fmax_157 = fmaxf(cur_34, pv_44);
    float hi_45 = _fmax_157;
    float _min_138 = fminf(cur_34, pv_44);
    float lo_46 = _min_138;
    cur_34 = ((up[3] != 0) ? hi_45 : lo_46);
    V[2] = cur_34;
    int s0_47 = ((sg * 16 + 3) * 8 + cg) * 17;
    int s1_48 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
    float x0_49 = pub[s0_47 + ln];
    float y0_50 = pub[s1_48 + lnr];
    float _min_139 = fminf(x0_49, y0_50);
    float lo0_51 = _min_139;
    float _fmax_158 = fmaxf(r_1, lo0_51);
    r_1 = _fmax_158;
    float _fmax_159 = fmaxf(x0_49, y0_50);
    float hi0_52 = _fmax_159;
    float cur_53 = hi0_52;
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 8);
    float pv_54 = _shfl_xor_12;
    float _fmax_160 = fmaxf(cur_53, pv_54);
    float hi_55_1 = _fmax_160;
    float _min_140 = fminf(cur_53, pv_54);
    float lo_56_1 = _min_140;
    cur_53 = ((up[0] != 0) ? hi_55_1 : lo_56_1);
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 4);
    float pv_57 = _shfl_xor_13;
    float _fmax_161 = fmaxf(cur_53, pv_57);
    float hi_58 = _fmax_161;
    float _min_141 = fminf(cur_53, pv_57);
    float lo_59 = _min_141;
    cur_53 = ((up[1] != 0) ? hi_58 : lo_59);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 2);
    float pv_60 = _shfl_xor_14;
    float _fmax_162 = fmaxf(cur_53, pv_60);
    float hi_61_1 = _fmax_162;
    float _min_142 = fminf(cur_53, pv_60);
    float lo_62_1 = _min_142;
    cur_53 = ((up[2] != 0) ? hi_61_1 : lo_62_1);
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cur_53, 1);
    float pv_63 = _shfl_xor_15;
    float _fmax_163 = fmaxf(cur_53, pv_63);
    float hi_64 = _fmax_163;
    float _min_143 = fminf(cur_53, pv_63);
    float lo_65 = _min_143;
    cur_53 = ((up[3] != 0) ? hi_64 : lo_65);
    V[3] = cur_53;
    int s0_66 = ((sg * 16 + 4) * 8 + cg) * 17;
    int s1_67 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
    float x0_68 = pub[s0_66 + ln];
    float y0_69 = pub[s1_67 + lnr];
    float _min_144 = fminf(x0_68, y0_69);
    float lo0_70 = _min_144;
    float _fmax_164 = fmaxf(r_1, lo0_70);
    r_1 = _fmax_164;
    float _fmax_165 = fmaxf(x0_68, y0_69);
    float hi0_71 = _fmax_165;
    float cur_72 = hi0_71;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 8);
    float pv_73 = _shfl_xor_16;
    float _fmax_166 = fmaxf(cur_72, pv_73);
    float hi_74 = _fmax_166;
    float _min_145 = fminf(cur_72, pv_73);
    float lo_75 = _min_145;
    cur_72 = ((up[0] != 0) ? hi_74 : lo_75);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 4);
    float pv_76 = _shfl_xor_17;
    float _fmax_167 = fmaxf(cur_72, pv_76);
    float hi_77_1 = _fmax_167;
    float _min_146 = fminf(cur_72, pv_76);
    float lo_78_1 = _min_146;
    cur_72 = ((up[1] != 0) ? hi_77_1 : lo_78_1);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 2);
    float pv_79 = _shfl_xor_18;
    float _fmax_168 = fmaxf(cur_72, pv_79);
    float hi_80 = _fmax_168;
    float _min_147 = fminf(cur_72, pv_79);
    float lo_81 = _min_147;
    cur_72 = ((up[2] != 0) ? hi_80 : lo_81);
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cur_72, 1);
    float pv_82 = _shfl_xor_19;
    float _fmax_169 = fmaxf(cur_72, pv_82);
    float hi_83_1 = _fmax_169;
    float _min_148 = fminf(cur_72, pv_82);
    float lo_84_1 = _min_148;
    cur_72 = ((up[3] != 0) ? hi_83_1 : lo_84_1);
    V[4] = cur_72;
    int s0_85 = ((sg * 16 + 5) * 8 + cg) * 17;
    int s1_86 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
    float x0_87 = pub[s0_85 + ln];
    float y0_88 = pub[s1_86 + lnr];
    float _min_149 = fminf(x0_87, y0_88);
    float lo0_89 = _min_149;
    float _fmax_170 = fmaxf(r_1, lo0_89);
    r_1 = _fmax_170;
    float _fmax_171 = fmaxf(x0_87, y0_88);
    float hi0_90 = _fmax_171;
    float cur_91 = hi0_90;
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 8);
    float pv_92 = _shfl_xor_20;
    float _fmax_172 = fmaxf(cur_91, pv_92);
    float hi_93_1 = _fmax_172;
    float _min_150 = fminf(cur_91, pv_92);
    float lo_94_1 = _min_150;
    cur_91 = ((up[0] != 0) ? hi_93_1 : lo_94_1);
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 4);
    float pv_95 = _shfl_xor_21;
    float _fmax_173 = fmaxf(cur_91, pv_95);
    float hi_96 = _fmax_173;
    float _min_151 = fminf(cur_91, pv_95);
    float lo_97 = _min_151;
    cur_91 = ((up[1] != 0) ? hi_96 : lo_97);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 2);
    float pv_98 = _shfl_xor_22;
    float _fmax_174 = fmaxf(cur_91, pv_98);
    float hi_99_1 = _fmax_174;
    float _min_152 = fminf(cur_91, pv_98);
    float lo_100_1 = _min_152;
    cur_91 = ((up[2] != 0) ? hi_99_1 : lo_100_1);
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cur_91, 1);
    float pv_101 = _shfl_xor_23;
    float _fmax_175 = fmaxf(cur_91, pv_101);
    float hi_102 = _fmax_175;
    float _min_153 = fminf(cur_91, pv_101);
    float lo_103 = _min_153;
    cur_91 = ((up[3] != 0) ? hi_102 : lo_103);
    V[5] = cur_91;
    int s0_104 = ((sg * 16 + 6) * 8 + cg) * 17;
    int s1_105 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
    float x0_106 = pub[s0_104 + ln];
    float y0_107 = pub[s1_105 + lnr];
    float _min_154 = fminf(x0_106, y0_107);
    float lo0_108 = _min_154;
    float _fmax_176 = fmaxf(r_1, lo0_108);
    r_1 = _fmax_176;
    float _fmax_177 = fmaxf(x0_106, y0_107);
    float hi0_109 = _fmax_177;
    float cur_110 = hi0_109;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 8);
    float pv_111 = _shfl_xor_24;
    float _fmax_178 = fmaxf(cur_110, pv_111);
    float hi_112 = _fmax_178;
    float _min_155 = fminf(cur_110, pv_111);
    float lo_113 = _min_155;
    cur_110 = ((up[0] != 0) ? hi_112 : lo_113);
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 4);
    float pv_114 = _shfl_xor_25;
    float _fmax_179 = fmaxf(cur_110, pv_114);
    float hi_115_1 = _fmax_179;
    float _min_156 = fminf(cur_110, pv_114);
    float lo_116_1 = _min_156;
    cur_110 = ((up[1] != 0) ? hi_115_1 : lo_116_1);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 2);
    float pv_117 = _shfl_xor_26;
    float _fmax_180 = fmaxf(cur_110, pv_117);
    float hi_118 = _fmax_180;
    float _min_157 = fminf(cur_110, pv_117);
    float lo_119 = _min_157;
    cur_110 = ((up[2] != 0) ? hi_118 : lo_119);
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cur_110, 1);
    float pv_120 = _shfl_xor_27;
    float _fmax_181 = fmaxf(cur_110, pv_120);
    float hi_121_1 = _fmax_181;
    float _min_158 = fminf(cur_110, pv_120);
    float lo_122_1 = _min_158;
    cur_110 = ((up[3] != 0) ? hi_121_1 : lo_122_1);
    V[6] = cur_110;
    int s0_123 = ((sg * 16 + 7) * 8 + cg) * 17;
    int s1_124 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
    float x0_125 = pub[s0_123 + ln];
    float y0_126 = pub[s1_124 + lnr];
    float _min_159 = fminf(x0_125, y0_126);
    float lo0_127 = _min_159;
    float _fmax_182 = fmaxf(r_1, lo0_127);
    r_1 = _fmax_182;
    float _fmax_183 = fmaxf(x0_125, y0_126);
    float hi0_128 = _fmax_183;
    float cur_129 = hi0_128;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 8);
    float pv_130 = _shfl_xor_28;
    float _fmax_184 = fmaxf(cur_129, pv_130);
    float hi_131_1 = _fmax_184;
    float _min_160 = fminf(cur_129, pv_130);
    float lo_132_1 = _min_160;
    cur_129 = ((up[0] != 0) ? hi_131_1 : lo_132_1);
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 4);
    float pv_133 = _shfl_xor_29;
    float _fmax_185 = fmaxf(cur_129, pv_133);
    float hi_134 = _fmax_185;
    float _min_161 = fminf(cur_129, pv_133);
    float lo_135 = _min_161;
    cur_129 = ((up[1] != 0) ? hi_134 : lo_135);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 2);
    float pv_136 = _shfl_xor_30;
    float _fmax_186 = fmaxf(cur_129, pv_136);
    float hi_137_1 = _fmax_186;
    float _min_162 = fminf(cur_129, pv_136);
    float lo_138_1 = _min_162;
    cur_129 = ((up[2] != 0) ? hi_137_1 : lo_138_1);
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cur_129, 1);
    float pv_139 = _shfl_xor_31;
    float _fmax_187 = fmaxf(cur_129, pv_139);
    float hi_140 = _fmax_187;
    float _min_163 = fminf(cur_129, pv_139);
    float lo_141 = _min_163;
    cur_129 = ((up[3] != 0) ? hi_140 : lo_141);
    V[7] = cur_129;
    float rs = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
    float _fmax_188 = fmaxf(r_1, rs);
    r_1 = _fmax_188;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float y1 = _shfl_xor_32;
    float _min_164 = fminf(V[0], y1);
    float lo1 = _min_164;
    float _fmax_189 = fmaxf(r_1, lo1);
    r_1 = _fmax_189;
    float _fmax_190 = fmaxf(V[0], y1);
    float hi1 = _fmax_190;
    float cur_142 = hi1;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 8);
    float pv_143 = _shfl_xor_33;
    float _fmax_191 = fmaxf(cur_142, pv_143);
    float hi_144 = _fmax_191;
    float _min_165 = fminf(cur_142, pv_143);
    float lo_145 = _min_165;
    cur_142 = ((up[0] != 0) ? hi_144 : lo_145);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 4);
    float pv_146 = _shfl_xor_34;
    float _fmax_192 = fmaxf(cur_142, pv_146);
    float hi_147_1 = _fmax_192;
    float _min_166 = fminf(cur_142, pv_146);
    float lo_148_1 = _min_166;
    cur_142 = ((up[1] != 0) ? hi_147_1 : lo_148_1);
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 2);
    float pv_149 = _shfl_xor_35;
    float _fmax_193 = fmaxf(cur_142, pv_149);
    float hi_150 = _fmax_193;
    float _min_167 = fminf(cur_142, pv_149);
    float lo_151 = _min_167;
    cur_142 = ((up[2] != 0) ? hi_150 : lo_151);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_142, 1);
    float pv_152 = _shfl_xor_36;
    float _fmax_194 = fmaxf(cur_142, pv_152);
    float hi_153_1 = _fmax_194;
    float _min_168 = fminf(cur_142, pv_152);
    float lo_154_1 = _min_168;
    cur_142 = ((up[3] != 0) ? hi_153_1 : lo_154_1);
    V[0] = cur_142;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_155 = _shfl_xor_37;
    float _min_169 = fminf(V[1], y1_155);
    float lo1_156 = _min_169;
    float _fmax_195 = fmaxf(r_1, lo1_156);
    r_1 = _fmax_195;
    float _fmax_196 = fmaxf(V[1], y1_155);
    float hi1_157 = _fmax_196;
    float cur_158 = hi1_157;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 8);
    float pv_159 = _shfl_xor_38;
    float _fmax_197 = fmaxf(cur_158, pv_159);
    float hi_160 = _fmax_197;
    float _min_170 = fminf(cur_158, pv_159);
    float lo_161 = _min_170;
    cur_158 = ((up[0] != 0) ? hi_160 : lo_161);
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 4);
    float pv_162 = _shfl_xor_39;
    float _fmax_198 = fmaxf(cur_158, pv_162);
    float hi_163_1 = _fmax_198;
    float _min_171 = fminf(cur_158, pv_162);
    float lo_164_1 = _min_171;
    cur_158 = ((up[1] != 0) ? hi_163_1 : lo_164_1);
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 2);
    float pv_165 = _shfl_xor_40;
    float _fmax_199 = fmaxf(cur_158, pv_165);
    float hi_166 = _fmax_199;
    float _min_172 = fminf(cur_158, pv_165);
    float lo_167 = _min_172;
    cur_158 = ((up[2] != 0) ? hi_166 : lo_167);
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cur_158, 1);
    float pv_168 = _shfl_xor_41;
    float _fmax_200 = fmaxf(cur_158, pv_168);
    float hi_169_1 = _fmax_200;
    float _min_173 = fminf(cur_158, pv_168);
    float lo_170_1 = _min_173;
    cur_158 = ((up[3] != 0) ? hi_169_1 : lo_170_1);
    V[1] = cur_158;
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_171 = _shfl_xor_42;
    float _min_174 = fminf(V[2], y1_171);
    float lo1_172 = _min_174;
    float _fmax_201 = fmaxf(r_1, lo1_172);
    r_1 = _fmax_201;
    float _fmax_202 = fmaxf(V[2], y1_171);
    float hi1_173 = _fmax_202;
    float cur_174 = hi1_173;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 8);
    float pv_175 = _shfl_xor_43;
    float _fmax_203 = fmaxf(cur_174, pv_175);
    float hi_176 = _fmax_203;
    float _min_175 = fminf(cur_174, pv_175);
    float lo_177 = _min_175;
    cur_174 = ((up[0] != 0) ? hi_176 : lo_177);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 4);
    float pv_178 = _shfl_xor_44;
    float _fmax_204 = fmaxf(cur_174, pv_178);
    float hi_179_1 = _fmax_204;
    float _min_176 = fminf(cur_174, pv_178);
    float lo_180_1 = _min_176;
    cur_174 = ((up[1] != 0) ? hi_179_1 : lo_180_1);
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 2);
    float pv_181 = _shfl_xor_45;
    float _fmax_205 = fmaxf(cur_174, pv_181);
    float hi_182 = _fmax_205;
    float _min_177 = fminf(cur_174, pv_181);
    float lo_183 = _min_177;
    cur_174 = ((up[2] != 0) ? hi_182 : lo_183);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_174, 1);
    float pv_184 = _shfl_xor_46;
    float _fmax_206 = fmaxf(cur_174, pv_184);
    float hi_185_1 = _fmax_206;
    float _min_178 = fminf(cur_174, pv_184);
    float lo_186_1 = _min_178;
    cur_174 = ((up[3] != 0) ? hi_185_1 : lo_186_1);
    V[2] = cur_174;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_187 = _shfl_xor_47;
    float _min_179 = fminf(V[3], y1_187);
    float lo1_188 = _min_179;
    float _fmax_207 = fmaxf(r_1, lo1_188);
    r_1 = _fmax_207;
    float _fmax_208 = fmaxf(V[3], y1_187);
    float hi1_189 = _fmax_208;
    float cur_190 = hi1_189;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 8);
    float pv_191 = _shfl_xor_48;
    float _fmax_209 = fmaxf(cur_190, pv_191);
    float hi_192 = _fmax_209;
    float _min_180 = fminf(cur_190, pv_191);
    float lo_193 = _min_180;
    cur_190 = ((up[0] != 0) ? hi_192 : lo_193);
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 4);
    float pv_194 = _shfl_xor_49;
    float _fmax_210 = fmaxf(cur_190, pv_194);
    float hi_195_1 = _fmax_210;
    float _min_181 = fminf(cur_190, pv_194);
    float lo_196_1 = _min_181;
    cur_190 = ((up[1] != 0) ? hi_195_1 : lo_196_1);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 2);
    float pv_197 = _shfl_xor_50;
    float _fmax_211 = fmaxf(cur_190, pv_197);
    float hi_198 = _fmax_211;
    float _min_182 = fminf(cur_190, pv_197);
    float lo_199 = _min_182;
    cur_190 = ((up[2] != 0) ? hi_198 : lo_199);
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cur_190, 1);
    float pv_200 = _shfl_xor_51;
    float _fmax_212 = fmaxf(cur_190, pv_200);
    float hi_201_1 = _fmax_212;
    float _min_183 = fminf(cur_190, pv_200);
    float lo_202_1 = _min_183;
    cur_190 = ((up[3] != 0) ? hi_201_1 : lo_202_1);
    V[3] = cur_190;
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_203 = _shfl_xor_52;
    float _min_184 = fminf(V[0], y1_203);
    float lo1_204 = _min_184;
    float _fmax_213 = fmaxf(r_1, lo1_204);
    r_1 = _fmax_213;
    float _fmax_214 = fmaxf(V[0], y1_203);
    float hi1_205 = _fmax_214;
    float cur_206 = hi1_205;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 8);
    float pv_207 = _shfl_xor_53;
    float _fmax_215 = fmaxf(cur_206, pv_207);
    float hi_208 = _fmax_215;
    float _min_185 = fminf(cur_206, pv_207);
    float lo_209 = _min_185;
    cur_206 = ((up[0] != 0) ? hi_208 : lo_209);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 4);
    float pv_210 = _shfl_xor_54;
    float _fmax_216 = fmaxf(cur_206, pv_210);
    float hi_211_1 = _fmax_216;
    float _min_186 = fminf(cur_206, pv_210);
    float lo_212_1 = _min_186;
    cur_206 = ((up[1] != 0) ? hi_211_1 : lo_212_1);
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 2);
    float pv_213 = _shfl_xor_55;
    float _fmax_217 = fmaxf(cur_206, pv_213);
    float hi_214 = _fmax_217;
    float _min_187 = fminf(cur_206, pv_213);
    float lo_215 = _min_187;
    cur_206 = ((up[2] != 0) ? hi_214 : lo_215);
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_206, 1);
    float pv_216 = _shfl_xor_56;
    float _fmax_218 = fmaxf(cur_206, pv_216);
    float hi_217_1 = _fmax_218;
    float _min_188 = fminf(cur_206, pv_216);
    float lo_218_1 = _min_188;
    cur_206 = ((up[3] != 0) ? hi_217_1 : lo_218_1);
    V[0] = cur_206;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_219 = _shfl_xor_57;
    float _min_189 = fminf(V[1], y1_219);
    float lo1_220 = _min_189;
    float _fmax_219 = fmaxf(r_1, lo1_220);
    r_1 = _fmax_219;
    float _fmax_220 = fmaxf(V[1], y1_219);
    float hi1_221 = _fmax_220;
    float cur_222 = hi1_221;
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 8);
    float pv_223 = _shfl_xor_58;
    float _fmax_221 = fmaxf(cur_222, pv_223);
    float hi_224 = _fmax_221;
    float _min_190 = fminf(cur_222, pv_223);
    float lo_225 = _min_190;
    cur_222 = ((up[0] != 0) ? hi_224 : lo_225);
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 4);
    float pv_226 = _shfl_xor_59;
    float _fmax_222 = fmaxf(cur_222, pv_226);
    float hi_227_1 = _fmax_222;
    float _min_191 = fminf(cur_222, pv_226);
    float lo_228_1 = _min_191;
    cur_222 = ((up[1] != 0) ? hi_227_1 : lo_228_1);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 2);
    float pv_229 = _shfl_xor_60;
    float _fmax_223 = fmaxf(cur_222, pv_229);
    float hi_230 = _fmax_223;
    float _min_192 = fminf(cur_222, pv_229);
    float lo_231 = _min_192;
    cur_222 = ((up[2] != 0) ? hi_230 : lo_231);
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cur_222, 1);
    float pv_232 = _shfl_xor_61;
    float _fmax_224 = fmaxf(cur_222, pv_232);
    float hi_233_1 = _fmax_224;
    float _min_193 = fminf(cur_222, pv_232);
    float lo_234_1 = _min_193;
    cur_222 = ((up[3] != 0) ? hi_233_1 : lo_234_1);
    V[1] = cur_222;
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_62;
    float _min_194 = fminf(V[0], yl);
    float lol = _min_194;
    float _fmax_225 = fmaxf(r_1, lol);
    r_1 = _fmax_225;
    float _fmax_226 = fmaxf(V[0], yl);
    float hil = _fmax_226;
    float cur_235 = hil;
    float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 8);
    float pv_236 = _shfl_xor_63;
    float _fmax_227 = fmaxf(cur_235, pv_236);
    float hi_237_1 = _fmax_227;
    float _min_195 = fminf(cur_235, pv_236);
    float lo_238_1 = _min_195;
    cur_235 = ((up[0] != 0) ? hi_237_1 : lo_238_1);
    float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 4);
    float pv_239 = _shfl_xor_64;
    float _fmax_228 = fmaxf(cur_235, pv_239);
    float hi_240 = _fmax_228;
    float _min_196 = fminf(cur_235, pv_239);
    float lo_241 = _min_196;
    cur_235 = ((up[1] != 0) ? hi_240 : lo_241);
    float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 2);
    float pv_242 = _shfl_xor_65;
    float _fmax_229 = fmaxf(cur_235, pv_242);
    float hi_243_1 = _fmax_229;
    float _min_197 = fminf(cur_235, pv_242);
    float lo_244_1 = _min_197;
    cur_235 = ((up[2] != 0) ? hi_243_1 : lo_244_1);
    float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cur_235, 1);
    float pv_245 = _shfl_xor_66;
    float _fmax_230 = fmaxf(cur_235, pv_245);
    float hi_246 = _fmax_230;
    float _min_198 = fminf(cur_235, pv_245);
    float lo_247 = _min_198;
    cur_235 = ((up[3] != 0) ? hi_246 : lo_247);
    V[0] = cur_235;
    float K = V[0];
    int qb = (sg * 8 + cg) * 32;
    q2[qb + ln] = K;
    q2[qb + 16 + ln] = r_1;
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    if (tid_1 < 128) {
        float r2 = neg_inf;
        float _fmax_231 = fmaxf(r2, q2[cg * 32 + 16 + ln]);
        r2 = _fmax_231;
        float _fmax_232 = fmaxf(r2, q2[(8 + cg) * 32 + 16 + ln]);
        r2 = _fmax_232;
        float V2[1];
        float x2 = q2[cg * 32 + ln];
        float y2 = q2[(8 + cg) * 32 + lnr];
        float _min_199 = fminf(x2, y2);
        float lo2 = _min_199;
        float _fmax_233 = fmaxf(r2, lo2);
        r2 = _fmax_233;
        float _fmax_234 = fmaxf(x2, y2);
        float hi2 = _fmax_234;
        V2[0] = hi2;
        K = V2[0];
        r_1 = r2;
    }
    rr1[0] = r_1;
    int ucol = blockIdx.x * 8 + cg;
    int commit = 0;
    if (g < 8 && ucol < total_q) {
        commit = 1;
    }
    if (tid_1 < 128) {
        float u16 = K;
        float cr = rr1[0];
        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, u16, 8);
        float _min_200 = fminf(u16, _shfl_xor_67);
        u16 = _min_200;
        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cr, 8);
        float _fmax_235 = fmaxf(cr, _shfl_xor_68);
        cr = _fmax_235;
        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, u16, 4);
        float _min_201 = fminf(u16, _shfl_xor_69);
        u16 = _min_201;
        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cr, 4);
        float _fmax_236 = fmaxf(cr, _shfl_xor_70);
        cr = _fmax_236;
        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, u16, 2);
        float _min_202 = fminf(u16, _shfl_xor_71);
        u16 = _min_202;
        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cr, 2);
        float _fmax_237 = fmaxf(cr, _shfl_xor_72);
        cr = _fmax_237;
        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, u16, 1);
        float _min_203 = fminf(u16, _shfl_xor_73);
        u16 = _min_203;
        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, cr, 1);
        float _fmax_238 = fmaxf(cr, _shfl_xor_74);
        cr = _fmax_238;
        unsigned int u16b = __as_u32(u16);
        unsigned int c16 = u16b & 4294966784u;
        unsigned int crb = __as_u32(cr) & 4294966784u;
        unsigned int q = 4294967295;
        if (c16 == crb && u16b < 4278190080u && c16 != 2139094528 && commit != 0) {
            q = c16;
            if (ln == 0) {
                flagw[0] = 1;
            }
        }
        if (ln == 0 && g < 8) {
            qcol[cg] = q;
        }
    }
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    unsigned int need2 = flagw[0];
    unsigned int qc = qcol[c];
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
        int lim2 = 0;
        if (flagged != 0) {
            lim2 = lim;
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
        unsigned int cb2[16];
        unsigned int nb2[16];
        float kb2[16];
        int t0_0_1 = w * 16;
        long long p_1_1 = cbase + (long long)t0_0_1 * nq64;
        cb2[0] = 4286578688;
        if (lim2 > t0_0_1) {
            cb2[0] = S[p_1_1];
        }
        cb2[1] = 4286578688;
        if (lim2 > t0_0_1 + 1) {
            cb2[1] = S[p_1_1 + nq64];
        }
        cb2[2] = 4286578688;
        if (lim2 > t0_0_1 + 2) {
            cb2[2] = S[p_1_1 + 2 * nq64];
        }
        cb2[3] = 4286578688;
        if (lim2 > t0_0_1 + 3) {
            cb2[3] = S[p_1_1 + 3 * nq64];
        }
        cb2[4] = 4286578688;
        if (lim2 > t0_0_1 + 4) {
            cb2[4] = S[p_1_1 + 4 * nq64];
        }
        cb2[5] = 4286578688;
        if (lim2 > t0_0_1 + 5) {
            cb2[5] = S[p_1_1 + 5 * nq64];
        }
        cb2[6] = 4286578688;
        if (lim2 > t0_0_1 + 6) {
            cb2[6] = S[p_1_1 + 6 * nq64];
        }
        cb2[7] = 4286578688;
        if (lim2 > t0_0_1 + 7) {
            cb2[7] = S[p_1_1 + 7 * nq64];
        }
        cb2[8] = 4286578688;
        if (lim2 > t0_0_1 + 8) {
            cb2[8] = S[p_1_1 + 8 * nq64];
        }
        cb2[9] = 4286578688;
        if (lim2 > t0_0_1 + 9) {
            cb2[9] = S[p_1_1 + 9 * nq64];
        }
        cb2[10] = 4286578688;
        if (lim2 > t0_0_1 + 10) {
            cb2[10] = S[p_1_1 + 10 * nq64];
        }
        cb2[11] = 4286578688;
        if (lim2 > t0_0_1 + 11) {
            cb2[11] = S[p_1_1 + 11 * nq64];
        }
        cb2[12] = 4286578688;
        if (lim2 > t0_0_1 + 12) {
            cb2[12] = S[p_1_1 + 12 * nq64];
        }
        cb2[13] = 4286578688;
        if (lim2 > t0_0_1 + 13) {
            cb2[13] = S[p_1_1 + 13 * nq64];
        }
        cb2[14] = 4286578688;
        if (lim2 > t0_0_1 + 14) {
            cb2[14] = S[p_1_1 + 14 * nq64];
        }
        cb2[15] = 4286578688;
        if (lim2 > t0_0_1 + 15) {
            cb2[15] = S[p_1_1 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
        #pragma unroll 1
        for (int j_1 = 0; j_1 < num_chunks; j_1++) {
            int t0_1 = ((j_1 + 1) * 32 + w) * 16;
            long long p_2 = cbase + (long long)t0_1 * nq64;
            nb2[0] = 4286578688;
            if (lim2 > t0_1) {
                nb2[0] = S[p_2];
            }
            nb2[1] = 4286578688;
            if (lim2 > t0_1 + 1) {
                nb2[1] = S[p_2 + nq64];
            }
            nb2[2] = 4286578688;
            if (lim2 > t0_1 + 2) {
                nb2[2] = S[p_2 + 2 * nq64];
            }
            nb2[3] = 4286578688;
            if (lim2 > t0_1 + 3) {
                nb2[3] = S[p_2 + 3 * nq64];
            }
            nb2[4] = 4286578688;
            if (lim2 > t0_1 + 4) {
                nb2[4] = S[p_2 + 4 * nq64];
            }
            nb2[5] = 4286578688;
            if (lim2 > t0_1 + 5) {
                nb2[5] = S[p_2 + 5 * nq64];
            }
            nb2[6] = 4286578688;
            if (lim2 > t0_1 + 6) {
                nb2[6] = S[p_2 + 6 * nq64];
            }
            nb2[7] = 4286578688;
            if (lim2 > t0_1 + 7) {
                nb2[7] = S[p_2 + 7 * nq64];
            }
            nb2[8] = 4286578688;
            if (lim2 > t0_1 + 8) {
                nb2[8] = S[p_2 + 8 * nq64];
            }
            nb2[9] = 4286578688;
            if (lim2 > t0_1 + 9) {
                nb2[9] = S[p_2 + 9 * nq64];
            }
            nb2[10] = 4286578688;
            if (lim2 > t0_1 + 10) {
                nb2[10] = S[p_2 + 10 * nq64];
            }
            nb2[11] = 4286578688;
            if (lim2 > t0_1 + 11) {
                nb2[11] = S[p_2 + 11 * nq64];
            }
            nb2[12] = 4286578688;
            if (lim2 > t0_1 + 12) {
                nb2[12] = S[p_2 + 12 * nq64];
            }
            nb2[13] = 4286578688;
            if (lim2 > t0_1 + 13) {
                nb2[13] = S[p_2 + 13 * nq64];
            }
            nb2[14] = 4286578688;
            if (lim2 > t0_1 + 14) {
                nb2[14] = S[p_2 + 14 * nq64];
            }
            nb2[15] = 4286578688;
            if (lim2 > t0_1 + 15) {
                nb2[15] = S[p_2 + 15 * nq64];
            }
            asm volatile("" ::: "memory");
            int t0_3 = (j_1 * 32 + w) * 16;
            float qf = __uint_as_float(qc);
            float sc_1 = __uint_as_float(cb2[0]);
            float _fmax_239 = fmaxf(sc_1, -1.7014118346046923e+38f);
            sc_1 = _fmax_239;
            float _min_204 = fminf(sc_1, 1.7014118346046923e+38f);
            sc_1 = _min_204;
            sc_1 = sc_1;
            float sc_4_1 = sc_1;
            unsigned int u = __as_u32(sc_4_1);
            unsigned int cls = u & 4294966784u;
            int f_16 = 0;
            if (t0_3 < fb || t0_3 >= lim - fe && lim > t0_3) {
                f_16 = 1;
            }
            if (f_16 != 0) {
                cls = 2139094528;
            }
            unsigned int key_1 = 0;
            if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                key_1 = 1073741824 | (unsigned int)t0_3;
            }
            if (cls == qc) {
                unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 511) & 511;
                key_1 = 536870912 | lowb << 9 | (unsigned int)t0_3;
            }
            kb2[0] = __uint_as_float(key_1);
            float sc_5_1 = __uint_as_float(cb2[1]);
            float _fmax_240 = fmaxf(sc_5_1, -1.7014118346046923e+38f);
            sc_5_1 = _fmax_240;
            float _min_205 = fminf(sc_5_1, 1.7014118346046923e+38f);
            sc_5_1 = _min_205;
            sc_5_1 = sc_5_1;
            float sc_6 = sc_5_1;
            unsigned int u_7 = __as_u32(sc_6);
            unsigned int cls_8 = u_7 & 4294966784u;
            int f_9_1 = 0;
            if (t0_3 + 1 < fb || t0_3 + 1 >= lim - fe && lim > t0_3 + 1) {
                f_9_1 = 1;
            }
            if (f_9_1 != 0) {
                cls_8 = 2139094528;
            }
            unsigned int key_10 = 0;
            if (qf < __uint_as_float(cls_8) && cls_8 < 4278190080u) {
                key_10 = 1073741824 | (unsigned int)(t0_3 + 1);
            }
            if (cls_8 == qc) {
                unsigned int lowb_1 = (u_7 ^ (unsigned int)((int)u_7 >> 31) & 511) & 511;
                key_10 = 536870912 | lowb_1 << 9 | (unsigned int)(t0_3 + 1);
            }
            kb2[1] = __uint_as_float(key_10);
            float sc_11_1 = __uint_as_float(cb2[2]);
            float _fmax_241 = fmaxf(sc_11_1, -1.7014118346046923e+38f);
            sc_11_1 = _fmax_241;
            float _min_206 = fminf(sc_11_1, 1.7014118346046923e+38f);
            sc_11_1 = _min_206;
            sc_11_1 = sc_11_1;
            float sc_12 = sc_11_1;
            unsigned int u_13 = __as_u32(sc_12);
            unsigned int cls_14 = u_13 & 4294966784u;
            int f_15_1 = 0;
            if (t0_3 + 2 < fb || t0_3 + 2 >= lim - fe && lim > t0_3 + 2) {
                f_15_1 = 1;
            }
            if (f_15_1 != 0) {
                cls_14 = 2139094528;
            }
            unsigned int key_16 = 0;
            if (qf < __uint_as_float(cls_14) && cls_14 < 4278190080u) {
                key_16 = 1073741824 | (unsigned int)(t0_3 + 2);
            }
            if (cls_14 == qc) {
                unsigned int lowb_2 = (u_13 ^ (unsigned int)((int)u_13 >> 31) & 511) & 511;
                key_16 = 536870912 | lowb_2 << 9 | (unsigned int)(t0_3 + 2);
            }
            kb2[2] = __uint_as_float(key_16);
            float sc_17_1 = __uint_as_float(cb2[3]);
            float _fmax_242 = fmaxf(sc_17_1, -1.7014118346046923e+38f);
            sc_17_1 = _fmax_242;
            float _min_207 = fminf(sc_17_1, 1.7014118346046923e+38f);
            sc_17_1 = _min_207;
            sc_17_1 = sc_17_1;
            float sc_18 = sc_17_1;
            unsigned int u_19 = __as_u32(sc_18);
            unsigned int cls_20 = u_19 & 4294966784u;
            int f_21 = 0;
            if (t0_3 + 3 < fb || t0_3 + 3 >= lim - fe && lim > t0_3 + 3) {
                f_21 = 1;
            }
            if (f_21 != 0) {
                cls_20 = 2139094528;
            }
            unsigned int key_22 = 0;
            if (qf < __uint_as_float(cls_20) && cls_20 < 4278190080u) {
                key_22 = 1073741824 | (unsigned int)(t0_3 + 3);
            }
            if (cls_20 == qc) {
                unsigned int lowb_3 = (u_19 ^ (unsigned int)((int)u_19 >> 31) & 511) & 511;
                key_22 = 536870912 | lowb_3 << 9 | (unsigned int)(t0_3 + 3);
            }
            kb2[3] = __uint_as_float(key_22);
            float sc_23_1 = __uint_as_float(cb2[4]);
            float _fmax_243 = fmaxf(sc_23_1, -1.7014118346046923e+38f);
            sc_23_1 = _fmax_243;
            float _min_208 = fminf(sc_23_1, 1.7014118346046923e+38f);
            sc_23_1 = _min_208;
            sc_23_1 = sc_23_1;
            float sc_24 = sc_23_1;
            unsigned int u_25 = __as_u32(sc_24);
            unsigned int cls_26 = u_25 & 4294966784u;
            int f_27 = 0;
            if (t0_3 + 4 < fb || t0_3 + 4 >= lim - fe && lim > t0_3 + 4) {
                f_27 = 1;
            }
            if (f_27 != 0) {
                cls_26 = 2139094528;
            }
            unsigned int key_28 = 0;
            if (qf < __uint_as_float(cls_26) && cls_26 < 4278190080u) {
                key_28 = 1073741824 | (unsigned int)(t0_3 + 4);
            }
            if (cls_26 == qc) {
                unsigned int lowb_4 = (u_25 ^ (unsigned int)((int)u_25 >> 31) & 511) & 511;
                key_28 = 536870912 | lowb_4 << 9 | (unsigned int)(t0_3 + 4);
            }
            kb2[4] = __uint_as_float(key_28);
            float sc_29_1 = __uint_as_float(cb2[5]);
            float _fmax_244 = fmaxf(sc_29_1, -1.7014118346046923e+38f);
            sc_29_1 = _fmax_244;
            float _min_209 = fminf(sc_29_1, 1.7014118346046923e+38f);
            sc_29_1 = _min_209;
            sc_29_1 = sc_29_1;
            float sc_30 = sc_29_1;
            unsigned int u_31 = __as_u32(sc_30);
            unsigned int cls_32 = u_31 & 4294966784u;
            int f_33 = 0;
            if (t0_3 + 5 < fb || t0_3 + 5 >= lim - fe && lim > t0_3 + 5) {
                f_33 = 1;
            }
            if (f_33 != 0) {
                cls_32 = 2139094528;
            }
            unsigned int key_34 = 0;
            if (qf < __uint_as_float(cls_32) && cls_32 < 4278190080u) {
                key_34 = 1073741824 | (unsigned int)(t0_3 + 5);
            }
            if (cls_32 == qc) {
                unsigned int lowb_5 = (u_31 ^ (unsigned int)((int)u_31 >> 31) & 511) & 511;
                key_34 = 536870912 | lowb_5 << 9 | (unsigned int)(t0_3 + 5);
            }
            kb2[5] = __uint_as_float(key_34);
            float sc_35_1 = __uint_as_float(cb2[6]);
            float _fmax_245 = fmaxf(sc_35_1, -1.7014118346046923e+38f);
            sc_35_1 = _fmax_245;
            float _min_210 = fminf(sc_35_1, 1.7014118346046923e+38f);
            sc_35_1 = _min_210;
            sc_35_1 = sc_35_1;
            float sc_36 = sc_35_1;
            unsigned int u_37 = __as_u32(sc_36);
            unsigned int cls_38 = u_37 & 4294966784u;
            int f_39 = 0;
            if (t0_3 + 6 < fb || t0_3 + 6 >= lim - fe && lim > t0_3 + 6) {
                f_39 = 1;
            }
            if (f_39 != 0) {
                cls_38 = 2139094528;
            }
            unsigned int key_40 = 0;
            if (qf < __uint_as_float(cls_38) && cls_38 < 4278190080u) {
                key_40 = 1073741824 | (unsigned int)(t0_3 + 6);
            }
            if (cls_38 == qc) {
                unsigned int lowb_6 = (u_37 ^ (unsigned int)((int)u_37 >> 31) & 511) & 511;
                key_40 = 536870912 | lowb_6 << 9 | (unsigned int)(t0_3 + 6);
            }
            kb2[6] = __uint_as_float(key_40);
            float sc_41_1 = __uint_as_float(cb2[7]);
            float _fmax_246 = fmaxf(sc_41_1, -1.7014118346046923e+38f);
            sc_41_1 = _fmax_246;
            float _min_211 = fminf(sc_41_1, 1.7014118346046923e+38f);
            sc_41_1 = _min_211;
            sc_41_1 = sc_41_1;
            float sc_42 = sc_41_1;
            unsigned int u_43 = __as_u32(sc_42);
            unsigned int cls_44 = u_43 & 4294966784u;
            int f_45 = 0;
            if (t0_3 + 7 < fb || t0_3 + 7 >= lim - fe && lim > t0_3 + 7) {
                f_45 = 1;
            }
            if (f_45 != 0) {
                cls_44 = 2139094528;
            }
            unsigned int key_46 = 0;
            if (qf < __uint_as_float(cls_44) && cls_44 < 4278190080u) {
                key_46 = 1073741824 | (unsigned int)(t0_3 + 7);
            }
            if (cls_44 == qc) {
                unsigned int lowb_7 = (u_43 ^ (unsigned int)((int)u_43 >> 31) & 511) & 511;
                key_46 = 536870912 | lowb_7 << 9 | (unsigned int)(t0_3 + 7);
            }
            kb2[7] = __uint_as_float(key_46);
            float sc_47_1 = __uint_as_float(cb2[8]);
            float _fmax_247 = fmaxf(sc_47_1, -1.7014118346046923e+38f);
            sc_47_1 = _fmax_247;
            float _min_212 = fminf(sc_47_1, 1.7014118346046923e+38f);
            sc_47_1 = _min_212;
            sc_47_1 = sc_47_1;
            float sc_48 = sc_47_1;
            unsigned int u_49 = __as_u32(sc_48);
            unsigned int cls_50 = u_49 & 4294966784u;
            int f_51 = 0;
            if (t0_3 + 8 < fb || t0_3 + 8 >= lim - fe && lim > t0_3 + 8) {
                f_51 = 1;
            }
            if (f_51 != 0) {
                cls_50 = 2139094528;
            }
            unsigned int key_52 = 0;
            if (qf < __uint_as_float(cls_50) && cls_50 < 4278190080u) {
                key_52 = 1073741824 | (unsigned int)(t0_3 + 8);
            }
            if (cls_50 == qc) {
                unsigned int lowb_8 = (u_49 ^ (unsigned int)((int)u_49 >> 31) & 511) & 511;
                key_52 = 536870912 | lowb_8 << 9 | (unsigned int)(t0_3 + 8);
            }
            kb2[8] = __uint_as_float(key_52);
            float sc_53 = __uint_as_float(cb2[9]);
            float _fmax_248 = fmaxf(sc_53, -1.7014118346046923e+38f);
            sc_53 = _fmax_248;
            float _min_213 = fminf(sc_53, 1.7014118346046923e+38f);
            sc_53 = _min_213;
            sc_53 = sc_53;
            float sc_54 = sc_53;
            unsigned int u_55 = __as_u32(sc_54);
            unsigned int cls_56 = u_55 & 4294966784u;
            int f_57 = 0;
            if (t0_3 + 9 < fb || t0_3 + 9 >= lim - fe && lim > t0_3 + 9) {
                f_57 = 1;
            }
            if (f_57 != 0) {
                cls_56 = 2139094528;
            }
            unsigned int key_58 = 0;
            if (qf < __uint_as_float(cls_56) && cls_56 < 4278190080u) {
                key_58 = 1073741824 | (unsigned int)(t0_3 + 9);
            }
            if (cls_56 == qc) {
                unsigned int lowb_9 = (u_55 ^ (unsigned int)((int)u_55 >> 31) & 511) & 511;
                key_58 = 536870912 | lowb_9 << 9 | (unsigned int)(t0_3 + 9);
            }
            kb2[9] = __uint_as_float(key_58);
            float sc_59 = __uint_as_float(cb2[10]);
            float _fmax_249 = fmaxf(sc_59, -1.7014118346046923e+38f);
            sc_59 = _fmax_249;
            float _min_214 = fminf(sc_59, 1.7014118346046923e+38f);
            sc_59 = _min_214;
            sc_59 = sc_59;
            float sc_60 = sc_59;
            unsigned int u_61 = __as_u32(sc_60);
            unsigned int cls_62 = u_61 & 4294966784u;
            int f_63 = 0;
            if (t0_3 + 10 < fb || t0_3 + 10 >= lim - fe && lim > t0_3 + 10) {
                f_63 = 1;
            }
            if (f_63 != 0) {
                cls_62 = 2139094528;
            }
            unsigned int key_64 = 0;
            if (qf < __uint_as_float(cls_62) && cls_62 < 4278190080u) {
                key_64 = 1073741824 | (unsigned int)(t0_3 + 10);
            }
            if (cls_62 == qc) {
                unsigned int lowb_10 = (u_61 ^ (unsigned int)((int)u_61 >> 31) & 511) & 511;
                key_64 = 536870912 | lowb_10 << 9 | (unsigned int)(t0_3 + 10);
            }
            kb2[10] = __uint_as_float(key_64);
            float sc_65 = __uint_as_float(cb2[11]);
            float _fmax_250 = fmaxf(sc_65, -1.7014118346046923e+38f);
            sc_65 = _fmax_250;
            float _min_215 = fminf(sc_65, 1.7014118346046923e+38f);
            sc_65 = _min_215;
            sc_65 = sc_65;
            float sc_66 = sc_65;
            unsigned int u_67 = __as_u32(sc_66);
            unsigned int cls_68 = u_67 & 4294966784u;
            int f_69 = 0;
            if (t0_3 + 11 < fb || t0_3 + 11 >= lim - fe && lim > t0_3 + 11) {
                f_69 = 1;
            }
            if (f_69 != 0) {
                cls_68 = 2139094528;
            }
            unsigned int key_70 = 0;
            if (qf < __uint_as_float(cls_68) && cls_68 < 4278190080u) {
                key_70 = 1073741824 | (unsigned int)(t0_3 + 11);
            }
            if (cls_68 == qc) {
                unsigned int lowb_11 = (u_67 ^ (unsigned int)((int)u_67 >> 31) & 511) & 511;
                key_70 = 536870912 | lowb_11 << 9 | (unsigned int)(t0_3 + 11);
            }
            kb2[11] = __uint_as_float(key_70);
            float sc_71 = __uint_as_float(cb2[12]);
            float _fmax_251 = fmaxf(sc_71, -1.7014118346046923e+38f);
            sc_71 = _fmax_251;
            float _min_216 = fminf(sc_71, 1.7014118346046923e+38f);
            sc_71 = _min_216;
            sc_71 = sc_71;
            float sc_72 = sc_71;
            unsigned int u_73 = __as_u32(sc_72);
            unsigned int cls_74 = u_73 & 4294966784u;
            int f_75 = 0;
            if (t0_3 + 12 < fb || t0_3 + 12 >= lim - fe && lim > t0_3 + 12) {
                f_75 = 1;
            }
            if (f_75 != 0) {
                cls_74 = 2139094528;
            }
            unsigned int key_76 = 0;
            if (qf < __uint_as_float(cls_74) && cls_74 < 4278190080u) {
                key_76 = 1073741824 | (unsigned int)(t0_3 + 12);
            }
            if (cls_74 == qc) {
                unsigned int lowb_12 = (u_73 ^ (unsigned int)((int)u_73 >> 31) & 511) & 511;
                key_76 = 536870912 | lowb_12 << 9 | (unsigned int)(t0_3 + 12);
            }
            kb2[12] = __uint_as_float(key_76);
            float sc_77 = __uint_as_float(cb2[13]);
            float _fmax_252 = fmaxf(sc_77, -1.7014118346046923e+38f);
            sc_77 = _fmax_252;
            float _min_217 = fminf(sc_77, 1.7014118346046923e+38f);
            sc_77 = _min_217;
            sc_77 = sc_77;
            float sc_78 = sc_77;
            unsigned int u_79 = __as_u32(sc_78);
            unsigned int cls_80 = u_79 & 4294966784u;
            int f_81 = 0;
            if (t0_3 + 13 < fb || t0_3 + 13 >= lim - fe && lim > t0_3 + 13) {
                f_81 = 1;
            }
            if (f_81 != 0) {
                cls_80 = 2139094528;
            }
            unsigned int key_82 = 0;
            if (qf < __uint_as_float(cls_80) && cls_80 < 4278190080u) {
                key_82 = 1073741824 | (unsigned int)(t0_3 + 13);
            }
            if (cls_80 == qc) {
                unsigned int lowb_13 = (u_79 ^ (unsigned int)((int)u_79 >> 31) & 511) & 511;
                key_82 = 536870912 | lowb_13 << 9 | (unsigned int)(t0_3 + 13);
            }
            kb2[13] = __uint_as_float(key_82);
            float sc_83 = __uint_as_float(cb2[14]);
            float _fmax_253 = fmaxf(sc_83, -1.7014118346046923e+38f);
            sc_83 = _fmax_253;
            float _min_218 = fminf(sc_83, 1.7014118346046923e+38f);
            sc_83 = _min_218;
            sc_83 = sc_83;
            float sc_84 = sc_83;
            unsigned int u_85 = __as_u32(sc_84);
            unsigned int cls_86 = u_85 & 4294966784u;
            int f_87 = 0;
            if (t0_3 + 14 < fb || t0_3 + 14 >= lim - fe && lim > t0_3 + 14) {
                f_87 = 1;
            }
            if (f_87 != 0) {
                cls_86 = 2139094528;
            }
            unsigned int key_88 = 0;
            if (qf < __uint_as_float(cls_86) && cls_86 < 4278190080u) {
                key_88 = 1073741824 | (unsigned int)(t0_3 + 14);
            }
            if (cls_86 == qc) {
                unsigned int lowb_14 = (u_85 ^ (unsigned int)((int)u_85 >> 31) & 511) & 511;
                key_88 = 536870912 | lowb_14 << 9 | (unsigned int)(t0_3 + 14);
            }
            kb2[14] = __uint_as_float(key_88);
            float sc_89 = __uint_as_float(cb2[15]);
            float _fmax_254 = fmaxf(sc_89, -1.7014118346046923e+38f);
            sc_89 = _fmax_254;
            float _min_219 = fminf(sc_89, 1.7014118346046923e+38f);
            sc_89 = _min_219;
            sc_89 = sc_89;
            float sc_90 = sc_89;
            unsigned int u_91 = __as_u32(sc_90);
            unsigned int cls_92 = u_91 & 4294966784u;
            int f_93 = 0;
            if (t0_3 + 15 < fb || t0_3 + 15 >= lim - fe && lim > t0_3 + 15) {
                f_93 = 1;
            }
            if (f_93 != 0) {
                cls_92 = 2139094528;
            }
            unsigned int key_94 = 0;
            if (qf < __uint_as_float(cls_92) && cls_92 < 4278190080u) {
                key_94 = 1073741824 | (unsigned int)(t0_3 + 15);
            }
            if (cls_92 == qc) {
                unsigned int lowb_15 = (u_91 ^ (unsigned int)((int)u_91 >> 31) & 511) & 511;
                key_94 = 536870912 | lowb_15 << 9 | (unsigned int)(t0_3 + 15);
            }
            kb2[15] = __uint_as_float(key_94);
            float _fmax_255 = fmaxf(kb2[0], kb2[13]);
            float hi_95_1 = _fmax_255;
            float _min_220 = fminf(kb2[0], kb2[13]);
            float lo_96_1 = _min_220;
            kb2[0] = hi_95_1;
            kb2[13] = lo_96_1;
            float _fmax_256 = fmaxf(kb2[1], kb2[12]);
            float hi_97_1 = _fmax_256;
            float _min_221 = fminf(kb2[1], kb2[12]);
            float lo_98_1 = _min_221;
            kb2[1] = hi_97_1;
            kb2[12] = lo_98_1;
            float _fmax_257 = fmaxf(kb2[2], kb2[15]);
            float hi_100 = _fmax_257;
            float _min_222 = fminf(kb2[2], kb2[15]);
            float lo_101 = _min_222;
            kb2[2] = hi_100;
            kb2[15] = lo_101;
            float _fmax_258 = fmaxf(kb2[3], kb2[14]);
            float hi_103_1 = _fmax_258;
            float _min_223 = fminf(kb2[3], kb2[14]);
            float lo_104_1 = _min_223;
            kb2[3] = hi_103_1;
            kb2[14] = lo_104_1;
            float _fmax_259 = fmaxf(kb2[4], kb2[8]);
            float hi_105_1 = _fmax_259;
            float _min_224 = fminf(kb2[4], kb2[8]);
            float lo_106_1 = _min_224;
            kb2[4] = hi_105_1;
            kb2[8] = lo_106_1;
            float _fmax_260 = fmaxf(kb2[5], kb2[6]);
            float hi_107_1 = _fmax_260;
            float _min_225 = fminf(kb2[5], kb2[6]);
            float lo_108_1 = _min_225;
            kb2[5] = hi_107_1;
            kb2[6] = lo_108_1;
            float _fmax_261 = fmaxf(kb2[7], kb2[11]);
            float hi_109_1 = _fmax_261;
            float _min_226 = fminf(kb2[7], kb2[11]);
            float lo_110_1 = _min_226;
            kb2[7] = hi_109_1;
            kb2[11] = lo_110_1;
            float _fmax_262 = fmaxf(kb2[9], kb2[10]);
            float hi_111_1 = _fmax_262;
            float _min_227 = fminf(kb2[9], kb2[10]);
            float lo_112_1 = _min_227;
            kb2[9] = hi_111_1;
            kb2[10] = lo_112_1;
            float _fmax_263 = fmaxf(kb2[0], kb2[5]);
            float hi_113_1 = _fmax_263;
            float _min_228 = fminf(kb2[0], kb2[5]);
            float lo_114_1 = _min_228;
            kb2[0] = hi_113_1;
            kb2[5] = lo_114_1;
            float _fmax_264 = fmaxf(kb2[1], kb2[7]);
            float hi_116 = _fmax_264;
            float _min_229 = fminf(kb2[1], kb2[7]);
            float lo_117 = _min_229;
            kb2[1] = hi_116;
            kb2[7] = lo_117;
            float _fmax_265 = fmaxf(kb2[2], kb2[9]);
            float hi_119_1 = _fmax_265;
            float _min_230 = fminf(kb2[2], kb2[9]);
            float lo_120_1 = _min_230;
            kb2[2] = hi_119_1;
            kb2[9] = lo_120_1;
            float _fmax_266 = fmaxf(kb2[3], kb2[4]);
            float hi_122 = _fmax_266;
            float _min_231 = fminf(kb2[3], kb2[4]);
            float lo_123 = _min_231;
            kb2[3] = hi_122;
            kb2[4] = lo_123;
            float _fmax_267 = fmaxf(kb2[6], kb2[13]);
            float hi_124 = _fmax_267;
            float _min_232 = fminf(kb2[6], kb2[13]);
            float lo_125 = _min_232;
            kb2[6] = hi_124;
            kb2[13] = lo_125;
            float _fmax_268 = fmaxf(kb2[8], kb2[14]);
            float hi_126 = _fmax_268;
            float _min_233 = fminf(kb2[8], kb2[14]);
            float lo_127 = _min_233;
            kb2[8] = hi_126;
            kb2[14] = lo_127;
            float _fmax_269 = fmaxf(kb2[10], kb2[15]);
            float hi_128 = _fmax_269;
            float _min_234 = fminf(kb2[10], kb2[15]);
            float lo_129 = _min_234;
            kb2[10] = hi_128;
            kb2[15] = lo_129;
            float _fmax_270 = fmaxf(kb2[11], kb2[12]);
            float hi_130 = _fmax_270;
            float _min_235 = fminf(kb2[11], kb2[12]);
            float lo_131 = _min_235;
            kb2[11] = hi_130;
            kb2[12] = lo_131;
            float _fmax_271 = fmaxf(kb2[0], kb2[1]);
            float hi_132 = _fmax_271;
            float _min_236 = fminf(kb2[0], kb2[1]);
            float lo_133 = _min_236;
            kb2[0] = hi_132;
            kb2[1] = lo_133;
            float _fmax_272 = fmaxf(kb2[2], kb2[3]);
            float hi_135_1 = _fmax_272;
            float _min_237 = fminf(kb2[2], kb2[3]);
            float lo_136_1 = _min_237;
            kb2[2] = hi_135_1;
            kb2[3] = lo_136_1;
            float _fmax_273 = fmaxf(kb2[4], kb2[5]);
            float hi_138 = _fmax_273;
            float _min_238 = fminf(kb2[4], kb2[5]);
            float lo_139 = _min_238;
            kb2[4] = hi_138;
            kb2[5] = lo_139;
            float _fmax_274 = fmaxf(kb2[6], kb2[8]);
            float hi_141_1 = _fmax_274;
            float _min_239 = fminf(kb2[6], kb2[8]);
            float lo_142_1 = _min_239;
            kb2[6] = hi_141_1;
            kb2[8] = lo_142_1;
            float _fmax_275 = fmaxf(kb2[7], kb2[9]);
            float hi_143_1 = _fmax_275;
            float _min_240 = fminf(kb2[7], kb2[9]);
            float lo_144_1 = _min_240;
            kb2[7] = hi_143_1;
            kb2[9] = lo_144_1;
            float _fmax_276 = fmaxf(kb2[10], kb2[11]);
            float hi_145_1 = _fmax_276;
            float _min_241 = fminf(kb2[10], kb2[11]);
            float lo_146_1 = _min_241;
            kb2[10] = hi_145_1;
            kb2[11] = lo_146_1;
            float _fmax_277 = fmaxf(kb2[12], kb2[13]);
            float hi_148 = _fmax_277;
            float _min_242 = fminf(kb2[12], kb2[13]);
            float lo_149 = _min_242;
            kb2[12] = hi_148;
            kb2[13] = lo_149;
            float _fmax_278 = fmaxf(kb2[14], kb2[15]);
            float hi_151_1 = _fmax_278;
            float _min_243 = fminf(kb2[14], kb2[15]);
            float lo_152_1 = _min_243;
            kb2[14] = hi_151_1;
            kb2[15] = lo_152_1;
            float _fmax_279 = fmaxf(kb2[0], kb2[2]);
            float hi_154 = _fmax_279;
            float _min_244 = fminf(kb2[0], kb2[2]);
            float lo_155 = _min_244;
            kb2[0] = hi_154;
            kb2[2] = lo_155;
            float _fmax_280 = fmaxf(kb2[1], kb2[3]);
            float hi_156 = _fmax_280;
            float _min_245 = fminf(kb2[1], kb2[3]);
            float lo_157 = _min_245;
            kb2[1] = hi_156;
            kb2[3] = lo_157;
            float _fmax_281 = fmaxf(kb2[4], kb2[10]);
            float hi_158 = _fmax_281;
            float _min_246 = fminf(kb2[4], kb2[10]);
            float lo_159 = _min_246;
            kb2[4] = hi_158;
            kb2[10] = lo_159;
            float _fmax_282 = fmaxf(kb2[5], kb2[11]);
            float hi_161_1 = _fmax_282;
            float _min_247 = fminf(kb2[5], kb2[11]);
            float lo_162_1 = _min_247;
            kb2[5] = hi_161_1;
            kb2[11] = lo_162_1;
            float _fmax_283 = fmaxf(kb2[6], kb2[7]);
            float hi_164 = _fmax_283;
            float _min_248 = fminf(kb2[6], kb2[7]);
            float lo_165 = _min_248;
            kb2[6] = hi_164;
            kb2[7] = lo_165;
            float _fmax_284 = fmaxf(kb2[8], kb2[9]);
            float hi_167_1 = _fmax_284;
            float _min_249 = fminf(kb2[8], kb2[9]);
            float lo_168_1 = _min_249;
            kb2[8] = hi_167_1;
            kb2[9] = lo_168_1;
            float _fmax_285 = fmaxf(kb2[12], kb2[14]);
            float hi_170 = _fmax_285;
            float _min_250 = fminf(kb2[12], kb2[14]);
            float lo_171 = _min_250;
            kb2[12] = hi_170;
            kb2[14] = lo_171;
            float _fmax_286 = fmaxf(kb2[13], kb2[15]);
            float hi_172 = _fmax_286;
            float _min_251 = fminf(kb2[13], kb2[15]);
            float lo_173 = _min_251;
            kb2[13] = hi_172;
            kb2[15] = lo_173;
            float _fmax_287 = fmaxf(kb2[1], kb2[2]);
            float hi_174 = _fmax_287;
            float _min_252 = fminf(kb2[1], kb2[2]);
            float lo_175 = _min_252;
            kb2[1] = hi_174;
            kb2[2] = lo_175;
            float _fmax_288 = fmaxf(kb2[3], kb2[12]);
            float hi_177_1 = _fmax_288;
            float _min_253 = fminf(kb2[3], kb2[12]);
            float lo_178_1 = _min_253;
            kb2[3] = hi_177_1;
            kb2[12] = lo_178_1;
            float _fmax_289 = fmaxf(kb2[4], kb2[6]);
            float hi_180 = _fmax_289;
            float _min_254 = fminf(kb2[4], kb2[6]);
            float lo_181 = _min_254;
            kb2[4] = hi_180;
            kb2[6] = lo_181;
            float _fmax_290 = fmaxf(kb2[5], kb2[7]);
            float hi_183_1 = _fmax_290;
            float _min_255 = fminf(kb2[5], kb2[7]);
            float lo_184_1 = _min_255;
            kb2[5] = hi_183_1;
            kb2[7] = lo_184_1;
            float _fmax_291 = fmaxf(kb2[8], kb2[10]);
            float hi_186 = _fmax_291;
            float _min_256 = fminf(kb2[8], kb2[10]);
            float lo_187 = _min_256;
            kb2[8] = hi_186;
            kb2[10] = lo_187;
            float _fmax_292 = fmaxf(kb2[9], kb2[11]);
            float hi_188 = _fmax_292;
            float _min_257 = fminf(kb2[9], kb2[11]);
            float lo_189 = _min_257;
            kb2[9] = hi_188;
            kb2[11] = lo_189;
            float _fmax_293 = fmaxf(kb2[13], kb2[14]);
            float hi_190 = _fmax_293;
            float _min_258 = fminf(kb2[13], kb2[14]);
            float lo_191 = _min_258;
            kb2[13] = hi_190;
            kb2[14] = lo_191;
            float _fmax_294 = fmaxf(kb2[1], kb2[4]);
            float hi_193_1 = _fmax_294;
            float _min_259 = fminf(kb2[1], kb2[4]);
            float lo_194_1 = _min_259;
            kb2[1] = hi_193_1;
            kb2[4] = lo_194_1;
            float _fmax_295 = fmaxf(kb2[2], kb2[6]);
            float hi_196 = _fmax_295;
            float _min_260 = fminf(kb2[2], kb2[6]);
            float lo_197 = _min_260;
            kb2[2] = hi_196;
            kb2[6] = lo_197;
            float _fmax_296 = fmaxf(kb2[5], kb2[8]);
            float hi_199_1 = _fmax_296;
            float _min_261 = fminf(kb2[5], kb2[8]);
            float lo_200_1 = _min_261;
            kb2[5] = hi_199_1;
            kb2[8] = lo_200_1;
            float _fmax_297 = fmaxf(kb2[7], kb2[10]);
            float hi_202 = _fmax_297;
            float _min_262 = fminf(kb2[7], kb2[10]);
            float lo_203 = _min_262;
            kb2[7] = hi_202;
            kb2[10] = lo_203;
            float _fmax_298 = fmaxf(kb2[9], kb2[13]);
            float hi_204 = _fmax_298;
            float _min_263 = fminf(kb2[9], kb2[13]);
            float lo_205 = _min_263;
            kb2[9] = hi_204;
            kb2[13] = lo_205;
            float _fmax_299 = fmaxf(kb2[11], kb2[14]);
            float hi_206 = _fmax_299;
            float _min_264 = fminf(kb2[11], kb2[14]);
            float lo_207 = _min_264;
            kb2[11] = hi_206;
            kb2[14] = lo_207;
            float _fmax_300 = fmaxf(kb2[2], kb2[4]);
            float hi_209_1 = _fmax_300;
            float _min_265 = fminf(kb2[2], kb2[4]);
            float lo_210_1 = _min_265;
            kb2[2] = hi_209_1;
            kb2[4] = lo_210_1;
            float _fmax_301 = fmaxf(kb2[3], kb2[6]);
            float hi_212 = _fmax_301;
            float _min_266 = fminf(kb2[3], kb2[6]);
            float lo_213 = _min_266;
            kb2[3] = hi_212;
            kb2[6] = lo_213;
            float _fmax_302 = fmaxf(kb2[9], kb2[12]);
            float hi_215_1 = _fmax_302;
            float _min_267 = fminf(kb2[9], kb2[12]);
            float lo_216_1 = _min_267;
            kb2[9] = hi_215_1;
            kb2[12] = lo_216_1;
            float _fmax_303 = fmaxf(kb2[11], kb2[13]);
            float hi_218 = _fmax_303;
            float _min_268 = fminf(kb2[11], kb2[13]);
            float lo_219 = _min_268;
            kb2[11] = hi_218;
            kb2[13] = lo_219;
            float _fmax_304 = fmaxf(kb2[3], kb2[5]);
            float hi_220 = _fmax_304;
            float _min_269 = fminf(kb2[3], kb2[5]);
            float lo_221 = _min_269;
            kb2[3] = hi_220;
            kb2[5] = lo_221;
            float _fmax_305 = fmaxf(kb2[6], kb2[8]);
            float hi_222 = _fmax_305;
            float _min_270 = fminf(kb2[6], kb2[8]);
            float lo_223 = _min_270;
            kb2[6] = hi_222;
            kb2[8] = lo_223;
            float _fmax_306 = fmaxf(kb2[7], kb2[9]);
            float hi_225_1 = _fmax_306;
            float _min_271 = fminf(kb2[7], kb2[9]);
            float lo_226_1 = _min_271;
            kb2[7] = hi_225_1;
            kb2[9] = lo_226_1;
            float _fmax_307 = fmaxf(kb2[10], kb2[12]);
            float hi_228 = _fmax_307;
            float _min_272 = fminf(kb2[10], kb2[12]);
            float lo_229 = _min_272;
            kb2[10] = hi_228;
            kb2[12] = lo_229;
            float _fmax_308 = fmaxf(kb2[3], kb2[4]);
            float hi_231_1 = _fmax_308;
            float _min_273 = fminf(kb2[3], kb2[4]);
            float lo_232_1 = _min_273;
            kb2[3] = hi_231_1;
            kb2[4] = lo_232_1;
            float _fmax_309 = fmaxf(kb2[5], kb2[6]);
            float hi_234 = _fmax_309;
            float _min_274 = fminf(kb2[5], kb2[6]);
            float lo_235 = _min_274;
            kb2[5] = hi_234;
            kb2[6] = lo_235;
            float _fmax_310 = fmaxf(kb2[7], kb2[8]);
            float hi_236 = _fmax_310;
            float _min_275 = fminf(kb2[7], kb2[8]);
            float lo_237 = _min_275;
            kb2[7] = hi_236;
            kb2[8] = lo_237;
            float _fmax_311 = fmaxf(kb2[9], kb2[10]);
            float hi_238 = _fmax_311;
            float _min_276 = fminf(kb2[9], kb2[10]);
            float lo_239 = _min_276;
            kb2[9] = hi_238;
            kb2[10] = lo_239;
            float _fmax_312 = fmaxf(kb2[11], kb2[12]);
            float hi_241_1 = _fmax_312;
            float _min_277 = fminf(kb2[11], kb2[12]);
            float lo_242_1 = _min_277;
            kb2[11] = hi_241_1;
            kb2[12] = lo_242_1;
            float _fmax_313 = fmaxf(kb2[6], kb2[7]);
            float hi_244 = _fmax_313;
            float _min_278 = fminf(kb2[6], kb2[7]);
            float lo_245 = _min_278;
            kb2[6] = hi_244;
            kb2[7] = lo_245;
            float _fmax_314 = fmaxf(kb2[8], kb2[9]);
            float hi_247_1 = _fmax_314;
            float _min_279 = fminf(kb2[8], kb2[9]);
            float lo_248_1 = _min_279;
            kb2[8] = hi_247_1;
            kb2[9] = lo_248_1;
            float _fmax_315 = fmaxf(a2[0], kb2[15]);
            float hi_249_1 = _fmax_315;
            a2[0] = hi_249_1;
            float _fmax_316 = fmaxf(a2[1], kb2[14]);
            float hi_250 = _fmax_316;
            a2[1] = hi_250;
            float _fmax_317 = fmaxf(a2[2], kb2[13]);
            float hi_251_1 = _fmax_317;
            a2[2] = hi_251_1;
            float _fmax_318 = fmaxf(a2[3], kb2[12]);
            float hi_252 = _fmax_318;
            a2[3] = hi_252;
            float _fmax_319 = fmaxf(a2[4], kb2[11]);
            float hi_253_1 = _fmax_319;
            a2[4] = hi_253_1;
            float _fmax_320 = fmaxf(a2[5], kb2[10]);
            float hi_254 = _fmax_320;
            a2[5] = hi_254;
            float _fmax_321 = fmaxf(a2[6], kb2[9]);
            float hi_255_1 = _fmax_321;
            a2[6] = hi_255_1;
            float _fmax_322 = fmaxf(a2[7], kb2[8]);
            float hi_256 = _fmax_322;
            a2[7] = hi_256;
            float _fmax_323 = fmaxf(a2[8], kb2[7]);
            float hi_257_1 = _fmax_323;
            a2[8] = hi_257_1;
            float _fmax_324 = fmaxf(a2[9], kb2[6]);
            float hi_258 = _fmax_324;
            a2[9] = hi_258;
            float _fmax_325 = fmaxf(a2[10], kb2[5]);
            float hi_259_1 = _fmax_325;
            a2[10] = hi_259_1;
            float _fmax_326 = fmaxf(a2[11], kb2[4]);
            float hi_260 = _fmax_326;
            a2[11] = hi_260;
            float _fmax_327 = fmaxf(a2[12], kb2[3]);
            float hi_261_1 = _fmax_327;
            a2[12] = hi_261_1;
            float _fmax_328 = fmaxf(a2[13], kb2[2]);
            float hi_262 = _fmax_328;
            a2[13] = hi_262;
            float _fmax_329 = fmaxf(a2[14], kb2[1]);
            float hi_263 = _fmax_329;
            a2[14] = hi_263;
            float _fmax_330 = fmaxf(a2[15], kb2[0]);
            float hi_264 = _fmax_330;
            a2[15] = hi_264;
            float _fmax_331 = fmaxf(a2[0], a2[8]);
            float hi_265 = _fmax_331;
            float _min_280 = fminf(a2[0], a2[8]);
            float lo_266 = _min_280;
            a2[0] = hi_265;
            a2[8] = lo_266;
            float _fmax_332 = fmaxf(a2[1], a2[9]);
            float hi_267 = _fmax_332;
            float _min_281 = fminf(a2[1], a2[9]);
            float lo_268 = _min_281;
            a2[1] = hi_267;
            a2[9] = lo_268;
            float _fmax_333 = fmaxf(a2[2], a2[10]);
            float hi_269 = _fmax_333;
            float _min_282 = fminf(a2[2], a2[10]);
            float lo_270 = _min_282;
            a2[2] = hi_269;
            a2[10] = lo_270;
            float _fmax_334 = fmaxf(a2[3], a2[11]);
            float hi_271 = _fmax_334;
            float _min_283 = fminf(a2[3], a2[11]);
            float lo_272 = _min_283;
            a2[3] = hi_271;
            a2[11] = lo_272;
            float _fmax_335 = fmaxf(a2[4], a2[12]);
            float hi_273 = _fmax_335;
            float _min_284 = fminf(a2[4], a2[12]);
            float lo_274 = _min_284;
            a2[4] = hi_273;
            a2[12] = lo_274;
            float _fmax_336 = fmaxf(a2[5], a2[13]);
            float hi_275 = _fmax_336;
            float _min_285 = fminf(a2[5], a2[13]);
            float lo_276 = _min_285;
            a2[5] = hi_275;
            a2[13] = lo_276;
            float _fmax_337 = fmaxf(a2[6], a2[14]);
            float hi_277 = _fmax_337;
            float _min_286 = fminf(a2[6], a2[14]);
            float lo_278 = _min_286;
            a2[6] = hi_277;
            a2[14] = lo_278;
            float _fmax_338 = fmaxf(a2[7], a2[15]);
            float hi_279 = _fmax_338;
            float _min_287 = fminf(a2[7], a2[15]);
            float lo_280 = _min_287;
            a2[7] = hi_279;
            a2[15] = lo_280;
            float _fmax_339 = fmaxf(a2[0], a2[4]);
            float hi_281 = _fmax_339;
            float _min_288 = fminf(a2[0], a2[4]);
            float lo_282 = _min_288;
            a2[0] = hi_281;
            a2[4] = lo_282;
            float _fmax_340 = fmaxf(a2[1], a2[5]);
            float hi_283 = _fmax_340;
            float _min_289 = fminf(a2[1], a2[5]);
            float lo_284 = _min_289;
            a2[1] = hi_283;
            a2[5] = lo_284;
            float _fmax_341 = fmaxf(a2[2], a2[6]);
            float hi_285 = _fmax_341;
            float _min_290 = fminf(a2[2], a2[6]);
            float lo_286 = _min_290;
            a2[2] = hi_285;
            a2[6] = lo_286;
            float _fmax_342 = fmaxf(a2[3], a2[7]);
            float hi_287 = _fmax_342;
            float _min_291 = fminf(a2[3], a2[7]);
            float lo_288 = _min_291;
            a2[3] = hi_287;
            a2[7] = lo_288;
            float _fmax_343 = fmaxf(a2[8], a2[12]);
            float hi_289 = _fmax_343;
            float _min_292 = fminf(a2[8], a2[12]);
            float lo_290 = _min_292;
            a2[8] = hi_289;
            a2[12] = lo_290;
            float _fmax_344 = fmaxf(a2[9], a2[13]);
            float hi_291 = _fmax_344;
            float _min_293 = fminf(a2[9], a2[13]);
            float lo_292 = _min_293;
            a2[9] = hi_291;
            a2[13] = lo_292;
            float _fmax_345 = fmaxf(a2[10], a2[14]);
            float hi_293 = _fmax_345;
            float _min_294 = fminf(a2[10], a2[14]);
            float lo_294 = _min_294;
            a2[10] = hi_293;
            a2[14] = lo_294;
            float _fmax_346 = fmaxf(a2[11], a2[15]);
            float hi_295 = _fmax_346;
            float _min_295 = fminf(a2[11], a2[15]);
            float lo_296 = _min_295;
            a2[11] = hi_295;
            a2[15] = lo_296;
            float _fmax_347 = fmaxf(a2[0], a2[2]);
            float hi_297 = _fmax_347;
            float _min_296 = fminf(a2[0], a2[2]);
            float lo_298 = _min_296;
            a2[0] = hi_297;
            a2[2] = lo_298;
            float _fmax_348 = fmaxf(a2[1], a2[3]);
            float hi_299 = _fmax_348;
            float _min_297 = fminf(a2[1], a2[3]);
            float lo_300 = _min_297;
            a2[1] = hi_299;
            a2[3] = lo_300;
            float _fmax_349 = fmaxf(a2[4], a2[6]);
            float hi_301 = _fmax_349;
            float _min_298 = fminf(a2[4], a2[6]);
            float lo_302 = _min_298;
            a2[4] = hi_301;
            a2[6] = lo_302;
            float _fmax_350 = fmaxf(a2[5], a2[7]);
            float hi_303 = _fmax_350;
            float _min_299 = fminf(a2[5], a2[7]);
            float lo_304 = _min_299;
            a2[5] = hi_303;
            a2[7] = lo_304;
            float _fmax_351 = fmaxf(a2[8], a2[10]);
            float hi_305 = _fmax_351;
            float _min_300 = fminf(a2[8], a2[10]);
            float lo_306 = _min_300;
            a2[8] = hi_305;
            a2[10] = lo_306;
            float _fmax_352 = fmaxf(a2[9], a2[11]);
            float hi_307 = _fmax_352;
            float _min_301 = fminf(a2[9], a2[11]);
            float lo_308 = _min_301;
            a2[9] = hi_307;
            a2[11] = lo_308;
            float _fmax_353 = fmaxf(a2[12], a2[14]);
            float hi_309 = _fmax_353;
            float _min_302 = fminf(a2[12], a2[14]);
            float lo_310 = _min_302;
            a2[12] = hi_309;
            a2[14] = lo_310;
            float _fmax_354 = fmaxf(a2[13], a2[15]);
            float hi_311 = _fmax_354;
            float _min_303 = fminf(a2[13], a2[15]);
            float lo_312 = _min_303;
            a2[13] = hi_311;
            a2[15] = lo_312;
            float _fmax_355 = fmaxf(a2[0], a2[1]);
            float hi_313 = _fmax_355;
            float _min_304 = fminf(a2[0], a2[1]);
            float lo_314 = _min_304;
            a2[0] = hi_313;
            a2[1] = lo_314;
            float _fmax_356 = fmaxf(a2[2], a2[3]);
            float hi_315 = _fmax_356;
            float _min_305 = fminf(a2[2], a2[3]);
            float lo_316 = _min_305;
            a2[2] = hi_315;
            a2[3] = lo_316;
            float _fmax_357 = fmaxf(a2[4], a2[5]);
            float hi_317 = _fmax_357;
            float _min_306 = fminf(a2[4], a2[5]);
            float lo_318 = _min_306;
            a2[4] = hi_317;
            a2[5] = lo_318;
            float _fmax_358 = fmaxf(a2[6], a2[7]);
            float hi_319 = _fmax_358;
            float _min_307 = fminf(a2[6], a2[7]);
            float lo_320 = _min_307;
            a2[6] = hi_319;
            a2[7] = lo_320;
            float _fmax_359 = fmaxf(a2[8], a2[9]);
            float hi_321 = _fmax_359;
            float _min_308 = fminf(a2[8], a2[9]);
            float lo_322 = _min_308;
            a2[8] = hi_321;
            a2[9] = lo_322;
            float _fmax_360 = fmaxf(a2[10], a2[11]);
            float hi_323 = _fmax_360;
            float _min_309 = fminf(a2[10], a2[11]);
            float lo_324 = _min_309;
            a2[10] = hi_323;
            a2[11] = lo_324;
            float _fmax_361 = fmaxf(a2[12], a2[13]);
            float hi_325 = _fmax_361;
            float _min_310 = fminf(a2[12], a2[13]);
            float lo_326 = _min_310;
            a2[12] = hi_325;
            a2[13] = lo_326;
            float _fmax_362 = fmaxf(a2[14], a2[15]);
            float hi_327 = _fmax_362;
            float _min_311 = fminf(a2[14], a2[15]);
            float lo_328 = _min_311;
            a2[14] = hi_327;
            a2[15] = lo_328;
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
        float rr2[1];
        int pb_2 = tid_1 * 17;
        pub[pb_2] = a2[0];
        pub[pb_2 + 1] = a2[1];
        pub[pb_2 + 2] = a2[2];
        pub[pb_2 + 3] = a2[3];
        pub[pb_2 + 4] = a2[4];
        pub[pb_2 + 5] = a2[5];
        pub[pb_2 + 6] = a2[6];
        pub[pb_2 + 7] = a2[7];
        pub[pb_2 + 8] = a2[8];
        pub[pb_2 + 9] = a2[9];
        pub[pb_2 + 10] = a2[10];
        pub[pb_2 + 11] = a2[11];
        pub[pb_2 + 12] = a2[12];
        pub[pb_2 + 13] = a2[13];
        pub[pb_2 + 14] = a2[14];
        pub[pb_2 + 15] = a2[15];
        pub[pb_2 + 16] = neg_inf;
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float r_3 = neg_inf;
        float V_4[8];
        int s0_5 = (sg * 16 * 8 + cg) * 17;
        int s1_6 = ((sg * 16 + 8) * 8 + cg) * 17;
        float x0_7 = pub[s0_5 + ln];
        float y0_8 = pub[s1_6 + lnr];
        float _min_312 = fminf(x0_7, y0_8);
        float lo0_9 = _min_312;
        float _fmax_363 = fmaxf(r_3, lo0_9);
        r_3 = _fmax_363;
        float _fmax_364 = fmaxf(x0_7, y0_8);
        float hi0_10 = _fmax_364;
        float cur_11 = hi0_10;
        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 8);
        float pv_12 = _shfl_xor_75;
        float _fmax_365 = fmaxf(cur_11, pv_12);
        float hi_13 = _fmax_365;
        float _min_313 = fminf(cur_11, pv_12);
        float lo_14 = _min_313;
        cur_11 = ((up[0] != 0) ? hi_13 : lo_14);
        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 4);
        float pv_15 = _shfl_xor_76;
        float _fmax_366 = fmaxf(cur_11, pv_15);
        float hi_16 = _fmax_366;
        float _min_314 = fminf(cur_11, pv_15);
        float lo_17 = _min_314;
        cur_11 = ((up[1] != 0) ? hi_16 : lo_17);
        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 2);
        float pv_18 = _shfl_xor_77;
        float _fmax_367 = fmaxf(cur_11, pv_18);
        float hi_19 = _fmax_367;
        float _min_315 = fminf(cur_11, pv_18);
        float lo_20 = _min_315;
        cur_11 = ((up[2] != 0) ? hi_19 : lo_20);
        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_11, 1);
        float pv_21 = _shfl_xor_78;
        float _fmax_368 = fmaxf(cur_11, pv_21);
        float hi_22 = _fmax_368;
        float _min_316 = fminf(cur_11, pv_21);
        float lo_23 = _min_316;
        cur_11 = ((up[3] != 0) ? hi_22 : lo_23);
        V_4[0] = cur_11;
        int s0_24 = ((sg * 16 + 1) * 8 + cg) * 17;
        int s1_25 = ((sg * 16 + 1 + 8) * 8 + cg) * 17;
        float x0_26 = pub[s0_24 + ln];
        float y0_27 = pub[s1_25 + lnr];
        float _min_317 = fminf(x0_26, y0_27);
        float lo0_28 = _min_317;
        float _fmax_369 = fmaxf(r_3, lo0_28);
        r_3 = _fmax_369;
        float _fmax_370 = fmaxf(x0_26, y0_27);
        float hi0_29 = _fmax_370;
        float cur_30 = hi0_29;
        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 8);
        float pv_31 = _shfl_xor_79;
        float _fmax_371 = fmaxf(cur_30, pv_31);
        float hi_32 = _fmax_371;
        float _min_318 = fminf(cur_30, pv_31);
        float lo_33 = _min_318;
        cur_30 = ((up[0] != 0) ? hi_32 : lo_33);
        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 4);
        float pv_34 = _shfl_xor_80;
        float _fmax_372 = fmaxf(cur_30, pv_34);
        float hi_35 = _fmax_372;
        float _min_319 = fminf(cur_30, pv_34);
        float lo_36 = _min_319;
        cur_30 = ((up[1] != 0) ? hi_35 : lo_36);
        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 2);
        float pv_37 = _shfl_xor_81;
        float _fmax_373 = fmaxf(cur_30, pv_37);
        float hi_38 = _fmax_373;
        float _min_320 = fminf(cur_30, pv_37);
        float lo_39 = _min_320;
        cur_30 = ((up[2] != 0) ? hi_38 : lo_39);
        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_30, 1);
        float pv_40 = _shfl_xor_82;
        float _fmax_374 = fmaxf(cur_30, pv_40);
        float hi_41 = _fmax_374;
        float _min_321 = fminf(cur_30, pv_40);
        float lo_42 = _min_321;
        cur_30 = ((up[3] != 0) ? hi_41 : lo_42);
        V_4[1] = cur_30;
        int s0_43 = ((sg * 16 + 2) * 8 + cg) * 17;
        int s1_44 = ((sg * 16 + 2 + 8) * 8 + cg) * 17;
        float x0_45 = pub[s0_43 + ln];
        float y0_46 = pub[s1_44 + lnr];
        float _min_322 = fminf(x0_45, y0_46);
        float lo0_47 = _min_322;
        float _fmax_375 = fmaxf(r_3, lo0_47);
        r_3 = _fmax_375;
        float _fmax_376 = fmaxf(x0_45, y0_46);
        float hi0_48 = _fmax_376;
        float cur_49 = hi0_48;
        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 8);
        float pv_50 = _shfl_xor_83;
        float _fmax_377 = fmaxf(cur_49, pv_50);
        float hi_51_1 = _fmax_377;
        float _min_323 = fminf(cur_49, pv_50);
        float lo_52_1 = _min_323;
        cur_49 = ((up[0] != 0) ? hi_51_1 : lo_52_1);
        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 4);
        float pv_53 = _shfl_xor_84;
        float _fmax_378 = fmaxf(cur_49, pv_53);
        float hi_54 = _fmax_378;
        float _min_324 = fminf(cur_49, pv_53);
        float lo_55 = _min_324;
        cur_49 = ((up[1] != 0) ? hi_54 : lo_55);
        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 2);
        float pv_56 = _shfl_xor_85;
        float _fmax_379 = fmaxf(cur_49, pv_56);
        float hi_57_1 = _fmax_379;
        float _min_325 = fminf(cur_49, pv_56);
        float lo_58_1 = _min_325;
        cur_49 = ((up[2] != 0) ? hi_57_1 : lo_58_1);
        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_49, 1);
        float pv_59 = _shfl_xor_86;
        float _fmax_380 = fmaxf(cur_49, pv_59);
        float hi_60 = _fmax_380;
        float _min_326 = fminf(cur_49, pv_59);
        float lo_61 = _min_326;
        cur_49 = ((up[3] != 0) ? hi_60 : lo_61);
        V_4[2] = cur_49;
        int s0_62 = ((sg * 16 + 3) * 8 + cg) * 17;
        int s1_63 = ((sg * 16 + 3 + 8) * 8 + cg) * 17;
        float x0_64 = pub[s0_62 + ln];
        float y0_65 = pub[s1_63 + lnr];
        float _min_327 = fminf(x0_64, y0_65);
        float lo0_66 = _min_327;
        float _fmax_381 = fmaxf(r_3, lo0_66);
        r_3 = _fmax_381;
        float _fmax_382 = fmaxf(x0_64, y0_65);
        float hi0_67 = _fmax_382;
        float cur_68 = hi0_67;
        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 8);
        float pv_69 = _shfl_xor_87;
        float _fmax_383 = fmaxf(cur_68, pv_69);
        float hi_70 = _fmax_383;
        float _min_328 = fminf(cur_68, pv_69);
        float lo_71 = _min_328;
        cur_68 = ((up[0] != 0) ? hi_70 : lo_71);
        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 4);
        float pv_72 = _shfl_xor_88;
        float _fmax_384 = fmaxf(cur_68, pv_72);
        float hi_73_1 = _fmax_384;
        float _min_329 = fminf(cur_68, pv_72);
        float lo_74_1 = _min_329;
        cur_68 = ((up[1] != 0) ? hi_73_1 : lo_74_1);
        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 2);
        float pv_75 = _shfl_xor_89;
        float _fmax_385 = fmaxf(cur_68, pv_75);
        float hi_76 = _fmax_385;
        float _min_330 = fminf(cur_68, pv_75);
        float lo_77 = _min_330;
        cur_68 = ((up[2] != 0) ? hi_76 : lo_77);
        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_68, 1);
        float pv_78 = _shfl_xor_90;
        float _fmax_386 = fmaxf(cur_68, pv_78);
        float hi_79_1 = _fmax_386;
        float _min_331 = fminf(cur_68, pv_78);
        float lo_80_1 = _min_331;
        cur_68 = ((up[3] != 0) ? hi_79_1 : lo_80_1);
        V_4[3] = cur_68;
        int s0_81 = ((sg * 16 + 4) * 8 + cg) * 17;
        int s1_82 = ((sg * 16 + 4 + 8) * 8 + cg) * 17;
        float x0_83 = pub[s0_81 + ln];
        float y0_84 = pub[s1_82 + lnr];
        float _min_332 = fminf(x0_83, y0_84);
        float lo0_85 = _min_332;
        float _fmax_387 = fmaxf(r_3, lo0_85);
        r_3 = _fmax_387;
        float _fmax_388 = fmaxf(x0_83, y0_84);
        float hi0_86 = _fmax_388;
        float cur_87 = hi0_86;
        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 8);
        float pv_88 = _shfl_xor_91;
        float _fmax_389 = fmaxf(cur_87, pv_88);
        float hi_89_1 = _fmax_389;
        float _min_333 = fminf(cur_87, pv_88);
        float lo_90_1 = _min_333;
        cur_87 = ((up[0] != 0) ? hi_89_1 : lo_90_1);
        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 4);
        float pv_91 = _shfl_xor_92;
        float _fmax_390 = fmaxf(cur_87, pv_91);
        float hi_92 = _fmax_390;
        float _min_334 = fminf(cur_87, pv_91);
        float lo_93 = _min_334;
        cur_87 = ((up[1] != 0) ? hi_92 : lo_93);
        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 2);
        float pv_94 = _shfl_xor_93;
        float _fmax_391 = fmaxf(cur_87, pv_94);
        float hi_95_2 = _fmax_391;
        float _min_335 = fminf(cur_87, pv_94);
        float lo_96_2 = _min_335;
        cur_87 = ((up[2] != 0) ? hi_95_2 : lo_96_2);
        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, cur_87, 1);
        float pv_97 = _shfl_xor_94;
        float _fmax_392 = fmaxf(cur_87, pv_97);
        float hi_98 = _fmax_392;
        float _min_336 = fminf(cur_87, pv_97);
        float lo_99 = _min_336;
        cur_87 = ((up[3] != 0) ? hi_98 : lo_99);
        V_4[4] = cur_87;
        int s0_100 = ((sg * 16 + 5) * 8 + cg) * 17;
        int s1_101 = ((sg * 16 + 5 + 8) * 8 + cg) * 17;
        float x0_102 = pub[s0_100 + ln];
        float y0_103 = pub[s1_101 + lnr];
        float _min_337 = fminf(x0_102, y0_103);
        float lo0_104 = _min_337;
        float _fmax_393 = fmaxf(r_3, lo0_104);
        r_3 = _fmax_393;
        float _fmax_394 = fmaxf(x0_102, y0_103);
        float hi0_105 = _fmax_394;
        float cur_106 = hi0_105;
        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 8);
        float pv_107 = _shfl_xor_95;
        float _fmax_395 = fmaxf(cur_106, pv_107);
        float hi_108 = _fmax_395;
        float _min_338 = fminf(cur_106, pv_107);
        float lo_109 = _min_338;
        cur_106 = ((up[0] != 0) ? hi_108 : lo_109);
        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 4);
        float pv_110 = _shfl_xor_96;
        float _fmax_396 = fmaxf(cur_106, pv_110);
        float hi_111_2 = _fmax_396;
        float _min_339 = fminf(cur_106, pv_110);
        float lo_112_2 = _min_339;
        cur_106 = ((up[1] != 0) ? hi_111_2 : lo_112_2);
        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 2);
        float pv_113 = _shfl_xor_97;
        float _fmax_397 = fmaxf(cur_106, pv_113);
        float hi_114 = _fmax_397;
        float _min_340 = fminf(cur_106, pv_113);
        float lo_115 = _min_340;
        cur_106 = ((up[2] != 0) ? hi_114 : lo_115);
        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, cur_106, 1);
        float pv_116 = _shfl_xor_98;
        float _fmax_398 = fmaxf(cur_106, pv_116);
        float hi_117_1 = _fmax_398;
        float _min_341 = fminf(cur_106, pv_116);
        float lo_118_1 = _min_341;
        cur_106 = ((up[3] != 0) ? hi_117_1 : lo_118_1);
        V_4[5] = cur_106;
        int s0_119 = ((sg * 16 + 6) * 8 + cg) * 17;
        int s1_120 = ((sg * 16 + 6 + 8) * 8 + cg) * 17;
        float x0_121 = pub[s0_119 + ln];
        float y0_122 = pub[s1_120 + lnr];
        float _min_342 = fminf(x0_121, y0_122);
        float lo0_123 = _min_342;
        float _fmax_399 = fmaxf(r_3, lo0_123);
        r_3 = _fmax_399;
        float _fmax_400 = fmaxf(x0_121, y0_122);
        float hi0_124 = _fmax_400;
        float cur_125 = hi0_124;
        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cur_125, 8);
        float pv_126 = _shfl_xor_99;
        float _fmax_401 = fmaxf(cur_125, pv_126);
        float hi_127_1 = _fmax_401;
        float _min_343 = fminf(cur_125, pv_126);
        float lo_128_1 = _min_343;
        cur_125 = ((up[0] != 0) ? hi_127_1 : lo_128_1);
        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, cur_125, 4);
        float pv_129 = _shfl_xor_100;
        float _fmax_402 = fmaxf(cur_125, pv_129);
        float hi_130_1 = _fmax_402;
        float _min_344 = fminf(cur_125, pv_129);
        float lo_131_1 = _min_344;
        cur_125 = ((up[1] != 0) ? hi_130_1 : lo_131_1);
        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cur_125, 2);
        float pv_132 = _shfl_xor_101;
        float _fmax_403 = fmaxf(cur_125, pv_132);
        float hi_133_1 = _fmax_403;
        float _min_345 = fminf(cur_125, pv_132);
        float lo_134_1 = _min_345;
        cur_125 = ((up[2] != 0) ? hi_133_1 : lo_134_1);
        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_125, 1);
        float pv_135 = _shfl_xor_102;
        float _fmax_404 = fmaxf(cur_125, pv_135);
        float hi_136 = _fmax_404;
        float _min_346 = fminf(cur_125, pv_135);
        float lo_137 = _min_346;
        cur_125 = ((up[3] != 0) ? hi_136 : lo_137);
        V_4[6] = cur_125;
        int s0_138 = ((sg * 16 + 7) * 8 + cg) * 17;
        int s1_139 = ((sg * 16 + 7 + 8) * 8 + cg) * 17;
        float x0_140 = pub[s0_138 + ln];
        float y0_141 = pub[s1_139 + lnr];
        float _min_347 = fminf(x0_140, y0_141);
        float lo0_142 = _min_347;
        float _fmax_405 = fmaxf(r_3, lo0_142);
        r_3 = _fmax_405;
        float _fmax_406 = fmaxf(x0_140, y0_141);
        float hi0_143 = _fmax_406;
        float cur_144 = hi0_143;
        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, cur_144, 8);
        float pv_145 = _shfl_xor_103;
        float _fmax_407 = fmaxf(cur_144, pv_145);
        float hi_146 = _fmax_407;
        float _min_348 = fminf(cur_144, pv_145);
        float lo_147 = _min_348;
        cur_144 = ((up[0] != 0) ? hi_146 : lo_147);
        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, cur_144, 4);
        float pv_148 = _shfl_xor_104;
        float _fmax_408 = fmaxf(cur_144, pv_148);
        float hi_149_1 = _fmax_408;
        float _min_349 = fminf(cur_144, pv_148);
        float lo_150_1 = _min_349;
        cur_144 = ((up[1] != 0) ? hi_149_1 : lo_150_1);
        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cur_144, 2);
        float pv_151 = _shfl_xor_105;
        float _fmax_409 = fmaxf(cur_144, pv_151);
        float hi_152 = _fmax_409;
        float _min_350 = fminf(cur_144, pv_151);
        float lo_153 = _min_350;
        cur_144 = ((up[2] != 0) ? hi_152 : lo_153);
        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_144, 1);
        float pv_154 = _shfl_xor_106;
        float _fmax_410 = fmaxf(cur_144, pv_154);
        float hi_155_1 = _fmax_410;
        float _min_351 = fminf(cur_144, pv_154);
        float lo_156_1 = _min_351;
        cur_144 = ((up[3] != 0) ? hi_155_1 : lo_156_1);
        V_4[7] = cur_144;
        float rs_157 = pub[((sg * 16 + ln) * 8 + cg) * 17 + 16];
        float _fmax_411 = fmaxf(r_3, rs_157);
        r_3 = _fmax_411;
        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, V_4[4], 15);
        float y1_158 = _shfl_xor_107;
        float _min_352 = fminf(V_4[0], y1_158);
        float lo1_159 = _min_352;
        float _fmax_412 = fmaxf(r_3, lo1_159);
        r_3 = _fmax_412;
        float _fmax_413 = fmaxf(V_4[0], y1_158);
        float hi1_160 = _fmax_413;
        float cur_161 = hi1_160;
        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, cur_161, 8);
        float pv_163 = _shfl_xor_108;
        float _fmax_414 = fmaxf(cur_161, pv_163);
        float hi_164_1 = _fmax_414;
        float _min_353 = fminf(cur_161, pv_163);
        float lo_165_1 = _min_353;
        cur_161 = ((up[0] != 0) ? hi_164_1 : lo_165_1);
        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cur_161, 4);
        float pv_166 = _shfl_xor_109;
        float _fmax_415 = fmaxf(cur_161, pv_166);
        float hi_167_2 = _fmax_415;
        float _min_354 = fminf(cur_161, pv_166);
        float lo_168_2 = _min_354;
        cur_161 = ((up[1] != 0) ? hi_167_2 : lo_168_2);
        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_161, 2);
        float pv_169 = _shfl_xor_110;
        float _fmax_416 = fmaxf(cur_161, pv_169);
        float hi_170_1 = _fmax_416;
        float _min_355 = fminf(cur_161, pv_169);
        float lo_171_1 = _min_355;
        cur_161 = ((up[2] != 0) ? hi_170_1 : lo_171_1);
        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cur_161, 1);
        float pv_172 = _shfl_xor_111;
        float _fmax_417 = fmaxf(cur_161, pv_172);
        float hi_173_1 = _fmax_417;
        float _min_356 = fminf(cur_161, pv_172);
        float lo_174_1 = _min_356;
        cur_161 = ((up[3] != 0) ? hi_173_1 : lo_174_1);
        V_4[0] = cur_161;
        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, V_4[5], 15);
        float y1_175 = _shfl_xor_112;
        float _min_357 = fminf(V_4[1], y1_175);
        float lo1_176 = _min_357;
        float _fmax_418 = fmaxf(r_3, lo1_176);
        r_3 = _fmax_418;
        float _fmax_419 = fmaxf(V_4[1], y1_175);
        float hi1_177 = _fmax_419;
        float cur_178 = hi1_177;
        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, cur_178, 8);
        float pv_179 = _shfl_xor_113;
        float _fmax_420 = fmaxf(cur_178, pv_179);
        float hi_180_1 = _fmax_420;
        float _min_358 = fminf(cur_178, pv_179);
        float lo_181_1 = _min_358;
        cur_178 = ((up[0] != 0) ? hi_180_1 : lo_181_1);
        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, cur_178, 4);
        float pv_182 = _shfl_xor_114;
        float _fmax_421 = fmaxf(cur_178, pv_182);
        float hi_183_2 = _fmax_421;
        float _min_359 = fminf(cur_178, pv_182);
        float lo_184_2 = _min_359;
        cur_178 = ((up[1] != 0) ? hi_183_2 : lo_184_2);
        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, cur_178, 2);
        float pv_185 = _shfl_xor_115;
        float _fmax_422 = fmaxf(cur_178, pv_185);
        float hi_186_1 = _fmax_422;
        float _min_360 = fminf(cur_178, pv_185);
        float lo_187_1 = _min_360;
        cur_178 = ((up[2] != 0) ? hi_186_1 : lo_187_1);
        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_178, 1);
        float pv_188 = _shfl_xor_116;
        float _fmax_423 = fmaxf(cur_178, pv_188);
        float hi_189_1 = _fmax_423;
        float _min_361 = fminf(cur_178, pv_188);
        float lo_190_1 = _min_361;
        cur_178 = ((up[3] != 0) ? hi_189_1 : lo_190_1);
        V_4[1] = cur_178;
        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, V_4[6], 15);
        float y1_191 = _shfl_xor_117;
        float _min_362 = fminf(V_4[2], y1_191);
        float lo1_192 = _min_362;
        float _fmax_424 = fmaxf(r_3, lo1_192);
        r_3 = _fmax_424;
        float _fmax_425 = fmaxf(V_4[2], y1_191);
        float hi1_193 = _fmax_425;
        float cur_194 = hi1_193;
        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 8);
        float pv_195 = _shfl_xor_118;
        float _fmax_426 = fmaxf(cur_194, pv_195);
        float hi_196_1 = _fmax_426;
        float _min_363 = fminf(cur_194, pv_195);
        float lo_197_1 = _min_363;
        cur_194 = ((up[0] != 0) ? hi_196_1 : lo_197_1);
        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 4);
        float pv_198 = _shfl_xor_119;
        float _fmax_427 = fmaxf(cur_194, pv_198);
        float hi_199_2 = _fmax_427;
        float _min_364 = fminf(cur_194, pv_198);
        float lo_200_2 = _min_364;
        cur_194 = ((up[1] != 0) ? hi_199_2 : lo_200_2);
        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 2);
        float pv_201 = _shfl_xor_120;
        float _fmax_428 = fmaxf(cur_194, pv_201);
        float hi_202_1 = _fmax_428;
        float _min_365 = fminf(cur_194, pv_201);
        float lo_203_1 = _min_365;
        cur_194 = ((up[2] != 0) ? hi_202_1 : lo_203_1);
        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cur_194, 1);
        float pv_204 = _shfl_xor_121;
        float _fmax_429 = fmaxf(cur_194, pv_204);
        float hi_205_1 = _fmax_429;
        float _min_366 = fminf(cur_194, pv_204);
        float lo_206_1 = _min_366;
        cur_194 = ((up[3] != 0) ? hi_205_1 : lo_206_1);
        V_4[2] = cur_194;
        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, V_4[7], 15);
        float y1_207 = _shfl_xor_122;
        float _min_367 = fminf(V_4[3], y1_207);
        float lo1_208 = _min_367;
        float _fmax_430 = fmaxf(r_3, lo1_208);
        r_3 = _fmax_430;
        float _fmax_431 = fmaxf(V_4[3], y1_207);
        float hi1_209 = _fmax_431;
        float cur_210 = hi1_209;
        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 8);
        float pv_211 = _shfl_xor_123;
        float _fmax_432 = fmaxf(cur_210, pv_211);
        float hi_212_1 = _fmax_432;
        float _min_368 = fminf(cur_210, pv_211);
        float lo_213_1 = _min_368;
        cur_210 = ((up[0] != 0) ? hi_212_1 : lo_213_1);
        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 4);
        float pv_214 = _shfl_xor_124;
        float _fmax_433 = fmaxf(cur_210, pv_214);
        float hi_215_2 = _fmax_433;
        float _min_369 = fminf(cur_210, pv_214);
        float lo_216_2 = _min_369;
        cur_210 = ((up[1] != 0) ? hi_215_2 : lo_216_2);
        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 2);
        float pv_217 = _shfl_xor_125;
        float _fmax_434 = fmaxf(cur_210, pv_217);
        float hi_218_1 = _fmax_434;
        float _min_370 = fminf(cur_210, pv_217);
        float lo_219_1 = _min_370;
        cur_210 = ((up[2] != 0) ? hi_218_1 : lo_219_1);
        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, cur_210, 1);
        float pv_220 = _shfl_xor_126;
        float _fmax_435 = fmaxf(cur_210, pv_220);
        float hi_221_1 = _fmax_435;
        float _min_371 = fminf(cur_210, pv_220);
        float lo_222_1 = _min_371;
        cur_210 = ((up[3] != 0) ? hi_221_1 : lo_222_1);
        V_4[3] = cur_210;
        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, V_4[2], 15);
        float y1_223 = _shfl_xor_127;
        float _min_372 = fminf(V_4[0], y1_223);
        float lo1_224 = _min_372;
        float _fmax_436 = fmaxf(r_3, lo1_224);
        r_3 = _fmax_436;
        float _fmax_437 = fmaxf(V_4[0], y1_223);
        float hi1_225 = _fmax_437;
        float cur_226 = hi1_225;
        float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 8);
        float pv_227 = _shfl_xor_128;
        float _fmax_438 = fmaxf(cur_226, pv_227);
        float hi_228_1 = _fmax_438;
        float _min_373 = fminf(cur_226, pv_227);
        float lo_229_1 = _min_373;
        cur_226 = ((up[0] != 0) ? hi_228_1 : lo_229_1);
        float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 4);
        float pv_230 = _shfl_xor_129;
        float _fmax_439 = fmaxf(cur_226, pv_230);
        float hi_231_2 = _fmax_439;
        float _min_374 = fminf(cur_226, pv_230);
        float lo_232_2 = _min_374;
        cur_226 = ((up[1] != 0) ? hi_231_2 : lo_232_2);
        float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 2);
        float pv_233 = _shfl_xor_130;
        float _fmax_440 = fmaxf(cur_226, pv_233);
        float hi_234_1 = _fmax_440;
        float _min_375 = fminf(cur_226, pv_233);
        float lo_235_1 = _min_375;
        cur_226 = ((up[2] != 0) ? hi_234_1 : lo_235_1);
        float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, cur_226, 1);
        float pv_237 = _shfl_xor_131;
        float _fmax_441 = fmaxf(cur_226, pv_237);
        float hi_238_1 = _fmax_441;
        float _min_376 = fminf(cur_226, pv_237);
        float lo_239_1 = _min_376;
        cur_226 = ((up[3] != 0) ? hi_238_1 : lo_239_1);
        V_4[0] = cur_226;
        float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, V_4[3], 15);
        float y1_240 = _shfl_xor_132;
        float _min_377 = fminf(V_4[1], y1_240);
        float lo1_241 = _min_377;
        float _fmax_442 = fmaxf(r_3, lo1_241);
        r_3 = _fmax_442;
        float _fmax_443 = fmaxf(V_4[1], y1_240);
        float hi1_242 = _fmax_443;
        float cur_243 = hi1_242;
        float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 8);
        float pv_244 = _shfl_xor_133;
        float _fmax_444 = fmaxf(cur_243, pv_244);
        float hi_245_1 = _fmax_444;
        float _min_378 = fminf(cur_243, pv_244);
        float lo_246_1 = _min_378;
        cur_243 = ((up[0] != 0) ? hi_245_1 : lo_246_1);
        float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 4);
        float pv_247 = _shfl_xor_134;
        float _fmax_445 = fmaxf(cur_243, pv_247);
        float hi_248 = _fmax_445;
        float _min_379 = fminf(cur_243, pv_247);
        float lo_249 = _min_379;
        cur_243 = ((up[1] != 0) ? hi_248 : lo_249);
        float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 2);
        float pv_250 = _shfl_xor_135;
        float _fmax_446 = fmaxf(cur_243, pv_250);
        float hi_251_2 = _fmax_446;
        float _min_380 = fminf(cur_243, pv_250);
        float lo_252_1 = _min_380;
        cur_243 = ((up[2] != 0) ? hi_251_2 : lo_252_1);
        float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, cur_243, 1);
        float pv_253 = _shfl_xor_136;
        float _fmax_447 = fmaxf(cur_243, pv_253);
        float hi_254_1 = _fmax_447;
        float _min_381 = fminf(cur_243, pv_253);
        float lo_255 = _min_381;
        cur_243 = ((up[3] != 0) ? hi_254_1 : lo_255);
        V_4[1] = cur_243;
        float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, V_4[1], 15);
        float yl_256 = _shfl_xor_137;
        float _min_382 = fminf(V_4[0], yl_256);
        float lol_257 = _min_382;
        float _fmax_448 = fmaxf(r_3, lol_257);
        r_3 = _fmax_448;
        float _fmax_449 = fmaxf(V_4[0], yl_256);
        float hil_258 = _fmax_449;
        float cur_259 = hil_258;
        float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 8);
        float pv_260 = _shfl_xor_138;
        float _fmax_450 = fmaxf(cur_259, pv_260);
        float hi_261_2 = _fmax_450;
        float _min_383 = fminf(cur_259, pv_260);
        float lo_262_1 = _min_383;
        cur_259 = ((up[0] != 0) ? hi_261_2 : lo_262_1);
        float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 4);
        float pv_263 = _shfl_xor_139;
        float _fmax_451 = fmaxf(cur_259, pv_263);
        float hi_264_1 = _fmax_451;
        float _min_384 = fminf(cur_259, pv_263);
        float lo_265 = _min_384;
        cur_259 = ((up[1] != 0) ? hi_264_1 : lo_265);
        float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 2);
        float pv_266 = _shfl_xor_140;
        float _fmax_452 = fmaxf(cur_259, pv_266);
        float hi_267_1 = _fmax_452;
        float _min_385 = fminf(cur_259, pv_266);
        float lo_268_1 = _min_385;
        cur_259 = ((up[2] != 0) ? hi_267_1 : lo_268_1);
        float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, cur_259, 1);
        float pv_269 = _shfl_xor_141;
        float _fmax_453 = fmaxf(cur_259, pv_269);
        float hi_270 = _fmax_453;
        float _min_386 = fminf(cur_259, pv_269);
        float lo_271 = _min_386;
        cur_259 = ((up[3] != 0) ? hi_270 : lo_271);
        V_4[0] = cur_259;
        float K_272 = V_4[0];
        int qb_273 = (sg * 8 + cg) * 32;
        q2[qb_273 + ln] = K_272;
        q2[qb_273 + 16 + ln] = r_3;
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (tid_1 < 128) {
            float r2_1 = neg_inf;
            float _fmax_454 = fmaxf(r2_1, q2[cg * 32 + 16 + ln]);
            r2_1 = _fmax_454;
            float _fmax_455 = fmaxf(r2_1, q2[(8 + cg) * 32 + 16 + ln]);
            r2_1 = _fmax_455;
            float V2_1[1];
            float x2_1 = q2[cg * 32 + ln];
            float y2_1 = q2[(8 + cg) * 32 + lnr];
            float _min_387 = fminf(x2_1, y2_1);
            float lo2_1 = _min_387;
            float _fmax_456 = fmaxf(r2_1, lo2_1);
            r2_1 = _fmax_456;
            float _fmax_457 = fmaxf(x2_1, y2_1);
            float hi2_1 = _fmax_457;
            V2_1[0] = hi2_1;
            K_272 = V2_1[0];
            r_3 = r2_1;
        }
        rr2[0] = r_3;
        K2 = K_272;
    }
    if (tid_1 < 128) {
        unsigned int kk = __as_u32(K);
        unsigned int idx = 512;
        if (kk < 4278190080u) {
            idx = kk & 511;
        }
        if (gflag != 0) {
            unsigned int k2 = __as_u32(K2);
            idx = 512;
            if (k2 != 0) {
                idx = k2 & 511;
            }
        }
        unsigned int rkey = idx << 4 | (unsigned int)ln;
        int rank = 0;
        unsigned int _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, rkey, 1);
        unsigned int ox = _shfl_xor_142;
        if (ox < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, rkey, 2);
        unsigned int ox_0 = _shfl_xor_143;
        if (ox_0 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, rkey, 3);
        unsigned int ox_1 = _shfl_xor_144;
        if (ox_1 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, rkey, 4);
        unsigned int ox_2 = _shfl_xor_145;
        if (ox_2 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, rkey, 5);
        unsigned int ox_3 = _shfl_xor_146;
        if (ox_3 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, rkey, 6);
        unsigned int ox_4 = _shfl_xor_147;
        if (ox_4 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, rkey, 7);
        unsigned int ox_5 = _shfl_xor_148;
        if (ox_5 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, rkey, 8);
        unsigned int ox_6 = _shfl_xor_149;
        if (ox_6 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, rkey, 9);
        unsigned int ox_7 = _shfl_xor_150;
        if (ox_7 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, rkey, 10);
        unsigned int ox_8 = _shfl_xor_151;
        if (ox_8 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, rkey, 11);
        unsigned int ox_9 = _shfl_xor_152;
        if (ox_9 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, rkey, 12);
        unsigned int ox_10 = _shfl_xor_153;
        if (ox_10 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, rkey, 13);
        unsigned int ox_11 = _shfl_xor_154;
        if (ox_11 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, rkey, 14);
        unsigned int ox_12 = _shfl_xor_155;
        if (ox_12 < rkey) {
            rank = rank + 1;
        }
        unsigned int _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, rkey, 15);
        unsigned int ox_13 = _shfl_xor_156;
        if (ox_13 < rkey) {
            rank = rank + 1;
        }
        int val = -1;
        if (idx <= 511) {
            val = (int)idx;
        }
        if (commit != 0) {
            long long obase = ((long long)ucol * (long long)num_heads + (long long)head) * 16;
            out[obase + (long long)rank] = val;
        }
    }
}

} // extern "C"
