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
#define SMEM_TREE_OFF 0
#define SMEM_TREE_STAGE_BYTES 8704
#define SMEM_TREE_STRIDE 8704
#define SMEM_QCOL_OFF 8704
#define SMEM_QCOL_STAGE_BYTES 128
#define SMEM_QCOL_STRIDE 128
#define SMEM_FLAGW_OFF 8832
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_TOTAL 8960
#define THREADS 128

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

__global__ __launch_bounds__(128) void
kernel_cake_hopper_msa_5b548344ff1532b02d11(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    float* tree = reinterpret_cast<float*>(smem_raw + 0);
    const int tree_addr = smem + 0;
    unsigned int* qcol = reinterpret_cast<unsigned int*>(smem_raw + 8704);
    const int qcol_addr = smem + 8704;
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 8832);
    const int flagw_addr = smem + 8832;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 32;
    int c = tid_1 - w * 32;
    int whi = w;
    int col = blockIdx.x * 32 + c;
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
    float kb[16];
    int t0 = w * 16;
    int t0_0 = t0;
    long long p = cbase + (long long)t0_0 * nq64;
    cb[0] = 4286578688;
    if (lim_full > t0_0) {
        cb[0] = S[p];
    }
    cb[1] = 4286578688;
    if (lim_full > t0_0 + 1) {
        cb[1] = S[p + nq64];
    }
    cb[2] = 4286578688;
    if (lim_full > t0_0 + 2) {
        cb[2] = S[p + 2 * nq64];
    }
    cb[3] = 4286578688;
    if (lim_full > t0_0 + 3) {
        cb[3] = S[p + 3 * nq64];
    }
    cb[4] = 4286578688;
    if (lim_full > t0_0 + 4) {
        cb[4] = S[p + 4 * nq64];
    }
    cb[5] = 4286578688;
    if (lim_full > t0_0 + 5) {
        cb[5] = S[p + 5 * nq64];
    }
    cb[6] = 4286578688;
    if (lim_full > t0_0 + 6) {
        cb[6] = S[p + 6 * nq64];
    }
    cb[7] = 4286578688;
    if (lim_full > t0_0 + 7) {
        cb[7] = S[p + 7 * nq64];
    }
    cb[8] = 4286578688;
    if (lim_full > t0_0 + 8) {
        cb[8] = S[p + 8 * nq64];
    }
    cb[9] = 4286578688;
    if (lim_full > t0_0 + 9) {
        cb[9] = S[p + 9 * nq64];
    }
    cb[10] = 4286578688;
    if (lim_full > t0_0 + 10) {
        cb[10] = S[p + 10 * nq64];
    }
    cb[11] = 4286578688;
    if (lim_full > t0_0 + 11) {
        cb[11] = S[p + 11 * nq64];
    }
    cb[12] = 4286578688;
    if (lim_full > t0_0 + 12) {
        cb[12] = S[p + 12 * nq64];
    }
    cb[13] = 4286578688;
    if (lim_full > t0_0 + 13) {
        cb[13] = S[p + 13 * nq64];
    }
    cb[14] = 4286578688;
    if (lim_full > t0_0 + 14) {
        cb[14] = S[p + 14 * nq64];
    }
    cb[15] = 4286578688;
    if (lim_full > t0_0 + 15) {
        cb[15] = S[p + 15 * nq64];
    }
    asm volatile("" ::: "memory");
    int t0_1 = w * 16;
    int t00 = t0_1;
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
    unsigned int nb[16];
    #pragma unroll 1
    for (int j = 0; j < num_chunks; j++) {
        int t0_2 = ((j + 1) * 4 + w) * 16;
        int t0_3 = t0_2;
        long long p_4 = cbase + (long long)t0_3 * nq64;
        nb[0] = 4286578688;
        if (lim > t0_3) {
            nb[0] = S[p_4];
        }
        nb[1] = 4286578688;
        if (lim > t0_3 + 1) {
            nb[1] = S[p_4 + nq64];
        }
        nb[2] = 4286578688;
        if (lim > t0_3 + 2) {
            nb[2] = S[p_4 + 2 * nq64];
        }
        nb[3] = 4286578688;
        if (lim > t0_3 + 3) {
            nb[3] = S[p_4 + 3 * nq64];
        }
        nb[4] = 4286578688;
        if (lim > t0_3 + 4) {
            nb[4] = S[p_4 + 4 * nq64];
        }
        nb[5] = 4286578688;
        if (lim > t0_3 + 5) {
            nb[5] = S[p_4 + 5 * nq64];
        }
        nb[6] = 4286578688;
        if (lim > t0_3 + 6) {
            nb[6] = S[p_4 + 6 * nq64];
        }
        nb[7] = 4286578688;
        if (lim > t0_3 + 7) {
            nb[7] = S[p_4 + 7 * nq64];
        }
        nb[8] = 4286578688;
        if (lim > t0_3 + 8) {
            nb[8] = S[p_4 + 8 * nq64];
        }
        nb[9] = 4286578688;
        if (lim > t0_3 + 9) {
            nb[9] = S[p_4 + 9 * nq64];
        }
        nb[10] = 4286578688;
        if (lim > t0_3 + 10) {
            nb[10] = S[p_4 + 10 * nq64];
        }
        nb[11] = 4286578688;
        if (lim > t0_3 + 11) {
            nb[11] = S[p_4 + 11 * nq64];
        }
        nb[12] = 4286578688;
        if (lim > t0_3 + 12) {
            nb[12] = S[p_4 + 12 * nq64];
        }
        nb[13] = 4286578688;
        if (lim > t0_3 + 13) {
            nb[13] = S[p_4 + 13 * nq64];
        }
        nb[14] = 4286578688;
        if (lim > t0_3 + 14) {
            nb[14] = S[p_4 + 14 * nq64];
        }
        nb[15] = 4286578688;
        if (lim > t0_3 + 15) {
            nb[15] = S[p_4 + 15 * nq64];
        }
        asm volatile("" ::: "memory");
        int t0_5 = (j * 4 + w) * 16;
        int t0_6 = t0_5;
        float sc = __uint_as_float(cb[0]);
        float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
        sc = _fmax_0;
        float _min_0 = fminf(sc, 1.7014118346046923e+38f);
        sc = _min_0;
        sc = sc;
        float sc_7 = sc;
        unsigned int key = __as_u32(sc_7) & 4294967232u | (unsigned int)t0_6;
        kb[0] = __uint_as_float(key);
        float sc_8 = __uint_as_float(cb[1]);
        float _fmax_1 = fmaxf(sc_8, -1.7014118346046923e+38f);
        sc_8 = _fmax_1;
        float _min_1 = fminf(sc_8, 1.7014118346046923e+38f);
        sc_8 = _min_1;
        sc_8 = sc_8;
        float sc_9 = sc_8;
        unsigned int key_10 = __as_u32(sc_9) & 4294967232u | (unsigned int)(t0_6 + 1);
        kb[1] = __uint_as_float(key_10);
        float sc_11 = __uint_as_float(cb[2]);
        float _fmax_2 = fmaxf(sc_11, -1.7014118346046923e+38f);
        sc_11 = _fmax_2;
        float _min_2 = fminf(sc_11, 1.7014118346046923e+38f);
        sc_11 = _min_2;
        sc_11 = sc_11;
        float sc_12 = sc_11;
        unsigned int key_13 = __as_u32(sc_12) & 4294967232u | (unsigned int)(t0_6 + 2);
        kb[2] = __uint_as_float(key_13);
        float sc_14 = __uint_as_float(cb[3]);
        float _fmax_3 = fmaxf(sc_14, -1.7014118346046923e+38f);
        sc_14 = _fmax_3;
        float _min_3 = fminf(sc_14, 1.7014118346046923e+38f);
        sc_14 = _min_3;
        sc_14 = sc_14;
        float sc_15 = sc_14;
        unsigned int key_16 = __as_u32(sc_15) & 4294967232u | (unsigned int)(t0_6 + 3);
        kb[3] = __uint_as_float(key_16);
        float sc_17 = __uint_as_float(cb[4]);
        float _fmax_4 = fmaxf(sc_17, -1.7014118346046923e+38f);
        sc_17 = _fmax_4;
        float _min_4 = fminf(sc_17, 1.7014118346046923e+38f);
        sc_17 = _min_4;
        sc_17 = sc_17;
        float sc_18 = sc_17;
        unsigned int key_19 = __as_u32(sc_18) & 4294967232u | (unsigned int)(t0_6 + 4);
        kb[4] = __uint_as_float(key_19);
        float sc_20 = __uint_as_float(cb[5]);
        float _fmax_5 = fmaxf(sc_20, -1.7014118346046923e+38f);
        sc_20 = _fmax_5;
        float _min_5 = fminf(sc_20, 1.7014118346046923e+38f);
        sc_20 = _min_5;
        sc_20 = sc_20;
        float sc_21 = sc_20;
        unsigned int key_22 = __as_u32(sc_21) & 4294967232u | (unsigned int)(t0_6 + 5);
        kb[5] = __uint_as_float(key_22);
        float sc_23 = __uint_as_float(cb[6]);
        float _fmax_6 = fmaxf(sc_23, -1.7014118346046923e+38f);
        sc_23 = _fmax_6;
        float _min_6 = fminf(sc_23, 1.7014118346046923e+38f);
        sc_23 = _min_6;
        sc_23 = sc_23;
        float sc_24 = sc_23;
        unsigned int key_25 = __as_u32(sc_24) & 4294967232u | (unsigned int)(t0_6 + 6);
        kb[6] = __uint_as_float(key_25);
        float sc_26 = __uint_as_float(cb[7]);
        float _fmax_7 = fmaxf(sc_26, -1.7014118346046923e+38f);
        sc_26 = _fmax_7;
        float _min_7 = fminf(sc_26, 1.7014118346046923e+38f);
        sc_26 = _min_7;
        sc_26 = sc_26;
        float sc_27 = sc_26;
        unsigned int key_28 = __as_u32(sc_27) & 4294967232u | (unsigned int)(t0_6 + 7);
        kb[7] = __uint_as_float(key_28);
        float sc_29 = __uint_as_float(cb[8]);
        float _fmax_8 = fmaxf(sc_29, -1.7014118346046923e+38f);
        sc_29 = _fmax_8;
        float _min_8 = fminf(sc_29, 1.7014118346046923e+38f);
        sc_29 = _min_8;
        sc_29 = sc_29;
        float sc_30 = sc_29;
        unsigned int key_31 = __as_u32(sc_30) & 4294967232u | (unsigned int)(t0_6 + 8);
        kb[8] = __uint_as_float(key_31);
        float sc_32 = __uint_as_float(cb[9]);
        float _fmax_9 = fmaxf(sc_32, -1.7014118346046923e+38f);
        sc_32 = _fmax_9;
        float _min_9 = fminf(sc_32, 1.7014118346046923e+38f);
        sc_32 = _min_9;
        sc_32 = sc_32;
        float sc_33 = sc_32;
        unsigned int key_34 = __as_u32(sc_33) & 4294967232u | (unsigned int)(t0_6 + 9);
        kb[9] = __uint_as_float(key_34);
        float sc_35 = __uint_as_float(cb[10]);
        float _fmax_10 = fmaxf(sc_35, -1.7014118346046923e+38f);
        sc_35 = _fmax_10;
        float _min_10 = fminf(sc_35, 1.7014118346046923e+38f);
        sc_35 = _min_10;
        sc_35 = sc_35;
        float sc_36 = sc_35;
        unsigned int key_37 = __as_u32(sc_36) & 4294967232u | (unsigned int)(t0_6 + 10);
        kb[10] = __uint_as_float(key_37);
        float sc_38 = __uint_as_float(cb[11]);
        float _fmax_11 = fmaxf(sc_38, -1.7014118346046923e+38f);
        sc_38 = _fmax_11;
        float _min_11 = fminf(sc_38, 1.7014118346046923e+38f);
        sc_38 = _min_11;
        sc_38 = sc_38;
        float sc_39 = sc_38;
        unsigned int key_40 = __as_u32(sc_39) & 4294967232u | (unsigned int)(t0_6 + 11);
        kb[11] = __uint_as_float(key_40);
        float sc_41 = __uint_as_float(cb[12]);
        float _fmax_12 = fmaxf(sc_41, -1.7014118346046923e+38f);
        sc_41 = _fmax_12;
        float _min_12 = fminf(sc_41, 1.7014118346046923e+38f);
        sc_41 = _min_12;
        sc_41 = sc_41;
        float sc_42 = sc_41;
        unsigned int key_43 = __as_u32(sc_42) & 4294967232u | (unsigned int)(t0_6 + 12);
        kb[12] = __uint_as_float(key_43);
        float sc_44 = __uint_as_float(cb[13]);
        float _fmax_13 = fmaxf(sc_44, -1.7014118346046923e+38f);
        sc_44 = _fmax_13;
        float _min_13 = fminf(sc_44, 1.7014118346046923e+38f);
        sc_44 = _min_13;
        sc_44 = sc_44;
        float sc_45 = sc_44;
        unsigned int key_46 = __as_u32(sc_45) & 4294967232u | (unsigned int)(t0_6 + 13);
        kb[13] = __uint_as_float(key_46);
        float sc_47 = __uint_as_float(cb[14]);
        float _fmax_14 = fmaxf(sc_47, -1.7014118346046923e+38f);
        sc_47 = _fmax_14;
        float _min_14 = fminf(sc_47, 1.7014118346046923e+38f);
        sc_47 = _min_14;
        sc_47 = sc_47;
        float sc_48 = sc_47;
        unsigned int key_49 = __as_u32(sc_48) & 4294967232u | (unsigned int)(t0_6 + 14);
        kb[14] = __uint_as_float(key_49);
        float sc_50 = __uint_as_float(cb[15]);
        float _fmax_15 = fmaxf(sc_50, -1.7014118346046923e+38f);
        sc_50 = _fmax_15;
        float _min_15 = fminf(sc_50, 1.7014118346046923e+38f);
        sc_50 = _min_15;
        sc_50 = sc_50;
        float sc_51 = sc_50;
        unsigned int key_52 = __as_u32(sc_51) & 4294967232u | (unsigned int)(t0_6 + 15);
        kb[15] = __uint_as_float(key_52);
        int f = 0;
        if (t0_6 < fb || t0_6 + 16 > lim - fe && t0_6 < lim) {
            f = 1;
        }
        int fch1 = f;
        if (fch1 != 0) {
            int f_0 = 0;
            if (t0_6 < fb || t0_6 >= lim - fe && lim > t0_6) {
                f_0 = 1;
            }
            if (f_0 != 0) {
                kb[0] = __uint_as_float(2139094976 | (unsigned int)t0_6);
            }
            int f_1 = 0;
            if (t0_6 + 1 < fb || t0_6 + 1 >= lim - fe && lim > t0_6 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                kb[1] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 1));
            }
            int f_2 = 0;
            if (t0_6 + 2 < fb || t0_6 + 2 >= lim - fe && lim > t0_6 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                kb[2] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 2));
            }
            int f_3 = 0;
            if (t0_6 + 3 < fb || t0_6 + 3 >= lim - fe && lim > t0_6 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                kb[3] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 3));
            }
            int f_4 = 0;
            if (t0_6 + 4 < fb || t0_6 + 4 >= lim - fe && lim > t0_6 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                kb[4] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 4));
            }
            int f_5 = 0;
            if (t0_6 + 5 < fb || t0_6 + 5 >= lim - fe && lim > t0_6 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                kb[5] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 5));
            }
            int f_6 = 0;
            if (t0_6 + 6 < fb || t0_6 + 6 >= lim - fe && lim > t0_6 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                kb[6] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 6));
            }
            int f_7 = 0;
            if (t0_6 + 7 < fb || t0_6 + 7 >= lim - fe && lim > t0_6 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                kb[7] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 7));
            }
            int f_8 = 0;
            if (t0_6 + 8 < fb || t0_6 + 8 >= lim - fe && lim > t0_6 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                kb[8] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 8));
            }
            int f_9 = 0;
            if (t0_6 + 9 < fb || t0_6 + 9 >= lim - fe && lim > t0_6 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                kb[9] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 9));
            }
            int f_10 = 0;
            if (t0_6 + 10 < fb || t0_6 + 10 >= lim - fe && lim > t0_6 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                kb[10] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 10));
            }
            int f_11 = 0;
            if (t0_6 + 11 < fb || t0_6 + 11 >= lim - fe && lim > t0_6 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                kb[11] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 11));
            }
            int f_12 = 0;
            if (t0_6 + 12 < fb || t0_6 + 12 >= lim - fe && lim > t0_6 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                kb[12] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 12));
            }
            int f_13 = 0;
            if (t0_6 + 13 < fb || t0_6 + 13 >= lim - fe && lim > t0_6 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                kb[13] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 13));
            }
            int f_14 = 0;
            if (t0_6 + 14 < fb || t0_6 + 14 >= lim - fe && lim > t0_6 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                kb[14] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 14));
            }
            int f_15 = 0;
            if (t0_6 + 15 < fb || t0_6 + 15 >= lim - fe && lim > t0_6 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                kb[15] = __uint_as_float(2139094976 | (unsigned int)(t0_6 + 15));
            }
        }
        float _fmax_16 = fmaxf(kb[0], kb[13]);
        float hi = _fmax_16;
        float _min_16 = fminf(kb[0], kb[13]);
        float lo = _min_16;
        kb[0] = hi;
        kb[13] = lo;
        float _fmax_17 = fmaxf(kb[1], kb[12]);
        float hi_53 = _fmax_17;
        float _min_17 = fminf(kb[1], kb[12]);
        float lo_54 = _min_17;
        kb[1] = hi_53;
        kb[12] = lo_54;
        float _fmax_18 = fmaxf(kb[2], kb[15]);
        float hi_55 = _fmax_18;
        float _min_18 = fminf(kb[2], kb[15]);
        float lo_56 = _min_18;
        kb[2] = hi_55;
        kb[15] = lo_56;
        float _fmax_19 = fmaxf(kb[3], kb[14]);
        float hi_57 = _fmax_19;
        float _min_19 = fminf(kb[3], kb[14]);
        float lo_58 = _min_19;
        kb[3] = hi_57;
        kb[14] = lo_58;
        float _fmax_20 = fmaxf(kb[4], kb[8]);
        float hi_59 = _fmax_20;
        float _min_20 = fminf(kb[4], kb[8]);
        float lo_60 = _min_20;
        kb[4] = hi_59;
        kb[8] = lo_60;
        float _fmax_21 = fmaxf(kb[5], kb[6]);
        float hi_61 = _fmax_21;
        float _min_21 = fminf(kb[5], kb[6]);
        float lo_62 = _min_21;
        kb[5] = hi_61;
        kb[6] = lo_62;
        float _fmax_22 = fmaxf(kb[7], kb[11]);
        float hi_63 = _fmax_22;
        float _min_22 = fminf(kb[7], kb[11]);
        float lo_64 = _min_22;
        kb[7] = hi_63;
        kb[11] = lo_64;
        float _fmax_23 = fmaxf(kb[9], kb[10]);
        float hi_65 = _fmax_23;
        float _min_23 = fminf(kb[9], kb[10]);
        float lo_66 = _min_23;
        kb[9] = hi_65;
        kb[10] = lo_66;
        float _fmax_24 = fmaxf(kb[0], kb[5]);
        float hi_67 = _fmax_24;
        float _min_24 = fminf(kb[0], kb[5]);
        float lo_68 = _min_24;
        kb[0] = hi_67;
        kb[5] = lo_68;
        float _fmax_25 = fmaxf(kb[1], kb[7]);
        float hi_69 = _fmax_25;
        float _min_25 = fminf(kb[1], kb[7]);
        float lo_70 = _min_25;
        kb[1] = hi_69;
        kb[7] = lo_70;
        float _fmax_26 = fmaxf(kb[2], kb[9]);
        float hi_71 = _fmax_26;
        float _min_26 = fminf(kb[2], kb[9]);
        float lo_72 = _min_26;
        kb[2] = hi_71;
        kb[9] = lo_72;
        float _fmax_27 = fmaxf(kb[3], kb[4]);
        float hi_73 = _fmax_27;
        float _min_27 = fminf(kb[3], kb[4]);
        float lo_74 = _min_27;
        kb[3] = hi_73;
        kb[4] = lo_74;
        float _fmax_28 = fmaxf(kb[6], kb[13]);
        float hi_75 = _fmax_28;
        float _min_28 = fminf(kb[6], kb[13]);
        float lo_76 = _min_28;
        kb[6] = hi_75;
        kb[13] = lo_76;
        float _fmax_29 = fmaxf(kb[8], kb[14]);
        float hi_77 = _fmax_29;
        float _min_29 = fminf(kb[8], kb[14]);
        float lo_78 = _min_29;
        kb[8] = hi_77;
        kb[14] = lo_78;
        float _fmax_30 = fmaxf(kb[10], kb[15]);
        float hi_79 = _fmax_30;
        float _min_30 = fminf(kb[10], kb[15]);
        float lo_80 = _min_30;
        kb[10] = hi_79;
        kb[15] = lo_80;
        float _fmax_31 = fmaxf(kb[11], kb[12]);
        float hi_81 = _fmax_31;
        float _min_31 = fminf(kb[11], kb[12]);
        float lo_82 = _min_31;
        kb[11] = hi_81;
        kb[12] = lo_82;
        float _fmax_32 = fmaxf(kb[0], kb[1]);
        float hi_83 = _fmax_32;
        float _min_32 = fminf(kb[0], kb[1]);
        float lo_84 = _min_32;
        kb[0] = hi_83;
        kb[1] = lo_84;
        float _fmax_33 = fmaxf(kb[2], kb[3]);
        float hi_85 = _fmax_33;
        float _min_33 = fminf(kb[2], kb[3]);
        float lo_86 = _min_33;
        kb[2] = hi_85;
        kb[3] = lo_86;
        float _fmax_34 = fmaxf(kb[4], kb[5]);
        float hi_87 = _fmax_34;
        float _min_34 = fminf(kb[4], kb[5]);
        float lo_88 = _min_34;
        kb[4] = hi_87;
        kb[5] = lo_88;
        float _fmax_35 = fmaxf(kb[6], kb[8]);
        float hi_89 = _fmax_35;
        float _min_35 = fminf(kb[6], kb[8]);
        float lo_90 = _min_35;
        kb[6] = hi_89;
        kb[8] = lo_90;
        float _fmax_36 = fmaxf(kb[7], kb[9]);
        float hi_91 = _fmax_36;
        float _min_36 = fminf(kb[7], kb[9]);
        float lo_92 = _min_36;
        kb[7] = hi_91;
        kb[9] = lo_92;
        float _fmax_37 = fmaxf(kb[10], kb[11]);
        float hi_93 = _fmax_37;
        float _min_37 = fminf(kb[10], kb[11]);
        float lo_94 = _min_37;
        kb[10] = hi_93;
        kb[11] = lo_94;
        float _fmax_38 = fmaxf(kb[12], kb[13]);
        float hi_95 = _fmax_38;
        float _min_38 = fminf(kb[12], kb[13]);
        float lo_96 = _min_38;
        kb[12] = hi_95;
        kb[13] = lo_96;
        float _fmax_39 = fmaxf(kb[14], kb[15]);
        float hi_97 = _fmax_39;
        float _min_39 = fminf(kb[14], kb[15]);
        float lo_98 = _min_39;
        kb[14] = hi_97;
        kb[15] = lo_98;
        float _fmax_40 = fmaxf(kb[0], kb[2]);
        float hi_99 = _fmax_40;
        float _min_40 = fminf(kb[0], kb[2]);
        float lo_100 = _min_40;
        kb[0] = hi_99;
        kb[2] = lo_100;
        float _fmax_41 = fmaxf(kb[1], kb[3]);
        float hi_101 = _fmax_41;
        float _min_41 = fminf(kb[1], kb[3]);
        float lo_102 = _min_41;
        kb[1] = hi_101;
        kb[3] = lo_102;
        float _fmax_42 = fmaxf(kb[4], kb[10]);
        float hi_103 = _fmax_42;
        float _min_42 = fminf(kb[4], kb[10]);
        float lo_104 = _min_42;
        kb[4] = hi_103;
        kb[10] = lo_104;
        float _fmax_43 = fmaxf(kb[5], kb[11]);
        float hi_105 = _fmax_43;
        float _min_43 = fminf(kb[5], kb[11]);
        float lo_106 = _min_43;
        kb[5] = hi_105;
        kb[11] = lo_106;
        float _fmax_44 = fmaxf(kb[6], kb[7]);
        float hi_107 = _fmax_44;
        float _min_44 = fminf(kb[6], kb[7]);
        float lo_108 = _min_44;
        kb[6] = hi_107;
        kb[7] = lo_108;
        float _fmax_45 = fmaxf(kb[8], kb[9]);
        float hi_109 = _fmax_45;
        float _min_45 = fminf(kb[8], kb[9]);
        float lo_110 = _min_45;
        kb[8] = hi_109;
        kb[9] = lo_110;
        float _fmax_46 = fmaxf(kb[12], kb[14]);
        float hi_111 = _fmax_46;
        float _min_46 = fminf(kb[12], kb[14]);
        float lo_112 = _min_46;
        kb[12] = hi_111;
        kb[14] = lo_112;
        float _fmax_47 = fmaxf(kb[13], kb[15]);
        float hi_113 = _fmax_47;
        float _min_47 = fminf(kb[13], kb[15]);
        float lo_114 = _min_47;
        kb[13] = hi_113;
        kb[15] = lo_114;
        float _fmax_48 = fmaxf(kb[1], kb[2]);
        float hi_115 = _fmax_48;
        float _min_48 = fminf(kb[1], kb[2]);
        float lo_116 = _min_48;
        kb[1] = hi_115;
        kb[2] = lo_116;
        float _fmax_49 = fmaxf(kb[3], kb[12]);
        float hi_117 = _fmax_49;
        float _min_49 = fminf(kb[3], kb[12]);
        float lo_118 = _min_49;
        kb[3] = hi_117;
        kb[12] = lo_118;
        float _fmax_50 = fmaxf(kb[4], kb[6]);
        float hi_119 = _fmax_50;
        float _min_50 = fminf(kb[4], kb[6]);
        float lo_120 = _min_50;
        kb[4] = hi_119;
        kb[6] = lo_120;
        float _fmax_51 = fmaxf(kb[5], kb[7]);
        float hi_121 = _fmax_51;
        float _min_51 = fminf(kb[5], kb[7]);
        float lo_122 = _min_51;
        kb[5] = hi_121;
        kb[7] = lo_122;
        float _fmax_52 = fmaxf(kb[8], kb[10]);
        float hi_123 = _fmax_52;
        float _min_52 = fminf(kb[8], kb[10]);
        float lo_124 = _min_52;
        kb[8] = hi_123;
        kb[10] = lo_124;
        float _fmax_53 = fmaxf(kb[9], kb[11]);
        float hi_125 = _fmax_53;
        float _min_53 = fminf(kb[9], kb[11]);
        float lo_126 = _min_53;
        kb[9] = hi_125;
        kb[11] = lo_126;
        float _fmax_54 = fmaxf(kb[13], kb[14]);
        float hi_127 = _fmax_54;
        float _min_54 = fminf(kb[13], kb[14]);
        float lo_128 = _min_54;
        kb[13] = hi_127;
        kb[14] = lo_128;
        float _fmax_55 = fmaxf(kb[1], kb[4]);
        float hi_129 = _fmax_55;
        float _min_55 = fminf(kb[1], kb[4]);
        float lo_130 = _min_55;
        kb[1] = hi_129;
        kb[4] = lo_130;
        float _fmax_56 = fmaxf(kb[2], kb[6]);
        float hi_131 = _fmax_56;
        float _min_56 = fminf(kb[2], kb[6]);
        float lo_132 = _min_56;
        kb[2] = hi_131;
        kb[6] = lo_132;
        float _fmax_57 = fmaxf(kb[5], kb[8]);
        float hi_133 = _fmax_57;
        float _min_57 = fminf(kb[5], kb[8]);
        float lo_134 = _min_57;
        kb[5] = hi_133;
        kb[8] = lo_134;
        float _fmax_58 = fmaxf(kb[7], kb[10]);
        float hi_135 = _fmax_58;
        float _min_58 = fminf(kb[7], kb[10]);
        float lo_136 = _min_58;
        kb[7] = hi_135;
        kb[10] = lo_136;
        float _fmax_59 = fmaxf(kb[9], kb[13]);
        float hi_137 = _fmax_59;
        float _min_59 = fminf(kb[9], kb[13]);
        float lo_138 = _min_59;
        kb[9] = hi_137;
        kb[13] = lo_138;
        float _fmax_60 = fmaxf(kb[11], kb[14]);
        float hi_139 = _fmax_60;
        float _min_60 = fminf(kb[11], kb[14]);
        float lo_140 = _min_60;
        kb[11] = hi_139;
        kb[14] = lo_140;
        float _fmax_61 = fmaxf(kb[2], kb[4]);
        float hi_141 = _fmax_61;
        float _min_61 = fminf(kb[2], kb[4]);
        float lo_142 = _min_61;
        kb[2] = hi_141;
        kb[4] = lo_142;
        float _fmax_62 = fmaxf(kb[3], kb[6]);
        float hi_143 = _fmax_62;
        float _min_62 = fminf(kb[3], kb[6]);
        float lo_144 = _min_62;
        kb[3] = hi_143;
        kb[6] = lo_144;
        float _fmax_63 = fmaxf(kb[9], kb[12]);
        float hi_145 = _fmax_63;
        float _min_63 = fminf(kb[9], kb[12]);
        float lo_146 = _min_63;
        kb[9] = hi_145;
        kb[12] = lo_146;
        float _fmax_64 = fmaxf(kb[11], kb[13]);
        float hi_147 = _fmax_64;
        float _min_64 = fminf(kb[11], kb[13]);
        float lo_148 = _min_64;
        kb[11] = hi_147;
        kb[13] = lo_148;
        float _fmax_65 = fmaxf(kb[3], kb[5]);
        float hi_149 = _fmax_65;
        float _min_65 = fminf(kb[3], kb[5]);
        float lo_150 = _min_65;
        kb[3] = hi_149;
        kb[5] = lo_150;
        float _fmax_66 = fmaxf(kb[6], kb[8]);
        float hi_151 = _fmax_66;
        float _min_66 = fminf(kb[6], kb[8]);
        float lo_152 = _min_66;
        kb[6] = hi_151;
        kb[8] = lo_152;
        float _fmax_67 = fmaxf(kb[7], kb[9]);
        float hi_153 = _fmax_67;
        float _min_67 = fminf(kb[7], kb[9]);
        float lo_154 = _min_67;
        kb[7] = hi_153;
        kb[9] = lo_154;
        float _fmax_68 = fmaxf(kb[10], kb[12]);
        float hi_155 = _fmax_68;
        float _min_68 = fminf(kb[10], kb[12]);
        float lo_156 = _min_68;
        kb[10] = hi_155;
        kb[12] = lo_156;
        float _fmax_69 = fmaxf(kb[3], kb[4]);
        float hi_157 = _fmax_69;
        float _min_69 = fminf(kb[3], kb[4]);
        float lo_158 = _min_69;
        kb[3] = hi_157;
        kb[4] = lo_158;
        float _fmax_70 = fmaxf(kb[5], kb[6]);
        float hi_159 = _fmax_70;
        float _min_70 = fminf(kb[5], kb[6]);
        float lo_160 = _min_70;
        kb[5] = hi_159;
        kb[6] = lo_160;
        float _fmax_71 = fmaxf(kb[7], kb[8]);
        float hi_161 = _fmax_71;
        float _min_71 = fminf(kb[7], kb[8]);
        float lo_162 = _min_71;
        kb[7] = hi_161;
        kb[8] = lo_162;
        float _fmax_72 = fmaxf(kb[9], kb[10]);
        float hi_163 = _fmax_72;
        float _min_72 = fminf(kb[9], kb[10]);
        float lo_164 = _min_72;
        kb[9] = hi_163;
        kb[10] = lo_164;
        float _fmax_73 = fmaxf(kb[11], kb[12]);
        float hi_165 = _fmax_73;
        float _min_73 = fminf(kb[11], kb[12]);
        float lo_166 = _min_73;
        kb[11] = hi_165;
        kb[12] = lo_166;
        float _fmax_74 = fmaxf(kb[6], kb[7]);
        float hi_167 = _fmax_74;
        float _min_74 = fminf(kb[6], kb[7]);
        float lo_168 = _min_74;
        kb[6] = hi_167;
        kb[7] = lo_168;
        float _fmax_75 = fmaxf(kb[8], kb[9]);
        float hi_169 = _fmax_75;
        float _min_75 = fminf(kb[8], kb[9]);
        float lo_170 = _min_75;
        kb[8] = hi_169;
        kb[9] = lo_170;
        float r = rej;
        float _fmax_76 = fmaxf(a[0], kb[15]);
        float hi_171 = _fmax_76;
        float _min_76 = fminf(a[0], kb[15]);
        float lo_172 = _min_76;
        a[0] = hi_171;
        float _fmax_77 = fmaxf(r, lo_172);
        r = _fmax_77;
        float _fmax_78 = fmaxf(a[1], kb[14]);
        float hi_173 = _fmax_78;
        float _min_77 = fminf(a[1], kb[14]);
        float lo_174 = _min_77;
        a[1] = hi_173;
        float _fmax_79 = fmaxf(r, lo_174);
        r = _fmax_79;
        float _fmax_80 = fmaxf(a[2], kb[13]);
        float hi_175 = _fmax_80;
        float _min_78 = fminf(a[2], kb[13]);
        float lo_176 = _min_78;
        a[2] = hi_175;
        float _fmax_81 = fmaxf(r, lo_176);
        r = _fmax_81;
        float _fmax_82 = fmaxf(a[3], kb[12]);
        float hi_177 = _fmax_82;
        float _min_79 = fminf(a[3], kb[12]);
        float lo_178 = _min_79;
        a[3] = hi_177;
        float _fmax_83 = fmaxf(r, lo_178);
        r = _fmax_83;
        float _fmax_84 = fmaxf(a[4], kb[11]);
        float hi_179 = _fmax_84;
        float _min_80 = fminf(a[4], kb[11]);
        float lo_180 = _min_80;
        a[4] = hi_179;
        float _fmax_85 = fmaxf(r, lo_180);
        r = _fmax_85;
        float _fmax_86 = fmaxf(a[5], kb[10]);
        float hi_181 = _fmax_86;
        float _min_81 = fminf(a[5], kb[10]);
        float lo_182 = _min_81;
        a[5] = hi_181;
        float _fmax_87 = fmaxf(r, lo_182);
        r = _fmax_87;
        float _fmax_88 = fmaxf(a[6], kb[9]);
        float hi_183 = _fmax_88;
        float _min_82 = fminf(a[6], kb[9]);
        float lo_184 = _min_82;
        a[6] = hi_183;
        float _fmax_89 = fmaxf(r, lo_184);
        r = _fmax_89;
        float _fmax_90 = fmaxf(a[7], kb[8]);
        float hi_185 = _fmax_90;
        float _min_83 = fminf(a[7], kb[8]);
        float lo_186 = _min_83;
        a[7] = hi_185;
        float _fmax_91 = fmaxf(r, lo_186);
        r = _fmax_91;
        float _fmax_92 = fmaxf(a[8], kb[7]);
        float hi_187 = _fmax_92;
        float _min_84 = fminf(a[8], kb[7]);
        float lo_188 = _min_84;
        a[8] = hi_187;
        float _fmax_93 = fmaxf(r, lo_188);
        r = _fmax_93;
        float _fmax_94 = fmaxf(a[9], kb[6]);
        float hi_189 = _fmax_94;
        float _min_85 = fminf(a[9], kb[6]);
        float lo_190 = _min_85;
        a[9] = hi_189;
        float _fmax_95 = fmaxf(r, lo_190);
        r = _fmax_95;
        float _fmax_96 = fmaxf(a[10], kb[5]);
        float hi_191 = _fmax_96;
        float _min_86 = fminf(a[10], kb[5]);
        float lo_192 = _min_86;
        a[10] = hi_191;
        float _fmax_97 = fmaxf(r, lo_192);
        r = _fmax_97;
        float _fmax_98 = fmaxf(a[11], kb[4]);
        float hi_193 = _fmax_98;
        float _min_87 = fminf(a[11], kb[4]);
        float lo_194 = _min_87;
        a[11] = hi_193;
        float _fmax_99 = fmaxf(r, lo_194);
        r = _fmax_99;
        float _fmax_100 = fmaxf(a[12], kb[3]);
        float hi_195 = _fmax_100;
        float _min_88 = fminf(a[12], kb[3]);
        float lo_196 = _min_88;
        a[12] = hi_195;
        float _fmax_101 = fmaxf(r, lo_196);
        r = _fmax_101;
        float _fmax_102 = fmaxf(a[13], kb[2]);
        float hi_197 = _fmax_102;
        float _min_89 = fminf(a[13], kb[2]);
        float lo_198 = _min_89;
        a[13] = hi_197;
        float _fmax_103 = fmaxf(r, lo_198);
        r = _fmax_103;
        float _fmax_104 = fmaxf(a[14], kb[1]);
        float hi_199 = _fmax_104;
        float _min_90 = fminf(a[14], kb[1]);
        float lo_200 = _min_90;
        a[14] = hi_199;
        float _fmax_105 = fmaxf(r, lo_200);
        r = _fmax_105;
        float _fmax_106 = fmaxf(a[15], kb[0]);
        float hi_201 = _fmax_106;
        float _min_91 = fminf(a[15], kb[0]);
        float lo_202 = _min_91;
        a[15] = hi_201;
        float _fmax_107 = fmaxf(r, lo_202);
        r = _fmax_107;
        float _fmax_108 = fmaxf(a[0], a[8]);
        float hi_203 = _fmax_108;
        float _min_92 = fminf(a[0], a[8]);
        float lo_204 = _min_92;
        a[0] = hi_203;
        a[8] = lo_204;
        float _fmax_109 = fmaxf(a[1], a[9]);
        float hi_205 = _fmax_109;
        float _min_93 = fminf(a[1], a[9]);
        float lo_206 = _min_93;
        a[1] = hi_205;
        a[9] = lo_206;
        float _fmax_110 = fmaxf(a[2], a[10]);
        float hi_207 = _fmax_110;
        float _min_94 = fminf(a[2], a[10]);
        float lo_208 = _min_94;
        a[2] = hi_207;
        a[10] = lo_208;
        float _fmax_111 = fmaxf(a[3], a[11]);
        float hi_209 = _fmax_111;
        float _min_95 = fminf(a[3], a[11]);
        float lo_210 = _min_95;
        a[3] = hi_209;
        a[11] = lo_210;
        float _fmax_112 = fmaxf(a[4], a[12]);
        float hi_211 = _fmax_112;
        float _min_96 = fminf(a[4], a[12]);
        float lo_212 = _min_96;
        a[4] = hi_211;
        a[12] = lo_212;
        float _fmax_113 = fmaxf(a[5], a[13]);
        float hi_213 = _fmax_113;
        float _min_97 = fminf(a[5], a[13]);
        float lo_214 = _min_97;
        a[5] = hi_213;
        a[13] = lo_214;
        float _fmax_114 = fmaxf(a[6], a[14]);
        float hi_215 = _fmax_114;
        float _min_98 = fminf(a[6], a[14]);
        float lo_216 = _min_98;
        a[6] = hi_215;
        a[14] = lo_216;
        float _fmax_115 = fmaxf(a[7], a[15]);
        float hi_217 = _fmax_115;
        float _min_99 = fminf(a[7], a[15]);
        float lo_218 = _min_99;
        a[7] = hi_217;
        a[15] = lo_218;
        float _fmax_116 = fmaxf(a[0], a[4]);
        float hi_219 = _fmax_116;
        float _min_100 = fminf(a[0], a[4]);
        float lo_220 = _min_100;
        a[0] = hi_219;
        a[4] = lo_220;
        float _fmax_117 = fmaxf(a[1], a[5]);
        float hi_221 = _fmax_117;
        float _min_101 = fminf(a[1], a[5]);
        float lo_222 = _min_101;
        a[1] = hi_221;
        a[5] = lo_222;
        float _fmax_118 = fmaxf(a[2], a[6]);
        float hi_223 = _fmax_118;
        float _min_102 = fminf(a[2], a[6]);
        float lo_224 = _min_102;
        a[2] = hi_223;
        a[6] = lo_224;
        float _fmax_119 = fmaxf(a[3], a[7]);
        float hi_225 = _fmax_119;
        float _min_103 = fminf(a[3], a[7]);
        float lo_226 = _min_103;
        a[3] = hi_225;
        a[7] = lo_226;
        float _fmax_120 = fmaxf(a[8], a[12]);
        float hi_227 = _fmax_120;
        float _min_104 = fminf(a[8], a[12]);
        float lo_228 = _min_104;
        a[8] = hi_227;
        a[12] = lo_228;
        float _fmax_121 = fmaxf(a[9], a[13]);
        float hi_229 = _fmax_121;
        float _min_105 = fminf(a[9], a[13]);
        float lo_230 = _min_105;
        a[9] = hi_229;
        a[13] = lo_230;
        float _fmax_122 = fmaxf(a[10], a[14]);
        float hi_231 = _fmax_122;
        float _min_106 = fminf(a[10], a[14]);
        float lo_232 = _min_106;
        a[10] = hi_231;
        a[14] = lo_232;
        float _fmax_123 = fmaxf(a[11], a[15]);
        float hi_233 = _fmax_123;
        float _min_107 = fminf(a[11], a[15]);
        float lo_234 = _min_107;
        a[11] = hi_233;
        a[15] = lo_234;
        float _fmax_124 = fmaxf(a[0], a[2]);
        float hi_235 = _fmax_124;
        float _min_108 = fminf(a[0], a[2]);
        float lo_236 = _min_108;
        a[0] = hi_235;
        a[2] = lo_236;
        float _fmax_125 = fmaxf(a[1], a[3]);
        float hi_237 = _fmax_125;
        float _min_109 = fminf(a[1], a[3]);
        float lo_238 = _min_109;
        a[1] = hi_237;
        a[3] = lo_238;
        float _fmax_126 = fmaxf(a[4], a[6]);
        float hi_239 = _fmax_126;
        float _min_110 = fminf(a[4], a[6]);
        float lo_240 = _min_110;
        a[4] = hi_239;
        a[6] = lo_240;
        float _fmax_127 = fmaxf(a[5], a[7]);
        float hi_241 = _fmax_127;
        float _min_111 = fminf(a[5], a[7]);
        float lo_242 = _min_111;
        a[5] = hi_241;
        a[7] = lo_242;
        float _fmax_128 = fmaxf(a[8], a[10]);
        float hi_243 = _fmax_128;
        float _min_112 = fminf(a[8], a[10]);
        float lo_244 = _min_112;
        a[8] = hi_243;
        a[10] = lo_244;
        float _fmax_129 = fmaxf(a[9], a[11]);
        float hi_245 = _fmax_129;
        float _min_113 = fminf(a[9], a[11]);
        float lo_246 = _min_113;
        a[9] = hi_245;
        a[11] = lo_246;
        float _fmax_130 = fmaxf(a[12], a[14]);
        float hi_247 = _fmax_130;
        float _min_114 = fminf(a[12], a[14]);
        float lo_248 = _min_114;
        a[12] = hi_247;
        a[14] = lo_248;
        float _fmax_131 = fmaxf(a[13], a[15]);
        float hi_249 = _fmax_131;
        float _min_115 = fminf(a[13], a[15]);
        float lo_250 = _min_115;
        a[13] = hi_249;
        a[15] = lo_250;
        float _fmax_132 = fmaxf(a[0], a[1]);
        float hi_251 = _fmax_132;
        float _min_116 = fminf(a[0], a[1]);
        float lo_252 = _min_116;
        a[0] = hi_251;
        a[1] = lo_252;
        float _fmax_133 = fmaxf(a[2], a[3]);
        float hi_253 = _fmax_133;
        float _min_117 = fminf(a[2], a[3]);
        float lo_254 = _min_117;
        a[2] = hi_253;
        a[3] = lo_254;
        float _fmax_134 = fmaxf(a[4], a[5]);
        float hi_255 = _fmax_134;
        float _min_118 = fminf(a[4], a[5]);
        float lo_256 = _min_118;
        a[4] = hi_255;
        a[5] = lo_256;
        float _fmax_135 = fmaxf(a[6], a[7]);
        float hi_257 = _fmax_135;
        float _min_119 = fminf(a[6], a[7]);
        float lo_258 = _min_119;
        a[6] = hi_257;
        a[7] = lo_258;
        float _fmax_136 = fmaxf(a[8], a[9]);
        float hi_259 = _fmax_136;
        float _min_120 = fminf(a[8], a[9]);
        float lo_260 = _min_120;
        a[8] = hi_259;
        a[9] = lo_260;
        float _fmax_137 = fmaxf(a[10], a[11]);
        float hi_261 = _fmax_137;
        float _min_121 = fminf(a[10], a[11]);
        float lo_262 = _min_121;
        a[10] = hi_261;
        a[11] = lo_262;
        float _fmax_138 = fmaxf(a[12], a[13]);
        float hi_263 = _fmax_138;
        float _min_122 = fminf(a[12], a[13]);
        float lo_264 = _min_122;
        a[12] = hi_263;
        a[13] = lo_264;
        float _fmax_139 = fmaxf(a[14], a[15]);
        float hi_265 = _fmax_139;
        float _min_123 = fminf(a[14], a[15]);
        float lo_266 = _min_123;
        a[14] = hi_265;
        a[15] = lo_266;
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
    for (int k = 0; k < 2; k++) {
        int bit = 1 << k;
        int low = whi & (bit << 1) - 1;
        if (low == bit) {
            int base = whi * 544 + c;
            tree[base] = a[0];
            tree[base + 32] = a[1];
            tree[base + 64] = a[2];
            tree[base + 96] = a[3];
            tree[base + 128] = a[4];
            tree[base + 160] = a[5];
            tree[base + 192] = a[6];
            tree[base + 224] = a[7];
            tree[base + 256] = a[8];
            tree[base + 288] = a[9];
            tree[base + 320] = a[10];
            tree[base + 352] = a[11];
            tree[base + 384] = a[12];
            tree[base + 416] = a[13];
            tree[base + 448] = a[14];
            tree[base + 480] = a[15];
            int base_0 = base;
            tree[base_0 + 512] = rej;
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (low == 0) {
            float sb[16];
            int rbase = (whi + bit) * 544 + c;
            sb[0] = tree[rbase];
            sb[1] = tree[rbase + 32];
            sb[2] = tree[rbase + 64];
            sb[3] = tree[rbase + 96];
            sb[4] = tree[rbase + 128];
            sb[5] = tree[rbase + 160];
            sb[6] = tree[rbase + 192];
            sb[7] = tree[rbase + 224];
            sb[8] = tree[rbase + 256];
            sb[9] = tree[rbase + 288];
            sb[10] = tree[rbase + 320];
            sb[11] = tree[rbase + 352];
            sb[12] = tree[rbase + 384];
            sb[13] = tree[rbase + 416];
            sb[14] = tree[rbase + 448];
            sb[15] = tree[rbase + 480];
            float srej = tree[rbase + 512];
            float _fmax_140 = fmaxf(rej, srej);
            rej = _fmax_140;
            float r_1 = rej;
            float _fmax_141 = fmaxf(a[0], sb[15]);
            float hi_1 = _fmax_141;
            float _min_124 = fminf(a[0], sb[15]);
            float lo_1 = _min_124;
            a[0] = hi_1;
            float _fmax_142 = fmaxf(r_1, lo_1);
            r_1 = _fmax_142;
            float _fmax_143 = fmaxf(a[1], sb[14]);
            float hi_0 = _fmax_143;
            float _min_125 = fminf(a[1], sb[14]);
            float lo_1_1 = _min_125;
            a[1] = hi_0;
            float _fmax_144 = fmaxf(r_1, lo_1_1);
            r_1 = _fmax_144;
            float _fmax_145 = fmaxf(a[2], sb[13]);
            float hi_2 = _fmax_145;
            float _min_126 = fminf(a[2], sb[13]);
            float lo_3 = _min_126;
            a[2] = hi_2;
            float _fmax_146 = fmaxf(r_1, lo_3);
            r_1 = _fmax_146;
            float _fmax_147 = fmaxf(a[3], sb[12]);
            float hi_4 = _fmax_147;
            float _min_127 = fminf(a[3], sb[12]);
            float lo_5 = _min_127;
            a[3] = hi_4;
            float _fmax_148 = fmaxf(r_1, lo_5);
            r_1 = _fmax_148;
            float _fmax_149 = fmaxf(a[4], sb[11]);
            float hi_6 = _fmax_149;
            float _min_128 = fminf(a[4], sb[11]);
            float lo_7 = _min_128;
            a[4] = hi_6;
            float _fmax_150 = fmaxf(r_1, lo_7);
            r_1 = _fmax_150;
            float _fmax_151 = fmaxf(a[5], sb[10]);
            float hi_8 = _fmax_151;
            float _min_129 = fminf(a[5], sb[10]);
            float lo_9 = _min_129;
            a[5] = hi_8;
            float _fmax_152 = fmaxf(r_1, lo_9);
            r_1 = _fmax_152;
            float _fmax_153 = fmaxf(a[6], sb[9]);
            float hi_10 = _fmax_153;
            float _min_130 = fminf(a[6], sb[9]);
            float lo_11 = _min_130;
            a[6] = hi_10;
            float _fmax_154 = fmaxf(r_1, lo_11);
            r_1 = _fmax_154;
            float _fmax_155 = fmaxf(a[7], sb[8]);
            float hi_12 = _fmax_155;
            float _min_131 = fminf(a[7], sb[8]);
            float lo_13 = _min_131;
            a[7] = hi_12;
            float _fmax_156 = fmaxf(r_1, lo_13);
            r_1 = _fmax_156;
            float _fmax_157 = fmaxf(a[8], sb[7]);
            float hi_14 = _fmax_157;
            float _min_132 = fminf(a[8], sb[7]);
            float lo_15 = _min_132;
            a[8] = hi_14;
            float _fmax_158 = fmaxf(r_1, lo_15);
            r_1 = _fmax_158;
            float _fmax_159 = fmaxf(a[9], sb[6]);
            float hi_16 = _fmax_159;
            float _min_133 = fminf(a[9], sb[6]);
            float lo_17 = _min_133;
            a[9] = hi_16;
            float _fmax_160 = fmaxf(r_1, lo_17);
            r_1 = _fmax_160;
            float _fmax_161 = fmaxf(a[10], sb[5]);
            float hi_18 = _fmax_161;
            float _min_134 = fminf(a[10], sb[5]);
            float lo_19 = _min_134;
            a[10] = hi_18;
            float _fmax_162 = fmaxf(r_1, lo_19);
            r_1 = _fmax_162;
            float _fmax_163 = fmaxf(a[11], sb[4]);
            float hi_20 = _fmax_163;
            float _min_135 = fminf(a[11], sb[4]);
            float lo_21 = _min_135;
            a[11] = hi_20;
            float _fmax_164 = fmaxf(r_1, lo_21);
            r_1 = _fmax_164;
            float _fmax_165 = fmaxf(a[12], sb[3]);
            float hi_22 = _fmax_165;
            float _min_136 = fminf(a[12], sb[3]);
            float lo_23 = _min_136;
            a[12] = hi_22;
            float _fmax_166 = fmaxf(r_1, lo_23);
            r_1 = _fmax_166;
            float _fmax_167 = fmaxf(a[13], sb[2]);
            float hi_24 = _fmax_167;
            float _min_137 = fminf(a[13], sb[2]);
            float lo_25 = _min_137;
            a[13] = hi_24;
            float _fmax_168 = fmaxf(r_1, lo_25);
            r_1 = _fmax_168;
            float _fmax_169 = fmaxf(a[14], sb[1]);
            float hi_26 = _fmax_169;
            float _min_138 = fminf(a[14], sb[1]);
            float lo_27 = _min_138;
            a[14] = hi_26;
            float _fmax_170 = fmaxf(r_1, lo_27);
            r_1 = _fmax_170;
            float _fmax_171 = fmaxf(a[15], sb[0]);
            float hi_28 = _fmax_171;
            float _min_139 = fminf(a[15], sb[0]);
            float lo_29 = _min_139;
            a[15] = hi_28;
            float _fmax_172 = fmaxf(r_1, lo_29);
            r_1 = _fmax_172;
            float _fmax_173 = fmaxf(a[0], a[8]);
            float hi_30 = _fmax_173;
            float _min_140 = fminf(a[0], a[8]);
            float lo_31 = _min_140;
            a[0] = hi_30;
            a[8] = lo_31;
            float _fmax_174 = fmaxf(a[1], a[9]);
            float hi_32 = _fmax_174;
            float _min_141 = fminf(a[1], a[9]);
            float lo_33 = _min_141;
            a[1] = hi_32;
            a[9] = lo_33;
            float _fmax_175 = fmaxf(a[2], a[10]);
            float hi_34 = _fmax_175;
            float _min_142 = fminf(a[2], a[10]);
            float lo_35 = _min_142;
            a[2] = hi_34;
            a[10] = lo_35;
            float _fmax_176 = fmaxf(a[3], a[11]);
            float hi_36 = _fmax_176;
            float _min_143 = fminf(a[3], a[11]);
            float lo_37 = _min_143;
            a[3] = hi_36;
            a[11] = lo_37;
            float _fmax_177 = fmaxf(a[4], a[12]);
            float hi_38 = _fmax_177;
            float _min_144 = fminf(a[4], a[12]);
            float lo_39 = _min_144;
            a[4] = hi_38;
            a[12] = lo_39;
            float _fmax_178 = fmaxf(a[5], a[13]);
            float hi_40 = _fmax_178;
            float _min_145 = fminf(a[5], a[13]);
            float lo_41 = _min_145;
            a[5] = hi_40;
            a[13] = lo_41;
            float _fmax_179 = fmaxf(a[6], a[14]);
            float hi_42 = _fmax_179;
            float _min_146 = fminf(a[6], a[14]);
            float lo_43 = _min_146;
            a[6] = hi_42;
            a[14] = lo_43;
            float _fmax_180 = fmaxf(a[7], a[15]);
            float hi_44 = _fmax_180;
            float _min_147 = fminf(a[7], a[15]);
            float lo_45 = _min_147;
            a[7] = hi_44;
            a[15] = lo_45;
            float _fmax_181 = fmaxf(a[0], a[4]);
            float hi_46 = _fmax_181;
            float _min_148 = fminf(a[0], a[4]);
            float lo_47 = _min_148;
            a[0] = hi_46;
            a[4] = lo_47;
            float _fmax_182 = fmaxf(a[1], a[5]);
            float hi_48 = _fmax_182;
            float _min_149 = fminf(a[1], a[5]);
            float lo_49 = _min_149;
            a[1] = hi_48;
            a[5] = lo_49;
            float _fmax_183 = fmaxf(a[2], a[6]);
            float hi_50 = _fmax_183;
            float _min_150 = fminf(a[2], a[6]);
            float lo_51 = _min_150;
            a[2] = hi_50;
            a[6] = lo_51;
            float _fmax_184 = fmaxf(a[3], a[7]);
            float hi_52 = _fmax_184;
            float _min_151 = fminf(a[3], a[7]);
            float lo_53 = _min_151;
            a[3] = hi_52;
            a[7] = lo_53;
            float _fmax_185 = fmaxf(a[8], a[12]);
            float hi_54 = _fmax_185;
            float _min_152 = fminf(a[8], a[12]);
            float lo_55 = _min_152;
            a[8] = hi_54;
            a[12] = lo_55;
            float _fmax_186 = fmaxf(a[9], a[13]);
            float hi_56 = _fmax_186;
            float _min_153 = fminf(a[9], a[13]);
            float lo_57 = _min_153;
            a[9] = hi_56;
            a[13] = lo_57;
            float _fmax_187 = fmaxf(a[10], a[14]);
            float hi_58 = _fmax_187;
            float _min_154 = fminf(a[10], a[14]);
            float lo_59 = _min_154;
            a[10] = hi_58;
            a[14] = lo_59;
            float _fmax_188 = fmaxf(a[11], a[15]);
            float hi_60 = _fmax_188;
            float _min_155 = fminf(a[11], a[15]);
            float lo_61 = _min_155;
            a[11] = hi_60;
            a[15] = lo_61;
            float _fmax_189 = fmaxf(a[0], a[2]);
            float hi_62 = _fmax_189;
            float _min_156 = fminf(a[0], a[2]);
            float lo_63 = _min_156;
            a[0] = hi_62;
            a[2] = lo_63;
            float _fmax_190 = fmaxf(a[1], a[3]);
            float hi_64 = _fmax_190;
            float _min_157 = fminf(a[1], a[3]);
            float lo_65 = _min_157;
            a[1] = hi_64;
            a[3] = lo_65;
            float _fmax_191 = fmaxf(a[4], a[6]);
            float hi_66 = _fmax_191;
            float _min_158 = fminf(a[4], a[6]);
            float lo_67 = _min_158;
            a[4] = hi_66;
            a[6] = lo_67;
            float _fmax_192 = fmaxf(a[5], a[7]);
            float hi_68 = _fmax_192;
            float _min_159 = fminf(a[5], a[7]);
            float lo_69 = _min_159;
            a[5] = hi_68;
            a[7] = lo_69;
            float _fmax_193 = fmaxf(a[8], a[10]);
            float hi_70 = _fmax_193;
            float _min_160 = fminf(a[8], a[10]);
            float lo_71 = _min_160;
            a[8] = hi_70;
            a[10] = lo_71;
            float _fmax_194 = fmaxf(a[9], a[11]);
            float hi_72 = _fmax_194;
            float _min_161 = fminf(a[9], a[11]);
            float lo_73 = _min_161;
            a[9] = hi_72;
            a[11] = lo_73;
            float _fmax_195 = fmaxf(a[12], a[14]);
            float hi_74 = _fmax_195;
            float _min_162 = fminf(a[12], a[14]);
            float lo_75 = _min_162;
            a[12] = hi_74;
            a[14] = lo_75;
            float _fmax_196 = fmaxf(a[13], a[15]);
            float hi_76 = _fmax_196;
            float _min_163 = fminf(a[13], a[15]);
            float lo_77 = _min_163;
            a[13] = hi_76;
            a[15] = lo_77;
            float _fmax_197 = fmaxf(a[0], a[1]);
            float hi_78 = _fmax_197;
            float _min_164 = fminf(a[0], a[1]);
            float lo_79 = _min_164;
            a[0] = hi_78;
            a[1] = lo_79;
            float _fmax_198 = fmaxf(a[2], a[3]);
            float hi_80 = _fmax_198;
            float _min_165 = fminf(a[2], a[3]);
            float lo_81 = _min_165;
            a[2] = hi_80;
            a[3] = lo_81;
            float _fmax_199 = fmaxf(a[4], a[5]);
            float hi_82 = _fmax_199;
            float _min_166 = fminf(a[4], a[5]);
            float lo_83 = _min_166;
            a[4] = hi_82;
            a[5] = lo_83;
            float _fmax_200 = fmaxf(a[6], a[7]);
            float hi_84 = _fmax_200;
            float _min_167 = fminf(a[6], a[7]);
            float lo_85 = _min_167;
            a[6] = hi_84;
            a[7] = lo_85;
            float _fmax_201 = fmaxf(a[8], a[9]);
            float hi_86 = _fmax_201;
            float _min_168 = fminf(a[8], a[9]);
            float lo_87 = _min_168;
            a[8] = hi_86;
            a[9] = lo_87;
            float _fmax_202 = fmaxf(a[10], a[11]);
            float hi_88 = _fmax_202;
            float _min_169 = fminf(a[10], a[11]);
            float lo_89 = _min_169;
            a[10] = hi_88;
            a[11] = lo_89;
            float _fmax_203 = fmaxf(a[12], a[13]);
            float hi_90 = _fmax_203;
            float _min_170 = fminf(a[12], a[13]);
            float lo_91 = _min_170;
            a[12] = hi_90;
            a[13] = lo_91;
            float _fmax_204 = fmaxf(a[14], a[15]);
            float hi_92 = _fmax_204;
            float _min_171 = fminf(a[14], a[15]);
            float lo_93 = _min_171;
            a[14] = hi_92;
            a[15] = lo_93;
            rej = r_1;
        }
    }
    if (w == 0) {
        unsigned int u16 = __as_u32(a[15]);
        unsigned int c16 = u16 & 4294967232u;
        unsigned int cr = __as_u32(rej) & 4294967232u;
        unsigned int q = 4294967295;
        if (c16 == cr && u16 < 4278190080u && c16 != 2139094976 && col < total_q) {
            q = c16;
            flagw[0] = 1;
        }
        qcol[c] = q;
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
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
        int _vote_0 = __any_sync(0xFFFFFFFF, flagged != 0);
        if (_vote_0 != 0) {
            unsigned int cb2[16];
            unsigned int nb2[16];
            float kb2[16];
            int t0_2_1 = w * 16;
            int t0_3_1 = t0_2_1;
            long long p_4_1 = cbase + (long long)t0_3_1 * nq64;
            cb2[0] = 4286578688;
            if (lim2 > t0_3_1) {
                cb2[0] = S[p_4_1];
            }
            cb2[1] = 4286578688;
            if (lim2 > t0_3_1 + 1) {
                cb2[1] = S[p_4_1 + nq64];
            }
            cb2[2] = 4286578688;
            if (lim2 > t0_3_1 + 2) {
                cb2[2] = S[p_4_1 + 2 * nq64];
            }
            cb2[3] = 4286578688;
            if (lim2 > t0_3_1 + 3) {
                cb2[3] = S[p_4_1 + 3 * nq64];
            }
            cb2[4] = 4286578688;
            if (lim2 > t0_3_1 + 4) {
                cb2[4] = S[p_4_1 + 4 * nq64];
            }
            cb2[5] = 4286578688;
            if (lim2 > t0_3_1 + 5) {
                cb2[5] = S[p_4_1 + 5 * nq64];
            }
            cb2[6] = 4286578688;
            if (lim2 > t0_3_1 + 6) {
                cb2[6] = S[p_4_1 + 6 * nq64];
            }
            cb2[7] = 4286578688;
            if (lim2 > t0_3_1 + 7) {
                cb2[7] = S[p_4_1 + 7 * nq64];
            }
            cb2[8] = 4286578688;
            if (lim2 > t0_3_1 + 8) {
                cb2[8] = S[p_4_1 + 8 * nq64];
            }
            cb2[9] = 4286578688;
            if (lim2 > t0_3_1 + 9) {
                cb2[9] = S[p_4_1 + 9 * nq64];
            }
            cb2[10] = 4286578688;
            if (lim2 > t0_3_1 + 10) {
                cb2[10] = S[p_4_1 + 10 * nq64];
            }
            cb2[11] = 4286578688;
            if (lim2 > t0_3_1 + 11) {
                cb2[11] = S[p_4_1 + 11 * nq64];
            }
            cb2[12] = 4286578688;
            if (lim2 > t0_3_1 + 12) {
                cb2[12] = S[p_4_1 + 12 * nq64];
            }
            cb2[13] = 4286578688;
            if (lim2 > t0_3_1 + 13) {
                cb2[13] = S[p_4_1 + 13 * nq64];
            }
            cb2[14] = 4286578688;
            if (lim2 > t0_3_1 + 14) {
                cb2[14] = S[p_4_1 + 14 * nq64];
            }
            cb2[15] = 4286578688;
            if (lim2 > t0_3_1 + 15) {
                cb2[15] = S[p_4_1 + 15 * nq64];
            }
            asm volatile("" ::: "memory");
            #pragma unroll 1
            for (int j_1 = 0; j_1 < num_chunks; j_1++) {
                int t0_4 = ((j_1 + 1) * 4 + w) * 16;
                int t0_5_1 = t0_4;
                long long p_6 = cbase + (long long)t0_5_1 * nq64;
                nb2[0] = 4286578688;
                if (lim2 > t0_5_1) {
                    nb2[0] = S[p_6];
                }
                nb2[1] = 4286578688;
                if (lim2 > t0_5_1 + 1) {
                    nb2[1] = S[p_6 + nq64];
                }
                nb2[2] = 4286578688;
                if (lim2 > t0_5_1 + 2) {
                    nb2[2] = S[p_6 + 2 * nq64];
                }
                nb2[3] = 4286578688;
                if (lim2 > t0_5_1 + 3) {
                    nb2[3] = S[p_6 + 3 * nq64];
                }
                nb2[4] = 4286578688;
                if (lim2 > t0_5_1 + 4) {
                    nb2[4] = S[p_6 + 4 * nq64];
                }
                nb2[5] = 4286578688;
                if (lim2 > t0_5_1 + 5) {
                    nb2[5] = S[p_6 + 5 * nq64];
                }
                nb2[6] = 4286578688;
                if (lim2 > t0_5_1 + 6) {
                    nb2[6] = S[p_6 + 6 * nq64];
                }
                nb2[7] = 4286578688;
                if (lim2 > t0_5_1 + 7) {
                    nb2[7] = S[p_6 + 7 * nq64];
                }
                nb2[8] = 4286578688;
                if (lim2 > t0_5_1 + 8) {
                    nb2[8] = S[p_6 + 8 * nq64];
                }
                nb2[9] = 4286578688;
                if (lim2 > t0_5_1 + 9) {
                    nb2[9] = S[p_6 + 9 * nq64];
                }
                nb2[10] = 4286578688;
                if (lim2 > t0_5_1 + 10) {
                    nb2[10] = S[p_6 + 10 * nq64];
                }
                nb2[11] = 4286578688;
                if (lim2 > t0_5_1 + 11) {
                    nb2[11] = S[p_6 + 11 * nq64];
                }
                nb2[12] = 4286578688;
                if (lim2 > t0_5_1 + 12) {
                    nb2[12] = S[p_6 + 12 * nq64];
                }
                nb2[13] = 4286578688;
                if (lim2 > t0_5_1 + 13) {
                    nb2[13] = S[p_6 + 13 * nq64];
                }
                nb2[14] = 4286578688;
                if (lim2 > t0_5_1 + 14) {
                    nb2[14] = S[p_6 + 14 * nq64];
                }
                nb2[15] = 4286578688;
                if (lim2 > t0_5_1 + 15) {
                    nb2[15] = S[p_6 + 15 * nq64];
                }
                asm volatile("" ::: "memory");
                int t0_7 = (j_1 * 4 + w) * 16;
                int t0_8 = t0_7;
                float qf = __uint_as_float(qc);
                float sc_1 = __uint_as_float(cb2[0]);
                float _fmax_205 = fmaxf(sc_1, -1.7014118346046923e+38f);
                sc_1 = _fmax_205;
                float _min_172 = fminf(sc_1, 1.7014118346046923e+38f);
                sc_1 = _min_172;
                sc_1 = sc_1;
                float sc_9_1 = sc_1;
                unsigned int u = __as_u32(sc_9_1);
                unsigned int cls = u & 4294967232u;
                int f_16 = 0;
                if (t0_8 < fb || t0_8 >= lim - fe && lim > t0_8) {
                    f_16 = 1;
                }
                if (f_16 != 0) {
                    cls = 2139094976;
                }
                unsigned int key_1 = 0;
                if (qf < __uint_as_float(cls) && cls < 4278190080u) {
                    key_1 = 1073741824 | (unsigned int)t0_8;
                }
                if (cls == qc) {
                    unsigned int lowb = (u ^ (unsigned int)((int)u >> 31) & 63) & 63;
                    key_1 = 536870912 | lowb << 6 | (unsigned int)t0_8;
                }
                kb2[0] = __uint_as_float(key_1);
                float sc_10 = __uint_as_float(cb2[1]);
                float _fmax_206 = fmaxf(sc_10, -1.7014118346046923e+38f);
                sc_10 = _fmax_206;
                float _min_173 = fminf(sc_10, 1.7014118346046923e+38f);
                sc_10 = _min_173;
                sc_10 = sc_10;
                float sc_11_1 = sc_10;
                unsigned int u_12 = __as_u32(sc_11_1);
                unsigned int cls_13 = u_12 & 4294967232u;
                int f_14_1 = 0;
                if (t0_8 + 1 < fb || t0_8 + 1 >= lim - fe && lim > t0_8 + 1) {
                    f_14_1 = 1;
                }
                if (f_14_1 != 0) {
                    cls_13 = 2139094976;
                }
                unsigned int key_15 = 0;
                if (qf < __uint_as_float(cls_13) && cls_13 < 4278190080u) {
                    key_15 = 1073741824 | (unsigned int)(t0_8 + 1);
                }
                if (cls_13 == qc) {
                    unsigned int lowb_1 = (u_12 ^ (unsigned int)((int)u_12 >> 31) & 63) & 63;
                    key_15 = 536870912 | lowb_1 << 6 | (unsigned int)(t0_8 + 1);
                }
                kb2[1] = __uint_as_float(key_15);
                float sc_16 = __uint_as_float(cb2[2]);
                float _fmax_207 = fmaxf(sc_16, -1.7014118346046923e+38f);
                sc_16 = _fmax_207;
                float _min_174 = fminf(sc_16, 1.7014118346046923e+38f);
                sc_16 = _min_174;
                sc_16 = sc_16;
                float sc_17_1 = sc_16;
                unsigned int u_18 = __as_u32(sc_17_1);
                unsigned int cls_19 = u_18 & 4294967232u;
                int f_20 = 0;
                if (t0_8 + 2 < fb || t0_8 + 2 >= lim - fe && lim > t0_8 + 2) {
                    f_20 = 1;
                }
                if (f_20 != 0) {
                    cls_19 = 2139094976;
                }
                unsigned int key_21 = 0;
                if (qf < __uint_as_float(cls_19) && cls_19 < 4278190080u) {
                    key_21 = 1073741824 | (unsigned int)(t0_8 + 2);
                }
                if (cls_19 == qc) {
                    unsigned int lowb_2 = (u_18 ^ (unsigned int)((int)u_18 >> 31) & 63) & 63;
                    key_21 = 536870912 | lowb_2 << 6 | (unsigned int)(t0_8 + 2);
                }
                kb2[2] = __uint_as_float(key_21);
                float sc_22 = __uint_as_float(cb2[3]);
                float _fmax_208 = fmaxf(sc_22, -1.7014118346046923e+38f);
                sc_22 = _fmax_208;
                float _min_175 = fminf(sc_22, 1.7014118346046923e+38f);
                sc_22 = _min_175;
                sc_22 = sc_22;
                float sc_23_1 = sc_22;
                unsigned int u_24 = __as_u32(sc_23_1);
                unsigned int cls_25 = u_24 & 4294967232u;
                int f_26 = 0;
                if (t0_8 + 3 < fb || t0_8 + 3 >= lim - fe && lim > t0_8 + 3) {
                    f_26 = 1;
                }
                if (f_26 != 0) {
                    cls_25 = 2139094976;
                }
                unsigned int key_27 = 0;
                if (qf < __uint_as_float(cls_25) && cls_25 < 4278190080u) {
                    key_27 = 1073741824 | (unsigned int)(t0_8 + 3);
                }
                if (cls_25 == qc) {
                    unsigned int lowb_3 = (u_24 ^ (unsigned int)((int)u_24 >> 31) & 63) & 63;
                    key_27 = 536870912 | lowb_3 << 6 | (unsigned int)(t0_8 + 3);
                }
                kb2[3] = __uint_as_float(key_27);
                float sc_28 = __uint_as_float(cb2[4]);
                float _fmax_209 = fmaxf(sc_28, -1.7014118346046923e+38f);
                sc_28 = _fmax_209;
                float _min_176 = fminf(sc_28, 1.7014118346046923e+38f);
                sc_28 = _min_176;
                sc_28 = sc_28;
                float sc_29_1 = sc_28;
                unsigned int u_30 = __as_u32(sc_29_1);
                unsigned int cls_31 = u_30 & 4294967232u;
                int f_32 = 0;
                if (t0_8 + 4 < fb || t0_8 + 4 >= lim - fe && lim > t0_8 + 4) {
                    f_32 = 1;
                }
                if (f_32 != 0) {
                    cls_31 = 2139094976;
                }
                unsigned int key_33 = 0;
                if (qf < __uint_as_float(cls_31) && cls_31 < 4278190080u) {
                    key_33 = 1073741824 | (unsigned int)(t0_8 + 4);
                }
                if (cls_31 == qc) {
                    unsigned int lowb_4 = (u_30 ^ (unsigned int)((int)u_30 >> 31) & 63) & 63;
                    key_33 = 536870912 | lowb_4 << 6 | (unsigned int)(t0_8 + 4);
                }
                kb2[4] = __uint_as_float(key_33);
                float sc_34 = __uint_as_float(cb2[5]);
                float _fmax_210 = fmaxf(sc_34, -1.7014118346046923e+38f);
                sc_34 = _fmax_210;
                float _min_177 = fminf(sc_34, 1.7014118346046923e+38f);
                sc_34 = _min_177;
                sc_34 = sc_34;
                float sc_35_1 = sc_34;
                unsigned int u_36 = __as_u32(sc_35_1);
                unsigned int cls_37 = u_36 & 4294967232u;
                int f_38 = 0;
                if (t0_8 + 5 < fb || t0_8 + 5 >= lim - fe && lim > t0_8 + 5) {
                    f_38 = 1;
                }
                if (f_38 != 0) {
                    cls_37 = 2139094976;
                }
                unsigned int key_39 = 0;
                if (qf < __uint_as_float(cls_37) && cls_37 < 4278190080u) {
                    key_39 = 1073741824 | (unsigned int)(t0_8 + 5);
                }
                if (cls_37 == qc) {
                    unsigned int lowb_5 = (u_36 ^ (unsigned int)((int)u_36 >> 31) & 63) & 63;
                    key_39 = 536870912 | lowb_5 << 6 | (unsigned int)(t0_8 + 5);
                }
                kb2[5] = __uint_as_float(key_39);
                float sc_40 = __uint_as_float(cb2[6]);
                float _fmax_211 = fmaxf(sc_40, -1.7014118346046923e+38f);
                sc_40 = _fmax_211;
                float _min_178 = fminf(sc_40, 1.7014118346046923e+38f);
                sc_40 = _min_178;
                sc_40 = sc_40;
                float sc_41_1 = sc_40;
                unsigned int u_42 = __as_u32(sc_41_1);
                unsigned int cls_43 = u_42 & 4294967232u;
                int f_44 = 0;
                if (t0_8 + 6 < fb || t0_8 + 6 >= lim - fe && lim > t0_8 + 6) {
                    f_44 = 1;
                }
                if (f_44 != 0) {
                    cls_43 = 2139094976;
                }
                unsigned int key_45 = 0;
                if (qf < __uint_as_float(cls_43) && cls_43 < 4278190080u) {
                    key_45 = 1073741824 | (unsigned int)(t0_8 + 6);
                }
                if (cls_43 == qc) {
                    unsigned int lowb_6 = (u_42 ^ (unsigned int)((int)u_42 >> 31) & 63) & 63;
                    key_45 = 536870912 | lowb_6 << 6 | (unsigned int)(t0_8 + 6);
                }
                kb2[6] = __uint_as_float(key_45);
                float sc_46 = __uint_as_float(cb2[7]);
                float _fmax_212 = fmaxf(sc_46, -1.7014118346046923e+38f);
                sc_46 = _fmax_212;
                float _min_179 = fminf(sc_46, 1.7014118346046923e+38f);
                sc_46 = _min_179;
                sc_46 = sc_46;
                float sc_47_1 = sc_46;
                unsigned int u_48 = __as_u32(sc_47_1);
                unsigned int cls_49 = u_48 & 4294967232u;
                int f_50 = 0;
                if (t0_8 + 7 < fb || t0_8 + 7 >= lim - fe && lim > t0_8 + 7) {
                    f_50 = 1;
                }
                if (f_50 != 0) {
                    cls_49 = 2139094976;
                }
                unsigned int key_51 = 0;
                if (qf < __uint_as_float(cls_49) && cls_49 < 4278190080u) {
                    key_51 = 1073741824 | (unsigned int)(t0_8 + 7);
                }
                if (cls_49 == qc) {
                    unsigned int lowb_7 = (u_48 ^ (unsigned int)((int)u_48 >> 31) & 63) & 63;
                    key_51 = 536870912 | lowb_7 << 6 | (unsigned int)(t0_8 + 7);
                }
                kb2[7] = __uint_as_float(key_51);
                float sc_52 = __uint_as_float(cb2[8]);
                float _fmax_213 = fmaxf(sc_52, -1.7014118346046923e+38f);
                sc_52 = _fmax_213;
                float _min_180 = fminf(sc_52, 1.7014118346046923e+38f);
                sc_52 = _min_180;
                sc_52 = sc_52;
                float sc_53 = sc_52;
                unsigned int u_54 = __as_u32(sc_53);
                unsigned int cls_55 = u_54 & 4294967232u;
                int f_56 = 0;
                if (t0_8 + 8 < fb || t0_8 + 8 >= lim - fe && lim > t0_8 + 8) {
                    f_56 = 1;
                }
                if (f_56 != 0) {
                    cls_55 = 2139094976;
                }
                unsigned int key_57 = 0;
                if (qf < __uint_as_float(cls_55) && cls_55 < 4278190080u) {
                    key_57 = 1073741824 | (unsigned int)(t0_8 + 8);
                }
                if (cls_55 == qc) {
                    unsigned int lowb_8 = (u_54 ^ (unsigned int)((int)u_54 >> 31) & 63) & 63;
                    key_57 = 536870912 | lowb_8 << 6 | (unsigned int)(t0_8 + 8);
                }
                kb2[8] = __uint_as_float(key_57);
                float sc_58 = __uint_as_float(cb2[9]);
                float _fmax_214 = fmaxf(sc_58, -1.7014118346046923e+38f);
                sc_58 = _fmax_214;
                float _min_181 = fminf(sc_58, 1.7014118346046923e+38f);
                sc_58 = _min_181;
                sc_58 = sc_58;
                float sc_59 = sc_58;
                unsigned int u_60 = __as_u32(sc_59);
                unsigned int cls_61 = u_60 & 4294967232u;
                int f_62 = 0;
                if (t0_8 + 9 < fb || t0_8 + 9 >= lim - fe && lim > t0_8 + 9) {
                    f_62 = 1;
                }
                if (f_62 != 0) {
                    cls_61 = 2139094976;
                }
                unsigned int key_63 = 0;
                if (qf < __uint_as_float(cls_61) && cls_61 < 4278190080u) {
                    key_63 = 1073741824 | (unsigned int)(t0_8 + 9);
                }
                if (cls_61 == qc) {
                    unsigned int lowb_9 = (u_60 ^ (unsigned int)((int)u_60 >> 31) & 63) & 63;
                    key_63 = 536870912 | lowb_9 << 6 | (unsigned int)(t0_8 + 9);
                }
                kb2[9] = __uint_as_float(key_63);
                float sc_64 = __uint_as_float(cb2[10]);
                float _fmax_215 = fmaxf(sc_64, -1.7014118346046923e+38f);
                sc_64 = _fmax_215;
                float _min_182 = fminf(sc_64, 1.7014118346046923e+38f);
                sc_64 = _min_182;
                sc_64 = sc_64;
                float sc_65 = sc_64;
                unsigned int u_66 = __as_u32(sc_65);
                unsigned int cls_67 = u_66 & 4294967232u;
                int f_68 = 0;
                if (t0_8 + 10 < fb || t0_8 + 10 >= lim - fe && lim > t0_8 + 10) {
                    f_68 = 1;
                }
                if (f_68 != 0) {
                    cls_67 = 2139094976;
                }
                unsigned int key_69 = 0;
                if (qf < __uint_as_float(cls_67) && cls_67 < 4278190080u) {
                    key_69 = 1073741824 | (unsigned int)(t0_8 + 10);
                }
                if (cls_67 == qc) {
                    unsigned int lowb_10 = (u_66 ^ (unsigned int)((int)u_66 >> 31) & 63) & 63;
                    key_69 = 536870912 | lowb_10 << 6 | (unsigned int)(t0_8 + 10);
                }
                kb2[10] = __uint_as_float(key_69);
                float sc_70 = __uint_as_float(cb2[11]);
                float _fmax_216 = fmaxf(sc_70, -1.7014118346046923e+38f);
                sc_70 = _fmax_216;
                float _min_183 = fminf(sc_70, 1.7014118346046923e+38f);
                sc_70 = _min_183;
                sc_70 = sc_70;
                float sc_71 = sc_70;
                unsigned int u_72 = __as_u32(sc_71);
                unsigned int cls_73 = u_72 & 4294967232u;
                int f_74 = 0;
                if (t0_8 + 11 < fb || t0_8 + 11 >= lim - fe && lim > t0_8 + 11) {
                    f_74 = 1;
                }
                if (f_74 != 0) {
                    cls_73 = 2139094976;
                }
                unsigned int key_75 = 0;
                if (qf < __uint_as_float(cls_73) && cls_73 < 4278190080u) {
                    key_75 = 1073741824 | (unsigned int)(t0_8 + 11);
                }
                if (cls_73 == qc) {
                    unsigned int lowb_11 = (u_72 ^ (unsigned int)((int)u_72 >> 31) & 63) & 63;
                    key_75 = 536870912 | lowb_11 << 6 | (unsigned int)(t0_8 + 11);
                }
                kb2[11] = __uint_as_float(key_75);
                float sc_76 = __uint_as_float(cb2[12]);
                float _fmax_217 = fmaxf(sc_76, -1.7014118346046923e+38f);
                sc_76 = _fmax_217;
                float _min_184 = fminf(sc_76, 1.7014118346046923e+38f);
                sc_76 = _min_184;
                sc_76 = sc_76;
                float sc_77 = sc_76;
                unsigned int u_78 = __as_u32(sc_77);
                unsigned int cls_79 = u_78 & 4294967232u;
                int f_80 = 0;
                if (t0_8 + 12 < fb || t0_8 + 12 >= lim - fe && lim > t0_8 + 12) {
                    f_80 = 1;
                }
                if (f_80 != 0) {
                    cls_79 = 2139094976;
                }
                unsigned int key_81 = 0;
                if (qf < __uint_as_float(cls_79) && cls_79 < 4278190080u) {
                    key_81 = 1073741824 | (unsigned int)(t0_8 + 12);
                }
                if (cls_79 == qc) {
                    unsigned int lowb_12 = (u_78 ^ (unsigned int)((int)u_78 >> 31) & 63) & 63;
                    key_81 = 536870912 | lowb_12 << 6 | (unsigned int)(t0_8 + 12);
                }
                kb2[12] = __uint_as_float(key_81);
                float sc_82 = __uint_as_float(cb2[13]);
                float _fmax_218 = fmaxf(sc_82, -1.7014118346046923e+38f);
                sc_82 = _fmax_218;
                float _min_185 = fminf(sc_82, 1.7014118346046923e+38f);
                sc_82 = _min_185;
                sc_82 = sc_82;
                float sc_83 = sc_82;
                unsigned int u_84 = __as_u32(sc_83);
                unsigned int cls_85 = u_84 & 4294967232u;
                int f_86 = 0;
                if (t0_8 + 13 < fb || t0_8 + 13 >= lim - fe && lim > t0_8 + 13) {
                    f_86 = 1;
                }
                if (f_86 != 0) {
                    cls_85 = 2139094976;
                }
                unsigned int key_87 = 0;
                if (qf < __uint_as_float(cls_85) && cls_85 < 4278190080u) {
                    key_87 = 1073741824 | (unsigned int)(t0_8 + 13);
                }
                if (cls_85 == qc) {
                    unsigned int lowb_13 = (u_84 ^ (unsigned int)((int)u_84 >> 31) & 63) & 63;
                    key_87 = 536870912 | lowb_13 << 6 | (unsigned int)(t0_8 + 13);
                }
                kb2[13] = __uint_as_float(key_87);
                float sc_88 = __uint_as_float(cb2[14]);
                float _fmax_219 = fmaxf(sc_88, -1.7014118346046923e+38f);
                sc_88 = _fmax_219;
                float _min_186 = fminf(sc_88, 1.7014118346046923e+38f);
                sc_88 = _min_186;
                sc_88 = sc_88;
                float sc_89 = sc_88;
                unsigned int u_90 = __as_u32(sc_89);
                unsigned int cls_91 = u_90 & 4294967232u;
                int f_92 = 0;
                if (t0_8 + 14 < fb || t0_8 + 14 >= lim - fe && lim > t0_8 + 14) {
                    f_92 = 1;
                }
                if (f_92 != 0) {
                    cls_91 = 2139094976;
                }
                unsigned int key_93 = 0;
                if (qf < __uint_as_float(cls_91) && cls_91 < 4278190080u) {
                    key_93 = 1073741824 | (unsigned int)(t0_8 + 14);
                }
                if (cls_91 == qc) {
                    unsigned int lowb_14 = (u_90 ^ (unsigned int)((int)u_90 >> 31) & 63) & 63;
                    key_93 = 536870912 | lowb_14 << 6 | (unsigned int)(t0_8 + 14);
                }
                kb2[14] = __uint_as_float(key_93);
                float sc_94 = __uint_as_float(cb2[15]);
                float _fmax_220 = fmaxf(sc_94, -1.7014118346046923e+38f);
                sc_94 = _fmax_220;
                float _min_187 = fminf(sc_94, 1.7014118346046923e+38f);
                sc_94 = _min_187;
                sc_94 = sc_94;
                float sc_95 = sc_94;
                unsigned int u_96 = __as_u32(sc_95);
                unsigned int cls_97 = u_96 & 4294967232u;
                int f_98 = 0;
                if (t0_8 + 15 < fb || t0_8 + 15 >= lim - fe && lim > t0_8 + 15) {
                    f_98 = 1;
                }
                if (f_98 != 0) {
                    cls_97 = 2139094976;
                }
                unsigned int key_99 = 0;
                if (qf < __uint_as_float(cls_97) && cls_97 < 4278190080u) {
                    key_99 = 1073741824 | (unsigned int)(t0_8 + 15);
                }
                if (cls_97 == qc) {
                    unsigned int lowb_15 = (u_96 ^ (unsigned int)((int)u_96 >> 31) & 63) & 63;
                    key_99 = 536870912 | lowb_15 << 6 | (unsigned int)(t0_8 + 15);
                }
                kb2[15] = __uint_as_float(key_99);
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
                int _vote_1 = __any_sync(0xFFFFFFFF, nzc != 0);
                if (_vote_1 != 0) {
                    float _fmax_221 = fmaxf(kb2[0], kb2[13]);
                    float hi_3 = _fmax_221;
                    float _min_188 = fminf(kb2[0], kb2[13]);
                    float lo_2 = _min_188;
                    kb2[0] = hi_3;
                    kb2[13] = lo_2;
                    float _fmax_222 = fmaxf(kb2[1], kb2[12]);
                    float hi_0_1 = _fmax_222;
                    float _min_189 = fminf(kb2[1], kb2[12]);
                    float lo_1_2 = _min_189;
                    kb2[1] = hi_0_1;
                    kb2[12] = lo_1_2;
                    float _fmax_223 = fmaxf(kb2[2], kb2[15]);
                    float hi_2_1 = _fmax_223;
                    float _min_190 = fminf(kb2[2], kb2[15]);
                    float lo_3_1 = _min_190;
                    kb2[2] = hi_2_1;
                    kb2[15] = lo_3_1;
                    float _fmax_224 = fmaxf(kb2[3], kb2[14]);
                    float hi_4_1 = _fmax_224;
                    float _min_191 = fminf(kb2[3], kb2[14]);
                    float lo_5_1 = _min_191;
                    kb2[3] = hi_4_1;
                    kb2[14] = lo_5_1;
                    float _fmax_225 = fmaxf(kb2[4], kb2[8]);
                    float hi_6_1 = _fmax_225;
                    float _min_192 = fminf(kb2[4], kb2[8]);
                    float lo_7_1 = _min_192;
                    kb2[4] = hi_6_1;
                    kb2[8] = lo_7_1;
                    float _fmax_226 = fmaxf(kb2[5], kb2[6]);
                    float hi_8_1 = _fmax_226;
                    float _min_193 = fminf(kb2[5], kb2[6]);
                    float lo_9_1 = _min_193;
                    kb2[5] = hi_8_1;
                    kb2[6] = lo_9_1;
                    float _fmax_227 = fmaxf(kb2[7], kb2[11]);
                    float hi_10_1 = _fmax_227;
                    float _min_194 = fminf(kb2[7], kb2[11]);
                    float lo_11_1 = _min_194;
                    kb2[7] = hi_10_1;
                    kb2[11] = lo_11_1;
                    float _fmax_228 = fmaxf(kb2[9], kb2[10]);
                    float hi_12_1 = _fmax_228;
                    float _min_195 = fminf(kb2[9], kb2[10]);
                    float lo_13_1 = _min_195;
                    kb2[9] = hi_12_1;
                    kb2[10] = lo_13_1;
                    float _fmax_229 = fmaxf(kb2[0], kb2[5]);
                    float hi_14_1 = _fmax_229;
                    float _min_196 = fminf(kb2[0], kb2[5]);
                    float lo_15_1 = _min_196;
                    kb2[0] = hi_14_1;
                    kb2[5] = lo_15_1;
                    float _fmax_230 = fmaxf(kb2[1], kb2[7]);
                    float hi_16_1 = _fmax_230;
                    float _min_197 = fminf(kb2[1], kb2[7]);
                    float lo_17_1 = _min_197;
                    kb2[1] = hi_16_1;
                    kb2[7] = lo_17_1;
                    float _fmax_231 = fmaxf(kb2[2], kb2[9]);
                    float hi_18_1 = _fmax_231;
                    float _min_198 = fminf(kb2[2], kb2[9]);
                    float lo_19_1 = _min_198;
                    kb2[2] = hi_18_1;
                    kb2[9] = lo_19_1;
                    float _fmax_232 = fmaxf(kb2[3], kb2[4]);
                    float hi_20_1 = _fmax_232;
                    float _min_199 = fminf(kb2[3], kb2[4]);
                    float lo_21_1 = _min_199;
                    kb2[3] = hi_20_1;
                    kb2[4] = lo_21_1;
                    float _fmax_233 = fmaxf(kb2[6], kb2[13]);
                    float hi_22_1 = _fmax_233;
                    float _min_200 = fminf(kb2[6], kb2[13]);
                    float lo_23_1 = _min_200;
                    kb2[6] = hi_22_1;
                    kb2[13] = lo_23_1;
                    float _fmax_234 = fmaxf(kb2[8], kb2[14]);
                    float hi_24_1 = _fmax_234;
                    float _min_201 = fminf(kb2[8], kb2[14]);
                    float lo_25_1 = _min_201;
                    kb2[8] = hi_24_1;
                    kb2[14] = lo_25_1;
                    float _fmax_235 = fmaxf(kb2[10], kb2[15]);
                    float hi_26_1 = _fmax_235;
                    float _min_202 = fminf(kb2[10], kb2[15]);
                    float lo_27_1 = _min_202;
                    kb2[10] = hi_26_1;
                    kb2[15] = lo_27_1;
                    float _fmax_236 = fmaxf(kb2[11], kb2[12]);
                    float hi_28_1 = _fmax_236;
                    float _min_203 = fminf(kb2[11], kb2[12]);
                    float lo_29_1 = _min_203;
                    kb2[11] = hi_28_1;
                    kb2[12] = lo_29_1;
                    float _fmax_237 = fmaxf(kb2[0], kb2[1]);
                    float hi_30_1 = _fmax_237;
                    float _min_204 = fminf(kb2[0], kb2[1]);
                    float lo_31_1 = _min_204;
                    kb2[0] = hi_30_1;
                    kb2[1] = lo_31_1;
                    float _fmax_238 = fmaxf(kb2[2], kb2[3]);
                    float hi_32_1 = _fmax_238;
                    float _min_205 = fminf(kb2[2], kb2[3]);
                    float lo_33_1 = _min_205;
                    kb2[2] = hi_32_1;
                    kb2[3] = lo_33_1;
                    float _fmax_239 = fmaxf(kb2[4], kb2[5]);
                    float hi_34_1 = _fmax_239;
                    float _min_206 = fminf(kb2[4], kb2[5]);
                    float lo_35_1 = _min_206;
                    kb2[4] = hi_34_1;
                    kb2[5] = lo_35_1;
                    float _fmax_240 = fmaxf(kb2[6], kb2[8]);
                    float hi_36_1 = _fmax_240;
                    float _min_207 = fminf(kb2[6], kb2[8]);
                    float lo_37_1 = _min_207;
                    kb2[6] = hi_36_1;
                    kb2[8] = lo_37_1;
                    float _fmax_241 = fmaxf(kb2[7], kb2[9]);
                    float hi_38_1 = _fmax_241;
                    float _min_208 = fminf(kb2[7], kb2[9]);
                    float lo_39_1 = _min_208;
                    kb2[7] = hi_38_1;
                    kb2[9] = lo_39_1;
                    float _fmax_242 = fmaxf(kb2[10], kb2[11]);
                    float hi_40_1 = _fmax_242;
                    float _min_209 = fminf(kb2[10], kb2[11]);
                    float lo_41_1 = _min_209;
                    kb2[10] = hi_40_1;
                    kb2[11] = lo_41_1;
                    float _fmax_243 = fmaxf(kb2[12], kb2[13]);
                    float hi_42_1 = _fmax_243;
                    float _min_210 = fminf(kb2[12], kb2[13]);
                    float lo_43_1 = _min_210;
                    kb2[12] = hi_42_1;
                    kb2[13] = lo_43_1;
                    float _fmax_244 = fmaxf(kb2[14], kb2[15]);
                    float hi_44_1 = _fmax_244;
                    float _min_211 = fminf(kb2[14], kb2[15]);
                    float lo_45_1 = _min_211;
                    kb2[14] = hi_44_1;
                    kb2[15] = lo_45_1;
                    float _fmax_245 = fmaxf(kb2[0], kb2[2]);
                    float hi_46_1 = _fmax_245;
                    float _min_212 = fminf(kb2[0], kb2[2]);
                    float lo_47_1 = _min_212;
                    kb2[0] = hi_46_1;
                    kb2[2] = lo_47_1;
                    float _fmax_246 = fmaxf(kb2[1], kb2[3]);
                    float hi_48_1 = _fmax_246;
                    float _min_213 = fminf(kb2[1], kb2[3]);
                    float lo_49_1 = _min_213;
                    kb2[1] = hi_48_1;
                    kb2[3] = lo_49_1;
                    float _fmax_247 = fmaxf(kb2[4], kb2[10]);
                    float hi_50_1 = _fmax_247;
                    float _min_214 = fminf(kb2[4], kb2[10]);
                    float lo_51_1 = _min_214;
                    kb2[4] = hi_50_1;
                    kb2[10] = lo_51_1;
                    float _fmax_248 = fmaxf(kb2[5], kb2[11]);
                    float hi_52_1 = _fmax_248;
                    float _min_215 = fminf(kb2[5], kb2[11]);
                    float lo_53_1 = _min_215;
                    kb2[5] = hi_52_1;
                    kb2[11] = lo_53_1;
                    float _fmax_249 = fmaxf(kb2[6], kb2[7]);
                    float hi_54_1 = _fmax_249;
                    float _min_216 = fminf(kb2[6], kb2[7]);
                    float lo_55_1 = _min_216;
                    kb2[6] = hi_54_1;
                    kb2[7] = lo_55_1;
                    float _fmax_250 = fmaxf(kb2[8], kb2[9]);
                    float hi_56_1 = _fmax_250;
                    float _min_217 = fminf(kb2[8], kb2[9]);
                    float lo_57_1 = _min_217;
                    kb2[8] = hi_56_1;
                    kb2[9] = lo_57_1;
                    float _fmax_251 = fmaxf(kb2[12], kb2[14]);
                    float hi_58_1 = _fmax_251;
                    float _min_218 = fminf(kb2[12], kb2[14]);
                    float lo_59_1 = _min_218;
                    kb2[12] = hi_58_1;
                    kb2[14] = lo_59_1;
                    float _fmax_252 = fmaxf(kb2[13], kb2[15]);
                    float hi_60_1 = _fmax_252;
                    float _min_219 = fminf(kb2[13], kb2[15]);
                    float lo_61_1 = _min_219;
                    kb2[13] = hi_60_1;
                    kb2[15] = lo_61_1;
                    float _fmax_253 = fmaxf(kb2[1], kb2[2]);
                    float hi_62_1 = _fmax_253;
                    float _min_220 = fminf(kb2[1], kb2[2]);
                    float lo_63_1 = _min_220;
                    kb2[1] = hi_62_1;
                    kb2[2] = lo_63_1;
                    float _fmax_254 = fmaxf(kb2[3], kb2[12]);
                    float hi_64_1 = _fmax_254;
                    float _min_221 = fminf(kb2[3], kb2[12]);
                    float lo_65_1 = _min_221;
                    kb2[3] = hi_64_1;
                    kb2[12] = lo_65_1;
                    float _fmax_255 = fmaxf(kb2[4], kb2[6]);
                    float hi_66_1 = _fmax_255;
                    float _min_222 = fminf(kb2[4], kb2[6]);
                    float lo_67_1 = _min_222;
                    kb2[4] = hi_66_1;
                    kb2[6] = lo_67_1;
                    float _fmax_256 = fmaxf(kb2[5], kb2[7]);
                    float hi_68_1 = _fmax_256;
                    float _min_223 = fminf(kb2[5], kb2[7]);
                    float lo_69_1 = _min_223;
                    kb2[5] = hi_68_1;
                    kb2[7] = lo_69_1;
                    float _fmax_257 = fmaxf(kb2[8], kb2[10]);
                    float hi_70_1 = _fmax_257;
                    float _min_224 = fminf(kb2[8], kb2[10]);
                    float lo_71_1 = _min_224;
                    kb2[8] = hi_70_1;
                    kb2[10] = lo_71_1;
                    float _fmax_258 = fmaxf(kb2[9], kb2[11]);
                    float hi_72_1 = _fmax_258;
                    float _min_225 = fminf(kb2[9], kb2[11]);
                    float lo_73_1 = _min_225;
                    kb2[9] = hi_72_1;
                    kb2[11] = lo_73_1;
                    float _fmax_259 = fmaxf(kb2[13], kb2[14]);
                    float hi_74_1 = _fmax_259;
                    float _min_226 = fminf(kb2[13], kb2[14]);
                    float lo_75_1 = _min_226;
                    kb2[13] = hi_74_1;
                    kb2[14] = lo_75_1;
                    float _fmax_260 = fmaxf(kb2[1], kb2[4]);
                    float hi_76_1 = _fmax_260;
                    float _min_227 = fminf(kb2[1], kb2[4]);
                    float lo_77_1 = _min_227;
                    kb2[1] = hi_76_1;
                    kb2[4] = lo_77_1;
                    float _fmax_261 = fmaxf(kb2[2], kb2[6]);
                    float hi_78_1 = _fmax_261;
                    float _min_228 = fminf(kb2[2], kb2[6]);
                    float lo_79_1 = _min_228;
                    kb2[2] = hi_78_1;
                    kb2[6] = lo_79_1;
                    float _fmax_262 = fmaxf(kb2[5], kb2[8]);
                    float hi_80_1 = _fmax_262;
                    float _min_229 = fminf(kb2[5], kb2[8]);
                    float lo_81_1 = _min_229;
                    kb2[5] = hi_80_1;
                    kb2[8] = lo_81_1;
                    float _fmax_263 = fmaxf(kb2[7], kb2[10]);
                    float hi_82_1 = _fmax_263;
                    float _min_230 = fminf(kb2[7], kb2[10]);
                    float lo_83_1 = _min_230;
                    kb2[7] = hi_82_1;
                    kb2[10] = lo_83_1;
                    float _fmax_264 = fmaxf(kb2[9], kb2[13]);
                    float hi_84_1 = _fmax_264;
                    float _min_231 = fminf(kb2[9], kb2[13]);
                    float lo_85_1 = _min_231;
                    kb2[9] = hi_84_1;
                    kb2[13] = lo_85_1;
                    float _fmax_265 = fmaxf(kb2[11], kb2[14]);
                    float hi_86_1 = _fmax_265;
                    float _min_232 = fminf(kb2[11], kb2[14]);
                    float lo_87_1 = _min_232;
                    kb2[11] = hi_86_1;
                    kb2[14] = lo_87_1;
                    float _fmax_266 = fmaxf(kb2[2], kb2[4]);
                    float hi_88_1 = _fmax_266;
                    float _min_233 = fminf(kb2[2], kb2[4]);
                    float lo_89_1 = _min_233;
                    kb2[2] = hi_88_1;
                    kb2[4] = lo_89_1;
                    float _fmax_267 = fmaxf(kb2[3], kb2[6]);
                    float hi_90_1 = _fmax_267;
                    float _min_234 = fminf(kb2[3], kb2[6]);
                    float lo_91_1 = _min_234;
                    kb2[3] = hi_90_1;
                    kb2[6] = lo_91_1;
                    float _fmax_268 = fmaxf(kb2[9], kb2[12]);
                    float hi_92_1 = _fmax_268;
                    float _min_235 = fminf(kb2[9], kb2[12]);
                    float lo_93_1 = _min_235;
                    kb2[9] = hi_92_1;
                    kb2[12] = lo_93_1;
                    float _fmax_269 = fmaxf(kb2[11], kb2[13]);
                    float hi_94 = _fmax_269;
                    float _min_236 = fminf(kb2[11], kb2[13]);
                    float lo_95 = _min_236;
                    kb2[11] = hi_94;
                    kb2[13] = lo_95;
                    float _fmax_270 = fmaxf(kb2[3], kb2[5]);
                    float hi_96 = _fmax_270;
                    float _min_237 = fminf(kb2[3], kb2[5]);
                    float lo_97 = _min_237;
                    kb2[3] = hi_96;
                    kb2[5] = lo_97;
                    float _fmax_271 = fmaxf(kb2[6], kb2[8]);
                    float hi_98 = _fmax_271;
                    float _min_238 = fminf(kb2[6], kb2[8]);
                    float lo_99 = _min_238;
                    kb2[6] = hi_98;
                    kb2[8] = lo_99;
                    float _fmax_272 = fmaxf(kb2[7], kb2[9]);
                    float hi_100 = _fmax_272;
                    float _min_239 = fminf(kb2[7], kb2[9]);
                    float lo_101 = _min_239;
                    kb2[7] = hi_100;
                    kb2[9] = lo_101;
                    float _fmax_273 = fmaxf(kb2[10], kb2[12]);
                    float hi_102 = _fmax_273;
                    float _min_240 = fminf(kb2[10], kb2[12]);
                    float lo_103 = _min_240;
                    kb2[10] = hi_102;
                    kb2[12] = lo_103;
                    float _fmax_274 = fmaxf(kb2[3], kb2[4]);
                    float hi_104 = _fmax_274;
                    float _min_241 = fminf(kb2[3], kb2[4]);
                    float lo_105 = _min_241;
                    kb2[3] = hi_104;
                    kb2[4] = lo_105;
                    float _fmax_275 = fmaxf(kb2[5], kb2[6]);
                    float hi_106 = _fmax_275;
                    float _min_242 = fminf(kb2[5], kb2[6]);
                    float lo_107 = _min_242;
                    kb2[5] = hi_106;
                    kb2[6] = lo_107;
                    float _fmax_276 = fmaxf(kb2[7], kb2[8]);
                    float hi_108 = _fmax_276;
                    float _min_243 = fminf(kb2[7], kb2[8]);
                    float lo_109 = _min_243;
                    kb2[7] = hi_108;
                    kb2[8] = lo_109;
                    float _fmax_277 = fmaxf(kb2[9], kb2[10]);
                    float hi_110 = _fmax_277;
                    float _min_244 = fminf(kb2[9], kb2[10]);
                    float lo_111 = _min_244;
                    kb2[9] = hi_110;
                    kb2[10] = lo_111;
                    float _fmax_278 = fmaxf(kb2[11], kb2[12]);
                    float hi_112 = _fmax_278;
                    float _min_245 = fminf(kb2[11], kb2[12]);
                    float lo_113 = _min_245;
                    kb2[11] = hi_112;
                    kb2[12] = lo_113;
                    float _fmax_279 = fmaxf(kb2[6], kb2[7]);
                    float hi_114 = _fmax_279;
                    float _min_246 = fminf(kb2[6], kb2[7]);
                    float lo_115 = _min_246;
                    kb2[6] = hi_114;
                    kb2[7] = lo_115;
                    float _fmax_280 = fmaxf(kb2[8], kb2[9]);
                    float hi_116 = _fmax_280;
                    float _min_247 = fminf(kb2[8], kb2[9]);
                    float lo_117 = _min_247;
                    kb2[8] = hi_116;
                    kb2[9] = lo_117;
                    float _fmax_281 = fmaxf(a2[0], kb2[15]);
                    float hi_118 = _fmax_281;
                    a2[0] = hi_118;
                    float _fmax_282 = fmaxf(a2[1], kb2[14]);
                    float hi_119_1 = _fmax_282;
                    a2[1] = hi_119_1;
                    float _fmax_283 = fmaxf(a2[2], kb2[13]);
                    float hi_120 = _fmax_283;
                    a2[2] = hi_120;
                    float _fmax_284 = fmaxf(a2[3], kb2[12]);
                    float hi_121_1 = _fmax_284;
                    a2[3] = hi_121_1;
                    float _fmax_285 = fmaxf(a2[4], kb2[11]);
                    float hi_122 = _fmax_285;
                    a2[4] = hi_122;
                    float _fmax_286 = fmaxf(a2[5], kb2[10]);
                    float hi_123_1 = _fmax_286;
                    a2[5] = hi_123_1;
                    float _fmax_287 = fmaxf(a2[6], kb2[9]);
                    float hi_124 = _fmax_287;
                    a2[6] = hi_124;
                    float _fmax_288 = fmaxf(a2[7], kb2[8]);
                    float hi_125_1 = _fmax_288;
                    a2[7] = hi_125_1;
                    float _fmax_289 = fmaxf(a2[8], kb2[7]);
                    float hi_126 = _fmax_289;
                    a2[8] = hi_126;
                    float _fmax_290 = fmaxf(a2[9], kb2[6]);
                    float hi_127_1 = _fmax_290;
                    a2[9] = hi_127_1;
                    float _fmax_291 = fmaxf(a2[10], kb2[5]);
                    float hi_128 = _fmax_291;
                    a2[10] = hi_128;
                    float _fmax_292 = fmaxf(a2[11], kb2[4]);
                    float hi_129_1 = _fmax_292;
                    a2[11] = hi_129_1;
                    float _fmax_293 = fmaxf(a2[12], kb2[3]);
                    float hi_130 = _fmax_293;
                    a2[12] = hi_130;
                    float _fmax_294 = fmaxf(a2[13], kb2[2]);
                    float hi_131_1 = _fmax_294;
                    a2[13] = hi_131_1;
                    float _fmax_295 = fmaxf(a2[14], kb2[1]);
                    float hi_132 = _fmax_295;
                    a2[14] = hi_132;
                    float _fmax_296 = fmaxf(a2[15], kb2[0]);
                    float hi_133_1 = _fmax_296;
                    a2[15] = hi_133_1;
                    float _fmax_297 = fmaxf(a2[0], a2[8]);
                    float hi_134 = _fmax_297;
                    float _min_248 = fminf(a2[0], a2[8]);
                    float lo_135 = _min_248;
                    a2[0] = hi_134;
                    a2[8] = lo_135;
                    float _fmax_298 = fmaxf(a2[1], a2[9]);
                    float hi_136 = _fmax_298;
                    float _min_249 = fminf(a2[1], a2[9]);
                    float lo_137 = _min_249;
                    a2[1] = hi_136;
                    a2[9] = lo_137;
                    float _fmax_299 = fmaxf(a2[2], a2[10]);
                    float hi_138 = _fmax_299;
                    float _min_250 = fminf(a2[2], a2[10]);
                    float lo_139 = _min_250;
                    a2[2] = hi_138;
                    a2[10] = lo_139;
                    float _fmax_300 = fmaxf(a2[3], a2[11]);
                    float hi_140 = _fmax_300;
                    float _min_251 = fminf(a2[3], a2[11]);
                    float lo_141 = _min_251;
                    a2[3] = hi_140;
                    a2[11] = lo_141;
                    float _fmax_301 = fmaxf(a2[4], a2[12]);
                    float hi_142 = _fmax_301;
                    float _min_252 = fminf(a2[4], a2[12]);
                    float lo_143 = _min_252;
                    a2[4] = hi_142;
                    a2[12] = lo_143;
                    float _fmax_302 = fmaxf(a2[5], a2[13]);
                    float hi_144 = _fmax_302;
                    float _min_253 = fminf(a2[5], a2[13]);
                    float lo_145 = _min_253;
                    a2[5] = hi_144;
                    a2[13] = lo_145;
                    float _fmax_303 = fmaxf(a2[6], a2[14]);
                    float hi_146 = _fmax_303;
                    float _min_254 = fminf(a2[6], a2[14]);
                    float lo_147 = _min_254;
                    a2[6] = hi_146;
                    a2[14] = lo_147;
                    float _fmax_304 = fmaxf(a2[7], a2[15]);
                    float hi_148 = _fmax_304;
                    float _min_255 = fminf(a2[7], a2[15]);
                    float lo_149 = _min_255;
                    a2[7] = hi_148;
                    a2[15] = lo_149;
                    float _fmax_305 = fmaxf(a2[0], a2[4]);
                    float hi_150 = _fmax_305;
                    float _min_256 = fminf(a2[0], a2[4]);
                    float lo_151 = _min_256;
                    a2[0] = hi_150;
                    a2[4] = lo_151;
                    float _fmax_306 = fmaxf(a2[1], a2[5]);
                    float hi_152 = _fmax_306;
                    float _min_257 = fminf(a2[1], a2[5]);
                    float lo_153 = _min_257;
                    a2[1] = hi_152;
                    a2[5] = lo_153;
                    float _fmax_307 = fmaxf(a2[2], a2[6]);
                    float hi_154 = _fmax_307;
                    float _min_258 = fminf(a2[2], a2[6]);
                    float lo_155 = _min_258;
                    a2[2] = hi_154;
                    a2[6] = lo_155;
                    float _fmax_308 = fmaxf(a2[3], a2[7]);
                    float hi_156 = _fmax_308;
                    float _min_259 = fminf(a2[3], a2[7]);
                    float lo_157 = _min_259;
                    a2[3] = hi_156;
                    a2[7] = lo_157;
                    float _fmax_309 = fmaxf(a2[8], a2[12]);
                    float hi_158 = _fmax_309;
                    float _min_260 = fminf(a2[8], a2[12]);
                    float lo_159 = _min_260;
                    a2[8] = hi_158;
                    a2[12] = lo_159;
                    float _fmax_310 = fmaxf(a2[9], a2[13]);
                    float hi_160 = _fmax_310;
                    float _min_261 = fminf(a2[9], a2[13]);
                    float lo_161 = _min_261;
                    a2[9] = hi_160;
                    a2[13] = lo_161;
                    float _fmax_311 = fmaxf(a2[10], a2[14]);
                    float hi_162 = _fmax_311;
                    float _min_262 = fminf(a2[10], a2[14]);
                    float lo_163 = _min_262;
                    a2[10] = hi_162;
                    a2[14] = lo_163;
                    float _fmax_312 = fmaxf(a2[11], a2[15]);
                    float hi_164 = _fmax_312;
                    float _min_263 = fminf(a2[11], a2[15]);
                    float lo_165 = _min_263;
                    a2[11] = hi_164;
                    a2[15] = lo_165;
                    float _fmax_313 = fmaxf(a2[0], a2[2]);
                    float hi_166 = _fmax_313;
                    float _min_264 = fminf(a2[0], a2[2]);
                    float lo_167 = _min_264;
                    a2[0] = hi_166;
                    a2[2] = lo_167;
                    float _fmax_314 = fmaxf(a2[1], a2[3]);
                    float hi_168 = _fmax_314;
                    float _min_265 = fminf(a2[1], a2[3]);
                    float lo_169 = _min_265;
                    a2[1] = hi_168;
                    a2[3] = lo_169;
                    float _fmax_315 = fmaxf(a2[4], a2[6]);
                    float hi_170 = _fmax_315;
                    float _min_266 = fminf(a2[4], a2[6]);
                    float lo_171 = _min_266;
                    a2[4] = hi_170;
                    a2[6] = lo_171;
                    float _fmax_316 = fmaxf(a2[5], a2[7]);
                    float hi_172 = _fmax_316;
                    float _min_267 = fminf(a2[5], a2[7]);
                    float lo_173 = _min_267;
                    a2[5] = hi_172;
                    a2[7] = lo_173;
                    float _fmax_317 = fmaxf(a2[8], a2[10]);
                    float hi_174 = _fmax_317;
                    float _min_268 = fminf(a2[8], a2[10]);
                    float lo_175 = _min_268;
                    a2[8] = hi_174;
                    a2[10] = lo_175;
                    float _fmax_318 = fmaxf(a2[9], a2[11]);
                    float hi_176 = _fmax_318;
                    float _min_269 = fminf(a2[9], a2[11]);
                    float lo_177 = _min_269;
                    a2[9] = hi_176;
                    a2[11] = lo_177;
                    float _fmax_319 = fmaxf(a2[12], a2[14]);
                    float hi_178 = _fmax_319;
                    float _min_270 = fminf(a2[12], a2[14]);
                    float lo_179 = _min_270;
                    a2[12] = hi_178;
                    a2[14] = lo_179;
                    float _fmax_320 = fmaxf(a2[13], a2[15]);
                    float hi_180 = _fmax_320;
                    float _min_271 = fminf(a2[13], a2[15]);
                    float lo_181 = _min_271;
                    a2[13] = hi_180;
                    a2[15] = lo_181;
                    float _fmax_321 = fmaxf(a2[0], a2[1]);
                    float hi_182 = _fmax_321;
                    float _min_272 = fminf(a2[0], a2[1]);
                    float lo_183 = _min_272;
                    a2[0] = hi_182;
                    a2[1] = lo_183;
                    float _fmax_322 = fmaxf(a2[2], a2[3]);
                    float hi_184 = _fmax_322;
                    float _min_273 = fminf(a2[2], a2[3]);
                    float lo_185 = _min_273;
                    a2[2] = hi_184;
                    a2[3] = lo_185;
                    float _fmax_323 = fmaxf(a2[4], a2[5]);
                    float hi_186 = _fmax_323;
                    float _min_274 = fminf(a2[4], a2[5]);
                    float lo_187 = _min_274;
                    a2[4] = hi_186;
                    a2[5] = lo_187;
                    float _fmax_324 = fmaxf(a2[6], a2[7]);
                    float hi_188 = _fmax_324;
                    float _min_275 = fminf(a2[6], a2[7]);
                    float lo_189 = _min_275;
                    a2[6] = hi_188;
                    a2[7] = lo_189;
                    float _fmax_325 = fmaxf(a2[8], a2[9]);
                    float hi_190 = _fmax_325;
                    float _min_276 = fminf(a2[8], a2[9]);
                    float lo_191 = _min_276;
                    a2[8] = hi_190;
                    a2[9] = lo_191;
                    float _fmax_326 = fmaxf(a2[10], a2[11]);
                    float hi_192 = _fmax_326;
                    float _min_277 = fminf(a2[10], a2[11]);
                    float lo_193 = _min_277;
                    a2[10] = hi_192;
                    a2[11] = lo_193;
                    float _fmax_327 = fmaxf(a2[12], a2[13]);
                    float hi_194 = _fmax_327;
                    float _min_278 = fminf(a2[12], a2[13]);
                    float lo_195 = _min_278;
                    a2[12] = hi_194;
                    a2[13] = lo_195;
                    float _fmax_328 = fmaxf(a2[14], a2[15]);
                    float hi_196 = _fmax_328;
                    float _min_279 = fminf(a2[14], a2[15]);
                    float lo_197 = _min_279;
                    a2[14] = hi_196;
                    a2[15] = lo_197;
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
        for (int k_1 = 0; k_1 < 2; k_1++) {
            int bit2 = 1 << k_1;
            int low2 = whi & (bit2 << 1) - 1;
            if (low2 == bit2) {
                int base_1 = whi * 544 + c;
                tree[base_1] = a2[0];
                tree[base_1 + 32] = a2[1];
                tree[base_1 + 64] = a2[2];
                tree[base_1 + 96] = a2[3];
                tree[base_1 + 128] = a2[4];
                tree[base_1 + 160] = a2[5];
                tree[base_1 + 192] = a2[6];
                tree[base_1 + 224] = a2[7];
                tree[base_1 + 256] = a2[8];
                tree[base_1 + 288] = a2[9];
                tree[base_1 + 320] = a2[10];
                tree[base_1 + 352] = a2[11];
                tree[base_1 + 384] = a2[12];
                tree[base_1 + 416] = a2[13];
                tree[base_1 + 448] = a2[14];
                tree[base_1 + 480] = a2[15];
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (low2 == 0) {
                float sb2[16];
                int rbase2 = (whi + bit2) * 544 + c;
                sb2[0] = tree[rbase2];
                sb2[1] = tree[rbase2 + 32];
                sb2[2] = tree[rbase2 + 64];
                sb2[3] = tree[rbase2 + 96];
                sb2[4] = tree[rbase2 + 128];
                sb2[5] = tree[rbase2 + 160];
                sb2[6] = tree[rbase2 + 192];
                sb2[7] = tree[rbase2 + 224];
                sb2[8] = tree[rbase2 + 256];
                sb2[9] = tree[rbase2 + 288];
                sb2[10] = tree[rbase2 + 320];
                sb2[11] = tree[rbase2 + 352];
                sb2[12] = tree[rbase2 + 384];
                sb2[13] = tree[rbase2 + 416];
                sb2[14] = tree[rbase2 + 448];
                sb2[15] = tree[rbase2 + 480];
                float _fmax_329 = fmaxf(a2[0], sb2[15]);
                float hi_5 = _fmax_329;
                a2[0] = hi_5;
                float _fmax_330 = fmaxf(a2[1], sb2[14]);
                float hi_0_2 = _fmax_330;
                a2[1] = hi_0_2;
                float _fmax_331 = fmaxf(a2[2], sb2[13]);
                float hi_1_1 = _fmax_331;
                a2[2] = hi_1_1;
                float _fmax_332 = fmaxf(a2[3], sb2[12]);
                float hi_2_2 = _fmax_332;
                a2[3] = hi_2_2;
                float _fmax_333 = fmaxf(a2[4], sb2[11]);
                float hi_3_1 = _fmax_333;
                a2[4] = hi_3_1;
                float _fmax_334 = fmaxf(a2[5], sb2[10]);
                float hi_4_2 = _fmax_334;
                a2[5] = hi_4_2;
                float _fmax_335 = fmaxf(a2[6], sb2[9]);
                float hi_5_1 = _fmax_335;
                a2[6] = hi_5_1;
                float _fmax_336 = fmaxf(a2[7], sb2[8]);
                float hi_6_2 = _fmax_336;
                a2[7] = hi_6_2;
                float _fmax_337 = fmaxf(a2[8], sb2[7]);
                float hi_7 = _fmax_337;
                a2[8] = hi_7;
                float _fmax_338 = fmaxf(a2[9], sb2[6]);
                float hi_8_2 = _fmax_338;
                a2[9] = hi_8_2;
                float _fmax_339 = fmaxf(a2[10], sb2[5]);
                float hi_9 = _fmax_339;
                a2[10] = hi_9;
                float _fmax_340 = fmaxf(a2[11], sb2[4]);
                float hi_10_2 = _fmax_340;
                a2[11] = hi_10_2;
                float _fmax_341 = fmaxf(a2[12], sb2[3]);
                float hi_11 = _fmax_341;
                a2[12] = hi_11;
                float _fmax_342 = fmaxf(a2[13], sb2[2]);
                float hi_12_2 = _fmax_342;
                a2[13] = hi_12_2;
                float _fmax_343 = fmaxf(a2[14], sb2[1]);
                float hi_13 = _fmax_343;
                a2[14] = hi_13;
                float _fmax_344 = fmaxf(a2[15], sb2[0]);
                float hi_14_2 = _fmax_344;
                a2[15] = hi_14_2;
                float _fmax_345 = fmaxf(a2[0], a2[8]);
                float hi_15 = _fmax_345;
                float _min_280 = fminf(a2[0], a2[8]);
                float lo_4 = _min_280;
                a2[0] = hi_15;
                a2[8] = lo_4;
                float _fmax_346 = fmaxf(a2[1], a2[9]);
                float hi_16_2 = _fmax_346;
                float _min_281 = fminf(a2[1], a2[9]);
                float lo_17_2 = _min_281;
                a2[1] = hi_16_2;
                a2[9] = lo_17_2;
                float _fmax_347 = fmaxf(a2[2], a2[10]);
                float hi_18_2 = _fmax_347;
                float _min_282 = fminf(a2[2], a2[10]);
                float lo_19_2 = _min_282;
                a2[2] = hi_18_2;
                a2[10] = lo_19_2;
                float _fmax_348 = fmaxf(a2[3], a2[11]);
                float hi_20_2 = _fmax_348;
                float _min_283 = fminf(a2[3], a2[11]);
                float lo_21_2 = _min_283;
                a2[3] = hi_20_2;
                a2[11] = lo_21_2;
                float _fmax_349 = fmaxf(a2[4], a2[12]);
                float hi_22_2 = _fmax_349;
                float _min_284 = fminf(a2[4], a2[12]);
                float lo_23_2 = _min_284;
                a2[4] = hi_22_2;
                a2[12] = lo_23_2;
                float _fmax_350 = fmaxf(a2[5], a2[13]);
                float hi_24_2 = _fmax_350;
                float _min_285 = fminf(a2[5], a2[13]);
                float lo_25_2 = _min_285;
                a2[5] = hi_24_2;
                a2[13] = lo_25_2;
                float _fmax_351 = fmaxf(a2[6], a2[14]);
                float hi_26_2 = _fmax_351;
                float _min_286 = fminf(a2[6], a2[14]);
                float lo_27_2 = _min_286;
                a2[6] = hi_26_2;
                a2[14] = lo_27_2;
                float _fmax_352 = fmaxf(a2[7], a2[15]);
                float hi_28_2 = _fmax_352;
                float _min_287 = fminf(a2[7], a2[15]);
                float lo_29_2 = _min_287;
                a2[7] = hi_28_2;
                a2[15] = lo_29_2;
                float _fmax_353 = fmaxf(a2[0], a2[4]);
                float hi_30_2 = _fmax_353;
                float _min_288 = fminf(a2[0], a2[4]);
                float lo_31_2 = _min_288;
                a2[0] = hi_30_2;
                a2[4] = lo_31_2;
                float _fmax_354 = fmaxf(a2[1], a2[5]);
                float hi_32_2 = _fmax_354;
                float _min_289 = fminf(a2[1], a2[5]);
                float lo_33_2 = _min_289;
                a2[1] = hi_32_2;
                a2[5] = lo_33_2;
                float _fmax_355 = fmaxf(a2[2], a2[6]);
                float hi_34_2 = _fmax_355;
                float _min_290 = fminf(a2[2], a2[6]);
                float lo_35_2 = _min_290;
                a2[2] = hi_34_2;
                a2[6] = lo_35_2;
                float _fmax_356 = fmaxf(a2[3], a2[7]);
                float hi_36_2 = _fmax_356;
                float _min_291 = fminf(a2[3], a2[7]);
                float lo_37_2 = _min_291;
                a2[3] = hi_36_2;
                a2[7] = lo_37_2;
                float _fmax_357 = fmaxf(a2[8], a2[12]);
                float hi_38_2 = _fmax_357;
                float _min_292 = fminf(a2[8], a2[12]);
                float lo_39_2 = _min_292;
                a2[8] = hi_38_2;
                a2[12] = lo_39_2;
                float _fmax_358 = fmaxf(a2[9], a2[13]);
                float hi_40_2 = _fmax_358;
                float _min_293 = fminf(a2[9], a2[13]);
                float lo_41_2 = _min_293;
                a2[9] = hi_40_2;
                a2[13] = lo_41_2;
                float _fmax_359 = fmaxf(a2[10], a2[14]);
                float hi_42_2 = _fmax_359;
                float _min_294 = fminf(a2[10], a2[14]);
                float lo_43_2 = _min_294;
                a2[10] = hi_42_2;
                a2[14] = lo_43_2;
                float _fmax_360 = fmaxf(a2[11], a2[15]);
                float hi_44_2 = _fmax_360;
                float _min_295 = fminf(a2[11], a2[15]);
                float lo_45_2 = _min_295;
                a2[11] = hi_44_2;
                a2[15] = lo_45_2;
                float _fmax_361 = fmaxf(a2[0], a2[2]);
                float hi_46_2 = _fmax_361;
                float _min_296 = fminf(a2[0], a2[2]);
                float lo_47_2 = _min_296;
                a2[0] = hi_46_2;
                a2[2] = lo_47_2;
                float _fmax_362 = fmaxf(a2[1], a2[3]);
                float hi_48_2 = _fmax_362;
                float _min_297 = fminf(a2[1], a2[3]);
                float lo_49_2 = _min_297;
                a2[1] = hi_48_2;
                a2[3] = lo_49_2;
                float _fmax_363 = fmaxf(a2[4], a2[6]);
                float hi_50_2 = _fmax_363;
                float _min_298 = fminf(a2[4], a2[6]);
                float lo_51_2 = _min_298;
                a2[4] = hi_50_2;
                a2[6] = lo_51_2;
                float _fmax_364 = fmaxf(a2[5], a2[7]);
                float hi_52_2 = _fmax_364;
                float _min_299 = fminf(a2[5], a2[7]);
                float lo_53_2 = _min_299;
                a2[5] = hi_52_2;
                a2[7] = lo_53_2;
                float _fmax_365 = fmaxf(a2[8], a2[10]);
                float hi_54_2 = _fmax_365;
                float _min_300 = fminf(a2[8], a2[10]);
                float lo_55_2 = _min_300;
                a2[8] = hi_54_2;
                a2[10] = lo_55_2;
                float _fmax_366 = fmaxf(a2[9], a2[11]);
                float hi_56_2 = _fmax_366;
                float _min_301 = fminf(a2[9], a2[11]);
                float lo_57_2 = _min_301;
                a2[9] = hi_56_2;
                a2[11] = lo_57_2;
                float _fmax_367 = fmaxf(a2[12], a2[14]);
                float hi_58_2 = _fmax_367;
                float _min_302 = fminf(a2[12], a2[14]);
                float lo_59_2 = _min_302;
                a2[12] = hi_58_2;
                a2[14] = lo_59_2;
                float _fmax_368 = fmaxf(a2[13], a2[15]);
                float hi_60_2 = _fmax_368;
                float _min_303 = fminf(a2[13], a2[15]);
                float lo_61_2 = _min_303;
                a2[13] = hi_60_2;
                a2[15] = lo_61_2;
                float _fmax_369 = fmaxf(a2[0], a2[1]);
                float hi_62_2 = _fmax_369;
                float _min_304 = fminf(a2[0], a2[1]);
                float lo_63_2 = _min_304;
                a2[0] = hi_62_2;
                a2[1] = lo_63_2;
                float _fmax_370 = fmaxf(a2[2], a2[3]);
                float hi_64_2 = _fmax_370;
                float _min_305 = fminf(a2[2], a2[3]);
                float lo_65_2 = _min_305;
                a2[2] = hi_64_2;
                a2[3] = lo_65_2;
                float _fmax_371 = fmaxf(a2[4], a2[5]);
                float hi_66_2 = _fmax_371;
                float _min_306 = fminf(a2[4], a2[5]);
                float lo_67_2 = _min_306;
                a2[4] = hi_66_2;
                a2[5] = lo_67_2;
                float _fmax_372 = fmaxf(a2[6], a2[7]);
                float hi_68_2 = _fmax_372;
                float _min_307 = fminf(a2[6], a2[7]);
                float lo_69_2 = _min_307;
                a2[6] = hi_68_2;
                a2[7] = lo_69_2;
                float _fmax_373 = fmaxf(a2[8], a2[9]);
                float hi_70_2 = _fmax_373;
                float _min_308 = fminf(a2[8], a2[9]);
                float lo_71_2 = _min_308;
                a2[8] = hi_70_2;
                a2[9] = lo_71_2;
                float _fmax_374 = fmaxf(a2[10], a2[11]);
                float hi_72_2 = _fmax_374;
                float _min_309 = fminf(a2[10], a2[11]);
                float lo_73_2 = _min_309;
                a2[10] = hi_72_2;
                a2[11] = lo_73_2;
                float _fmax_375 = fmaxf(a2[12], a2[13]);
                float hi_74_2 = _fmax_375;
                float _min_310 = fminf(a2[12], a2[13]);
                float lo_75_2 = _min_310;
                a2[12] = hi_74_2;
                a2[13] = lo_75_2;
                float _fmax_376 = fmaxf(a2[14], a2[15]);
                float hi_76_2 = _fmax_376;
                float _min_311 = fminf(a2[14], a2[15]);
                float lo_77_2 = _min_311;
                a2[14] = hi_76_2;
                a2[15] = lo_77_2;
            }
        }
    }
    if (w == 0 && col < total_q) {
        unsigned int o[16];
        unsigned int k1 = __as_u32(a[0]);
        unsigned int idx = 4294967295;
        if (k1 < 4278190080u) {
            idx = k1 & 63;
        }
        if (flagged != 0) {
            unsigned int k2 = __as_u32(a2[0]);
            idx = 4294967295;
            if (k2 != 0) {
                idx = k2 & 63;
            }
        }
        o[0] = idx;
        unsigned int k1_0 = __as_u32(a[1]);
        unsigned int idx_1 = 4294967295;
        if (k1_0 < 4278190080u) {
            idx_1 = k1_0 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_1 = __as_u32(a2[1]);
            idx_1 = 4294967295;
            if (k2_1 != 0) {
                idx_1 = k2_1 & 63;
            }
        }
        o[1] = idx_1;
        unsigned int k1_2 = __as_u32(a[2]);
        unsigned int idx_3 = 4294967295;
        if (k1_2 < 4278190080u) {
            idx_3 = k1_2 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_2 = __as_u32(a2[2]);
            idx_3 = 4294967295;
            if (k2_2 != 0) {
                idx_3 = k2_2 & 63;
            }
        }
        o[2] = idx_3;
        unsigned int k1_4 = __as_u32(a[3]);
        unsigned int idx_5 = 4294967295;
        if (k1_4 < 4278190080u) {
            idx_5 = k1_4 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_3 = __as_u32(a2[3]);
            idx_5 = 4294967295;
            if (k2_3 != 0) {
                idx_5 = k2_3 & 63;
            }
        }
        o[3] = idx_5;
        unsigned int k1_6 = __as_u32(a[4]);
        unsigned int idx_7 = 4294967295;
        if (k1_6 < 4278190080u) {
            idx_7 = k1_6 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_4 = __as_u32(a2[4]);
            idx_7 = 4294967295;
            if (k2_4 != 0) {
                idx_7 = k2_4 & 63;
            }
        }
        o[4] = idx_7;
        unsigned int k1_8 = __as_u32(a[5]);
        unsigned int idx_9 = 4294967295;
        if (k1_8 < 4278190080u) {
            idx_9 = k1_8 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_5 = __as_u32(a2[5]);
            idx_9 = 4294967295;
            if (k2_5 != 0) {
                idx_9 = k2_5 & 63;
            }
        }
        o[5] = idx_9;
        unsigned int k1_10 = __as_u32(a[6]);
        unsigned int idx_11 = 4294967295;
        if (k1_10 < 4278190080u) {
            idx_11 = k1_10 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_6 = __as_u32(a2[6]);
            idx_11 = 4294967295;
            if (k2_6 != 0) {
                idx_11 = k2_6 & 63;
            }
        }
        o[6] = idx_11;
        unsigned int k1_12 = __as_u32(a[7]);
        unsigned int idx_13 = 4294967295;
        if (k1_12 < 4278190080u) {
            idx_13 = k1_12 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_7 = __as_u32(a2[7]);
            idx_13 = 4294967295;
            if (k2_7 != 0) {
                idx_13 = k2_7 & 63;
            }
        }
        o[7] = idx_13;
        unsigned int k1_14 = __as_u32(a[8]);
        unsigned int idx_15 = 4294967295;
        if (k1_14 < 4278190080u) {
            idx_15 = k1_14 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_8 = __as_u32(a2[8]);
            idx_15 = 4294967295;
            if (k2_8 != 0) {
                idx_15 = k2_8 & 63;
            }
        }
        o[8] = idx_15;
        unsigned int k1_16 = __as_u32(a[9]);
        unsigned int idx_17 = 4294967295;
        if (k1_16 < 4278190080u) {
            idx_17 = k1_16 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_9 = __as_u32(a2[9]);
            idx_17 = 4294967295;
            if (k2_9 != 0) {
                idx_17 = k2_9 & 63;
            }
        }
        o[9] = idx_17;
        unsigned int k1_18 = __as_u32(a[10]);
        unsigned int idx_19 = 4294967295;
        if (k1_18 < 4278190080u) {
            idx_19 = k1_18 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_10 = __as_u32(a2[10]);
            idx_19 = 4294967295;
            if (k2_10 != 0) {
                idx_19 = k2_10 & 63;
            }
        }
        o[10] = idx_19;
        unsigned int k1_20 = __as_u32(a[11]);
        unsigned int idx_21 = 4294967295;
        if (k1_20 < 4278190080u) {
            idx_21 = k1_20 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_11 = __as_u32(a2[11]);
            idx_21 = 4294967295;
            if (k2_11 != 0) {
                idx_21 = k2_11 & 63;
            }
        }
        o[11] = idx_21;
        unsigned int k1_22 = __as_u32(a[12]);
        unsigned int idx_23 = 4294967295;
        if (k1_22 < 4278190080u) {
            idx_23 = k1_22 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_12 = __as_u32(a2[12]);
            idx_23 = 4294967295;
            if (k2_12 != 0) {
                idx_23 = k2_12 & 63;
            }
        }
        o[12] = idx_23;
        unsigned int k1_24 = __as_u32(a[13]);
        unsigned int idx_25 = 4294967295;
        if (k1_24 < 4278190080u) {
            idx_25 = k1_24 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_13 = __as_u32(a2[13]);
            idx_25 = 4294967295;
            if (k2_13 != 0) {
                idx_25 = k2_13 & 63;
            }
        }
        o[13] = idx_25;
        unsigned int k1_26 = __as_u32(a[14]);
        unsigned int idx_27 = 4294967295;
        if (k1_26 < 4278190080u) {
            idx_27 = k1_26 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_14 = __as_u32(a2[14]);
            idx_27 = 4294967295;
            if (k2_14 != 0) {
                idx_27 = k2_14 & 63;
            }
        }
        o[14] = idx_27;
        unsigned int k1_28 = __as_u32(a[15]);
        unsigned int idx_29 = 4294967295;
        if (k1_28 < 4278190080u) {
            idx_29 = k1_28 & 63;
        }
        if (flagged != 0) {
            unsigned int k2_15 = __as_u32(a2[15]);
            idx_29 = 4294967295;
            if (k2_15 != 0) {
                idx_29 = k2_15 & 63;
            }
        }
        o[15] = idx_29;
        unsigned int _max_0 = ((o[0]) > (o[13]) ? (o[0]) : (o[13]));
        unsigned int hi_17 = _max_0;
        unsigned int _min_312 = ((o[0]) < (o[13]) ? (o[0]) : (o[13]));
        unsigned int lo_6 = _min_312;
        o[0] = lo_6;
        o[13] = hi_17;
        unsigned int _max_1 = ((o[1]) > (o[12]) ? (o[1]) : (o[12]));
        unsigned int hi_30_3 = _max_1;
        unsigned int _min_313 = ((o[1]) < (o[12]) ? (o[1]) : (o[12]));
        unsigned int lo_31_3 = _min_313;
        o[1] = lo_31_3;
        o[12] = hi_30_3;
        unsigned int _max_2 = ((o[2]) > (o[15]) ? (o[2]) : (o[15]));
        unsigned int hi_32_3 = _max_2;
        unsigned int _min_314 = ((o[2]) < (o[15]) ? (o[2]) : (o[15]));
        unsigned int lo_33_3 = _min_314;
        o[2] = lo_33_3;
        o[15] = hi_32_3;
        unsigned int _max_3 = ((o[3]) > (o[14]) ? (o[3]) : (o[14]));
        unsigned int hi_34_3 = _max_3;
        unsigned int _min_315 = ((o[3]) < (o[14]) ? (o[3]) : (o[14]));
        unsigned int lo_35_3 = _min_315;
        o[3] = lo_35_3;
        o[14] = hi_34_3;
        unsigned int _max_4 = ((o[4]) > (o[8]) ? (o[4]) : (o[8]));
        unsigned int hi_36_3 = _max_4;
        unsigned int _min_316 = ((o[4]) < (o[8]) ? (o[4]) : (o[8]));
        unsigned int lo_37_3 = _min_316;
        o[4] = lo_37_3;
        o[8] = hi_36_3;
        unsigned int _max_5 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int hi_38_3 = _max_5;
        unsigned int _min_317 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_39_3 = _min_317;
        o[5] = lo_39_3;
        o[6] = hi_38_3;
        unsigned int _max_6 = ((o[7]) > (o[11]) ? (o[7]) : (o[11]));
        unsigned int hi_40_3 = _max_6;
        unsigned int _min_318 = ((o[7]) < (o[11]) ? (o[7]) : (o[11]));
        unsigned int lo_41_3 = _min_318;
        o[7] = lo_41_3;
        o[11] = hi_40_3;
        unsigned int _max_7 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int hi_42_3 = _max_7;
        unsigned int _min_319 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_43_3 = _min_319;
        o[9] = lo_43_3;
        o[10] = hi_42_3;
        unsigned int _max_8 = ((o[0]) > (o[5]) ? (o[0]) : (o[5]));
        unsigned int hi_44_3 = _max_8;
        unsigned int _min_320 = ((o[0]) < (o[5]) ? (o[0]) : (o[5]));
        unsigned int lo_45_3 = _min_320;
        o[0] = lo_45_3;
        o[5] = hi_44_3;
        unsigned int _max_9 = ((o[1]) > (o[7]) ? (o[1]) : (o[7]));
        unsigned int hi_46_3 = _max_9;
        unsigned int _min_321 = ((o[1]) < (o[7]) ? (o[1]) : (o[7]));
        unsigned int lo_47_3 = _min_321;
        o[1] = lo_47_3;
        o[7] = hi_46_3;
        unsigned int _max_10 = ((o[2]) > (o[9]) ? (o[2]) : (o[9]));
        unsigned int hi_48_3 = _max_10;
        unsigned int _min_322 = ((o[2]) < (o[9]) ? (o[2]) : (o[9]));
        unsigned int lo_49_3 = _min_322;
        o[2] = lo_49_3;
        o[9] = hi_48_3;
        unsigned int _max_11 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int hi_50_3 = _max_11;
        unsigned int _min_323 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_51_3 = _min_323;
        o[3] = lo_51_3;
        o[4] = hi_50_3;
        unsigned int _max_12 = ((o[6]) > (o[13]) ? (o[6]) : (o[13]));
        unsigned int hi_52_3 = _max_12;
        unsigned int _min_324 = ((o[6]) < (o[13]) ? (o[6]) : (o[13]));
        unsigned int lo_53_3 = _min_324;
        o[6] = lo_53_3;
        o[13] = hi_52_3;
        unsigned int _max_13 = ((o[8]) > (o[14]) ? (o[8]) : (o[14]));
        unsigned int hi_54_3 = _max_13;
        unsigned int _min_325 = ((o[8]) < (o[14]) ? (o[8]) : (o[14]));
        unsigned int lo_55_3 = _min_325;
        o[8] = lo_55_3;
        o[14] = hi_54_3;
        unsigned int _max_14 = ((o[10]) > (o[15]) ? (o[10]) : (o[15]));
        unsigned int hi_56_3 = _max_14;
        unsigned int _min_326 = ((o[10]) < (o[15]) ? (o[10]) : (o[15]));
        unsigned int lo_57_3 = _min_326;
        o[10] = lo_57_3;
        o[15] = hi_56_3;
        unsigned int _max_15 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int hi_58_3 = _max_15;
        unsigned int _min_327 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_59_3 = _min_327;
        o[11] = lo_59_3;
        o[12] = hi_58_3;
        unsigned int _max_16 = ((o[0]) > (o[1]) ? (o[0]) : (o[1]));
        unsigned int hi_60_3 = _max_16;
        unsigned int _min_328 = ((o[0]) < (o[1]) ? (o[0]) : (o[1]));
        unsigned int lo_61_3 = _min_328;
        o[0] = lo_61_3;
        o[1] = hi_60_3;
        unsigned int _max_17 = ((o[2]) > (o[3]) ? (o[2]) : (o[3]));
        unsigned int hi_62_3 = _max_17;
        unsigned int _min_329 = ((o[2]) < (o[3]) ? (o[2]) : (o[3]));
        unsigned int lo_63_3 = _min_329;
        o[2] = lo_63_3;
        o[3] = hi_62_3;
        unsigned int _max_18 = ((o[4]) > (o[5]) ? (o[4]) : (o[5]));
        unsigned int hi_64_3 = _max_18;
        unsigned int _min_330 = ((o[4]) < (o[5]) ? (o[4]) : (o[5]));
        unsigned int lo_65_3 = _min_330;
        o[4] = lo_65_3;
        o[5] = hi_64_3;
        unsigned int _max_19 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int hi_66_3 = _max_19;
        unsigned int _min_331 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_67_3 = _min_331;
        o[6] = lo_67_3;
        o[8] = hi_66_3;
        unsigned int _max_20 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int hi_68_3 = _max_20;
        unsigned int _min_332 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_69_3 = _min_332;
        o[7] = lo_69_3;
        o[9] = hi_68_3;
        unsigned int _max_21 = ((o[10]) > (o[11]) ? (o[10]) : (o[11]));
        unsigned int hi_70_3 = _max_21;
        unsigned int _min_333 = ((o[10]) < (o[11]) ? (o[10]) : (o[11]));
        unsigned int lo_71_3 = _min_333;
        o[10] = lo_71_3;
        o[11] = hi_70_3;
        unsigned int _max_22 = ((o[12]) > (o[13]) ? (o[12]) : (o[13]));
        unsigned int hi_72_3 = _max_22;
        unsigned int _min_334 = ((o[12]) < (o[13]) ? (o[12]) : (o[13]));
        unsigned int lo_73_3 = _min_334;
        o[12] = lo_73_3;
        o[13] = hi_72_3;
        unsigned int _max_23 = ((o[14]) > (o[15]) ? (o[14]) : (o[15]));
        unsigned int hi_74_3 = _max_23;
        unsigned int _min_335 = ((o[14]) < (o[15]) ? (o[14]) : (o[15]));
        unsigned int lo_75_3 = _min_335;
        o[14] = lo_75_3;
        o[15] = hi_74_3;
        unsigned int _max_24 = ((o[0]) > (o[2]) ? (o[0]) : (o[2]));
        unsigned int hi_76_3 = _max_24;
        unsigned int _min_336 = ((o[0]) < (o[2]) ? (o[0]) : (o[2]));
        unsigned int lo_77_3 = _min_336;
        o[0] = lo_77_3;
        o[2] = hi_76_3;
        unsigned int _max_25 = ((o[1]) > (o[3]) ? (o[1]) : (o[3]));
        unsigned int hi_78_2 = _max_25;
        unsigned int _min_337 = ((o[1]) < (o[3]) ? (o[1]) : (o[3]));
        unsigned int lo_79_2 = _min_337;
        o[1] = lo_79_2;
        o[3] = hi_78_2;
        unsigned int _max_26 = ((o[4]) > (o[10]) ? (o[4]) : (o[10]));
        unsigned int hi_80_2 = _max_26;
        unsigned int _min_338 = ((o[4]) < (o[10]) ? (o[4]) : (o[10]));
        unsigned int lo_81_2 = _min_338;
        o[4] = lo_81_2;
        o[10] = hi_80_2;
        unsigned int _max_27 = ((o[5]) > (o[11]) ? (o[5]) : (o[11]));
        unsigned int hi_82_2 = _max_27;
        unsigned int _min_339 = ((o[5]) < (o[11]) ? (o[5]) : (o[11]));
        unsigned int lo_83_2 = _min_339;
        o[5] = lo_83_2;
        o[11] = hi_82_2;
        unsigned int _max_28 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int hi_84_2 = _max_28;
        unsigned int _min_340 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_85_2 = _min_340;
        o[6] = lo_85_2;
        o[7] = hi_84_2;
        unsigned int _max_29 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int hi_86_2 = _max_29;
        unsigned int _min_341 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_87_2 = _min_341;
        o[8] = lo_87_2;
        o[9] = hi_86_2;
        unsigned int _max_30 = ((o[12]) > (o[14]) ? (o[12]) : (o[14]));
        unsigned int hi_88_2 = _max_30;
        unsigned int _min_342 = ((o[12]) < (o[14]) ? (o[12]) : (o[14]));
        unsigned int lo_89_2 = _min_342;
        o[12] = lo_89_2;
        o[14] = hi_88_2;
        unsigned int _max_31 = ((o[13]) > (o[15]) ? (o[13]) : (o[15]));
        unsigned int hi_90_2 = _max_31;
        unsigned int _min_343 = ((o[13]) < (o[15]) ? (o[13]) : (o[15]));
        unsigned int lo_91_2 = _min_343;
        o[13] = lo_91_2;
        o[15] = hi_90_2;
        unsigned int _max_32 = ((o[1]) > (o[2]) ? (o[1]) : (o[2]));
        unsigned int hi_92_2 = _max_32;
        unsigned int _min_344 = ((o[1]) < (o[2]) ? (o[1]) : (o[2]));
        unsigned int lo_93_2 = _min_344;
        o[1] = lo_93_2;
        o[2] = hi_92_2;
        unsigned int _max_33 = ((o[3]) > (o[12]) ? (o[3]) : (o[12]));
        unsigned int hi_94_1 = _max_33;
        unsigned int _min_345 = ((o[3]) < (o[12]) ? (o[3]) : (o[12]));
        unsigned int lo_95_1 = _min_345;
        o[3] = lo_95_1;
        o[12] = hi_94_1;
        unsigned int _max_34 = ((o[4]) > (o[6]) ? (o[4]) : (o[6]));
        unsigned int hi_96_1 = _max_34;
        unsigned int _min_346 = ((o[4]) < (o[6]) ? (o[4]) : (o[6]));
        unsigned int lo_97_1 = _min_346;
        o[4] = lo_97_1;
        o[6] = hi_96_1;
        unsigned int _max_35 = ((o[5]) > (o[7]) ? (o[5]) : (o[7]));
        unsigned int hi_98_1 = _max_35;
        unsigned int _min_347 = ((o[5]) < (o[7]) ? (o[5]) : (o[7]));
        unsigned int lo_99_1 = _min_347;
        o[5] = lo_99_1;
        o[7] = hi_98_1;
        unsigned int _max_36 = ((o[8]) > (o[10]) ? (o[8]) : (o[10]));
        unsigned int hi_100_1 = _max_36;
        unsigned int _min_348 = ((o[8]) < (o[10]) ? (o[8]) : (o[10]));
        unsigned int lo_101_1 = _min_348;
        o[8] = lo_101_1;
        o[10] = hi_100_1;
        unsigned int _max_37 = ((o[9]) > (o[11]) ? (o[9]) : (o[11]));
        unsigned int hi_102_1 = _max_37;
        unsigned int _min_349 = ((o[9]) < (o[11]) ? (o[9]) : (o[11]));
        unsigned int lo_103_1 = _min_349;
        o[9] = lo_103_1;
        o[11] = hi_102_1;
        unsigned int _max_38 = ((o[13]) > (o[14]) ? (o[13]) : (o[14]));
        unsigned int hi_104_1 = _max_38;
        unsigned int _min_350 = ((o[13]) < (o[14]) ? (o[13]) : (o[14]));
        unsigned int lo_105_1 = _min_350;
        o[13] = lo_105_1;
        o[14] = hi_104_1;
        unsigned int _max_39 = ((o[1]) > (o[4]) ? (o[1]) : (o[4]));
        unsigned int hi_106_1 = _max_39;
        unsigned int _min_351 = ((o[1]) < (o[4]) ? (o[1]) : (o[4]));
        unsigned int lo_107_1 = _min_351;
        o[1] = lo_107_1;
        o[4] = hi_106_1;
        unsigned int _max_40 = ((o[2]) > (o[6]) ? (o[2]) : (o[6]));
        unsigned int hi_108_1 = _max_40;
        unsigned int _min_352 = ((o[2]) < (o[6]) ? (o[2]) : (o[6]));
        unsigned int lo_109_1 = _min_352;
        o[2] = lo_109_1;
        o[6] = hi_108_1;
        unsigned int _max_41 = ((o[5]) > (o[8]) ? (o[5]) : (o[8]));
        unsigned int hi_110_1 = _max_41;
        unsigned int _min_353 = ((o[5]) < (o[8]) ? (o[5]) : (o[8]));
        unsigned int lo_111_1 = _min_353;
        o[5] = lo_111_1;
        o[8] = hi_110_1;
        unsigned int _max_42 = ((o[7]) > (o[10]) ? (o[7]) : (o[10]));
        unsigned int hi_112_1 = _max_42;
        unsigned int _min_354 = ((o[7]) < (o[10]) ? (o[7]) : (o[10]));
        unsigned int lo_113_1 = _min_354;
        o[7] = lo_113_1;
        o[10] = hi_112_1;
        unsigned int _max_43 = ((o[9]) > (o[13]) ? (o[9]) : (o[13]));
        unsigned int hi_114_1 = _max_43;
        unsigned int _min_355 = ((o[9]) < (o[13]) ? (o[9]) : (o[13]));
        unsigned int lo_115_1 = _min_355;
        o[9] = lo_115_1;
        o[13] = hi_114_1;
        unsigned int _max_44 = ((o[11]) > (o[14]) ? (o[11]) : (o[14]));
        unsigned int hi_116_1 = _max_44;
        unsigned int _min_356 = ((o[11]) < (o[14]) ? (o[11]) : (o[14]));
        unsigned int lo_117_1 = _min_356;
        o[11] = lo_117_1;
        o[14] = hi_116_1;
        unsigned int _max_45 = ((o[2]) > (o[4]) ? (o[2]) : (o[4]));
        unsigned int hi_118_1 = _max_45;
        unsigned int _min_357 = ((o[2]) < (o[4]) ? (o[2]) : (o[4]));
        unsigned int lo_119 = _min_357;
        o[2] = lo_119;
        o[4] = hi_118_1;
        unsigned int _max_46 = ((o[3]) > (o[6]) ? (o[3]) : (o[6]));
        unsigned int hi_120_1 = _max_46;
        unsigned int _min_358 = ((o[3]) < (o[6]) ? (o[3]) : (o[6]));
        unsigned int lo_121 = _min_358;
        o[3] = lo_121;
        o[6] = hi_120_1;
        unsigned int _max_47 = ((o[9]) > (o[12]) ? (o[9]) : (o[12]));
        unsigned int hi_122_1 = _max_47;
        unsigned int _min_359 = ((o[9]) < (o[12]) ? (o[9]) : (o[12]));
        unsigned int lo_123 = _min_359;
        o[9] = lo_123;
        o[12] = hi_122_1;
        unsigned int _max_48 = ((o[11]) > (o[13]) ? (o[11]) : (o[13]));
        unsigned int hi_124_1 = _max_48;
        unsigned int _min_360 = ((o[11]) < (o[13]) ? (o[11]) : (o[13]));
        unsigned int lo_125 = _min_360;
        o[11] = lo_125;
        o[13] = hi_124_1;
        unsigned int _max_49 = ((o[3]) > (o[5]) ? (o[3]) : (o[5]));
        unsigned int hi_126_1 = _max_49;
        unsigned int _min_361 = ((o[3]) < (o[5]) ? (o[3]) : (o[5]));
        unsigned int lo_127 = _min_361;
        o[3] = lo_127;
        o[5] = hi_126_1;
        unsigned int _max_50 = ((o[6]) > (o[8]) ? (o[6]) : (o[8]));
        unsigned int hi_128_1 = _max_50;
        unsigned int _min_362 = ((o[6]) < (o[8]) ? (o[6]) : (o[8]));
        unsigned int lo_129 = _min_362;
        o[6] = lo_129;
        o[8] = hi_128_1;
        unsigned int _max_51 = ((o[7]) > (o[9]) ? (o[7]) : (o[9]));
        unsigned int hi_130_1 = _max_51;
        unsigned int _min_363 = ((o[7]) < (o[9]) ? (o[7]) : (o[9]));
        unsigned int lo_131 = _min_363;
        o[7] = lo_131;
        o[9] = hi_130_1;
        unsigned int _max_52 = ((o[10]) > (o[12]) ? (o[10]) : (o[12]));
        unsigned int hi_132_1 = _max_52;
        unsigned int _min_364 = ((o[10]) < (o[12]) ? (o[10]) : (o[12]));
        unsigned int lo_133 = _min_364;
        o[10] = lo_133;
        o[12] = hi_132_1;
        unsigned int _max_53 = ((o[3]) > (o[4]) ? (o[3]) : (o[4]));
        unsigned int hi_134_1 = _max_53;
        unsigned int _min_365 = ((o[3]) < (o[4]) ? (o[3]) : (o[4]));
        unsigned int lo_135_1 = _min_365;
        o[3] = lo_135_1;
        o[4] = hi_134_1;
        unsigned int _max_54 = ((o[5]) > (o[6]) ? (o[5]) : (o[6]));
        unsigned int hi_136_1 = _max_54;
        unsigned int _min_366 = ((o[5]) < (o[6]) ? (o[5]) : (o[6]));
        unsigned int lo_137_1 = _min_366;
        o[5] = lo_137_1;
        o[6] = hi_136_1;
        unsigned int _max_55 = ((o[7]) > (o[8]) ? (o[7]) : (o[8]));
        unsigned int hi_138_1 = _max_55;
        unsigned int _min_367 = ((o[7]) < (o[8]) ? (o[7]) : (o[8]));
        unsigned int lo_139_1 = _min_367;
        o[7] = lo_139_1;
        o[8] = hi_138_1;
        unsigned int _max_56 = ((o[9]) > (o[10]) ? (o[9]) : (o[10]));
        unsigned int hi_140_1 = _max_56;
        unsigned int _min_368 = ((o[9]) < (o[10]) ? (o[9]) : (o[10]));
        unsigned int lo_141_1 = _min_368;
        o[9] = lo_141_1;
        o[10] = hi_140_1;
        unsigned int _max_57 = ((o[11]) > (o[12]) ? (o[11]) : (o[12]));
        unsigned int hi_142_1 = _max_57;
        unsigned int _min_369 = ((o[11]) < (o[12]) ? (o[11]) : (o[12]));
        unsigned int lo_143_1 = _min_369;
        o[11] = lo_143_1;
        o[12] = hi_142_1;
        unsigned int _max_58 = ((o[6]) > (o[7]) ? (o[6]) : (o[7]));
        unsigned int hi_144_1 = _max_58;
        unsigned int _min_370 = ((o[6]) < (o[7]) ? (o[6]) : (o[7]));
        unsigned int lo_145_1 = _min_370;
        o[6] = lo_145_1;
        o[7] = hi_144_1;
        unsigned int _max_59 = ((o[8]) > (o[9]) ? (o[8]) : (o[9]));
        unsigned int hi_146_1 = _max_59;
        unsigned int _min_371 = ((o[8]) < (o[9]) ? (o[8]) : (o[9]));
        unsigned int lo_147_1 = _min_371;
        o[8] = lo_147_1;
        o[9] = hi_146_1;
        long long obase = ((long long)col * (long long)num_heads + (long long)head) * 16;
        {
            int4 _iv4 = make_int4(o[0 + 0], o[0 + 1], o[0 + 2], o[0 + 3]);
            *reinterpret_cast<int4*>(out + obase) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[4 + 0], o[4 + 1], o[4 + 2], o[4 + 3]);
            *reinterpret_cast<int4*>(out + obase + 4) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[8 + 0], o[8 + 1], o[8 + 2], o[8 + 3]);
            *reinterpret_cast<int4*>(out + obase + 8) = _iv4;
        }
        {
            int4 _iv4 = make_int4(o[12 + 0], o[12 + 1], o[12 + 2], o[12 + 3]);
            *reinterpret_cast<int4*>(out + obase + 12) = _iv4;
        }
    }
}

} // extern "C"
