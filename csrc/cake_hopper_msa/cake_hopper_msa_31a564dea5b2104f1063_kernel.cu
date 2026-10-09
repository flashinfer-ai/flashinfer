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
#define SMEM_QCOL_STAGE_BYTES 128
#define SMEM_QCOL_STRIDE 128
#define SMEM_FLAGW_OFF 128
#define SMEM_FLAGW_STAGE_BYTES 16
#define SMEM_FLAGW_STRIDE 16
#define SMEM_PUB_OFF 144
#define SMEM_PUB_STAGE_BYTES 65536
#define SMEM_PUB_STRIDE 65536
#define SMEM_TOTAL 65792
#define THREADS 512

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

__global__ __launch_bounds__(512) void
kernel_cake_hopper_msa_31a564dea5b2104f1063(unsigned int* __restrict__ S, int* __restrict__ nvp, int* __restrict__ out, int tiles, int total_q, int num_heads, int num_chunks, int fb, int fe)
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
    unsigned int* flagw = reinterpret_cast<unsigned int*>(smem_raw + 128);
    const int flagw_addr = smem + 128;
    float* pub = reinterpret_cast<float*>(smem_raw + 144);
    const int pub_addr = smem + 144;

    // === Task calls (dependency order) ===
    int tid_1 = threadIdx.x;
    int w = tid_1 / 32;
    int c = tid_1 - w * 32;
    int whi = w;
    int col = blockIdx.x * 32 + c;
    int head = blockIdx.y;
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
    float ax[16];
    ax[0] = 0.0f;
    ax[1] = 0.0f;
    ax[2] = 0.0f;
    ax[3] = 0.0f;
    ax[4] = 0.0f;
    ax[5] = 0.0f;
    ax[6] = 0.0f;
    ax[7] = 0.0f;
    ax[8] = 0.0f;
    ax[9] = 0.0f;
    ax[10] = 0.0f;
    ax[11] = 0.0f;
    ax[12] = 0.0f;
    ax[13] = 0.0f;
    ax[14] = 0.0f;
    ax[15] = 0.0f;
    float rej = neg_inf;
    unsigned int cb[16];
    float kb[16];
    float kx[16];
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
        int t0_2 = ((j + 1) * 16 + w) * 16;
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
        int t0_5 = (j * 16 + w) * 16;
        float sc = __uint_as_float(cb[0]);
        float _fmax_0 = fmaxf(sc, -1.7014118346046923e+38f);
        sc = _fmax_0;
        float _min_0 = fminf(sc, 1.7014118346046923e+38f);
        sc = _min_0;
        sc = sc;
        kb[0] = sc;
        kx[0] = __uint_as_float((unsigned int)t0_5);
        float sc_6 = __uint_as_float(cb[1]);
        float _fmax_1 = fmaxf(sc_6, -1.7014118346046923e+38f);
        sc_6 = _fmax_1;
        float _min_1 = fminf(sc_6, 1.7014118346046923e+38f);
        sc_6 = _min_1;
        sc_6 = sc_6;
        kb[1] = sc_6;
        kx[1] = __uint_as_float((unsigned int)(t0_5 + 1));
        float sc_7 = __uint_as_float(cb[2]);
        float _fmax_2 = fmaxf(sc_7, -1.7014118346046923e+38f);
        sc_7 = _fmax_2;
        float _min_2 = fminf(sc_7, 1.7014118346046923e+38f);
        sc_7 = _min_2;
        sc_7 = sc_7;
        kb[2] = sc_7;
        kx[2] = __uint_as_float((unsigned int)(t0_5 + 2));
        float sc_8 = __uint_as_float(cb[3]);
        float _fmax_3 = fmaxf(sc_8, -1.7014118346046923e+38f);
        sc_8 = _fmax_3;
        float _min_3 = fminf(sc_8, 1.7014118346046923e+38f);
        sc_8 = _min_3;
        sc_8 = sc_8;
        kb[3] = sc_8;
        kx[3] = __uint_as_float((unsigned int)(t0_5 + 3));
        float sc_9 = __uint_as_float(cb[4]);
        float _fmax_4 = fmaxf(sc_9, -1.7014118346046923e+38f);
        sc_9 = _fmax_4;
        float _min_4 = fminf(sc_9, 1.7014118346046923e+38f);
        sc_9 = _min_4;
        sc_9 = sc_9;
        kb[4] = sc_9;
        kx[4] = __uint_as_float((unsigned int)(t0_5 + 4));
        float sc_10 = __uint_as_float(cb[5]);
        float _fmax_5 = fmaxf(sc_10, -1.7014118346046923e+38f);
        sc_10 = _fmax_5;
        float _min_5 = fminf(sc_10, 1.7014118346046923e+38f);
        sc_10 = _min_5;
        sc_10 = sc_10;
        kb[5] = sc_10;
        kx[5] = __uint_as_float((unsigned int)(t0_5 + 5));
        float sc_11 = __uint_as_float(cb[6]);
        float _fmax_6 = fmaxf(sc_11, -1.7014118346046923e+38f);
        sc_11 = _fmax_6;
        float _min_6 = fminf(sc_11, 1.7014118346046923e+38f);
        sc_11 = _min_6;
        sc_11 = sc_11;
        kb[6] = sc_11;
        kx[6] = __uint_as_float((unsigned int)(t0_5 + 6));
        float sc_12 = __uint_as_float(cb[7]);
        float _fmax_7 = fmaxf(sc_12, -1.7014118346046923e+38f);
        sc_12 = _fmax_7;
        float _min_7 = fminf(sc_12, 1.7014118346046923e+38f);
        sc_12 = _min_7;
        sc_12 = sc_12;
        kb[7] = sc_12;
        kx[7] = __uint_as_float((unsigned int)(t0_5 + 7));
        float sc_13 = __uint_as_float(cb[8]);
        float _fmax_8 = fmaxf(sc_13, -1.7014118346046923e+38f);
        sc_13 = _fmax_8;
        float _min_8 = fminf(sc_13, 1.7014118346046923e+38f);
        sc_13 = _min_8;
        sc_13 = sc_13;
        kb[8] = sc_13;
        kx[8] = __uint_as_float((unsigned int)(t0_5 + 8));
        float sc_14 = __uint_as_float(cb[9]);
        float _fmax_9 = fmaxf(sc_14, -1.7014118346046923e+38f);
        sc_14 = _fmax_9;
        float _min_9 = fminf(sc_14, 1.7014118346046923e+38f);
        sc_14 = _min_9;
        sc_14 = sc_14;
        kb[9] = sc_14;
        kx[9] = __uint_as_float((unsigned int)(t0_5 + 9));
        float sc_15 = __uint_as_float(cb[10]);
        float _fmax_10 = fmaxf(sc_15, -1.7014118346046923e+38f);
        sc_15 = _fmax_10;
        float _min_10 = fminf(sc_15, 1.7014118346046923e+38f);
        sc_15 = _min_10;
        sc_15 = sc_15;
        kb[10] = sc_15;
        kx[10] = __uint_as_float((unsigned int)(t0_5 + 10));
        float sc_16 = __uint_as_float(cb[11]);
        float _fmax_11 = fmaxf(sc_16, -1.7014118346046923e+38f);
        sc_16 = _fmax_11;
        float _min_11 = fminf(sc_16, 1.7014118346046923e+38f);
        sc_16 = _min_11;
        sc_16 = sc_16;
        kb[11] = sc_16;
        kx[11] = __uint_as_float((unsigned int)(t0_5 + 11));
        float sc_17 = __uint_as_float(cb[12]);
        float _fmax_12 = fmaxf(sc_17, -1.7014118346046923e+38f);
        sc_17 = _fmax_12;
        float _min_12 = fminf(sc_17, 1.7014118346046923e+38f);
        sc_17 = _min_12;
        sc_17 = sc_17;
        kb[12] = sc_17;
        kx[12] = __uint_as_float((unsigned int)(t0_5 + 12));
        float sc_18 = __uint_as_float(cb[13]);
        float _fmax_13 = fmaxf(sc_18, -1.7014118346046923e+38f);
        sc_18 = _fmax_13;
        float _min_13 = fminf(sc_18, 1.7014118346046923e+38f);
        sc_18 = _min_13;
        sc_18 = sc_18;
        kb[13] = sc_18;
        kx[13] = __uint_as_float((unsigned int)(t0_5 + 13));
        float sc_19 = __uint_as_float(cb[14]);
        float _fmax_14 = fmaxf(sc_19, -1.7014118346046923e+38f);
        sc_19 = _fmax_14;
        float _min_14 = fminf(sc_19, 1.7014118346046923e+38f);
        sc_19 = _min_14;
        sc_19 = sc_19;
        kb[14] = sc_19;
        kx[14] = __uint_as_float((unsigned int)(t0_5 + 14));
        float sc_20 = __uint_as_float(cb[15]);
        float _fmax_15 = fmaxf(sc_20, -1.7014118346046923e+38f);
        sc_20 = _fmax_15;
        float _min_15 = fminf(sc_20, 1.7014118346046923e+38f);
        sc_20 = _min_15;
        sc_20 = sc_20;
        kb[15] = sc_20;
        kx[15] = __uint_as_float((unsigned int)(t0_5 + 15));
        int f = 0;
        if (t0_5 < fb || t0_5 + 16 > lim - fe && t0_5 < lim) {
            f = 1;
        }
        int fcp = f;
        if (fcp != 0) {
            int f_0 = 0;
            if (t0_5 < fb || t0_5 >= lim - fe && lim > t0_5) {
                f_0 = 1;
            }
            if (f_0 != 0) {
                kb[0] = 3.4028234663852886e+38f;
            }
            int f_1 = 0;
            if (t0_5 + 1 < fb || t0_5 + 1 >= lim - fe && lim > t0_5 + 1) {
                f_1 = 1;
            }
            if (f_1 != 0) {
                kb[1] = 3.4028234663852886e+38f;
            }
            int f_2 = 0;
            if (t0_5 + 2 < fb || t0_5 + 2 >= lim - fe && lim > t0_5 + 2) {
                f_2 = 1;
            }
            if (f_2 != 0) {
                kb[2] = 3.4028234663852886e+38f;
            }
            int f_3 = 0;
            if (t0_5 + 3 < fb || t0_5 + 3 >= lim - fe && lim > t0_5 + 3) {
                f_3 = 1;
            }
            if (f_3 != 0) {
                kb[3] = 3.4028234663852886e+38f;
            }
            int f_4 = 0;
            if (t0_5 + 4 < fb || t0_5 + 4 >= lim - fe && lim > t0_5 + 4) {
                f_4 = 1;
            }
            if (f_4 != 0) {
                kb[4] = 3.4028234663852886e+38f;
            }
            int f_5 = 0;
            if (t0_5 + 5 < fb || t0_5 + 5 >= lim - fe && lim > t0_5 + 5) {
                f_5 = 1;
            }
            if (f_5 != 0) {
                kb[5] = 3.4028234663852886e+38f;
            }
            int f_6 = 0;
            if (t0_5 + 6 < fb || t0_5 + 6 >= lim - fe && lim > t0_5 + 6) {
                f_6 = 1;
            }
            if (f_6 != 0) {
                kb[6] = 3.4028234663852886e+38f;
            }
            int f_7 = 0;
            if (t0_5 + 7 < fb || t0_5 + 7 >= lim - fe && lim > t0_5 + 7) {
                f_7 = 1;
            }
            if (f_7 != 0) {
                kb[7] = 3.4028234663852886e+38f;
            }
            int f_8 = 0;
            if (t0_5 + 8 < fb || t0_5 + 8 >= lim - fe && lim > t0_5 + 8) {
                f_8 = 1;
            }
            if (f_8 != 0) {
                kb[8] = 3.4028234663852886e+38f;
            }
            int f_9 = 0;
            if (t0_5 + 9 < fb || t0_5 + 9 >= lim - fe && lim > t0_5 + 9) {
                f_9 = 1;
            }
            if (f_9 != 0) {
                kb[9] = 3.4028234663852886e+38f;
            }
            int f_10 = 0;
            if (t0_5 + 10 < fb || t0_5 + 10 >= lim - fe && lim > t0_5 + 10) {
                f_10 = 1;
            }
            if (f_10 != 0) {
                kb[10] = 3.4028234663852886e+38f;
            }
            int f_11 = 0;
            if (t0_5 + 11 < fb || t0_5 + 11 >= lim - fe && lim > t0_5 + 11) {
                f_11 = 1;
            }
            if (f_11 != 0) {
                kb[11] = 3.4028234663852886e+38f;
            }
            int f_12 = 0;
            if (t0_5 + 12 < fb || t0_5 + 12 >= lim - fe && lim > t0_5 + 12) {
                f_12 = 1;
            }
            if (f_12 != 0) {
                kb[12] = 3.4028234663852886e+38f;
            }
            int f_13 = 0;
            if (t0_5 + 13 < fb || t0_5 + 13 >= lim - fe && lim > t0_5 + 13) {
                f_13 = 1;
            }
            if (f_13 != 0) {
                kb[13] = 3.4028234663852886e+38f;
            }
            int f_14 = 0;
            if (t0_5 + 14 < fb || t0_5 + 14 >= lim - fe && lim > t0_5 + 14) {
                f_14 = 1;
            }
            if (f_14 != 0) {
                kb[14] = 3.4028234663852886e+38f;
            }
            int f_15 = 0;
            if (t0_5 + 15 < fb || t0_5 + 15 >= lim - fe && lim > t0_5 + 15) {
                f_15 = 1;
            }
            if (f_15 != 0) {
                kb[15] = 3.4028234663852886e+38f;
            }
        }
        float _fmax_16 = fmaxf(kb[0], kb[13]);
        float hi = _fmax_16;
        float _min_16 = fminf(kb[0], kb[13]);
        float lo = _min_16;
        float ihi = ((kb[0] >= kb[13]) ? kx[0] : kx[13]);
        float ilo = ((kb[0] >= kb[13]) ? kx[13] : kx[0]);
        kb[0] = hi;
        kb[13] = lo;
        kx[0] = ihi;
        kx[13] = ilo;
        float _fmax_17 = fmaxf(kb[1], kb[12]);
        float hi_21 = _fmax_17;
        float _min_17 = fminf(kb[1], kb[12]);
        float lo_22 = _min_17;
        float ihi_23 = ((kb[1] >= kb[12]) ? kx[1] : kx[12]);
        float ilo_24 = ((kb[1] >= kb[12]) ? kx[12] : kx[1]);
        kb[1] = hi_21;
        kb[12] = lo_22;
        kx[1] = ihi_23;
        kx[12] = ilo_24;
        float _fmax_18 = fmaxf(kb[2], kb[15]);
        float hi_25 = _fmax_18;
        float _min_18 = fminf(kb[2], kb[15]);
        float lo_26 = _min_18;
        float ihi_27 = ((kb[2] >= kb[15]) ? kx[2] : kx[15]);
        float ilo_28 = ((kb[2] >= kb[15]) ? kx[15] : kx[2]);
        kb[2] = hi_25;
        kb[15] = lo_26;
        kx[2] = ihi_27;
        kx[15] = ilo_28;
        float _fmax_19 = fmaxf(kb[3], kb[14]);
        float hi_29 = _fmax_19;
        float _min_19 = fminf(kb[3], kb[14]);
        float lo_30 = _min_19;
        float ihi_31 = ((kb[3] >= kb[14]) ? kx[3] : kx[14]);
        float ilo_32 = ((kb[3] >= kb[14]) ? kx[14] : kx[3]);
        kb[3] = hi_29;
        kb[14] = lo_30;
        kx[3] = ihi_31;
        kx[14] = ilo_32;
        float _fmax_20 = fmaxf(kb[4], kb[8]);
        float hi_33 = _fmax_20;
        float _min_20 = fminf(kb[4], kb[8]);
        float lo_34 = _min_20;
        float ihi_35 = ((kb[4] >= kb[8]) ? kx[4] : kx[8]);
        float ilo_36 = ((kb[4] >= kb[8]) ? kx[8] : kx[4]);
        kb[4] = hi_33;
        kb[8] = lo_34;
        kx[4] = ihi_35;
        kx[8] = ilo_36;
        float _fmax_21 = fmaxf(kb[5], kb[6]);
        float hi_37 = _fmax_21;
        float _min_21 = fminf(kb[5], kb[6]);
        float lo_38 = _min_21;
        float ihi_39 = ((kb[5] >= kb[6]) ? kx[5] : kx[6]);
        float ilo_40 = ((kb[5] >= kb[6]) ? kx[6] : kx[5]);
        kb[5] = hi_37;
        kb[6] = lo_38;
        kx[5] = ihi_39;
        kx[6] = ilo_40;
        float _fmax_22 = fmaxf(kb[7], kb[11]);
        float hi_41 = _fmax_22;
        float _min_22 = fminf(kb[7], kb[11]);
        float lo_42 = _min_22;
        float ihi_43 = ((kb[7] >= kb[11]) ? kx[7] : kx[11]);
        float ilo_44 = ((kb[7] >= kb[11]) ? kx[11] : kx[7]);
        kb[7] = hi_41;
        kb[11] = lo_42;
        kx[7] = ihi_43;
        kx[11] = ilo_44;
        float _fmax_23 = fmaxf(kb[9], kb[10]);
        float hi_45 = _fmax_23;
        float _min_23 = fminf(kb[9], kb[10]);
        float lo_46 = _min_23;
        float ihi_47 = ((kb[9] >= kb[10]) ? kx[9] : kx[10]);
        float ilo_48 = ((kb[9] >= kb[10]) ? kx[10] : kx[9]);
        kb[9] = hi_45;
        kb[10] = lo_46;
        kx[9] = ihi_47;
        kx[10] = ilo_48;
        float _fmax_24 = fmaxf(kb[0], kb[5]);
        float hi_49 = _fmax_24;
        float _min_24 = fminf(kb[0], kb[5]);
        float lo_50 = _min_24;
        float ihi_51 = ((kb[0] >= kb[5]) ? kx[0] : kx[5]);
        float ilo_52 = ((kb[0] >= kb[5]) ? kx[5] : kx[0]);
        kb[0] = hi_49;
        kb[5] = lo_50;
        kx[0] = ihi_51;
        kx[5] = ilo_52;
        float _fmax_25 = fmaxf(kb[1], kb[7]);
        float hi_53 = _fmax_25;
        float _min_25 = fminf(kb[1], kb[7]);
        float lo_54 = _min_25;
        float ihi_55 = ((kb[1] >= kb[7]) ? kx[1] : kx[7]);
        float ilo_56 = ((kb[1] >= kb[7]) ? kx[7] : kx[1]);
        kb[1] = hi_53;
        kb[7] = lo_54;
        kx[1] = ihi_55;
        kx[7] = ilo_56;
        float _fmax_26 = fmaxf(kb[2], kb[9]);
        float hi_57 = _fmax_26;
        float _min_26 = fminf(kb[2], kb[9]);
        float lo_58 = _min_26;
        float ihi_59 = ((kb[2] >= kb[9]) ? kx[2] : kx[9]);
        float ilo_60 = ((kb[2] >= kb[9]) ? kx[9] : kx[2]);
        kb[2] = hi_57;
        kb[9] = lo_58;
        kx[2] = ihi_59;
        kx[9] = ilo_60;
        float _fmax_27 = fmaxf(kb[3], kb[4]);
        float hi_61 = _fmax_27;
        float _min_27 = fminf(kb[3], kb[4]);
        float lo_62 = _min_27;
        float ihi_63 = ((kb[3] >= kb[4]) ? kx[3] : kx[4]);
        float ilo_64 = ((kb[3] >= kb[4]) ? kx[4] : kx[3]);
        kb[3] = hi_61;
        kb[4] = lo_62;
        kx[3] = ihi_63;
        kx[4] = ilo_64;
        float _fmax_28 = fmaxf(kb[6], kb[13]);
        float hi_65 = _fmax_28;
        float _min_28 = fminf(kb[6], kb[13]);
        float lo_66 = _min_28;
        float ihi_67 = ((kb[6] >= kb[13]) ? kx[6] : kx[13]);
        float ilo_68 = ((kb[6] >= kb[13]) ? kx[13] : kx[6]);
        kb[6] = hi_65;
        kb[13] = lo_66;
        kx[6] = ihi_67;
        kx[13] = ilo_68;
        float _fmax_29 = fmaxf(kb[8], kb[14]);
        float hi_69 = _fmax_29;
        float _min_29 = fminf(kb[8], kb[14]);
        float lo_70 = _min_29;
        float ihi_71 = ((kb[8] >= kb[14]) ? kx[8] : kx[14]);
        float ilo_72 = ((kb[8] >= kb[14]) ? kx[14] : kx[8]);
        kb[8] = hi_69;
        kb[14] = lo_70;
        kx[8] = ihi_71;
        kx[14] = ilo_72;
        float _fmax_30 = fmaxf(kb[10], kb[15]);
        float hi_73 = _fmax_30;
        float _min_30 = fminf(kb[10], kb[15]);
        float lo_74 = _min_30;
        float ihi_75 = ((kb[10] >= kb[15]) ? kx[10] : kx[15]);
        float ilo_76 = ((kb[10] >= kb[15]) ? kx[15] : kx[10]);
        kb[10] = hi_73;
        kb[15] = lo_74;
        kx[10] = ihi_75;
        kx[15] = ilo_76;
        float _fmax_31 = fmaxf(kb[11], kb[12]);
        float hi_77 = _fmax_31;
        float _min_31 = fminf(kb[11], kb[12]);
        float lo_78 = _min_31;
        float ihi_79 = ((kb[11] >= kb[12]) ? kx[11] : kx[12]);
        float ilo_80 = ((kb[11] >= kb[12]) ? kx[12] : kx[11]);
        kb[11] = hi_77;
        kb[12] = lo_78;
        kx[11] = ihi_79;
        kx[12] = ilo_80;
        float _fmax_32 = fmaxf(kb[0], kb[1]);
        float hi_81 = _fmax_32;
        float _min_32 = fminf(kb[0], kb[1]);
        float lo_82 = _min_32;
        float ihi_83 = ((kb[0] >= kb[1]) ? kx[0] : kx[1]);
        float ilo_84 = ((kb[0] >= kb[1]) ? kx[1] : kx[0]);
        kb[0] = hi_81;
        kb[1] = lo_82;
        kx[0] = ihi_83;
        kx[1] = ilo_84;
        float _fmax_33 = fmaxf(kb[2], kb[3]);
        float hi_85 = _fmax_33;
        float _min_33 = fminf(kb[2], kb[3]);
        float lo_86 = _min_33;
        float ihi_87 = ((kb[2] >= kb[3]) ? kx[2] : kx[3]);
        float ilo_88 = ((kb[2] >= kb[3]) ? kx[3] : kx[2]);
        kb[2] = hi_85;
        kb[3] = lo_86;
        kx[2] = ihi_87;
        kx[3] = ilo_88;
        float _fmax_34 = fmaxf(kb[4], kb[5]);
        float hi_89 = _fmax_34;
        float _min_34 = fminf(kb[4], kb[5]);
        float lo_90 = _min_34;
        float ihi_91 = ((kb[4] >= kb[5]) ? kx[4] : kx[5]);
        float ilo_92 = ((kb[4] >= kb[5]) ? kx[5] : kx[4]);
        kb[4] = hi_89;
        kb[5] = lo_90;
        kx[4] = ihi_91;
        kx[5] = ilo_92;
        float _fmax_35 = fmaxf(kb[6], kb[8]);
        float hi_93 = _fmax_35;
        float _min_35 = fminf(kb[6], kb[8]);
        float lo_94 = _min_35;
        float ihi_95 = ((kb[6] >= kb[8]) ? kx[6] : kx[8]);
        float ilo_96 = ((kb[6] >= kb[8]) ? kx[8] : kx[6]);
        kb[6] = hi_93;
        kb[8] = lo_94;
        kx[6] = ihi_95;
        kx[8] = ilo_96;
        float _fmax_36 = fmaxf(kb[7], kb[9]);
        float hi_97 = _fmax_36;
        float _min_36 = fminf(kb[7], kb[9]);
        float lo_98 = _min_36;
        float ihi_99 = ((kb[7] >= kb[9]) ? kx[7] : kx[9]);
        float ilo_100 = ((kb[7] >= kb[9]) ? kx[9] : kx[7]);
        kb[7] = hi_97;
        kb[9] = lo_98;
        kx[7] = ihi_99;
        kx[9] = ilo_100;
        float _fmax_37 = fmaxf(kb[10], kb[11]);
        float hi_101 = _fmax_37;
        float _min_37 = fminf(kb[10], kb[11]);
        float lo_102 = _min_37;
        float ihi_103 = ((kb[10] >= kb[11]) ? kx[10] : kx[11]);
        float ilo_104 = ((kb[10] >= kb[11]) ? kx[11] : kx[10]);
        kb[10] = hi_101;
        kb[11] = lo_102;
        kx[10] = ihi_103;
        kx[11] = ilo_104;
        float _fmax_38 = fmaxf(kb[12], kb[13]);
        float hi_105 = _fmax_38;
        float _min_38 = fminf(kb[12], kb[13]);
        float lo_106 = _min_38;
        float ihi_107 = ((kb[12] >= kb[13]) ? kx[12] : kx[13]);
        float ilo_108 = ((kb[12] >= kb[13]) ? kx[13] : kx[12]);
        kb[12] = hi_105;
        kb[13] = lo_106;
        kx[12] = ihi_107;
        kx[13] = ilo_108;
        float _fmax_39 = fmaxf(kb[14], kb[15]);
        float hi_109 = _fmax_39;
        float _min_39 = fminf(kb[14], kb[15]);
        float lo_110 = _min_39;
        float ihi_111 = ((kb[14] >= kb[15]) ? kx[14] : kx[15]);
        float ilo_112 = ((kb[14] >= kb[15]) ? kx[15] : kx[14]);
        kb[14] = hi_109;
        kb[15] = lo_110;
        kx[14] = ihi_111;
        kx[15] = ilo_112;
        float _fmax_40 = fmaxf(kb[0], kb[2]);
        float hi_113 = _fmax_40;
        float _min_40 = fminf(kb[0], kb[2]);
        float lo_114 = _min_40;
        float ihi_115 = ((kb[0] >= kb[2]) ? kx[0] : kx[2]);
        float ilo_116 = ((kb[0] >= kb[2]) ? kx[2] : kx[0]);
        kb[0] = hi_113;
        kb[2] = lo_114;
        kx[0] = ihi_115;
        kx[2] = ilo_116;
        float _fmax_41 = fmaxf(kb[1], kb[3]);
        float hi_117 = _fmax_41;
        float _min_41 = fminf(kb[1], kb[3]);
        float lo_118 = _min_41;
        float ihi_119 = ((kb[1] >= kb[3]) ? kx[1] : kx[3]);
        float ilo_120 = ((kb[1] >= kb[3]) ? kx[3] : kx[1]);
        kb[1] = hi_117;
        kb[3] = lo_118;
        kx[1] = ihi_119;
        kx[3] = ilo_120;
        float _fmax_42 = fmaxf(kb[4], kb[10]);
        float hi_121 = _fmax_42;
        float _min_42 = fminf(kb[4], kb[10]);
        float lo_122 = _min_42;
        float ihi_123 = ((kb[4] >= kb[10]) ? kx[4] : kx[10]);
        float ilo_124 = ((kb[4] >= kb[10]) ? kx[10] : kx[4]);
        kb[4] = hi_121;
        kb[10] = lo_122;
        kx[4] = ihi_123;
        kx[10] = ilo_124;
        float _fmax_43 = fmaxf(kb[5], kb[11]);
        float hi_125 = _fmax_43;
        float _min_43 = fminf(kb[5], kb[11]);
        float lo_126 = _min_43;
        float ihi_127 = ((kb[5] >= kb[11]) ? kx[5] : kx[11]);
        float ilo_128 = ((kb[5] >= kb[11]) ? kx[11] : kx[5]);
        kb[5] = hi_125;
        kb[11] = lo_126;
        kx[5] = ihi_127;
        kx[11] = ilo_128;
        float _fmax_44 = fmaxf(kb[6], kb[7]);
        float hi_129 = _fmax_44;
        float _min_44 = fminf(kb[6], kb[7]);
        float lo_130 = _min_44;
        float ihi_131 = ((kb[6] >= kb[7]) ? kx[6] : kx[7]);
        float ilo_132 = ((kb[6] >= kb[7]) ? kx[7] : kx[6]);
        kb[6] = hi_129;
        kb[7] = lo_130;
        kx[6] = ihi_131;
        kx[7] = ilo_132;
        float _fmax_45 = fmaxf(kb[8], kb[9]);
        float hi_133 = _fmax_45;
        float _min_45 = fminf(kb[8], kb[9]);
        float lo_134 = _min_45;
        float ihi_135 = ((kb[8] >= kb[9]) ? kx[8] : kx[9]);
        float ilo_136 = ((kb[8] >= kb[9]) ? kx[9] : kx[8]);
        kb[8] = hi_133;
        kb[9] = lo_134;
        kx[8] = ihi_135;
        kx[9] = ilo_136;
        float _fmax_46 = fmaxf(kb[12], kb[14]);
        float hi_137 = _fmax_46;
        float _min_46 = fminf(kb[12], kb[14]);
        float lo_138 = _min_46;
        float ihi_139 = ((kb[12] >= kb[14]) ? kx[12] : kx[14]);
        float ilo_140 = ((kb[12] >= kb[14]) ? kx[14] : kx[12]);
        kb[12] = hi_137;
        kb[14] = lo_138;
        kx[12] = ihi_139;
        kx[14] = ilo_140;
        float _fmax_47 = fmaxf(kb[13], kb[15]);
        float hi_141 = _fmax_47;
        float _min_47 = fminf(kb[13], kb[15]);
        float lo_142 = _min_47;
        float ihi_143 = ((kb[13] >= kb[15]) ? kx[13] : kx[15]);
        float ilo_144 = ((kb[13] >= kb[15]) ? kx[15] : kx[13]);
        kb[13] = hi_141;
        kb[15] = lo_142;
        kx[13] = ihi_143;
        kx[15] = ilo_144;
        float _fmax_48 = fmaxf(kb[1], kb[2]);
        float hi_145 = _fmax_48;
        float _min_48 = fminf(kb[1], kb[2]);
        float lo_146 = _min_48;
        float ihi_147 = ((kb[1] >= kb[2]) ? kx[1] : kx[2]);
        float ilo_148 = ((kb[1] >= kb[2]) ? kx[2] : kx[1]);
        kb[1] = hi_145;
        kb[2] = lo_146;
        kx[1] = ihi_147;
        kx[2] = ilo_148;
        float _fmax_49 = fmaxf(kb[3], kb[12]);
        float hi_149 = _fmax_49;
        float _min_49 = fminf(kb[3], kb[12]);
        float lo_150 = _min_49;
        float ihi_151 = ((kb[3] >= kb[12]) ? kx[3] : kx[12]);
        float ilo_152 = ((kb[3] >= kb[12]) ? kx[12] : kx[3]);
        kb[3] = hi_149;
        kb[12] = lo_150;
        kx[3] = ihi_151;
        kx[12] = ilo_152;
        float _fmax_50 = fmaxf(kb[4], kb[6]);
        float hi_153 = _fmax_50;
        float _min_50 = fminf(kb[4], kb[6]);
        float lo_154 = _min_50;
        float ihi_155 = ((kb[4] >= kb[6]) ? kx[4] : kx[6]);
        float ilo_156 = ((kb[4] >= kb[6]) ? kx[6] : kx[4]);
        kb[4] = hi_153;
        kb[6] = lo_154;
        kx[4] = ihi_155;
        kx[6] = ilo_156;
        float _fmax_51 = fmaxf(kb[5], kb[7]);
        float hi_157 = _fmax_51;
        float _min_51 = fminf(kb[5], kb[7]);
        float lo_158 = _min_51;
        float ihi_159 = ((kb[5] >= kb[7]) ? kx[5] : kx[7]);
        float ilo_160 = ((kb[5] >= kb[7]) ? kx[7] : kx[5]);
        kb[5] = hi_157;
        kb[7] = lo_158;
        kx[5] = ihi_159;
        kx[7] = ilo_160;
        float _fmax_52 = fmaxf(kb[8], kb[10]);
        float hi_161 = _fmax_52;
        float _min_52 = fminf(kb[8], kb[10]);
        float lo_162 = _min_52;
        float ihi_163 = ((kb[8] >= kb[10]) ? kx[8] : kx[10]);
        float ilo_164 = ((kb[8] >= kb[10]) ? kx[10] : kx[8]);
        kb[8] = hi_161;
        kb[10] = lo_162;
        kx[8] = ihi_163;
        kx[10] = ilo_164;
        float _fmax_53 = fmaxf(kb[9], kb[11]);
        float hi_165 = _fmax_53;
        float _min_53 = fminf(kb[9], kb[11]);
        float lo_166 = _min_53;
        float ihi_167 = ((kb[9] >= kb[11]) ? kx[9] : kx[11]);
        float ilo_168 = ((kb[9] >= kb[11]) ? kx[11] : kx[9]);
        kb[9] = hi_165;
        kb[11] = lo_166;
        kx[9] = ihi_167;
        kx[11] = ilo_168;
        float _fmax_54 = fmaxf(kb[13], kb[14]);
        float hi_169 = _fmax_54;
        float _min_54 = fminf(kb[13], kb[14]);
        float lo_170 = _min_54;
        float ihi_171 = ((kb[13] >= kb[14]) ? kx[13] : kx[14]);
        float ilo_172 = ((kb[13] >= kb[14]) ? kx[14] : kx[13]);
        kb[13] = hi_169;
        kb[14] = lo_170;
        kx[13] = ihi_171;
        kx[14] = ilo_172;
        float _fmax_55 = fmaxf(kb[1], kb[4]);
        float hi_173 = _fmax_55;
        float _min_55 = fminf(kb[1], kb[4]);
        float lo_174 = _min_55;
        float ihi_175 = ((kb[1] >= kb[4]) ? kx[1] : kx[4]);
        float ilo_176 = ((kb[1] >= kb[4]) ? kx[4] : kx[1]);
        kb[1] = hi_173;
        kb[4] = lo_174;
        kx[1] = ihi_175;
        kx[4] = ilo_176;
        float _fmax_56 = fmaxf(kb[2], kb[6]);
        float hi_177 = _fmax_56;
        float _min_56 = fminf(kb[2], kb[6]);
        float lo_178 = _min_56;
        float ihi_179 = ((kb[2] >= kb[6]) ? kx[2] : kx[6]);
        float ilo_180 = ((kb[2] >= kb[6]) ? kx[6] : kx[2]);
        kb[2] = hi_177;
        kb[6] = lo_178;
        kx[2] = ihi_179;
        kx[6] = ilo_180;
        float _fmax_57 = fmaxf(kb[5], kb[8]);
        float hi_181 = _fmax_57;
        float _min_57 = fminf(kb[5], kb[8]);
        float lo_182 = _min_57;
        float ihi_183 = ((kb[5] >= kb[8]) ? kx[5] : kx[8]);
        float ilo_184 = ((kb[5] >= kb[8]) ? kx[8] : kx[5]);
        kb[5] = hi_181;
        kb[8] = lo_182;
        kx[5] = ihi_183;
        kx[8] = ilo_184;
        float _fmax_58 = fmaxf(kb[7], kb[10]);
        float hi_185 = _fmax_58;
        float _min_58 = fminf(kb[7], kb[10]);
        float lo_186 = _min_58;
        float ihi_187 = ((kb[7] >= kb[10]) ? kx[7] : kx[10]);
        float ilo_188 = ((kb[7] >= kb[10]) ? kx[10] : kx[7]);
        kb[7] = hi_185;
        kb[10] = lo_186;
        kx[7] = ihi_187;
        kx[10] = ilo_188;
        float _fmax_59 = fmaxf(kb[9], kb[13]);
        float hi_189 = _fmax_59;
        float _min_59 = fminf(kb[9], kb[13]);
        float lo_190 = _min_59;
        float ihi_191 = ((kb[9] >= kb[13]) ? kx[9] : kx[13]);
        float ilo_192 = ((kb[9] >= kb[13]) ? kx[13] : kx[9]);
        kb[9] = hi_189;
        kb[13] = lo_190;
        kx[9] = ihi_191;
        kx[13] = ilo_192;
        float _fmax_60 = fmaxf(kb[11], kb[14]);
        float hi_193 = _fmax_60;
        float _min_60 = fminf(kb[11], kb[14]);
        float lo_194 = _min_60;
        float ihi_195 = ((kb[11] >= kb[14]) ? kx[11] : kx[14]);
        float ilo_196 = ((kb[11] >= kb[14]) ? kx[14] : kx[11]);
        kb[11] = hi_193;
        kb[14] = lo_194;
        kx[11] = ihi_195;
        kx[14] = ilo_196;
        float _fmax_61 = fmaxf(kb[2], kb[4]);
        float hi_197 = _fmax_61;
        float _min_61 = fminf(kb[2], kb[4]);
        float lo_198 = _min_61;
        float ihi_199 = ((kb[2] >= kb[4]) ? kx[2] : kx[4]);
        float ilo_200 = ((kb[2] >= kb[4]) ? kx[4] : kx[2]);
        kb[2] = hi_197;
        kb[4] = lo_198;
        kx[2] = ihi_199;
        kx[4] = ilo_200;
        float _fmax_62 = fmaxf(kb[3], kb[6]);
        float hi_201 = _fmax_62;
        float _min_62 = fminf(kb[3], kb[6]);
        float lo_202 = _min_62;
        float ihi_203 = ((kb[3] >= kb[6]) ? kx[3] : kx[6]);
        float ilo_204 = ((kb[3] >= kb[6]) ? kx[6] : kx[3]);
        kb[3] = hi_201;
        kb[6] = lo_202;
        kx[3] = ihi_203;
        kx[6] = ilo_204;
        float _fmax_63 = fmaxf(kb[9], kb[12]);
        float hi_205 = _fmax_63;
        float _min_63 = fminf(kb[9], kb[12]);
        float lo_206 = _min_63;
        float ihi_207 = ((kb[9] >= kb[12]) ? kx[9] : kx[12]);
        float ilo_208 = ((kb[9] >= kb[12]) ? kx[12] : kx[9]);
        kb[9] = hi_205;
        kb[12] = lo_206;
        kx[9] = ihi_207;
        kx[12] = ilo_208;
        float _fmax_64 = fmaxf(kb[11], kb[13]);
        float hi_209 = _fmax_64;
        float _min_64 = fminf(kb[11], kb[13]);
        float lo_210 = _min_64;
        float ihi_211 = ((kb[11] >= kb[13]) ? kx[11] : kx[13]);
        float ilo_212 = ((kb[11] >= kb[13]) ? kx[13] : kx[11]);
        kb[11] = hi_209;
        kb[13] = lo_210;
        kx[11] = ihi_211;
        kx[13] = ilo_212;
        float _fmax_65 = fmaxf(kb[3], kb[5]);
        float hi_213 = _fmax_65;
        float _min_65 = fminf(kb[3], kb[5]);
        float lo_214 = _min_65;
        float ihi_215 = ((kb[3] >= kb[5]) ? kx[3] : kx[5]);
        float ilo_216 = ((kb[3] >= kb[5]) ? kx[5] : kx[3]);
        kb[3] = hi_213;
        kb[5] = lo_214;
        kx[3] = ihi_215;
        kx[5] = ilo_216;
        float _fmax_66 = fmaxf(kb[6], kb[8]);
        float hi_217 = _fmax_66;
        float _min_66 = fminf(kb[6], kb[8]);
        float lo_218 = _min_66;
        float ihi_219 = ((kb[6] >= kb[8]) ? kx[6] : kx[8]);
        float ilo_220 = ((kb[6] >= kb[8]) ? kx[8] : kx[6]);
        kb[6] = hi_217;
        kb[8] = lo_218;
        kx[6] = ihi_219;
        kx[8] = ilo_220;
        float _fmax_67 = fmaxf(kb[7], kb[9]);
        float hi_221 = _fmax_67;
        float _min_67 = fminf(kb[7], kb[9]);
        float lo_222 = _min_67;
        float ihi_223 = ((kb[7] >= kb[9]) ? kx[7] : kx[9]);
        float ilo_224 = ((kb[7] >= kb[9]) ? kx[9] : kx[7]);
        kb[7] = hi_221;
        kb[9] = lo_222;
        kx[7] = ihi_223;
        kx[9] = ilo_224;
        float _fmax_68 = fmaxf(kb[10], kb[12]);
        float hi_225 = _fmax_68;
        float _min_68 = fminf(kb[10], kb[12]);
        float lo_226 = _min_68;
        float ihi_227 = ((kb[10] >= kb[12]) ? kx[10] : kx[12]);
        float ilo_228 = ((kb[10] >= kb[12]) ? kx[12] : kx[10]);
        kb[10] = hi_225;
        kb[12] = lo_226;
        kx[10] = ihi_227;
        kx[12] = ilo_228;
        float _fmax_69 = fmaxf(kb[3], kb[4]);
        float hi_229 = _fmax_69;
        float _min_69 = fminf(kb[3], kb[4]);
        float lo_230 = _min_69;
        float ihi_231 = ((kb[3] >= kb[4]) ? kx[3] : kx[4]);
        float ilo_232 = ((kb[3] >= kb[4]) ? kx[4] : kx[3]);
        kb[3] = hi_229;
        kb[4] = lo_230;
        kx[3] = ihi_231;
        kx[4] = ilo_232;
        float _fmax_70 = fmaxf(kb[5], kb[6]);
        float hi_233 = _fmax_70;
        float _min_70 = fminf(kb[5], kb[6]);
        float lo_234 = _min_70;
        float ihi_235 = ((kb[5] >= kb[6]) ? kx[5] : kx[6]);
        float ilo_236 = ((kb[5] >= kb[6]) ? kx[6] : kx[5]);
        kb[5] = hi_233;
        kb[6] = lo_234;
        kx[5] = ihi_235;
        kx[6] = ilo_236;
        float _fmax_71 = fmaxf(kb[7], kb[8]);
        float hi_237 = _fmax_71;
        float _min_71 = fminf(kb[7], kb[8]);
        float lo_238 = _min_71;
        float ihi_239 = ((kb[7] >= kb[8]) ? kx[7] : kx[8]);
        float ilo_240 = ((kb[7] >= kb[8]) ? kx[8] : kx[7]);
        kb[7] = hi_237;
        kb[8] = lo_238;
        kx[7] = ihi_239;
        kx[8] = ilo_240;
        float _fmax_72 = fmaxf(kb[9], kb[10]);
        float hi_241 = _fmax_72;
        float _min_72 = fminf(kb[9], kb[10]);
        float lo_242 = _min_72;
        float ihi_243 = ((kb[9] >= kb[10]) ? kx[9] : kx[10]);
        float ilo_244 = ((kb[9] >= kb[10]) ? kx[10] : kx[9]);
        kb[9] = hi_241;
        kb[10] = lo_242;
        kx[9] = ihi_243;
        kx[10] = ilo_244;
        float _fmax_73 = fmaxf(kb[11], kb[12]);
        float hi_245 = _fmax_73;
        float _min_73 = fminf(kb[11], kb[12]);
        float lo_246 = _min_73;
        float ihi_247 = ((kb[11] >= kb[12]) ? kx[11] : kx[12]);
        float ilo_248 = ((kb[11] >= kb[12]) ? kx[12] : kx[11]);
        kb[11] = hi_245;
        kb[12] = lo_246;
        kx[11] = ihi_247;
        kx[12] = ilo_248;
        float _fmax_74 = fmaxf(kb[6], kb[7]);
        float hi_249 = _fmax_74;
        float _min_74 = fminf(kb[6], kb[7]);
        float lo_250 = _min_74;
        float ihi_251 = ((kb[6] >= kb[7]) ? kx[6] : kx[7]);
        float ilo_252 = ((kb[6] >= kb[7]) ? kx[7] : kx[6]);
        kb[6] = hi_249;
        kb[7] = lo_250;
        kx[6] = ihi_251;
        kx[7] = ilo_252;
        float _fmax_75 = fmaxf(kb[8], kb[9]);
        float hi_253 = _fmax_75;
        float _min_75 = fminf(kb[8], kb[9]);
        float lo_254 = _min_75;
        float ihi_255 = ((kb[8] >= kb[9]) ? kx[8] : kx[9]);
        float ilo_256 = ((kb[8] >= kb[9]) ? kx[9] : kx[8]);
        kb[8] = hi_253;
        kb[9] = lo_254;
        kx[8] = ihi_255;
        kx[9] = ilo_256;
        float _fmax_76 = fmaxf(a[0], kb[15]);
        float hi_257 = _fmax_76;
        float ihi_258 = ((a[0] >= kb[15]) ? ax[0] : kx[15]);
        a[0] = hi_257;
        ax[0] = ihi_258;
        float _fmax_77 = fmaxf(a[1], kb[14]);
        float hi_259 = _fmax_77;
        float ihi_260 = ((a[1] >= kb[14]) ? ax[1] : kx[14]);
        a[1] = hi_259;
        ax[1] = ihi_260;
        float _fmax_78 = fmaxf(a[2], kb[13]);
        float hi_261 = _fmax_78;
        float ihi_262 = ((a[2] >= kb[13]) ? ax[2] : kx[13]);
        a[2] = hi_261;
        ax[2] = ihi_262;
        float _fmax_79 = fmaxf(a[3], kb[12]);
        float hi_263 = _fmax_79;
        float ihi_264 = ((a[3] >= kb[12]) ? ax[3] : kx[12]);
        a[3] = hi_263;
        ax[3] = ihi_264;
        float _fmax_80 = fmaxf(a[4], kb[11]);
        float hi_265 = _fmax_80;
        float ihi_266 = ((a[4] >= kb[11]) ? ax[4] : kx[11]);
        a[4] = hi_265;
        ax[4] = ihi_266;
        float _fmax_81 = fmaxf(a[5], kb[10]);
        float hi_267 = _fmax_81;
        float ihi_268 = ((a[5] >= kb[10]) ? ax[5] : kx[10]);
        a[5] = hi_267;
        ax[5] = ihi_268;
        float _fmax_82 = fmaxf(a[6], kb[9]);
        float hi_269 = _fmax_82;
        float ihi_270 = ((a[6] >= kb[9]) ? ax[6] : kx[9]);
        a[6] = hi_269;
        ax[6] = ihi_270;
        float _fmax_83 = fmaxf(a[7], kb[8]);
        float hi_271 = _fmax_83;
        float ihi_272 = ((a[7] >= kb[8]) ? ax[7] : kx[8]);
        a[7] = hi_271;
        ax[7] = ihi_272;
        float _fmax_84 = fmaxf(a[8], kb[7]);
        float hi_273 = _fmax_84;
        float ihi_274 = ((a[8] >= kb[7]) ? ax[8] : kx[7]);
        a[8] = hi_273;
        ax[8] = ihi_274;
        float _fmax_85 = fmaxf(a[9], kb[6]);
        float hi_275 = _fmax_85;
        float ihi_276 = ((a[9] >= kb[6]) ? ax[9] : kx[6]);
        a[9] = hi_275;
        ax[9] = ihi_276;
        float _fmax_86 = fmaxf(a[10], kb[5]);
        float hi_277 = _fmax_86;
        float ihi_278 = ((a[10] >= kb[5]) ? ax[10] : kx[5]);
        a[10] = hi_277;
        ax[10] = ihi_278;
        float _fmax_87 = fmaxf(a[11], kb[4]);
        float hi_279 = _fmax_87;
        float ihi_280 = ((a[11] >= kb[4]) ? ax[11] : kx[4]);
        a[11] = hi_279;
        ax[11] = ihi_280;
        float _fmax_88 = fmaxf(a[12], kb[3]);
        float hi_281 = _fmax_88;
        float ihi_282 = ((a[12] >= kb[3]) ? ax[12] : kx[3]);
        a[12] = hi_281;
        ax[12] = ihi_282;
        float _fmax_89 = fmaxf(a[13], kb[2]);
        float hi_283 = _fmax_89;
        float ihi_284 = ((a[13] >= kb[2]) ? ax[13] : kx[2]);
        a[13] = hi_283;
        ax[13] = ihi_284;
        float _fmax_90 = fmaxf(a[14], kb[1]);
        float hi_285 = _fmax_90;
        float ihi_286 = ((a[14] >= kb[1]) ? ax[14] : kx[1]);
        a[14] = hi_285;
        ax[14] = ihi_286;
        float _fmax_91 = fmaxf(a[15], kb[0]);
        float hi_287 = _fmax_91;
        float ihi_288 = ((a[15] >= kb[0]) ? ax[15] : kx[0]);
        a[15] = hi_287;
        ax[15] = ihi_288;
        float _fmax_92 = fmaxf(a[0], a[8]);
        float hi_289 = _fmax_92;
        float _min_76 = fminf(a[0], a[8]);
        float lo_290 = _min_76;
        float ihi_291 = ((a[0] >= a[8]) ? ax[0] : ax[8]);
        float ilo_292 = ((a[0] >= a[8]) ? ax[8] : ax[0]);
        a[0] = hi_289;
        a[8] = lo_290;
        ax[0] = ihi_291;
        ax[8] = ilo_292;
        float _fmax_93 = fmaxf(a[1], a[9]);
        float hi_293 = _fmax_93;
        float _min_77 = fminf(a[1], a[9]);
        float lo_294 = _min_77;
        float ihi_295 = ((a[1] >= a[9]) ? ax[1] : ax[9]);
        float ilo_296 = ((a[1] >= a[9]) ? ax[9] : ax[1]);
        a[1] = hi_293;
        a[9] = lo_294;
        ax[1] = ihi_295;
        ax[9] = ilo_296;
        float _fmax_94 = fmaxf(a[2], a[10]);
        float hi_297 = _fmax_94;
        float _min_78 = fminf(a[2], a[10]);
        float lo_298 = _min_78;
        float ihi_299 = ((a[2] >= a[10]) ? ax[2] : ax[10]);
        float ilo_300 = ((a[2] >= a[10]) ? ax[10] : ax[2]);
        a[2] = hi_297;
        a[10] = lo_298;
        ax[2] = ihi_299;
        ax[10] = ilo_300;
        float _fmax_95 = fmaxf(a[3], a[11]);
        float hi_301 = _fmax_95;
        float _min_79 = fminf(a[3], a[11]);
        float lo_302 = _min_79;
        float ihi_303 = ((a[3] >= a[11]) ? ax[3] : ax[11]);
        float ilo_304 = ((a[3] >= a[11]) ? ax[11] : ax[3]);
        a[3] = hi_301;
        a[11] = lo_302;
        ax[3] = ihi_303;
        ax[11] = ilo_304;
        float _fmax_96 = fmaxf(a[4], a[12]);
        float hi_305 = _fmax_96;
        float _min_80 = fminf(a[4], a[12]);
        float lo_306 = _min_80;
        float ihi_307 = ((a[4] >= a[12]) ? ax[4] : ax[12]);
        float ilo_308 = ((a[4] >= a[12]) ? ax[12] : ax[4]);
        a[4] = hi_305;
        a[12] = lo_306;
        ax[4] = ihi_307;
        ax[12] = ilo_308;
        float _fmax_97 = fmaxf(a[5], a[13]);
        float hi_309 = _fmax_97;
        float _min_81 = fminf(a[5], a[13]);
        float lo_310 = _min_81;
        float ihi_311 = ((a[5] >= a[13]) ? ax[5] : ax[13]);
        float ilo_312 = ((a[5] >= a[13]) ? ax[13] : ax[5]);
        a[5] = hi_309;
        a[13] = lo_310;
        ax[5] = ihi_311;
        ax[13] = ilo_312;
        float _fmax_98 = fmaxf(a[6], a[14]);
        float hi_313 = _fmax_98;
        float _min_82 = fminf(a[6], a[14]);
        float lo_314 = _min_82;
        float ihi_315 = ((a[6] >= a[14]) ? ax[6] : ax[14]);
        float ilo_316 = ((a[6] >= a[14]) ? ax[14] : ax[6]);
        a[6] = hi_313;
        a[14] = lo_314;
        ax[6] = ihi_315;
        ax[14] = ilo_316;
        float _fmax_99 = fmaxf(a[7], a[15]);
        float hi_317 = _fmax_99;
        float _min_83 = fminf(a[7], a[15]);
        float lo_318 = _min_83;
        float ihi_319 = ((a[7] >= a[15]) ? ax[7] : ax[15]);
        float ilo_320 = ((a[7] >= a[15]) ? ax[15] : ax[7]);
        a[7] = hi_317;
        a[15] = lo_318;
        ax[7] = ihi_319;
        ax[15] = ilo_320;
        float _fmax_100 = fmaxf(a[0], a[4]);
        float hi_321 = _fmax_100;
        float _min_84 = fminf(a[0], a[4]);
        float lo_322 = _min_84;
        float ihi_323 = ((a[0] >= a[4]) ? ax[0] : ax[4]);
        float ilo_324 = ((a[0] >= a[4]) ? ax[4] : ax[0]);
        a[0] = hi_321;
        a[4] = lo_322;
        ax[0] = ihi_323;
        ax[4] = ilo_324;
        float _fmax_101 = fmaxf(a[1], a[5]);
        float hi_325 = _fmax_101;
        float _min_85 = fminf(a[1], a[5]);
        float lo_326 = _min_85;
        float ihi_327 = ((a[1] >= a[5]) ? ax[1] : ax[5]);
        float ilo_328 = ((a[1] >= a[5]) ? ax[5] : ax[1]);
        a[1] = hi_325;
        a[5] = lo_326;
        ax[1] = ihi_327;
        ax[5] = ilo_328;
        float _fmax_102 = fmaxf(a[2], a[6]);
        float hi_329 = _fmax_102;
        float _min_86 = fminf(a[2], a[6]);
        float lo_330 = _min_86;
        float ihi_331 = ((a[2] >= a[6]) ? ax[2] : ax[6]);
        float ilo_332 = ((a[2] >= a[6]) ? ax[6] : ax[2]);
        a[2] = hi_329;
        a[6] = lo_330;
        ax[2] = ihi_331;
        ax[6] = ilo_332;
        float _fmax_103 = fmaxf(a[3], a[7]);
        float hi_333 = _fmax_103;
        float _min_87 = fminf(a[3], a[7]);
        float lo_334 = _min_87;
        float ihi_335 = ((a[3] >= a[7]) ? ax[3] : ax[7]);
        float ilo_336 = ((a[3] >= a[7]) ? ax[7] : ax[3]);
        a[3] = hi_333;
        a[7] = lo_334;
        ax[3] = ihi_335;
        ax[7] = ilo_336;
        float _fmax_104 = fmaxf(a[8], a[12]);
        float hi_337 = _fmax_104;
        float _min_88 = fminf(a[8], a[12]);
        float lo_338 = _min_88;
        float ihi_339 = ((a[8] >= a[12]) ? ax[8] : ax[12]);
        float ilo_340 = ((a[8] >= a[12]) ? ax[12] : ax[8]);
        a[8] = hi_337;
        a[12] = lo_338;
        ax[8] = ihi_339;
        ax[12] = ilo_340;
        float _fmax_105 = fmaxf(a[9], a[13]);
        float hi_341 = _fmax_105;
        float _min_89 = fminf(a[9], a[13]);
        float lo_342 = _min_89;
        float ihi_343 = ((a[9] >= a[13]) ? ax[9] : ax[13]);
        float ilo_344 = ((a[9] >= a[13]) ? ax[13] : ax[9]);
        a[9] = hi_341;
        a[13] = lo_342;
        ax[9] = ihi_343;
        ax[13] = ilo_344;
        float _fmax_106 = fmaxf(a[10], a[14]);
        float hi_345 = _fmax_106;
        float _min_90 = fminf(a[10], a[14]);
        float lo_346 = _min_90;
        float ihi_347 = ((a[10] >= a[14]) ? ax[10] : ax[14]);
        float ilo_348 = ((a[10] >= a[14]) ? ax[14] : ax[10]);
        a[10] = hi_345;
        a[14] = lo_346;
        ax[10] = ihi_347;
        ax[14] = ilo_348;
        float _fmax_107 = fmaxf(a[11], a[15]);
        float hi_349 = _fmax_107;
        float _min_91 = fminf(a[11], a[15]);
        float lo_350 = _min_91;
        float ihi_351 = ((a[11] >= a[15]) ? ax[11] : ax[15]);
        float ilo_352 = ((a[11] >= a[15]) ? ax[15] : ax[11]);
        a[11] = hi_349;
        a[15] = lo_350;
        ax[11] = ihi_351;
        ax[15] = ilo_352;
        float _fmax_108 = fmaxf(a[0], a[2]);
        float hi_353 = _fmax_108;
        float _min_92 = fminf(a[0], a[2]);
        float lo_354 = _min_92;
        float ihi_355 = ((a[0] >= a[2]) ? ax[0] : ax[2]);
        float ilo_356 = ((a[0] >= a[2]) ? ax[2] : ax[0]);
        a[0] = hi_353;
        a[2] = lo_354;
        ax[0] = ihi_355;
        ax[2] = ilo_356;
        float _fmax_109 = fmaxf(a[1], a[3]);
        float hi_357 = _fmax_109;
        float _min_93 = fminf(a[1], a[3]);
        float lo_358 = _min_93;
        float ihi_359 = ((a[1] >= a[3]) ? ax[1] : ax[3]);
        float ilo_360 = ((a[1] >= a[3]) ? ax[3] : ax[1]);
        a[1] = hi_357;
        a[3] = lo_358;
        ax[1] = ihi_359;
        ax[3] = ilo_360;
        float _fmax_110 = fmaxf(a[4], a[6]);
        float hi_361 = _fmax_110;
        float _min_94 = fminf(a[4], a[6]);
        float lo_362 = _min_94;
        float ihi_363 = ((a[4] >= a[6]) ? ax[4] : ax[6]);
        float ilo_364 = ((a[4] >= a[6]) ? ax[6] : ax[4]);
        a[4] = hi_361;
        a[6] = lo_362;
        ax[4] = ihi_363;
        ax[6] = ilo_364;
        float _fmax_111 = fmaxf(a[5], a[7]);
        float hi_365 = _fmax_111;
        float _min_95 = fminf(a[5], a[7]);
        float lo_366 = _min_95;
        float ihi_367 = ((a[5] >= a[7]) ? ax[5] : ax[7]);
        float ilo_368 = ((a[5] >= a[7]) ? ax[7] : ax[5]);
        a[5] = hi_365;
        a[7] = lo_366;
        ax[5] = ihi_367;
        ax[7] = ilo_368;
        float _fmax_112 = fmaxf(a[8], a[10]);
        float hi_369 = _fmax_112;
        float _min_96 = fminf(a[8], a[10]);
        float lo_370 = _min_96;
        float ihi_371 = ((a[8] >= a[10]) ? ax[8] : ax[10]);
        float ilo_372 = ((a[8] >= a[10]) ? ax[10] : ax[8]);
        a[8] = hi_369;
        a[10] = lo_370;
        ax[8] = ihi_371;
        ax[10] = ilo_372;
        float _fmax_113 = fmaxf(a[9], a[11]);
        float hi_373 = _fmax_113;
        float _min_97 = fminf(a[9], a[11]);
        float lo_374 = _min_97;
        float ihi_375 = ((a[9] >= a[11]) ? ax[9] : ax[11]);
        float ilo_376 = ((a[9] >= a[11]) ? ax[11] : ax[9]);
        a[9] = hi_373;
        a[11] = lo_374;
        ax[9] = ihi_375;
        ax[11] = ilo_376;
        float _fmax_114 = fmaxf(a[12], a[14]);
        float hi_377 = _fmax_114;
        float _min_98 = fminf(a[12], a[14]);
        float lo_378 = _min_98;
        float ihi_379 = ((a[12] >= a[14]) ? ax[12] : ax[14]);
        float ilo_380 = ((a[12] >= a[14]) ? ax[14] : ax[12]);
        a[12] = hi_377;
        a[14] = lo_378;
        ax[12] = ihi_379;
        ax[14] = ilo_380;
        float _fmax_115 = fmaxf(a[13], a[15]);
        float hi_381 = _fmax_115;
        float _min_99 = fminf(a[13], a[15]);
        float lo_382 = _min_99;
        float ihi_383 = ((a[13] >= a[15]) ? ax[13] : ax[15]);
        float ilo_384 = ((a[13] >= a[15]) ? ax[15] : ax[13]);
        a[13] = hi_381;
        a[15] = lo_382;
        ax[13] = ihi_383;
        ax[15] = ilo_384;
        float _fmax_116 = fmaxf(a[0], a[1]);
        float hi_385 = _fmax_116;
        float _min_100 = fminf(a[0], a[1]);
        float lo_386 = _min_100;
        float ihi_387 = ((a[0] >= a[1]) ? ax[0] : ax[1]);
        float ilo_388 = ((a[0] >= a[1]) ? ax[1] : ax[0]);
        a[0] = hi_385;
        a[1] = lo_386;
        ax[0] = ihi_387;
        ax[1] = ilo_388;
        float _fmax_117 = fmaxf(a[2], a[3]);
        float hi_389 = _fmax_117;
        float _min_101 = fminf(a[2], a[3]);
        float lo_390 = _min_101;
        float ihi_391 = ((a[2] >= a[3]) ? ax[2] : ax[3]);
        float ilo_392 = ((a[2] >= a[3]) ? ax[3] : ax[2]);
        a[2] = hi_389;
        a[3] = lo_390;
        ax[2] = ihi_391;
        ax[3] = ilo_392;
        float _fmax_118 = fmaxf(a[4], a[5]);
        float hi_393 = _fmax_118;
        float _min_102 = fminf(a[4], a[5]);
        float lo_394 = _min_102;
        float ihi_395 = ((a[4] >= a[5]) ? ax[4] : ax[5]);
        float ilo_396 = ((a[4] >= a[5]) ? ax[5] : ax[4]);
        a[4] = hi_393;
        a[5] = lo_394;
        ax[4] = ihi_395;
        ax[5] = ilo_396;
        float _fmax_119 = fmaxf(a[6], a[7]);
        float hi_397 = _fmax_119;
        float _min_103 = fminf(a[6], a[7]);
        float lo_398 = _min_103;
        float ihi_399 = ((a[6] >= a[7]) ? ax[6] : ax[7]);
        float ilo_400 = ((a[6] >= a[7]) ? ax[7] : ax[6]);
        a[6] = hi_397;
        a[7] = lo_398;
        ax[6] = ihi_399;
        ax[7] = ilo_400;
        float _fmax_120 = fmaxf(a[8], a[9]);
        float hi_401 = _fmax_120;
        float _min_104 = fminf(a[8], a[9]);
        float lo_402 = _min_104;
        float ihi_403 = ((a[8] >= a[9]) ? ax[8] : ax[9]);
        float ilo_404 = ((a[8] >= a[9]) ? ax[9] : ax[8]);
        a[8] = hi_401;
        a[9] = lo_402;
        ax[8] = ihi_403;
        ax[9] = ilo_404;
        float _fmax_121 = fmaxf(a[10], a[11]);
        float hi_405 = _fmax_121;
        float _min_105 = fminf(a[10], a[11]);
        float lo_406 = _min_105;
        float ihi_407 = ((a[10] >= a[11]) ? ax[10] : ax[11]);
        float ilo_408 = ((a[10] >= a[11]) ? ax[11] : ax[10]);
        a[10] = hi_405;
        a[11] = lo_406;
        ax[10] = ihi_407;
        ax[11] = ilo_408;
        float _fmax_122 = fmaxf(a[12], a[13]);
        float hi_409 = _fmax_122;
        float _min_106 = fminf(a[12], a[13]);
        float lo_410 = _min_106;
        float ihi_411 = ((a[12] >= a[13]) ? ax[12] : ax[13]);
        float ilo_412 = ((a[12] >= a[13]) ? ax[13] : ax[12]);
        a[12] = hi_409;
        a[13] = lo_410;
        ax[12] = ihi_411;
        ax[13] = ilo_412;
        float _fmax_123 = fmaxf(a[14], a[15]);
        float hi_413 = _fmax_123;
        float _min_107 = fminf(a[14], a[15]);
        float lo_414 = _min_107;
        float ihi_415 = ((a[14] >= a[15]) ? ax[14] : ax[15]);
        float ilo_416 = ((a[14] >= a[15]) ? ax[15] : ax[14]);
        a[14] = hi_413;
        a[15] = lo_414;
        ax[14] = ihi_415;
        ax[15] = ilo_416;
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
    float kp1[1];
    float xp1[1];
    int pb = tid_1 * 32;
    pub[pb] = a[0];
    pub[pb + 16] = ax[0];
    pub[pb + 1] = a[1];
    pub[pb + 16 + 1] = ax[1];
    pub[pb + 2] = a[2];
    pub[pb + 16 + 2] = ax[2];
    pub[pb + 3] = a[3];
    pub[pb + 16 + 3] = ax[3];
    pub[pb + 4] = a[4];
    pub[pb + 16 + 4] = ax[4];
    pub[pb + 5] = a[5];
    pub[pb + 16 + 5] = ax[5];
    pub[pb + 6] = a[6];
    pub[pb + 16 + 6] = ax[6];
    pub[pb + 7] = a[7];
    pub[pb + 16 + 7] = ax[7];
    pub[pb + 8] = a[8];
    pub[pb + 16 + 8] = ax[8];
    pub[pb + 9] = a[9];
    pub[pb + 16 + 9] = ax[9];
    pub[pb + 10] = a[10];
    pub[pb + 16 + 10] = ax[10];
    pub[pb + 11] = a[11];
    pub[pb + 16 + 11] = ax[11];
    pub[pb + 12] = a[12];
    pub[pb + 16 + 12] = ax[12];
    pub[pb + 13] = a[13];
    pub[pb + 16 + 13] = ax[13];
    pub[pb + 14] = a[14];
    pub[pb + 16 + 14] = ax[14];
    pub[pb + 15] = a[15];
    pub[pb + 16 + 15] = ax[15];
    asm volatile("barrier.sync 8, 512;" ::: "memory");
    float V[8];
    float X[8];
    int s0 = (sg * 16 * 32 + cg) * 32;
    int s1 = ((sg * 16 + 8) * 32 + cg) * 32;
    float x0 = pub[s0 + ln];
    float i0 = pub[s0 + 16 + ln];
    float y0 = pub[s1 + lnr];
    float j0 = pub[s1 + 16 + lnr];
    float _fmax_124 = fmaxf(x0, y0);
    float hi0 = _fmax_124;
    float ih0 = ((x0 >= y0) ? i0 : j0);
    float cur = hi0;
    float cx = ih0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, cur, 8);
    float pv = _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, cx, 8);
    float px = _shfl_xor_1;
    float _fmax_125 = fmaxf(cur, pv);
    float hi_1 = _fmax_125;
    float _min_108 = fminf(cur, pv);
    float lo_1 = _min_108;
    float ihi_1 = ((cur >= pv) ? cx : px);
    float ilo_1 = ((cur <= pv) ? cx : px);
    cur = ((up[0] != 0) ? hi_1 : lo_1);
    cx = ((up[0] != 0) ? ihi_1 : ilo_1);
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, cur, 4);
    float pv_2 = _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, cx, 4);
    float px_3 = _shfl_xor_3;
    float _fmax_126 = fmaxf(cur, pv_2);
    float hi_4 = _fmax_126;
    float _min_109 = fminf(cur, pv_2);
    float lo_5 = _min_109;
    float ihi_6 = ((cur >= pv_2) ? cx : px_3);
    float ilo_7 = ((cur <= pv_2) ? cx : px_3);
    cur = ((up[1] != 0) ? hi_4 : lo_5);
    cx = ((up[1] != 0) ? ihi_6 : ilo_7);
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, cur, 2);
    float pv_8 = _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, cx, 2);
    float px_9 = _shfl_xor_5;
    float _fmax_127 = fmaxf(cur, pv_8);
    float hi_10 = _fmax_127;
    float _min_110 = fminf(cur, pv_8);
    float lo_11 = _min_110;
    float ihi_12 = ((cur >= pv_8) ? cx : px_9);
    float ilo_13 = ((cur <= pv_8) ? cx : px_9);
    cur = ((up[2] != 0) ? hi_10 : lo_11);
    cx = ((up[2] != 0) ? ihi_12 : ilo_13);
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, cur, 1);
    float pv_14 = _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, cx, 1);
    float px_15 = _shfl_xor_7;
    float _fmax_128 = fmaxf(cur, pv_14);
    float hi_16 = _fmax_128;
    float _min_111 = fminf(cur, pv_14);
    float lo_17 = _min_111;
    float ihi_18 = ((cur >= pv_14) ? cx : px_15);
    float ilo_19 = ((cur <= pv_14) ? cx : px_15);
    cur = ((up[3] != 0) ? hi_16 : lo_17);
    cx = ((up[3] != 0) ? ihi_18 : ilo_19);
    V[0] = cur;
    X[0] = cx;
    int s0_20 = ((sg * 16 + 1) * 32 + cg) * 32;
    int s1_21 = ((sg * 16 + 1 + 8) * 32 + cg) * 32;
    float x0_22 = pub[s0_20 + ln];
    float i0_23 = pub[s0_20 + 16 + ln];
    float y0_24 = pub[s1_21 + lnr];
    float j0_25 = pub[s1_21 + 16 + lnr];
    float _fmax_129 = fmaxf(x0_22, y0_24);
    float hi0_26 = _fmax_129;
    float ih0_27 = ((x0_22 >= y0_24) ? i0_23 : j0_25);
    float cur_28 = hi0_26;
    float cx_29 = ih0_27;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 8);
    float pv_30 = _shfl_xor_8;
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, cx_29, 8);
    float px_31 = _shfl_xor_9;
    float _fmax_130 = fmaxf(cur_28, pv_30);
    float hi_32 = _fmax_130;
    float _min_112 = fminf(cur_28, pv_30);
    float lo_33 = _min_112;
    float ihi_34 = ((cur_28 >= pv_30) ? cx_29 : px_31);
    float ilo_35 = ((cur_28 <= pv_30) ? cx_29 : px_31);
    cur_28 = ((up[0] != 0) ? hi_32 : lo_33);
    cx_29 = ((up[0] != 0) ? ihi_34 : ilo_35);
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 4);
    float pv_36 = _shfl_xor_10;
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, cx_29, 4);
    float px_37 = _shfl_xor_11;
    float _fmax_131 = fmaxf(cur_28, pv_36);
    float hi_38 = _fmax_131;
    float _min_113 = fminf(cur_28, pv_36);
    float lo_39 = _min_113;
    float ihi_40 = ((cur_28 >= pv_36) ? cx_29 : px_37);
    float ilo_41 = ((cur_28 <= pv_36) ? cx_29 : px_37);
    cur_28 = ((up[1] != 0) ? hi_38 : lo_39);
    cx_29 = ((up[1] != 0) ? ihi_40 : ilo_41);
    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 2);
    float pv_42 = _shfl_xor_12;
    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, cx_29, 2);
    float px_43 = _shfl_xor_13;
    float _fmax_132 = fmaxf(cur_28, pv_42);
    float hi_44 = _fmax_132;
    float _min_114 = fminf(cur_28, pv_42);
    float lo_45 = _min_114;
    float ihi_46 = ((cur_28 >= pv_42) ? cx_29 : px_43);
    float ilo_47 = ((cur_28 <= pv_42) ? cx_29 : px_43);
    cur_28 = ((up[2] != 0) ? hi_44 : lo_45);
    cx_29 = ((up[2] != 0) ? ihi_46 : ilo_47);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, cur_28, 1);
    float pv_48 = _shfl_xor_14;
    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, cx_29, 1);
    float px_49 = _shfl_xor_15;
    float _fmax_133 = fmaxf(cur_28, pv_48);
    float hi_50 = _fmax_133;
    float _min_115 = fminf(cur_28, pv_48);
    float lo_51 = _min_115;
    float ihi_52 = ((cur_28 >= pv_48) ? cx_29 : px_49);
    float ilo_53 = ((cur_28 <= pv_48) ? cx_29 : px_49);
    cur_28 = ((up[3] != 0) ? hi_50 : lo_51);
    cx_29 = ((up[3] != 0) ? ihi_52 : ilo_53);
    V[1] = cur_28;
    X[1] = cx_29;
    int s0_54 = ((sg * 16 + 2) * 32 + cg) * 32;
    int s1_55 = ((sg * 16 + 2 + 8) * 32 + cg) * 32;
    float x0_56 = pub[s0_54 + ln];
    float i0_57 = pub[s0_54 + 16 + ln];
    float y0_58 = pub[s1_55 + lnr];
    float j0_59 = pub[s1_55 + 16 + lnr];
    float _fmax_134 = fmaxf(x0_56, y0_58);
    float hi0_60 = _fmax_134;
    float ih0_61 = ((x0_56 >= y0_58) ? i0_57 : j0_59);
    float cur_62 = hi0_60;
    float cx_63 = ih0_61;
    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, cur_62, 8);
    float pv_64 = _shfl_xor_16;
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, cx_63, 8);
    float px_65 = _shfl_xor_17;
    float _fmax_135 = fmaxf(cur_62, pv_64);
    float hi_66 = _fmax_135;
    float _min_116 = fminf(cur_62, pv_64);
    float lo_67 = _min_116;
    float ihi_68 = ((cur_62 >= pv_64) ? cx_63 : px_65);
    float ilo_69 = ((cur_62 <= pv_64) ? cx_63 : px_65);
    cur_62 = ((up[0] != 0) ? hi_66 : lo_67);
    cx_63 = ((up[0] != 0) ? ihi_68 : ilo_69);
    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, cur_62, 4);
    float pv_70 = _shfl_xor_18;
    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, cx_63, 4);
    float px_71 = _shfl_xor_19;
    float _fmax_136 = fmaxf(cur_62, pv_70);
    float hi_72 = _fmax_136;
    float _min_117 = fminf(cur_62, pv_70);
    float lo_73 = _min_117;
    float ihi_74 = ((cur_62 >= pv_70) ? cx_63 : px_71);
    float ilo_75 = ((cur_62 <= pv_70) ? cx_63 : px_71);
    cur_62 = ((up[1] != 0) ? hi_72 : lo_73);
    cx_63 = ((up[1] != 0) ? ihi_74 : ilo_75);
    float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, cur_62, 2);
    float pv_76 = _shfl_xor_20;
    float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, cx_63, 2);
    float px_77 = _shfl_xor_21;
    float _fmax_137 = fmaxf(cur_62, pv_76);
    float hi_78 = _fmax_137;
    float _min_118 = fminf(cur_62, pv_76);
    float lo_79 = _min_118;
    float ihi_80 = ((cur_62 >= pv_76) ? cx_63 : px_77);
    float ilo_81 = ((cur_62 <= pv_76) ? cx_63 : px_77);
    cur_62 = ((up[2] != 0) ? hi_78 : lo_79);
    cx_63 = ((up[2] != 0) ? ihi_80 : ilo_81);
    float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, cur_62, 1);
    float pv_82 = _shfl_xor_22;
    float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, cx_63, 1);
    float px_83 = _shfl_xor_23;
    float _fmax_138 = fmaxf(cur_62, pv_82);
    float hi_84 = _fmax_138;
    float _min_119 = fminf(cur_62, pv_82);
    float lo_85 = _min_119;
    float ihi_86 = ((cur_62 >= pv_82) ? cx_63 : px_83);
    float ilo_87 = ((cur_62 <= pv_82) ? cx_63 : px_83);
    cur_62 = ((up[3] != 0) ? hi_84 : lo_85);
    cx_63 = ((up[3] != 0) ? ihi_86 : ilo_87);
    V[2] = cur_62;
    X[2] = cx_63;
    int s0_88 = ((sg * 16 + 3) * 32 + cg) * 32;
    int s1_89 = ((sg * 16 + 3 + 8) * 32 + cg) * 32;
    float x0_90 = pub[s0_88 + ln];
    float i0_91 = pub[s0_88 + 16 + ln];
    float y0_92 = pub[s1_89 + lnr];
    float j0_93 = pub[s1_89 + 16 + lnr];
    float _fmax_139 = fmaxf(x0_90, y0_92);
    float hi0_94 = _fmax_139;
    float ih0_95 = ((x0_90 >= y0_92) ? i0_91 : j0_93);
    float cur_96 = hi0_94;
    float cx_97 = ih0_95;
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, cur_96, 8);
    float pv_98 = _shfl_xor_24;
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, cx_97, 8);
    float px_99 = _shfl_xor_25;
    float _fmax_140 = fmaxf(cur_96, pv_98);
    float hi_100 = _fmax_140;
    float _min_120 = fminf(cur_96, pv_98);
    float lo_101 = _min_120;
    float ihi_102 = ((cur_96 >= pv_98) ? cx_97 : px_99);
    float ilo_103 = ((cur_96 <= pv_98) ? cx_97 : px_99);
    cur_96 = ((up[0] != 0) ? hi_100 : lo_101);
    cx_97 = ((up[0] != 0) ? ihi_102 : ilo_103);
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, cur_96, 4);
    float pv_104 = _shfl_xor_26;
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, cx_97, 4);
    float px_105 = _shfl_xor_27;
    float _fmax_141 = fmaxf(cur_96, pv_104);
    float hi_106 = _fmax_141;
    float _min_121 = fminf(cur_96, pv_104);
    float lo_107 = _min_121;
    float ihi_108 = ((cur_96 >= pv_104) ? cx_97 : px_105);
    float ilo_109 = ((cur_96 <= pv_104) ? cx_97 : px_105);
    cur_96 = ((up[1] != 0) ? hi_106 : lo_107);
    cx_97 = ((up[1] != 0) ? ihi_108 : ilo_109);
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, cur_96, 2);
    float pv_110 = _shfl_xor_28;
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, cx_97, 2);
    float px_111 = _shfl_xor_29;
    float _fmax_142 = fmaxf(cur_96, pv_110);
    float hi_112 = _fmax_142;
    float _min_122 = fminf(cur_96, pv_110);
    float lo_113 = _min_122;
    float ihi_114 = ((cur_96 >= pv_110) ? cx_97 : px_111);
    float ilo_115 = ((cur_96 <= pv_110) ? cx_97 : px_111);
    cur_96 = ((up[2] != 0) ? hi_112 : lo_113);
    cx_97 = ((up[2] != 0) ? ihi_114 : ilo_115);
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, cur_96, 1);
    float pv_116 = _shfl_xor_30;
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, cx_97, 1);
    float px_117 = _shfl_xor_31;
    float _fmax_143 = fmaxf(cur_96, pv_116);
    float hi_118 = _fmax_143;
    float _min_123 = fminf(cur_96, pv_116);
    float lo_119 = _min_123;
    float ihi_120 = ((cur_96 >= pv_116) ? cx_97 : px_117);
    float ilo_121 = ((cur_96 <= pv_116) ? cx_97 : px_117);
    cur_96 = ((up[3] != 0) ? hi_118 : lo_119);
    cx_97 = ((up[3] != 0) ? ihi_120 : ilo_121);
    V[3] = cur_96;
    X[3] = cx_97;
    int s0_122 = ((sg * 16 + 4) * 32 + cg) * 32;
    int s1_123 = ((sg * 16 + 4 + 8) * 32 + cg) * 32;
    float x0_124 = pub[s0_122 + ln];
    float i0_125 = pub[s0_122 + 16 + ln];
    float y0_126 = pub[s1_123 + lnr];
    float j0_127 = pub[s1_123 + 16 + lnr];
    float _fmax_144 = fmaxf(x0_124, y0_126);
    float hi0_128 = _fmax_144;
    float ih0_129 = ((x0_124 >= y0_126) ? i0_125 : j0_127);
    float cur_130 = hi0_128;
    float cx_131 = ih0_129;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 8);
    float pv_132 = _shfl_xor_32;
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, cx_131, 8);
    float px_133 = _shfl_xor_33;
    float _fmax_145 = fmaxf(cur_130, pv_132);
    float hi_134 = _fmax_145;
    float _min_124 = fminf(cur_130, pv_132);
    float lo_135 = _min_124;
    float ihi_136 = ((cur_130 >= pv_132) ? cx_131 : px_133);
    float ilo_137 = ((cur_130 <= pv_132) ? cx_131 : px_133);
    cur_130 = ((up[0] != 0) ? hi_134 : lo_135);
    cx_131 = ((up[0] != 0) ? ihi_136 : ilo_137);
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 4);
    float pv_138 = _shfl_xor_34;
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, cx_131, 4);
    float px_139 = _shfl_xor_35;
    float _fmax_146 = fmaxf(cur_130, pv_138);
    float hi_140 = _fmax_146;
    float _min_125 = fminf(cur_130, pv_138);
    float lo_141 = _min_125;
    float ihi_142 = ((cur_130 >= pv_138) ? cx_131 : px_139);
    float ilo_143 = ((cur_130 <= pv_138) ? cx_131 : px_139);
    cur_130 = ((up[1] != 0) ? hi_140 : lo_141);
    cx_131 = ((up[1] != 0) ? ihi_142 : ilo_143);
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 2);
    float pv_144 = _shfl_xor_36;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, cx_131, 2);
    float px_145 = _shfl_xor_37;
    float _fmax_147 = fmaxf(cur_130, pv_144);
    float hi_146 = _fmax_147;
    float _min_126 = fminf(cur_130, pv_144);
    float lo_147 = _min_126;
    float ihi_148 = ((cur_130 >= pv_144) ? cx_131 : px_145);
    float ilo_149 = ((cur_130 <= pv_144) ? cx_131 : px_145);
    cur_130 = ((up[2] != 0) ? hi_146 : lo_147);
    cx_131 = ((up[2] != 0) ? ihi_148 : ilo_149);
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, cur_130, 1);
    float pv_150 = _shfl_xor_38;
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, cx_131, 1);
    float px_151 = _shfl_xor_39;
    float _fmax_148 = fmaxf(cur_130, pv_150);
    float hi_152 = _fmax_148;
    float _min_127 = fminf(cur_130, pv_150);
    float lo_153 = _min_127;
    float ihi_154 = ((cur_130 >= pv_150) ? cx_131 : px_151);
    float ilo_155 = ((cur_130 <= pv_150) ? cx_131 : px_151);
    cur_130 = ((up[3] != 0) ? hi_152 : lo_153);
    cx_131 = ((up[3] != 0) ? ihi_154 : ilo_155);
    V[4] = cur_130;
    X[4] = cx_131;
    int s0_156 = ((sg * 16 + 5) * 32 + cg) * 32;
    int s1_157 = ((sg * 16 + 5 + 8) * 32 + cg) * 32;
    float x0_158 = pub[s0_156 + ln];
    float i0_159 = pub[s0_156 + 16 + ln];
    float y0_160 = pub[s1_157 + lnr];
    float j0_161 = pub[s1_157 + 16 + lnr];
    float _fmax_149 = fmaxf(x0_158, y0_160);
    float hi0_162 = _fmax_149;
    float ih0_163 = ((x0_158 >= y0_160) ? i0_159 : j0_161);
    float cur_164 = hi0_162;
    float cx_165 = ih0_163;
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, cur_164, 8);
    float pv_166 = _shfl_xor_40;
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, cx_165, 8);
    float px_167 = _shfl_xor_41;
    float _fmax_150 = fmaxf(cur_164, pv_166);
    float hi_168 = _fmax_150;
    float _min_128 = fminf(cur_164, pv_166);
    float lo_169 = _min_128;
    float ihi_170 = ((cur_164 >= pv_166) ? cx_165 : px_167);
    float ilo_171 = ((cur_164 <= pv_166) ? cx_165 : px_167);
    cur_164 = ((up[0] != 0) ? hi_168 : lo_169);
    cx_165 = ((up[0] != 0) ? ihi_170 : ilo_171);
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, cur_164, 4);
    float pv_172 = _shfl_xor_42;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, cx_165, 4);
    float px_173 = _shfl_xor_43;
    float _fmax_151 = fmaxf(cur_164, pv_172);
    float hi_174 = _fmax_151;
    float _min_129 = fminf(cur_164, pv_172);
    float lo_175 = _min_129;
    float ihi_176 = ((cur_164 >= pv_172) ? cx_165 : px_173);
    float ilo_177 = ((cur_164 <= pv_172) ? cx_165 : px_173);
    cur_164 = ((up[1] != 0) ? hi_174 : lo_175);
    cx_165 = ((up[1] != 0) ? ihi_176 : ilo_177);
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, cur_164, 2);
    float pv_178 = _shfl_xor_44;
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, cx_165, 2);
    float px_179 = _shfl_xor_45;
    float _fmax_152 = fmaxf(cur_164, pv_178);
    float hi_180 = _fmax_152;
    float _min_130 = fminf(cur_164, pv_178);
    float lo_181 = _min_130;
    float ihi_182 = ((cur_164 >= pv_178) ? cx_165 : px_179);
    float ilo_183 = ((cur_164 <= pv_178) ? cx_165 : px_179);
    cur_164 = ((up[2] != 0) ? hi_180 : lo_181);
    cx_165 = ((up[2] != 0) ? ihi_182 : ilo_183);
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, cur_164, 1);
    float pv_184 = _shfl_xor_46;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, cx_165, 1);
    float px_185 = _shfl_xor_47;
    float _fmax_153 = fmaxf(cur_164, pv_184);
    float hi_186 = _fmax_153;
    float _min_131 = fminf(cur_164, pv_184);
    float lo_187 = _min_131;
    float ihi_188 = ((cur_164 >= pv_184) ? cx_165 : px_185);
    float ilo_189 = ((cur_164 <= pv_184) ? cx_165 : px_185);
    cur_164 = ((up[3] != 0) ? hi_186 : lo_187);
    cx_165 = ((up[3] != 0) ? ihi_188 : ilo_189);
    V[5] = cur_164;
    X[5] = cx_165;
    int s0_190 = ((sg * 16 + 6) * 32 + cg) * 32;
    int s1_191 = ((sg * 16 + 6 + 8) * 32 + cg) * 32;
    float x0_192 = pub[s0_190 + ln];
    float i0_193 = pub[s0_190 + 16 + ln];
    float y0_194 = pub[s1_191 + lnr];
    float j0_195 = pub[s1_191 + 16 + lnr];
    float _fmax_154 = fmaxf(x0_192, y0_194);
    float hi0_196 = _fmax_154;
    float ih0_197 = ((x0_192 >= y0_194) ? i0_193 : j0_195);
    float cur_198 = hi0_196;
    float cx_199 = ih0_197;
    float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, cur_198, 8);
    float pv_200 = _shfl_xor_48;
    float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, cx_199, 8);
    float px_201 = _shfl_xor_49;
    float _fmax_155 = fmaxf(cur_198, pv_200);
    float hi_202 = _fmax_155;
    float _min_132 = fminf(cur_198, pv_200);
    float lo_203 = _min_132;
    float ihi_204 = ((cur_198 >= pv_200) ? cx_199 : px_201);
    float ilo_205 = ((cur_198 <= pv_200) ? cx_199 : px_201);
    cur_198 = ((up[0] != 0) ? hi_202 : lo_203);
    cx_199 = ((up[0] != 0) ? ihi_204 : ilo_205);
    float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, cur_198, 4);
    float pv_206 = _shfl_xor_50;
    float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, cx_199, 4);
    float px_207 = _shfl_xor_51;
    float _fmax_156 = fmaxf(cur_198, pv_206);
    float hi_208 = _fmax_156;
    float _min_133 = fminf(cur_198, pv_206);
    float lo_209 = _min_133;
    float ihi_210 = ((cur_198 >= pv_206) ? cx_199 : px_207);
    float ilo_211 = ((cur_198 <= pv_206) ? cx_199 : px_207);
    cur_198 = ((up[1] != 0) ? hi_208 : lo_209);
    cx_199 = ((up[1] != 0) ? ihi_210 : ilo_211);
    float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, cur_198, 2);
    float pv_212 = _shfl_xor_52;
    float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, cx_199, 2);
    float px_213 = _shfl_xor_53;
    float _fmax_157 = fmaxf(cur_198, pv_212);
    float hi_214 = _fmax_157;
    float _min_134 = fminf(cur_198, pv_212);
    float lo_215 = _min_134;
    float ihi_216 = ((cur_198 >= pv_212) ? cx_199 : px_213);
    float ilo_217 = ((cur_198 <= pv_212) ? cx_199 : px_213);
    cur_198 = ((up[2] != 0) ? hi_214 : lo_215);
    cx_199 = ((up[2] != 0) ? ihi_216 : ilo_217);
    float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, cur_198, 1);
    float pv_218 = _shfl_xor_54;
    float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, cx_199, 1);
    float px_219 = _shfl_xor_55;
    float _fmax_158 = fmaxf(cur_198, pv_218);
    float hi_220 = _fmax_158;
    float _min_135 = fminf(cur_198, pv_218);
    float lo_221 = _min_135;
    float ihi_222 = ((cur_198 >= pv_218) ? cx_199 : px_219);
    float ilo_223 = ((cur_198 <= pv_218) ? cx_199 : px_219);
    cur_198 = ((up[3] != 0) ? hi_220 : lo_221);
    cx_199 = ((up[3] != 0) ? ihi_222 : ilo_223);
    V[6] = cur_198;
    X[6] = cx_199;
    int s0_224 = ((sg * 16 + 7) * 32 + cg) * 32;
    int s1_225 = ((sg * 16 + 7 + 8) * 32 + cg) * 32;
    float x0_226 = pub[s0_224 + ln];
    float i0_227 = pub[s0_224 + 16 + ln];
    float y0_228 = pub[s1_225 + lnr];
    float j0_229 = pub[s1_225 + 16 + lnr];
    float _fmax_159 = fmaxf(x0_226, y0_228);
    float hi0_230 = _fmax_159;
    float ih0_231 = ((x0_226 >= y0_228) ? i0_227 : j0_229);
    float cur_232 = hi0_230;
    float cx_233 = ih0_231;
    float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, cur_232, 8);
    float pv_234 = _shfl_xor_56;
    float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, cx_233, 8);
    float px_235 = _shfl_xor_57;
    float _fmax_160 = fmaxf(cur_232, pv_234);
    float hi_236 = _fmax_160;
    float _min_136 = fminf(cur_232, pv_234);
    float lo_237 = _min_136;
    float ihi_238 = ((cur_232 >= pv_234) ? cx_233 : px_235);
    float ilo_239 = ((cur_232 <= pv_234) ? cx_233 : px_235);
    cur_232 = ((up[0] != 0) ? hi_236 : lo_237);
    cx_233 = ((up[0] != 0) ? ihi_238 : ilo_239);
    float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, cur_232, 4);
    float pv_240 = _shfl_xor_58;
    float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, cx_233, 4);
    float px_241 = _shfl_xor_59;
    float _fmax_161 = fmaxf(cur_232, pv_240);
    float hi_242 = _fmax_161;
    float _min_137 = fminf(cur_232, pv_240);
    float lo_243 = _min_137;
    float ihi_244 = ((cur_232 >= pv_240) ? cx_233 : px_241);
    float ilo_245 = ((cur_232 <= pv_240) ? cx_233 : px_241);
    cur_232 = ((up[1] != 0) ? hi_242 : lo_243);
    cx_233 = ((up[1] != 0) ? ihi_244 : ilo_245);
    float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, cur_232, 2);
    float pv_246 = _shfl_xor_60;
    float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, cx_233, 2);
    float px_247 = _shfl_xor_61;
    float _fmax_162 = fmaxf(cur_232, pv_246);
    float hi_248 = _fmax_162;
    float _min_138 = fminf(cur_232, pv_246);
    float lo_249 = _min_138;
    float ihi_250 = ((cur_232 >= pv_246) ? cx_233 : px_247);
    float ilo_251 = ((cur_232 <= pv_246) ? cx_233 : px_247);
    cur_232 = ((up[2] != 0) ? hi_248 : lo_249);
    cx_233 = ((up[2] != 0) ? ihi_250 : ilo_251);
    float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, cur_232, 1);
    float pv_252 = _shfl_xor_62;
    float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, cx_233, 1);
    float px_253 = _shfl_xor_63;
    float _fmax_163 = fmaxf(cur_232, pv_252);
    float hi_254 = _fmax_163;
    float _min_139 = fminf(cur_232, pv_252);
    float lo_255 = _min_139;
    float ihi_256 = ((cur_232 >= pv_252) ? cx_233 : px_253);
    float ilo_257 = ((cur_232 <= pv_252) ? cx_233 : px_253);
    cur_232 = ((up[3] != 0) ? hi_254 : lo_255);
    cx_233 = ((up[3] != 0) ? ihi_256 : ilo_257);
    V[7] = cur_232;
    X[7] = cx_233;
    float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, V[4], 15);
    float y1 = _shfl_xor_64;
    float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, X[4], 15);
    float j1 = _shfl_xor_65;
    float _fmax_164 = fmaxf(V[0], y1);
    float hi1 = _fmax_164;
    float ih1 = ((V[0] >= y1) ? X[0] : j1);
    float cur_258 = hi1;
    float cx_259 = ih1;
    float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 8);
    float pv_260 = _shfl_xor_66;
    float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, cx_259, 8);
    float px_261 = _shfl_xor_67;
    float _fmax_165 = fmaxf(cur_258, pv_260);
    float hi_262 = _fmax_165;
    float _min_140 = fminf(cur_258, pv_260);
    float lo_263 = _min_140;
    float ihi_264_1 = ((cur_258 >= pv_260) ? cx_259 : px_261);
    float ilo_265 = ((cur_258 <= pv_260) ? cx_259 : px_261);
    cur_258 = ((up[0] != 0) ? hi_262 : lo_263);
    cx_259 = ((up[0] != 0) ? ihi_264_1 : ilo_265);
    float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 4);
    float pv_266 = _shfl_xor_68;
    float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, cx_259, 4);
    float px_267 = _shfl_xor_69;
    float _fmax_166 = fmaxf(cur_258, pv_266);
    float hi_268 = _fmax_166;
    float _min_141 = fminf(cur_258, pv_266);
    float lo_269 = _min_141;
    float ihi_270_1 = ((cur_258 >= pv_266) ? cx_259 : px_267);
    float ilo_271 = ((cur_258 <= pv_266) ? cx_259 : px_267);
    cur_258 = ((up[1] != 0) ? hi_268 : lo_269);
    cx_259 = ((up[1] != 0) ? ihi_270_1 : ilo_271);
    float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 2);
    float pv_272 = _shfl_xor_70;
    float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, cx_259, 2);
    float px_273 = _shfl_xor_71;
    float _fmax_167 = fmaxf(cur_258, pv_272);
    float hi_274 = _fmax_167;
    float _min_142 = fminf(cur_258, pv_272);
    float lo_275 = _min_142;
    float ihi_276_1 = ((cur_258 >= pv_272) ? cx_259 : px_273);
    float ilo_277 = ((cur_258 <= pv_272) ? cx_259 : px_273);
    cur_258 = ((up[2] != 0) ? hi_274 : lo_275);
    cx_259 = ((up[2] != 0) ? ihi_276_1 : ilo_277);
    float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, cur_258, 1);
    float pv_278 = _shfl_xor_72;
    float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, cx_259, 1);
    float px_279 = _shfl_xor_73;
    float _fmax_168 = fmaxf(cur_258, pv_278);
    float hi_280 = _fmax_168;
    float _min_143 = fminf(cur_258, pv_278);
    float lo_281 = _min_143;
    float ihi_282_1 = ((cur_258 >= pv_278) ? cx_259 : px_279);
    float ilo_283 = ((cur_258 <= pv_278) ? cx_259 : px_279);
    cur_258 = ((up[3] != 0) ? hi_280 : lo_281);
    cx_259 = ((up[3] != 0) ? ihi_282_1 : ilo_283);
    V[0] = cur_258;
    X[0] = cx_259;
    float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, V[5], 15);
    float y1_284 = _shfl_xor_74;
    float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, X[5], 15);
    float j1_285 = _shfl_xor_75;
    float _fmax_169 = fmaxf(V[1], y1_284);
    float hi1_286 = _fmax_169;
    float ih1_287 = ((V[1] >= y1_284) ? X[1] : j1_285);
    float cur_288 = hi1_286;
    float cx_289 = ih1_287;
    float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, cur_288, 8);
    float pv_290 = _shfl_xor_76;
    float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, cx_289, 8);
    float px_291 = _shfl_xor_77;
    float _fmax_170 = fmaxf(cur_288, pv_290);
    float hi_292 = _fmax_170;
    float _min_144 = fminf(cur_288, pv_290);
    float lo_293 = _min_144;
    float ihi_294 = ((cur_288 >= pv_290) ? cx_289 : px_291);
    float ilo_295 = ((cur_288 <= pv_290) ? cx_289 : px_291);
    cur_288 = ((up[0] != 0) ? hi_292 : lo_293);
    cx_289 = ((up[0] != 0) ? ihi_294 : ilo_295);
    float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, cur_288, 4);
    float pv_296 = _shfl_xor_78;
    float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, cx_289, 4);
    float px_297 = _shfl_xor_79;
    float _fmax_171 = fmaxf(cur_288, pv_296);
    float hi_298 = _fmax_171;
    float _min_145 = fminf(cur_288, pv_296);
    float lo_299 = _min_145;
    float ihi_300 = ((cur_288 >= pv_296) ? cx_289 : px_297);
    float ilo_301 = ((cur_288 <= pv_296) ? cx_289 : px_297);
    cur_288 = ((up[1] != 0) ? hi_298 : lo_299);
    cx_289 = ((up[1] != 0) ? ihi_300 : ilo_301);
    float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, cur_288, 2);
    float pv_302 = _shfl_xor_80;
    float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, cx_289, 2);
    float px_303 = _shfl_xor_81;
    float _fmax_172 = fmaxf(cur_288, pv_302);
    float hi_304 = _fmax_172;
    float _min_146 = fminf(cur_288, pv_302);
    float lo_305 = _min_146;
    float ihi_306 = ((cur_288 >= pv_302) ? cx_289 : px_303);
    float ilo_307 = ((cur_288 <= pv_302) ? cx_289 : px_303);
    cur_288 = ((up[2] != 0) ? hi_304 : lo_305);
    cx_289 = ((up[2] != 0) ? ihi_306 : ilo_307);
    float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, cur_288, 1);
    float pv_308 = _shfl_xor_82;
    float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, cx_289, 1);
    float px_309 = _shfl_xor_83;
    float _fmax_173 = fmaxf(cur_288, pv_308);
    float hi_310 = _fmax_173;
    float _min_147 = fminf(cur_288, pv_308);
    float lo_311 = _min_147;
    float ihi_312 = ((cur_288 >= pv_308) ? cx_289 : px_309);
    float ilo_313 = ((cur_288 <= pv_308) ? cx_289 : px_309);
    cur_288 = ((up[3] != 0) ? hi_310 : lo_311);
    cx_289 = ((up[3] != 0) ? ihi_312 : ilo_313);
    V[1] = cur_288;
    X[1] = cx_289;
    float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, V[6], 15);
    float y1_314 = _shfl_xor_84;
    float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, X[6], 15);
    float j1_315 = _shfl_xor_85;
    float _fmax_174 = fmaxf(V[2], y1_314);
    float hi1_316 = _fmax_174;
    float ih1_317 = ((V[2] >= y1_314) ? X[2] : j1_315);
    float cur_318 = hi1_316;
    float cx_319 = ih1_317;
    float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, cur_318, 8);
    float pv_320 = _shfl_xor_86;
    float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, cx_319, 8);
    float px_321 = _shfl_xor_87;
    float _fmax_175 = fmaxf(cur_318, pv_320);
    float hi_322 = _fmax_175;
    float _min_148 = fminf(cur_318, pv_320);
    float lo_323 = _min_148;
    float ihi_324 = ((cur_318 >= pv_320) ? cx_319 : px_321);
    float ilo_325 = ((cur_318 <= pv_320) ? cx_319 : px_321);
    cur_318 = ((up[0] != 0) ? hi_322 : lo_323);
    cx_319 = ((up[0] != 0) ? ihi_324 : ilo_325);
    float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, cur_318, 4);
    float pv_326 = _shfl_xor_88;
    float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, cx_319, 4);
    float px_327 = _shfl_xor_89;
    float _fmax_176 = fmaxf(cur_318, pv_326);
    float hi_328 = _fmax_176;
    float _min_149 = fminf(cur_318, pv_326);
    float lo_329 = _min_149;
    float ihi_330 = ((cur_318 >= pv_326) ? cx_319 : px_327);
    float ilo_331 = ((cur_318 <= pv_326) ? cx_319 : px_327);
    cur_318 = ((up[1] != 0) ? hi_328 : lo_329);
    cx_319 = ((up[1] != 0) ? ihi_330 : ilo_331);
    float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, cur_318, 2);
    float pv_332 = _shfl_xor_90;
    float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, cx_319, 2);
    float px_333 = _shfl_xor_91;
    float _fmax_177 = fmaxf(cur_318, pv_332);
    float hi_334 = _fmax_177;
    float _min_150 = fminf(cur_318, pv_332);
    float lo_335 = _min_150;
    float ihi_336 = ((cur_318 >= pv_332) ? cx_319 : px_333);
    float ilo_337 = ((cur_318 <= pv_332) ? cx_319 : px_333);
    cur_318 = ((up[2] != 0) ? hi_334 : lo_335);
    cx_319 = ((up[2] != 0) ? ihi_336 : ilo_337);
    float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, cur_318, 1);
    float pv_338 = _shfl_xor_92;
    float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, cx_319, 1);
    float px_339 = _shfl_xor_93;
    float _fmax_178 = fmaxf(cur_318, pv_338);
    float hi_340 = _fmax_178;
    float _min_151 = fminf(cur_318, pv_338);
    float lo_341 = _min_151;
    float ihi_342 = ((cur_318 >= pv_338) ? cx_319 : px_339);
    float ilo_343 = ((cur_318 <= pv_338) ? cx_319 : px_339);
    cur_318 = ((up[3] != 0) ? hi_340 : lo_341);
    cx_319 = ((up[3] != 0) ? ihi_342 : ilo_343);
    V[2] = cur_318;
    X[2] = cx_319;
    float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, V[7], 15);
    float y1_344 = _shfl_xor_94;
    float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, X[7], 15);
    float j1_345 = _shfl_xor_95;
    float _fmax_179 = fmaxf(V[3], y1_344);
    float hi1_346 = _fmax_179;
    float ih1_347 = ((V[3] >= y1_344) ? X[3] : j1_345);
    float cur_348 = hi1_346;
    float cx_349 = ih1_347;
    float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, cur_348, 8);
    float pv_350 = _shfl_xor_96;
    float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, cx_349, 8);
    float px_351 = _shfl_xor_97;
    float _fmax_180 = fmaxf(cur_348, pv_350);
    float hi_352 = _fmax_180;
    float _min_152 = fminf(cur_348, pv_350);
    float lo_353 = _min_152;
    float ihi_354 = ((cur_348 >= pv_350) ? cx_349 : px_351);
    float ilo_355 = ((cur_348 <= pv_350) ? cx_349 : px_351);
    cur_348 = ((up[0] != 0) ? hi_352 : lo_353);
    cx_349 = ((up[0] != 0) ? ihi_354 : ilo_355);
    float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, cur_348, 4);
    float pv_356 = _shfl_xor_98;
    float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, cx_349, 4);
    float px_357 = _shfl_xor_99;
    float _fmax_181 = fmaxf(cur_348, pv_356);
    float hi_358 = _fmax_181;
    float _min_153 = fminf(cur_348, pv_356);
    float lo_359 = _min_153;
    float ihi_360 = ((cur_348 >= pv_356) ? cx_349 : px_357);
    float ilo_361 = ((cur_348 <= pv_356) ? cx_349 : px_357);
    cur_348 = ((up[1] != 0) ? hi_358 : lo_359);
    cx_349 = ((up[1] != 0) ? ihi_360 : ilo_361);
    float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, cur_348, 2);
    float pv_362 = _shfl_xor_100;
    float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, cx_349, 2);
    float px_363 = _shfl_xor_101;
    float _fmax_182 = fmaxf(cur_348, pv_362);
    float hi_364 = _fmax_182;
    float _min_154 = fminf(cur_348, pv_362);
    float lo_365 = _min_154;
    float ihi_366 = ((cur_348 >= pv_362) ? cx_349 : px_363);
    float ilo_367 = ((cur_348 <= pv_362) ? cx_349 : px_363);
    cur_348 = ((up[2] != 0) ? hi_364 : lo_365);
    cx_349 = ((up[2] != 0) ? ihi_366 : ilo_367);
    float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, cur_348, 1);
    float pv_368 = _shfl_xor_102;
    float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, cx_349, 1);
    float px_369 = _shfl_xor_103;
    float _fmax_183 = fmaxf(cur_348, pv_368);
    float hi_370 = _fmax_183;
    float _min_155 = fminf(cur_348, pv_368);
    float lo_371 = _min_155;
    float ihi_372 = ((cur_348 >= pv_368) ? cx_349 : px_369);
    float ilo_373 = ((cur_348 <= pv_368) ? cx_349 : px_369);
    cur_348 = ((up[3] != 0) ? hi_370 : lo_371);
    cx_349 = ((up[3] != 0) ? ihi_372 : ilo_373);
    V[3] = cur_348;
    X[3] = cx_349;
    float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, V[2], 15);
    float y1_374 = _shfl_xor_104;
    float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, X[2], 15);
    float j1_375 = _shfl_xor_105;
    float _fmax_184 = fmaxf(V[0], y1_374);
    float hi1_376 = _fmax_184;
    float ih1_377 = ((V[0] >= y1_374) ? X[0] : j1_375);
    float cur_378 = hi1_376;
    float cx_379 = ih1_377;
    float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, cur_378, 8);
    float pv_380 = _shfl_xor_106;
    float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, cx_379, 8);
    float px_381 = _shfl_xor_107;
    float _fmax_185 = fmaxf(cur_378, pv_380);
    float hi_382 = _fmax_185;
    float _min_156 = fminf(cur_378, pv_380);
    float lo_383 = _min_156;
    float ihi_384 = ((cur_378 >= pv_380) ? cx_379 : px_381);
    float ilo_385 = ((cur_378 <= pv_380) ? cx_379 : px_381);
    cur_378 = ((up[0] != 0) ? hi_382 : lo_383);
    cx_379 = ((up[0] != 0) ? ihi_384 : ilo_385);
    float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, cur_378, 4);
    float pv_386 = _shfl_xor_108;
    float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, cx_379, 4);
    float px_387 = _shfl_xor_109;
    float _fmax_186 = fmaxf(cur_378, pv_386);
    float hi_388 = _fmax_186;
    float _min_157 = fminf(cur_378, pv_386);
    float lo_389 = _min_157;
    float ihi_390 = ((cur_378 >= pv_386) ? cx_379 : px_387);
    float ilo_391 = ((cur_378 <= pv_386) ? cx_379 : px_387);
    cur_378 = ((up[1] != 0) ? hi_388 : lo_389);
    cx_379 = ((up[1] != 0) ? ihi_390 : ilo_391);
    float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, cur_378, 2);
    float pv_392 = _shfl_xor_110;
    float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, cx_379, 2);
    float px_393 = _shfl_xor_111;
    float _fmax_187 = fmaxf(cur_378, pv_392);
    float hi_394 = _fmax_187;
    float _min_158 = fminf(cur_378, pv_392);
    float lo_395 = _min_158;
    float ihi_396 = ((cur_378 >= pv_392) ? cx_379 : px_393);
    float ilo_397 = ((cur_378 <= pv_392) ? cx_379 : px_393);
    cur_378 = ((up[2] != 0) ? hi_394 : lo_395);
    cx_379 = ((up[2] != 0) ? ihi_396 : ilo_397);
    float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, cur_378, 1);
    float pv_398 = _shfl_xor_112;
    float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, cx_379, 1);
    float px_399 = _shfl_xor_113;
    float _fmax_188 = fmaxf(cur_378, pv_398);
    float hi_400 = _fmax_188;
    float _min_159 = fminf(cur_378, pv_398);
    float lo_401 = _min_159;
    float ihi_402 = ((cur_378 >= pv_398) ? cx_379 : px_399);
    float ilo_403 = ((cur_378 <= pv_398) ? cx_379 : px_399);
    cur_378 = ((up[3] != 0) ? hi_400 : lo_401);
    cx_379 = ((up[3] != 0) ? ihi_402 : ilo_403);
    V[0] = cur_378;
    X[0] = cx_379;
    float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, V[3], 15);
    float y1_404 = _shfl_xor_114;
    float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, X[3], 15);
    float j1_405 = _shfl_xor_115;
    float _fmax_189 = fmaxf(V[1], y1_404);
    float hi1_406 = _fmax_189;
    float ih1_407 = ((V[1] >= y1_404) ? X[1] : j1_405);
    float cur_408 = hi1_406;
    float cx_409 = ih1_407;
    float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, cur_408, 8);
    float pv_410 = _shfl_xor_116;
    float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, cx_409, 8);
    float px_411 = _shfl_xor_117;
    float _fmax_190 = fmaxf(cur_408, pv_410);
    float hi_412 = _fmax_190;
    float _min_160 = fminf(cur_408, pv_410);
    float lo_413 = _min_160;
    float ihi_414 = ((cur_408 >= pv_410) ? cx_409 : px_411);
    float ilo_415 = ((cur_408 <= pv_410) ? cx_409 : px_411);
    cur_408 = ((up[0] != 0) ? hi_412 : lo_413);
    cx_409 = ((up[0] != 0) ? ihi_414 : ilo_415);
    float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, cur_408, 4);
    float pv_416 = _shfl_xor_118;
    float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, cx_409, 4);
    float px_417 = _shfl_xor_119;
    float _fmax_191 = fmaxf(cur_408, pv_416);
    float hi_418 = _fmax_191;
    float _min_161 = fminf(cur_408, pv_416);
    float lo_419 = _min_161;
    float ihi_420 = ((cur_408 >= pv_416) ? cx_409 : px_417);
    float ilo_421 = ((cur_408 <= pv_416) ? cx_409 : px_417);
    cur_408 = ((up[1] != 0) ? hi_418 : lo_419);
    cx_409 = ((up[1] != 0) ? ihi_420 : ilo_421);
    float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, cur_408, 2);
    float pv_422 = _shfl_xor_120;
    float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, cx_409, 2);
    float px_423 = _shfl_xor_121;
    float _fmax_192 = fmaxf(cur_408, pv_422);
    float hi_424 = _fmax_192;
    float _min_162 = fminf(cur_408, pv_422);
    float lo_425 = _min_162;
    float ihi_426 = ((cur_408 >= pv_422) ? cx_409 : px_423);
    float ilo_427 = ((cur_408 <= pv_422) ? cx_409 : px_423);
    cur_408 = ((up[2] != 0) ? hi_424 : lo_425);
    cx_409 = ((up[2] != 0) ? ihi_426 : ilo_427);
    float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, cur_408, 1);
    float pv_428 = _shfl_xor_122;
    float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, cx_409, 1);
    float px_429 = _shfl_xor_123;
    float _fmax_193 = fmaxf(cur_408, pv_428);
    float hi_430 = _fmax_193;
    float _min_163 = fminf(cur_408, pv_428);
    float lo_431 = _min_163;
    float ihi_432 = ((cur_408 >= pv_428) ? cx_409 : px_429);
    float ilo_433 = ((cur_408 <= pv_428) ? cx_409 : px_429);
    cur_408 = ((up[3] != 0) ? hi_430 : lo_431);
    cx_409 = ((up[3] != 0) ? ihi_432 : ilo_433);
    V[1] = cur_408;
    X[1] = cx_409;
    float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, V[1], 15);
    float yl = _shfl_xor_124;
    float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, X[1], 15);
    float jl = _shfl_xor_125;
    float _fmax_194 = fmaxf(V[0], yl);
    float hil = _fmax_194;
    float ihl = ((V[0] >= yl) ? X[0] : jl);
    V[0] = hil;
    X[0] = ihl;
    float K = V[0];
    float KX = X[0];
    kp1[0] = K;
    xp1[0] = KX;
    int ucolp = blockIdx.x * 32 + cg;
    if (tid_1 < 512) {
        unsigned int kkp = __as_u32(kp1[0]);
        unsigned int idxp = 2048;
        if (kkp < 4278190080u) {
            idxp = __as_u32(xp1[0]);
        }
        unsigned int rkeyp = idxp << 4 | (unsigned int)ln;
        int rankp = 0;
        unsigned int _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 1);
        unsigned int oxp = _shfl_xor_126;
        if (oxp < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 2);
        unsigned int oxp_0 = _shfl_xor_127;
        if (oxp_0 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 3);
        unsigned int oxp_1 = _shfl_xor_128;
        if (oxp_1 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 4);
        unsigned int oxp_2 = _shfl_xor_129;
        if (oxp_2 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 5);
        unsigned int oxp_3 = _shfl_xor_130;
        if (oxp_3 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 6);
        unsigned int oxp_4 = _shfl_xor_131;
        if (oxp_4 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 7);
        unsigned int oxp_5 = _shfl_xor_132;
        if (oxp_5 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 8);
        unsigned int oxp_6 = _shfl_xor_133;
        if (oxp_6 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 9);
        unsigned int oxp_7 = _shfl_xor_134;
        if (oxp_7 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 10);
        unsigned int oxp_8 = _shfl_xor_135;
        if (oxp_8 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 11);
        unsigned int oxp_9 = _shfl_xor_136;
        if (oxp_9 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 12);
        unsigned int oxp_10 = _shfl_xor_137;
        if (oxp_10 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 13);
        unsigned int oxp_11 = _shfl_xor_138;
        if (oxp_11 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 14);
        unsigned int oxp_12 = _shfl_xor_139;
        if (oxp_12 < rkeyp) {
            rankp = rankp + 1;
        }
        unsigned int _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, rkeyp, 15);
        unsigned int oxp_13 = _shfl_xor_140;
        if (oxp_13 < rkeyp) {
            rankp = rankp + 1;
        }
        int valp = -1;
        if (idxp <= 2047) {
            valp = (int)idxp;
        }
        if (g < 32 && ucolp < total_q) {
            long long obasep = ((long long)ucolp * (long long)num_heads + (long long)head) * 16;
            out[obasep + (long long)rankp] = valp;
        }
    }
}

} // extern "C"
