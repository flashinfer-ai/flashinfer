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
#define SMEM_SSTAGE_U_OFF 0
#define SMEM_SSTAGE_U_STAGE_BYTES 2048
#define SMEM_SSTAGE_U_STRIDE 2048
#define SMEM_SSTAGE_F_OFF 0
#define SMEM_SSTAGE_F_STAGE_BYTES 2048
#define SMEM_SSTAGE_F_STRIDE 2048
#define SMEM_SPARTS_OFF 2048
#define SMEM_SPARTS_STAGE_BYTES 2048
#define SMEM_SPARTS_STRIDE 2048
#define SMEM_SCR_OFF 4096
#define SMEM_SCR_STAGE_BYTES 128
#define SMEM_SCR_STRIDE 128
#define SMEM_TOTAL 4224
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
kernel_cake_mhc_train_bwd_reduce__s27_wcvt_spred(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_chunks, int accumulate)
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
    unsigned int* sstage_u = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int sstage_u_addr = smem + 0;
    float* sstage_f = reinterpret_cast<float*>(smem_raw + 0);
    const int sstage_f_addr = smem + 0;
    double* sparts = reinterpret_cast<double*>(smem_raw + 2048);
    const int sparts_addr = smem + 2048;
    int* scr = reinterpret_cast<int*>(smem_raw + 4096);
    const int scr_addr = smem + 4096;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int b = bid;
    int tid_0 = tid;
    if (b < 384) {
        long long e0 = (long long)(b * 256 + tid_0) * 4;
        float out[4];
        double a[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            a[k] = 0.0;
        }
        float _vec_load_0[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + e0 + 0);
            _vec_load_0[0 + 0] = _v4.x;
            _vec_load_0[0 + 1] = _v4.y;
            _vec_load_0[0 + 2] = _v4.z;
            _vec_load_0[0 + 3] = _v4.w;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            a[k_1] = ((_vec_load_0[k_1] != 0.0f) ? (double)_vec_load_0[k_1] : 0.0);
        }
        #pragma unroll 2
        for (int p = 1; p < n_parts; p++) {
            float vec_load[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 393216 + e0) + 0);
                vec_load[0 + 0] = _v4.x;
                vec_load[0 + 1] = _v4.y;
                vec_load[0 + 2] = _v4.z;
                vec_load[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                a[k_2] = a[k_2] + (double)vec_load[k_2];
            }
        }
        if (accumulate != 0) {
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn + e0 + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_3 = 0; k_3 < 4; k_3++) {
                a[k_3] = a[k_3] + (double)_vec_load_2[k_3];
            }
        }
        #pragma unroll
        for (int k_4 = 0; k_4 < 4; k_4++) {
            out[k_4] = (float)a[k_4];
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        int col = b - 384;
        long long col64 = (long long)col;
        long long scol64 = col64 - 24;
        double tot[1];
        float res[1];
        tot[0] = 0.0;
        int n_tiles = (n_chunks + 511) / 512;
        int lane_1 = tid_0 & 31;
        int wrp = tid_0 >> 5;
        #pragma unroll 1
        for (int ti = 0; ti < n_tiles; ti++) {
            int c0 = ti * 512;
            int rem = n_chunks - c0;
            int rows = ((rem < 512) ? rem : 512);
            int last = rows - 1;
            long long c064 = (long long)c0;
            int emax = -4096;
            int negl = -4096;
            int bad = 0;
            double part = 0.0;
            #pragma unroll
            for (int k_5 = 0; k_5 < 2; k_5++) {
                int row = k_5 * 256 + tid_0;
                int inb = (int)(row < rows);
                unsigned int cbr[1];
                cbr[0] = 0;
                if (inb != 0) {
                    long long srcp = (c064 + (long long)row) * 32 + col64;
                    float xp = param_partial[srcp];
                    cbr[0] = __as_u32(xp);
                    sstage_u[row] = cbr[0];
                }
                unsigned int cb = cbr[0];
                int ce = (int)(cb >> 23 & 255);
                int cvalid = (int)((cb & 2147483647) != 0);
                unsigned int csig = cb & 8388607 | ((ce != 0) ? 8388608 : 0);
                int _find_first_set_3 = __ffs(csig);
                int cl = ((ce != 0) ? ce : 1) - 150 + (_find_first_set_3 - 1);
                int _max_4 = ((emax) > (((cvalid != 0) ? ce - 127 : -4096)) ? (emax) : (((cvalid != 0) ? ce - 127 : -4096)));
                emax = _max_4;
                int _max_5 = ((negl) > (((cvalid != 0) ? -cl : -4096)) ? (negl) : (((cvalid != 0) ? -cl : -4096)));
                negl = _max_5;
                int _max_6 = ((bad) > ((int)(cvalid != 0 && ce == 255)) ? (bad) : ((int)(cvalid != 0 && ce == 255)));
                bad = _max_6;
                part = part + (double)__uint_as_float(cb);
            }
            int _warp_redux_7;
            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_7) : "r"(emax));
            emax = _warp_redux_7;
            int _warp_redux_8;
            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_8) : "r"(negl));
            negl = _warp_redux_8;
            int _warp_redux_9;
            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(bad));
            bad = _warp_redux_9;
            if (lane_1 == 0) {
                scr[wrp * 4] = emax;
                scr[wrp * 4 + 1] = negl;
                scr[wrp * 4 + 2] = bad;
            }
            sparts[tid_0] = part;
            __syncthreads();
            #pragma unroll
            for (int lv = 0; lv < 8; lv++) {
                int half = 256 >> lv + 1;
                if (tid_0 < half) {
                    sparts[tid_0] = sparts[tid_0] + sparts[tid_0 + half];
                }
                __syncthreads();
            }
            int gmax = -4096;
            int gneg = -4096;
            int gbad = 0;
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                int _max_10 = ((gmax) > (scr[w * 4]) ? (gmax) : (scr[w * 4]));
                gmax = _max_10;
                int _max_11 = ((gneg) > (scr[w * 4 + 1]) ? (gneg) : (scr[w * 4 + 1]));
                gneg = _max_11;
                int _max_12 = ((gbad) > (scr[w * 4 + 2]) ? (gbad) : (scr[w * 4 + 2]));
                gbad = _max_12;
            }
            int ok = (int)(gbad == 0 && gmax + gneg <= 43 && n_tiles == 1);
            if (tid_0 == 0) {
                if (ok != 0) {
                    tot[0] = tot[0] + sparts[0];
                } else {
                    #pragma unroll 8
                    for (int rloc = 0; rloc < rows; rloc++) {
                        float xf = sstage_f[rloc];
                        tot[0] = tot[0] + (double)xf;
                    }
                }
            }
            __syncthreads();
        }
        if (tid_0 == 0) {
            if (accumulate != 0) {
                if (col < 24) {
                    tot[0] = tot[0] + (double)dbase[col64];
                } else {
                    tot[0] = tot[0] + (double)dscale[scol64];
                }
            }
            res[0] = (float)tot[0];
            if (col < 24) {
                *(reinterpret_cast<float*>(dbase + col64) + (0)) = res[0];
            } else {
                *(reinterpret_cast<float*>(dscale + scol64) + (0)) = res[0];
            }
        }
    }
}

} // extern "C"
