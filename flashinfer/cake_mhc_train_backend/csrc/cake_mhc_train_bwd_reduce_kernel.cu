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
#define SMEM_STAGE_OFF 0
#define SMEM_STAGE_STAGE_BYTES 65536
#define SMEM_STAGE_STRIDE 65536
#define SMEM_TOTAL 65536
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_train_bwd_reduce(float* __restrict__ dfn_part, float* __restrict__ param_partial, float* __restrict__ dfn, float* __restrict__ dscale, float* __restrict__ dbase, int n_parts, int n_chunks, int accumulate)
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
    float* stage = reinterpret_cast<float*>(smem_raw + 0);
    const int stage_addr = smem + 0;

    // === Task calls (dependency order) ===
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
        #pragma unroll 8
        for (int p = 0; p < n_parts; p++) {
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn_part + ((long long)p * 393216 + e0) + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_1 = 0; k_1 < 4; k_1++) {
                a[k_1] = a[k_1] + (double)_vec_load_0[k_1];
            }
        }
        if (accumulate != 0) {
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(dfn + e0 + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 4; k_2++) {
                a[k_2] = a[k_2] + (double)_vec_load_1[k_2];
            }
        }
        #pragma unroll
        for (int k_3 = 0; k_3 < 4; k_3++) {
            out[k_3] = (float)a[k_3];
        }
        {
            float4 _v4 = make_float4(out[0 + 0], out[0 + 1], out[0 + 2], out[0 + 3]);
            *reinterpret_cast<float4*>((dfn + e0) + 0) = _v4;
        }
    } else {
        long long tid64 = (long long)tid_0;
        long long scol = (long long)(tid_0 - 24);
        double tot[1];
        float ta[1];
        float tc[1];
        float res[1];
        tot[0] = 0.0;
        ta[0] = 0.0f;
        tc[0] = 0.0f;
        int n_tiles = (n_chunks + 511) / 512;
        #pragma unroll 1
        for (int ti = 0; ti < n_tiles; ti++) {
            int c0 = ti * 512;
            int rem = n_chunks - c0;
            int rows = ((rem < 512) ? rem : 512);
            int last = rows - 1;
            long long c064 = (long long)c0;
            #pragma unroll
            for (int k_4 = 0; k_4 < 16; k_4++) {
                int q = k_4 * 256 + tid_0;
                int row = q >> 3;
                int rowc = ((row < rows) ? row : last);
                long long src = (c064 + (long long)rowc) * 32 + (long long)((q & 7) * 4);
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(param_partial + src + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    stage[q * 4 + j] = _vec_load_2[j];
                }
            }
            __syncthreads();
            if (tid_0 < 27) {
                #pragma unroll 8
                for (int rloc = 0; rloc < rows; rloc++) {
                    float x = stage[rloc * 32 + tid_0];
                    tot[0] = tot[0] + (double)x;
                }
            }
            __syncthreads();
        }
        if (tid_0 < 27) {
            if (accumulate != 0) {
                if (tid_0 < 24) {
                    tot[0] = tot[0] + (double)dbase[tid64];
                } else {
                    tot[0] = tot[0] + (double)dscale[scol];
                }
            }
            res[0] = (float)tot[0];
            if (tid_0 < 24) {
                *(reinterpret_cast<float*>(dbase + tid64) + (0)) = res[0];
            } else {
                *(reinterpret_cast<float*>(dscale + scol) + (0)) = res[0];
            }
        }
    }
}

} // extern "C"
