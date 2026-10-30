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
#define SMEM_RED_FOLD_OFF 0
#define SMEM_RED_FOLD_STAGE_BYTES 4096
#define SMEM_RED_FOLD_STRIDE 4096
#define SMEM_GRP_FOLD_OFF 4096
#define SMEM_GRP_FOLD_STAGE_BYTES 512
#define SMEM_GRP_FOLD_STRIDE 512
#define SMEM_TOTAL 4608
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_rmsnorm_train_520453292492bd2725be(float* __restrict__ partial, float* __restrict__ dw, int n_chunks)
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
    float* red_fold = reinterpret_cast<float*>(smem_raw + 0);
    const int red_fold_addr = smem + 0;
    float* grp_fold = reinterpret_cast<float*>(smem_raw + 4096);
    const int grp_fold_addr = smem + 4096;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int s = bid;
    int rsub = tid / 8;
    int cg = tid - rsub * 8;
    int col = s * 32 + cg * 4;
    float a[4];
    float c[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        a[e] = 0.0f;
        c[e] = 0.0f;
    }
    int n_trips = (n_chunks + 31) / 32;
    #pragma unroll 4
    for (int k = 0; k < n_trips; k++) {
        int cc = rsub + k * 32;
        if (cc < n_chunks) {
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)cc * 2048 + (unsigned long long)col);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                float y = __fmaf_rn(_vec_load_0[e_1], 1.0f, -c[e_1]);
                float t = a[e_1] + y;
                c[e_1] = t - a[e_1] - y;
                a[e_1] = t;
            }
        }
    }
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        red_fold[rsub * 32 + cg * 4 + e_2] = a[e_2] - c[e_2];
    }
    __syncthreads();
    if (tid < 32) {
        int gi = tid / 8;
        int lane_1 = tid - gi * 8;
        float o[4];
        float oc[4];
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            o[e_3] = 0.0f;
            oc[e_3] = 0.0f;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 8; k_1++) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                float d = red_fold[(gi * 8 + k_1) * 32 + lane_1 * 4 + e_4];
                float y_1 = __fmaf_rn(d, 1.0f, -oc[e_4]);
                float t_1 = o[e_4] + y_1;
                oc[e_4] = t_1 - o[e_4] - y_1;
                o[e_4] = t_1;
            }
        }
        #pragma unroll
        for (int e_5 = 0; e_5 < 4; e_5++) {
            grp_fold[gi * 32 + lane_1 * 4 + e_5] = o[e_5] - oc[e_5];
        }
    }
    __syncthreads();
    if (tid < 8) {
        float o2[4];
        float oc2[4];
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            o2[e_6] = 0.0f;
            oc2[e_6] = 0.0f;
        }
        #pragma unroll
        for (int gk = 0; gk < 4; gk++) {
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                float d2 = grp_fold[gk * 32 + tid * 4 + e_7];
                float y_2 = __fmaf_rn(d2, 1.0f, -oc2[e_7]);
                float t_2 = o2[e_7] + y_2;
                oc2[e_7] = t_2 - o2[e_7] - y_2;
                o2[e_7] = t_2;
            }
        }
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            o2[e_8] = o2[e_8] - oc2[e_8];
        }
        int ocol2 = s * 32 + tid * 4;
        {
            float4 _v4 = make_float4(o2[0 + 0], o2[0 + 1], o2[0 + 2], o2[0 + 3]);
            *reinterpret_cast<float4*>((dw + (unsigned long long)ocol2) + 0) = _v4;
        }
    }
    __syncthreads();
}

} // extern "C"
