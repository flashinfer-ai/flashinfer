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
#define SMEM_RED_FOLD_OFF 0
#define SMEM_RED_FOLD_STAGE_BYTES 6144
#define SMEM_RED_FOLD_STRIDE 6144
#define SMEM_TOTAL 6144
#define THREADS 384

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(384) void
kernel_cake_rmsnorm_det_train_60bfcb24560bbae612c3(float* __restrict__ partial, float* __restrict__ dw, int n_chunks)
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

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int s = bid;
    int rsub = tid / 16;
    int cg = tid - rsub * 16;
    int col = s * 64 + cg * 4;
    float a4[4];
    float c4[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        a4[e] = 0.0f;
        c4[e] = 0.0f;
    }
    int n_trips = (n_chunks + 23) / 24;
    #pragma unroll 4
    for (int k = 0; k < n_trips; k++) {
        int cc = rsub + k * 24;
        if (cc < n_chunks) {
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)cc * 6144 + (unsigned long long)col);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                float y = __fmaf_rn(_vec_load_0[e_1], 1.0f, -c4[e_1]);
                float t = a4[e_1] + y;
                c4[e_1] = t - a4[e_1] - y;
                a4[e_1] = t;
            }
        }
    }
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        red_fold[rsub * 64 + cg * 4 + e_2] = a4[e_2] - c4[e_2];
    }
    __syncthreads();
    if (tid < 16) {
        float o4[4];
        float oc4[4];
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            o4[e_3] = 0.0f;
            oc4[e_3] = 0.0f;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 24; k_1++) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                float d = red_fold[k_1 * 64 + tid * 4 + e_4];
                float y_1 = __fmaf_rn(d, 1.0f, -oc4[e_4]);
                float t_1 = o4[e_4] + y_1;
                oc4[e_4] = t_1 - o4[e_4] - y_1;
                o4[e_4] = t_1;
            }
        }
        #pragma unroll
        for (int e_5 = 0; e_5 < 4; e_5++) {
            o4[e_5] = o4[e_5] - oc4[e_5];
        }
        int ocol = s * 64 + tid * 4;
        {
            float4 _v4 = make_float4(o4[0 + 0], o4[0 + 1], o4[0 + 2], o4[0 + 3]);
            *reinterpret_cast<float4*>(dw + (unsigned long long)ocol) = _v4;
        }
    }
    __syncthreads();
}

} // extern "C"
