/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 256
#define SMEM_PARTIALS_STRIDE 256
#define SMEM_TOTAL 256
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_lm_head_loss_f56852ff841fd920822b(float* __restrict__ term, double* __restrict__ loss_acc, float* __restrict__ loss_out, int rows_c, int first_chunk, int last_chunk, int mode, float loss_div)
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
    double* partials = reinterpret_cast<double*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (dependency order) ===
    double zero = (double)0.0f;
    double acc = (double)0.0f;
    #pragma unroll 4
    for (int i = tid; i < rows_c; i += 256) {
        acc += (double)term[i];
    }
    double _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, acc, 16);
    acc = acc + _shfl_xor_0;
    double _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, acc, 8);
    acc = acc + _shfl_xor_1;
    double _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, acc, 4);
    acc = acc + _shfl_xor_2;
    double _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, acc, 2);
    acc = acc + _shfl_xor_3;
    double _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, acc, 1);
    acc = acc + _shfl_xor_4;
    if (lane == 0) {
        partials[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        double p = ((lane < 8) ? partials[lane] : zero);
        double _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, p, 16);
        p = p + _shfl_xor_5;
        double _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, p, 8);
        p = p + _shfl_xor_6;
        double _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, p, 4);
        p = p + _shfl_xor_7;
        double _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, p, 2);
        p = p + _shfl_xor_8;
        double _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, p, 1);
        p = p + _shfl_xor_9;
        if (lane == 0) {
            double prev = loss_acc[0];
            double total = ((first_chunk != 0) ? p : prev + p);
            loss_acc[0] = total;
            if (last_chunk != 0) {
                double neg = zero - total;
                double ld64 = (double)loss_div;
                double fin = ((mode == 0) ? neg / ld64 : neg);
                loss_out[0] = (float)fin;
            }
        }
    }
}

} // extern "C"
