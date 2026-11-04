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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kda_train_ee7092e42aef1607a922(double* __restrict__ param_acc, float* __restrict__ dA_log, float* __restrict__ ddt_bias, int num_heads)
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

    // Prologue
    double* wsum = reinterpret_cast<double*>(smem_raw + 0);
    const int wsum_addr = smem + 0;

    // === Task calls (declaration order) ===
    int head = blockIdx.x;
    int d = tid;
    int base = head * 256;
    double a = param_acc[base + d];
    double b = param_acc[base + 128 + d];
    ddt_bias[head * 128 + d] = (float)b;
    double zero64 = (double)0.0f;
    param_acc[base + d] = zero64;
    param_acc[base + 128 + d] = zero64;
    double v = a;
    double _shfl_xor_sync_0 = __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v = v + _shfl_xor_sync_0;
    double _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, v, 8);
    v = v + _shfl_xor_sync_1;
    double _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, v, 4);
    v = v + _shfl_xor_sync_2;
    double _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, v, 2);
    v = v + _shfl_xor_sync_3;
    double _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, v, 1);
    v = v + _shfl_xor_sync_4;
    if (lane == 0) {
        wsum[warp] = v;
    }
    __syncthreads();
    if (tid == 0) {
        double total = wsum[0] + wsum[1];
        total = total + wsum[2];
        total = total + wsum[3];
        dA_log[head] = (float)total;
    }
}

} // extern "C"
