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
#define SMEM_SUMS_OFF 0
#define SMEM_SUMS_STAGE_BYTES 2048
#define SMEM_SUMS_STRIDE 2048
#define SMEM_GROUPS_OFF 2048
#define SMEM_GROUPS_STAGE_BYTES 256
#define SMEM_GROUPS_STRIDE 256
#define SMEM_TOTAL 2304
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_rmsnorm_silu_gate_train_47d2327da7f419fce725(float* __restrict__ partial, float* __restrict__ dw, int chunks, int accumulate)
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
    double* sums = reinterpret_cast<double*>(smem_raw + 0);
    const int sums_addr = smem + 0;
    double* groups = reinterpret_cast<double*>(smem_raw + 2048);
    const int groups_addr = smem + 2048;

    // === Task calls (declaration order) ===
    int lane_col = tid & 0;
    int column = bid * 1 + lane_col;
    double total = 0.0;
    #pragma unroll 8
    for (int c = tid >> 0; c < chunks; c += 256) {
        total = total + (double)partial[(long long)c * 128 + (long long)column];
    }
    sums[tid] = total;
    __syncthreads();
    if (tid < 32) {
        double part = 0.0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            part = part + sums[((tid >> 0) * 8 + j) * 1 + lane_col];
        }
        groups[tid] = part;
    }
    __syncthreads();
    if (tid < 1) {
        double folded = 0.0;
        #pragma unroll
        for (int g = 0; g < 32; g++) {
            folded = folded + groups[g * 1 + lane_col];
        }
        if (accumulate != 0) {
            folded = folded + (double)dw[column];
        }
        dw[column] = (float)folded;
    }
}

} // extern "C"
