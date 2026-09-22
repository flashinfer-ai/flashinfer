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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_PART_OFF 0
#define SMEM_PART_STAGE_BYTES 16
#define SMEM_PART_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kda_chunk_train_93b2b013aa41ffaa96dc(float* __restrict__ dA_part, float* __restrict__ dbias_part, float* __restrict__ dA_log, float* __restrict__ dt_bias_grad, int num_chunks, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* part = reinterpret_cast<float*>(smem_raw + 0);
    const int part_addr = smem + 0;

    // === Task calls (dependency order) ===
    int head = blockIdx.x;
    int dim = tid;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    #pragma unroll 1
    for (int c = 0; c < num_chunks; c++) {
        long long index = ((long long)c * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        sum_a = sum_a + dA_part[index];
        sum_b = sum_b + dbias_part[index];
    }
    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(sum_b);
    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
    dt_bias_grad[head * 128 + dim] = _cvt_f32_0;
    float _warp_reduce_0 = sum_a;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float wa = _warp_reduce_0;
    if (lane == 0) {
        part[warp] = wa;
    }
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            dA_log[head] = part[0] + part[1] + (part[2] + part[3]);
        }
    }
}

} // extern "C"
