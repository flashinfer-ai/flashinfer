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
#define SMEM_WEIGHTS_OFF 0
#define SMEM_WEIGHTS_STAGE_BYTES 256
#define SMEM_WEIGHTS_STRIDE 256
#define SMEM_TOTAL 256
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_cake_nvfp4_mla_decode_dc4be86c62fbaf46c704(__nv_bfloat16* __restrict__ o, float* __restrict__ lse, float* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ row_splits, int num_heads, int max_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* weights = reinterpret_cast<float*>(smem_raw + 0);
    const int weights_addr = smem + 0;

    // === Task calls (dependency order) ===
    float NEG_INF = -CUDART_INF_F;
    int h_idx = blockIdx.x;
    int q_idx = blockIdx.y;
    int tidx = tid;
    int row = q_idx * num_heads + h_idx;
    int used = row_splits[q_idx];
    int lse_base = row * max_splits;
    if (warp == 0) {
        float local_lse[2];
        float lse_max = NEG_INF;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int sk = tidx + i * 32;
            float lv = ((sk < used) ? partial_lse[lse_base + sk] : NEG_INF);
            local_lse[i] = lv;
            float _max_0 = max_noftz(lse_max, lv);
            lse_max = _max_0;
        }
        float _warp_reduce_0 = lse_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        lse_max = _warp_reduce_0;
        lse_max = ((lse_max != NEG_INF) ? lse_max : 0.0f);
        float sum_w = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; i_1++) {
            float _exp2_0 = approx_exp2(local_lse[i_1] - lse_max);
            sum_w = sum_w + _exp2_0;
        }
        float _warp_reduce_1 = sum_w;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        sum_w = _warp_reduce_1;
        float _log2_0;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(sum_w));
        float global_lse2 = lse_max + _log2_0;
        if (tidx == 0) {
            *(reinterpret_cast<float*>(lse + row) + (0)) = global_lse2 * 0.6931471805599453f;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; i_2++) {
            int sk2 = tidx + i_2 * 32;
            if (sk2 < used) {
                float _exp2_1 = approx_exp2(local_lse[i_2] - global_lse2);
                weights[sk2] = _exp2_1;
            }
        }
    }
    asm volatile("barrier.sync 4, 128;" ::: "memory");
    float acc[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }
    int o_base = row * max_splits * 512 + tidx * 4;
    #pragma unroll 1
    for (int i_3 = 0; i_3 < used; i_3++) {
        float w_i = weights[i_3];
        #pragma unroll
        for (int j_1 = 0; j_1 < 4; j_1++) {
            acc[j_1] = acc[j_1] + partial_o[o_base + i_3 * 512 + j_1] * w_i;
        }
    }
    int out_base = row * 512 + tidx * 4;
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; j_2++) {
        *(reinterpret_cast<__nv_bfloat16*>(o + (out_base + j_2)) + (0)) = __float2bfloat16_rn((__nv_bfloat16)acc[j_2]);
    }
}

} // extern "C"
