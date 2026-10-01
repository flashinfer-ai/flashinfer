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
// Generated source; do not edit manually.
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "MLA requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) MlaTensorMap { uint64_t opaque[16]; };
struct __align__(64) MlaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(MlaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(MlaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) MlaTensorMapPack { MlaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(MlaTensorMap) >= alignof(CUtensorMap), "MlaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define MLA_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_WEIGHTS_OFF 0
#define SMEM_WEIGHTS_STAGE_BYTES 16384
#define SMEM_WEIGHTS_STRIDE 16384
#define SMEM_TOTAL 16384
#define THREADS 32

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_full_abi_tail_bf16(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ KV, float* __restrict__ fp8_lut, int* __restrict__ page_table, int* __restrict__ sparse_indices, int* __restrict__ row_batches, int* __restrict__ row_seq_lens, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, float* __restrict__ sinks, int num_heads, int qk_dim, int value_dim, int kv_stride, int page_size, int page_table_width, int sparse_width, int use_sparse, float softmax_scale, float bmm2_scale, int enable_sink, int write_lse)
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
    float* weights = reinterpret_cast<float*>(smem_raw + 0);
    const int weights_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row = blockIdx.x;
    int head = blockIdx.y;
    int batch = row_batches[row];
    int seq_len = row_seq_lens[row];
    int q_base = (row * num_heads + head) * qk_dim;
    float local_max = -MLA_INF;
    #pragma unroll 1
    for (int token = lane; token < seq_len; token += 32) {
        int physical_token = 0;
        if (use_sparse != 0) {
            physical_token = sparse_indices[row * sparse_width + token];
        } else {
            int logical_page = token / page_size;
            int token_in_page = token - logical_page * page_size;
            int physical_page = page_table[batch * page_table_width + logical_page];
            physical_token = physical_page * page_size + token_in_page;
        }
        int kv_base = physical_token * kv_stride;
        float score = 0.0f;
        #pragma unroll 4
        for (int dim = 0; dim < qk_dim; dim++) {
            float q_value = Q[q_base + dim];
            float kv_value = KV[kv_base + dim];
            score = score + q_value * kv_value;
        }
        score = score * softmax_scale;
        weights[token] = score;
        float _max_0 = max_noftz(local_max, score);
        local_max = _max_0;
    }
    if (enable_sink != 0 && lane == 0) {
        float _max_1 = max_noftz(local_max, sinks[head]);
        local_max = _max_1;
    }
    float _warp_reduce_0 = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float row_max = _warp_reduce_0;
    __syncthreads();
    float local_sum = 0.0f;
    #pragma unroll 1
    for (int token_1 = lane; token_1 < seq_len; token_1 += 32) {
        float _exp2_0 = approx_exp2((weights[token_1] - row_max) * 1.4426950408889634f);
        float weight = _exp2_0;
        weights[token_1] = weight;
        local_sum = local_sum + weight;
    }
    if (enable_sink != 0 && lane == 0) {
        float _exp2_1 = approx_exp2((sinks[head] - row_max) * 1.4426950408889634f);
        local_sum = local_sum + _exp2_1;
    }
    float _warp_reduce_1 = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float row_sum = _warp_reduce_1;
    float _rcp_0 = approx_rcp(row_sum);
    float inv_sum = _rcp_0;
    __syncthreads();
    int o_base = (row * num_heads + head) * value_dim;
    #pragma unroll 1
    for (int dim_1 = lane; dim_1 < value_dim; dim_1 += 32) {
        float accum = 0.0f;
        #pragma unroll 1
        for (int token_2 = 0; token_2 < seq_len; token_2++) {
            int physical_token_1 = 0;
            if (use_sparse != 0) {
                physical_token_1 = sparse_indices[row * sparse_width + token_2];
            } else {
                int logical_page_1 = token_2 / page_size;
                int token_in_page_1 = token_2 - logical_page_1 * page_size;
                int physical_page_1 = page_table[batch * page_table_width + logical_page_1];
                physical_token_1 = physical_page_1 * page_size + token_in_page_1;
            }
            int kv_base_1 = physical_token_1 * kv_stride;
            float value = KV[kv_base_1 + dim_1];
            accum = accum + weights[token_2] * value;
        }
        *(reinterpret_cast<__nv_bfloat16*>(O + (o_base + dim_1)) + (0)) = __float2bfloat16_rn(accum * inv_sum * bmm2_scale);
    }
    if (write_lse != 0 && lane == 0) {
        float _log2_0;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
        float lse = row_max * 1.4426950408889634f + _log2_0;
        *(reinterpret_cast<float*>(LSE + (row * num_heads + head)) + (0)) = lse;
    }
}

} // extern "C"

