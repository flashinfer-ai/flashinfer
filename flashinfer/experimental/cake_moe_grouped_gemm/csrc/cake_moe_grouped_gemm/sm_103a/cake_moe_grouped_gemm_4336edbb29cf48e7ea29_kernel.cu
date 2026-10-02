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
#define THREADS 256
#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define CLUSTER_M 256
#define EPI_CHUNK 16

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_moe_grouped_gemm_4336edbb29cf48e7ea29(float* __restrict__ partials, float* __restrict__ C, int* __restrict__ offs, int num_groups, int N, int K, int ldc, int stride_e, int num_clusters_1, int tail_splits, int raster_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    const int threads_per_row = 256 / EPI_CHUNK;
    const int rows_per_cta = THREADS / threads_per_row;
    const int row_blocks = CLUSTER_M / rows_per_cta;
    int tid_r = tid;
    int bid_r = bid;
    int j_r = bid_r / row_blocks;
    int rb_r = bid_r - j_r * row_blocks;
    int row_in_tile = rb_r * rows_per_cta + tid_r / threads_per_row;
    int col_r = tid_r % threads_per_row * EPI_CHUNK;
    int grid_n_r = N / BLOCK_N;
    int grid_k_r = K / 256;
    int tiles_per_group_r = grid_n_r * grid_k_r;
    int total_tiles_r = num_groups * tiles_per_group_r;
    int tail_base_r = total_tiles_r / num_clusters_1 * num_clusters_1;
    int t_r = tail_base_r + j_r;
    int e_r = t_r / tiles_per_group_r;
    int rem_r = t_r - e_r * tiles_per_group_r;
    int n_block_r = 0;
    int k_block_r = 0;
    {
        int nc_r = rem_r / grid_k_r;
        int kr_r = rem_r / grid_n_r;
        n_block_r = nc_r * (1 - raster_rows) + (rem_r - kr_r * grid_n_r) * raster_rows;
        k_block_r = (rem_r - nc_r * grid_k_r) * (1 - raster_rows) + kr_r * raster_rows;
    }
    int end_r = offs[e_r];
    int prev_r = e_r - 1;
    if (prev_r < 0) {
        prev_r = 0;
    }
    int start_r = offs[prev_r];
    if (e_r == 0) {
        start_r = 0;
    }
    int kb_r = (end_r - start_r + (BLOCK_K - 1)) / BLOCK_K;
    if (kb_r == 0) {
        kb_r = 1;
    }
    float acc[EPI_CHUNK];
    #pragma unroll
    for (int i0 = 0; i0 < EPI_CHUNK; i0++) {
        acc[i0] = 0.0f;
    }
    int count_r = 0;
    #pragma unroll 1
    for (int s_r = 0; s_r < tail_splits; s_r++) {
        int lo_r = s_r * kb_r / tail_splits;
        int hi_r = (s_r + 1) * kb_r / tail_splits;
        if (hi_r > lo_r) {
            count_r = count_r + 1;
            int slot_r = j_r * tail_splits + s_r;
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partials + (slot_r * (CLUSTER_M * 256) + row_in_tile * 256 + col_r) + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partials + (slot_r * (CLUSTER_M * 256) + row_in_tile * 256 + col_r + 4) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float _vec_load_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partials + (slot_r * (CLUSTER_M * 256) + row_in_tile * 256 + col_r + 8) + 0);
                _vec_load_2[0 + 0] = _v4.x;
                _vec_load_2[0 + 1] = _v4.y;
                _vec_load_2[0 + 2] = _v4.z;
                _vec_load_2[0 + 3] = _v4.w;
            }
            float _vec_load_3[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(partials + (slot_r * (CLUSTER_M * 256) + row_in_tile * 256 + col_r + 12) + 0);
                _vec_load_3[0 + 0] = _v4.x;
                _vec_load_3[0 + 1] = _v4.y;
                _vec_load_3[0 + 2] = _v4.z;
                _vec_load_3[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int i1 = 0; i1 < 4; i1++) {
                acc[i1] = acc[i1] + _vec_load_0[i1];
                acc[4 + i1] = acc[4 + i1] + _vec_load_1[i1];
                acc[8 + i1] = acc[8 + i1] + _vec_load_2[i1];
                acc[12 + i1] = acc[12 + i1] + _vec_load_3[i1];
            }
        }
    }
    if (count_r > 1) {
        {
            {
                float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                *reinterpret_cast<float4*>(C + (e_r * stride_e + (n_block_r * BLOCK_N + row_in_tile) * ldc + k_block_r * 256 + col_r) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                *reinterpret_cast<float4*>(C + (e_r * stride_e + (n_block_r * BLOCK_N + row_in_tile) * ldc + k_block_r * 256 + col_r + 4) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                *reinterpret_cast<float4*>(C + (e_r * stride_e + (n_block_r * BLOCK_N + row_in_tile) * ldc + k_block_r * 256 + col_r + 8) + 0) = _v4;
            }
            {
                float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                *reinterpret_cast<float4*>(C + (e_r * stride_e + (n_block_r * BLOCK_N + row_in_tile) * ldc + k_block_r * 256 + col_r + 12) + 0) = _v4;
            }
        }
    }
}

} // extern "C"
