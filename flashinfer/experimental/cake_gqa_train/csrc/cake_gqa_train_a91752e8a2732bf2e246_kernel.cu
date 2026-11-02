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
#define THREADS 256

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gqa_train_a91752e8a2732bf2e246(__nv_bfloat16* __restrict__ O, __nv_bfloat16* __restrict__ dO, float* __restrict__ LSE, int* __restrict__ cu_seqlens_q, float* __restrict__ LSE_log2, float* __restrict__ dPsum, float* __restrict__ dQ_accum, int T_q, int H_q, int num_segments, int rows_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int pair = blockIdx.y;
    int row0 = blockIdx.x * 32;
    int half = tid / 16;
    int col = lane % 16 * 16;
    unsigned long long stats_base = (unsigned long long)pair * (unsigned long long)rows_pad + (unsigned long long)row0;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int limit = (row0 / 128 + 1) * 128;
    int started_count = 0;
    #pragma unroll 1
    for (int scan_base = 0; scan_base < num_segments; scan_base += 32) {
        int b = scan_base + lane;
        int _min_0 = ((b) < (num_segments - 1) ? (b) : (num_segments - 1));
        int cu_b = cu_seqlens_q[_min_0];
        unsigned int started = __ballot_sync(0xFFFFFFFF, b < num_segments && 2 * cu_b + 128 * b < limit);
        started_count = started_count + (int)(unsigned int)__popc(started);
    }
    int seg = started_count - 1;
    int seg_start = cu_seqlens_q[seg];
    int seg_len = cu_seqlens_q[seg + 1] - seg_start;
    int local0 = row0 - (2 * seg_start + 128 * seg) / 128 * 128;
    int in_window = ((local0 < (2 * seg_len + 128 - 1) / 128 * 128) ? 1 : 0);
    unsigned int o_words[16];
    unsigned int do_words[16];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int local = local0 + i * 16 + half;
        int load_token = ((seg_start + (local >> 1)) < (T_q - 1) ? (seg_start + (local >> 1)) : (T_q - 1));
        unsigned long long row_offset = ((unsigned long long)load_token * (unsigned long long)H_q + (unsigned long long)(2 * pair + (local & 1))) * 256 + (unsigned long long)col;
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(o_words[i * 8 + 0]), "=r"(o_words[i * 8 + 1]), "=r"(o_words[i * 8 + 2]), "=r"(o_words[i * 8 + 3]), "=r"(o_words[i * 8 + 4]), "=r"(o_words[i * 8 + 5]), "=r"(o_words[i * 8 + 6]), "=r"(o_words[i * 8 + 7]) : "l"((const void*)((const char*)(O + row_offset + 0) + 0)) : "memory");
        }
        {
            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(do_words[i * 8 + 0]), "=r"(do_words[i * 8 + 1]), "=r"(do_words[i * 8 + 2]), "=r"(do_words[i * 8 + 3]), "=r"(do_words[i * 8 + 4]), "=r"(do_words[i * 8 + 5]), "=r"(do_words[i * 8 + 6]), "=r"(do_words[i * 8 + 7]) : "l"((const void*)((const char*)(dO + row_offset + 0) + 0)) : "memory");
        }
    }
    int lse_local = local0 + tid % 32;
    int lse_token = ((seg_start + (lse_local >> 1)) < (T_q - 1) ? (seg_start + (lse_local >> 1)) : (T_q - 1));
    float lse = LSE[(unsigned long long)(2 * pair + (lse_local & 1)) * (unsigned long long)T_q + (unsigned long long)lse_token];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned int zeros[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        zeros[j] = 0;
    }
    unsigned long long chunk_base = stats_base * 256 + (unsigned long long)(tid * 8);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; i_1++) {
        unsigned long long elem = chunk_base + (unsigned long long)(i_1 * 4096);
        {
            const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(zeros);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dQ_accum + elem))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
        }
        {
            const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(zeros);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(reinterpret_cast<unsigned int*>(dQ_accum + (elem + 2048)))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
        }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; i_2++) {
        int local_1 = local0 + i_2 * 16 + half;
        float even = 0.0f;
        float odd = 0.0f;
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            float sums[2] = {even, odd};
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sums[0]) : "h"(static_cast<uint16_t>((o_words[i_2 * 8 + j_1]))), "h"(static_cast<uint16_t>((do_words[i_2 * 8 + j_1]))));
            asm("fma.rn.f32.bf16 %0, %1, %2, %0;" : "+f"(sums[1]) : "h"(static_cast<uint16_t>((o_words[i_2 * 8 + j_1]) >> 16)), "h"(static_cast<uint16_t>((do_words[i_2 * 8 + j_1]) >> 16)));
            even = sums[0];
            odd = sums[1];
        }
        float row_sum = even + odd;
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
        row_sum = row_sum + _shfl_xor_sync_5;
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
        row_sum = row_sum + _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
        row_sum = row_sum + _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
        row_sum = row_sum + _shfl_xor_sync_8;
        if (lane % 16 == 0 && in_window != 0) {
            *(reinterpret_cast<float*>(dPsum + (stats_base + (unsigned long long)(i_2 * 16 + half))) + (0)) = ((local_1 >> 1 < seg_len) ? row_sum : 0.0f);
        }
    }
    if (tid < 32 && in_window != 0) {
        float lse_log2 = ((lse == -CUDART_INF_F) ? 0.0f : mul_f32_rn_noftz(lse, 1.442695f));
        *(reinterpret_cast<float*>(LSE_log2 + (stats_base + (unsigned long long)tid)) + (0)) = ((lse_local >> 1 < seg_len) ? lse_log2 : CUDART_INF_F);
    }
}

} // extern "C"
