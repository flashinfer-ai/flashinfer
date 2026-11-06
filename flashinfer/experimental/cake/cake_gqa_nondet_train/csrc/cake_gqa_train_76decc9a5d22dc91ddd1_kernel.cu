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


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gqa_train_76decc9a5d22dc91ddd1(float* __restrict__ dQ_accum, int* __restrict__ cu_seqlens_q, __nv_bfloat16* __restrict__ dQ, int T_q, int H_q, int num_segments, int rows_pad, float softmax_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int pair = blockIdx.y;
    int row0 = blockIdx.x * 64;
    int warp_row = row0 % 128 + warp * 8;
    unsigned long long tile_base = ((unsigned long long)pair * (unsigned long long)rows_pad + (unsigned long long)(row0 - row0 % 128)) * 256;
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
    int token_local0 = local0 + warp * 8 >> 1;
    int tokens_left = ((in_window != 0) ? seg_len - token_local0 : 0);
    float acc[64];
    int span_row = lane % 4;
    int span_group = lane / 4;
    #pragma unroll
    for (int tb = 0; tb < 1; tb++) {
        int row = warp_row + 2 * (tb * 4 + span_row);
        #pragma unroll
        for (int gb = 0; gb < 4; gb++) {
            #pragma unroll
            for (int hc = 0; hc < 2; hc++) {
                {
                    unsigned _ldv8_1_0;
                    unsigned _ldv8_1_1;
                    unsigned _ldv8_1_2;
                    unsigned _ldv8_1_3;
                    unsigned _ldv8_1_4;
                    unsigned _ldv8_1_5;
                    unsigned _ldv8_1_6;
                    unsigned _ldv8_1_7;
                    asm volatile(
                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(dQ_accum + (tile_base + (unsigned long long)(row / 64 * 16384 + row % 64 * 4 + hc * 256 + (gb * 8 + span_group) * 512)) + (0))) : "memory");
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 0] = __uint_as_float(_ldv8_1_0);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 1] = __uint_as_float(_ldv8_1_1);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 2] = __uint_as_float(_ldv8_1_2);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 3] = __uint_as_float(_ldv8_1_3);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 4] = __uint_as_float(_ldv8_1_4);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 5] = __uint_as_float(_ldv8_1_5);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 6] = __uint_as_float(_ldv8_1_6);
                    acc[((tb * 4 + gb) * 2 + hc) * 8 + 7] = __uint_as_float(_ldv8_1_7);
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float2 scale2 = make_float2(softmax_scale, softmax_scale);
    #pragma unroll
    for (int tb_1 = 0; tb_1 < 1; tb_1++) {
        int tk = tb_1 * 4 + span_row;
        int token = ((seg_start + token_local0 + tk) < (T_q - 1) ? (seg_start + token_local0 + tk) : (T_q - 1));
        #pragma unroll
        for (int gb_1 = 0; gb_1 < 4; gb_1++) {
            #pragma unroll
            for (int hc_1 = 0; hc_1 < 2; hc_1++) {
                unsigned int out[4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int a = ((tb_1 * 4 + gb_1) * 2 + hc_1) * 8 + 2 * j;
                    float2 _f32x2_mul_a_2 = make_float2(acc[a], acc[a + 1]);
                    float2 scaled;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_2), "l"(*(const unsigned long long*)&scale2));
                    out[j] = __as_u32(__float22bfloat162_rn(scaled));
                }
                if (tk < tokens_left) {
                    unsigned long long dst = ((unsigned long long)token * (unsigned long long)H_q + (unsigned long long)(2 * pair)) * 256 + (unsigned long long)(hc_1 * 128 + (gb_1 * 8 + span_group) * 4);
                    reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dQ + dst))[0] = reinterpret_cast<int2*>(out)[0];
                    reinterpret_cast<int2*>(reinterpret_cast<unsigned int*>(dQ + (dst + 256)))[0] = reinterpret_cast<int2*>(out + (2))[0];
                }
            }
        }
    }
}

} // extern "C"
