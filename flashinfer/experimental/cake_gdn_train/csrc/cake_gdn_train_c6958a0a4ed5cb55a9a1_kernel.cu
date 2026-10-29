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
#define SMEM_X_BLOCK_OFF 0
#define SMEM_X_BLOCK_STAGE_BYTES 4096
#define SMEM_X_BLOCK_STRIDE 4096
#define SMEM_SLABS_OFF 4096
#define SMEM_SLABS_STAGE_BYTES 33024
#define SMEM_SLABS_STRIDE 33024
#define SMEM_TOTAL 37120
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_c6958a0a4ed5cb55a9a1(float* __restrict__ m, float* __restrict__ c, float* __restrict__ x0, float* __restrict__ y, int* __restrict__ chain_start, long long* __restrict__ elem_m, long long* __restrict__ elem_c, long long* __restrict__ elem_y, long long* __restrict__ x0_index, long long* __restrict__ y_final, int ncols, int transpose_m, int write_states, int cols_per_cta)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
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
    float* x_block = reinterpret_cast<float*>(smem_raw + 0);
    const int x_block_addr = smem + 0;
    float* slabs = reinterpret_cast<float*>(smem_raw + 4096);
    const int slabs_addr = smem + 4096;

    // === Task calls (declaration order) ===
    int nblk = ncols / 8;
    int chain = blockIdx.x / nblk;
    int col0 = blockIdx.x % nblk * 8;
    int col_tile = col0 / 128;
    int v0 = col0 % 128;
    int k = tid;
    long long row_off = (long long)col_tile * 16384 + (long long)(k * 128 + v0);
    int e_begin = chain_start[chain];
    int e_end = chain_start[chain + 1];
    unsigned int x_addr = x_block_addr + (unsigned int)(k * 32);
    int w4r = warp * 4 + lane / 8;
    int q4 = lane % 8 * 4;
    float vals[8];
    float acc[8];
    float xv[4];
    float ring[64];
    float pf[32];
    long long x0_off = x0_index[chain];
    if (x0_off >= 0) {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + row_off);
            vals[0 + 0] = _v4.x;
            vals[0 + 1] = _v4.y;
            vals[0 + 2] = _v4.z;
            vals[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + row_off + 4);
            vals[4 + 0] = _v4.x;
            vals[4 + 1] = _v4.y;
            vals[4 + 2] = _v4.z;
            vals[4 + 3] = _v4.w;
        }
    } else {
        int diag = k - v0;
        if (col_tile == 0) {
            diag = -1;
        }
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            vals[v] = ((v == diag) ? 1.0f : 0.0f);
        }
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr), "r"(__float_as_uint(vals[0])), "r"(__float_as_uint(vals[1])), "r"(__float_as_uint(vals[2])), "r"(__float_as_uint(vals[3])) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr + 16), "r"(__float_as_uint(vals[4])), "r"(__float_as_uint(vals[5])), "r"(__float_as_uint(vals[6])), "r"(__float_as_uint(vals[7])) : "memory");
    if (e_begin < e_end) {
        if (write_states != 0) {
            long long y_first = elem_y[e_begin];
            if (y_first >= 0) {
                {
                    float4 _v4 = make_float4(vals[0 + 0], vals[0 + 1], vals[0 + 2], vals[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_first + row_off) = _v4;
                }
                {
                    float4 _v4 = make_float4(vals[4 + 0], vals[4 + 1], vals[4 + 2], vals[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_first + row_off + 4) = _v4;
                }
            }
        }
    } else {
        long long y_last = y_final[chain];
        if (y_last >= 0) {
            {
                float4 _v4 = make_float4(vals[0 + 0], vals[0 + 1], vals[0 + 2], vals[0 + 3]);
                *reinterpret_cast<float4*>(y + y_last + row_off) = _v4;
            }
            {
                float4 _v4 = make_float4(vals[4 + 0], vals[4 + 1], vals[4 + 2], vals[4 + 3]);
                *reinterpret_cast<float4*>(y + y_last + row_off + 4) = _v4;
            }
        }
    }
    if (transpose_m == 0) {
        if (e_begin < e_end) {
            long long m_first = elem_m[e_begin];
            #pragma unroll
            for (int f = 0; f < 16; f++) {
                ring[f] = m[m_first + (long long)(f * 128 + k)];
            }
            #pragma unroll
            for (int f_1 = 0; f_1 < 16; f_1++) {
                ring[16 + f_1] = m[m_first + (long long)((16 + f_1) * 128 + k)];
            }
            #pragma unroll
            for (int f_2 = 0; f_2 < 16; f_2++) {
                ring[32 + f_2] = m[m_first + (long long)((32 + f_2) * 128 + k)];
            }
        }
        __syncthreads();
        #pragma unroll 1
        for (int e = e_begin; e < e_end; e++) {
            long long cur = elem_m[e];
            int has_next = 0;
            long long nxt = 0;
            if (e_end > e + 1) {
                has_next = 1;
                nxt = elem_m[e + 1];
            }
            long long c_off = elem_c[e];
            if (col_tile != 0) {
                c_off = -1;
            }
            if (c_off >= 0) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_off + row_off);
                    acc[0 + 0] = _v4.x;
                    acc[0 + 1] = _v4.y;
                    acc[0 + 2] = _v4.z;
                    acc[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_off + row_off + 4);
                    acc[4 + 0] = _v4.x;
                    acc[4 + 1] = _v4.y;
                    acc[4 + 2] = _v4.z;
                    acc[4 + 3] = _v4.w;
                }
            } else {
                acc[0] = 0.0f;
                acc[1] = 0.0f;
                acc[2] = 0.0f;
                acc[3] = 0.0f;
                acc[4] = 0.0f;
                acc[5] = 0.0f;
                acc[6] = 0.0f;
                acc[7] = 0.0f;
            }
            #pragma unroll
            for (int f_3 = 0; f_3 < 16; f_3++) {
                ring[48 + f_3] = m[cur + (long long)((48 + f_3) * 128 + k)];
            }
            #pragma unroll
            for (int f_0 = 0; f_0 < 16; f_0++) {
                float mv_0 = ring[f_0];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(f_0 * 8 * 4)));
                acc[0] = __fmaf_rn(mv_0, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_0, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_0, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_0, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((f_0 * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_0, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_0, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_0, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_0, xv[3], acc[7]);
            }
            #pragma unroll
            for (int f_1_1 = 0; f_1_1 < 16; f_1_1++) {
                ring[f_1_1] = m[cur + (long long)((64 + f_1_1) * 128 + k)];
            }
            #pragma unroll
            for (int f_2_1 = 0; f_2_1 < 16; f_2_1++) {
                float mv_1 = ring[16 + f_2_1];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((16 + f_2_1) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_1, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_1, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_1, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_1, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((16 + f_2_1) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_1, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_1, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_1, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_1, xv[3], acc[7]);
            }
            #pragma unroll
            for (int f_3_1 = 0; f_3_1 < 16; f_3_1++) {
                ring[16 + f_3_1] = m[cur + (long long)((80 + f_3_1) * 128 + k)];
            }
            #pragma unroll
            for (int f_4 = 0; f_4 < 16; f_4++) {
                float mv_2 = ring[32 + f_4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((32 + f_4) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_2, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_2, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_2, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_2, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((32 + f_4) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_2, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_2, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_2, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_2, xv[3], acc[7]);
            }
            #pragma unroll
            for (int f_5 = 0; f_5 < 16; f_5++) {
                ring[32 + f_5] = m[cur + (long long)((96 + f_5) * 128 + k)];
            }
            #pragma unroll
            for (int f_6 = 0; f_6 < 16; f_6++) {
                float mv_3 = ring[48 + f_6];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((48 + f_6) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_3, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_3, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_3, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_3, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((48 + f_6) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_3, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_3, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_3, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_3, xv[3], acc[7]);
            }
            #pragma unroll
            for (int f_7 = 0; f_7 < 16; f_7++) {
                ring[48 + f_7] = m[cur + (long long)((112 + f_7) * 128 + k)];
            }
            #pragma unroll
            for (int f_8 = 0; f_8 < 16; f_8++) {
                float mv_4 = ring[f_8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((64 + f_8) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_4, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_4, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_4, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_4, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((64 + f_8) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_4, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_4, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_4, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_4, xv[3], acc[7]);
            }
            if (has_next != 0) {
                #pragma unroll
                for (int f_9 = 0; f_9 < 16; f_9++) {
                    ring[f_9] = m[nxt + (long long)(f_9 * 128 + k)];
                }
            }
            #pragma unroll
            for (int f_10 = 0; f_10 < 16; f_10++) {
                float mv_5 = ring[16 + f_10];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((80 + f_10) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_5, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_5, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_5, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_5, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((80 + f_10) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_5, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_5, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_5, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_5, xv[3], acc[7]);
            }
            if (has_next != 0) {
                #pragma unroll
                for (int f_11 = 0; f_11 < 16; f_11++) {
                    ring[16 + f_11] = m[nxt + (long long)((16 + f_11) * 128 + k)];
                }
            }
            #pragma unroll
            for (int f_12 = 0; f_12 < 16; f_12++) {
                float mv_6 = ring[32 + f_12];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((96 + f_12) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_6, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_6, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_6, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_6, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((96 + f_12) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_6, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_6, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_6, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_6, xv[3], acc[7]);
            }
            if (has_next != 0) {
                #pragma unroll
                for (int f_13 = 0; f_13 < 16; f_13++) {
                    ring[32 + f_13] = m[nxt + (long long)((32 + f_13) * 128 + k)];
                }
            }
            #pragma unroll
            for (int f_14 = 0; f_14 < 16; f_14++) {
                float mv_7 = ring[48 + f_14];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((112 + f_14) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_7, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_7, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_7, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_7, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((112 + f_14) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_7, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_7, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_7, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_7, xv[3], acc[7]);
            }
            __syncthreads();
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr + 16), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            if (has_next != 0) {
                if (write_states != 0) {
                    long long y_next = elem_y[e + 1];
                    if (y_next >= 0) {
                        {
                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                            *reinterpret_cast<float4*>(y + y_next + row_off) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                            *reinterpret_cast<float4*>(y + y_next + row_off + 4) = _v4;
                        }
                    }
                }
            } else {
                long long y_out = y_final[chain];
                if (y_out >= 0) {
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>(y + y_out + row_off) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                        *reinterpret_cast<float4*>(y + y_out + row_off + 4) = _v4;
                    }
                }
            }
            __syncthreads();
        }
    } else {
        if (e_begin < e_end) {
            long long m_first_1 = elem_m[e_begin];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)(w4r * 128 + q4));
                pf[0 + 0] = _v4.x;
                pf[0 + 1] = _v4.y;
                pf[0 + 2] = _v4.z;
                pf[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((16 + w4r) * 128 + q4));
                pf[4 + 0] = _v4.x;
                pf[4 + 1] = _v4.y;
                pf[4 + 2] = _v4.z;
                pf[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((32 + w4r) * 128 + q4));
                pf[8 + 0] = _v4.x;
                pf[8 + 1] = _v4.y;
                pf[8 + 2] = _v4.z;
                pf[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((48 + w4r) * 128 + q4));
                pf[12 + 0] = _v4.x;
                pf[12 + 1] = _v4.y;
                pf[12 + 2] = _v4.z;
                pf[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((64 + w4r) * 128 + q4));
                pf[16 + 0] = _v4.x;
                pf[16 + 1] = _v4.y;
                pf[16 + 2] = _v4.z;
                pf[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((80 + w4r) * 128 + q4));
                pf[20 + 0] = _v4.x;
                pf[20 + 1] = _v4.y;
                pf[20 + 2] = _v4.z;
                pf[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((96 + w4r) * 128 + q4));
                pf[24 + 0] = _v4.x;
                pf[24 + 1] = _v4.y;
                pf[24 + 2] = _v4.z;
                pf[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + m_first_1 + (long long)((112 + w4r) * 128 + q4));
                pf[28 + 0] = _v4.x;
                pf[28 + 1] = _v4.y;
                pf[28 + 2] = _v4.z;
                pf[28 + 3] = _v4.w;
            }
            slabs[q4 * 129 + w4r] = pf[0];
            slabs[(q4 + 1) * 129 + w4r] = pf[1];
            slabs[(q4 + 2) * 129 + w4r] = pf[2];
            slabs[(q4 + 3) * 129 + w4r] = pf[3];
            slabs[q4 * 129 + 16 + w4r] = pf[4];
            slabs[(q4 + 1) * 129 + 16 + w4r] = pf[5];
            slabs[(q4 + 2) * 129 + 16 + w4r] = pf[6];
            slabs[(q4 + 3) * 129 + 16 + w4r] = pf[7];
            slabs[q4 * 129 + 32 + w4r] = pf[8];
            slabs[(q4 + 1) * 129 + 32 + w4r] = pf[9];
            slabs[(q4 + 2) * 129 + 32 + w4r] = pf[10];
            slabs[(q4 + 3) * 129 + 32 + w4r] = pf[11];
            slabs[q4 * 129 + 48 + w4r] = pf[12];
            slabs[(q4 + 1) * 129 + 48 + w4r] = pf[13];
            slabs[(q4 + 2) * 129 + 48 + w4r] = pf[14];
            slabs[(q4 + 3) * 129 + 48 + w4r] = pf[15];
            slabs[q4 * 129 + 64 + w4r] = pf[16];
            slabs[(q4 + 1) * 129 + 64 + w4r] = pf[17];
            slabs[(q4 + 2) * 129 + 64 + w4r] = pf[18];
            slabs[(q4 + 3) * 129 + 64 + w4r] = pf[19];
            slabs[q4 * 129 + 80 + w4r] = pf[20];
            slabs[(q4 + 1) * 129 + 80 + w4r] = pf[21];
            slabs[(q4 + 2) * 129 + 80 + w4r] = pf[22];
            slabs[(q4 + 3) * 129 + 80 + w4r] = pf[23];
            slabs[q4 * 129 + 96 + w4r] = pf[24];
            slabs[(q4 + 1) * 129 + 96 + w4r] = pf[25];
            slabs[(q4 + 2) * 129 + 96 + w4r] = pf[26];
            slabs[(q4 + 3) * 129 + 96 + w4r] = pf[27];
            slabs[q4 * 129 + 112 + w4r] = pf[28];
            slabs[(q4 + 1) * 129 + 112 + w4r] = pf[29];
            slabs[(q4 + 2) * 129 + 112 + w4r] = pf[30];
            slabs[(q4 + 3) * 129 + 112 + w4r] = pf[31];
        }
        __syncthreads();
        #pragma unroll 1
        for (int e_1 = e_begin; e_1 < e_end; e_1++) {
            long long cur_1 = elem_m[e_1];
            int has_next_1 = 0;
            long long nxt_1 = 0;
            if (e_end > e_1 + 1) {
                has_next_1 = 1;
                nxt_1 = elem_m[e_1 + 1];
            }
            long long c_off_1 = elem_c[e_1];
            if (col_tile != 0) {
                c_off_1 = -1;
            }
            if (c_off_1 >= 0) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_off_1 + row_off);
                    acc[0 + 0] = _v4.x;
                    acc[0 + 1] = _v4.y;
                    acc[0 + 2] = _v4.z;
                    acc[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_off_1 + row_off + 4);
                    acc[4 + 0] = _v4.x;
                    acc[4 + 1] = _v4.y;
                    acc[4 + 2] = _v4.z;
                    acc[4 + 3] = _v4.w;
                }
            } else {
                acc[0] = 0.0f;
                acc[1] = 0.0f;
                acc[2] = 0.0f;
                acc[3] = 0.0f;
                acc[4] = 0.0f;
                acc[5] = 0.0f;
                acc[6] = 0.0f;
                acc[7] = 0.0f;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)(w4r * 128 + 32 + q4));
                pf[0 + 0] = _v4.x;
                pf[0 + 1] = _v4.y;
                pf[0 + 2] = _v4.z;
                pf[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((16 + w4r) * 128 + 32 + q4));
                pf[4 + 0] = _v4.x;
                pf[4 + 1] = _v4.y;
                pf[4 + 2] = _v4.z;
                pf[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((32 + w4r) * 128 + 32 + q4));
                pf[8 + 0] = _v4.x;
                pf[8 + 1] = _v4.y;
                pf[8 + 2] = _v4.z;
                pf[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((48 + w4r) * 128 + 32 + q4));
                pf[12 + 0] = _v4.x;
                pf[12 + 1] = _v4.y;
                pf[12 + 2] = _v4.z;
                pf[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((64 + w4r) * 128 + 32 + q4));
                pf[16 + 0] = _v4.x;
                pf[16 + 1] = _v4.y;
                pf[16 + 2] = _v4.z;
                pf[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((80 + w4r) * 128 + 32 + q4));
                pf[20 + 0] = _v4.x;
                pf[20 + 1] = _v4.y;
                pf[20 + 2] = _v4.z;
                pf[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((96 + w4r) * 128 + 32 + q4));
                pf[24 + 0] = _v4.x;
                pf[24 + 1] = _v4.y;
                pf[24 + 2] = _v4.z;
                pf[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((112 + w4r) * 128 + 32 + q4));
                pf[28 + 0] = _v4.x;
                pf[28 + 1] = _v4.y;
                pf[28 + 2] = _v4.z;
                pf[28 + 3] = _v4.w;
            }
            #pragma unroll
            for (int kk = 0; kk < 32; kk++) {
                float mv = slabs[kk * 129 + k];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(kk * 8 * 4)));
                acc[0] = __fmaf_rn(mv, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((kk * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv, xv[3], acc[7]);
            }
            slabs[4128 + q4 * 129 + w4r] = pf[0];
            slabs[4128 + (q4 + 1) * 129 + w4r] = pf[1];
            slabs[4128 + (q4 + 2) * 129 + w4r] = pf[2];
            slabs[4128 + (q4 + 3) * 129 + w4r] = pf[3];
            slabs[4128 + q4 * 129 + 16 + w4r] = pf[4];
            slabs[4128 + (q4 + 1) * 129 + 16 + w4r] = pf[5];
            slabs[4128 + (q4 + 2) * 129 + 16 + w4r] = pf[6];
            slabs[4128 + (q4 + 3) * 129 + 16 + w4r] = pf[7];
            slabs[4128 + q4 * 129 + 32 + w4r] = pf[8];
            slabs[4128 + (q4 + 1) * 129 + 32 + w4r] = pf[9];
            slabs[4128 + (q4 + 2) * 129 + 32 + w4r] = pf[10];
            slabs[4128 + (q4 + 3) * 129 + 32 + w4r] = pf[11];
            slabs[4128 + q4 * 129 + 48 + w4r] = pf[12];
            slabs[4128 + (q4 + 1) * 129 + 48 + w4r] = pf[13];
            slabs[4128 + (q4 + 2) * 129 + 48 + w4r] = pf[14];
            slabs[4128 + (q4 + 3) * 129 + 48 + w4r] = pf[15];
            slabs[4128 + q4 * 129 + 64 + w4r] = pf[16];
            slabs[4128 + (q4 + 1) * 129 + 64 + w4r] = pf[17];
            slabs[4128 + (q4 + 2) * 129 + 64 + w4r] = pf[18];
            slabs[4128 + (q4 + 3) * 129 + 64 + w4r] = pf[19];
            slabs[4128 + q4 * 129 + 80 + w4r] = pf[20];
            slabs[4128 + (q4 + 1) * 129 + 80 + w4r] = pf[21];
            slabs[4128 + (q4 + 2) * 129 + 80 + w4r] = pf[22];
            slabs[4128 + (q4 + 3) * 129 + 80 + w4r] = pf[23];
            slabs[4128 + q4 * 129 + 96 + w4r] = pf[24];
            slabs[4128 + (q4 + 1) * 129 + 96 + w4r] = pf[25];
            slabs[4128 + (q4 + 2) * 129 + 96 + w4r] = pf[26];
            slabs[4128 + (q4 + 3) * 129 + 96 + w4r] = pf[27];
            slabs[4128 + q4 * 129 + 112 + w4r] = pf[28];
            slabs[4128 + (q4 + 1) * 129 + 112 + w4r] = pf[29];
            slabs[4128 + (q4 + 2) * 129 + 112 + w4r] = pf[30];
            slabs[4128 + (q4 + 3) * 129 + 112 + w4r] = pf[31];
            __syncthreads();
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)(w4r * 128 + 64 + q4));
                pf[0 + 0] = _v4.x;
                pf[0 + 1] = _v4.y;
                pf[0 + 2] = _v4.z;
                pf[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((16 + w4r) * 128 + 64 + q4));
                pf[4 + 0] = _v4.x;
                pf[4 + 1] = _v4.y;
                pf[4 + 2] = _v4.z;
                pf[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((32 + w4r) * 128 + 64 + q4));
                pf[8 + 0] = _v4.x;
                pf[8 + 1] = _v4.y;
                pf[8 + 2] = _v4.z;
                pf[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((48 + w4r) * 128 + 64 + q4));
                pf[12 + 0] = _v4.x;
                pf[12 + 1] = _v4.y;
                pf[12 + 2] = _v4.z;
                pf[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((64 + w4r) * 128 + 64 + q4));
                pf[16 + 0] = _v4.x;
                pf[16 + 1] = _v4.y;
                pf[16 + 2] = _v4.z;
                pf[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((80 + w4r) * 128 + 64 + q4));
                pf[20 + 0] = _v4.x;
                pf[20 + 1] = _v4.y;
                pf[20 + 2] = _v4.z;
                pf[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((96 + w4r) * 128 + 64 + q4));
                pf[24 + 0] = _v4.x;
                pf[24 + 1] = _v4.y;
                pf[24 + 2] = _v4.z;
                pf[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((112 + w4r) * 128 + 64 + q4));
                pf[28 + 0] = _v4.x;
                pf[28 + 1] = _v4.y;
                pf[28 + 2] = _v4.z;
                pf[28 + 3] = _v4.w;
            }
            #pragma unroll
            for (int kk_1 = 0; kk_1 < 32; kk_1++) {
                float mv_8 = slabs[4128 + kk_1 * 129 + k];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((32 + kk_1) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_8, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_8, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_8, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_8, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((32 + kk_1) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_8, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_8, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_8, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_8, xv[3], acc[7]);
            }
            slabs[q4 * 129 + w4r] = pf[0];
            slabs[(q4 + 1) * 129 + w4r] = pf[1];
            slabs[(q4 + 2) * 129 + w4r] = pf[2];
            slabs[(q4 + 3) * 129 + w4r] = pf[3];
            slabs[q4 * 129 + 16 + w4r] = pf[4];
            slabs[(q4 + 1) * 129 + 16 + w4r] = pf[5];
            slabs[(q4 + 2) * 129 + 16 + w4r] = pf[6];
            slabs[(q4 + 3) * 129 + 16 + w4r] = pf[7];
            slabs[q4 * 129 + 32 + w4r] = pf[8];
            slabs[(q4 + 1) * 129 + 32 + w4r] = pf[9];
            slabs[(q4 + 2) * 129 + 32 + w4r] = pf[10];
            slabs[(q4 + 3) * 129 + 32 + w4r] = pf[11];
            slabs[q4 * 129 + 48 + w4r] = pf[12];
            slabs[(q4 + 1) * 129 + 48 + w4r] = pf[13];
            slabs[(q4 + 2) * 129 + 48 + w4r] = pf[14];
            slabs[(q4 + 3) * 129 + 48 + w4r] = pf[15];
            slabs[q4 * 129 + 64 + w4r] = pf[16];
            slabs[(q4 + 1) * 129 + 64 + w4r] = pf[17];
            slabs[(q4 + 2) * 129 + 64 + w4r] = pf[18];
            slabs[(q4 + 3) * 129 + 64 + w4r] = pf[19];
            slabs[q4 * 129 + 80 + w4r] = pf[20];
            slabs[(q4 + 1) * 129 + 80 + w4r] = pf[21];
            slabs[(q4 + 2) * 129 + 80 + w4r] = pf[22];
            slabs[(q4 + 3) * 129 + 80 + w4r] = pf[23];
            slabs[q4 * 129 + 96 + w4r] = pf[24];
            slabs[(q4 + 1) * 129 + 96 + w4r] = pf[25];
            slabs[(q4 + 2) * 129 + 96 + w4r] = pf[26];
            slabs[(q4 + 3) * 129 + 96 + w4r] = pf[27];
            slabs[q4 * 129 + 112 + w4r] = pf[28];
            slabs[(q4 + 1) * 129 + 112 + w4r] = pf[29];
            slabs[(q4 + 2) * 129 + 112 + w4r] = pf[30];
            slabs[(q4 + 3) * 129 + 112 + w4r] = pf[31];
            __syncthreads();
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)(w4r * 128 + 96 + q4));
                pf[0 + 0] = _v4.x;
                pf[0 + 1] = _v4.y;
                pf[0 + 2] = _v4.z;
                pf[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((16 + w4r) * 128 + 96 + q4));
                pf[4 + 0] = _v4.x;
                pf[4 + 1] = _v4.y;
                pf[4 + 2] = _v4.z;
                pf[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((32 + w4r) * 128 + 96 + q4));
                pf[8 + 0] = _v4.x;
                pf[8 + 1] = _v4.y;
                pf[8 + 2] = _v4.z;
                pf[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((48 + w4r) * 128 + 96 + q4));
                pf[12 + 0] = _v4.x;
                pf[12 + 1] = _v4.y;
                pf[12 + 2] = _v4.z;
                pf[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((64 + w4r) * 128 + 96 + q4));
                pf[16 + 0] = _v4.x;
                pf[16 + 1] = _v4.y;
                pf[16 + 2] = _v4.z;
                pf[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((80 + w4r) * 128 + 96 + q4));
                pf[20 + 0] = _v4.x;
                pf[20 + 1] = _v4.y;
                pf[20 + 2] = _v4.z;
                pf[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((96 + w4r) * 128 + 96 + q4));
                pf[24 + 0] = _v4.x;
                pf[24 + 1] = _v4.y;
                pf[24 + 2] = _v4.z;
                pf[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(m + cur_1 + (long long)((112 + w4r) * 128 + 96 + q4));
                pf[28 + 0] = _v4.x;
                pf[28 + 1] = _v4.y;
                pf[28 + 2] = _v4.z;
                pf[28 + 3] = _v4.w;
            }
            #pragma unroll
            for (int kk_2 = 0; kk_2 < 32; kk_2++) {
                float mv_9 = slabs[kk_2 * 129 + k];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((64 + kk_2) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_9, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_9, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_9, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_9, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((64 + kk_2) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_9, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_9, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_9, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_9, xv[3], acc[7]);
            }
            slabs[4128 + q4 * 129 + w4r] = pf[0];
            slabs[4128 + (q4 + 1) * 129 + w4r] = pf[1];
            slabs[4128 + (q4 + 2) * 129 + w4r] = pf[2];
            slabs[4128 + (q4 + 3) * 129 + w4r] = pf[3];
            slabs[4128 + q4 * 129 + 16 + w4r] = pf[4];
            slabs[4128 + (q4 + 1) * 129 + 16 + w4r] = pf[5];
            slabs[4128 + (q4 + 2) * 129 + 16 + w4r] = pf[6];
            slabs[4128 + (q4 + 3) * 129 + 16 + w4r] = pf[7];
            slabs[4128 + q4 * 129 + 32 + w4r] = pf[8];
            slabs[4128 + (q4 + 1) * 129 + 32 + w4r] = pf[9];
            slabs[4128 + (q4 + 2) * 129 + 32 + w4r] = pf[10];
            slabs[4128 + (q4 + 3) * 129 + 32 + w4r] = pf[11];
            slabs[4128 + q4 * 129 + 48 + w4r] = pf[12];
            slabs[4128 + (q4 + 1) * 129 + 48 + w4r] = pf[13];
            slabs[4128 + (q4 + 2) * 129 + 48 + w4r] = pf[14];
            slabs[4128 + (q4 + 3) * 129 + 48 + w4r] = pf[15];
            slabs[4128 + q4 * 129 + 64 + w4r] = pf[16];
            slabs[4128 + (q4 + 1) * 129 + 64 + w4r] = pf[17];
            slabs[4128 + (q4 + 2) * 129 + 64 + w4r] = pf[18];
            slabs[4128 + (q4 + 3) * 129 + 64 + w4r] = pf[19];
            slabs[4128 + q4 * 129 + 80 + w4r] = pf[20];
            slabs[4128 + (q4 + 1) * 129 + 80 + w4r] = pf[21];
            slabs[4128 + (q4 + 2) * 129 + 80 + w4r] = pf[22];
            slabs[4128 + (q4 + 3) * 129 + 80 + w4r] = pf[23];
            slabs[4128 + q4 * 129 + 96 + w4r] = pf[24];
            slabs[4128 + (q4 + 1) * 129 + 96 + w4r] = pf[25];
            slabs[4128 + (q4 + 2) * 129 + 96 + w4r] = pf[26];
            slabs[4128 + (q4 + 3) * 129 + 96 + w4r] = pf[27];
            slabs[4128 + q4 * 129 + 112 + w4r] = pf[28];
            slabs[4128 + (q4 + 1) * 129 + 112 + w4r] = pf[29];
            slabs[4128 + (q4 + 2) * 129 + 112 + w4r] = pf[30];
            slabs[4128 + (q4 + 3) * 129 + 112 + w4r] = pf[31];
            __syncthreads();
            if (has_next_1 != 0) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)(w4r * 128 + q4));
                    pf[0 + 0] = _v4.x;
                    pf[0 + 1] = _v4.y;
                    pf[0 + 2] = _v4.z;
                    pf[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((16 + w4r) * 128 + q4));
                    pf[4 + 0] = _v4.x;
                    pf[4 + 1] = _v4.y;
                    pf[4 + 2] = _v4.z;
                    pf[4 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((32 + w4r) * 128 + q4));
                    pf[8 + 0] = _v4.x;
                    pf[8 + 1] = _v4.y;
                    pf[8 + 2] = _v4.z;
                    pf[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((48 + w4r) * 128 + q4));
                    pf[12 + 0] = _v4.x;
                    pf[12 + 1] = _v4.y;
                    pf[12 + 2] = _v4.z;
                    pf[12 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((64 + w4r) * 128 + q4));
                    pf[16 + 0] = _v4.x;
                    pf[16 + 1] = _v4.y;
                    pf[16 + 2] = _v4.z;
                    pf[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((80 + w4r) * 128 + q4));
                    pf[20 + 0] = _v4.x;
                    pf[20 + 1] = _v4.y;
                    pf[20 + 2] = _v4.z;
                    pf[20 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((96 + w4r) * 128 + q4));
                    pf[24 + 0] = _v4.x;
                    pf[24 + 1] = _v4.y;
                    pf[24 + 2] = _v4.z;
                    pf[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(m + nxt_1 + (long long)((112 + w4r) * 128 + q4));
                    pf[28 + 0] = _v4.x;
                    pf[28 + 1] = _v4.y;
                    pf[28 + 2] = _v4.z;
                    pf[28 + 3] = _v4.w;
                }
            }
            #pragma unroll
            for (int kk_3 = 0; kk_3 < 32; kk_3++) {
                float mv_10 = slabs[4128 + kk_3 * 129 + k];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)((96 + kk_3) * 8 * 4)));
                acc[0] = __fmaf_rn(mv_10, xv[0], acc[0]);
                acc[1] = __fmaf_rn(mv_10, xv[1], acc[1]);
                acc[2] = __fmaf_rn(mv_10, xv[2], acc[2]);
                acc[3] = __fmaf_rn(mv_10, xv[3], acc[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_block_addr + (unsigned int)(((96 + kk_3) * 8 + 4) * 4)));
                acc[4] = __fmaf_rn(mv_10, xv[0], acc[4]);
                acc[5] = __fmaf_rn(mv_10, xv[1], acc[5]);
                acc[6] = __fmaf_rn(mv_10, xv[2], acc[6]);
                acc[7] = __fmaf_rn(mv_10, xv[3], acc[7]);
            }
            if (has_next_1 != 0) {
                slabs[q4 * 129 + w4r] = pf[0];
                slabs[(q4 + 1) * 129 + w4r] = pf[1];
                slabs[(q4 + 2) * 129 + w4r] = pf[2];
                slabs[(q4 + 3) * 129 + w4r] = pf[3];
                slabs[q4 * 129 + 16 + w4r] = pf[4];
                slabs[(q4 + 1) * 129 + 16 + w4r] = pf[5];
                slabs[(q4 + 2) * 129 + 16 + w4r] = pf[6];
                slabs[(q4 + 3) * 129 + 16 + w4r] = pf[7];
                slabs[q4 * 129 + 32 + w4r] = pf[8];
                slabs[(q4 + 1) * 129 + 32 + w4r] = pf[9];
                slabs[(q4 + 2) * 129 + 32 + w4r] = pf[10];
                slabs[(q4 + 3) * 129 + 32 + w4r] = pf[11];
                slabs[q4 * 129 + 48 + w4r] = pf[12];
                slabs[(q4 + 1) * 129 + 48 + w4r] = pf[13];
                slabs[(q4 + 2) * 129 + 48 + w4r] = pf[14];
                slabs[(q4 + 3) * 129 + 48 + w4r] = pf[15];
                slabs[q4 * 129 + 64 + w4r] = pf[16];
                slabs[(q4 + 1) * 129 + 64 + w4r] = pf[17];
                slabs[(q4 + 2) * 129 + 64 + w4r] = pf[18];
                slabs[(q4 + 3) * 129 + 64 + w4r] = pf[19];
                slabs[q4 * 129 + 80 + w4r] = pf[20];
                slabs[(q4 + 1) * 129 + 80 + w4r] = pf[21];
                slabs[(q4 + 2) * 129 + 80 + w4r] = pf[22];
                slabs[(q4 + 3) * 129 + 80 + w4r] = pf[23];
                slabs[q4 * 129 + 96 + w4r] = pf[24];
                slabs[(q4 + 1) * 129 + 96 + w4r] = pf[25];
                slabs[(q4 + 2) * 129 + 96 + w4r] = pf[26];
                slabs[(q4 + 3) * 129 + 96 + w4r] = pf[27];
                slabs[q4 * 129 + 112 + w4r] = pf[28];
                slabs[(q4 + 1) * 129 + 112 + w4r] = pf[29];
                slabs[(q4 + 2) * 129 + 112 + w4r] = pf[30];
                slabs[(q4 + 3) * 129 + 112 + w4r] = pf[31];
            }
            __syncthreads();
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr + 16), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            if (has_next_1 != 0) {
                if (write_states != 0) {
                    long long y_next_1 = elem_y[e_1 + 1];
                    if (y_next_1 >= 0) {
                        {
                            float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                            *reinterpret_cast<float4*>(y + y_next_1 + row_off) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                            *reinterpret_cast<float4*>(y + y_next_1 + row_off + 4) = _v4;
                        }
                    }
                }
            } else {
                long long y_out_1 = y_final[chain];
                if (y_out_1 >= 0) {
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>(y + y_out_1 + row_off) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                        *reinterpret_cast<float4*>(y + y_out_1 + row_off + 4) = _v4;
                    }
                }
            }
            __syncthreads();
        }
    }
}

} // extern "C"
