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

__global__ __launch_bounds__(256) void
kernel_cake_gdn_train_df2a0e535ca9a0e8f86f(float* __restrict__ seg_state, float* __restrict__ init_state, float* __restrict__ state_out, int* __restrict__ seq_seg_start, int* __restrict__ seg_flag, int num_v_heads, int use_init)
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
    float* s_x = reinterpret_cast<float*>(smem_raw + 0);
    const int s_x_addr = smem + 0;

    // === Task calls (declaration order) ===
    int seq = blockIdx.x;
    int hv = blockIdx.y;
    int row = tid / 2;
    int half = tid % 2;
    int col = blockIdx.z * 16 + half * 8;
    long long nvh = (long long)num_v_heads;
    long long hv64 = (long long)hv;
    int first = seq_seg_start[seq];
    int last = seq_seg_start[seq + 1] - 1;
    float x[8];
    float acc[8];
    float xv[8];
    float m[4];
    float b[8];
    long long cell = (long long)row * 128ll + (long long)col;
    unsigned int slot = (unsigned int)((row * 16 + half * 8) * 4);
    if (use_init != 0) {
        long long init_base = ((long long)seq * nvh + hv64) * 16384ll + cell;
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base);
            x[0 + 0] = _v4.x;
            x[0 + 1] = _v4.y;
            x[0 + 2] = _v4.z;
            x[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + 4ll);
            x[4 + 0] = _v4.x;
            x[4 + 1] = _v4.y;
            x[4 + 2] = _v4.z;
            x[4 + 3] = _v4.w;
        }
    } else {
        x[0] = 0.0f;
        x[1] = 0.0f;
        x[2] = 0.0f;
        x[3] = 0.0f;
        x[4] = 0.0f;
        x[5] = 0.0f;
        x[6] = 0.0f;
        x[7] = 0.0f;
    }
    if (last >= first) {
        long long last_base = ((long long)last * nvh + hv64) * 16384ll + cell;
        {
            float4 _v4 = make_float4(x[0 + 0], x[0 + 1], x[0 + 2], x[0 + 3]);
            *reinterpret_cast<float4*>(state_out + last_base) = _v4;
        }
        {
            float4 _v4 = make_float4(x[4 + 0], x[4 + 1], x[4 + 2], x[4 + 3]);
            *reinterpret_cast<float4*>(state_out + last_base + 4ll) = _v4;
        }
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + slot), "r"(__float_as_uint(x[0])), "r"(__float_as_uint(x[1])), "r"(__float_as_uint(x[2])), "r"(__float_as_uint(x[3])) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (slot + 16)), "r"(__float_as_uint(x[4])), "r"(__float_as_uint(x[5])), "r"(__float_as_uint(x[6])), "r"(__float_as_uint(x[7])) : "memory");
    __syncthreads();
    #pragma unroll 1
    for (int i = 0; i < last - first; i++) {
        int s = last - i;
        unsigned int cur = (unsigned int)(i % 2) * 8192;
        unsigned int nxt = (unsigned int)((i + 1) % 2) * 8192;
        long long map_base = ((long long)s * nvh + hv64) * 32768ll + (long long)row * 256ll;
        int has_m = seg_flag[(long long)s * nvh + hv64];
        if (i == 0) {
            if (use_init == 0) {
                has_m = 0;
            }
        }
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        if (has_m != 0) {
            #pragma unroll
            for (int kc = 0; kc < 32; kc++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_base + (long long)(128 + kc * 4));
                    m[0 + 0] = _v4.x;
                    m[0 + 1] = _v4.y;
                    m[0 + 2] = _v4.z;
                    m[0 + 3] = _v4.w;
                }
                unsigned int x_row_0 = s_x_addr + cur + (unsigned int)((kc * 4 * 16 + half * 8) * 4);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_row_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                    : "r"(x_row_0 + 16));
                acc[0] = __fmaf_rn(m[0], xv[0], acc[0]);
                acc[1] = __fmaf_rn(m[0], xv[1], acc[1]);
                acc[2] = __fmaf_rn(m[0], xv[2], acc[2]);
                acc[3] = __fmaf_rn(m[0], xv[3], acc[3]);
                acc[4] = __fmaf_rn(m[0], xv[4], acc[4]);
                acc[5] = __fmaf_rn(m[0], xv[5], acc[5]);
                acc[6] = __fmaf_rn(m[0], xv[6], acc[6]);
                acc[7] = __fmaf_rn(m[0], xv[7], acc[7]);
                unsigned int x_row_1 = s_x_addr + cur + (unsigned int)(((kc * 4 + 1) * 16 + half * 8) * 4);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_row_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                    : "r"(x_row_1 + 16));
                acc[0] = __fmaf_rn(m[1], xv[0], acc[0]);
                acc[1] = __fmaf_rn(m[1], xv[1], acc[1]);
                acc[2] = __fmaf_rn(m[1], xv[2], acc[2]);
                acc[3] = __fmaf_rn(m[1], xv[3], acc[3]);
                acc[4] = __fmaf_rn(m[1], xv[4], acc[4]);
                acc[5] = __fmaf_rn(m[1], xv[5], acc[5]);
                acc[6] = __fmaf_rn(m[1], xv[6], acc[6]);
                acc[7] = __fmaf_rn(m[1], xv[7], acc[7]);
                unsigned int x_row_2 = s_x_addr + cur + (unsigned int)(((kc * 4 + 2) * 16 + half * 8) * 4);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_row_2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                    : "r"(x_row_2 + 16));
                acc[0] = __fmaf_rn(m[2], xv[0], acc[0]);
                acc[1] = __fmaf_rn(m[2], xv[1], acc[1]);
                acc[2] = __fmaf_rn(m[2], xv[2], acc[2]);
                acc[3] = __fmaf_rn(m[2], xv[3], acc[3]);
                acc[4] = __fmaf_rn(m[2], xv[4], acc[4]);
                acc[5] = __fmaf_rn(m[2], xv[5], acc[5]);
                acc[6] = __fmaf_rn(m[2], xv[6], acc[6]);
                acc[7] = __fmaf_rn(m[2], xv[7], acc[7]);
                unsigned int x_row_3 = s_x_addr + cur + (unsigned int)(((kc * 4 + 3) * 16 + half * 8) * 4);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                    : "r"(x_row_3));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                    : "r"(x_row_3 + 16));
                acc[0] = __fmaf_rn(m[3], xv[0], acc[0]);
                acc[1] = __fmaf_rn(m[3], xv[1], acc[1]);
                acc[2] = __fmaf_rn(m[3], xv[2], acc[2]);
                acc[3] = __fmaf_rn(m[3], xv[3], acc[3]);
                acc[4] = __fmaf_rn(m[3], xv[4], acc[4]);
                acc[5] = __fmaf_rn(m[3], xv[5], acc[5]);
                acc[6] = __fmaf_rn(m[3], xv[6], acc[6]);
                acc[7] = __fmaf_rn(m[3], xv[7], acc[7]);
            }
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_base + (long long)col);
            b[0 + 0] = _v4.x;
            b[0 + 1] = _v4.y;
            b[0 + 2] = _v4.z;
            b[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_base + (long long)(col + 4));
            b[4 + 0] = _v4.x;
            b[4 + 1] = _v4.y;
            b[4 + 2] = _v4.z;
            b[4 + 3] = _v4.w;
        }
        x[0] = acc[0] + b[0];
        x[1] = acc[1] + b[1];
        x[2] = acc[2] + b[2];
        x[3] = acc[3] + b[3];
        x[4] = acc[4] + b[4];
        x[5] = acc[5] + b[5];
        x[6] = acc[6] + b[6];
        x[7] = acc[7] + b[7];
        long long out_base = ((long long)(s - 1) * nvh + hv64) * 16384ll + cell;
        {
            float4 _v4 = make_float4(x[0 + 0], x[0 + 1], x[0 + 2], x[0 + 3]);
            *reinterpret_cast<float4*>(state_out + out_base) = _v4;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + nxt + slot), "r"(__float_as_uint(x[0])), "r"(__float_as_uint(x[1])), "r"(__float_as_uint(x[2])), "r"(__float_as_uint(x[3])) : "memory");
        {
            float4 _v4 = make_float4(x[4 + 0], x[4 + 1], x[4 + 2], x[4 + 3]);
            *reinterpret_cast<float4*>(state_out + out_base + 4ll) = _v4;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + nxt + (slot + 16)), "r"(__float_as_uint(x[4])), "r"(__float_as_uint(x[5])), "r"(__float_as_uint(x[6])), "r"(__float_as_uint(x[7])) : "memory");
        __syncthreads();
    }
}

} // extern "C"
