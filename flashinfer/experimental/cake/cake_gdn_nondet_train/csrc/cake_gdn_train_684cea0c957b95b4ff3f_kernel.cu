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
kernel_cake_gdn_train_684cea0c957b95b4ff3f(float* __restrict__ seg_state, float* __restrict__ init_state, float* __restrict__ state_out, int* __restrict__ seq_seg_start, int* __restrict__ seg_flag, int num_v_heads, int use_init)
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
    float mrow[128];
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
    int steps = last - first;
    if (steps > 0) {
        long long map_first = ((long long)last * nvh + hv64) * 32768ll + (long long)row * 256ll;
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 128ll);
            mrow[0 + 0] = _v4.x;
            mrow[0 + 1] = _v4.y;
            mrow[0 + 2] = _v4.z;
            mrow[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 132ll);
            mrow[4 + 0] = _v4.x;
            mrow[4 + 1] = _v4.y;
            mrow[4 + 2] = _v4.z;
            mrow[4 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 136ll);
            mrow[8 + 0] = _v4.x;
            mrow[8 + 1] = _v4.y;
            mrow[8 + 2] = _v4.z;
            mrow[8 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 140ll);
            mrow[12 + 0] = _v4.x;
            mrow[12 + 1] = _v4.y;
            mrow[12 + 2] = _v4.z;
            mrow[12 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 144ll);
            mrow[16 + 0] = _v4.x;
            mrow[16 + 1] = _v4.y;
            mrow[16 + 2] = _v4.z;
            mrow[16 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 148ll);
            mrow[20 + 0] = _v4.x;
            mrow[20 + 1] = _v4.y;
            mrow[20 + 2] = _v4.z;
            mrow[20 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 152ll);
            mrow[24 + 0] = _v4.x;
            mrow[24 + 1] = _v4.y;
            mrow[24 + 2] = _v4.z;
            mrow[24 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 156ll);
            mrow[28 + 0] = _v4.x;
            mrow[28 + 1] = _v4.y;
            mrow[28 + 2] = _v4.z;
            mrow[28 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 160ll);
            mrow[32 + 0] = _v4.x;
            mrow[32 + 1] = _v4.y;
            mrow[32 + 2] = _v4.z;
            mrow[32 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 164ll);
            mrow[36 + 0] = _v4.x;
            mrow[36 + 1] = _v4.y;
            mrow[36 + 2] = _v4.z;
            mrow[36 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 168ll);
            mrow[40 + 0] = _v4.x;
            mrow[40 + 1] = _v4.y;
            mrow[40 + 2] = _v4.z;
            mrow[40 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 172ll);
            mrow[44 + 0] = _v4.x;
            mrow[44 + 1] = _v4.y;
            mrow[44 + 2] = _v4.z;
            mrow[44 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 176ll);
            mrow[48 + 0] = _v4.x;
            mrow[48 + 1] = _v4.y;
            mrow[48 + 2] = _v4.z;
            mrow[48 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 180ll);
            mrow[52 + 0] = _v4.x;
            mrow[52 + 1] = _v4.y;
            mrow[52 + 2] = _v4.z;
            mrow[52 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 184ll);
            mrow[56 + 0] = _v4.x;
            mrow[56 + 1] = _v4.y;
            mrow[56 + 2] = _v4.z;
            mrow[56 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 188ll);
            mrow[60 + 0] = _v4.x;
            mrow[60 + 1] = _v4.y;
            mrow[60 + 2] = _v4.z;
            mrow[60 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 192ll);
            mrow[64 + 0] = _v4.x;
            mrow[64 + 1] = _v4.y;
            mrow[64 + 2] = _v4.z;
            mrow[64 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 196ll);
            mrow[68 + 0] = _v4.x;
            mrow[68 + 1] = _v4.y;
            mrow[68 + 2] = _v4.z;
            mrow[68 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 200ll);
            mrow[72 + 0] = _v4.x;
            mrow[72 + 1] = _v4.y;
            mrow[72 + 2] = _v4.z;
            mrow[72 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 204ll);
            mrow[76 + 0] = _v4.x;
            mrow[76 + 1] = _v4.y;
            mrow[76 + 2] = _v4.z;
            mrow[76 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 208ll);
            mrow[80 + 0] = _v4.x;
            mrow[80 + 1] = _v4.y;
            mrow[80 + 2] = _v4.z;
            mrow[80 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 212ll);
            mrow[84 + 0] = _v4.x;
            mrow[84 + 1] = _v4.y;
            mrow[84 + 2] = _v4.z;
            mrow[84 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 216ll);
            mrow[88 + 0] = _v4.x;
            mrow[88 + 1] = _v4.y;
            mrow[88 + 2] = _v4.z;
            mrow[88 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 220ll);
            mrow[92 + 0] = _v4.x;
            mrow[92 + 1] = _v4.y;
            mrow[92 + 2] = _v4.z;
            mrow[92 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 224ll);
            mrow[96 + 0] = _v4.x;
            mrow[96 + 1] = _v4.y;
            mrow[96 + 2] = _v4.z;
            mrow[96 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 228ll);
            mrow[100 + 0] = _v4.x;
            mrow[100 + 1] = _v4.y;
            mrow[100 + 2] = _v4.z;
            mrow[100 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 232ll);
            mrow[104 + 0] = _v4.x;
            mrow[104 + 1] = _v4.y;
            mrow[104 + 2] = _v4.z;
            mrow[104 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 236ll);
            mrow[108 + 0] = _v4.x;
            mrow[108 + 1] = _v4.y;
            mrow[108 + 2] = _v4.z;
            mrow[108 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 240ll);
            mrow[112 + 0] = _v4.x;
            mrow[112 + 1] = _v4.y;
            mrow[112 + 2] = _v4.z;
            mrow[112 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 244ll);
            mrow[116 + 0] = _v4.x;
            mrow[116 + 1] = _v4.y;
            mrow[116 + 2] = _v4.z;
            mrow[116 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 248ll);
            mrow[120 + 0] = _v4.x;
            mrow[120 + 1] = _v4.y;
            mrow[120 + 2] = _v4.z;
            mrow[120 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + 252ll);
            mrow[124 + 0] = _v4.x;
            mrow[124 + 1] = _v4.y;
            mrow[124 + 2] = _v4.z;
            mrow[124 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + (long long)col);
            b[0 + 0] = _v4.x;
            b[0 + 1] = _v4.y;
            b[0 + 2] = _v4.z;
            b[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_first + (long long)(col + 4));
            b[4 + 0] = _v4.x;
            b[4 + 1] = _v4.y;
            b[4 + 2] = _v4.z;
            b[4 + 3] = _v4.w;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int i = 0; i < steps; i++) {
        int s = last - i;
        unsigned int cur = (unsigned int)(i % 2) * 8192;
        unsigned int nxt = (unsigned int)((i + 1) % 2) * 8192;
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
            unsigned int x_row_0 = s_x_addr + cur + (unsigned int)(half * 8 * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_0));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_0 + 16));
            acc[0] = __fmaf_rn(mrow[0], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[0], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[0], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[0], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[0], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[0], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[0], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[0], xv[7], acc[7]);
            unsigned int x_row_1 = s_x_addr + cur + (unsigned int)((16 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_1));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_1 + 16));
            acc[0] = __fmaf_rn(mrow[1], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[1], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[1], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[1], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[1], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[1], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[1], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[1], xv[7], acc[7]);
            unsigned int x_row_2 = s_x_addr + cur + (unsigned int)((32 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_2));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_2 + 16));
            acc[0] = __fmaf_rn(mrow[2], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[2], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[2], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[2], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[2], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[2], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[2], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[2], xv[7], acc[7]);
            unsigned int x_row_3 = s_x_addr + cur + (unsigned int)((48 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_3));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_3 + 16));
            acc[0] = __fmaf_rn(mrow[3], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[3], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[3], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[3], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[3], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[3], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[3], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[3], xv[7], acc[7]);
            unsigned int x_row_4 = s_x_addr + cur + (unsigned int)((64 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_4));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_4 + 16));
            acc[0] = __fmaf_rn(mrow[4], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[4], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[4], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[4], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[4], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[4], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[4], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[4], xv[7], acc[7]);
            unsigned int x_row_5 = s_x_addr + cur + (unsigned int)((80 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_5));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_5 + 16));
            acc[0] = __fmaf_rn(mrow[5], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[5], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[5], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[5], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[5], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[5], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[5], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[5], xv[7], acc[7]);
            unsigned int x_row_6 = s_x_addr + cur + (unsigned int)((96 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_6));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_6 + 16));
            acc[0] = __fmaf_rn(mrow[6], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[6], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[6], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[6], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[6], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[6], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[6], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[6], xv[7], acc[7]);
            unsigned int x_row_7 = s_x_addr + cur + (unsigned int)((112 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_7));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_7 + 16));
            acc[0] = __fmaf_rn(mrow[7], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[7], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[7], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[7], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[7], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[7], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[7], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[7], xv[7], acc[7]);
            unsigned int x_row_8 = s_x_addr + cur + (unsigned int)((128 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_8));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_8 + 16));
            acc[0] = __fmaf_rn(mrow[8], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[8], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[8], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[8], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[8], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[8], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[8], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[8], xv[7], acc[7]);
            unsigned int x_row_9 = s_x_addr + cur + (unsigned int)((144 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_9));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_9 + 16));
            acc[0] = __fmaf_rn(mrow[9], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[9], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[9], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[9], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[9], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[9], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[9], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[9], xv[7], acc[7]);
            unsigned int x_row_10 = s_x_addr + cur + (unsigned int)((160 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_10));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_10 + 16));
            acc[0] = __fmaf_rn(mrow[10], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[10], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[10], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[10], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[10], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[10], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[10], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[10], xv[7], acc[7]);
            unsigned int x_row_11 = s_x_addr + cur + (unsigned int)((176 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_11));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_11 + 16));
            acc[0] = __fmaf_rn(mrow[11], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[11], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[11], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[11], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[11], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[11], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[11], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[11], xv[7], acc[7]);
            unsigned int x_row_12 = s_x_addr + cur + (unsigned int)((192 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_12));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_12 + 16));
            acc[0] = __fmaf_rn(mrow[12], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[12], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[12], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[12], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[12], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[12], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[12], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[12], xv[7], acc[7]);
            unsigned int x_row_13 = s_x_addr + cur + (unsigned int)((208 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_13));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_13 + 16));
            acc[0] = __fmaf_rn(mrow[13], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[13], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[13], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[13], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[13], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[13], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[13], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[13], xv[7], acc[7]);
            unsigned int x_row_14 = s_x_addr + cur + (unsigned int)((224 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_14));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_14 + 16));
            acc[0] = __fmaf_rn(mrow[14], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[14], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[14], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[14], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[14], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[14], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[14], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[14], xv[7], acc[7]);
            unsigned int x_row_15 = s_x_addr + cur + (unsigned int)((240 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_15));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_15 + 16));
            acc[0] = __fmaf_rn(mrow[15], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[15], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[15], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[15], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[15], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[15], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[15], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[15], xv[7], acc[7]);
            unsigned int x_row_16 = s_x_addr + cur + (unsigned int)((256 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_16));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_16 + 16));
            acc[0] = __fmaf_rn(mrow[16], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[16], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[16], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[16], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[16], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[16], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[16], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[16], xv[7], acc[7]);
            unsigned int x_row_17 = s_x_addr + cur + (unsigned int)((272 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_17));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_17 + 16));
            acc[0] = __fmaf_rn(mrow[17], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[17], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[17], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[17], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[17], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[17], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[17], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[17], xv[7], acc[7]);
            unsigned int x_row_18 = s_x_addr + cur + (unsigned int)((288 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_18));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_18 + 16));
            acc[0] = __fmaf_rn(mrow[18], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[18], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[18], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[18], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[18], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[18], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[18], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[18], xv[7], acc[7]);
            unsigned int x_row_19 = s_x_addr + cur + (unsigned int)((304 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_19));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_19 + 16));
            acc[0] = __fmaf_rn(mrow[19], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[19], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[19], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[19], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[19], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[19], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[19], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[19], xv[7], acc[7]);
            unsigned int x_row_20 = s_x_addr + cur + (unsigned int)((320 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_20));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_20 + 16));
            acc[0] = __fmaf_rn(mrow[20], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[20], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[20], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[20], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[20], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[20], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[20], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[20], xv[7], acc[7]);
            unsigned int x_row_21 = s_x_addr + cur + (unsigned int)((336 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_21));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_21 + 16));
            acc[0] = __fmaf_rn(mrow[21], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[21], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[21], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[21], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[21], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[21], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[21], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[21], xv[7], acc[7]);
            unsigned int x_row_22 = s_x_addr + cur + (unsigned int)((352 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_22));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_22 + 16));
            acc[0] = __fmaf_rn(mrow[22], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[22], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[22], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[22], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[22], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[22], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[22], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[22], xv[7], acc[7]);
            unsigned int x_row_23 = s_x_addr + cur + (unsigned int)((368 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_23));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_23 + 16));
            acc[0] = __fmaf_rn(mrow[23], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[23], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[23], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[23], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[23], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[23], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[23], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[23], xv[7], acc[7]);
            unsigned int x_row_24 = s_x_addr + cur + (unsigned int)((384 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_24));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_24 + 16));
            acc[0] = __fmaf_rn(mrow[24], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[24], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[24], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[24], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[24], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[24], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[24], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[24], xv[7], acc[7]);
            unsigned int x_row_25 = s_x_addr + cur + (unsigned int)((400 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_25));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_25 + 16));
            acc[0] = __fmaf_rn(mrow[25], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[25], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[25], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[25], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[25], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[25], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[25], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[25], xv[7], acc[7]);
            unsigned int x_row_26 = s_x_addr + cur + (unsigned int)((416 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_26));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_26 + 16));
            acc[0] = __fmaf_rn(mrow[26], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[26], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[26], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[26], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[26], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[26], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[26], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[26], xv[7], acc[7]);
            unsigned int x_row_27 = s_x_addr + cur + (unsigned int)((432 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_27));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_27 + 16));
            acc[0] = __fmaf_rn(mrow[27], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[27], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[27], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[27], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[27], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[27], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[27], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[27], xv[7], acc[7]);
            unsigned int x_row_28 = s_x_addr + cur + (unsigned int)((448 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_28));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_28 + 16));
            acc[0] = __fmaf_rn(mrow[28], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[28], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[28], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[28], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[28], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[28], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[28], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[28], xv[7], acc[7]);
            unsigned int x_row_29 = s_x_addr + cur + (unsigned int)((464 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_29));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_29 + 16));
            acc[0] = __fmaf_rn(mrow[29], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[29], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[29], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[29], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[29], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[29], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[29], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[29], xv[7], acc[7]);
            unsigned int x_row_30 = s_x_addr + cur + (unsigned int)((480 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_30));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_30 + 16));
            acc[0] = __fmaf_rn(mrow[30], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[30], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[30], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[30], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[30], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[30], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[30], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[30], xv[7], acc[7]);
            unsigned int x_row_31 = s_x_addr + cur + (unsigned int)((496 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_31));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_31 + 16));
            acc[0] = __fmaf_rn(mrow[31], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[31], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[31], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[31], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[31], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[31], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[31], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[31], xv[7], acc[7]);
            unsigned int x_row_32 = s_x_addr + cur + (unsigned int)((512 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_32));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_32 + 16));
            acc[0] = __fmaf_rn(mrow[32], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[32], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[32], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[32], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[32], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[32], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[32], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[32], xv[7], acc[7]);
            unsigned int x_row_33 = s_x_addr + cur + (unsigned int)((528 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_33));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_33 + 16));
            acc[0] = __fmaf_rn(mrow[33], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[33], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[33], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[33], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[33], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[33], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[33], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[33], xv[7], acc[7]);
            unsigned int x_row_34 = s_x_addr + cur + (unsigned int)((544 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_34));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_34 + 16));
            acc[0] = __fmaf_rn(mrow[34], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[34], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[34], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[34], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[34], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[34], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[34], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[34], xv[7], acc[7]);
            unsigned int x_row_35 = s_x_addr + cur + (unsigned int)((560 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_35));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_35 + 16));
            acc[0] = __fmaf_rn(mrow[35], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[35], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[35], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[35], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[35], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[35], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[35], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[35], xv[7], acc[7]);
            unsigned int x_row_36 = s_x_addr + cur + (unsigned int)((576 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_36));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_36 + 16));
            acc[0] = __fmaf_rn(mrow[36], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[36], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[36], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[36], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[36], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[36], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[36], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[36], xv[7], acc[7]);
            unsigned int x_row_37 = s_x_addr + cur + (unsigned int)((592 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_37));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_37 + 16));
            acc[0] = __fmaf_rn(mrow[37], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[37], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[37], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[37], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[37], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[37], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[37], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[37], xv[7], acc[7]);
            unsigned int x_row_38 = s_x_addr + cur + (unsigned int)((608 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_38));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_38 + 16));
            acc[0] = __fmaf_rn(mrow[38], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[38], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[38], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[38], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[38], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[38], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[38], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[38], xv[7], acc[7]);
            unsigned int x_row_39 = s_x_addr + cur + (unsigned int)((624 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_39));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_39 + 16));
            acc[0] = __fmaf_rn(mrow[39], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[39], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[39], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[39], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[39], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[39], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[39], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[39], xv[7], acc[7]);
            unsigned int x_row_40 = s_x_addr + cur + (unsigned int)((640 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_40));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_40 + 16));
            acc[0] = __fmaf_rn(mrow[40], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[40], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[40], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[40], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[40], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[40], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[40], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[40], xv[7], acc[7]);
            unsigned int x_row_41 = s_x_addr + cur + (unsigned int)((656 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_41));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_41 + 16));
            acc[0] = __fmaf_rn(mrow[41], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[41], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[41], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[41], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[41], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[41], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[41], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[41], xv[7], acc[7]);
            unsigned int x_row_42 = s_x_addr + cur + (unsigned int)((672 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_42));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_42 + 16));
            acc[0] = __fmaf_rn(mrow[42], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[42], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[42], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[42], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[42], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[42], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[42], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[42], xv[7], acc[7]);
            unsigned int x_row_43 = s_x_addr + cur + (unsigned int)((688 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_43));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_43 + 16));
            acc[0] = __fmaf_rn(mrow[43], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[43], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[43], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[43], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[43], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[43], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[43], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[43], xv[7], acc[7]);
            unsigned int x_row_44 = s_x_addr + cur + (unsigned int)((704 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_44));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_44 + 16));
            acc[0] = __fmaf_rn(mrow[44], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[44], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[44], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[44], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[44], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[44], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[44], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[44], xv[7], acc[7]);
            unsigned int x_row_45 = s_x_addr + cur + (unsigned int)((720 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_45));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_45 + 16));
            acc[0] = __fmaf_rn(mrow[45], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[45], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[45], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[45], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[45], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[45], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[45], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[45], xv[7], acc[7]);
            unsigned int x_row_46 = s_x_addr + cur + (unsigned int)((736 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_46));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_46 + 16));
            acc[0] = __fmaf_rn(mrow[46], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[46], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[46], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[46], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[46], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[46], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[46], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[46], xv[7], acc[7]);
            unsigned int x_row_47 = s_x_addr + cur + (unsigned int)((752 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_47));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_47 + 16));
            acc[0] = __fmaf_rn(mrow[47], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[47], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[47], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[47], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[47], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[47], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[47], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[47], xv[7], acc[7]);
            unsigned int x_row_48 = s_x_addr + cur + (unsigned int)((768 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_48));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_48 + 16));
            acc[0] = __fmaf_rn(mrow[48], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[48], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[48], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[48], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[48], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[48], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[48], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[48], xv[7], acc[7]);
            unsigned int x_row_49 = s_x_addr + cur + (unsigned int)((784 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_49));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_49 + 16));
            acc[0] = __fmaf_rn(mrow[49], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[49], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[49], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[49], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[49], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[49], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[49], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[49], xv[7], acc[7]);
            unsigned int x_row_50 = s_x_addr + cur + (unsigned int)((800 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_50));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_50 + 16));
            acc[0] = __fmaf_rn(mrow[50], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[50], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[50], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[50], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[50], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[50], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[50], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[50], xv[7], acc[7]);
            unsigned int x_row_51 = s_x_addr + cur + (unsigned int)((816 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_51));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_51 + 16));
            acc[0] = __fmaf_rn(mrow[51], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[51], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[51], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[51], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[51], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[51], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[51], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[51], xv[7], acc[7]);
            unsigned int x_row_52 = s_x_addr + cur + (unsigned int)((832 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_52));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_52 + 16));
            acc[0] = __fmaf_rn(mrow[52], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[52], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[52], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[52], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[52], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[52], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[52], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[52], xv[7], acc[7]);
            unsigned int x_row_53 = s_x_addr + cur + (unsigned int)((848 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_53));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_53 + 16));
            acc[0] = __fmaf_rn(mrow[53], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[53], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[53], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[53], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[53], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[53], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[53], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[53], xv[7], acc[7]);
            unsigned int x_row_54 = s_x_addr + cur + (unsigned int)((864 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_54));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_54 + 16));
            acc[0] = __fmaf_rn(mrow[54], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[54], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[54], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[54], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[54], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[54], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[54], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[54], xv[7], acc[7]);
            unsigned int x_row_55 = s_x_addr + cur + (unsigned int)((880 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_55));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_55 + 16));
            acc[0] = __fmaf_rn(mrow[55], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[55], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[55], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[55], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[55], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[55], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[55], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[55], xv[7], acc[7]);
            unsigned int x_row_56 = s_x_addr + cur + (unsigned int)((896 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_56));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_56 + 16));
            acc[0] = __fmaf_rn(mrow[56], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[56], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[56], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[56], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[56], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[56], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[56], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[56], xv[7], acc[7]);
            unsigned int x_row_57 = s_x_addr + cur + (unsigned int)((912 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_57));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_57 + 16));
            acc[0] = __fmaf_rn(mrow[57], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[57], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[57], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[57], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[57], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[57], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[57], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[57], xv[7], acc[7]);
            unsigned int x_row_58 = s_x_addr + cur + (unsigned int)((928 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_58));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_58 + 16));
            acc[0] = __fmaf_rn(mrow[58], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[58], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[58], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[58], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[58], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[58], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[58], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[58], xv[7], acc[7]);
            unsigned int x_row_59 = s_x_addr + cur + (unsigned int)((944 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_59));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_59 + 16));
            acc[0] = __fmaf_rn(mrow[59], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[59], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[59], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[59], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[59], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[59], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[59], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[59], xv[7], acc[7]);
            unsigned int x_row_60 = s_x_addr + cur + (unsigned int)((960 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_60));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_60 + 16));
            acc[0] = __fmaf_rn(mrow[60], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[60], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[60], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[60], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[60], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[60], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[60], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[60], xv[7], acc[7]);
            unsigned int x_row_61 = s_x_addr + cur + (unsigned int)((976 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_61));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_61 + 16));
            acc[0] = __fmaf_rn(mrow[61], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[61], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[61], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[61], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[61], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[61], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[61], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[61], xv[7], acc[7]);
            unsigned int x_row_62 = s_x_addr + cur + (unsigned int)((992 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_62));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_62 + 16));
            acc[0] = __fmaf_rn(mrow[62], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[62], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[62], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[62], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[62], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[62], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[62], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[62], xv[7], acc[7]);
            unsigned int x_row_63 = s_x_addr + cur + (unsigned int)((1008 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_63));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_63 + 16));
            acc[0] = __fmaf_rn(mrow[63], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[63], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[63], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[63], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[63], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[63], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[63], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[63], xv[7], acc[7]);
            unsigned int x_row_64 = s_x_addr + cur + (unsigned int)((1024 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_64));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_64 + 16));
            acc[0] = __fmaf_rn(mrow[64], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[64], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[64], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[64], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[64], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[64], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[64], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[64], xv[7], acc[7]);
            unsigned int x_row_65 = s_x_addr + cur + (unsigned int)((1040 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_65));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_65 + 16));
            acc[0] = __fmaf_rn(mrow[65], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[65], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[65], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[65], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[65], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[65], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[65], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[65], xv[7], acc[7]);
            unsigned int x_row_66 = s_x_addr + cur + (unsigned int)((1056 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_66));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_66 + 16));
            acc[0] = __fmaf_rn(mrow[66], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[66], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[66], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[66], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[66], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[66], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[66], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[66], xv[7], acc[7]);
            unsigned int x_row_67 = s_x_addr + cur + (unsigned int)((1072 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_67));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_67 + 16));
            acc[0] = __fmaf_rn(mrow[67], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[67], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[67], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[67], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[67], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[67], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[67], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[67], xv[7], acc[7]);
            unsigned int x_row_68 = s_x_addr + cur + (unsigned int)((1088 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_68));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_68 + 16));
            acc[0] = __fmaf_rn(mrow[68], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[68], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[68], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[68], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[68], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[68], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[68], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[68], xv[7], acc[7]);
            unsigned int x_row_69 = s_x_addr + cur + (unsigned int)((1104 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_69));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_69 + 16));
            acc[0] = __fmaf_rn(mrow[69], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[69], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[69], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[69], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[69], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[69], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[69], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[69], xv[7], acc[7]);
            unsigned int x_row_70 = s_x_addr + cur + (unsigned int)((1120 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_70));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_70 + 16));
            acc[0] = __fmaf_rn(mrow[70], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[70], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[70], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[70], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[70], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[70], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[70], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[70], xv[7], acc[7]);
            unsigned int x_row_71 = s_x_addr + cur + (unsigned int)((1136 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_71));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_71 + 16));
            acc[0] = __fmaf_rn(mrow[71], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[71], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[71], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[71], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[71], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[71], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[71], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[71], xv[7], acc[7]);
            unsigned int x_row_72 = s_x_addr + cur + (unsigned int)((1152 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_72));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_72 + 16));
            acc[0] = __fmaf_rn(mrow[72], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[72], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[72], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[72], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[72], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[72], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[72], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[72], xv[7], acc[7]);
            unsigned int x_row_73 = s_x_addr + cur + (unsigned int)((1168 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_73));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_73 + 16));
            acc[0] = __fmaf_rn(mrow[73], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[73], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[73], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[73], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[73], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[73], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[73], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[73], xv[7], acc[7]);
            unsigned int x_row_74 = s_x_addr + cur + (unsigned int)((1184 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_74));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_74 + 16));
            acc[0] = __fmaf_rn(mrow[74], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[74], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[74], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[74], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[74], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[74], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[74], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[74], xv[7], acc[7]);
            unsigned int x_row_75 = s_x_addr + cur + (unsigned int)((1200 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_75));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_75 + 16));
            acc[0] = __fmaf_rn(mrow[75], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[75], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[75], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[75], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[75], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[75], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[75], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[75], xv[7], acc[7]);
            unsigned int x_row_76 = s_x_addr + cur + (unsigned int)((1216 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_76));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_76 + 16));
            acc[0] = __fmaf_rn(mrow[76], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[76], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[76], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[76], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[76], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[76], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[76], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[76], xv[7], acc[7]);
            unsigned int x_row_77 = s_x_addr + cur + (unsigned int)((1232 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_77));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_77 + 16));
            acc[0] = __fmaf_rn(mrow[77], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[77], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[77], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[77], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[77], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[77], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[77], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[77], xv[7], acc[7]);
            unsigned int x_row_78 = s_x_addr + cur + (unsigned int)((1248 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_78));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_78 + 16));
            acc[0] = __fmaf_rn(mrow[78], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[78], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[78], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[78], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[78], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[78], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[78], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[78], xv[7], acc[7]);
            unsigned int x_row_79 = s_x_addr + cur + (unsigned int)((1264 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_79));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_79 + 16));
            acc[0] = __fmaf_rn(mrow[79], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[79], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[79], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[79], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[79], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[79], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[79], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[79], xv[7], acc[7]);
            unsigned int x_row_80 = s_x_addr + cur + (unsigned int)((1280 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_80));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_80 + 16));
            acc[0] = __fmaf_rn(mrow[80], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[80], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[80], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[80], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[80], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[80], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[80], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[80], xv[7], acc[7]);
            unsigned int x_row_81 = s_x_addr + cur + (unsigned int)((1296 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_81));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_81 + 16));
            acc[0] = __fmaf_rn(mrow[81], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[81], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[81], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[81], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[81], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[81], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[81], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[81], xv[7], acc[7]);
            unsigned int x_row_82 = s_x_addr + cur + (unsigned int)((1312 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_82));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_82 + 16));
            acc[0] = __fmaf_rn(mrow[82], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[82], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[82], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[82], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[82], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[82], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[82], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[82], xv[7], acc[7]);
            unsigned int x_row_83 = s_x_addr + cur + (unsigned int)((1328 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_83));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_83 + 16));
            acc[0] = __fmaf_rn(mrow[83], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[83], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[83], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[83], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[83], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[83], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[83], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[83], xv[7], acc[7]);
            unsigned int x_row_84 = s_x_addr + cur + (unsigned int)((1344 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_84));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_84 + 16));
            acc[0] = __fmaf_rn(mrow[84], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[84], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[84], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[84], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[84], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[84], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[84], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[84], xv[7], acc[7]);
            unsigned int x_row_85 = s_x_addr + cur + (unsigned int)((1360 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_85));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_85 + 16));
            acc[0] = __fmaf_rn(mrow[85], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[85], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[85], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[85], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[85], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[85], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[85], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[85], xv[7], acc[7]);
            unsigned int x_row_86 = s_x_addr + cur + (unsigned int)((1376 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_86));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_86 + 16));
            acc[0] = __fmaf_rn(mrow[86], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[86], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[86], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[86], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[86], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[86], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[86], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[86], xv[7], acc[7]);
            unsigned int x_row_87 = s_x_addr + cur + (unsigned int)((1392 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_87));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_87 + 16));
            acc[0] = __fmaf_rn(mrow[87], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[87], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[87], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[87], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[87], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[87], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[87], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[87], xv[7], acc[7]);
            unsigned int x_row_88 = s_x_addr + cur + (unsigned int)((1408 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_88));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_88 + 16));
            acc[0] = __fmaf_rn(mrow[88], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[88], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[88], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[88], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[88], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[88], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[88], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[88], xv[7], acc[7]);
            unsigned int x_row_89 = s_x_addr + cur + (unsigned int)((1424 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_89));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_89 + 16));
            acc[0] = __fmaf_rn(mrow[89], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[89], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[89], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[89], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[89], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[89], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[89], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[89], xv[7], acc[7]);
            unsigned int x_row_90 = s_x_addr + cur + (unsigned int)((1440 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_90));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_90 + 16));
            acc[0] = __fmaf_rn(mrow[90], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[90], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[90], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[90], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[90], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[90], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[90], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[90], xv[7], acc[7]);
            unsigned int x_row_91 = s_x_addr + cur + (unsigned int)((1456 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_91));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_91 + 16));
            acc[0] = __fmaf_rn(mrow[91], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[91], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[91], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[91], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[91], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[91], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[91], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[91], xv[7], acc[7]);
            unsigned int x_row_92 = s_x_addr + cur + (unsigned int)((1472 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_92));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_92 + 16));
            acc[0] = __fmaf_rn(mrow[92], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[92], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[92], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[92], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[92], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[92], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[92], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[92], xv[7], acc[7]);
            unsigned int x_row_93 = s_x_addr + cur + (unsigned int)((1488 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_93));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_93 + 16));
            acc[0] = __fmaf_rn(mrow[93], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[93], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[93], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[93], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[93], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[93], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[93], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[93], xv[7], acc[7]);
            unsigned int x_row_94 = s_x_addr + cur + (unsigned int)((1504 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_94));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_94 + 16));
            acc[0] = __fmaf_rn(mrow[94], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[94], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[94], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[94], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[94], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[94], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[94], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[94], xv[7], acc[7]);
            unsigned int x_row_95 = s_x_addr + cur + (unsigned int)((1520 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_95));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_95 + 16));
            acc[0] = __fmaf_rn(mrow[95], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[95], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[95], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[95], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[95], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[95], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[95], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[95], xv[7], acc[7]);
            unsigned int x_row_96 = s_x_addr + cur + (unsigned int)((1536 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_96));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_96 + 16));
            acc[0] = __fmaf_rn(mrow[96], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[96], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[96], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[96], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[96], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[96], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[96], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[96], xv[7], acc[7]);
            unsigned int x_row_97 = s_x_addr + cur + (unsigned int)((1552 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_97));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_97 + 16));
            acc[0] = __fmaf_rn(mrow[97], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[97], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[97], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[97], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[97], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[97], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[97], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[97], xv[7], acc[7]);
            unsigned int x_row_98 = s_x_addr + cur + (unsigned int)((1568 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_98));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_98 + 16));
            acc[0] = __fmaf_rn(mrow[98], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[98], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[98], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[98], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[98], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[98], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[98], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[98], xv[7], acc[7]);
            unsigned int x_row_99 = s_x_addr + cur + (unsigned int)((1584 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_99));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_99 + 16));
            acc[0] = __fmaf_rn(mrow[99], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[99], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[99], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[99], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[99], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[99], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[99], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[99], xv[7], acc[7]);
            unsigned int x_row_100 = s_x_addr + cur + (unsigned int)((1600 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_100));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_100 + 16));
            acc[0] = __fmaf_rn(mrow[100], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[100], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[100], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[100], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[100], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[100], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[100], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[100], xv[7], acc[7]);
            unsigned int x_row_101 = s_x_addr + cur + (unsigned int)((1616 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_101));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_101 + 16));
            acc[0] = __fmaf_rn(mrow[101], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[101], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[101], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[101], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[101], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[101], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[101], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[101], xv[7], acc[7]);
            unsigned int x_row_102 = s_x_addr + cur + (unsigned int)((1632 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_102));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_102 + 16));
            acc[0] = __fmaf_rn(mrow[102], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[102], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[102], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[102], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[102], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[102], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[102], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[102], xv[7], acc[7]);
            unsigned int x_row_103 = s_x_addr + cur + (unsigned int)((1648 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_103));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_103 + 16));
            acc[0] = __fmaf_rn(mrow[103], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[103], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[103], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[103], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[103], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[103], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[103], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[103], xv[7], acc[7]);
            unsigned int x_row_104 = s_x_addr + cur + (unsigned int)((1664 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_104));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_104 + 16));
            acc[0] = __fmaf_rn(mrow[104], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[104], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[104], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[104], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[104], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[104], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[104], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[104], xv[7], acc[7]);
            unsigned int x_row_105 = s_x_addr + cur + (unsigned int)((1680 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_105));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_105 + 16));
            acc[0] = __fmaf_rn(mrow[105], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[105], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[105], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[105], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[105], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[105], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[105], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[105], xv[7], acc[7]);
            unsigned int x_row_106 = s_x_addr + cur + (unsigned int)((1696 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_106));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_106 + 16));
            acc[0] = __fmaf_rn(mrow[106], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[106], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[106], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[106], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[106], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[106], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[106], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[106], xv[7], acc[7]);
            unsigned int x_row_107 = s_x_addr + cur + (unsigned int)((1712 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_107));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_107 + 16));
            acc[0] = __fmaf_rn(mrow[107], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[107], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[107], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[107], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[107], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[107], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[107], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[107], xv[7], acc[7]);
            unsigned int x_row_108 = s_x_addr + cur + (unsigned int)((1728 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_108));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_108 + 16));
            acc[0] = __fmaf_rn(mrow[108], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[108], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[108], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[108], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[108], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[108], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[108], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[108], xv[7], acc[7]);
            unsigned int x_row_109 = s_x_addr + cur + (unsigned int)((1744 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_109));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_109 + 16));
            acc[0] = __fmaf_rn(mrow[109], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[109], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[109], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[109], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[109], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[109], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[109], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[109], xv[7], acc[7]);
            unsigned int x_row_110 = s_x_addr + cur + (unsigned int)((1760 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_110));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_110 + 16));
            acc[0] = __fmaf_rn(mrow[110], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[110], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[110], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[110], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[110], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[110], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[110], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[110], xv[7], acc[7]);
            unsigned int x_row_111 = s_x_addr + cur + (unsigned int)((1776 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_111));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_111 + 16));
            acc[0] = __fmaf_rn(mrow[111], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[111], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[111], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[111], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[111], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[111], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[111], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[111], xv[7], acc[7]);
            unsigned int x_row_112 = s_x_addr + cur + (unsigned int)((1792 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_112));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_112 + 16));
            acc[0] = __fmaf_rn(mrow[112], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[112], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[112], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[112], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[112], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[112], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[112], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[112], xv[7], acc[7]);
            unsigned int x_row_113 = s_x_addr + cur + (unsigned int)((1808 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_113));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_113 + 16));
            acc[0] = __fmaf_rn(mrow[113], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[113], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[113], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[113], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[113], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[113], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[113], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[113], xv[7], acc[7]);
            unsigned int x_row_114 = s_x_addr + cur + (unsigned int)((1824 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_114));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_114 + 16));
            acc[0] = __fmaf_rn(mrow[114], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[114], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[114], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[114], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[114], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[114], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[114], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[114], xv[7], acc[7]);
            unsigned int x_row_115 = s_x_addr + cur + (unsigned int)((1840 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_115));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_115 + 16));
            acc[0] = __fmaf_rn(mrow[115], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[115], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[115], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[115], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[115], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[115], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[115], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[115], xv[7], acc[7]);
            unsigned int x_row_116 = s_x_addr + cur + (unsigned int)((1856 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_116));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_116 + 16));
            acc[0] = __fmaf_rn(mrow[116], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[116], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[116], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[116], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[116], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[116], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[116], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[116], xv[7], acc[7]);
            unsigned int x_row_117 = s_x_addr + cur + (unsigned int)((1872 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_117));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_117 + 16));
            acc[0] = __fmaf_rn(mrow[117], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[117], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[117], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[117], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[117], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[117], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[117], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[117], xv[7], acc[7]);
            unsigned int x_row_118 = s_x_addr + cur + (unsigned int)((1888 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_118));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_118 + 16));
            acc[0] = __fmaf_rn(mrow[118], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[118], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[118], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[118], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[118], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[118], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[118], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[118], xv[7], acc[7]);
            unsigned int x_row_119 = s_x_addr + cur + (unsigned int)((1904 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_119));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_119 + 16));
            acc[0] = __fmaf_rn(mrow[119], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[119], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[119], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[119], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[119], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[119], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[119], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[119], xv[7], acc[7]);
            unsigned int x_row_120 = s_x_addr + cur + (unsigned int)((1920 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_120));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_120 + 16));
            acc[0] = __fmaf_rn(mrow[120], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[120], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[120], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[120], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[120], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[120], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[120], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[120], xv[7], acc[7]);
            unsigned int x_row_121 = s_x_addr + cur + (unsigned int)((1936 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_121));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_121 + 16));
            acc[0] = __fmaf_rn(mrow[121], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[121], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[121], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[121], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[121], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[121], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[121], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[121], xv[7], acc[7]);
            unsigned int x_row_122 = s_x_addr + cur + (unsigned int)((1952 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_122));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_122 + 16));
            acc[0] = __fmaf_rn(mrow[122], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[122], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[122], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[122], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[122], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[122], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[122], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[122], xv[7], acc[7]);
            unsigned int x_row_123 = s_x_addr + cur + (unsigned int)((1968 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_123));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_123 + 16));
            acc[0] = __fmaf_rn(mrow[123], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[123], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[123], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[123], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[123], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[123], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[123], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[123], xv[7], acc[7]);
            unsigned int x_row_124 = s_x_addr + cur + (unsigned int)((1984 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_124));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_124 + 16));
            acc[0] = __fmaf_rn(mrow[124], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[124], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[124], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[124], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[124], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[124], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[124], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[124], xv[7], acc[7]);
            unsigned int x_row_125 = s_x_addr + cur + (unsigned int)((2000 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_125));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_125 + 16));
            acc[0] = __fmaf_rn(mrow[125], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[125], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[125], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[125], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[125], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[125], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[125], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[125], xv[7], acc[7]);
            unsigned int x_row_126 = s_x_addr + cur + (unsigned int)((2016 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_126));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_126 + 16));
            acc[0] = __fmaf_rn(mrow[126], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[126], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[126], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[126], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[126], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[126], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[126], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[126], xv[7], acc[7]);
            unsigned int x_row_127 = s_x_addr + cur + (unsigned int)((2032 + half * 8) * 4);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                : "r"(x_row_127));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(4) + 3]))
                : "r"(x_row_127 + 16));
            acc[0] = __fmaf_rn(mrow[127], xv[0], acc[0]);
            acc[1] = __fmaf_rn(mrow[127], xv[1], acc[1]);
            acc[2] = __fmaf_rn(mrow[127], xv[2], acc[2]);
            acc[3] = __fmaf_rn(mrow[127], xv[3], acc[3]);
            acc[4] = __fmaf_rn(mrow[127], xv[4], acc[4]);
            acc[5] = __fmaf_rn(mrow[127], xv[5], acc[5]);
            acc[6] = __fmaf_rn(mrow[127], xv[6], acc[6]);
            acc[7] = __fmaf_rn(mrow[127], xv[7], acc[7]);
        }
        x[0] = acc[0] + b[0];
        x[1] = acc[1] + b[1];
        x[2] = acc[2] + b[2];
        x[3] = acc[3] + b[3];
        x[4] = acc[4] + b[4];
        x[5] = acc[5] + b[5];
        x[6] = acc[6] + b[6];
        x[7] = acc[7] + b[7];
        if (i + 1 < steps) {
            long long map_next = ((long long)(s - 1) * nvh + hv64) * 32768ll + (long long)row * 256ll;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 128ll);
                mrow[0 + 0] = _v4.x;
                mrow[0 + 1] = _v4.y;
                mrow[0 + 2] = _v4.z;
                mrow[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 132ll);
                mrow[4 + 0] = _v4.x;
                mrow[4 + 1] = _v4.y;
                mrow[4 + 2] = _v4.z;
                mrow[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 136ll);
                mrow[8 + 0] = _v4.x;
                mrow[8 + 1] = _v4.y;
                mrow[8 + 2] = _v4.z;
                mrow[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 140ll);
                mrow[12 + 0] = _v4.x;
                mrow[12 + 1] = _v4.y;
                mrow[12 + 2] = _v4.z;
                mrow[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 144ll);
                mrow[16 + 0] = _v4.x;
                mrow[16 + 1] = _v4.y;
                mrow[16 + 2] = _v4.z;
                mrow[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 148ll);
                mrow[20 + 0] = _v4.x;
                mrow[20 + 1] = _v4.y;
                mrow[20 + 2] = _v4.z;
                mrow[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 152ll);
                mrow[24 + 0] = _v4.x;
                mrow[24 + 1] = _v4.y;
                mrow[24 + 2] = _v4.z;
                mrow[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 156ll);
                mrow[28 + 0] = _v4.x;
                mrow[28 + 1] = _v4.y;
                mrow[28 + 2] = _v4.z;
                mrow[28 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 160ll);
                mrow[32 + 0] = _v4.x;
                mrow[32 + 1] = _v4.y;
                mrow[32 + 2] = _v4.z;
                mrow[32 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 164ll);
                mrow[36 + 0] = _v4.x;
                mrow[36 + 1] = _v4.y;
                mrow[36 + 2] = _v4.z;
                mrow[36 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 168ll);
                mrow[40 + 0] = _v4.x;
                mrow[40 + 1] = _v4.y;
                mrow[40 + 2] = _v4.z;
                mrow[40 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 172ll);
                mrow[44 + 0] = _v4.x;
                mrow[44 + 1] = _v4.y;
                mrow[44 + 2] = _v4.z;
                mrow[44 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 176ll);
                mrow[48 + 0] = _v4.x;
                mrow[48 + 1] = _v4.y;
                mrow[48 + 2] = _v4.z;
                mrow[48 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 180ll);
                mrow[52 + 0] = _v4.x;
                mrow[52 + 1] = _v4.y;
                mrow[52 + 2] = _v4.z;
                mrow[52 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 184ll);
                mrow[56 + 0] = _v4.x;
                mrow[56 + 1] = _v4.y;
                mrow[56 + 2] = _v4.z;
                mrow[56 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 188ll);
                mrow[60 + 0] = _v4.x;
                mrow[60 + 1] = _v4.y;
                mrow[60 + 2] = _v4.z;
                mrow[60 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 192ll);
                mrow[64 + 0] = _v4.x;
                mrow[64 + 1] = _v4.y;
                mrow[64 + 2] = _v4.z;
                mrow[64 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 196ll);
                mrow[68 + 0] = _v4.x;
                mrow[68 + 1] = _v4.y;
                mrow[68 + 2] = _v4.z;
                mrow[68 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 200ll);
                mrow[72 + 0] = _v4.x;
                mrow[72 + 1] = _v4.y;
                mrow[72 + 2] = _v4.z;
                mrow[72 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 204ll);
                mrow[76 + 0] = _v4.x;
                mrow[76 + 1] = _v4.y;
                mrow[76 + 2] = _v4.z;
                mrow[76 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 208ll);
                mrow[80 + 0] = _v4.x;
                mrow[80 + 1] = _v4.y;
                mrow[80 + 2] = _v4.z;
                mrow[80 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 212ll);
                mrow[84 + 0] = _v4.x;
                mrow[84 + 1] = _v4.y;
                mrow[84 + 2] = _v4.z;
                mrow[84 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 216ll);
                mrow[88 + 0] = _v4.x;
                mrow[88 + 1] = _v4.y;
                mrow[88 + 2] = _v4.z;
                mrow[88 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 220ll);
                mrow[92 + 0] = _v4.x;
                mrow[92 + 1] = _v4.y;
                mrow[92 + 2] = _v4.z;
                mrow[92 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 224ll);
                mrow[96 + 0] = _v4.x;
                mrow[96 + 1] = _v4.y;
                mrow[96 + 2] = _v4.z;
                mrow[96 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 228ll);
                mrow[100 + 0] = _v4.x;
                mrow[100 + 1] = _v4.y;
                mrow[100 + 2] = _v4.z;
                mrow[100 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 232ll);
                mrow[104 + 0] = _v4.x;
                mrow[104 + 1] = _v4.y;
                mrow[104 + 2] = _v4.z;
                mrow[104 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 236ll);
                mrow[108 + 0] = _v4.x;
                mrow[108 + 1] = _v4.y;
                mrow[108 + 2] = _v4.z;
                mrow[108 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 240ll);
                mrow[112 + 0] = _v4.x;
                mrow[112 + 1] = _v4.y;
                mrow[112 + 2] = _v4.z;
                mrow[112 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 244ll);
                mrow[116 + 0] = _v4.x;
                mrow[116 + 1] = _v4.y;
                mrow[116 + 2] = _v4.z;
                mrow[116 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 248ll);
                mrow[120 + 0] = _v4.x;
                mrow[120 + 1] = _v4.y;
                mrow[120 + 2] = _v4.z;
                mrow[120 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + 252ll);
                mrow[124 + 0] = _v4.x;
                mrow[124 + 1] = _v4.y;
                mrow[124 + 2] = _v4.z;
                mrow[124 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + (long long)col);
                b[0 + 0] = _v4.x;
                b[0 + 1] = _v4.y;
                b[0 + 2] = _v4.z;
                b[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + map_next + (long long)(col + 4));
                b[4 + 0] = _v4.x;
                b[4 + 1] = _v4.y;
                b[4 + 2] = _v4.z;
                b[4 + 3] = _v4.w;
            }
        }
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
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
