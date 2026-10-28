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
#define SMEM_S_MHI_OFF 0
#define SMEM_S_MHI_STAGE_BYTES 32768
#define SMEM_S_MHI_STRIDE 32768
#define SMEM_S_MLO_OFF 32768
#define SMEM_S_MLO_STAGE_BYTES 32768
#define SMEM_S_MLO_STRIDE 32768
#define SMEM_S_HHI_OFF 65536
#define SMEM_S_HHI_STAGE_BYTES 32768
#define SMEM_S_HHI_STRIDE 32768
#define SMEM_S_HLO_OFF 98304
#define SMEM_S_HLO_STAGE_BYTES 32768
#define SMEM_S_HLO_STRIDE 32768
#define SMEM_TOTAL 131072
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_gdn_det_train_6c5de747bed9e9773c92(float* __restrict__ seg_state, float* __restrict__ init_state, float* __restrict__ state_out, int* __restrict__ seq_seg_start, int* __restrict__ seg_flag, int num_v_heads, int use_init, int reverse)
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
    __nv_bfloat16* s_mhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_mhi_addr = smem + 0;
    __nv_bfloat16* s_mlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_mlo_addr = smem + 32768;
    __nv_bfloat16* s_hhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_hhi_addr = smem + 65536;
    __nv_bfloat16* s_hlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_hlo_addr = smem + 98304;

    // === Task calls (declaration order) ===
    int seq = blockIdx.x;
    int hv = blockIdx.y;
    int vb0 = blockIdx.z * 32;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int quad = lane_0 % 4;
    int m0 = warp_1 * 16;
    int row_a = m0 + lane_0 / 4;
    int m_idx = lane_0 / 8;
    int r_idx = lane_0 % 8;
    int st_row = m0 + m_idx % 2 * 8 + r_idx;
    int st_col = m_idx / 2 * 8;
    long long nvh = (long long)num_v_heads;
    long long hv64 = (long long)hv;
    int seg_first = seq_seg_start[seq];
    int num_segs = seq_seg_start[seq + 1] - seg_first;
    int step = 1;
    if (reverse != 0) {
        seg_first = seg_first + num_segs - 1;
        step = -1;
    }
    int no_init = 1 - use_init;
    float acc[16];
    unsigned int a_hi[4];
    unsigned int a_lo[4];
    unsigned int b_hi[4];
    unsigned int b_lo[4];
    unsigned int pk_hi[4];
    unsigned int pk_lo[4];
    float res[8];
    float blk[8];
    float bp[2];
    long long init_base = ((long long)seq * nvh + hv64) * 16384;
    long long first_base = ((long long)seg_first * nvh + hv64) * 16384;
    float vals[4];
    unsigned int hi[2];
    unsigned int lo[2];
    float res_3[4];
    int idx = tid * 4;
    int row = idx / 32;
    int col = vb0 + idx % 32;
    if (no_init != 0) {
        vals[0] = 0.0f;
        vals[1] = 0.0f;
        vals[2] = 0.0f;
        vals[3] = 0.0f;
    } else {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + (long long)row * 128 + (long long)col);
            vals[0 + 0] = _v4.x;
            vals[0 + 1] = _v4.y;
            vals[0 + 2] = _v4.z;
            vals[0 + 3] = _v4.w;
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
        hi[_lp] = *(uint32_t*)&_bf2;
    }
    res_3[0] = vals[0] - __uint_as_float(hi[0] << 16);
    res_3[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
    res_3[2] = vals[2] - __uint_as_float(hi[1] << 16);
    res_3[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
        lo[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hhi_addr + (unsigned int)(col / 64 * 16384 + (row * 128 + col % 64 * 2 ^ (row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hlo_addr + (unsigned int)(col / 64 * 16384 + (row * 128 + col % 64 * 2 ^ (row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
    int idx_4 = (256 + tid) * 4;
    int row_5 = idx_4 / 32;
    int col_6 = vb0 + idx_4 % 32;
    if (no_init != 0) {
        vals[0] = 0.0f;
        vals[1] = 0.0f;
        vals[2] = 0.0f;
        vals[3] = 0.0f;
    } else {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + (long long)row_5 * 128 + (long long)col_6);
            vals[0 + 0] = _v4.x;
            vals[0 + 1] = _v4.y;
            vals[0 + 2] = _v4.z;
            vals[0 + 3] = _v4.w;
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
        hi[_lp] = *(uint32_t*)&_bf2;
    }
    res_3[0] = vals[0] - __uint_as_float(hi[0] << 16);
    res_3[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
    res_3[2] = vals[2] - __uint_as_float(hi[1] << 16);
    res_3[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
        lo[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hhi_addr + (unsigned int)(col_6 / 64 * 16384 + (row_5 * 128 + col_6 % 64 * 2 ^ (row_5 * 128 + col_6 % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hlo_addr + (unsigned int)(col_6 / 64 * 16384 + (row_5 * 128 + col_6 % 64 * 2 ^ (row_5 * 128 + col_6 % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
    int idx_7 = (512 + tid) * 4;
    int row_8 = idx_7 / 32;
    int col_9 = vb0 + idx_7 % 32;
    if (no_init != 0) {
        vals[0] = 0.0f;
        vals[1] = 0.0f;
        vals[2] = 0.0f;
        vals[3] = 0.0f;
    } else {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + (long long)row_8 * 128 + (long long)col_9);
            vals[0 + 0] = _v4.x;
            vals[0 + 1] = _v4.y;
            vals[0 + 2] = _v4.z;
            vals[0 + 3] = _v4.w;
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
        hi[_lp] = *(uint32_t*)&_bf2;
    }
    res_3[0] = vals[0] - __uint_as_float(hi[0] << 16);
    res_3[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
    res_3[2] = vals[2] - __uint_as_float(hi[1] << 16);
    res_3[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
        lo[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hhi_addr + (unsigned int)(col_9 / 64 * 16384 + (row_8 * 128 + col_9 % 64 * 2 ^ (row_8 * 128 + col_9 % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hlo_addr + (unsigned int)(col_9 / 64 * 16384 + (row_8 * 128 + col_9 % 64 * 2 ^ (row_8 * 128 + col_9 % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
    int idx_10 = (768 + tid) * 4;
    int row_11 = idx_10 / 32;
    int col_12 = vb0 + idx_10 % 32;
    if (no_init != 0) {
        vals[0] = 0.0f;
        vals[1] = 0.0f;
        vals[2] = 0.0f;
        vals[3] = 0.0f;
    } else {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + (long long)row_11 * 128 + (long long)col_12);
            vals[0 + 0] = _v4.x;
            vals[0 + 1] = _v4.y;
            vals[0 + 2] = _v4.z;
            vals[0 + 3] = _v4.w;
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
        hi[_lp] = *(uint32_t*)&_bf2;
    }
    res_3[0] = vals[0] - __uint_as_float(hi[0] << 16);
    res_3[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
    res_3[2] = vals[2] - __uint_as_float(hi[1] << 16);
    res_3[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
    #pragma unroll
    for (int _lp = 0; _lp < 2; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
        lo[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hhi_addr + (unsigned int)(col_12 / 64 * 16384 + (row_11 * 128 + col_12 % 64 * 2 ^ (row_11 * 128 + col_12 % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_hlo_addr + (unsigned int)(col_12 / 64 * 16384 + (row_11 * 128 + col_12 % 64 * 2 ^ (row_11 * 128 + col_12 % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
    float cp[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx_0 = (i * 256 + tid_2) * 4;
        long long cp_off = (long long)(idx_0 / 32) * 128 + (long long)(vb0 + idx_0 % 32);
        if (no_init != 0) {
            cp[0] = 0.0f;
            cp[1] = 0.0f;
            cp[2] = 0.0f;
            cp[3] = 0.0f;
        } else {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(init_state + init_base + cp_off);
                cp[0 + 0] = _v4.x;
                cp[0 + 1] = _v4.y;
                cp[0 + 2] = _v4.z;
                cp[0 + 3] = _v4.w;
            }
        }
        {
            float4 _v4 = make_float4(cp[0 + 0], cp[0 + 1], cp[0 + 2], cp[0 + 3]);
            *reinterpret_cast<float4*>(state_out + first_base + cp_off) = _v4;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int sidx = 1; sidx < num_segs; sidx++) {
        int prev = seg_first + (sidx - 1) * step;
        int cur = prev + step;
        long long pre_base = ((long long)prev * nvh + hv64) * 32768;
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        acc[8] = 0.0f;
        acc[9] = 0.0f;
        acc[10] = 0.0f;
        acc[11] = 0.0f;
        acc[12] = 0.0f;
        acc[13] = 0.0f;
        acc[14] = 0.0f;
        acc[15] = 0.0f;
        int has_m = seg_flag[(long long)prev * nvh + hv64];
        if (sidx == 1) {
            if (no_init != 0) {
                has_m = 0;
            }
        }
        if (has_m != 0) {
            float vals_0[4];
            unsigned int hi_1[2];
            unsigned int lo_2[2];
            float res_4[4];
            int idx_5 = tid * 4;
            int row_6 = idx_5 / 128;
            int col_7 = idx_5 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_6 * 256 + (long long)col_7);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_7 / 64 * 16384 + (row_6 * 128 + col_7 % 64 * 2 ^ (row_6 * 128 + col_7 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_7 / 64 * 16384 + (row_6 * 128 + col_7 % 64 * 2 ^ (row_6 * 128 + col_7 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_8 = (256 + tid) * 4;
            int row_9 = idx_8 / 128;
            int col_10 = idx_8 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_9 * 256 + (long long)col_10);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_10 / 64 * 16384 + (row_9 * 128 + col_10 % 64 * 2 ^ (row_9 * 128 + col_10 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_10 / 64 * 16384 + (row_9 * 128 + col_10 % 64 * 2 ^ (row_9 * 128 + col_10 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_11 = (512 + tid) * 4;
            int row_12 = idx_11 / 128;
            int col_13 = idx_11 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_12 * 256 + (long long)col_13);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_13 / 64 * 16384 + (row_12 * 128 + col_13 % 64 * 2 ^ (row_12 * 128 + col_13 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_13 / 64 * 16384 + (row_12 * 128 + col_13 % 64 * 2 ^ (row_12 * 128 + col_13 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_14 = (768 + tid) * 4;
            int row_15 = idx_14 / 128;
            int col_16 = idx_14 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_15 * 256 + (long long)col_16);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_16 / 64 * 16384 + (row_15 * 128 + col_16 % 64 * 2 ^ (row_15 * 128 + col_16 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_16 / 64 * 16384 + (row_15 * 128 + col_16 % 64 * 2 ^ (row_15 * 128 + col_16 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_17 = (1024 + tid) * 4;
            int row_18 = idx_17 / 128;
            int col_19 = idx_17 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_18 * 256 + (long long)col_19);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_19 / 64 * 16384 + (row_18 * 128 + col_19 % 64 * 2 ^ (row_18 * 128 + col_19 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_19 / 64 * 16384 + (row_18 * 128 + col_19 % 64 * 2 ^ (row_18 * 128 + col_19 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_20 = (1280 + tid) * 4;
            int row_21 = idx_20 / 128;
            int col_22 = idx_20 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_21 * 256 + (long long)col_22);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_22 / 64 * 16384 + (row_21 * 128 + col_22 % 64 * 2 ^ (row_21 * 128 + col_22 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_22 / 64 * 16384 + (row_21 * 128 + col_22 % 64 * 2 ^ (row_21 * 128 + col_22 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_23 = (1536 + tid) * 4;
            int row_24 = idx_23 / 128;
            int col_25 = idx_23 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_24 * 256 + (long long)col_25);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_25 / 64 * 16384 + (row_24 * 128 + col_25 % 64 * 2 ^ (row_24 * 128 + col_25 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_25 / 64 * 16384 + (row_24 * 128 + col_25 % 64 * 2 ^ (row_24 * 128 + col_25 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_26 = (1792 + tid) * 4;
            int row_27 = idx_26 / 128;
            int col_28 = idx_26 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_27 * 256 + (long long)col_28);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_28 / 64 * 16384 + (row_27 * 128 + col_28 % 64 * 2 ^ (row_27 * 128 + col_28 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_28 / 64 * 16384 + (row_27 * 128 + col_28 % 64 * 2 ^ (row_27 * 128 + col_28 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_29 = (2048 + tid) * 4;
            int row_30 = idx_29 / 128;
            int col_31 = idx_29 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_30 * 256 + (long long)col_31);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_31 / 64 * 16384 + (row_30 * 128 + col_31 % 64 * 2 ^ (row_30 * 128 + col_31 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_31 / 64 * 16384 + (row_30 * 128 + col_31 % 64 * 2 ^ (row_30 * 128 + col_31 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_32 = (2304 + tid) * 4;
            int row_33 = idx_32 / 128;
            int col_34 = idx_32 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_33 * 256 + (long long)col_34);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_34 / 64 * 16384 + (row_33 * 128 + col_34 % 64 * 2 ^ (row_33 * 128 + col_34 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_34 / 64 * 16384 + (row_33 * 128 + col_34 % 64 * 2 ^ (row_33 * 128 + col_34 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_35 = (2560 + tid) * 4;
            int row_36 = idx_35 / 128;
            int col_37 = idx_35 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_36 * 256 + (long long)col_37);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_37 / 64 * 16384 + (row_36 * 128 + col_37 % 64 * 2 ^ (row_36 * 128 + col_37 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_37 / 64 * 16384 + (row_36 * 128 + col_37 % 64 * 2 ^ (row_36 * 128 + col_37 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_38 = (2816 + tid) * 4;
            int row_39 = idx_38 / 128;
            int col_40 = idx_38 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_39 * 256 + (long long)col_40);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_40 / 64 * 16384 + (row_39 * 128 + col_40 % 64 * 2 ^ (row_39 * 128 + col_40 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_40 / 64 * 16384 + (row_39 * 128 + col_40 % 64 * 2 ^ (row_39 * 128 + col_40 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_41 = (3072 + tid) * 4;
            int row_42 = idx_41 / 128;
            int col_43 = idx_41 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_42 * 256 + (long long)col_43);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_43 / 64 * 16384 + (row_42 * 128 + col_43 % 64 * 2 ^ (row_42 * 128 + col_43 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_43 / 64 * 16384 + (row_42 * 128 + col_43 % 64 * 2 ^ (row_42 * 128 + col_43 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_44 = (3328 + tid) * 4;
            int row_45 = idx_44 / 128;
            int col_46 = idx_44 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_45 * 256 + (long long)col_46);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_46 / 64 * 16384 + (row_45 * 128 + col_46 % 64 * 2 ^ (row_45 * 128 + col_46 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_46 / 64 * 16384 + (row_45 * 128 + col_46 % 64 * 2 ^ (row_45 * 128 + col_46 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_47 = (3584 + tid) * 4;
            int row_48 = idx_47 / 128;
            int col_49 = idx_47 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_48 * 256 + (long long)col_49);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_49 / 64 * 16384 + (row_48 * 128 + col_49 % 64 * 2 ^ (row_48 * 128 + col_49 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_49 / 64 * 16384 + (row_48 * 128 + col_49 % 64 * 2 ^ (row_48 * 128 + col_49 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
            int idx_50 = (3840 + tid) * 4;
            int row_51 = idx_50 / 128;
            int col_52 = idx_50 % 128;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(seg_state + pre_base + 128 + (long long)row_51 * 256 + (long long)col_52);
                vals_0[0 + 0] = _v4.x;
                vals_0[0 + 1] = _v4.y;
                vals_0[0 + 2] = _v4.z;
                vals_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                hi_1[_lp] = *(uint32_t*)&_bf2;
            }
            res_4[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
            res_4[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
            res_4[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
            res_4[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                lo_2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_52 / 64 * 16384 + (row_51 * 128 + col_52 % 64 * 2 ^ (row_51 * 128 + col_52 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_52 / 64 * 16384 + (row_51 * 128 + col_52 % 64 * 2 ^ (row_51 * 128 + col_52 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
        }
        __syncthreads();
        if (has_m != 0) {
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                    : "r"(s_mhi_addr + (unsigned int)((kk * 16 + lane / 16 * 8) / 64 * 16384 + ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                    : "r"(s_mlo_addr + (unsigned int)((kk * 16 + lane / 16 * 8) / 64 * 16384 + ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                #pragma unroll
                for (int nb = 0; nb < 2; nb++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_hhi_addr + (unsigned int)((vb0 + nb * 16 + lane / 16 * 8) / 64 * 16384 + ((kk * 16 + lane % 16) * 128 + (vb0 + nb * 16 + lane / 16 * 8) % 64 * 2 ^ ((kk * 16 + lane % 16) * 128 + (vb0 + nb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_hlo_addr + (unsigned int)((vb0 + nb * 16 + lane / 16 * 8) / 64 * 16384 + ((kk * 16 + lane % 16) * 128 + (vb0 + nb * 16 + lane / 16 * 8) % 64 * 2 ^ ((kk * 16 + lane % 16) * 128 + (vb0 + nb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8)[0]), "+f"((acc + nb * 8)[1]), "+f"((acc + nb * 8)[2]), "+f"((acc + nb * 8)[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8 + 4)[0]), "+f"((acc + nb * 8 + 4)[1]), "+f"((acc + nb * 8 + 4)[2]), "+f"((acc + nb * 8 + 4)[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8)[0]), "+f"((acc + nb * 8)[1]), "+f"((acc + nb * 8)[2]), "+f"((acc + nb * 8)[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8 + 4)[0]), "+f"((acc + nb * 8 + 4)[1]), "+f"((acc + nb * 8 + 4)[2]), "+f"((acc + nb * 8 + 4)[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8)[0]), "+f"((acc + nb * 8)[1]), "+f"((acc + nb * 8)[2]), "+f"((acc + nb * 8)[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nb * 8 + 4)[0]), "+f"((acc + nb * 8 + 4)[1]), "+f"((acc + nb * 8 + 4)[2]), "+f"((acc + nb * 8 + 4)[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                }
            }
        }
        long long cur_base = ((long long)cur * nvh + hv64) * 16384;
        #pragma unroll
        for (int nb2 = 0; nb2 < 2; nb2++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int col_0 = vb0 + nb2 * 16 + half * 8 + quad * 2;
                {
                    float2 _v2_21 = *reinterpret_cast<const float2*>(seg_state + pre_base + (long long)row_a * 256 + (long long)col_0);
                    bp[0] = _v2_21.x;
                    bp[0 + 1] = _v2_21.y;
                }
                acc[nb2 * 8 + half * 4] = acc[nb2 * 8 + half * 4] + bp[0];
                acc[nb2 * 8 + half * 4 + 1] = acc[nb2 * 8 + half * 4 + 1] + bp[1];
                {
                    float2 _v2_22 = *reinterpret_cast<const float2*>(seg_state + pre_base + (long long)(row_a + 8) * 256 + (long long)col_0);
                    bp[0] = _v2_22.x;
                    bp[0 + 1] = _v2_22.y;
                }
                acc[nb2 * 8 + half * 4 + 2] = acc[nb2 * 8 + half * 4 + 2] + bp[0];
                acc[nb2 * 8 + half * 4 + 3] = acc[nb2 * 8 + half * 4 + 3] + bp[1];
                {
                    float2 _v2 = make_float2(acc[nb2 * 8 + half * 4 + 0], acc[nb2 * 8 + half * 4 + 1]);
                    *reinterpret_cast<float2*>(state_out + cur_base + (long long)row_a * 128 + (long long)col_0) = _v2;
                }
                {
                    float2 _v2 = make_float2(acc[nb2 * 8 + half * 4 + 2 + 0], acc[nb2 * 8 + half * 4 + 2 + 1]);
                    *reinterpret_cast<float2*>(state_out + cur_base + (long long)(row_a + 8) * 128 + (long long)col_0) = _v2;
                }
            }
        }
        int rebuild = 0;
        if (num_segs > sidx + 1) {
            rebuild = seg_flag[(long long)cur * nvh + hv64];
        }
        __syncthreads();
        if (rebuild != 0) {
            #pragma unroll
            for (int nb3 = 0; nb3 < 2; nb3++) {
                #pragma unroll
                for (int e0 = 0; e0 < 8; e0++) {
                    blk[e0] = acc[nb3 * 8 + e0];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(blk[_lp*2 + 0], blk[_lp*2+1 + 0]));
                    pk_hi[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    res[2 * e] = blk[2 * e] - __uint_as_float(pk_hi[e] << 16);
                    res[2 * e + 1] = blk[2 * e + 1] - __uint_as_float(pk_hi[e] & 4294901760u);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 0], res[_lp*2+1 + 0]));
                    pk_lo[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(s_hhi_addr + (unsigned int)((vb0 + nb3 * 16 + st_col) / 64 * 16384 + (st_row * 128 + (vb0 + nb3 * 16 + st_col) % 64 * 2 ^ (st_row * 128 + (vb0 + nb3 * 16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&pk_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_hi[3]))
                    : "memory");
                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(s_hlo_addr + (unsigned int)((vb0 + nb3 * 16 + st_col) / 64 * 16384 + (st_row * 128 + (vb0 + nb3 * 16 + st_col) % 64 * 2 ^ (st_row * 128 + (vb0 + nb3 * 16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&pk_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_lo[3]))
                    : "memory");
            }
        }
        __syncthreads();
    }
}

} // extern "C"
