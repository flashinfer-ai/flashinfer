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

#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>
#if __CUDA_ARCH__ == 1070
#else


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}
#endif

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_det_train_7634c7541a6836a5146a_kcp(float* __restrict__ m, float* __restrict__ c, float* __restrict__ x0, float* __restrict__ y, int* __restrict__ chain_start, long long* __restrict__ elem_m, long long* __restrict__ elem_c, long long* __restrict__ elem_y, long long* __restrict__ x0_index, long long* __restrict__ y_final, int ncols, int transpose_m, int write_states, int cols_per_cta)
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

    // Prologue
    float* xs = reinterpret_cast<float*>(smem_raw + 0);
    const int xs_addr = smem + 0;
    float* ms = reinterpret_cast<float*>(smem_raw + 36864);
    const int ms_addr = smem + 36864;

    // === Task calls (declaration order) ===
    int nblk = ncols / 32;
    int chain = blockIdx.x / nblk;
    int col0 = blockIdx.x % nblk * 32;
    int col_tile = col0 / 128;
    int v0 = col0 % 128;
    int warp_1 = warp;
    int g = lane / 8;
    int cg = lane % 8;
    int e_begin = chain_start[chain];
    int e_end = chain_start[chain + 1];
    unsigned int ring_w = ms_addr + (unsigned int)(warp_1 * 18432);
    float acc[32];
    float cn[32];
    float mv[32];
    float xv[4];
    if (transpose_m == 0) {
        int row0 = warp_1 * 32 + g * 4;
        long long g_off = (long long)col_tile * 16384ll + (long long)(row0 * 128 + v0 + cg * 4);
        unsigned int x_thr = xs_addr + (unsigned int)((row0 * 36 + cg * 4) * 4);
        unsigned int x_rd = xs_addr + (unsigned int)(cg * 16);
        unsigned int m_rd = ring_w + (unsigned int)(g * 16);
        int cp_i = lane / 8;
        int cp_j = lane % 8;
        int cp_src = cp_i * 128 + warp_1 * 32 + 4 * cp_j;
        unsigned int cp_dst = ring_w + (unsigned int)((cp_i * 32 + 4 * cp_j) * 4);
        long long m_cur = 0;
        if (e_begin < e_end) {
            m_cur = elem_m[e_begin];
            long long src0 = m_cur + (long long)cp_src;
            unsigned int dst0 = cp_dst + 0;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0), "l"(m + src0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 512), "l"(m + (src0 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 1024), "l"(m + (src0 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 1536), "l"(m + (src0 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 2048), "l"(m + (src0 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 2560), "l"(m + (src0 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 3072), "l"(m + (src0 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0 + 3584), "l"(m + (src0 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_0 = m_cur + (long long)(cp_src + 4096);
            unsigned int dst0_0 = cp_dst + 4608;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0), "l"(m + src0_0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 512), "l"(m + (src0_0 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 1024), "l"(m + (src0_0 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 1536), "l"(m + (src0_0 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 2048), "l"(m + (src0_0 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 2560), "l"(m + (src0_0 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 3072), "l"(m + (src0_0 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 3584), "l"(m + (src0_0 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_1 = m_cur + (long long)(cp_src + 8192);
            unsigned int dst0_1 = cp_dst + 9216;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1), "l"(m + src0_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 512), "l"(m + (src0_1 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 1024), "l"(m + (src0_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 1536), "l"(m + (src0_1 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 2048), "l"(m + (src0_1 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 2560), "l"(m + (src0_1 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 3072), "l"(m + (src0_1 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 3584), "l"(m + (src0_1 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long c_first = -1;
            if (col_tile == 0) {
                c_first = elem_c[e_begin];
            }
            if (c_first >= 0ll) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off);
                    cn[0 + 0] = _v4.x;
                    cn[0 + 1] = _v4.y;
                    cn[0 + 2] = _v4.z;
                    cn[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 128ll);
                    cn[4 + 0] = _v4.x;
                    cn[4 + 1] = _v4.y;
                    cn[4 + 2] = _v4.z;
                    cn[4 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 256ll);
                    cn[8 + 0] = _v4.x;
                    cn[8 + 1] = _v4.y;
                    cn[8 + 2] = _v4.z;
                    cn[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 384ll);
                    cn[12 + 0] = _v4.x;
                    cn[12 + 1] = _v4.y;
                    cn[12 + 2] = _v4.z;
                    cn[12 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 2048ll);
                    cn[16 + 0] = _v4.x;
                    cn[16 + 1] = _v4.y;
                    cn[16 + 2] = _v4.z;
                    cn[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 2176ll);
                    cn[20 + 0] = _v4.x;
                    cn[20 + 1] = _v4.y;
                    cn[20 + 2] = _v4.z;
                    cn[20 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 2304ll);
                    cn[24 + 0] = _v4.x;
                    cn[24 + 1] = _v4.y;
                    cn[24 + 2] = _v4.z;
                    cn[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first + g_off + 2432ll);
                    cn[28 + 0] = _v4.x;
                    cn[28 + 1] = _v4.y;
                    cn[28 + 2] = _v4.z;
                    cn[28 + 3] = _v4.w;
                }
            } else {
                cn[0] = 0.0f;
                cn[1] = 0.0f;
                cn[2] = 0.0f;
                cn[3] = 0.0f;
                cn[4] = 0.0f;
                cn[5] = 0.0f;
                cn[6] = 0.0f;
                cn[7] = 0.0f;
                cn[8] = 0.0f;
                cn[9] = 0.0f;
                cn[10] = 0.0f;
                cn[11] = 0.0f;
                cn[12] = 0.0f;
                cn[13] = 0.0f;
                cn[14] = 0.0f;
                cn[15] = 0.0f;
                cn[16] = 0.0f;
                cn[17] = 0.0f;
                cn[18] = 0.0f;
                cn[19] = 0.0f;
                cn[20] = 0.0f;
                cn[21] = 0.0f;
                cn[22] = 0.0f;
                cn[23] = 0.0f;
                cn[24] = 0.0f;
                cn[25] = 0.0f;
                cn[26] = 0.0f;
                cn[27] = 0.0f;
                cn[28] = 0.0f;
                cn[29] = 0.0f;
                cn[30] = 0.0f;
                cn[31] = 0.0f;
            }
        }
        long long x0_off = x0_index[chain];
        if (x0_off >= 0ll) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off);
                acc[0 + 0] = _v4.x;
                acc[0 + 1] = _v4.y;
                acc[0 + 2] = _v4.z;
                acc[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 128ll);
                acc[4 + 0] = _v4.x;
                acc[4 + 1] = _v4.y;
                acc[4 + 2] = _v4.z;
                acc[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 256ll);
                acc[8 + 0] = _v4.x;
                acc[8 + 1] = _v4.y;
                acc[8 + 2] = _v4.z;
                acc[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 384ll);
                acc[12 + 0] = _v4.x;
                acc[12 + 1] = _v4.y;
                acc[12 + 2] = _v4.z;
                acc[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 2048ll);
                acc[16 + 0] = _v4.x;
                acc[16 + 1] = _v4.y;
                acc[16 + 2] = _v4.z;
                acc[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 2176ll);
                acc[20 + 0] = _v4.x;
                acc[20 + 1] = _v4.y;
                acc[20 + 2] = _v4.z;
                acc[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 2304ll);
                acc[24 + 0] = _v4.x;
                acc[24 + 1] = _v4.y;
                acc[24 + 2] = _v4.z;
                acc[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off + g_off + 2432ll);
                acc[28 + 0] = _v4.x;
                acc[28 + 1] = _v4.y;
                acc[28 + 2] = _v4.z;
                acc[28 + 3] = _v4.w;
            }
        } else {
            int diag = row0 - v0 - cg * 4;
            if (col_tile == 0) {
                diag = -4096;
            }
            acc[0] = ((diag == 0) ? 1.0f : 0.0f);
            acc[1] = ((diag == 1) ? 1.0f : 0.0f);
            acc[2] = ((diag == 2) ? 1.0f : 0.0f);
            acc[3] = ((diag == 3) ? 1.0f : 0.0f);
            acc[4] = ((diag + 1 == 0) ? 1.0f : 0.0f);
            acc[5] = ((diag + 1 == 1) ? 1.0f : 0.0f);
            acc[6] = ((diag + 1 == 2) ? 1.0f : 0.0f);
            acc[7] = ((diag + 1 == 3) ? 1.0f : 0.0f);
            acc[8] = ((diag + 2 == 0) ? 1.0f : 0.0f);
            acc[9] = ((diag + 2 == 1) ? 1.0f : 0.0f);
            acc[10] = ((diag + 2 == 2) ? 1.0f : 0.0f);
            acc[11] = ((diag + 2 == 3) ? 1.0f : 0.0f);
            acc[12] = ((diag + 3 == 0) ? 1.0f : 0.0f);
            acc[13] = ((diag + 3 == 1) ? 1.0f : 0.0f);
            acc[14] = ((diag + 3 == 2) ? 1.0f : 0.0f);
            acc[15] = ((diag + 3 == 3) ? 1.0f : 0.0f);
            acc[16] = ((diag + 16 == 0) ? 1.0f : 0.0f);
            acc[17] = ((diag + 16 == 1) ? 1.0f : 0.0f);
            acc[18] = ((diag + 16 == 2) ? 1.0f : 0.0f);
            acc[19] = ((diag + 16 == 3) ? 1.0f : 0.0f);
            acc[20] = ((diag + 17 == 0) ? 1.0f : 0.0f);
            acc[21] = ((diag + 17 == 1) ? 1.0f : 0.0f);
            acc[22] = ((diag + 17 == 2) ? 1.0f : 0.0f);
            acc[23] = ((diag + 17 == 3) ? 1.0f : 0.0f);
            acc[24] = ((diag + 18 == 0) ? 1.0f : 0.0f);
            acc[25] = ((diag + 18 == 1) ? 1.0f : 0.0f);
            acc[26] = ((diag + 18 == 2) ? 1.0f : 0.0f);
            acc[27] = ((diag + 18 == 3) ? 1.0f : 0.0f);
            acc[28] = ((diag + 19 == 0) ? 1.0f : 0.0f);
            acc[29] = ((diag + 19 == 1) ? 1.0f : 0.0f);
            acc[30] = ((diag + 19 == 2) ? 1.0f : 0.0f);
            acc[31] = ((diag + 19 == 3) ? 1.0f : 0.0f);
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 144), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 288), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 432), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 2304), "r"(__float_as_uint(acc[16])), "r"(__float_as_uint(acc[17])), "r"(__float_as_uint(acc[18])), "r"(__float_as_uint(acc[19])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 2448), "r"(__float_as_uint(acc[20])), "r"(__float_as_uint(acc[21])), "r"(__float_as_uint(acc[22])), "r"(__float_as_uint(acc[23])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 2592), "r"(__float_as_uint(acc[24])), "r"(__float_as_uint(acc[25])), "r"(__float_as_uint(acc[26])), "r"(__float_as_uint(acc[27])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 2736), "r"(__float_as_uint(acc[28])), "r"(__float_as_uint(acc[29])), "r"(__float_as_uint(acc[30])), "r"(__float_as_uint(acc[31])) : "memory");
        if (e_begin < e_end) {
            if (write_states != 0) {
                long long y_first = elem_y[e_begin];
                if (y_first >= 0ll) {
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 128ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 256ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 384ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 2048ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 2176ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 2304ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                        *reinterpret_cast<float4*>(y + y_first + g_off + 2432ll) = _v4;
                    }
                }
            }
        } else {
            long long y_last = y_final[chain];
            if (y_last >= 0ll) {
                {
                    float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 128ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 256ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 384ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 2176ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 2304ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                    *reinterpret_cast<float4*>(y + y_last + g_off + 2432ll) = _v4;
                }
            }
        }
        __syncthreads();
        unsigned int buf = 0;
        #pragma unroll 1
        for (int e = e_begin; e < e_end; e++) {
            int has_next = 0;
            long long m_next = 0;
            long long c_next = -1;
            long long y_pub = -1;
            if (e + 1 < e_end) {
                has_next = 1;
                m_next = elem_m[e + 1];
                if (col_tile == 0) {
                    c_next = elem_c[e + 1];
                }
                if (write_states != 0) {
                    y_pub = elem_y[e + 1];
                }
            } else {
                y_pub = y_final[chain];
            }
            acc[0] = cn[0];
            acc[1] = cn[1];
            acc[2] = cn[2];
            acc[3] = cn[3];
            acc[4] = cn[4];
            acc[5] = cn[5];
            acc[6] = cn[6];
            acc[7] = cn[7];
            acc[8] = cn[8];
            acc[9] = cn[9];
            acc[10] = cn[10];
            acc[11] = cn[11];
            acc[12] = cn[12];
            acc[13] = cn[13];
            acc[14] = cn[14];
            acc[15] = cn[15];
            acc[16] = cn[16];
            acc[17] = cn[17];
            acc[18] = cn[18];
            acc[19] = cn[19];
            acc[20] = cn[20];
            acc[21] = cn[21];
            acc[22] = cn[22];
            acc[23] = cn[23];
            acc[24] = cn[24];
            acc[25] = cn[25];
            acc[26] = cn[26];
            acc[27] = cn[27];
            acc[28] = cn[28];
            acc[29] = cn[29];
            acc[30] = cn[30];
            acc[31] = cn[31];
            unsigned int x_rd_e = x_rd + buf * 18432;
            #pragma unroll 1
            for (int ch = 0; ch < 4; ch++) {
                asm volatile("cp.async.wait_group 2;");
                __syncwarp();
                int nc = ch + 3;
                if (nc < 4) {
                    long long src0_2 = m_cur + (long long)(cp_src + nc * 4096);
                    unsigned int dst0_2 = cp_dst + (unsigned int)(nc % 4 * 4608);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2), "l"(m + src0_2));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 512), "l"(m + (src0_2 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 1024), "l"(m + (src0_2 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 1536), "l"(m + (src0_2 + 1536ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 2048), "l"(m + (src0_2 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 2560), "l"(m + (src0_2 + 2560ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 3072), "l"(m + (src0_2 + 3072ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 3584), "l"(m + (src0_2 + 3584ll)));
                } else if (has_next != 0) {
                    long long src0_3 = m_next + (long long)(cp_src + (nc - 4) * 4096);
                    unsigned int dst0_3 = cp_dst + (unsigned int)(nc % 4 * 4608);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3), "l"(m + src0_3));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 512), "l"(m + (src0_3 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 1024), "l"(m + (src0_3 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 1536), "l"(m + (src0_3 + 1536ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 2048), "l"(m + (src0_3 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 2560), "l"(m + (src0_3 + 2560ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 3072), "l"(m + (src0_3 + 3072ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 3584), "l"(m + (src0_3 + 3584ll)));
                }
                asm volatile("cp.async.commit_group;");
                if (ch == 1) {
                    if (c_next >= 0ll) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off);
                            cn[0 + 0] = _v4.x;
                            cn[0 + 1] = _v4.y;
                            cn[0 + 2] = _v4.z;
                            cn[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 128ll);
                            cn[4 + 0] = _v4.x;
                            cn[4 + 1] = _v4.y;
                            cn[4 + 2] = _v4.z;
                            cn[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 256ll);
                            cn[8 + 0] = _v4.x;
                            cn[8 + 1] = _v4.y;
                            cn[8 + 2] = _v4.z;
                            cn[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 384ll);
                            cn[12 + 0] = _v4.x;
                            cn[12 + 1] = _v4.y;
                            cn[12 + 2] = _v4.z;
                            cn[12 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 2048ll);
                            cn[16 + 0] = _v4.x;
                            cn[16 + 1] = _v4.y;
                            cn[16 + 2] = _v4.z;
                            cn[16 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 2176ll);
                            cn[20 + 0] = _v4.x;
                            cn[20 + 1] = _v4.y;
                            cn[20 + 2] = _v4.z;
                            cn[20 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 2304ll);
                            cn[24 + 0] = _v4.x;
                            cn[24 + 1] = _v4.y;
                            cn[24 + 2] = _v4.z;
                            cn[24 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next + g_off + 2432ll);
                            cn[28 + 0] = _v4.x;
                            cn[28 + 1] = _v4.y;
                            cn[28 + 2] = _v4.z;
                            cn[28 + 3] = _v4.w;
                        }
                    } else {
                        cn[0] = 0.0f;
                        cn[1] = 0.0f;
                        cn[2] = 0.0f;
                        cn[3] = 0.0f;
                        cn[4] = 0.0f;
                        cn[5] = 0.0f;
                        cn[6] = 0.0f;
                        cn[7] = 0.0f;
                        cn[8] = 0.0f;
                        cn[9] = 0.0f;
                        cn[10] = 0.0f;
                        cn[11] = 0.0f;
                        cn[12] = 0.0f;
                        cn[13] = 0.0f;
                        cn[14] = 0.0f;
                        cn[15] = 0.0f;
                        cn[16] = 0.0f;
                        cn[17] = 0.0f;
                        cn[18] = 0.0f;
                        cn[19] = 0.0f;
                        cn[20] = 0.0f;
                        cn[21] = 0.0f;
                        cn[22] = 0.0f;
                        cn[23] = 0.0f;
                        cn[24] = 0.0f;
                        cn[25] = 0.0f;
                        cn[26] = 0.0f;
                        cn[27] = 0.0f;
                        cn[28] = 0.0f;
                        cn[29] = 0.0f;
                        cn[30] = 0.0f;
                        cn[31] = 0.0f;
                    }
                }
                #pragma unroll
                for (int kk = 0; kk < 32; kk++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[0])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 3]))
                        : "r"(m_rd + (unsigned int)(ch % 4 * 4608) + (unsigned int)(kk * 128)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[4])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 3]))
                        : "r"(m_rd + (unsigned int)(ch % 4 * 4608) + (unsigned int)(kk * 128) + 64));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e + (unsigned int)(ch * 4608) + (unsigned int)(kk * 144)));
#if __CUDA_ARCH__ == 1070
                    acc[0] = __fmaf_rn(mv[0], xv[0], acc[0]);
                    acc[1] = __fmaf_rn(mv[0], xv[1], acc[1]);
                    acc[2] = __fmaf_rn(mv[0], xv[2], acc[2]);
                    acc[3] = __fmaf_rn(mv[0], xv[3], acc[3]);
                    acc[4] = __fmaf_rn(mv[1], xv[0], acc[4]);
                    acc[5] = __fmaf_rn(mv[1], xv[1], acc[5]);
                    acc[6] = __fmaf_rn(mv[1], xv[2], acc[6]);
                    acc[7] = __fmaf_rn(mv[1], xv[3], acc[7]);
                    acc[8] = __fmaf_rn(mv[2], xv[0], acc[8]);
                    acc[9] = __fmaf_rn(mv[2], xv[1], acc[9]);
                    acc[10] = __fmaf_rn(mv[2], xv[2], acc[10]);
                    acc[11] = __fmaf_rn(mv[2], xv[3], acc[11]);
                    acc[12] = __fmaf_rn(mv[3], xv[0], acc[12]);
                    acc[13] = __fmaf_rn(mv[3], xv[1], acc[13]);
                    acc[14] = __fmaf_rn(mv[3], xv[2], acc[14]);
                    acc[15] = __fmaf_rn(mv[3], xv[3], acc[15]);
                    acc[16] = __fmaf_rn(mv[4], xv[0], acc[16]);
                    acc[17] = __fmaf_rn(mv[4], xv[1], acc[17]);
                    acc[18] = __fmaf_rn(mv[4], xv[2], acc[18]);
                    acc[19] = __fmaf_rn(mv[4], xv[3], acc[19]);
                    acc[20] = __fmaf_rn(mv[5], xv[0], acc[20]);
                    acc[21] = __fmaf_rn(mv[5], xv[1], acc[21]);
                    acc[22] = __fmaf_rn(mv[5], xv[2], acc[22]);
                    acc[23] = __fmaf_rn(mv[5], xv[3], acc[23]);
                    acc[24] = __fmaf_rn(mv[6], xv[0], acc[24]);
                    acc[25] = __fmaf_rn(mv[6], xv[1], acc[25]);
                    acc[26] = __fmaf_rn(mv[6], xv[2], acc[26]);
                    acc[27] = __fmaf_rn(mv[6], xv[3], acc[27]);
                    acc[28] = __fmaf_rn(mv[7], xv[0], acc[28]);
                    acc[29] = __fmaf_rn(mv[7], xv[1], acc[29]);
                    acc[30] = __fmaf_rn(mv[7], xv[2], acc[30]);
                    acc[31] = __fmaf_rn(mv[7], xv[3], acc[31]);
#else
                    float2 m2_0 = make_float2(mv[0], mv[0]);
                    float2 a2_0 = make_float2(acc[0], acc[1]);
                    a2_0 = fma_f32x2_rn_ftz(m2_0, make_float2(xv[0], xv[1]), a2_0);
                    acc[0] = a2_0.x;
                    acc[1] = a2_0.y;
                    float2 a2_1 = make_float2(acc[2], acc[3]);
                    a2_1 = fma_f32x2_rn_ftz(m2_0, make_float2(xv[2], xv[3]), a2_1);
                    acc[2] = a2_1.x;
                    acc[3] = a2_1.y;
                    float2 m2_1 = make_float2(mv[1], mv[1]);
                    float2 a2_2 = make_float2(acc[4], acc[5]);
                    a2_2 = fma_f32x2_rn_ftz(m2_1, make_float2(xv[0], xv[1]), a2_2);
                    acc[4] = a2_2.x;
                    acc[5] = a2_2.y;
                    float2 a2_3 = make_float2(acc[6], acc[7]);
                    a2_3 = fma_f32x2_rn_ftz(m2_1, make_float2(xv[2], xv[3]), a2_3);
                    acc[6] = a2_3.x;
                    acc[7] = a2_3.y;
                    float2 m2_2 = make_float2(mv[2], mv[2]);
                    float2 a2_4 = make_float2(acc[8], acc[9]);
                    a2_4 = fma_f32x2_rn_ftz(m2_2, make_float2(xv[0], xv[1]), a2_4);
                    acc[8] = a2_4.x;
                    acc[9] = a2_4.y;
                    float2 a2_5 = make_float2(acc[10], acc[11]);
                    a2_5 = fma_f32x2_rn_ftz(m2_2, make_float2(xv[2], xv[3]), a2_5);
                    acc[10] = a2_5.x;
                    acc[11] = a2_5.y;
                    float2 m2_3 = make_float2(mv[3], mv[3]);
                    float2 a2_6 = make_float2(acc[12], acc[13]);
                    a2_6 = fma_f32x2_rn_ftz(m2_3, make_float2(xv[0], xv[1]), a2_6);
                    acc[12] = a2_6.x;
                    acc[13] = a2_6.y;
                    float2 a2_7 = make_float2(acc[14], acc[15]);
                    a2_7 = fma_f32x2_rn_ftz(m2_3, make_float2(xv[2], xv[3]), a2_7);
                    acc[14] = a2_7.x;
                    acc[15] = a2_7.y;
                    float2 m2_4 = make_float2(mv[4], mv[4]);
                    float2 a2_8 = make_float2(acc[16], acc[17]);
                    a2_8 = fma_f32x2_rn_ftz(m2_4, make_float2(xv[0], xv[1]), a2_8);
                    acc[16] = a2_8.x;
                    acc[17] = a2_8.y;
                    float2 a2_9 = make_float2(acc[18], acc[19]);
                    a2_9 = fma_f32x2_rn_ftz(m2_4, make_float2(xv[2], xv[3]), a2_9);
                    acc[18] = a2_9.x;
                    acc[19] = a2_9.y;
                    float2 m2_5 = make_float2(mv[5], mv[5]);
                    float2 a2_10 = make_float2(acc[20], acc[21]);
                    a2_10 = fma_f32x2_rn_ftz(m2_5, make_float2(xv[0], xv[1]), a2_10);
                    acc[20] = a2_10.x;
                    acc[21] = a2_10.y;
                    float2 a2_11 = make_float2(acc[22], acc[23]);
                    a2_11 = fma_f32x2_rn_ftz(m2_5, make_float2(xv[2], xv[3]), a2_11);
                    acc[22] = a2_11.x;
                    acc[23] = a2_11.y;
                    float2 m2_6 = make_float2(mv[6], mv[6]);
                    float2 a2_12 = make_float2(acc[24], acc[25]);
                    a2_12 = fma_f32x2_rn_ftz(m2_6, make_float2(xv[0], xv[1]), a2_12);
                    acc[24] = a2_12.x;
                    acc[25] = a2_12.y;
                    float2 a2_13 = make_float2(acc[26], acc[27]);
                    a2_13 = fma_f32x2_rn_ftz(m2_6, make_float2(xv[2], xv[3]), a2_13);
                    acc[26] = a2_13.x;
                    acc[27] = a2_13.y;
                    float2 m2_7 = make_float2(mv[7], mv[7]);
                    float2 a2_14 = make_float2(acc[28], acc[29]);
                    a2_14 = fma_f32x2_rn_ftz(m2_7, make_float2(xv[0], xv[1]), a2_14);
                    acc[28] = a2_14.x;
                    acc[29] = a2_14.y;
                    float2 a2_15 = make_float2(acc[30], acc[31]);
                    a2_15 = fma_f32x2_rn_ftz(m2_7, make_float2(xv[2], xv[3]), a2_15);
                    acc[30] = a2_15.x;
                    acc[31] = a2_15.y;
#endif
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432)), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 144), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 288), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 432), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 2304), "r"(__float_as_uint(acc[16])), "r"(__float_as_uint(acc[17])), "r"(__float_as_uint(acc[18])), "r"(__float_as_uint(acc[19])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 2448), "r"(__float_as_uint(acc[20])), "r"(__float_as_uint(acc[21])), "r"(__float_as_uint(acc[22])), "r"(__float_as_uint(acc[23])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 2592), "r"(__float_as_uint(acc[24])), "r"(__float_as_uint(acc[25])), "r"(__float_as_uint(acc[26])), "r"(__float_as_uint(acc[27])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (18432 - buf * 18432) + 2736), "r"(__float_as_uint(acc[28])), "r"(__float_as_uint(acc[29])), "r"(__float_as_uint(acc[30])), "r"(__float_as_uint(acc[31])) : "memory");
            if (y_pub >= 0ll) {
                {
                    float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 128ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 256ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 384ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 2176ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 2304ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub + g_off + 2432ll) = _v4;
                }
            }
            __syncthreads();
            buf = 1 - buf;
            m_cur = m_next;
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    } else {
        int row0_1 = warp_1 * 32 + g;
        long long g_off_1 = (long long)col_tile * 16384ll + (long long)(row0_1 * 128 + v0 + cg * 4);
        unsigned int x_thr_1 = xs_addr + (unsigned int)((row0_1 * 36 + cg * 4) * 4);
        unsigned int x_rd_1 = xs_addr + (unsigned int)(cg * 16);
        unsigned int m_rd_1 = ring_w + (unsigned int)(g * 144);
        int cp_i_1 = lane / 8;
        int cp_j_1 = lane % 8;
        int cp_src_1 = cp_i_1 * 128 + warp_1 * 4096 + 4 * cp_j_1;
        unsigned int cp_dst_1 = ring_w + (unsigned int)((cp_i_1 * 36 + 4 * cp_j_1) * 4);
        long long m_cur_1 = 0;
        if (e_begin < e_end) {
            m_cur_1 = elem_m[e_begin];
            long long src0_4 = m_cur_1 + (long long)cp_src_1;
            unsigned int dst0_4 = cp_dst_1 + 0;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4), "l"(m + src0_4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 576), "l"(m + (src0_4 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 1152), "l"(m + (src0_4 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 1728), "l"(m + (src0_4 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 2304), "l"(m + (src0_4 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 2880), "l"(m + (src0_4 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 3456), "l"(m + (src0_4 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 4032), "l"(m + (src0_4 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_2_1 = m_cur_1 + (long long)(cp_src_1 + 32);
            unsigned int dst0_2_1 = cp_dst_1 + 4608;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1), "l"(m + src0_2_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 576), "l"(m + (src0_2_1 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 1152), "l"(m + (src0_2_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 1728), "l"(m + (src0_2_1 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 2304), "l"(m + (src0_2_1 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 2880), "l"(m + (src0_2_1 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 3456), "l"(m + (src0_2_1 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 4032), "l"(m + (src0_2_1 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_3_1 = m_cur_1 + (long long)(cp_src_1 + 64);
            unsigned int dst0_3_1 = cp_dst_1 + 9216;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1), "l"(m + src0_3_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 576), "l"(m + (src0_3_1 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 1152), "l"(m + (src0_3_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 1728), "l"(m + (src0_3_1 + 1536ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 2304), "l"(m + (src0_3_1 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 2880), "l"(m + (src0_3_1 + 2560ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 3456), "l"(m + (src0_3_1 + 3072ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 4032), "l"(m + (src0_3_1 + 3584ll)));
            asm volatile("cp.async.commit_group;");
            long long c_first_1 = -1;
            if (col_tile == 0) {
                c_first_1 = elem_c[e_begin];
            }
            if (c_first_1 >= 0ll) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1);
                    cn[0 + 0] = _v4.x;
                    cn[0 + 1] = _v4.y;
                    cn[0 + 2] = _v4.z;
                    cn[0 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 512ll);
                    cn[4 + 0] = _v4.x;
                    cn[4 + 1] = _v4.y;
                    cn[4 + 2] = _v4.z;
                    cn[4 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 1024ll);
                    cn[8 + 0] = _v4.x;
                    cn[8 + 1] = _v4.y;
                    cn[8 + 2] = _v4.z;
                    cn[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 1536ll);
                    cn[12 + 0] = _v4.x;
                    cn[12 + 1] = _v4.y;
                    cn[12 + 2] = _v4.z;
                    cn[12 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 2048ll);
                    cn[16 + 0] = _v4.x;
                    cn[16 + 1] = _v4.y;
                    cn[16 + 2] = _v4.z;
                    cn[16 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 2560ll);
                    cn[20 + 0] = _v4.x;
                    cn[20 + 1] = _v4.y;
                    cn[20 + 2] = _v4.z;
                    cn[20 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 3072ll);
                    cn[24 + 0] = _v4.x;
                    cn[24 + 1] = _v4.y;
                    cn[24 + 2] = _v4.z;
                    cn[24 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 3584ll);
                    cn[28 + 0] = _v4.x;
                    cn[28 + 1] = _v4.y;
                    cn[28 + 2] = _v4.z;
                    cn[28 + 3] = _v4.w;
                }
            } else {
                cn[0] = 0.0f;
                cn[1] = 0.0f;
                cn[2] = 0.0f;
                cn[3] = 0.0f;
                cn[4] = 0.0f;
                cn[5] = 0.0f;
                cn[6] = 0.0f;
                cn[7] = 0.0f;
                cn[8] = 0.0f;
                cn[9] = 0.0f;
                cn[10] = 0.0f;
                cn[11] = 0.0f;
                cn[12] = 0.0f;
                cn[13] = 0.0f;
                cn[14] = 0.0f;
                cn[15] = 0.0f;
                cn[16] = 0.0f;
                cn[17] = 0.0f;
                cn[18] = 0.0f;
                cn[19] = 0.0f;
                cn[20] = 0.0f;
                cn[21] = 0.0f;
                cn[22] = 0.0f;
                cn[23] = 0.0f;
                cn[24] = 0.0f;
                cn[25] = 0.0f;
                cn[26] = 0.0f;
                cn[27] = 0.0f;
                cn[28] = 0.0f;
                cn[29] = 0.0f;
                cn[30] = 0.0f;
                cn[31] = 0.0f;
            }
        }
        long long x0_off_1 = x0_index[chain];
        if (x0_off_1 >= 0ll) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1);
                acc[0 + 0] = _v4.x;
                acc[0 + 1] = _v4.y;
                acc[0 + 2] = _v4.z;
                acc[0 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 512ll);
                acc[4 + 0] = _v4.x;
                acc[4 + 1] = _v4.y;
                acc[4 + 2] = _v4.z;
                acc[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 1024ll);
                acc[8 + 0] = _v4.x;
                acc[8 + 1] = _v4.y;
                acc[8 + 2] = _v4.z;
                acc[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 1536ll);
                acc[12 + 0] = _v4.x;
                acc[12 + 1] = _v4.y;
                acc[12 + 2] = _v4.z;
                acc[12 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 2048ll);
                acc[16 + 0] = _v4.x;
                acc[16 + 1] = _v4.y;
                acc[16 + 2] = _v4.z;
                acc[16 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 2560ll);
                acc[20 + 0] = _v4.x;
                acc[20 + 1] = _v4.y;
                acc[20 + 2] = _v4.z;
                acc[20 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 3072ll);
                acc[24 + 0] = _v4.x;
                acc[24 + 1] = _v4.y;
                acc[24 + 2] = _v4.z;
                acc[24 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 3584ll);
                acc[28 + 0] = _v4.x;
                acc[28 + 1] = _v4.y;
                acc[28 + 2] = _v4.z;
                acc[28 + 3] = _v4.w;
            }
        } else {
            int diag_1 = row0_1 - v0 - cg * 4;
            if (col_tile == 0) {
                diag_1 = -4096;
            }
            acc[0] = ((diag_1 == 0) ? 1.0f : 0.0f);
            acc[1] = ((diag_1 == 1) ? 1.0f : 0.0f);
            acc[2] = ((diag_1 == 2) ? 1.0f : 0.0f);
            acc[3] = ((diag_1 == 3) ? 1.0f : 0.0f);
            acc[4] = ((diag_1 + 4 == 0) ? 1.0f : 0.0f);
            acc[5] = ((diag_1 + 4 == 1) ? 1.0f : 0.0f);
            acc[6] = ((diag_1 + 4 == 2) ? 1.0f : 0.0f);
            acc[7] = ((diag_1 + 4 == 3) ? 1.0f : 0.0f);
            acc[8] = ((diag_1 + 8 == 0) ? 1.0f : 0.0f);
            acc[9] = ((diag_1 + 8 == 1) ? 1.0f : 0.0f);
            acc[10] = ((diag_1 + 8 == 2) ? 1.0f : 0.0f);
            acc[11] = ((diag_1 + 8 == 3) ? 1.0f : 0.0f);
            acc[12] = ((diag_1 + 12 == 0) ? 1.0f : 0.0f);
            acc[13] = ((diag_1 + 12 == 1) ? 1.0f : 0.0f);
            acc[14] = ((diag_1 + 12 == 2) ? 1.0f : 0.0f);
            acc[15] = ((diag_1 + 12 == 3) ? 1.0f : 0.0f);
            acc[16] = ((diag_1 + 16 == 0) ? 1.0f : 0.0f);
            acc[17] = ((diag_1 + 16 == 1) ? 1.0f : 0.0f);
            acc[18] = ((diag_1 + 16 == 2) ? 1.0f : 0.0f);
            acc[19] = ((diag_1 + 16 == 3) ? 1.0f : 0.0f);
            acc[20] = ((diag_1 + 20 == 0) ? 1.0f : 0.0f);
            acc[21] = ((diag_1 + 20 == 1) ? 1.0f : 0.0f);
            acc[22] = ((diag_1 + 20 == 2) ? 1.0f : 0.0f);
            acc[23] = ((diag_1 + 20 == 3) ? 1.0f : 0.0f);
            acc[24] = ((diag_1 + 24 == 0) ? 1.0f : 0.0f);
            acc[25] = ((diag_1 + 24 == 1) ? 1.0f : 0.0f);
            acc[26] = ((diag_1 + 24 == 2) ? 1.0f : 0.0f);
            acc[27] = ((diag_1 + 24 == 3) ? 1.0f : 0.0f);
            acc[28] = ((diag_1 + 28 == 0) ? 1.0f : 0.0f);
            acc[29] = ((diag_1 + 28 == 1) ? 1.0f : 0.0f);
            acc[30] = ((diag_1 + 28 == 2) ? 1.0f : 0.0f);
            acc[31] = ((diag_1 + 28 == 3) ? 1.0f : 0.0f);
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 576), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 1152), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 1728), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 2304), "r"(__float_as_uint(acc[16])), "r"(__float_as_uint(acc[17])), "r"(__float_as_uint(acc[18])), "r"(__float_as_uint(acc[19])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 2880), "r"(__float_as_uint(acc[20])), "r"(__float_as_uint(acc[21])), "r"(__float_as_uint(acc[22])), "r"(__float_as_uint(acc[23])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 3456), "r"(__float_as_uint(acc[24])), "r"(__float_as_uint(acc[25])), "r"(__float_as_uint(acc[26])), "r"(__float_as_uint(acc[27])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 4032), "r"(__float_as_uint(acc[28])), "r"(__float_as_uint(acc[29])), "r"(__float_as_uint(acc[30])), "r"(__float_as_uint(acc[31])) : "memory");
        if (e_begin < e_end) {
            if (write_states != 0) {
                long long y_first_1 = elem_y[e_begin];
                if (y_first_1 >= 0ll) {
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 512ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 1024ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 1536ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 2048ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 2560ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 3072ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 3584ll) = _v4;
                    }
                }
            }
        } else {
            long long y_last_1 = y_final[chain];
            if (y_last_1 >= 0ll) {
                {
                    float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 512ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 1024ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 1536ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 2560ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 3072ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 3584ll) = _v4;
                }
            }
        }
        __syncthreads();
        unsigned int buf_1 = 0;
        #pragma unroll 1
        for (int e_1 = e_begin; e_1 < e_end; e_1++) {
            int has_next_1 = 0;
            long long m_next_1 = 0;
            long long c_next_1 = -1;
            long long y_pub_1 = -1;
            if (e_1 + 1 < e_end) {
                has_next_1 = 1;
                m_next_1 = elem_m[e_1 + 1];
                if (col_tile == 0) {
                    c_next_1 = elem_c[e_1 + 1];
                }
                if (write_states != 0) {
                    y_pub_1 = elem_y[e_1 + 1];
                }
            } else {
                y_pub_1 = y_final[chain];
            }
            acc[0] = cn[0];
            acc[1] = cn[1];
            acc[2] = cn[2];
            acc[3] = cn[3];
            acc[4] = cn[4];
            acc[5] = cn[5];
            acc[6] = cn[6];
            acc[7] = cn[7];
            acc[8] = cn[8];
            acc[9] = cn[9];
            acc[10] = cn[10];
            acc[11] = cn[11];
            acc[12] = cn[12];
            acc[13] = cn[13];
            acc[14] = cn[14];
            acc[15] = cn[15];
            acc[16] = cn[16];
            acc[17] = cn[17];
            acc[18] = cn[18];
            acc[19] = cn[19];
            acc[20] = cn[20];
            acc[21] = cn[21];
            acc[22] = cn[22];
            acc[23] = cn[23];
            acc[24] = cn[24];
            acc[25] = cn[25];
            acc[26] = cn[26];
            acc[27] = cn[27];
            acc[28] = cn[28];
            acc[29] = cn[29];
            acc[30] = cn[30];
            acc[31] = cn[31];
            unsigned int x_rd_e_1 = x_rd_1 + buf_1 * 18432;
            #pragma unroll 1
            for (int ch_1 = 0; ch_1 < 4; ch_1++) {
                asm volatile("cp.async.wait_group 2;");
                __syncwarp();
                int nc_1 = ch_1 + 3;
                if (nc_1 < 4) {
                    long long src0_5 = m_cur_1 + (long long)(cp_src_1 + nc_1 * 32);
                    unsigned int dst0_5 = cp_dst_1 + (unsigned int)(nc_1 % 4 * 4608);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5), "l"(m + src0_5));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 576), "l"(m + (src0_5 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 1152), "l"(m + (src0_5 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 1728), "l"(m + (src0_5 + 1536ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 2304), "l"(m + (src0_5 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 2880), "l"(m + (src0_5 + 2560ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 3456), "l"(m + (src0_5 + 3072ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 4032), "l"(m + (src0_5 + 3584ll)));
                } else if (has_next_1 != 0) {
                    long long src0_6 = m_next_1 + (long long)(cp_src_1 + (nc_1 - 4) * 32);
                    unsigned int dst0_6 = cp_dst_1 + (unsigned int)(nc_1 % 4 * 4608);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6), "l"(m + src0_6));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 576), "l"(m + (src0_6 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 1152), "l"(m + (src0_6 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 1728), "l"(m + (src0_6 + 1536ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 2304), "l"(m + (src0_6 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 2880), "l"(m + (src0_6 + 2560ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 3456), "l"(m + (src0_6 + 3072ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 4032), "l"(m + (src0_6 + 3584ll)));
                }
                asm volatile("cp.async.commit_group;");
                if (ch_1 == 1) {
                    if (c_next_1 >= 0ll) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1);
                            cn[0 + 0] = _v4.x;
                            cn[0 + 1] = _v4.y;
                            cn[0 + 2] = _v4.z;
                            cn[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 512ll);
                            cn[4 + 0] = _v4.x;
                            cn[4 + 1] = _v4.y;
                            cn[4 + 2] = _v4.z;
                            cn[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 1024ll);
                            cn[8 + 0] = _v4.x;
                            cn[8 + 1] = _v4.y;
                            cn[8 + 2] = _v4.z;
                            cn[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 1536ll);
                            cn[12 + 0] = _v4.x;
                            cn[12 + 1] = _v4.y;
                            cn[12 + 2] = _v4.z;
                            cn[12 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 2048ll);
                            cn[16 + 0] = _v4.x;
                            cn[16 + 1] = _v4.y;
                            cn[16 + 2] = _v4.z;
                            cn[16 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 2560ll);
                            cn[20 + 0] = _v4.x;
                            cn[20 + 1] = _v4.y;
                            cn[20 + 2] = _v4.z;
                            cn[20 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 3072ll);
                            cn[24 + 0] = _v4.x;
                            cn[24 + 1] = _v4.y;
                            cn[24 + 2] = _v4.z;
                            cn[24 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 3584ll);
                            cn[28 + 0] = _v4.x;
                            cn[28 + 1] = _v4.y;
                            cn[28 + 2] = _v4.z;
                            cn[28 + 3] = _v4.w;
                        }
                    } else {
                        cn[0] = 0.0f;
                        cn[1] = 0.0f;
                        cn[2] = 0.0f;
                        cn[3] = 0.0f;
                        cn[4] = 0.0f;
                        cn[5] = 0.0f;
                        cn[6] = 0.0f;
                        cn[7] = 0.0f;
                        cn[8] = 0.0f;
                        cn[9] = 0.0f;
                        cn[10] = 0.0f;
                        cn[11] = 0.0f;
                        cn[12] = 0.0f;
                        cn[13] = 0.0f;
                        cn[14] = 0.0f;
                        cn[15] = 0.0f;
                        cn[16] = 0.0f;
                        cn[17] = 0.0f;
                        cn[18] = 0.0f;
                        cn[19] = 0.0f;
                        cn[20] = 0.0f;
                        cn[21] = 0.0f;
                        cn[22] = 0.0f;
                        cn[23] = 0.0f;
                        cn[24] = 0.0f;
                        cn[25] = 0.0f;
                        cn[26] = 0.0f;
                        cn[27] = 0.0f;
                        cn[28] = 0.0f;
                        cn[29] = 0.0f;
                        cn[30] = 0.0f;
                        cn[31] = 0.0f;
                    }
                }
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[0])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[4])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 576));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[8])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 1152));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[12])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 1728));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[16])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(16) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 2304));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[20])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(20) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 2880));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[24])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(24) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 3456));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[28])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(28) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 4608) + (unsigned int)(q * 16) + 4032));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 4608) + (unsigned int)(q * 576)));
#if __CUDA_ARCH__ == 1070
                    acc[0] = __fmaf_rn(mv[0], xv[0], acc[0]);
                    acc[1] = __fmaf_rn(mv[0], xv[1], acc[1]);
                    acc[2] = __fmaf_rn(mv[0], xv[2], acc[2]);
                    acc[3] = __fmaf_rn(mv[0], xv[3], acc[3]);
                    acc[4] = __fmaf_rn(mv[4], xv[0], acc[4]);
                    acc[5] = __fmaf_rn(mv[4], xv[1], acc[5]);
                    acc[6] = __fmaf_rn(mv[4], xv[2], acc[6]);
                    acc[7] = __fmaf_rn(mv[4], xv[3], acc[7]);
                    acc[8] = __fmaf_rn(mv[8], xv[0], acc[8]);
                    acc[9] = __fmaf_rn(mv[8], xv[1], acc[9]);
                    acc[10] = __fmaf_rn(mv[8], xv[2], acc[10]);
                    acc[11] = __fmaf_rn(mv[8], xv[3], acc[11]);
                    acc[12] = __fmaf_rn(mv[12], xv[0], acc[12]);
                    acc[13] = __fmaf_rn(mv[12], xv[1], acc[13]);
                    acc[14] = __fmaf_rn(mv[12], xv[2], acc[14]);
                    acc[15] = __fmaf_rn(mv[12], xv[3], acc[15]);
                    acc[16] = __fmaf_rn(mv[16], xv[0], acc[16]);
                    acc[17] = __fmaf_rn(mv[16], xv[1], acc[17]);
                    acc[18] = __fmaf_rn(mv[16], xv[2], acc[18]);
                    acc[19] = __fmaf_rn(mv[16], xv[3], acc[19]);
                    acc[20] = __fmaf_rn(mv[20], xv[0], acc[20]);
                    acc[21] = __fmaf_rn(mv[20], xv[1], acc[21]);
                    acc[22] = __fmaf_rn(mv[20], xv[2], acc[22]);
                    acc[23] = __fmaf_rn(mv[20], xv[3], acc[23]);
                    acc[24] = __fmaf_rn(mv[24], xv[0], acc[24]);
                    acc[25] = __fmaf_rn(mv[24], xv[1], acc[25]);
                    acc[26] = __fmaf_rn(mv[24], xv[2], acc[26]);
                    acc[27] = __fmaf_rn(mv[24], xv[3], acc[27]);
                    acc[28] = __fmaf_rn(mv[28], xv[0], acc[28]);
                    acc[29] = __fmaf_rn(mv[28], xv[1], acc[29]);
                    acc[30] = __fmaf_rn(mv[28], xv[2], acc[30]);
                    acc[31] = __fmaf_rn(mv[28], xv[3], acc[31]);
#else
                    float2 m2_8 = make_float2(mv[0], mv[0]);
                    float2 a2_16 = make_float2(acc[0], acc[1]);
                    a2_16 = fma_f32x2_rn_ftz(m2_8, make_float2(xv[0], xv[1]), a2_16);
                    acc[0] = a2_16.x;
                    acc[1] = a2_16.y;
                    float2 a2_17 = make_float2(acc[2], acc[3]);
                    a2_17 = fma_f32x2_rn_ftz(m2_8, make_float2(xv[2], xv[3]), a2_17);
                    acc[2] = a2_17.x;
                    acc[3] = a2_17.y;
                    float2 m2_9 = make_float2(mv[4], mv[4]);
                    float2 a2_18 = make_float2(acc[4], acc[5]);
                    a2_18 = fma_f32x2_rn_ftz(m2_9, make_float2(xv[0], xv[1]), a2_18);
                    acc[4] = a2_18.x;
                    acc[5] = a2_18.y;
                    float2 a2_19 = make_float2(acc[6], acc[7]);
                    a2_19 = fma_f32x2_rn_ftz(m2_9, make_float2(xv[2], xv[3]), a2_19);
                    acc[6] = a2_19.x;
                    acc[7] = a2_19.y;
                    float2 m2_10 = make_float2(mv[8], mv[8]);
                    float2 a2_20 = make_float2(acc[8], acc[9]);
                    a2_20 = fma_f32x2_rn_ftz(m2_10, make_float2(xv[0], xv[1]), a2_20);
                    acc[8] = a2_20.x;
                    acc[9] = a2_20.y;
                    float2 a2_21 = make_float2(acc[10], acc[11]);
                    a2_21 = fma_f32x2_rn_ftz(m2_10, make_float2(xv[2], xv[3]), a2_21);
                    acc[10] = a2_21.x;
                    acc[11] = a2_21.y;
                    float2 m2_11 = make_float2(mv[12], mv[12]);
                    float2 a2_22 = make_float2(acc[12], acc[13]);
                    a2_22 = fma_f32x2_rn_ftz(m2_11, make_float2(xv[0], xv[1]), a2_22);
                    acc[12] = a2_22.x;
                    acc[13] = a2_22.y;
                    float2 a2_23 = make_float2(acc[14], acc[15]);
                    a2_23 = fma_f32x2_rn_ftz(m2_11, make_float2(xv[2], xv[3]), a2_23);
                    acc[14] = a2_23.x;
                    acc[15] = a2_23.y;
                    float2 m2_12 = make_float2(mv[16], mv[16]);
                    float2 a2_24 = make_float2(acc[16], acc[17]);
                    a2_24 = fma_f32x2_rn_ftz(m2_12, make_float2(xv[0], xv[1]), a2_24);
                    acc[16] = a2_24.x;
                    acc[17] = a2_24.y;
                    float2 a2_25 = make_float2(acc[18], acc[19]);
                    a2_25 = fma_f32x2_rn_ftz(m2_12, make_float2(xv[2], xv[3]), a2_25);
                    acc[18] = a2_25.x;
                    acc[19] = a2_25.y;
                    float2 m2_13 = make_float2(mv[20], mv[20]);
                    float2 a2_26 = make_float2(acc[20], acc[21]);
                    a2_26 = fma_f32x2_rn_ftz(m2_13, make_float2(xv[0], xv[1]), a2_26);
                    acc[20] = a2_26.x;
                    acc[21] = a2_26.y;
                    float2 a2_27 = make_float2(acc[22], acc[23]);
                    a2_27 = fma_f32x2_rn_ftz(m2_13, make_float2(xv[2], xv[3]), a2_27);
                    acc[22] = a2_27.x;
                    acc[23] = a2_27.y;
                    float2 m2_14 = make_float2(mv[24], mv[24]);
                    float2 a2_28 = make_float2(acc[24], acc[25]);
                    a2_28 = fma_f32x2_rn_ftz(m2_14, make_float2(xv[0], xv[1]), a2_28);
                    acc[24] = a2_28.x;
                    acc[25] = a2_28.y;
                    float2 a2_29 = make_float2(acc[26], acc[27]);
                    a2_29 = fma_f32x2_rn_ftz(m2_14, make_float2(xv[2], xv[3]), a2_29);
                    acc[26] = a2_29.x;
                    acc[27] = a2_29.y;
                    float2 m2_15 = make_float2(mv[28], mv[28]);
                    float2 a2_30 = make_float2(acc[28], acc[29]);
                    a2_30 = fma_f32x2_rn_ftz(m2_15, make_float2(xv[0], xv[1]), a2_30);
                    acc[28] = a2_30.x;
                    acc[29] = a2_30.y;
                    float2 a2_31 = make_float2(acc[30], acc[31]);
                    a2_31 = fma_f32x2_rn_ftz(m2_15, make_float2(xv[2], xv[3]), a2_31);
                    acc[30] = a2_31.x;
                    acc[31] = a2_31.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 4608) + (unsigned int)(q * 576) + 144));
#if __CUDA_ARCH__ == 1070
                    acc[0] = __fmaf_rn(mv[1], xv[0], acc[0]);
                    acc[1] = __fmaf_rn(mv[1], xv[1], acc[1]);
                    acc[2] = __fmaf_rn(mv[1], xv[2], acc[2]);
                    acc[3] = __fmaf_rn(mv[1], xv[3], acc[3]);
                    acc[4] = __fmaf_rn(mv[5], xv[0], acc[4]);
                    acc[5] = __fmaf_rn(mv[5], xv[1], acc[5]);
                    acc[6] = __fmaf_rn(mv[5], xv[2], acc[6]);
                    acc[7] = __fmaf_rn(mv[5], xv[3], acc[7]);
                    acc[8] = __fmaf_rn(mv[9], xv[0], acc[8]);
                    acc[9] = __fmaf_rn(mv[9], xv[1], acc[9]);
                    acc[10] = __fmaf_rn(mv[9], xv[2], acc[10]);
                    acc[11] = __fmaf_rn(mv[9], xv[3], acc[11]);
                    acc[12] = __fmaf_rn(mv[13], xv[0], acc[12]);
                    acc[13] = __fmaf_rn(mv[13], xv[1], acc[13]);
                    acc[14] = __fmaf_rn(mv[13], xv[2], acc[14]);
                    acc[15] = __fmaf_rn(mv[13], xv[3], acc[15]);
                    acc[16] = __fmaf_rn(mv[17], xv[0], acc[16]);
                    acc[17] = __fmaf_rn(mv[17], xv[1], acc[17]);
                    acc[18] = __fmaf_rn(mv[17], xv[2], acc[18]);
                    acc[19] = __fmaf_rn(mv[17], xv[3], acc[19]);
                    acc[20] = __fmaf_rn(mv[21], xv[0], acc[20]);
                    acc[21] = __fmaf_rn(mv[21], xv[1], acc[21]);
                    acc[22] = __fmaf_rn(mv[21], xv[2], acc[22]);
                    acc[23] = __fmaf_rn(mv[21], xv[3], acc[23]);
                    acc[24] = __fmaf_rn(mv[25], xv[0], acc[24]);
                    acc[25] = __fmaf_rn(mv[25], xv[1], acc[25]);
                    acc[26] = __fmaf_rn(mv[25], xv[2], acc[26]);
                    acc[27] = __fmaf_rn(mv[25], xv[3], acc[27]);
                    acc[28] = __fmaf_rn(mv[29], xv[0], acc[28]);
                    acc[29] = __fmaf_rn(mv[29], xv[1], acc[29]);
                    acc[30] = __fmaf_rn(mv[29], xv[2], acc[30]);
                    acc[31] = __fmaf_rn(mv[29], xv[3], acc[31]);
#else
                    float2 m2_16 = make_float2(mv[1], mv[1]);
                    float2 a2_32 = make_float2(acc[0], acc[1]);
                    a2_32 = fma_f32x2_rn_ftz(m2_16, make_float2(xv[0], xv[1]), a2_32);
                    acc[0] = a2_32.x;
                    acc[1] = a2_32.y;
                    float2 a2_33 = make_float2(acc[2], acc[3]);
                    a2_33 = fma_f32x2_rn_ftz(m2_16, make_float2(xv[2], xv[3]), a2_33);
                    acc[2] = a2_33.x;
                    acc[3] = a2_33.y;
                    float2 m2_17 = make_float2(mv[5], mv[5]);
                    float2 a2_34 = make_float2(acc[4], acc[5]);
                    a2_34 = fma_f32x2_rn_ftz(m2_17, make_float2(xv[0], xv[1]), a2_34);
                    acc[4] = a2_34.x;
                    acc[5] = a2_34.y;
                    float2 a2_35 = make_float2(acc[6], acc[7]);
                    a2_35 = fma_f32x2_rn_ftz(m2_17, make_float2(xv[2], xv[3]), a2_35);
                    acc[6] = a2_35.x;
                    acc[7] = a2_35.y;
                    float2 m2_18 = make_float2(mv[9], mv[9]);
                    float2 a2_36 = make_float2(acc[8], acc[9]);
                    a2_36 = fma_f32x2_rn_ftz(m2_18, make_float2(xv[0], xv[1]), a2_36);
                    acc[8] = a2_36.x;
                    acc[9] = a2_36.y;
                    float2 a2_37 = make_float2(acc[10], acc[11]);
                    a2_37 = fma_f32x2_rn_ftz(m2_18, make_float2(xv[2], xv[3]), a2_37);
                    acc[10] = a2_37.x;
                    acc[11] = a2_37.y;
                    float2 m2_19 = make_float2(mv[13], mv[13]);
                    float2 a2_38 = make_float2(acc[12], acc[13]);
                    a2_38 = fma_f32x2_rn_ftz(m2_19, make_float2(xv[0], xv[1]), a2_38);
                    acc[12] = a2_38.x;
                    acc[13] = a2_38.y;
                    float2 a2_39 = make_float2(acc[14], acc[15]);
                    a2_39 = fma_f32x2_rn_ftz(m2_19, make_float2(xv[2], xv[3]), a2_39);
                    acc[14] = a2_39.x;
                    acc[15] = a2_39.y;
                    float2 m2_20 = make_float2(mv[17], mv[17]);
                    float2 a2_40 = make_float2(acc[16], acc[17]);
                    a2_40 = fma_f32x2_rn_ftz(m2_20, make_float2(xv[0], xv[1]), a2_40);
                    acc[16] = a2_40.x;
                    acc[17] = a2_40.y;
                    float2 a2_41 = make_float2(acc[18], acc[19]);
                    a2_41 = fma_f32x2_rn_ftz(m2_20, make_float2(xv[2], xv[3]), a2_41);
                    acc[18] = a2_41.x;
                    acc[19] = a2_41.y;
                    float2 m2_21 = make_float2(mv[21], mv[21]);
                    float2 a2_42 = make_float2(acc[20], acc[21]);
                    a2_42 = fma_f32x2_rn_ftz(m2_21, make_float2(xv[0], xv[1]), a2_42);
                    acc[20] = a2_42.x;
                    acc[21] = a2_42.y;
                    float2 a2_43 = make_float2(acc[22], acc[23]);
                    a2_43 = fma_f32x2_rn_ftz(m2_21, make_float2(xv[2], xv[3]), a2_43);
                    acc[22] = a2_43.x;
                    acc[23] = a2_43.y;
                    float2 m2_22 = make_float2(mv[25], mv[25]);
                    float2 a2_44 = make_float2(acc[24], acc[25]);
                    a2_44 = fma_f32x2_rn_ftz(m2_22, make_float2(xv[0], xv[1]), a2_44);
                    acc[24] = a2_44.x;
                    acc[25] = a2_44.y;
                    float2 a2_45 = make_float2(acc[26], acc[27]);
                    a2_45 = fma_f32x2_rn_ftz(m2_22, make_float2(xv[2], xv[3]), a2_45);
                    acc[26] = a2_45.x;
                    acc[27] = a2_45.y;
                    float2 m2_23 = make_float2(mv[29], mv[29]);
                    float2 a2_46 = make_float2(acc[28], acc[29]);
                    a2_46 = fma_f32x2_rn_ftz(m2_23, make_float2(xv[0], xv[1]), a2_46);
                    acc[28] = a2_46.x;
                    acc[29] = a2_46.y;
                    float2 a2_47 = make_float2(acc[30], acc[31]);
                    a2_47 = fma_f32x2_rn_ftz(m2_23, make_float2(xv[2], xv[3]), a2_47);
                    acc[30] = a2_47.x;
                    acc[31] = a2_47.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 4608) + (unsigned int)(q * 576) + 288));
#if __CUDA_ARCH__ == 1070
                    acc[0] = __fmaf_rn(mv[2], xv[0], acc[0]);
                    acc[1] = __fmaf_rn(mv[2], xv[1], acc[1]);
                    acc[2] = __fmaf_rn(mv[2], xv[2], acc[2]);
                    acc[3] = __fmaf_rn(mv[2], xv[3], acc[3]);
                    acc[4] = __fmaf_rn(mv[6], xv[0], acc[4]);
                    acc[5] = __fmaf_rn(mv[6], xv[1], acc[5]);
                    acc[6] = __fmaf_rn(mv[6], xv[2], acc[6]);
                    acc[7] = __fmaf_rn(mv[6], xv[3], acc[7]);
                    acc[8] = __fmaf_rn(mv[10], xv[0], acc[8]);
                    acc[9] = __fmaf_rn(mv[10], xv[1], acc[9]);
                    acc[10] = __fmaf_rn(mv[10], xv[2], acc[10]);
                    acc[11] = __fmaf_rn(mv[10], xv[3], acc[11]);
                    acc[12] = __fmaf_rn(mv[14], xv[0], acc[12]);
                    acc[13] = __fmaf_rn(mv[14], xv[1], acc[13]);
                    acc[14] = __fmaf_rn(mv[14], xv[2], acc[14]);
                    acc[15] = __fmaf_rn(mv[14], xv[3], acc[15]);
                    acc[16] = __fmaf_rn(mv[18], xv[0], acc[16]);
                    acc[17] = __fmaf_rn(mv[18], xv[1], acc[17]);
                    acc[18] = __fmaf_rn(mv[18], xv[2], acc[18]);
                    acc[19] = __fmaf_rn(mv[18], xv[3], acc[19]);
                    acc[20] = __fmaf_rn(mv[22], xv[0], acc[20]);
                    acc[21] = __fmaf_rn(mv[22], xv[1], acc[21]);
                    acc[22] = __fmaf_rn(mv[22], xv[2], acc[22]);
                    acc[23] = __fmaf_rn(mv[22], xv[3], acc[23]);
                    acc[24] = __fmaf_rn(mv[26], xv[0], acc[24]);
                    acc[25] = __fmaf_rn(mv[26], xv[1], acc[25]);
                    acc[26] = __fmaf_rn(mv[26], xv[2], acc[26]);
                    acc[27] = __fmaf_rn(mv[26], xv[3], acc[27]);
                    acc[28] = __fmaf_rn(mv[30], xv[0], acc[28]);
                    acc[29] = __fmaf_rn(mv[30], xv[1], acc[29]);
                    acc[30] = __fmaf_rn(mv[30], xv[2], acc[30]);
                    acc[31] = __fmaf_rn(mv[30], xv[3], acc[31]);
#else
                    float2 m2_24 = make_float2(mv[2], mv[2]);
                    float2 a2_48 = make_float2(acc[0], acc[1]);
                    a2_48 = fma_f32x2_rn_ftz(m2_24, make_float2(xv[0], xv[1]), a2_48);
                    acc[0] = a2_48.x;
                    acc[1] = a2_48.y;
                    float2 a2_49 = make_float2(acc[2], acc[3]);
                    a2_49 = fma_f32x2_rn_ftz(m2_24, make_float2(xv[2], xv[3]), a2_49);
                    acc[2] = a2_49.x;
                    acc[3] = a2_49.y;
                    float2 m2_25 = make_float2(mv[6], mv[6]);
                    float2 a2_50 = make_float2(acc[4], acc[5]);
                    a2_50 = fma_f32x2_rn_ftz(m2_25, make_float2(xv[0], xv[1]), a2_50);
                    acc[4] = a2_50.x;
                    acc[5] = a2_50.y;
                    float2 a2_51 = make_float2(acc[6], acc[7]);
                    a2_51 = fma_f32x2_rn_ftz(m2_25, make_float2(xv[2], xv[3]), a2_51);
                    acc[6] = a2_51.x;
                    acc[7] = a2_51.y;
                    float2 m2_26 = make_float2(mv[10], mv[10]);
                    float2 a2_52 = make_float2(acc[8], acc[9]);
                    a2_52 = fma_f32x2_rn_ftz(m2_26, make_float2(xv[0], xv[1]), a2_52);
                    acc[8] = a2_52.x;
                    acc[9] = a2_52.y;
                    float2 a2_53 = make_float2(acc[10], acc[11]);
                    a2_53 = fma_f32x2_rn_ftz(m2_26, make_float2(xv[2], xv[3]), a2_53);
                    acc[10] = a2_53.x;
                    acc[11] = a2_53.y;
                    float2 m2_27 = make_float2(mv[14], mv[14]);
                    float2 a2_54 = make_float2(acc[12], acc[13]);
                    a2_54 = fma_f32x2_rn_ftz(m2_27, make_float2(xv[0], xv[1]), a2_54);
                    acc[12] = a2_54.x;
                    acc[13] = a2_54.y;
                    float2 a2_55 = make_float2(acc[14], acc[15]);
                    a2_55 = fma_f32x2_rn_ftz(m2_27, make_float2(xv[2], xv[3]), a2_55);
                    acc[14] = a2_55.x;
                    acc[15] = a2_55.y;
                    float2 m2_28 = make_float2(mv[18], mv[18]);
                    float2 a2_56 = make_float2(acc[16], acc[17]);
                    a2_56 = fma_f32x2_rn_ftz(m2_28, make_float2(xv[0], xv[1]), a2_56);
                    acc[16] = a2_56.x;
                    acc[17] = a2_56.y;
                    float2 a2_57 = make_float2(acc[18], acc[19]);
                    a2_57 = fma_f32x2_rn_ftz(m2_28, make_float2(xv[2], xv[3]), a2_57);
                    acc[18] = a2_57.x;
                    acc[19] = a2_57.y;
                    float2 m2_29 = make_float2(mv[22], mv[22]);
                    float2 a2_58 = make_float2(acc[20], acc[21]);
                    a2_58 = fma_f32x2_rn_ftz(m2_29, make_float2(xv[0], xv[1]), a2_58);
                    acc[20] = a2_58.x;
                    acc[21] = a2_58.y;
                    float2 a2_59 = make_float2(acc[22], acc[23]);
                    a2_59 = fma_f32x2_rn_ftz(m2_29, make_float2(xv[2], xv[3]), a2_59);
                    acc[22] = a2_59.x;
                    acc[23] = a2_59.y;
                    float2 m2_30 = make_float2(mv[26], mv[26]);
                    float2 a2_60 = make_float2(acc[24], acc[25]);
                    a2_60 = fma_f32x2_rn_ftz(m2_30, make_float2(xv[0], xv[1]), a2_60);
                    acc[24] = a2_60.x;
                    acc[25] = a2_60.y;
                    float2 a2_61 = make_float2(acc[26], acc[27]);
                    a2_61 = fma_f32x2_rn_ftz(m2_30, make_float2(xv[2], xv[3]), a2_61);
                    acc[26] = a2_61.x;
                    acc[27] = a2_61.y;
                    float2 m2_31 = make_float2(mv[30], mv[30]);
                    float2 a2_62 = make_float2(acc[28], acc[29]);
                    a2_62 = fma_f32x2_rn_ftz(m2_31, make_float2(xv[0], xv[1]), a2_62);
                    acc[28] = a2_62.x;
                    acc[29] = a2_62.y;
                    float2 a2_63 = make_float2(acc[30], acc[31]);
                    a2_63 = fma_f32x2_rn_ftz(m2_31, make_float2(xv[2], xv[3]), a2_63);
                    acc[30] = a2_63.x;
                    acc[31] = a2_63.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 4608) + (unsigned int)(q * 576) + 432));
#if __CUDA_ARCH__ == 1070
                    acc[0] = __fmaf_rn(mv[3], xv[0], acc[0]);
                    acc[1] = __fmaf_rn(mv[3], xv[1], acc[1]);
                    acc[2] = __fmaf_rn(mv[3], xv[2], acc[2]);
                    acc[3] = __fmaf_rn(mv[3], xv[3], acc[3]);
                    acc[4] = __fmaf_rn(mv[7], xv[0], acc[4]);
                    acc[5] = __fmaf_rn(mv[7], xv[1], acc[5]);
                    acc[6] = __fmaf_rn(mv[7], xv[2], acc[6]);
                    acc[7] = __fmaf_rn(mv[7], xv[3], acc[7]);
                    acc[8] = __fmaf_rn(mv[11], xv[0], acc[8]);
                    acc[9] = __fmaf_rn(mv[11], xv[1], acc[9]);
                    acc[10] = __fmaf_rn(mv[11], xv[2], acc[10]);
                    acc[11] = __fmaf_rn(mv[11], xv[3], acc[11]);
                    acc[12] = __fmaf_rn(mv[15], xv[0], acc[12]);
                    acc[13] = __fmaf_rn(mv[15], xv[1], acc[13]);
                    acc[14] = __fmaf_rn(mv[15], xv[2], acc[14]);
                    acc[15] = __fmaf_rn(mv[15], xv[3], acc[15]);
                    acc[16] = __fmaf_rn(mv[19], xv[0], acc[16]);
                    acc[17] = __fmaf_rn(mv[19], xv[1], acc[17]);
                    acc[18] = __fmaf_rn(mv[19], xv[2], acc[18]);
                    acc[19] = __fmaf_rn(mv[19], xv[3], acc[19]);
                    acc[20] = __fmaf_rn(mv[23], xv[0], acc[20]);
                    acc[21] = __fmaf_rn(mv[23], xv[1], acc[21]);
                    acc[22] = __fmaf_rn(mv[23], xv[2], acc[22]);
                    acc[23] = __fmaf_rn(mv[23], xv[3], acc[23]);
                    acc[24] = __fmaf_rn(mv[27], xv[0], acc[24]);
                    acc[25] = __fmaf_rn(mv[27], xv[1], acc[25]);
                    acc[26] = __fmaf_rn(mv[27], xv[2], acc[26]);
                    acc[27] = __fmaf_rn(mv[27], xv[3], acc[27]);
                    acc[28] = __fmaf_rn(mv[31], xv[0], acc[28]);
                    acc[29] = __fmaf_rn(mv[31], xv[1], acc[29]);
                    acc[30] = __fmaf_rn(mv[31], xv[2], acc[30]);
                    acc[31] = __fmaf_rn(mv[31], xv[3], acc[31]);
#else
                    float2 m2_32 = make_float2(mv[3], mv[3]);
                    float2 a2_64 = make_float2(acc[0], acc[1]);
                    a2_64 = fma_f32x2_rn_ftz(m2_32, make_float2(xv[0], xv[1]), a2_64);
                    acc[0] = a2_64.x;
                    acc[1] = a2_64.y;
                    float2 a2_65 = make_float2(acc[2], acc[3]);
                    a2_65 = fma_f32x2_rn_ftz(m2_32, make_float2(xv[2], xv[3]), a2_65);
                    acc[2] = a2_65.x;
                    acc[3] = a2_65.y;
                    float2 m2_33 = make_float2(mv[7], mv[7]);
                    float2 a2_66 = make_float2(acc[4], acc[5]);
                    a2_66 = fma_f32x2_rn_ftz(m2_33, make_float2(xv[0], xv[1]), a2_66);
                    acc[4] = a2_66.x;
                    acc[5] = a2_66.y;
                    float2 a2_67 = make_float2(acc[6], acc[7]);
                    a2_67 = fma_f32x2_rn_ftz(m2_33, make_float2(xv[2], xv[3]), a2_67);
                    acc[6] = a2_67.x;
                    acc[7] = a2_67.y;
                    float2 m2_34 = make_float2(mv[11], mv[11]);
                    float2 a2_68 = make_float2(acc[8], acc[9]);
                    a2_68 = fma_f32x2_rn_ftz(m2_34, make_float2(xv[0], xv[1]), a2_68);
                    acc[8] = a2_68.x;
                    acc[9] = a2_68.y;
                    float2 a2_69 = make_float2(acc[10], acc[11]);
                    a2_69 = fma_f32x2_rn_ftz(m2_34, make_float2(xv[2], xv[3]), a2_69);
                    acc[10] = a2_69.x;
                    acc[11] = a2_69.y;
                    float2 m2_35 = make_float2(mv[15], mv[15]);
                    float2 a2_70 = make_float2(acc[12], acc[13]);
                    a2_70 = fma_f32x2_rn_ftz(m2_35, make_float2(xv[0], xv[1]), a2_70);
                    acc[12] = a2_70.x;
                    acc[13] = a2_70.y;
                    float2 a2_71 = make_float2(acc[14], acc[15]);
                    a2_71 = fma_f32x2_rn_ftz(m2_35, make_float2(xv[2], xv[3]), a2_71);
                    acc[14] = a2_71.x;
                    acc[15] = a2_71.y;
                    float2 m2_36 = make_float2(mv[19], mv[19]);
                    float2 a2_72 = make_float2(acc[16], acc[17]);
                    a2_72 = fma_f32x2_rn_ftz(m2_36, make_float2(xv[0], xv[1]), a2_72);
                    acc[16] = a2_72.x;
                    acc[17] = a2_72.y;
                    float2 a2_73 = make_float2(acc[18], acc[19]);
                    a2_73 = fma_f32x2_rn_ftz(m2_36, make_float2(xv[2], xv[3]), a2_73);
                    acc[18] = a2_73.x;
                    acc[19] = a2_73.y;
                    float2 m2_37 = make_float2(mv[23], mv[23]);
                    float2 a2_74 = make_float2(acc[20], acc[21]);
                    a2_74 = fma_f32x2_rn_ftz(m2_37, make_float2(xv[0], xv[1]), a2_74);
                    acc[20] = a2_74.x;
                    acc[21] = a2_74.y;
                    float2 a2_75 = make_float2(acc[22], acc[23]);
                    a2_75 = fma_f32x2_rn_ftz(m2_37, make_float2(xv[2], xv[3]), a2_75);
                    acc[22] = a2_75.x;
                    acc[23] = a2_75.y;
                    float2 m2_38 = make_float2(mv[27], mv[27]);
                    float2 a2_76 = make_float2(acc[24], acc[25]);
                    a2_76 = fma_f32x2_rn_ftz(m2_38, make_float2(xv[0], xv[1]), a2_76);
                    acc[24] = a2_76.x;
                    acc[25] = a2_76.y;
                    float2 a2_77 = make_float2(acc[26], acc[27]);
                    a2_77 = fma_f32x2_rn_ftz(m2_38, make_float2(xv[2], xv[3]), a2_77);
                    acc[26] = a2_77.x;
                    acc[27] = a2_77.y;
                    float2 m2_39 = make_float2(mv[31], mv[31]);
                    float2 a2_78 = make_float2(acc[28], acc[29]);
                    a2_78 = fma_f32x2_rn_ftz(m2_39, make_float2(xv[0], xv[1]), a2_78);
                    acc[28] = a2_78.x;
                    acc[29] = a2_78.y;
                    float2 a2_79 = make_float2(acc[30], acc[31]);
                    a2_79 = fma_f32x2_rn_ftz(m2_39, make_float2(xv[2], xv[3]), a2_79);
                    acc[30] = a2_79.x;
                    acc[31] = a2_79.y;
#endif
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432)), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 576), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 1152), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 1728), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 2304), "r"(__float_as_uint(acc[16])), "r"(__float_as_uint(acc[17])), "r"(__float_as_uint(acc[18])), "r"(__float_as_uint(acc[19])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 2880), "r"(__float_as_uint(acc[20])), "r"(__float_as_uint(acc[21])), "r"(__float_as_uint(acc[22])), "r"(__float_as_uint(acc[23])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 3456), "r"(__float_as_uint(acc[24])), "r"(__float_as_uint(acc[25])), "r"(__float_as_uint(acc[26])), "r"(__float_as_uint(acc[27])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (18432 - buf_1 * 18432) + 4032), "r"(__float_as_uint(acc[28])), "r"(__float_as_uint(acc[29])), "r"(__float_as_uint(acc[30])), "r"(__float_as_uint(acc[31])) : "memory");
            if (y_pub_1 >= 0ll) {
                {
                    float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 512ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 1024ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 1536ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[16 + 0], acc[16 + 1], acc[16 + 2], acc[16 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[20 + 0], acc[20 + 1], acc[20 + 2], acc[20 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 2560ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[24 + 0], acc[24 + 1], acc[24 + 2], acc[24 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 3072ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[28 + 0], acc[28 + 1], acc[28 + 2], acc[28 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 3584ll) = _v4;
                }
            }
            __syncthreads();
            buf_1 = 1 - buf_1;
            m_cur_1 = m_next_1;
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
}

} // extern "C"
