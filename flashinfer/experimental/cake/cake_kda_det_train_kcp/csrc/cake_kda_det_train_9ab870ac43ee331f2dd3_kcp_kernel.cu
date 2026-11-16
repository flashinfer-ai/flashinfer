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

#define LAUNCH_MIN_BLOCKS 3

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
kernel_cake_kda_det_train_9ab870ac43ee331f2dd3_kcp(float* __restrict__ m, float* __restrict__ c, float* __restrict__ x0, float* __restrict__ y, int* __restrict__ chain_start, long long* __restrict__ elem_m, long long* __restrict__ elem_c, long long* __restrict__ elem_y, long long* __restrict__ x0_index, long long* __restrict__ y_final, int ncols, int transpose_m, int write_states, int cols_per_cta)
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
    float* ms = reinterpret_cast<float*>(smem_raw + 20480);
    const int ms_addr = smem + 20480;

    // === Task calls (declaration order) ===
    int nblk = ncols / 16;
    int chain = blockIdx.x / nblk;
    int col0 = blockIdx.x % nblk * 16;
    int col_tile = col0 / 128;
    int v0 = col0 % 128;
    int warp_1 = warp;
    int g = lane / 4;
    int cg = lane % 4;
    int e_begin = chain_start[chain];
    int e_end = chain_start[chain + 1];
    unsigned int ring_w = ms_addr + (unsigned int)(warp_1 * 10240);
    float acc[16];
    float cn[16];
    float mv[16];
    float xv[4];
    if (transpose_m == 0) {
        int row0 = warp_1 * 32 + g * 4;
        long long g_off = (long long)col_tile * 16384ll + (long long)(row0 * 128 + v0 + cg * 4);
        unsigned int x_thr = xs_addr + (unsigned int)((row0 * 20 + cg * 4) * 4);
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
            asm volatile("cp.async.commit_group;");
            long long src0_0 = m_cur + (long long)(cp_src + 2048);
            unsigned int dst0_0 = cp_dst + 2560;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0), "l"(m + src0_0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 512), "l"(m + (src0_0 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 1024), "l"(m + (src0_0 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_0 + 1536), "l"(m + (src0_0 + 1536ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_1 = m_cur + (long long)(cp_src + 4096);
            unsigned int dst0_1 = cp_dst + 5120;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1), "l"(m + src0_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 512), "l"(m + (src0_1 + 512ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 1024), "l"(m + (src0_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_1 + 1536), "l"(m + (src0_1 + 1536ll)));
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
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 80), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 160), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + 240), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
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
            unsigned int x_rd_e = x_rd + buf * 10240;
            #pragma unroll 1
            for (int ch = 0; ch < 8; ch++) {
                asm volatile("cp.async.wait_group 2;");
                __syncwarp();
                int nc = ch + 3;
                if (nc < 8) {
                    long long src0_2 = m_cur + (long long)(cp_src + nc * 2048);
                    unsigned int dst0_2 = cp_dst + (unsigned int)(nc % 4 * 2560);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2), "l"(m + src0_2));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 512), "l"(m + (src0_2 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 1024), "l"(m + (src0_2 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_2 + 1536), "l"(m + (src0_2 + 1536ll)));
                } else if (has_next != 0) {
                    long long src0_3 = m_next + (long long)(cp_src + (nc - 8) * 2048);
                    unsigned int dst0_3 = cp_dst + (unsigned int)(nc % 4 * 2560);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3), "l"(m + src0_3));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 512), "l"(m + (src0_3 + 512ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 1024), "l"(m + (src0_3 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_3 + 1536), "l"(m + (src0_3 + 1536ll)));
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
                    }
                }
                #pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[0])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 3]))
                        : "r"(m_rd + (unsigned int)(ch % 4 * 2560) + (unsigned int)(kk * 128)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e + (unsigned int)(ch * 1280) + (unsigned int)(kk * 80)));
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
#endif
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (10240 - buf * 10240)), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (10240 - buf * 10240) + 80), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (10240 - buf * 10240) + 160), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr + (10240 - buf * 10240) + 240), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
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
        unsigned int x_thr_1 = xs_addr + (unsigned int)((row0_1 * 20 + cg * 4) * 4);
        unsigned int x_rd_1 = xs_addr + (unsigned int)(cg * 16);
        unsigned int m_rd_1 = ring_w + (unsigned int)(g * 80);
        int cp_i_1 = lane / 4;
        int cp_j_1 = lane % 4;
        int cp_src_1 = cp_i_1 * 128 + warp_1 * 4096 + 4 * cp_j_1;
        unsigned int cp_dst_1 = ring_w + (unsigned int)((cp_i_1 * 20 + 4 * cp_j_1) * 4);
        long long m_cur_1 = 0;
        if (e_begin < e_end) {
            m_cur_1 = elem_m[e_begin];
            long long src0_4 = m_cur_1 + (long long)cp_src_1;
            unsigned int dst0_4 = cp_dst_1 + 0;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4), "l"(m + src0_4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 640), "l"(m + (src0_4 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 1280), "l"(m + (src0_4 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_4 + 1920), "l"(m + (src0_4 + 3072ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_2_1 = m_cur_1 + (long long)(cp_src_1 + 16);
            unsigned int dst0_2_1 = cp_dst_1 + 2560;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1), "l"(m + src0_2_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 640), "l"(m + (src0_2_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 1280), "l"(m + (src0_2_1 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_2_1 + 1920), "l"(m + (src0_2_1 + 3072ll)));
            asm volatile("cp.async.commit_group;");
            long long src0_3_1 = m_cur_1 + (long long)(cp_src_1 + 32);
            unsigned int dst0_3_1 = cp_dst_1 + 5120;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1), "l"(m + src0_3_1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 640), "l"(m + (src0_3_1 + 1024ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 1280), "l"(m + (src0_3_1 + 2048ll)));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(dst0_3_1 + 1920), "l"(m + (src0_3_1 + 3072ll)));
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
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 1024ll);
                    cn[4 + 0] = _v4.x;
                    cn[4 + 1] = _v4.y;
                    cn[4 + 2] = _v4.z;
                    cn[4 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 2048ll);
                    cn[8 + 0] = _v4.x;
                    cn[8 + 1] = _v4.y;
                    cn[8 + 2] = _v4.z;
                    cn[8 + 3] = _v4.w;
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(c + c_first_1 + g_off_1 + 3072ll);
                    cn[12 + 0] = _v4.x;
                    cn[12 + 1] = _v4.y;
                    cn[12 + 2] = _v4.z;
                    cn[12 + 3] = _v4.w;
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
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 1024ll);
                acc[4 + 0] = _v4.x;
                acc[4 + 1] = _v4.y;
                acc[4 + 2] = _v4.z;
                acc[4 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 2048ll);
                acc[8 + 0] = _v4.x;
                acc[8 + 1] = _v4.y;
                acc[8 + 2] = _v4.z;
                acc[8 + 3] = _v4.w;
            }
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x0 + x0_off_1 + g_off_1 + 3072ll);
                acc[12 + 0] = _v4.x;
                acc[12 + 1] = _v4.y;
                acc[12 + 2] = _v4.z;
                acc[12 + 3] = _v4.w;
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
            acc[4] = ((diag_1 + 8 == 0) ? 1.0f : 0.0f);
            acc[5] = ((diag_1 + 8 == 1) ? 1.0f : 0.0f);
            acc[6] = ((diag_1 + 8 == 2) ? 1.0f : 0.0f);
            acc[7] = ((diag_1 + 8 == 3) ? 1.0f : 0.0f);
            acc[8] = ((diag_1 + 16 == 0) ? 1.0f : 0.0f);
            acc[9] = ((diag_1 + 16 == 1) ? 1.0f : 0.0f);
            acc[10] = ((diag_1 + 16 == 2) ? 1.0f : 0.0f);
            acc[11] = ((diag_1 + 16 == 3) ? 1.0f : 0.0f);
            acc[12] = ((diag_1 + 24 == 0) ? 1.0f : 0.0f);
            acc[13] = ((diag_1 + 24 == 1) ? 1.0f : 0.0f);
            acc[14] = ((diag_1 + 24 == 2) ? 1.0f : 0.0f);
            acc[15] = ((diag_1 + 24 == 3) ? 1.0f : 0.0f);
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 640), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 1280), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + 1920), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
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
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 1024ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 2048ll) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                        *reinterpret_cast<float4*>(y + y_first_1 + g_off_1 + 3072ll) = _v4;
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
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 1024ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_last_1 + g_off_1 + 3072ll) = _v4;
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
            unsigned int x_rd_e_1 = x_rd_1 + buf_1 * 10240;
            #pragma unroll 1
            for (int ch_1 = 0; ch_1 < 8; ch_1++) {
                asm volatile("cp.async.wait_group 2;");
                __syncwarp();
                int nc_1 = ch_1 + 3;
                if (nc_1 < 8) {
                    long long src0_5 = m_cur_1 + (long long)(cp_src_1 + nc_1 * 16);
                    unsigned int dst0_5 = cp_dst_1 + (unsigned int)(nc_1 % 4 * 2560);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5), "l"(m + src0_5));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 640), "l"(m + (src0_5 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 1280), "l"(m + (src0_5 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_5 + 1920), "l"(m + (src0_5 + 3072ll)));
                } else if (has_next_1 != 0) {
                    long long src0_6 = m_next_1 + (long long)(cp_src_1 + (nc_1 - 8) * 16);
                    unsigned int dst0_6 = cp_dst_1 + (unsigned int)(nc_1 % 4 * 2560);
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6), "l"(m + src0_6));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 640), "l"(m + (src0_6 + 1024ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 1280), "l"(m + (src0_6 + 2048ll)));
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(dst0_6 + 1920), "l"(m + (src0_6 + 3072ll)));
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
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 1024ll);
                            cn[4 + 0] = _v4.x;
                            cn[4 + 1] = _v4.y;
                            cn[4 + 2] = _v4.z;
                            cn[4 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 2048ll);
                            cn[8 + 0] = _v4.x;
                            cn[8 + 1] = _v4.y;
                            cn[8 + 2] = _v4.z;
                            cn[8 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c + c_next_1 + g_off_1 + 3072ll);
                            cn[12 + 0] = _v4.x;
                            cn[12 + 1] = _v4.y;
                            cn[12 + 2] = _v4.z;
                            cn[12 + 3] = _v4.w;
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
                    }
                }
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[0])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(0) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 2560) + (unsigned int)(q * 16)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[4])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(4) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 2560) + (unsigned int)(q * 16) + 640));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[8])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(8) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 2560) + (unsigned int)(q * 16) + 1280));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mv[12])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mv[(12) + 3]))
                        : "r"(m_rd_1 + (unsigned int)(ch_1 % 4 * 2560) + (unsigned int)(q * 16) + 1920));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 1280) + (unsigned int)(q * 320)));
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
#else
                    float2 m2_4 = make_float2(mv[0], mv[0]);
                    float2 a2_8 = make_float2(acc[0], acc[1]);
                    a2_8 = fma_f32x2_rn_ftz(m2_4, make_float2(xv[0], xv[1]), a2_8);
                    acc[0] = a2_8.x;
                    acc[1] = a2_8.y;
                    float2 a2_9 = make_float2(acc[2], acc[3]);
                    a2_9 = fma_f32x2_rn_ftz(m2_4, make_float2(xv[2], xv[3]), a2_9);
                    acc[2] = a2_9.x;
                    acc[3] = a2_9.y;
                    float2 m2_5 = make_float2(mv[4], mv[4]);
                    float2 a2_10 = make_float2(acc[4], acc[5]);
                    a2_10 = fma_f32x2_rn_ftz(m2_5, make_float2(xv[0], xv[1]), a2_10);
                    acc[4] = a2_10.x;
                    acc[5] = a2_10.y;
                    float2 a2_11 = make_float2(acc[6], acc[7]);
                    a2_11 = fma_f32x2_rn_ftz(m2_5, make_float2(xv[2], xv[3]), a2_11);
                    acc[6] = a2_11.x;
                    acc[7] = a2_11.y;
                    float2 m2_6 = make_float2(mv[8], mv[8]);
                    float2 a2_12 = make_float2(acc[8], acc[9]);
                    a2_12 = fma_f32x2_rn_ftz(m2_6, make_float2(xv[0], xv[1]), a2_12);
                    acc[8] = a2_12.x;
                    acc[9] = a2_12.y;
                    float2 a2_13 = make_float2(acc[10], acc[11]);
                    a2_13 = fma_f32x2_rn_ftz(m2_6, make_float2(xv[2], xv[3]), a2_13);
                    acc[10] = a2_13.x;
                    acc[11] = a2_13.y;
                    float2 m2_7 = make_float2(mv[12], mv[12]);
                    float2 a2_14 = make_float2(acc[12], acc[13]);
                    a2_14 = fma_f32x2_rn_ftz(m2_7, make_float2(xv[0], xv[1]), a2_14);
                    acc[12] = a2_14.x;
                    acc[13] = a2_14.y;
                    float2 a2_15 = make_float2(acc[14], acc[15]);
                    a2_15 = fma_f32x2_rn_ftz(m2_7, make_float2(xv[2], xv[3]), a2_15);
                    acc[14] = a2_15.x;
                    acc[15] = a2_15.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 1280) + (unsigned int)(q * 320) + 80));
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
#else
                    float2 m2_8 = make_float2(mv[1], mv[1]);
                    float2 a2_16 = make_float2(acc[0], acc[1]);
                    a2_16 = fma_f32x2_rn_ftz(m2_8, make_float2(xv[0], xv[1]), a2_16);
                    acc[0] = a2_16.x;
                    acc[1] = a2_16.y;
                    float2 a2_17 = make_float2(acc[2], acc[3]);
                    a2_17 = fma_f32x2_rn_ftz(m2_8, make_float2(xv[2], xv[3]), a2_17);
                    acc[2] = a2_17.x;
                    acc[3] = a2_17.y;
                    float2 m2_9 = make_float2(mv[5], mv[5]);
                    float2 a2_18 = make_float2(acc[4], acc[5]);
                    a2_18 = fma_f32x2_rn_ftz(m2_9, make_float2(xv[0], xv[1]), a2_18);
                    acc[4] = a2_18.x;
                    acc[5] = a2_18.y;
                    float2 a2_19 = make_float2(acc[6], acc[7]);
                    a2_19 = fma_f32x2_rn_ftz(m2_9, make_float2(xv[2], xv[3]), a2_19);
                    acc[6] = a2_19.x;
                    acc[7] = a2_19.y;
                    float2 m2_10 = make_float2(mv[9], mv[9]);
                    float2 a2_20 = make_float2(acc[8], acc[9]);
                    a2_20 = fma_f32x2_rn_ftz(m2_10, make_float2(xv[0], xv[1]), a2_20);
                    acc[8] = a2_20.x;
                    acc[9] = a2_20.y;
                    float2 a2_21 = make_float2(acc[10], acc[11]);
                    a2_21 = fma_f32x2_rn_ftz(m2_10, make_float2(xv[2], xv[3]), a2_21);
                    acc[10] = a2_21.x;
                    acc[11] = a2_21.y;
                    float2 m2_11 = make_float2(mv[13], mv[13]);
                    float2 a2_22 = make_float2(acc[12], acc[13]);
                    a2_22 = fma_f32x2_rn_ftz(m2_11, make_float2(xv[0], xv[1]), a2_22);
                    acc[12] = a2_22.x;
                    acc[13] = a2_22.y;
                    float2 a2_23 = make_float2(acc[14], acc[15]);
                    a2_23 = fma_f32x2_rn_ftz(m2_11, make_float2(xv[2], xv[3]), a2_23);
                    acc[14] = a2_23.x;
                    acc[15] = a2_23.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 1280) + (unsigned int)(q * 320) + 160));
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
#else
                    float2 m2_12 = make_float2(mv[2], mv[2]);
                    float2 a2_24 = make_float2(acc[0], acc[1]);
                    a2_24 = fma_f32x2_rn_ftz(m2_12, make_float2(xv[0], xv[1]), a2_24);
                    acc[0] = a2_24.x;
                    acc[1] = a2_24.y;
                    float2 a2_25 = make_float2(acc[2], acc[3]);
                    a2_25 = fma_f32x2_rn_ftz(m2_12, make_float2(xv[2], xv[3]), a2_25);
                    acc[2] = a2_25.x;
                    acc[3] = a2_25.y;
                    float2 m2_13 = make_float2(mv[6], mv[6]);
                    float2 a2_26 = make_float2(acc[4], acc[5]);
                    a2_26 = fma_f32x2_rn_ftz(m2_13, make_float2(xv[0], xv[1]), a2_26);
                    acc[4] = a2_26.x;
                    acc[5] = a2_26.y;
                    float2 a2_27 = make_float2(acc[6], acc[7]);
                    a2_27 = fma_f32x2_rn_ftz(m2_13, make_float2(xv[2], xv[3]), a2_27);
                    acc[6] = a2_27.x;
                    acc[7] = a2_27.y;
                    float2 m2_14 = make_float2(mv[10], mv[10]);
                    float2 a2_28 = make_float2(acc[8], acc[9]);
                    a2_28 = fma_f32x2_rn_ftz(m2_14, make_float2(xv[0], xv[1]), a2_28);
                    acc[8] = a2_28.x;
                    acc[9] = a2_28.y;
                    float2 a2_29 = make_float2(acc[10], acc[11]);
                    a2_29 = fma_f32x2_rn_ftz(m2_14, make_float2(xv[2], xv[3]), a2_29);
                    acc[10] = a2_29.x;
                    acc[11] = a2_29.y;
                    float2 m2_15 = make_float2(mv[14], mv[14]);
                    float2 a2_30 = make_float2(acc[12], acc[13]);
                    a2_30 = fma_f32x2_rn_ftz(m2_15, make_float2(xv[0], xv[1]), a2_30);
                    acc[12] = a2_30.x;
                    acc[13] = a2_30.y;
                    float2 a2_31 = make_float2(acc[14], acc[15]);
                    a2_31 = fma_f32x2_rn_ftz(m2_15, make_float2(xv[2], xv[3]), a2_31);
                    acc[14] = a2_31.x;
                    acc[15] = a2_31.y;
#endif
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv[(0) + 3]))
                        : "r"(x_rd_e_1 + (unsigned int)(ch_1 * 1280) + (unsigned int)(q * 320) + 240));
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
#else
                    float2 m2_16 = make_float2(mv[3], mv[3]);
                    float2 a2_32 = make_float2(acc[0], acc[1]);
                    a2_32 = fma_f32x2_rn_ftz(m2_16, make_float2(xv[0], xv[1]), a2_32);
                    acc[0] = a2_32.x;
                    acc[1] = a2_32.y;
                    float2 a2_33 = make_float2(acc[2], acc[3]);
                    a2_33 = fma_f32x2_rn_ftz(m2_16, make_float2(xv[2], xv[3]), a2_33);
                    acc[2] = a2_33.x;
                    acc[3] = a2_33.y;
                    float2 m2_17 = make_float2(mv[7], mv[7]);
                    float2 a2_34 = make_float2(acc[4], acc[5]);
                    a2_34 = fma_f32x2_rn_ftz(m2_17, make_float2(xv[0], xv[1]), a2_34);
                    acc[4] = a2_34.x;
                    acc[5] = a2_34.y;
                    float2 a2_35 = make_float2(acc[6], acc[7]);
                    a2_35 = fma_f32x2_rn_ftz(m2_17, make_float2(xv[2], xv[3]), a2_35);
                    acc[6] = a2_35.x;
                    acc[7] = a2_35.y;
                    float2 m2_18 = make_float2(mv[11], mv[11]);
                    float2 a2_36 = make_float2(acc[8], acc[9]);
                    a2_36 = fma_f32x2_rn_ftz(m2_18, make_float2(xv[0], xv[1]), a2_36);
                    acc[8] = a2_36.x;
                    acc[9] = a2_36.y;
                    float2 a2_37 = make_float2(acc[10], acc[11]);
                    a2_37 = fma_f32x2_rn_ftz(m2_18, make_float2(xv[2], xv[3]), a2_37);
                    acc[10] = a2_37.x;
                    acc[11] = a2_37.y;
                    float2 m2_19 = make_float2(mv[15], mv[15]);
                    float2 a2_38 = make_float2(acc[12], acc[13]);
                    a2_38 = fma_f32x2_rn_ftz(m2_19, make_float2(xv[0], xv[1]), a2_38);
                    acc[12] = a2_38.x;
                    acc[13] = a2_38.y;
                    float2 a2_39 = make_float2(acc[14], acc[15]);
                    a2_39 = fma_f32x2_rn_ftz(m2_19, make_float2(xv[2], xv[3]), a2_39);
                    acc[14] = a2_39.x;
                    acc[15] = a2_39.y;
#endif
                }
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (10240 - buf_1 * 10240)), "r"(__float_as_uint(acc[0])), "r"(__float_as_uint(acc[1])), "r"(__float_as_uint(acc[2])), "r"(__float_as_uint(acc[3])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (10240 - buf_1 * 10240) + 640), "r"(__float_as_uint(acc[4])), "r"(__float_as_uint(acc[5])), "r"(__float_as_uint(acc[6])), "r"(__float_as_uint(acc[7])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (10240 - buf_1 * 10240) + 1280), "r"(__float_as_uint(acc[8])), "r"(__float_as_uint(acc[9])), "r"(__float_as_uint(acc[10])), "r"(__float_as_uint(acc[11])) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_thr_1 + (10240 - buf_1 * 10240) + 1920), "r"(__float_as_uint(acc[12])), "r"(__float_as_uint(acc[13])), "r"(__float_as_uint(acc[14])), "r"(__float_as_uint(acc[15])) : "memory");
            if (y_pub_1 >= 0ll) {
                {
                    float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[4 + 0], acc[4 + 1], acc[4 + 2], acc[4 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 1024ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[8 + 0], acc[8 + 1], acc[8 + 2], acc[8 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 2048ll) = _v4;
                }
                {
                    float4 _v4 = make_float4(acc[12 + 0], acc[12 + 1], acc[12 + 2], acc[12 + 3]);
                    *reinterpret_cast<float4*>(y + y_pub_1 + g_off_1 + 3072ll) = _v4;
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
