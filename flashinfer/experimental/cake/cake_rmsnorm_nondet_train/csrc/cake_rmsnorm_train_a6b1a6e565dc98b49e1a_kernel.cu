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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}


extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(8,1,1) void
kernel_cake_rmsnorm_train_a6b1a6e565dc98b49e1a(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int mbar_base = smem;
    #define recv_bar_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* wpart = reinterpret_cast<float*>(smem_raw + 1152);
    const int wpart_addr = smem + 1152;
    float* pairs = reinterpret_cast<float*>(smem_raw + 1200);
    const int pairs_addr = smem + 1200;
    float* recv = reinterpret_cast<float*>(smem_raw + 1024);
    const int recv_addr = smem + 1024;

    // Mbarrier init (1 barriers at smem_raw[0..8))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // recv_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 0, 128);
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive(recv_bar_addr);
        }
    }
    int rg = warp / 3;
    int ct = tid - rg * 96;
    int col = ct * 8;
    int rank = cta_rank;
    col = rank * 768 + ct * 8;
    float ww[8];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(w + ((unsigned long long)col))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_0_0);
        ww[0 + 1] = __uint_as_float(_ldv8_0_1);
        ww[0 + 2] = __uint_as_float(_ldv8_0_2);
        ww[0 + 3] = __uint_as_float(_ldv8_0_3);
        ww[0 + 4] = __uint_as_float(_ldv8_0_4);
        ww[0 + 5] = __uint_as_float(_ldv8_0_5);
        ww[0 + 6] = __uint_as_float(_ldv8_0_6);
        ww[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[8];
    float rr[2];
    int lrow_0 = rg;
    if (lrow_0 < T) {
        rr[0] = r[lrow_0];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_0 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&gw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_0 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&xw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(g_h + (unsigned long long)lrow_0 * (unsigned long long)g_h_stride_p0 + (unsigned long long)col);
            uint4* _vdst_3 = reinterpret_cast<uint4*>(&hw[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_3[_blk] = _vptr_3[_blk];
            }
        }
    }
    int lrow_1 = 2 + rg;
    if (lrow_1 < T) {
        rr[1] = r[lrow_1];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(g + (unsigned long long)lrow_1 * (unsigned long long)g_stride_p0 + (unsigned long long)col);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&gw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + (unsigned long long)lrow_1 * (unsigned long long)x_stride_p0 + (unsigned long long)col);
            uint4* _vdst_5 = reinterpret_cast<uint4*>(&xw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_5[_blk] = _vptr_5[_blk];
            }
        }
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(g_h + (unsigned long long)lrow_1 * (unsigned long long)g_h_stride_p0 + (unsigned long long)col);
            uint4* _vdst_6 = reinterpret_cast<uint4*>(&hw[4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_6[_blk] = _vptr_6[_blk];
            }
        }
    }
    float acc[8];
    float comp[8];
    float dq[2];
    acc[0] = 0.0f;
    comp[0] = 0.0f;
    acc[1] = 0.0f;
    comp[1] = 0.0f;
    acc[2] = 0.0f;
    comp[2] = 0.0f;
    acc[3] = 0.0f;
    comp[3] = 0.0f;
    acc[4] = 0.0f;
    comp[4] = 0.0f;
    acc[5] = 0.0f;
    comp[5] = 0.0f;
    acc[6] = 0.0f;
    comp[6] = 0.0f;
    acc[7] = 0.0f;
    comp[7] = 0.0f;
    dq[0] = 0.0f;
    if (rg < T) {
        float2 rr2 = make_float2(rr[0], rr[0]);
        float dot = 0.0f;
        float g0_0 = __uint_as_float(gw[0] << 16);
        float g1_0 = __uint_as_float(gw[0] & 4294901760u);
        float x0_0 = __uint_as_float(xw[0] << 16);
        float x1_0 = __uint_as_float(xw[0] & 4294901760u);
        dot = __fmaf_rn(g0_0 * ww[0], x0_0, dot);
        dot = __fmaf_rn(g1_0 * ww[1], x1_0, dot);
        float2 _f32x2_mul_a_7 = make_float2(g0_0, g1_0);
        float2 _f32x2_mul_b_8 = make_float2(x0_0, x1_0);
        float2 gx;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
        float2 p;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rr2));
        float2 a = make_float2(acc[0], acc[1]);
        float2 s = add_f32x2_noftz(a, p);
        float2 bn = sub_f32x2_noftz(a, s);
        float2 av = add_f32x2_noftz(s, bn);
        float2 db = fma_f32x2_rn_noftz(gx, rr2, bn);
        float2 c = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a, av)), db);
        acc[0] = s.x;
        acc[1] = s.y;
        comp[0] = c.x;
        comp[1] = c.y;
        float g0_1 = __uint_as_float(gw[1] << 16);
        float g1_1 = __uint_as_float(gw[1] & 4294901760u);
        float x0_1 = __uint_as_float(xw[1] << 16);
        float x1_1 = __uint_as_float(xw[1] & 4294901760u);
        dot = __fmaf_rn(g0_1 * ww[2], x0_1, dot);
        dot = __fmaf_rn(g1_1 * ww[3], x1_1, dot);
        float2 _f32x2_mul_a_9 = make_float2(g0_1, g1_1);
        float2 _f32x2_mul_b_10 = make_float2(x0_1, x1_1);
        float2 gx_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float2 p_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&gx_0), "l"(*(const unsigned long long*)&rr2));
        float2 a_0 = make_float2(acc[2], acc[3]);
        float2 s_0 = add_f32x2_noftz(a_0, p_0);
        float2 bn_0 = sub_f32x2_noftz(a_0, s_0);
        float2 av_0 = add_f32x2_noftz(s_0, bn_0);
        float2 db_0 = fma_f32x2_rn_noftz(gx_0, rr2, bn_0);
        float2 c_0 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_0, av_0)), db_0);
        acc[2] = s_0.x;
        acc[3] = s_0.y;
        comp[2] = c_0.x;
        comp[3] = c_0.y;
        float g0_2 = __uint_as_float(gw[2] << 16);
        float g1_2 = __uint_as_float(gw[2] & 4294901760u);
        float x0_2 = __uint_as_float(xw[2] << 16);
        float x1_2 = __uint_as_float(xw[2] & 4294901760u);
        dot = __fmaf_rn(g0_2 * ww[4], x0_2, dot);
        dot = __fmaf_rn(g1_2 * ww[5], x1_2, dot);
        float2 _f32x2_mul_a_11 = make_float2(g0_2, g1_2);
        float2 _f32x2_mul_b_12 = make_float2(x0_2, x1_2);
        float2 gx_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float2 p_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&gx_1), "l"(*(const unsigned long long*)&rr2));
        float2 a_1 = make_float2(acc[4], acc[5]);
        float2 s_1 = add_f32x2_noftz(a_1, p_1);
        float2 bn_1 = sub_f32x2_noftz(a_1, s_1);
        float2 av_1 = add_f32x2_noftz(s_1, bn_1);
        float2 db_1 = fma_f32x2_rn_noftz(gx_1, rr2, bn_1);
        float2 c_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_1, av_1)), db_1);
        acc[4] = s_1.x;
        acc[5] = s_1.y;
        comp[4] = c_1.x;
        comp[5] = c_1.y;
        float g0_3 = __uint_as_float(gw[3] << 16);
        float g1_3 = __uint_as_float(gw[3] & 4294901760u);
        float x0_3 = __uint_as_float(xw[3] << 16);
        float x1_3 = __uint_as_float(xw[3] & 4294901760u);
        dot = __fmaf_rn(g0_3 * ww[6], x0_3, dot);
        dot = __fmaf_rn(g1_3 * ww[7], x1_3, dot);
        float2 _f32x2_mul_a_13 = make_float2(g0_3, g1_3);
        float2 _f32x2_mul_b_14 = make_float2(x0_3, x1_3);
        float2 gx_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float2 p_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&gx_2), "l"(*(const unsigned long long*)&rr2));
        float2 a_2 = make_float2(acc[6], acc[7]);
        float2 s_2 = add_f32x2_noftz(a_2, p_2);
        float2 bn_2 = sub_f32x2_noftz(a_2, s_2);
        float2 av_2 = add_f32x2_noftz(s_2, bn_2);
        float2 db_2 = fma_f32x2_rn_noftz(gx_2, rr2, bn_2);
        float2 c_2 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_2, av_2)), db_2);
        acc[6] = s_2.x;
        acc[7] = s_2.y;
        comp[6] = c_2.x;
        comp[7] = c_2.y;
        dq[0] = dot;
    }
    dq[1] = 0.0f;
    if (2 + rg < T) {
        float2 rr2_1 = make_float2(rr[1], rr[1]);
        float dot_1 = 0.0f;
        float g0_4 = __uint_as_float(gw[4] << 16);
        float g1_4 = __uint_as_float(gw[4] & 4294901760u);
        float x0_4 = __uint_as_float(xw[4] << 16);
        float x1_4 = __uint_as_float(xw[4] & 4294901760u);
        dot_1 = __fmaf_rn(g0_4 * ww[0], x0_4, dot_1);
        dot_1 = __fmaf_rn(g1_4 * ww[1], x1_4, dot_1);
        float2 _f32x2_mul_a_15 = make_float2(g0_4, g1_4);
        float2 _f32x2_mul_b_16 = make_float2(x0_4, x1_4);
        float2 gx_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float2 p_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3) : "l"(*(const unsigned long long*)&gx_3), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_3 = make_float2(acc[0], acc[1]);
        float2 s_3 = add_f32x2_noftz(a_3, p_3);
        float2 bn_3 = sub_f32x2_noftz(a_3, s_3);
        float2 av_3 = add_f32x2_noftz(s_3, bn_3);
        float2 db_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, bn_3);
        float2 c_3 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a_3, av_3)), db_3);
        acc[0] = s_3.x;
        acc[1] = s_3.y;
        comp[0] = c_3.x;
        comp[1] = c_3.y;
        float g0_5 = __uint_as_float(gw[5] << 16);
        float g1_5 = __uint_as_float(gw[5] & 4294901760u);
        float x0_5 = __uint_as_float(xw[5] << 16);
        float x1_5 = __uint_as_float(xw[5] & 4294901760u);
        dot_1 = __fmaf_rn(g0_5 * ww[2], x0_5, dot_1);
        dot_1 = __fmaf_rn(g1_5 * ww[3], x1_5, dot_1);
        float2 _f32x2_mul_a_17 = make_float2(g0_5, g1_5);
        float2 _f32x2_mul_b_18 = make_float2(x0_5, x1_5);
        float2 gx_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float2 p_3_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3_1) : "l"(*(const unsigned long long*)&gx_3_1), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_3_1 = make_float2(acc[2], acc[3]);
        float2 s_3_1 = add_f32x2_noftz(a_3_1, p_3_1);
        float2 bn_3_1 = sub_f32x2_noftz(a_3_1, s_3_1);
        float2 av_3_1 = add_f32x2_noftz(s_3_1, bn_3_1);
        float2 db_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, bn_3_1);
        float2 c_3_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_3_1, av_3_1)), db_3_1);
        acc[2] = s_3_1.x;
        acc[3] = s_3_1.y;
        comp[2] = c_3_1.x;
        comp[3] = c_3_1.y;
        float g0_6 = __uint_as_float(gw[6] << 16);
        float g1_6 = __uint_as_float(gw[6] & 4294901760u);
        float x0_6 = __uint_as_float(xw[6] << 16);
        float x1_6 = __uint_as_float(xw[6] & 4294901760u);
        dot_1 = __fmaf_rn(g0_6 * ww[4], x0_6, dot_1);
        dot_1 = __fmaf_rn(g1_6 * ww[5], x1_6, dot_1);
        float2 _f32x2_mul_a_19 = make_float2(g0_6, g1_6);
        float2 _f32x2_mul_b_20 = make_float2(x0_6, x1_6);
        float2 gx_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float2 p_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_4) : "l"(*(const unsigned long long*)&gx_4), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_4 = make_float2(acc[4], acc[5]);
        float2 s_4 = add_f32x2_noftz(a_4, p_4);
        float2 bn_4 = sub_f32x2_noftz(a_4, s_4);
        float2 av_4 = add_f32x2_noftz(s_4, bn_4);
        float2 db_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, bn_4);
        float2 c_4 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_4, av_4)), db_4);
        acc[4] = s_4.x;
        acc[5] = s_4.y;
        comp[4] = c_4.x;
        comp[5] = c_4.y;
        float g0_7 = __uint_as_float(gw[7] << 16);
        float g1_7 = __uint_as_float(gw[7] & 4294901760u);
        float x0_7 = __uint_as_float(xw[7] << 16);
        float x1_7 = __uint_as_float(xw[7] & 4294901760u);
        dot_1 = __fmaf_rn(g0_7 * ww[6], x0_7, dot_1);
        dot_1 = __fmaf_rn(g1_7 * ww[7], x1_7, dot_1);
        float2 _f32x2_mul_a_21 = make_float2(g0_7, g1_7);
        float2 _f32x2_mul_b_22 = make_float2(x0_7, x1_7);
        float2 gx_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float2 p_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_5) : "l"(*(const unsigned long long*)&gx_5), "l"(*(const unsigned long long*)&rr2_1));
        float2 a_5 = make_float2(acc[6], acc[7]);
        float2 s_5 = add_f32x2_noftz(a_5, p_5);
        float2 bn_5 = sub_f32x2_noftz(a_5, s_5);
        float2 av_5 = add_f32x2_noftz(s_5, bn_5);
        float2 db_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, bn_5);
        float2 c_5 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_5, av_5)), db_5);
        acc[6] = s_5.x;
        acc[7] = s_5.y;
        comp[6] = c_5.x;
        comp[7] = c_5.y;
        dq[1] = dot_1;
    }
    float _warp_reduce_23 = dq[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
    dq[0] = _warp_reduce_23;
    float _warp_reduce_24 = dq[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
    dq[1] = _warp_reduce_24;
    if (lane == 0) {
        wpart[rg * 3 + warp - rg * 3] = dq[0];
        wpart[(2 + rg) * 3 + warp - rg * 3] = dq[1];
    }
    pairs[rg * 2 * 8 * 96 + ct] = acc[0];
    pairs[(rg * 2 * 8 + 1) * 96 + ct] = comp[0];
    pairs[(rg * 2 * 8 + 2) * 96 + ct] = acc[1];
    pairs[(rg * 2 * 8 + 2 + 1) * 96 + ct] = comp[1];
    pairs[(rg * 2 * 8 + 4) * 96 + ct] = acc[2];
    pairs[(rg * 2 * 8 + 4 + 1) * 96 + ct] = comp[2];
    pairs[(rg * 2 * 8 + 6) * 96 + ct] = acc[3];
    pairs[(rg * 2 * 8 + 6 + 1) * 96 + ct] = comp[3];
    pairs[(rg * 2 * 8 + 8) * 96 + ct] = acc[4];
    pairs[(rg * 2 * 8 + 8 + 1) * 96 + ct] = comp[4];
    pairs[(rg * 2 * 8 + 10) * 96 + ct] = acc[5];
    pairs[(rg * 2 * 8 + 10 + 1) * 96 + ct] = comp[5];
    pairs[(rg * 2 * 8 + 12) * 96 + ct] = acc[6];
    pairs[(rg * 2 * 8 + 12 + 1) * 96 + ct] = comp[6];
    pairs[(rg * 2 * 8 + 14) * 96 + ct] = acc[7];
    pairs[(rg * 2 * 8 + 14 + 1) * 96 + ct] = comp[7];
    __syncthreads();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid < 8) {
        int dst = tid;
        int qq = tid - dst;
        float pv[4];
        pv[0] = wpart[4 * qq * 3];
        pv[0] = pv[0] + wpart[4 * qq * 3 + 1];
        pv[0] = pv[0] + wpart[4 * qq * 3 + 1 + 1];
        pv[1] = wpart[(4 * qq + 1) * 3];
        pv[1] = pv[1] + wpart[(4 * qq + 1) * 3 + 1];
        pv[1] = pv[1] + wpart[(4 * qq + 1) * 3 + 1 + 1];
        pv[2] = wpart[(4 * qq + 2) * 3];
        pv[2] = pv[2] + wpart[(4 * qq + 2) * 3 + 1];
        pv[2] = pv[2] + wpart[(4 * qq + 2) * 3 + 1 + 1];
        pv[3] = wpart[(4 * qq + 3) * 3];
        pv[3] = pv[3] + wpart[(4 * qq + 3) * 3 + 1];
        pv[3] = pv[3] + wpart[(4 * qq + 3) * 3 + 1 + 1];
        uint32_t _cluster_mapa_25;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_25) : "r"(recv_addr + (unsigned int)((rank * 4 + 4 * qq) * 4)), "r"(dst));
        uint32_t _mbar_peer_26;
        asm(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_mbar_peer_26) : "r"(recv_bar_addr), "r"(dst));
        asm volatile(
            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
            :: "r"(_cluster_mapa_25), "r"(__float_as_uint(pv[0])), "r"(__float_as_uint(pv[1])), "r"(__float_as_uint(pv[2])), "r"(__float_as_uint(pv[3])), "r"(_mbar_peer_26) : "memory");
    }
    float dwo[4];
    float sc[2];
    int jc_0 = rg * 4;
    sc[0] = pairs[2 * jc_0 * 96 + ct];
    sc[1] = pairs[(2 * jc_0 + 1) * 96 + ct];
    float s_6 = sc[0] + pairs[(16 + 2 * jc_0) * 96 + ct];
    float bp = s_6 - sc[0];
    float ap = s_6 - bp;
    float e = sc[0] - ap + (pairs[(16 + 2 * jc_0) * 96 + ct] - bp);
    sc[0] = s_6;
    sc[1] = sc[1] + pairs[(16 + 2 * jc_0 + 1) * 96 + ct] + e;
    dwo[0] = sc[0] + sc[1];
    int jc_1 = rg * 4 + 1;
    sc[0] = pairs[2 * jc_1 * 96 + ct];
    sc[1] = pairs[(2 * jc_1 + 1) * 96 + ct];
    float s_6_1 = sc[0] + pairs[(16 + 2 * jc_1) * 96 + ct];
    float bp_0 = s_6_1 - sc[0];
    float ap_0 = s_6_1 - bp_0;
    float e_0 = sc[0] - ap_0 + (pairs[(16 + 2 * jc_1) * 96 + ct] - bp_0);
    sc[0] = s_6_1;
    sc[1] = sc[1] + pairs[(16 + 2 * jc_1 + 1) * 96 + ct] + e_0;
    dwo[1] = sc[0] + sc[1];
    int jc_2 = rg * 4 + 2;
    sc[0] = pairs[2 * jc_2 * 96 + ct];
    sc[1] = pairs[(2 * jc_2 + 1) * 96 + ct];
    float s_7 = sc[0] + pairs[(16 + 2 * jc_2) * 96 + ct];
    float bp_1 = s_7 - sc[0];
    float ap_1 = s_7 - bp_1;
    float e_1 = sc[0] - ap_1 + (pairs[(16 + 2 * jc_2) * 96 + ct] - bp_1);
    sc[0] = s_7;
    sc[1] = sc[1] + pairs[(16 + 2 * jc_2 + 1) * 96 + ct] + e_1;
    dwo[2] = sc[0] + sc[1];
    int jc_3 = rg * 4 + 3;
    sc[0] = pairs[2 * jc_3 * 96 + ct];
    sc[1] = pairs[(2 * jc_3 + 1) * 96 + ct];
    float s_8 = sc[0] + pairs[(16 + 2 * jc_3) * 96 + ct];
    float bp_2 = s_8 - sc[0];
    float ap_2 = s_8 - bp_2;
    float e_2 = sc[0] - ap_2 + (pairs[(16 + 2 * jc_3) * 96 + ct] - bp_2);
    sc[0] = s_8;
    sc[1] = sc[1] + pairs[(16 + 2 * jc_3 + 1) * 96 + ct] + e_2;
    dwo[3] = sc[0] + sc[1];
    {
        float4 _v4 = make_float4(dwo[0 + 0], dwo[0 + 1], dwo[0 + 2], dwo[0 + 3]);
        *reinterpret_cast<float4*>(dw + col + rg * 4) = _v4;
    }
    mbarrier_wait(recv_bar_addr, 0);
    int row_0 = rg;
    if (row_0 < T) {
        float tot_0 = dq[0];
        tot_0 = recv[row_0];
        tot_0 = tot_0 + recv[4 + row_0];
        tot_0 = tot_0 + recv[8 + row_0];
        tot_0 = tot_0 + recv[12 + row_0];
        tot_0 = tot_0 + recv[16 + row_0];
        tot_0 = tot_0 + recv[20 + row_0];
        tot_0 = tot_0 + recv[24 + row_0];
        tot_0 = tot_0 + recv[28 + row_0];
        float rq = rr[0];
        float cq = __fdiv_rn(tot_0, 6144.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        float out[8];
        float g0_8 = __uint_as_float(gw[0] << 16);
        float g1_8 = __uint_as_float(gw[0] & 4294901760u);
        float x0_8 = __uint_as_float(xw[0] << 16);
        float x1_8 = __uint_as_float(xw[0] & 4294901760u);
        float y0_0 = __fmaf_rn(nkx, x0_8, rq * (g0_8 * ww[0]));
        float y1_0 = __fmaf_rn(nkx, x1_8, rq * (g1_8 * ww[1]));
        out[0] = y0_0 + __uint_as_float(hw[0] << 16);
        out[1] = y1_0 + __uint_as_float(hw[0] & 4294901760u);
        float g0_9 = __uint_as_float(gw[1] << 16);
        float g1_9 = __uint_as_float(gw[1] & 4294901760u);
        float x0_9 = __uint_as_float(xw[1] << 16);
        float x1_9 = __uint_as_float(xw[1] & 4294901760u);
        float y0_1 = __fmaf_rn(nkx, x0_9, rq * (g0_9 * ww[2]));
        float y1_1 = __fmaf_rn(nkx, x1_9, rq * (g1_9 * ww[3]));
        out[2] = y0_1 + __uint_as_float(hw[1] << 16);
        out[3] = y1_1 + __uint_as_float(hw[1] & 4294901760u);
        float g0_10 = __uint_as_float(gw[2] << 16);
        float g1_10 = __uint_as_float(gw[2] & 4294901760u);
        float x0_10 = __uint_as_float(xw[2] << 16);
        float x1_10 = __uint_as_float(xw[2] & 4294901760u);
        float y0_2 = __fmaf_rn(nkx, x0_10, rq * (g0_10 * ww[4]));
        float y1_2 = __fmaf_rn(nkx, x1_10, rq * (g1_10 * ww[5]));
        out[4] = y0_2 + __uint_as_float(hw[2] << 16);
        out[5] = y1_2 + __uint_as_float(hw[2] & 4294901760u);
        float g0_11 = __uint_as_float(gw[3] << 16);
        float g1_11 = __uint_as_float(gw[3] & 4294901760u);
        float x0_11 = __uint_as_float(xw[3] << 16);
        float x1_11 = __uint_as_float(xw[3] & 4294901760u);
        float y0_3 = __fmaf_rn(nkx, x0_11, rq * (g0_11 * ww[6]));
        float y1_3 = __fmaf_rn(nkx, x1_11, rq * (g1_11 * ww[7]));
        out[6] = y0_3 + __uint_as_float(hw[3] << 16);
        out[7] = y1_3 + __uint_as_float(hw[3] & 4294901760u);
        unsigned int packed[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_0 * 6144ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
    }
    int row_1 = 2 + rg;
    if (row_1 < T) {
        float tot_1 = dq[1];
        tot_1 = recv[row_1];
        tot_1 = tot_1 + recv[4 + row_1];
        tot_1 = tot_1 + recv[8 + row_1];
        tot_1 = tot_1 + recv[12 + row_1];
        tot_1 = tot_1 + recv[16 + row_1];
        tot_1 = tot_1 + recv[20 + row_1];
        tot_1 = tot_1 + recv[24 + row_1];
        tot_1 = tot_1 + recv[28 + row_1];
        float rq_1 = rr[1];
        float cq_1 = __fdiv_rn(tot_1, 6144.0f);
        float kx_1 = rq_1 * rq_1 * rq_1 * cq_1;
        float nkx_1 = -kx_1;
        float out_1[8];
        float g0_12 = __uint_as_float(gw[4] << 16);
        float g1_12 = __uint_as_float(gw[4] & 4294901760u);
        float x0_12 = __uint_as_float(xw[4] << 16);
        float x1_12 = __uint_as_float(xw[4] & 4294901760u);
        float y0_4 = __fmaf_rn(nkx_1, x0_12, rq_1 * (g0_12 * ww[0]));
        float y1_4 = __fmaf_rn(nkx_1, x1_12, rq_1 * (g1_12 * ww[1]));
        out_1[0] = y0_4 + __uint_as_float(hw[4] << 16);
        out_1[1] = y1_4 + __uint_as_float(hw[4] & 4294901760u);
        float g0_13 = __uint_as_float(gw[5] << 16);
        float g1_13 = __uint_as_float(gw[5] & 4294901760u);
        float x0_13 = __uint_as_float(xw[5] << 16);
        float x1_13 = __uint_as_float(xw[5] & 4294901760u);
        float y0_5 = __fmaf_rn(nkx_1, x0_13, rq_1 * (g0_13 * ww[2]));
        float y1_5 = __fmaf_rn(nkx_1, x1_13, rq_1 * (g1_13 * ww[3]));
        out_1[2] = y0_5 + __uint_as_float(hw[5] << 16);
        out_1[3] = y1_5 + __uint_as_float(hw[5] & 4294901760u);
        float g0_14 = __uint_as_float(gw[6] << 16);
        float g1_14 = __uint_as_float(gw[6] & 4294901760u);
        float x0_14 = __uint_as_float(xw[6] << 16);
        float x1_14 = __uint_as_float(xw[6] & 4294901760u);
        float y0_6 = __fmaf_rn(nkx_1, x0_14, rq_1 * (g0_14 * ww[4]));
        float y1_6 = __fmaf_rn(nkx_1, x1_14, rq_1 * (g1_14 * ww[5]));
        out_1[4] = y0_6 + __uint_as_float(hw[6] << 16);
        out_1[5] = y1_6 + __uint_as_float(hw[6] & 4294901760u);
        float g0_15 = __uint_as_float(gw[7] << 16);
        float g1_15 = __uint_as_float(gw[7] & 4294901760u);
        float x0_15 = __uint_as_float(xw[7] << 16);
        float x1_15 = __uint_as_float(xw[7] & 4294901760u);
        float y0_7 = __fmaf_rn(nkx_1, x0_15, rq_1 * (g0_15 * ww[6]));
        float y1_7 = __fmaf_rn(nkx_1, x1_15, rq_1 * (g1_15 * ww[7]));
        out_1[6] = y0_7 + __uint_as_float(hw[7] << 16);
        out_1[7] = y1_7 + __uint_as_float(hw[7] & 4294901760u);
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(dx + ((unsigned long long)row_1 * 6144ull + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed_1)[0];
    }
}

} // extern "C"
