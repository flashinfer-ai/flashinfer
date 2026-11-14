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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_2e9171b7753650cda515(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    double* sums = reinterpret_cast<double*>(smem_raw + 1152);
    const int sums_addr = smem + 1152;
    float* sumsf = reinterpret_cast<float*>(smem_raw + 1152);
    const int sumsf_addr = smem + 1152;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1152);
    const int dwbuf_addr = smem + 1152;
    int* meta = reinterpret_cast<int*>(smem_raw + 1120);
    const int meta_addr = smem + 1120;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 17536);
    const int ring_addr = smem + 17536;

    // Mbarrier init (4 barriers at smem_raw[0..32))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 4;
    int tir = tid - rg * 128;
    float ww[16];
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
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)(tir * 8)))) : "memory");
        ww[0 + 0] = __uint_as_float(_ldv8_1_0);
        ww[0 + 1] = __uint_as_float(_ldv8_1_1);
        ww[0 + 2] = __uint_as_float(_ldv8_1_2);
        ww[0 + 3] = __uint_as_float(_ldv8_1_3);
        ww[0 + 4] = __uint_as_float(_ldv8_1_4);
        ww[0 + 5] = __uint_as_float(_ldv8_1_5);
        ww[0 + 6] = __uint_as_float(_ldv8_1_6);
        ww[0 + 7] = __uint_as_float(_ldv8_1_7);
    }
    {
        unsigned _ldv8_2_0;
        unsigned _ldv8_2_1;
        unsigned _ldv8_2_2;
        unsigned _ldv8_2_3;
        unsigned _ldv8_2_4;
        unsigned _ldv8_2_5;
        unsigned _ldv8_2_6;
        unsigned _ldv8_2_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + ((unsigned long long)((128 + tir) * 8)))) : "memory");
        ww[8 + 0] = __uint_as_float(_ldv8_2_0);
        ww[8 + 1] = __uint_as_float(_ldv8_2_1);
        ww[8 + 2] = __uint_as_float(_ldv8_2_2);
        ww[8 + 3] = __uint_as_float(_ldv8_2_3);
        ww[8 + 4] = __uint_as_float(_ldv8_2_4);
        ww[8 + 5] = __uint_as_float(_ldv8_2_5);
        ww[8 + 6] = __uint_as_float(_ldv8_2_6);
        ww[8 + 7] = __uint_as_float(_ldv8_2_7);
    }
    float acc[16];
    float comp[1];
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
    int c = 0;
    int first_c = 0;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[1];
    float rr[1];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        flags[4] = (int)nxt;
    }
    __syncthreads();
    int tk = flags[4];
    #pragma unroll 1
    for (int ci = 0; ci < 32; ci++) {
        if (tk >= 32) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int vl = tid % 8;
        int rl = tid / 8;
        unsigned int gw_0[32];
        unsigned int xw_0[32];
        unsigned int hw_0[1];
        float rr_0[8];
        float acc_0[8];
        float comp_0[8];
        acc_0[0] = 0.0f;
        comp_0[0] = 0.0f;
        acc_0[1] = 0.0f;
        comp_0[1] = 0.0f;
        acc_0[2] = 0.0f;
        comp_0[2] = 0.0f;
        acc_0[3] = 0.0f;
        comp_0[3] = 0.0f;
        acc_0[4] = 0.0f;
        comp_0[4] = 0.0f;
        acc_0[5] = 0.0f;
        comp_0[5] = 0.0f;
        acc_0[6] = 0.0f;
        comp_0[6] = 0.0f;
        acc_0[7] = 0.0f;
        comp_0[7] = 0.0f;
        #pragma unroll 1
        for (int t0 = rl; t0 < T; t0 += 256) {
            if (t0 < T) {
                unsigned long long g_base = (unsigned long long)t0 * (unsigned long long)g_stride;
                unsigned long long x_base = (unsigned long long)t0 * (unsigned long long)x_stride;
                rr_0[0] = r[t0];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&gw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&xw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 32 < T) {
                unsigned long long g_base_1 = (unsigned long long)(t0 + 32) * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)(t0 + 32) * (unsigned long long)x_stride;
                rr_0[1] = r[t0 + 32];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&gw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&xw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 64 < T) {
                unsigned long long g_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)(t0 + 64) * (unsigned long long)x_stride;
                rr_0[2] = r[t0 + 64];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 96 < T) {
                unsigned long long g_base_3 = (unsigned long long)(t0 + 96) * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)(t0 + 96) * (unsigned long long)x_stride;
                rr_0[3] = r[t0 + 96];
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_10 = reinterpret_cast<uint4*>(&gw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_11 = reinterpret_cast<uint4*>(&xw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 128 < T) {
                unsigned long long g_base_4 = (unsigned long long)(t0 + 128) * (unsigned long long)g_stride;
                unsigned long long x_base_4 = (unsigned long long)(t0 + 128) * (unsigned long long)x_stride;
                rr_0[4] = r[t0 + 128];
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 160 < T) {
                unsigned long long g_base_5 = (unsigned long long)(t0 + 160) * (unsigned long long)g_stride;
                unsigned long long x_base_5 = (unsigned long long)(t0 + 160) * (unsigned long long)x_stride;
                rr_0[5] = r[t0 + 160];
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 192 < T) {
                unsigned long long g_base_6 = (unsigned long long)(t0 + 192) * (unsigned long long)g_stride;
                unsigned long long x_base_6 = (unsigned long long)(t0 + 192) * (unsigned long long)x_stride;
                rr_0[6] = r[t0 + 192];
                {
                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw_0[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw_0[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 224 < T) {
                unsigned long long g_base_7 = (unsigned long long)(t0 + 224) * (unsigned long long)g_stride;
                unsigned long long x_base_7 = (unsigned long long)(t0 + 224) * (unsigned long long)x_stride;
                rr_0[7] = r[t0 + 224];
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw_0[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)((tk * 8 + vl) * 8));
                    uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw_0[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 < T) {
                float2 rr2_0 = make_float2(rr_0[0], rr_0[0]);
                float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(gw_0[0] << 16), __uint_as_float(gw_0[0] & 4294901760u));
                float2 _f32x2_mul_b_21 = make_float2(__uint_as_float(xw_0[0] << 16), __uint_as_float(xw_0[0] & 4294901760u));
                float2 gx;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                float2 p;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rr2_0));
                float2 a = make_float2(acc_0[0], acc_0[1]);
                float2 s = add_f32x2_noftz(a, p);
                float2 bn = sub_f32x2_noftz(a, s);
                float2 av = add_f32x2_noftz(s, bn);
                float2 db = fma_f32x2_rn_noftz(gx, rr2_0, bn);
                float2 c_0 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a, av)), db);
                acc_0[0] = s.x;
                acc_0[1] = s.y;
                comp_0[0] = c_0.x;
                comp_0[1] = c_0.y;
                float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(gw_0[1] << 16), __uint_as_float(gw_0[1] & 4294901760u));
                float2 _f32x2_mul_b_23 = make_float2(__uint_as_float(xw_0[1] << 16), __uint_as_float(xw_0[1] & 4294901760u));
                float2 gx_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
                float2 p_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&gx_0), "l"(*(const unsigned long long*)&rr2_0));
                float2 a_0 = make_float2(acc_0[2], acc_0[3]);
                float2 s_0 = add_f32x2_noftz(a_0, p_0);
                float2 bn_0 = sub_f32x2_noftz(a_0, s_0);
                float2 av_0 = add_f32x2_noftz(s_0, bn_0);
                float2 db_0 = fma_f32x2_rn_noftz(gx_0, rr2_0, bn_0);
                float2 c_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_0, av_0)), db_0);
                acc_0[2] = s_0.x;
                acc_0[3] = s_0.y;
                comp_0[2] = c_1.x;
                comp_0[3] = c_1.y;
                float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(gw_0[2] << 16), __uint_as_float(gw_0[2] & 4294901760u));
                float2 _f32x2_mul_b_25 = make_float2(__uint_as_float(xw_0[2] << 16), __uint_as_float(xw_0[2] & 4294901760u));
                float2 gx_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_25));
                float2 p_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&gx_1), "l"(*(const unsigned long long*)&rr2_0));
                float2 a_1 = make_float2(acc_0[4], acc_0[5]);
                float2 s_1 = add_f32x2_noftz(a_1, p_1);
                float2 bn_1 = sub_f32x2_noftz(a_1, s_1);
                float2 av_1 = add_f32x2_noftz(s_1, bn_1);
                float2 db_1 = fma_f32x2_rn_noftz(gx_1, rr2_0, bn_1);
                float2 c_2 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_1, av_1)), db_1);
                acc_0[4] = s_1.x;
                acc_0[5] = s_1.y;
                comp_0[4] = c_2.x;
                comp_0[5] = c_2.y;
                float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(gw_0[3] << 16), __uint_as_float(gw_0[3] & 4294901760u));
                float2 _f32x2_mul_b_27 = make_float2(__uint_as_float(xw_0[3] << 16), __uint_as_float(xw_0[3] & 4294901760u));
                float2 gx_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
                float2 p_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&gx_2), "l"(*(const unsigned long long*)&rr2_0));
                float2 a_2 = make_float2(acc_0[6], acc_0[7]);
                float2 s_2 = add_f32x2_noftz(a_2, p_2);
                float2 bn_2 = sub_f32x2_noftz(a_2, s_2);
                float2 av_2 = add_f32x2_noftz(s_2, bn_2);
                float2 db_2 = fma_f32x2_rn_noftz(gx_2, rr2_0, bn_2);
                float2 c_3 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_2, av_2)), db_2);
                acc_0[6] = s_2.x;
                acc_0[7] = s_2.y;
                comp_0[6] = c_3.x;
                comp_0[7] = c_3.y;
            }
            if (t0 + 32 < T) {
                float2 rr2_1 = make_float2(rr_0[1], rr_0[1]);
                float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(gw_0[4] << 16), __uint_as_float(gw_0[4] & 4294901760u));
                float2 _f32x2_mul_b_29 = make_float2(__uint_as_float(xw_0[4] << 16), __uint_as_float(xw_0[4] & 4294901760u));
                float2 gx_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_29));
                float2 p_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3) : "l"(*(const unsigned long long*)&gx_3), "l"(*(const unsigned long long*)&rr2_1));
                float2 a_3 = make_float2(acc_0[0], acc_0[1]);
                float2 s_3 = add_f32x2_noftz(a_3, p_3);
                float2 bn_3 = sub_f32x2_noftz(a_3, s_3);
                float2 av_3 = add_f32x2_noftz(s_3, bn_3);
                float2 db_3 = fma_f32x2_rn_noftz(gx_3, rr2_1, bn_3);
                float2 c_4 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_3, av_3)), db_3);
                acc_0[0] = s_3.x;
                acc_0[1] = s_3.y;
                comp_0[0] = c_4.x;
                comp_0[1] = c_4.y;
                float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(gw_0[5] << 16), __uint_as_float(gw_0[5] & 4294901760u));
                float2 _f32x2_mul_b_31 = make_float2(__uint_as_float(xw_0[5] << 16), __uint_as_float(xw_0[5] & 4294901760u));
                float2 gx_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
                float2 p_3_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3_1) : "l"(*(const unsigned long long*)&gx_3_1), "l"(*(const unsigned long long*)&rr2_1));
                float2 a_3_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_3_1 = add_f32x2_noftz(a_3_1, p_3_1);
                float2 bn_3_1 = sub_f32x2_noftz(a_3_1, s_3_1);
                float2 av_3_1 = add_f32x2_noftz(s_3_1, bn_3_1);
                float2 db_3_1 = fma_f32x2_rn_noftz(gx_3_1, rr2_1, bn_3_1);
                float2 c_5 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_3_1, av_3_1)), db_3_1);
                acc_0[2] = s_3_1.x;
                acc_0[3] = s_3_1.y;
                comp_0[2] = c_5.x;
                comp_0[3] = c_5.y;
                float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(gw_0[6] << 16), __uint_as_float(gw_0[6] & 4294901760u));
                float2 _f32x2_mul_b_33 = make_float2(__uint_as_float(xw_0[6] << 16), __uint_as_float(xw_0[6] & 4294901760u));
                float2 gx_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_33));
                float2 p_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_4) : "l"(*(const unsigned long long*)&gx_4), "l"(*(const unsigned long long*)&rr2_1));
                float2 a_4 = make_float2(acc_0[4], acc_0[5]);
                float2 s_4 = add_f32x2_noftz(a_4, p_4);
                float2 bn_4 = sub_f32x2_noftz(a_4, s_4);
                float2 av_4 = add_f32x2_noftz(s_4, bn_4);
                float2 db_4 = fma_f32x2_rn_noftz(gx_4, rr2_1, bn_4);
                float2 c_6 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_4, av_4)), db_4);
                acc_0[4] = s_4.x;
                acc_0[5] = s_4.y;
                comp_0[4] = c_6.x;
                comp_0[5] = c_6.y;
                float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(gw_0[7] << 16), __uint_as_float(gw_0[7] & 4294901760u));
                float2 _f32x2_mul_b_35 = make_float2(__uint_as_float(xw_0[7] << 16), __uint_as_float(xw_0[7] & 4294901760u));
                float2 gx_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_35));
                float2 p_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_5) : "l"(*(const unsigned long long*)&gx_5), "l"(*(const unsigned long long*)&rr2_1));
                float2 a_5 = make_float2(acc_0[6], acc_0[7]);
                float2 s_5 = add_f32x2_noftz(a_5, p_5);
                float2 bn_5 = sub_f32x2_noftz(a_5, s_5);
                float2 av_5 = add_f32x2_noftz(s_5, bn_5);
                float2 db_5 = fma_f32x2_rn_noftz(gx_5, rr2_1, bn_5);
                float2 c_7 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_5, av_5)), db_5);
                acc_0[6] = s_5.x;
                acc_0[7] = s_5.y;
                comp_0[6] = c_7.x;
                comp_0[7] = c_7.y;
            }
            if (t0 + 64 < T) {
                float2 rr2_2 = make_float2(rr_0[2], rr_0[2]);
                float2 _f32x2_mul_a_36 = make_float2(__uint_as_float(gw_0[8] << 16), __uint_as_float(gw_0[8] & 4294901760u));
                float2 _f32x2_mul_b_37 = make_float2(__uint_as_float(xw_0[8] << 16), __uint_as_float(xw_0[8] & 4294901760u));
                float2 gx_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                float2 p_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6) : "l"(*(const unsigned long long*)&gx_6), "l"(*(const unsigned long long*)&rr2_2));
                float2 a_6 = make_float2(acc_0[0], acc_0[1]);
                float2 s_6 = add_f32x2_noftz(a_6, p_6);
                float2 bn_6 = sub_f32x2_noftz(a_6, s_6);
                float2 av_6 = add_f32x2_noftz(s_6, bn_6);
                float2 db_6 = fma_f32x2_rn_noftz(gx_6, rr2_2, bn_6);
                float2 c_8 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_6, av_6)), db_6);
                acc_0[0] = s_6.x;
                acc_0[1] = s_6.y;
                comp_0[0] = c_8.x;
                comp_0[1] = c_8.y;
                float2 _f32x2_mul_a_38 = make_float2(__uint_as_float(gw_0[9] << 16), __uint_as_float(gw_0[9] & 4294901760u));
                float2 _f32x2_mul_b_39 = make_float2(__uint_as_float(xw_0[9] << 16), __uint_as_float(xw_0[9] & 4294901760u));
                float2 gx_6_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
                float2 p_6_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6_1) : "l"(*(const unsigned long long*)&gx_6_1), "l"(*(const unsigned long long*)&rr2_2));
                float2 a_6_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_6_1 = add_f32x2_noftz(a_6_1, p_6_1);
                float2 bn_6_1 = sub_f32x2_noftz(a_6_1, s_6_1);
                float2 av_6_1 = add_f32x2_noftz(s_6_1, bn_6_1);
                float2 db_6_1 = fma_f32x2_rn_noftz(gx_6_1, rr2_2, bn_6_1);
                float2 c_9 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_6_1, av_6_1)), db_6_1);
                acc_0[2] = s_6_1.x;
                acc_0[3] = s_6_1.y;
                comp_0[2] = c_9.x;
                comp_0[3] = c_9.y;
                float2 _f32x2_mul_a_40 = make_float2(__uint_as_float(gw_0[10] << 16), __uint_as_float(gw_0[10] & 4294901760u));
                float2 _f32x2_mul_b_41 = make_float2(__uint_as_float(xw_0[10] << 16), __uint_as_float(xw_0[10] & 4294901760u));
                float2 gx_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
                float2 p_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_7) : "l"(*(const unsigned long long*)&gx_7), "l"(*(const unsigned long long*)&rr2_2));
                float2 a_7 = make_float2(acc_0[4], acc_0[5]);
                float2 s_7 = add_f32x2_noftz(a_7, p_7);
                float2 bn_7 = sub_f32x2_noftz(a_7, s_7);
                float2 av_7 = add_f32x2_noftz(s_7, bn_7);
                float2 db_7 = fma_f32x2_rn_noftz(gx_7, rr2_2, bn_7);
                float2 c_10 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_7, av_7)), db_7);
                acc_0[4] = s_7.x;
                acc_0[5] = s_7.y;
                comp_0[4] = c_10.x;
                comp_0[5] = c_10.y;
                float2 _f32x2_mul_a_42 = make_float2(__uint_as_float(gw_0[11] << 16), __uint_as_float(gw_0[11] & 4294901760u));
                float2 _f32x2_mul_b_43 = make_float2(__uint_as_float(xw_0[11] << 16), __uint_as_float(xw_0[11] & 4294901760u));
                float2 gx_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
                float2 p_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_8) : "l"(*(const unsigned long long*)&gx_8), "l"(*(const unsigned long long*)&rr2_2));
                float2 a_8 = make_float2(acc_0[6], acc_0[7]);
                float2 s_8 = add_f32x2_noftz(a_8, p_8);
                float2 bn_8 = sub_f32x2_noftz(a_8, s_8);
                float2 av_8 = add_f32x2_noftz(s_8, bn_8);
                float2 db_8 = fma_f32x2_rn_noftz(gx_8, rr2_2, bn_8);
                float2 c_11 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_8, av_8)), db_8);
                acc_0[6] = s_8.x;
                acc_0[7] = s_8.y;
                comp_0[6] = c_11.x;
                comp_0[7] = c_11.y;
            }
            if (t0 + 96 < T) {
                float2 rr2_3 = make_float2(rr_0[3], rr_0[3]);
                float2 _f32x2_mul_a_44 = make_float2(__uint_as_float(gw_0[12] << 16), __uint_as_float(gw_0[12] & 4294901760u));
                float2 _f32x2_mul_b_45 = make_float2(__uint_as_float(xw_0[12] << 16), __uint_as_float(xw_0[12] & 4294901760u));
                float2 gx_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
                float2 p_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9) : "l"(*(const unsigned long long*)&gx_9), "l"(*(const unsigned long long*)&rr2_3));
                float2 a_9 = make_float2(acc_0[0], acc_0[1]);
                float2 s_9 = add_f32x2_noftz(a_9, p_9);
                float2 bn_9 = sub_f32x2_noftz(a_9, s_9);
                float2 av_9 = add_f32x2_noftz(s_9, bn_9);
                float2 db_9 = fma_f32x2_rn_noftz(gx_9, rr2_3, bn_9);
                float2 c_12 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_9, av_9)), db_9);
                acc_0[0] = s_9.x;
                acc_0[1] = s_9.y;
                comp_0[0] = c_12.x;
                comp_0[1] = c_12.y;
                float2 _f32x2_mul_a_46 = make_float2(__uint_as_float(gw_0[13] << 16), __uint_as_float(gw_0[13] & 4294901760u));
                float2 _f32x2_mul_b_47 = make_float2(__uint_as_float(xw_0[13] << 16), __uint_as_float(xw_0[13] & 4294901760u));
                float2 gx_9_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
                float2 p_9_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9_1) : "l"(*(const unsigned long long*)&gx_9_1), "l"(*(const unsigned long long*)&rr2_3));
                float2 a_9_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_9_1 = add_f32x2_noftz(a_9_1, p_9_1);
                float2 bn_9_1 = sub_f32x2_noftz(a_9_1, s_9_1);
                float2 av_9_1 = add_f32x2_noftz(s_9_1, bn_9_1);
                float2 db_9_1 = fma_f32x2_rn_noftz(gx_9_1, rr2_3, bn_9_1);
                float2 c_13 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_9_1, av_9_1)), db_9_1);
                acc_0[2] = s_9_1.x;
                acc_0[3] = s_9_1.y;
                comp_0[2] = c_13.x;
                comp_0[3] = c_13.y;
                float2 _f32x2_mul_a_48 = make_float2(__uint_as_float(gw_0[14] << 16), __uint_as_float(gw_0[14] & 4294901760u));
                float2 _f32x2_mul_b_49 = make_float2(__uint_as_float(xw_0[14] << 16), __uint_as_float(xw_0[14] & 4294901760u));
                float2 gx_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
                float2 p_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_10) : "l"(*(const unsigned long long*)&gx_10), "l"(*(const unsigned long long*)&rr2_3));
                float2 a_10 = make_float2(acc_0[4], acc_0[5]);
                float2 s_10 = add_f32x2_noftz(a_10, p_10);
                float2 bn_10 = sub_f32x2_noftz(a_10, s_10);
                float2 av_10 = add_f32x2_noftz(s_10, bn_10);
                float2 db_10 = fma_f32x2_rn_noftz(gx_10, rr2_3, bn_10);
                float2 c_14 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_10, av_10)), db_10);
                acc_0[4] = s_10.x;
                acc_0[5] = s_10.y;
                comp_0[4] = c_14.x;
                comp_0[5] = c_14.y;
                float2 _f32x2_mul_a_50 = make_float2(__uint_as_float(gw_0[15] << 16), __uint_as_float(gw_0[15] & 4294901760u));
                float2 _f32x2_mul_b_51 = make_float2(__uint_as_float(xw_0[15] << 16), __uint_as_float(xw_0[15] & 4294901760u));
                float2 gx_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
                float2 p_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_11) : "l"(*(const unsigned long long*)&gx_11), "l"(*(const unsigned long long*)&rr2_3));
                float2 a_11 = make_float2(acc_0[6], acc_0[7]);
                float2 s_11 = add_f32x2_noftz(a_11, p_11);
                float2 bn_11 = sub_f32x2_noftz(a_11, s_11);
                float2 av_11 = add_f32x2_noftz(s_11, bn_11);
                float2 db_11 = fma_f32x2_rn_noftz(gx_11, rr2_3, bn_11);
                float2 c_15 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_11, av_11)), db_11);
                acc_0[6] = s_11.x;
                acc_0[7] = s_11.y;
                comp_0[6] = c_15.x;
                comp_0[7] = c_15.y;
            }
            if (t0 + 128 < T) {
                float2 rr2_4 = make_float2(rr_0[4], rr_0[4]);
                float2 _f32x2_mul_a_52 = make_float2(__uint_as_float(gw_0[16] << 16), __uint_as_float(gw_0[16] & 4294901760u));
                float2 _f32x2_mul_b_53 = make_float2(__uint_as_float(xw_0[16] << 16), __uint_as_float(xw_0[16] & 4294901760u));
                float2 gx_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
                float2 p_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12) : "l"(*(const unsigned long long*)&gx_12), "l"(*(const unsigned long long*)&rr2_4));
                float2 a_12 = make_float2(acc_0[0], acc_0[1]);
                float2 s_12 = add_f32x2_noftz(a_12, p_12);
                float2 bn_12 = sub_f32x2_noftz(a_12, s_12);
                float2 av_12 = add_f32x2_noftz(s_12, bn_12);
                float2 db_12 = fma_f32x2_rn_noftz(gx_12, rr2_4, bn_12);
                float2 c_16 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_12, av_12)), db_12);
                acc_0[0] = s_12.x;
                acc_0[1] = s_12.y;
                comp_0[0] = c_16.x;
                comp_0[1] = c_16.y;
                float2 _f32x2_mul_a_54 = make_float2(__uint_as_float(gw_0[17] << 16), __uint_as_float(gw_0[17] & 4294901760u));
                float2 _f32x2_mul_b_55 = make_float2(__uint_as_float(xw_0[17] << 16), __uint_as_float(xw_0[17] & 4294901760u));
                float2 gx_12_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
                float2 p_12_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12_1) : "l"(*(const unsigned long long*)&gx_12_1), "l"(*(const unsigned long long*)&rr2_4));
                float2 a_12_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_12_1 = add_f32x2_noftz(a_12_1, p_12_1);
                float2 bn_12_1 = sub_f32x2_noftz(a_12_1, s_12_1);
                float2 av_12_1 = add_f32x2_noftz(s_12_1, bn_12_1);
                float2 db_12_1 = fma_f32x2_rn_noftz(gx_12_1, rr2_4, bn_12_1);
                float2 c_17 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_12_1, av_12_1)), db_12_1);
                acc_0[2] = s_12_1.x;
                acc_0[3] = s_12_1.y;
                comp_0[2] = c_17.x;
                comp_0[3] = c_17.y;
                float2 _f32x2_mul_a_56 = make_float2(__uint_as_float(gw_0[18] << 16), __uint_as_float(gw_0[18] & 4294901760u));
                float2 _f32x2_mul_b_57 = make_float2(__uint_as_float(xw_0[18] << 16), __uint_as_float(xw_0[18] & 4294901760u));
                float2 gx_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
                float2 p_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_13) : "l"(*(const unsigned long long*)&gx_13), "l"(*(const unsigned long long*)&rr2_4));
                float2 a_13 = make_float2(acc_0[4], acc_0[5]);
                float2 s_13 = add_f32x2_noftz(a_13, p_13);
                float2 bn_13 = sub_f32x2_noftz(a_13, s_13);
                float2 av_13 = add_f32x2_noftz(s_13, bn_13);
                float2 db_13 = fma_f32x2_rn_noftz(gx_13, rr2_4, bn_13);
                float2 c_18 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_13, av_13)), db_13);
                acc_0[4] = s_13.x;
                acc_0[5] = s_13.y;
                comp_0[4] = c_18.x;
                comp_0[5] = c_18.y;
                float2 _f32x2_mul_a_58 = make_float2(__uint_as_float(gw_0[19] << 16), __uint_as_float(gw_0[19] & 4294901760u));
                float2 _f32x2_mul_b_59 = make_float2(__uint_as_float(xw_0[19] << 16), __uint_as_float(xw_0[19] & 4294901760u));
                float2 gx_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                float2 p_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_14) : "l"(*(const unsigned long long*)&gx_14), "l"(*(const unsigned long long*)&rr2_4));
                float2 a_14 = make_float2(acc_0[6], acc_0[7]);
                float2 s_14 = add_f32x2_noftz(a_14, p_14);
                float2 bn_14 = sub_f32x2_noftz(a_14, s_14);
                float2 av_14 = add_f32x2_noftz(s_14, bn_14);
                float2 db_14 = fma_f32x2_rn_noftz(gx_14, rr2_4, bn_14);
                float2 c_19 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_14, av_14)), db_14);
                acc_0[6] = s_14.x;
                acc_0[7] = s_14.y;
                comp_0[6] = c_19.x;
                comp_0[7] = c_19.y;
            }
            if (t0 + 160 < T) {
                float2 rr2_5 = make_float2(rr_0[5], rr_0[5]);
                float2 _f32x2_mul_a_60 = make_float2(__uint_as_float(gw_0[20] << 16), __uint_as_float(gw_0[20] & 4294901760u));
                float2 _f32x2_mul_b_61 = make_float2(__uint_as_float(xw_0[20] << 16), __uint_as_float(xw_0[20] & 4294901760u));
                float2 gx_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
                float2 p_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15) : "l"(*(const unsigned long long*)&gx_15), "l"(*(const unsigned long long*)&rr2_5));
                float2 a_15 = make_float2(acc_0[0], acc_0[1]);
                float2 s_15 = add_f32x2_noftz(a_15, p_15);
                float2 bn_15 = sub_f32x2_noftz(a_15, s_15);
                float2 av_15 = add_f32x2_noftz(s_15, bn_15);
                float2 db_15 = fma_f32x2_rn_noftz(gx_15, rr2_5, bn_15);
                float2 c_20 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_15, av_15)), db_15);
                acc_0[0] = s_15.x;
                acc_0[1] = s_15.y;
                comp_0[0] = c_20.x;
                comp_0[1] = c_20.y;
                float2 _f32x2_mul_a_62 = make_float2(__uint_as_float(gw_0[21] << 16), __uint_as_float(gw_0[21] & 4294901760u));
                float2 _f32x2_mul_b_63 = make_float2(__uint_as_float(xw_0[21] << 16), __uint_as_float(xw_0[21] & 4294901760u));
                float2 gx_15_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_15_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
                float2 p_15_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_15_1) : "l"(*(const unsigned long long*)&gx_15_1), "l"(*(const unsigned long long*)&rr2_5));
                float2 a_15_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_15_1 = add_f32x2_noftz(a_15_1, p_15_1);
                float2 bn_15_1 = sub_f32x2_noftz(a_15_1, s_15_1);
                float2 av_15_1 = add_f32x2_noftz(s_15_1, bn_15_1);
                float2 db_15_1 = fma_f32x2_rn_noftz(gx_15_1, rr2_5, bn_15_1);
                float2 c_21 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_15_1, av_15_1)), db_15_1);
                acc_0[2] = s_15_1.x;
                acc_0[3] = s_15_1.y;
                comp_0[2] = c_21.x;
                comp_0[3] = c_21.y;
                float2 _f32x2_mul_a_64 = make_float2(__uint_as_float(gw_0[22] << 16), __uint_as_float(gw_0[22] & 4294901760u));
                float2 _f32x2_mul_b_65 = make_float2(__uint_as_float(xw_0[22] << 16), __uint_as_float(xw_0[22] & 4294901760u));
                float2 gx_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
                float2 p_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_16) : "l"(*(const unsigned long long*)&gx_16), "l"(*(const unsigned long long*)&rr2_5));
                float2 a_16 = make_float2(acc_0[4], acc_0[5]);
                float2 s_16 = add_f32x2_noftz(a_16, p_16);
                float2 bn_16 = sub_f32x2_noftz(a_16, s_16);
                float2 av_16 = add_f32x2_noftz(s_16, bn_16);
                float2 db_16 = fma_f32x2_rn_noftz(gx_16, rr2_5, bn_16);
                float2 c_22 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_16, av_16)), db_16);
                acc_0[4] = s_16.x;
                acc_0[5] = s_16.y;
                comp_0[4] = c_22.x;
                comp_0[5] = c_22.y;
                float2 _f32x2_mul_a_66 = make_float2(__uint_as_float(gw_0[23] << 16), __uint_as_float(gw_0[23] & 4294901760u));
                float2 _f32x2_mul_b_67 = make_float2(__uint_as_float(xw_0[23] << 16), __uint_as_float(xw_0[23] & 4294901760u));
                float2 gx_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
                float2 p_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_17) : "l"(*(const unsigned long long*)&gx_17), "l"(*(const unsigned long long*)&rr2_5));
                float2 a_17 = make_float2(acc_0[6], acc_0[7]);
                float2 s_17 = add_f32x2_noftz(a_17, p_17);
                float2 bn_17 = sub_f32x2_noftz(a_17, s_17);
                float2 av_17 = add_f32x2_noftz(s_17, bn_17);
                float2 db_17 = fma_f32x2_rn_noftz(gx_17, rr2_5, bn_17);
                float2 c_23 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_17, av_17)), db_17);
                acc_0[6] = s_17.x;
                acc_0[7] = s_17.y;
                comp_0[6] = c_23.x;
                comp_0[7] = c_23.y;
            }
            if (t0 + 192 < T) {
                float2 rr2_6 = make_float2(rr_0[6], rr_0[6]);
                float2 _f32x2_mul_a_68 = make_float2(__uint_as_float(gw_0[24] << 16), __uint_as_float(gw_0[24] & 4294901760u));
                float2 _f32x2_mul_b_69 = make_float2(__uint_as_float(xw_0[24] << 16), __uint_as_float(xw_0[24] & 4294901760u));
                float2 gx_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
                float2 p_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18) : "l"(*(const unsigned long long*)&gx_18), "l"(*(const unsigned long long*)&rr2_6));
                float2 a_18 = make_float2(acc_0[0], acc_0[1]);
                float2 s_18 = add_f32x2_noftz(a_18, p_18);
                float2 bn_18 = sub_f32x2_noftz(a_18, s_18);
                float2 av_18 = add_f32x2_noftz(s_18, bn_18);
                float2 db_18 = fma_f32x2_rn_noftz(gx_18, rr2_6, bn_18);
                float2 c_24 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_18, av_18)), db_18);
                acc_0[0] = s_18.x;
                acc_0[1] = s_18.y;
                comp_0[0] = c_24.x;
                comp_0[1] = c_24.y;
                float2 _f32x2_mul_a_70 = make_float2(__uint_as_float(gw_0[25] << 16), __uint_as_float(gw_0[25] & 4294901760u));
                float2 _f32x2_mul_b_71 = make_float2(__uint_as_float(xw_0[25] << 16), __uint_as_float(xw_0[25] & 4294901760u));
                float2 gx_18_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_18_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
                float2 p_18_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_18_1) : "l"(*(const unsigned long long*)&gx_18_1), "l"(*(const unsigned long long*)&rr2_6));
                float2 a_18_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_18_1 = add_f32x2_noftz(a_18_1, p_18_1);
                float2 bn_18_1 = sub_f32x2_noftz(a_18_1, s_18_1);
                float2 av_18_1 = add_f32x2_noftz(s_18_1, bn_18_1);
                float2 db_18_1 = fma_f32x2_rn_noftz(gx_18_1, rr2_6, bn_18_1);
                float2 c_25 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_18_1, av_18_1)), db_18_1);
                acc_0[2] = s_18_1.x;
                acc_0[3] = s_18_1.y;
                comp_0[2] = c_25.x;
                comp_0[3] = c_25.y;
                float2 _f32x2_mul_a_72 = make_float2(__uint_as_float(gw_0[26] << 16), __uint_as_float(gw_0[26] & 4294901760u));
                float2 _f32x2_mul_b_73 = make_float2(__uint_as_float(xw_0[26] << 16), __uint_as_float(xw_0[26] & 4294901760u));
                float2 gx_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
                float2 p_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_19) : "l"(*(const unsigned long long*)&gx_19), "l"(*(const unsigned long long*)&rr2_6));
                float2 a_19 = make_float2(acc_0[4], acc_0[5]);
                float2 s_19 = add_f32x2_noftz(a_19, p_19);
                float2 bn_19 = sub_f32x2_noftz(a_19, s_19);
                float2 av_19 = add_f32x2_noftz(s_19, bn_19);
                float2 db_19 = fma_f32x2_rn_noftz(gx_19, rr2_6, bn_19);
                float2 c_26 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_19, av_19)), db_19);
                acc_0[4] = s_19.x;
                acc_0[5] = s_19.y;
                comp_0[4] = c_26.x;
                comp_0[5] = c_26.y;
                float2 _f32x2_mul_a_74 = make_float2(__uint_as_float(gw_0[27] << 16), __uint_as_float(gw_0[27] & 4294901760u));
                float2 _f32x2_mul_b_75 = make_float2(__uint_as_float(xw_0[27] << 16), __uint_as_float(xw_0[27] & 4294901760u));
                float2 gx_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                float2 p_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_20) : "l"(*(const unsigned long long*)&gx_20), "l"(*(const unsigned long long*)&rr2_6));
                float2 a_20 = make_float2(acc_0[6], acc_0[7]);
                float2 s_20 = add_f32x2_noftz(a_20, p_20);
                float2 bn_20 = sub_f32x2_noftz(a_20, s_20);
                float2 av_20 = add_f32x2_noftz(s_20, bn_20);
                float2 db_20 = fma_f32x2_rn_noftz(gx_20, rr2_6, bn_20);
                float2 c_27 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_20, av_20)), db_20);
                acc_0[6] = s_20.x;
                acc_0[7] = s_20.y;
                comp_0[6] = c_27.x;
                comp_0[7] = c_27.y;
            }
            if (t0 + 224 < T) {
                float2 rr2_7 = make_float2(rr_0[7], rr_0[7]);
                float2 _f32x2_mul_a_76 = make_float2(__uint_as_float(gw_0[28] << 16), __uint_as_float(gw_0[28] & 4294901760u));
                float2 _f32x2_mul_b_77 = make_float2(__uint_as_float(xw_0[28] << 16), __uint_as_float(xw_0[28] & 4294901760u));
                float2 gx_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
                float2 p_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21) : "l"(*(const unsigned long long*)&gx_21), "l"(*(const unsigned long long*)&rr2_7));
                float2 a_21 = make_float2(acc_0[0], acc_0[1]);
                float2 s_21 = add_f32x2_noftz(a_21, p_21);
                float2 bn_21 = sub_f32x2_noftz(a_21, s_21);
                float2 av_21 = add_f32x2_noftz(s_21, bn_21);
                float2 db_21 = fma_f32x2_rn_noftz(gx_21, rr2_7, bn_21);
                float2 c_28 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[0], comp_0[1]), sub_f32x2_noftz(a_21, av_21)), db_21);
                acc_0[0] = s_21.x;
                acc_0[1] = s_21.y;
                comp_0[0] = c_28.x;
                comp_0[1] = c_28.y;
                float2 _f32x2_mul_a_78 = make_float2(__uint_as_float(gw_0[29] << 16), __uint_as_float(gw_0[29] & 4294901760u));
                float2 _f32x2_mul_b_79 = make_float2(__uint_as_float(xw_0[29] << 16), __uint_as_float(xw_0[29] & 4294901760u));
                float2 gx_21_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_21_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
                float2 p_21_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_21_1) : "l"(*(const unsigned long long*)&gx_21_1), "l"(*(const unsigned long long*)&rr2_7));
                float2 a_21_1 = make_float2(acc_0[2], acc_0[3]);
                float2 s_21_1 = add_f32x2_noftz(a_21_1, p_21_1);
                float2 bn_21_1 = sub_f32x2_noftz(a_21_1, s_21_1);
                float2 av_21_1 = add_f32x2_noftz(s_21_1, bn_21_1);
                float2 db_21_1 = fma_f32x2_rn_noftz(gx_21_1, rr2_7, bn_21_1);
                float2 c_29 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[2], comp_0[3]), sub_f32x2_noftz(a_21_1, av_21_1)), db_21_1);
                acc_0[2] = s_21_1.x;
                acc_0[3] = s_21_1.y;
                comp_0[2] = c_29.x;
                comp_0[3] = c_29.y;
                float2 _f32x2_mul_a_80 = make_float2(__uint_as_float(gw_0[30] << 16), __uint_as_float(gw_0[30] & 4294901760u));
                float2 _f32x2_mul_b_81 = make_float2(__uint_as_float(xw_0[30] << 16), __uint_as_float(xw_0[30] & 4294901760u));
                float2 gx_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
                float2 p_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_22) : "l"(*(const unsigned long long*)&gx_22), "l"(*(const unsigned long long*)&rr2_7));
                float2 a_22 = make_float2(acc_0[4], acc_0[5]);
                float2 s_22 = add_f32x2_noftz(a_22, p_22);
                float2 bn_22 = sub_f32x2_noftz(a_22, s_22);
                float2 av_22 = add_f32x2_noftz(s_22, bn_22);
                float2 db_22 = fma_f32x2_rn_noftz(gx_22, rr2_7, bn_22);
                float2 c_30 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[4], comp_0[5]), sub_f32x2_noftz(a_22, av_22)), db_22);
                acc_0[4] = s_22.x;
                acc_0[5] = s_22.y;
                comp_0[4] = c_30.x;
                comp_0[5] = c_30.y;
                float2 _f32x2_mul_a_82 = make_float2(__uint_as_float(gw_0[31] << 16), __uint_as_float(gw_0[31] & 4294901760u));
                float2 _f32x2_mul_b_83 = make_float2(__uint_as_float(xw_0[31] << 16), __uint_as_float(xw_0[31] & 4294901760u));
                float2 gx_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
                float2 p_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_23) : "l"(*(const unsigned long long*)&gx_23), "l"(*(const unsigned long long*)&rr2_7));
                float2 a_23 = make_float2(acc_0[6], acc_0[7]);
                float2 s_23 = add_f32x2_noftz(a_23, p_23);
                float2 bn_23 = sub_f32x2_noftz(a_23, s_23);
                float2 av_23 = add_f32x2_noftz(s_23, bn_23);
                float2 db_23 = fma_f32x2_rn_noftz(gx_23, rr2_7, bn_23);
                float2 c_31 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp_0[6], comp_0[7]), sub_f32x2_noftz(a_23, av_23)), db_23);
                acc_0[6] = s_23.x;
                acc_0[7] = s_23.y;
                comp_0[6] = c_31.x;
                comp_0[7] = c_31.y;
            }
        }
        sumsf[2 * (rl * 64 + vl * 8)] = acc_0[0];
        sumsf[2 * (rl * 64 + vl * 8) + 1] = comp_0[0];
        sumsf[2 * (rl * 64 + vl * 8 + 1)] = acc_0[1];
        sumsf[2 * (rl * 64 + vl * 8 + 1) + 1] = comp_0[1];
        sumsf[2 * (rl * 64 + vl * 8 + 2)] = acc_0[2];
        sumsf[2 * (rl * 64 + vl * 8 + 2) + 1] = comp_0[2];
        sumsf[2 * (rl * 64 + vl * 8 + 3)] = acc_0[3];
        sumsf[2 * (rl * 64 + vl * 8 + 3) + 1] = comp_0[3];
        sumsf[2 * (rl * 64 + vl * 8 + 4)] = acc_0[4];
        sumsf[2 * (rl * 64 + vl * 8 + 4) + 1] = comp_0[4];
        sumsf[2 * (rl * 64 + vl * 8 + 5)] = acc_0[5];
        sumsf[2 * (rl * 64 + vl * 8 + 5) + 1] = comp_0[5];
        sumsf[2 * (rl * 64 + vl * 8 + 6)] = acc_0[6];
        sumsf[2 * (rl * 64 + vl * 8 + 6) + 1] = comp_0[6];
        sumsf[2 * (rl * 64 + vl * 8 + 7)] = acc_0[7];
        sumsf[2 * (rl * 64 + vl * 8 + 7) + 1] = comp_0[7];
        __syncthreads();
        int cl = tid % 64;
        int s8 = tid / 64 * 8;
        float p_24[3];
        p_24[0] = 0.0f;
        p_24[1] = 0.0f;
        p_24[2] = 0.0f;
        float a_24 = p_24[0] + sumsf[2 * (s8 * 64 + cl)];
        float bp = a_24 - p_24[0];
        float ap = a_24 - bp;
        float e0 = p_24[0] - ap + (sumsf[2 * (s8 * 64 + cl)] - bp);
        p_24[0] = a_24;
        float b = p_24[1] + e0;
        float bq = b - p_24[1];
        float aq = b - bq;
        float e1 = p_24[1] - aq + (e0 - bq);
        p_24[1] = b;
        p_24[2] = p_24[2] + e1;
        float a_24_1 = p_24[0] + sumsf[2 * (s8 * 64 + cl) + 1];
        float bp_0 = a_24_1 - p_24[0];
        float ap_0 = a_24_1 - bp_0;
        float e0_0 = p_24[0] - ap_0 + (sumsf[2 * (s8 * 64 + cl) + 1] - bp_0);
        p_24[0] = a_24_1;
        float b_0 = p_24[1] + e0_0;
        float bq_0 = b_0 - p_24[1];
        float aq_0 = b_0 - bq_0;
        float e1_0 = p_24[1] - aq_0 + (e0_0 - bq_0);
        p_24[1] = b_0;
        p_24[2] = p_24[2] + e1_0;
        float a_25 = p_24[0] + sumsf[2 * ((s8 + 1) * 64 + cl)];
        float bp_1 = a_25 - p_24[0];
        float ap_1 = a_25 - bp_1;
        float e0_1 = p_24[0] - ap_1 + (sumsf[2 * ((s8 + 1) * 64 + cl)] - bp_1);
        p_24[0] = a_25;
        float b_1 = p_24[1] + e0_1;
        float bq_1 = b_1 - p_24[1];
        float aq_1 = b_1 - bq_1;
        float e1_1 = p_24[1] - aq_1 + (e0_1 - bq_1);
        p_24[1] = b_1;
        p_24[2] = p_24[2] + e1_1;
        float a_26 = p_24[0] + sumsf[2 * ((s8 + 1) * 64 + cl) + 1];
        float bp_2 = a_26 - p_24[0];
        float ap_2 = a_26 - bp_2;
        float e0_2 = p_24[0] - ap_2 + (sumsf[2 * ((s8 + 1) * 64 + cl) + 1] - bp_2);
        p_24[0] = a_26;
        float b_2 = p_24[1] + e0_2;
        float bq_2 = b_2 - p_24[1];
        float aq_2 = b_2 - bq_2;
        float e1_2 = p_24[1] - aq_2 + (e0_2 - bq_2);
        p_24[1] = b_2;
        p_24[2] = p_24[2] + e1_2;
        float a_27 = p_24[0] + sumsf[2 * ((s8 + 2) * 64 + cl)];
        float bp_3 = a_27 - p_24[0];
        float ap_3 = a_27 - bp_3;
        float e0_3 = p_24[0] - ap_3 + (sumsf[2 * ((s8 + 2) * 64 + cl)] - bp_3);
        p_24[0] = a_27;
        float b_3 = p_24[1] + e0_3;
        float bq_3 = b_3 - p_24[1];
        float aq_3 = b_3 - bq_3;
        float e1_3 = p_24[1] - aq_3 + (e0_3 - bq_3);
        p_24[1] = b_3;
        p_24[2] = p_24[2] + e1_3;
        float a_28 = p_24[0] + sumsf[2 * ((s8 + 2) * 64 + cl) + 1];
        float bp_4 = a_28 - p_24[0];
        float ap_4 = a_28 - bp_4;
        float e0_4 = p_24[0] - ap_4 + (sumsf[2 * ((s8 + 2) * 64 + cl) + 1] - bp_4);
        p_24[0] = a_28;
        float b_4 = p_24[1] + e0_4;
        float bq_4 = b_4 - p_24[1];
        float aq_4 = b_4 - bq_4;
        float e1_4 = p_24[1] - aq_4 + (e0_4 - bq_4);
        p_24[1] = b_4;
        p_24[2] = p_24[2] + e1_4;
        float a_29 = p_24[0] + sumsf[2 * ((s8 + 3) * 64 + cl)];
        float bp_5 = a_29 - p_24[0];
        float ap_5 = a_29 - bp_5;
        float e0_5 = p_24[0] - ap_5 + (sumsf[2 * ((s8 + 3) * 64 + cl)] - bp_5);
        p_24[0] = a_29;
        float b_5 = p_24[1] + e0_5;
        float bq_5 = b_5 - p_24[1];
        float aq_5 = b_5 - bq_5;
        float e1_5 = p_24[1] - aq_5 + (e0_5 - bq_5);
        p_24[1] = b_5;
        p_24[2] = p_24[2] + e1_5;
        float a_30 = p_24[0] + sumsf[2 * ((s8 + 3) * 64 + cl) + 1];
        float bp_6 = a_30 - p_24[0];
        float ap_6 = a_30 - bp_6;
        float e0_6 = p_24[0] - ap_6 + (sumsf[2 * ((s8 + 3) * 64 + cl) + 1] - bp_6);
        p_24[0] = a_30;
        float b_6 = p_24[1] + e0_6;
        float bq_6 = b_6 - p_24[1];
        float aq_6 = b_6 - bq_6;
        float e1_6 = p_24[1] - aq_6 + (e0_6 - bq_6);
        p_24[1] = b_6;
        p_24[2] = p_24[2] + e1_6;
        float a_31 = p_24[0] + sumsf[2 * ((s8 + 4) * 64 + cl)];
        float bp_7 = a_31 - p_24[0];
        float ap_7 = a_31 - bp_7;
        float e0_7 = p_24[0] - ap_7 + (sumsf[2 * ((s8 + 4) * 64 + cl)] - bp_7);
        p_24[0] = a_31;
        float b_7 = p_24[1] + e0_7;
        float bq_7 = b_7 - p_24[1];
        float aq_7 = b_7 - bq_7;
        float e1_7 = p_24[1] - aq_7 + (e0_7 - bq_7);
        p_24[1] = b_7;
        p_24[2] = p_24[2] + e1_7;
        float a_32 = p_24[0] + sumsf[2 * ((s8 + 4) * 64 + cl) + 1];
        float bp_8 = a_32 - p_24[0];
        float ap_8 = a_32 - bp_8;
        float e0_8 = p_24[0] - ap_8 + (sumsf[2 * ((s8 + 4) * 64 + cl) + 1] - bp_8);
        p_24[0] = a_32;
        float b_8 = p_24[1] + e0_8;
        float bq_8 = b_8 - p_24[1];
        float aq_8 = b_8 - bq_8;
        float e1_8 = p_24[1] - aq_8 + (e0_8 - bq_8);
        p_24[1] = b_8;
        p_24[2] = p_24[2] + e1_8;
        float a_33 = p_24[0] + sumsf[2 * ((s8 + 5) * 64 + cl)];
        float bp_9 = a_33 - p_24[0];
        float ap_9 = a_33 - bp_9;
        float e0_9 = p_24[0] - ap_9 + (sumsf[2 * ((s8 + 5) * 64 + cl)] - bp_9);
        p_24[0] = a_33;
        float b_9 = p_24[1] + e0_9;
        float bq_9 = b_9 - p_24[1];
        float aq_9 = b_9 - bq_9;
        float e1_9 = p_24[1] - aq_9 + (e0_9 - bq_9);
        p_24[1] = b_9;
        p_24[2] = p_24[2] + e1_9;
        float a_34 = p_24[0] + sumsf[2 * ((s8 + 5) * 64 + cl) + 1];
        float bp_10 = a_34 - p_24[0];
        float ap_10 = a_34 - bp_10;
        float e0_10 = p_24[0] - ap_10 + (sumsf[2 * ((s8 + 5) * 64 + cl) + 1] - bp_10);
        p_24[0] = a_34;
        float b_10 = p_24[1] + e0_10;
        float bq_10 = b_10 - p_24[1];
        float aq_10 = b_10 - bq_10;
        float e1_10 = p_24[1] - aq_10 + (e0_10 - bq_10);
        p_24[1] = b_10;
        p_24[2] = p_24[2] + e1_10;
        float a_35 = p_24[0] + sumsf[2 * ((s8 + 6) * 64 + cl)];
        float bp_11 = a_35 - p_24[0];
        float ap_11 = a_35 - bp_11;
        float e0_11 = p_24[0] - ap_11 + (sumsf[2 * ((s8 + 6) * 64 + cl)] - bp_11);
        p_24[0] = a_35;
        float b_11 = p_24[1] + e0_11;
        float bq_11 = b_11 - p_24[1];
        float aq_11 = b_11 - bq_11;
        float e1_11 = p_24[1] - aq_11 + (e0_11 - bq_11);
        p_24[1] = b_11;
        p_24[2] = p_24[2] + e1_11;
        float a_36 = p_24[0] + sumsf[2 * ((s8 + 6) * 64 + cl) + 1];
        float bp_12 = a_36 - p_24[0];
        float ap_12 = a_36 - bp_12;
        float e0_12 = p_24[0] - ap_12 + (sumsf[2 * ((s8 + 6) * 64 + cl) + 1] - bp_12);
        p_24[0] = a_36;
        float b_12 = p_24[1] + e0_12;
        float bq_12 = b_12 - p_24[1];
        float aq_12 = b_12 - bq_12;
        float e1_12 = p_24[1] - aq_12 + (e0_12 - bq_12);
        p_24[1] = b_12;
        p_24[2] = p_24[2] + e1_12;
        float a_37 = p_24[0] + sumsf[2 * ((s8 + 7) * 64 + cl)];
        float bp_13 = a_37 - p_24[0];
        float ap_13 = a_37 - bp_13;
        float e0_13 = p_24[0] - ap_13 + (sumsf[2 * ((s8 + 7) * 64 + cl)] - bp_13);
        p_24[0] = a_37;
        float b_13 = p_24[1] + e0_13;
        float bq_13 = b_13 - p_24[1];
        float aq_13 = b_13 - bq_13;
        float e1_13 = p_24[1] - aq_13 + (e0_13 - bq_13);
        p_24[1] = b_13;
        p_24[2] = p_24[2] + e1_13;
        float a_38 = p_24[0] + sumsf[2 * ((s8 + 7) * 64 + cl) + 1];
        float bp_14 = a_38 - p_24[0];
        float ap_14 = a_38 - bp_14;
        float e0_14 = p_24[0] - ap_14 + (sumsf[2 * ((s8 + 7) * 64 + cl) + 1] - bp_14);
        p_24[0] = a_38;
        float b_14 = p_24[1] + e0_14;
        float bq_14 = b_14 - p_24[1];
        float aq_14 = b_14 - bq_14;
        float e1_14 = p_24[1] - aq_14 + (e0_14 - bq_14);
        p_24[1] = b_14;
        p_24[2] = p_24[2] + e1_14;
        sumsf[2 * (s8 * 64 + cl)] = p_24[0];
        sumsf[2 * (s8 * 64 + cl) + 1] = p_24[1];
        sumsf[2 * ((s8 + 1) * 64 + cl)] = p_24[2];
        __syncthreads();
        if (tid < 64) {
            float q[3];
            q[0] = 0.0f;
            q[1] = 0.0f;
            q[2] = 0.0f;
            float a_39 = q[0] + sumsf[2 * cl];
            float bp_15 = a_39 - q[0];
            float ap_15 = a_39 - bp_15;
            float e0_15 = q[0] - ap_15 + (sumsf[2 * cl] - bp_15);
            q[0] = a_39;
            float b_15 = q[1] + e0_15;
            float bq_15 = b_15 - q[1];
            float aq_15 = b_15 - bq_15;
            float e1_15 = q[1] - aq_15 + (e0_15 - bq_15);
            q[1] = b_15;
            q[2] = q[2] + e1_15;
            float a_40 = q[0] + sumsf[2 * cl + 1];
            float bp_16 = a_40 - q[0];
            float ap_16 = a_40 - bp_16;
            float e0_16 = q[0] - ap_16 + (sumsf[2 * cl + 1] - bp_16);
            q[0] = a_40;
            float b_16 = q[1] + e0_16;
            float bq_16 = b_16 - q[1];
            float aq_16 = b_16 - bq_16;
            float e1_16 = q[1] - aq_16 + (e0_16 - bq_16);
            q[1] = b_16;
            q[2] = q[2] + e1_16;
            float a_41 = q[0] + sumsf[2 * (64 + cl)];
            float bp_17 = a_41 - q[0];
            float ap_17 = a_41 - bp_17;
            float e0_17 = q[0] - ap_17 + (sumsf[2 * (64 + cl)] - bp_17);
            q[0] = a_41;
            float b_17 = q[1] + e0_17;
            float bq_17 = b_17 - q[1];
            float aq_17 = b_17 - bq_17;
            float e1_17 = q[1] - aq_17 + (e0_17 - bq_17);
            q[1] = b_17;
            q[2] = q[2] + e1_17;
            float a_42 = q[0] + sumsf[2 * (512 + cl)];
            float bp_18 = a_42 - q[0];
            float ap_18 = a_42 - bp_18;
            float e0_18 = q[0] - ap_18 + (sumsf[2 * (512 + cl)] - bp_18);
            q[0] = a_42;
            float b_18 = q[1] + e0_18;
            float bq_18 = b_18 - q[1];
            float aq_18 = b_18 - bq_18;
            float e1_18 = q[1] - aq_18 + (e0_18 - bq_18);
            q[1] = b_18;
            q[2] = q[2] + e1_18;
            float a_43 = q[0] + sumsf[2 * (512 + cl) + 1];
            float bp_19 = a_43 - q[0];
            float ap_19 = a_43 - bp_19;
            float e0_19 = q[0] - ap_19 + (sumsf[2 * (512 + cl) + 1] - bp_19);
            q[0] = a_43;
            float b_19 = q[1] + e0_19;
            float bq_19 = b_19 - q[1];
            float aq_19 = b_19 - bq_19;
            float e1_19 = q[1] - aq_19 + (e0_19 - bq_19);
            q[1] = b_19;
            q[2] = q[2] + e1_19;
            float a_44 = q[0] + sumsf[2 * (576 + cl)];
            float bp_20 = a_44 - q[0];
            float ap_20 = a_44 - bp_20;
            float e0_20 = q[0] - ap_20 + (sumsf[2 * (576 + cl)] - bp_20);
            q[0] = a_44;
            float b_20 = q[1] + e0_20;
            float bq_20 = b_20 - q[1];
            float aq_20 = b_20 - bq_20;
            float e1_20 = q[1] - aq_20 + (e0_20 - bq_20);
            q[1] = b_20;
            q[2] = q[2] + e1_20;
            float a_45 = q[0] + sumsf[2 * (1024 + cl)];
            float bp_21 = a_45 - q[0];
            float ap_21 = a_45 - bp_21;
            float e0_21 = q[0] - ap_21 + (sumsf[2 * (1024 + cl)] - bp_21);
            q[0] = a_45;
            float b_21 = q[1] + e0_21;
            float bq_21 = b_21 - q[1];
            float aq_21 = b_21 - bq_21;
            float e1_21 = q[1] - aq_21 + (e0_21 - bq_21);
            q[1] = b_21;
            q[2] = q[2] + e1_21;
            float a_46 = q[0] + sumsf[2 * (1024 + cl) + 1];
            float bp_22 = a_46 - q[0];
            float ap_22 = a_46 - bp_22;
            float e0_22 = q[0] - ap_22 + (sumsf[2 * (1024 + cl) + 1] - bp_22);
            q[0] = a_46;
            float b_22 = q[1] + e0_22;
            float bq_22 = b_22 - q[1];
            float aq_22 = b_22 - bq_22;
            float e1_22 = q[1] - aq_22 + (e0_22 - bq_22);
            q[1] = b_22;
            q[2] = q[2] + e1_22;
            float a_47 = q[0] + sumsf[2 * (1088 + cl)];
            float bp_23 = a_47 - q[0];
            float ap_23 = a_47 - bp_23;
            float e0_23 = q[0] - ap_23 + (sumsf[2 * (1088 + cl)] - bp_23);
            q[0] = a_47;
            float b_23 = q[1] + e0_23;
            float bq_23 = b_23 - q[1];
            float aq_23 = b_23 - bq_23;
            float e1_23 = q[1] - aq_23 + (e0_23 - bq_23);
            q[1] = b_23;
            q[2] = q[2] + e1_23;
            float a_48 = q[0] + sumsf[2 * (1536 + cl)];
            float bp_24 = a_48 - q[0];
            float ap_24 = a_48 - bp_24;
            float e0_24 = q[0] - ap_24 + (sumsf[2 * (1536 + cl)] - bp_24);
            q[0] = a_48;
            float b_24 = q[1] + e0_24;
            float bq_24 = b_24 - q[1];
            float aq_24 = b_24 - bq_24;
            float e1_24 = q[1] - aq_24 + (e0_24 - bq_24);
            q[1] = b_24;
            q[2] = q[2] + e1_24;
            float a_49 = q[0] + sumsf[2 * (1536 + cl) + 1];
            float bp_25 = a_49 - q[0];
            float ap_25 = a_49 - bp_25;
            float e0_25 = q[0] - ap_25 + (sumsf[2 * (1536 + cl) + 1] - bp_25);
            q[0] = a_49;
            float b_25 = q[1] + e0_25;
            float bq_25 = b_25 - q[1];
            float aq_25 = b_25 - bq_25;
            float e1_25 = q[1] - aq_25 + (e0_25 - bq_25);
            q[1] = b_25;
            q[2] = q[2] + e1_25;
            float a_50 = q[0] + sumsf[2 * (1600 + cl)];
            float bp_26 = a_50 - q[0];
            float ap_26 = a_50 - bp_26;
            float e0_26 = q[0] - ap_26 + (sumsf[2 * (1600 + cl)] - bp_26);
            q[0] = a_50;
            float b_26 = q[1] + e0_26;
            float bq_26 = b_26 - q[1];
            float aq_26 = b_26 - bq_26;
            float e1_26 = q[1] - aq_26 + (e0_26 - bq_26);
            q[1] = b_26;
            q[2] = q[2] + e1_26;
            float a1 = q[1] + q[2];
            float b1p = a1 - q[1];
            float a1p = a1 - b1p;
            float b1 = q[1] - a1p + (q[2] - b1p);
            float hi = q[0] + a1;
            float b2p = hi - q[0];
            float a2p = hi - b2p;
            float lo = q[0] - a2p + (a1 - b2p);
            float m = lo + b1;
            float b3p = m - lo;
            float a3p = m - b3p;
            float e = lo - a3p + (b1 - b3p);
            unsigned int bits = __as_u32(m);
            if (e != 0.0f) {
                if ((bits & 1) == 0) {
                    bits = bits + 1 - 2 * ((__as_u32(e) ^ bits) >> 31);
                }
            }
            *(reinterpret_cast<float*>(dw) + (tk * 64 + cl)) = hi + __uint_as_float(bits);
        }
        if (tid == 0) {
            flags[5 + (ci & 1)] = (int)nxt;
        }
        __syncthreads();
        tk = flags[5 + (ci & 1)];
    }
    if (tid == 0) {
        pc = (int)nxt - 32;
        first_c = pc;
        if (pc < n_chunks) {
            int stages = (T + 1) / 2;
            int sl = rows_per_chunk;
            if (sl > stages) {
                sl = stages;
            }
            if (sl < 1) {
                sl = 1;
            }
            int share = (stages + sl - 1) / sl;
            int tail = (share + 11) / 12;
            if (tail < 1) {
                tail = 1;
            }
            int rem = stages;
            int size = (share + 1) / 2;
            int first = 0;
            int base = 0;
            int span = 0;
            int halving = 1;
            int begin_it = 0;
            #pragma unroll 1
            for (int k = 0; k < 5; k++) {
                if (halving != 0) {
                    if (size > tail) {
                        if (rem >= sl * size) {
                            if (pc < first + sl) {
                                span = size;
                                begin_it = base + (pc - first) * size;
                                halving = 0;
                            } else {
                                first = first + sl;
                                base = base + sl * size;
                                rem = rem - sl * size;
                                size = (size + 1) / 2;
                            }
                        } else {
                            halving = 0;
                        }
                    } else {
                        halving = 0;
                    }
                }
            }
            if (span == 0) {
                int quot = rem / sl;
                int extra = rem - quot * sl;
                if (pc < first + extra) {
                    span = quot + 1;
                    begin_it = base + (pc - first) * (quot + 1);
                } else if (pc < first + sl) {
                    span = quot;
                    begin_it = base + extra * (quot + 1) + (pc - first - extra) * quot;
                } else {
                    begin_it = stages;
                }
            }
            int _min_84 = ((begin_it * 2) < (T) ? (begin_it * 2) : (T));
            prow = _min_84;
            int _min_85 = ((begin_it * 2 + span * 2) < (T) ? (begin_it * 2 + span * 2) : (T));
            pend = _min_85;
            unsigned int _atomic_fetch_add_86 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_86;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 32;
                if (pc < n_chunks) {
                    int stages_1 = (T + 1) / 2;
                    int sl_1 = rows_per_chunk;
                    if (sl_1 > stages_1) {
                        sl_1 = stages_1;
                    }
                    if (sl_1 < 1) {
                        sl_1 = 1;
                    }
                    int share_1 = (stages_1 + sl_1 - 1) / sl_1;
                    int tail_1 = (share_1 + 11) / 12;
                    if (tail_1 < 1) {
                        tail_1 = 1;
                    }
                    int rem_1 = stages_1;
                    int size_1 = (share_1 + 1) / 2;
                    int first_1 = 0;
                    int base_1 = 0;
                    int span_1 = 0;
                    int halving_1 = 1;
                    int begin_it_1 = 0;
                    #pragma unroll 1
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving_1 != 0) {
                            if (size_1 > tail_1) {
                                if (rem_1 >= sl_1 * size_1) {
                                    if (pc < first_1 + sl_1) {
                                        span_1 = size_1;
                                        begin_it_1 = base_1 + (pc - first_1) * size_1;
                                        halving_1 = 0;
                                    } else {
                                        first_1 = first_1 + sl_1;
                                        base_1 = base_1 + sl_1 * size_1;
                                        rem_1 = rem_1 - sl_1 * size_1;
                                        size_1 = (size_1 + 1) / 2;
                                    }
                                } else {
                                    halving_1 = 0;
                                }
                            } else {
                                halving_1 = 0;
                            }
                        }
                    }
                    if (span_1 == 0) {
                        int quot_1 = rem_1 / sl_1;
                        int extra_1 = rem_1 - quot_1 * sl_1;
                        if (pc < first_1 + extra_1) {
                            span_1 = quot_1 + 1;
                            begin_it_1 = base_1 + (pc - first_1) * (quot_1 + 1);
                        } else if (pc < first_1 + sl_1) {
                            span_1 = quot_1;
                            begin_it_1 = base_1 + extra_1 * (quot_1 + 1) + (pc - first_1 - extra_1) * quot_1;
                        } else {
                            begin_it_1 = stages_1;
                        }
                    }
                    int _min_87 = ((begin_it_1 * 2) < (T) ? (begin_it_1 * 2) : (T));
                    prow = _min_87;
                    int _min_88 = ((begin_it_1 * 2 + span_1 * 2) < (T) ? (begin_it_1 * 2 + span_1 * 2) : (T));
                    pend = _min_88;
                    unsigned int _atomic_fetch_add_89 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_89;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_90 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval = _min_90;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 16384);
            int sbase = ring_at + slot * 16384;
            int _min_91 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_91);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_92 = ((nval - 1) < (1) ? (nval - 1) : (1));
            long long row_i_1 = (long long)(prow + _min_92);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 32;
                if (pc < n_chunks) {
                    int stages_2 = (T + 1) / 2;
                    int sl_2 = rows_per_chunk;
                    if (sl_2 > stages_2) {
                        sl_2 = stages_2;
                    }
                    if (sl_2 < 1) {
                        sl_2 = 1;
                    }
                    int share_2 = (stages_2 + sl_2 - 1) / sl_2;
                    int tail_2 = (share_2 + 11) / 12;
                    if (tail_2 < 1) {
                        tail_2 = 1;
                    }
                    int rem_2 = stages_2;
                    int size_2 = (share_2 + 1) / 2;
                    int first_2 = 0;
                    int base_2 = 0;
                    int span_2 = 0;
                    int halving_2 = 1;
                    int begin_it_2 = 0;
                    #pragma unroll 1
                    for (int k_2 = 0; k_2 < 5; k_2++) {
                        if (halving_2 != 0) {
                            if (size_2 > tail_2) {
                                if (rem_2 >= sl_2 * size_2) {
                                    if (pc < first_2 + sl_2) {
                                        span_2 = size_2;
                                        begin_it_2 = base_2 + (pc - first_2) * size_2;
                                        halving_2 = 0;
                                    } else {
                                        first_2 = first_2 + sl_2;
                                        base_2 = base_2 + sl_2 * size_2;
                                        rem_2 = rem_2 - sl_2 * size_2;
                                        size_2 = (size_2 + 1) / 2;
                                    }
                                } else {
                                    halving_2 = 0;
                                }
                            } else {
                                halving_2 = 0;
                            }
                        }
                    }
                    if (span_2 == 0) {
                        int quot_2 = rem_2 / sl_2;
                        int extra_2 = rem_2 - quot_2 * sl_2;
                        if (pc < first_2 + extra_2) {
                            span_2 = quot_2 + 1;
                            begin_it_2 = base_2 + (pc - first_2) * (quot_2 + 1);
                        } else if (pc < first_2 + sl_2) {
                            span_2 = quot_2;
                            begin_it_2 = base_2 + extra_2 * (quot_2 + 1) + (pc - first_2 - extra_2) * quot_2;
                        } else {
                            begin_it_2 = stages_2;
                        }
                    }
                    int _min_93 = ((begin_it_2 * 2) < (T) ? (begin_it_2 * 2) : (T));
                    prow = _min_93;
                    int _min_94 = ((begin_it_2 * 2 + span_2 * 2) < (T) ? (begin_it_2 * 2 + span_2 * 2) : (T));
                    pend = _min_94;
                    unsigned int _atomic_fetch_add_95 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_95;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_96 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_0 = _min_96;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            int sbase_1 = ring_at + slot_0 * 16384;
            int _min_97 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_97);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_98 = ((nval_0 - 1) < (1) ? (nval_0 - 1) : (1));
            long long row_i_3 = (long long)(prow + _min_98);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 32;
                if (pc < n_chunks) {
                    int stages_3 = (T + 1) / 2;
                    int sl_3 = rows_per_chunk;
                    if (sl_3 > stages_3) {
                        sl_3 = stages_3;
                    }
                    if (sl_3 < 1) {
                        sl_3 = 1;
                    }
                    int share_3 = (stages_3 + sl_3 - 1) / sl_3;
                    int tail_3 = (share_3 + 11) / 12;
                    if (tail_3 < 1) {
                        tail_3 = 1;
                    }
                    int rem_3 = stages_3;
                    int size_3 = (share_3 + 1) / 2;
                    int first_3 = 0;
                    int base_3 = 0;
                    int span_3 = 0;
                    int halving_3 = 1;
                    int begin_it_3 = 0;
                    #pragma unroll 1
                    for (int k_3 = 0; k_3 < 5; k_3++) {
                        if (halving_3 != 0) {
                            if (size_3 > tail_3) {
                                if (rem_3 >= sl_3 * size_3) {
                                    if (pc < first_3 + sl_3) {
                                        span_3 = size_3;
                                        begin_it_3 = base_3 + (pc - first_3) * size_3;
                                        halving_3 = 0;
                                    } else {
                                        first_3 = first_3 + sl_3;
                                        base_3 = base_3 + sl_3 * size_3;
                                        rem_3 = rem_3 - sl_3 * size_3;
                                        size_3 = (size_3 + 1) / 2;
                                    }
                                } else {
                                    halving_3 = 0;
                                }
                            } else {
                                halving_3 = 0;
                            }
                        }
                    }
                    if (span_3 == 0) {
                        int quot_3 = rem_3 / sl_3;
                        int extra_3 = rem_3 - quot_3 * sl_3;
                        if (pc < first_3 + extra_3) {
                            span_3 = quot_3 + 1;
                            begin_it_3 = base_3 + (pc - first_3) * (quot_3 + 1);
                        } else if (pc < first_3 + sl_3) {
                            span_3 = quot_3;
                            begin_it_3 = base_3 + extra_3 * (quot_3 + 1) + (pc - first_3 - extra_3) * quot_3;
                        } else {
                            begin_it_3 = stages_3;
                        }
                    }
                    int _min_99 = ((begin_it_3 * 2) < (T) ? (begin_it_3 * 2) : (T));
                    prow = _min_99;
                    int _min_100 = ((begin_it_3 * 2 + span_3 * 2) < (T) ? (begin_it_3 * 2 + span_3 * 2) : (T));
                    pend = _min_100;
                    unsigned int _atomic_fetch_add_101 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_101;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_102 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_1 = _min_102;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            int sbase_2 = ring_at + slot_1 * 16384;
            int _min_103 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_4 = (long long)(prow + _min_103);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_4 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_4 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_104 = ((nval_1 - 1) < (1) ? (nval_1 - 1) : (1));
            long long row_i_5 = (long long)(prow + _min_104);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_5 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_5 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_1;
        int slot_2 = 3;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 32;
                if (pc < n_chunks) {
                    int stages_4 = (T + 1) / 2;
                    int sl_4 = rows_per_chunk;
                    if (sl_4 > stages_4) {
                        sl_4 = stages_4;
                    }
                    if (sl_4 < 1) {
                        sl_4 = 1;
                    }
                    int share_4 = (stages_4 + sl_4 - 1) / sl_4;
                    int tail_4 = (share_4 + 11) / 12;
                    if (tail_4 < 1) {
                        tail_4 = 1;
                    }
                    int rem_4 = stages_4;
                    int size_4 = (share_4 + 1) / 2;
                    int first_4 = 0;
                    int base_4 = 0;
                    int span_4 = 0;
                    int halving_4 = 1;
                    int begin_it_4 = 0;
                    #pragma unroll 1
                    for (int k_4 = 0; k_4 < 5; k_4++) {
                        if (halving_4 != 0) {
                            if (size_4 > tail_4) {
                                if (rem_4 >= sl_4 * size_4) {
                                    if (pc < first_4 + sl_4) {
                                        span_4 = size_4;
                                        begin_it_4 = base_4 + (pc - first_4) * size_4;
                                        halving_4 = 0;
                                    } else {
                                        first_4 = first_4 + sl_4;
                                        base_4 = base_4 + sl_4 * size_4;
                                        rem_4 = rem_4 - sl_4 * size_4;
                                        size_4 = (size_4 + 1) / 2;
                                    }
                                } else {
                                    halving_4 = 0;
                                }
                            } else {
                                halving_4 = 0;
                            }
                        }
                    }
                    if (span_4 == 0) {
                        int quot_4 = rem_4 / sl_4;
                        int extra_4 = rem_4 - quot_4 * sl_4;
                        if (pc < first_4 + extra_4) {
                            span_4 = quot_4 + 1;
                            begin_it_4 = base_4 + (pc - first_4) * (quot_4 + 1);
                        } else if (pc < first_4 + sl_4) {
                            span_4 = quot_4;
                            begin_it_4 = base_4 + extra_4 * (quot_4 + 1) + (pc - first_4 - extra_4) * quot_4;
                        } else {
                            begin_it_4 = stages_4;
                        }
                    }
                    int _min_105 = ((begin_it_4 * 2) < (T) ? (begin_it_4 * 2) : (T));
                    prow = _min_105;
                    int _min_106 = ((begin_it_4 * 2 + span_4 * 2) < (T) ? (begin_it_4 * 2 + span_4 * 2) : (T));
                    pend = _min_106;
                    unsigned int _atomic_fetch_add_107 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_107;
                }
            }
        }
        int nval_2 = 0;
        if (prow < pend) {
            int _min_108 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_2 = _min_108;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_2;
        if (nval_2 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 24, 16384);
            int sbase_3 = ring_at + slot_2 * 16384;
            int _min_109 = ((nval_2 - 1) < (0) ? (nval_2 - 1) : (0));
            long long row_i_6 = (long long)(prow + _min_109);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_6 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_6 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_110 = ((nval_2 - 1) < (1) ? (nval_2 - 1) : (1));
            long long row_i_7 = (long long)(prow + _min_110);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_7 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_7 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_2;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_3 = meta[1];
    rr[0] = 0.0f;
    if (nval_3 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_5 = 0; k_5 < T + n_chunks; k_5++) {
        if (nval_3 == 0) {
            break;
        }
        int _stage_full = (int)((unsigned int)k_5 % 4);
        int _parity_full = (int)((unsigned int)k_5 / 4 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_5 % 4 * 16384 + rg * 4096;
        __int128_t _smem_raw_111;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_111) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_111)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_111)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_111)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_111)[3];
        __int128_t _smem_raw_112;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_112) : "r"(rbase + 8192 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_112)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_112)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_112)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_112)[3];
        __int128_t _smem_raw_113;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_113) : "r"(rbase + (128 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_113)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_113)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_113)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_113)[3];
        __int128_t _smem_raw_114;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_114) : "r"(rbase + 8192 + (128 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_114)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_114)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_114)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_114)[3];
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        float dq = 0.0f;
        dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * ww[0], __uint_as_float(xw[0] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * ww[1], __uint_as_float(xw[0] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * ww[2], __uint_as_float(xw[1] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * ww[3], __uint_as_float(xw[1] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * ww[4], __uint_as_float(xw[2] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * ww[5], __uint_as_float(xw[2] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * ww[6], __uint_as_float(xw[3] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * ww[7], __uint_as_float(xw[3] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * ww[8], __uint_as_float(xw[4] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * ww[9], __uint_as_float(xw[4] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * ww[10], __uint_as_float(xw[5] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * ww[11], __uint_as_float(xw[5] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * ww[12], __uint_as_float(xw[6] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * ww[13], __uint_as_float(xw[6] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * ww[14], __uint_as_float(xw[7] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * ww[15], __uint_as_float(xw[7] & 4294901760u), dq);
        float _warp_reduce_115 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_115 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_115, offset);
        dot[0] = _warp_reduce_115;
        if (lane == 0) {
            dots[(k_5 & 1) * 8 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_5 & 1) * 8 + rg * 4];
        tot += dots[(k_5 & 1) * 8 + rg * 4 + 1];
        tot += dots[(k_5 & 1) * 8 + rg * 4 + 2];
        tot += dots[(k_5 & 1) * 8 + rg * 4 + 3];
        dot[0] = tot;
        if (tid == 0) {
            int slot_3 = (k_5 + 4) % 4;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt - 32;
                    if (pc < n_chunks) {
                        int stages_5 = (T + 1) / 2;
                        int sl_5 = rows_per_chunk;
                        if (sl_5 > stages_5) {
                            sl_5 = stages_5;
                        }
                        if (sl_5 < 1) {
                            sl_5 = 1;
                        }
                        int share_5 = (stages_5 + sl_5 - 1) / sl_5;
                        int tail_5 = (share_5 + 11) / 12;
                        if (tail_5 < 1) {
                            tail_5 = 1;
                        }
                        int rem_5 = stages_5;
                        int size_5 = (share_5 + 1) / 2;
                        int first_5 = 0;
                        int base_5 = 0;
                        int span_5 = 0;
                        int halving_5 = 1;
                        int begin_it_5 = 0;
                        #pragma unroll 1
                        for (int k_0 = 0; k_0 < 5; k_0++) {
                            if (halving_5 != 0) {
                                if (size_5 > tail_5) {
                                    if (rem_5 >= sl_5 * size_5) {
                                        if (pc < first_5 + sl_5) {
                                            span_5 = size_5;
                                            begin_it_5 = base_5 + (pc - first_5) * size_5;
                                            halving_5 = 0;
                                        } else {
                                            first_5 = first_5 + sl_5;
                                            base_5 = base_5 + sl_5 * size_5;
                                            rem_5 = rem_5 - sl_5 * size_5;
                                            size_5 = (size_5 + 1) / 2;
                                        }
                                    } else {
                                        halving_5 = 0;
                                    }
                                } else {
                                    halving_5 = 0;
                                }
                            }
                        }
                        if (span_5 == 0) {
                            int quot_5 = rem_5 / sl_5;
                            int extra_5 = rem_5 - quot_5 * sl_5;
                            if (pc < first_5 + extra_5) {
                                span_5 = quot_5 + 1;
                                begin_it_5 = base_5 + (pc - first_5) * (quot_5 + 1);
                            } else if (pc < first_5 + sl_5) {
                                span_5 = quot_5;
                                begin_it_5 = base_5 + extra_5 * (quot_5 + 1) + (pc - first_5 - extra_5) * quot_5;
                            } else {
                                begin_it_5 = stages_5;
                            }
                        }
                        int _min_116 = ((begin_it_5 * 2) < (T) ? (begin_it_5 * 2) : (T));
                        prow = _min_116;
                        int _min_117 = ((begin_it_5 * 2 + span_5 * 2) < (T) ? (begin_it_5 * 2 + span_5 * 2) : (T));
                        pend = _min_117;
                        unsigned int _atomic_fetch_add_118 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_118;
                    }
                }
            }
            int nval_3_1 = 0;
            if (prow < pend) {
                int _min_119 = ((pend - prow) < (2) ? (pend - prow) : (2));
                nval_3_1 = _min_119;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_3_1;
            if (nval_3_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8, 16384);
                int sbase_4 = ring_at + slot_3 * 16384;
                int _min_120 = ((nval_3_1 - 1) < (0) ? (nval_3_1 - 1) : (0));
                long long row_i_8 = (long long)(prow + _min_120);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4), "l"(reinterpret_cast<const uint8_t*>(g + row_i_8 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_8 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                int _min_121 = ((nval_3_1 - 1) < (1) ? (nval_3_1 - 1) : (1));
                long long row_i_9 = (long long)(prow + _min_121);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + row_i_9 * g_stride)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_9 * x_stride)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
            }
            prow = prow + nval_3_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 2048.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_3) {
            float out_0[8];
            float g0_0 = __uint_as_float(gw[0] << 16);
            float g1_0 = __uint_as_float(gw[0] & 4294901760u);
            float x0_0 = __uint_as_float(xw[0] << 16);
            float x1_0 = __uint_as_float(xw[0] & 4294901760u);
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * ww[0]));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * ww[1]));
            out_0[0] = y0_0;
            out_0[1] = y1_0;
            float g0_1 = __uint_as_float(gw[1] << 16);
            float g1_1 = __uint_as_float(gw[1] & 4294901760u);
            float x0_1 = __uint_as_float(xw[1] << 16);
            float x1_1 = __uint_as_float(xw[1] & 4294901760u);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * ww[2]));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * ww[3]));
            out_0[2] = y0_1;
            out_0[3] = y1_1;
            float g0_2 = __uint_as_float(gw[2] << 16);
            float g1_2 = __uint_as_float(gw[2] & 4294901760u);
            float x0_2 = __uint_as_float(xw[2] << 16);
            float x1_2 = __uint_as_float(xw[2] & 4294901760u);
            float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * ww[4]));
            float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * ww[5]));
            out_0[4] = y0_2;
            out_0[5] = y1_2;
            float g0_3 = __uint_as_float(gw[3] << 16);
            float g1_3 = __uint_as_float(gw[3] & 4294901760u);
            float x0_3 = __uint_as_float(xw[3] << 16);
            float x1_3 = __uint_as_float(xw[3] & 4294901760u);
            float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * ww[6]));
            float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * ww[7]));
            out_0[6] = y0_3;
            out_0[7] = y1_3;
            unsigned int packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 2048ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            float out_1[8];
            float g0_4 = __uint_as_float(gw[4] << 16);
            float g1_4 = __uint_as_float(gw[4] & 4294901760u);
            float x0_4 = __uint_as_float(xw[4] << 16);
            float x1_4 = __uint_as_float(xw[4] & 4294901760u);
            float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[8]));
            float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[9]));
            out_1[0] = y0_4;
            out_1[1] = y1_4;
            float g0_5 = __uint_as_float(gw[5] << 16);
            float g1_5 = __uint_as_float(gw[5] & 4294901760u);
            float x0_5 = __uint_as_float(xw[5] << 16);
            float x1_5 = __uint_as_float(xw[5] & 4294901760u);
            float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[10]));
            float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[11]));
            out_1[2] = y0_5;
            out_1[3] = y1_5;
            float g0_6 = __uint_as_float(gw[6] << 16);
            float g1_6 = __uint_as_float(gw[6] & 4294901760u);
            float x0_6 = __uint_as_float(xw[6] << 16);
            float x1_6 = __uint_as_float(xw[6] & 4294901760u);
            float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[12]));
            float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[13]));
            out_1[4] = y0_6;
            out_1[5] = y1_6;
            float g0_7 = __uint_as_float(gw[7] << 16);
            float g1_7 = __uint_as_float(gw[7] & 4294901760u);
            float x0_7 = __uint_as_float(xw[7] << 16);
            float x1_7 = __uint_as_float(xw[7] & 4294901760u);
            float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[14]));
            float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[15]));
            out_1[6] = y0_7;
            out_1[7] = y1_7;
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 2048ull + (unsigned long long)((128 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        row0 = meta[2 * ((k_5 + 1) % 4)];
        nval_3 = meta[2 * ((k_5 + 1) % 4) + 1];
        if (nval_3 > rg) {
            rr[0] = r[row0 + rg];
        }
    }
    c = pc;
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
