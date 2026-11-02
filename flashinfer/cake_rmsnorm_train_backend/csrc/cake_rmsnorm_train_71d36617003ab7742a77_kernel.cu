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
#define SMEM_FLAGS_OFF 1024
#define SMEM_FLAGS_STAGE_BYTES 32
#define SMEM_FLAGS_STRIDE 32
#define SMEM_DOTS_OFF 1056
#define SMEM_DOTS_STAGE_BYTES 96
#define SMEM_DOTS_STRIDE 96
#define SMEM_SUMS_OFF 1184
#define SMEM_SUMS_STAGE_BYTES 24576
#define SMEM_SUMS_STRIDE 24576
#define SMEM_DWBUF_OFF 1184
#define SMEM_DWBUF_STAGE_BYTES 24576
#define SMEM_DWBUF_STRIDE 24576
#define SMEM_META_OFF 1152
#define SMEM_META_STAGE_BYTES 24
#define SMEM_META_STRIDE 24
#define SMEM_RING_OFF 25856
#define SMEM_RING_STAGE_BYTES 110592
#define SMEM_RING_STRIDE 110592
#define SMEM_TOTAL 136448
#define THREADS 384
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



extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_71d36617003ab7742a77(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    double* sums = reinterpret_cast<double*>(smem_raw + 1184);
    const int sums_addr = smem + 1184;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1184);
    const int dwbuf_addr = smem + 1184;
    int* meta = reinterpret_cast<int*>(smem_raw + 1152);
    const int meta_addr = smem + 1152;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 25856);
    const int ring_addr = smem + 25856;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
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
    int rg = warp / 12;
    int tir = tid - rg * 384;
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
            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(w + ((unsigned long long)((384 + tir) * 8)))) : "memory");
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
    unsigned int hw[8];
    float rr[2];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        flags[4] = (int)nxt;
    }
    __syncthreads();
    int tk = flags[4];
    #pragma unroll 1
    for (int ci = 0; ci < 96; ci++) {
        if (tk >= 96) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int vl = tid % 8;
        int rl = tid / 8;
        unsigned int gw_0[24];
        unsigned int xw_0[24];
        unsigned int hw_0[1];
        float rr_0[6];
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
        for (int t0 = rl; t0 < T; t0 += 288) {
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
            if (t0 + 48 < T) {
                unsigned long long g_base_1 = (unsigned long long)(t0 + 48) * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)(t0 + 48) * (unsigned long long)x_stride;
                rr_0[1] = r[t0 + 48];
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
            if (t0 + 96 < T) {
                unsigned long long g_base_2 = (unsigned long long)(t0 + 96) * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)(t0 + 96) * (unsigned long long)x_stride;
                rr_0[2] = r[t0 + 96];
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
            if (t0 + 144 < T) {
                unsigned long long g_base_3 = (unsigned long long)(t0 + 144) * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)(t0 + 144) * (unsigned long long)x_stride;
                rr_0[3] = r[t0 + 144];
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
            if (t0 + 192 < T) {
                unsigned long long g_base_4 = (unsigned long long)(t0 + 192) * (unsigned long long)g_stride;
                unsigned long long x_base_4 = (unsigned long long)(t0 + 192) * (unsigned long long)x_stride;
                rr_0[4] = r[t0 + 192];
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
            if (t0 + 240 < T) {
                unsigned long long g_base_5 = (unsigned long long)(t0 + 240) * (unsigned long long)g_stride;
                unsigned long long x_base_5 = (unsigned long long)(t0 + 240) * (unsigned long long)x_stride;
                rr_0[5] = r[t0 + 240];
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
            if (t0 < T) {
                float y = __fmaf_rn(__uint_as_float(gw_0[0] << 16) * __uint_as_float(xw_0[0] << 16), rr_0[0], -comp_0[0]);
                float t = acc_0[0] + y;
                comp_0[0] = t - acc_0[0] - y;
                acc_0[0] = t;
                float y_0 = __fmaf_rn(__uint_as_float(gw_0[0] & 4294901760u) * __uint_as_float(xw_0[0] & 4294901760u), rr_0[0], -comp_0[1]);
                float t_0 = acc_0[1] + y_0;
                comp_0[1] = t_0 - acc_0[1] - y_0;
                acc_0[1] = t_0;
                float y_1 = __fmaf_rn(__uint_as_float(gw_0[1] << 16) * __uint_as_float(xw_0[1] << 16), rr_0[0], -comp_0[2]);
                float t_1 = acc_0[2] + y_1;
                comp_0[2] = t_1 - acc_0[2] - y_1;
                acc_0[2] = t_1;
                float y_2 = __fmaf_rn(__uint_as_float(gw_0[1] & 4294901760u) * __uint_as_float(xw_0[1] & 4294901760u), rr_0[0], -comp_0[3]);
                float t_2 = acc_0[3] + y_2;
                comp_0[3] = t_2 - acc_0[3] - y_2;
                acc_0[3] = t_2;
                float y_3 = __fmaf_rn(__uint_as_float(gw_0[2] << 16) * __uint_as_float(xw_0[2] << 16), rr_0[0], -comp_0[4]);
                float t_3 = acc_0[4] + y_3;
                comp_0[4] = t_3 - acc_0[4] - y_3;
                acc_0[4] = t_3;
                float y_4 = __fmaf_rn(__uint_as_float(gw_0[2] & 4294901760u) * __uint_as_float(xw_0[2] & 4294901760u), rr_0[0], -comp_0[5]);
                float t_4 = acc_0[5] + y_4;
                comp_0[5] = t_4 - acc_0[5] - y_4;
                acc_0[5] = t_4;
                float y_5 = __fmaf_rn(__uint_as_float(gw_0[3] << 16) * __uint_as_float(xw_0[3] << 16), rr_0[0], -comp_0[6]);
                float t_5 = acc_0[6] + y_5;
                comp_0[6] = t_5 - acc_0[6] - y_5;
                acc_0[6] = t_5;
                float y_6 = __fmaf_rn(__uint_as_float(gw_0[3] & 4294901760u) * __uint_as_float(xw_0[3] & 4294901760u), rr_0[0], -comp_0[7]);
                float t_6 = acc_0[7] + y_6;
                comp_0[7] = t_6 - acc_0[7] - y_6;
                acc_0[7] = t_6;
            }
            if (t0 + 48 < T) {
                float y_7 = __fmaf_rn(__uint_as_float(gw_0[4] << 16) * __uint_as_float(xw_0[4] << 16), rr_0[1], -comp_0[0]);
                float t_7 = acc_0[0] + y_7;
                comp_0[0] = t_7 - acc_0[0] - y_7;
                acc_0[0] = t_7;
                float y_7_1 = __fmaf_rn(__uint_as_float(gw_0[4] & 4294901760u) * __uint_as_float(xw_0[4] & 4294901760u), rr_0[1], -comp_0[1]);
                float t_7_1 = acc_0[1] + y_7_1;
                comp_0[1] = t_7_1 - acc_0[1] - y_7_1;
                acc_0[1] = t_7_1;
                float y_8 = __fmaf_rn(__uint_as_float(gw_0[5] << 16) * __uint_as_float(xw_0[5] << 16), rr_0[1], -comp_0[2]);
                float t_8 = acc_0[2] + y_8;
                comp_0[2] = t_8 - acc_0[2] - y_8;
                acc_0[2] = t_8;
                float y_9 = __fmaf_rn(__uint_as_float(gw_0[5] & 4294901760u) * __uint_as_float(xw_0[5] & 4294901760u), rr_0[1], -comp_0[3]);
                float t_9 = acc_0[3] + y_9;
                comp_0[3] = t_9 - acc_0[3] - y_9;
                acc_0[3] = t_9;
                float y_10 = __fmaf_rn(__uint_as_float(gw_0[6] << 16) * __uint_as_float(xw_0[6] << 16), rr_0[1], -comp_0[4]);
                float t_10 = acc_0[4] + y_10;
                comp_0[4] = t_10 - acc_0[4] - y_10;
                acc_0[4] = t_10;
                float y_11 = __fmaf_rn(__uint_as_float(gw_0[6] & 4294901760u) * __uint_as_float(xw_0[6] & 4294901760u), rr_0[1], -comp_0[5]);
                float t_11 = acc_0[5] + y_11;
                comp_0[5] = t_11 - acc_0[5] - y_11;
                acc_0[5] = t_11;
                float y_12 = __fmaf_rn(__uint_as_float(gw_0[7] << 16) * __uint_as_float(xw_0[7] << 16), rr_0[1], -comp_0[6]);
                float t_12 = acc_0[6] + y_12;
                comp_0[6] = t_12 - acc_0[6] - y_12;
                acc_0[6] = t_12;
                float y_13 = __fmaf_rn(__uint_as_float(gw_0[7] & 4294901760u) * __uint_as_float(xw_0[7] & 4294901760u), rr_0[1], -comp_0[7]);
                float t_13 = acc_0[7] + y_13;
                comp_0[7] = t_13 - acc_0[7] - y_13;
                acc_0[7] = t_13;
            }
            if (t0 + 96 < T) {
                float y_14 = __fmaf_rn(__uint_as_float(gw_0[8] << 16) * __uint_as_float(xw_0[8] << 16), rr_0[2], -comp_0[0]);
                float t_14 = acc_0[0] + y_14;
                comp_0[0] = t_14 - acc_0[0] - y_14;
                acc_0[0] = t_14;
                float y_14_1 = __fmaf_rn(__uint_as_float(gw_0[8] & 4294901760u) * __uint_as_float(xw_0[8] & 4294901760u), rr_0[2], -comp_0[1]);
                float t_14_1 = acc_0[1] + y_14_1;
                comp_0[1] = t_14_1 - acc_0[1] - y_14_1;
                acc_0[1] = t_14_1;
                float y_15 = __fmaf_rn(__uint_as_float(gw_0[9] << 16) * __uint_as_float(xw_0[9] << 16), rr_0[2], -comp_0[2]);
                float t_15 = acc_0[2] + y_15;
                comp_0[2] = t_15 - acc_0[2] - y_15;
                acc_0[2] = t_15;
                float y_16 = __fmaf_rn(__uint_as_float(gw_0[9] & 4294901760u) * __uint_as_float(xw_0[9] & 4294901760u), rr_0[2], -comp_0[3]);
                float t_16 = acc_0[3] + y_16;
                comp_0[3] = t_16 - acc_0[3] - y_16;
                acc_0[3] = t_16;
                float y_17 = __fmaf_rn(__uint_as_float(gw_0[10] << 16) * __uint_as_float(xw_0[10] << 16), rr_0[2], -comp_0[4]);
                float t_17 = acc_0[4] + y_17;
                comp_0[4] = t_17 - acc_0[4] - y_17;
                acc_0[4] = t_17;
                float y_18 = __fmaf_rn(__uint_as_float(gw_0[10] & 4294901760u) * __uint_as_float(xw_0[10] & 4294901760u), rr_0[2], -comp_0[5]);
                float t_18 = acc_0[5] + y_18;
                comp_0[5] = t_18 - acc_0[5] - y_18;
                acc_0[5] = t_18;
                float y_19 = __fmaf_rn(__uint_as_float(gw_0[11] << 16) * __uint_as_float(xw_0[11] << 16), rr_0[2], -comp_0[6]);
                float t_19 = acc_0[6] + y_19;
                comp_0[6] = t_19 - acc_0[6] - y_19;
                acc_0[6] = t_19;
                float y_20 = __fmaf_rn(__uint_as_float(gw_0[11] & 4294901760u) * __uint_as_float(xw_0[11] & 4294901760u), rr_0[2], -comp_0[7]);
                float t_20 = acc_0[7] + y_20;
                comp_0[7] = t_20 - acc_0[7] - y_20;
                acc_0[7] = t_20;
            }
            if (t0 + 144 < T) {
                float y_21 = __fmaf_rn(__uint_as_float(gw_0[12] << 16) * __uint_as_float(xw_0[12] << 16), rr_0[3], -comp_0[0]);
                float t_21 = acc_0[0] + y_21;
                comp_0[0] = t_21 - acc_0[0] - y_21;
                acc_0[0] = t_21;
                float y_21_1 = __fmaf_rn(__uint_as_float(gw_0[12] & 4294901760u) * __uint_as_float(xw_0[12] & 4294901760u), rr_0[3], -comp_0[1]);
                float t_21_1 = acc_0[1] + y_21_1;
                comp_0[1] = t_21_1 - acc_0[1] - y_21_1;
                acc_0[1] = t_21_1;
                float y_22 = __fmaf_rn(__uint_as_float(gw_0[13] << 16) * __uint_as_float(xw_0[13] << 16), rr_0[3], -comp_0[2]);
                float t_22 = acc_0[2] + y_22;
                comp_0[2] = t_22 - acc_0[2] - y_22;
                acc_0[2] = t_22;
                float y_23 = __fmaf_rn(__uint_as_float(gw_0[13] & 4294901760u) * __uint_as_float(xw_0[13] & 4294901760u), rr_0[3], -comp_0[3]);
                float t_23 = acc_0[3] + y_23;
                comp_0[3] = t_23 - acc_0[3] - y_23;
                acc_0[3] = t_23;
                float y_24 = __fmaf_rn(__uint_as_float(gw_0[14] << 16) * __uint_as_float(xw_0[14] << 16), rr_0[3], -comp_0[4]);
                float t_24 = acc_0[4] + y_24;
                comp_0[4] = t_24 - acc_0[4] - y_24;
                acc_0[4] = t_24;
                float y_25 = __fmaf_rn(__uint_as_float(gw_0[14] & 4294901760u) * __uint_as_float(xw_0[14] & 4294901760u), rr_0[3], -comp_0[5]);
                float t_25 = acc_0[5] + y_25;
                comp_0[5] = t_25 - acc_0[5] - y_25;
                acc_0[5] = t_25;
                float y_26 = __fmaf_rn(__uint_as_float(gw_0[15] << 16) * __uint_as_float(xw_0[15] << 16), rr_0[3], -comp_0[6]);
                float t_26 = acc_0[6] + y_26;
                comp_0[6] = t_26 - acc_0[6] - y_26;
                acc_0[6] = t_26;
                float y_27 = __fmaf_rn(__uint_as_float(gw_0[15] & 4294901760u) * __uint_as_float(xw_0[15] & 4294901760u), rr_0[3], -comp_0[7]);
                float t_27 = acc_0[7] + y_27;
                comp_0[7] = t_27 - acc_0[7] - y_27;
                acc_0[7] = t_27;
            }
            if (t0 + 192 < T) {
                float y_28 = __fmaf_rn(__uint_as_float(gw_0[16] << 16) * __uint_as_float(xw_0[16] << 16), rr_0[4], -comp_0[0]);
                float t_28 = acc_0[0] + y_28;
                comp_0[0] = t_28 - acc_0[0] - y_28;
                acc_0[0] = t_28;
                float y_28_1 = __fmaf_rn(__uint_as_float(gw_0[16] & 4294901760u) * __uint_as_float(xw_0[16] & 4294901760u), rr_0[4], -comp_0[1]);
                float t_28_1 = acc_0[1] + y_28_1;
                comp_0[1] = t_28_1 - acc_0[1] - y_28_1;
                acc_0[1] = t_28_1;
                float y_29 = __fmaf_rn(__uint_as_float(gw_0[17] << 16) * __uint_as_float(xw_0[17] << 16), rr_0[4], -comp_0[2]);
                float t_29 = acc_0[2] + y_29;
                comp_0[2] = t_29 - acc_0[2] - y_29;
                acc_0[2] = t_29;
                float y_30 = __fmaf_rn(__uint_as_float(gw_0[17] & 4294901760u) * __uint_as_float(xw_0[17] & 4294901760u), rr_0[4], -comp_0[3]);
                float t_30 = acc_0[3] + y_30;
                comp_0[3] = t_30 - acc_0[3] - y_30;
                acc_0[3] = t_30;
                float y_31 = __fmaf_rn(__uint_as_float(gw_0[18] << 16) * __uint_as_float(xw_0[18] << 16), rr_0[4], -comp_0[4]);
                float t_31 = acc_0[4] + y_31;
                comp_0[4] = t_31 - acc_0[4] - y_31;
                acc_0[4] = t_31;
                float y_32 = __fmaf_rn(__uint_as_float(gw_0[18] & 4294901760u) * __uint_as_float(xw_0[18] & 4294901760u), rr_0[4], -comp_0[5]);
                float t_32 = acc_0[5] + y_32;
                comp_0[5] = t_32 - acc_0[5] - y_32;
                acc_0[5] = t_32;
                float y_33 = __fmaf_rn(__uint_as_float(gw_0[19] << 16) * __uint_as_float(xw_0[19] << 16), rr_0[4], -comp_0[6]);
                float t_33 = acc_0[6] + y_33;
                comp_0[6] = t_33 - acc_0[6] - y_33;
                acc_0[6] = t_33;
                float y_34 = __fmaf_rn(__uint_as_float(gw_0[19] & 4294901760u) * __uint_as_float(xw_0[19] & 4294901760u), rr_0[4], -comp_0[7]);
                float t_34 = acc_0[7] + y_34;
                comp_0[7] = t_34 - acc_0[7] - y_34;
                acc_0[7] = t_34;
            }
            if (t0 + 240 < T) {
                float y_35 = __fmaf_rn(__uint_as_float(gw_0[20] << 16) * __uint_as_float(xw_0[20] << 16), rr_0[5], -comp_0[0]);
                float t_35 = acc_0[0] + y_35;
                comp_0[0] = t_35 - acc_0[0] - y_35;
                acc_0[0] = t_35;
                float y_35_1 = __fmaf_rn(__uint_as_float(gw_0[20] & 4294901760u) * __uint_as_float(xw_0[20] & 4294901760u), rr_0[5], -comp_0[1]);
                float t_35_1 = acc_0[1] + y_35_1;
                comp_0[1] = t_35_1 - acc_0[1] - y_35_1;
                acc_0[1] = t_35_1;
                float y_36 = __fmaf_rn(__uint_as_float(gw_0[21] << 16) * __uint_as_float(xw_0[21] << 16), rr_0[5], -comp_0[2]);
                float t_36 = acc_0[2] + y_36;
                comp_0[2] = t_36 - acc_0[2] - y_36;
                acc_0[2] = t_36;
                float y_37 = __fmaf_rn(__uint_as_float(gw_0[21] & 4294901760u) * __uint_as_float(xw_0[21] & 4294901760u), rr_0[5], -comp_0[3]);
                float t_37 = acc_0[3] + y_37;
                comp_0[3] = t_37 - acc_0[3] - y_37;
                acc_0[3] = t_37;
                float y_38 = __fmaf_rn(__uint_as_float(gw_0[22] << 16) * __uint_as_float(xw_0[22] << 16), rr_0[5], -comp_0[4]);
                float t_38 = acc_0[4] + y_38;
                comp_0[4] = t_38 - acc_0[4] - y_38;
                acc_0[4] = t_38;
                float y_39 = __fmaf_rn(__uint_as_float(gw_0[22] & 4294901760u) * __uint_as_float(xw_0[22] & 4294901760u), rr_0[5], -comp_0[5]);
                float t_39 = acc_0[5] + y_39;
                comp_0[5] = t_39 - acc_0[5] - y_39;
                acc_0[5] = t_39;
                float y_40 = __fmaf_rn(__uint_as_float(gw_0[23] << 16) * __uint_as_float(xw_0[23] << 16), rr_0[5], -comp_0[6]);
                float t_40 = acc_0[6] + y_40;
                comp_0[6] = t_40 - acc_0[6] - y_40;
                acc_0[6] = t_40;
                float y_41 = __fmaf_rn(__uint_as_float(gw_0[23] & 4294901760u) * __uint_as_float(xw_0[23] & 4294901760u), rr_0[5], -comp_0[7]);
                float t_41 = acc_0[7] + y_41;
                comp_0[7] = t_41 - acc_0[7] - y_41;
                acc_0[7] = t_41;
            }
        }
        sums[rl * 64 + vl * 8] = (double)acc_0[0] - (double)comp_0[0];
        sums[rl * 64 + vl * 8 + 1] = (double)acc_0[1] - (double)comp_0[1];
        sums[rl * 64 + vl * 8 + 2] = (double)acc_0[2] - (double)comp_0[2];
        sums[rl * 64 + vl * 8 + 3] = (double)acc_0[3] - (double)comp_0[3];
        sums[rl * 64 + vl * 8 + 4] = (double)acc_0[4] - (double)comp_0[4];
        sums[rl * 64 + vl * 8 + 5] = (double)acc_0[5] - (double)comp_0[5];
        sums[rl * 64 + vl * 8 + 6] = (double)acc_0[6] - (double)comp_0[6];
        sums[rl * 64 + vl * 8 + 7] = (double)acc_0[7] - (double)comp_0[7];
        __syncthreads();
        int cl = tid % 64;
        int s8 = tid / 64 * 8;
        double part = sums[s8 * 64 + cl];
        part = part + sums[(s8 + 1) * 64 + cl];
        part = part + sums[(s8 + 1 + 1) * 64 + cl];
        part = part + sums[(s8 + 2 + 1) * 64 + cl];
        part = part + sums[(s8 + 3 + 1) * 64 + cl];
        part = part + sums[(s8 + 4 + 1) * 64 + cl];
        part = part + sums[(s8 + 5 + 1) * 64 + cl];
        part = part + sums[(s8 + 6 + 1) * 64 + cl];
        sums[s8 * 64 + cl] = part;
        __syncthreads();
        if (tid < 64) {
            double tot = sums[cl];
            tot = tot + sums[512 + cl];
            tot = tot + sums[1024 + cl];
            tot = tot + sums[1536 + cl];
            tot = tot + sums[2048 + cl];
            tot = tot + sums[2560 + cl];
            *(reinterpret_cast<float*>(dw) + (tk * 64 + cl)) = (float)tot;
        }
        if (tid == 0) {
            flags[5 + (ci & 1)] = (int)nxt;
        }
        __syncthreads();
        tk = flags[5 + (ci & 1)];
    }
    if (tid == 0) {
        pc = (int)nxt - 96;
        first_c = pc;
        if (pc < n_chunks) {
            int stages = T / 1;
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
            int _min_16 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_16;
            int _min_17 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_17;
            unsigned int _atomic_fetch_add_18 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_18;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
                    int stages_1 = T / 1;
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
                    int _min_19 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_19;
                    int _min_20 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_20;
                    unsigned int _atomic_fetch_add_21 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_21;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_22 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_22;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 36864);
            int sbase = ring_at + slot * 36864;
            int _min_23 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_23);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_0 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_0 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_0 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
                    int stages_2 = T / 1;
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
                    int _min_24 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_24;
                    int _min_25 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_25;
                    unsigned int _atomic_fetch_add_26 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_26;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_27 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_27;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 36864);
            int sbase_1 = ring_at + slot_0 * 36864;
            int _min_28 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_28);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_1 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_1 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_1 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 96;
                if (pc < n_chunks) {
                    int stages_3 = T / 1;
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
                    int _min_29 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_29;
                    int _min_30 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_30;
                    unsigned int _atomic_fetch_add_31 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_31;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_32 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_32;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 36864);
            int sbase_2 = ring_at + slot_1 * 36864;
            int _min_33 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_33);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_2 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_2 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_2 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_1;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_2 = meta[1];
    rr[0] = 0.0f;
    if (nval_2 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_4 = 0; k_4 < T + n_chunks; k_4++) {
        if (nval_2 == 0) {
            break;
        }
        row0_n = meta[2 * ((k_4 + 1) % 3)];
        nval_n = meta[2 * ((k_4 + 1) % 3) + 1];
        if (nval_n > rg) {
            rr[1] = r[row0_n + rg];
        }
        int _stage_full = (int)((unsigned int)k_4 % 3);
        int _parity_full = (int)((unsigned int)k_4 / 3 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_4 % 3 * 36864 + rg * 12288;
        __int128_t _smem_raw_34;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_34) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_34)[3];
        __int128_t _smem_raw_35;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_35) : "r"(rbase + 12288 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_35)[3];
        __int128_t _smem_raw_36;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_36) : "r"(rbase + 24576 + tir * 8 * 2));
        hw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[0];
        hw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[1];
        hw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[2];
        hw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_36)[3];
        __int128_t _smem_raw_37;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_37) : "r"(rbase + (384 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_37)[3];
        __int128_t _smem_raw_38;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_38) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[3];
        __int128_t _smem_raw_39;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_39) : "r"(rbase + 24576 + (384 + tir) * 8 * 2));
        hw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[0];
        hw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[1];
        hw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[2];
        hw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[3];
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
        float _warp_reduce_40 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_40 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_40, offset);
        dot[0] = _warp_reduce_40;
        if (lane == 0) {
            dots[(k_4 & 1) * 12 + warp] = dot[0];
        }
        __syncthreads();
        float tot_1 = 0.0f;
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 1];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 2];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 3];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 4];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 5];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 6];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 7];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 8];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 9];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 10];
        tot_1 += dots[(k_4 & 1) * 12 + rg * 12 + 11];
        dot[0] = tot_1;
        if (tid == 0) {
            int slot_2 = (k_4 + 3) % 3;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt - 96;
                    if (pc < n_chunks) {
                        int stages_4 = T / 1;
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
                        for (int k_0 = 0; k_0 < 5; k_0++) {
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
                        int _min_41 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                        prow = _min_41;
                        int _min_42 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                        pend = _min_42;
                        unsigned int _atomic_fetch_add_43 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_43;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_44 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_2_1 = _min_44;
            }
            meta[2 * slot_2] = prow;
            meta[2 * slot_2 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 36864);
                int sbase_3 = ring_at + slot_2 * 36864;
                int _min_45 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_3 = (long long)(prow + _min_45);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_3 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_3 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_3 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 6144.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_2) {
            float out_0[8];
            float g0_0 = __uint_as_float(gw[0] << 16);
            float g1_0 = __uint_as_float(gw[0] & 4294901760u);
            float x0_0 = __uint_as_float(xw[0] << 16);
            float x1_0 = __uint_as_float(xw[0] & 4294901760u);
            float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * ww[0]));
            float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * ww[1]));
            out_0[0] = y0_0 + __uint_as_float(hw[0] << 16);
            out_0[1] = y1_0 + __uint_as_float(hw[0] & 4294901760u);
            float g0_1 = __uint_as_float(gw[1] << 16);
            float g1_1 = __uint_as_float(gw[1] & 4294901760u);
            float x0_1 = __uint_as_float(xw[1] << 16);
            float x1_1 = __uint_as_float(xw[1] & 4294901760u);
            float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * ww[2]));
            float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * ww[3]));
            out_0[2] = y0_1 + __uint_as_float(hw[1] << 16);
            out_0[3] = y1_1 + __uint_as_float(hw[1] & 4294901760u);
            float g0_2 = __uint_as_float(gw[2] << 16);
            float g1_2 = __uint_as_float(gw[2] & 4294901760u);
            float x0_2 = __uint_as_float(xw[2] << 16);
            float x1_2 = __uint_as_float(xw[2] & 4294901760u);
            float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * ww[4]));
            float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * ww[5]));
            out_0[4] = y0_2 + __uint_as_float(hw[2] << 16);
            out_0[5] = y1_2 + __uint_as_float(hw[2] & 4294901760u);
            float g0_3 = __uint_as_float(gw[3] << 16);
            float g1_3 = __uint_as_float(gw[3] & 4294901760u);
            float x0_3 = __uint_as_float(xw[3] << 16);
            float x1_3 = __uint_as_float(xw[3] & 4294901760u);
            float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * ww[6]));
            float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * ww[7]));
            out_0[6] = y0_3 + __uint_as_float(hw[3] << 16);
            out_0[7] = y1_3 + __uint_as_float(hw[3] & 4294901760u);
            unsigned int packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144 + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            float out_1[8];
            float g0_4 = __uint_as_float(gw[4] << 16);
            float g1_4 = __uint_as_float(gw[4] & 4294901760u);
            float x0_4 = __uint_as_float(xw[4] << 16);
            float x1_4 = __uint_as_float(xw[4] & 4294901760u);
            float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * ww[8]));
            float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * ww[9]));
            out_1[0] = y0_4 + __uint_as_float(hw[4] << 16);
            out_1[1] = y1_4 + __uint_as_float(hw[4] & 4294901760u);
            float g0_5 = __uint_as_float(gw[5] << 16);
            float g1_5 = __uint_as_float(gw[5] & 4294901760u);
            float x0_5 = __uint_as_float(xw[5] << 16);
            float x1_5 = __uint_as_float(xw[5] & 4294901760u);
            float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * ww[10]));
            float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * ww[11]));
            out_1[2] = y0_5 + __uint_as_float(hw[5] << 16);
            out_1[3] = y1_5 + __uint_as_float(hw[5] & 4294901760u);
            float g0_6 = __uint_as_float(gw[6] << 16);
            float g1_6 = __uint_as_float(gw[6] & 4294901760u);
            float x0_6 = __uint_as_float(xw[6] << 16);
            float x1_6 = __uint_as_float(xw[6] & 4294901760u);
            float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * ww[12]));
            float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * ww[13]));
            out_1[4] = y0_6 + __uint_as_float(hw[6] << 16);
            out_1[5] = y1_6 + __uint_as_float(hw[6] & 4294901760u);
            float g0_7 = __uint_as_float(gw[7] << 16);
            float g1_7 = __uint_as_float(gw[7] & 4294901760u);
            float x0_7 = __uint_as_float(xw[7] << 16);
            float x1_7 = __uint_as_float(xw[7] & 4294901760u);
            float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * ww[14]));
            float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * ww[15]));
            out_1[6] = y0_7 + __uint_as_float(hw[7] << 16);
            out_1[7] = y1_7 + __uint_as_float(hw[7] & 4294901760u);
            unsigned int packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144 + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        row0 = row0_n;
        nval_2 = nval_n;
        rr[0] = rr[1];
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
