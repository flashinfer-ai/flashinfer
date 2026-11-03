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
#define SMEM_DOTS_STAGE_BYTES 64
#define SMEM_DOTS_STRIDE 64
#define SMEM_SUMS_OFF 1152
#define SMEM_SUMS_STAGE_BYTES 16384
#define SMEM_SUMS_STRIDE 16384
#define SMEM_DWBUF_OFF 1152
#define SMEM_DWBUF_STAGE_BYTES 8192
#define SMEM_DWBUF_STRIDE 8192
#define SMEM_META_OFF 1120
#define SMEM_META_STAGE_BYTES 32
#define SMEM_META_STRIDE 32
#define SMEM_RING_OFF 17536
#define SMEM_RING_STAGE_BYTES 65536
#define SMEM_RING_STRIDE 65536
#define SMEM_TOTAL 83072
#define THREADS 256
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



extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_17493054fed5409b6df9(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    double* sums = reinterpret_cast<double*>(smem_raw + 1152);
    const int sums_addr = smem + 1152;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1152);
    const int dwbuf_addr = smem + 1152;
    int* meta = reinterpret_cast<int*>(smem_raw + 1120);
    const int meta_addr = smem + 1120;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 17536);
    const int ring_addr = smem + 17536;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

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
    for (int ci = 0; ci < 64; ci++) {
        if (tk >= 64) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_3;
        }
        int vl = tid % 4;
        int rl = tid / 4;
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
        for (int t0 = rl; t0 < T; t0 += 512) {
            if (t0 < T) {
                unsigned long long g_base = (unsigned long long)t0 * (unsigned long long)g_stride;
                unsigned long long x_base = (unsigned long long)t0 * (unsigned long long)x_stride;
                rr_0[0] = r[t0];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(g + g_base + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_4 = reinterpret_cast<uint4*>(&gw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_4[_blk].x), "=r"(_vdst_4[_blk].y), "=r"(_vdst_4[_blk].z), "=r"(_vdst_4[_blk].w) : "l"((const void*)(_vptr_4 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(x + x_base + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_5 = reinterpret_cast<uint4*>(&xw_0[0]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_5[_blk].x), "=r"(_vdst_5[_blk].y), "=r"(_vdst_5[_blk].z), "=r"(_vdst_5[_blk].w) : "l"((const void*)(_vptr_5 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 64 < T) {
                unsigned long long g_base_1 = (unsigned long long)(t0 + 64) * (unsigned long long)g_stride;
                unsigned long long x_base_1 = (unsigned long long)(t0 + 64) * (unsigned long long)x_stride;
                rr_0[1] = r[t0 + 64];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(g + g_base_1 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_6 = reinterpret_cast<uint4*>(&gw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_6[_blk].x), "=r"(_vdst_6[_blk].y), "=r"(_vdst_6[_blk].z), "=r"(_vdst_6[_blk].w) : "l"((const void*)(_vptr_6 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(x + x_base_1 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_7 = reinterpret_cast<uint4*>(&xw_0[4]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_7[_blk].x), "=r"(_vdst_7[_blk].y), "=r"(_vdst_7[_blk].z), "=r"(_vdst_7[_blk].w) : "l"((const void*)(_vptr_7 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 128 < T) {
                unsigned long long g_base_2 = (unsigned long long)(t0 + 128) * (unsigned long long)g_stride;
                unsigned long long x_base_2 = (unsigned long long)(t0 + 128) * (unsigned long long)x_stride;
                rr_0[2] = r[t0 + 128];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(g + g_base_2 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_8 = reinterpret_cast<uint4*>(&gw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_8[_blk].x), "=r"(_vdst_8[_blk].y), "=r"(_vdst_8[_blk].z), "=r"(_vdst_8[_blk].w) : "l"((const void*)(_vptr_8 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(x + x_base_2 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_9 = reinterpret_cast<uint4*>(&xw_0[8]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_9[_blk].x), "=r"(_vdst_9[_blk].y), "=r"(_vdst_9[_blk].z), "=r"(_vdst_9[_blk].w) : "l"((const void*)(_vptr_9 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 192 < T) {
                unsigned long long g_base_3 = (unsigned long long)(t0 + 192) * (unsigned long long)g_stride;
                unsigned long long x_base_3 = (unsigned long long)(t0 + 192) * (unsigned long long)x_stride;
                rr_0[3] = r[t0 + 192];
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(g + g_base_3 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_10 = reinterpret_cast<uint4*>(&gw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_10[_blk].x), "=r"(_vdst_10[_blk].y), "=r"(_vdst_10[_blk].z), "=r"(_vdst_10[_blk].w) : "l"((const void*)(_vptr_10 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_11 = reinterpret_cast<const uint4*>(x + x_base_3 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_11 = reinterpret_cast<uint4*>(&xw_0[12]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_11[_blk].x), "=r"(_vdst_11[_blk].y), "=r"(_vdst_11[_blk].z), "=r"(_vdst_11[_blk].w) : "l"((const void*)(_vptr_11 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 256 < T) {
                unsigned long long g_base_4 = (unsigned long long)(t0 + 256) * (unsigned long long)g_stride;
                unsigned long long x_base_4 = (unsigned long long)(t0 + 256) * (unsigned long long)x_stride;
                rr_0[4] = r[t0 + 256];
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(g + g_base_4 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_12 = reinterpret_cast<uint4*>(&gw_0[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_12[_blk].x), "=r"(_vdst_12[_blk].y), "=r"(_vdst_12[_blk].z), "=r"(_vdst_12[_blk].w) : "l"((const void*)(_vptr_12 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(x + x_base_4 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_13 = reinterpret_cast<uint4*>(&xw_0[16]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_13[_blk].x), "=r"(_vdst_13[_blk].y), "=r"(_vdst_13[_blk].z), "=r"(_vdst_13[_blk].w) : "l"((const void*)(_vptr_13 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 320 < T) {
                unsigned long long g_base_5 = (unsigned long long)(t0 + 320) * (unsigned long long)g_stride;
                unsigned long long x_base_5 = (unsigned long long)(t0 + 320) * (unsigned long long)x_stride;
                rr_0[5] = r[t0 + 320];
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(g + g_base_5 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_14 = reinterpret_cast<uint4*>(&gw_0[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_14[_blk].x), "=r"(_vdst_14[_blk].y), "=r"(_vdst_14[_blk].z), "=r"(_vdst_14[_blk].w) : "l"((const void*)(_vptr_14 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(x + x_base_5 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_15 = reinterpret_cast<uint4*>(&xw_0[20]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_15[_blk].x), "=r"(_vdst_15[_blk].y), "=r"(_vdst_15[_blk].z), "=r"(_vdst_15[_blk].w) : "l"((const void*)(_vptr_15 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 384 < T) {
                unsigned long long g_base_6 = (unsigned long long)(t0 + 384) * (unsigned long long)g_stride;
                unsigned long long x_base_6 = (unsigned long long)(t0 + 384) * (unsigned long long)x_stride;
                rr_0[6] = r[t0 + 384];
                {
                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(g + g_base_6 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_16 = reinterpret_cast<uint4*>(&gw_0[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_16[_blk].x), "=r"(_vdst_16[_blk].y), "=r"(_vdst_16[_blk].z), "=r"(_vdst_16[_blk].w) : "l"((const void*)(_vptr_16 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(x + x_base_6 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_17 = reinterpret_cast<uint4*>(&xw_0[24]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_17[_blk].x), "=r"(_vdst_17[_blk].y), "=r"(_vdst_17[_blk].z), "=r"(_vdst_17[_blk].w) : "l"((const void*)(_vptr_17 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
            if (t0 + 448 < T) {
                unsigned long long g_base_7 = (unsigned long long)(t0 + 448) * (unsigned long long)g_stride;
                unsigned long long x_base_7 = (unsigned long long)(t0 + 448) * (unsigned long long)x_stride;
                rr_0[7] = r[t0 + 448];
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(g + g_base_7 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_18 = reinterpret_cast<uint4*>(&gw_0[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_18[_blk].x), "=r"(_vdst_18[_blk].y), "=r"(_vdst_18[_blk].z), "=r"(_vdst_18[_blk].w) : "l"((const void*)(_vptr_18 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(x + x_base_7 + (unsigned long long)((tk * 4 + vl) * 8));
                    uint4* _vdst_19 = reinterpret_cast<uint4*>(&xw_0[28]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                            : "=r"(_vdst_19[_blk].x), "=r"(_vdst_19[_blk].y), "=r"(_vdst_19[_blk].z), "=r"(_vdst_19[_blk].w) : "l"((const void*)(_vptr_19 + _blk)), "l"(0x14F0000000000000ULL) : "memory");
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
            if (t0 + 64 < T) {
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
            if (t0 + 128 < T) {
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
            if (t0 + 192 < T) {
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
            if (t0 + 256 < T) {
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
            if (t0 + 320 < T) {
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
            if (t0 + 384 < T) {
                float y_42 = __fmaf_rn(__uint_as_float(gw_0[24] << 16) * __uint_as_float(xw_0[24] << 16), rr_0[6], -comp_0[0]);
                float t_42 = acc_0[0] + y_42;
                comp_0[0] = t_42 - acc_0[0] - y_42;
                acc_0[0] = t_42;
                float y_42_1 = __fmaf_rn(__uint_as_float(gw_0[24] & 4294901760u) * __uint_as_float(xw_0[24] & 4294901760u), rr_0[6], -comp_0[1]);
                float t_42_1 = acc_0[1] + y_42_1;
                comp_0[1] = t_42_1 - acc_0[1] - y_42_1;
                acc_0[1] = t_42_1;
                float y_43 = __fmaf_rn(__uint_as_float(gw_0[25] << 16) * __uint_as_float(xw_0[25] << 16), rr_0[6], -comp_0[2]);
                float t_43 = acc_0[2] + y_43;
                comp_0[2] = t_43 - acc_0[2] - y_43;
                acc_0[2] = t_43;
                float y_44 = __fmaf_rn(__uint_as_float(gw_0[25] & 4294901760u) * __uint_as_float(xw_0[25] & 4294901760u), rr_0[6], -comp_0[3]);
                float t_44 = acc_0[3] + y_44;
                comp_0[3] = t_44 - acc_0[3] - y_44;
                acc_0[3] = t_44;
                float y_45 = __fmaf_rn(__uint_as_float(gw_0[26] << 16) * __uint_as_float(xw_0[26] << 16), rr_0[6], -comp_0[4]);
                float t_45 = acc_0[4] + y_45;
                comp_0[4] = t_45 - acc_0[4] - y_45;
                acc_0[4] = t_45;
                float y_46 = __fmaf_rn(__uint_as_float(gw_0[26] & 4294901760u) * __uint_as_float(xw_0[26] & 4294901760u), rr_0[6], -comp_0[5]);
                float t_46 = acc_0[5] + y_46;
                comp_0[5] = t_46 - acc_0[5] - y_46;
                acc_0[5] = t_46;
                float y_47 = __fmaf_rn(__uint_as_float(gw_0[27] << 16) * __uint_as_float(xw_0[27] << 16), rr_0[6], -comp_0[6]);
                float t_47 = acc_0[6] + y_47;
                comp_0[6] = t_47 - acc_0[6] - y_47;
                acc_0[6] = t_47;
                float y_48 = __fmaf_rn(__uint_as_float(gw_0[27] & 4294901760u) * __uint_as_float(xw_0[27] & 4294901760u), rr_0[6], -comp_0[7]);
                float t_48 = acc_0[7] + y_48;
                comp_0[7] = t_48 - acc_0[7] - y_48;
                acc_0[7] = t_48;
            }
            if (t0 + 448 < T) {
                float y_49 = __fmaf_rn(__uint_as_float(gw_0[28] << 16) * __uint_as_float(xw_0[28] << 16), rr_0[7], -comp_0[0]);
                float t_49 = acc_0[0] + y_49;
                comp_0[0] = t_49 - acc_0[0] - y_49;
                acc_0[0] = t_49;
                float y_49_1 = __fmaf_rn(__uint_as_float(gw_0[28] & 4294901760u) * __uint_as_float(xw_0[28] & 4294901760u), rr_0[7], -comp_0[1]);
                float t_49_1 = acc_0[1] + y_49_1;
                comp_0[1] = t_49_1 - acc_0[1] - y_49_1;
                acc_0[1] = t_49_1;
                float y_50 = __fmaf_rn(__uint_as_float(gw_0[29] << 16) * __uint_as_float(xw_0[29] << 16), rr_0[7], -comp_0[2]);
                float t_50 = acc_0[2] + y_50;
                comp_0[2] = t_50 - acc_0[2] - y_50;
                acc_0[2] = t_50;
                float y_51 = __fmaf_rn(__uint_as_float(gw_0[29] & 4294901760u) * __uint_as_float(xw_0[29] & 4294901760u), rr_0[7], -comp_0[3]);
                float t_51 = acc_0[3] + y_51;
                comp_0[3] = t_51 - acc_0[3] - y_51;
                acc_0[3] = t_51;
                float y_52 = __fmaf_rn(__uint_as_float(gw_0[30] << 16) * __uint_as_float(xw_0[30] << 16), rr_0[7], -comp_0[4]);
                float t_52 = acc_0[4] + y_52;
                comp_0[4] = t_52 - acc_0[4] - y_52;
                acc_0[4] = t_52;
                float y_53 = __fmaf_rn(__uint_as_float(gw_0[30] & 4294901760u) * __uint_as_float(xw_0[30] & 4294901760u), rr_0[7], -comp_0[5]);
                float t_53 = acc_0[5] + y_53;
                comp_0[5] = t_53 - acc_0[5] - y_53;
                acc_0[5] = t_53;
                float y_54 = __fmaf_rn(__uint_as_float(gw_0[31] << 16) * __uint_as_float(xw_0[31] << 16), rr_0[7], -comp_0[6]);
                float t_54 = acc_0[6] + y_54;
                comp_0[6] = t_54 - acc_0[6] - y_54;
                acc_0[6] = t_54;
                float y_55 = __fmaf_rn(__uint_as_float(gw_0[31] & 4294901760u) * __uint_as_float(xw_0[31] & 4294901760u), rr_0[7], -comp_0[7]);
                float t_55 = acc_0[7] + y_55;
                comp_0[7] = t_55 - acc_0[7] - y_55;
                acc_0[7] = t_55;
            }
        }
        sums[rl * 32 + vl * 8] = (double)acc_0[0] - (double)comp_0[0];
        sums[rl * 32 + vl * 8 + 1] = (double)acc_0[1] - (double)comp_0[1];
        sums[rl * 32 + vl * 8 + 2] = (double)acc_0[2] - (double)comp_0[2];
        sums[rl * 32 + vl * 8 + 3] = (double)acc_0[3] - (double)comp_0[3];
        sums[rl * 32 + vl * 8 + 4] = (double)acc_0[4] - (double)comp_0[4];
        sums[rl * 32 + vl * 8 + 5] = (double)acc_0[5] - (double)comp_0[5];
        sums[rl * 32 + vl * 8 + 6] = (double)acc_0[6] - (double)comp_0[6];
        sums[rl * 32 + vl * 8 + 7] = (double)acc_0[7] - (double)comp_0[7];
        __syncthreads();
        int cl = tid % 32;
        int s8 = tid / 32 * 8;
        double part = sums[s8 * 32 + cl];
        part = part + sums[(s8 + 1) * 32 + cl];
        part = part + sums[(s8 + 1 + 1) * 32 + cl];
        part = part + sums[(s8 + 2 + 1) * 32 + cl];
        part = part + sums[(s8 + 3 + 1) * 32 + cl];
        part = part + sums[(s8 + 4 + 1) * 32 + cl];
        part = part + sums[(s8 + 5 + 1) * 32 + cl];
        part = part + sums[(s8 + 6 + 1) * 32 + cl];
        sums[s8 * 32 + cl] = part;
        __syncthreads();
        if (tid < 32) {
            double tot = sums[cl];
            tot = tot + sums[256 + cl];
            tot = tot + sums[512 + cl];
            tot = tot + sums[768 + cl];
            tot = tot + sums[1024 + cl];
            tot = tot + sums[1280 + cl];
            tot = tot + sums[1536 + cl];
            tot = tot + sums[1792 + cl];
            *(reinterpret_cast<float*>(dw) + (tk * 32 + cl)) = (float)tot;
        }
        if (tid == 0) {
            flags[5 + (ci & 1)] = (int)nxt;
        }
        __syncthreads();
        tk = flags[5 + (ci & 1)];
    }
    if (tid == 0) {
        pc = (int)nxt - 64;
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
            int _min_20 = ((begin_it * 2) < (T) ? (begin_it * 2) : (T));
            prow = _min_20;
            int _min_21 = ((begin_it * 2 + span * 2) < (T) ? (begin_it * 2 + span * 2) : (T));
            pend = _min_21;
            unsigned int _atomic_fetch_add_22 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_22;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 64;
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
                    int _min_23 = ((begin_it_1 * 2) < (T) ? (begin_it_1 * 2) : (T));
                    prow = _min_23;
                    int _min_24 = ((begin_it_1 * 2 + span_1 * 2) < (T) ? (begin_it_1 * 2 + span_1 * 2) : (T));
                    pend = _min_24;
                    unsigned int _atomic_fetch_add_25 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_25;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_26 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval = _min_26;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 16384);
            int sbase = ring_at + slot * 16384;
            int _min_27 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_27);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_0 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 8192), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_0 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_28 = ((nval - 1) < (1) ? (nval - 1) : (1));
            long long row_i_1 = (long long)(prow + _min_28);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 4096), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_1 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_1 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 64;
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
                    int _min_29 = ((begin_it_2 * 2) < (T) ? (begin_it_2 * 2) : (T));
                    prow = _min_29;
                    int _min_30 = ((begin_it_2 * 2 + span_2 * 2) < (T) ? (begin_it_2 * 2 + span_2 * 2) : (T));
                    pend = _min_30;
                    unsigned int _atomic_fetch_add_31 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_31;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_32 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_0 = _min_32;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            int sbase_1 = ring_at + slot_0 * 16384;
            int _min_33 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_33);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_2 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_2 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_34 = ((nval_0 - 1) < (1) ? (nval_0 - 1) : (1));
            long long row_i_3 = (long long)(prow + _min_34);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_3 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_3 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 64;
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
                    int _min_35 = ((begin_it_3 * 2) < (T) ? (begin_it_3 * 2) : (T));
                    prow = _min_35;
                    int _min_36 = ((begin_it_3 * 2 + span_3 * 2) < (T) ? (begin_it_3 * 2 + span_3 * 2) : (T));
                    pend = _min_36;
                    unsigned int _atomic_fetch_add_37 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_37;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_38 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_1 = _min_38;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            int sbase_2 = ring_at + slot_1 * 16384;
            int _min_39 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_4 = (long long)(prow + _min_39);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_4 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_4 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_40 = ((nval_1 - 1) < (1) ? (nval_1 - 1) : (1));
            long long row_i_5 = (long long)(prow + _min_40);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_5 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_5 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_1;
        int slot_2 = 3;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt - 64;
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
                    int _min_41 = ((begin_it_4 * 2) < (T) ? (begin_it_4 * 2) : (T));
                    prow = _min_41;
                    int _min_42 = ((begin_it_4 * 2 + span_4 * 2) < (T) ? (begin_it_4 * 2 + span_4 * 2) : (T));
                    pend = _min_42;
                    unsigned int _atomic_fetch_add_43 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_43;
                }
            }
        }
        int nval_2 = 0;
        if (prow < pend) {
            int _min_44 = ((pend - prow) < (2) ? (pend - prow) : (2));
            nval_2 = _min_44;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_2;
        if (nval_2 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 24, 16384);
            int sbase_3 = ring_at + slot_2 * 16384;
            int _min_45 = ((nval_2 - 1) < (0) ? (nval_2 - 1) : (0));
            long long row_i_6 = (long long)(prow + _min_45);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_6 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_6 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            int _min_46 = ((nval_2 - 1) < (1) ? (nval_2 - 1) : (1));
            long long row_i_7 = (long long)(prow + _min_46);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_7 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + 24),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_7 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + 24),
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
        __int128_t _smem_raw_47;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_47) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_47)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_47)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_47)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_47)[3];
        __int128_t _smem_raw_48;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_48) : "r"(rbase + 8192 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_48)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_48)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_48)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_48)[3];
        __int128_t _smem_raw_49;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_49) : "r"(rbase + (128 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_49)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_49)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_49)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_49)[3];
        __int128_t _smem_raw_50;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_50) : "r"(rbase + 8192 + (128 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_50)[3];
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
        float _warp_reduce_51 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_51 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_51, offset);
        dot[0] = _warp_reduce_51;
        if (lane == 0) {
            dots[(k_5 & 1) * 8 + warp] = dot[0];
        }
        __syncthreads();
        float tot_1 = 0.0f;
        tot_1 += dots[(k_5 & 1) * 8 + rg * 4];
        tot_1 += dots[(k_5 & 1) * 8 + rg * 4 + 1];
        tot_1 += dots[(k_5 & 1) * 8 + rg * 4 + 2];
        tot_1 += dots[(k_5 & 1) * 8 + rg * 4 + 3];
        dot[0] = tot_1;
        if (tid == 0) {
            int slot_3 = (k_5 + 4) % 4;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt - 64;
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
                        int _min_52 = ((begin_it_5 * 2) < (T) ? (begin_it_5 * 2) : (T));
                        prow = _min_52;
                        int _min_53 = ((begin_it_5 * 2 + span_5 * 2) < (T) ? (begin_it_5 * 2 + span_5 * 2) : (T));
                        pend = _min_53;
                        unsigned int _atomic_fetch_add_54 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_54;
                    }
                }
            }
            int nval_3_1 = 0;
            if (prow < pend) {
                int _min_55 = ((pend - prow) < (2) ? (pend - prow) : (2));
                nval_3_1 = _min_55;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_3_1;
            if (nval_3_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8, 16384);
                int sbase_4 = ring_at + slot_3 * 16384;
                int _min_56 = ((nval_3_1 - 1) < (0) ? (nval_3_1 - 1) : (0));
                long long row_i_8 = (long long)(prow + _min_56);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_8 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_8 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                int _min_57 = ((nval_3_1 - 1) < (1) ? (nval_3_1 - 1) : (1));
                long long row_i_9 = (long long)(prow + _min_57);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 4096), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_9 * g_stride))), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_9 * x_stride))), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 2048 + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 2048 + (unsigned long long)((128 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
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
