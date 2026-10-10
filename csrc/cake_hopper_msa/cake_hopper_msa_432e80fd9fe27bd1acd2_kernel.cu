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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_Q_S_OFF 1024
#define SMEM_Q_S_STAGE_BYTES 24576
#define SMEM_Q_S_STRIDE 24576
#define SMEM_K_S_OFF 25600
#define SMEM_K_S_STAGE_BYTES 16384
#define SMEM_K_S_STRIDE 16384
#define SMEM_TOTAL 91136
#define THREADS 416
#define TRACE 0
#define TRACE_WG 0
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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(416, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_432e80fd9fe27bd1acd2(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, float* __restrict__ out, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int has_qoff, int pt_stride, int max_k_tiles, int total_q, int num_heads, int nsplit, unsigned long long* __restrict__ trace)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* q_s = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_s_addr = smem + 1024;
    uint8_t* k_s = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int k_s_addr = smem + 25600;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=12
            mbarrier_init(smem + 32, 12);
            mbarrier_init(smem + 40, 12);
            mbarrier_init(smem + 48, 12);
            mbarrier_init(smem + 56, 12);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 12) {
        { // producer_main
            if (warp == 12) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
                int bx = blockIdx.x;
                int split = blockIdx.y;
                int b = blockIdx.z;
                int mt = gridDim.x / num_heads - 1 - bx / num_heads;
                {
                    split = bx / num_heads;
                    mt = gridDim.y - 1 - blockIdx.y;
                }
                int pt_base = b * pt_stride + split;
                int pg_first = page_table[pt_base];
                int h = bx % num_heads;
                int qlo = cu_seqlens_q[b];
                int sqb = cu_seqlens_q[b + 1] - qlo;
                int sk = seqused_k[b];
                int pfx = sk - sqb;
                if (has_qoff != 0) {
                    pfx = q_offset[b];
                }
                int m0 = mt * 192;
                int mlim = sqb - 192;
                if (m0 > mlim) {
                    m0 = mlim;
                }
                if (m0 < 0) {
                    m0 = 0;
                }
                int nb = (sk + 127) / 128;
                int qmax = m0 + 191;
                if (qmax > sqb - 1) {
                    qmax = sqb - 1;
                }
                qmax = qmax + pfx;
                int t_lim = qmax / 128 + 1;
                if (t_lim > nb) {
                    t_lim = nb;
                }
                if (t_lim > max_k_tiles) {
                    t_lim = max_k_tiles;
                }
                int n_comp = (t_lim - split + nsplit - 1) / nsplit;
                if (t_lim <= split) {
                    n_comp = 0;
                }
                if (elect_sync()) {
                    if (n_comp > 0) {
                        int pg_next = pg_first;
                        #pragma unroll 1
                        for (int i = 0; i < n_comp; i++) {
                            int stage = i % 4;
                            int pphase = i / 4 + 1 & 1;
                            int page = pg_next;
                            if (n_comp > i + 1) {
                                pg_next = page_table[pt_base + (i + 1) * nsplit];
                            }
                            mbarrier_wait(empty_addr + (stage) * 8, pphase);
                            if (i == 0) {
                                mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 40960);
                                tma_3d_gmem2smem(q_s_addr, (&Q), 0, h, qlo + m0, full_addr + (stage) * 8);
                            }
                            if (i != 0) {
                                mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 16384);
                            }
                            tma_2d_gmem2smem(k_s_addr + (unsigned int)(stage * 16384), (&K), 0, page * 128, full_addr + (stage) * 8);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: math ----
    if (warp <= 11) {
        { // math_main
            unsigned long long stamps[6];
            stamps[0] = 0;
            stamps[1] = 0;
            stamps[2] = 0;
            stamps[3] = 0;
            stamps[4] = 0;
            stamps[5] = 0;
            int bx2 = blockIdx.x;
            int split2 = blockIdx.y;
            int b2 = blockIdx.z;
            int mt2 = gridDim.x / num_heads - 1 - bx2 / num_heads;
            {
                split2 = bx2 / num_heads;
                mt2 = gridDim.y - 1 - blockIdx.y;
            }
            int h2 = bx2 % num_heads;
            int qlo2 = cu_seqlens_q[b2];
            int sqb2 = cu_seqlens_q[b2 + 1] - qlo2;
            int sk2 = seqused_k[b2];
            int pfx2 = sk2 - sqb2;
            if (has_qoff != 0) {
                pfx2 = q_offset[b2];
            }
            int m02 = mt2 * 192;
            int mlim2 = sqb2 - 192;
            if (m02 > mlim2) {
                m02 = mlim2;
            }
            if (m02 < 0) {
                m02 = 0;
            }
            int nb2 = (sk2 + 127) / 128;
            int qmax2 = m02 + 191;
            if (qmax2 > sqb2 - 1) {
                qmax2 = sqb2 - 1;
            }
            qmax2 = qmax2 + pfx2;
            int t_lim2 = qmax2 / 128 + 1;
            if (t_lim2 > nb2) {
                t_lim2 = nb2;
            }
            if (t_lim2 > max_k_tiles) {
                t_lim2 = max_k_tiles;
            }
            int n_comp2 = (t_lim2 - split2 + nsplit - 1) / nsplit;
            if (t_lim2 <= split2) {
                n_comp2 = 0;
            }
            int n_dead = (max_k_tiles - t_lim2 - split2 + nsplit - 1) / nsplit;
            if (split2 >= max_k_tiles - t_lim2) {
                n_dead = 0;
            }
            int tid_1 = threadIdx.x;
            int lane_0 = lane;
            int wgi = warp / 4;
            int wrp = warp - wgi * 4;
            int g = lane_0 / 4;
            int cq = lane_0 - g * 4;
            int r0 = wgi * 64 + wrp * 16 + g;
            int rows_valid = sqb2 - m02;
            int obase = h2 * max_k_tiles * total_q + qlo2 + m02;
            int sk1 = sk2 - 1;
            int lim0 = pfx2 + m02 + r0;
            if (lim0 > sk1) {
                lim0 = sk1;
            }
            int lim1 = pfx2 + m02 + r0 + 8;
            if (lim1 > sk1) {
                lim1 = sk1;
            }
            int mask_thr = pfx2 + m02;
            if (mask_thr > sk1) {
                mask_thr = sk1;
            }
            float neg = -CAKE_INF;
            unsigned int acc[32];
            unsigned int acc1[32];
            uint64_t _wgmma_desc_0 = (((uint64_t)(((q_s_addr + (unsigned int)(wgi * 64 * 128))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
            uint64_t _wgmma_desc_1 = (((uint64_t)(((k_s_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
            if (n_comp2 > 0) {
                mbarrier_wait(full_addr, 0);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
                    : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                    : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                    : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                    : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                    : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                    : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                    : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                    : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                int npair = (n_comp2 - 1) / 2;
                #pragma unroll 1
                for (int jp = 0; jp < npair; jp++) {
                    int ia = 2 * jp;
                    int ib = ia + 1;
                    int ic = ia + 2;
                    int sa = ia % 4;
                    int sb = ib % 4;
                    int sc = ic % 4;
                    mbarrier_wait(full_addr + (sb) * 8, ib / 4 & 1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_2 = (((uint64_t)(((k_s_addr + (unsigned int)(sb * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_2 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_2 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_2 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(empty_addr + (sa) * 8);
                    }
                    int k0 = (split2 + ia * nsplit) * 128;
                    float m0v = neg;
                    float m1v = neg;
                    int need_mask = 0;
                    if (mask_thr < k0 + 127) {
                        need_mask = 1;
                    }
                    if (need_mask == 0) {
                        unsigned int w0 = acc[0];
                        unsigned int w1 = acc[1];
                        uint32_t _f16x2_max_0;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_0) : "r"(w0), "r"(acc[2]));
                        w0 = _f16x2_max_0;
                        uint32_t _f16x2_max_1;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_1) : "r"(w1), "r"(acc[3]));
                        w1 = _f16x2_max_1;
                        uint32_t _f16x2_max_2;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_2) : "r"(w0), "r"(acc[4]));
                        w0 = _f16x2_max_2;
                        uint32_t _f16x2_max_3;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_3) : "r"(w1), "r"(acc[5]));
                        w1 = _f16x2_max_3;
                        uint32_t _f16x2_max_4;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_4) : "r"(w0), "r"(acc[6]));
                        w0 = _f16x2_max_4;
                        uint32_t _f16x2_max_5;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_5) : "r"(w1), "r"(acc[7]));
                        w1 = _f16x2_max_5;
                        uint32_t _f16x2_max_6;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_6) : "r"(w0), "r"(acc[8]));
                        w0 = _f16x2_max_6;
                        uint32_t _f16x2_max_7;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_7) : "r"(w1), "r"(acc[9]));
                        w1 = _f16x2_max_7;
                        uint32_t _f16x2_max_8;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_8) : "r"(w0), "r"(acc[10]));
                        w0 = _f16x2_max_8;
                        uint32_t _f16x2_max_9;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_9) : "r"(w1), "r"(acc[11]));
                        w1 = _f16x2_max_9;
                        uint32_t _f16x2_max_10;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_10) : "r"(w0), "r"(acc[12]));
                        w0 = _f16x2_max_10;
                        uint32_t _f16x2_max_11;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_11) : "r"(w1), "r"(acc[13]));
                        w1 = _f16x2_max_11;
                        uint32_t _f16x2_max_12;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_12) : "r"(w0), "r"(acc[14]));
                        w0 = _f16x2_max_12;
                        uint32_t _f16x2_max_13;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_13) : "r"(w1), "r"(acc[15]));
                        w1 = _f16x2_max_13;
                        uint32_t _f16x2_max_14;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_14) : "r"(w0), "r"(acc[16]));
                        w0 = _f16x2_max_14;
                        uint32_t _f16x2_max_15;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_15) : "r"(w1), "r"(acc[17]));
                        w1 = _f16x2_max_15;
                        uint32_t _f16x2_max_16;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_16) : "r"(w0), "r"(acc[18]));
                        w0 = _f16x2_max_16;
                        uint32_t _f16x2_max_17;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_17) : "r"(w1), "r"(acc[19]));
                        w1 = _f16x2_max_17;
                        uint32_t _f16x2_max_18;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_18) : "r"(w0), "r"(acc[20]));
                        w0 = _f16x2_max_18;
                        uint32_t _f16x2_max_19;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_19) : "r"(w1), "r"(acc[21]));
                        w1 = _f16x2_max_19;
                        uint32_t _f16x2_max_20;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_20) : "r"(w0), "r"(acc[22]));
                        w0 = _f16x2_max_20;
                        uint32_t _f16x2_max_21;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_21) : "r"(w1), "r"(acc[23]));
                        w1 = _f16x2_max_21;
                        uint32_t _f16x2_max_22;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_22) : "r"(w0), "r"(acc[24]));
                        w0 = _f16x2_max_22;
                        uint32_t _f16x2_max_23;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_23) : "r"(w1), "r"(acc[25]));
                        w1 = _f16x2_max_23;
                        uint32_t _f16x2_max_24;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_24) : "r"(w0), "r"(acc[26]));
                        w0 = _f16x2_max_24;
                        uint32_t _f16x2_max_25;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_25) : "r"(w1), "r"(acc[27]));
                        w1 = _f16x2_max_25;
                        uint32_t _f16x2_max_26;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_26) : "r"(w0), "r"(acc[28]));
                        w0 = _f16x2_max_26;
                        uint32_t _f16x2_max_27;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_27) : "r"(w1), "r"(acc[29]));
                        w1 = _f16x2_max_27;
                        uint32_t _f16x2_max_28;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_28) : "r"(w0), "r"(acc[30]));
                        w0 = _f16x2_max_28;
                        uint32_t _f16x2_max_29;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_29) : "r"(w1), "r"(acc[31]));
                        w1 = _f16x2_max_29;
                        uint16_t _f16_max_0;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_0) : "h"((uint16_t)(w0 & 65535)), "h"((uint16_t)(w0 >> 16)));
                        float _cvt_f32_f16_0;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_f16_max_0)));
                        m0v = _cvt_f32_f16_0;
                        uint16_t _f16_max_1;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_1) : "h"((uint16_t)(w1 & 65535)), "h"((uint16_t)(w1 >> 16)));
                        float _cvt_f32_f16_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_f16_max_1)));
                        m1v = _cvt_f32_f16_1;
                    }
                    if (need_mask != 0) {
                        int c0 = lim0 - k0 - 2 * cq;
                        int c1 = lim1 - k0 - 2 * cq;
                        float v = neg;
                        {
                            if (c0 >= 0) {
                                {
                                    float _cvt_f32_f16_2;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(acc[0] & 65535)));
                                    v = _cvt_f32_f16_2;
                                }
                            }
                            float _max_0 = max_noftz(m0v, v);
                            m0v = _max_0;
                        }
                        float v_0 = neg;
                        {
                            if (c0 >= 1) {
                                {
                                    float _cvt_f32_f16_7;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(acc[0] >> 16)));
                                    v_0 = _cvt_f32_f16_7;
                                }
                            }
                            float _max_2 = max_noftz(m0v, v_0);
                            m0v = _max_2;
                        }
                        float v_1 = neg;
                        {
                            if (c1 >= 0) {
                                {
                                    float _cvt_f32_f16_12;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(acc[1] & 65535)));
                                    v_1 = _cvt_f32_f16_12;
                                }
                            }
                            float _max_5 = max_noftz(m1v, v_1);
                            m1v = _max_5;
                        }
                        float v_2 = neg;
                        {
                            if (c1 >= 1) {
                                {
                                    float _cvt_f32_f16_17;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(acc[1] >> 16)));
                                    v_2 = _cvt_f32_f16_17;
                                }
                            }
                            float _max_7 = max_noftz(m1v, v_2);
                            m1v = _max_7;
                        }
                        float v_3 = neg;
                        {
                            if (c0 >= 8) {
                                {
                                    float _cvt_f32_f16_18;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(acc[2] & 65535)));
                                    v_3 = _cvt_f32_f16_18;
                                }
                            }
                            float _max_8 = max_noftz(m0v, v_3);
                            m0v = _max_8;
                        }
                        float v_4 = neg;
                        {
                            if (c0 >= 9) {
                                {
                                    float _cvt_f32_f16_23;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(acc[2] >> 16)));
                                    v_4 = _cvt_f32_f16_23;
                                }
                            }
                            float _max_10 = max_noftz(m0v, v_4);
                            m0v = _max_10;
                        }
                        float v_5 = neg;
                        {
                            if (c1 >= 8) {
                                {
                                    float _cvt_f32_f16_28;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(acc[3] & 65535)));
                                    v_5 = _cvt_f32_f16_28;
                                }
                            }
                            float _max_13 = max_noftz(m1v, v_5);
                            m1v = _max_13;
                        }
                        float v_6 = neg;
                        {
                            if (c1 >= 9) {
                                {
                                    float _cvt_f32_f16_33;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(acc[3] >> 16)));
                                    v_6 = _cvt_f32_f16_33;
                                }
                            }
                            float _max_15 = max_noftz(m1v, v_6);
                            m1v = _max_15;
                        }
                        float v_7 = neg;
                        {
                            if (c0 >= 16) {
                                {
                                    float _cvt_f32_f16_34;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(acc[4] & 65535)));
                                    v_7 = _cvt_f32_f16_34;
                                }
                            }
                            float _max_16 = max_noftz(m0v, v_7);
                            m0v = _max_16;
                        }
                        float v_8 = neg;
                        {
                            if (c0 >= 17) {
                                {
                                    float _cvt_f32_f16_39;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_39) : "h"((uint16_t)(acc[4] >> 16)));
                                    v_8 = _cvt_f32_f16_39;
                                }
                            }
                            float _max_18 = max_noftz(m0v, v_8);
                            m0v = _max_18;
                        }
                        float v_9 = neg;
                        {
                            if (c1 >= 16) {
                                {
                                    float _cvt_f32_f16_44;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_44) : "h"((uint16_t)(acc[5] & 65535)));
                                    v_9 = _cvt_f32_f16_44;
                                }
                            }
                            float _max_21 = max_noftz(m1v, v_9);
                            m1v = _max_21;
                        }
                        float v_10 = neg;
                        {
                            if (c1 >= 17) {
                                {
                                    float _cvt_f32_f16_49;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_49) : "h"((uint16_t)(acc[5] >> 16)));
                                    v_10 = _cvt_f32_f16_49;
                                }
                            }
                            float _max_23 = max_noftz(m1v, v_10);
                            m1v = _max_23;
                        }
                        float v_11 = neg;
                        {
                            if (c0 >= 24) {
                                {
                                    float _cvt_f32_f16_50;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_50) : "h"((uint16_t)(acc[6] & 65535)));
                                    v_11 = _cvt_f32_f16_50;
                                }
                            }
                            float _max_24 = max_noftz(m0v, v_11);
                            m0v = _max_24;
                        }
                        float v_12 = neg;
                        {
                            if (c0 >= 25) {
                                {
                                    float _cvt_f32_f16_55;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_55) : "h"((uint16_t)(acc[6] >> 16)));
                                    v_12 = _cvt_f32_f16_55;
                                }
                            }
                            float _max_26 = max_noftz(m0v, v_12);
                            m0v = _max_26;
                        }
                        float v_13 = neg;
                        {
                            if (c1 >= 24) {
                                {
                                    float _cvt_f32_f16_60;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_60) : "h"((uint16_t)(acc[7] & 65535)));
                                    v_13 = _cvt_f32_f16_60;
                                }
                            }
                            float _max_29 = max_noftz(m1v, v_13);
                            m1v = _max_29;
                        }
                        float v_14 = neg;
                        {
                            if (c1 >= 25) {
                                {
                                    float _cvt_f32_f16_65;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_65) : "h"((uint16_t)(acc[7] >> 16)));
                                    v_14 = _cvt_f32_f16_65;
                                }
                            }
                            float _max_31 = max_noftz(m1v, v_14);
                            m1v = _max_31;
                        }
                        float v_15 = neg;
                        {
                            if (c0 >= 32) {
                                {
                                    float _cvt_f32_f16_66;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_66) : "h"((uint16_t)(acc[8] & 65535)));
                                    v_15 = _cvt_f32_f16_66;
                                }
                            }
                            float _max_32 = max_noftz(m0v, v_15);
                            m0v = _max_32;
                        }
                        float v_16 = neg;
                        {
                            if (c0 >= 33) {
                                {
                                    float _cvt_f32_f16_71;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_71) : "h"((uint16_t)(acc[8] >> 16)));
                                    v_16 = _cvt_f32_f16_71;
                                }
                            }
                            float _max_34 = max_noftz(m0v, v_16);
                            m0v = _max_34;
                        }
                        float v_17 = neg;
                        {
                            if (c1 >= 32) {
                                {
                                    float _cvt_f32_f16_76;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_76) : "h"((uint16_t)(acc[9] & 65535)));
                                    v_17 = _cvt_f32_f16_76;
                                }
                            }
                            float _max_37 = max_noftz(m1v, v_17);
                            m1v = _max_37;
                        }
                        float v_18 = neg;
                        {
                            if (c1 >= 33) {
                                {
                                    float _cvt_f32_f16_81;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_81) : "h"((uint16_t)(acc[9] >> 16)));
                                    v_18 = _cvt_f32_f16_81;
                                }
                            }
                            float _max_39 = max_noftz(m1v, v_18);
                            m1v = _max_39;
                        }
                        float v_19 = neg;
                        {
                            if (c0 >= 40) {
                                {
                                    float _cvt_f32_f16_82;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_82) : "h"((uint16_t)(acc[10] & 65535)));
                                    v_19 = _cvt_f32_f16_82;
                                }
                            }
                            float _max_40 = max_noftz(m0v, v_19);
                            m0v = _max_40;
                        }
                        float v_20 = neg;
                        {
                            if (c0 >= 41) {
                                {
                                    float _cvt_f32_f16_87;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_87) : "h"((uint16_t)(acc[10] >> 16)));
                                    v_20 = _cvt_f32_f16_87;
                                }
                            }
                            float _max_42 = max_noftz(m0v, v_20);
                            m0v = _max_42;
                        }
                        float v_21 = neg;
                        {
                            if (c1 >= 40) {
                                {
                                    float _cvt_f32_f16_92;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_92) : "h"((uint16_t)(acc[11] & 65535)));
                                    v_21 = _cvt_f32_f16_92;
                                }
                            }
                            float _max_45 = max_noftz(m1v, v_21);
                            m1v = _max_45;
                        }
                        float v_22 = neg;
                        {
                            if (c1 >= 41) {
                                {
                                    float _cvt_f32_f16_97;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_97) : "h"((uint16_t)(acc[11] >> 16)));
                                    v_22 = _cvt_f32_f16_97;
                                }
                            }
                            float _max_47 = max_noftz(m1v, v_22);
                            m1v = _max_47;
                        }
                        float v_23 = neg;
                        {
                            if (c0 >= 48) {
                                {
                                    float _cvt_f32_f16_98;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_98) : "h"((uint16_t)(acc[12] & 65535)));
                                    v_23 = _cvt_f32_f16_98;
                                }
                            }
                            float _max_48 = max_noftz(m0v, v_23);
                            m0v = _max_48;
                        }
                        float v_24 = neg;
                        {
                            if (c0 >= 49) {
                                {
                                    float _cvt_f32_f16_103;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_103) : "h"((uint16_t)(acc[12] >> 16)));
                                    v_24 = _cvt_f32_f16_103;
                                }
                            }
                            float _max_50 = max_noftz(m0v, v_24);
                            m0v = _max_50;
                        }
                        float v_25 = neg;
                        {
                            if (c1 >= 48) {
                                {
                                    float _cvt_f32_f16_108;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_108) : "h"((uint16_t)(acc[13] & 65535)));
                                    v_25 = _cvt_f32_f16_108;
                                }
                            }
                            float _max_53 = max_noftz(m1v, v_25);
                            m1v = _max_53;
                        }
                        float v_26 = neg;
                        {
                            if (c1 >= 49) {
                                {
                                    float _cvt_f32_f16_113;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_113) : "h"((uint16_t)(acc[13] >> 16)));
                                    v_26 = _cvt_f32_f16_113;
                                }
                            }
                            float _max_55 = max_noftz(m1v, v_26);
                            m1v = _max_55;
                        }
                        float v_27 = neg;
                        {
                            if (c0 >= 56) {
                                {
                                    float _cvt_f32_f16_114;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_114) : "h"((uint16_t)(acc[14] & 65535)));
                                    v_27 = _cvt_f32_f16_114;
                                }
                            }
                            float _max_56 = max_noftz(m0v, v_27);
                            m0v = _max_56;
                        }
                        float v_28 = neg;
                        {
                            if (c0 >= 57) {
                                {
                                    float _cvt_f32_f16_119;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_119) : "h"((uint16_t)(acc[14] >> 16)));
                                    v_28 = _cvt_f32_f16_119;
                                }
                            }
                            float _max_58 = max_noftz(m0v, v_28);
                            m0v = _max_58;
                        }
                        float v_29 = neg;
                        {
                            if (c1 >= 56) {
                                {
                                    float _cvt_f32_f16_124;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_124) : "h"((uint16_t)(acc[15] & 65535)));
                                    v_29 = _cvt_f32_f16_124;
                                }
                            }
                            float _max_61 = max_noftz(m1v, v_29);
                            m1v = _max_61;
                        }
                        float v_30 = neg;
                        {
                            if (c1 >= 57) {
                                {
                                    float _cvt_f32_f16_129;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_129) : "h"((uint16_t)(acc[15] >> 16)));
                                    v_30 = _cvt_f32_f16_129;
                                }
                            }
                            float _max_63 = max_noftz(m1v, v_30);
                            m1v = _max_63;
                        }
                        float v_31 = neg;
                        {
                            if (c0 >= 64) {
                                {
                                    float _cvt_f32_f16_130;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_130) : "h"((uint16_t)(acc[16] & 65535)));
                                    v_31 = _cvt_f32_f16_130;
                                }
                            }
                            float _max_64 = max_noftz(m0v, v_31);
                            m0v = _max_64;
                        }
                        float v_32 = neg;
                        {
                            if (c0 >= 65) {
                                {
                                    float _cvt_f32_f16_135;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_135) : "h"((uint16_t)(acc[16] >> 16)));
                                    v_32 = _cvt_f32_f16_135;
                                }
                            }
                            float _max_66 = max_noftz(m0v, v_32);
                            m0v = _max_66;
                        }
                        float v_33 = neg;
                        {
                            if (c1 >= 64) {
                                {
                                    float _cvt_f32_f16_140;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_140) : "h"((uint16_t)(acc[17] & 65535)));
                                    v_33 = _cvt_f32_f16_140;
                                }
                            }
                            float _max_69 = max_noftz(m1v, v_33);
                            m1v = _max_69;
                        }
                        float v_34 = neg;
                        {
                            if (c1 >= 65) {
                                {
                                    float _cvt_f32_f16_145;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_145) : "h"((uint16_t)(acc[17] >> 16)));
                                    v_34 = _cvt_f32_f16_145;
                                }
                            }
                            float _max_71 = max_noftz(m1v, v_34);
                            m1v = _max_71;
                        }
                        float v_35 = neg;
                        {
                            if (c0 >= 72) {
                                {
                                    float _cvt_f32_f16_146;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_146) : "h"((uint16_t)(acc[18] & 65535)));
                                    v_35 = _cvt_f32_f16_146;
                                }
                            }
                            float _max_72 = max_noftz(m0v, v_35);
                            m0v = _max_72;
                        }
                        float v_36 = neg;
                        {
                            if (c0 >= 73) {
                                {
                                    float _cvt_f32_f16_151;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_151) : "h"((uint16_t)(acc[18] >> 16)));
                                    v_36 = _cvt_f32_f16_151;
                                }
                            }
                            float _max_74 = max_noftz(m0v, v_36);
                            m0v = _max_74;
                        }
                        float v_37 = neg;
                        {
                            if (c1 >= 72) {
                                {
                                    float _cvt_f32_f16_156;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_156) : "h"((uint16_t)(acc[19] & 65535)));
                                    v_37 = _cvt_f32_f16_156;
                                }
                            }
                            float _max_77 = max_noftz(m1v, v_37);
                            m1v = _max_77;
                        }
                        float v_38 = neg;
                        {
                            if (c1 >= 73) {
                                {
                                    float _cvt_f32_f16_161;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_161) : "h"((uint16_t)(acc[19] >> 16)));
                                    v_38 = _cvt_f32_f16_161;
                                }
                            }
                            float _max_79 = max_noftz(m1v, v_38);
                            m1v = _max_79;
                        }
                        float v_39 = neg;
                        {
                            if (c0 >= 80) {
                                {
                                    float _cvt_f32_f16_162;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_162) : "h"((uint16_t)(acc[20] & 65535)));
                                    v_39 = _cvt_f32_f16_162;
                                }
                            }
                            float _max_80 = max_noftz(m0v, v_39);
                            m0v = _max_80;
                        }
                        float v_40 = neg;
                        {
                            if (c0 >= 81) {
                                {
                                    float _cvt_f32_f16_167;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_167) : "h"((uint16_t)(acc[20] >> 16)));
                                    v_40 = _cvt_f32_f16_167;
                                }
                            }
                            float _max_82 = max_noftz(m0v, v_40);
                            m0v = _max_82;
                        }
                        float v_41 = neg;
                        {
                            if (c1 >= 80) {
                                {
                                    float _cvt_f32_f16_172;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_172) : "h"((uint16_t)(acc[21] & 65535)));
                                    v_41 = _cvt_f32_f16_172;
                                }
                            }
                            float _max_85 = max_noftz(m1v, v_41);
                            m1v = _max_85;
                        }
                        float v_42 = neg;
                        {
                            if (c1 >= 81) {
                                {
                                    float _cvt_f32_f16_177;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_177) : "h"((uint16_t)(acc[21] >> 16)));
                                    v_42 = _cvt_f32_f16_177;
                                }
                            }
                            float _max_87 = max_noftz(m1v, v_42);
                            m1v = _max_87;
                        }
                        float v_43 = neg;
                        {
                            if (c0 >= 88) {
                                {
                                    float _cvt_f32_f16_178;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_178) : "h"((uint16_t)(acc[22] & 65535)));
                                    v_43 = _cvt_f32_f16_178;
                                }
                            }
                            float _max_88 = max_noftz(m0v, v_43);
                            m0v = _max_88;
                        }
                        float v_44 = neg;
                        {
                            if (c0 >= 89) {
                                {
                                    float _cvt_f32_f16_183;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_183) : "h"((uint16_t)(acc[22] >> 16)));
                                    v_44 = _cvt_f32_f16_183;
                                }
                            }
                            float _max_90 = max_noftz(m0v, v_44);
                            m0v = _max_90;
                        }
                        float v_45 = neg;
                        {
                            if (c1 >= 88) {
                                {
                                    float _cvt_f32_f16_188;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_188) : "h"((uint16_t)(acc[23] & 65535)));
                                    v_45 = _cvt_f32_f16_188;
                                }
                            }
                            float _max_93 = max_noftz(m1v, v_45);
                            m1v = _max_93;
                        }
                        float v_46 = neg;
                        {
                            if (c1 >= 89) {
                                {
                                    float _cvt_f32_f16_193;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_193) : "h"((uint16_t)(acc[23] >> 16)));
                                    v_46 = _cvt_f32_f16_193;
                                }
                            }
                            float _max_95 = max_noftz(m1v, v_46);
                            m1v = _max_95;
                        }
                        float v_47 = neg;
                        {
                            if (c0 >= 96) {
                                {
                                    float _cvt_f32_f16_194;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_194) : "h"((uint16_t)(acc[24] & 65535)));
                                    v_47 = _cvt_f32_f16_194;
                                }
                            }
                            float _max_96 = max_noftz(m0v, v_47);
                            m0v = _max_96;
                        }
                        float v_48 = neg;
                        {
                            if (c0 >= 97) {
                                {
                                    float _cvt_f32_f16_199;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_199) : "h"((uint16_t)(acc[24] >> 16)));
                                    v_48 = _cvt_f32_f16_199;
                                }
                            }
                            float _max_98 = max_noftz(m0v, v_48);
                            m0v = _max_98;
                        }
                        float v_49 = neg;
                        {
                            if (c1 >= 96) {
                                {
                                    float _cvt_f32_f16_204;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_204) : "h"((uint16_t)(acc[25] & 65535)));
                                    v_49 = _cvt_f32_f16_204;
                                }
                            }
                            float _max_101 = max_noftz(m1v, v_49);
                            m1v = _max_101;
                        }
                        float v_50 = neg;
                        {
                            if (c1 >= 97) {
                                {
                                    float _cvt_f32_f16_209;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_209) : "h"((uint16_t)(acc[25] >> 16)));
                                    v_50 = _cvt_f32_f16_209;
                                }
                            }
                            float _max_103 = max_noftz(m1v, v_50);
                            m1v = _max_103;
                        }
                        float v_51 = neg;
                        {
                            if (c0 >= 104) {
                                {
                                    float _cvt_f32_f16_210;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_210) : "h"((uint16_t)(acc[26] & 65535)));
                                    v_51 = _cvt_f32_f16_210;
                                }
                            }
                            float _max_104 = max_noftz(m0v, v_51);
                            m0v = _max_104;
                        }
                        float v_52 = neg;
                        {
                            if (c0 >= 105) {
                                {
                                    float _cvt_f32_f16_215;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_215) : "h"((uint16_t)(acc[26] >> 16)));
                                    v_52 = _cvt_f32_f16_215;
                                }
                            }
                            float _max_106 = max_noftz(m0v, v_52);
                            m0v = _max_106;
                        }
                        float v_53 = neg;
                        {
                            if (c1 >= 104) {
                                {
                                    float _cvt_f32_f16_220;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_220) : "h"((uint16_t)(acc[27] & 65535)));
                                    v_53 = _cvt_f32_f16_220;
                                }
                            }
                            float _max_109 = max_noftz(m1v, v_53);
                            m1v = _max_109;
                        }
                        float v_54 = neg;
                        {
                            if (c1 >= 105) {
                                {
                                    float _cvt_f32_f16_225;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_225) : "h"((uint16_t)(acc[27] >> 16)));
                                    v_54 = _cvt_f32_f16_225;
                                }
                            }
                            float _max_111 = max_noftz(m1v, v_54);
                            m1v = _max_111;
                        }
                        float v_55 = neg;
                        {
                            if (c0 >= 112) {
                                {
                                    float _cvt_f32_f16_226;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_226) : "h"((uint16_t)(acc[28] & 65535)));
                                    v_55 = _cvt_f32_f16_226;
                                }
                            }
                            float _max_112 = max_noftz(m0v, v_55);
                            m0v = _max_112;
                        }
                        float v_56 = neg;
                        {
                            if (c0 >= 113) {
                                {
                                    float _cvt_f32_f16_231;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_231) : "h"((uint16_t)(acc[28] >> 16)));
                                    v_56 = _cvt_f32_f16_231;
                                }
                            }
                            float _max_114 = max_noftz(m0v, v_56);
                            m0v = _max_114;
                        }
                        float v_57 = neg;
                        {
                            if (c1 >= 112) {
                                {
                                    float _cvt_f32_f16_236;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_236) : "h"((uint16_t)(acc[29] & 65535)));
                                    v_57 = _cvt_f32_f16_236;
                                }
                            }
                            float _max_117 = max_noftz(m1v, v_57);
                            m1v = _max_117;
                        }
                        float v_58 = neg;
                        {
                            if (c1 >= 113) {
                                {
                                    float _cvt_f32_f16_241;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_241) : "h"((uint16_t)(acc[29] >> 16)));
                                    v_58 = _cvt_f32_f16_241;
                                }
                            }
                            float _max_119 = max_noftz(m1v, v_58);
                            m1v = _max_119;
                        }
                        float v_59 = neg;
                        {
                            if (c0 >= 120) {
                                {
                                    float _cvt_f32_f16_242;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_242) : "h"((uint16_t)(acc[30] & 65535)));
                                    v_59 = _cvt_f32_f16_242;
                                }
                            }
                            float _max_120 = max_noftz(m0v, v_59);
                            m0v = _max_120;
                        }
                        float v_60 = neg;
                        {
                            if (c0 >= 121) {
                                {
                                    float _cvt_f32_f16_247;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_247) : "h"((uint16_t)(acc[30] >> 16)));
                                    v_60 = _cvt_f32_f16_247;
                                }
                            }
                            float _max_122 = max_noftz(m0v, v_60);
                            m0v = _max_122;
                        }
                        float v_61 = neg;
                        {
                            if (c1 >= 120) {
                                {
                                    float _cvt_f32_f16_252;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_252) : "h"((uint16_t)(acc[31] & 65535)));
                                    v_61 = _cvt_f32_f16_252;
                                }
                            }
                            float _max_125 = max_noftz(m1v, v_61);
                            m1v = _max_125;
                        }
                        float v_62 = neg;
                        {
                            if (c1 >= 121) {
                                {
                                    float _cvt_f32_f16_257;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_257) : "h"((uint16_t)(acc[31] >> 16)));
                                    v_62 = _cvt_f32_f16_257;
                                }
                            }
                            float _max_127 = max_noftz(m1v, v_62);
                            m1v = _max_127;
                        }
                    }
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, m0v, 1);
                    float _max_128 = max_noftz(m0v, _shfl_xor_0);
                    m0v = _max_128;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, m0v, 2);
                    float _max_129 = max_noftz(m0v, _shfl_xor_1);
                    m0v = _max_129;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, m1v, 1);
                    float _max_130 = max_noftz(m1v, _shfl_xor_2);
                    m1v = _max_130;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, m1v, 2);
                    float _max_131 = max_noftz(m1v, _shfl_xor_3);
                    m1v = _max_131;
                    if (cq == 0) {
                        int orow = obase + (split2 + ia * nsplit) * total_q + r0;
                        if (r0 < rows_valid) {
                            out[orow] = m0v;
                        }
                        if (rows_valid > r0 + 8) {
                            out[orow + 8] = m1v;
                        }
                    }
                    mbarrier_wait(full_addr + (sc) * 8, ic / 4 & 1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_3 = (((uint64_t)(((k_s_addr + (unsigned int)(sc * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
                        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                        : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_3)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                        : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_3 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                        : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_3 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3]), "+r"(acc[4]), "+r"(acc[5]), "+r"(acc[6]), "+r"(acc[7]), "+r"(acc[8]), "+r"(acc[9]), "+r"(acc[10]), "+r"(acc[11]), "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15]), "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19]), "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23]), "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27]), "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
                        : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_3 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(empty_addr + (sb) * 8);
                    }
                    int k0_0 = (split2 + ib * nsplit) * 128;
                    float m0v_1 = neg;
                    float m1v_2 = neg;
                    int need_mask_3 = 0;
                    if (mask_thr < k0_0 + 127) {
                        need_mask_3 = 1;
                    }
                    if (need_mask_3 == 0) {
                        unsigned int w0_1 = acc1[0];
                        unsigned int w1_1 = acc1[1];
                        uint32_t _f16x2_max_30;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_30) : "r"(w0_1), "r"(acc1[2]));
                        w0_1 = _f16x2_max_30;
                        uint32_t _f16x2_max_31;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_31) : "r"(w1_1), "r"(acc1[3]));
                        w1_1 = _f16x2_max_31;
                        uint32_t _f16x2_max_32;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_32) : "r"(w0_1), "r"(acc1[4]));
                        w0_1 = _f16x2_max_32;
                        uint32_t _f16x2_max_33;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_33) : "r"(w1_1), "r"(acc1[5]));
                        w1_1 = _f16x2_max_33;
                        uint32_t _f16x2_max_34;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_34) : "r"(w0_1), "r"(acc1[6]));
                        w0_1 = _f16x2_max_34;
                        uint32_t _f16x2_max_35;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_35) : "r"(w1_1), "r"(acc1[7]));
                        w1_1 = _f16x2_max_35;
                        uint32_t _f16x2_max_36;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_36) : "r"(w0_1), "r"(acc1[8]));
                        w0_1 = _f16x2_max_36;
                        uint32_t _f16x2_max_37;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_37) : "r"(w1_1), "r"(acc1[9]));
                        w1_1 = _f16x2_max_37;
                        uint32_t _f16x2_max_38;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_38) : "r"(w0_1), "r"(acc1[10]));
                        w0_1 = _f16x2_max_38;
                        uint32_t _f16x2_max_39;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_39) : "r"(w1_1), "r"(acc1[11]));
                        w1_1 = _f16x2_max_39;
                        uint32_t _f16x2_max_40;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_40) : "r"(w0_1), "r"(acc1[12]));
                        w0_1 = _f16x2_max_40;
                        uint32_t _f16x2_max_41;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_41) : "r"(w1_1), "r"(acc1[13]));
                        w1_1 = _f16x2_max_41;
                        uint32_t _f16x2_max_42;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_42) : "r"(w0_1), "r"(acc1[14]));
                        w0_1 = _f16x2_max_42;
                        uint32_t _f16x2_max_43;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_43) : "r"(w1_1), "r"(acc1[15]));
                        w1_1 = _f16x2_max_43;
                        uint32_t _f16x2_max_44;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_44) : "r"(w0_1), "r"(acc1[16]));
                        w0_1 = _f16x2_max_44;
                        uint32_t _f16x2_max_45;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_45) : "r"(w1_1), "r"(acc1[17]));
                        w1_1 = _f16x2_max_45;
                        uint32_t _f16x2_max_46;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_46) : "r"(w0_1), "r"(acc1[18]));
                        w0_1 = _f16x2_max_46;
                        uint32_t _f16x2_max_47;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_47) : "r"(w1_1), "r"(acc1[19]));
                        w1_1 = _f16x2_max_47;
                        uint32_t _f16x2_max_48;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_48) : "r"(w0_1), "r"(acc1[20]));
                        w0_1 = _f16x2_max_48;
                        uint32_t _f16x2_max_49;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_49) : "r"(w1_1), "r"(acc1[21]));
                        w1_1 = _f16x2_max_49;
                        uint32_t _f16x2_max_50;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_50) : "r"(w0_1), "r"(acc1[22]));
                        w0_1 = _f16x2_max_50;
                        uint32_t _f16x2_max_51;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_51) : "r"(w1_1), "r"(acc1[23]));
                        w1_1 = _f16x2_max_51;
                        uint32_t _f16x2_max_52;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_52) : "r"(w0_1), "r"(acc1[24]));
                        w0_1 = _f16x2_max_52;
                        uint32_t _f16x2_max_53;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_53) : "r"(w1_1), "r"(acc1[25]));
                        w1_1 = _f16x2_max_53;
                        uint32_t _f16x2_max_54;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_54) : "r"(w0_1), "r"(acc1[26]));
                        w0_1 = _f16x2_max_54;
                        uint32_t _f16x2_max_55;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_55) : "r"(w1_1), "r"(acc1[27]));
                        w1_1 = _f16x2_max_55;
                        uint32_t _f16x2_max_56;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_56) : "r"(w0_1), "r"(acc1[28]));
                        w0_1 = _f16x2_max_56;
                        uint32_t _f16x2_max_57;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_57) : "r"(w1_1), "r"(acc1[29]));
                        w1_1 = _f16x2_max_57;
                        uint32_t _f16x2_max_58;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_58) : "r"(w0_1), "r"(acc1[30]));
                        w0_1 = _f16x2_max_58;
                        uint32_t _f16x2_max_59;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_59) : "r"(w1_1), "r"(acc1[31]));
                        w1_1 = _f16x2_max_59;
                        uint16_t _f16_max_2;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_2) : "h"((uint16_t)(w0_1 & 65535)), "h"((uint16_t)(w0_1 >> 16)));
                        float _cvt_f32_f16_258;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_258) : "h"((uint16_t)(_f16_max_2)));
                        m0v_1 = _cvt_f32_f16_258;
                        uint16_t _f16_max_3;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_3) : "h"((uint16_t)(w1_1 & 65535)), "h"((uint16_t)(w1_1 >> 16)));
                        float _cvt_f32_f16_259;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_259) : "h"((uint16_t)(_f16_max_3)));
                        m1v_2 = _cvt_f32_f16_259;
                    }
                    if (need_mask_3 != 0) {
                        int c0_1 = lim0 - k0_0 - 2 * cq;
                        int c1_1 = lim1 - k0_0 - 2 * cq;
                        float v_63 = neg;
                        {
                            if (c0_1 >= 0) {
                                {
                                    float _cvt_f32_f16_260;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_260) : "h"((uint16_t)(acc1[0] & 65535)));
                                    v_63 = _cvt_f32_f16_260;
                                }
                            }
                            float _max_132 = max_noftz(m0v_1, v_63);
                            m0v_1 = _max_132;
                        }
                        float v_0_1 = neg;
                        {
                            if (c0_1 >= 1) {
                                {
                                    float _cvt_f32_f16_265;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_265) : "h"((uint16_t)(acc1[0] >> 16)));
                                    v_0_1 = _cvt_f32_f16_265;
                                }
                            }
                            float _max_134 = max_noftz(m0v_1, v_0_1);
                            m0v_1 = _max_134;
                        }
                        float v_1_1 = neg;
                        {
                            if (c1_1 >= 0) {
                                {
                                    float _cvt_f32_f16_270;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_270) : "h"((uint16_t)(acc1[1] & 65535)));
                                    v_1_1 = _cvt_f32_f16_270;
                                }
                            }
                            float _max_137 = max_noftz(m1v_2, v_1_1);
                            m1v_2 = _max_137;
                        }
                        float v_2_1 = neg;
                        {
                            if (c1_1 >= 1) {
                                {
                                    float _cvt_f32_f16_275;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_275) : "h"((uint16_t)(acc1[1] >> 16)));
                                    v_2_1 = _cvt_f32_f16_275;
                                }
                            }
                            float _max_139 = max_noftz(m1v_2, v_2_1);
                            m1v_2 = _max_139;
                        }
                        float v_3_1 = neg;
                        {
                            if (c0_1 >= 8) {
                                {
                                    float _cvt_f32_f16_276;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_276) : "h"((uint16_t)(acc1[2] & 65535)));
                                    v_3_1 = _cvt_f32_f16_276;
                                }
                            }
                            float _max_140 = max_noftz(m0v_1, v_3_1);
                            m0v_1 = _max_140;
                        }
                        float v_4_1 = neg;
                        {
                            if (c0_1 >= 9) {
                                {
                                    float _cvt_f32_f16_281;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_281) : "h"((uint16_t)(acc1[2] >> 16)));
                                    v_4_1 = _cvt_f32_f16_281;
                                }
                            }
                            float _max_142 = max_noftz(m0v_1, v_4_1);
                            m0v_1 = _max_142;
                        }
                        float v_5_1 = neg;
                        {
                            if (c1_1 >= 8) {
                                {
                                    float _cvt_f32_f16_286;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_286) : "h"((uint16_t)(acc1[3] & 65535)));
                                    v_5_1 = _cvt_f32_f16_286;
                                }
                            }
                            float _max_145 = max_noftz(m1v_2, v_5_1);
                            m1v_2 = _max_145;
                        }
                        float v_6_1 = neg;
                        {
                            if (c1_1 >= 9) {
                                {
                                    float _cvt_f32_f16_291;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_291) : "h"((uint16_t)(acc1[3] >> 16)));
                                    v_6_1 = _cvt_f32_f16_291;
                                }
                            }
                            float _max_147 = max_noftz(m1v_2, v_6_1);
                            m1v_2 = _max_147;
                        }
                        float v_7_1 = neg;
                        {
                            if (c0_1 >= 16) {
                                {
                                    float _cvt_f32_f16_292;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_292) : "h"((uint16_t)(acc1[4] & 65535)));
                                    v_7_1 = _cvt_f32_f16_292;
                                }
                            }
                            float _max_148 = max_noftz(m0v_1, v_7_1);
                            m0v_1 = _max_148;
                        }
                        float v_8_1 = neg;
                        {
                            if (c0_1 >= 17) {
                                {
                                    float _cvt_f32_f16_297;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_297) : "h"((uint16_t)(acc1[4] >> 16)));
                                    v_8_1 = _cvt_f32_f16_297;
                                }
                            }
                            float _max_150 = max_noftz(m0v_1, v_8_1);
                            m0v_1 = _max_150;
                        }
                        float v_9_1 = neg;
                        {
                            if (c1_1 >= 16) {
                                {
                                    float _cvt_f32_f16_302;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_302) : "h"((uint16_t)(acc1[5] & 65535)));
                                    v_9_1 = _cvt_f32_f16_302;
                                }
                            }
                            float _max_153 = max_noftz(m1v_2, v_9_1);
                            m1v_2 = _max_153;
                        }
                        float v_10_1 = neg;
                        {
                            if (c1_1 >= 17) {
                                {
                                    float _cvt_f32_f16_307;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_307) : "h"((uint16_t)(acc1[5] >> 16)));
                                    v_10_1 = _cvt_f32_f16_307;
                                }
                            }
                            float _max_155 = max_noftz(m1v_2, v_10_1);
                            m1v_2 = _max_155;
                        }
                        float v_11_1 = neg;
                        {
                            if (c0_1 >= 24) {
                                {
                                    float _cvt_f32_f16_308;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_308) : "h"((uint16_t)(acc1[6] & 65535)));
                                    v_11_1 = _cvt_f32_f16_308;
                                }
                            }
                            float _max_156 = max_noftz(m0v_1, v_11_1);
                            m0v_1 = _max_156;
                        }
                        float v_12_1 = neg;
                        {
                            if (c0_1 >= 25) {
                                {
                                    float _cvt_f32_f16_313;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_313) : "h"((uint16_t)(acc1[6] >> 16)));
                                    v_12_1 = _cvt_f32_f16_313;
                                }
                            }
                            float _max_158 = max_noftz(m0v_1, v_12_1);
                            m0v_1 = _max_158;
                        }
                        float v_13_1 = neg;
                        {
                            if (c1_1 >= 24) {
                                {
                                    float _cvt_f32_f16_318;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_318) : "h"((uint16_t)(acc1[7] & 65535)));
                                    v_13_1 = _cvt_f32_f16_318;
                                }
                            }
                            float _max_161 = max_noftz(m1v_2, v_13_1);
                            m1v_2 = _max_161;
                        }
                        float v_14_1 = neg;
                        {
                            if (c1_1 >= 25) {
                                {
                                    float _cvt_f32_f16_323;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_323) : "h"((uint16_t)(acc1[7] >> 16)));
                                    v_14_1 = _cvt_f32_f16_323;
                                }
                            }
                            float _max_163 = max_noftz(m1v_2, v_14_1);
                            m1v_2 = _max_163;
                        }
                        float v_15_1 = neg;
                        {
                            if (c0_1 >= 32) {
                                {
                                    float _cvt_f32_f16_324;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_324) : "h"((uint16_t)(acc1[8] & 65535)));
                                    v_15_1 = _cvt_f32_f16_324;
                                }
                            }
                            float _max_164 = max_noftz(m0v_1, v_15_1);
                            m0v_1 = _max_164;
                        }
                        float v_16_1 = neg;
                        {
                            if (c0_1 >= 33) {
                                {
                                    float _cvt_f32_f16_329;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_329) : "h"((uint16_t)(acc1[8] >> 16)));
                                    v_16_1 = _cvt_f32_f16_329;
                                }
                            }
                            float _max_166 = max_noftz(m0v_1, v_16_1);
                            m0v_1 = _max_166;
                        }
                        float v_17_1 = neg;
                        {
                            if (c1_1 >= 32) {
                                {
                                    float _cvt_f32_f16_334;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_334) : "h"((uint16_t)(acc1[9] & 65535)));
                                    v_17_1 = _cvt_f32_f16_334;
                                }
                            }
                            float _max_169 = max_noftz(m1v_2, v_17_1);
                            m1v_2 = _max_169;
                        }
                        float v_18_1 = neg;
                        {
                            if (c1_1 >= 33) {
                                {
                                    float _cvt_f32_f16_339;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_339) : "h"((uint16_t)(acc1[9] >> 16)));
                                    v_18_1 = _cvt_f32_f16_339;
                                }
                            }
                            float _max_171 = max_noftz(m1v_2, v_18_1);
                            m1v_2 = _max_171;
                        }
                        float v_19_1 = neg;
                        {
                            if (c0_1 >= 40) {
                                {
                                    float _cvt_f32_f16_340;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_340) : "h"((uint16_t)(acc1[10] & 65535)));
                                    v_19_1 = _cvt_f32_f16_340;
                                }
                            }
                            float _max_172 = max_noftz(m0v_1, v_19_1);
                            m0v_1 = _max_172;
                        }
                        float v_20_1 = neg;
                        {
                            if (c0_1 >= 41) {
                                {
                                    float _cvt_f32_f16_345;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_345) : "h"((uint16_t)(acc1[10] >> 16)));
                                    v_20_1 = _cvt_f32_f16_345;
                                }
                            }
                            float _max_174 = max_noftz(m0v_1, v_20_1);
                            m0v_1 = _max_174;
                        }
                        float v_21_1 = neg;
                        {
                            if (c1_1 >= 40) {
                                {
                                    float _cvt_f32_f16_350;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_350) : "h"((uint16_t)(acc1[11] & 65535)));
                                    v_21_1 = _cvt_f32_f16_350;
                                }
                            }
                            float _max_177 = max_noftz(m1v_2, v_21_1);
                            m1v_2 = _max_177;
                        }
                        float v_22_1 = neg;
                        {
                            if (c1_1 >= 41) {
                                {
                                    float _cvt_f32_f16_355;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_355) : "h"((uint16_t)(acc1[11] >> 16)));
                                    v_22_1 = _cvt_f32_f16_355;
                                }
                            }
                            float _max_179 = max_noftz(m1v_2, v_22_1);
                            m1v_2 = _max_179;
                        }
                        float v_23_1 = neg;
                        {
                            if (c0_1 >= 48) {
                                {
                                    float _cvt_f32_f16_356;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_356) : "h"((uint16_t)(acc1[12] & 65535)));
                                    v_23_1 = _cvt_f32_f16_356;
                                }
                            }
                            float _max_180 = max_noftz(m0v_1, v_23_1);
                            m0v_1 = _max_180;
                        }
                        float v_24_1 = neg;
                        {
                            if (c0_1 >= 49) {
                                {
                                    float _cvt_f32_f16_361;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_361) : "h"((uint16_t)(acc1[12] >> 16)));
                                    v_24_1 = _cvt_f32_f16_361;
                                }
                            }
                            float _max_182 = max_noftz(m0v_1, v_24_1);
                            m0v_1 = _max_182;
                        }
                        float v_25_1 = neg;
                        {
                            if (c1_1 >= 48) {
                                {
                                    float _cvt_f32_f16_366;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_366) : "h"((uint16_t)(acc1[13] & 65535)));
                                    v_25_1 = _cvt_f32_f16_366;
                                }
                            }
                            float _max_185 = max_noftz(m1v_2, v_25_1);
                            m1v_2 = _max_185;
                        }
                        float v_26_1 = neg;
                        {
                            if (c1_1 >= 49) {
                                {
                                    float _cvt_f32_f16_371;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_371) : "h"((uint16_t)(acc1[13] >> 16)));
                                    v_26_1 = _cvt_f32_f16_371;
                                }
                            }
                            float _max_187 = max_noftz(m1v_2, v_26_1);
                            m1v_2 = _max_187;
                        }
                        float v_27_1 = neg;
                        {
                            if (c0_1 >= 56) {
                                {
                                    float _cvt_f32_f16_372;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_372) : "h"((uint16_t)(acc1[14] & 65535)));
                                    v_27_1 = _cvt_f32_f16_372;
                                }
                            }
                            float _max_188 = max_noftz(m0v_1, v_27_1);
                            m0v_1 = _max_188;
                        }
                        float v_28_1 = neg;
                        {
                            if (c0_1 >= 57) {
                                {
                                    float _cvt_f32_f16_377;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_377) : "h"((uint16_t)(acc1[14] >> 16)));
                                    v_28_1 = _cvt_f32_f16_377;
                                }
                            }
                            float _max_190 = max_noftz(m0v_1, v_28_1);
                            m0v_1 = _max_190;
                        }
                        float v_29_1 = neg;
                        {
                            if (c1_1 >= 56) {
                                {
                                    float _cvt_f32_f16_382;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_382) : "h"((uint16_t)(acc1[15] & 65535)));
                                    v_29_1 = _cvt_f32_f16_382;
                                }
                            }
                            float _max_193 = max_noftz(m1v_2, v_29_1);
                            m1v_2 = _max_193;
                        }
                        float v_30_1 = neg;
                        {
                            if (c1_1 >= 57) {
                                {
                                    float _cvt_f32_f16_387;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_387) : "h"((uint16_t)(acc1[15] >> 16)));
                                    v_30_1 = _cvt_f32_f16_387;
                                }
                            }
                            float _max_195 = max_noftz(m1v_2, v_30_1);
                            m1v_2 = _max_195;
                        }
                        float v_31_1 = neg;
                        {
                            if (c0_1 >= 64) {
                                {
                                    float _cvt_f32_f16_388;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_388) : "h"((uint16_t)(acc1[16] & 65535)));
                                    v_31_1 = _cvt_f32_f16_388;
                                }
                            }
                            float _max_196 = max_noftz(m0v_1, v_31_1);
                            m0v_1 = _max_196;
                        }
                        float v_32_1 = neg;
                        {
                            if (c0_1 >= 65) {
                                {
                                    float _cvt_f32_f16_393;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_393) : "h"((uint16_t)(acc1[16] >> 16)));
                                    v_32_1 = _cvt_f32_f16_393;
                                }
                            }
                            float _max_198 = max_noftz(m0v_1, v_32_1);
                            m0v_1 = _max_198;
                        }
                        float v_33_1 = neg;
                        {
                            if (c1_1 >= 64) {
                                {
                                    float _cvt_f32_f16_398;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_398) : "h"((uint16_t)(acc1[17] & 65535)));
                                    v_33_1 = _cvt_f32_f16_398;
                                }
                            }
                            float _max_201 = max_noftz(m1v_2, v_33_1);
                            m1v_2 = _max_201;
                        }
                        float v_34_1 = neg;
                        {
                            if (c1_1 >= 65) {
                                {
                                    float _cvt_f32_f16_403;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_403) : "h"((uint16_t)(acc1[17] >> 16)));
                                    v_34_1 = _cvt_f32_f16_403;
                                }
                            }
                            float _max_203 = max_noftz(m1v_2, v_34_1);
                            m1v_2 = _max_203;
                        }
                        float v_35_1 = neg;
                        {
                            if (c0_1 >= 72) {
                                {
                                    float _cvt_f32_f16_404;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_404) : "h"((uint16_t)(acc1[18] & 65535)));
                                    v_35_1 = _cvt_f32_f16_404;
                                }
                            }
                            float _max_204 = max_noftz(m0v_1, v_35_1);
                            m0v_1 = _max_204;
                        }
                        float v_36_1 = neg;
                        {
                            if (c0_1 >= 73) {
                                {
                                    float _cvt_f32_f16_409;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_409) : "h"((uint16_t)(acc1[18] >> 16)));
                                    v_36_1 = _cvt_f32_f16_409;
                                }
                            }
                            float _max_206 = max_noftz(m0v_1, v_36_1);
                            m0v_1 = _max_206;
                        }
                        float v_37_1 = neg;
                        {
                            if (c1_1 >= 72) {
                                {
                                    float _cvt_f32_f16_414;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_414) : "h"((uint16_t)(acc1[19] & 65535)));
                                    v_37_1 = _cvt_f32_f16_414;
                                }
                            }
                            float _max_209 = max_noftz(m1v_2, v_37_1);
                            m1v_2 = _max_209;
                        }
                        float v_38_1 = neg;
                        {
                            if (c1_1 >= 73) {
                                {
                                    float _cvt_f32_f16_419;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_419) : "h"((uint16_t)(acc1[19] >> 16)));
                                    v_38_1 = _cvt_f32_f16_419;
                                }
                            }
                            float _max_211 = max_noftz(m1v_2, v_38_1);
                            m1v_2 = _max_211;
                        }
                        float v_39_1 = neg;
                        {
                            if (c0_1 >= 80) {
                                {
                                    float _cvt_f32_f16_420;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_420) : "h"((uint16_t)(acc1[20] & 65535)));
                                    v_39_1 = _cvt_f32_f16_420;
                                }
                            }
                            float _max_212 = max_noftz(m0v_1, v_39_1);
                            m0v_1 = _max_212;
                        }
                        float v_40_1 = neg;
                        {
                            if (c0_1 >= 81) {
                                {
                                    float _cvt_f32_f16_425;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_425) : "h"((uint16_t)(acc1[20] >> 16)));
                                    v_40_1 = _cvt_f32_f16_425;
                                }
                            }
                            float _max_214 = max_noftz(m0v_1, v_40_1);
                            m0v_1 = _max_214;
                        }
                        float v_41_1 = neg;
                        {
                            if (c1_1 >= 80) {
                                {
                                    float _cvt_f32_f16_430;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_430) : "h"((uint16_t)(acc1[21] & 65535)));
                                    v_41_1 = _cvt_f32_f16_430;
                                }
                            }
                            float _max_217 = max_noftz(m1v_2, v_41_1);
                            m1v_2 = _max_217;
                        }
                        float v_42_1 = neg;
                        {
                            if (c1_1 >= 81) {
                                {
                                    float _cvt_f32_f16_435;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_435) : "h"((uint16_t)(acc1[21] >> 16)));
                                    v_42_1 = _cvt_f32_f16_435;
                                }
                            }
                            float _max_219 = max_noftz(m1v_2, v_42_1);
                            m1v_2 = _max_219;
                        }
                        float v_43_1 = neg;
                        {
                            if (c0_1 >= 88) {
                                {
                                    float _cvt_f32_f16_436;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_436) : "h"((uint16_t)(acc1[22] & 65535)));
                                    v_43_1 = _cvt_f32_f16_436;
                                }
                            }
                            float _max_220 = max_noftz(m0v_1, v_43_1);
                            m0v_1 = _max_220;
                        }
                        float v_44_1 = neg;
                        {
                            if (c0_1 >= 89) {
                                {
                                    float _cvt_f32_f16_441;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_441) : "h"((uint16_t)(acc1[22] >> 16)));
                                    v_44_1 = _cvt_f32_f16_441;
                                }
                            }
                            float _max_222 = max_noftz(m0v_1, v_44_1);
                            m0v_1 = _max_222;
                        }
                        float v_45_1 = neg;
                        {
                            if (c1_1 >= 88) {
                                {
                                    float _cvt_f32_f16_446;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_446) : "h"((uint16_t)(acc1[23] & 65535)));
                                    v_45_1 = _cvt_f32_f16_446;
                                }
                            }
                            float _max_225 = max_noftz(m1v_2, v_45_1);
                            m1v_2 = _max_225;
                        }
                        float v_46_1 = neg;
                        {
                            if (c1_1 >= 89) {
                                {
                                    float _cvt_f32_f16_451;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_451) : "h"((uint16_t)(acc1[23] >> 16)));
                                    v_46_1 = _cvt_f32_f16_451;
                                }
                            }
                            float _max_227 = max_noftz(m1v_2, v_46_1);
                            m1v_2 = _max_227;
                        }
                        float v_47_1 = neg;
                        {
                            if (c0_1 >= 96) {
                                {
                                    float _cvt_f32_f16_452;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_452) : "h"((uint16_t)(acc1[24] & 65535)));
                                    v_47_1 = _cvt_f32_f16_452;
                                }
                            }
                            float _max_228 = max_noftz(m0v_1, v_47_1);
                            m0v_1 = _max_228;
                        }
                        float v_48_1 = neg;
                        {
                            if (c0_1 >= 97) {
                                {
                                    float _cvt_f32_f16_457;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_457) : "h"((uint16_t)(acc1[24] >> 16)));
                                    v_48_1 = _cvt_f32_f16_457;
                                }
                            }
                            float _max_230 = max_noftz(m0v_1, v_48_1);
                            m0v_1 = _max_230;
                        }
                        float v_49_1 = neg;
                        {
                            if (c1_1 >= 96) {
                                {
                                    float _cvt_f32_f16_462;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_462) : "h"((uint16_t)(acc1[25] & 65535)));
                                    v_49_1 = _cvt_f32_f16_462;
                                }
                            }
                            float _max_233 = max_noftz(m1v_2, v_49_1);
                            m1v_2 = _max_233;
                        }
                        float v_50_1 = neg;
                        {
                            if (c1_1 >= 97) {
                                {
                                    float _cvt_f32_f16_467;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_467) : "h"((uint16_t)(acc1[25] >> 16)));
                                    v_50_1 = _cvt_f32_f16_467;
                                }
                            }
                            float _max_235 = max_noftz(m1v_2, v_50_1);
                            m1v_2 = _max_235;
                        }
                        float v_51_1 = neg;
                        {
                            if (c0_1 >= 104) {
                                {
                                    float _cvt_f32_f16_468;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_468) : "h"((uint16_t)(acc1[26] & 65535)));
                                    v_51_1 = _cvt_f32_f16_468;
                                }
                            }
                            float _max_236 = max_noftz(m0v_1, v_51_1);
                            m0v_1 = _max_236;
                        }
                        float v_52_1 = neg;
                        {
                            if (c0_1 >= 105) {
                                {
                                    float _cvt_f32_f16_473;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_473) : "h"((uint16_t)(acc1[26] >> 16)));
                                    v_52_1 = _cvt_f32_f16_473;
                                }
                            }
                            float _max_238 = max_noftz(m0v_1, v_52_1);
                            m0v_1 = _max_238;
                        }
                        float v_53_1 = neg;
                        {
                            if (c1_1 >= 104) {
                                {
                                    float _cvt_f32_f16_478;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_478) : "h"((uint16_t)(acc1[27] & 65535)));
                                    v_53_1 = _cvt_f32_f16_478;
                                }
                            }
                            float _max_241 = max_noftz(m1v_2, v_53_1);
                            m1v_2 = _max_241;
                        }
                        float v_54_1 = neg;
                        {
                            if (c1_1 >= 105) {
                                {
                                    float _cvt_f32_f16_483;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_483) : "h"((uint16_t)(acc1[27] >> 16)));
                                    v_54_1 = _cvt_f32_f16_483;
                                }
                            }
                            float _max_243 = max_noftz(m1v_2, v_54_1);
                            m1v_2 = _max_243;
                        }
                        float v_55_1 = neg;
                        {
                            if (c0_1 >= 112) {
                                {
                                    float _cvt_f32_f16_484;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_484) : "h"((uint16_t)(acc1[28] & 65535)));
                                    v_55_1 = _cvt_f32_f16_484;
                                }
                            }
                            float _max_244 = max_noftz(m0v_1, v_55_1);
                            m0v_1 = _max_244;
                        }
                        float v_56_1 = neg;
                        {
                            if (c0_1 >= 113) {
                                {
                                    float _cvt_f32_f16_489;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_489) : "h"((uint16_t)(acc1[28] >> 16)));
                                    v_56_1 = _cvt_f32_f16_489;
                                }
                            }
                            float _max_246 = max_noftz(m0v_1, v_56_1);
                            m0v_1 = _max_246;
                        }
                        float v_57_1 = neg;
                        {
                            if (c1_1 >= 112) {
                                {
                                    float _cvt_f32_f16_494;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_494) : "h"((uint16_t)(acc1[29] & 65535)));
                                    v_57_1 = _cvt_f32_f16_494;
                                }
                            }
                            float _max_249 = max_noftz(m1v_2, v_57_1);
                            m1v_2 = _max_249;
                        }
                        float v_58_1 = neg;
                        {
                            if (c1_1 >= 113) {
                                {
                                    float _cvt_f32_f16_499;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_499) : "h"((uint16_t)(acc1[29] >> 16)));
                                    v_58_1 = _cvt_f32_f16_499;
                                }
                            }
                            float _max_251 = max_noftz(m1v_2, v_58_1);
                            m1v_2 = _max_251;
                        }
                        float v_59_1 = neg;
                        {
                            if (c0_1 >= 120) {
                                {
                                    float _cvt_f32_f16_500;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_500) : "h"((uint16_t)(acc1[30] & 65535)));
                                    v_59_1 = _cvt_f32_f16_500;
                                }
                            }
                            float _max_252 = max_noftz(m0v_1, v_59_1);
                            m0v_1 = _max_252;
                        }
                        float v_60_1 = neg;
                        {
                            if (c0_1 >= 121) {
                                {
                                    float _cvt_f32_f16_505;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_505) : "h"((uint16_t)(acc1[30] >> 16)));
                                    v_60_1 = _cvt_f32_f16_505;
                                }
                            }
                            float _max_254 = max_noftz(m0v_1, v_60_1);
                            m0v_1 = _max_254;
                        }
                        float v_61_1 = neg;
                        {
                            if (c1_1 >= 120) {
                                {
                                    float _cvt_f32_f16_510;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_510) : "h"((uint16_t)(acc1[31] & 65535)));
                                    v_61_1 = _cvt_f32_f16_510;
                                }
                            }
                            float _max_257 = max_noftz(m1v_2, v_61_1);
                            m1v_2 = _max_257;
                        }
                        float v_62_1 = neg;
                        {
                            if (c1_1 >= 121) {
                                {
                                    float _cvt_f32_f16_515;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_515) : "h"((uint16_t)(acc1[31] >> 16)));
                                    v_62_1 = _cvt_f32_f16_515;
                                }
                            }
                            float _max_259 = max_noftz(m1v_2, v_62_1);
                            m1v_2 = _max_259;
                        }
                    }
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, m0v_1, 1);
                    float _max_260 = max_noftz(m0v_1, _shfl_xor_4);
                    m0v_1 = _max_260;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, m0v_1, 2);
                    float _max_261 = max_noftz(m0v_1, _shfl_xor_5);
                    m0v_1 = _max_261;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, m1v_2, 1);
                    float _max_262 = max_noftz(m1v_2, _shfl_xor_6);
                    m1v_2 = _max_262;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, m1v_2, 2);
                    float _max_263 = max_noftz(m1v_2, _shfl_xor_7);
                    m1v_2 = _max_263;
                    if (cq == 0) {
                        int orow_1 = obase + (split2 + ib * nsplit) * total_q + r0;
                        if (r0 < rows_valid) {
                            out[orow_1] = m0v_1;
                        }
                        if (rows_valid > r0 + 8) {
                            out[orow_1 + 8] = m1v_2;
                        }
                    }
                }
                int il = 2 * npair;
                int sl = il % 4;
                if (n_comp2 - il == 2) {
                    int im = il + 1;
                    int sm = im % 4;
                    mbarrier_wait(full_addr + (sm) * 8, im / 4 & 1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_4 = (((uint64_t)(((k_s_addr + (unsigned int)(sm * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_4 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_4 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1;\n}\n"
                        : "+r"(acc1[0]), "+r"(acc1[1]), "+r"(acc1[2]), "+r"(acc1[3]), "+r"(acc1[4]), "+r"(acc1[5]), "+r"(acc1[6]), "+r"(acc1[7]), "+r"(acc1[8]), "+r"(acc1[9]), "+r"(acc1[10]), "+r"(acc1[11]), "+r"(acc1[12]), "+r"(acc1[13]), "+r"(acc1[14]), "+r"(acc1[15]), "+r"(acc1[16]), "+r"(acc1[17]), "+r"(acc1[18]), "+r"(acc1[19]), "+r"(acc1[20]), "+r"(acc1[21]), "+r"(acc1[22]), "+r"(acc1[23]), "+r"(acc1[24]), "+r"(acc1[25]), "+r"(acc1[26]), "+r"(acc1[27]), "+r"(acc1[28]), "+r"(acc1[29]), "+r"(acc1[30]), "+r"(acc1[31])
                        : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_4 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(empty_addr + (sl) * 8);
                    }
                    int k0_1 = (split2 + il * nsplit) * 128;
                    float m0v_2 = neg;
                    float m1v_1 = neg;
                    int need_mask_1 = 0;
                    if (mask_thr < k0_1 + 127) {
                        need_mask_1 = 1;
                    }
                    if (need_mask_1 == 0) {
                        unsigned int w0_2 = acc[0];
                        unsigned int w1_2 = acc[1];
                        uint32_t _f16x2_max_60;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_60) : "r"(w0_2), "r"(acc[2]));
                        w0_2 = _f16x2_max_60;
                        uint32_t _f16x2_max_61;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_61) : "r"(w1_2), "r"(acc[3]));
                        w1_2 = _f16x2_max_61;
                        uint32_t _f16x2_max_62;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_62) : "r"(w0_2), "r"(acc[4]));
                        w0_2 = _f16x2_max_62;
                        uint32_t _f16x2_max_63;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_63) : "r"(w1_2), "r"(acc[5]));
                        w1_2 = _f16x2_max_63;
                        uint32_t _f16x2_max_64;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_64) : "r"(w0_2), "r"(acc[6]));
                        w0_2 = _f16x2_max_64;
                        uint32_t _f16x2_max_65;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_65) : "r"(w1_2), "r"(acc[7]));
                        w1_2 = _f16x2_max_65;
                        uint32_t _f16x2_max_66;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_66) : "r"(w0_2), "r"(acc[8]));
                        w0_2 = _f16x2_max_66;
                        uint32_t _f16x2_max_67;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_67) : "r"(w1_2), "r"(acc[9]));
                        w1_2 = _f16x2_max_67;
                        uint32_t _f16x2_max_68;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_68) : "r"(w0_2), "r"(acc[10]));
                        w0_2 = _f16x2_max_68;
                        uint32_t _f16x2_max_69;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_69) : "r"(w1_2), "r"(acc[11]));
                        w1_2 = _f16x2_max_69;
                        uint32_t _f16x2_max_70;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_70) : "r"(w0_2), "r"(acc[12]));
                        w0_2 = _f16x2_max_70;
                        uint32_t _f16x2_max_71;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_71) : "r"(w1_2), "r"(acc[13]));
                        w1_2 = _f16x2_max_71;
                        uint32_t _f16x2_max_72;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_72) : "r"(w0_2), "r"(acc[14]));
                        w0_2 = _f16x2_max_72;
                        uint32_t _f16x2_max_73;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_73) : "r"(w1_2), "r"(acc[15]));
                        w1_2 = _f16x2_max_73;
                        uint32_t _f16x2_max_74;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_74) : "r"(w0_2), "r"(acc[16]));
                        w0_2 = _f16x2_max_74;
                        uint32_t _f16x2_max_75;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_75) : "r"(w1_2), "r"(acc[17]));
                        w1_2 = _f16x2_max_75;
                        uint32_t _f16x2_max_76;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_76) : "r"(w0_2), "r"(acc[18]));
                        w0_2 = _f16x2_max_76;
                        uint32_t _f16x2_max_77;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_77) : "r"(w1_2), "r"(acc[19]));
                        w1_2 = _f16x2_max_77;
                        uint32_t _f16x2_max_78;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_78) : "r"(w0_2), "r"(acc[20]));
                        w0_2 = _f16x2_max_78;
                        uint32_t _f16x2_max_79;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_79) : "r"(w1_2), "r"(acc[21]));
                        w1_2 = _f16x2_max_79;
                        uint32_t _f16x2_max_80;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_80) : "r"(w0_2), "r"(acc[22]));
                        w0_2 = _f16x2_max_80;
                        uint32_t _f16x2_max_81;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_81) : "r"(w1_2), "r"(acc[23]));
                        w1_2 = _f16x2_max_81;
                        uint32_t _f16x2_max_82;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_82) : "r"(w0_2), "r"(acc[24]));
                        w0_2 = _f16x2_max_82;
                        uint32_t _f16x2_max_83;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_83) : "r"(w1_2), "r"(acc[25]));
                        w1_2 = _f16x2_max_83;
                        uint32_t _f16x2_max_84;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_84) : "r"(w0_2), "r"(acc[26]));
                        w0_2 = _f16x2_max_84;
                        uint32_t _f16x2_max_85;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_85) : "r"(w1_2), "r"(acc[27]));
                        w1_2 = _f16x2_max_85;
                        uint32_t _f16x2_max_86;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_86) : "r"(w0_2), "r"(acc[28]));
                        w0_2 = _f16x2_max_86;
                        uint32_t _f16x2_max_87;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_87) : "r"(w1_2), "r"(acc[29]));
                        w1_2 = _f16x2_max_87;
                        uint32_t _f16x2_max_88;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_88) : "r"(w0_2), "r"(acc[30]));
                        w0_2 = _f16x2_max_88;
                        uint32_t _f16x2_max_89;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_89) : "r"(w1_2), "r"(acc[31]));
                        w1_2 = _f16x2_max_89;
                        uint16_t _f16_max_4;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_4) : "h"((uint16_t)(w0_2 & 65535)), "h"((uint16_t)(w0_2 >> 16)));
                        float _cvt_f32_f16_516;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_516) : "h"((uint16_t)(_f16_max_4)));
                        m0v_2 = _cvt_f32_f16_516;
                        uint16_t _f16_max_5;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_5) : "h"((uint16_t)(w1_2 & 65535)), "h"((uint16_t)(w1_2 >> 16)));
                        float _cvt_f32_f16_517;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_517) : "h"((uint16_t)(_f16_max_5)));
                        m1v_1 = _cvt_f32_f16_517;
                    }
                    if (need_mask_1 != 0) {
                        int c0_2 = lim0 - k0_1 - 2 * cq;
                        int c1_2 = lim1 - k0_1 - 2 * cq;
                        float v_64 = neg;
                        {
                            if (c0_2 >= 0) {
                                {
                                    float _cvt_f32_f16_518;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_518) : "h"((uint16_t)(acc[0] & 65535)));
                                    v_64 = _cvt_f32_f16_518;
                                }
                            }
                            float _max_264 = max_noftz(m0v_2, v_64);
                            m0v_2 = _max_264;
                        }
                        float v_0_2 = neg;
                        {
                            if (c0_2 >= 1) {
                                {
                                    float _cvt_f32_f16_523;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_523) : "h"((uint16_t)(acc[0] >> 16)));
                                    v_0_2 = _cvt_f32_f16_523;
                                }
                            }
                            float _max_266 = max_noftz(m0v_2, v_0_2);
                            m0v_2 = _max_266;
                        }
                        float v_1_2 = neg;
                        {
                            if (c1_2 >= 0) {
                                {
                                    float _cvt_f32_f16_528;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_528) : "h"((uint16_t)(acc[1] & 65535)));
                                    v_1_2 = _cvt_f32_f16_528;
                                }
                            }
                            float _max_269 = max_noftz(m1v_1, v_1_2);
                            m1v_1 = _max_269;
                        }
                        float v_2_2 = neg;
                        {
                            if (c1_2 >= 1) {
                                {
                                    float _cvt_f32_f16_533;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_533) : "h"((uint16_t)(acc[1] >> 16)));
                                    v_2_2 = _cvt_f32_f16_533;
                                }
                            }
                            float _max_271 = max_noftz(m1v_1, v_2_2);
                            m1v_1 = _max_271;
                        }
                        float v_3_2 = neg;
                        {
                            if (c0_2 >= 8) {
                                {
                                    float _cvt_f32_f16_534;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_534) : "h"((uint16_t)(acc[2] & 65535)));
                                    v_3_2 = _cvt_f32_f16_534;
                                }
                            }
                            float _max_272 = max_noftz(m0v_2, v_3_2);
                            m0v_2 = _max_272;
                        }
                        float v_4_2 = neg;
                        {
                            if (c0_2 >= 9) {
                                {
                                    float _cvt_f32_f16_539;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_539) : "h"((uint16_t)(acc[2] >> 16)));
                                    v_4_2 = _cvt_f32_f16_539;
                                }
                            }
                            float _max_274 = max_noftz(m0v_2, v_4_2);
                            m0v_2 = _max_274;
                        }
                        float v_5_2 = neg;
                        {
                            if (c1_2 >= 8) {
                                {
                                    float _cvt_f32_f16_544;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_544) : "h"((uint16_t)(acc[3] & 65535)));
                                    v_5_2 = _cvt_f32_f16_544;
                                }
                            }
                            float _max_277 = max_noftz(m1v_1, v_5_2);
                            m1v_1 = _max_277;
                        }
                        float v_6_2 = neg;
                        {
                            if (c1_2 >= 9) {
                                {
                                    float _cvt_f32_f16_549;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_549) : "h"((uint16_t)(acc[3] >> 16)));
                                    v_6_2 = _cvt_f32_f16_549;
                                }
                            }
                            float _max_279 = max_noftz(m1v_1, v_6_2);
                            m1v_1 = _max_279;
                        }
                        float v_7_2 = neg;
                        {
                            if (c0_2 >= 16) {
                                {
                                    float _cvt_f32_f16_550;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_550) : "h"((uint16_t)(acc[4] & 65535)));
                                    v_7_2 = _cvt_f32_f16_550;
                                }
                            }
                            float _max_280 = max_noftz(m0v_2, v_7_2);
                            m0v_2 = _max_280;
                        }
                        float v_8_2 = neg;
                        {
                            if (c0_2 >= 17) {
                                {
                                    float _cvt_f32_f16_555;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_555) : "h"((uint16_t)(acc[4] >> 16)));
                                    v_8_2 = _cvt_f32_f16_555;
                                }
                            }
                            float _max_282 = max_noftz(m0v_2, v_8_2);
                            m0v_2 = _max_282;
                        }
                        float v_9_2 = neg;
                        {
                            if (c1_2 >= 16) {
                                {
                                    float _cvt_f32_f16_560;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_560) : "h"((uint16_t)(acc[5] & 65535)));
                                    v_9_2 = _cvt_f32_f16_560;
                                }
                            }
                            float _max_285 = max_noftz(m1v_1, v_9_2);
                            m1v_1 = _max_285;
                        }
                        float v_10_2 = neg;
                        {
                            if (c1_2 >= 17) {
                                {
                                    float _cvt_f32_f16_565;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_565) : "h"((uint16_t)(acc[5] >> 16)));
                                    v_10_2 = _cvt_f32_f16_565;
                                }
                            }
                            float _max_287 = max_noftz(m1v_1, v_10_2);
                            m1v_1 = _max_287;
                        }
                        float v_11_2 = neg;
                        {
                            if (c0_2 >= 24) {
                                {
                                    float _cvt_f32_f16_566;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_566) : "h"((uint16_t)(acc[6] & 65535)));
                                    v_11_2 = _cvt_f32_f16_566;
                                }
                            }
                            float _max_288 = max_noftz(m0v_2, v_11_2);
                            m0v_2 = _max_288;
                        }
                        float v_12_2 = neg;
                        {
                            if (c0_2 >= 25) {
                                {
                                    float _cvt_f32_f16_571;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_571) : "h"((uint16_t)(acc[6] >> 16)));
                                    v_12_2 = _cvt_f32_f16_571;
                                }
                            }
                            float _max_290 = max_noftz(m0v_2, v_12_2);
                            m0v_2 = _max_290;
                        }
                        float v_13_2 = neg;
                        {
                            if (c1_2 >= 24) {
                                {
                                    float _cvt_f32_f16_576;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_576) : "h"((uint16_t)(acc[7] & 65535)));
                                    v_13_2 = _cvt_f32_f16_576;
                                }
                            }
                            float _max_293 = max_noftz(m1v_1, v_13_2);
                            m1v_1 = _max_293;
                        }
                        float v_14_2 = neg;
                        {
                            if (c1_2 >= 25) {
                                {
                                    float _cvt_f32_f16_581;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_581) : "h"((uint16_t)(acc[7] >> 16)));
                                    v_14_2 = _cvt_f32_f16_581;
                                }
                            }
                            float _max_295 = max_noftz(m1v_1, v_14_2);
                            m1v_1 = _max_295;
                        }
                        float v_15_2 = neg;
                        {
                            if (c0_2 >= 32) {
                                {
                                    float _cvt_f32_f16_582;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_582) : "h"((uint16_t)(acc[8] & 65535)));
                                    v_15_2 = _cvt_f32_f16_582;
                                }
                            }
                            float _max_296 = max_noftz(m0v_2, v_15_2);
                            m0v_2 = _max_296;
                        }
                        float v_16_2 = neg;
                        {
                            if (c0_2 >= 33) {
                                {
                                    float _cvt_f32_f16_587;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_587) : "h"((uint16_t)(acc[8] >> 16)));
                                    v_16_2 = _cvt_f32_f16_587;
                                }
                            }
                            float _max_298 = max_noftz(m0v_2, v_16_2);
                            m0v_2 = _max_298;
                        }
                        float v_17_2 = neg;
                        {
                            if (c1_2 >= 32) {
                                {
                                    float _cvt_f32_f16_592;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_592) : "h"((uint16_t)(acc[9] & 65535)));
                                    v_17_2 = _cvt_f32_f16_592;
                                }
                            }
                            float _max_301 = max_noftz(m1v_1, v_17_2);
                            m1v_1 = _max_301;
                        }
                        float v_18_2 = neg;
                        {
                            if (c1_2 >= 33) {
                                {
                                    float _cvt_f32_f16_597;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_597) : "h"((uint16_t)(acc[9] >> 16)));
                                    v_18_2 = _cvt_f32_f16_597;
                                }
                            }
                            float _max_303 = max_noftz(m1v_1, v_18_2);
                            m1v_1 = _max_303;
                        }
                        float v_19_2 = neg;
                        {
                            if (c0_2 >= 40) {
                                {
                                    float _cvt_f32_f16_598;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_598) : "h"((uint16_t)(acc[10] & 65535)));
                                    v_19_2 = _cvt_f32_f16_598;
                                }
                            }
                            float _max_304 = max_noftz(m0v_2, v_19_2);
                            m0v_2 = _max_304;
                        }
                        float v_20_2 = neg;
                        {
                            if (c0_2 >= 41) {
                                {
                                    float _cvt_f32_f16_603;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_603) : "h"((uint16_t)(acc[10] >> 16)));
                                    v_20_2 = _cvt_f32_f16_603;
                                }
                            }
                            float _max_306 = max_noftz(m0v_2, v_20_2);
                            m0v_2 = _max_306;
                        }
                        float v_21_2 = neg;
                        {
                            if (c1_2 >= 40) {
                                {
                                    float _cvt_f32_f16_608;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_608) : "h"((uint16_t)(acc[11] & 65535)));
                                    v_21_2 = _cvt_f32_f16_608;
                                }
                            }
                            float _max_309 = max_noftz(m1v_1, v_21_2);
                            m1v_1 = _max_309;
                        }
                        float v_22_2 = neg;
                        {
                            if (c1_2 >= 41) {
                                {
                                    float _cvt_f32_f16_613;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_613) : "h"((uint16_t)(acc[11] >> 16)));
                                    v_22_2 = _cvt_f32_f16_613;
                                }
                            }
                            float _max_311 = max_noftz(m1v_1, v_22_2);
                            m1v_1 = _max_311;
                        }
                        float v_23_2 = neg;
                        {
                            if (c0_2 >= 48) {
                                {
                                    float _cvt_f32_f16_614;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_614) : "h"((uint16_t)(acc[12] & 65535)));
                                    v_23_2 = _cvt_f32_f16_614;
                                }
                            }
                            float _max_312 = max_noftz(m0v_2, v_23_2);
                            m0v_2 = _max_312;
                        }
                        float v_24_2 = neg;
                        {
                            if (c0_2 >= 49) {
                                {
                                    float _cvt_f32_f16_619;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_619) : "h"((uint16_t)(acc[12] >> 16)));
                                    v_24_2 = _cvt_f32_f16_619;
                                }
                            }
                            float _max_314 = max_noftz(m0v_2, v_24_2);
                            m0v_2 = _max_314;
                        }
                        float v_25_2 = neg;
                        {
                            if (c1_2 >= 48) {
                                {
                                    float _cvt_f32_f16_624;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_624) : "h"((uint16_t)(acc[13] & 65535)));
                                    v_25_2 = _cvt_f32_f16_624;
                                }
                            }
                            float _max_317 = max_noftz(m1v_1, v_25_2);
                            m1v_1 = _max_317;
                        }
                        float v_26_2 = neg;
                        {
                            if (c1_2 >= 49) {
                                {
                                    float _cvt_f32_f16_629;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_629) : "h"((uint16_t)(acc[13] >> 16)));
                                    v_26_2 = _cvt_f32_f16_629;
                                }
                            }
                            float _max_319 = max_noftz(m1v_1, v_26_2);
                            m1v_1 = _max_319;
                        }
                        float v_27_2 = neg;
                        {
                            if (c0_2 >= 56) {
                                {
                                    float _cvt_f32_f16_630;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_630) : "h"((uint16_t)(acc[14] & 65535)));
                                    v_27_2 = _cvt_f32_f16_630;
                                }
                            }
                            float _max_320 = max_noftz(m0v_2, v_27_2);
                            m0v_2 = _max_320;
                        }
                        float v_28_2 = neg;
                        {
                            if (c0_2 >= 57) {
                                {
                                    float _cvt_f32_f16_635;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_635) : "h"((uint16_t)(acc[14] >> 16)));
                                    v_28_2 = _cvt_f32_f16_635;
                                }
                            }
                            float _max_322 = max_noftz(m0v_2, v_28_2);
                            m0v_2 = _max_322;
                        }
                        float v_29_2 = neg;
                        {
                            if (c1_2 >= 56) {
                                {
                                    float _cvt_f32_f16_640;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_640) : "h"((uint16_t)(acc[15] & 65535)));
                                    v_29_2 = _cvt_f32_f16_640;
                                }
                            }
                            float _max_325 = max_noftz(m1v_1, v_29_2);
                            m1v_1 = _max_325;
                        }
                        float v_30_2 = neg;
                        {
                            if (c1_2 >= 57) {
                                {
                                    float _cvt_f32_f16_645;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_645) : "h"((uint16_t)(acc[15] >> 16)));
                                    v_30_2 = _cvt_f32_f16_645;
                                }
                            }
                            float _max_327 = max_noftz(m1v_1, v_30_2);
                            m1v_1 = _max_327;
                        }
                        float v_31_2 = neg;
                        {
                            if (c0_2 >= 64) {
                                {
                                    float _cvt_f32_f16_646;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_646) : "h"((uint16_t)(acc[16] & 65535)));
                                    v_31_2 = _cvt_f32_f16_646;
                                }
                            }
                            float _max_328 = max_noftz(m0v_2, v_31_2);
                            m0v_2 = _max_328;
                        }
                        float v_32_2 = neg;
                        {
                            if (c0_2 >= 65) {
                                {
                                    float _cvt_f32_f16_651;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_651) : "h"((uint16_t)(acc[16] >> 16)));
                                    v_32_2 = _cvt_f32_f16_651;
                                }
                            }
                            float _max_330 = max_noftz(m0v_2, v_32_2);
                            m0v_2 = _max_330;
                        }
                        float v_33_2 = neg;
                        {
                            if (c1_2 >= 64) {
                                {
                                    float _cvt_f32_f16_656;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_656) : "h"((uint16_t)(acc[17] & 65535)));
                                    v_33_2 = _cvt_f32_f16_656;
                                }
                            }
                            float _max_333 = max_noftz(m1v_1, v_33_2);
                            m1v_1 = _max_333;
                        }
                        float v_34_2 = neg;
                        {
                            if (c1_2 >= 65) {
                                {
                                    float _cvt_f32_f16_661;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_661) : "h"((uint16_t)(acc[17] >> 16)));
                                    v_34_2 = _cvt_f32_f16_661;
                                }
                            }
                            float _max_335 = max_noftz(m1v_1, v_34_2);
                            m1v_1 = _max_335;
                        }
                        float v_35_2 = neg;
                        {
                            if (c0_2 >= 72) {
                                {
                                    float _cvt_f32_f16_662;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_662) : "h"((uint16_t)(acc[18] & 65535)));
                                    v_35_2 = _cvt_f32_f16_662;
                                }
                            }
                            float _max_336 = max_noftz(m0v_2, v_35_2);
                            m0v_2 = _max_336;
                        }
                        float v_36_2 = neg;
                        {
                            if (c0_2 >= 73) {
                                {
                                    float _cvt_f32_f16_667;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_667) : "h"((uint16_t)(acc[18] >> 16)));
                                    v_36_2 = _cvt_f32_f16_667;
                                }
                            }
                            float _max_338 = max_noftz(m0v_2, v_36_2);
                            m0v_2 = _max_338;
                        }
                        float v_37_2 = neg;
                        {
                            if (c1_2 >= 72) {
                                {
                                    float _cvt_f32_f16_672;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_672) : "h"((uint16_t)(acc[19] & 65535)));
                                    v_37_2 = _cvt_f32_f16_672;
                                }
                            }
                            float _max_341 = max_noftz(m1v_1, v_37_2);
                            m1v_1 = _max_341;
                        }
                        float v_38_2 = neg;
                        {
                            if (c1_2 >= 73) {
                                {
                                    float _cvt_f32_f16_677;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_677) : "h"((uint16_t)(acc[19] >> 16)));
                                    v_38_2 = _cvt_f32_f16_677;
                                }
                            }
                            float _max_343 = max_noftz(m1v_1, v_38_2);
                            m1v_1 = _max_343;
                        }
                        float v_39_2 = neg;
                        {
                            if (c0_2 >= 80) {
                                {
                                    float _cvt_f32_f16_678;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_678) : "h"((uint16_t)(acc[20] & 65535)));
                                    v_39_2 = _cvt_f32_f16_678;
                                }
                            }
                            float _max_344 = max_noftz(m0v_2, v_39_2);
                            m0v_2 = _max_344;
                        }
                        float v_40_2 = neg;
                        {
                            if (c0_2 >= 81) {
                                {
                                    float _cvt_f32_f16_683;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_683) : "h"((uint16_t)(acc[20] >> 16)));
                                    v_40_2 = _cvt_f32_f16_683;
                                }
                            }
                            float _max_346 = max_noftz(m0v_2, v_40_2);
                            m0v_2 = _max_346;
                        }
                        float v_41_2 = neg;
                        {
                            if (c1_2 >= 80) {
                                {
                                    float _cvt_f32_f16_688;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_688) : "h"((uint16_t)(acc[21] & 65535)));
                                    v_41_2 = _cvt_f32_f16_688;
                                }
                            }
                            float _max_349 = max_noftz(m1v_1, v_41_2);
                            m1v_1 = _max_349;
                        }
                        float v_42_2 = neg;
                        {
                            if (c1_2 >= 81) {
                                {
                                    float _cvt_f32_f16_693;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_693) : "h"((uint16_t)(acc[21] >> 16)));
                                    v_42_2 = _cvt_f32_f16_693;
                                }
                            }
                            float _max_351 = max_noftz(m1v_1, v_42_2);
                            m1v_1 = _max_351;
                        }
                        float v_43_2 = neg;
                        {
                            if (c0_2 >= 88) {
                                {
                                    float _cvt_f32_f16_694;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_694) : "h"((uint16_t)(acc[22] & 65535)));
                                    v_43_2 = _cvt_f32_f16_694;
                                }
                            }
                            float _max_352 = max_noftz(m0v_2, v_43_2);
                            m0v_2 = _max_352;
                        }
                        float v_44_2 = neg;
                        {
                            if (c0_2 >= 89) {
                                {
                                    float _cvt_f32_f16_699;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_699) : "h"((uint16_t)(acc[22] >> 16)));
                                    v_44_2 = _cvt_f32_f16_699;
                                }
                            }
                            float _max_354 = max_noftz(m0v_2, v_44_2);
                            m0v_2 = _max_354;
                        }
                        float v_45_2 = neg;
                        {
                            if (c1_2 >= 88) {
                                {
                                    float _cvt_f32_f16_704;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_704) : "h"((uint16_t)(acc[23] & 65535)));
                                    v_45_2 = _cvt_f32_f16_704;
                                }
                            }
                            float _max_357 = max_noftz(m1v_1, v_45_2);
                            m1v_1 = _max_357;
                        }
                        float v_46_2 = neg;
                        {
                            if (c1_2 >= 89) {
                                {
                                    float _cvt_f32_f16_709;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_709) : "h"((uint16_t)(acc[23] >> 16)));
                                    v_46_2 = _cvt_f32_f16_709;
                                }
                            }
                            float _max_359 = max_noftz(m1v_1, v_46_2);
                            m1v_1 = _max_359;
                        }
                        float v_47_2 = neg;
                        {
                            if (c0_2 >= 96) {
                                {
                                    float _cvt_f32_f16_710;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_710) : "h"((uint16_t)(acc[24] & 65535)));
                                    v_47_2 = _cvt_f32_f16_710;
                                }
                            }
                            float _max_360 = max_noftz(m0v_2, v_47_2);
                            m0v_2 = _max_360;
                        }
                        float v_48_2 = neg;
                        {
                            if (c0_2 >= 97) {
                                {
                                    float _cvt_f32_f16_715;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_715) : "h"((uint16_t)(acc[24] >> 16)));
                                    v_48_2 = _cvt_f32_f16_715;
                                }
                            }
                            float _max_362 = max_noftz(m0v_2, v_48_2);
                            m0v_2 = _max_362;
                        }
                        float v_49_2 = neg;
                        {
                            if (c1_2 >= 96) {
                                {
                                    float _cvt_f32_f16_720;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_720) : "h"((uint16_t)(acc[25] & 65535)));
                                    v_49_2 = _cvt_f32_f16_720;
                                }
                            }
                            float _max_365 = max_noftz(m1v_1, v_49_2);
                            m1v_1 = _max_365;
                        }
                        float v_50_2 = neg;
                        {
                            if (c1_2 >= 97) {
                                {
                                    float _cvt_f32_f16_725;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_725) : "h"((uint16_t)(acc[25] >> 16)));
                                    v_50_2 = _cvt_f32_f16_725;
                                }
                            }
                            float _max_367 = max_noftz(m1v_1, v_50_2);
                            m1v_1 = _max_367;
                        }
                        float v_51_2 = neg;
                        {
                            if (c0_2 >= 104) {
                                {
                                    float _cvt_f32_f16_726;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_726) : "h"((uint16_t)(acc[26] & 65535)));
                                    v_51_2 = _cvt_f32_f16_726;
                                }
                            }
                            float _max_368 = max_noftz(m0v_2, v_51_2);
                            m0v_2 = _max_368;
                        }
                        float v_52_2 = neg;
                        {
                            if (c0_2 >= 105) {
                                {
                                    float _cvt_f32_f16_731;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_731) : "h"((uint16_t)(acc[26] >> 16)));
                                    v_52_2 = _cvt_f32_f16_731;
                                }
                            }
                            float _max_370 = max_noftz(m0v_2, v_52_2);
                            m0v_2 = _max_370;
                        }
                        float v_53_2 = neg;
                        {
                            if (c1_2 >= 104) {
                                {
                                    float _cvt_f32_f16_736;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_736) : "h"((uint16_t)(acc[27] & 65535)));
                                    v_53_2 = _cvt_f32_f16_736;
                                }
                            }
                            float _max_373 = max_noftz(m1v_1, v_53_2);
                            m1v_1 = _max_373;
                        }
                        float v_54_2 = neg;
                        {
                            if (c1_2 >= 105) {
                                {
                                    float _cvt_f32_f16_741;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_741) : "h"((uint16_t)(acc[27] >> 16)));
                                    v_54_2 = _cvt_f32_f16_741;
                                }
                            }
                            float _max_375 = max_noftz(m1v_1, v_54_2);
                            m1v_1 = _max_375;
                        }
                        float v_55_2 = neg;
                        {
                            if (c0_2 >= 112) {
                                {
                                    float _cvt_f32_f16_742;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_742) : "h"((uint16_t)(acc[28] & 65535)));
                                    v_55_2 = _cvt_f32_f16_742;
                                }
                            }
                            float _max_376 = max_noftz(m0v_2, v_55_2);
                            m0v_2 = _max_376;
                        }
                        float v_56_2 = neg;
                        {
                            if (c0_2 >= 113) {
                                {
                                    float _cvt_f32_f16_747;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_747) : "h"((uint16_t)(acc[28] >> 16)));
                                    v_56_2 = _cvt_f32_f16_747;
                                }
                            }
                            float _max_378 = max_noftz(m0v_2, v_56_2);
                            m0v_2 = _max_378;
                        }
                        float v_57_2 = neg;
                        {
                            if (c1_2 >= 112) {
                                {
                                    float _cvt_f32_f16_752;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_752) : "h"((uint16_t)(acc[29] & 65535)));
                                    v_57_2 = _cvt_f32_f16_752;
                                }
                            }
                            float _max_381 = max_noftz(m1v_1, v_57_2);
                            m1v_1 = _max_381;
                        }
                        float v_58_2 = neg;
                        {
                            if (c1_2 >= 113) {
                                {
                                    float _cvt_f32_f16_757;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_757) : "h"((uint16_t)(acc[29] >> 16)));
                                    v_58_2 = _cvt_f32_f16_757;
                                }
                            }
                            float _max_383 = max_noftz(m1v_1, v_58_2);
                            m1v_1 = _max_383;
                        }
                        float v_59_2 = neg;
                        {
                            if (c0_2 >= 120) {
                                {
                                    float _cvt_f32_f16_758;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_758) : "h"((uint16_t)(acc[30] & 65535)));
                                    v_59_2 = _cvt_f32_f16_758;
                                }
                            }
                            float _max_384 = max_noftz(m0v_2, v_59_2);
                            m0v_2 = _max_384;
                        }
                        float v_60_2 = neg;
                        {
                            if (c0_2 >= 121) {
                                {
                                    float _cvt_f32_f16_763;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_763) : "h"((uint16_t)(acc[30] >> 16)));
                                    v_60_2 = _cvt_f32_f16_763;
                                }
                            }
                            float _max_386 = max_noftz(m0v_2, v_60_2);
                            m0v_2 = _max_386;
                        }
                        float v_61_2 = neg;
                        {
                            if (c1_2 >= 120) {
                                {
                                    float _cvt_f32_f16_768;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_768) : "h"((uint16_t)(acc[31] & 65535)));
                                    v_61_2 = _cvt_f32_f16_768;
                                }
                            }
                            float _max_389 = max_noftz(m1v_1, v_61_2);
                            m1v_1 = _max_389;
                        }
                        float v_62_2 = neg;
                        {
                            if (c1_2 >= 121) {
                                {
                                    float _cvt_f32_f16_773;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_773) : "h"((uint16_t)(acc[31] >> 16)));
                                    v_62_2 = _cvt_f32_f16_773;
                                }
                            }
                            float _max_391 = max_noftz(m1v_1, v_62_2);
                            m1v_1 = _max_391;
                        }
                    }
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, m0v_2, 1);
                    float _max_392 = max_noftz(m0v_2, _shfl_xor_8);
                    m0v_2 = _max_392;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, m0v_2, 2);
                    float _max_393 = max_noftz(m0v_2, _shfl_xor_9);
                    m0v_2 = _max_393;
                    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, m1v_1, 1);
                    float _max_394 = max_noftz(m1v_1, _shfl_xor_10);
                    m1v_1 = _max_394;
                    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, m1v_1, 2);
                    float _max_395 = max_noftz(m1v_1, _shfl_xor_11);
                    m1v_1 = _max_395;
                    if (cq == 0) {
                        int orow_2 = obase + (split2 + il * nsplit) * total_q + r0;
                        if (r0 < rows_valid) {
                            out[orow_2] = m0v_2;
                        }
                        if (rows_valid > r0 + 8) {
                            out[orow_2 + 8] = m1v_1;
                        }
                    }
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(empty_addr + (sm) * 8);
                    }
                    int k0_0_1 = (split2 + im * nsplit) * 128;
                    float m0v_1_1 = neg;
                    float m1v_2_1 = neg;
                    int need_mask_3_1 = 0;
                    if (mask_thr < k0_0_1 + 127) {
                        need_mask_3_1 = 1;
                    }
                    if (need_mask_3_1 == 0) {
                        unsigned int w0_3 = acc1[0];
                        unsigned int w1_3 = acc1[1];
                        uint32_t _f16x2_max_90;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_90) : "r"(w0_3), "r"(acc1[2]));
                        w0_3 = _f16x2_max_90;
                        uint32_t _f16x2_max_91;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_91) : "r"(w1_3), "r"(acc1[3]));
                        w1_3 = _f16x2_max_91;
                        uint32_t _f16x2_max_92;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_92) : "r"(w0_3), "r"(acc1[4]));
                        w0_3 = _f16x2_max_92;
                        uint32_t _f16x2_max_93;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_93) : "r"(w1_3), "r"(acc1[5]));
                        w1_3 = _f16x2_max_93;
                        uint32_t _f16x2_max_94;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_94) : "r"(w0_3), "r"(acc1[6]));
                        w0_3 = _f16x2_max_94;
                        uint32_t _f16x2_max_95;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_95) : "r"(w1_3), "r"(acc1[7]));
                        w1_3 = _f16x2_max_95;
                        uint32_t _f16x2_max_96;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_96) : "r"(w0_3), "r"(acc1[8]));
                        w0_3 = _f16x2_max_96;
                        uint32_t _f16x2_max_97;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_97) : "r"(w1_3), "r"(acc1[9]));
                        w1_3 = _f16x2_max_97;
                        uint32_t _f16x2_max_98;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_98) : "r"(w0_3), "r"(acc1[10]));
                        w0_3 = _f16x2_max_98;
                        uint32_t _f16x2_max_99;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_99) : "r"(w1_3), "r"(acc1[11]));
                        w1_3 = _f16x2_max_99;
                        uint32_t _f16x2_max_100;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_100) : "r"(w0_3), "r"(acc1[12]));
                        w0_3 = _f16x2_max_100;
                        uint32_t _f16x2_max_101;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_101) : "r"(w1_3), "r"(acc1[13]));
                        w1_3 = _f16x2_max_101;
                        uint32_t _f16x2_max_102;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_102) : "r"(w0_3), "r"(acc1[14]));
                        w0_3 = _f16x2_max_102;
                        uint32_t _f16x2_max_103;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_103) : "r"(w1_3), "r"(acc1[15]));
                        w1_3 = _f16x2_max_103;
                        uint32_t _f16x2_max_104;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_104) : "r"(w0_3), "r"(acc1[16]));
                        w0_3 = _f16x2_max_104;
                        uint32_t _f16x2_max_105;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_105) : "r"(w1_3), "r"(acc1[17]));
                        w1_3 = _f16x2_max_105;
                        uint32_t _f16x2_max_106;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_106) : "r"(w0_3), "r"(acc1[18]));
                        w0_3 = _f16x2_max_106;
                        uint32_t _f16x2_max_107;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_107) : "r"(w1_3), "r"(acc1[19]));
                        w1_3 = _f16x2_max_107;
                        uint32_t _f16x2_max_108;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_108) : "r"(w0_3), "r"(acc1[20]));
                        w0_3 = _f16x2_max_108;
                        uint32_t _f16x2_max_109;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_109) : "r"(w1_3), "r"(acc1[21]));
                        w1_3 = _f16x2_max_109;
                        uint32_t _f16x2_max_110;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_110) : "r"(w0_3), "r"(acc1[22]));
                        w0_3 = _f16x2_max_110;
                        uint32_t _f16x2_max_111;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_111) : "r"(w1_3), "r"(acc1[23]));
                        w1_3 = _f16x2_max_111;
                        uint32_t _f16x2_max_112;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_112) : "r"(w0_3), "r"(acc1[24]));
                        w0_3 = _f16x2_max_112;
                        uint32_t _f16x2_max_113;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_113) : "r"(w1_3), "r"(acc1[25]));
                        w1_3 = _f16x2_max_113;
                        uint32_t _f16x2_max_114;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_114) : "r"(w0_3), "r"(acc1[26]));
                        w0_3 = _f16x2_max_114;
                        uint32_t _f16x2_max_115;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_115) : "r"(w1_3), "r"(acc1[27]));
                        w1_3 = _f16x2_max_115;
                        uint32_t _f16x2_max_116;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_116) : "r"(w0_3), "r"(acc1[28]));
                        w0_3 = _f16x2_max_116;
                        uint32_t _f16x2_max_117;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_117) : "r"(w1_3), "r"(acc1[29]));
                        w1_3 = _f16x2_max_117;
                        uint32_t _f16x2_max_118;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_118) : "r"(w0_3), "r"(acc1[30]));
                        w0_3 = _f16x2_max_118;
                        uint32_t _f16x2_max_119;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_119) : "r"(w1_3), "r"(acc1[31]));
                        w1_3 = _f16x2_max_119;
                        uint16_t _f16_max_6;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_6) : "h"((uint16_t)(w0_3 & 65535)), "h"((uint16_t)(w0_3 >> 16)));
                        float _cvt_f32_f16_774;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_774) : "h"((uint16_t)(_f16_max_6)));
                        m0v_1_1 = _cvt_f32_f16_774;
                        uint16_t _f16_max_7;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_7) : "h"((uint16_t)(w1_3 & 65535)), "h"((uint16_t)(w1_3 >> 16)));
                        float _cvt_f32_f16_775;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_775) : "h"((uint16_t)(_f16_max_7)));
                        m1v_2_1 = _cvt_f32_f16_775;
                    }
                    if (need_mask_3_1 != 0) {
                        int c0_3 = lim0 - k0_0_1 - 2 * cq;
                        int c1_3 = lim1 - k0_0_1 - 2 * cq;
                        float v_65 = neg;
                        {
                            if (c0_3 >= 0) {
                                {
                                    float _cvt_f32_f16_776;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_776) : "h"((uint16_t)(acc1[0] & 65535)));
                                    v_65 = _cvt_f32_f16_776;
                                }
                            }
                            float _max_396 = max_noftz(m0v_1_1, v_65);
                            m0v_1_1 = _max_396;
                        }
                        float v_0_3 = neg;
                        {
                            if (c0_3 >= 1) {
                                {
                                    float _cvt_f32_f16_781;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_781) : "h"((uint16_t)(acc1[0] >> 16)));
                                    v_0_3 = _cvt_f32_f16_781;
                                }
                            }
                            float _max_398 = max_noftz(m0v_1_1, v_0_3);
                            m0v_1_1 = _max_398;
                        }
                        float v_1_3 = neg;
                        {
                            if (c1_3 >= 0) {
                                {
                                    float _cvt_f32_f16_786;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_786) : "h"((uint16_t)(acc1[1] & 65535)));
                                    v_1_3 = _cvt_f32_f16_786;
                                }
                            }
                            float _max_401 = max_noftz(m1v_2_1, v_1_3);
                            m1v_2_1 = _max_401;
                        }
                        float v_2_3 = neg;
                        {
                            if (c1_3 >= 1) {
                                {
                                    float _cvt_f32_f16_791;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_791) : "h"((uint16_t)(acc1[1] >> 16)));
                                    v_2_3 = _cvt_f32_f16_791;
                                }
                            }
                            float _max_403 = max_noftz(m1v_2_1, v_2_3);
                            m1v_2_1 = _max_403;
                        }
                        float v_3_3 = neg;
                        {
                            if (c0_3 >= 8) {
                                {
                                    float _cvt_f32_f16_792;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_792) : "h"((uint16_t)(acc1[2] & 65535)));
                                    v_3_3 = _cvt_f32_f16_792;
                                }
                            }
                            float _max_404 = max_noftz(m0v_1_1, v_3_3);
                            m0v_1_1 = _max_404;
                        }
                        float v_4_3 = neg;
                        {
                            if (c0_3 >= 9) {
                                {
                                    float _cvt_f32_f16_797;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_797) : "h"((uint16_t)(acc1[2] >> 16)));
                                    v_4_3 = _cvt_f32_f16_797;
                                }
                            }
                            float _max_406 = max_noftz(m0v_1_1, v_4_3);
                            m0v_1_1 = _max_406;
                        }
                        float v_5_3 = neg;
                        {
                            if (c1_3 >= 8) {
                                {
                                    float _cvt_f32_f16_802;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_802) : "h"((uint16_t)(acc1[3] & 65535)));
                                    v_5_3 = _cvt_f32_f16_802;
                                }
                            }
                            float _max_409 = max_noftz(m1v_2_1, v_5_3);
                            m1v_2_1 = _max_409;
                        }
                        float v_6_3 = neg;
                        {
                            if (c1_3 >= 9) {
                                {
                                    float _cvt_f32_f16_807;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_807) : "h"((uint16_t)(acc1[3] >> 16)));
                                    v_6_3 = _cvt_f32_f16_807;
                                }
                            }
                            float _max_411 = max_noftz(m1v_2_1, v_6_3);
                            m1v_2_1 = _max_411;
                        }
                        float v_7_3 = neg;
                        {
                            if (c0_3 >= 16) {
                                {
                                    float _cvt_f32_f16_808;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_808) : "h"((uint16_t)(acc1[4] & 65535)));
                                    v_7_3 = _cvt_f32_f16_808;
                                }
                            }
                            float _max_412 = max_noftz(m0v_1_1, v_7_3);
                            m0v_1_1 = _max_412;
                        }
                        float v_8_3 = neg;
                        {
                            if (c0_3 >= 17) {
                                {
                                    float _cvt_f32_f16_813;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_813) : "h"((uint16_t)(acc1[4] >> 16)));
                                    v_8_3 = _cvt_f32_f16_813;
                                }
                            }
                            float _max_414 = max_noftz(m0v_1_1, v_8_3);
                            m0v_1_1 = _max_414;
                        }
                        float v_9_3 = neg;
                        {
                            if (c1_3 >= 16) {
                                {
                                    float _cvt_f32_f16_818;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_818) : "h"((uint16_t)(acc1[5] & 65535)));
                                    v_9_3 = _cvt_f32_f16_818;
                                }
                            }
                            float _max_417 = max_noftz(m1v_2_1, v_9_3);
                            m1v_2_1 = _max_417;
                        }
                        float v_10_3 = neg;
                        {
                            if (c1_3 >= 17) {
                                {
                                    float _cvt_f32_f16_823;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_823) : "h"((uint16_t)(acc1[5] >> 16)));
                                    v_10_3 = _cvt_f32_f16_823;
                                }
                            }
                            float _max_419 = max_noftz(m1v_2_1, v_10_3);
                            m1v_2_1 = _max_419;
                        }
                        float v_11_3 = neg;
                        {
                            if (c0_3 >= 24) {
                                {
                                    float _cvt_f32_f16_824;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_824) : "h"((uint16_t)(acc1[6] & 65535)));
                                    v_11_3 = _cvt_f32_f16_824;
                                }
                            }
                            float _max_420 = max_noftz(m0v_1_1, v_11_3);
                            m0v_1_1 = _max_420;
                        }
                        float v_12_3 = neg;
                        {
                            if (c0_3 >= 25) {
                                {
                                    float _cvt_f32_f16_829;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_829) : "h"((uint16_t)(acc1[6] >> 16)));
                                    v_12_3 = _cvt_f32_f16_829;
                                }
                            }
                            float _max_422 = max_noftz(m0v_1_1, v_12_3);
                            m0v_1_1 = _max_422;
                        }
                        float v_13_3 = neg;
                        {
                            if (c1_3 >= 24) {
                                {
                                    float _cvt_f32_f16_834;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_834) : "h"((uint16_t)(acc1[7] & 65535)));
                                    v_13_3 = _cvt_f32_f16_834;
                                }
                            }
                            float _max_425 = max_noftz(m1v_2_1, v_13_3);
                            m1v_2_1 = _max_425;
                        }
                        float v_14_3 = neg;
                        {
                            if (c1_3 >= 25) {
                                {
                                    float _cvt_f32_f16_839;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_839) : "h"((uint16_t)(acc1[7] >> 16)));
                                    v_14_3 = _cvt_f32_f16_839;
                                }
                            }
                            float _max_427 = max_noftz(m1v_2_1, v_14_3);
                            m1v_2_1 = _max_427;
                        }
                        float v_15_3 = neg;
                        {
                            if (c0_3 >= 32) {
                                {
                                    float _cvt_f32_f16_840;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_840) : "h"((uint16_t)(acc1[8] & 65535)));
                                    v_15_3 = _cvt_f32_f16_840;
                                }
                            }
                            float _max_428 = max_noftz(m0v_1_1, v_15_3);
                            m0v_1_1 = _max_428;
                        }
                        float v_16_3 = neg;
                        {
                            if (c0_3 >= 33) {
                                {
                                    float _cvt_f32_f16_845;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_845) : "h"((uint16_t)(acc1[8] >> 16)));
                                    v_16_3 = _cvt_f32_f16_845;
                                }
                            }
                            float _max_430 = max_noftz(m0v_1_1, v_16_3);
                            m0v_1_1 = _max_430;
                        }
                        float v_17_3 = neg;
                        {
                            if (c1_3 >= 32) {
                                {
                                    float _cvt_f32_f16_850;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_850) : "h"((uint16_t)(acc1[9] & 65535)));
                                    v_17_3 = _cvt_f32_f16_850;
                                }
                            }
                            float _max_433 = max_noftz(m1v_2_1, v_17_3);
                            m1v_2_1 = _max_433;
                        }
                        float v_18_3 = neg;
                        {
                            if (c1_3 >= 33) {
                                {
                                    float _cvt_f32_f16_855;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_855) : "h"((uint16_t)(acc1[9] >> 16)));
                                    v_18_3 = _cvt_f32_f16_855;
                                }
                            }
                            float _max_435 = max_noftz(m1v_2_1, v_18_3);
                            m1v_2_1 = _max_435;
                        }
                        float v_19_3 = neg;
                        {
                            if (c0_3 >= 40) {
                                {
                                    float _cvt_f32_f16_856;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_856) : "h"((uint16_t)(acc1[10] & 65535)));
                                    v_19_3 = _cvt_f32_f16_856;
                                }
                            }
                            float _max_436 = max_noftz(m0v_1_1, v_19_3);
                            m0v_1_1 = _max_436;
                        }
                        float v_20_3 = neg;
                        {
                            if (c0_3 >= 41) {
                                {
                                    float _cvt_f32_f16_861;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_861) : "h"((uint16_t)(acc1[10] >> 16)));
                                    v_20_3 = _cvt_f32_f16_861;
                                }
                            }
                            float _max_438 = max_noftz(m0v_1_1, v_20_3);
                            m0v_1_1 = _max_438;
                        }
                        float v_21_3 = neg;
                        {
                            if (c1_3 >= 40) {
                                {
                                    float _cvt_f32_f16_866;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_866) : "h"((uint16_t)(acc1[11] & 65535)));
                                    v_21_3 = _cvt_f32_f16_866;
                                }
                            }
                            float _max_441 = max_noftz(m1v_2_1, v_21_3);
                            m1v_2_1 = _max_441;
                        }
                        float v_22_3 = neg;
                        {
                            if (c1_3 >= 41) {
                                {
                                    float _cvt_f32_f16_871;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_871) : "h"((uint16_t)(acc1[11] >> 16)));
                                    v_22_3 = _cvt_f32_f16_871;
                                }
                            }
                            float _max_443 = max_noftz(m1v_2_1, v_22_3);
                            m1v_2_1 = _max_443;
                        }
                        float v_23_3 = neg;
                        {
                            if (c0_3 >= 48) {
                                {
                                    float _cvt_f32_f16_872;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_872) : "h"((uint16_t)(acc1[12] & 65535)));
                                    v_23_3 = _cvt_f32_f16_872;
                                }
                            }
                            float _max_444 = max_noftz(m0v_1_1, v_23_3);
                            m0v_1_1 = _max_444;
                        }
                        float v_24_3 = neg;
                        {
                            if (c0_3 >= 49) {
                                {
                                    float _cvt_f32_f16_877;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_877) : "h"((uint16_t)(acc1[12] >> 16)));
                                    v_24_3 = _cvt_f32_f16_877;
                                }
                            }
                            float _max_446 = max_noftz(m0v_1_1, v_24_3);
                            m0v_1_1 = _max_446;
                        }
                        float v_25_3 = neg;
                        {
                            if (c1_3 >= 48) {
                                {
                                    float _cvt_f32_f16_882;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_882) : "h"((uint16_t)(acc1[13] & 65535)));
                                    v_25_3 = _cvt_f32_f16_882;
                                }
                            }
                            float _max_449 = max_noftz(m1v_2_1, v_25_3);
                            m1v_2_1 = _max_449;
                        }
                        float v_26_3 = neg;
                        {
                            if (c1_3 >= 49) {
                                {
                                    float _cvt_f32_f16_887;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_887) : "h"((uint16_t)(acc1[13] >> 16)));
                                    v_26_3 = _cvt_f32_f16_887;
                                }
                            }
                            float _max_451 = max_noftz(m1v_2_1, v_26_3);
                            m1v_2_1 = _max_451;
                        }
                        float v_27_3 = neg;
                        {
                            if (c0_3 >= 56) {
                                {
                                    float _cvt_f32_f16_888;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_888) : "h"((uint16_t)(acc1[14] & 65535)));
                                    v_27_3 = _cvt_f32_f16_888;
                                }
                            }
                            float _max_452 = max_noftz(m0v_1_1, v_27_3);
                            m0v_1_1 = _max_452;
                        }
                        float v_28_3 = neg;
                        {
                            if (c0_3 >= 57) {
                                {
                                    float _cvt_f32_f16_893;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_893) : "h"((uint16_t)(acc1[14] >> 16)));
                                    v_28_3 = _cvt_f32_f16_893;
                                }
                            }
                            float _max_454 = max_noftz(m0v_1_1, v_28_3);
                            m0v_1_1 = _max_454;
                        }
                        float v_29_3 = neg;
                        {
                            if (c1_3 >= 56) {
                                {
                                    float _cvt_f32_f16_898;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_898) : "h"((uint16_t)(acc1[15] & 65535)));
                                    v_29_3 = _cvt_f32_f16_898;
                                }
                            }
                            float _max_457 = max_noftz(m1v_2_1, v_29_3);
                            m1v_2_1 = _max_457;
                        }
                        float v_30_3 = neg;
                        {
                            if (c1_3 >= 57) {
                                {
                                    float _cvt_f32_f16_903;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_903) : "h"((uint16_t)(acc1[15] >> 16)));
                                    v_30_3 = _cvt_f32_f16_903;
                                }
                            }
                            float _max_459 = max_noftz(m1v_2_1, v_30_3);
                            m1v_2_1 = _max_459;
                        }
                        float v_31_3 = neg;
                        {
                            if (c0_3 >= 64) {
                                {
                                    float _cvt_f32_f16_904;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_904) : "h"((uint16_t)(acc1[16] & 65535)));
                                    v_31_3 = _cvt_f32_f16_904;
                                }
                            }
                            float _max_460 = max_noftz(m0v_1_1, v_31_3);
                            m0v_1_1 = _max_460;
                        }
                        float v_32_3 = neg;
                        {
                            if (c0_3 >= 65) {
                                {
                                    float _cvt_f32_f16_909;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_909) : "h"((uint16_t)(acc1[16] >> 16)));
                                    v_32_3 = _cvt_f32_f16_909;
                                }
                            }
                            float _max_462 = max_noftz(m0v_1_1, v_32_3);
                            m0v_1_1 = _max_462;
                        }
                        float v_33_3 = neg;
                        {
                            if (c1_3 >= 64) {
                                {
                                    float _cvt_f32_f16_914;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_914) : "h"((uint16_t)(acc1[17] & 65535)));
                                    v_33_3 = _cvt_f32_f16_914;
                                }
                            }
                            float _max_465 = max_noftz(m1v_2_1, v_33_3);
                            m1v_2_1 = _max_465;
                        }
                        float v_34_3 = neg;
                        {
                            if (c1_3 >= 65) {
                                {
                                    float _cvt_f32_f16_919;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_919) : "h"((uint16_t)(acc1[17] >> 16)));
                                    v_34_3 = _cvt_f32_f16_919;
                                }
                            }
                            float _max_467 = max_noftz(m1v_2_1, v_34_3);
                            m1v_2_1 = _max_467;
                        }
                        float v_35_3 = neg;
                        {
                            if (c0_3 >= 72) {
                                {
                                    float _cvt_f32_f16_920;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_920) : "h"((uint16_t)(acc1[18] & 65535)));
                                    v_35_3 = _cvt_f32_f16_920;
                                }
                            }
                            float _max_468 = max_noftz(m0v_1_1, v_35_3);
                            m0v_1_1 = _max_468;
                        }
                        float v_36_3 = neg;
                        {
                            if (c0_3 >= 73) {
                                {
                                    float _cvt_f32_f16_925;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_925) : "h"((uint16_t)(acc1[18] >> 16)));
                                    v_36_3 = _cvt_f32_f16_925;
                                }
                            }
                            float _max_470 = max_noftz(m0v_1_1, v_36_3);
                            m0v_1_1 = _max_470;
                        }
                        float v_37_3 = neg;
                        {
                            if (c1_3 >= 72) {
                                {
                                    float _cvt_f32_f16_930;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_930) : "h"((uint16_t)(acc1[19] & 65535)));
                                    v_37_3 = _cvt_f32_f16_930;
                                }
                            }
                            float _max_473 = max_noftz(m1v_2_1, v_37_3);
                            m1v_2_1 = _max_473;
                        }
                        float v_38_3 = neg;
                        {
                            if (c1_3 >= 73) {
                                {
                                    float _cvt_f32_f16_935;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_935) : "h"((uint16_t)(acc1[19] >> 16)));
                                    v_38_3 = _cvt_f32_f16_935;
                                }
                            }
                            float _max_475 = max_noftz(m1v_2_1, v_38_3);
                            m1v_2_1 = _max_475;
                        }
                        float v_39_3 = neg;
                        {
                            if (c0_3 >= 80) {
                                {
                                    float _cvt_f32_f16_936;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_936) : "h"((uint16_t)(acc1[20] & 65535)));
                                    v_39_3 = _cvt_f32_f16_936;
                                }
                            }
                            float _max_476 = max_noftz(m0v_1_1, v_39_3);
                            m0v_1_1 = _max_476;
                        }
                        float v_40_3 = neg;
                        {
                            if (c0_3 >= 81) {
                                {
                                    float _cvt_f32_f16_941;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_941) : "h"((uint16_t)(acc1[20] >> 16)));
                                    v_40_3 = _cvt_f32_f16_941;
                                }
                            }
                            float _max_478 = max_noftz(m0v_1_1, v_40_3);
                            m0v_1_1 = _max_478;
                        }
                        float v_41_3 = neg;
                        {
                            if (c1_3 >= 80) {
                                {
                                    float _cvt_f32_f16_946;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_946) : "h"((uint16_t)(acc1[21] & 65535)));
                                    v_41_3 = _cvt_f32_f16_946;
                                }
                            }
                            float _max_481 = max_noftz(m1v_2_1, v_41_3);
                            m1v_2_1 = _max_481;
                        }
                        float v_42_3 = neg;
                        {
                            if (c1_3 >= 81) {
                                {
                                    float _cvt_f32_f16_951;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_951) : "h"((uint16_t)(acc1[21] >> 16)));
                                    v_42_3 = _cvt_f32_f16_951;
                                }
                            }
                            float _max_483 = max_noftz(m1v_2_1, v_42_3);
                            m1v_2_1 = _max_483;
                        }
                        float v_43_3 = neg;
                        {
                            if (c0_3 >= 88) {
                                {
                                    float _cvt_f32_f16_952;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_952) : "h"((uint16_t)(acc1[22] & 65535)));
                                    v_43_3 = _cvt_f32_f16_952;
                                }
                            }
                            float _max_484 = max_noftz(m0v_1_1, v_43_3);
                            m0v_1_1 = _max_484;
                        }
                        float v_44_3 = neg;
                        {
                            if (c0_3 >= 89) {
                                {
                                    float _cvt_f32_f16_957;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_957) : "h"((uint16_t)(acc1[22] >> 16)));
                                    v_44_3 = _cvt_f32_f16_957;
                                }
                            }
                            float _max_486 = max_noftz(m0v_1_1, v_44_3);
                            m0v_1_1 = _max_486;
                        }
                        float v_45_3 = neg;
                        {
                            if (c1_3 >= 88) {
                                {
                                    float _cvt_f32_f16_962;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_962) : "h"((uint16_t)(acc1[23] & 65535)));
                                    v_45_3 = _cvt_f32_f16_962;
                                }
                            }
                            float _max_489 = max_noftz(m1v_2_1, v_45_3);
                            m1v_2_1 = _max_489;
                        }
                        float v_46_3 = neg;
                        {
                            if (c1_3 >= 89) {
                                {
                                    float _cvt_f32_f16_967;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_967) : "h"((uint16_t)(acc1[23] >> 16)));
                                    v_46_3 = _cvt_f32_f16_967;
                                }
                            }
                            float _max_491 = max_noftz(m1v_2_1, v_46_3);
                            m1v_2_1 = _max_491;
                        }
                        float v_47_3 = neg;
                        {
                            if (c0_3 >= 96) {
                                {
                                    float _cvt_f32_f16_968;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_968) : "h"((uint16_t)(acc1[24] & 65535)));
                                    v_47_3 = _cvt_f32_f16_968;
                                }
                            }
                            float _max_492 = max_noftz(m0v_1_1, v_47_3);
                            m0v_1_1 = _max_492;
                        }
                        float v_48_3 = neg;
                        {
                            if (c0_3 >= 97) {
                                {
                                    float _cvt_f32_f16_973;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_973) : "h"((uint16_t)(acc1[24] >> 16)));
                                    v_48_3 = _cvt_f32_f16_973;
                                }
                            }
                            float _max_494 = max_noftz(m0v_1_1, v_48_3);
                            m0v_1_1 = _max_494;
                        }
                        float v_49_3 = neg;
                        {
                            if (c1_3 >= 96) {
                                {
                                    float _cvt_f32_f16_978;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_978) : "h"((uint16_t)(acc1[25] & 65535)));
                                    v_49_3 = _cvt_f32_f16_978;
                                }
                            }
                            float _max_497 = max_noftz(m1v_2_1, v_49_3);
                            m1v_2_1 = _max_497;
                        }
                        float v_50_3 = neg;
                        {
                            if (c1_3 >= 97) {
                                {
                                    float _cvt_f32_f16_983;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_983) : "h"((uint16_t)(acc1[25] >> 16)));
                                    v_50_3 = _cvt_f32_f16_983;
                                }
                            }
                            float _max_499 = max_noftz(m1v_2_1, v_50_3);
                            m1v_2_1 = _max_499;
                        }
                        float v_51_3 = neg;
                        {
                            if (c0_3 >= 104) {
                                {
                                    float _cvt_f32_f16_984;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_984) : "h"((uint16_t)(acc1[26] & 65535)));
                                    v_51_3 = _cvt_f32_f16_984;
                                }
                            }
                            float _max_500 = max_noftz(m0v_1_1, v_51_3);
                            m0v_1_1 = _max_500;
                        }
                        float v_52_3 = neg;
                        {
                            if (c0_3 >= 105) {
                                {
                                    float _cvt_f32_f16_989;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_989) : "h"((uint16_t)(acc1[26] >> 16)));
                                    v_52_3 = _cvt_f32_f16_989;
                                }
                            }
                            float _max_502 = max_noftz(m0v_1_1, v_52_3);
                            m0v_1_1 = _max_502;
                        }
                        float v_53_3 = neg;
                        {
                            if (c1_3 >= 104) {
                                {
                                    float _cvt_f32_f16_994;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_994) : "h"((uint16_t)(acc1[27] & 65535)));
                                    v_53_3 = _cvt_f32_f16_994;
                                }
                            }
                            float _max_505 = max_noftz(m1v_2_1, v_53_3);
                            m1v_2_1 = _max_505;
                        }
                        float v_54_3 = neg;
                        {
                            if (c1_3 >= 105) {
                                {
                                    float _cvt_f32_f16_999;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_999) : "h"((uint16_t)(acc1[27] >> 16)));
                                    v_54_3 = _cvt_f32_f16_999;
                                }
                            }
                            float _max_507 = max_noftz(m1v_2_1, v_54_3);
                            m1v_2_1 = _max_507;
                        }
                        float v_55_3 = neg;
                        {
                            if (c0_3 >= 112) {
                                {
                                    float _cvt_f32_f16_1000;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1000) : "h"((uint16_t)(acc1[28] & 65535)));
                                    v_55_3 = _cvt_f32_f16_1000;
                                }
                            }
                            float _max_508 = max_noftz(m0v_1_1, v_55_3);
                            m0v_1_1 = _max_508;
                        }
                        float v_56_3 = neg;
                        {
                            if (c0_3 >= 113) {
                                {
                                    float _cvt_f32_f16_1005;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1005) : "h"((uint16_t)(acc1[28] >> 16)));
                                    v_56_3 = _cvt_f32_f16_1005;
                                }
                            }
                            float _max_510 = max_noftz(m0v_1_1, v_56_3);
                            m0v_1_1 = _max_510;
                        }
                        float v_57_3 = neg;
                        {
                            if (c1_3 >= 112) {
                                {
                                    float _cvt_f32_f16_1010;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1010) : "h"((uint16_t)(acc1[29] & 65535)));
                                    v_57_3 = _cvt_f32_f16_1010;
                                }
                            }
                            float _max_513 = max_noftz(m1v_2_1, v_57_3);
                            m1v_2_1 = _max_513;
                        }
                        float v_58_3 = neg;
                        {
                            if (c1_3 >= 113) {
                                {
                                    float _cvt_f32_f16_1015;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1015) : "h"((uint16_t)(acc1[29] >> 16)));
                                    v_58_3 = _cvt_f32_f16_1015;
                                }
                            }
                            float _max_515 = max_noftz(m1v_2_1, v_58_3);
                            m1v_2_1 = _max_515;
                        }
                        float v_59_3 = neg;
                        {
                            if (c0_3 >= 120) {
                                {
                                    float _cvt_f32_f16_1016;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1016) : "h"((uint16_t)(acc1[30] & 65535)));
                                    v_59_3 = _cvt_f32_f16_1016;
                                }
                            }
                            float _max_516 = max_noftz(m0v_1_1, v_59_3);
                            m0v_1_1 = _max_516;
                        }
                        float v_60_3 = neg;
                        {
                            if (c0_3 >= 121) {
                                {
                                    float _cvt_f32_f16_1021;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1021) : "h"((uint16_t)(acc1[30] >> 16)));
                                    v_60_3 = _cvt_f32_f16_1021;
                                }
                            }
                            float _max_518 = max_noftz(m0v_1_1, v_60_3);
                            m0v_1_1 = _max_518;
                        }
                        float v_61_3 = neg;
                        {
                            if (c1_3 >= 120) {
                                {
                                    float _cvt_f32_f16_1026;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1026) : "h"((uint16_t)(acc1[31] & 65535)));
                                    v_61_3 = _cvt_f32_f16_1026;
                                }
                            }
                            float _max_521 = max_noftz(m1v_2_1, v_61_3);
                            m1v_2_1 = _max_521;
                        }
                        float v_62_3 = neg;
                        {
                            if (c1_3 >= 121) {
                                {
                                    float _cvt_f32_f16_1031;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1031) : "h"((uint16_t)(acc1[31] >> 16)));
                                    v_62_3 = _cvt_f32_f16_1031;
                                }
                            }
                            float _max_523 = max_noftz(m1v_2_1, v_62_3);
                            m1v_2_1 = _max_523;
                        }
                    }
                    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, m0v_1_1, 1);
                    float _max_524 = max_noftz(m0v_1_1, _shfl_xor_12);
                    m0v_1_1 = _max_524;
                    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, m0v_1_1, 2);
                    float _max_525 = max_noftz(m0v_1_1, _shfl_xor_13);
                    m0v_1_1 = _max_525;
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, m1v_2_1, 1);
                    float _max_526 = max_noftz(m1v_2_1, _shfl_xor_14);
                    m1v_2_1 = _max_526;
                    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, m1v_2_1, 2);
                    float _max_527 = max_noftz(m1v_2_1, _shfl_xor_15);
                    m1v_2_1 = _max_527;
                    if (cq == 0) {
                        int orow_3 = obase + (split2 + im * nsplit) * total_q + r0;
                        if (r0 < rows_valid) {
                            out[orow_3] = m0v_1_1;
                        }
                        if (rows_valid > r0 + 8) {
                            out[orow_3 + 8] = m1v_2_1;
                        }
                    }
                }
                if (n_comp2 - il != 2) {
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(empty_addr + (sl) * 8);
                    }
                    int k0_2 = (split2 + il * nsplit) * 128;
                    float m0v_3 = neg;
                    float m1v_3 = neg;
                    int need_mask_2 = 0;
                    if (mask_thr < k0_2 + 127) {
                        need_mask_2 = 1;
                    }
                    if (need_mask_2 == 0) {
                        unsigned int w0_4 = acc[0];
                        unsigned int w1_4 = acc[1];
                        uint32_t _f16x2_max_120;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_120) : "r"(w0_4), "r"(acc[2]));
                        w0_4 = _f16x2_max_120;
                        uint32_t _f16x2_max_121;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_121) : "r"(w1_4), "r"(acc[3]));
                        w1_4 = _f16x2_max_121;
                        uint32_t _f16x2_max_122;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_122) : "r"(w0_4), "r"(acc[4]));
                        w0_4 = _f16x2_max_122;
                        uint32_t _f16x2_max_123;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_123) : "r"(w1_4), "r"(acc[5]));
                        w1_4 = _f16x2_max_123;
                        uint32_t _f16x2_max_124;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_124) : "r"(w0_4), "r"(acc[6]));
                        w0_4 = _f16x2_max_124;
                        uint32_t _f16x2_max_125;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_125) : "r"(w1_4), "r"(acc[7]));
                        w1_4 = _f16x2_max_125;
                        uint32_t _f16x2_max_126;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_126) : "r"(w0_4), "r"(acc[8]));
                        w0_4 = _f16x2_max_126;
                        uint32_t _f16x2_max_127;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_127) : "r"(w1_4), "r"(acc[9]));
                        w1_4 = _f16x2_max_127;
                        uint32_t _f16x2_max_128;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_128) : "r"(w0_4), "r"(acc[10]));
                        w0_4 = _f16x2_max_128;
                        uint32_t _f16x2_max_129;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_129) : "r"(w1_4), "r"(acc[11]));
                        w1_4 = _f16x2_max_129;
                        uint32_t _f16x2_max_130;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_130) : "r"(w0_4), "r"(acc[12]));
                        w0_4 = _f16x2_max_130;
                        uint32_t _f16x2_max_131;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_131) : "r"(w1_4), "r"(acc[13]));
                        w1_4 = _f16x2_max_131;
                        uint32_t _f16x2_max_132;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_132) : "r"(w0_4), "r"(acc[14]));
                        w0_4 = _f16x2_max_132;
                        uint32_t _f16x2_max_133;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_133) : "r"(w1_4), "r"(acc[15]));
                        w1_4 = _f16x2_max_133;
                        uint32_t _f16x2_max_134;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_134) : "r"(w0_4), "r"(acc[16]));
                        w0_4 = _f16x2_max_134;
                        uint32_t _f16x2_max_135;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_135) : "r"(w1_4), "r"(acc[17]));
                        w1_4 = _f16x2_max_135;
                        uint32_t _f16x2_max_136;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_136) : "r"(w0_4), "r"(acc[18]));
                        w0_4 = _f16x2_max_136;
                        uint32_t _f16x2_max_137;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_137) : "r"(w1_4), "r"(acc[19]));
                        w1_4 = _f16x2_max_137;
                        uint32_t _f16x2_max_138;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_138) : "r"(w0_4), "r"(acc[20]));
                        w0_4 = _f16x2_max_138;
                        uint32_t _f16x2_max_139;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_139) : "r"(w1_4), "r"(acc[21]));
                        w1_4 = _f16x2_max_139;
                        uint32_t _f16x2_max_140;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_140) : "r"(w0_4), "r"(acc[22]));
                        w0_4 = _f16x2_max_140;
                        uint32_t _f16x2_max_141;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_141) : "r"(w1_4), "r"(acc[23]));
                        w1_4 = _f16x2_max_141;
                        uint32_t _f16x2_max_142;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_142) : "r"(w0_4), "r"(acc[24]));
                        w0_4 = _f16x2_max_142;
                        uint32_t _f16x2_max_143;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_143) : "r"(w1_4), "r"(acc[25]));
                        w1_4 = _f16x2_max_143;
                        uint32_t _f16x2_max_144;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_144) : "r"(w0_4), "r"(acc[26]));
                        w0_4 = _f16x2_max_144;
                        uint32_t _f16x2_max_145;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_145) : "r"(w1_4), "r"(acc[27]));
                        w1_4 = _f16x2_max_145;
                        uint32_t _f16x2_max_146;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_146) : "r"(w0_4), "r"(acc[28]));
                        w0_4 = _f16x2_max_146;
                        uint32_t _f16x2_max_147;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_147) : "r"(w1_4), "r"(acc[29]));
                        w1_4 = _f16x2_max_147;
                        uint32_t _f16x2_max_148;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_148) : "r"(w0_4), "r"(acc[30]));
                        w0_4 = _f16x2_max_148;
                        uint32_t _f16x2_max_149;
                        asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_149) : "r"(w1_4), "r"(acc[31]));
                        w1_4 = _f16x2_max_149;
                        uint16_t _f16_max_8;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_8) : "h"((uint16_t)(w0_4 & 65535)), "h"((uint16_t)(w0_4 >> 16)));
                        float _cvt_f32_f16_1032;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1032) : "h"((uint16_t)(_f16_max_8)));
                        m0v_3 = _cvt_f32_f16_1032;
                        uint16_t _f16_max_9;
                        asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_9) : "h"((uint16_t)(w1_4 & 65535)), "h"((uint16_t)(w1_4 >> 16)));
                        float _cvt_f32_f16_1033;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1033) : "h"((uint16_t)(_f16_max_9)));
                        m1v_3 = _cvt_f32_f16_1033;
                    }
                    if (need_mask_2 != 0) {
                        int c0_4 = lim0 - k0_2 - 2 * cq;
                        int c1_4 = lim1 - k0_2 - 2 * cq;
                        float v_66 = neg;
                        {
                            if (c0_4 >= 0) {
                                {
                                    float _cvt_f32_f16_1034;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1034) : "h"((uint16_t)(acc[0] & 65535)));
                                    v_66 = _cvt_f32_f16_1034;
                                }
                            }
                            float _max_528 = max_noftz(m0v_3, v_66);
                            m0v_3 = _max_528;
                        }
                        float v_0_4 = neg;
                        {
                            if (c0_4 >= 1) {
                                {
                                    float _cvt_f32_f16_1039;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1039) : "h"((uint16_t)(acc[0] >> 16)));
                                    v_0_4 = _cvt_f32_f16_1039;
                                }
                            }
                            float _max_530 = max_noftz(m0v_3, v_0_4);
                            m0v_3 = _max_530;
                        }
                        float v_1_4 = neg;
                        {
                            if (c1_4 >= 0) {
                                {
                                    float _cvt_f32_f16_1044;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1044) : "h"((uint16_t)(acc[1] & 65535)));
                                    v_1_4 = _cvt_f32_f16_1044;
                                }
                            }
                            float _max_533 = max_noftz(m1v_3, v_1_4);
                            m1v_3 = _max_533;
                        }
                        float v_2_4 = neg;
                        {
                            if (c1_4 >= 1) {
                                {
                                    float _cvt_f32_f16_1049;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1049) : "h"((uint16_t)(acc[1] >> 16)));
                                    v_2_4 = _cvt_f32_f16_1049;
                                }
                            }
                            float _max_535 = max_noftz(m1v_3, v_2_4);
                            m1v_3 = _max_535;
                        }
                        float v_3_4 = neg;
                        {
                            if (c0_4 >= 8) {
                                {
                                    float _cvt_f32_f16_1050;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1050) : "h"((uint16_t)(acc[2] & 65535)));
                                    v_3_4 = _cvt_f32_f16_1050;
                                }
                            }
                            float _max_536 = max_noftz(m0v_3, v_3_4);
                            m0v_3 = _max_536;
                        }
                        float v_4_4 = neg;
                        {
                            if (c0_4 >= 9) {
                                {
                                    float _cvt_f32_f16_1055;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1055) : "h"((uint16_t)(acc[2] >> 16)));
                                    v_4_4 = _cvt_f32_f16_1055;
                                }
                            }
                            float _max_538 = max_noftz(m0v_3, v_4_4);
                            m0v_3 = _max_538;
                        }
                        float v_5_4 = neg;
                        {
                            if (c1_4 >= 8) {
                                {
                                    float _cvt_f32_f16_1060;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1060) : "h"((uint16_t)(acc[3] & 65535)));
                                    v_5_4 = _cvt_f32_f16_1060;
                                }
                            }
                            float _max_541 = max_noftz(m1v_3, v_5_4);
                            m1v_3 = _max_541;
                        }
                        float v_6_4 = neg;
                        {
                            if (c1_4 >= 9) {
                                {
                                    float _cvt_f32_f16_1065;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1065) : "h"((uint16_t)(acc[3] >> 16)));
                                    v_6_4 = _cvt_f32_f16_1065;
                                }
                            }
                            float _max_543 = max_noftz(m1v_3, v_6_4);
                            m1v_3 = _max_543;
                        }
                        float v_7_4 = neg;
                        {
                            if (c0_4 >= 16) {
                                {
                                    float _cvt_f32_f16_1066;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1066) : "h"((uint16_t)(acc[4] & 65535)));
                                    v_7_4 = _cvt_f32_f16_1066;
                                }
                            }
                            float _max_544 = max_noftz(m0v_3, v_7_4);
                            m0v_3 = _max_544;
                        }
                        float v_8_4 = neg;
                        {
                            if (c0_4 >= 17) {
                                {
                                    float _cvt_f32_f16_1071;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1071) : "h"((uint16_t)(acc[4] >> 16)));
                                    v_8_4 = _cvt_f32_f16_1071;
                                }
                            }
                            float _max_546 = max_noftz(m0v_3, v_8_4);
                            m0v_3 = _max_546;
                        }
                        float v_9_4 = neg;
                        {
                            if (c1_4 >= 16) {
                                {
                                    float _cvt_f32_f16_1076;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1076) : "h"((uint16_t)(acc[5] & 65535)));
                                    v_9_4 = _cvt_f32_f16_1076;
                                }
                            }
                            float _max_549 = max_noftz(m1v_3, v_9_4);
                            m1v_3 = _max_549;
                        }
                        float v_10_4 = neg;
                        {
                            if (c1_4 >= 17) {
                                {
                                    float _cvt_f32_f16_1081;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1081) : "h"((uint16_t)(acc[5] >> 16)));
                                    v_10_4 = _cvt_f32_f16_1081;
                                }
                            }
                            float _max_551 = max_noftz(m1v_3, v_10_4);
                            m1v_3 = _max_551;
                        }
                        float v_11_4 = neg;
                        {
                            if (c0_4 >= 24) {
                                {
                                    float _cvt_f32_f16_1082;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1082) : "h"((uint16_t)(acc[6] & 65535)));
                                    v_11_4 = _cvt_f32_f16_1082;
                                }
                            }
                            float _max_552 = max_noftz(m0v_3, v_11_4);
                            m0v_3 = _max_552;
                        }
                        float v_12_4 = neg;
                        {
                            if (c0_4 >= 25) {
                                {
                                    float _cvt_f32_f16_1087;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1087) : "h"((uint16_t)(acc[6] >> 16)));
                                    v_12_4 = _cvt_f32_f16_1087;
                                }
                            }
                            float _max_554 = max_noftz(m0v_3, v_12_4);
                            m0v_3 = _max_554;
                        }
                        float v_13_4 = neg;
                        {
                            if (c1_4 >= 24) {
                                {
                                    float _cvt_f32_f16_1092;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1092) : "h"((uint16_t)(acc[7] & 65535)));
                                    v_13_4 = _cvt_f32_f16_1092;
                                }
                            }
                            float _max_557 = max_noftz(m1v_3, v_13_4);
                            m1v_3 = _max_557;
                        }
                        float v_14_4 = neg;
                        {
                            if (c1_4 >= 25) {
                                {
                                    float _cvt_f32_f16_1097;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1097) : "h"((uint16_t)(acc[7] >> 16)));
                                    v_14_4 = _cvt_f32_f16_1097;
                                }
                            }
                            float _max_559 = max_noftz(m1v_3, v_14_4);
                            m1v_3 = _max_559;
                        }
                        float v_15_4 = neg;
                        {
                            if (c0_4 >= 32) {
                                {
                                    float _cvt_f32_f16_1098;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1098) : "h"((uint16_t)(acc[8] & 65535)));
                                    v_15_4 = _cvt_f32_f16_1098;
                                }
                            }
                            float _max_560 = max_noftz(m0v_3, v_15_4);
                            m0v_3 = _max_560;
                        }
                        float v_16_4 = neg;
                        {
                            if (c0_4 >= 33) {
                                {
                                    float _cvt_f32_f16_1103;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1103) : "h"((uint16_t)(acc[8] >> 16)));
                                    v_16_4 = _cvt_f32_f16_1103;
                                }
                            }
                            float _max_562 = max_noftz(m0v_3, v_16_4);
                            m0v_3 = _max_562;
                        }
                        float v_17_4 = neg;
                        {
                            if (c1_4 >= 32) {
                                {
                                    float _cvt_f32_f16_1108;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1108) : "h"((uint16_t)(acc[9] & 65535)));
                                    v_17_4 = _cvt_f32_f16_1108;
                                }
                            }
                            float _max_565 = max_noftz(m1v_3, v_17_4);
                            m1v_3 = _max_565;
                        }
                        float v_18_4 = neg;
                        {
                            if (c1_4 >= 33) {
                                {
                                    float _cvt_f32_f16_1113;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1113) : "h"((uint16_t)(acc[9] >> 16)));
                                    v_18_4 = _cvt_f32_f16_1113;
                                }
                            }
                            float _max_567 = max_noftz(m1v_3, v_18_4);
                            m1v_3 = _max_567;
                        }
                        float v_19_4 = neg;
                        {
                            if (c0_4 >= 40) {
                                {
                                    float _cvt_f32_f16_1114;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1114) : "h"((uint16_t)(acc[10] & 65535)));
                                    v_19_4 = _cvt_f32_f16_1114;
                                }
                            }
                            float _max_568 = max_noftz(m0v_3, v_19_4);
                            m0v_3 = _max_568;
                        }
                        float v_20_4 = neg;
                        {
                            if (c0_4 >= 41) {
                                {
                                    float _cvt_f32_f16_1119;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1119) : "h"((uint16_t)(acc[10] >> 16)));
                                    v_20_4 = _cvt_f32_f16_1119;
                                }
                            }
                            float _max_570 = max_noftz(m0v_3, v_20_4);
                            m0v_3 = _max_570;
                        }
                        float v_21_4 = neg;
                        {
                            if (c1_4 >= 40) {
                                {
                                    float _cvt_f32_f16_1124;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1124) : "h"((uint16_t)(acc[11] & 65535)));
                                    v_21_4 = _cvt_f32_f16_1124;
                                }
                            }
                            float _max_573 = max_noftz(m1v_3, v_21_4);
                            m1v_3 = _max_573;
                        }
                        float v_22_4 = neg;
                        {
                            if (c1_4 >= 41) {
                                {
                                    float _cvt_f32_f16_1129;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1129) : "h"((uint16_t)(acc[11] >> 16)));
                                    v_22_4 = _cvt_f32_f16_1129;
                                }
                            }
                            float _max_575 = max_noftz(m1v_3, v_22_4);
                            m1v_3 = _max_575;
                        }
                        float v_23_4 = neg;
                        {
                            if (c0_4 >= 48) {
                                {
                                    float _cvt_f32_f16_1130;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1130) : "h"((uint16_t)(acc[12] & 65535)));
                                    v_23_4 = _cvt_f32_f16_1130;
                                }
                            }
                            float _max_576 = max_noftz(m0v_3, v_23_4);
                            m0v_3 = _max_576;
                        }
                        float v_24_4 = neg;
                        {
                            if (c0_4 >= 49) {
                                {
                                    float _cvt_f32_f16_1135;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1135) : "h"((uint16_t)(acc[12] >> 16)));
                                    v_24_4 = _cvt_f32_f16_1135;
                                }
                            }
                            float _max_578 = max_noftz(m0v_3, v_24_4);
                            m0v_3 = _max_578;
                        }
                        float v_25_4 = neg;
                        {
                            if (c1_4 >= 48) {
                                {
                                    float _cvt_f32_f16_1140;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1140) : "h"((uint16_t)(acc[13] & 65535)));
                                    v_25_4 = _cvt_f32_f16_1140;
                                }
                            }
                            float _max_581 = max_noftz(m1v_3, v_25_4);
                            m1v_3 = _max_581;
                        }
                        float v_26_4 = neg;
                        {
                            if (c1_4 >= 49) {
                                {
                                    float _cvt_f32_f16_1145;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1145) : "h"((uint16_t)(acc[13] >> 16)));
                                    v_26_4 = _cvt_f32_f16_1145;
                                }
                            }
                            float _max_583 = max_noftz(m1v_3, v_26_4);
                            m1v_3 = _max_583;
                        }
                        float v_27_4 = neg;
                        {
                            if (c0_4 >= 56) {
                                {
                                    float _cvt_f32_f16_1146;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1146) : "h"((uint16_t)(acc[14] & 65535)));
                                    v_27_4 = _cvt_f32_f16_1146;
                                }
                            }
                            float _max_584 = max_noftz(m0v_3, v_27_4);
                            m0v_3 = _max_584;
                        }
                        float v_28_4 = neg;
                        {
                            if (c0_4 >= 57) {
                                {
                                    float _cvt_f32_f16_1151;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1151) : "h"((uint16_t)(acc[14] >> 16)));
                                    v_28_4 = _cvt_f32_f16_1151;
                                }
                            }
                            float _max_586 = max_noftz(m0v_3, v_28_4);
                            m0v_3 = _max_586;
                        }
                        float v_29_4 = neg;
                        {
                            if (c1_4 >= 56) {
                                {
                                    float _cvt_f32_f16_1156;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1156) : "h"((uint16_t)(acc[15] & 65535)));
                                    v_29_4 = _cvt_f32_f16_1156;
                                }
                            }
                            float _max_589 = max_noftz(m1v_3, v_29_4);
                            m1v_3 = _max_589;
                        }
                        float v_30_4 = neg;
                        {
                            if (c1_4 >= 57) {
                                {
                                    float _cvt_f32_f16_1161;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1161) : "h"((uint16_t)(acc[15] >> 16)));
                                    v_30_4 = _cvt_f32_f16_1161;
                                }
                            }
                            float _max_591 = max_noftz(m1v_3, v_30_4);
                            m1v_3 = _max_591;
                        }
                        float v_31_4 = neg;
                        {
                            if (c0_4 >= 64) {
                                {
                                    float _cvt_f32_f16_1162;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1162) : "h"((uint16_t)(acc[16] & 65535)));
                                    v_31_4 = _cvt_f32_f16_1162;
                                }
                            }
                            float _max_592 = max_noftz(m0v_3, v_31_4);
                            m0v_3 = _max_592;
                        }
                        float v_32_4 = neg;
                        {
                            if (c0_4 >= 65) {
                                {
                                    float _cvt_f32_f16_1167;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1167) : "h"((uint16_t)(acc[16] >> 16)));
                                    v_32_4 = _cvt_f32_f16_1167;
                                }
                            }
                            float _max_594 = max_noftz(m0v_3, v_32_4);
                            m0v_3 = _max_594;
                        }
                        float v_33_4 = neg;
                        {
                            if (c1_4 >= 64) {
                                {
                                    float _cvt_f32_f16_1172;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1172) : "h"((uint16_t)(acc[17] & 65535)));
                                    v_33_4 = _cvt_f32_f16_1172;
                                }
                            }
                            float _max_597 = max_noftz(m1v_3, v_33_4);
                            m1v_3 = _max_597;
                        }
                        float v_34_4 = neg;
                        {
                            if (c1_4 >= 65) {
                                {
                                    float _cvt_f32_f16_1177;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1177) : "h"((uint16_t)(acc[17] >> 16)));
                                    v_34_4 = _cvt_f32_f16_1177;
                                }
                            }
                            float _max_599 = max_noftz(m1v_3, v_34_4);
                            m1v_3 = _max_599;
                        }
                        float v_35_4 = neg;
                        {
                            if (c0_4 >= 72) {
                                {
                                    float _cvt_f32_f16_1178;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1178) : "h"((uint16_t)(acc[18] & 65535)));
                                    v_35_4 = _cvt_f32_f16_1178;
                                }
                            }
                            float _max_600 = max_noftz(m0v_3, v_35_4);
                            m0v_3 = _max_600;
                        }
                        float v_36_4 = neg;
                        {
                            if (c0_4 >= 73) {
                                {
                                    float _cvt_f32_f16_1183;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1183) : "h"((uint16_t)(acc[18] >> 16)));
                                    v_36_4 = _cvt_f32_f16_1183;
                                }
                            }
                            float _max_602 = max_noftz(m0v_3, v_36_4);
                            m0v_3 = _max_602;
                        }
                        float v_37_4 = neg;
                        {
                            if (c1_4 >= 72) {
                                {
                                    float _cvt_f32_f16_1188;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1188) : "h"((uint16_t)(acc[19] & 65535)));
                                    v_37_4 = _cvt_f32_f16_1188;
                                }
                            }
                            float _max_605 = max_noftz(m1v_3, v_37_4);
                            m1v_3 = _max_605;
                        }
                        float v_38_4 = neg;
                        {
                            if (c1_4 >= 73) {
                                {
                                    float _cvt_f32_f16_1193;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1193) : "h"((uint16_t)(acc[19] >> 16)));
                                    v_38_4 = _cvt_f32_f16_1193;
                                }
                            }
                            float _max_607 = max_noftz(m1v_3, v_38_4);
                            m1v_3 = _max_607;
                        }
                        float v_39_4 = neg;
                        {
                            if (c0_4 >= 80) {
                                {
                                    float _cvt_f32_f16_1194;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1194) : "h"((uint16_t)(acc[20] & 65535)));
                                    v_39_4 = _cvt_f32_f16_1194;
                                }
                            }
                            float _max_608 = max_noftz(m0v_3, v_39_4);
                            m0v_3 = _max_608;
                        }
                        float v_40_4 = neg;
                        {
                            if (c0_4 >= 81) {
                                {
                                    float _cvt_f32_f16_1199;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1199) : "h"((uint16_t)(acc[20] >> 16)));
                                    v_40_4 = _cvt_f32_f16_1199;
                                }
                            }
                            float _max_610 = max_noftz(m0v_3, v_40_4);
                            m0v_3 = _max_610;
                        }
                        float v_41_4 = neg;
                        {
                            if (c1_4 >= 80) {
                                {
                                    float _cvt_f32_f16_1204;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1204) : "h"((uint16_t)(acc[21] & 65535)));
                                    v_41_4 = _cvt_f32_f16_1204;
                                }
                            }
                            float _max_613 = max_noftz(m1v_3, v_41_4);
                            m1v_3 = _max_613;
                        }
                        float v_42_4 = neg;
                        {
                            if (c1_4 >= 81) {
                                {
                                    float _cvt_f32_f16_1209;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1209) : "h"((uint16_t)(acc[21] >> 16)));
                                    v_42_4 = _cvt_f32_f16_1209;
                                }
                            }
                            float _max_615 = max_noftz(m1v_3, v_42_4);
                            m1v_3 = _max_615;
                        }
                        float v_43_4 = neg;
                        {
                            if (c0_4 >= 88) {
                                {
                                    float _cvt_f32_f16_1210;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1210) : "h"((uint16_t)(acc[22] & 65535)));
                                    v_43_4 = _cvt_f32_f16_1210;
                                }
                            }
                            float _max_616 = max_noftz(m0v_3, v_43_4);
                            m0v_3 = _max_616;
                        }
                        float v_44_4 = neg;
                        {
                            if (c0_4 >= 89) {
                                {
                                    float _cvt_f32_f16_1215;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1215) : "h"((uint16_t)(acc[22] >> 16)));
                                    v_44_4 = _cvt_f32_f16_1215;
                                }
                            }
                            float _max_618 = max_noftz(m0v_3, v_44_4);
                            m0v_3 = _max_618;
                        }
                        float v_45_4 = neg;
                        {
                            if (c1_4 >= 88) {
                                {
                                    float _cvt_f32_f16_1220;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1220) : "h"((uint16_t)(acc[23] & 65535)));
                                    v_45_4 = _cvt_f32_f16_1220;
                                }
                            }
                            float _max_621 = max_noftz(m1v_3, v_45_4);
                            m1v_3 = _max_621;
                        }
                        float v_46_4 = neg;
                        {
                            if (c1_4 >= 89) {
                                {
                                    float _cvt_f32_f16_1225;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1225) : "h"((uint16_t)(acc[23] >> 16)));
                                    v_46_4 = _cvt_f32_f16_1225;
                                }
                            }
                            float _max_623 = max_noftz(m1v_3, v_46_4);
                            m1v_3 = _max_623;
                        }
                        float v_47_4 = neg;
                        {
                            if (c0_4 >= 96) {
                                {
                                    float _cvt_f32_f16_1226;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1226) : "h"((uint16_t)(acc[24] & 65535)));
                                    v_47_4 = _cvt_f32_f16_1226;
                                }
                            }
                            float _max_624 = max_noftz(m0v_3, v_47_4);
                            m0v_3 = _max_624;
                        }
                        float v_48_4 = neg;
                        {
                            if (c0_4 >= 97) {
                                {
                                    float _cvt_f32_f16_1231;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1231) : "h"((uint16_t)(acc[24] >> 16)));
                                    v_48_4 = _cvt_f32_f16_1231;
                                }
                            }
                            float _max_626 = max_noftz(m0v_3, v_48_4);
                            m0v_3 = _max_626;
                        }
                        float v_49_4 = neg;
                        {
                            if (c1_4 >= 96) {
                                {
                                    float _cvt_f32_f16_1236;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1236) : "h"((uint16_t)(acc[25] & 65535)));
                                    v_49_4 = _cvt_f32_f16_1236;
                                }
                            }
                            float _max_629 = max_noftz(m1v_3, v_49_4);
                            m1v_3 = _max_629;
                        }
                        float v_50_4 = neg;
                        {
                            if (c1_4 >= 97) {
                                {
                                    float _cvt_f32_f16_1241;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1241) : "h"((uint16_t)(acc[25] >> 16)));
                                    v_50_4 = _cvt_f32_f16_1241;
                                }
                            }
                            float _max_631 = max_noftz(m1v_3, v_50_4);
                            m1v_3 = _max_631;
                        }
                        float v_51_4 = neg;
                        {
                            if (c0_4 >= 104) {
                                {
                                    float _cvt_f32_f16_1242;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1242) : "h"((uint16_t)(acc[26] & 65535)));
                                    v_51_4 = _cvt_f32_f16_1242;
                                }
                            }
                            float _max_632 = max_noftz(m0v_3, v_51_4);
                            m0v_3 = _max_632;
                        }
                        float v_52_4 = neg;
                        {
                            if (c0_4 >= 105) {
                                {
                                    float _cvt_f32_f16_1247;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1247) : "h"((uint16_t)(acc[26] >> 16)));
                                    v_52_4 = _cvt_f32_f16_1247;
                                }
                            }
                            float _max_634 = max_noftz(m0v_3, v_52_4);
                            m0v_3 = _max_634;
                        }
                        float v_53_4 = neg;
                        {
                            if (c1_4 >= 104) {
                                {
                                    float _cvt_f32_f16_1252;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1252) : "h"((uint16_t)(acc[27] & 65535)));
                                    v_53_4 = _cvt_f32_f16_1252;
                                }
                            }
                            float _max_637 = max_noftz(m1v_3, v_53_4);
                            m1v_3 = _max_637;
                        }
                        float v_54_4 = neg;
                        {
                            if (c1_4 >= 105) {
                                {
                                    float _cvt_f32_f16_1257;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1257) : "h"((uint16_t)(acc[27] >> 16)));
                                    v_54_4 = _cvt_f32_f16_1257;
                                }
                            }
                            float _max_639 = max_noftz(m1v_3, v_54_4);
                            m1v_3 = _max_639;
                        }
                        float v_55_4 = neg;
                        {
                            if (c0_4 >= 112) {
                                {
                                    float _cvt_f32_f16_1258;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1258) : "h"((uint16_t)(acc[28] & 65535)));
                                    v_55_4 = _cvt_f32_f16_1258;
                                }
                            }
                            float _max_640 = max_noftz(m0v_3, v_55_4);
                            m0v_3 = _max_640;
                        }
                        float v_56_4 = neg;
                        {
                            if (c0_4 >= 113) {
                                {
                                    float _cvt_f32_f16_1263;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1263) : "h"((uint16_t)(acc[28] >> 16)));
                                    v_56_4 = _cvt_f32_f16_1263;
                                }
                            }
                            float _max_642 = max_noftz(m0v_3, v_56_4);
                            m0v_3 = _max_642;
                        }
                        float v_57_4 = neg;
                        {
                            if (c1_4 >= 112) {
                                {
                                    float _cvt_f32_f16_1268;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1268) : "h"((uint16_t)(acc[29] & 65535)));
                                    v_57_4 = _cvt_f32_f16_1268;
                                }
                            }
                            float _max_645 = max_noftz(m1v_3, v_57_4);
                            m1v_3 = _max_645;
                        }
                        float v_58_4 = neg;
                        {
                            if (c1_4 >= 113) {
                                {
                                    float _cvt_f32_f16_1273;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1273) : "h"((uint16_t)(acc[29] >> 16)));
                                    v_58_4 = _cvt_f32_f16_1273;
                                }
                            }
                            float _max_647 = max_noftz(m1v_3, v_58_4);
                            m1v_3 = _max_647;
                        }
                        float v_59_4 = neg;
                        {
                            if (c0_4 >= 120) {
                                {
                                    float _cvt_f32_f16_1274;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1274) : "h"((uint16_t)(acc[30] & 65535)));
                                    v_59_4 = _cvt_f32_f16_1274;
                                }
                            }
                            float _max_648 = max_noftz(m0v_3, v_59_4);
                            m0v_3 = _max_648;
                        }
                        float v_60_4 = neg;
                        {
                            if (c0_4 >= 121) {
                                {
                                    float _cvt_f32_f16_1279;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1279) : "h"((uint16_t)(acc[30] >> 16)));
                                    v_60_4 = _cvt_f32_f16_1279;
                                }
                            }
                            float _max_650 = max_noftz(m0v_3, v_60_4);
                            m0v_3 = _max_650;
                        }
                        float v_61_4 = neg;
                        {
                            if (c1_4 >= 120) {
                                {
                                    float _cvt_f32_f16_1284;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1284) : "h"((uint16_t)(acc[31] & 65535)));
                                    v_61_4 = _cvt_f32_f16_1284;
                                }
                            }
                            float _max_653 = max_noftz(m1v_3, v_61_4);
                            m1v_3 = _max_653;
                        }
                        float v_62_4 = neg;
                        {
                            if (c1_4 >= 121) {
                                {
                                    float _cvt_f32_f16_1289;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1289) : "h"((uint16_t)(acc[31] >> 16)));
                                    v_62_4 = _cvt_f32_f16_1289;
                                }
                            }
                            float _max_655 = max_noftz(m1v_3, v_62_4);
                            m1v_3 = _max_655;
                        }
                    }
                    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, m0v_3, 1);
                    float _max_656 = max_noftz(m0v_3, _shfl_xor_16);
                    m0v_3 = _max_656;
                    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, m0v_3, 2);
                    float _max_657 = max_noftz(m0v_3, _shfl_xor_17);
                    m0v_3 = _max_657;
                    float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, m1v_3, 1);
                    float _max_658 = max_noftz(m1v_3, _shfl_xor_18);
                    m1v_3 = _max_658;
                    float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, m1v_3, 2);
                    float _max_659 = max_noftz(m1v_3, _shfl_xor_19);
                    m1v_3 = _max_659;
                    if (cq == 0) {
                        int orow_4 = obase + (split2 + il * nsplit) * total_q + r0;
                        if (r0 < rows_valid) {
                            out[orow_4] = m0v_3;
                        }
                        if (rows_valid > r0 + 8) {
                            out[orow_4 + 8] = m1v_3;
                        }
                    }
                }
            }
            float negs[4];
            negs[0] = neg;
            negs[1] = neg;
            negs[2] = neg;
            negs[3] = neg;
            int tq = (tid_1 & 31) * 4;
            int tw = tid_1 >> 5;
            int n_dead_it4 = (n_dead + 12 - 1) / 12;
            #pragma unroll 1
            for (int d = 0; d < n_dead_it4; d++) {
                int di4 = 12 * d + tw;
                if (di4 < n_dead) {
                    int td4 = t_lim2 + split2 + di4 * nsplit;
                    int ob = obase + td4 * total_q;
                    int al = ob & 3;
                    if (al == 0) {
                        if (rows_valid > tq + 3) {
                            {
                                float4 _v4 = make_float4(negs[0 + 0], negs[0 + 1], negs[0 + 2], negs[0 + 3]);
                                *reinterpret_cast<float4*>(out + ob + tq) = _v4;
                            }
                        }
                        if (rows_valid <= tq + 3) {
                            if (rows_valid > tq) {
                                out[ob + tq] = neg;
                            }
                            if (rows_valid > tq + 1) {
                                out[ob + tq + 1] = neg;
                            }
                            if (rows_valid > tq + 2) {
                                out[ob + tq + 2] = neg;
                            }
                            if (rows_valid > tq + 3) {
                                out[ob + tq + 3] = neg;
                            }
                        }
                    }
                    if (al != 0) {
                        if (rows_valid > tq) {
                            out[ob + tq] = neg;
                        }
                        if (rows_valid > tq + 1) {
                            out[ob + tq + 1] = neg;
                        }
                        if (rows_valid > tq + 2) {
                            out[ob + tq + 2] = neg;
                        }
                        if (rows_valid > tq + 3) {
                            out[ob + tq + 3] = neg;
                        }
                    }
                    int tqB = tq + 128;
                    int rvB = rows_valid;
                    if (rvB > 192) {
                        rvB = 192;
                    }
                    if (al == 0) {
                        if (rvB > tqB + 3) {
                            {
                                float4 _v4 = make_float4(negs[0 + 0], negs[0 + 1], negs[0 + 2], negs[0 + 3]);
                                *reinterpret_cast<float4*>(out + ob + tqB) = _v4;
                            }
                        }
                        if (rvB <= tqB + 3) {
                            if (rvB > tqB) {
                                out[ob + tqB] = neg;
                            }
                            if (rvB > tqB + 1) {
                                out[ob + tqB + 1] = neg;
                            }
                            if (rvB > tqB + 2) {
                                out[ob + tqB + 2] = neg;
                            }
                            if (rvB > tqB + 3) {
                                out[ob + tqB + 3] = neg;
                            }
                        }
                    }
                    if (al != 0) {
                        if (rvB > tqB) {
                            out[ob + tqB] = neg;
                        }
                        if (rvB > tqB + 1) {
                            out[ob + tqB + 1] = neg;
                        }
                        if (rvB > tqB + 2) {
                            out[ob + tqB + 2] = neg;
                        }
                        if (rvB > tqB + 3) {
                            out[ob + tqB + 3] = neg;
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
