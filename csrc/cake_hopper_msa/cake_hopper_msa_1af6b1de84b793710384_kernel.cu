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
#define SMEM_Q_S_STAGE_BYTES 32768
#define SMEM_Q_S_STRIDE 32768
#define SMEM_K_S_OFF 33792
#define SMEM_K_S_STAGE_BYTES 16384
#define SMEM_K_S_STRIDE 16384
#define SMEM_TOTAL 99328
#define THREADS 544
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

__global__ __launch_bounds__(544, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_1af6b1de84b793710384(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, float* __restrict__ out, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int has_qoff, int pt_stride, int max_k_tiles, int total_q, int num_heads, int nsplit, unsigned long long* __restrict__ trace)
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
    uint8_t* k_s = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int k_s_addr = smem + 33792;

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
            // empty: 4 barriers, init_count=16
            mbarrier_init(smem + 32, 16);
            mbarrier_init(smem + 40, 16);
            mbarrier_init(smem + 48, 16);
            mbarrier_init(smem + 56, 16);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp == 16) {
        { // producer_main
            if (warp == 16) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
                int bx = blockIdx.x;
                int split = blockIdx.y;
                int b = blockIdx.z;
                int mt = gridDim.x / num_heads - 1 - bx / num_heads;
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
                int m0 = mt * 256;
                int mlim = sqb - 256;
                if (m0 > mlim) {
                    m0 = mlim;
                }
                if (m0 < 0) {
                    m0 = 0;
                }
                int nb = (sk + 127) / 128;
                int qmax = m0 + 255;
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
                                mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 49152);
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
    if (warp <= 15) {
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
            int h2 = bx2 % num_heads;
            int qlo2 = cu_seqlens_q[b2];
            int sqb2 = cu_seqlens_q[b2 + 1] - qlo2;
            int sk2 = seqused_k[b2];
            int pfx2 = sk2 - sqb2;
            if (has_qoff != 0) {
                pfx2 = q_offset[b2];
            }
            int m02 = mt2 * 256;
            int mlim2 = sqb2 - 256;
            if (m02 > mlim2) {
                m02 = mlim2;
            }
            if (m02 < 0) {
                m02 = 0;
            }
            int nb2 = (sk2 + 127) / 128;
            int qmax2 = m02 + 255;
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
            float acc[64];
            int sodd = cq & 1;
            int srow = r0 + 8 * sodd;
            int st_ok = 0;
            if (cq < 2) {
                if (srow < rows_valid) {
                    st_ok = 1;
                }
            }
            int sobase = obase + srow;
            int first = 1;
            uint64_t _wgmma_desc_0 = (((uint64_t)(((q_s_addr + (unsigned int)(wgi * 64 * 128))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
            #pragma unroll 1
            for (int i_1 = 0; i_1 < n_comp2; i_1++) {
                int stage2 = i_1 % 4;
                int cphase = i_1 / 4 & 1;
                int t = split2 + i_1 * nsplit;
                mbarrier_wait(full_addr + (stage2) * 8, cphase);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                uint64_t _wgmma_desc_1 = (((uint64_t)(((k_s_addr + (unsigned int)(stage2 * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1;\n}\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
                    : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
                    : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
                    : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])
                    : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(empty_addr + (stage2) * 8);
                }
                int k0 = t * 128;
                float m0v = neg;
                float m1v = neg;
                int need_mask = 0;
                if (mask_thr < k0 + 127) {
                    need_mask = 1;
                }
                if (need_mask == 0) {
                    float _max_0 = max_noftz(acc[0], acc[1]);
                    float _max_1 = max_noftz(m0v, _max_0);
                    m0v = _max_1;
                    float _max_2 = max_noftz(acc[2], acc[3]);
                    float _max_3 = max_noftz(m1v, _max_2);
                    m1v = _max_3;
                    float _max_4 = max_noftz(acc[4], acc[5]);
                    float _max_5 = max_noftz(m0v, _max_4);
                    m0v = _max_5;
                    float _max_6 = max_noftz(acc[6], acc[7]);
                    float _max_7 = max_noftz(m1v, _max_6);
                    m1v = _max_7;
                    float _max_8 = max_noftz(acc[8], acc[9]);
                    float _max_9 = max_noftz(m0v, _max_8);
                    m0v = _max_9;
                    float _max_10 = max_noftz(acc[10], acc[11]);
                    float _max_11 = max_noftz(m1v, _max_10);
                    m1v = _max_11;
                    float _max_12 = max_noftz(acc[12], acc[13]);
                    float _max_13 = max_noftz(m0v, _max_12);
                    m0v = _max_13;
                    float _max_14 = max_noftz(acc[14], acc[15]);
                    float _max_15 = max_noftz(m1v, _max_14);
                    m1v = _max_15;
                    float _max_16 = max_noftz(acc[16], acc[17]);
                    float _max_17 = max_noftz(m0v, _max_16);
                    m0v = _max_17;
                    float _max_18 = max_noftz(acc[18], acc[19]);
                    float _max_19 = max_noftz(m1v, _max_18);
                    m1v = _max_19;
                    float _max_20 = max_noftz(acc[20], acc[21]);
                    float _max_21 = max_noftz(m0v, _max_20);
                    m0v = _max_21;
                    float _max_22 = max_noftz(acc[22], acc[23]);
                    float _max_23 = max_noftz(m1v, _max_22);
                    m1v = _max_23;
                    float _max_24 = max_noftz(acc[24], acc[25]);
                    float _max_25 = max_noftz(m0v, _max_24);
                    m0v = _max_25;
                    float _max_26 = max_noftz(acc[26], acc[27]);
                    float _max_27 = max_noftz(m1v, _max_26);
                    m1v = _max_27;
                    float _max_28 = max_noftz(acc[28], acc[29]);
                    float _max_29 = max_noftz(m0v, _max_28);
                    m0v = _max_29;
                    float _max_30 = max_noftz(acc[30], acc[31]);
                    float _max_31 = max_noftz(m1v, _max_30);
                    m1v = _max_31;
                    float _max_32 = max_noftz(acc[32], acc[33]);
                    float _max_33 = max_noftz(m0v, _max_32);
                    m0v = _max_33;
                    float _max_34 = max_noftz(acc[34], acc[35]);
                    float _max_35 = max_noftz(m1v, _max_34);
                    m1v = _max_35;
                    float _max_36 = max_noftz(acc[36], acc[37]);
                    float _max_37 = max_noftz(m0v, _max_36);
                    m0v = _max_37;
                    float _max_38 = max_noftz(acc[38], acc[39]);
                    float _max_39 = max_noftz(m1v, _max_38);
                    m1v = _max_39;
                    float _max_40 = max_noftz(acc[40], acc[41]);
                    float _max_41 = max_noftz(m0v, _max_40);
                    m0v = _max_41;
                    float _max_42 = max_noftz(acc[42], acc[43]);
                    float _max_43 = max_noftz(m1v, _max_42);
                    m1v = _max_43;
                    float _max_44 = max_noftz(acc[44], acc[45]);
                    float _max_45 = max_noftz(m0v, _max_44);
                    m0v = _max_45;
                    float _max_46 = max_noftz(acc[46], acc[47]);
                    float _max_47 = max_noftz(m1v, _max_46);
                    m1v = _max_47;
                    float _max_48 = max_noftz(acc[48], acc[49]);
                    float _max_49 = max_noftz(m0v, _max_48);
                    m0v = _max_49;
                    float _max_50 = max_noftz(acc[50], acc[51]);
                    float _max_51 = max_noftz(m1v, _max_50);
                    m1v = _max_51;
                    float _max_52 = max_noftz(acc[52], acc[53]);
                    float _max_53 = max_noftz(m0v, _max_52);
                    m0v = _max_53;
                    float _max_54 = max_noftz(acc[54], acc[55]);
                    float _max_55 = max_noftz(m1v, _max_54);
                    m1v = _max_55;
                    float _max_56 = max_noftz(acc[56], acc[57]);
                    float _max_57 = max_noftz(m0v, _max_56);
                    m0v = _max_57;
                    float _max_58 = max_noftz(acc[58], acc[59]);
                    float _max_59 = max_noftz(m1v, _max_58);
                    m1v = _max_59;
                    float _max_60 = max_noftz(acc[60], acc[61]);
                    float _max_61 = max_noftz(m0v, _max_60);
                    m0v = _max_61;
                    float _max_62 = max_noftz(acc[62], acc[63]);
                    float _max_63 = max_noftz(m1v, _max_62);
                    m1v = _max_63;
                }
                if (need_mask != 0) {
                    int c0 = lim0 - k0 - 2 * cq;
                    int c1 = lim1 - k0 - 2 * cq;
                    float v = neg;
                    {
                        if (c0 >= 0) {
                            v = acc[0];
                        }
                        float _max_64 = max_noftz(m0v, v);
                        m0v = _max_64;
                    }
                    float v_0 = neg;
                    {
                        if (c0 >= 1) {
                            v_0 = acc[1];
                        }
                        float _max_66 = max_noftz(m0v, v_0);
                        m0v = _max_66;
                    }
                    float v_1 = neg;
                    {
                        if (c1 >= 0) {
                            v_1 = acc[2];
                        }
                        float _max_69 = max_noftz(m1v, v_1);
                        m1v = _max_69;
                    }
                    float v_2 = neg;
                    {
                        if (c1 >= 1) {
                            v_2 = acc[3];
                        }
                        float _max_71 = max_noftz(m1v, v_2);
                        m1v = _max_71;
                    }
                    float v_3 = neg;
                    {
                        if (c0 >= 8) {
                            v_3 = acc[4];
                        }
                        float _max_72 = max_noftz(m0v, v_3);
                        m0v = _max_72;
                    }
                    float v_4 = neg;
                    {
                        if (c0 >= 9) {
                            v_4 = acc[5];
                        }
                        float _max_74 = max_noftz(m0v, v_4);
                        m0v = _max_74;
                    }
                    float v_5 = neg;
                    {
                        if (c1 >= 8) {
                            v_5 = acc[6];
                        }
                        float _max_77 = max_noftz(m1v, v_5);
                        m1v = _max_77;
                    }
                    float v_6 = neg;
                    {
                        if (c1 >= 9) {
                            v_6 = acc[7];
                        }
                        float _max_79 = max_noftz(m1v, v_6);
                        m1v = _max_79;
                    }
                    float v_7 = neg;
                    {
                        if (c0 >= 16) {
                            v_7 = acc[8];
                        }
                        float _max_80 = max_noftz(m0v, v_7);
                        m0v = _max_80;
                    }
                    float v_8 = neg;
                    {
                        if (c0 >= 17) {
                            v_8 = acc[9];
                        }
                        float _max_82 = max_noftz(m0v, v_8);
                        m0v = _max_82;
                    }
                    float v_9 = neg;
                    {
                        if (c1 >= 16) {
                            v_9 = acc[10];
                        }
                        float _max_85 = max_noftz(m1v, v_9);
                        m1v = _max_85;
                    }
                    float v_10 = neg;
                    {
                        if (c1 >= 17) {
                            v_10 = acc[11];
                        }
                        float _max_87 = max_noftz(m1v, v_10);
                        m1v = _max_87;
                    }
                    float v_11 = neg;
                    {
                        if (c0 >= 24) {
                            v_11 = acc[12];
                        }
                        float _max_88 = max_noftz(m0v, v_11);
                        m0v = _max_88;
                    }
                    float v_12 = neg;
                    {
                        if (c0 >= 25) {
                            v_12 = acc[13];
                        }
                        float _max_90 = max_noftz(m0v, v_12);
                        m0v = _max_90;
                    }
                    float v_13 = neg;
                    {
                        if (c1 >= 24) {
                            v_13 = acc[14];
                        }
                        float _max_93 = max_noftz(m1v, v_13);
                        m1v = _max_93;
                    }
                    float v_14 = neg;
                    {
                        if (c1 >= 25) {
                            v_14 = acc[15];
                        }
                        float _max_95 = max_noftz(m1v, v_14);
                        m1v = _max_95;
                    }
                    float v_15 = neg;
                    {
                        if (c0 >= 32) {
                            v_15 = acc[16];
                        }
                        float _max_96 = max_noftz(m0v, v_15);
                        m0v = _max_96;
                    }
                    float v_16 = neg;
                    {
                        if (c0 >= 33) {
                            v_16 = acc[17];
                        }
                        float _max_98 = max_noftz(m0v, v_16);
                        m0v = _max_98;
                    }
                    float v_17 = neg;
                    {
                        if (c1 >= 32) {
                            v_17 = acc[18];
                        }
                        float _max_101 = max_noftz(m1v, v_17);
                        m1v = _max_101;
                    }
                    float v_18 = neg;
                    {
                        if (c1 >= 33) {
                            v_18 = acc[19];
                        }
                        float _max_103 = max_noftz(m1v, v_18);
                        m1v = _max_103;
                    }
                    float v_19 = neg;
                    {
                        if (c0 >= 40) {
                            v_19 = acc[20];
                        }
                        float _max_104 = max_noftz(m0v, v_19);
                        m0v = _max_104;
                    }
                    float v_20 = neg;
                    {
                        if (c0 >= 41) {
                            v_20 = acc[21];
                        }
                        float _max_106 = max_noftz(m0v, v_20);
                        m0v = _max_106;
                    }
                    float v_21 = neg;
                    {
                        if (c1 >= 40) {
                            v_21 = acc[22];
                        }
                        float _max_109 = max_noftz(m1v, v_21);
                        m1v = _max_109;
                    }
                    float v_22 = neg;
                    {
                        if (c1 >= 41) {
                            v_22 = acc[23];
                        }
                        float _max_111 = max_noftz(m1v, v_22);
                        m1v = _max_111;
                    }
                    float v_23 = neg;
                    {
                        if (c0 >= 48) {
                            v_23 = acc[24];
                        }
                        float _max_112 = max_noftz(m0v, v_23);
                        m0v = _max_112;
                    }
                    float v_24 = neg;
                    {
                        if (c0 >= 49) {
                            v_24 = acc[25];
                        }
                        float _max_114 = max_noftz(m0v, v_24);
                        m0v = _max_114;
                    }
                    float v_25 = neg;
                    {
                        if (c1 >= 48) {
                            v_25 = acc[26];
                        }
                        float _max_117 = max_noftz(m1v, v_25);
                        m1v = _max_117;
                    }
                    float v_26 = neg;
                    {
                        if (c1 >= 49) {
                            v_26 = acc[27];
                        }
                        float _max_119 = max_noftz(m1v, v_26);
                        m1v = _max_119;
                    }
                    float v_27 = neg;
                    {
                        if (c0 >= 56) {
                            v_27 = acc[28];
                        }
                        float _max_120 = max_noftz(m0v, v_27);
                        m0v = _max_120;
                    }
                    float v_28 = neg;
                    {
                        if (c0 >= 57) {
                            v_28 = acc[29];
                        }
                        float _max_122 = max_noftz(m0v, v_28);
                        m0v = _max_122;
                    }
                    float v_29 = neg;
                    {
                        if (c1 >= 56) {
                            v_29 = acc[30];
                        }
                        float _max_125 = max_noftz(m1v, v_29);
                        m1v = _max_125;
                    }
                    float v_30 = neg;
                    {
                        if (c1 >= 57) {
                            v_30 = acc[31];
                        }
                        float _max_127 = max_noftz(m1v, v_30);
                        m1v = _max_127;
                    }
                    float v_31 = neg;
                    {
                        if (c0 >= 64) {
                            v_31 = acc[32];
                        }
                        float _max_128 = max_noftz(m0v, v_31);
                        m0v = _max_128;
                    }
                    float v_32 = neg;
                    {
                        if (c0 >= 65) {
                            v_32 = acc[33];
                        }
                        float _max_130 = max_noftz(m0v, v_32);
                        m0v = _max_130;
                    }
                    float v_33 = neg;
                    {
                        if (c1 >= 64) {
                            v_33 = acc[34];
                        }
                        float _max_133 = max_noftz(m1v, v_33);
                        m1v = _max_133;
                    }
                    float v_34 = neg;
                    {
                        if (c1 >= 65) {
                            v_34 = acc[35];
                        }
                        float _max_135 = max_noftz(m1v, v_34);
                        m1v = _max_135;
                    }
                    float v_35 = neg;
                    {
                        if (c0 >= 72) {
                            v_35 = acc[36];
                        }
                        float _max_136 = max_noftz(m0v, v_35);
                        m0v = _max_136;
                    }
                    float v_36 = neg;
                    {
                        if (c0 >= 73) {
                            v_36 = acc[37];
                        }
                        float _max_138 = max_noftz(m0v, v_36);
                        m0v = _max_138;
                    }
                    float v_37 = neg;
                    {
                        if (c1 >= 72) {
                            v_37 = acc[38];
                        }
                        float _max_141 = max_noftz(m1v, v_37);
                        m1v = _max_141;
                    }
                    float v_38 = neg;
                    {
                        if (c1 >= 73) {
                            v_38 = acc[39];
                        }
                        float _max_143 = max_noftz(m1v, v_38);
                        m1v = _max_143;
                    }
                    float v_39 = neg;
                    {
                        if (c0 >= 80) {
                            v_39 = acc[40];
                        }
                        float _max_144 = max_noftz(m0v, v_39);
                        m0v = _max_144;
                    }
                    float v_40 = neg;
                    {
                        if (c0 >= 81) {
                            v_40 = acc[41];
                        }
                        float _max_146 = max_noftz(m0v, v_40);
                        m0v = _max_146;
                    }
                    float v_41 = neg;
                    {
                        if (c1 >= 80) {
                            v_41 = acc[42];
                        }
                        float _max_149 = max_noftz(m1v, v_41);
                        m1v = _max_149;
                    }
                    float v_42 = neg;
                    {
                        if (c1 >= 81) {
                            v_42 = acc[43];
                        }
                        float _max_151 = max_noftz(m1v, v_42);
                        m1v = _max_151;
                    }
                    float v_43 = neg;
                    {
                        if (c0 >= 88) {
                            v_43 = acc[44];
                        }
                        float _max_152 = max_noftz(m0v, v_43);
                        m0v = _max_152;
                    }
                    float v_44 = neg;
                    {
                        if (c0 >= 89) {
                            v_44 = acc[45];
                        }
                        float _max_154 = max_noftz(m0v, v_44);
                        m0v = _max_154;
                    }
                    float v_45 = neg;
                    {
                        if (c1 >= 88) {
                            v_45 = acc[46];
                        }
                        float _max_157 = max_noftz(m1v, v_45);
                        m1v = _max_157;
                    }
                    float v_46 = neg;
                    {
                        if (c1 >= 89) {
                            v_46 = acc[47];
                        }
                        float _max_159 = max_noftz(m1v, v_46);
                        m1v = _max_159;
                    }
                    float v_47 = neg;
                    {
                        if (c0 >= 96) {
                            v_47 = acc[48];
                        }
                        float _max_160 = max_noftz(m0v, v_47);
                        m0v = _max_160;
                    }
                    float v_48 = neg;
                    {
                        if (c0 >= 97) {
                            v_48 = acc[49];
                        }
                        float _max_162 = max_noftz(m0v, v_48);
                        m0v = _max_162;
                    }
                    float v_49 = neg;
                    {
                        if (c1 >= 96) {
                            v_49 = acc[50];
                        }
                        float _max_165 = max_noftz(m1v, v_49);
                        m1v = _max_165;
                    }
                    float v_50 = neg;
                    {
                        if (c1 >= 97) {
                            v_50 = acc[51];
                        }
                        float _max_167 = max_noftz(m1v, v_50);
                        m1v = _max_167;
                    }
                    float v_51 = neg;
                    {
                        if (c0 >= 104) {
                            v_51 = acc[52];
                        }
                        float _max_168 = max_noftz(m0v, v_51);
                        m0v = _max_168;
                    }
                    float v_52 = neg;
                    {
                        if (c0 >= 105) {
                            v_52 = acc[53];
                        }
                        float _max_170 = max_noftz(m0v, v_52);
                        m0v = _max_170;
                    }
                    float v_53 = neg;
                    {
                        if (c1 >= 104) {
                            v_53 = acc[54];
                        }
                        float _max_173 = max_noftz(m1v, v_53);
                        m1v = _max_173;
                    }
                    float v_54 = neg;
                    {
                        if (c1 >= 105) {
                            v_54 = acc[55];
                        }
                        float _max_175 = max_noftz(m1v, v_54);
                        m1v = _max_175;
                    }
                    float v_55 = neg;
                    {
                        if (c0 >= 112) {
                            v_55 = acc[56];
                        }
                        float _max_176 = max_noftz(m0v, v_55);
                        m0v = _max_176;
                    }
                    float v_56 = neg;
                    {
                        if (c0 >= 113) {
                            v_56 = acc[57];
                        }
                        float _max_178 = max_noftz(m0v, v_56);
                        m0v = _max_178;
                    }
                    float v_57 = neg;
                    {
                        if (c1 >= 112) {
                            v_57 = acc[58];
                        }
                        float _max_181 = max_noftz(m1v, v_57);
                        m1v = _max_181;
                    }
                    float v_58 = neg;
                    {
                        if (c1 >= 113) {
                            v_58 = acc[59];
                        }
                        float _max_183 = max_noftz(m1v, v_58);
                        m1v = _max_183;
                    }
                    float v_59 = neg;
                    {
                        if (c0 >= 120) {
                            v_59 = acc[60];
                        }
                        float _max_184 = max_noftz(m0v, v_59);
                        m0v = _max_184;
                    }
                    float v_60 = neg;
                    {
                        if (c0 >= 121) {
                            v_60 = acc[61];
                        }
                        float _max_186 = max_noftz(m0v, v_60);
                        m0v = _max_186;
                    }
                    float v_61 = neg;
                    {
                        if (c1 >= 120) {
                            v_61 = acc[62];
                        }
                        float _max_189 = max_noftz(m1v, v_61);
                        m1v = _max_189;
                    }
                    float v_62 = neg;
                    {
                        if (c1 >= 121) {
                            v_62 = acc[63];
                        }
                        float _max_191 = max_noftz(m1v, v_62);
                        m1v = _max_191;
                    }
                }
                float keep = m0v;
                float send = m1v;
                if (sodd != 0) {
                    keep = m1v;
                    send = m0v;
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                float _max_192 = max_noftz(keep, _shfl_xor_0);
                keep = _max_192;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, keep, 2);
                float _max_193 = max_noftz(keep, _shfl_xor_1);
                keep = _max_193;
                if (st_ok != 0) {
                    out[sobase + t * total_q] = keep;
                }
            }
            float negs[4];
            negs[0] = neg;
            negs[1] = neg;
            negs[2] = neg;
            negs[3] = neg;
            int tq = (tid_1 & 31) * 4;
            int tw = tid_1 >> 5;
            int n_dead_it4 = (n_dead + 16 - 1) / 16;
            #pragma unroll 1
            for (int d = 0; d < n_dead_it4; d++) {
                int di4 = 16 * d + tw;
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
                    if (al == 0) {
                        if (rows_valid > tqB + 3) {
                            {
                                float4 _v4 = make_float4(negs[0 + 0], negs[0 + 1], negs[0 + 2], negs[0 + 3]);
                                *reinterpret_cast<float4*>(out + ob + tqB) = _v4;
                            }
                        }
                        if (rows_valid <= tqB + 3) {
                            if (rows_valid > tqB) {
                                out[ob + tqB] = neg;
                            }
                            if (rows_valid > tqB + 1) {
                                out[ob + tqB + 1] = neg;
                            }
                            if (rows_valid > tqB + 2) {
                                out[ob + tqB + 2] = neg;
                            }
                            if (rows_valid > tqB + 3) {
                                out[ob + tqB + 3] = neg;
                            }
                        }
                    }
                    if (al != 0) {
                        if (rows_valid > tqB) {
                            out[ob + tqB] = neg;
                        }
                        if (rows_valid > tqB + 1) {
                            out[ob + tqB + 1] = neg;
                        }
                        if (rows_valid > tqB + 2) {
                            out[ob + tqB + 2] = neg;
                        }
                        if (rows_valid > tqB + 3) {
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
