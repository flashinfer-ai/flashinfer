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
#define TMEM_NCOLS 192
#define TMEM_ACCUM_OFFSET 0
#define NUM_STAGE_PIPE_STAGES 8
#define NUM_PACK_PIPE_STAGES 14
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_SF_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_CVT_OFF 132096
#define SMEM_SMEM_B_CVT_STAGE_BYTES 6144
#define SMEM_SMEM_B_CVT_STRIDE 6144
#define SMEM_SMEM_BPACK_OFF 181248
#define SMEM_SMEM_BPACK_STAGE_BYTES 1536
#define SMEM_SMEM_BPACK_STRIDE 1536
#define SMEM_SMEM_SF_OFF 202752
#define SMEM_SMEM_SF_STAGE_BYTES 768
#define SMEM_SMEM_SF_STRIDE 768
#define SMEM_TOTAL 205824
#define HAS_ALPHA 1
#define ENABLE_PDL 1

__device__ __forceinline__ uint32_t cvt_bf16x2_e4m3x2_p112(uint16_t src) {
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(dst) : "h"(src));
#else
    uint32_t _f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(src));
    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
    float _f0;
    float _f1;
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(dst) : "f"(_f1), "f"(_f0));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst) : "r"(dst), "r"(0x77807780u));
#endif
    return dst;
}

__device__ __forceinline__ uint32_t cvt_bf16x2_e2m1x2_prescaled(uint32_t src) {
    uint16_t _fp4_u16 = (uint16_t)(src & 0xFFu);
    uint32_t dst;
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _fp4b, _fp4z;                 \n\t"
        "  mov.b16 {_fp4b, _fp4z}, %1;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _fp4b;        }"
        : "=r"(dst) : "h"(_fp4_u16));
#else
    uint32_t _fp4_f16x2;
    asm("{ .reg .b8 _fp4b, _fp4z0, _fp4z1, _fp4z2;      \n\t"
        "  mov.b32 {_fp4b, _fp4z0, _fp4z1, _fp4z2}, %1;   \n\t"
        "  cvt.rn.f16x2.e2m1x2 %0, _fp4b;              }"
        : "=r"(_fp4_f16x2) : "r"((uint32_t)_fp4_u16));
    dst = ((_fp4_f16x2 & 0x7FFF7FFFu) >> 3) | (_fp4_f16x2 & 0x80008000u);
#endif
    return dst;
}

__device__ __forceinline__ void cvt_bf16x8_e2m1x16_e4m3(uint32_t packed0, uint32_t packed1, uint32_t scale_byte, uint32_t (&dst)[8]) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("{ .reg .b8 _q0, _q1, _q2, _q3, _q4, _q5, _q6, _q7, _sb, _sz0, _sz1, _sz2; \n\t"
        "  .reg .b16 _sx2;                            \n\t"
        "  .reg .b32 _sc;                             \n\t"
        "  mov.b32 {_q0, _q1, _q2, _q3}, %8;          \n\t"
        "  mov.b32 {_q4, _q5, _q6, _q7}, %9;          \n\t"
        "  mov.b32 {_sb, _sz0, _sz1, _sz2}, %10;      \n\t"
        "  mov.b16 _sx2, {_sb, _sb};                  \n\t"
        "  cvt.rn.bf16x2.e4m3x2 _sc, _sx2;            \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %0, _q0;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, _q1;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %2, _q2;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %3, _q3;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %4, _q4;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %5, _q5;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %6, _q6;        \n\t"
        "  cvt.rn.bf16x2.e2m1x2 %7, _q7;        \n\t"
        "  mul.rn.bf16x2 %0, %0, _sc;            \n\t"
        "  mul.rn.bf16x2 %1, %1, _sc;            \n\t"
        "  mul.rn.bf16x2 %2, %2, _sc;            \n\t"
        "  mul.rn.bf16x2 %3, %3, _sc;            \n\t"
        "  mul.rn.bf16x2 %4, %4, _sc;            \n\t"
        "  mul.rn.bf16x2 %5, %5, _sc;            \n\t"
        "  mul.rn.bf16x2 %6, %6, _sc;            \n\t"
        "  mul.rn.bf16x2 %7, %7, _sc;            \n\t"
        "}"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]), "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(packed0), "r"(packed1), "r"(scale_byte));
#else
    uint32_t _scale_p112 = cvt_bf16x2_e4m3x2_p112((uint16_t)(scale_byte | (scale_byte << 8)));
#pragma unroll
    for (int _k = 0; _k < 4; ++_k) {
        uint32_t _lo = cvt_bf16x2_e2m1x2_prescaled(packed0 >> (8 * _k));
        uint32_t _hi = cvt_bf16x2_e2m1x2_prescaled(packed1 >> (8 * _k));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[_k]) : "r"(_lo), "r"(_scale_p112));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(dst[4 + _k]) : "r"(_hi), "r"(_scale_p112));
    }
#endif
}

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(512) __cluster_dims__(2,1,1) void
kernel_cake_blackwell_bf16_fp4_f09a6b112cd48156248b(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap B_descale, float* __restrict__ alpha, __nv_bfloat16* __restrict__ C, int M, int N, int K, int grid_m, int grid_n)
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
    #define stage_full_addr (mbar_base + 0)
    #define stage_done_addr (mbar_base + 64)
    #define bsrc_full_addr (mbar_base + 128)
    #define bsrc_done_addr (mbar_base + 240)
    #define sf_full_addr (mbar_base + 352)
    #define sf_done_addr (mbar_base + 384)
    #define mainloop_done_addr (mbar_base + 416)
    #define epilogue_done_addr (mbar_base + 432)
    #define tmem_dealloc_bar_addr (mbar_base + 448)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b_cvt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_b_cvt_addr = smem + 132096;
    uint8_t* smem_bpack = reinterpret_cast<uint8_t*>(smem_raw + 181248);
    const int smem_bpack_addr = smem + 181248;
    uint8_t* smem_sf = reinterpret_cast<uint8_t*>(smem_raw + 202752);
    const int smem_sf_addr = smem + 202752;
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 6) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 7) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_descale))) : "memory"); }

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 57 barriers)
    // Mbarriers at smem_raw[0..456)

    if (warp == 0) {
        // --- pipeline 'stage_pipe' ---
        // stage_full: 8 barriers, init_count=6
        // stage_done: 8 barriers, init_count=1
        // --- pipeline 'pack_pipe' ---
        // bsrc_full: 14 barriers, init_count=1
        // bsrc_done: 14 barriers, init_count=2
        // --- pipeline 'sf_pipe' ---
        // sf_full: 4 barriers, init_count=1
        // sf_done: 4 barriers, init_count=8
        // --- pipeline 'mainloop_pipe' ---
        // mainloop_done: 2 barriers, init_count=1
        // epilogue_done: 2 barriers, init_count=8
        // tmem_dealloc_bar: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 2;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(6)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(24), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(22), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(20), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(12), "r"((uint32_t)(2)));
        if (lane < 25) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 456);
    if (warp == 0) {
        int _tmem_hold = smem + 456;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    if (warp <= 4) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(7), "r"(160) : "memory");
    }
    const int taddr = (warp <= 4) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            float alpha_value = 1.0f;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            alpha_value = alpha[0];
            int vec_ok = ((N % 16 == 0) ? 1 : 0);
            float vals[16];
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < grid_m * grid_n; tile_idx += num_clusters) {
                int m_tile = tile_idx / (unsigned int)grid_n;
                int n_tile = tile_idx % (unsigned int)grid_n;
                int off_m = m_tile * 256;
                int tmem_row = epi_warp * 32;
                off_m = off_m + cta_rank * 128;
                tmem_row = tmem_row + cta_rank * 128;
                int off_n = n_tile * 96;
                int row_global = off_m + epi_tid;
                long long row_linear = (long long)row_global * (long long)N;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tmem_col = epi_stage * 96;
                int tmem_addr = taddr + (unsigned int)((tmem_row << 16) + tmem_col);
                float _tmem_load_0[48];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                tmem_ld_x16(&_tmem_load_0[32], tmem_addr + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int sub16 = 0; sub16 < 3; sub16++) {
                    int col_global = off_n + sub16 * 16;
                    if (row_global < M && col_global < N) {
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            vals[i] = _tmem_load_0[sub16 * 16 + i] * alpha_value;
                        }
                        if (vec_ok == 1 && col_global + 16 <= N) {
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 2], vals[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 4], vals[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 6], vals[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 8], vals[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 10], vals[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 12], vals[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 14], vals[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(C + (row_linear + (long long)col_global))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                            }
                        } else {
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 16; i_1++) {
                                if (col_global + i_1 < N) {
                                    float value = vals[i_1];
                                    *(reinterpret_cast<__nv_bfloat16*>(C + (row_linear + (long long)(col_global + i_1))) + (0)) = (__nv_bfloat16)value;
                                }
                            }
                        }
                    }
                }
                int tmem_col_0 = epi_stage * 96 + 48;
                int tmem_addr_1 = taddr + (unsigned int)((tmem_row << 16) + tmem_col_0);
                float _tmem_load_1[48];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(tmem_addr_1));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                tmem_ld_x16(&_tmem_load_1[32], tmem_addr_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                #pragma unroll
                for (int sub16_1 = 0; sub16_1 < 3; sub16_1++) {
                    int col_global_1 = off_n + 48 + sub16_1 * 16;
                    if (row_global < M && col_global_1 < N) {
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 16; i_2++) {
                            vals[i_2] = _tmem_load_1[sub16_1 * 16 + i_2] * alpha_value;
                        }
                        if (vec_ok == 1 && col_global_1 + 16 <= N) {
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 2], vals[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 4], vals[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 6], vals[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 8], vals[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 10], vals[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 12], vals[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 14], vals[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            {
                                const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(C + (row_linear + (long long)col_global_1))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                            }
                        } else {
                            #pragma unroll
                            for (int i_3 = 0; i_3 < 16; i_3++) {
                                if (col_global_1 + i_3 < N) {
                                    float value_1 = vals[i_3];
                                    *(reinterpret_cast<__nv_bfloat16*>(C + (row_linear + (long long)(col_global_1 + i_3))) + (0)) = (__nv_bfloat16)value_1;
                                }
                            }
                        }
                    }
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_epi_stage = 0;
            int num_iters = K / 64;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_stage_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < grid_m * grid_n; tile_idx_1 += num_clusters) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                        mbarrier_wait(stage_full_addr + (mma_stage) * 8, _phase_stage_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1024;
                        int _mma_b_lo_0 = (((smem_b_cvt_addr) >> 4) & 0x3FFF) + (mma_stage) * 384;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270009488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 96))), "r"(((init_flag) ? 0 : 1)) : "memory");
                        elect_commit_cg2_multicast(stage_done_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 8) { mma_stage = 0; _phase_stage_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            mbarrier_wait(tmem_dealloc_bar_addr, 0);
            if (cta_rank == 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            unsigned int load_a_stage = 0;
            int num_iters_1 = K / 64;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_stage_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < grid_m * grid_n; tile_idx_2 += num_clusters) {
                    int m_tile_1 = tile_idx_2 / (unsigned int)grid_n;
                    int off_m_1 = m_tile_1 * 256;
                    off_m_1 = off_m_1 + cta_rank * 128;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < num_iters_1; iter_k_1++) {
                        mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                        int a_k = iter_k_1 * 64;
                        tma_2d_gmem2smem_cta2(smem_a_addr + load_a_stage * 16384, (&A), a_k, off_m_1, ((stage_full_addr + (load_a_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((stage_full_addr + (load_a_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        load_a_stage += 1;
                        if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                    }
                }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
                mbarrier_wait(stage_done_addr + (load_a_stage) * 8, _phase_stage_done);
                load_a_stage += 1;
                if (load_a_stage == 8) { load_a_stage = 0; _phase_stage_done ^= 1; }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 6) {
        { // load_b_main
            unsigned int load_b_stage = 0;
            int num_groups = K / 256;
            unsigned int _phase_bsrc_done = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < grid_m * grid_n; tile_idx_3 += num_clusters) {
                    int n_tile_1 = tile_idx_3 % (unsigned int)grid_n;
                    int b_row = n_tile_1 * 96;
                    b_row = b_row + cta_rank * 48;
                    #pragma unroll 1
                    for (int group = 0; group < num_groups; group++) {
                        #pragma unroll 1
                        for (int sub = 0; sub < 4; sub += 2) {
                            mbarrier_wait(bsrc_done_addr + (load_b_stage) * 8, _phase_bsrc_done);
                            tma_2d_gmem2smem(smem_bpack_addr + load_b_stage * 1536, (&B), (group * 4 + sub) * 32, b_row, bsrc_full_addr + (load_b_stage) * 8);
                            mbarrier_arrive_expect_tx(bsrc_full_addr + (load_b_stage) * 8, 1536);
                            if (load_b_stage + 2 >= 14) { _phase_bsrc_done ^= 1; }
                            load_b_stage = (load_b_stage + 2) % 14;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: load_b2 ----
    if (warp == 7) {
        { // load_b2_main
            unsigned int load_b_stage_1 = 1;
            unsigned int sf_stage = 0;
            int num_groups_1 = K / 256;
            unsigned int _phase_sf_done = 1;
            unsigned int _phase_bsrc_done_1 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < grid_m * grid_n; tile_idx_4 += num_clusters) {
                    int n_tile_2 = tile_idx_4 % (unsigned int)grid_n;
                    int b_row_1 = n_tile_2 * 96;
                    b_row_1 = b_row_1 + cta_rank * 48;
                    #pragma unroll 1
                    for (int group_1 = 0; group_1 < num_groups_1; group_1++) {
                        mbarrier_wait(sf_done_addr + (sf_stage) * 8, _phase_sf_done);
                        tma_2d_gmem2smem(smem_sf_addr + sf_stage * 768, (&B_descale), group_1 * 16, b_row_1, sf_full_addr + (sf_stage) * 8);
                        mbarrier_arrive_expect_tx(sf_full_addr + (sf_stage) * 8, 768);
                        sf_stage += 1;
                        if (sf_stage == 4) { sf_stage = 0; _phase_sf_done ^= 1; }
                        #pragma unroll 1
                        for (int sub_1 = 1; sub_1 < 4; sub_1 += 2) {
                            mbarrier_wait(bsrc_done_addr + (load_b_stage_1) * 8, _phase_bsrc_done_1);
                            tma_2d_gmem2smem(smem_bpack_addr + load_b_stage_1 * 1536, (&B), (group_1 * 4 + sub_1) * 32, b_row_1, bsrc_full_addr + (load_b_stage_1) * 8);
                            mbarrier_arrive_expect_tx(bsrc_full_addr + (load_b_stage_1) * 8, 1536);
                            if (load_b_stage_1 + 2 >= 14) { _phase_bsrc_done_1 ^= 1; }
                            load_b_stage_1 = (load_b_stage_1 + 2) % 14;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 15) {
        { // convert_main
            int warp_id_in_role = warp - 8;
            int group_2 = warp_id_in_role / 2;
            unsigned int cvt_stage = (unsigned int)group_2;
            unsigned int cvt_pack_stage = (unsigned int)group_2;
            unsigned int cvt_sf_stage = (unsigned int)(group_2 / 4);
            int num_iters_2 = K / 64;
            int phase4 = 0;
            int convert_tid = warp_id_in_role % 2 * 32 + lane;
            int row0 = convert_tid / 4;
            int k_block = convert_tid - row0 * 4;
            int swz = row0 & 7;
            int chunk_lo = k_block * 2 ^ swz;
            int chunk_hi = k_block * 2 + 1 ^ swz;
            unsigned int raw_words[2];
            unsigned int scale_word[1];
            unsigned int _phase_stage_done_1 = 1;
            unsigned int _phase_bsrc_full = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < grid_m * grid_n; tile_idx_5 += num_clusters) {
                int first = group_2 - phase4 & 3;
                #pragma unroll 1
                for (int local = first; local < num_iters_2; local += 4) {
                    int sub_2 = local & 3;
                    mbarrier_wait(stage_done_addr + (cvt_stage) * 8, _phase_stage_done_1);
                    mbarrier_wait(bsrc_full_addr + (cvt_pack_stage) * 8, _phase_bsrc_full);
                    mbarrier_wait(sf_full_addr + (cvt_sf_stage) * 8, _phase_sf_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int sf_base = smem_sf_addr + cvt_sf_stage * 768;
                    int packed_base = smem_bpack_addr + cvt_pack_stage * 1536;
                    int bcvt_base = smem_b_cvt_addr + cvt_stage * 6144;
                    #pragma unroll
                    for (int half = 0; half < 3; half++) {
                        int row = row0 + half * 16;
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_words[(0) + 1]))
                            : "r"(packed_base + row * 32 + k_block * 8));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_word[0])) : "r"(sf_base + row * 16 + sub_2 * 4));
                        uint8_t scale_byte = (uint8_t)(scale_word[0] >> (unsigned int)(k_block * 8) & 255);
                        uint32_t _fp4_dequant_block16_bf16_0[8];
                        {
                            uint32_t _scale_byte = ((uint32_t)(scale_byte)) & 0xFFu;
                            cvt_bf16x8_e2m1x16_e4m3((uint32_t)(raw_words[0]), (uint32_t)(raw_words[1]), _scale_byte, _fp4_dequant_block16_bf16_0);
                        }
                        int row_base = bcvt_base + row * 128;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (chunk_lo << 4)), "r"(_fp4_dequant_block16_bf16_0[0]), "r"(_fp4_dequant_block16_bf16_0[1]), "r"(_fp4_dequant_block16_bf16_0[2]), "r"(_fp4_dequant_block16_bf16_0[3]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + (chunk_hi << 4)), "r"(_fp4_dequant_block16_bf16_0[4]), "r"(_fp4_dequant_block16_bf16_0[5]), "r"(_fp4_dequant_block16_bf16_0[6]), "r"(_fp4_dequant_block16_bf16_0[7]));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(bsrc_done_addr + (cvt_pack_stage) * 8);
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((stage_full_addr + (cvt_stage) * 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(sf_done_addr + (cvt_sf_stage) * 8);
                    }
                    if (cvt_stage + 4 >= 8) { _phase_stage_done_1 ^= 1; }
                    cvt_stage = (cvt_stage + 4) % 8;
                    if (cvt_pack_stage + 4 >= 14) { _phase_bsrc_full ^= 1; }
                    cvt_pack_stage = (cvt_pack_stage + 4) % 14;
                    cvt_sf_stage += 1;
                    if (cvt_sf_stage == 4) { cvt_sf_stage = 0; _phase_sf_full ^= 1; }
                }
                phase4 = phase4 + num_iters_2 & 3;
            }
            mbarrier_wait(stage_done_addr + (cvt_stage) * 8, _phase_stage_done_1);
            if (cvt_stage + 4 >= 8) { _phase_stage_done_1 ^= 1; }
            cvt_stage = (cvt_stage + 4) % 8;
            mbarrier_wait(stage_done_addr + (cvt_stage) * 8, _phase_stage_done_1);
            if (cvt_stage + 4 >= 8) { _phase_stage_done_1 ^= 1; }
            cvt_stage = (cvt_stage + 4) % 8;
        }
    }

    // Cleanup
}

} // extern "C"
