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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 1024
#define SMEM_LAND_STRIDE 1024
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 3072
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 117760
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_BN_OFF 117760
#define SMEM_SMEM_BN_STAGE_BYTES 16384
#define SMEM_SMEM_BN_STRIDE 16384
#define SMEM_SMEM_AG_OFF 3072
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 11264
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 232448

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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(192) void
kernel_cake_kimi_k3_latent_moe_a5b95a7689e0a7181ffd(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 56)
    #define acc_full_addr (mbar_base + 112)
    #define acc_empty_addr (mbar_base + 120)
    #define red_bar_addr (mbar_base + 128)
    #define issue_bar_addr (mbar_base + 136)
    #define rows_bar_addr (mbar_base + 144)
    #define bn_bar_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_a_addr = smem + 3072;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 117760);
    const int smem_b_addr = smem + 117760;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 117760);
    const int smem_bn_addr = smem + 117760;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_ag_addr = smem + 3072;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_au_addr = smem + 11264;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // a_empty: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int g_e = bid;
            int epi_warp = warp % 4;
            int tid_1 = epi_warp * 32 + lane;
            int tl_e = g_e * 16;
            int tile = g_e;
            int slot = 0;
            int cls = ((tile < 7) ? 0 : ((tile < 35) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = (tile - 7) * 128;
            int row_s = (tile - 7 - 28) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float red[128];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                    : "r"(lane_addr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                    : "r"(lane_addr + 256));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[64] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[65] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[66] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[67] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[68] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[69] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[70] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[71] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[72] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[73] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[74] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[75] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[76] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[77] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[78] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[79] = _tmem_load_1[15];
                red[16] = _tmem_load_0[16];
                red[80] = _tmem_load_1[16];
                red[17] = _tmem_load_0[17];
                red[81] = _tmem_load_1[17];
                red[18] = _tmem_load_0[18];
                red[82] = _tmem_load_1[18];
                red[19] = _tmem_load_0[19];
                red[83] = _tmem_load_1[19];
                red[20] = _tmem_load_0[20];
                red[84] = _tmem_load_1[20];
                red[21] = _tmem_load_0[21];
                red[85] = _tmem_load_1[21];
                red[22] = _tmem_load_0[22];
                red[86] = _tmem_load_1[22];
                red[23] = _tmem_load_0[23];
                red[87] = _tmem_load_1[23];
                red[24] = _tmem_load_0[24];
                red[88] = _tmem_load_1[24];
                red[25] = _tmem_load_0[25];
                red[89] = _tmem_load_1[25];
                red[26] = _tmem_load_0[26];
                red[90] = _tmem_load_1[26];
                red[27] = _tmem_load_0[27];
                red[91] = _tmem_load_1[27];
                red[28] = _tmem_load_0[28];
                red[92] = _tmem_load_1[28];
                red[29] = _tmem_load_0[29];
                red[93] = _tmem_load_1[29];
                red[30] = _tmem_load_0[30];
                red[94] = _tmem_load_1[30];
                red[31] = _tmem_load_0[31];
                red[95] = _tmem_load_1[31];
                red[32] = _tmem_load_0[32];
                red[96] = _tmem_load_1[32];
                red[33] = _tmem_load_0[33];
                red[97] = _tmem_load_1[33];
                red[34] = _tmem_load_0[34];
                red[98] = _tmem_load_1[34];
                red[35] = _tmem_load_0[35];
                red[99] = _tmem_load_1[35];
                red[36] = _tmem_load_0[36];
                red[100] = _tmem_load_1[36];
                red[37] = _tmem_load_0[37];
                red[101] = _tmem_load_1[37];
                red[38] = _tmem_load_0[38];
                red[102] = _tmem_load_1[38];
                red[39] = _tmem_load_0[39];
                red[103] = _tmem_load_1[39];
                red[40] = _tmem_load_0[40];
                red[104] = _tmem_load_1[40];
                red[41] = _tmem_load_0[41];
                red[105] = _tmem_load_1[41];
                red[42] = _tmem_load_0[42];
                red[106] = _tmem_load_1[42];
                red[43] = _tmem_load_0[43];
                red[107] = _tmem_load_1[43];
                red[44] = _tmem_load_0[44];
                red[108] = _tmem_load_1[44];
                red[45] = _tmem_load_0[45];
                red[109] = _tmem_load_1[45];
                red[46] = _tmem_load_0[46];
                red[110] = _tmem_load_1[46];
                red[47] = _tmem_load_0[47];
                red[111] = _tmem_load_1[47];
                red[48] = _tmem_load_0[48];
                red[112] = _tmem_load_1[48];
                red[49] = _tmem_load_0[49];
                red[113] = _tmem_load_1[49];
                red[50] = _tmem_load_0[50];
                red[114] = _tmem_load_1[50];
                red[51] = _tmem_load_0[51];
                red[115] = _tmem_load_1[51];
                red[52] = _tmem_load_0[52];
                red[116] = _tmem_load_1[52];
                red[53] = _tmem_load_0[53];
                red[117] = _tmem_load_1[53];
                red[54] = _tmem_load_0[54];
                red[118] = _tmem_load_1[54];
                red[55] = _tmem_load_0[55];
                red[119] = _tmem_load_1[55];
                red[56] = _tmem_load_0[56];
                red[120] = _tmem_load_1[56];
                red[57] = _tmem_load_0[57];
                red[121] = _tmem_load_1[57];
                red[58] = _tmem_load_0[58];
                red[122] = _tmem_load_1[58];
                red[59] = _tmem_load_0[59];
                red[123] = _tmem_load_1[59];
                red[60] = _tmem_load_0[60];
                red[124] = _tmem_load_1[60];
                red[61] = _tmem_load_0[61];
                red[125] = _tmem_load_1[61];
                red[62] = _tmem_load_0[62];
                red[126] = _tmem_load_1[62];
                red[63] = _tmem_load_0[63];
                red[127] = _tmem_load_1[63];
            } else {
                float _tmem_load_2[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[64]), "=f"(_tmem_load_2[65]), "=f"(_tmem_load_2[66]), "=f"(_tmem_load_2[67]), "=f"(_tmem_load_2[68]), "=f"(_tmem_load_2[69]), "=f"(_tmem_load_2[70]), "=f"(_tmem_load_2[71]), "=f"(_tmem_load_2[72]), "=f"(_tmem_load_2[73]), "=f"(_tmem_load_2[74]), "=f"(_tmem_load_2[75]), "=f"(_tmem_load_2[76]), "=f"(_tmem_load_2[77]), "=f"(_tmem_load_2[78]), "=f"(_tmem_load_2[79]), "=f"(_tmem_load_2[80]), "=f"(_tmem_load_2[81]), "=f"(_tmem_load_2[82]), "=f"(_tmem_load_2[83]), "=f"(_tmem_load_2[84]), "=f"(_tmem_load_2[85]), "=f"(_tmem_load_2[86]), "=f"(_tmem_load_2[87]), "=f"(_tmem_load_2[88]), "=f"(_tmem_load_2[89]), "=f"(_tmem_load_2[90]), "=f"(_tmem_load_2[91]), "=f"(_tmem_load_2[92]), "=f"(_tmem_load_2[93]), "=f"(_tmem_load_2[94]), "=f"(_tmem_load_2[95])
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[96]), "=f"(_tmem_load_2[97]), "=f"(_tmem_load_2[98]), "=f"(_tmem_load_2[99]), "=f"(_tmem_load_2[100]), "=f"(_tmem_load_2[101]), "=f"(_tmem_load_2[102]), "=f"(_tmem_load_2[103]), "=f"(_tmem_load_2[104]), "=f"(_tmem_load_2[105]), "=f"(_tmem_load_2[106]), "=f"(_tmem_load_2[107]), "=f"(_tmem_load_2[108]), "=f"(_tmem_load_2[109]), "=f"(_tmem_load_2[110]), "=f"(_tmem_load_2[111]), "=f"(_tmem_load_2[112]), "=f"(_tmem_load_2[113]), "=f"(_tmem_load_2[114]), "=f"(_tmem_load_2[115]), "=f"(_tmem_load_2[116]), "=f"(_tmem_load_2[117]), "=f"(_tmem_load_2[118]), "=f"(_tmem_load_2[119]), "=f"(_tmem_load_2[120]), "=f"(_tmem_load_2[121]), "=f"(_tmem_load_2[122]), "=f"(_tmem_load_2[123]), "=f"(_tmem_load_2[124]), "=f"(_tmem_load_2[125]), "=f"(_tmem_load_2[126]), "=f"(_tmem_load_2[127])
                    : "r"(lane_addr + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_2[0];
                red[1] = _tmem_load_2[1];
                red[2] = _tmem_load_2[2];
                red[3] = _tmem_load_2[3];
                red[4] = _tmem_load_2[4];
                red[5] = _tmem_load_2[5];
                red[6] = _tmem_load_2[6];
                red[7] = _tmem_load_2[7];
                red[8] = _tmem_load_2[8];
                red[9] = _tmem_load_2[9];
                red[10] = _tmem_load_2[10];
                red[11] = _tmem_load_2[11];
                red[12] = _tmem_load_2[12];
                red[13] = _tmem_load_2[13];
                red[14] = _tmem_load_2[14];
                red[15] = _tmem_load_2[15];
                red[16] = _tmem_load_2[16];
                red[17] = _tmem_load_2[17];
                red[18] = _tmem_load_2[18];
                red[19] = _tmem_load_2[19];
                red[20] = _tmem_load_2[20];
                red[21] = _tmem_load_2[21];
                red[22] = _tmem_load_2[22];
                red[23] = _tmem_load_2[23];
                red[24] = _tmem_load_2[24];
                red[25] = _tmem_load_2[25];
                red[26] = _tmem_load_2[26];
                red[27] = _tmem_load_2[27];
                red[28] = _tmem_load_2[28];
                red[29] = _tmem_load_2[29];
                red[30] = _tmem_load_2[30];
                red[31] = _tmem_load_2[31];
                red[32] = _tmem_load_2[32];
                red[33] = _tmem_load_2[33];
                red[34] = _tmem_load_2[34];
                red[35] = _tmem_load_2[35];
                red[36] = _tmem_load_2[36];
                red[37] = _tmem_load_2[37];
                red[38] = _tmem_load_2[38];
                red[39] = _tmem_load_2[39];
                red[40] = _tmem_load_2[40];
                red[41] = _tmem_load_2[41];
                red[42] = _tmem_load_2[42];
                red[43] = _tmem_load_2[43];
                red[44] = _tmem_load_2[44];
                red[45] = _tmem_load_2[45];
                red[46] = _tmem_load_2[46];
                red[47] = _tmem_load_2[47];
                red[48] = _tmem_load_2[48];
                red[49] = _tmem_load_2[49];
                red[50] = _tmem_load_2[50];
                red[51] = _tmem_load_2[51];
                red[52] = _tmem_load_2[52];
                red[53] = _tmem_load_2[53];
                red[54] = _tmem_load_2[54];
                red[55] = _tmem_load_2[55];
                red[56] = _tmem_load_2[56];
                red[57] = _tmem_load_2[57];
                red[58] = _tmem_load_2[58];
                red[59] = _tmem_load_2[59];
                red[60] = _tmem_load_2[60];
                red[61] = _tmem_load_2[61];
                red[62] = _tmem_load_2[62];
                red[63] = _tmem_load_2[63];
                red[64] = _tmem_load_2[64];
                red[65] = _tmem_load_2[65];
                red[66] = _tmem_load_2[66];
                red[67] = _tmem_load_2[67];
                red[68] = _tmem_load_2[68];
                red[69] = _tmem_load_2[69];
                red[70] = _tmem_load_2[70];
                red[71] = _tmem_load_2[71];
                red[72] = _tmem_load_2[72];
                red[73] = _tmem_load_2[73];
                red[74] = _tmem_load_2[74];
                red[75] = _tmem_load_2[75];
                red[76] = _tmem_load_2[76];
                red[77] = _tmem_load_2[77];
                red[78] = _tmem_load_2[78];
                red[79] = _tmem_load_2[79];
                red[80] = _tmem_load_2[80];
                red[81] = _tmem_load_2[81];
                red[82] = _tmem_load_2[82];
                red[83] = _tmem_load_2[83];
                red[84] = _tmem_load_2[84];
                red[85] = _tmem_load_2[85];
                red[86] = _tmem_load_2[86];
                red[87] = _tmem_load_2[87];
                red[88] = _tmem_load_2[88];
                red[89] = _tmem_load_2[89];
                red[90] = _tmem_load_2[90];
                red[91] = _tmem_load_2[91];
                red[92] = _tmem_load_2[92];
                red[93] = _tmem_load_2[93];
                red[94] = _tmem_load_2[94];
                red[95] = _tmem_load_2[95];
                red[96] = _tmem_load_2[96];
                red[97] = _tmem_load_2[97];
                red[98] = _tmem_load_2[98];
                red[99] = _tmem_load_2[99];
                red[100] = _tmem_load_2[100];
                red[101] = _tmem_load_2[101];
                red[102] = _tmem_load_2[102];
                red[103] = _tmem_load_2[103];
                red[104] = _tmem_load_2[104];
                red[105] = _tmem_load_2[105];
                red[106] = _tmem_load_2[106];
                red[107] = _tmem_load_2[107];
                red[108] = _tmem_load_2[108];
                red[109] = _tmem_load_2[109];
                red[110] = _tmem_load_2[110];
                red[111] = _tmem_load_2[111];
                red[112] = _tmem_load_2[112];
                red[113] = _tmem_load_2[113];
                red[114] = _tmem_load_2[114];
                red[115] = _tmem_load_2[115];
                red[116] = _tmem_load_2[116];
                red[117] = _tmem_load_2[117];
                red[118] = _tmem_load_2[118];
                red[119] = _tmem_load_2[119];
                red[120] = _tmem_load_2[120];
                red[121] = _tmem_load_2[121];
                red[122] = _tmem_load_2[122];
                red[123] = _tmem_load_2[123];
                red[124] = _tmem_load_2[124];
                red[125] = _tmem_load_2[125];
                red[126] = _tmem_load_2[126];
                red[127] = _tmem_load_2[127];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr);
                }
            }
            if (fin == 1) {
                if (cls == 2) {
                    int tok_1_0 = lane_pair * 2;
                    int frow_2_0 = row_base;
                    if (tok_1_0 < num_tokens) {
                        float gk_1_0 = (float)(__nv_bfloat16)red[0];
                        float uk_1_0 = (float)(__nv_bfloat16)red[64];
                        float _exp2_0 = approx_exp2((-gk_1_0) * 1.442695f);
                        float sig_1_0 = approx_rcp(1.0f + _exp2_0);
                        float _tanh_1 = tanhf(gk_1_0 * 0.25f);
                        float aa_1_0 = 4.0f * _tanh_1 * sig_1_0;
                        float _tanh_2 = tanhf(uk_1_0 * 0.04f);
                        float bb_1_0 = 25.0f * _tanh_2;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_0 * 6144 + row_s + frow_2_0)) + (0)) = (__nv_bfloat16)(aa_1_0 * bb_1_0);
                    }
                    int tok_1_1 = lane_pair * 2 + 1;
                    int frow_2_1 = row_base;
                    if (tok_1_1 < num_tokens) {
                        float gk_1_1 = (float)(__nv_bfloat16)red[1];
                        float uk_1_1 = (float)(__nv_bfloat16)red[65];
                        float _exp2_3 = approx_exp2((-gk_1_1) * 1.442695f);
                        float sig_1_1 = approx_rcp(1.0f + _exp2_3);
                        float _tanh_4 = tanhf(gk_1_1 * 0.25f);
                        float aa_1_1 = 4.0f * _tanh_4 * sig_1_1;
                        float _tanh_5 = tanhf(uk_1_1 * 0.04f);
                        float bb_1_1 = 25.0f * _tanh_5;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_1 * 6144 + row_s + frow_2_1)) + (0)) = (__nv_bfloat16)(aa_1_1 * bb_1_1);
                    }
                    int tok_1_2 = lane_pair * 2;
                    int frow_2_2 = row_base + 8;
                    if (tok_1_2 < num_tokens) {
                        float gk_1_2 = (float)(__nv_bfloat16)red[2];
                        float uk_1_2 = (float)(__nv_bfloat16)red[66];
                        float _exp2_6 = approx_exp2((-gk_1_2) * 1.442695f);
                        float sig_1_2 = approx_rcp(1.0f + _exp2_6);
                        float _tanh_7 = tanhf(gk_1_2 * 0.25f);
                        float aa_1_2 = 4.0f * _tanh_7 * sig_1_2;
                        float _tanh_8 = tanhf(uk_1_2 * 0.04f);
                        float bb_1_2 = 25.0f * _tanh_8;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_2 * 6144 + row_s + frow_2_2)) + (0)) = (__nv_bfloat16)(aa_1_2 * bb_1_2);
                    }
                    int tok_1_3 = lane_pair * 2 + 1;
                    int frow_2_3 = row_base + 8;
                    if (tok_1_3 < num_tokens) {
                        float gk_1_3 = (float)(__nv_bfloat16)red[3];
                        float uk_1_3 = (float)(__nv_bfloat16)red[67];
                        float _exp2_9 = approx_exp2((-gk_1_3) * 1.442695f);
                        float sig_1_3 = approx_rcp(1.0f + _exp2_9);
                        float _tanh_10 = tanhf(gk_1_3 * 0.25f);
                        float aa_1_3 = 4.0f * _tanh_10 * sig_1_3;
                        float _tanh_11 = tanhf(uk_1_3 * 0.04f);
                        float bb_1_3 = 25.0f * _tanh_11;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_3 * 6144 + row_s + frow_2_3)) + (0)) = (__nv_bfloat16)(aa_1_3 * bb_1_3);
                    }
                    int tok_1_4 = 8 + lane_pair * 2;
                    int frow_2_4 = row_base;
                    if (tok_1_4 < num_tokens) {
                        float gk_1_4 = (float)(__nv_bfloat16)red[4];
                        float uk_1_4 = (float)(__nv_bfloat16)red[68];
                        float _exp2_12 = approx_exp2((-gk_1_4) * 1.442695f);
                        float sig_1_4 = approx_rcp(1.0f + _exp2_12);
                        float _tanh_13 = tanhf(gk_1_4 * 0.25f);
                        float aa_1_4 = 4.0f * _tanh_13 * sig_1_4;
                        float _tanh_14 = tanhf(uk_1_4 * 0.04f);
                        float bb_1_4 = 25.0f * _tanh_14;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_4 * 6144 + row_s + frow_2_4)) + (0)) = (__nv_bfloat16)(aa_1_4 * bb_1_4);
                    }
                    int tok_1_5 = 8 + lane_pair * 2 + 1;
                    int frow_2_5 = row_base;
                    if (tok_1_5 < num_tokens) {
                        float gk_1_5 = (float)(__nv_bfloat16)red[5];
                        float uk_1_5 = (float)(__nv_bfloat16)red[69];
                        float _exp2_15 = approx_exp2((-gk_1_5) * 1.442695f);
                        float sig_1_5 = approx_rcp(1.0f + _exp2_15);
                        float _tanh_16 = tanhf(gk_1_5 * 0.25f);
                        float aa_1_5 = 4.0f * _tanh_16 * sig_1_5;
                        float _tanh_17 = tanhf(uk_1_5 * 0.04f);
                        float bb_1_5 = 25.0f * _tanh_17;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_5 * 6144 + row_s + frow_2_5)) + (0)) = (__nv_bfloat16)(aa_1_5 * bb_1_5);
                    }
                    int tok_1_6 = 8 + lane_pair * 2;
                    int frow_2_6 = row_base + 8;
                    if (tok_1_6 < num_tokens) {
                        float gk_1_6 = (float)(__nv_bfloat16)red[6];
                        float uk_1_6 = (float)(__nv_bfloat16)red[70];
                        float _exp2_18 = approx_exp2((-gk_1_6) * 1.442695f);
                        float sig_1_6 = approx_rcp(1.0f + _exp2_18);
                        float _tanh_19 = tanhf(gk_1_6 * 0.25f);
                        float aa_1_6 = 4.0f * _tanh_19 * sig_1_6;
                        float _tanh_20 = tanhf(uk_1_6 * 0.04f);
                        float bb_1_6 = 25.0f * _tanh_20;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_6 * 6144 + row_s + frow_2_6)) + (0)) = (__nv_bfloat16)(aa_1_6 * bb_1_6);
                    }
                    int tok_1_7 = 8 + lane_pair * 2 + 1;
                    int frow_2_7 = row_base + 8;
                    if (tok_1_7 < num_tokens) {
                        float gk_1_7 = (float)(__nv_bfloat16)red[7];
                        float uk_1_7 = (float)(__nv_bfloat16)red[71];
                        float _exp2_21 = approx_exp2((-gk_1_7) * 1.442695f);
                        float sig_1_7 = approx_rcp(1.0f + _exp2_21);
                        float _tanh_22 = tanhf(gk_1_7 * 0.25f);
                        float aa_1_7 = 4.0f * _tanh_22 * sig_1_7;
                        float _tanh_23 = tanhf(uk_1_7 * 0.04f);
                        float bb_1_7 = 25.0f * _tanh_23;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_7 * 6144 + row_s + frow_2_7)) + (0)) = (__nv_bfloat16)(aa_1_7 * bb_1_7);
                    }
                    int tok_1_8 = 16 + lane_pair * 2;
                    int frow_2_8 = row_base;
                    if (tok_1_8 < num_tokens) {
                        float gk_1_8 = (float)(__nv_bfloat16)red[8];
                        float uk_1_8 = (float)(__nv_bfloat16)red[72];
                        float _exp2_24 = approx_exp2((-gk_1_8) * 1.442695f);
                        float sig_1_8 = approx_rcp(1.0f + _exp2_24);
                        float _tanh_25 = tanhf(gk_1_8 * 0.25f);
                        float aa_1_8 = 4.0f * _tanh_25 * sig_1_8;
                        float _tanh_26 = tanhf(uk_1_8 * 0.04f);
                        float bb_1_8 = 25.0f * _tanh_26;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_8 * 6144 + row_s + frow_2_8)) + (0)) = (__nv_bfloat16)(aa_1_8 * bb_1_8);
                    }
                    int tok_1_9 = 16 + lane_pair * 2 + 1;
                    int frow_2_9 = row_base;
                    if (tok_1_9 < num_tokens) {
                        float gk_1_9 = (float)(__nv_bfloat16)red[9];
                        float uk_1_9 = (float)(__nv_bfloat16)red[73];
                        float _exp2_27 = approx_exp2((-gk_1_9) * 1.442695f);
                        float sig_1_9 = approx_rcp(1.0f + _exp2_27);
                        float _tanh_28 = tanhf(gk_1_9 * 0.25f);
                        float aa_1_9 = 4.0f * _tanh_28 * sig_1_9;
                        float _tanh_29 = tanhf(uk_1_9 * 0.04f);
                        float bb_1_9 = 25.0f * _tanh_29;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_9 * 6144 + row_s + frow_2_9)) + (0)) = (__nv_bfloat16)(aa_1_9 * bb_1_9);
                    }
                    int tok_1_10 = 16 + lane_pair * 2;
                    int frow_2_10 = row_base + 8;
                    if (tok_1_10 < num_tokens) {
                        float gk_1_10 = (float)(__nv_bfloat16)red[10];
                        float uk_1_10 = (float)(__nv_bfloat16)red[74];
                        float _exp2_30 = approx_exp2((-gk_1_10) * 1.442695f);
                        float sig_1_10 = approx_rcp(1.0f + _exp2_30);
                        float _tanh_31 = tanhf(gk_1_10 * 0.25f);
                        float aa_1_10 = 4.0f * _tanh_31 * sig_1_10;
                        float _tanh_32 = tanhf(uk_1_10 * 0.04f);
                        float bb_1_10 = 25.0f * _tanh_32;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_10 * 6144 + row_s + frow_2_10)) + (0)) = (__nv_bfloat16)(aa_1_10 * bb_1_10);
                    }
                    int tok_1_11 = 16 + lane_pair * 2 + 1;
                    int frow_2_11 = row_base + 8;
                    if (tok_1_11 < num_tokens) {
                        float gk_1_11 = (float)(__nv_bfloat16)red[11];
                        float uk_1_11 = (float)(__nv_bfloat16)red[75];
                        float _exp2_33 = approx_exp2((-gk_1_11) * 1.442695f);
                        float sig_1_11 = approx_rcp(1.0f + _exp2_33);
                        float _tanh_34 = tanhf(gk_1_11 * 0.25f);
                        float aa_1_11 = 4.0f * _tanh_34 * sig_1_11;
                        float _tanh_35 = tanhf(uk_1_11 * 0.04f);
                        float bb_1_11 = 25.0f * _tanh_35;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_11 * 6144 + row_s + frow_2_11)) + (0)) = (__nv_bfloat16)(aa_1_11 * bb_1_11);
                    }
                    int tok_1_12 = 24 + lane_pair * 2;
                    int frow_2_12 = row_base;
                    if (tok_1_12 < num_tokens) {
                        float gk_1_12 = (float)(__nv_bfloat16)red[12];
                        float uk_1_12 = (float)(__nv_bfloat16)red[76];
                        float _exp2_36 = approx_exp2((-gk_1_12) * 1.442695f);
                        float sig_1_12 = approx_rcp(1.0f + _exp2_36);
                        float _tanh_37 = tanhf(gk_1_12 * 0.25f);
                        float aa_1_12 = 4.0f * _tanh_37 * sig_1_12;
                        float _tanh_38 = tanhf(uk_1_12 * 0.04f);
                        float bb_1_12 = 25.0f * _tanh_38;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_12 * 6144 + row_s + frow_2_12)) + (0)) = (__nv_bfloat16)(aa_1_12 * bb_1_12);
                    }
                    int tok_1_13 = 24 + lane_pair * 2 + 1;
                    int frow_2_13 = row_base;
                    if (tok_1_13 < num_tokens) {
                        float gk_1_13 = (float)(__nv_bfloat16)red[13];
                        float uk_1_13 = (float)(__nv_bfloat16)red[77];
                        float _exp2_39 = approx_exp2((-gk_1_13) * 1.442695f);
                        float sig_1_13 = approx_rcp(1.0f + _exp2_39);
                        float _tanh_40 = tanhf(gk_1_13 * 0.25f);
                        float aa_1_13 = 4.0f * _tanh_40 * sig_1_13;
                        float _tanh_41 = tanhf(uk_1_13 * 0.04f);
                        float bb_1_13 = 25.0f * _tanh_41;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_13 * 6144 + row_s + frow_2_13)) + (0)) = (__nv_bfloat16)(aa_1_13 * bb_1_13);
                    }
                    int tok_1_14 = 24 + lane_pair * 2;
                    int frow_2_14 = row_base + 8;
                    if (tok_1_14 < num_tokens) {
                        float gk_1_14 = (float)(__nv_bfloat16)red[14];
                        float uk_1_14 = (float)(__nv_bfloat16)red[78];
                        float _exp2_42 = approx_exp2((-gk_1_14) * 1.442695f);
                        float sig_1_14 = approx_rcp(1.0f + _exp2_42);
                        float _tanh_43 = tanhf(gk_1_14 * 0.25f);
                        float aa_1_14 = 4.0f * _tanh_43 * sig_1_14;
                        float _tanh_44 = tanhf(uk_1_14 * 0.04f);
                        float bb_1_14 = 25.0f * _tanh_44;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_14 * 6144 + row_s + frow_2_14)) + (0)) = (__nv_bfloat16)(aa_1_14 * bb_1_14);
                    }
                    int tok_1_15 = 24 + lane_pair * 2 + 1;
                    int frow_2_15 = row_base + 8;
                    if (tok_1_15 < num_tokens) {
                        float gk_1_15 = (float)(__nv_bfloat16)red[15];
                        float uk_1_15 = (float)(__nv_bfloat16)red[79];
                        float _exp2_45 = approx_exp2((-gk_1_15) * 1.442695f);
                        float sig_1_15 = approx_rcp(1.0f + _exp2_45);
                        float _tanh_46 = tanhf(gk_1_15 * 0.25f);
                        float aa_1_15 = 4.0f * _tanh_46 * sig_1_15;
                        float _tanh_47 = tanhf(uk_1_15 * 0.04f);
                        float bb_1_15 = 25.0f * _tanh_47;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_15 * 6144 + row_s + frow_2_15)) + (0)) = (__nv_bfloat16)(aa_1_15 * bb_1_15);
                    }
                    int tok_1_16 = 32 + lane_pair * 2;
                    int frow_2_16 = row_base;
                    if (tok_1_16 < num_tokens) {
                        float gk_1_16 = (float)(__nv_bfloat16)red[16];
                        float uk_1_16 = (float)(__nv_bfloat16)red[80];
                        float _exp2_48 = approx_exp2((-gk_1_16) * 1.442695f);
                        float sig_1_16 = approx_rcp(1.0f + _exp2_48);
                        float _tanh_49 = tanhf(gk_1_16 * 0.25f);
                        float aa_1_16 = 4.0f * _tanh_49 * sig_1_16;
                        float _tanh_50 = tanhf(uk_1_16 * 0.04f);
                        float bb_1_16 = 25.0f * _tanh_50;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_16 * 6144 + row_s + frow_2_16)) + (0)) = (__nv_bfloat16)(aa_1_16 * bb_1_16);
                    }
                    int tok_1_17 = 32 + lane_pair * 2 + 1;
                    int frow_2_17 = row_base;
                    if (tok_1_17 < num_tokens) {
                        float gk_1_17 = (float)(__nv_bfloat16)red[17];
                        float uk_1_17 = (float)(__nv_bfloat16)red[81];
                        float _exp2_51 = approx_exp2((-gk_1_17) * 1.442695f);
                        float sig_1_17 = approx_rcp(1.0f + _exp2_51);
                        float _tanh_52 = tanhf(gk_1_17 * 0.25f);
                        float aa_1_17 = 4.0f * _tanh_52 * sig_1_17;
                        float _tanh_53 = tanhf(uk_1_17 * 0.04f);
                        float bb_1_17 = 25.0f * _tanh_53;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_17 * 6144 + row_s + frow_2_17)) + (0)) = (__nv_bfloat16)(aa_1_17 * bb_1_17);
                    }
                    int tok_1_18 = 32 + lane_pair * 2;
                    int frow_2_18 = row_base + 8;
                    if (tok_1_18 < num_tokens) {
                        float gk_1_18 = (float)(__nv_bfloat16)red[18];
                        float uk_1_18 = (float)(__nv_bfloat16)red[82];
                        float _exp2_54 = approx_exp2((-gk_1_18) * 1.442695f);
                        float sig_1_18 = approx_rcp(1.0f + _exp2_54);
                        float _tanh_55 = tanhf(gk_1_18 * 0.25f);
                        float aa_1_18 = 4.0f * _tanh_55 * sig_1_18;
                        float _tanh_56 = tanhf(uk_1_18 * 0.04f);
                        float bb_1_18 = 25.0f * _tanh_56;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_18 * 6144 + row_s + frow_2_18)) + (0)) = (__nv_bfloat16)(aa_1_18 * bb_1_18);
                    }
                    int tok_1_19 = 32 + lane_pair * 2 + 1;
                    int frow_2_19 = row_base + 8;
                    if (tok_1_19 < num_tokens) {
                        float gk_1_19 = (float)(__nv_bfloat16)red[19];
                        float uk_1_19 = (float)(__nv_bfloat16)red[83];
                        float _exp2_57 = approx_exp2((-gk_1_19) * 1.442695f);
                        float sig_1_19 = approx_rcp(1.0f + _exp2_57);
                        float _tanh_58 = tanhf(gk_1_19 * 0.25f);
                        float aa_1_19 = 4.0f * _tanh_58 * sig_1_19;
                        float _tanh_59 = tanhf(uk_1_19 * 0.04f);
                        float bb_1_19 = 25.0f * _tanh_59;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_19 * 6144 + row_s + frow_2_19)) + (0)) = (__nv_bfloat16)(aa_1_19 * bb_1_19);
                    }
                    int tok_1_20 = 40 + lane_pair * 2;
                    int frow_2_20 = row_base;
                    if (tok_1_20 < num_tokens) {
                        float gk_1_20 = (float)(__nv_bfloat16)red[20];
                        float uk_1_20 = (float)(__nv_bfloat16)red[84];
                        float _exp2_60 = approx_exp2((-gk_1_20) * 1.442695f);
                        float sig_1_20 = approx_rcp(1.0f + _exp2_60);
                        float _tanh_61 = tanhf(gk_1_20 * 0.25f);
                        float aa_1_20 = 4.0f * _tanh_61 * sig_1_20;
                        float _tanh_62 = tanhf(uk_1_20 * 0.04f);
                        float bb_1_20 = 25.0f * _tanh_62;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_20 * 6144 + row_s + frow_2_20)) + (0)) = (__nv_bfloat16)(aa_1_20 * bb_1_20);
                    }
                    int tok_1_21 = 40 + lane_pair * 2 + 1;
                    int frow_2_21 = row_base;
                    if (tok_1_21 < num_tokens) {
                        float gk_1_21 = (float)(__nv_bfloat16)red[21];
                        float uk_1_21 = (float)(__nv_bfloat16)red[85];
                        float _exp2_63 = approx_exp2((-gk_1_21) * 1.442695f);
                        float sig_1_21 = approx_rcp(1.0f + _exp2_63);
                        float _tanh_64 = tanhf(gk_1_21 * 0.25f);
                        float aa_1_21 = 4.0f * _tanh_64 * sig_1_21;
                        float _tanh_65 = tanhf(uk_1_21 * 0.04f);
                        float bb_1_21 = 25.0f * _tanh_65;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_21 * 6144 + row_s + frow_2_21)) + (0)) = (__nv_bfloat16)(aa_1_21 * bb_1_21);
                    }
                    int tok_1_22 = 40 + lane_pair * 2;
                    int frow_2_22 = row_base + 8;
                    if (tok_1_22 < num_tokens) {
                        float gk_1_22 = (float)(__nv_bfloat16)red[22];
                        float uk_1_22 = (float)(__nv_bfloat16)red[86];
                        float _exp2_66 = approx_exp2((-gk_1_22) * 1.442695f);
                        float sig_1_22 = approx_rcp(1.0f + _exp2_66);
                        float _tanh_67 = tanhf(gk_1_22 * 0.25f);
                        float aa_1_22 = 4.0f * _tanh_67 * sig_1_22;
                        float _tanh_68 = tanhf(uk_1_22 * 0.04f);
                        float bb_1_22 = 25.0f * _tanh_68;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_22 * 6144 + row_s + frow_2_22)) + (0)) = (__nv_bfloat16)(aa_1_22 * bb_1_22);
                    }
                    int tok_1_23 = 40 + lane_pair * 2 + 1;
                    int frow_2_23 = row_base + 8;
                    if (tok_1_23 < num_tokens) {
                        float gk_1_23 = (float)(__nv_bfloat16)red[23];
                        float uk_1_23 = (float)(__nv_bfloat16)red[87];
                        float _exp2_69 = approx_exp2((-gk_1_23) * 1.442695f);
                        float sig_1_23 = approx_rcp(1.0f + _exp2_69);
                        float _tanh_70 = tanhf(gk_1_23 * 0.25f);
                        float aa_1_23 = 4.0f * _tanh_70 * sig_1_23;
                        float _tanh_71 = tanhf(uk_1_23 * 0.04f);
                        float bb_1_23 = 25.0f * _tanh_71;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_23 * 6144 + row_s + frow_2_23)) + (0)) = (__nv_bfloat16)(aa_1_23 * bb_1_23);
                    }
                    int tok_1_24 = 48 + lane_pair * 2;
                    int frow_2_24 = row_base;
                    if (tok_1_24 < num_tokens) {
                        float gk_1_24 = (float)(__nv_bfloat16)red[24];
                        float uk_1_24 = (float)(__nv_bfloat16)red[88];
                        float _exp2_72 = approx_exp2((-gk_1_24) * 1.442695f);
                        float sig_1_24 = approx_rcp(1.0f + _exp2_72);
                        float _tanh_73 = tanhf(gk_1_24 * 0.25f);
                        float aa_1_24 = 4.0f * _tanh_73 * sig_1_24;
                        float _tanh_74 = tanhf(uk_1_24 * 0.04f);
                        float bb_1_24 = 25.0f * _tanh_74;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_24 * 6144 + row_s + frow_2_24)) + (0)) = (__nv_bfloat16)(aa_1_24 * bb_1_24);
                    }
                    int tok_1_25 = 48 + lane_pair * 2 + 1;
                    int frow_2_25 = row_base;
                    if (tok_1_25 < num_tokens) {
                        float gk_1_25 = (float)(__nv_bfloat16)red[25];
                        float uk_1_25 = (float)(__nv_bfloat16)red[89];
                        float _exp2_75 = approx_exp2((-gk_1_25) * 1.442695f);
                        float sig_1_25 = approx_rcp(1.0f + _exp2_75);
                        float _tanh_76 = tanhf(gk_1_25 * 0.25f);
                        float aa_1_25 = 4.0f * _tanh_76 * sig_1_25;
                        float _tanh_77 = tanhf(uk_1_25 * 0.04f);
                        float bb_1_25 = 25.0f * _tanh_77;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_25 * 6144 + row_s + frow_2_25)) + (0)) = (__nv_bfloat16)(aa_1_25 * bb_1_25);
                    }
                    int tok_1_26 = 48 + lane_pair * 2;
                    int frow_2_26 = row_base + 8;
                    if (tok_1_26 < num_tokens) {
                        float gk_1_26 = (float)(__nv_bfloat16)red[26];
                        float uk_1_26 = (float)(__nv_bfloat16)red[90];
                        float _exp2_78 = approx_exp2((-gk_1_26) * 1.442695f);
                        float sig_1_26 = approx_rcp(1.0f + _exp2_78);
                        float _tanh_79 = tanhf(gk_1_26 * 0.25f);
                        float aa_1_26 = 4.0f * _tanh_79 * sig_1_26;
                        float _tanh_80 = tanhf(uk_1_26 * 0.04f);
                        float bb_1_26 = 25.0f * _tanh_80;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_26 * 6144 + row_s + frow_2_26)) + (0)) = (__nv_bfloat16)(aa_1_26 * bb_1_26);
                    }
                    int tok_1_27 = 48 + lane_pair * 2 + 1;
                    int frow_2_27 = row_base + 8;
                    if (tok_1_27 < num_tokens) {
                        float gk_1_27 = (float)(__nv_bfloat16)red[27];
                        float uk_1_27 = (float)(__nv_bfloat16)red[91];
                        float _exp2_81 = approx_exp2((-gk_1_27) * 1.442695f);
                        float sig_1_27 = approx_rcp(1.0f + _exp2_81);
                        float _tanh_82 = tanhf(gk_1_27 * 0.25f);
                        float aa_1_27 = 4.0f * _tanh_82 * sig_1_27;
                        float _tanh_83 = tanhf(uk_1_27 * 0.04f);
                        float bb_1_27 = 25.0f * _tanh_83;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_27 * 6144 + row_s + frow_2_27)) + (0)) = (__nv_bfloat16)(aa_1_27 * bb_1_27);
                    }
                    int tok_1_28 = 56 + lane_pair * 2;
                    int frow_2_28 = row_base;
                    if (tok_1_28 < num_tokens) {
                        float gk_1_28 = (float)(__nv_bfloat16)red[28];
                        float uk_1_28 = (float)(__nv_bfloat16)red[92];
                        float _exp2_84 = approx_exp2((-gk_1_28) * 1.442695f);
                        float sig_1_28 = approx_rcp(1.0f + _exp2_84);
                        float _tanh_85 = tanhf(gk_1_28 * 0.25f);
                        float aa_1_28 = 4.0f * _tanh_85 * sig_1_28;
                        float _tanh_86 = tanhf(uk_1_28 * 0.04f);
                        float bb_1_28 = 25.0f * _tanh_86;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_28 * 6144 + row_s + frow_2_28)) + (0)) = (__nv_bfloat16)(aa_1_28 * bb_1_28);
                    }
                    int tok_1_29 = 56 + lane_pair * 2 + 1;
                    int frow_2_29 = row_base;
                    if (tok_1_29 < num_tokens) {
                        float gk_1_29 = (float)(__nv_bfloat16)red[29];
                        float uk_1_29 = (float)(__nv_bfloat16)red[93];
                        float _exp2_87 = approx_exp2((-gk_1_29) * 1.442695f);
                        float sig_1_29 = approx_rcp(1.0f + _exp2_87);
                        float _tanh_88 = tanhf(gk_1_29 * 0.25f);
                        float aa_1_29 = 4.0f * _tanh_88 * sig_1_29;
                        float _tanh_89 = tanhf(uk_1_29 * 0.04f);
                        float bb_1_29 = 25.0f * _tanh_89;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_29 * 6144 + row_s + frow_2_29)) + (0)) = (__nv_bfloat16)(aa_1_29 * bb_1_29);
                    }
                    int tok_1_30 = 56 + lane_pair * 2;
                    int frow_2_30 = row_base + 8;
                    if (tok_1_30 < num_tokens) {
                        float gk_1_30 = (float)(__nv_bfloat16)red[30];
                        float uk_1_30 = (float)(__nv_bfloat16)red[94];
                        float _exp2_90 = approx_exp2((-gk_1_30) * 1.442695f);
                        float sig_1_30 = approx_rcp(1.0f + _exp2_90);
                        float _tanh_91 = tanhf(gk_1_30 * 0.25f);
                        float aa_1_30 = 4.0f * _tanh_91 * sig_1_30;
                        float _tanh_92 = tanhf(uk_1_30 * 0.04f);
                        float bb_1_30 = 25.0f * _tanh_92;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_30 * 6144 + row_s + frow_2_30)) + (0)) = (__nv_bfloat16)(aa_1_30 * bb_1_30);
                    }
                    int tok_1_31 = 56 + lane_pair * 2 + 1;
                    int frow_2_31 = row_base + 8;
                    if (tok_1_31 < num_tokens) {
                        float gk_1_31 = (float)(__nv_bfloat16)red[31];
                        float uk_1_31 = (float)(__nv_bfloat16)red[95];
                        float _exp2_93 = approx_exp2((-gk_1_31) * 1.442695f);
                        float sig_1_31 = approx_rcp(1.0f + _exp2_93);
                        float _tanh_94 = tanhf(gk_1_31 * 0.25f);
                        float aa_1_31 = 4.0f * _tanh_94 * sig_1_31;
                        float _tanh_95 = tanhf(uk_1_31 * 0.04f);
                        float bb_1_31 = 25.0f * _tanh_95;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_31 * 6144 + row_s + frow_2_31)) + (0)) = (__nv_bfloat16)(aa_1_31 * bb_1_31);
                    }
                    int tok_1_32 = 64 + lane_pair * 2;
                    int frow_2_32 = row_base;
                    if (tok_1_32 < num_tokens) {
                        float gk_1_32 = (float)(__nv_bfloat16)red[32];
                        float uk_1_32 = (float)(__nv_bfloat16)red[96];
                        float _exp2_96 = approx_exp2((-gk_1_32) * 1.442695f);
                        float sig_1_32 = approx_rcp(1.0f + _exp2_96);
                        float _tanh_97 = tanhf(gk_1_32 * 0.25f);
                        float aa_1_32 = 4.0f * _tanh_97 * sig_1_32;
                        float _tanh_98 = tanhf(uk_1_32 * 0.04f);
                        float bb_1_32 = 25.0f * _tanh_98;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_32 * 6144 + row_s + frow_2_32)) + (0)) = (__nv_bfloat16)(aa_1_32 * bb_1_32);
                    }
                    int tok_1_33 = 64 + lane_pair * 2 + 1;
                    int frow_2_33 = row_base;
                    if (tok_1_33 < num_tokens) {
                        float gk_1_33 = (float)(__nv_bfloat16)red[33];
                        float uk_1_33 = (float)(__nv_bfloat16)red[97];
                        float _exp2_99 = approx_exp2((-gk_1_33) * 1.442695f);
                        float sig_1_33 = approx_rcp(1.0f + _exp2_99);
                        float _tanh_100 = tanhf(gk_1_33 * 0.25f);
                        float aa_1_33 = 4.0f * _tanh_100 * sig_1_33;
                        float _tanh_101 = tanhf(uk_1_33 * 0.04f);
                        float bb_1_33 = 25.0f * _tanh_101;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_33 * 6144 + row_s + frow_2_33)) + (0)) = (__nv_bfloat16)(aa_1_33 * bb_1_33);
                    }
                    int tok_1_34 = 64 + lane_pair * 2;
                    int frow_2_34 = row_base + 8;
                    if (tok_1_34 < num_tokens) {
                        float gk_1_34 = (float)(__nv_bfloat16)red[34];
                        float uk_1_34 = (float)(__nv_bfloat16)red[98];
                        float _exp2_102 = approx_exp2((-gk_1_34) * 1.442695f);
                        float sig_1_34 = approx_rcp(1.0f + _exp2_102);
                        float _tanh_103 = tanhf(gk_1_34 * 0.25f);
                        float aa_1_34 = 4.0f * _tanh_103 * sig_1_34;
                        float _tanh_104 = tanhf(uk_1_34 * 0.04f);
                        float bb_1_34 = 25.0f * _tanh_104;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_34 * 6144 + row_s + frow_2_34)) + (0)) = (__nv_bfloat16)(aa_1_34 * bb_1_34);
                    }
                    int tok_1_35 = 64 + lane_pair * 2 + 1;
                    int frow_2_35 = row_base + 8;
                    if (tok_1_35 < num_tokens) {
                        float gk_1_35 = (float)(__nv_bfloat16)red[35];
                        float uk_1_35 = (float)(__nv_bfloat16)red[99];
                        float _exp2_105 = approx_exp2((-gk_1_35) * 1.442695f);
                        float sig_1_35 = approx_rcp(1.0f + _exp2_105);
                        float _tanh_106 = tanhf(gk_1_35 * 0.25f);
                        float aa_1_35 = 4.0f * _tanh_106 * sig_1_35;
                        float _tanh_107 = tanhf(uk_1_35 * 0.04f);
                        float bb_1_35 = 25.0f * _tanh_107;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_35 * 6144 + row_s + frow_2_35)) + (0)) = (__nv_bfloat16)(aa_1_35 * bb_1_35);
                    }
                    int tok_1_36 = 72 + lane_pair * 2;
                    int frow_2_36 = row_base;
                    if (tok_1_36 < num_tokens) {
                        float gk_1_36 = (float)(__nv_bfloat16)red[36];
                        float uk_1_36 = (float)(__nv_bfloat16)red[100];
                        float _exp2_108 = approx_exp2((-gk_1_36) * 1.442695f);
                        float sig_1_36 = approx_rcp(1.0f + _exp2_108);
                        float _tanh_109 = tanhf(gk_1_36 * 0.25f);
                        float aa_1_36 = 4.0f * _tanh_109 * sig_1_36;
                        float _tanh_110 = tanhf(uk_1_36 * 0.04f);
                        float bb_1_36 = 25.0f * _tanh_110;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_36 * 6144 + row_s + frow_2_36)) + (0)) = (__nv_bfloat16)(aa_1_36 * bb_1_36);
                    }
                    int tok_1_37 = 72 + lane_pair * 2 + 1;
                    int frow_2_37 = row_base;
                    if (tok_1_37 < num_tokens) {
                        float gk_1_37 = (float)(__nv_bfloat16)red[37];
                        float uk_1_37 = (float)(__nv_bfloat16)red[101];
                        float _exp2_111 = approx_exp2((-gk_1_37) * 1.442695f);
                        float sig_1_37 = approx_rcp(1.0f + _exp2_111);
                        float _tanh_112 = tanhf(gk_1_37 * 0.25f);
                        float aa_1_37 = 4.0f * _tanh_112 * sig_1_37;
                        float _tanh_113 = tanhf(uk_1_37 * 0.04f);
                        float bb_1_37 = 25.0f * _tanh_113;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_37 * 6144 + row_s + frow_2_37)) + (0)) = (__nv_bfloat16)(aa_1_37 * bb_1_37);
                    }
                    int tok_1_38 = 72 + lane_pair * 2;
                    int frow_2_38 = row_base + 8;
                    if (tok_1_38 < num_tokens) {
                        float gk_1_38 = (float)(__nv_bfloat16)red[38];
                        float uk_1_38 = (float)(__nv_bfloat16)red[102];
                        float _exp2_114 = approx_exp2((-gk_1_38) * 1.442695f);
                        float sig_1_38 = approx_rcp(1.0f + _exp2_114);
                        float _tanh_115 = tanhf(gk_1_38 * 0.25f);
                        float aa_1_38 = 4.0f * _tanh_115 * sig_1_38;
                        float _tanh_116 = tanhf(uk_1_38 * 0.04f);
                        float bb_1_38 = 25.0f * _tanh_116;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_38 * 6144 + row_s + frow_2_38)) + (0)) = (__nv_bfloat16)(aa_1_38 * bb_1_38);
                    }
                    int tok_1_39 = 72 + lane_pair * 2 + 1;
                    int frow_2_39 = row_base + 8;
                    if (tok_1_39 < num_tokens) {
                        float gk_1_39 = (float)(__nv_bfloat16)red[39];
                        float uk_1_39 = (float)(__nv_bfloat16)red[103];
                        float _exp2_117 = approx_exp2((-gk_1_39) * 1.442695f);
                        float sig_1_39 = approx_rcp(1.0f + _exp2_117);
                        float _tanh_118 = tanhf(gk_1_39 * 0.25f);
                        float aa_1_39 = 4.0f * _tanh_118 * sig_1_39;
                        float _tanh_119 = tanhf(uk_1_39 * 0.04f);
                        float bb_1_39 = 25.0f * _tanh_119;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_39 * 6144 + row_s + frow_2_39)) + (0)) = (__nv_bfloat16)(aa_1_39 * bb_1_39);
                    }
                    int tok_1_40 = 80 + lane_pair * 2;
                    int frow_2_40 = row_base;
                    if (tok_1_40 < num_tokens) {
                        float gk_1_40 = (float)(__nv_bfloat16)red[40];
                        float uk_1_40 = (float)(__nv_bfloat16)red[104];
                        float _exp2_120 = approx_exp2((-gk_1_40) * 1.442695f);
                        float sig_1_40 = approx_rcp(1.0f + _exp2_120);
                        float _tanh_121 = tanhf(gk_1_40 * 0.25f);
                        float aa_1_40 = 4.0f * _tanh_121 * sig_1_40;
                        float _tanh_122 = tanhf(uk_1_40 * 0.04f);
                        float bb_1_40 = 25.0f * _tanh_122;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_40 * 6144 + row_s + frow_2_40)) + (0)) = (__nv_bfloat16)(aa_1_40 * bb_1_40);
                    }
                    int tok_1_41 = 80 + lane_pair * 2 + 1;
                    int frow_2_41 = row_base;
                    if (tok_1_41 < num_tokens) {
                        float gk_1_41 = (float)(__nv_bfloat16)red[41];
                        float uk_1_41 = (float)(__nv_bfloat16)red[105];
                        float _exp2_123 = approx_exp2((-gk_1_41) * 1.442695f);
                        float sig_1_41 = approx_rcp(1.0f + _exp2_123);
                        float _tanh_124 = tanhf(gk_1_41 * 0.25f);
                        float aa_1_41 = 4.0f * _tanh_124 * sig_1_41;
                        float _tanh_125 = tanhf(uk_1_41 * 0.04f);
                        float bb_1_41 = 25.0f * _tanh_125;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_41 * 6144 + row_s + frow_2_41)) + (0)) = (__nv_bfloat16)(aa_1_41 * bb_1_41);
                    }
                    int tok_1_42 = 80 + lane_pair * 2;
                    int frow_2_42 = row_base + 8;
                    if (tok_1_42 < num_tokens) {
                        float gk_1_42 = (float)(__nv_bfloat16)red[42];
                        float uk_1_42 = (float)(__nv_bfloat16)red[106];
                        float _exp2_126 = approx_exp2((-gk_1_42) * 1.442695f);
                        float sig_1_42 = approx_rcp(1.0f + _exp2_126);
                        float _tanh_127 = tanhf(gk_1_42 * 0.25f);
                        float aa_1_42 = 4.0f * _tanh_127 * sig_1_42;
                        float _tanh_128 = tanhf(uk_1_42 * 0.04f);
                        float bb_1_42 = 25.0f * _tanh_128;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_42 * 6144 + row_s + frow_2_42)) + (0)) = (__nv_bfloat16)(aa_1_42 * bb_1_42);
                    }
                    int tok_1_43 = 80 + lane_pair * 2 + 1;
                    int frow_2_43 = row_base + 8;
                    if (tok_1_43 < num_tokens) {
                        float gk_1_43 = (float)(__nv_bfloat16)red[43];
                        float uk_1_43 = (float)(__nv_bfloat16)red[107];
                        float _exp2_129 = approx_exp2((-gk_1_43) * 1.442695f);
                        float sig_1_43 = approx_rcp(1.0f + _exp2_129);
                        float _tanh_130 = tanhf(gk_1_43 * 0.25f);
                        float aa_1_43 = 4.0f * _tanh_130 * sig_1_43;
                        float _tanh_131 = tanhf(uk_1_43 * 0.04f);
                        float bb_1_43 = 25.0f * _tanh_131;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_43 * 6144 + row_s + frow_2_43)) + (0)) = (__nv_bfloat16)(aa_1_43 * bb_1_43);
                    }
                    int tok_1_44 = 88 + lane_pair * 2;
                    int frow_2_44 = row_base;
                    if (tok_1_44 < num_tokens) {
                        float gk_1_44 = (float)(__nv_bfloat16)red[44];
                        float uk_1_44 = (float)(__nv_bfloat16)red[108];
                        float _exp2_132 = approx_exp2((-gk_1_44) * 1.442695f);
                        float sig_1_44 = approx_rcp(1.0f + _exp2_132);
                        float _tanh_133 = tanhf(gk_1_44 * 0.25f);
                        float aa_1_44 = 4.0f * _tanh_133 * sig_1_44;
                        float _tanh_134 = tanhf(uk_1_44 * 0.04f);
                        float bb_1_44 = 25.0f * _tanh_134;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_44 * 6144 + row_s + frow_2_44)) + (0)) = (__nv_bfloat16)(aa_1_44 * bb_1_44);
                    }
                    int tok_1_45 = 88 + lane_pair * 2 + 1;
                    int frow_2_45 = row_base;
                    if (tok_1_45 < num_tokens) {
                        float gk_1_45 = (float)(__nv_bfloat16)red[45];
                        float uk_1_45 = (float)(__nv_bfloat16)red[109];
                        float _exp2_135 = approx_exp2((-gk_1_45) * 1.442695f);
                        float sig_1_45 = approx_rcp(1.0f + _exp2_135);
                        float _tanh_136 = tanhf(gk_1_45 * 0.25f);
                        float aa_1_45 = 4.0f * _tanh_136 * sig_1_45;
                        float _tanh_137 = tanhf(uk_1_45 * 0.04f);
                        float bb_1_45 = 25.0f * _tanh_137;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_45 * 6144 + row_s + frow_2_45)) + (0)) = (__nv_bfloat16)(aa_1_45 * bb_1_45);
                    }
                    int tok_1_46 = 88 + lane_pair * 2;
                    int frow_2_46 = row_base + 8;
                    if (tok_1_46 < num_tokens) {
                        float gk_1_46 = (float)(__nv_bfloat16)red[46];
                        float uk_1_46 = (float)(__nv_bfloat16)red[110];
                        float _exp2_138 = approx_exp2((-gk_1_46) * 1.442695f);
                        float sig_1_46 = approx_rcp(1.0f + _exp2_138);
                        float _tanh_139 = tanhf(gk_1_46 * 0.25f);
                        float aa_1_46 = 4.0f * _tanh_139 * sig_1_46;
                        float _tanh_140 = tanhf(uk_1_46 * 0.04f);
                        float bb_1_46 = 25.0f * _tanh_140;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_46 * 6144 + row_s + frow_2_46)) + (0)) = (__nv_bfloat16)(aa_1_46 * bb_1_46);
                    }
                    int tok_1_47 = 88 + lane_pair * 2 + 1;
                    int frow_2_47 = row_base + 8;
                    if (tok_1_47 < num_tokens) {
                        float gk_1_47 = (float)(__nv_bfloat16)red[47];
                        float uk_1_47 = (float)(__nv_bfloat16)red[111];
                        float _exp2_141 = approx_exp2((-gk_1_47) * 1.442695f);
                        float sig_1_47 = approx_rcp(1.0f + _exp2_141);
                        float _tanh_142 = tanhf(gk_1_47 * 0.25f);
                        float aa_1_47 = 4.0f * _tanh_142 * sig_1_47;
                        float _tanh_143 = tanhf(uk_1_47 * 0.04f);
                        float bb_1_47 = 25.0f * _tanh_143;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_47 * 6144 + row_s + frow_2_47)) + (0)) = (__nv_bfloat16)(aa_1_47 * bb_1_47);
                    }
                    int tok_1_48 = 96 + lane_pair * 2;
                    int frow_2_48 = row_base;
                    if (tok_1_48 < num_tokens) {
                        float gk_1_48 = (float)(__nv_bfloat16)red[48];
                        float uk_1_48 = (float)(__nv_bfloat16)red[112];
                        float _exp2_144 = approx_exp2((-gk_1_48) * 1.442695f);
                        float sig_1_48 = approx_rcp(1.0f + _exp2_144);
                        float _tanh_145 = tanhf(gk_1_48 * 0.25f);
                        float aa_1_48 = 4.0f * _tanh_145 * sig_1_48;
                        float _tanh_146 = tanhf(uk_1_48 * 0.04f);
                        float bb_1_48 = 25.0f * _tanh_146;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_48 * 6144 + row_s + frow_2_48)) + (0)) = (__nv_bfloat16)(aa_1_48 * bb_1_48);
                    }
                    int tok_1_49 = 96 + lane_pair * 2 + 1;
                    int frow_2_49 = row_base;
                    if (tok_1_49 < num_tokens) {
                        float gk_1_49 = (float)(__nv_bfloat16)red[49];
                        float uk_1_49 = (float)(__nv_bfloat16)red[113];
                        float _exp2_147 = approx_exp2((-gk_1_49) * 1.442695f);
                        float sig_1_49 = approx_rcp(1.0f + _exp2_147);
                        float _tanh_148 = tanhf(gk_1_49 * 0.25f);
                        float aa_1_49 = 4.0f * _tanh_148 * sig_1_49;
                        float _tanh_149 = tanhf(uk_1_49 * 0.04f);
                        float bb_1_49 = 25.0f * _tanh_149;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_49 * 6144 + row_s + frow_2_49)) + (0)) = (__nv_bfloat16)(aa_1_49 * bb_1_49);
                    }
                    int tok_1_50 = 96 + lane_pair * 2;
                    int frow_2_50 = row_base + 8;
                    if (tok_1_50 < num_tokens) {
                        float gk_1_50 = (float)(__nv_bfloat16)red[50];
                        float uk_1_50 = (float)(__nv_bfloat16)red[114];
                        float _exp2_150 = approx_exp2((-gk_1_50) * 1.442695f);
                        float sig_1_50 = approx_rcp(1.0f + _exp2_150);
                        float _tanh_151 = tanhf(gk_1_50 * 0.25f);
                        float aa_1_50 = 4.0f * _tanh_151 * sig_1_50;
                        float _tanh_152 = tanhf(uk_1_50 * 0.04f);
                        float bb_1_50 = 25.0f * _tanh_152;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_50 * 6144 + row_s + frow_2_50)) + (0)) = (__nv_bfloat16)(aa_1_50 * bb_1_50);
                    }
                    int tok_1_51 = 96 + lane_pair * 2 + 1;
                    int frow_2_51 = row_base + 8;
                    if (tok_1_51 < num_tokens) {
                        float gk_1_51 = (float)(__nv_bfloat16)red[51];
                        float uk_1_51 = (float)(__nv_bfloat16)red[115];
                        float _exp2_153 = approx_exp2((-gk_1_51) * 1.442695f);
                        float sig_1_51 = approx_rcp(1.0f + _exp2_153);
                        float _tanh_154 = tanhf(gk_1_51 * 0.25f);
                        float aa_1_51 = 4.0f * _tanh_154 * sig_1_51;
                        float _tanh_155 = tanhf(uk_1_51 * 0.04f);
                        float bb_1_51 = 25.0f * _tanh_155;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_51 * 6144 + row_s + frow_2_51)) + (0)) = (__nv_bfloat16)(aa_1_51 * bb_1_51);
                    }
                    int tok_1_52 = 104 + lane_pair * 2;
                    int frow_2_52 = row_base;
                    if (tok_1_52 < num_tokens) {
                        float gk_1_52 = (float)(__nv_bfloat16)red[52];
                        float uk_1_52 = (float)(__nv_bfloat16)red[116];
                        float _exp2_156 = approx_exp2((-gk_1_52) * 1.442695f);
                        float sig_1_52 = approx_rcp(1.0f + _exp2_156);
                        float _tanh_157 = tanhf(gk_1_52 * 0.25f);
                        float aa_1_52 = 4.0f * _tanh_157 * sig_1_52;
                        float _tanh_158 = tanhf(uk_1_52 * 0.04f);
                        float bb_1_52 = 25.0f * _tanh_158;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_52 * 6144 + row_s + frow_2_52)) + (0)) = (__nv_bfloat16)(aa_1_52 * bb_1_52);
                    }
                    int tok_1_53 = 104 + lane_pair * 2 + 1;
                    int frow_2_53 = row_base;
                    if (tok_1_53 < num_tokens) {
                        float gk_1_53 = (float)(__nv_bfloat16)red[53];
                        float uk_1_53 = (float)(__nv_bfloat16)red[117];
                        float _exp2_159 = approx_exp2((-gk_1_53) * 1.442695f);
                        float sig_1_53 = approx_rcp(1.0f + _exp2_159);
                        float _tanh_160 = tanhf(gk_1_53 * 0.25f);
                        float aa_1_53 = 4.0f * _tanh_160 * sig_1_53;
                        float _tanh_161 = tanhf(uk_1_53 * 0.04f);
                        float bb_1_53 = 25.0f * _tanh_161;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_53 * 6144 + row_s + frow_2_53)) + (0)) = (__nv_bfloat16)(aa_1_53 * bb_1_53);
                    }
                    int tok_1_54 = 104 + lane_pair * 2;
                    int frow_2_54 = row_base + 8;
                    if (tok_1_54 < num_tokens) {
                        float gk_1_54 = (float)(__nv_bfloat16)red[54];
                        float uk_1_54 = (float)(__nv_bfloat16)red[118];
                        float _exp2_162 = approx_exp2((-gk_1_54) * 1.442695f);
                        float sig_1_54 = approx_rcp(1.0f + _exp2_162);
                        float _tanh_163 = tanhf(gk_1_54 * 0.25f);
                        float aa_1_54 = 4.0f * _tanh_163 * sig_1_54;
                        float _tanh_164 = tanhf(uk_1_54 * 0.04f);
                        float bb_1_54 = 25.0f * _tanh_164;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_54 * 6144 + row_s + frow_2_54)) + (0)) = (__nv_bfloat16)(aa_1_54 * bb_1_54);
                    }
                    int tok_1_55 = 104 + lane_pair * 2 + 1;
                    int frow_2_55 = row_base + 8;
                    if (tok_1_55 < num_tokens) {
                        float gk_1_55 = (float)(__nv_bfloat16)red[55];
                        float uk_1_55 = (float)(__nv_bfloat16)red[119];
                        float _exp2_165 = approx_exp2((-gk_1_55) * 1.442695f);
                        float sig_1_55 = approx_rcp(1.0f + _exp2_165);
                        float _tanh_166 = tanhf(gk_1_55 * 0.25f);
                        float aa_1_55 = 4.0f * _tanh_166 * sig_1_55;
                        float _tanh_167 = tanhf(uk_1_55 * 0.04f);
                        float bb_1_55 = 25.0f * _tanh_167;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_55 * 6144 + row_s + frow_2_55)) + (0)) = (__nv_bfloat16)(aa_1_55 * bb_1_55);
                    }
                    int tok_1_56 = 112 + lane_pair * 2;
                    int frow_2_56 = row_base;
                    if (tok_1_56 < num_tokens) {
                        float gk_1_56 = (float)(__nv_bfloat16)red[56];
                        float uk_1_56 = (float)(__nv_bfloat16)red[120];
                        float _exp2_168 = approx_exp2((-gk_1_56) * 1.442695f);
                        float sig_1_56 = approx_rcp(1.0f + _exp2_168);
                        float _tanh_169 = tanhf(gk_1_56 * 0.25f);
                        float aa_1_56 = 4.0f * _tanh_169 * sig_1_56;
                        float _tanh_170 = tanhf(uk_1_56 * 0.04f);
                        float bb_1_56 = 25.0f * _tanh_170;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_56 * 6144 + row_s + frow_2_56)) + (0)) = (__nv_bfloat16)(aa_1_56 * bb_1_56);
                    }
                    int tok_1_57 = 112 + lane_pair * 2 + 1;
                    int frow_2_57 = row_base;
                    if (tok_1_57 < num_tokens) {
                        float gk_1_57 = (float)(__nv_bfloat16)red[57];
                        float uk_1_57 = (float)(__nv_bfloat16)red[121];
                        float _exp2_171 = approx_exp2((-gk_1_57) * 1.442695f);
                        float sig_1_57 = approx_rcp(1.0f + _exp2_171);
                        float _tanh_172 = tanhf(gk_1_57 * 0.25f);
                        float aa_1_57 = 4.0f * _tanh_172 * sig_1_57;
                        float _tanh_173 = tanhf(uk_1_57 * 0.04f);
                        float bb_1_57 = 25.0f * _tanh_173;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_57 * 6144 + row_s + frow_2_57)) + (0)) = (__nv_bfloat16)(aa_1_57 * bb_1_57);
                    }
                    int tok_1_58 = 112 + lane_pair * 2;
                    int frow_2_58 = row_base + 8;
                    if (tok_1_58 < num_tokens) {
                        float gk_1_58 = (float)(__nv_bfloat16)red[58];
                        float uk_1_58 = (float)(__nv_bfloat16)red[122];
                        float _exp2_174 = approx_exp2((-gk_1_58) * 1.442695f);
                        float sig_1_58 = approx_rcp(1.0f + _exp2_174);
                        float _tanh_175 = tanhf(gk_1_58 * 0.25f);
                        float aa_1_58 = 4.0f * _tanh_175 * sig_1_58;
                        float _tanh_176 = tanhf(uk_1_58 * 0.04f);
                        float bb_1_58 = 25.0f * _tanh_176;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_58 * 6144 + row_s + frow_2_58)) + (0)) = (__nv_bfloat16)(aa_1_58 * bb_1_58);
                    }
                    int tok_1_59 = 112 + lane_pair * 2 + 1;
                    int frow_2_59 = row_base + 8;
                    if (tok_1_59 < num_tokens) {
                        float gk_1_59 = (float)(__nv_bfloat16)red[59];
                        float uk_1_59 = (float)(__nv_bfloat16)red[123];
                        float _exp2_177 = approx_exp2((-gk_1_59) * 1.442695f);
                        float sig_1_59 = approx_rcp(1.0f + _exp2_177);
                        float _tanh_178 = tanhf(gk_1_59 * 0.25f);
                        float aa_1_59 = 4.0f * _tanh_178 * sig_1_59;
                        float _tanh_179 = tanhf(uk_1_59 * 0.04f);
                        float bb_1_59 = 25.0f * _tanh_179;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_59 * 6144 + row_s + frow_2_59)) + (0)) = (__nv_bfloat16)(aa_1_59 * bb_1_59);
                    }
                    int tok_1_60 = 120 + lane_pair * 2;
                    int frow_2_60 = row_base;
                    if (tok_1_60 < num_tokens) {
                        float gk_1_60 = (float)(__nv_bfloat16)red[60];
                        float uk_1_60 = (float)(__nv_bfloat16)red[124];
                        float _exp2_180 = approx_exp2((-gk_1_60) * 1.442695f);
                        float sig_1_60 = approx_rcp(1.0f + _exp2_180);
                        float _tanh_181 = tanhf(gk_1_60 * 0.25f);
                        float aa_1_60 = 4.0f * _tanh_181 * sig_1_60;
                        float _tanh_182 = tanhf(uk_1_60 * 0.04f);
                        float bb_1_60 = 25.0f * _tanh_182;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_60 * 6144 + row_s + frow_2_60)) + (0)) = (__nv_bfloat16)(aa_1_60 * bb_1_60);
                    }
                    int tok_1_61 = 120 + lane_pair * 2 + 1;
                    int frow_2_61 = row_base;
                    if (tok_1_61 < num_tokens) {
                        float gk_1_61 = (float)(__nv_bfloat16)red[61];
                        float uk_1_61 = (float)(__nv_bfloat16)red[125];
                        float _exp2_183 = approx_exp2((-gk_1_61) * 1.442695f);
                        float sig_1_61 = approx_rcp(1.0f + _exp2_183);
                        float _tanh_184 = tanhf(gk_1_61 * 0.25f);
                        float aa_1_61 = 4.0f * _tanh_184 * sig_1_61;
                        float _tanh_185 = tanhf(uk_1_61 * 0.04f);
                        float bb_1_61 = 25.0f * _tanh_185;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_61 * 6144 + row_s + frow_2_61)) + (0)) = (__nv_bfloat16)(aa_1_61 * bb_1_61);
                    }
                    int tok_1_62 = 120 + lane_pair * 2;
                    int frow_2_62 = row_base + 8;
                    if (tok_1_62 < num_tokens) {
                        float gk_1_62 = (float)(__nv_bfloat16)red[62];
                        float uk_1_62 = (float)(__nv_bfloat16)red[126];
                        float _exp2_186 = approx_exp2((-gk_1_62) * 1.442695f);
                        float sig_1_62 = approx_rcp(1.0f + _exp2_186);
                        float _tanh_187 = tanhf(gk_1_62 * 0.25f);
                        float aa_1_62 = 4.0f * _tanh_187 * sig_1_62;
                        float _tanh_188 = tanhf(uk_1_62 * 0.04f);
                        float bb_1_62 = 25.0f * _tanh_188;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_62 * 6144 + row_s + frow_2_62)) + (0)) = (__nv_bfloat16)(aa_1_62 * bb_1_62);
                    }
                    int tok_1_63 = 120 + lane_pair * 2 + 1;
                    int frow_2_63 = row_base + 8;
                    if (tok_1_63 < num_tokens) {
                        float gk_1_63 = (float)(__nv_bfloat16)red[63];
                        float uk_1_63 = (float)(__nv_bfloat16)red[127];
                        float _exp2_189 = approx_exp2((-gk_1_63) * 1.442695f);
                        float sig_1_63 = approx_rcp(1.0f + _exp2_189);
                        float _tanh_190 = tanhf(gk_1_63 * 0.25f);
                        float aa_1_63 = 4.0f * _tanh_190 * sig_1_63;
                        float _tanh_191 = tanhf(uk_1_63 * 0.04f);
                        float bb_1_63 = 25.0f * _tanh_191;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_1_63 * 6144 + row_s + frow_2_63)) + (0)) = (__nv_bfloat16)(aa_1_63 * bb_1_63);
                    }
                } else {
                    if (num_tokens > 0) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (row_r + tid_1)) + (0)) = red[0];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = (__nv_bfloat16)red[0];
                        }
                    }
                    if (num_tokens > 1) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (896 + row_r + tid_1)) + (0)) = red[1];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (3584 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[1];
                        }
                    }
                    if (num_tokens > 2) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (1792 + row_r + tid_1)) + (0)) = red[2];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[2];
                        }
                    }
                    if (num_tokens > 3) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (2688 + row_r + tid_1)) + (0)) = red[3];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (10752 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[3];
                        }
                    }
                    if (num_tokens > 4) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (3584 + row_r + tid_1)) + (0)) = red[4];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[4];
                        }
                    }
                    if (num_tokens > 5) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (4480 + row_r + tid_1)) + (0)) = red[5];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (17920 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[5];
                        }
                    }
                    if (num_tokens > 6) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (5376 + row_r + tid_1)) + (0)) = red[6];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[6];
                        }
                    }
                    if (num_tokens > 7) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (6272 + row_r + tid_1)) + (0)) = red[7];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (25088 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[7];
                        }
                    }
                    if (num_tokens > 8) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (7168 + row_r + tid_1)) + (0)) = red[8];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[8];
                        }
                    }
                    if (num_tokens > 9) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8064 + row_r + tid_1)) + (0)) = red[9];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (32256 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[9];
                        }
                    }
                    if (num_tokens > 10) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8960 + row_r + tid_1)) + (0)) = red[10];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[10];
                        }
                    }
                    if (num_tokens > 11) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (9856 + row_r + tid_1)) + (0)) = red[11];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (39424 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[11];
                        }
                    }
                    if (num_tokens > 12) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (10752 + row_r + tid_1)) + (0)) = red[12];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[12];
                        }
                    }
                    if (num_tokens > 13) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (11648 + row_r + tid_1)) + (0)) = red[13];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (46592 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[13];
                        }
                    }
                    if (num_tokens > 14) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (12544 + row_r + tid_1)) + (0)) = red[14];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[14];
                        }
                    }
                    if (num_tokens > 15) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (13440 + row_r + tid_1)) + (0)) = red[15];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (53760 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[15];
                        }
                    }
                    if (num_tokens > 16) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (14336 + row_r + tid_1)) + (0)) = red[16];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (57344 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[16];
                        }
                    }
                    if (num_tokens > 17) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (15232 + row_r + tid_1)) + (0)) = red[17];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (60928 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[17];
                        }
                    }
                    if (num_tokens > 18) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (16128 + row_r + tid_1)) + (0)) = red[18];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (64512 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[18];
                        }
                    }
                    if (num_tokens > 19) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17024 + row_r + tid_1)) + (0)) = red[19];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (68096 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[19];
                        }
                    }
                    if (num_tokens > 20) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17920 + row_r + tid_1)) + (0)) = red[20];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (71680 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[20];
                        }
                    }
                    if (num_tokens > 21) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (18816 + row_r + tid_1)) + (0)) = red[21];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (75264 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[21];
                        }
                    }
                    if (num_tokens > 22) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (19712 + row_r + tid_1)) + (0)) = red[22];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (78848 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[22];
                        }
                    }
                    if (num_tokens > 23) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (20608 + row_r + tid_1)) + (0)) = red[23];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (82432 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[23];
                        }
                    }
                    if (num_tokens > 24) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (21504 + row_r + tid_1)) + (0)) = red[24];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (86016 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[24];
                        }
                    }
                    if (num_tokens > 25) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (22400 + row_r + tid_1)) + (0)) = red[25];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (89600 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[25];
                        }
                    }
                    if (num_tokens > 26) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (23296 + row_r + tid_1)) + (0)) = red[26];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (93184 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[26];
                        }
                    }
                    if (num_tokens > 27) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (24192 + row_r + tid_1)) + (0)) = red[27];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (96768 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[27];
                        }
                    }
                    if (num_tokens > 28) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25088 + row_r + tid_1)) + (0)) = red[28];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (100352 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[28];
                        }
                    }
                    if (num_tokens > 29) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25984 + row_r + tid_1)) + (0)) = red[29];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (103936 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[29];
                        }
                    }
                    if (num_tokens > 30) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (26880 + row_r + tid_1)) + (0)) = red[30];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (107520 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[30];
                        }
                    }
                    if (num_tokens > 31) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (27776 + row_r + tid_1)) + (0)) = red[31];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (111104 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[31];
                        }
                    }
                    if (num_tokens > 32) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (28672 + row_r + tid_1)) + (0)) = red[32];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (114688 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[32];
                        }
                    }
                    if (num_tokens > 33) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (29568 + row_r + tid_1)) + (0)) = red[33];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (118272 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[33];
                        }
                    }
                    if (num_tokens > 34) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (30464 + row_r + tid_1)) + (0)) = red[34];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (121856 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[34];
                        }
                    }
                    if (num_tokens > 35) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (31360 + row_r + tid_1)) + (0)) = red[35];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (125440 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[35];
                        }
                    }
                    if (num_tokens > 36) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (32256 + row_r + tid_1)) + (0)) = red[36];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (129024 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[36];
                        }
                    }
                    if (num_tokens > 37) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (33152 + row_r + tid_1)) + (0)) = red[37];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (132608 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[37];
                        }
                    }
                    if (num_tokens > 38) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34048 + row_r + tid_1)) + (0)) = red[38];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (136192 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[38];
                        }
                    }
                    if (num_tokens > 39) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34944 + row_r + tid_1)) + (0)) = red[39];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (139776 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[39];
                        }
                    }
                    if (num_tokens > 40) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (35840 + row_r + tid_1)) + (0)) = red[40];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (143360 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[40];
                        }
                    }
                    if (num_tokens > 41) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (36736 + row_r + tid_1)) + (0)) = red[41];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (146944 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[41];
                        }
                    }
                    if (num_tokens > 42) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (37632 + row_r + tid_1)) + (0)) = red[42];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (150528 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[42];
                        }
                    }
                    if (num_tokens > 43) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (38528 + row_r + tid_1)) + (0)) = red[43];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (154112 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[43];
                        }
                    }
                    if (num_tokens > 44) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (39424 + row_r + tid_1)) + (0)) = red[44];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (157696 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[44];
                        }
                    }
                    if (num_tokens > 45) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (40320 + row_r + tid_1)) + (0)) = red[45];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (161280 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[45];
                        }
                    }
                    if (num_tokens > 46) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (41216 + row_r + tid_1)) + (0)) = red[46];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (164864 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[46];
                        }
                    }
                    if (num_tokens > 47) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (42112 + row_r + tid_1)) + (0)) = red[47];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (168448 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[47];
                        }
                    }
                    if (num_tokens > 48) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43008 + row_r + tid_1)) + (0)) = red[48];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (172032 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[48];
                        }
                    }
                    if (num_tokens > 49) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43904 + row_r + tid_1)) + (0)) = red[49];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (175616 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[49];
                        }
                    }
                    if (num_tokens > 50) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (44800 + row_r + tid_1)) + (0)) = red[50];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (179200 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[50];
                        }
                    }
                    if (num_tokens > 51) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (45696 + row_r + tid_1)) + (0)) = red[51];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (182784 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[51];
                        }
                    }
                    if (num_tokens > 52) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (46592 + row_r + tid_1)) + (0)) = red[52];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (186368 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[52];
                        }
                    }
                    if (num_tokens > 53) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (47488 + row_r + tid_1)) + (0)) = red[53];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (189952 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[53];
                        }
                    }
                    if (num_tokens > 54) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (48384 + row_r + tid_1)) + (0)) = red[54];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (193536 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[54];
                        }
                    }
                    if (num_tokens > 55) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (49280 + row_r + tid_1)) + (0)) = red[55];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (197120 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[55];
                        }
                    }
                    if (num_tokens > 56) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (50176 + row_r + tid_1)) + (0)) = red[56];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (200704 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[56];
                        }
                    }
                    if (num_tokens > 57) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51072 + row_r + tid_1)) + (0)) = red[57];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (204288 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[57];
                        }
                    }
                    if (num_tokens > 58) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51968 + row_r + tid_1)) + (0)) = red[58];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (207872 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[58];
                        }
                    }
                    if (num_tokens > 59) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (52864 + row_r + tid_1)) + (0)) = red[59];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (211456 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[59];
                        }
                    }
                    if (num_tokens > 60) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (53760 + row_r + tid_1)) + (0)) = red[60];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (215040 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[60];
                        }
                    }
                    if (num_tokens > 61) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (54656 + row_r + tid_1)) + (0)) = red[61];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (218624 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[61];
                        }
                    }
                    if (num_tokens > 62) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (55552 + row_r + tid_1)) + (0)) = red[62];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (222208 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[62];
                        }
                    }
                    if (num_tokens > 63) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (56448 + row_r + tid_1)) + (0)) = red[63];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (225792 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[63];
                        }
                    }
                    if (num_tokens > 64) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (57344 + row_r + tid_1)) + (0)) = red[64];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (229376 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[64];
                        }
                    }
                    if (num_tokens > 65) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (58240 + row_r + tid_1)) + (0)) = red[65];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (232960 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[65];
                        }
                    }
                    if (num_tokens > 66) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (59136 + row_r + tid_1)) + (0)) = red[66];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (236544 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[66];
                        }
                    }
                    if (num_tokens > 67) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (60032 + row_r + tid_1)) + (0)) = red[67];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (240128 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[67];
                        }
                    }
                    if (num_tokens > 68) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (60928 + row_r + tid_1)) + (0)) = red[68];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (243712 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[68];
                        }
                    }
                    if (num_tokens > 69) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (61824 + row_r + tid_1)) + (0)) = red[69];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (247296 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[69];
                        }
                    }
                    if (num_tokens > 70) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (62720 + row_r + tid_1)) + (0)) = red[70];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (250880 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[70];
                        }
                    }
                    if (num_tokens > 71) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (63616 + row_r + tid_1)) + (0)) = red[71];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (254464 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[71];
                        }
                    }
                    if (num_tokens > 72) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (64512 + row_r + tid_1)) + (0)) = red[72];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (258048 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[72];
                        }
                    }
                    if (num_tokens > 73) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (65408 + row_r + tid_1)) + (0)) = red[73];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (261632 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[73];
                        }
                    }
                    if (num_tokens > 74) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (66304 + row_r + tid_1)) + (0)) = red[74];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (265216 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[74];
                        }
                    }
                    if (num_tokens > 75) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (67200 + row_r + tid_1)) + (0)) = red[75];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (268800 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[75];
                        }
                    }
                    if (num_tokens > 76) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (68096 + row_r + tid_1)) + (0)) = red[76];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (272384 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[76];
                        }
                    }
                    if (num_tokens > 77) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (68992 + row_r + tid_1)) + (0)) = red[77];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (275968 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[77];
                        }
                    }
                    if (num_tokens > 78) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (69888 + row_r + tid_1)) + (0)) = red[78];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (279552 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[78];
                        }
                    }
                    if (num_tokens > 79) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (70784 + row_r + tid_1)) + (0)) = red[79];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (283136 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[79];
                        }
                    }
                    if (num_tokens > 80) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (71680 + row_r + tid_1)) + (0)) = red[80];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (286720 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[80];
                        }
                    }
                    if (num_tokens > 81) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (72576 + row_r + tid_1)) + (0)) = red[81];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (290304 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[81];
                        }
                    }
                    if (num_tokens > 82) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (73472 + row_r + tid_1)) + (0)) = red[82];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (293888 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[82];
                        }
                    }
                    if (num_tokens > 83) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (74368 + row_r + tid_1)) + (0)) = red[83];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (297472 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[83];
                        }
                    }
                    if (num_tokens > 84) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (75264 + row_r + tid_1)) + (0)) = red[84];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (301056 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[84];
                        }
                    }
                    if (num_tokens > 85) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (76160 + row_r + tid_1)) + (0)) = red[85];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (304640 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[85];
                        }
                    }
                    if (num_tokens > 86) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (77056 + row_r + tid_1)) + (0)) = red[86];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (308224 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[86];
                        }
                    }
                    if (num_tokens > 87) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (77952 + row_r + tid_1)) + (0)) = red[87];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (311808 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[87];
                        }
                    }
                    if (num_tokens > 88) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (78848 + row_r + tid_1)) + (0)) = red[88];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (315392 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[88];
                        }
                    }
                    if (num_tokens > 89) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (79744 + row_r + tid_1)) + (0)) = red[89];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (318976 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[89];
                        }
                    }
                    if (num_tokens > 90) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (80640 + row_r + tid_1)) + (0)) = red[90];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (322560 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[90];
                        }
                    }
                    if (num_tokens > 91) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (81536 + row_r + tid_1)) + (0)) = red[91];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (326144 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[91];
                        }
                    }
                    if (num_tokens > 92) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (82432 + row_r + tid_1)) + (0)) = red[92];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (329728 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[92];
                        }
                    }
                    if (num_tokens > 93) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (83328 + row_r + tid_1)) + (0)) = red[93];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (333312 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[93];
                        }
                    }
                    if (num_tokens > 94) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (84224 + row_r + tid_1)) + (0)) = red[94];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (336896 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[94];
                        }
                    }
                    if (num_tokens > 95) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (85120 + row_r + tid_1)) + (0)) = red[95];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (340480 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[95];
                        }
                    }
                    if (num_tokens > 96) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (86016 + row_r + tid_1)) + (0)) = red[96];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (344064 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[96];
                        }
                    }
                    if (num_tokens > 97) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (86912 + row_r + tid_1)) + (0)) = red[97];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (347648 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[97];
                        }
                    }
                    if (num_tokens > 98) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (87808 + row_r + tid_1)) + (0)) = red[98];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (351232 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[98];
                        }
                    }
                    if (num_tokens > 99) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (88704 + row_r + tid_1)) + (0)) = red[99];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (354816 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[99];
                        }
                    }
                    if (num_tokens > 100) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (89600 + row_r + tid_1)) + (0)) = red[100];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (358400 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[100];
                        }
                    }
                    if (num_tokens > 101) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (90496 + row_r + tid_1)) + (0)) = red[101];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (361984 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[101];
                        }
                    }
                    if (num_tokens > 102) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (91392 + row_r + tid_1)) + (0)) = red[102];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (365568 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[102];
                        }
                    }
                    if (num_tokens > 103) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (92288 + row_r + tid_1)) + (0)) = red[103];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (369152 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[103];
                        }
                    }
                    if (num_tokens > 104) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (93184 + row_r + tid_1)) + (0)) = red[104];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (372736 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[104];
                        }
                    }
                    if (num_tokens > 105) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (94080 + row_r + tid_1)) + (0)) = red[105];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (376320 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[105];
                        }
                    }
                    if (num_tokens > 106) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (94976 + row_r + tid_1)) + (0)) = red[106];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (379904 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[106];
                        }
                    }
                    if (num_tokens > 107) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (95872 + row_r + tid_1)) + (0)) = red[107];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (383488 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[107];
                        }
                    }
                    if (num_tokens > 108) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (96768 + row_r + tid_1)) + (0)) = red[108];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (387072 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[108];
                        }
                    }
                    if (num_tokens > 109) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (97664 + row_r + tid_1)) + (0)) = red[109];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (390656 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[109];
                        }
                    }
                    if (num_tokens > 110) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (98560 + row_r + tid_1)) + (0)) = red[110];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (394240 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[110];
                        }
                    }
                    if (num_tokens > 111) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (99456 + row_r + tid_1)) + (0)) = red[111];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (397824 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[111];
                        }
                    }
                    if (num_tokens > 112) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (100352 + row_r + tid_1)) + (0)) = red[112];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (401408 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[112];
                        }
                    }
                    if (num_tokens > 113) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (101248 + row_r + tid_1)) + (0)) = red[113];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (404992 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[113];
                        }
                    }
                    if (num_tokens > 114) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (102144 + row_r + tid_1)) + (0)) = red[114];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (408576 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[114];
                        }
                    }
                    if (num_tokens > 115) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (103040 + row_r + tid_1)) + (0)) = red[115];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (412160 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[115];
                        }
                    }
                    if (num_tokens > 116) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (103936 + row_r + tid_1)) + (0)) = red[116];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (415744 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[116];
                        }
                    }
                    if (num_tokens > 117) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (104832 + row_r + tid_1)) + (0)) = red[117];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (419328 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[117];
                        }
                    }
                    if (num_tokens > 118) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (105728 + row_r + tid_1)) + (0)) = red[118];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (422912 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[118];
                        }
                    }
                    if (num_tokens > 119) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (106624 + row_r + tid_1)) + (0)) = red[119];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (426496 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[119];
                        }
                    }
                    if (num_tokens > 120) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (107520 + row_r + tid_1)) + (0)) = red[120];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (430080 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[120];
                        }
                    }
                    if (num_tokens > 121) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (108416 + row_r + tid_1)) + (0)) = red[121];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (433664 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[121];
                        }
                    }
                    if (num_tokens > 122) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (109312 + row_r + tid_1)) + (0)) = red[122];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (437248 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[122];
                        }
                    }
                    if (num_tokens > 123) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (110208 + row_r + tid_1)) + (0)) = red[123];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (440832 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[123];
                        }
                    }
                    if (num_tokens > 124) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (111104 + row_r + tid_1)) + (0)) = red[124];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (444416 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[124];
                        }
                    }
                    if (num_tokens > 125) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (112000 + row_r + tid_1)) + (0)) = red[125];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (448000 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[125];
                        }
                    }
                    if (num_tokens > 126) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (112896 + row_r + tid_1)) + (0)) = red[126];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (451584 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[126];
                        }
                    }
                    if (num_tokens > 127) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (113792 + row_r + tid_1)) + (0)) = red[127];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (455168 + row_l + tid_1)) + (0)) = (__nv_bfloat16)red[127];
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int g_l = bid;
            int tl_l = g_l * 16;
            int tile_l = g_l;
            int rank_l = 0;
            int d_lo_l = ((rank_l == 0) ? 0 : 0);
            int d_cnt_l = ((rank_l == 0) ? 0 : 0);
            int u_lo_l = ((rank_l == 0) ? 0 : 112);
            int u_cnt_l = ((rank_l == 0) ? 112 : 0);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int pro = 0;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_R))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_L))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_S))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_2))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_1))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_2))) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = 0;
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 7) ? 0 : ((tile_1 < 35) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = (tile_1 - 7) * 128;
                    int row_s_1 = (tile_1 - 7 - 28) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    int kb1 = k1_off + m;
                    tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
                    if (need_a == 1) {
                        int kc1 = k1_off + m;
                        if (cls_1 == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384), "l"((&A_R)), "r"(0), "r"(row_r_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (cls_1 == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384), "l"((&A_L)), "r"(0), "r"(row_l_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (cls_1 == 2) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384), "l"((&A_S)), "r"(0), "r"(row_s_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + stage * 16384 + 8192), "l"((&A_S)), "r"(0), "r"(6144 + row_s_1), "r"(kc1),
                                   "r"(a_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 32768);
                    stage += 1;
                    if (stage == 7) { stage = 0; _phase_a_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int g_m = bid;
            int tl_m = g_m * 16;
            int tile_2 = g_m;
            int rank_m = 0;
            int u_count_m = ((rank_m == 0) ? 112 : 0);
            int d_cnt_m = ((rank_m == 0) ? 0 : 0);
            unsigned int stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 7) ? 0 : ((tile_2 < 35) ? 1 : 2));
                mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    int init_sub = init_flag;
                    if (cls_2 == 2) {
                        int _smem_desc_lo_0 = (((smem_ag_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (128))), "r"(((init_sub) ? 0 : 1)) : "memory");
                        int _smem_desc_lo_2 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc + (256))), "r"(((init_sub) ? 0 : 1)) : "memory");
                    } else {
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)) : "memory");
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 7) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
