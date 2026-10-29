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
#define SMEM_LAND_STAGE_BYTES 32768
#define SMEM_LAND_STRIDE 32768
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 2048
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 116736
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_BN_OFF 116736
#define SMEM_SMEM_BN_STAGE_BYTES 16384
#define SMEM_SMEM_BN_STRIDE 16384
#define SMEM_SMEM_AG_OFF 2048
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 10240
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 231424

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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_121e58fe80b2705b094f(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define free_bar_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_a_addr = smem + 2048;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 116736);
    const int smem_b_addr = smem + 116736;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 116736);
    const int smem_bn_addr = smem + 116736;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_ag_addr = smem + 2048;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_au_addr = smem + 10240;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

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
            // free_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 128, 32768);
            mbarrier_expect_tx(smem + 160, 8);
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 0) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
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
            tile = g_e / 2;
            slot = g_e % 2;
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
            unsigned long long zero_free = 0;
            if (slot == 0) {
                if (warp == 0) {
                    if (elect_sync()) {
                        uint32_t _cluster_mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_0) : "r"(free_word_addr), "r"(1));
                        uint32_t _mbar_peer_1;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_1) : "r"(free_bar_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_0), "l"(zero_free), "r"(_mbar_peer_1) : "memory");
                    }
                }
            }
            if (slot == 1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        uint32_t _cluster_mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_2) : "r"(free_word_addr), "r"(0));
                        uint32_t _mbar_peer_3;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_3) : "r"(free_bar_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_2), "l"(zero_free), "r"(_mbar_peer_3) : "memory");
                    }
                }
            }
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
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(free_bar_addr);
                }
            }
            mbarrier_wait_cluster_hint(free_bar_addr, 0, 10000000);
            if (slot == 0) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(1));
                if (cls == 2) {
                    uint32_t _mbar_peer_4;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_4) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0), "r"(__float_as_uint(red[32])), "r"(__float_as_uint(red[33])), "r"(__float_as_uint(red[34])), "r"(__float_as_uint(red[35])), "r"(_mbar_peer_4) : "memory");
                    uint32_t _mbar_peer_5;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_5) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[36])), "r"(__float_as_uint(red[37])), "r"(__float_as_uint(red[38])), "r"(__float_as_uint(red[39])), "r"(_mbar_peer_5) : "memory");
                    uint32_t _mbar_peer_6;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_6) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(red[40])), "r"(__float_as_uint(red[41])), "r"(__float_as_uint(red[42])), "r"(__float_as_uint(red[43])), "r"(_mbar_peer_6) : "memory");
                    uint32_t _mbar_peer_7;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_7) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(red[44])), "r"(__float_as_uint(red[45])), "r"(__float_as_uint(red[46])), "r"(__float_as_uint(red[47])), "r"(_mbar_peer_7) : "memory");
                    uint32_t _mbar_peer_8;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_8) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 8192), "r"(__float_as_uint(red[48])), "r"(__float_as_uint(red[49])), "r"(__float_as_uint(red[50])), "r"(__float_as_uint(red[51])), "r"(_mbar_peer_8) : "memory");
                    uint32_t _mbar_peer_9;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_9) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 10240), "r"(__float_as_uint(red[52])), "r"(__float_as_uint(red[53])), "r"(__float_as_uint(red[54])), "r"(__float_as_uint(red[55])), "r"(_mbar_peer_9) : "memory");
                    uint32_t _mbar_peer_10;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_10) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 12288), "r"(__float_as_uint(red[56])), "r"(__float_as_uint(red[57])), "r"(__float_as_uint(red[58])), "r"(__float_as_uint(red[59])), "r"(_mbar_peer_10) : "memory");
                    uint32_t _mbar_peer_11;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_11) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 14336), "r"(__float_as_uint(red[60])), "r"(__float_as_uint(red[61])), "r"(__float_as_uint(red[62])), "r"(__float_as_uint(red[63])), "r"(_mbar_peer_11) : "memory");
                    uint32_t _mbar_peer_12;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_12) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 16384), "r"(__float_as_uint(red[96])), "r"(__float_as_uint(red[97])), "r"(__float_as_uint(red[98])), "r"(__float_as_uint(red[99])), "r"(_mbar_peer_12) : "memory");
                    uint32_t _mbar_peer_13;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_13) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 18432), "r"(__float_as_uint(red[100])), "r"(__float_as_uint(red[101])), "r"(__float_as_uint(red[102])), "r"(__float_as_uint(red[103])), "r"(_mbar_peer_13) : "memory");
                    uint32_t _mbar_peer_14;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_14) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 20480), "r"(__float_as_uint(red[104])), "r"(__float_as_uint(red[105])), "r"(__float_as_uint(red[106])), "r"(__float_as_uint(red[107])), "r"(_mbar_peer_14) : "memory");
                    uint32_t _mbar_peer_15;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_15) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 22528), "r"(__float_as_uint(red[108])), "r"(__float_as_uint(red[109])), "r"(__float_as_uint(red[110])), "r"(__float_as_uint(red[111])), "r"(_mbar_peer_15) : "memory");
                    uint32_t _mbar_peer_16;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_16) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 24576), "r"(__float_as_uint(red[112])), "r"(__float_as_uint(red[113])), "r"(__float_as_uint(red[114])), "r"(__float_as_uint(red[115])), "r"(_mbar_peer_16) : "memory");
                    uint32_t _mbar_peer_17;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_17) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 26624), "r"(__float_as_uint(red[116])), "r"(__float_as_uint(red[117])), "r"(__float_as_uint(red[118])), "r"(__float_as_uint(red[119])), "r"(_mbar_peer_17) : "memory");
                    uint32_t _mbar_peer_18;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_18) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 28672), "r"(__float_as_uint(red[120])), "r"(__float_as_uint(red[121])), "r"(__float_as_uint(red[122])), "r"(__float_as_uint(red[123])), "r"(_mbar_peer_18) : "memory");
                    uint32_t _mbar_peer_19;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_19) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 30720), "r"(__float_as_uint(red[124])), "r"(__float_as_uint(red[125])), "r"(__float_as_uint(red[126])), "r"(__float_as_uint(red[127])), "r"(_mbar_peer_19) : "memory");
                } else {
                    uint32_t _mbar_peer_20;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_20) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0), "r"(__float_as_uint(red[64])), "r"(__float_as_uint(red[65])), "r"(__float_as_uint(red[66])), "r"(__float_as_uint(red[67])), "r"(_mbar_peer_20) : "memory");
                    uint32_t _mbar_peer_21;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_21) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[68])), "r"(__float_as_uint(red[69])), "r"(__float_as_uint(red[70])), "r"(__float_as_uint(red[71])), "r"(_mbar_peer_21) : "memory");
                    uint32_t _mbar_peer_22;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_22) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(red[72])), "r"(__float_as_uint(red[73])), "r"(__float_as_uint(red[74])), "r"(__float_as_uint(red[75])), "r"(_mbar_peer_22) : "memory");
                    uint32_t _mbar_peer_23;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_23) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(red[76])), "r"(__float_as_uint(red[77])), "r"(__float_as_uint(red[78])), "r"(__float_as_uint(red[79])), "r"(_mbar_peer_23) : "memory");
                    uint32_t _mbar_peer_24;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_24) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 8192), "r"(__float_as_uint(red[80])), "r"(__float_as_uint(red[81])), "r"(__float_as_uint(red[82])), "r"(__float_as_uint(red[83])), "r"(_mbar_peer_24) : "memory");
                    uint32_t _mbar_peer_25;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_25) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 10240), "r"(__float_as_uint(red[84])), "r"(__float_as_uint(red[85])), "r"(__float_as_uint(red[86])), "r"(__float_as_uint(red[87])), "r"(_mbar_peer_25) : "memory");
                    uint32_t _mbar_peer_26;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_26) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 12288), "r"(__float_as_uint(red[88])), "r"(__float_as_uint(red[89])), "r"(__float_as_uint(red[90])), "r"(__float_as_uint(red[91])), "r"(_mbar_peer_26) : "memory");
                    uint32_t _mbar_peer_27;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_27) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 14336), "r"(__float_as_uint(red[92])), "r"(__float_as_uint(red[93])), "r"(__float_as_uint(red[94])), "r"(__float_as_uint(red[95])), "r"(_mbar_peer_27) : "memory");
                    uint32_t _mbar_peer_28;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_28) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 16384), "r"(__float_as_uint(red[96])), "r"(__float_as_uint(red[97])), "r"(__float_as_uint(red[98])), "r"(__float_as_uint(red[99])), "r"(_mbar_peer_28) : "memory");
                    uint32_t _mbar_peer_29;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_29) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 18432), "r"(__float_as_uint(red[100])), "r"(__float_as_uint(red[101])), "r"(__float_as_uint(red[102])), "r"(__float_as_uint(red[103])), "r"(_mbar_peer_29) : "memory");
                    uint32_t _mbar_peer_30;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_30) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 20480), "r"(__float_as_uint(red[104])), "r"(__float_as_uint(red[105])), "r"(__float_as_uint(red[106])), "r"(__float_as_uint(red[107])), "r"(_mbar_peer_30) : "memory");
                    uint32_t _mbar_peer_31;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_31) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 22528), "r"(__float_as_uint(red[108])), "r"(__float_as_uint(red[109])), "r"(__float_as_uint(red[110])), "r"(__float_as_uint(red[111])), "r"(_mbar_peer_31) : "memory");
                    uint32_t _mbar_peer_32;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_32) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 24576), "r"(__float_as_uint(red[112])), "r"(__float_as_uint(red[113])), "r"(__float_as_uint(red[114])), "r"(__float_as_uint(red[115])), "r"(_mbar_peer_32) : "memory");
                    uint32_t _mbar_peer_33;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_33) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 26624), "r"(__float_as_uint(red[116])), "r"(__float_as_uint(red[117])), "r"(__float_as_uint(red[118])), "r"(__float_as_uint(red[119])), "r"(_mbar_peer_33) : "memory");
                    uint32_t _mbar_peer_34;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_34) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 28672), "r"(__float_as_uint(red[120])), "r"(__float_as_uint(red[121])), "r"(__float_as_uint(red[122])), "r"(__float_as_uint(red[123])), "r"(_mbar_peer_34) : "memory");
                    uint32_t _mbar_peer_35;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_35) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 30720), "r"(__float_as_uint(red[124])), "r"(__float_as_uint(red[125])), "r"(__float_as_uint(red[126])), "r"(__float_as_uint(red[127])), "r"(_mbar_peer_35) : "memory");
                }
            }
            if (slot == 1) {
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                if (cls == 2) {
                    uint32_t _mbar_peer_36;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_36) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_36) : "memory");
                    uint32_t _mbar_peer_37;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_37) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_37) : "memory");
                    uint32_t _mbar_peer_38;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_38) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mbar_peer_38) : "memory");
                    uint32_t _mbar_peer_39;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_39) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mbar_peer_39) : "memory");
                    uint32_t _mbar_peer_40;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_40) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 8192), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mbar_peer_40) : "memory");
                    uint32_t _mbar_peer_41;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_41) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 10240), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mbar_peer_41) : "memory");
                    uint32_t _mbar_peer_42;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_42) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 12288), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mbar_peer_42) : "memory");
                    uint32_t _mbar_peer_43;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_43) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 14336), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mbar_peer_43) : "memory");
                    uint32_t _mbar_peer_44;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_44) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 16384), "r"(__float_as_uint(red[64])), "r"(__float_as_uint(red[65])), "r"(__float_as_uint(red[66])), "r"(__float_as_uint(red[67])), "r"(_mbar_peer_44) : "memory");
                    uint32_t _mbar_peer_45;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_45) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 18432), "r"(__float_as_uint(red[68])), "r"(__float_as_uint(red[69])), "r"(__float_as_uint(red[70])), "r"(__float_as_uint(red[71])), "r"(_mbar_peer_45) : "memory");
                    uint32_t _mbar_peer_46;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_46) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 20480), "r"(__float_as_uint(red[72])), "r"(__float_as_uint(red[73])), "r"(__float_as_uint(red[74])), "r"(__float_as_uint(red[75])), "r"(_mbar_peer_46) : "memory");
                    uint32_t _mbar_peer_47;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_47) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 22528), "r"(__float_as_uint(red[76])), "r"(__float_as_uint(red[77])), "r"(__float_as_uint(red[78])), "r"(__float_as_uint(red[79])), "r"(_mbar_peer_47) : "memory");
                    uint32_t _mbar_peer_48;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_48) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 24576), "r"(__float_as_uint(red[80])), "r"(__float_as_uint(red[81])), "r"(__float_as_uint(red[82])), "r"(__float_as_uint(red[83])), "r"(_mbar_peer_48) : "memory");
                    uint32_t _mbar_peer_49;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_49) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 26624), "r"(__float_as_uint(red[84])), "r"(__float_as_uint(red[85])), "r"(__float_as_uint(red[86])), "r"(__float_as_uint(red[87])), "r"(_mbar_peer_49) : "memory");
                    uint32_t _mbar_peer_50;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_50) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 28672), "r"(__float_as_uint(red[88])), "r"(__float_as_uint(red[89])), "r"(__float_as_uint(red[90])), "r"(__float_as_uint(red[91])), "r"(_mbar_peer_50) : "memory");
                    uint32_t _mbar_peer_51;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_51) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 30720), "r"(__float_as_uint(red[92])), "r"(__float_as_uint(red[93])), "r"(__float_as_uint(red[94])), "r"(__float_as_uint(red[95])), "r"(_mbar_peer_51) : "memory");
                } else {
                    uint32_t _mbar_peer_52;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_52) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_52) : "memory");
                    uint32_t _mbar_peer_53;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_53) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_53) : "memory");
                    uint32_t _mbar_peer_54;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_54) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mbar_peer_54) : "memory");
                    uint32_t _mbar_peer_55;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_55) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mbar_peer_55) : "memory");
                    uint32_t _mbar_peer_56;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_56) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 8192), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mbar_peer_56) : "memory");
                    uint32_t _mbar_peer_57;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_57) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 10240), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mbar_peer_57) : "memory");
                    uint32_t _mbar_peer_58;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_58) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 12288), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mbar_peer_58) : "memory");
                    uint32_t _mbar_peer_59;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_59) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 14336), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mbar_peer_59) : "memory");
                    uint32_t _mbar_peer_60;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_60) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 16384), "r"(__float_as_uint(red[32])), "r"(__float_as_uint(red[33])), "r"(__float_as_uint(red[34])), "r"(__float_as_uint(red[35])), "r"(_mbar_peer_60) : "memory");
                    uint32_t _mbar_peer_61;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_61) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 18432), "r"(__float_as_uint(red[36])), "r"(__float_as_uint(red[37])), "r"(__float_as_uint(red[38])), "r"(__float_as_uint(red[39])), "r"(_mbar_peer_61) : "memory");
                    uint32_t _mbar_peer_62;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_62) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 20480), "r"(__float_as_uint(red[40])), "r"(__float_as_uint(red[41])), "r"(__float_as_uint(red[42])), "r"(__float_as_uint(red[43])), "r"(_mbar_peer_62) : "memory");
                    uint32_t _mbar_peer_63;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_63) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 22528), "r"(__float_as_uint(red[44])), "r"(__float_as_uint(red[45])), "r"(__float_as_uint(red[46])), "r"(__float_as_uint(red[47])), "r"(_mbar_peer_63) : "memory");
                    uint32_t _mbar_peer_64;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_64) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 24576), "r"(__float_as_uint(red[48])), "r"(__float_as_uint(red[49])), "r"(__float_as_uint(red[50])), "r"(__float_as_uint(red[51])), "r"(_mbar_peer_64) : "memory");
                    uint32_t _mbar_peer_65;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_65) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 26624), "r"(__float_as_uint(red[52])), "r"(__float_as_uint(red[53])), "r"(__float_as_uint(red[54])), "r"(__float_as_uint(red[55])), "r"(_mbar_peer_65) : "memory");
                    uint32_t _mbar_peer_66;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_66) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 28672), "r"(__float_as_uint(red[56])), "r"(__float_as_uint(red[57])), "r"(__float_as_uint(red[58])), "r"(__float_as_uint(red[59])), "r"(_mbar_peer_66) : "memory");
                    uint32_t _mbar_peer_67;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_67) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 30720), "r"(__float_as_uint(red[60])), "r"(__float_as_uint(red[61])), "r"(__float_as_uint(red[62])), "r"(__float_as_uint(red[63])), "r"(_mbar_peer_67) : "memory");
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(red_bar_addr);
                }
            }
            mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
            if (slot == 0) {
                if (cls == 2) {
                    red[0] = red[0] + land[tid_1 * 4];
                    red[1] = red[1] + land[tid_1 * 4 + 1];
                    red[2] = red[2] + land[tid_1 * 4 + 2];
                    red[3] = red[3] + land[tid_1 * 4 + 3];
                    red[4] = red[4] + land[(128 + tid_1) * 4];
                    red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                    red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                    red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
                    red[8] = red[8] + land[(256 + tid_1) * 4];
                    red[9] = red[9] + land[(256 + tid_1) * 4 + 1];
                    red[10] = red[10] + land[(256 + tid_1) * 4 + 2];
                    red[11] = red[11] + land[(256 + tid_1) * 4 + 3];
                    red[12] = red[12] + land[(384 + tid_1) * 4];
                    red[13] = red[13] + land[(384 + tid_1) * 4 + 1];
                    red[14] = red[14] + land[(384 + tid_1) * 4 + 2];
                    red[15] = red[15] + land[(384 + tid_1) * 4 + 3];
                    red[16] = red[16] + land[(512 + tid_1) * 4];
                    red[17] = red[17] + land[(512 + tid_1) * 4 + 1];
                    red[18] = red[18] + land[(512 + tid_1) * 4 + 2];
                    red[19] = red[19] + land[(512 + tid_1) * 4 + 3];
                    red[20] = red[20] + land[(640 + tid_1) * 4];
                    red[21] = red[21] + land[(640 + tid_1) * 4 + 1];
                    red[22] = red[22] + land[(640 + tid_1) * 4 + 2];
                    red[23] = red[23] + land[(640 + tid_1) * 4 + 3];
                    red[24] = red[24] + land[(768 + tid_1) * 4];
                    red[25] = red[25] + land[(768 + tid_1) * 4 + 1];
                    red[26] = red[26] + land[(768 + tid_1) * 4 + 2];
                    red[27] = red[27] + land[(768 + tid_1) * 4 + 3];
                    red[28] = red[28] + land[(896 + tid_1) * 4];
                    red[29] = red[29] + land[(896 + tid_1) * 4 + 1];
                    red[30] = red[30] + land[(896 + tid_1) * 4 + 2];
                    red[31] = red[31] + land[(896 + tid_1) * 4 + 3];
                    red[64] = red[64] + land[(1024 + tid_1) * 4];
                    red[65] = red[65] + land[(1024 + tid_1) * 4 + 1];
                    red[66] = red[66] + land[(1024 + tid_1) * 4 + 2];
                    red[67] = red[67] + land[(1024 + tid_1) * 4 + 3];
                    red[68] = red[68] + land[(1152 + tid_1) * 4];
                    red[69] = red[69] + land[(1152 + tid_1) * 4 + 1];
                    red[70] = red[70] + land[(1152 + tid_1) * 4 + 2];
                    red[71] = red[71] + land[(1152 + tid_1) * 4 + 3];
                    red[72] = red[72] + land[(1280 + tid_1) * 4];
                    red[73] = red[73] + land[(1280 + tid_1) * 4 + 1];
                    red[74] = red[74] + land[(1280 + tid_1) * 4 + 2];
                    red[75] = red[75] + land[(1280 + tid_1) * 4 + 3];
                    red[76] = red[76] + land[(1408 + tid_1) * 4];
                    red[77] = red[77] + land[(1408 + tid_1) * 4 + 1];
                    red[78] = red[78] + land[(1408 + tid_1) * 4 + 2];
                    red[79] = red[79] + land[(1408 + tid_1) * 4 + 3];
                    red[80] = red[80] + land[(1536 + tid_1) * 4];
                    red[81] = red[81] + land[(1536 + tid_1) * 4 + 1];
                    red[82] = red[82] + land[(1536 + tid_1) * 4 + 2];
                    red[83] = red[83] + land[(1536 + tid_1) * 4 + 3];
                    red[84] = red[84] + land[(1664 + tid_1) * 4];
                    red[85] = red[85] + land[(1664 + tid_1) * 4 + 1];
                    red[86] = red[86] + land[(1664 + tid_1) * 4 + 2];
                    red[87] = red[87] + land[(1664 + tid_1) * 4 + 3];
                    red[88] = red[88] + land[(1792 + tid_1) * 4];
                    red[89] = red[89] + land[(1792 + tid_1) * 4 + 1];
                    red[90] = red[90] + land[(1792 + tid_1) * 4 + 2];
                    red[91] = red[91] + land[(1792 + tid_1) * 4 + 3];
                    red[92] = red[92] + land[(1920 + tid_1) * 4];
                    red[93] = red[93] + land[(1920 + tid_1) * 4 + 1];
                    red[94] = red[94] + land[(1920 + tid_1) * 4 + 2];
                    red[95] = red[95] + land[(1920 + tid_1) * 4 + 3];
                } else {
                    red[0] = red[0] + land[tid_1 * 4];
                    red[1] = red[1] + land[tid_1 * 4 + 1];
                    red[2] = red[2] + land[tid_1 * 4 + 2];
                    red[3] = red[3] + land[tid_1 * 4 + 3];
                    red[4] = red[4] + land[(128 + tid_1) * 4];
                    red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                    red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                    red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
                    red[8] = red[8] + land[(256 + tid_1) * 4];
                    red[9] = red[9] + land[(256 + tid_1) * 4 + 1];
                    red[10] = red[10] + land[(256 + tid_1) * 4 + 2];
                    red[11] = red[11] + land[(256 + tid_1) * 4 + 3];
                    red[12] = red[12] + land[(384 + tid_1) * 4];
                    red[13] = red[13] + land[(384 + tid_1) * 4 + 1];
                    red[14] = red[14] + land[(384 + tid_1) * 4 + 2];
                    red[15] = red[15] + land[(384 + tid_1) * 4 + 3];
                    red[16] = red[16] + land[(512 + tid_1) * 4];
                    red[17] = red[17] + land[(512 + tid_1) * 4 + 1];
                    red[18] = red[18] + land[(512 + tid_1) * 4 + 2];
                    red[19] = red[19] + land[(512 + tid_1) * 4 + 3];
                    red[20] = red[20] + land[(640 + tid_1) * 4];
                    red[21] = red[21] + land[(640 + tid_1) * 4 + 1];
                    red[22] = red[22] + land[(640 + tid_1) * 4 + 2];
                    red[23] = red[23] + land[(640 + tid_1) * 4 + 3];
                    red[24] = red[24] + land[(768 + tid_1) * 4];
                    red[25] = red[25] + land[(768 + tid_1) * 4 + 1];
                    red[26] = red[26] + land[(768 + tid_1) * 4 + 2];
                    red[27] = red[27] + land[(768 + tid_1) * 4 + 3];
                    red[28] = red[28] + land[(896 + tid_1) * 4];
                    red[29] = red[29] + land[(896 + tid_1) * 4 + 1];
                    red[30] = red[30] + land[(896 + tid_1) * 4 + 2];
                    red[31] = red[31] + land[(896 + tid_1) * 4 + 3];
                    red[32] = red[32] + land[(1024 + tid_1) * 4];
                    red[33] = red[33] + land[(1024 + tid_1) * 4 + 1];
                    red[34] = red[34] + land[(1024 + tid_1) * 4 + 2];
                    red[35] = red[35] + land[(1024 + tid_1) * 4 + 3];
                    red[36] = red[36] + land[(1152 + tid_1) * 4];
                    red[37] = red[37] + land[(1152 + tid_1) * 4 + 1];
                    red[38] = red[38] + land[(1152 + tid_1) * 4 + 2];
                    red[39] = red[39] + land[(1152 + tid_1) * 4 + 3];
                    red[40] = red[40] + land[(1280 + tid_1) * 4];
                    red[41] = red[41] + land[(1280 + tid_1) * 4 + 1];
                    red[42] = red[42] + land[(1280 + tid_1) * 4 + 2];
                    red[43] = red[43] + land[(1280 + tid_1) * 4 + 3];
                    red[44] = red[44] + land[(1408 + tid_1) * 4];
                    red[45] = red[45] + land[(1408 + tid_1) * 4 + 1];
                    red[46] = red[46] + land[(1408 + tid_1) * 4 + 2];
                    red[47] = red[47] + land[(1408 + tid_1) * 4 + 3];
                    red[48] = red[48] + land[(1536 + tid_1) * 4];
                    red[49] = red[49] + land[(1536 + tid_1) * 4 + 1];
                    red[50] = red[50] + land[(1536 + tid_1) * 4 + 2];
                    red[51] = red[51] + land[(1536 + tid_1) * 4 + 3];
                    red[52] = red[52] + land[(1664 + tid_1) * 4];
                    red[53] = red[53] + land[(1664 + tid_1) * 4 + 1];
                    red[54] = red[54] + land[(1664 + tid_1) * 4 + 2];
                    red[55] = red[55] + land[(1664 + tid_1) * 4 + 3];
                    red[56] = red[56] + land[(1792 + tid_1) * 4];
                    red[57] = red[57] + land[(1792 + tid_1) * 4 + 1];
                    red[58] = red[58] + land[(1792 + tid_1) * 4 + 2];
                    red[59] = red[59] + land[(1792 + tid_1) * 4 + 3];
                    red[60] = red[60] + land[(1920 + tid_1) * 4];
                    red[61] = red[61] + land[(1920 + tid_1) * 4 + 1];
                    red[62] = red[62] + land[(1920 + tid_1) * 4 + 2];
                    red[63] = red[63] + land[(1920 + tid_1) * 4 + 3];
                }
            }
            if (slot == 1) {
                if (cls == 2) {
                    red[32] = red[32] + land[tid_1 * 4];
                    red[33] = red[33] + land[tid_1 * 4 + 1];
                    red[34] = red[34] + land[tid_1 * 4 + 2];
                    red[35] = red[35] + land[tid_1 * 4 + 3];
                    red[36] = red[36] + land[(128 + tid_1) * 4];
                    red[37] = red[37] + land[(128 + tid_1) * 4 + 1];
                    red[38] = red[38] + land[(128 + tid_1) * 4 + 2];
                    red[39] = red[39] + land[(128 + tid_1) * 4 + 3];
                    red[40] = red[40] + land[(256 + tid_1) * 4];
                    red[41] = red[41] + land[(256 + tid_1) * 4 + 1];
                    red[42] = red[42] + land[(256 + tid_1) * 4 + 2];
                    red[43] = red[43] + land[(256 + tid_1) * 4 + 3];
                    red[44] = red[44] + land[(384 + tid_1) * 4];
                    red[45] = red[45] + land[(384 + tid_1) * 4 + 1];
                    red[46] = red[46] + land[(384 + tid_1) * 4 + 2];
                    red[47] = red[47] + land[(384 + tid_1) * 4 + 3];
                    red[48] = red[48] + land[(512 + tid_1) * 4];
                    red[49] = red[49] + land[(512 + tid_1) * 4 + 1];
                    red[50] = red[50] + land[(512 + tid_1) * 4 + 2];
                    red[51] = red[51] + land[(512 + tid_1) * 4 + 3];
                    red[52] = red[52] + land[(640 + tid_1) * 4];
                    red[53] = red[53] + land[(640 + tid_1) * 4 + 1];
                    red[54] = red[54] + land[(640 + tid_1) * 4 + 2];
                    red[55] = red[55] + land[(640 + tid_1) * 4 + 3];
                    red[56] = red[56] + land[(768 + tid_1) * 4];
                    red[57] = red[57] + land[(768 + tid_1) * 4 + 1];
                    red[58] = red[58] + land[(768 + tid_1) * 4 + 2];
                    red[59] = red[59] + land[(768 + tid_1) * 4 + 3];
                    red[60] = red[60] + land[(896 + tid_1) * 4];
                    red[61] = red[61] + land[(896 + tid_1) * 4 + 1];
                    red[62] = red[62] + land[(896 + tid_1) * 4 + 2];
                    red[63] = red[63] + land[(896 + tid_1) * 4 + 3];
                    red[96] = red[96] + land[(1024 + tid_1) * 4];
                    red[97] = red[97] + land[(1024 + tid_1) * 4 + 1];
                    red[98] = red[98] + land[(1024 + tid_1) * 4 + 2];
                    red[99] = red[99] + land[(1024 + tid_1) * 4 + 3];
                    red[100] = red[100] + land[(1152 + tid_1) * 4];
                    red[101] = red[101] + land[(1152 + tid_1) * 4 + 1];
                    red[102] = red[102] + land[(1152 + tid_1) * 4 + 2];
                    red[103] = red[103] + land[(1152 + tid_1) * 4 + 3];
                    red[104] = red[104] + land[(1280 + tid_1) * 4];
                    red[105] = red[105] + land[(1280 + tid_1) * 4 + 1];
                    red[106] = red[106] + land[(1280 + tid_1) * 4 + 2];
                    red[107] = red[107] + land[(1280 + tid_1) * 4 + 3];
                    red[108] = red[108] + land[(1408 + tid_1) * 4];
                    red[109] = red[109] + land[(1408 + tid_1) * 4 + 1];
                    red[110] = red[110] + land[(1408 + tid_1) * 4 + 2];
                    red[111] = red[111] + land[(1408 + tid_1) * 4 + 3];
                    red[112] = red[112] + land[(1536 + tid_1) * 4];
                    red[113] = red[113] + land[(1536 + tid_1) * 4 + 1];
                    red[114] = red[114] + land[(1536 + tid_1) * 4 + 2];
                    red[115] = red[115] + land[(1536 + tid_1) * 4 + 3];
                    red[116] = red[116] + land[(1664 + tid_1) * 4];
                    red[117] = red[117] + land[(1664 + tid_1) * 4 + 1];
                    red[118] = red[118] + land[(1664 + tid_1) * 4 + 2];
                    red[119] = red[119] + land[(1664 + tid_1) * 4 + 3];
                    red[120] = red[120] + land[(1792 + tid_1) * 4];
                    red[121] = red[121] + land[(1792 + tid_1) * 4 + 1];
                    red[122] = red[122] + land[(1792 + tid_1) * 4 + 2];
                    red[123] = red[123] + land[(1792 + tid_1) * 4 + 3];
                    red[124] = red[124] + land[(1920 + tid_1) * 4];
                    red[125] = red[125] + land[(1920 + tid_1) * 4 + 1];
                    red[126] = red[126] + land[(1920 + tid_1) * 4 + 2];
                    red[127] = red[127] + land[(1920 + tid_1) * 4 + 3];
                } else {
                    red[64] = red[64] + land[tid_1 * 4];
                    red[65] = red[65] + land[tid_1 * 4 + 1];
                    red[66] = red[66] + land[tid_1 * 4 + 2];
                    red[67] = red[67] + land[tid_1 * 4 + 3];
                    red[68] = red[68] + land[(128 + tid_1) * 4];
                    red[69] = red[69] + land[(128 + tid_1) * 4 + 1];
                    red[70] = red[70] + land[(128 + tid_1) * 4 + 2];
                    red[71] = red[71] + land[(128 + tid_1) * 4 + 3];
                    red[72] = red[72] + land[(256 + tid_1) * 4];
                    red[73] = red[73] + land[(256 + tid_1) * 4 + 1];
                    red[74] = red[74] + land[(256 + tid_1) * 4 + 2];
                    red[75] = red[75] + land[(256 + tid_1) * 4 + 3];
                    red[76] = red[76] + land[(384 + tid_1) * 4];
                    red[77] = red[77] + land[(384 + tid_1) * 4 + 1];
                    red[78] = red[78] + land[(384 + tid_1) * 4 + 2];
                    red[79] = red[79] + land[(384 + tid_1) * 4 + 3];
                    red[80] = red[80] + land[(512 + tid_1) * 4];
                    red[81] = red[81] + land[(512 + tid_1) * 4 + 1];
                    red[82] = red[82] + land[(512 + tid_1) * 4 + 2];
                    red[83] = red[83] + land[(512 + tid_1) * 4 + 3];
                    red[84] = red[84] + land[(640 + tid_1) * 4];
                    red[85] = red[85] + land[(640 + tid_1) * 4 + 1];
                    red[86] = red[86] + land[(640 + tid_1) * 4 + 2];
                    red[87] = red[87] + land[(640 + tid_1) * 4 + 3];
                    red[88] = red[88] + land[(768 + tid_1) * 4];
                    red[89] = red[89] + land[(768 + tid_1) * 4 + 1];
                    red[90] = red[90] + land[(768 + tid_1) * 4 + 2];
                    red[91] = red[91] + land[(768 + tid_1) * 4 + 3];
                    red[92] = red[92] + land[(896 + tid_1) * 4];
                    red[93] = red[93] + land[(896 + tid_1) * 4 + 1];
                    red[94] = red[94] + land[(896 + tid_1) * 4 + 2];
                    red[95] = red[95] + land[(896 + tid_1) * 4 + 3];
                    red[96] = red[96] + land[(1024 + tid_1) * 4];
                    red[97] = red[97] + land[(1024 + tid_1) * 4 + 1];
                    red[98] = red[98] + land[(1024 + tid_1) * 4 + 2];
                    red[99] = red[99] + land[(1024 + tid_1) * 4 + 3];
                    red[100] = red[100] + land[(1152 + tid_1) * 4];
                    red[101] = red[101] + land[(1152 + tid_1) * 4 + 1];
                    red[102] = red[102] + land[(1152 + tid_1) * 4 + 2];
                    red[103] = red[103] + land[(1152 + tid_1) * 4 + 3];
                    red[104] = red[104] + land[(1280 + tid_1) * 4];
                    red[105] = red[105] + land[(1280 + tid_1) * 4 + 1];
                    red[106] = red[106] + land[(1280 + tid_1) * 4 + 2];
                    red[107] = red[107] + land[(1280 + tid_1) * 4 + 3];
                    red[108] = red[108] + land[(1408 + tid_1) * 4];
                    red[109] = red[109] + land[(1408 + tid_1) * 4 + 1];
                    red[110] = red[110] + land[(1408 + tid_1) * 4 + 2];
                    red[111] = red[111] + land[(1408 + tid_1) * 4 + 3];
                    red[112] = red[112] + land[(1536 + tid_1) * 4];
                    red[113] = red[113] + land[(1536 + tid_1) * 4 + 1];
                    red[114] = red[114] + land[(1536 + tid_1) * 4 + 2];
                    red[115] = red[115] + land[(1536 + tid_1) * 4 + 3];
                    red[116] = red[116] + land[(1664 + tid_1) * 4];
                    red[117] = red[117] + land[(1664 + tid_1) * 4 + 1];
                    red[118] = red[118] + land[(1664 + tid_1) * 4 + 2];
                    red[119] = red[119] + land[(1664 + tid_1) * 4 + 3];
                    red[120] = red[120] + land[(1792 + tid_1) * 4];
                    red[121] = red[121] + land[(1792 + tid_1) * 4 + 1];
                    red[122] = red[122] + land[(1792 + tid_1) * 4 + 2];
                    red[123] = red[123] + land[(1792 + tid_1) * 4 + 3];
                    red[124] = red[124] + land[(1920 + tid_1) * 4];
                    red[125] = red[125] + land[(1920 + tid_1) * 4 + 1];
                    red[126] = red[126] + land[(1920 + tid_1) * 4 + 2];
                    red[127] = red[127] + land[(1920 + tid_1) * 4 + 3];
                }
            }
            if (fin == 1) {
                if (slot == 0) {
                    if (cls == 2) {
                        int sf_tok_1_0 = lane_pair * 2;
                        int sf_frow_2_0 = row_base;
                        if (sf_tok_1_0 < num_tokens) {
                            float sf_gk_1_0 = (float)(__nv_bfloat16)red[0];
                            float sf_uk_1_0 = (float)(__nv_bfloat16)red[64];
                            float _exp2_68 = approx_exp2((-sf_gk_1_0) * 1.442695f);
                            float sf_sig_1_0 = approx_rcp(1.0f + _exp2_68);
                            float _tanh_69 = tanhf(sf_gk_1_0 * 0.25f);
                            float sf_aa_1_0 = 4.0f * _tanh_69 * sf_sig_1_0;
                            float _tanh_70 = tanhf(sf_uk_1_0 * 0.04f);
                            float sf_bb_1_0 = 25.0f * _tanh_70;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_0 * 768 + row_s + sf_frow_2_0)) + (0)) = (__nv_bfloat16)(sf_aa_1_0 * sf_bb_1_0);
                        }
                        int sf_tok_1_1 = lane_pair * 2 + 1;
                        int sf_frow_2_1 = row_base;
                        if (sf_tok_1_1 < num_tokens) {
                            float sf_gk_1_1 = (float)(__nv_bfloat16)red[1];
                            float sf_uk_1_1 = (float)(__nv_bfloat16)red[65];
                            float _exp2_71 = approx_exp2((-sf_gk_1_1) * 1.442695f);
                            float sf_sig_1_1 = approx_rcp(1.0f + _exp2_71);
                            float _tanh_72 = tanhf(sf_gk_1_1 * 0.25f);
                            float sf_aa_1_1 = 4.0f * _tanh_72 * sf_sig_1_1;
                            float _tanh_73 = tanhf(sf_uk_1_1 * 0.04f);
                            float sf_bb_1_1 = 25.0f * _tanh_73;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_1 * 768 + row_s + sf_frow_2_1)) + (0)) = (__nv_bfloat16)(sf_aa_1_1 * sf_bb_1_1);
                        }
                        int sf_tok_1_2 = lane_pair * 2;
                        int sf_frow_2_2 = row_base + 8;
                        if (sf_tok_1_2 < num_tokens) {
                            float sf_gk_1_2 = (float)(__nv_bfloat16)red[2];
                            float sf_uk_1_2 = (float)(__nv_bfloat16)red[66];
                            float _exp2_74 = approx_exp2((-sf_gk_1_2) * 1.442695f);
                            float sf_sig_1_2 = approx_rcp(1.0f + _exp2_74);
                            float _tanh_75 = tanhf(sf_gk_1_2 * 0.25f);
                            float sf_aa_1_2 = 4.0f * _tanh_75 * sf_sig_1_2;
                            float _tanh_76 = tanhf(sf_uk_1_2 * 0.04f);
                            float sf_bb_1_2 = 25.0f * _tanh_76;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_2 * 768 + row_s + sf_frow_2_2)) + (0)) = (__nv_bfloat16)(sf_aa_1_2 * sf_bb_1_2);
                        }
                        int sf_tok_1_3 = lane_pair * 2 + 1;
                        int sf_frow_2_3 = row_base + 8;
                        if (sf_tok_1_3 < num_tokens) {
                            float sf_gk_1_3 = (float)(__nv_bfloat16)red[3];
                            float sf_uk_1_3 = (float)(__nv_bfloat16)red[67];
                            float _exp2_77 = approx_exp2((-sf_gk_1_3) * 1.442695f);
                            float sf_sig_1_3 = approx_rcp(1.0f + _exp2_77);
                            float _tanh_78 = tanhf(sf_gk_1_3 * 0.25f);
                            float sf_aa_1_3 = 4.0f * _tanh_78 * sf_sig_1_3;
                            float _tanh_79 = tanhf(sf_uk_1_3 * 0.04f);
                            float sf_bb_1_3 = 25.0f * _tanh_79;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_3 * 768 + row_s + sf_frow_2_3)) + (0)) = (__nv_bfloat16)(sf_aa_1_3 * sf_bb_1_3);
                        }
                        int sf_tok_1_4 = 8 + lane_pair * 2;
                        int sf_frow_2_4 = row_base;
                        if (sf_tok_1_4 < num_tokens) {
                            float sf_gk_1_4 = (float)(__nv_bfloat16)red[4];
                            float sf_uk_1_4 = (float)(__nv_bfloat16)red[68];
                            float _exp2_80 = approx_exp2((-sf_gk_1_4) * 1.442695f);
                            float sf_sig_1_4 = approx_rcp(1.0f + _exp2_80);
                            float _tanh_81 = tanhf(sf_gk_1_4 * 0.25f);
                            float sf_aa_1_4 = 4.0f * _tanh_81 * sf_sig_1_4;
                            float _tanh_82 = tanhf(sf_uk_1_4 * 0.04f);
                            float sf_bb_1_4 = 25.0f * _tanh_82;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_4 * 768 + row_s + sf_frow_2_4)) + (0)) = (__nv_bfloat16)(sf_aa_1_4 * sf_bb_1_4);
                        }
                        int sf_tok_1_5 = 8 + lane_pair * 2 + 1;
                        int sf_frow_2_5 = row_base;
                        if (sf_tok_1_5 < num_tokens) {
                            float sf_gk_1_5 = (float)(__nv_bfloat16)red[5];
                            float sf_uk_1_5 = (float)(__nv_bfloat16)red[69];
                            float _exp2_83 = approx_exp2((-sf_gk_1_5) * 1.442695f);
                            float sf_sig_1_5 = approx_rcp(1.0f + _exp2_83);
                            float _tanh_84 = tanhf(sf_gk_1_5 * 0.25f);
                            float sf_aa_1_5 = 4.0f * _tanh_84 * sf_sig_1_5;
                            float _tanh_85 = tanhf(sf_uk_1_5 * 0.04f);
                            float sf_bb_1_5 = 25.0f * _tanh_85;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_5 * 768 + row_s + sf_frow_2_5)) + (0)) = (__nv_bfloat16)(sf_aa_1_5 * sf_bb_1_5);
                        }
                        int sf_tok_1_6 = 8 + lane_pair * 2;
                        int sf_frow_2_6 = row_base + 8;
                        if (sf_tok_1_6 < num_tokens) {
                            float sf_gk_1_6 = (float)(__nv_bfloat16)red[6];
                            float sf_uk_1_6 = (float)(__nv_bfloat16)red[70];
                            float _exp2_86 = approx_exp2((-sf_gk_1_6) * 1.442695f);
                            float sf_sig_1_6 = approx_rcp(1.0f + _exp2_86);
                            float _tanh_87 = tanhf(sf_gk_1_6 * 0.25f);
                            float sf_aa_1_6 = 4.0f * _tanh_87 * sf_sig_1_6;
                            float _tanh_88 = tanhf(sf_uk_1_6 * 0.04f);
                            float sf_bb_1_6 = 25.0f * _tanh_88;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_6 * 768 + row_s + sf_frow_2_6)) + (0)) = (__nv_bfloat16)(sf_aa_1_6 * sf_bb_1_6);
                        }
                        int sf_tok_1_7 = 8 + lane_pair * 2 + 1;
                        int sf_frow_2_7 = row_base + 8;
                        if (sf_tok_1_7 < num_tokens) {
                            float sf_gk_1_7 = (float)(__nv_bfloat16)red[7];
                            float sf_uk_1_7 = (float)(__nv_bfloat16)red[71];
                            float _exp2_89 = approx_exp2((-sf_gk_1_7) * 1.442695f);
                            float sf_sig_1_7 = approx_rcp(1.0f + _exp2_89);
                            float _tanh_90 = tanhf(sf_gk_1_7 * 0.25f);
                            float sf_aa_1_7 = 4.0f * _tanh_90 * sf_sig_1_7;
                            float _tanh_91 = tanhf(sf_uk_1_7 * 0.04f);
                            float sf_bb_1_7 = 25.0f * _tanh_91;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_7 * 768 + row_s + sf_frow_2_7)) + (0)) = (__nv_bfloat16)(sf_aa_1_7 * sf_bb_1_7);
                        }
                        int sf_tok_1_8 = 16 + lane_pair * 2;
                        int sf_frow_2_8 = row_base;
                        if (sf_tok_1_8 < num_tokens) {
                            float sf_gk_1_8 = (float)(__nv_bfloat16)red[8];
                            float sf_uk_1_8 = (float)(__nv_bfloat16)red[72];
                            float _exp2_92 = approx_exp2((-sf_gk_1_8) * 1.442695f);
                            float sf_sig_1_8 = approx_rcp(1.0f + _exp2_92);
                            float _tanh_93 = tanhf(sf_gk_1_8 * 0.25f);
                            float sf_aa_1_8 = 4.0f * _tanh_93 * sf_sig_1_8;
                            float _tanh_94 = tanhf(sf_uk_1_8 * 0.04f);
                            float sf_bb_1_8 = 25.0f * _tanh_94;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_8 * 768 + row_s + sf_frow_2_8)) + (0)) = (__nv_bfloat16)(sf_aa_1_8 * sf_bb_1_8);
                        }
                        int sf_tok_1_9 = 16 + lane_pair * 2 + 1;
                        int sf_frow_2_9 = row_base;
                        if (sf_tok_1_9 < num_tokens) {
                            float sf_gk_1_9 = (float)(__nv_bfloat16)red[9];
                            float sf_uk_1_9 = (float)(__nv_bfloat16)red[73];
                            float _exp2_95 = approx_exp2((-sf_gk_1_9) * 1.442695f);
                            float sf_sig_1_9 = approx_rcp(1.0f + _exp2_95);
                            float _tanh_96 = tanhf(sf_gk_1_9 * 0.25f);
                            float sf_aa_1_9 = 4.0f * _tanh_96 * sf_sig_1_9;
                            float _tanh_97 = tanhf(sf_uk_1_9 * 0.04f);
                            float sf_bb_1_9 = 25.0f * _tanh_97;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_9 * 768 + row_s + sf_frow_2_9)) + (0)) = (__nv_bfloat16)(sf_aa_1_9 * sf_bb_1_9);
                        }
                        int sf_tok_1_10 = 16 + lane_pair * 2;
                        int sf_frow_2_10 = row_base + 8;
                        if (sf_tok_1_10 < num_tokens) {
                            float sf_gk_1_10 = (float)(__nv_bfloat16)red[10];
                            float sf_uk_1_10 = (float)(__nv_bfloat16)red[74];
                            float _exp2_98 = approx_exp2((-sf_gk_1_10) * 1.442695f);
                            float sf_sig_1_10 = approx_rcp(1.0f + _exp2_98);
                            float _tanh_99 = tanhf(sf_gk_1_10 * 0.25f);
                            float sf_aa_1_10 = 4.0f * _tanh_99 * sf_sig_1_10;
                            float _tanh_100 = tanhf(sf_uk_1_10 * 0.04f);
                            float sf_bb_1_10 = 25.0f * _tanh_100;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_10 * 768 + row_s + sf_frow_2_10)) + (0)) = (__nv_bfloat16)(sf_aa_1_10 * sf_bb_1_10);
                        }
                        int sf_tok_1_11 = 16 + lane_pair * 2 + 1;
                        int sf_frow_2_11 = row_base + 8;
                        if (sf_tok_1_11 < num_tokens) {
                            float sf_gk_1_11 = (float)(__nv_bfloat16)red[11];
                            float sf_uk_1_11 = (float)(__nv_bfloat16)red[75];
                            float _exp2_101 = approx_exp2((-sf_gk_1_11) * 1.442695f);
                            float sf_sig_1_11 = approx_rcp(1.0f + _exp2_101);
                            float _tanh_102 = tanhf(sf_gk_1_11 * 0.25f);
                            float sf_aa_1_11 = 4.0f * _tanh_102 * sf_sig_1_11;
                            float _tanh_103 = tanhf(sf_uk_1_11 * 0.04f);
                            float sf_bb_1_11 = 25.0f * _tanh_103;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_11 * 768 + row_s + sf_frow_2_11)) + (0)) = (__nv_bfloat16)(sf_aa_1_11 * sf_bb_1_11);
                        }
                        int sf_tok_1_12 = 24 + lane_pair * 2;
                        int sf_frow_2_12 = row_base;
                        if (sf_tok_1_12 < num_tokens) {
                            float sf_gk_1_12 = (float)(__nv_bfloat16)red[12];
                            float sf_uk_1_12 = (float)(__nv_bfloat16)red[76];
                            float _exp2_104 = approx_exp2((-sf_gk_1_12) * 1.442695f);
                            float sf_sig_1_12 = approx_rcp(1.0f + _exp2_104);
                            float _tanh_105 = tanhf(sf_gk_1_12 * 0.25f);
                            float sf_aa_1_12 = 4.0f * _tanh_105 * sf_sig_1_12;
                            float _tanh_106 = tanhf(sf_uk_1_12 * 0.04f);
                            float sf_bb_1_12 = 25.0f * _tanh_106;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_12 * 768 + row_s + sf_frow_2_12)) + (0)) = (__nv_bfloat16)(sf_aa_1_12 * sf_bb_1_12);
                        }
                        int sf_tok_1_13 = 24 + lane_pair * 2 + 1;
                        int sf_frow_2_13 = row_base;
                        if (sf_tok_1_13 < num_tokens) {
                            float sf_gk_1_13 = (float)(__nv_bfloat16)red[13];
                            float sf_uk_1_13 = (float)(__nv_bfloat16)red[77];
                            float _exp2_107 = approx_exp2((-sf_gk_1_13) * 1.442695f);
                            float sf_sig_1_13 = approx_rcp(1.0f + _exp2_107);
                            float _tanh_108 = tanhf(sf_gk_1_13 * 0.25f);
                            float sf_aa_1_13 = 4.0f * _tanh_108 * sf_sig_1_13;
                            float _tanh_109 = tanhf(sf_uk_1_13 * 0.04f);
                            float sf_bb_1_13 = 25.0f * _tanh_109;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_13 * 768 + row_s + sf_frow_2_13)) + (0)) = (__nv_bfloat16)(sf_aa_1_13 * sf_bb_1_13);
                        }
                        int sf_tok_1_14 = 24 + lane_pair * 2;
                        int sf_frow_2_14 = row_base + 8;
                        if (sf_tok_1_14 < num_tokens) {
                            float sf_gk_1_14 = (float)(__nv_bfloat16)red[14];
                            float sf_uk_1_14 = (float)(__nv_bfloat16)red[78];
                            float _exp2_110 = approx_exp2((-sf_gk_1_14) * 1.442695f);
                            float sf_sig_1_14 = approx_rcp(1.0f + _exp2_110);
                            float _tanh_111 = tanhf(sf_gk_1_14 * 0.25f);
                            float sf_aa_1_14 = 4.0f * _tanh_111 * sf_sig_1_14;
                            float _tanh_112 = tanhf(sf_uk_1_14 * 0.04f);
                            float sf_bb_1_14 = 25.0f * _tanh_112;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_14 * 768 + row_s + sf_frow_2_14)) + (0)) = (__nv_bfloat16)(sf_aa_1_14 * sf_bb_1_14);
                        }
                        int sf_tok_1_15 = 24 + lane_pair * 2 + 1;
                        int sf_frow_2_15 = row_base + 8;
                        if (sf_tok_1_15 < num_tokens) {
                            float sf_gk_1_15 = (float)(__nv_bfloat16)red[15];
                            float sf_uk_1_15 = (float)(__nv_bfloat16)red[79];
                            float _exp2_113 = approx_exp2((-sf_gk_1_15) * 1.442695f);
                            float sf_sig_1_15 = approx_rcp(1.0f + _exp2_113);
                            float _tanh_114 = tanhf(sf_gk_1_15 * 0.25f);
                            float sf_aa_1_15 = 4.0f * _tanh_114 * sf_sig_1_15;
                            float _tanh_115 = tanhf(sf_uk_1_15 * 0.04f);
                            float sf_bb_1_15 = 25.0f * _tanh_115;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_15 * 768 + row_s + sf_frow_2_15)) + (0)) = (__nv_bfloat16)(sf_aa_1_15 * sf_bb_1_15);
                        }
                        int sf_tok_1_16 = 32 + lane_pair * 2;
                        int sf_frow_2_16 = row_base;
                        if (sf_tok_1_16 < num_tokens) {
                            float sf_gk_1_16 = (float)(__nv_bfloat16)red[16];
                            float sf_uk_1_16 = (float)(__nv_bfloat16)red[80];
                            float _exp2_116 = approx_exp2((-sf_gk_1_16) * 1.442695f);
                            float sf_sig_1_16 = approx_rcp(1.0f + _exp2_116);
                            float _tanh_117 = tanhf(sf_gk_1_16 * 0.25f);
                            float sf_aa_1_16 = 4.0f * _tanh_117 * sf_sig_1_16;
                            float _tanh_118 = tanhf(sf_uk_1_16 * 0.04f);
                            float sf_bb_1_16 = 25.0f * _tanh_118;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_16 * 768 + row_s + sf_frow_2_16)) + (0)) = (__nv_bfloat16)(sf_aa_1_16 * sf_bb_1_16);
                        }
                        int sf_tok_1_17 = 32 + lane_pair * 2 + 1;
                        int sf_frow_2_17 = row_base;
                        if (sf_tok_1_17 < num_tokens) {
                            float sf_gk_1_17 = (float)(__nv_bfloat16)red[17];
                            float sf_uk_1_17 = (float)(__nv_bfloat16)red[81];
                            float _exp2_119 = approx_exp2((-sf_gk_1_17) * 1.442695f);
                            float sf_sig_1_17 = approx_rcp(1.0f + _exp2_119);
                            float _tanh_120 = tanhf(sf_gk_1_17 * 0.25f);
                            float sf_aa_1_17 = 4.0f * _tanh_120 * sf_sig_1_17;
                            float _tanh_121 = tanhf(sf_uk_1_17 * 0.04f);
                            float sf_bb_1_17 = 25.0f * _tanh_121;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_17 * 768 + row_s + sf_frow_2_17)) + (0)) = (__nv_bfloat16)(sf_aa_1_17 * sf_bb_1_17);
                        }
                        int sf_tok_1_18 = 32 + lane_pair * 2;
                        int sf_frow_2_18 = row_base + 8;
                        if (sf_tok_1_18 < num_tokens) {
                            float sf_gk_1_18 = (float)(__nv_bfloat16)red[18];
                            float sf_uk_1_18 = (float)(__nv_bfloat16)red[82];
                            float _exp2_122 = approx_exp2((-sf_gk_1_18) * 1.442695f);
                            float sf_sig_1_18 = approx_rcp(1.0f + _exp2_122);
                            float _tanh_123 = tanhf(sf_gk_1_18 * 0.25f);
                            float sf_aa_1_18 = 4.0f * _tanh_123 * sf_sig_1_18;
                            float _tanh_124 = tanhf(sf_uk_1_18 * 0.04f);
                            float sf_bb_1_18 = 25.0f * _tanh_124;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_18 * 768 + row_s + sf_frow_2_18)) + (0)) = (__nv_bfloat16)(sf_aa_1_18 * sf_bb_1_18);
                        }
                        int sf_tok_1_19 = 32 + lane_pair * 2 + 1;
                        int sf_frow_2_19 = row_base + 8;
                        if (sf_tok_1_19 < num_tokens) {
                            float sf_gk_1_19 = (float)(__nv_bfloat16)red[19];
                            float sf_uk_1_19 = (float)(__nv_bfloat16)red[83];
                            float _exp2_125 = approx_exp2((-sf_gk_1_19) * 1.442695f);
                            float sf_sig_1_19 = approx_rcp(1.0f + _exp2_125);
                            float _tanh_126 = tanhf(sf_gk_1_19 * 0.25f);
                            float sf_aa_1_19 = 4.0f * _tanh_126 * sf_sig_1_19;
                            float _tanh_127 = tanhf(sf_uk_1_19 * 0.04f);
                            float sf_bb_1_19 = 25.0f * _tanh_127;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_19 * 768 + row_s + sf_frow_2_19)) + (0)) = (__nv_bfloat16)(sf_aa_1_19 * sf_bb_1_19);
                        }
                        int sf_tok_1_20 = 40 + lane_pair * 2;
                        int sf_frow_2_20 = row_base;
                        if (sf_tok_1_20 < num_tokens) {
                            float sf_gk_1_20 = (float)(__nv_bfloat16)red[20];
                            float sf_uk_1_20 = (float)(__nv_bfloat16)red[84];
                            float _exp2_128 = approx_exp2((-sf_gk_1_20) * 1.442695f);
                            float sf_sig_1_20 = approx_rcp(1.0f + _exp2_128);
                            float _tanh_129 = tanhf(sf_gk_1_20 * 0.25f);
                            float sf_aa_1_20 = 4.0f * _tanh_129 * sf_sig_1_20;
                            float _tanh_130 = tanhf(sf_uk_1_20 * 0.04f);
                            float sf_bb_1_20 = 25.0f * _tanh_130;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_20 * 768 + row_s + sf_frow_2_20)) + (0)) = (__nv_bfloat16)(sf_aa_1_20 * sf_bb_1_20);
                        }
                        int sf_tok_1_21 = 40 + lane_pair * 2 + 1;
                        int sf_frow_2_21 = row_base;
                        if (sf_tok_1_21 < num_tokens) {
                            float sf_gk_1_21 = (float)(__nv_bfloat16)red[21];
                            float sf_uk_1_21 = (float)(__nv_bfloat16)red[85];
                            float _exp2_131 = approx_exp2((-sf_gk_1_21) * 1.442695f);
                            float sf_sig_1_21 = approx_rcp(1.0f + _exp2_131);
                            float _tanh_132 = tanhf(sf_gk_1_21 * 0.25f);
                            float sf_aa_1_21 = 4.0f * _tanh_132 * sf_sig_1_21;
                            float _tanh_133 = tanhf(sf_uk_1_21 * 0.04f);
                            float sf_bb_1_21 = 25.0f * _tanh_133;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_21 * 768 + row_s + sf_frow_2_21)) + (0)) = (__nv_bfloat16)(sf_aa_1_21 * sf_bb_1_21);
                        }
                        int sf_tok_1_22 = 40 + lane_pair * 2;
                        int sf_frow_2_22 = row_base + 8;
                        if (sf_tok_1_22 < num_tokens) {
                            float sf_gk_1_22 = (float)(__nv_bfloat16)red[22];
                            float sf_uk_1_22 = (float)(__nv_bfloat16)red[86];
                            float _exp2_134 = approx_exp2((-sf_gk_1_22) * 1.442695f);
                            float sf_sig_1_22 = approx_rcp(1.0f + _exp2_134);
                            float _tanh_135 = tanhf(sf_gk_1_22 * 0.25f);
                            float sf_aa_1_22 = 4.0f * _tanh_135 * sf_sig_1_22;
                            float _tanh_136 = tanhf(sf_uk_1_22 * 0.04f);
                            float sf_bb_1_22 = 25.0f * _tanh_136;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_22 * 768 + row_s + sf_frow_2_22)) + (0)) = (__nv_bfloat16)(sf_aa_1_22 * sf_bb_1_22);
                        }
                        int sf_tok_1_23 = 40 + lane_pair * 2 + 1;
                        int sf_frow_2_23 = row_base + 8;
                        if (sf_tok_1_23 < num_tokens) {
                            float sf_gk_1_23 = (float)(__nv_bfloat16)red[23];
                            float sf_uk_1_23 = (float)(__nv_bfloat16)red[87];
                            float _exp2_137 = approx_exp2((-sf_gk_1_23) * 1.442695f);
                            float sf_sig_1_23 = approx_rcp(1.0f + _exp2_137);
                            float _tanh_138 = tanhf(sf_gk_1_23 * 0.25f);
                            float sf_aa_1_23 = 4.0f * _tanh_138 * sf_sig_1_23;
                            float _tanh_139 = tanhf(sf_uk_1_23 * 0.04f);
                            float sf_bb_1_23 = 25.0f * _tanh_139;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_23 * 768 + row_s + sf_frow_2_23)) + (0)) = (__nv_bfloat16)(sf_aa_1_23 * sf_bb_1_23);
                        }
                        int sf_tok_1_24 = 48 + lane_pair * 2;
                        int sf_frow_2_24 = row_base;
                        if (sf_tok_1_24 < num_tokens) {
                            float sf_gk_1_24 = (float)(__nv_bfloat16)red[24];
                            float sf_uk_1_24 = (float)(__nv_bfloat16)red[88];
                            float _exp2_140 = approx_exp2((-sf_gk_1_24) * 1.442695f);
                            float sf_sig_1_24 = approx_rcp(1.0f + _exp2_140);
                            float _tanh_141 = tanhf(sf_gk_1_24 * 0.25f);
                            float sf_aa_1_24 = 4.0f * _tanh_141 * sf_sig_1_24;
                            float _tanh_142 = tanhf(sf_uk_1_24 * 0.04f);
                            float sf_bb_1_24 = 25.0f * _tanh_142;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_24 * 768 + row_s + sf_frow_2_24)) + (0)) = (__nv_bfloat16)(sf_aa_1_24 * sf_bb_1_24);
                        }
                        int sf_tok_1_25 = 48 + lane_pair * 2 + 1;
                        int sf_frow_2_25 = row_base;
                        if (sf_tok_1_25 < num_tokens) {
                            float sf_gk_1_25 = (float)(__nv_bfloat16)red[25];
                            float sf_uk_1_25 = (float)(__nv_bfloat16)red[89];
                            float _exp2_143 = approx_exp2((-sf_gk_1_25) * 1.442695f);
                            float sf_sig_1_25 = approx_rcp(1.0f + _exp2_143);
                            float _tanh_144 = tanhf(sf_gk_1_25 * 0.25f);
                            float sf_aa_1_25 = 4.0f * _tanh_144 * sf_sig_1_25;
                            float _tanh_145 = tanhf(sf_uk_1_25 * 0.04f);
                            float sf_bb_1_25 = 25.0f * _tanh_145;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_25 * 768 + row_s + sf_frow_2_25)) + (0)) = (__nv_bfloat16)(sf_aa_1_25 * sf_bb_1_25);
                        }
                        int sf_tok_1_26 = 48 + lane_pair * 2;
                        int sf_frow_2_26 = row_base + 8;
                        if (sf_tok_1_26 < num_tokens) {
                            float sf_gk_1_26 = (float)(__nv_bfloat16)red[26];
                            float sf_uk_1_26 = (float)(__nv_bfloat16)red[90];
                            float _exp2_146 = approx_exp2((-sf_gk_1_26) * 1.442695f);
                            float sf_sig_1_26 = approx_rcp(1.0f + _exp2_146);
                            float _tanh_147 = tanhf(sf_gk_1_26 * 0.25f);
                            float sf_aa_1_26 = 4.0f * _tanh_147 * sf_sig_1_26;
                            float _tanh_148 = tanhf(sf_uk_1_26 * 0.04f);
                            float sf_bb_1_26 = 25.0f * _tanh_148;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_26 * 768 + row_s + sf_frow_2_26)) + (0)) = (__nv_bfloat16)(sf_aa_1_26 * sf_bb_1_26);
                        }
                        int sf_tok_1_27 = 48 + lane_pair * 2 + 1;
                        int sf_frow_2_27 = row_base + 8;
                        if (sf_tok_1_27 < num_tokens) {
                            float sf_gk_1_27 = (float)(__nv_bfloat16)red[27];
                            float sf_uk_1_27 = (float)(__nv_bfloat16)red[91];
                            float _exp2_149 = approx_exp2((-sf_gk_1_27) * 1.442695f);
                            float sf_sig_1_27 = approx_rcp(1.0f + _exp2_149);
                            float _tanh_150 = tanhf(sf_gk_1_27 * 0.25f);
                            float sf_aa_1_27 = 4.0f * _tanh_150 * sf_sig_1_27;
                            float _tanh_151 = tanhf(sf_uk_1_27 * 0.04f);
                            float sf_bb_1_27 = 25.0f * _tanh_151;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_27 * 768 + row_s + sf_frow_2_27)) + (0)) = (__nv_bfloat16)(sf_aa_1_27 * sf_bb_1_27);
                        }
                        int sf_tok_1_28 = 56 + lane_pair * 2;
                        int sf_frow_2_28 = row_base;
                        if (sf_tok_1_28 < num_tokens) {
                            float sf_gk_1_28 = (float)(__nv_bfloat16)red[28];
                            float sf_uk_1_28 = (float)(__nv_bfloat16)red[92];
                            float _exp2_152 = approx_exp2((-sf_gk_1_28) * 1.442695f);
                            float sf_sig_1_28 = approx_rcp(1.0f + _exp2_152);
                            float _tanh_153 = tanhf(sf_gk_1_28 * 0.25f);
                            float sf_aa_1_28 = 4.0f * _tanh_153 * sf_sig_1_28;
                            float _tanh_154 = tanhf(sf_uk_1_28 * 0.04f);
                            float sf_bb_1_28 = 25.0f * _tanh_154;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_28 * 768 + row_s + sf_frow_2_28)) + (0)) = (__nv_bfloat16)(sf_aa_1_28 * sf_bb_1_28);
                        }
                        int sf_tok_1_29 = 56 + lane_pair * 2 + 1;
                        int sf_frow_2_29 = row_base;
                        if (sf_tok_1_29 < num_tokens) {
                            float sf_gk_1_29 = (float)(__nv_bfloat16)red[29];
                            float sf_uk_1_29 = (float)(__nv_bfloat16)red[93];
                            float _exp2_155 = approx_exp2((-sf_gk_1_29) * 1.442695f);
                            float sf_sig_1_29 = approx_rcp(1.0f + _exp2_155);
                            float _tanh_156 = tanhf(sf_gk_1_29 * 0.25f);
                            float sf_aa_1_29 = 4.0f * _tanh_156 * sf_sig_1_29;
                            float _tanh_157 = tanhf(sf_uk_1_29 * 0.04f);
                            float sf_bb_1_29 = 25.0f * _tanh_157;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_29 * 768 + row_s + sf_frow_2_29)) + (0)) = (__nv_bfloat16)(sf_aa_1_29 * sf_bb_1_29);
                        }
                        int sf_tok_1_30 = 56 + lane_pair * 2;
                        int sf_frow_2_30 = row_base + 8;
                        if (sf_tok_1_30 < num_tokens) {
                            float sf_gk_1_30 = (float)(__nv_bfloat16)red[30];
                            float sf_uk_1_30 = (float)(__nv_bfloat16)red[94];
                            float _exp2_158 = approx_exp2((-sf_gk_1_30) * 1.442695f);
                            float sf_sig_1_30 = approx_rcp(1.0f + _exp2_158);
                            float _tanh_159 = tanhf(sf_gk_1_30 * 0.25f);
                            float sf_aa_1_30 = 4.0f * _tanh_159 * sf_sig_1_30;
                            float _tanh_160 = tanhf(sf_uk_1_30 * 0.04f);
                            float sf_bb_1_30 = 25.0f * _tanh_160;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_30 * 768 + row_s + sf_frow_2_30)) + (0)) = (__nv_bfloat16)(sf_aa_1_30 * sf_bb_1_30);
                        }
                        int sf_tok_1_31 = 56 + lane_pair * 2 + 1;
                        int sf_frow_2_31 = row_base + 8;
                        if (sf_tok_1_31 < num_tokens) {
                            float sf_gk_1_31 = (float)(__nv_bfloat16)red[31];
                            float sf_uk_1_31 = (float)(__nv_bfloat16)red[95];
                            float _exp2_161 = approx_exp2((-sf_gk_1_31) * 1.442695f);
                            float sf_sig_1_31 = approx_rcp(1.0f + _exp2_161);
                            float _tanh_162 = tanhf(sf_gk_1_31 * 0.25f);
                            float sf_aa_1_31 = 4.0f * _tanh_162 * sf_sig_1_31;
                            float _tanh_163 = tanhf(sf_uk_1_31 * 0.04f);
                            float sf_bb_1_31 = 25.0f * _tanh_163;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_31 * 768 + row_s + sf_frow_2_31)) + (0)) = (__nv_bfloat16)(sf_aa_1_31 * sf_bb_1_31);
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
                    }
                }
                if (slot == 1) {
                    if (cls == 2) {
                        int sf_tok_1_32 = 64 + lane_pair * 2;
                        int sf_frow_2_32 = row_base;
                        if (sf_tok_1_32 < num_tokens) {
                            float sf_gk_1_32 = (float)(__nv_bfloat16)red[32];
                            float sf_uk_1_32 = (float)(__nv_bfloat16)red[96];
                            float _exp2_164 = approx_exp2((-sf_gk_1_32) * 1.442695f);
                            float sf_sig_1_32 = approx_rcp(1.0f + _exp2_164);
                            float _tanh_165 = tanhf(sf_gk_1_32 * 0.25f);
                            float sf_aa_1_32 = 4.0f * _tanh_165 * sf_sig_1_32;
                            float _tanh_166 = tanhf(sf_uk_1_32 * 0.04f);
                            float sf_bb_1_32 = 25.0f * _tanh_166;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_32 * 768 + row_s + sf_frow_2_32)) + (0)) = (__nv_bfloat16)(sf_aa_1_32 * sf_bb_1_32);
                        }
                        int sf_tok_1_33 = 64 + lane_pair * 2 + 1;
                        int sf_frow_2_33 = row_base;
                        if (sf_tok_1_33 < num_tokens) {
                            float sf_gk_1_33 = (float)(__nv_bfloat16)red[33];
                            float sf_uk_1_33 = (float)(__nv_bfloat16)red[97];
                            float _exp2_167 = approx_exp2((-sf_gk_1_33) * 1.442695f);
                            float sf_sig_1_33 = approx_rcp(1.0f + _exp2_167);
                            float _tanh_168 = tanhf(sf_gk_1_33 * 0.25f);
                            float sf_aa_1_33 = 4.0f * _tanh_168 * sf_sig_1_33;
                            float _tanh_169 = tanhf(sf_uk_1_33 * 0.04f);
                            float sf_bb_1_33 = 25.0f * _tanh_169;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_33 * 768 + row_s + sf_frow_2_33)) + (0)) = (__nv_bfloat16)(sf_aa_1_33 * sf_bb_1_33);
                        }
                        int sf_tok_1_34 = 64 + lane_pair * 2;
                        int sf_frow_2_34 = row_base + 8;
                        if (sf_tok_1_34 < num_tokens) {
                            float sf_gk_1_34 = (float)(__nv_bfloat16)red[34];
                            float sf_uk_1_34 = (float)(__nv_bfloat16)red[98];
                            float _exp2_170 = approx_exp2((-sf_gk_1_34) * 1.442695f);
                            float sf_sig_1_34 = approx_rcp(1.0f + _exp2_170);
                            float _tanh_171 = tanhf(sf_gk_1_34 * 0.25f);
                            float sf_aa_1_34 = 4.0f * _tanh_171 * sf_sig_1_34;
                            float _tanh_172 = tanhf(sf_uk_1_34 * 0.04f);
                            float sf_bb_1_34 = 25.0f * _tanh_172;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_34 * 768 + row_s + sf_frow_2_34)) + (0)) = (__nv_bfloat16)(sf_aa_1_34 * sf_bb_1_34);
                        }
                        int sf_tok_1_35 = 64 + lane_pair * 2 + 1;
                        int sf_frow_2_35 = row_base + 8;
                        if (sf_tok_1_35 < num_tokens) {
                            float sf_gk_1_35 = (float)(__nv_bfloat16)red[35];
                            float sf_uk_1_35 = (float)(__nv_bfloat16)red[99];
                            float _exp2_173 = approx_exp2((-sf_gk_1_35) * 1.442695f);
                            float sf_sig_1_35 = approx_rcp(1.0f + _exp2_173);
                            float _tanh_174 = tanhf(sf_gk_1_35 * 0.25f);
                            float sf_aa_1_35 = 4.0f * _tanh_174 * sf_sig_1_35;
                            float _tanh_175 = tanhf(sf_uk_1_35 * 0.04f);
                            float sf_bb_1_35 = 25.0f * _tanh_175;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_35 * 768 + row_s + sf_frow_2_35)) + (0)) = (__nv_bfloat16)(sf_aa_1_35 * sf_bb_1_35);
                        }
                        int sf_tok_1_36 = 72 + lane_pair * 2;
                        int sf_frow_2_36 = row_base;
                        if (sf_tok_1_36 < num_tokens) {
                            float sf_gk_1_36 = (float)(__nv_bfloat16)red[36];
                            float sf_uk_1_36 = (float)(__nv_bfloat16)red[100];
                            float _exp2_176 = approx_exp2((-sf_gk_1_36) * 1.442695f);
                            float sf_sig_1_36 = approx_rcp(1.0f + _exp2_176);
                            float _tanh_177 = tanhf(sf_gk_1_36 * 0.25f);
                            float sf_aa_1_36 = 4.0f * _tanh_177 * sf_sig_1_36;
                            float _tanh_178 = tanhf(sf_uk_1_36 * 0.04f);
                            float sf_bb_1_36 = 25.0f * _tanh_178;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_36 * 768 + row_s + sf_frow_2_36)) + (0)) = (__nv_bfloat16)(sf_aa_1_36 * sf_bb_1_36);
                        }
                        int sf_tok_1_37 = 72 + lane_pair * 2 + 1;
                        int sf_frow_2_37 = row_base;
                        if (sf_tok_1_37 < num_tokens) {
                            float sf_gk_1_37 = (float)(__nv_bfloat16)red[37];
                            float sf_uk_1_37 = (float)(__nv_bfloat16)red[101];
                            float _exp2_179 = approx_exp2((-sf_gk_1_37) * 1.442695f);
                            float sf_sig_1_37 = approx_rcp(1.0f + _exp2_179);
                            float _tanh_180 = tanhf(sf_gk_1_37 * 0.25f);
                            float sf_aa_1_37 = 4.0f * _tanh_180 * sf_sig_1_37;
                            float _tanh_181 = tanhf(sf_uk_1_37 * 0.04f);
                            float sf_bb_1_37 = 25.0f * _tanh_181;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_37 * 768 + row_s + sf_frow_2_37)) + (0)) = (__nv_bfloat16)(sf_aa_1_37 * sf_bb_1_37);
                        }
                        int sf_tok_1_38 = 72 + lane_pair * 2;
                        int sf_frow_2_38 = row_base + 8;
                        if (sf_tok_1_38 < num_tokens) {
                            float sf_gk_1_38 = (float)(__nv_bfloat16)red[38];
                            float sf_uk_1_38 = (float)(__nv_bfloat16)red[102];
                            float _exp2_182 = approx_exp2((-sf_gk_1_38) * 1.442695f);
                            float sf_sig_1_38 = approx_rcp(1.0f + _exp2_182);
                            float _tanh_183 = tanhf(sf_gk_1_38 * 0.25f);
                            float sf_aa_1_38 = 4.0f * _tanh_183 * sf_sig_1_38;
                            float _tanh_184 = tanhf(sf_uk_1_38 * 0.04f);
                            float sf_bb_1_38 = 25.0f * _tanh_184;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_38 * 768 + row_s + sf_frow_2_38)) + (0)) = (__nv_bfloat16)(sf_aa_1_38 * sf_bb_1_38);
                        }
                        int sf_tok_1_39 = 72 + lane_pair * 2 + 1;
                        int sf_frow_2_39 = row_base + 8;
                        if (sf_tok_1_39 < num_tokens) {
                            float sf_gk_1_39 = (float)(__nv_bfloat16)red[39];
                            float sf_uk_1_39 = (float)(__nv_bfloat16)red[103];
                            float _exp2_185 = approx_exp2((-sf_gk_1_39) * 1.442695f);
                            float sf_sig_1_39 = approx_rcp(1.0f + _exp2_185);
                            float _tanh_186 = tanhf(sf_gk_1_39 * 0.25f);
                            float sf_aa_1_39 = 4.0f * _tanh_186 * sf_sig_1_39;
                            float _tanh_187 = tanhf(sf_uk_1_39 * 0.04f);
                            float sf_bb_1_39 = 25.0f * _tanh_187;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_39 * 768 + row_s + sf_frow_2_39)) + (0)) = (__nv_bfloat16)(sf_aa_1_39 * sf_bb_1_39);
                        }
                        int sf_tok_1_40 = 80 + lane_pair * 2;
                        int sf_frow_2_40 = row_base;
                        if (sf_tok_1_40 < num_tokens) {
                            float sf_gk_1_40 = (float)(__nv_bfloat16)red[40];
                            float sf_uk_1_40 = (float)(__nv_bfloat16)red[104];
                            float _exp2_188 = approx_exp2((-sf_gk_1_40) * 1.442695f);
                            float sf_sig_1_40 = approx_rcp(1.0f + _exp2_188);
                            float _tanh_189 = tanhf(sf_gk_1_40 * 0.25f);
                            float sf_aa_1_40 = 4.0f * _tanh_189 * sf_sig_1_40;
                            float _tanh_190 = tanhf(sf_uk_1_40 * 0.04f);
                            float sf_bb_1_40 = 25.0f * _tanh_190;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_40 * 768 + row_s + sf_frow_2_40)) + (0)) = (__nv_bfloat16)(sf_aa_1_40 * sf_bb_1_40);
                        }
                        int sf_tok_1_41 = 80 + lane_pair * 2 + 1;
                        int sf_frow_2_41 = row_base;
                        if (sf_tok_1_41 < num_tokens) {
                            float sf_gk_1_41 = (float)(__nv_bfloat16)red[41];
                            float sf_uk_1_41 = (float)(__nv_bfloat16)red[105];
                            float _exp2_191 = approx_exp2((-sf_gk_1_41) * 1.442695f);
                            float sf_sig_1_41 = approx_rcp(1.0f + _exp2_191);
                            float _tanh_192 = tanhf(sf_gk_1_41 * 0.25f);
                            float sf_aa_1_41 = 4.0f * _tanh_192 * sf_sig_1_41;
                            float _tanh_193 = tanhf(sf_uk_1_41 * 0.04f);
                            float sf_bb_1_41 = 25.0f * _tanh_193;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_41 * 768 + row_s + sf_frow_2_41)) + (0)) = (__nv_bfloat16)(sf_aa_1_41 * sf_bb_1_41);
                        }
                        int sf_tok_1_42 = 80 + lane_pair * 2;
                        int sf_frow_2_42 = row_base + 8;
                        if (sf_tok_1_42 < num_tokens) {
                            float sf_gk_1_42 = (float)(__nv_bfloat16)red[42];
                            float sf_uk_1_42 = (float)(__nv_bfloat16)red[106];
                            float _exp2_194 = approx_exp2((-sf_gk_1_42) * 1.442695f);
                            float sf_sig_1_42 = approx_rcp(1.0f + _exp2_194);
                            float _tanh_195 = tanhf(sf_gk_1_42 * 0.25f);
                            float sf_aa_1_42 = 4.0f * _tanh_195 * sf_sig_1_42;
                            float _tanh_196 = tanhf(sf_uk_1_42 * 0.04f);
                            float sf_bb_1_42 = 25.0f * _tanh_196;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_42 * 768 + row_s + sf_frow_2_42)) + (0)) = (__nv_bfloat16)(sf_aa_1_42 * sf_bb_1_42);
                        }
                        int sf_tok_1_43 = 80 + lane_pair * 2 + 1;
                        int sf_frow_2_43 = row_base + 8;
                        if (sf_tok_1_43 < num_tokens) {
                            float sf_gk_1_43 = (float)(__nv_bfloat16)red[43];
                            float sf_uk_1_43 = (float)(__nv_bfloat16)red[107];
                            float _exp2_197 = approx_exp2((-sf_gk_1_43) * 1.442695f);
                            float sf_sig_1_43 = approx_rcp(1.0f + _exp2_197);
                            float _tanh_198 = tanhf(sf_gk_1_43 * 0.25f);
                            float sf_aa_1_43 = 4.0f * _tanh_198 * sf_sig_1_43;
                            float _tanh_199 = tanhf(sf_uk_1_43 * 0.04f);
                            float sf_bb_1_43 = 25.0f * _tanh_199;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_43 * 768 + row_s + sf_frow_2_43)) + (0)) = (__nv_bfloat16)(sf_aa_1_43 * sf_bb_1_43);
                        }
                        int sf_tok_1_44 = 88 + lane_pair * 2;
                        int sf_frow_2_44 = row_base;
                        if (sf_tok_1_44 < num_tokens) {
                            float sf_gk_1_44 = (float)(__nv_bfloat16)red[44];
                            float sf_uk_1_44 = (float)(__nv_bfloat16)red[108];
                            float _exp2_200 = approx_exp2((-sf_gk_1_44) * 1.442695f);
                            float sf_sig_1_44 = approx_rcp(1.0f + _exp2_200);
                            float _tanh_201 = tanhf(sf_gk_1_44 * 0.25f);
                            float sf_aa_1_44 = 4.0f * _tanh_201 * sf_sig_1_44;
                            float _tanh_202 = tanhf(sf_uk_1_44 * 0.04f);
                            float sf_bb_1_44 = 25.0f * _tanh_202;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_44 * 768 + row_s + sf_frow_2_44)) + (0)) = (__nv_bfloat16)(sf_aa_1_44 * sf_bb_1_44);
                        }
                        int sf_tok_1_45 = 88 + lane_pair * 2 + 1;
                        int sf_frow_2_45 = row_base;
                        if (sf_tok_1_45 < num_tokens) {
                            float sf_gk_1_45 = (float)(__nv_bfloat16)red[45];
                            float sf_uk_1_45 = (float)(__nv_bfloat16)red[109];
                            float _exp2_203 = approx_exp2((-sf_gk_1_45) * 1.442695f);
                            float sf_sig_1_45 = approx_rcp(1.0f + _exp2_203);
                            float _tanh_204 = tanhf(sf_gk_1_45 * 0.25f);
                            float sf_aa_1_45 = 4.0f * _tanh_204 * sf_sig_1_45;
                            float _tanh_205 = tanhf(sf_uk_1_45 * 0.04f);
                            float sf_bb_1_45 = 25.0f * _tanh_205;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_45 * 768 + row_s + sf_frow_2_45)) + (0)) = (__nv_bfloat16)(sf_aa_1_45 * sf_bb_1_45);
                        }
                        int sf_tok_1_46 = 88 + lane_pair * 2;
                        int sf_frow_2_46 = row_base + 8;
                        if (sf_tok_1_46 < num_tokens) {
                            float sf_gk_1_46 = (float)(__nv_bfloat16)red[46];
                            float sf_uk_1_46 = (float)(__nv_bfloat16)red[110];
                            float _exp2_206 = approx_exp2((-sf_gk_1_46) * 1.442695f);
                            float sf_sig_1_46 = approx_rcp(1.0f + _exp2_206);
                            float _tanh_207 = tanhf(sf_gk_1_46 * 0.25f);
                            float sf_aa_1_46 = 4.0f * _tanh_207 * sf_sig_1_46;
                            float _tanh_208 = tanhf(sf_uk_1_46 * 0.04f);
                            float sf_bb_1_46 = 25.0f * _tanh_208;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_46 * 768 + row_s + sf_frow_2_46)) + (0)) = (__nv_bfloat16)(sf_aa_1_46 * sf_bb_1_46);
                        }
                        int sf_tok_1_47 = 88 + lane_pair * 2 + 1;
                        int sf_frow_2_47 = row_base + 8;
                        if (sf_tok_1_47 < num_tokens) {
                            float sf_gk_1_47 = (float)(__nv_bfloat16)red[47];
                            float sf_uk_1_47 = (float)(__nv_bfloat16)red[111];
                            float _exp2_209 = approx_exp2((-sf_gk_1_47) * 1.442695f);
                            float sf_sig_1_47 = approx_rcp(1.0f + _exp2_209);
                            float _tanh_210 = tanhf(sf_gk_1_47 * 0.25f);
                            float sf_aa_1_47 = 4.0f * _tanh_210 * sf_sig_1_47;
                            float _tanh_211 = tanhf(sf_uk_1_47 * 0.04f);
                            float sf_bb_1_47 = 25.0f * _tanh_211;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_47 * 768 + row_s + sf_frow_2_47)) + (0)) = (__nv_bfloat16)(sf_aa_1_47 * sf_bb_1_47);
                        }
                        int sf_tok_1_48 = 96 + lane_pair * 2;
                        int sf_frow_2_48 = row_base;
                        if (sf_tok_1_48 < num_tokens) {
                            float sf_gk_1_48 = (float)(__nv_bfloat16)red[48];
                            float sf_uk_1_48 = (float)(__nv_bfloat16)red[112];
                            float _exp2_212 = approx_exp2((-sf_gk_1_48) * 1.442695f);
                            float sf_sig_1_48 = approx_rcp(1.0f + _exp2_212);
                            float _tanh_213 = tanhf(sf_gk_1_48 * 0.25f);
                            float sf_aa_1_48 = 4.0f * _tanh_213 * sf_sig_1_48;
                            float _tanh_214 = tanhf(sf_uk_1_48 * 0.04f);
                            float sf_bb_1_48 = 25.0f * _tanh_214;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_48 * 768 + row_s + sf_frow_2_48)) + (0)) = (__nv_bfloat16)(sf_aa_1_48 * sf_bb_1_48);
                        }
                        int sf_tok_1_49 = 96 + lane_pair * 2 + 1;
                        int sf_frow_2_49 = row_base;
                        if (sf_tok_1_49 < num_tokens) {
                            float sf_gk_1_49 = (float)(__nv_bfloat16)red[49];
                            float sf_uk_1_49 = (float)(__nv_bfloat16)red[113];
                            float _exp2_215 = approx_exp2((-sf_gk_1_49) * 1.442695f);
                            float sf_sig_1_49 = approx_rcp(1.0f + _exp2_215);
                            float _tanh_216 = tanhf(sf_gk_1_49 * 0.25f);
                            float sf_aa_1_49 = 4.0f * _tanh_216 * sf_sig_1_49;
                            float _tanh_217 = tanhf(sf_uk_1_49 * 0.04f);
                            float sf_bb_1_49 = 25.0f * _tanh_217;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_49 * 768 + row_s + sf_frow_2_49)) + (0)) = (__nv_bfloat16)(sf_aa_1_49 * sf_bb_1_49);
                        }
                        int sf_tok_1_50 = 96 + lane_pair * 2;
                        int sf_frow_2_50 = row_base + 8;
                        if (sf_tok_1_50 < num_tokens) {
                            float sf_gk_1_50 = (float)(__nv_bfloat16)red[50];
                            float sf_uk_1_50 = (float)(__nv_bfloat16)red[114];
                            float _exp2_218 = approx_exp2((-sf_gk_1_50) * 1.442695f);
                            float sf_sig_1_50 = approx_rcp(1.0f + _exp2_218);
                            float _tanh_219 = tanhf(sf_gk_1_50 * 0.25f);
                            float sf_aa_1_50 = 4.0f * _tanh_219 * sf_sig_1_50;
                            float _tanh_220 = tanhf(sf_uk_1_50 * 0.04f);
                            float sf_bb_1_50 = 25.0f * _tanh_220;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_50 * 768 + row_s + sf_frow_2_50)) + (0)) = (__nv_bfloat16)(sf_aa_1_50 * sf_bb_1_50);
                        }
                        int sf_tok_1_51 = 96 + lane_pair * 2 + 1;
                        int sf_frow_2_51 = row_base + 8;
                        if (sf_tok_1_51 < num_tokens) {
                            float sf_gk_1_51 = (float)(__nv_bfloat16)red[51];
                            float sf_uk_1_51 = (float)(__nv_bfloat16)red[115];
                            float _exp2_221 = approx_exp2((-sf_gk_1_51) * 1.442695f);
                            float sf_sig_1_51 = approx_rcp(1.0f + _exp2_221);
                            float _tanh_222 = tanhf(sf_gk_1_51 * 0.25f);
                            float sf_aa_1_51 = 4.0f * _tanh_222 * sf_sig_1_51;
                            float _tanh_223 = tanhf(sf_uk_1_51 * 0.04f);
                            float sf_bb_1_51 = 25.0f * _tanh_223;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_51 * 768 + row_s + sf_frow_2_51)) + (0)) = (__nv_bfloat16)(sf_aa_1_51 * sf_bb_1_51);
                        }
                        int sf_tok_1_52 = 104 + lane_pair * 2;
                        int sf_frow_2_52 = row_base;
                        if (sf_tok_1_52 < num_tokens) {
                            float sf_gk_1_52 = (float)(__nv_bfloat16)red[52];
                            float sf_uk_1_52 = (float)(__nv_bfloat16)red[116];
                            float _exp2_224 = approx_exp2((-sf_gk_1_52) * 1.442695f);
                            float sf_sig_1_52 = approx_rcp(1.0f + _exp2_224);
                            float _tanh_225 = tanhf(sf_gk_1_52 * 0.25f);
                            float sf_aa_1_52 = 4.0f * _tanh_225 * sf_sig_1_52;
                            float _tanh_226 = tanhf(sf_uk_1_52 * 0.04f);
                            float sf_bb_1_52 = 25.0f * _tanh_226;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_52 * 768 + row_s + sf_frow_2_52)) + (0)) = (__nv_bfloat16)(sf_aa_1_52 * sf_bb_1_52);
                        }
                        int sf_tok_1_53 = 104 + lane_pair * 2 + 1;
                        int sf_frow_2_53 = row_base;
                        if (sf_tok_1_53 < num_tokens) {
                            float sf_gk_1_53 = (float)(__nv_bfloat16)red[53];
                            float sf_uk_1_53 = (float)(__nv_bfloat16)red[117];
                            float _exp2_227 = approx_exp2((-sf_gk_1_53) * 1.442695f);
                            float sf_sig_1_53 = approx_rcp(1.0f + _exp2_227);
                            float _tanh_228 = tanhf(sf_gk_1_53 * 0.25f);
                            float sf_aa_1_53 = 4.0f * _tanh_228 * sf_sig_1_53;
                            float _tanh_229 = tanhf(sf_uk_1_53 * 0.04f);
                            float sf_bb_1_53 = 25.0f * _tanh_229;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_53 * 768 + row_s + sf_frow_2_53)) + (0)) = (__nv_bfloat16)(sf_aa_1_53 * sf_bb_1_53);
                        }
                        int sf_tok_1_54 = 104 + lane_pair * 2;
                        int sf_frow_2_54 = row_base + 8;
                        if (sf_tok_1_54 < num_tokens) {
                            float sf_gk_1_54 = (float)(__nv_bfloat16)red[54];
                            float sf_uk_1_54 = (float)(__nv_bfloat16)red[118];
                            float _exp2_230 = approx_exp2((-sf_gk_1_54) * 1.442695f);
                            float sf_sig_1_54 = approx_rcp(1.0f + _exp2_230);
                            float _tanh_231 = tanhf(sf_gk_1_54 * 0.25f);
                            float sf_aa_1_54 = 4.0f * _tanh_231 * sf_sig_1_54;
                            float _tanh_232 = tanhf(sf_uk_1_54 * 0.04f);
                            float sf_bb_1_54 = 25.0f * _tanh_232;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_54 * 768 + row_s + sf_frow_2_54)) + (0)) = (__nv_bfloat16)(sf_aa_1_54 * sf_bb_1_54);
                        }
                        int sf_tok_1_55 = 104 + lane_pair * 2 + 1;
                        int sf_frow_2_55 = row_base + 8;
                        if (sf_tok_1_55 < num_tokens) {
                            float sf_gk_1_55 = (float)(__nv_bfloat16)red[55];
                            float sf_uk_1_55 = (float)(__nv_bfloat16)red[119];
                            float _exp2_233 = approx_exp2((-sf_gk_1_55) * 1.442695f);
                            float sf_sig_1_55 = approx_rcp(1.0f + _exp2_233);
                            float _tanh_234 = tanhf(sf_gk_1_55 * 0.25f);
                            float sf_aa_1_55 = 4.0f * _tanh_234 * sf_sig_1_55;
                            float _tanh_235 = tanhf(sf_uk_1_55 * 0.04f);
                            float sf_bb_1_55 = 25.0f * _tanh_235;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_55 * 768 + row_s + sf_frow_2_55)) + (0)) = (__nv_bfloat16)(sf_aa_1_55 * sf_bb_1_55);
                        }
                        int sf_tok_1_56 = 112 + lane_pair * 2;
                        int sf_frow_2_56 = row_base;
                        if (sf_tok_1_56 < num_tokens) {
                            float sf_gk_1_56 = (float)(__nv_bfloat16)red[56];
                            float sf_uk_1_56 = (float)(__nv_bfloat16)red[120];
                            float _exp2_236 = approx_exp2((-sf_gk_1_56) * 1.442695f);
                            float sf_sig_1_56 = approx_rcp(1.0f + _exp2_236);
                            float _tanh_237 = tanhf(sf_gk_1_56 * 0.25f);
                            float sf_aa_1_56 = 4.0f * _tanh_237 * sf_sig_1_56;
                            float _tanh_238 = tanhf(sf_uk_1_56 * 0.04f);
                            float sf_bb_1_56 = 25.0f * _tanh_238;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_56 * 768 + row_s + sf_frow_2_56)) + (0)) = (__nv_bfloat16)(sf_aa_1_56 * sf_bb_1_56);
                        }
                        int sf_tok_1_57 = 112 + lane_pair * 2 + 1;
                        int sf_frow_2_57 = row_base;
                        if (sf_tok_1_57 < num_tokens) {
                            float sf_gk_1_57 = (float)(__nv_bfloat16)red[57];
                            float sf_uk_1_57 = (float)(__nv_bfloat16)red[121];
                            float _exp2_239 = approx_exp2((-sf_gk_1_57) * 1.442695f);
                            float sf_sig_1_57 = approx_rcp(1.0f + _exp2_239);
                            float _tanh_240 = tanhf(sf_gk_1_57 * 0.25f);
                            float sf_aa_1_57 = 4.0f * _tanh_240 * sf_sig_1_57;
                            float _tanh_241 = tanhf(sf_uk_1_57 * 0.04f);
                            float sf_bb_1_57 = 25.0f * _tanh_241;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_57 * 768 + row_s + sf_frow_2_57)) + (0)) = (__nv_bfloat16)(sf_aa_1_57 * sf_bb_1_57);
                        }
                        int sf_tok_1_58 = 112 + lane_pair * 2;
                        int sf_frow_2_58 = row_base + 8;
                        if (sf_tok_1_58 < num_tokens) {
                            float sf_gk_1_58 = (float)(__nv_bfloat16)red[58];
                            float sf_uk_1_58 = (float)(__nv_bfloat16)red[122];
                            float _exp2_242 = approx_exp2((-sf_gk_1_58) * 1.442695f);
                            float sf_sig_1_58 = approx_rcp(1.0f + _exp2_242);
                            float _tanh_243 = tanhf(sf_gk_1_58 * 0.25f);
                            float sf_aa_1_58 = 4.0f * _tanh_243 * sf_sig_1_58;
                            float _tanh_244 = tanhf(sf_uk_1_58 * 0.04f);
                            float sf_bb_1_58 = 25.0f * _tanh_244;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_58 * 768 + row_s + sf_frow_2_58)) + (0)) = (__nv_bfloat16)(sf_aa_1_58 * sf_bb_1_58);
                        }
                        int sf_tok_1_59 = 112 + lane_pair * 2 + 1;
                        int sf_frow_2_59 = row_base + 8;
                        if (sf_tok_1_59 < num_tokens) {
                            float sf_gk_1_59 = (float)(__nv_bfloat16)red[59];
                            float sf_uk_1_59 = (float)(__nv_bfloat16)red[123];
                            float _exp2_245 = approx_exp2((-sf_gk_1_59) * 1.442695f);
                            float sf_sig_1_59 = approx_rcp(1.0f + _exp2_245);
                            float _tanh_246 = tanhf(sf_gk_1_59 * 0.25f);
                            float sf_aa_1_59 = 4.0f * _tanh_246 * sf_sig_1_59;
                            float _tanh_247 = tanhf(sf_uk_1_59 * 0.04f);
                            float sf_bb_1_59 = 25.0f * _tanh_247;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_59 * 768 + row_s + sf_frow_2_59)) + (0)) = (__nv_bfloat16)(sf_aa_1_59 * sf_bb_1_59);
                        }
                        int sf_tok_1_60 = 120 + lane_pair * 2;
                        int sf_frow_2_60 = row_base;
                        if (sf_tok_1_60 < num_tokens) {
                            float sf_gk_1_60 = (float)(__nv_bfloat16)red[60];
                            float sf_uk_1_60 = (float)(__nv_bfloat16)red[124];
                            float _exp2_248 = approx_exp2((-sf_gk_1_60) * 1.442695f);
                            float sf_sig_1_60 = approx_rcp(1.0f + _exp2_248);
                            float _tanh_249 = tanhf(sf_gk_1_60 * 0.25f);
                            float sf_aa_1_60 = 4.0f * _tanh_249 * sf_sig_1_60;
                            float _tanh_250 = tanhf(sf_uk_1_60 * 0.04f);
                            float sf_bb_1_60 = 25.0f * _tanh_250;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_60 * 768 + row_s + sf_frow_2_60)) + (0)) = (__nv_bfloat16)(sf_aa_1_60 * sf_bb_1_60);
                        }
                        int sf_tok_1_61 = 120 + lane_pair * 2 + 1;
                        int sf_frow_2_61 = row_base;
                        if (sf_tok_1_61 < num_tokens) {
                            float sf_gk_1_61 = (float)(__nv_bfloat16)red[61];
                            float sf_uk_1_61 = (float)(__nv_bfloat16)red[125];
                            float _exp2_251 = approx_exp2((-sf_gk_1_61) * 1.442695f);
                            float sf_sig_1_61 = approx_rcp(1.0f + _exp2_251);
                            float _tanh_252 = tanhf(sf_gk_1_61 * 0.25f);
                            float sf_aa_1_61 = 4.0f * _tanh_252 * sf_sig_1_61;
                            float _tanh_253 = tanhf(sf_uk_1_61 * 0.04f);
                            float sf_bb_1_61 = 25.0f * _tanh_253;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_61 * 768 + row_s + sf_frow_2_61)) + (0)) = (__nv_bfloat16)(sf_aa_1_61 * sf_bb_1_61);
                        }
                        int sf_tok_1_62 = 120 + lane_pair * 2;
                        int sf_frow_2_62 = row_base + 8;
                        if (sf_tok_1_62 < num_tokens) {
                            float sf_gk_1_62 = (float)(__nv_bfloat16)red[62];
                            float sf_uk_1_62 = (float)(__nv_bfloat16)red[126];
                            float _exp2_254 = approx_exp2((-sf_gk_1_62) * 1.442695f);
                            float sf_sig_1_62 = approx_rcp(1.0f + _exp2_254);
                            float _tanh_255 = tanhf(sf_gk_1_62 * 0.25f);
                            float sf_aa_1_62 = 4.0f * _tanh_255 * sf_sig_1_62;
                            float _tanh_256 = tanhf(sf_uk_1_62 * 0.04f);
                            float sf_bb_1_62 = 25.0f * _tanh_256;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_62 * 768 + row_s + sf_frow_2_62)) + (0)) = (__nv_bfloat16)(sf_aa_1_62 * sf_bb_1_62);
                        }
                        int sf_tok_1_63 = 120 + lane_pair * 2 + 1;
                        int sf_frow_2_63 = row_base + 8;
                        if (sf_tok_1_63 < num_tokens) {
                            float sf_gk_1_63 = (float)(__nv_bfloat16)red[63];
                            float sf_uk_1_63 = (float)(__nv_bfloat16)red[127];
                            float _exp2_257 = approx_exp2((-sf_gk_1_63) * 1.442695f);
                            float sf_sig_1_63 = approx_rcp(1.0f + _exp2_257);
                            float _tanh_258 = tanhf(sf_gk_1_63 * 0.25f);
                            float sf_aa_1_63 = 4.0f * _tanh_258 * sf_sig_1_63;
                            float _tanh_259 = tanhf(sf_uk_1_63 * 0.04f);
                            float sf_bb_1_63 = 25.0f * _tanh_259;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_63 * 768 + row_s + sf_frow_2_63)) + (0)) = (__nv_bfloat16)(sf_aa_1_63 * sf_bb_1_63);
                        }
                    } else {
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
            tile_l = g_l / 2;
            rank_l = g_l % 2;
            int d_lo_l = ((rank_l == 0) ? 0 : 0);
            int d_cnt_l = ((rank_l == 0) ? 0 : 0);
            int u_lo_l = ((rank_l == 0) ? 0 : 56);
            int u_cnt_l = ((rank_l == 0) ? 56 : 56);
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
                                :: "r"(smem_a_addr + stage * 16384 + 8192), "l"((&A_S)), "r"(0), "r"(768 + row_s_1), "r"(kc1),
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
            tile_2 = g_m / 2;
            rank_m = g_m % 2;
            int u_count_m = ((rank_m == 0) ? 56 : 56);
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
