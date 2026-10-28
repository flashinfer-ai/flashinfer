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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 10
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 8192
#define SMEM_LAND_STRIDE 8192
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 10240
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 174080
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_SMEM_BN_OFF 174080
#define SMEM_SMEM_BN_STAGE_BYTES 4096
#define SMEM_SMEM_BN_STRIDE 4096
#define SMEM_SMEM_AG_OFF 10240
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 18432
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 215040

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
kernel_cake_kimi_k3_latent_moe_a023716bc017311f5827(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 160)
    #define acc_empty_addr (mbar_base + 168)
    #define red_bar_addr (mbar_base + 176)
    #define issue_bar_addr (mbar_base + 184)
    #define rows_bar_addr (mbar_base + 192)
    #define bn_bar_addr (mbar_base + 200)

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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_a_addr = smem + 10240;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int smem_b_addr = smem + 174080;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int smem_bn_addr = smem + 174080;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_ag_addr = smem + 10240;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 18432);
    const int smem_au_addr = smem + 18432;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 10 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // a_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 176, 8192);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
            float red[32];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[16] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[17] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[18] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[19] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[20] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[21] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[22] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[23] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[24] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[25] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[26] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[27] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[28] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[29] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[30] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[31] = _tmem_load_1[15];
            } else {
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_addr));
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
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr);
                }
            }
            if (slot == 0) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(1));
                if (cls == 2) {
                    uint32_t _mbar_peer_0;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_0) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mbar_peer_0) : "memory");
                    uint32_t _mbar_peer_1;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_1) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mbar_peer_1) : "memory");
                    uint32_t _mbar_peer_2;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_2) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mbar_peer_2) : "memory");
                    uint32_t _mbar_peer_3;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_3) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mbar_peer_3) : "memory");
                } else {
                    uint32_t _mbar_peer_4;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_4) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mbar_peer_4) : "memory");
                    uint32_t _mbar_peer_5;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_5) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 2048), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mbar_peer_5) : "memory");
                    uint32_t _mbar_peer_6;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_6) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 4096), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mbar_peer_6) : "memory");
                    uint32_t _mbar_peer_7;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_7) : "r"(red_bar_addr), "r"(1));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mbar_peer_7) : "memory");
                }
            }
            if (slot == 1) {
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                if (cls == 2) {
                    uint32_t _mbar_peer_8;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_8) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_8) : "memory");
                    uint32_t _mbar_peer_9;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_9) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_9) : "memory");
                    uint32_t _mbar_peer_10;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_10) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mbar_peer_10) : "memory");
                    uint32_t _mbar_peer_11;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_11) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mbar_peer_11) : "memory");
                } else {
                    uint32_t _mbar_peer_12;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_12) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mbar_peer_12) : "memory");
                    uint32_t _mbar_peer_13;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_13) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mbar_peer_13) : "memory");
                    uint32_t _mbar_peer_14;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_14) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mbar_peer_14) : "memory");
                    uint32_t _mbar_peer_15;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_15) : "r"(red_bar_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mbar_peer_15) : "memory");
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
                    red[16] = red[16] + land[(256 + tid_1) * 4];
                    red[17] = red[17] + land[(256 + tid_1) * 4 + 1];
                    red[18] = red[18] + land[(256 + tid_1) * 4 + 2];
                    red[19] = red[19] + land[(256 + tid_1) * 4 + 3];
                    red[20] = red[20] + land[(384 + tid_1) * 4];
                    red[21] = red[21] + land[(384 + tid_1) * 4 + 1];
                    red[22] = red[22] + land[(384 + tid_1) * 4 + 2];
                    red[23] = red[23] + land[(384 + tid_1) * 4 + 3];
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
                }
            }
            if (slot == 1) {
                if (cls == 2) {
                    red[8] = red[8] + land[tid_1 * 4];
                    red[9] = red[9] + land[tid_1 * 4 + 1];
                    red[10] = red[10] + land[tid_1 * 4 + 2];
                    red[11] = red[11] + land[tid_1 * 4 + 3];
                    red[12] = red[12] + land[(128 + tid_1) * 4];
                    red[13] = red[13] + land[(128 + tid_1) * 4 + 1];
                    red[14] = red[14] + land[(128 + tid_1) * 4 + 2];
                    red[15] = red[15] + land[(128 + tid_1) * 4 + 3];
                    red[24] = red[24] + land[(256 + tid_1) * 4];
                    red[25] = red[25] + land[(256 + tid_1) * 4 + 1];
                    red[26] = red[26] + land[(256 + tid_1) * 4 + 2];
                    red[27] = red[27] + land[(256 + tid_1) * 4 + 3];
                    red[28] = red[28] + land[(384 + tid_1) * 4];
                    red[29] = red[29] + land[(384 + tid_1) * 4 + 1];
                    red[30] = red[30] + land[(384 + tid_1) * 4 + 2];
                    red[31] = red[31] + land[(384 + tid_1) * 4 + 3];
                } else {
                    red[16] = red[16] + land[tid_1 * 4];
                    red[17] = red[17] + land[tid_1 * 4 + 1];
                    red[18] = red[18] + land[tid_1 * 4 + 2];
                    red[19] = red[19] + land[tid_1 * 4 + 3];
                    red[20] = red[20] + land[(128 + tid_1) * 4];
                    red[21] = red[21] + land[(128 + tid_1) * 4 + 1];
                    red[22] = red[22] + land[(128 + tid_1) * 4 + 2];
                    red[23] = red[23] + land[(128 + tid_1) * 4 + 3];
                    red[24] = red[24] + land[(256 + tid_1) * 4];
                    red[25] = red[25] + land[(256 + tid_1) * 4 + 1];
                    red[26] = red[26] + land[(256 + tid_1) * 4 + 2];
                    red[27] = red[27] + land[(256 + tid_1) * 4 + 3];
                    red[28] = red[28] + land[(384 + tid_1) * 4];
                    red[29] = red[29] + land[(384 + tid_1) * 4 + 1];
                    red[30] = red[30] + land[(384 + tid_1) * 4 + 2];
                    red[31] = red[31] + land[(384 + tid_1) * 4 + 3];
                }
            }
            if (fin == 1) {
                if (slot == 0) {
                    if (cls == 2) {
                        int sf_tok_1_0 = lane_pair * 2;
                        int sf_frow_2_0 = row_base;
                        if (sf_tok_1_0 < num_tokens) {
                            float sf_gk_1_0 = (float)(__nv_bfloat16)red[0];
                            float sf_uk_1_0 = (float)(__nv_bfloat16)red[16];
                            float _exp2_16 = approx_exp2((-sf_gk_1_0) * 1.442695f);
                            float sf_sig_1_0 = approx_rcp(1.0f + _exp2_16);
                            float _tanh_17 = tanhf(sf_gk_1_0 * 0.25f);
                            float sf_aa_1_0 = 4.0f * _tanh_17 * sf_sig_1_0;
                            float _tanh_18 = tanhf(sf_uk_1_0 * 0.04f);
                            float sf_bb_1_0 = 25.0f * _tanh_18;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_0 * 768 + row_s + sf_frow_2_0)) + (0)) = (__nv_bfloat16)(sf_aa_1_0 * sf_bb_1_0);
                        }
                        int sf_tok_1_1 = lane_pair * 2 + 1;
                        int sf_frow_2_1 = row_base;
                        if (sf_tok_1_1 < num_tokens) {
                            float sf_gk_1_1 = (float)(__nv_bfloat16)red[1];
                            float sf_uk_1_1 = (float)(__nv_bfloat16)red[17];
                            float _exp2_19 = approx_exp2((-sf_gk_1_1) * 1.442695f);
                            float sf_sig_1_1 = approx_rcp(1.0f + _exp2_19);
                            float _tanh_20 = tanhf(sf_gk_1_1 * 0.25f);
                            float sf_aa_1_1 = 4.0f * _tanh_20 * sf_sig_1_1;
                            float _tanh_21 = tanhf(sf_uk_1_1 * 0.04f);
                            float sf_bb_1_1 = 25.0f * _tanh_21;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_1 * 768 + row_s + sf_frow_2_1)) + (0)) = (__nv_bfloat16)(sf_aa_1_1 * sf_bb_1_1);
                        }
                        int sf_tok_1_2 = lane_pair * 2;
                        int sf_frow_2_2 = row_base + 8;
                        if (sf_tok_1_2 < num_tokens) {
                            float sf_gk_1_2 = (float)(__nv_bfloat16)red[2];
                            float sf_uk_1_2 = (float)(__nv_bfloat16)red[18];
                            float _exp2_22 = approx_exp2((-sf_gk_1_2) * 1.442695f);
                            float sf_sig_1_2 = approx_rcp(1.0f + _exp2_22);
                            float _tanh_23 = tanhf(sf_gk_1_2 * 0.25f);
                            float sf_aa_1_2 = 4.0f * _tanh_23 * sf_sig_1_2;
                            float _tanh_24 = tanhf(sf_uk_1_2 * 0.04f);
                            float sf_bb_1_2 = 25.0f * _tanh_24;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_2 * 768 + row_s + sf_frow_2_2)) + (0)) = (__nv_bfloat16)(sf_aa_1_2 * sf_bb_1_2);
                        }
                        int sf_tok_1_3 = lane_pair * 2 + 1;
                        int sf_frow_2_3 = row_base + 8;
                        if (sf_tok_1_3 < num_tokens) {
                            float sf_gk_1_3 = (float)(__nv_bfloat16)red[3];
                            float sf_uk_1_3 = (float)(__nv_bfloat16)red[19];
                            float _exp2_25 = approx_exp2((-sf_gk_1_3) * 1.442695f);
                            float sf_sig_1_3 = approx_rcp(1.0f + _exp2_25);
                            float _tanh_26 = tanhf(sf_gk_1_3 * 0.25f);
                            float sf_aa_1_3 = 4.0f * _tanh_26 * sf_sig_1_3;
                            float _tanh_27 = tanhf(sf_uk_1_3 * 0.04f);
                            float sf_bb_1_3 = 25.0f * _tanh_27;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_3 * 768 + row_s + sf_frow_2_3)) + (0)) = (__nv_bfloat16)(sf_aa_1_3 * sf_bb_1_3);
                        }
                        int sf_tok_1_4 = 8 + lane_pair * 2;
                        int sf_frow_2_4 = row_base;
                        if (sf_tok_1_4 < num_tokens) {
                            float sf_gk_1_4 = (float)(__nv_bfloat16)red[4];
                            float sf_uk_1_4 = (float)(__nv_bfloat16)red[20];
                            float _exp2_28 = approx_exp2((-sf_gk_1_4) * 1.442695f);
                            float sf_sig_1_4 = approx_rcp(1.0f + _exp2_28);
                            float _tanh_29 = tanhf(sf_gk_1_4 * 0.25f);
                            float sf_aa_1_4 = 4.0f * _tanh_29 * sf_sig_1_4;
                            float _tanh_30 = tanhf(sf_uk_1_4 * 0.04f);
                            float sf_bb_1_4 = 25.0f * _tanh_30;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_4 * 768 + row_s + sf_frow_2_4)) + (0)) = (__nv_bfloat16)(sf_aa_1_4 * sf_bb_1_4);
                        }
                        int sf_tok_1_5 = 8 + lane_pair * 2 + 1;
                        int sf_frow_2_5 = row_base;
                        if (sf_tok_1_5 < num_tokens) {
                            float sf_gk_1_5 = (float)(__nv_bfloat16)red[5];
                            float sf_uk_1_5 = (float)(__nv_bfloat16)red[21];
                            float _exp2_31 = approx_exp2((-sf_gk_1_5) * 1.442695f);
                            float sf_sig_1_5 = approx_rcp(1.0f + _exp2_31);
                            float _tanh_32 = tanhf(sf_gk_1_5 * 0.25f);
                            float sf_aa_1_5 = 4.0f * _tanh_32 * sf_sig_1_5;
                            float _tanh_33 = tanhf(sf_uk_1_5 * 0.04f);
                            float sf_bb_1_5 = 25.0f * _tanh_33;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_5 * 768 + row_s + sf_frow_2_5)) + (0)) = (__nv_bfloat16)(sf_aa_1_5 * sf_bb_1_5);
                        }
                        int sf_tok_1_6 = 8 + lane_pair * 2;
                        int sf_frow_2_6 = row_base + 8;
                        if (sf_tok_1_6 < num_tokens) {
                            float sf_gk_1_6 = (float)(__nv_bfloat16)red[6];
                            float sf_uk_1_6 = (float)(__nv_bfloat16)red[22];
                            float _exp2_34 = approx_exp2((-sf_gk_1_6) * 1.442695f);
                            float sf_sig_1_6 = approx_rcp(1.0f + _exp2_34);
                            float _tanh_35 = tanhf(sf_gk_1_6 * 0.25f);
                            float sf_aa_1_6 = 4.0f * _tanh_35 * sf_sig_1_6;
                            float _tanh_36 = tanhf(sf_uk_1_6 * 0.04f);
                            float sf_bb_1_6 = 25.0f * _tanh_36;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_6 * 768 + row_s + sf_frow_2_6)) + (0)) = (__nv_bfloat16)(sf_aa_1_6 * sf_bb_1_6);
                        }
                        int sf_tok_1_7 = 8 + lane_pair * 2 + 1;
                        int sf_frow_2_7 = row_base + 8;
                        if (sf_tok_1_7 < num_tokens) {
                            float sf_gk_1_7 = (float)(__nv_bfloat16)red[7];
                            float sf_uk_1_7 = (float)(__nv_bfloat16)red[23];
                            float _exp2_37 = approx_exp2((-sf_gk_1_7) * 1.442695f);
                            float sf_sig_1_7 = approx_rcp(1.0f + _exp2_37);
                            float _tanh_38 = tanhf(sf_gk_1_7 * 0.25f);
                            float sf_aa_1_7 = 4.0f * _tanh_38 * sf_sig_1_7;
                            float _tanh_39 = tanhf(sf_uk_1_7 * 0.04f);
                            float sf_bb_1_7 = 25.0f * _tanh_39;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_7 * 768 + row_s + sf_frow_2_7)) + (0)) = (__nv_bfloat16)(sf_aa_1_7 * sf_bb_1_7);
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
                    }
                }
                if (slot == 1) {
                    if (cls == 2) {
                        int sf_tok_1_8 = 16 + lane_pair * 2;
                        int sf_frow_2_8 = row_base;
                        if (sf_tok_1_8 < num_tokens) {
                            float sf_gk_1_8 = (float)(__nv_bfloat16)red[8];
                            float sf_uk_1_8 = (float)(__nv_bfloat16)red[24];
                            float _exp2_40 = approx_exp2((-sf_gk_1_8) * 1.442695f);
                            float sf_sig_1_8 = approx_rcp(1.0f + _exp2_40);
                            float _tanh_41 = tanhf(sf_gk_1_8 * 0.25f);
                            float sf_aa_1_8 = 4.0f * _tanh_41 * sf_sig_1_8;
                            float _tanh_42 = tanhf(sf_uk_1_8 * 0.04f);
                            float sf_bb_1_8 = 25.0f * _tanh_42;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_8 * 768 + row_s + sf_frow_2_8)) + (0)) = (__nv_bfloat16)(sf_aa_1_8 * sf_bb_1_8);
                        }
                        int sf_tok_1_9 = 16 + lane_pair * 2 + 1;
                        int sf_frow_2_9 = row_base;
                        if (sf_tok_1_9 < num_tokens) {
                            float sf_gk_1_9 = (float)(__nv_bfloat16)red[9];
                            float sf_uk_1_9 = (float)(__nv_bfloat16)red[25];
                            float _exp2_43 = approx_exp2((-sf_gk_1_9) * 1.442695f);
                            float sf_sig_1_9 = approx_rcp(1.0f + _exp2_43);
                            float _tanh_44 = tanhf(sf_gk_1_9 * 0.25f);
                            float sf_aa_1_9 = 4.0f * _tanh_44 * sf_sig_1_9;
                            float _tanh_45 = tanhf(sf_uk_1_9 * 0.04f);
                            float sf_bb_1_9 = 25.0f * _tanh_45;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_9 * 768 + row_s + sf_frow_2_9)) + (0)) = (__nv_bfloat16)(sf_aa_1_9 * sf_bb_1_9);
                        }
                        int sf_tok_1_10 = 16 + lane_pair * 2;
                        int sf_frow_2_10 = row_base + 8;
                        if (sf_tok_1_10 < num_tokens) {
                            float sf_gk_1_10 = (float)(__nv_bfloat16)red[10];
                            float sf_uk_1_10 = (float)(__nv_bfloat16)red[26];
                            float _exp2_46 = approx_exp2((-sf_gk_1_10) * 1.442695f);
                            float sf_sig_1_10 = approx_rcp(1.0f + _exp2_46);
                            float _tanh_47 = tanhf(sf_gk_1_10 * 0.25f);
                            float sf_aa_1_10 = 4.0f * _tanh_47 * sf_sig_1_10;
                            float _tanh_48 = tanhf(sf_uk_1_10 * 0.04f);
                            float sf_bb_1_10 = 25.0f * _tanh_48;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_10 * 768 + row_s + sf_frow_2_10)) + (0)) = (__nv_bfloat16)(sf_aa_1_10 * sf_bb_1_10);
                        }
                        int sf_tok_1_11 = 16 + lane_pair * 2 + 1;
                        int sf_frow_2_11 = row_base + 8;
                        if (sf_tok_1_11 < num_tokens) {
                            float sf_gk_1_11 = (float)(__nv_bfloat16)red[11];
                            float sf_uk_1_11 = (float)(__nv_bfloat16)red[27];
                            float _exp2_49 = approx_exp2((-sf_gk_1_11) * 1.442695f);
                            float sf_sig_1_11 = approx_rcp(1.0f + _exp2_49);
                            float _tanh_50 = tanhf(sf_gk_1_11 * 0.25f);
                            float sf_aa_1_11 = 4.0f * _tanh_50 * sf_sig_1_11;
                            float _tanh_51 = tanhf(sf_uk_1_11 * 0.04f);
                            float sf_bb_1_11 = 25.0f * _tanh_51;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_11 * 768 + row_s + sf_frow_2_11)) + (0)) = (__nv_bfloat16)(sf_aa_1_11 * sf_bb_1_11);
                        }
                        int sf_tok_1_12 = 24 + lane_pair * 2;
                        int sf_frow_2_12 = row_base;
                        if (sf_tok_1_12 < num_tokens) {
                            float sf_gk_1_12 = (float)(__nv_bfloat16)red[12];
                            float sf_uk_1_12 = (float)(__nv_bfloat16)red[28];
                            float _exp2_52 = approx_exp2((-sf_gk_1_12) * 1.442695f);
                            float sf_sig_1_12 = approx_rcp(1.0f + _exp2_52);
                            float _tanh_53 = tanhf(sf_gk_1_12 * 0.25f);
                            float sf_aa_1_12 = 4.0f * _tanh_53 * sf_sig_1_12;
                            float _tanh_54 = tanhf(sf_uk_1_12 * 0.04f);
                            float sf_bb_1_12 = 25.0f * _tanh_54;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_12 * 768 + row_s + sf_frow_2_12)) + (0)) = (__nv_bfloat16)(sf_aa_1_12 * sf_bb_1_12);
                        }
                        int sf_tok_1_13 = 24 + lane_pair * 2 + 1;
                        int sf_frow_2_13 = row_base;
                        if (sf_tok_1_13 < num_tokens) {
                            float sf_gk_1_13 = (float)(__nv_bfloat16)red[13];
                            float sf_uk_1_13 = (float)(__nv_bfloat16)red[29];
                            float _exp2_55 = approx_exp2((-sf_gk_1_13) * 1.442695f);
                            float sf_sig_1_13 = approx_rcp(1.0f + _exp2_55);
                            float _tanh_56 = tanhf(sf_gk_1_13 * 0.25f);
                            float sf_aa_1_13 = 4.0f * _tanh_56 * sf_sig_1_13;
                            float _tanh_57 = tanhf(sf_uk_1_13 * 0.04f);
                            float sf_bb_1_13 = 25.0f * _tanh_57;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_13 * 768 + row_s + sf_frow_2_13)) + (0)) = (__nv_bfloat16)(sf_aa_1_13 * sf_bb_1_13);
                        }
                        int sf_tok_1_14 = 24 + lane_pair * 2;
                        int sf_frow_2_14 = row_base + 8;
                        if (sf_tok_1_14 < num_tokens) {
                            float sf_gk_1_14 = (float)(__nv_bfloat16)red[14];
                            float sf_uk_1_14 = (float)(__nv_bfloat16)red[30];
                            float _exp2_58 = approx_exp2((-sf_gk_1_14) * 1.442695f);
                            float sf_sig_1_14 = approx_rcp(1.0f + _exp2_58);
                            float _tanh_59 = tanhf(sf_gk_1_14 * 0.25f);
                            float sf_aa_1_14 = 4.0f * _tanh_59 * sf_sig_1_14;
                            float _tanh_60 = tanhf(sf_uk_1_14 * 0.04f);
                            float sf_bb_1_14 = 25.0f * _tanh_60;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_14 * 768 + row_s + sf_frow_2_14)) + (0)) = (__nv_bfloat16)(sf_aa_1_14 * sf_bb_1_14);
                        }
                        int sf_tok_1_15 = 24 + lane_pair * 2 + 1;
                        int sf_frow_2_15 = row_base + 8;
                        if (sf_tok_1_15 < num_tokens) {
                            float sf_gk_1_15 = (float)(__nv_bfloat16)red[15];
                            float sf_uk_1_15 = (float)(__nv_bfloat16)red[31];
                            float _exp2_61 = approx_exp2((-sf_gk_1_15) * 1.442695f);
                            float sf_sig_1_15 = approx_rcp(1.0f + _exp2_61);
                            float _tanh_62 = tanhf(sf_gk_1_15 * 0.25f);
                            float sf_aa_1_15 = 4.0f * _tanh_62 * sf_sig_1_15;
                            float _tanh_63 = tanhf(sf_uk_1_15 * 0.04f);
                            float sf_bb_1_15 = 25.0f * _tanh_63;
                            *(reinterpret_cast<__nv_bfloat16*>(out_s + (sf_tok_1_15 * 768 + row_s + sf_frow_2_15)) + (0)) = (__nv_bfloat16)(sf_aa_1_15 * sf_bb_1_15);
                        }
                    } else {
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
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
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
                    tma_3d_gmem2smem(smem_b_addr + stage * 4096, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
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
                    mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 20480);
                    stage += 1;
                    if (stage == 10) { stage = 0; _phase_a_empty ^= 1; }
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
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 67634320;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc + (32))), "r"(((init_sub) ? 0 : 1)) : "memory");
                        int _smem_desc_lo_2 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 67634320;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_acc + (64))), "r"(((init_sub) ? 0 : 1)) : "memory");
                    } else {
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    if (stage_1 == 10) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
