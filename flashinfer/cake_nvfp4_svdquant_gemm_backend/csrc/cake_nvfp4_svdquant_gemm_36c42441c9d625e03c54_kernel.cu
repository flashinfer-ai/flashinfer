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
#define TMEM_NCOLS 160
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 136
#define TMEM_TMEM_SFX_OFFSET 144
#define NUM_TMA_PIPE_STAGES 11
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 8192
#define SMEM_SMEM_W_STRIDE 19968
#define SMEM_SMEM_SFW_OFF 9216
#define SMEM_SMEM_SFW_STAGE_BYTES 1024
#define SMEM_SMEM_SFW_STRIDE 19968
#define SMEM_SMEM_X_OFF 10240
#define SMEM_SMEM_X_STAGE_BYTES 8704
#define SMEM_SMEM_X_STRIDE 19968
#define SMEM_SMEM_SFX_OFF 18944
#define SMEM_SMEM_SFX_STAGE_BYTES 2048
#define SMEM_SMEM_SFX_STRIDE 19968
#define SMEM_SMEM_L1_OFF 1024
#define SMEM_SMEM_L1_STAGE_BYTES 8192
#define SMEM_SMEM_L1_STRIDE 19968
#define SMEM_SMEM_D_OFF 10240
#define SMEM_SMEM_D_STAGE_BYTES 8704
#define SMEM_SMEM_D_STRIDE 19968
#define SMEM_SMEM_OUT_OFF 1024
#define SMEM_SMEM_OUT_STAGE_BYTES 9216
#define SMEM_SMEM_OUT_STRIDE 9216
#define SMEM_INBOX_OFF 60928
#define SMEM_INBOX_STAGE_BYTES 55296
#define SMEM_INBOX_STRIDE 55296
#define SMEM_TOTAL 220672
#define THREADS 256
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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



__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_nvfp4_svdquant_gemm_36c42441c9d625e03c54(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, const __grid_constant__ CUtensorMap L1, float* __restrict__ alpha, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int rank_tiles, int grid_n, int has_bias)
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
    #define tma_full_addr (mbar_base + 0)
    #define tma_empty_addr (mbar_base + 88)
    #define acc_full_addr (mbar_base + 176)
    #define tmem_dealloc_addr (mbar_base + 184)
    #define inbox_bar_addr (mbar_base + 192)
    #define done_bar_addr (mbar_base + 200)
    #define open_bar_addr (mbar_base + 208)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_sfw_addr = smem + 9216;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 10240);
    const int smem_x_addr = smem + 10240;
    uint8_t* smem_sfx = reinterpret_cast<uint8_t*>(smem_raw + 18944);
    const int smem_sfx_addr = smem + 18944;
    __nv_bfloat16* smem_l1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_l1_addr = smem + 1024;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_d_addr = smem + 10240;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_out_addr = smem + 1024;
    uint8_t* inbox = reinterpret_cast<uint8_t*>(smem_raw + 60928);
    const int inbox_addr = smem + 60928;
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1))) : "memory"); }
    if (warp == 3) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 11 barriers, init_count=1
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
            mbarrier_init(smem + 80, 1);
            // tma_empty: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            // inbox_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // done_bar: 1 barriers, init_count=3
            mbarrier_init(smem + 200, 3);
            // open_bar: 3 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 192, 55296);
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 160 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp == 0 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(3), "r"(160) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp == 0 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfw = taddr + 136;
    const int tmem_tmem_sfx = taddr + 144;

    // ---- Role: mma ----
    if (warp == 0) {
        { // mma_main
            int mm_rank = (int)cta_rank;
            int mm_k_tiles = K_tiles / 4;
            int mm_chunks = (rank_tiles - mm_rank + 3) / 4;
            unsigned int mma_stage = 0;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int k_tile_mma = 0; k_tile_mma < mm_k_tiles; k_tile_mma++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                unsigned int read_stage = mma_stage;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw_addr) >> 4) + (read_stage) * 1248)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx_addr) >> 4) + (read_stage) * 1248)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfx + 4), make_sf_cp_desc_lo_sbo128((((smem_sfx_addr) >> 4) + (read_stage) * 1248 + 32)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw_addr) >> 4) + (read_stage) * 1248 + 32)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 8, make_sf_cp_desc_lo_sbo128((((smem_sfx_addr) >> 4) + (read_stage) * 1248 + 64)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfx + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfx_addr) >> 4) + (read_stage) * 1248 + 64 + 32)));
                }
                int init_flag = ((k_tile_mma == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_w_addr) >> 4) & 0x3FFF) + (read_stage) * 1248);
                int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (read_stage) * 1248);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8220480U, tmem_tmem_sfw + 0, tmem_tmem_sfx + 0, ((init_flag) ? 0 : 1));
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_w_addr + 32) >> 4) & 0x3FFF) + (read_stage) * 1248);
                int _mma_b_lo_1 = make_warp_uniform((((smem_x_addr + 32) >> 4) & 0x3FFF) + (read_stage) * 1248);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8220480U, tmem_tmem_sfw + 4 + 0, tmem_tmem_sfx + 8 + 0, 1);
                    }
                }
                elect_commit(tma_empty_addr + (read_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 11) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            #pragma unroll 1
            for (unsigned int chunk_mma = 0; chunk_mma < mm_chunks; chunk_mma++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                unsigned int read_stage_1 = mma_stage;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_2 = make_warp_uniform((((smem_l1_addr) >> 4) & 0x3FFF) + (read_stage_1) * 1248);
                int _mma_b_lo_2 = make_warp_uniform((((smem_d_addr) >> 4) & 0x3FFF) + (read_stage_1) * 1248);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136447120, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136447120, 1);
                    }
                }
                elect_commit(tma_empty_addr + (read_stage_1) * 8);
                mma_stage += 1;
                if (mma_stage == 11) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(acc_full_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: idle ----
    if (warp == 1) {
        // idle — no tasks assigned
    }
    // ---- Role: load ----
    if (warp == 2) {
        { // load_main
            int ld_tile = (int)cluster_id;
            int ld_rank = (int)cta_rank;
            int ld_n0 = ld_tile * 128;
            int ld_k_tiles = K_tiles / 4;
            int ld_k_begin = ld_rank * ld_k_tiles;
            int ld_chunks = (rank_tiles - ld_rank + 3) / 4;
            unsigned int load_stage = 0;
            unsigned int _phase_tma_empty = 1;
            #pragma unroll 1
            for (unsigned int k_tile = 0; k_tile < ld_k_tiles; k_tile++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                unsigned int write_stage = load_stage;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(tma_full_addr + (write_stage) * 8, 19968);
                }
                unsigned int kt = (unsigned int)ld_k_begin + k_tile;
                if (elect_sync()) {
                    tma_3d_gmem2smem(smem_w_addr + write_stage * 19968, (&B), 0, ld_n0, kt, tma_full_addr + (write_stage) * 8);
                    tma_3d_gmem2smem(smem_sfw_addr + write_stage * 19968, (&SFB), 0, kt * 2, ld_tile, tma_full_addr + (write_stage) * 8);
                    tma_3d_gmem2smem(smem_x_addr + write_stage * 19968, (&A), 0, 0, kt, tma_full_addr + (write_stage) * 8);
                    tma_3d_gmem2smem(smem_sfx_addr + write_stage * 19968, (&SFA), 0, 0, kt * 2, tma_full_addr + (write_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
            #pragma unroll 1
            for (unsigned int chunk_i = 0; chunk_i < ld_chunks; chunk_i++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                unsigned int write_stage_1 = load_stage;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(tma_full_addr + (write_stage_1) * 8, 16896);
                }
                int rank_offset = (ld_rank + (int)chunk_i * 4) * 32;
                if (elect_sync()) {
                    tma_2d_gmem2smem(smem_l1_addr + write_stage_1 * 19968, (&L1), rank_offset, ld_n0, tma_full_addr + (write_stage_1) * 8);
                    tma_2d_gmem2smem(smem_d_addr + write_stage_1 * 19968, (&D), rank_offset, 0, tma_full_addr + (write_stage_1) * 8);
                }
                load_stage += 1;
                if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
            int ld_t = (ld_k_tiles + ld_chunks) % 11;
            int ld_open_after = ((ld_t <= 3) ? 5 - ld_t : ((ld_t <= 5) ? 10 : 16 - ld_t));
            #pragma unroll 1
            for (int _tail_open = 0; _tail_open < ld_open_after + 1; _tail_open++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                load_stage += 1;
                if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
            if (elect_sync()) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(open_bar_addr + 16), "r"((ld_rank + 1 + 0) % 4) : "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(open_bar_addr + 8), "r"((ld_rank + 1 + 1) % 4) : "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(open_bar_addr), "r"((ld_rank + 1 + 2) % 4) : "memory");
            }
            #pragma unroll 1
            for (int _tail_rest = ld_open_after + 1; _tail_rest < 11; _tail_rest++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                load_stage += 1;
                if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
        }
    }
    // ---- Role: epilogue_load ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int ep_tile = (int)cluster_id;
            int ep_rank = (int)cta_rank;
            int ep_n0 = ep_tile * 128;
            int n_local = epi_warp * 32 + (int)lane;
            int m_begin = ep_rank * 36;
            unsigned int lane_row = (unsigned int)(epi_warp * 32 << 16);
            float alpha_value = alpha[0];
            float bias_value = 0.0f;
            if (has_bias != 0) {
                float _vec_load_0[1];
                {
                    __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(bias + ep_n0 + n_local);
                    _vec_load_0[0] = __bfloat162float(_bf16_0);
                }
                bias_value = _vec_load_0[0];
            }
            unsigned int lane_off = (unsigned int)(n_local * 16);
            int peer_rank0 = (ep_rank + 1) % 4;
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(inbox_addr + 4096 + lane_off), "r"(peer_rank0));
            int peer_rank1 = (ep_rank + 1 + 1) % 4;
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(inbox_addr + 2048 + lane_off), "r"(peer_rank1));
            int peer_rank2 = (ep_rank + 1 + 2) % 4;
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(inbox_addr + lane_off), "r"(peer_rank2));
            mbarrier_wait(acc_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (warp == 4) {
                if (elect_sync()) {
                    mbarrier_arrive(inbox_bar_addr);
                }
            }
            int peer_src0 = taddr + lane_row + (unsigned int)(peer_rank0 * 36);
            float _tmem_load_0[36];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                : "r"(peer_src0));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            tmem_ld_x4(&_tmem_load_0[32], peer_src0 + 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            mbarrier_wait(open_bar_addr, 0);
            uint32_t _mbar_peer_1;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_1) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0), "r"(__float_as_uint(_tmem_load_0[0])), "r"(__float_as_uint(_tmem_load_0[1])), "r"(__float_as_uint(_tmem_load_0[2])), "r"(__float_as_uint(_tmem_load_0[3])), "r"(_mbar_peer_1) : "memory");
            uint32_t _mbar_peer_2;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_2) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 6144), "r"(__float_as_uint(_tmem_load_0[4])), "r"(__float_as_uint(_tmem_load_0[5])), "r"(__float_as_uint(_tmem_load_0[6])), "r"(__float_as_uint(_tmem_load_0[7])), "r"(_mbar_peer_2) : "memory");
            uint32_t _mbar_peer_3;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_3) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 12288), "r"(__float_as_uint(_tmem_load_0[8])), "r"(__float_as_uint(_tmem_load_0[9])), "r"(__float_as_uint(_tmem_load_0[10])), "r"(__float_as_uint(_tmem_load_0[11])), "r"(_mbar_peer_3) : "memory");
            uint32_t _mbar_peer_4;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_4) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 18432), "r"(__float_as_uint(_tmem_load_0[12])), "r"(__float_as_uint(_tmem_load_0[13])), "r"(__float_as_uint(_tmem_load_0[14])), "r"(__float_as_uint(_tmem_load_0[15])), "r"(_mbar_peer_4) : "memory");
            uint32_t _mbar_peer_5;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_5) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 24576), "r"(__float_as_uint(_tmem_load_0[16])), "r"(__float_as_uint(_tmem_load_0[17])), "r"(__float_as_uint(_tmem_load_0[18])), "r"(__float_as_uint(_tmem_load_0[19])), "r"(_mbar_peer_5) : "memory");
            uint32_t _mbar_peer_6;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_6) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 30720), "r"(__float_as_uint(_tmem_load_0[20])), "r"(__float_as_uint(_tmem_load_0[21])), "r"(__float_as_uint(_tmem_load_0[22])), "r"(__float_as_uint(_tmem_load_0[23])), "r"(_mbar_peer_6) : "memory");
            uint32_t _mbar_peer_7;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_7) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 36864), "r"(__float_as_uint(_tmem_load_0[24])), "r"(__float_as_uint(_tmem_load_0[25])), "r"(__float_as_uint(_tmem_load_0[26])), "r"(__float_as_uint(_tmem_load_0[27])), "r"(_mbar_peer_7) : "memory");
            uint32_t _mbar_peer_8;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_8) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 43008), "r"(__float_as_uint(_tmem_load_0[28])), "r"(__float_as_uint(_tmem_load_0[29])), "r"(__float_as_uint(_tmem_load_0[30])), "r"(__float_as_uint(_tmem_load_0[31])), "r"(_mbar_peer_8) : "memory");
            uint32_t _mbar_peer_9;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_9) : "r"(inbox_bar_addr), "r"(peer_rank0));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_0 + 49152), "r"(__float_as_uint(_tmem_load_0[32])), "r"(__float_as_uint(_tmem_load_0[33])), "r"(__float_as_uint(_tmem_load_0[34])), "r"(__float_as_uint(_tmem_load_0[35])), "r"(_mbar_peer_9) : "memory");
            int peer_src1 = taddr + lane_row + (unsigned int)(peer_rank1 * 36);
            float _tmem_load_1[36];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                : "r"(peer_src1));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            tmem_ld_x4(&_tmem_load_1[32], peer_src1 + 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            mbarrier_wait(open_bar_addr + 8, 0);
            uint32_t _mbar_peer_10;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_10) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1), "r"(__float_as_uint(_tmem_load_1[0])), "r"(__float_as_uint(_tmem_load_1[1])), "r"(__float_as_uint(_tmem_load_1[2])), "r"(__float_as_uint(_tmem_load_1[3])), "r"(_mbar_peer_10) : "memory");
            uint32_t _mbar_peer_11;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_11) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(_tmem_load_1[4])), "r"(__float_as_uint(_tmem_load_1[5])), "r"(__float_as_uint(_tmem_load_1[6])), "r"(__float_as_uint(_tmem_load_1[7])), "r"(_mbar_peer_11) : "memory");
            uint32_t _mbar_peer_12;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_12) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 12288), "r"(__float_as_uint(_tmem_load_1[8])), "r"(__float_as_uint(_tmem_load_1[9])), "r"(__float_as_uint(_tmem_load_1[10])), "r"(__float_as_uint(_tmem_load_1[11])), "r"(_mbar_peer_12) : "memory");
            uint32_t _mbar_peer_13;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_13) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 18432), "r"(__float_as_uint(_tmem_load_1[12])), "r"(__float_as_uint(_tmem_load_1[13])), "r"(__float_as_uint(_tmem_load_1[14])), "r"(__float_as_uint(_tmem_load_1[15])), "r"(_mbar_peer_13) : "memory");
            uint32_t _mbar_peer_14;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_14) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 24576), "r"(__float_as_uint(_tmem_load_1[16])), "r"(__float_as_uint(_tmem_load_1[17])), "r"(__float_as_uint(_tmem_load_1[18])), "r"(__float_as_uint(_tmem_load_1[19])), "r"(_mbar_peer_14) : "memory");
            uint32_t _mbar_peer_15;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_15) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 30720), "r"(__float_as_uint(_tmem_load_1[20])), "r"(__float_as_uint(_tmem_load_1[21])), "r"(__float_as_uint(_tmem_load_1[22])), "r"(__float_as_uint(_tmem_load_1[23])), "r"(_mbar_peer_15) : "memory");
            uint32_t _mbar_peer_16;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_16) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 36864), "r"(__float_as_uint(_tmem_load_1[24])), "r"(__float_as_uint(_tmem_load_1[25])), "r"(__float_as_uint(_tmem_load_1[26])), "r"(__float_as_uint(_tmem_load_1[27])), "r"(_mbar_peer_16) : "memory");
            uint32_t _mbar_peer_17;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_17) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 43008), "r"(__float_as_uint(_tmem_load_1[28])), "r"(__float_as_uint(_tmem_load_1[29])), "r"(__float_as_uint(_tmem_load_1[30])), "r"(__float_as_uint(_tmem_load_1[31])), "r"(_mbar_peer_17) : "memory");
            uint32_t _mbar_peer_18;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_18) : "r"(inbox_bar_addr), "r"(peer_rank1));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_1 + 49152), "r"(__float_as_uint(_tmem_load_1[32])), "r"(__float_as_uint(_tmem_load_1[33])), "r"(__float_as_uint(_tmem_load_1[34])), "r"(__float_as_uint(_tmem_load_1[35])), "r"(_mbar_peer_18) : "memory");
            int peer_src2 = taddr + lane_row + (unsigned int)(peer_rank2 * 36);
            float _tmem_load_2[36];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                : "r"(peer_src2));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            tmem_ld_x4(&_tmem_load_2[32], peer_src2 + 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            mbarrier_wait(open_bar_addr + 16, 0);
            uint32_t _mbar_peer_19;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_19) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2), "r"(__float_as_uint(_tmem_load_2[0])), "r"(__float_as_uint(_tmem_load_2[1])), "r"(__float_as_uint(_tmem_load_2[2])), "r"(__float_as_uint(_tmem_load_2[3])), "r"(_mbar_peer_19) : "memory");
            uint32_t _mbar_peer_20;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_20) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 6144), "r"(__float_as_uint(_tmem_load_2[4])), "r"(__float_as_uint(_tmem_load_2[5])), "r"(__float_as_uint(_tmem_load_2[6])), "r"(__float_as_uint(_tmem_load_2[7])), "r"(_mbar_peer_20) : "memory");
            uint32_t _mbar_peer_21;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_21) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 12288), "r"(__float_as_uint(_tmem_load_2[8])), "r"(__float_as_uint(_tmem_load_2[9])), "r"(__float_as_uint(_tmem_load_2[10])), "r"(__float_as_uint(_tmem_load_2[11])), "r"(_mbar_peer_21) : "memory");
            uint32_t _mbar_peer_22;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_22) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 18432), "r"(__float_as_uint(_tmem_load_2[12])), "r"(__float_as_uint(_tmem_load_2[13])), "r"(__float_as_uint(_tmem_load_2[14])), "r"(__float_as_uint(_tmem_load_2[15])), "r"(_mbar_peer_22) : "memory");
            uint32_t _mbar_peer_23;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_23) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 24576), "r"(__float_as_uint(_tmem_load_2[16])), "r"(__float_as_uint(_tmem_load_2[17])), "r"(__float_as_uint(_tmem_load_2[18])), "r"(__float_as_uint(_tmem_load_2[19])), "r"(_mbar_peer_23) : "memory");
            uint32_t _mbar_peer_24;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_24) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 30720), "r"(__float_as_uint(_tmem_load_2[20])), "r"(__float_as_uint(_tmem_load_2[21])), "r"(__float_as_uint(_tmem_load_2[22])), "r"(__float_as_uint(_tmem_load_2[23])), "r"(_mbar_peer_24) : "memory");
            uint32_t _mbar_peer_25;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_25) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 36864), "r"(__float_as_uint(_tmem_load_2[24])), "r"(__float_as_uint(_tmem_load_2[25])), "r"(__float_as_uint(_tmem_load_2[26])), "r"(__float_as_uint(_tmem_load_2[27])), "r"(_mbar_peer_25) : "memory");
            uint32_t _mbar_peer_26;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_26) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 43008), "r"(__float_as_uint(_tmem_load_2[28])), "r"(__float_as_uint(_tmem_load_2[29])), "r"(__float_as_uint(_tmem_load_2[30])), "r"(__float_as_uint(_tmem_load_2[31])), "r"(_mbar_peer_26) : "memory");
            uint32_t _mbar_peer_27;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_27) : "r"(inbox_bar_addr), "r"(peer_rank2));
            asm volatile(
                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                :: "r"(_mapa_2 + 49152), "r"(__float_as_uint(_tmem_load_2[32])), "r"(__float_as_uint(_tmem_load_2[33])), "r"(__float_as_uint(_tmem_load_2[34])), "r"(__float_as_uint(_tmem_load_2[35])), "r"(_mbar_peer_27) : "memory");
            int own_src = taddr + lane_row + (unsigned int)(ep_rank * 36);
            float _tmem_load_3[36];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                : "r"(own_src));
            tmem_ld_x4(&_tmem_load_3[32], own_src + 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            mbarrier_wait_cluster_hint(inbox_bar_addr, 0, 10000000);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (warp == 4) {
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(done_bar_addr), "r"(peer_rank0) : "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(done_bar_addr), "r"(peer_rank1) : "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.relaxed.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(done_bar_addr), "r"(peer_rank2) : "memory");
                }
            }
            float acc[36];
            unsigned int mail_words[4];
            if (ep_rank == 0) {
                #pragma unroll
                for (int i = 0; i < 36; i++) {
                    acc[i] = _tmem_load_3[i];
                }
            } else {
                unsigned int slot_off0 = (unsigned int)((-ep_rank + 3) % 4 * 2048);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0));
                acc[0] = __uint_as_float(mail_words[0]);
                acc[1] = __uint_as_float(mail_words[1]);
                acc[2] = __uint_as_float(mail_words[2]);
                acc[3] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 6144));
                acc[4] = __uint_as_float(mail_words[0]);
                acc[5] = __uint_as_float(mail_words[1]);
                acc[6] = __uint_as_float(mail_words[2]);
                acc[7] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 12288));
                acc[8] = __uint_as_float(mail_words[0]);
                acc[9] = __uint_as_float(mail_words[1]);
                acc[10] = __uint_as_float(mail_words[2]);
                acc[11] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 18432));
                acc[12] = __uint_as_float(mail_words[0]);
                acc[13] = __uint_as_float(mail_words[1]);
                acc[14] = __uint_as_float(mail_words[2]);
                acc[15] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 24576));
                acc[16] = __uint_as_float(mail_words[0]);
                acc[17] = __uint_as_float(mail_words[1]);
                acc[18] = __uint_as_float(mail_words[2]);
                acc[19] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 30720));
                acc[20] = __uint_as_float(mail_words[0]);
                acc[21] = __uint_as_float(mail_words[1]);
                acc[22] = __uint_as_float(mail_words[2]);
                acc[23] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 36864));
                acc[24] = __uint_as_float(mail_words[0]);
                acc[25] = __uint_as_float(mail_words[1]);
                acc[26] = __uint_as_float(mail_words[2]);
                acc[27] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 43008));
                acc[28] = __uint_as_float(mail_words[0]);
                acc[29] = __uint_as_float(mail_words[1]);
                acc[30] = __uint_as_float(mail_words[2]);
                acc[31] = __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off0 + 49152));
                acc[32] = __uint_as_float(mail_words[0]);
                acc[33] = __uint_as_float(mail_words[1]);
                acc[34] = __uint_as_float(mail_words[2]);
                acc[35] = __uint_as_float(mail_words[3]);
            }
            if (ep_rank == 1) {
                #pragma unroll
                for (int i_1 = 0; i_1 < 36; i_1++) {
                    acc[i_1] = acc[i_1] + _tmem_load_3[i_1];
                }
            } else {
                unsigned int slot_off1 = (unsigned int)((1 - ep_rank + 3) % 4 * 2048);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1));
                acc[0] = acc[0] + __uint_as_float(mail_words[0]);
                acc[1] = acc[1] + __uint_as_float(mail_words[1]);
                acc[2] = acc[2] + __uint_as_float(mail_words[2]);
                acc[3] = acc[3] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 6144));
                acc[4] = acc[4] + __uint_as_float(mail_words[0]);
                acc[5] = acc[5] + __uint_as_float(mail_words[1]);
                acc[6] = acc[6] + __uint_as_float(mail_words[2]);
                acc[7] = acc[7] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 12288));
                acc[8] = acc[8] + __uint_as_float(mail_words[0]);
                acc[9] = acc[9] + __uint_as_float(mail_words[1]);
                acc[10] = acc[10] + __uint_as_float(mail_words[2]);
                acc[11] = acc[11] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 18432));
                acc[12] = acc[12] + __uint_as_float(mail_words[0]);
                acc[13] = acc[13] + __uint_as_float(mail_words[1]);
                acc[14] = acc[14] + __uint_as_float(mail_words[2]);
                acc[15] = acc[15] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 24576));
                acc[16] = acc[16] + __uint_as_float(mail_words[0]);
                acc[17] = acc[17] + __uint_as_float(mail_words[1]);
                acc[18] = acc[18] + __uint_as_float(mail_words[2]);
                acc[19] = acc[19] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 30720));
                acc[20] = acc[20] + __uint_as_float(mail_words[0]);
                acc[21] = acc[21] + __uint_as_float(mail_words[1]);
                acc[22] = acc[22] + __uint_as_float(mail_words[2]);
                acc[23] = acc[23] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 36864));
                acc[24] = acc[24] + __uint_as_float(mail_words[0]);
                acc[25] = acc[25] + __uint_as_float(mail_words[1]);
                acc[26] = acc[26] + __uint_as_float(mail_words[2]);
                acc[27] = acc[27] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 43008));
                acc[28] = acc[28] + __uint_as_float(mail_words[0]);
                acc[29] = acc[29] + __uint_as_float(mail_words[1]);
                acc[30] = acc[30] + __uint_as_float(mail_words[2]);
                acc[31] = acc[31] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off1 + 49152));
                acc[32] = acc[32] + __uint_as_float(mail_words[0]);
                acc[33] = acc[33] + __uint_as_float(mail_words[1]);
                acc[34] = acc[34] + __uint_as_float(mail_words[2]);
                acc[35] = acc[35] + __uint_as_float(mail_words[3]);
            }
            if (ep_rank == 2) {
                #pragma unroll
                for (int i_2 = 0; i_2 < 36; i_2++) {
                    acc[i_2] = acc[i_2] + _tmem_load_3[i_2];
                }
            } else {
                unsigned int slot_off2 = (unsigned int)((2 - ep_rank + 3) % 4 * 2048);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2));
                acc[0] = acc[0] + __uint_as_float(mail_words[0]);
                acc[1] = acc[1] + __uint_as_float(mail_words[1]);
                acc[2] = acc[2] + __uint_as_float(mail_words[2]);
                acc[3] = acc[3] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 6144));
                acc[4] = acc[4] + __uint_as_float(mail_words[0]);
                acc[5] = acc[5] + __uint_as_float(mail_words[1]);
                acc[6] = acc[6] + __uint_as_float(mail_words[2]);
                acc[7] = acc[7] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 12288));
                acc[8] = acc[8] + __uint_as_float(mail_words[0]);
                acc[9] = acc[9] + __uint_as_float(mail_words[1]);
                acc[10] = acc[10] + __uint_as_float(mail_words[2]);
                acc[11] = acc[11] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 18432));
                acc[12] = acc[12] + __uint_as_float(mail_words[0]);
                acc[13] = acc[13] + __uint_as_float(mail_words[1]);
                acc[14] = acc[14] + __uint_as_float(mail_words[2]);
                acc[15] = acc[15] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 24576));
                acc[16] = acc[16] + __uint_as_float(mail_words[0]);
                acc[17] = acc[17] + __uint_as_float(mail_words[1]);
                acc[18] = acc[18] + __uint_as_float(mail_words[2]);
                acc[19] = acc[19] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 30720));
                acc[20] = acc[20] + __uint_as_float(mail_words[0]);
                acc[21] = acc[21] + __uint_as_float(mail_words[1]);
                acc[22] = acc[22] + __uint_as_float(mail_words[2]);
                acc[23] = acc[23] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 36864));
                acc[24] = acc[24] + __uint_as_float(mail_words[0]);
                acc[25] = acc[25] + __uint_as_float(mail_words[1]);
                acc[26] = acc[26] + __uint_as_float(mail_words[2]);
                acc[27] = acc[27] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 43008));
                acc[28] = acc[28] + __uint_as_float(mail_words[0]);
                acc[29] = acc[29] + __uint_as_float(mail_words[1]);
                acc[30] = acc[30] + __uint_as_float(mail_words[2]);
                acc[31] = acc[31] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off2 + 49152));
                acc[32] = acc[32] + __uint_as_float(mail_words[0]);
                acc[33] = acc[33] + __uint_as_float(mail_words[1]);
                acc[34] = acc[34] + __uint_as_float(mail_words[2]);
                acc[35] = acc[35] + __uint_as_float(mail_words[3]);
            }
            if (ep_rank == 3) {
                #pragma unroll
                for (int i_3 = 0; i_3 < 36; i_3++) {
                    acc[i_3] = acc[i_3] + _tmem_load_3[i_3];
                }
            } else {
                unsigned int slot_off3 = (unsigned int)((3 - ep_rank + 3) % 4 * 2048);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3));
                acc[0] = acc[0] + __uint_as_float(mail_words[0]);
                acc[1] = acc[1] + __uint_as_float(mail_words[1]);
                acc[2] = acc[2] + __uint_as_float(mail_words[2]);
                acc[3] = acc[3] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 6144));
                acc[4] = acc[4] + __uint_as_float(mail_words[0]);
                acc[5] = acc[5] + __uint_as_float(mail_words[1]);
                acc[6] = acc[6] + __uint_as_float(mail_words[2]);
                acc[7] = acc[7] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 12288));
                acc[8] = acc[8] + __uint_as_float(mail_words[0]);
                acc[9] = acc[9] + __uint_as_float(mail_words[1]);
                acc[10] = acc[10] + __uint_as_float(mail_words[2]);
                acc[11] = acc[11] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 18432));
                acc[12] = acc[12] + __uint_as_float(mail_words[0]);
                acc[13] = acc[13] + __uint_as_float(mail_words[1]);
                acc[14] = acc[14] + __uint_as_float(mail_words[2]);
                acc[15] = acc[15] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 24576));
                acc[16] = acc[16] + __uint_as_float(mail_words[0]);
                acc[17] = acc[17] + __uint_as_float(mail_words[1]);
                acc[18] = acc[18] + __uint_as_float(mail_words[2]);
                acc[19] = acc[19] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 30720));
                acc[20] = acc[20] + __uint_as_float(mail_words[0]);
                acc[21] = acc[21] + __uint_as_float(mail_words[1]);
                acc[22] = acc[22] + __uint_as_float(mail_words[2]);
                acc[23] = acc[23] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 36864));
                acc[24] = acc[24] + __uint_as_float(mail_words[0]);
                acc[25] = acc[25] + __uint_as_float(mail_words[1]);
                acc[26] = acc[26] + __uint_as_float(mail_words[2]);
                acc[27] = acc[27] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 43008));
                acc[28] = acc[28] + __uint_as_float(mail_words[0]);
                acc[29] = acc[29] + __uint_as_float(mail_words[1]);
                acc[30] = acc[30] + __uint_as_float(mail_words[2]);
                acc[31] = acc[31] + __uint_as_float(mail_words[3]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mail_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mail_words[(0) + 3]))
                    : "r"(inbox_addr + lane_off + slot_off3 + 49152));
                acc[32] = acc[32] + __uint_as_float(mail_words[0]);
                acc[33] = acc[33] + __uint_as_float(mail_words[1]);
                acc[34] = acc[34] + __uint_as_float(mail_words[2]);
                acc[35] = acc[35] + __uint_as_float(mail_words[3]);
            }
            float result[36];
            #pragma unroll
            for (int i_4 = 0; i_4 < 36; i_4++) {
                result[i_4] = __fmaf_rn(acc[i_4], alpha_value, bias_value);
            }
            unsigned int result_bf16[18];
            #pragma unroll
            for (int _lp = 0; _lp < 18; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                result_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int lane_odd = (unsigned int)lane & 1;
            unsigned int swap_mask = -lane_odd;
            unsigned int half_shift = lane_odd * 16;
            unsigned int row_base = smem_out_addr + lane_odd * 256 + (unsigned int)n_local / 2 * 4;
            #pragma unroll
            for (int w = 0; w < 18; w++) {
                unsigned int mine = result_bf16[w];
                unsigned int theirs = __shfl_xor_sync(0xFFFFFFFF, mine, 1);
                unsigned int swap = (mine ^ theirs) & swap_mask;
                unsigned int first = mine ^ swap;
                unsigned int second = theirs ^ swap;
                unsigned int word = first >> half_shift & 65535 | (second >> half_shift & 65535) << 16;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(row_base + (unsigned int)(w * 512)), "r"(word) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    tma_store_2d((&out), ep_n0, m_begin, smem_out_addr);
                }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            mbarrier_wait(done_bar_addr, 0);
        }
    }

    // Cleanup
}

} // extern "C"
