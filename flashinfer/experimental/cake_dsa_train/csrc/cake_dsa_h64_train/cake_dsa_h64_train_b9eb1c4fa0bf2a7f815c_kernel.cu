/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define TMEM_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 73728
#define SMEM_Q_K_STRIDE 73728
#define SMEM_Q_MN_OFF 1024
#define SMEM_Q_MN_STAGE_BYTES 16384
#define SMEM_Q_MN_STRIDE 16384
#define SMEM_QR_MN_OFF 66560
#define SMEM_QR_MN_STAGE_BYTES 8192
#define SMEM_QR_MN_STRIDE 8192
#define SMEM_DO_K_OFF 74752
#define SMEM_DO_K_STAGE_BYTES 65536
#define SMEM_DO_K_STRIDE 65536
#define SMEM_DO_MN_OFF 74752
#define SMEM_DO_MN_STAGE_BYTES 16384
#define SMEM_DO_MN_STRIDE 16384
#define SMEM_K_K_OFF 140288
#define SMEM_K_K_STAGE_BYTES 73728
#define SMEM_K_K_STRIDE 73728
#define SMEM_V_K_OFF 140288
#define SMEM_V_K_STAGE_BYTES 65536
#define SMEM_V_K_STRIDE 65536
#define SMEM_K_MN_OFF 140288
#define SMEM_K_MN_STAGE_BYTES 16384
#define SMEM_K_MN_STRIDE 16384
#define SMEM_KR_MN_OFF 205824
#define SMEM_KR_MN_STAGE_BYTES 8192
#define SMEM_KR_MN_STRIDE 8192
#define SMEM_K_BLK_OFF 140288
#define SMEM_K_BLK_STAGE_BYTES 8192
#define SMEM_K_BLK_STRIDE 8192
#define SMEM_P_K_OFF 214016
#define SMEM_P_K_STAGE_BYTES 8192
#define SMEM_P_K_STRIDE 8192
#define SMEM_DS_K_OFF 222208
#define SMEM_DS_K_STAGE_BYTES 8192
#define SMEM_DS_K_STRIDE 8192
#define SMEM_DS_MN_OFF 222208
#define SMEM_DS_MN_STAGE_BYTES 8192
#define SMEM_DS_MN_STRIDE 8192
#define SMEM_STATS_OFF 230400
#define SMEM_STATS_STAGE_BYTES 512
#define SMEM_STATS_STRIDE 512
#define SMEM_TILE_IDX_OFF 230912
#define SMEM_TILE_IDX_STAGE_BYTES 640
#define SMEM_TILE_IDX_STRIDE 640
#define SMEM_VALIDITY8_OFF 230912
#define SMEM_VALIDITY8_STAGE_BYTES 640
#define SMEM_VALIDITY8_STRIDE 640
#define SMEM_VALIDITY32_OFF 230912
#define SMEM_VALIDITY32_STAGE_BYTES 640
#define SMEM_VALIDITY32_STRIDE 640
#define SMEM_BLOCKS_WORD_OFF 231552
#define SMEM_BLOCKS_WORD_STAGE_BYTES 4
#define SMEM_BLOCKS_WORD_STRIDE 4
#define SMEM_TOTAL 231680
#define THREADS 640

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





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(640, 1) void
kernel_cake_dsa_h64_train_b9eb1c4fa0bf2a7f815c(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap dout, const __grid_constant__ CUtensorMap dq_latent, const __grid_constant__ CUtensorMap dq_rope, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap k_rope, float* __restrict__ lse, float* __restrict__ delta, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, int* __restrict__ indices, int* __restrict__ topk_length, float* __restrict__ dkv_f32, float* __restrict__ dkr_f32, int dkv_stride, int dkr_stride, int dkr_col0, int* __restrict__ dkv_dst_map, int dkv_has_map, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int pass_lo, int pass_hi, int dq_mode, float* __restrict__ dq_partial, int* __restrict__ key_scratch, int* __restrict__ pass_counts)
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
    #define qdo_full_addr (mbar_base + 0)
    #define stats_full_addr (mbar_base + 8)
    #define blocks_ready_addr (mbar_base + 16)
    #define dq_done_addr (mbar_base + 24)
    #define idx_full_addr (mbar_base + 32)
    #define idx_free_addr (mbar_base + 48)
    #define k_full_addr (mbar_base + 64)
    #define k_free_addr (mbar_base + 72)
    #define s_full_addr (mbar_base + 80)
    #define s_free_addr (mbar_base + 88)
    #define dp_full_addr (mbar_base + 96)
    #define dp_free_addr (mbar_base + 104)
    #define p_full_addr (mbar_base + 112)
    #define p_free_addr (mbar_base + 120)
    #define ds_full_addr (mbar_base + 128)
    #define ds_free_addr (mbar_base + 136)
    #define dkr_full_addr (mbar_base + 144)
    #define dkr_drained_addr (mbar_base + 152)
    #define dkv_a_full_addr (mbar_base + 160)
    #define dkv_a_drained_addr (mbar_base + 168)
    #define dkv_b_full_addr (mbar_base + 176)
    #define dkv_b_drained_addr (mbar_base + 184)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_mn_addr = smem + 1024;
    __nv_bfloat16* qr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int qr_mn_addr = smem + 66560;
    __nv_bfloat16* do_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int do_k_addr = smem + 74752;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int do_mn_addr = smem + 74752;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_k_addr = smem + 140288;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int v_k_addr = smem + 140288;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_mn_addr = smem + 140288;
    __nv_bfloat16* kr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int kr_mn_addr = smem + 205824;
    __nv_bfloat16* k_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_blk_addr = smem + 140288;
    __nv_bfloat16* p_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int p_k_addr = smem + 214016;
    __nv_bfloat16* ds_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 222208);
    const int ds_k_addr = smem + 222208;
    __nv_bfloat16* ds_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 222208);
    const int ds_mn_addr = smem + 222208;
    float* stats = reinterpret_cast<float*>(smem_raw + 230400);
    const int stats_addr = smem + 230400;
    int* tile_idx = reinterpret_cast<int*>(smem_raw + 230912);
    const int tile_idx_addr = smem + 230912;
    uint8_t* validity8 = reinterpret_cast<uint8_t*>(smem_raw + 230912);
    const int validity8_addr = smem + 230912;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 230912);
    const int validity32_addr = smem + 230912;
    int* blocks_word = reinterpret_cast<int*>(smem_raw + 231552);
    const int blocks_word_addr = smem + 231552;
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dout))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k_rope))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_latent))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_rope))) : "memory"); }

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // qdo_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // stats_full: 1 barriers, init_count=32
            mbarrier_init(smem + 8, 32);
            // blocks_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // dq_done: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // idx_full: 2 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            // idx_free: 2 barriers, init_count=16
            mbarrier_init(smem + 48, 16);
            mbarrier_init(smem + 56, 16);
            // k_full: 1 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            // k_free: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // s_free: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // dp_free: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // p_free: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // ds_full: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // ds_free: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // dkr_full: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // dkr_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            // dkv_a_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // dkv_a_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            // dkv_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // dkv_b_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 4) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: gather ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // gather_main
            int pw = warp;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks = blocks_word[0];
            if (elect_sync()) {
                #pragma unroll 1
                for (int i = 0; i < blocks; i++) {
                    int slot = i % 2;
                    mbarrier_wait(idx_full_addr + (slot) * 8, i / 2 & 1);
                    int rows[16];
                    rows[0] = tile_idx[slot * 80 + pw * 4];
                    rows[1] = tile_idx[slot * 80 + pw * 4 + 1];
                    rows[2] = tile_idx[slot * 80 + pw * 4 + 2];
                    rows[3] = tile_idx[slot * 80 + pw * 4 + 3];
                    rows[4] = tile_idx[slot * 80 + 16 + pw * 4];
                    rows[5] = tile_idx[slot * 80 + 16 + pw * 4 + 1];
                    rows[6] = tile_idx[slot * 80 + 16 + pw * 4 + 2];
                    rows[7] = tile_idx[slot * 80 + 16 + pw * 4 + 3];
                    rows[8] = tile_idx[slot * 80 + 32 + pw * 4];
                    rows[9] = tile_idx[slot * 80 + 32 + pw * 4 + 1];
                    rows[10] = tile_idx[slot * 80 + 32 + pw * 4 + 2];
                    rows[11] = tile_idx[slot * 80 + 32 + pw * 4 + 3];
                    rows[12] = tile_idx[slot * 80 + 48 + pw * 4];
                    rows[13] = tile_idx[slot * 80 + 48 + pw * 4 + 1];
                    rows[14] = tile_idx[slot * 80 + 48 + pw * 4 + 2];
                    rows[15] = tile_idx[slot * 80 + 48 + pw * 4 + 3];
                    mbarrier_arrive(idx_free_addr + (slot) * 8);
                    mbarrier_wait(k_free_addr, i & 1 ^ 1);
                    mbarrier_arrive_expect_tx(k_full_addr, 18432);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + (unsigned int)(pw * 512)), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 2048 + (unsigned int)(pw * 512)), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 4096 + (unsigned int)(pw * 512)), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 6144 + (unsigned int)(pw * 512)), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // compute_main
            int token = token_base + token_step * blockIdx.x;
            int lane_0 = lane;
            int w = warp - 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_1 = blocks_word[0];
            mbarrier_wait(stats_full_addr, 0);
            int h0 = w * 16 + lane_0 / 4;
            float lse0 = stats[h0];
            float lse1 = stats[h0 + 8];
            float dl0 = stats[64 + h0];
            float dl1 = stats[64 + h0 + 8];
            int kb = 2 * (lane_0 % 4);
            int r8 = lane_0 % 8;
            int m4 = lane_0 / 8;
            int kbase = m4 / 2 * 8 + r8;
            int hchunk = (2 * w + m4 % 2 ^ r8) * 16;
            #pragma unroll 1
            for (int i_1 = 0; i_1 < blocks_1; i_1++) {
                unsigned int par = i_1 & 1;
                int slot_1 = i_1 % 2;
                mbarrier_wait(idx_full_addr + (slot_1) * 8, i_1 / 2 & 1);
                unsigned int v0 = validity32[slot_1 * 80 + 64];
                unsigned int v1 = validity32[slot_1 * 80 + 65];
                if (lane_0 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_1) * 8);
                }
                mbarrier_wait(s_full_addr, par);
                float s[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s[0])), "=r"(*reinterpret_cast<uint32_t*>(&s[1])), "=r"(*reinterpret_cast<uint32_t*>(&s[2])), "=r"(*reinterpret_cast<uint32_t*>(&s[3])), "=r"(*reinterpret_cast<uint32_t*>(&s[4])), "=r"(*reinterpret_cast<uint32_t*>(&s[5])), "=r"(*reinterpret_cast<uint32_t*>(&s[6])), "=r"(*reinterpret_cast<uint32_t*>(&s[7])), "=r"(*reinterpret_cast<uint32_t*>(&s[8])), "=r"(*reinterpret_cast<uint32_t*>(&s[9])), "=r"(*reinterpret_cast<uint32_t*>(&s[10])), "=r"(*reinterpret_cast<uint32_t*>(&s[11])), "=r"(*reinterpret_cast<uint32_t*>(&s[12])), "=r"(*reinterpret_cast<uint32_t*>(&s[13])), "=r"(*reinterpret_cast<uint32_t*>(&s[14])), "=r"(*reinterpret_cast<uint32_t*>(&s[15])), "=r"(*reinterpret_cast<uint32_t*>(&s[16])), "=r"(*reinterpret_cast<uint32_t*>(&s[17])), "=r"(*reinterpret_cast<uint32_t*>(&s[18])), "=r"(*reinterpret_cast<uint32_t*>(&s[19])), "=r"(*reinterpret_cast<uint32_t*>(&s[20])), "=r"(*reinterpret_cast<uint32_t*>(&s[21])), "=r"(*reinterpret_cast<uint32_t*>(&s[22])), "=r"(*reinterpret_cast<uint32_t*>(&s[23])), "=r"(*reinterpret_cast<uint32_t*>(&s[24])), "=r"(*reinterpret_cast<uint32_t*>(&s[25])), "=r"(*reinterpret_cast<uint32_t*>(&s[26])), "=r"(*reinterpret_cast<uint32_t*>(&s[27])), "=r"(*reinterpret_cast<uint32_t*>(&s[28])), "=r"(*reinterpret_cast<uint32_t*>(&s[29])), "=r"(*reinterpret_cast<uint32_t*>(&s[30])), "=r"(*reinterpret_cast<uint32_t*>(&s[31]))
                    : "r"(tmem_tmem));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float p[32];
                float _fma_0 = __fmaf_rn(s[0], scale_log2, -lse0);
                float _exp2_0 = approx_exp2(_fma_0);
                p[0] = _exp2_0;
                float _fma_1 = __fmaf_rn(s[2], scale_log2, -lse1);
                float _exp2_1 = approx_exp2(_fma_1);
                p[2] = _exp2_1;
                unsigned int bit = ((1) ? v0 : v1) >> (unsigned int)kb & 1;
                if (bit == 0) {
                    p[0] = 0.0f;
                    p[2] = 0.0f;
                }
                float _fma_2 = __fmaf_rn(s[1], scale_log2, -lse0);
                float _exp2_2 = approx_exp2(_fma_2);
                p[1] = _exp2_2;
                float _fma_3 = __fmaf_rn(s[3], scale_log2, -lse1);
                float _exp2_3 = approx_exp2(_fma_3);
                p[3] = _exp2_3;
                unsigned int bit_0 = ((1) ? v0 : v1) >> (unsigned int)(1 + kb) & 1;
                if (bit_0 == 0) {
                    p[1] = 0.0f;
                    p[3] = 0.0f;
                }
                float _fma_4 = __fmaf_rn(s[4], scale_log2, -lse0);
                float _exp2_4 = approx_exp2(_fma_4);
                p[4] = _exp2_4;
                float _fma_5 = __fmaf_rn(s[6], scale_log2, -lse1);
                float _exp2_5 = approx_exp2(_fma_5);
                p[6] = _exp2_5;
                unsigned int bit_1 = ((1) ? v0 : v1) >> (unsigned int)(8 + kb) & 1;
                if (bit_1 == 0) {
                    p[4] = 0.0f;
                    p[6] = 0.0f;
                }
                float _fma_6 = __fmaf_rn(s[5], scale_log2, -lse0);
                float _exp2_6 = approx_exp2(_fma_6);
                p[5] = _exp2_6;
                float _fma_7 = __fmaf_rn(s[7], scale_log2, -lse1);
                float _exp2_7 = approx_exp2(_fma_7);
                p[7] = _exp2_7;
                unsigned int bit_2 = ((1) ? v0 : v1) >> (unsigned int)(9 + kb) & 1;
                if (bit_2 == 0) {
                    p[5] = 0.0f;
                    p[7] = 0.0f;
                }
                float _fma_8 = __fmaf_rn(s[8], scale_log2, -lse0);
                float _exp2_8 = approx_exp2(_fma_8);
                p[8] = _exp2_8;
                float _fma_9 = __fmaf_rn(s[10], scale_log2, -lse1);
                float _exp2_9 = approx_exp2(_fma_9);
                p[10] = _exp2_9;
                unsigned int bit_3 = ((1) ? v0 : v1) >> (unsigned int)(16 + kb) & 1;
                if (bit_3 == 0) {
                    p[8] = 0.0f;
                    p[10] = 0.0f;
                }
                float _fma_10 = __fmaf_rn(s[9], scale_log2, -lse0);
                float _exp2_10 = approx_exp2(_fma_10);
                p[9] = _exp2_10;
                float _fma_11 = __fmaf_rn(s[11], scale_log2, -lse1);
                float _exp2_11 = approx_exp2(_fma_11);
                p[11] = _exp2_11;
                unsigned int bit_4 = ((1) ? v0 : v1) >> (unsigned int)(17 + kb) & 1;
                if (bit_4 == 0) {
                    p[9] = 0.0f;
                    p[11] = 0.0f;
                }
                float _fma_12 = __fmaf_rn(s[12], scale_log2, -lse0);
                float _exp2_12 = approx_exp2(_fma_12);
                p[12] = _exp2_12;
                float _fma_13 = __fmaf_rn(s[14], scale_log2, -lse1);
                float _exp2_13 = approx_exp2(_fma_13);
                p[14] = _exp2_13;
                unsigned int bit_5 = ((1) ? v0 : v1) >> (unsigned int)(24 + kb) & 1;
                if (bit_5 == 0) {
                    p[12] = 0.0f;
                    p[14] = 0.0f;
                }
                float _fma_14 = __fmaf_rn(s[13], scale_log2, -lse0);
                float _exp2_14 = approx_exp2(_fma_14);
                p[13] = _exp2_14;
                float _fma_15 = __fmaf_rn(s[15], scale_log2, -lse1);
                float _exp2_15 = approx_exp2(_fma_15);
                p[15] = _exp2_15;
                unsigned int bit_6 = ((1) ? v0 : v1) >> (unsigned int)(25 + kb) & 1;
                if (bit_6 == 0) {
                    p[13] = 0.0f;
                    p[15] = 0.0f;
                }
                float _fma_16 = __fmaf_rn(s[16], scale_log2, -lse0);
                float _exp2_16 = approx_exp2(_fma_16);
                p[16] = _exp2_16;
                float _fma_17 = __fmaf_rn(s[18], scale_log2, -lse1);
                float _exp2_17 = approx_exp2(_fma_17);
                p[18] = _exp2_17;
                unsigned int bit_7 = ((0) ? v0 : v1) >> (unsigned int)kb & 1;
                if (bit_7 == 0) {
                    p[16] = 0.0f;
                    p[18] = 0.0f;
                }
                float _fma_18 = __fmaf_rn(s[17], scale_log2, -lse0);
                float _exp2_18 = approx_exp2(_fma_18);
                p[17] = _exp2_18;
                float _fma_19 = __fmaf_rn(s[19], scale_log2, -lse1);
                float _exp2_19 = approx_exp2(_fma_19);
                p[19] = _exp2_19;
                unsigned int bit_8 = ((0) ? v0 : v1) >> (unsigned int)(1 + kb) & 1;
                if (bit_8 == 0) {
                    p[17] = 0.0f;
                    p[19] = 0.0f;
                }
                float _fma_20 = __fmaf_rn(s[20], scale_log2, -lse0);
                float _exp2_20 = approx_exp2(_fma_20);
                p[20] = _exp2_20;
                float _fma_21 = __fmaf_rn(s[22], scale_log2, -lse1);
                float _exp2_21 = approx_exp2(_fma_21);
                p[22] = _exp2_21;
                unsigned int bit_9 = ((0) ? v0 : v1) >> (unsigned int)(8 + kb) & 1;
                if (bit_9 == 0) {
                    p[20] = 0.0f;
                    p[22] = 0.0f;
                }
                float _fma_22 = __fmaf_rn(s[21], scale_log2, -lse0);
                float _exp2_22 = approx_exp2(_fma_22);
                p[21] = _exp2_22;
                float _fma_23 = __fmaf_rn(s[23], scale_log2, -lse1);
                float _exp2_23 = approx_exp2(_fma_23);
                p[23] = _exp2_23;
                unsigned int bit_10 = ((0) ? v0 : v1) >> (unsigned int)(9 + kb) & 1;
                if (bit_10 == 0) {
                    p[21] = 0.0f;
                    p[23] = 0.0f;
                }
                float _fma_24 = __fmaf_rn(s[24], scale_log2, -lse0);
                float _exp2_24 = approx_exp2(_fma_24);
                p[24] = _exp2_24;
                float _fma_25 = __fmaf_rn(s[26], scale_log2, -lse1);
                float _exp2_25 = approx_exp2(_fma_25);
                p[26] = _exp2_25;
                unsigned int bit_11 = ((0) ? v0 : v1) >> (unsigned int)(16 + kb) & 1;
                if (bit_11 == 0) {
                    p[24] = 0.0f;
                    p[26] = 0.0f;
                }
                float _fma_26 = __fmaf_rn(s[25], scale_log2, -lse0);
                float _exp2_26 = approx_exp2(_fma_26);
                p[25] = _exp2_26;
                float _fma_27 = __fmaf_rn(s[27], scale_log2, -lse1);
                float _exp2_27 = approx_exp2(_fma_27);
                p[27] = _exp2_27;
                unsigned int bit_12 = ((0) ? v0 : v1) >> (unsigned int)(17 + kb) & 1;
                if (bit_12 == 0) {
                    p[25] = 0.0f;
                    p[27] = 0.0f;
                }
                float _fma_28 = __fmaf_rn(s[28], scale_log2, -lse0);
                float _exp2_28 = approx_exp2(_fma_28);
                p[28] = _exp2_28;
                float _fma_29 = __fmaf_rn(s[30], scale_log2, -lse1);
                float _exp2_29 = approx_exp2(_fma_29);
                p[30] = _exp2_29;
                unsigned int bit_13 = ((0) ? v0 : v1) >> (unsigned int)(24 + kb) & 1;
                if (bit_13 == 0) {
                    p[28] = 0.0f;
                    p[30] = 0.0f;
                }
                float _fma_30 = __fmaf_rn(s[29], scale_log2, -lse0);
                float _exp2_30 = approx_exp2(_fma_30);
                p[29] = _exp2_30;
                float _fma_31 = __fmaf_rn(s[31], scale_log2, -lse1);
                float _exp2_31 = approx_exp2(_fma_31);
                p[31] = _exp2_31;
                unsigned int bit_14 = ((0) ? v0 : v1) >> (unsigned int)(25 + kb) & 1;
                if (bit_14 == 0) {
                    p[29] = 0.0f;
                    p[31] = 0.0f;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(p_free_addr, par ^ 1);
                uint32_t p_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                    p_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int paddr = p_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(paddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[3]))
                    : "memory");
                unsigned int paddr_15 = p_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(paddr_15);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[7]))
                    : "memory");
                unsigned int paddr_16 = p_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(paddr_16);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[11]))
                    : "memory");
                unsigned int paddr_17 = p_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(paddr_17);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(dp_full_addr, par);
                float dp[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[1])), "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[3])), "=r"(*reinterpret_cast<uint32_t*>(&dp[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp[5])), "=r"(*reinterpret_cast<uint32_t*>(&dp[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp[7])), "=r"(*reinterpret_cast<uint32_t*>(&dp[8])), "=r"(*reinterpret_cast<uint32_t*>(&dp[9])), "=r"(*reinterpret_cast<uint32_t*>(&dp[10])), "=r"(*reinterpret_cast<uint32_t*>(&dp[11])), "=r"(*reinterpret_cast<uint32_t*>(&dp[12])), "=r"(*reinterpret_cast<uint32_t*>(&dp[13])), "=r"(*reinterpret_cast<uint32_t*>(&dp[14])), "=r"(*reinterpret_cast<uint32_t*>(&dp[15])), "=r"(*reinterpret_cast<uint32_t*>(&dp[16])), "=r"(*reinterpret_cast<uint32_t*>(&dp[17])), "=r"(*reinterpret_cast<uint32_t*>(&dp[18])), "=r"(*reinterpret_cast<uint32_t*>(&dp[19])), "=r"(*reinterpret_cast<uint32_t*>(&dp[20])), "=r"(*reinterpret_cast<uint32_t*>(&dp[21])), "=r"(*reinterpret_cast<uint32_t*>(&dp[22])), "=r"(*reinterpret_cast<uint32_t*>(&dp[23])), "=r"(*reinterpret_cast<uint32_t*>(&dp[24])), "=r"(*reinterpret_cast<uint32_t*>(&dp[25])), "=r"(*reinterpret_cast<uint32_t*>(&dp[26])), "=r"(*reinterpret_cast<uint32_t*>(&dp[27])), "=r"(*reinterpret_cast<uint32_t*>(&dp[28])), "=r"(*reinterpret_cast<uint32_t*>(&dp[29])), "=r"(*reinterpret_cast<uint32_t*>(&dp[30])), "=r"(*reinterpret_cast<uint32_t*>(&dp[31]))
                    : "r"(tmem_tmem + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ds[32];
                ds[0] = p[0] * (dp[0] - dl0) * sm_scale;
                ds[2] = p[2] * (dp[2] - dl1) * sm_scale;
                ds[1] = p[1] * (dp[1] - dl0) * sm_scale;
                ds[3] = p[3] * (dp[3] - dl1) * sm_scale;
                ds[4] = p[4] * (dp[4] - dl0) * sm_scale;
                ds[6] = p[6] * (dp[6] - dl1) * sm_scale;
                ds[5] = p[5] * (dp[5] - dl0) * sm_scale;
                ds[7] = p[7] * (dp[7] - dl1) * sm_scale;
                ds[8] = p[8] * (dp[8] - dl0) * sm_scale;
                ds[10] = p[10] * (dp[10] - dl1) * sm_scale;
                ds[9] = p[9] * (dp[9] - dl0) * sm_scale;
                ds[11] = p[11] * (dp[11] - dl1) * sm_scale;
                ds[12] = p[12] * (dp[12] - dl0) * sm_scale;
                ds[14] = p[14] * (dp[14] - dl1) * sm_scale;
                ds[13] = p[13] * (dp[13] - dl0) * sm_scale;
                ds[15] = p[15] * (dp[15] - dl1) * sm_scale;
                ds[16] = p[16] * (dp[16] - dl0) * sm_scale;
                ds[18] = p[18] * (dp[18] - dl1) * sm_scale;
                ds[17] = p[17] * (dp[17] - dl0) * sm_scale;
                ds[19] = p[19] * (dp[19] - dl1) * sm_scale;
                ds[20] = p[20] * (dp[20] - dl0) * sm_scale;
                ds[22] = p[22] * (dp[22] - dl1) * sm_scale;
                ds[21] = p[21] * (dp[21] - dl0) * sm_scale;
                ds[23] = p[23] * (dp[23] - dl1) * sm_scale;
                ds[24] = p[24] * (dp[24] - dl0) * sm_scale;
                ds[26] = p[26] * (dp[26] - dl1) * sm_scale;
                ds[25] = p[25] * (dp[25] - dl0) * sm_scale;
                ds[27] = p[27] * (dp[27] - dl1) * sm_scale;
                ds[28] = p[28] * (dp[28] - dl0) * sm_scale;
                ds[30] = p[30] * (dp[30] - dl1) * sm_scale;
                ds[29] = p[29] * (dp[29] - dl0) * sm_scale;
                ds[31] = p[31] * (dp[31] - dl1) * sm_scale;
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dp_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(ds_free_addr, par ^ 1);
                uint32_t ds_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ds[_lp*2 + 0], ds[_lp*2+1 + 0]));
                    ds_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int daddr = ds_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(daddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[3]))
                    : "memory");
                unsigned int daddr_18 = ds_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(daddr_18);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[7]))
                    : "memory");
                unsigned int daddr_19 = ds_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(daddr_19);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[11]))
                    : "memory");
                unsigned int daddr_20 = ds_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(daddr_20);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(ds_full_addr);
            }
            mbarrier_wait(dq_done_addr, 0);
            int ctid = w * 32 + lane_0;
            int bidc = blockIdx.x;
            long long pbase = (long long)bidc * 36864;
            float dq[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq[31]))
                : "r"(tmem_tmem + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx = pbase + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq[0 + 0], dq[0 + 1], dq[0 + 2], dq[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[4 + 0], dq[4 + 1], dq[4 + 2], dq[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[8 + 0], dq[8 + 1], dq[8 + 2], dq[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[12 + 0], dq[12 + 1], dq[12 + 2], dq[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[16 + 0], dq[16 + 1], dq[16 + 2], dq[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[20 + 0], dq[20 + 1], dq[20 + 2], dq[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[24 + 0], dq[24 + 1], dq[24 + 2], dq[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq[28 + 0], dq[28 + 1], dq[28 + 2], dq[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_1[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx + 0);
                    _vec_load_1[0 + 0] = _v4.x;
                    _vec_load_1[0 + 1] = _v4.y;
                    _vec_load_1[0 + 2] = _v4.z;
                    _vec_load_1[0 + 3] = _v4.w;
                }
                float _vec_load_2[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 512) + 0);
                    _vec_load_2[0 + 0] = _v4.x;
                    _vec_load_2[0 + 1] = _v4.y;
                    _vec_load_2[0 + 2] = _v4.z;
                    _vec_load_2[0 + 3] = _v4.w;
                }
                float _vec_load_3[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 1024) + 0);
                    _vec_load_3[0 + 0] = _v4.x;
                    _vec_load_3[0 + 1] = _v4.y;
                    _vec_load_3[0 + 2] = _v4.z;
                    _vec_load_3[0 + 3] = _v4.w;
                }
                float _vec_load_4[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 1536) + 0);
                    _vec_load_4[0 + 0] = _v4.x;
                    _vec_load_4[0 + 1] = _v4.y;
                    _vec_load_4[0 + 2] = _v4.z;
                    _vec_load_4[0 + 3] = _v4.w;
                }
                float _vec_load_5[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 2048) + 0);
                    _vec_load_5[0 + 0] = _v4.x;
                    _vec_load_5[0 + 1] = _v4.y;
                    _vec_load_5[0 + 2] = _v4.z;
                    _vec_load_5[0 + 3] = _v4.w;
                }
                float _vec_load_6[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 2560) + 0);
                    _vec_load_6[0 + 0] = _v4.x;
                    _vec_load_6[0 + 1] = _v4.y;
                    _vec_load_6[0 + 2] = _v4.z;
                    _vec_load_6[0 + 3] = _v4.w;
                }
                float _vec_load_7[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 3072) + 0);
                    _vec_load_7[0 + 0] = _v4.x;
                    _vec_load_7[0 + 1] = _v4.y;
                    _vec_load_7[0 + 2] = _v4.z;
                    _vec_load_7[0 + 3] = _v4.w;
                }
                float _vec_load_8[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx + 3584) + 0);
                    _vec_load_8[0 + 0] = _v4.x;
                    _vec_load_8[0 + 1] = _v4.y;
                    _vec_load_8[0 + 2] = _v4.z;
                    _vec_load_8[0 + 3] = _v4.w;
                }
                dq[0] = dq[0] + _vec_load_1[0];
                dq[4] = dq[4] + _vec_load_2[0];
                dq[8] = dq[8] + _vec_load_3[0];
                dq[12] = dq[12] + _vec_load_4[0];
                dq[16] = dq[16] + _vec_load_5[0];
                dq[20] = dq[20] + _vec_load_6[0];
                dq[24] = dq[24] + _vec_load_7[0];
                dq[28] = dq[28] + _vec_load_8[0];
                dq[1] = dq[1] + _vec_load_1[1];
                dq[5] = dq[5] + _vec_load_2[1];
                dq[9] = dq[9] + _vec_load_3[1];
                dq[13] = dq[13] + _vec_load_4[1];
                dq[17] = dq[17] + _vec_load_5[1];
                dq[21] = dq[21] + _vec_load_6[1];
                dq[25] = dq[25] + _vec_load_7[1];
                dq[29] = dq[29] + _vec_load_8[1];
                dq[2] = dq[2] + _vec_load_1[2];
                dq[6] = dq[6] + _vec_load_2[2];
                dq[10] = dq[10] + _vec_load_3[2];
                dq[14] = dq[14] + _vec_load_4[2];
                dq[18] = dq[18] + _vec_load_5[2];
                dq[22] = dq[22] + _vec_load_6[2];
                dq[26] = dq[26] + _vec_load_7[2];
                dq[30] = dq[30] + _vec_load_8[2];
                dq[3] = dq[3] + _vec_load_1[3];
                dq[7] = dq[7] + _vec_load_2[3];
                dq[11] = dq[11] + _vec_load_3[3];
                dq[15] = dq[15] + _vec_load_4[3];
                dq[19] = dq[19] + _vec_load_5[3];
                dq[23] = dq[23] + _vec_load_6[3];
                dq[27] = dq[27] + _vec_load_7[3];
                dq[31] = dq[31] + _vec_load_8[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq[0 + 0], dq[0 + 1], dq[0 + 2], dq[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[4 + 0], dq[4 + 1], dq[4 + 2], dq[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[8 + 0], dq[8 + 1], dq[8 + 2], dq[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[12 + 0], dq[12 + 1], dq[12 + 2], dq[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[16 + 0], dq[16 + 1], dq[16 + 2], dq[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[20 + 0], dq[20 + 1], dq[20 + 2], dq[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[24 + 0], dq[24 + 1], dq[24 + 2], dq[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq[28 + 0], dq[28 + 1], dq[28 + 2], dq[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq[_lp*2 + 0], dq[_lp*2+1 + 0]));
                dq_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead = 8 * (m4 / 2) + r8;
            int qchunk = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead * 128) + (unsigned int)((qchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(qaddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[3]))
                : "memory");
            int qhead_1 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_2 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_3 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_1 * 128) + (unsigned int)((qchunk_2 ^ r8) * 16);
            uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(qaddr_3);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[7]))
                : "memory");
            int qhead_4 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_5 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_6 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_4 * 128) + (unsigned int)((qchunk_5 ^ r8) * 16);
            uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(qaddr_6);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[11]))
                : "memory");
            int qhead_7 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_8 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_9 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_7 * 128) + (unsigned int)((qchunk_8 ^ r8) * 16);
            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(qaddr_9);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[15]))
                : "memory");
            float dq_10[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[31]))
                : "r"(tmem_tmem + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_11 = pbase + 4096 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_10[0 + 0], dq_10[0 + 1], dq_10[0 + 2], dq_10[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[4 + 0], dq_10[4 + 1], dq_10[4 + 2], dq_10[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[8 + 0], dq_10[8 + 1], dq_10[8 + 2], dq_10[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[12 + 0], dq_10[12 + 1], dq_10[12 + 2], dq_10[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[16 + 0], dq_10[16 + 1], dq_10[16 + 2], dq_10[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[20 + 0], dq_10[20 + 1], dq_10[20 + 2], dq_10[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[24 + 0], dq_10[24 + 1], dq_10[24 + 2], dq_10[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_10[28 + 0], dq_10[28 + 1], dq_10[28 + 2], dq_10[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_11 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_9[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_11 + 0);
                    _vec_load_9[0 + 0] = _v4.x;
                    _vec_load_9[0 + 1] = _v4.y;
                    _vec_load_9[0 + 2] = _v4.z;
                    _vec_load_9[0 + 3] = _v4.w;
                }
                float _vec_load_10[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 512) + 0);
                    _vec_load_10[0 + 0] = _v4.x;
                    _vec_load_10[0 + 1] = _v4.y;
                    _vec_load_10[0 + 2] = _v4.z;
                    _vec_load_10[0 + 3] = _v4.w;
                }
                float _vec_load_11[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 1024) + 0);
                    _vec_load_11[0 + 0] = _v4.x;
                    _vec_load_11[0 + 1] = _v4.y;
                    _vec_load_11[0 + 2] = _v4.z;
                    _vec_load_11[0 + 3] = _v4.w;
                }
                float _vec_load_12[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 1536) + 0);
                    _vec_load_12[0 + 0] = _v4.x;
                    _vec_load_12[0 + 1] = _v4.y;
                    _vec_load_12[0 + 2] = _v4.z;
                    _vec_load_12[0 + 3] = _v4.w;
                }
                float _vec_load_13[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 2048) + 0);
                    _vec_load_13[0 + 0] = _v4.x;
                    _vec_load_13[0 + 1] = _v4.y;
                    _vec_load_13[0 + 2] = _v4.z;
                    _vec_load_13[0 + 3] = _v4.w;
                }
                float _vec_load_14[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 2560) + 0);
                    _vec_load_14[0 + 0] = _v4.x;
                    _vec_load_14[0 + 1] = _v4.y;
                    _vec_load_14[0 + 2] = _v4.z;
                    _vec_load_14[0 + 3] = _v4.w;
                }
                float _vec_load_15[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 3072) + 0);
                    _vec_load_15[0 + 0] = _v4.x;
                    _vec_load_15[0 + 1] = _v4.y;
                    _vec_load_15[0 + 2] = _v4.z;
                    _vec_load_15[0 + 3] = _v4.w;
                }
                float _vec_load_16[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_11 + 3584) + 0);
                    _vec_load_16[0 + 0] = _v4.x;
                    _vec_load_16[0 + 1] = _v4.y;
                    _vec_load_16[0 + 2] = _v4.z;
                    _vec_load_16[0 + 3] = _v4.w;
                }
                dq_10[0] = dq_10[0] + _vec_load_9[0];
                dq_10[4] = dq_10[4] + _vec_load_10[0];
                dq_10[8] = dq_10[8] + _vec_load_11[0];
                dq_10[12] = dq_10[12] + _vec_load_12[0];
                dq_10[16] = dq_10[16] + _vec_load_13[0];
                dq_10[20] = dq_10[20] + _vec_load_14[0];
                dq_10[24] = dq_10[24] + _vec_load_15[0];
                dq_10[28] = dq_10[28] + _vec_load_16[0];
                dq_10[1] = dq_10[1] + _vec_load_9[1];
                dq_10[5] = dq_10[5] + _vec_load_10[1];
                dq_10[9] = dq_10[9] + _vec_load_11[1];
                dq_10[13] = dq_10[13] + _vec_load_12[1];
                dq_10[17] = dq_10[17] + _vec_load_13[1];
                dq_10[21] = dq_10[21] + _vec_load_14[1];
                dq_10[25] = dq_10[25] + _vec_load_15[1];
                dq_10[29] = dq_10[29] + _vec_load_16[1];
                dq_10[2] = dq_10[2] + _vec_load_9[2];
                dq_10[6] = dq_10[6] + _vec_load_10[2];
                dq_10[10] = dq_10[10] + _vec_load_11[2];
                dq_10[14] = dq_10[14] + _vec_load_12[2];
                dq_10[18] = dq_10[18] + _vec_load_13[2];
                dq_10[22] = dq_10[22] + _vec_load_14[2];
                dq_10[26] = dq_10[26] + _vec_load_15[2];
                dq_10[30] = dq_10[30] + _vec_load_16[2];
                dq_10[3] = dq_10[3] + _vec_load_9[3];
                dq_10[7] = dq_10[7] + _vec_load_10[3];
                dq_10[11] = dq_10[11] + _vec_load_11[3];
                dq_10[15] = dq_10[15] + _vec_load_12[3];
                dq_10[19] = dq_10[19] + _vec_load_13[3];
                dq_10[23] = dq_10[23] + _vec_load_14[3];
                dq_10[27] = dq_10[27] + _vec_load_15[3];
                dq_10[31] = dq_10[31] + _vec_load_16[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_10[0 + 0], dq_10[0 + 1], dq_10[0 + 2], dq_10[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[4 + 0], dq_10[4 + 1], dq_10[4 + 2], dq_10[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[8 + 0], dq_10[8 + 1], dq_10[8 + 2], dq_10[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[12 + 0], dq_10[12 + 1], dq_10[12 + 2], dq_10[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[16 + 0], dq_10[16 + 1], dq_10[16 + 2], dq_10[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[20 + 0], dq_10[20 + 1], dq_10[20 + 2], dq_10[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[24 + 0], dq_10[24 + 1], dq_10[24 + 2], dq_10[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_10[28 + 0], dq_10[28 + 1], dq_10[28 + 2], dq_10[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_11 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_10_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_10[_lp*2 + 0], dq_10[_lp*2+1 + 0]));
                dq_10_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_12 = 8 * (m4 / 2) + r8;
            int qchunk_13 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_14 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_12 * 128) + (unsigned int)((qchunk_13 ^ r8) * 16);
            uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(qaddr_14);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[3]))
                : "memory");
            int qhead_15 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_16 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_17 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_15 * 128) + (unsigned int)((qchunk_16 ^ r8) * 16);
            uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(qaddr_17);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[7]))
                : "memory");
            int qhead_18 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_19 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_20 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_18 * 128) + (unsigned int)((qchunk_19 ^ r8) * 16);
            uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(qaddr_20);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[11]))
                : "memory");
            int qhead_21 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_22 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_23 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_21 * 128) + (unsigned int)((qchunk_22 ^ r8) * 16);
            uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(qaddr_23);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[15]))
                : "memory");
            float dq_24[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_24[31]))
                : "r"(tmem_tmem + 192 + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_25 = pbase + 8192 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_24[0 + 0], dq_24[0 + 1], dq_24[0 + 2], dq_24[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[4 + 0], dq_24[4 + 1], dq_24[4 + 2], dq_24[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[8 + 0], dq_24[8 + 1], dq_24[8 + 2], dq_24[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[12 + 0], dq_24[12 + 1], dq_24[12 + 2], dq_24[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[16 + 0], dq_24[16 + 1], dq_24[16 + 2], dq_24[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[20 + 0], dq_24[20 + 1], dq_24[20 + 2], dq_24[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[24 + 0], dq_24[24 + 1], dq_24[24 + 2], dq_24[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_24[28 + 0], dq_24[28 + 1], dq_24[28 + 2], dq_24[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_25 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_17[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_25 + 0);
                    _vec_load_17[0 + 0] = _v4.x;
                    _vec_load_17[0 + 1] = _v4.y;
                    _vec_load_17[0 + 2] = _v4.z;
                    _vec_load_17[0 + 3] = _v4.w;
                }
                float _vec_load_18[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 512) + 0);
                    _vec_load_18[0 + 0] = _v4.x;
                    _vec_load_18[0 + 1] = _v4.y;
                    _vec_load_18[0 + 2] = _v4.z;
                    _vec_load_18[0 + 3] = _v4.w;
                }
                float _vec_load_19[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 1024) + 0);
                    _vec_load_19[0 + 0] = _v4.x;
                    _vec_load_19[0 + 1] = _v4.y;
                    _vec_load_19[0 + 2] = _v4.z;
                    _vec_load_19[0 + 3] = _v4.w;
                }
                float _vec_load_20[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 1536) + 0);
                    _vec_load_20[0 + 0] = _v4.x;
                    _vec_load_20[0 + 1] = _v4.y;
                    _vec_load_20[0 + 2] = _v4.z;
                    _vec_load_20[0 + 3] = _v4.w;
                }
                float _vec_load_21[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 2048) + 0);
                    _vec_load_21[0 + 0] = _v4.x;
                    _vec_load_21[0 + 1] = _v4.y;
                    _vec_load_21[0 + 2] = _v4.z;
                    _vec_load_21[0 + 3] = _v4.w;
                }
                float _vec_load_22[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 2560) + 0);
                    _vec_load_22[0 + 0] = _v4.x;
                    _vec_load_22[0 + 1] = _v4.y;
                    _vec_load_22[0 + 2] = _v4.z;
                    _vec_load_22[0 + 3] = _v4.w;
                }
                float _vec_load_23[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 3072) + 0);
                    _vec_load_23[0 + 0] = _v4.x;
                    _vec_load_23[0 + 1] = _v4.y;
                    _vec_load_23[0 + 2] = _v4.z;
                    _vec_load_23[0 + 3] = _v4.w;
                }
                float _vec_load_24[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_25 + 3584) + 0);
                    _vec_load_24[0 + 0] = _v4.x;
                    _vec_load_24[0 + 1] = _v4.y;
                    _vec_load_24[0 + 2] = _v4.z;
                    _vec_load_24[0 + 3] = _v4.w;
                }
                dq_24[0] = dq_24[0] + _vec_load_17[0];
                dq_24[4] = dq_24[4] + _vec_load_18[0];
                dq_24[8] = dq_24[8] + _vec_load_19[0];
                dq_24[12] = dq_24[12] + _vec_load_20[0];
                dq_24[16] = dq_24[16] + _vec_load_21[0];
                dq_24[20] = dq_24[20] + _vec_load_22[0];
                dq_24[24] = dq_24[24] + _vec_load_23[0];
                dq_24[28] = dq_24[28] + _vec_load_24[0];
                dq_24[1] = dq_24[1] + _vec_load_17[1];
                dq_24[5] = dq_24[5] + _vec_load_18[1];
                dq_24[9] = dq_24[9] + _vec_load_19[1];
                dq_24[13] = dq_24[13] + _vec_load_20[1];
                dq_24[17] = dq_24[17] + _vec_load_21[1];
                dq_24[21] = dq_24[21] + _vec_load_22[1];
                dq_24[25] = dq_24[25] + _vec_load_23[1];
                dq_24[29] = dq_24[29] + _vec_load_24[1];
                dq_24[2] = dq_24[2] + _vec_load_17[2];
                dq_24[6] = dq_24[6] + _vec_load_18[2];
                dq_24[10] = dq_24[10] + _vec_load_19[2];
                dq_24[14] = dq_24[14] + _vec_load_20[2];
                dq_24[18] = dq_24[18] + _vec_load_21[2];
                dq_24[22] = dq_24[22] + _vec_load_22[2];
                dq_24[26] = dq_24[26] + _vec_load_23[2];
                dq_24[30] = dq_24[30] + _vec_load_24[2];
                dq_24[3] = dq_24[3] + _vec_load_17[3];
                dq_24[7] = dq_24[7] + _vec_load_18[3];
                dq_24[11] = dq_24[11] + _vec_load_19[3];
                dq_24[15] = dq_24[15] + _vec_load_20[3];
                dq_24[19] = dq_24[19] + _vec_load_21[3];
                dq_24[23] = dq_24[23] + _vec_load_22[3];
                dq_24[27] = dq_24[27] + _vec_load_23[3];
                dq_24[31] = dq_24[31] + _vec_load_24[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_24[0 + 0], dq_24[0 + 1], dq_24[0 + 2], dq_24[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[4 + 0], dq_24[4 + 1], dq_24[4 + 2], dq_24[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[8 + 0], dq_24[8 + 1], dq_24[8 + 2], dq_24[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[12 + 0], dq_24[12 + 1], dq_24[12 + 2], dq_24[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[16 + 0], dq_24[16 + 1], dq_24[16 + 2], dq_24[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[20 + 0], dq_24[20 + 1], dq_24[20 + 2], dq_24[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[24 + 0], dq_24[24 + 1], dq_24[24 + 2], dq_24[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_24[28 + 0], dq_24[28 + 1], dq_24[28 + 2], dq_24[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_25 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_24_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_24[_lp*2 + 0], dq_24[_lp*2+1 + 0]));
                dq_24_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_26 = 8 * (m4 / 2) + r8;
            int qchunk_27 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_28 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_26 * 128) + (unsigned int)((qchunk_27 ^ r8) * 16);
            uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(qaddr_28);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[3]))
                : "memory");
            int qhead_29 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_30 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_31 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_29 * 128) + (unsigned int)((qchunk_30 ^ r8) * 16);
            uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(qaddr_31);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[7]))
                : "memory");
            int qhead_32 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_33 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_34 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_32 * 128) + (unsigned int)((qchunk_33 ^ r8) * 16);
            uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(qaddr_34);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[11]))
                : "memory");
            int qhead_35 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_36 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_37 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_35 * 128) + (unsigned int)((qchunk_36 ^ r8) * 16);
            uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(qaddr_37);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_24_bf16[15]))
                : "memory");
            float dq_38[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_38[31]))
                : "r"(tmem_tmem + 192 + 64 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_39 = pbase + 12288 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_38[0 + 0], dq_38[0 + 1], dq_38[0 + 2], dq_38[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[4 + 0], dq_38[4 + 1], dq_38[4 + 2], dq_38[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[8 + 0], dq_38[8 + 1], dq_38[8 + 2], dq_38[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[12 + 0], dq_38[12 + 1], dq_38[12 + 2], dq_38[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[16 + 0], dq_38[16 + 1], dq_38[16 + 2], dq_38[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[20 + 0], dq_38[20 + 1], dq_38[20 + 2], dq_38[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[24 + 0], dq_38[24 + 1], dq_38[24 + 2], dq_38[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_38[28 + 0], dq_38[28 + 1], dq_38[28 + 2], dq_38[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_39 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_25[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_39 + 0);
                    _vec_load_25[0 + 0] = _v4.x;
                    _vec_load_25[0 + 1] = _v4.y;
                    _vec_load_25[0 + 2] = _v4.z;
                    _vec_load_25[0 + 3] = _v4.w;
                }
                float _vec_load_26[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 512) + 0);
                    _vec_load_26[0 + 0] = _v4.x;
                    _vec_load_26[0 + 1] = _v4.y;
                    _vec_load_26[0 + 2] = _v4.z;
                    _vec_load_26[0 + 3] = _v4.w;
                }
                float _vec_load_27[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 1024) + 0);
                    _vec_load_27[0 + 0] = _v4.x;
                    _vec_load_27[0 + 1] = _v4.y;
                    _vec_load_27[0 + 2] = _v4.z;
                    _vec_load_27[0 + 3] = _v4.w;
                }
                float _vec_load_28[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 1536) + 0);
                    _vec_load_28[0 + 0] = _v4.x;
                    _vec_load_28[0 + 1] = _v4.y;
                    _vec_load_28[0 + 2] = _v4.z;
                    _vec_load_28[0 + 3] = _v4.w;
                }
                float _vec_load_29[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 2048) + 0);
                    _vec_load_29[0 + 0] = _v4.x;
                    _vec_load_29[0 + 1] = _v4.y;
                    _vec_load_29[0 + 2] = _v4.z;
                    _vec_load_29[0 + 3] = _v4.w;
                }
                float _vec_load_30[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 2560) + 0);
                    _vec_load_30[0 + 0] = _v4.x;
                    _vec_load_30[0 + 1] = _v4.y;
                    _vec_load_30[0 + 2] = _v4.z;
                    _vec_load_30[0 + 3] = _v4.w;
                }
                float _vec_load_31[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 3072) + 0);
                    _vec_load_31[0 + 0] = _v4.x;
                    _vec_load_31[0 + 1] = _v4.y;
                    _vec_load_31[0 + 2] = _v4.z;
                    _vec_load_31[0 + 3] = _v4.w;
                }
                float _vec_load_32[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_39 + 3584) + 0);
                    _vec_load_32[0 + 0] = _v4.x;
                    _vec_load_32[0 + 1] = _v4.y;
                    _vec_load_32[0 + 2] = _v4.z;
                    _vec_load_32[0 + 3] = _v4.w;
                }
                dq_38[0] = dq_38[0] + _vec_load_25[0];
                dq_38[4] = dq_38[4] + _vec_load_26[0];
                dq_38[8] = dq_38[8] + _vec_load_27[0];
                dq_38[12] = dq_38[12] + _vec_load_28[0];
                dq_38[16] = dq_38[16] + _vec_load_29[0];
                dq_38[20] = dq_38[20] + _vec_load_30[0];
                dq_38[24] = dq_38[24] + _vec_load_31[0];
                dq_38[28] = dq_38[28] + _vec_load_32[0];
                dq_38[1] = dq_38[1] + _vec_load_25[1];
                dq_38[5] = dq_38[5] + _vec_load_26[1];
                dq_38[9] = dq_38[9] + _vec_load_27[1];
                dq_38[13] = dq_38[13] + _vec_load_28[1];
                dq_38[17] = dq_38[17] + _vec_load_29[1];
                dq_38[21] = dq_38[21] + _vec_load_30[1];
                dq_38[25] = dq_38[25] + _vec_load_31[1];
                dq_38[29] = dq_38[29] + _vec_load_32[1];
                dq_38[2] = dq_38[2] + _vec_load_25[2];
                dq_38[6] = dq_38[6] + _vec_load_26[2];
                dq_38[10] = dq_38[10] + _vec_load_27[2];
                dq_38[14] = dq_38[14] + _vec_load_28[2];
                dq_38[18] = dq_38[18] + _vec_load_29[2];
                dq_38[22] = dq_38[22] + _vec_load_30[2];
                dq_38[26] = dq_38[26] + _vec_load_31[2];
                dq_38[30] = dq_38[30] + _vec_load_32[2];
                dq_38[3] = dq_38[3] + _vec_load_25[3];
                dq_38[7] = dq_38[7] + _vec_load_26[3];
                dq_38[11] = dq_38[11] + _vec_load_27[3];
                dq_38[15] = dq_38[15] + _vec_load_28[3];
                dq_38[19] = dq_38[19] + _vec_load_29[3];
                dq_38[23] = dq_38[23] + _vec_load_30[3];
                dq_38[27] = dq_38[27] + _vec_load_31[3];
                dq_38[31] = dq_38[31] + _vec_load_32[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_38[0 + 0], dq_38[0 + 1], dq_38[0 + 2], dq_38[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[4 + 0], dq_38[4 + 1], dq_38[4 + 2], dq_38[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[8 + 0], dq_38[8 + 1], dq_38[8 + 2], dq_38[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[12 + 0], dq_38[12 + 1], dq_38[12 + 2], dq_38[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[16 + 0], dq_38[16 + 1], dq_38[16 + 2], dq_38[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[20 + 0], dq_38[20 + 1], dq_38[20 + 2], dq_38[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[24 + 0], dq_38[24 + 1], dq_38[24 + 2], dq_38[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_38[28 + 0], dq_38[28 + 1], dq_38[28 + 2], dq_38[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_39 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_38_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_38[_lp*2 + 0], dq_38[_lp*2+1 + 0]));
                dq_38_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_40 = 8 * (m4 / 2) + r8;
            int qchunk_41 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_42 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_40 * 128) + (unsigned int)((qchunk_41 ^ r8) * 16);
            uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(qaddr_42);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[3]))
                : "memory");
            int qhead_43 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_44 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_45 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_43 * 128) + (unsigned int)((qchunk_44 ^ r8) * 16);
            uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(qaddr_45);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[7]))
                : "memory");
            int qhead_46 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_47 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_48 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_46 * 128) + (unsigned int)((qchunk_47 ^ r8) * 16);
            uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(qaddr_48);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[11]))
                : "memory");
            int qhead_49 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_50 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_51 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_49 * 128) + (unsigned int)((qchunk_50 ^ r8) * 16);
            uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(qaddr_51);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_38_bf16[15]))
                : "memory");
            float dq_52[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_52[31]))
                : "r"(tmem_tmem + 192 + 128));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_53 = pbase + 16384 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_52[0 + 0], dq_52[0 + 1], dq_52[0 + 2], dq_52[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[4 + 0], dq_52[4 + 1], dq_52[4 + 2], dq_52[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[8 + 0], dq_52[8 + 1], dq_52[8 + 2], dq_52[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[12 + 0], dq_52[12 + 1], dq_52[12 + 2], dq_52[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[16 + 0], dq_52[16 + 1], dq_52[16 + 2], dq_52[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[20 + 0], dq_52[20 + 1], dq_52[20 + 2], dq_52[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[24 + 0], dq_52[24 + 1], dq_52[24 + 2], dq_52[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_52[28 + 0], dq_52[28 + 1], dq_52[28 + 2], dq_52[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_53 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_33[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_53 + 0);
                    _vec_load_33[0 + 0] = _v4.x;
                    _vec_load_33[0 + 1] = _v4.y;
                    _vec_load_33[0 + 2] = _v4.z;
                    _vec_load_33[0 + 3] = _v4.w;
                }
                float _vec_load_34[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 512) + 0);
                    _vec_load_34[0 + 0] = _v4.x;
                    _vec_load_34[0 + 1] = _v4.y;
                    _vec_load_34[0 + 2] = _v4.z;
                    _vec_load_34[0 + 3] = _v4.w;
                }
                float _vec_load_35[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 1024) + 0);
                    _vec_load_35[0 + 0] = _v4.x;
                    _vec_load_35[0 + 1] = _v4.y;
                    _vec_load_35[0 + 2] = _v4.z;
                    _vec_load_35[0 + 3] = _v4.w;
                }
                float _vec_load_36[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 1536) + 0);
                    _vec_load_36[0 + 0] = _v4.x;
                    _vec_load_36[0 + 1] = _v4.y;
                    _vec_load_36[0 + 2] = _v4.z;
                    _vec_load_36[0 + 3] = _v4.w;
                }
                float _vec_load_37[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 2048) + 0);
                    _vec_load_37[0 + 0] = _v4.x;
                    _vec_load_37[0 + 1] = _v4.y;
                    _vec_load_37[0 + 2] = _v4.z;
                    _vec_load_37[0 + 3] = _v4.w;
                }
                float _vec_load_38[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 2560) + 0);
                    _vec_load_38[0 + 0] = _v4.x;
                    _vec_load_38[0 + 1] = _v4.y;
                    _vec_load_38[0 + 2] = _v4.z;
                    _vec_load_38[0 + 3] = _v4.w;
                }
                float _vec_load_39[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 3072) + 0);
                    _vec_load_39[0 + 0] = _v4.x;
                    _vec_load_39[0 + 1] = _v4.y;
                    _vec_load_39[0 + 2] = _v4.z;
                    _vec_load_39[0 + 3] = _v4.w;
                }
                float _vec_load_40[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_53 + 3584) + 0);
                    _vec_load_40[0 + 0] = _v4.x;
                    _vec_load_40[0 + 1] = _v4.y;
                    _vec_load_40[0 + 2] = _v4.z;
                    _vec_load_40[0 + 3] = _v4.w;
                }
                dq_52[0] = dq_52[0] + _vec_load_33[0];
                dq_52[4] = dq_52[4] + _vec_load_34[0];
                dq_52[8] = dq_52[8] + _vec_load_35[0];
                dq_52[12] = dq_52[12] + _vec_load_36[0];
                dq_52[16] = dq_52[16] + _vec_load_37[0];
                dq_52[20] = dq_52[20] + _vec_load_38[0];
                dq_52[24] = dq_52[24] + _vec_load_39[0];
                dq_52[28] = dq_52[28] + _vec_load_40[0];
                dq_52[1] = dq_52[1] + _vec_load_33[1];
                dq_52[5] = dq_52[5] + _vec_load_34[1];
                dq_52[9] = dq_52[9] + _vec_load_35[1];
                dq_52[13] = dq_52[13] + _vec_load_36[1];
                dq_52[17] = dq_52[17] + _vec_load_37[1];
                dq_52[21] = dq_52[21] + _vec_load_38[1];
                dq_52[25] = dq_52[25] + _vec_load_39[1];
                dq_52[29] = dq_52[29] + _vec_load_40[1];
                dq_52[2] = dq_52[2] + _vec_load_33[2];
                dq_52[6] = dq_52[6] + _vec_load_34[2];
                dq_52[10] = dq_52[10] + _vec_load_35[2];
                dq_52[14] = dq_52[14] + _vec_load_36[2];
                dq_52[18] = dq_52[18] + _vec_load_37[2];
                dq_52[22] = dq_52[22] + _vec_load_38[2];
                dq_52[26] = dq_52[26] + _vec_load_39[2];
                dq_52[30] = dq_52[30] + _vec_load_40[2];
                dq_52[3] = dq_52[3] + _vec_load_33[3];
                dq_52[7] = dq_52[7] + _vec_load_34[3];
                dq_52[11] = dq_52[11] + _vec_load_35[3];
                dq_52[15] = dq_52[15] + _vec_load_36[3];
                dq_52[19] = dq_52[19] + _vec_load_37[3];
                dq_52[23] = dq_52[23] + _vec_load_38[3];
                dq_52[27] = dq_52[27] + _vec_load_39[3];
                dq_52[31] = dq_52[31] + _vec_load_40[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_52[0 + 0], dq_52[0 + 1], dq_52[0 + 2], dq_52[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[4 + 0], dq_52[4 + 1], dq_52[4 + 2], dq_52[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[8 + 0], dq_52[8 + 1], dq_52[8 + 2], dq_52[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[12 + 0], dq_52[12 + 1], dq_52[12 + 2], dq_52[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[16 + 0], dq_52[16 + 1], dq_52[16 + 2], dq_52[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[20 + 0], dq_52[20 + 1], dq_52[20 + 2], dq_52[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[24 + 0], dq_52[24 + 1], dq_52[24 + 2], dq_52[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_52[28 + 0], dq_52[28 + 1], dq_52[28 + 2], dq_52[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_53 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_52_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_52[_lp*2 + 0], dq_52[_lp*2+1 + 0]));
                dq_52_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_54 = 8 * (m4 / 2) + r8;
            int qchunk_55 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_56 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_54 * 128) + (unsigned int)((qchunk_55 ^ r8) * 16);
            uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(qaddr_56);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[3]))
                : "memory");
            int qhead_57 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_58 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_59 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_57 * 128) + (unsigned int)((qchunk_58 ^ r8) * 16);
            uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(qaddr_59);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[7]))
                : "memory");
            int qhead_60 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_61 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_62 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_60 * 128) + (unsigned int)((qchunk_61 ^ r8) * 16);
            uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(qaddr_62);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[11]))
                : "memory");
            int qhead_63 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_64 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_65 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_63 * 128) + (unsigned int)((qchunk_64 ^ r8) * 16);
            uint32_t _stmatrix_addr_67 = static_cast<uint32_t>(qaddr_65);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_67), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_52_bf16[15]))
                : "memory");
            float dq_66[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_66[31]))
                : "r"(tmem_tmem + 192 + 128 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_67 = pbase + 20480 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_66[0 + 0], dq_66[0 + 1], dq_66[0 + 2], dq_66[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[4 + 0], dq_66[4 + 1], dq_66[4 + 2], dq_66[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[8 + 0], dq_66[8 + 1], dq_66[8 + 2], dq_66[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[12 + 0], dq_66[12 + 1], dq_66[12 + 2], dq_66[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[16 + 0], dq_66[16 + 1], dq_66[16 + 2], dq_66[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[20 + 0], dq_66[20 + 1], dq_66[20 + 2], dq_66[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[24 + 0], dq_66[24 + 1], dq_66[24 + 2], dq_66[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_66[28 + 0], dq_66[28 + 1], dq_66[28 + 2], dq_66[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_67 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_41[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_67 + 0);
                    _vec_load_41[0 + 0] = _v4.x;
                    _vec_load_41[0 + 1] = _v4.y;
                    _vec_load_41[0 + 2] = _v4.z;
                    _vec_load_41[0 + 3] = _v4.w;
                }
                float _vec_load_42[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 512) + 0);
                    _vec_load_42[0 + 0] = _v4.x;
                    _vec_load_42[0 + 1] = _v4.y;
                    _vec_load_42[0 + 2] = _v4.z;
                    _vec_load_42[0 + 3] = _v4.w;
                }
                float _vec_load_43[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 1024) + 0);
                    _vec_load_43[0 + 0] = _v4.x;
                    _vec_load_43[0 + 1] = _v4.y;
                    _vec_load_43[0 + 2] = _v4.z;
                    _vec_load_43[0 + 3] = _v4.w;
                }
                float _vec_load_44[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 1536) + 0);
                    _vec_load_44[0 + 0] = _v4.x;
                    _vec_load_44[0 + 1] = _v4.y;
                    _vec_load_44[0 + 2] = _v4.z;
                    _vec_load_44[0 + 3] = _v4.w;
                }
                float _vec_load_45[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 2048) + 0);
                    _vec_load_45[0 + 0] = _v4.x;
                    _vec_load_45[0 + 1] = _v4.y;
                    _vec_load_45[0 + 2] = _v4.z;
                    _vec_load_45[0 + 3] = _v4.w;
                }
                float _vec_load_46[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 2560) + 0);
                    _vec_load_46[0 + 0] = _v4.x;
                    _vec_load_46[0 + 1] = _v4.y;
                    _vec_load_46[0 + 2] = _v4.z;
                    _vec_load_46[0 + 3] = _v4.w;
                }
                float _vec_load_47[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 3072) + 0);
                    _vec_load_47[0 + 0] = _v4.x;
                    _vec_load_47[0 + 1] = _v4.y;
                    _vec_load_47[0 + 2] = _v4.z;
                    _vec_load_47[0 + 3] = _v4.w;
                }
                float _vec_load_48[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_67 + 3584) + 0);
                    _vec_load_48[0 + 0] = _v4.x;
                    _vec_load_48[0 + 1] = _v4.y;
                    _vec_load_48[0 + 2] = _v4.z;
                    _vec_load_48[0 + 3] = _v4.w;
                }
                dq_66[0] = dq_66[0] + _vec_load_41[0];
                dq_66[4] = dq_66[4] + _vec_load_42[0];
                dq_66[8] = dq_66[8] + _vec_load_43[0];
                dq_66[12] = dq_66[12] + _vec_load_44[0];
                dq_66[16] = dq_66[16] + _vec_load_45[0];
                dq_66[20] = dq_66[20] + _vec_load_46[0];
                dq_66[24] = dq_66[24] + _vec_load_47[0];
                dq_66[28] = dq_66[28] + _vec_load_48[0];
                dq_66[1] = dq_66[1] + _vec_load_41[1];
                dq_66[5] = dq_66[5] + _vec_load_42[1];
                dq_66[9] = dq_66[9] + _vec_load_43[1];
                dq_66[13] = dq_66[13] + _vec_load_44[1];
                dq_66[17] = dq_66[17] + _vec_load_45[1];
                dq_66[21] = dq_66[21] + _vec_load_46[1];
                dq_66[25] = dq_66[25] + _vec_load_47[1];
                dq_66[29] = dq_66[29] + _vec_load_48[1];
                dq_66[2] = dq_66[2] + _vec_load_41[2];
                dq_66[6] = dq_66[6] + _vec_load_42[2];
                dq_66[10] = dq_66[10] + _vec_load_43[2];
                dq_66[14] = dq_66[14] + _vec_load_44[2];
                dq_66[18] = dq_66[18] + _vec_load_45[2];
                dq_66[22] = dq_66[22] + _vec_load_46[2];
                dq_66[26] = dq_66[26] + _vec_load_47[2];
                dq_66[30] = dq_66[30] + _vec_load_48[2];
                dq_66[3] = dq_66[3] + _vec_load_41[3];
                dq_66[7] = dq_66[7] + _vec_load_42[3];
                dq_66[11] = dq_66[11] + _vec_load_43[3];
                dq_66[15] = dq_66[15] + _vec_load_44[3];
                dq_66[19] = dq_66[19] + _vec_load_45[3];
                dq_66[23] = dq_66[23] + _vec_load_46[3];
                dq_66[27] = dq_66[27] + _vec_load_47[3];
                dq_66[31] = dq_66[31] + _vec_load_48[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_66[0 + 0], dq_66[0 + 1], dq_66[0 + 2], dq_66[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[4 + 0], dq_66[4 + 1], dq_66[4 + 2], dq_66[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[8 + 0], dq_66[8 + 1], dq_66[8 + 2], dq_66[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[12 + 0], dq_66[12 + 1], dq_66[12 + 2], dq_66[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[16 + 0], dq_66[16 + 1], dq_66[16 + 2], dq_66[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[20 + 0], dq_66[20 + 1], dq_66[20 + 2], dq_66[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[24 + 0], dq_66[24 + 1], dq_66[24 + 2], dq_66[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_66[28 + 0], dq_66[28 + 1], dq_66[28 + 2], dq_66[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_67 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_66_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_66[_lp*2 + 0], dq_66[_lp*2+1 + 0]));
                dq_66_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_68 = 8 * (m4 / 2) + r8;
            int qchunk_69 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_70 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_68 * 128) + (unsigned int)((qchunk_69 ^ r8) * 16);
            uint32_t _stmatrix_addr_76 = static_cast<uint32_t>(qaddr_70);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[3]))
                : "memory");
            int qhead_71 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_72 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_73 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_71 * 128) + (unsigned int)((qchunk_72 ^ r8) * 16);
            uint32_t _stmatrix_addr_77 = static_cast<uint32_t>(qaddr_73);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_77), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[7]))
                : "memory");
            int qhead_74 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_75 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_76 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_74 * 128) + (unsigned int)((qchunk_75 ^ r8) * 16);
            uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(qaddr_76);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[11]))
                : "memory");
            int qhead_77 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_78 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_79 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_77 * 128) + (unsigned int)((qchunk_78 ^ r8) * 16);
            uint32_t _stmatrix_addr_79 = static_cast<uint32_t>(qaddr_79);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_66_bf16[15]))
                : "memory");
            float dq_80[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_80[31]))
                : "r"(tmem_tmem + 192 + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_81 = pbase + 24576 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_80[0 + 0], dq_80[0 + 1], dq_80[0 + 2], dq_80[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[4 + 0], dq_80[4 + 1], dq_80[4 + 2], dq_80[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[8 + 0], dq_80[8 + 1], dq_80[8 + 2], dq_80[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[12 + 0], dq_80[12 + 1], dq_80[12 + 2], dq_80[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[16 + 0], dq_80[16 + 1], dq_80[16 + 2], dq_80[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[20 + 0], dq_80[20 + 1], dq_80[20 + 2], dq_80[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[24 + 0], dq_80[24 + 1], dq_80[24 + 2], dq_80[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_80[28 + 0], dq_80[28 + 1], dq_80[28 + 2], dq_80[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_81 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_49[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_81 + 0);
                    _vec_load_49[0 + 0] = _v4.x;
                    _vec_load_49[0 + 1] = _v4.y;
                    _vec_load_49[0 + 2] = _v4.z;
                    _vec_load_49[0 + 3] = _v4.w;
                }
                float _vec_load_50[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 512) + 0);
                    _vec_load_50[0 + 0] = _v4.x;
                    _vec_load_50[0 + 1] = _v4.y;
                    _vec_load_50[0 + 2] = _v4.z;
                    _vec_load_50[0 + 3] = _v4.w;
                }
                float _vec_load_51[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 1024) + 0);
                    _vec_load_51[0 + 0] = _v4.x;
                    _vec_load_51[0 + 1] = _v4.y;
                    _vec_load_51[0 + 2] = _v4.z;
                    _vec_load_51[0 + 3] = _v4.w;
                }
                float _vec_load_52[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 1536) + 0);
                    _vec_load_52[0 + 0] = _v4.x;
                    _vec_load_52[0 + 1] = _v4.y;
                    _vec_load_52[0 + 2] = _v4.z;
                    _vec_load_52[0 + 3] = _v4.w;
                }
                float _vec_load_53[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 2048) + 0);
                    _vec_load_53[0 + 0] = _v4.x;
                    _vec_load_53[0 + 1] = _v4.y;
                    _vec_load_53[0 + 2] = _v4.z;
                    _vec_load_53[0 + 3] = _v4.w;
                }
                float _vec_load_54[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 2560) + 0);
                    _vec_load_54[0 + 0] = _v4.x;
                    _vec_load_54[0 + 1] = _v4.y;
                    _vec_load_54[0 + 2] = _v4.z;
                    _vec_load_54[0 + 3] = _v4.w;
                }
                float _vec_load_55[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 3072) + 0);
                    _vec_load_55[0 + 0] = _v4.x;
                    _vec_load_55[0 + 1] = _v4.y;
                    _vec_load_55[0 + 2] = _v4.z;
                    _vec_load_55[0 + 3] = _v4.w;
                }
                float _vec_load_56[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_81 + 3584) + 0);
                    _vec_load_56[0 + 0] = _v4.x;
                    _vec_load_56[0 + 1] = _v4.y;
                    _vec_load_56[0 + 2] = _v4.z;
                    _vec_load_56[0 + 3] = _v4.w;
                }
                dq_80[0] = dq_80[0] + _vec_load_49[0];
                dq_80[4] = dq_80[4] + _vec_load_50[0];
                dq_80[8] = dq_80[8] + _vec_load_51[0];
                dq_80[12] = dq_80[12] + _vec_load_52[0];
                dq_80[16] = dq_80[16] + _vec_load_53[0];
                dq_80[20] = dq_80[20] + _vec_load_54[0];
                dq_80[24] = dq_80[24] + _vec_load_55[0];
                dq_80[28] = dq_80[28] + _vec_load_56[0];
                dq_80[1] = dq_80[1] + _vec_load_49[1];
                dq_80[5] = dq_80[5] + _vec_load_50[1];
                dq_80[9] = dq_80[9] + _vec_load_51[1];
                dq_80[13] = dq_80[13] + _vec_load_52[1];
                dq_80[17] = dq_80[17] + _vec_load_53[1];
                dq_80[21] = dq_80[21] + _vec_load_54[1];
                dq_80[25] = dq_80[25] + _vec_load_55[1];
                dq_80[29] = dq_80[29] + _vec_load_56[1];
                dq_80[2] = dq_80[2] + _vec_load_49[2];
                dq_80[6] = dq_80[6] + _vec_load_50[2];
                dq_80[10] = dq_80[10] + _vec_load_51[2];
                dq_80[14] = dq_80[14] + _vec_load_52[2];
                dq_80[18] = dq_80[18] + _vec_load_53[2];
                dq_80[22] = dq_80[22] + _vec_load_54[2];
                dq_80[26] = dq_80[26] + _vec_load_55[2];
                dq_80[30] = dq_80[30] + _vec_load_56[2];
                dq_80[3] = dq_80[3] + _vec_load_49[3];
                dq_80[7] = dq_80[7] + _vec_load_50[3];
                dq_80[11] = dq_80[11] + _vec_load_51[3];
                dq_80[15] = dq_80[15] + _vec_load_52[3];
                dq_80[19] = dq_80[19] + _vec_load_53[3];
                dq_80[23] = dq_80[23] + _vec_load_54[3];
                dq_80[27] = dq_80[27] + _vec_load_55[3];
                dq_80[31] = dq_80[31] + _vec_load_56[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_80[0 + 0], dq_80[0 + 1], dq_80[0 + 2], dq_80[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[4 + 0], dq_80[4 + 1], dq_80[4 + 2], dq_80[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[8 + 0], dq_80[8 + 1], dq_80[8 + 2], dq_80[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[12 + 0], dq_80[12 + 1], dq_80[12 + 2], dq_80[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[16 + 0], dq_80[16 + 1], dq_80[16 + 2], dq_80[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[20 + 0], dq_80[20 + 1], dq_80[20 + 2], dq_80[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[24 + 0], dq_80[24 + 1], dq_80[24 + 2], dq_80[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_80[28 + 0], dq_80[28 + 1], dq_80[28 + 2], dq_80[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_81 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_80_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_80[_lp*2 + 0], dq_80[_lp*2+1 + 0]));
                dq_80_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_82 = 8 * (m4 / 2) + r8;
            int qchunk_83 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_84 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_82 * 128) + (unsigned int)((qchunk_83 ^ r8) * 16);
            uint32_t _stmatrix_addr_88 = static_cast<uint32_t>(qaddr_84);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_88), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[3]))
                : "memory");
            int qhead_85 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_86 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_87 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_85 * 128) + (unsigned int)((qchunk_86 ^ r8) * 16);
            uint32_t _stmatrix_addr_89 = static_cast<uint32_t>(qaddr_87);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_89), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[7]))
                : "memory");
            int qhead_88 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_89 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_90 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_88 * 128) + (unsigned int)((qchunk_89 ^ r8) * 16);
            uint32_t _stmatrix_addr_90 = static_cast<uint32_t>(qaddr_90);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_90), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[11]))
                : "memory");
            int qhead_91 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_92 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_93 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_91 * 128) + (unsigned int)((qchunk_92 ^ r8) * 16);
            uint32_t _stmatrix_addr_91 = static_cast<uint32_t>(qaddr_93);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_91), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_80_bf16[15]))
                : "memory");
            float dq_94[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_94[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_94[31]))
                : "r"(tmem_tmem + 192 + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_95 = pbase + 28672 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dq_94[0 + 0], dq_94[0 + 1], dq_94[0 + 2], dq_94[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[4 + 0], dq_94[4 + 1], dq_94[4 + 2], dq_94[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[8 + 0], dq_94[8 + 1], dq_94[8 + 2], dq_94[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[12 + 0], dq_94[12 + 1], dq_94[12 + 2], dq_94[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[16 + 0], dq_94[16 + 1], dq_94[16 + 2], dq_94[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[20 + 0], dq_94[20 + 1], dq_94[20 + 2], dq_94[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[24 + 0], dq_94[24 + 1], dq_94[24 + 2], dq_94[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dq_94[28 + 0], dq_94[28 + 1], dq_94[28 + 2], dq_94[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_95 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_57[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_95 + 0);
                    _vec_load_57[0 + 0] = _v4.x;
                    _vec_load_57[0 + 1] = _v4.y;
                    _vec_load_57[0 + 2] = _v4.z;
                    _vec_load_57[0 + 3] = _v4.w;
                }
                float _vec_load_58[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 512) + 0);
                    _vec_load_58[0 + 0] = _v4.x;
                    _vec_load_58[0 + 1] = _v4.y;
                    _vec_load_58[0 + 2] = _v4.z;
                    _vec_load_58[0 + 3] = _v4.w;
                }
                float _vec_load_59[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 1024) + 0);
                    _vec_load_59[0 + 0] = _v4.x;
                    _vec_load_59[0 + 1] = _v4.y;
                    _vec_load_59[0 + 2] = _v4.z;
                    _vec_load_59[0 + 3] = _v4.w;
                }
                float _vec_load_60[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 1536) + 0);
                    _vec_load_60[0 + 0] = _v4.x;
                    _vec_load_60[0 + 1] = _v4.y;
                    _vec_load_60[0 + 2] = _v4.z;
                    _vec_load_60[0 + 3] = _v4.w;
                }
                float _vec_load_61[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 2048) + 0);
                    _vec_load_61[0 + 0] = _v4.x;
                    _vec_load_61[0 + 1] = _v4.y;
                    _vec_load_61[0 + 2] = _v4.z;
                    _vec_load_61[0 + 3] = _v4.w;
                }
                float _vec_load_62[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 2560) + 0);
                    _vec_load_62[0 + 0] = _v4.x;
                    _vec_load_62[0 + 1] = _v4.y;
                    _vec_load_62[0 + 2] = _v4.z;
                    _vec_load_62[0 + 3] = _v4.w;
                }
                float _vec_load_63[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 3072) + 0);
                    _vec_load_63[0 + 0] = _v4.x;
                    _vec_load_63[0 + 1] = _v4.y;
                    _vec_load_63[0 + 2] = _v4.z;
                    _vec_load_63[0 + 3] = _v4.w;
                }
                float _vec_load_64[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_95 + 3584) + 0);
                    _vec_load_64[0 + 0] = _v4.x;
                    _vec_load_64[0 + 1] = _v4.y;
                    _vec_load_64[0 + 2] = _v4.z;
                    _vec_load_64[0 + 3] = _v4.w;
                }
                dq_94[0] = dq_94[0] + _vec_load_57[0];
                dq_94[4] = dq_94[4] + _vec_load_58[0];
                dq_94[8] = dq_94[8] + _vec_load_59[0];
                dq_94[12] = dq_94[12] + _vec_load_60[0];
                dq_94[16] = dq_94[16] + _vec_load_61[0];
                dq_94[20] = dq_94[20] + _vec_load_62[0];
                dq_94[24] = dq_94[24] + _vec_load_63[0];
                dq_94[28] = dq_94[28] + _vec_load_64[0];
                dq_94[1] = dq_94[1] + _vec_load_57[1];
                dq_94[5] = dq_94[5] + _vec_load_58[1];
                dq_94[9] = dq_94[9] + _vec_load_59[1];
                dq_94[13] = dq_94[13] + _vec_load_60[1];
                dq_94[17] = dq_94[17] + _vec_load_61[1];
                dq_94[21] = dq_94[21] + _vec_load_62[1];
                dq_94[25] = dq_94[25] + _vec_load_63[1];
                dq_94[29] = dq_94[29] + _vec_load_64[1];
                dq_94[2] = dq_94[2] + _vec_load_57[2];
                dq_94[6] = dq_94[6] + _vec_load_58[2];
                dq_94[10] = dq_94[10] + _vec_load_59[2];
                dq_94[14] = dq_94[14] + _vec_load_60[2];
                dq_94[18] = dq_94[18] + _vec_load_61[2];
                dq_94[22] = dq_94[22] + _vec_load_62[2];
                dq_94[26] = dq_94[26] + _vec_load_63[2];
                dq_94[30] = dq_94[30] + _vec_load_64[2];
                dq_94[3] = dq_94[3] + _vec_load_57[3];
                dq_94[7] = dq_94[7] + _vec_load_58[3];
                dq_94[11] = dq_94[11] + _vec_load_59[3];
                dq_94[15] = dq_94[15] + _vec_load_60[3];
                dq_94[19] = dq_94[19] + _vec_load_61[3];
                dq_94[23] = dq_94[23] + _vec_load_62[3];
                dq_94[27] = dq_94[27] + _vec_load_63[3];
                dq_94[31] = dq_94[31] + _vec_load_64[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dq_94[0 + 0], dq_94[0 + 1], dq_94[0 + 2], dq_94[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[4 + 0], dq_94[4 + 1], dq_94[4 + 2], dq_94[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[8 + 0], dq_94[8 + 1], dq_94[8 + 2], dq_94[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[12 + 0], dq_94[12 + 1], dq_94[12 + 2], dq_94[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[16 + 0], dq_94[16 + 1], dq_94[16 + 2], dq_94[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[20 + 0], dq_94[20 + 1], dq_94[20 + 2], dq_94[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[24 + 0], dq_94[24 + 1], dq_94[24 + 2], dq_94[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dq_94[28 + 0], dq_94[28 + 1], dq_94[28 + 2], dq_94[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_95 + 3584) = _v4;
                    }
                }
            }
            uint32_t dq_94_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_94[_lp*2 + 0], dq_94[_lp*2+1 + 0]));
                dq_94_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_96 = 8 * (m4 / 2) + r8;
            int qchunk_97 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_98 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_96 * 128) + (unsigned int)((qchunk_97 ^ r8) * 16);
            uint32_t _stmatrix_addr_100 = static_cast<uint32_t>(qaddr_98);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_100), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[3]))
                : "memory");
            int qhead_99 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_100 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_101 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_99 * 128) + (unsigned int)((qchunk_100 ^ r8) * 16);
            uint32_t _stmatrix_addr_101 = static_cast<uint32_t>(qaddr_101);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_101), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[7]))
                : "memory");
            int qhead_102 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_103 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_104 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_102 * 128) + (unsigned int)((qchunk_103 ^ r8) * 16);
            uint32_t _stmatrix_addr_102 = static_cast<uint32_t>(qaddr_104);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_102), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[11]))
                : "memory");
            int qhead_105 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_106 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_107 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_105 * 128) + (unsigned int)((qchunk_106 ^ r8) * 16);
            uint32_t _stmatrix_addr_103 = static_cast<uint32_t>(qaddr_107);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_103), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_94_bf16[15]))
                : "memory");
            float dqr[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dqr[0])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[1])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[2])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[3])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[4])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[5])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[6])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[7])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[8])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[9])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[10])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[11])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[12])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[13])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[14])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[15])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[16])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[17])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[18])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[19])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[20])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[21])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[22])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[23])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[24])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[25])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[26])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[27])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[28])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[29])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[30])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[31]))
                : "r"(tmem_tmem + 448));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            long long pidx_108 = pbase + 32768 + (long long)(ctid * 4);
            if (dq_mode == 1) {
                {
                    float4 _v4 = make_float4(dqr[0 + 0], dqr[0 + 1], dqr[0 + 2], dqr[0 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[4 + 0], dqr[4 + 1], dqr[4 + 2], dqr[4 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 512) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[8 + 0], dqr[8 + 1], dqr[8 + 2], dqr[8 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 1024) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[12 + 0], dqr[12 + 1], dqr[12 + 2], dqr[12 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 1536) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[16 + 0], dqr[16 + 1], dqr[16 + 2], dqr[16 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 2048) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[20 + 0], dqr[20 + 1], dqr[20 + 2], dqr[20 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 2560) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[24 + 0], dqr[24 + 1], dqr[24 + 2], dqr[24 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 3072) = _v4;
                }
                {
                    float4 _v4 = make_float4(dqr[28 + 0], dqr[28 + 1], dqr[28 + 2], dqr[28 + 3]);
                    *reinterpret_cast<float4*>(dq_partial + pidx_108 + 3584) = _v4;
                }
            }
            if (dq_mode >= 2) {
                float _vec_load_65[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + pidx_108 + 0);
                    _vec_load_65[0 + 0] = _v4.x;
                    _vec_load_65[0 + 1] = _v4.y;
                    _vec_load_65[0 + 2] = _v4.z;
                    _vec_load_65[0 + 3] = _v4.w;
                }
                float _vec_load_66[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 512) + 0);
                    _vec_load_66[0 + 0] = _v4.x;
                    _vec_load_66[0 + 1] = _v4.y;
                    _vec_load_66[0 + 2] = _v4.z;
                    _vec_load_66[0 + 3] = _v4.w;
                }
                float _vec_load_67[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 1024) + 0);
                    _vec_load_67[0 + 0] = _v4.x;
                    _vec_load_67[0 + 1] = _v4.y;
                    _vec_load_67[0 + 2] = _v4.z;
                    _vec_load_67[0 + 3] = _v4.w;
                }
                float _vec_load_68[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 1536) + 0);
                    _vec_load_68[0 + 0] = _v4.x;
                    _vec_load_68[0 + 1] = _v4.y;
                    _vec_load_68[0 + 2] = _v4.z;
                    _vec_load_68[0 + 3] = _v4.w;
                }
                float _vec_load_69[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 2048) + 0);
                    _vec_load_69[0 + 0] = _v4.x;
                    _vec_load_69[0 + 1] = _v4.y;
                    _vec_load_69[0 + 2] = _v4.z;
                    _vec_load_69[0 + 3] = _v4.w;
                }
                float _vec_load_70[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 2560) + 0);
                    _vec_load_70[0 + 0] = _v4.x;
                    _vec_load_70[0 + 1] = _v4.y;
                    _vec_load_70[0 + 2] = _v4.z;
                    _vec_load_70[0 + 3] = _v4.w;
                }
                float _vec_load_71[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 3072) + 0);
                    _vec_load_71[0 + 0] = _v4.x;
                    _vec_load_71[0 + 1] = _v4.y;
                    _vec_load_71[0 + 2] = _v4.z;
                    _vec_load_71[0 + 3] = _v4.w;
                }
                float _vec_load_72[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(dq_partial + (pidx_108 + 3584) + 0);
                    _vec_load_72[0 + 0] = _v4.x;
                    _vec_load_72[0 + 1] = _v4.y;
                    _vec_load_72[0 + 2] = _v4.z;
                    _vec_load_72[0 + 3] = _v4.w;
                }
                dqr[0] = dqr[0] + _vec_load_65[0];
                dqr[4] = dqr[4] + _vec_load_66[0];
                dqr[8] = dqr[8] + _vec_load_67[0];
                dqr[12] = dqr[12] + _vec_load_68[0];
                dqr[16] = dqr[16] + _vec_load_69[0];
                dqr[20] = dqr[20] + _vec_load_70[0];
                dqr[24] = dqr[24] + _vec_load_71[0];
                dqr[28] = dqr[28] + _vec_load_72[0];
                dqr[1] = dqr[1] + _vec_load_65[1];
                dqr[5] = dqr[5] + _vec_load_66[1];
                dqr[9] = dqr[9] + _vec_load_67[1];
                dqr[13] = dqr[13] + _vec_load_68[1];
                dqr[17] = dqr[17] + _vec_load_69[1];
                dqr[21] = dqr[21] + _vec_load_70[1];
                dqr[25] = dqr[25] + _vec_load_71[1];
                dqr[29] = dqr[29] + _vec_load_72[1];
                dqr[2] = dqr[2] + _vec_load_65[2];
                dqr[6] = dqr[6] + _vec_load_66[2];
                dqr[10] = dqr[10] + _vec_load_67[2];
                dqr[14] = dqr[14] + _vec_load_68[2];
                dqr[18] = dqr[18] + _vec_load_69[2];
                dqr[22] = dqr[22] + _vec_load_70[2];
                dqr[26] = dqr[26] + _vec_load_71[2];
                dqr[30] = dqr[30] + _vec_load_72[2];
                dqr[3] = dqr[3] + _vec_load_65[3];
                dqr[7] = dqr[7] + _vec_load_66[3];
                dqr[11] = dqr[11] + _vec_load_67[3];
                dqr[15] = dqr[15] + _vec_load_68[3];
                dqr[19] = dqr[19] + _vec_load_69[3];
                dqr[23] = dqr[23] + _vec_load_70[3];
                dqr[27] = dqr[27] + _vec_load_71[3];
                dqr[31] = dqr[31] + _vec_load_72[3];
                if (dq_mode == 2) {
                    {
                        float4 _v4 = make_float4(dqr[0 + 0], dqr[0 + 1], dqr[0 + 2], dqr[0 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[4 + 0], dqr[4 + 1], dqr[4 + 2], dqr[4 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 512) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[8 + 0], dqr[8 + 1], dqr[8 + 2], dqr[8 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 1024) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[12 + 0], dqr[12 + 1], dqr[12 + 2], dqr[12 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 1536) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[16 + 0], dqr[16 + 1], dqr[16 + 2], dqr[16 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 2048) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[20 + 0], dqr[20 + 1], dqr[20 + 2], dqr[20 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 2560) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[24 + 0], dqr[24 + 1], dqr[24 + 2], dqr[24 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 3072) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(dqr[28 + 0], dqr[28 + 1], dqr[28 + 2], dqr[28 + 3]);
                        *reinterpret_cast<float4*>(dq_partial + pidx_108 + 3584) = _v4;
                    }
                }
            }
            uint32_t dqr_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dqr[_lp*2 + 0], dqr[_lp*2+1 + 0]));
                dqr_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int rhead = 8 * (m4 / 2) + r8;
            int rchunk = 2 * w + m4 % 2;
            unsigned int raddr = k_blk_addr + 65536 + (unsigned int)(rhead * 128) + (unsigned int)((rchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_112 = static_cast<uint32_t>(raddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_112), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[3]))
                : "memory");
            int rhead_109 = 16 + 8 * (m4 / 2) + r8;
            int rchunk_110 = 2 * w + m4 % 2;
            unsigned int raddr_111 = k_blk_addr + 65536 + (unsigned int)(rhead_109 * 128) + (unsigned int)((rchunk_110 ^ r8) * 16);
            uint32_t _stmatrix_addr_113 = static_cast<uint32_t>(raddr_111);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_113), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[7]))
                : "memory");
            int rhead_112 = 32 + 8 * (m4 / 2) + r8;
            int rchunk_113 = 2 * w + m4 % 2;
            unsigned int raddr_114 = k_blk_addr + 65536 + (unsigned int)(rhead_112 * 128) + (unsigned int)((rchunk_113 ^ r8) * 16);
            uint32_t _stmatrix_addr_114 = static_cast<uint32_t>(raddr_114);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_114), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[11]))
                : "memory");
            int rhead_115 = 48 + 8 * (m4 / 2) + r8;
            int rchunk_116 = 2 * w + m4 % 2;
            unsigned int raddr_117 = k_blk_addr + 65536 + (unsigned int)(rhead_115 * 128) + (unsigned int)((rchunk_116 ^ r8) * 16);
            uint32_t _stmatrix_addr_115 = static_cast<uint32_t>(raddr_117);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_115), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[15]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w == 0) {
                if (dq_mode == 0 || dq_mode == 3) {
                    if (elect_sync()) {
                        tma_store_4d((&dq_latent), 0, 0, 0, token, k_blk_addr);
                        tma_store_4d((&dq_latent), 0, 0, 1, token, k_blk_addr + 8192);
                        tma_store_4d((&dq_latent), 0, 0, 2, token, k_blk_addr + 16384);
                        tma_store_4d((&dq_latent), 0, 0, 3, token, k_blk_addr + 24576);
                        tma_store_4d((&dq_latent), 0, 0, 4, token, k_blk_addr + 32768);
                        tma_store_4d((&dq_latent), 0, 0, 5, token, k_blk_addr + 40960);
                        tma_store_4d((&dq_latent), 0, 0, 6, token, k_blk_addr + 49152);
                        tma_store_4d((&dq_latent), 0, 0, 7, token, k_blk_addr + 57344);
                        tma_store_4d((&dq_rope), 0, 0, 0, token, k_blk_addr + 65536);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }
    // ---- Role: reduce ----
    if (warp >= 8 && warp <= 15) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.inc.sync.aligned.u32 104;");
#else
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
#endif
        { // reduce_main
            int lane_0_1 = lane;
            int w2 = (warp - 8) % 4;
            int wg = (warp - 8) / 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_2 = blocks_word[0];
            int lpos = 32 * w2 + 4 * (lane_0_1 / 4);
            int rpos = 16 * w2 + 2 * (lane_0_1 / 4);
            int npos = 32 * w2 + 8 * (lane_0_1 >> 2 & 3) + 4 * (lane_0_1 >> 4);
            int nrpos = 16 * w2 + (lane_0_1 >> 2 & 6) + 8 * (lane_0_1 >> 2 & 1);
            int spos = 32 * w2 + (lane_0_1 >> 2);
            int srpos = 16 * w2 + (lane_0_1 >> 2);
            #pragma unroll 1
            for (int i_2 = 0; i_2 < blocks_2; i_2++) {
                unsigned int par_1 = i_2 & 1;
                int slot_2 = i_2 % 2;
                mbarrier_wait(idx_full_addr + (slot_2) * 8, i_2 / 2 & 1);
                int keys[8];
                keys[0] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4)];
                keys[1] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4) + 1];
                keys[2] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4)];
                keys[3] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4) + 1];
                keys[4] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4)];
                keys[5] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4) + 1];
                keys[6] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4)];
                keys[7] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4) + 1];
                if (lane_0_1 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_2) * 8);
                }
                int rows_1[8];
                rows_1[0] = keys[0];
                rows_1[1] = keys[1];
                rows_1[2] = keys[2];
                rows_1[3] = keys[3];
                rows_1[4] = keys[4];
                rows_1[5] = keys[5];
                rows_1[6] = keys[6];
                rows_1[7] = keys[7];
                if (dkv_has_map != 0) {
                    if (keys[0] >= 0) {
                        rows_1[0] = dkv_dst_map[(long long)keys[0]];
                    }
                    if (keys[1] >= 0) {
                        rows_1[1] = dkv_dst_map[(long long)keys[1]];
                    }
                    if (keys[2] >= 0) {
                        rows_1[2] = dkv_dst_map[(long long)keys[2]];
                    }
                    if (keys[3] >= 0) {
                        rows_1[3] = dkv_dst_map[(long long)keys[3]];
                    }
                    if (keys[4] >= 0) {
                        rows_1[4] = dkv_dst_map[(long long)keys[4]];
                    }
                    if (keys[5] >= 0) {
                        rows_1[5] = dkv_dst_map[(long long)keys[5]];
                    }
                    if (keys[6] >= 0) {
                        rows_1[6] = dkv_dst_map[(long long)keys[6]];
                    }
                    if (keys[7] >= 0) {
                        rows_1[7] = dkv_dst_map[(long long)keys[7]];
                    }
                }
                mbarrier_wait(dkv_a_full_addr, par_1);
                float a0[32];
                float a1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_a_drained_addr);
                float rk[16];
                {
                    mbarrier_wait(dkr_full_addr, par_1);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rk[0])), "=r"(*reinterpret_cast<uint32_t*>(&rk[1])), "=r"(*reinterpret_cast<uint32_t*>(&rk[2])), "=r"(*reinterpret_cast<uint32_t*>(&rk[3])), "=r"(*reinterpret_cast<uint32_t*>(&rk[4])), "=r"(*reinterpret_cast<uint32_t*>(&rk[5])), "=r"(*reinterpret_cast<uint32_t*>(&rk[6])), "=r"(*reinterpret_cast<uint32_t*>(&rk[7])), "=r"(*reinterpret_cast<uint32_t*>(&rk[8])), "=r"(*reinterpret_cast<uint32_t*>(&rk[9])), "=r"(*reinterpret_cast<uint32_t*>(&rk[10])), "=r"(*reinterpret_cast<uint32_t*>(&rk[11])), "=r"(*reinterpret_cast<uint32_t*>(&rk[12])), "=r"(*reinterpret_cast<uint32_t*>(&rk[13])), "=r"(*reinterpret_cast<uint32_t*>(&rk[14])), "=r"(*reinterpret_cast<uint32_t*>(&rk[15]))
                        : "r"(tmem_tmem + wg * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkr_drained_addr);
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16])), 8);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1)), 8);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16])), 8);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5)), 8);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[0] : rk[2]), 4);
                int key = keys[0];
                if (key >= 0) {
                    int row = rows_1[0];
                    long long base = (long long)row * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_2 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_3 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16]) : _shfl_xor_2)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1) : _shfl_xor_3)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_6 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_7 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16]) : _shfl_xor_6)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5) : _shfl_xor_7)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_8 : rk[0])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[2] : _shfl_xor_8)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17])), 8);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10)), 8);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17])), 8);
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14)), 8);
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[1] : rk[3]), 4);
                int key_0 = keys[1];
                if (key_0 >= 0) {
                    int row_1 = rows_1[1];
                    long long base_1 = (long long)row_1 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_11 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_12 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17]) : _shfl_xor_11)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10) : _shfl_xor_12)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_15 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_16 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17]) : _shfl_xor_15)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14) : _shfl_xor_16)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_1 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_17 : rk[1])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[3] : _shfl_xor_17)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20])), 8);
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19)), 8);
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20])), 8);
                float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23)), 8);
                float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[4] : rk[6]), 4);
                int key_1 = keys[2];
                if (key_1 >= 0) {
                    int row_2 = rows_1[2];
                    long long base_2 = (long long)row_2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_20 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_21 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20]) : _shfl_xor_20)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19) : _shfl_xor_21)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_24 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_25 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20]) : _shfl_xor_24)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23) : _shfl_xor_25)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_2 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_26 : rk[4])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[6] : _shfl_xor_26)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21])), 8);
                float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28)), 8);
                float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21])), 8);
                float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32)), 8);
                float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[5] : rk[7]), 4);
                int key_2 = keys[3];
                if (key_2 >= 0) {
                    int row_3 = rows_1[3];
                    long long base_3 = (long long)row_3 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_29 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_30 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21]) : _shfl_xor_29)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28) : _shfl_xor_30)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_33 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_34 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21]) : _shfl_xor_33)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32) : _shfl_xor_34)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_3 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_35 : rk[5])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[7] : _shfl_xor_35)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24])), 8);
                float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37)), 8);
                float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24])), 8);
                float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41)), 8);
                float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[8] : rk[10]), 4);
                int key_3 = keys[4];
                if (key_3 >= 0) {
                    int row_4 = rows_1[4];
                    long long base_4 = (long long)row_4 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_38 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_39 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24]) : _shfl_xor_38)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37) : _shfl_xor_39)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_42 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_43 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24]) : _shfl_xor_42)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41) : _shfl_xor_43)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_4 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_44 : rk[8])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[10] : _shfl_xor_44)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25])), 8);
                float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46)), 8);
                float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25])), 8);
                float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50)), 8);
                float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[9] : rk[11]), 4);
                int key_4 = keys[5];
                if (key_4 >= 0) {
                    int row_5 = rows_1[5];
                    long long base_5 = (long long)row_5 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_47 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_48 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25]) : _shfl_xor_47)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46) : _shfl_xor_48)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_51 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_52 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25]) : _shfl_xor_51)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50) : _shfl_xor_52)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_5 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_53 : rk[9])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[11] : _shfl_xor_53)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28])), 8);
                float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55)), 8);
                float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28])), 8);
                float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59)), 8);
                float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[12] : rk[14]), 4);
                int key_5 = keys[6];
                if (key_5 >= 0) {
                    int row_6 = rows_1[6];
                    long long base_6 = (long long)row_6 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_56 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_57 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28]) : _shfl_xor_56)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55) : _shfl_xor_57)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_60 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_61 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28]) : _shfl_xor_60)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59) : _shfl_xor_61)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_6 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_62 : rk[12])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[14] : _shfl_xor_62)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29])), 8);
                float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64)), 8);
                float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29])), 8);
                float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68)), 8);
                float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[13] : rk[15]), 4);
                int key_6 = keys[7];
                if (key_6 >= 0) {
                    int row_7 = rows_1[7];
                    long long base_7 = (long long)row_7 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_65 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_66 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29]) : _shfl_xor_65)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64) : _shfl_xor_66)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_69 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_70 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29]) : _shfl_xor_69)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68) : _shfl_xor_70)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_7 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_71 : rk[13])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[15] : _shfl_xor_71)), "l"(0x14F0000000000000ULL) : "memory");
                }
                mbarrier_wait(dkv_b_full_addr, par_1);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_b_drained_addr);
                float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16])), 8);
                float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73)), 8);
                float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16])), 8);
                float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77)), 8);
                int key2 = keys[0];
                if (key2 >= 0) {
                    int row2 = rows_1[0];
                    long long base2 = (long long)row2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_74 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_75 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16]) : _shfl_xor_74)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73) : _shfl_xor_75)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_78 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_79 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16]) : _shfl_xor_78)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77) : _shfl_xor_79)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17])), 8);
                float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81)), 8);
                float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17])), 8);
                float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85)), 8);
                int key2_7 = keys[1];
                if (key2_7 >= 0) {
                    int row2_1 = rows_1[1];
                    long long base2_1 = (long long)row2_1 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_82 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_83 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17]) : _shfl_xor_82)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81) : _shfl_xor_83)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_86 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_87 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17]) : _shfl_xor_86)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85) : _shfl_xor_87)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20])), 8);
                float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89)), 8);
                float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20])), 8);
                float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93)), 8);
                int key2_8 = keys[2];
                if (key2_8 >= 0) {
                    int row2_2 = rows_1[2];
                    long long base2_2 = (long long)row2_2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_90 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_91 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20]) : _shfl_xor_90)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89) : _shfl_xor_91)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_94 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_95 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20]) : _shfl_xor_94)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93) : _shfl_xor_95)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21])), 8);
                float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97)), 8);
                float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21])), 8);
                float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101)), 8);
                int key2_9 = keys[3];
                if (key2_9 >= 0) {
                    int row2_3 = rows_1[3];
                    long long base2_3 = (long long)row2_3 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_98 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_99 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21]) : _shfl_xor_98)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97) : _shfl_xor_99)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_102 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_103 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21]) : _shfl_xor_102)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101) : _shfl_xor_103)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24])), 8);
                float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105)), 8);
                float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24])), 8);
                float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109)), 8);
                int key2_10 = keys[4];
                if (key2_10 >= 0) {
                    int row2_4 = rows_1[4];
                    long long base2_4 = (long long)row2_4 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_106 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_107 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24]) : _shfl_xor_106)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105) : _shfl_xor_107)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_110 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_111 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24]) : _shfl_xor_110)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109) : _shfl_xor_111)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25])), 8);
                float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113)), 8);
                float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25])), 8);
                float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117)), 8);
                int key2_11 = keys[5];
                if (key2_11 >= 0) {
                    int row2_5 = rows_1[5];
                    long long base2_5 = (long long)row2_5 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_114 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_115 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25]) : _shfl_xor_114)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113) : _shfl_xor_115)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_118 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_119 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25]) : _shfl_xor_118)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117) : _shfl_xor_119)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28])), 8);
                float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121)), 8);
                float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28])), 8);
                float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125)), 8);
                int key2_12 = keys[6];
                if (key2_12 >= 0) {
                    int row2_6 = rows_1[6];
                    long long base2_6 = (long long)row2_6 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_122 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_123 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28]) : _shfl_xor_122)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121) : _shfl_xor_123)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_126 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_127 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28]) : _shfl_xor_126)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125) : _shfl_xor_127)), "l"(0x14F0000000000000ULL) : "memory");
                }
                float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29])), 8);
                float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129)), 8);
                float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29])), 8);
                float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133)), 8);
                int key2_13 = keys[7];
                if (key2_13 >= 0) {
                    int row2_7 = rows_1[7];
                    long long base2_7 = (long long)row2_7 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_130 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_131 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29]) : _shfl_xor_130)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129) : _shfl_xor_131)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_134 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_135 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29]) : _shfl_xor_134)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133) : _shfl_xor_135)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 16) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
#endif
        { // mma_main
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_3 = blocks_word[0];
            mbarrier_wait(qdo_full_addr, 0);
            if (elect_sync()) {
                #pragma unroll 1
                for (int i_3 = 0; i_3 < blocks_3; i_3++) {
                    unsigned int par_2 = i_3 & 1;
                    if (i_3 == 0) {
                        mbarrier_wait(k_full_addr, 0);
                        int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(dp_free_addr, par_2 ^ 1);
                    int _mma_a_lo_1 = ((do_k_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_1 = ((v_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (1048576))), "r"(0));
                    tcgen05_commit(dp_full_addr);
                    mbarrier_wait(p_full_addr, par_2);
                    if (i_3 > 0) {
                        mbarrier_wait(dkv_b_drained_addr, par_2 ^ 1);
                    }
                    int _mma_a_lo_2 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_2 = (((p_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_3 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    mbarrier_wait(ds_full_addr, par_2);
                    int _mma_a_lo_4 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_4 = (((ds_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem + (192))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_5 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_4), "r"((tmem_tmem + (256))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_6 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_4), "r"((tmem_tmem + (320))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_7 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_4), "r"((tmem_tmem + (384))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_8 = (((kr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 68256912;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_4), "r"((tmem_tmem + (448))), "r"(((i_3 == 0) ? 0 : 1)));
                    tcgen05_commit(k_free_addr);
                    int _mma_a_lo_9 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_9 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem + (64))), "r"(1));
                    int _mma_a_lo_10 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_9), "r"((tmem_tmem + (128))), "r"(1));
                    tcgen05_commit(dkv_a_full_addr);
                    int _mma_a_lo_11 = (((qr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 68191376;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_9), "r"(tmem_tmem), "r"(0));
                    tcgen05_commit(dkr_full_addr);
                    mbarrier_wait(dkv_a_drained_addr, par_2);
                    int _mma_a_lo_12 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_13 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    tcgen05_commit(p_free_addr);
                    {
                        int _mma_a_lo_14 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
                        int _mma_b_lo_14 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"((tmem_tmem + (64))), "r"(1));
                        int _mma_a_lo_15 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_14), "r"((tmem_tmem + (128))), "r"(1));
                        tcgen05_commit(dkv_b_full_addr);
                        tcgen05_commit(ds_free_addr);
                    }
                    if (blocks_3 > i_3 + 1) {
                        mbarrier_wait(dkr_drained_addr, par_2);
                        mbarrier_wait(k_full_addr, par_2 ^ 1);
                        mbarrier_wait(s_free_addr, par_2);
                        int _mma_a_lo_16 = ((q_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_16 = ((k_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                }
                tcgen05_commit(dq_done_addr);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 17) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
#endif
        { // load_main
            int token_1 = token_base + token_step * blockIdx.x;
            int lane_0_2 = lane;
            if (elect_sync()) {
                tma_4d_gmem2smem(q_k_addr, (&q_latent), 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr, (&dout), 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 8192, (&q_latent), 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 8192, (&dout), 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 16384, (&q_latent), 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 16384, (&dout), 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 24576, (&q_latent), 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 24576, (&dout), 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 32768, (&q_latent), 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 32768, (&dout), 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 40960, (&q_latent), 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 40960, (&dout), 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 49152, (&q_latent), 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 49152, (&dout), 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 57344, (&q_latent), 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 57344, (&dout), 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 65536, (&q_rope), 0, 0, 0, token_1, qdo_full_addr);
                mbarrier_arrive_expect_tx(qdo_full_addr, 139264);
            }
            stats[lane_0_2] = lse[token_1 * 64 + lane_0_2] * 1.4426950408889634f;
            stats[32 + lane_0_2] = lse[token_1 * 64 + 32 + lane_0_2] * 1.4426950408889634f;
            stats[64 + lane_0_2] = delta[token_1 * 64 + lane_0_2];
            stats[96 + lane_0_2] = delta[token_1 * 64 + 32 + lane_0_2];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(stats_full_addr);
        }
    }
    // ---- Role: metadata ----
    if (warp == 18) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
#endif
        { // metadata_main
            int token_2 = token_base + token_step * blockIdx.x;
            int lane_0_3 = lane;
            int active = topk;
            if (has_topk_length != 0) {
                int _max_0 = ((topk_length[token_2]) > (0) ? (topk_length[token_2]) : (0));
                int _min_0 = ((_max_0) < (topk) ? (_max_0) : (topk));
                active = _min_0;
            }
            long long row_base = (long long)indices_offset + (long long)token_2 * (long long)idx_stride;
            int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
            int bidv = blockIdx.x;
            long long scratch_base = (long long)bidv * (long long)topk;
            int count = pass_counts[bidv];
            int _max_1 = (((count + 63) / 64) > (1) ? ((count + 63) / 64) : (1));
            int blocks_4 = _max_1;
            if (lane_0_3 == 0) {
                blocks_word[0] = blocks_4;
                mbarrier_arrive(blocks_ready_addr);
            }
            if (lane_0_3 < 8) {
                #pragma unroll 1
                for (int i_4 = 0; i_4 < blocks_4; i_4++) {
                    int n = blocks_4 - 1 - i_4;
                    int slot_3 = i_4 % 2;
                    unsigned int mask = 0;
                    int position = n * 64 + lane_0_3 * 8;
                    int values[8];
                    int cpos2 = n * 64 + lane_0_3 * 8;
                    if (count >= n * 64 + 64 && (topk & 7) == 0) {
                        int _vec_load_0[8];
                        {
                            uint32_t _iv_0_0;
                            uint32_t _iv_0_1;
                            uint32_t _iv_0_2;
                            uint32_t _iv_0_3;
                            uint32_t _iv_0_4;
                            uint32_t _iv_0_5;
                            uint32_t _iv_0_6;
                            uint32_t _iv_0_7;
                            asm volatile("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(key_scratch + (scratch_base + (long long)cpos2) + (0))) : "memory");
                            _vec_load_0[0 + 0] = (int32_t)_iv_0_0;
                            _vec_load_0[0 + 1] = (int32_t)_iv_0_1;
                            _vec_load_0[0 + 2] = (int32_t)_iv_0_2;
                            _vec_load_0[0 + 3] = (int32_t)_iv_0_3;
                            _vec_load_0[0 + 4] = (int32_t)_iv_0_4;
                            _vec_load_0[0 + 5] = (int32_t)_iv_0_5;
                            _vec_load_0[0 + 6] = (int32_t)_iv_0_6;
                            _vec_load_0[0 + 7] = (int32_t)_iv_0_7;
                        }
                        values[0] = _vec_load_0[0];
                        values[1] = _vec_load_0[1];
                        values[2] = _vec_load_0[2];
                        values[3] = _vec_load_0[3];
                        values[4] = _vec_load_0[4];
                        values[5] = _vec_load_0[5];
                        values[6] = _vec_load_0[6];
                        values[7] = _vec_load_0[7];
                    } else {
                        int kk = -1;
                        if (count > cpos2) {
                            kk = key_scratch[scratch_base + (long long)cpos2];
                        }
                        values[0] = kk;
                        int kk_0 = -1;
                        if (count > cpos2 + 1) {
                            kk_0 = key_scratch[scratch_base + (long long)(cpos2 + 1)];
                        }
                        values[1] = kk_0;
                        int kk_1 = -1;
                        if (count > cpos2 + 2) {
                            kk_1 = key_scratch[scratch_base + (long long)(cpos2 + 2)];
                        }
                        values[2] = kk_1;
                        int kk_2 = -1;
                        if (count > cpos2 + 3) {
                            kk_2 = key_scratch[scratch_base + (long long)(cpos2 + 3)];
                        }
                        values[3] = kk_2;
                        int kk_3 = -1;
                        if (count > cpos2 + 4) {
                            kk_3 = key_scratch[scratch_base + (long long)(cpos2 + 4)];
                        }
                        values[4] = kk_3;
                        int kk_4 = -1;
                        if (count > cpos2 + 5) {
                            kk_4 = key_scratch[scratch_base + (long long)(cpos2 + 5)];
                        }
                        values[5] = kk_4;
                        int kk_5 = -1;
                        if (count > cpos2 + 6) {
                            kk_5 = key_scratch[scratch_base + (long long)(cpos2 + 6)];
                        }
                        values[6] = kk_5;
                        int kk_6 = -1;
                        if (count > cpos2 + 7) {
                            kk_6 = key_scratch[scratch_base + (long long)(cpos2 + 7)];
                        }
                        values[7] = kk_6;
                    }
                    mbarrier_wait(idx_free_addr + (slot_3) * 8, i_4 / 2 & 1 ^ 1);
                    int ok = (int)(values[0] >= 0 && values[0] < num_kv);
                    if (ok != 0) {
                        mask = mask | 1;
                    } else {
                        values[0] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8] = values[0];
                    int ok_0 = (int)(values[1] >= 0 && values[1] < num_kv);
                    if (ok_0 != 0) {
                        mask = mask | 2;
                    } else {
                        values[1] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 1] = values[1];
                    int ok_1 = (int)(values[2] >= 0 && values[2] < num_kv);
                    if (ok_1 != 0) {
                        mask = mask | 4;
                    } else {
                        values[2] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 2] = values[2];
                    int ok_2 = (int)(values[3] >= 0 && values[3] < num_kv);
                    if (ok_2 != 0) {
                        mask = mask | 8;
                    } else {
                        values[3] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 3] = values[3];
                    int ok_3 = (int)(values[4] >= 0 && values[4] < num_kv);
                    if (ok_3 != 0) {
                        mask = mask | 16;
                    } else {
                        values[4] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 4] = values[4];
                    int ok_4 = (int)(values[5] >= 0 && values[5] < num_kv);
                    if (ok_4 != 0) {
                        mask = mask | 32;
                    } else {
                        values[5] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 5] = values[5];
                    int ok_5 = (int)(values[6] >= 0 && values[6] < num_kv);
                    if (ok_5 != 0) {
                        mask = mask | 64;
                    } else {
                        values[6] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 6] = values[6];
                    int ok_6 = (int)(values[7] >= 0 && values[7] < num_kv);
                    if (ok_6 != 0) {
                        mask = mask | 128;
                    } else {
                        values[7] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 7] = values[7];
                    validity8[slot_3 * 320 + 256 + lane_0_3] = mask;
                    mbarrier_arrive(idx_full_addr + (slot_3) * 8);
                }
            }
        }
    }
    // ---- Role: spare ----
    if (warp == 19) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
#endif
        { // spare_main
            if (dq_mode >= 2) {
                int sbid = blockIdx.x;
                long long pfb = (long long)sbid * 36864 + (long long)(lane * 32);
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + pfb)));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 1024))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 2048))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 3072))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 4096))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 5120))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 6144))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 7168))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 8192))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 9216))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 10240))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 11264))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 12288))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 13312))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 14336))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 15360))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 16384))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 17408))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 18432))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 19456))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 20480))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 21504))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 22528))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 23552))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 24576))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 25600))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 26624))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 27648))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 28672))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 29696))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 30720))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 31744))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 32768))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 33792))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 34816))));
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(dq_partial + (pfb + 35840))));
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
