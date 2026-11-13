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


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_train_4487388f2950ac4f97a5(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap dout, const __grid_constant__ CUtensorMap dq_latent, const __grid_constant__ CUtensorMap dq_rope, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap k_rope, float* __restrict__ lse, float* __restrict__ delta, int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ dkv_acc, int* __restrict__ dkr_acc, int dkv_stride, int dkr_stride, int dkr_col0, int* __restrict__ dkv_dst_map, int dkv_has_map, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int* __restrict__ e_lat, int* __restrict__ e_rope, unsigned int* __restrict__ cnt, unsigned long long* __restrict__ cnt64, int* __restrict__ redo_mark, int* __restrict__ redo_count, int dq_mode, float* __restrict__ dq_partial)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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

    // Prologue
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

    // Kernel setup ops
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dout))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k_rope))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_latent))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_rope))) : "memory"); }

    // Mbarrier init (24 barriers at smem_raw[0..192))

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
            // idx_free: 2 barriers, init_count=388
            mbarrier_init(smem + 48, 388);
            mbarrier_init(smem + 56, 388);
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

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem = taddr;

    // ---- Role: gather ----
    if (warp <= 3) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
#endif
        { // gather_main
            int pw = warp;
            int ntok = 1;
            int tord = 0;
            #pragma unroll 1
            for (int tk = 0; tk < ntok; tk++) {
                int _parity_blocks_ready = (int)((unsigned int)(int)tk % 2);
                mbarrier_wait(blocks_ready_addr, _parity_blocks_ready);
                int blocks = blocks_word[0];
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int i = 0; i < blocks; i++) {
                        int slot = (tord + i) % 2;
                        int _stage_idx_full = (int)((unsigned int)(tord + i) % 2);
                        int _parity_idx_full = (int)((unsigned int)(tord + i) / 2 % 2);
                        mbarrier_wait(idx_full_addr + (_stage_idx_full) * 8, _parity_idx_full);
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
                        mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord + i) % 2)) * 8);
                        int _parity_k_free = (int)((unsigned int)(tord + i) % 2) ^ 1;
                        mbarrier_wait(k_free_addr, _parity_k_free);
                        mbarrier_arrive_expect_tx(k_full_addr, 18432);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(0), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(0), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(0), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(0), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 8192 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(64), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 8192 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(64), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 8192 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(64), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 8192 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(64), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 16384 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(128), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 16384 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(128), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 16384 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(128), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 16384 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(128), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 24576 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(192), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 24576 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(192), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 24576 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(192), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 24576 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(192), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 32768 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(256), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 32768 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(256), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 32768 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(256), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 32768 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(256), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 40960 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(320), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 40960 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(320), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 40960 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(320), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 40960 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(320), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 49152 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(384), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 49152 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(384), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 49152 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(384), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 49152 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(384), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 57344 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(448), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 57344 + 2048 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(448), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 57344 + 4096 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(448), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 57344 + 6144 + (unsigned int)(pw * 512)), "l"((&kv_latent)), "r"(448), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 65536 + (unsigned int)(pw * 512)), "l"((&k_rope)), "r"(0), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 65536 + 2048 + (unsigned int)(pw * 512)), "l"((&k_rope)), "r"(0), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 65536 + 4096 + (unsigned int)(pw * 512)), "l"((&k_rope)), "r"(0), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                            ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(k_blk_addr + 65536 + 6144 + (unsigned int)(pw * 512)), "l"((&k_rope)), "r"(0), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                tord += blocks;
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // compute_main
            int lane_0 = lane;
            int w = warp - 4;
            int lane_base = (tid & 96) << 16;
            int ntok_1 = 1;
            int tord_1 = 0;
            #pragma unroll 1
            for (int tk_1 = 0; tk_1 < ntok_1; tk_1++) {
                int tok = blockIdx.x + tk_1 * gridDim.x;
                int token = token_base + token_step * tok;
                int _parity_blocks_ready_1 = (int)((unsigned int)(int)tk_1 % 2);
                mbarrier_wait(blocks_ready_addr, _parity_blocks_ready_1);
                int blocks_1 = blocks_word[0];
                int _parity_stats_full = (int)((unsigned int)(int)tk_1 % 2);
                mbarrier_wait(stats_full_addr, _parity_stats_full);
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
                    int slot_1 = (tord_1 + i_1) % 2;
                    int _stage_idx_full_1 = (int)((unsigned int)(tord_1 + i_1) % 2);
                    int _parity_idx_full_1 = (int)((unsigned int)(tord_1 + i_1) / 2 % 2);
                    mbarrier_wait(idx_full_addr + (_stage_idx_full_1) * 8, _parity_idx_full_1);
                    unsigned int v0 = validity32[slot_1 * 80 + 64];
                    unsigned int v1 = validity32[slot_1 * 80 + 65];
                    mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord_1 + i_1) % 2)) * 8);
                    int _parity_s_full = (int)((unsigned int)(tord_1 + i_1) % 2);
                    mbarrier_wait(s_full_addr, _parity_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float s[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s[0])), "=r"(*reinterpret_cast<uint32_t*>(&s[1])), "=r"(*reinterpret_cast<uint32_t*>(&s[2])), "=r"(*reinterpret_cast<uint32_t*>(&s[3])), "=r"(*reinterpret_cast<uint32_t*>(&s[4])), "=r"(*reinterpret_cast<uint32_t*>(&s[5])), "=r"(*reinterpret_cast<uint32_t*>(&s[6])), "=r"(*reinterpret_cast<uint32_t*>(&s[7])), "=r"(*reinterpret_cast<uint32_t*>(&s[8])), "=r"(*reinterpret_cast<uint32_t*>(&s[9])), "=r"(*reinterpret_cast<uint32_t*>(&s[10])), "=r"(*reinterpret_cast<uint32_t*>(&s[11])), "=r"(*reinterpret_cast<uint32_t*>(&s[12])), "=r"(*reinterpret_cast<uint32_t*>(&s[13])), "=r"(*reinterpret_cast<uint32_t*>(&s[14])), "=r"(*reinterpret_cast<uint32_t*>(&s[15])), "=r"(*reinterpret_cast<uint32_t*>(&s[16])), "=r"(*reinterpret_cast<uint32_t*>(&s[17])), "=r"(*reinterpret_cast<uint32_t*>(&s[18])), "=r"(*reinterpret_cast<uint32_t*>(&s[19])), "=r"(*reinterpret_cast<uint32_t*>(&s[20])), "=r"(*reinterpret_cast<uint32_t*>(&s[21])), "=r"(*reinterpret_cast<uint32_t*>(&s[22])), "=r"(*reinterpret_cast<uint32_t*>(&s[23])), "=r"(*reinterpret_cast<uint32_t*>(&s[24])), "=r"(*reinterpret_cast<uint32_t*>(&s[25])), "=r"(*reinterpret_cast<uint32_t*>(&s[26])), "=r"(*reinterpret_cast<uint32_t*>(&s[27])), "=r"(*reinterpret_cast<uint32_t*>(&s[28])), "=r"(*reinterpret_cast<uint32_t*>(&s[29])), "=r"(*reinterpret_cast<uint32_t*>(&s[30])), "=r"(*reinterpret_cast<uint32_t*>(&s[31]))
                        : "r"(tmem_tmem + lane_base + 0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float p[32];
                    float _exp2_0 = approx_exp2(__fmaf_rn(s[0], scale_log2, -lse0));
                    p[0] = _exp2_0;
                    float _exp2_1 = approx_exp2(__fmaf_rn(s[2], scale_log2, -lse1));
                    p[2] = _exp2_1;
                    unsigned int bit_15_0 = v0 >> (unsigned int)kb & 1;
                    if (bit_15_0 == 0) {
                        p[0] = 0.0f;
                        p[2] = 0.0f;
                    }
                    float _exp2_2 = approx_exp2(__fmaf_rn(s[1], scale_log2, -lse0));
                    p[1] = _exp2_2;
                    float _exp2_3 = approx_exp2(__fmaf_rn(s[3], scale_log2, -lse1));
                    p[3] = _exp2_3;
                    unsigned int bit_15_1 = v0 >> (unsigned int)(1 + kb) & 1;
                    if (bit_15_1 == 0) {
                        p[1] = 0.0f;
                        p[3] = 0.0f;
                    }
                    float _exp2_4 = approx_exp2(__fmaf_rn(s[4], scale_log2, -lse0));
                    p[4] = _exp2_4;
                    float _exp2_5 = approx_exp2(__fmaf_rn(s[6], scale_log2, -lse1));
                    p[6] = _exp2_5;
                    unsigned int bit_15_2 = v0 >> (unsigned int)(8 + kb) & 1;
                    if (bit_15_2 == 0) {
                        p[4] = 0.0f;
                        p[6] = 0.0f;
                    }
                    float _exp2_6 = approx_exp2(__fmaf_rn(s[5], scale_log2, -lse0));
                    p[5] = _exp2_6;
                    float _exp2_7 = approx_exp2(__fmaf_rn(s[7], scale_log2, -lse1));
                    p[7] = _exp2_7;
                    unsigned int bit_15_3 = v0 >> (unsigned int)(9 + kb) & 1;
                    if (bit_15_3 == 0) {
                        p[5] = 0.0f;
                        p[7] = 0.0f;
                    }
                    float _exp2_8 = approx_exp2(__fmaf_rn(s[8], scale_log2, -lse0));
                    p[8] = _exp2_8;
                    float _exp2_9 = approx_exp2(__fmaf_rn(s[10], scale_log2, -lse1));
                    p[10] = _exp2_9;
                    unsigned int bit_15_4 = v0 >> (unsigned int)(16 + kb) & 1;
                    if (bit_15_4 == 0) {
                        p[8] = 0.0f;
                        p[10] = 0.0f;
                    }
                    float _exp2_10 = approx_exp2(__fmaf_rn(s[9], scale_log2, -lse0));
                    p[9] = _exp2_10;
                    float _exp2_11 = approx_exp2(__fmaf_rn(s[11], scale_log2, -lse1));
                    p[11] = _exp2_11;
                    unsigned int bit_15_5 = v0 >> (unsigned int)(17 + kb) & 1;
                    if (bit_15_5 == 0) {
                        p[9] = 0.0f;
                        p[11] = 0.0f;
                    }
                    float _exp2_12 = approx_exp2(__fmaf_rn(s[12], scale_log2, -lse0));
                    p[12] = _exp2_12;
                    float _exp2_13 = approx_exp2(__fmaf_rn(s[14], scale_log2, -lse1));
                    p[14] = _exp2_13;
                    unsigned int bit_15_6 = v0 >> (unsigned int)(24 + kb) & 1;
                    if (bit_15_6 == 0) {
                        p[12] = 0.0f;
                        p[14] = 0.0f;
                    }
                    float _exp2_14 = approx_exp2(__fmaf_rn(s[13], scale_log2, -lse0));
                    p[13] = _exp2_14;
                    float _exp2_15 = approx_exp2(__fmaf_rn(s[15], scale_log2, -lse1));
                    p[15] = _exp2_15;
                    unsigned int bit_15_7 = v0 >> (unsigned int)(25 + kb) & 1;
                    if (bit_15_7 == 0) {
                        p[13] = 0.0f;
                        p[15] = 0.0f;
                    }
                    float _exp2_16 = approx_exp2(__fmaf_rn(s[16], scale_log2, -lse0));
                    p[16] = _exp2_16;
                    float _exp2_17 = approx_exp2(__fmaf_rn(s[18], scale_log2, -lse1));
                    p[18] = _exp2_17;
                    unsigned int bit_15_8 = v1 >> (unsigned int)kb & 1;
                    if (bit_15_8 == 0) {
                        p[16] = 0.0f;
                        p[18] = 0.0f;
                    }
                    float _exp2_18 = approx_exp2(__fmaf_rn(s[17], scale_log2, -lse0));
                    p[17] = _exp2_18;
                    float _exp2_19 = approx_exp2(__fmaf_rn(s[19], scale_log2, -lse1));
                    p[19] = _exp2_19;
                    unsigned int bit_15_9 = v1 >> (unsigned int)(1 + kb) & 1;
                    if (bit_15_9 == 0) {
                        p[17] = 0.0f;
                        p[19] = 0.0f;
                    }
                    float _exp2_20 = approx_exp2(__fmaf_rn(s[20], scale_log2, -lse0));
                    p[20] = _exp2_20;
                    float _exp2_21 = approx_exp2(__fmaf_rn(s[22], scale_log2, -lse1));
                    p[22] = _exp2_21;
                    unsigned int bit_15_10 = v1 >> (unsigned int)(8 + kb) & 1;
                    if (bit_15_10 == 0) {
                        p[20] = 0.0f;
                        p[22] = 0.0f;
                    }
                    float _exp2_22 = approx_exp2(__fmaf_rn(s[21], scale_log2, -lse0));
                    p[21] = _exp2_22;
                    float _exp2_23 = approx_exp2(__fmaf_rn(s[23], scale_log2, -lse1));
                    p[23] = _exp2_23;
                    unsigned int bit_15_11 = v1 >> (unsigned int)(9 + kb) & 1;
                    if (bit_15_11 == 0) {
                        p[21] = 0.0f;
                        p[23] = 0.0f;
                    }
                    float _exp2_24 = approx_exp2(__fmaf_rn(s[24], scale_log2, -lse0));
                    p[24] = _exp2_24;
                    float _exp2_25 = approx_exp2(__fmaf_rn(s[26], scale_log2, -lse1));
                    p[26] = _exp2_25;
                    unsigned int bit_15_12 = v1 >> (unsigned int)(16 + kb) & 1;
                    if (bit_15_12 == 0) {
                        p[24] = 0.0f;
                        p[26] = 0.0f;
                    }
                    float _exp2_26 = approx_exp2(__fmaf_rn(s[25], scale_log2, -lse0));
                    p[25] = _exp2_26;
                    float _exp2_27 = approx_exp2(__fmaf_rn(s[27], scale_log2, -lse1));
                    p[27] = _exp2_27;
                    unsigned int bit_15_13 = v1 >> (unsigned int)(17 + kb) & 1;
                    if (bit_15_13 == 0) {
                        p[25] = 0.0f;
                        p[27] = 0.0f;
                    }
                    float _exp2_28 = approx_exp2(__fmaf_rn(s[28], scale_log2, -lse0));
                    p[28] = _exp2_28;
                    float _exp2_29 = approx_exp2(__fmaf_rn(s[30], scale_log2, -lse1));
                    p[30] = _exp2_29;
                    unsigned int bit_15_14 = v1 >> (unsigned int)(24 + kb) & 1;
                    if (bit_15_14 == 0) {
                        p[28] = 0.0f;
                        p[30] = 0.0f;
                    }
                    float _exp2_30 = approx_exp2(__fmaf_rn(s[29], scale_log2, -lse0));
                    p[29] = _exp2_30;
                    float _exp2_31 = approx_exp2(__fmaf_rn(s[31], scale_log2, -lse1));
                    p[31] = _exp2_31;
                    unsigned int bit_15_15 = v1 >> (unsigned int)(25 + kb) & 1;
                    if (bit_15_15 == 0) {
                        p[29] = 0.0f;
                        p[31] = 0.0f;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(s_free_addr);
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int _parity_p_free = (int)((unsigned int)(tord_1 + i_1) % 2) ^ 1;
                    mbarrier_wait(p_free_addr, _parity_p_free);
                    unsigned int p_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                        p_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int paddr_1_0 = p_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(paddr_1_0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[3]))
                        : "memory");
                    unsigned int paddr_1_1 = p_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(paddr_1_1);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[7]))
                        : "memory");
                    unsigned int paddr_1_2 = p_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(paddr_1_2);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[11]))
                        : "memory");
                    unsigned int paddr_1_3 = p_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(paddr_1_3);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[15]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    int _parity_dp_full = (int)((unsigned int)(tord_1 + i_1) % 2);
                    mbarrier_wait(dp_full_addr, _parity_dp_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float dp[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[1])), "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[3])), "=r"(*reinterpret_cast<uint32_t*>(&dp[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp[5])), "=r"(*reinterpret_cast<uint32_t*>(&dp[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp[7])), "=r"(*reinterpret_cast<uint32_t*>(&dp[8])), "=r"(*reinterpret_cast<uint32_t*>(&dp[9])), "=r"(*reinterpret_cast<uint32_t*>(&dp[10])), "=r"(*reinterpret_cast<uint32_t*>(&dp[11])), "=r"(*reinterpret_cast<uint32_t*>(&dp[12])), "=r"(*reinterpret_cast<uint32_t*>(&dp[13])), "=r"(*reinterpret_cast<uint32_t*>(&dp[14])), "=r"(*reinterpret_cast<uint32_t*>(&dp[15])), "=r"(*reinterpret_cast<uint32_t*>(&dp[16])), "=r"(*reinterpret_cast<uint32_t*>(&dp[17])), "=r"(*reinterpret_cast<uint32_t*>(&dp[18])), "=r"(*reinterpret_cast<uint32_t*>(&dp[19])), "=r"(*reinterpret_cast<uint32_t*>(&dp[20])), "=r"(*reinterpret_cast<uint32_t*>(&dp[21])), "=r"(*reinterpret_cast<uint32_t*>(&dp[22])), "=r"(*reinterpret_cast<uint32_t*>(&dp[23])), "=r"(*reinterpret_cast<uint32_t*>(&dp[24])), "=r"(*reinterpret_cast<uint32_t*>(&dp[25])), "=r"(*reinterpret_cast<uint32_t*>(&dp[26])), "=r"(*reinterpret_cast<uint32_t*>(&dp[27])), "=r"(*reinterpret_cast<uint32_t*>(&dp[28])), "=r"(*reinterpret_cast<uint32_t*>(&dp[29])), "=r"(*reinterpret_cast<uint32_t*>(&dp[30])), "=r"(*reinterpret_cast<uint32_t*>(&dp[31]))
                        : "r"(tmem_tmem + lane_base + 0 + 1048576));
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
                    int _parity_ds_free = (int)((unsigned int)(tord_1 + i_1) % 2) ^ 1;
                    mbarrier_wait(ds_free_addr, _parity_ds_free);
                    unsigned int ds_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ds[_lp*2 + 0], ds[_lp*2+1 + 0]));
                        ds_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int daddr_1_0 = ds_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(daddr_1_0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[3]))
                        : "memory");
                    unsigned int daddr_1_1 = ds_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(daddr_1_1);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[7]))
                        : "memory");
                    unsigned int daddr_1_2 = ds_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(daddr_1_2);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[11]))
                        : "memory");
                    unsigned int daddr_1_3 = ds_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                    uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(daddr_1_3);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[15]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(ds_full_addr);
                }
                int _parity_dq_done = (int)((unsigned int)(int)tk_1 % 2);
                mbarrier_wait(dq_done_addr, _parity_dq_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long pbase = (long long)tok * 36864ll + (long long)((w * 32 + lane_0) * 4);
                float dq_1_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_0[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 0 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_0[_lp*2 + 0], dq_1_0[_lp*2+1 + 0]));
                    dq_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_0 = 8 * (m4 / 2) + r8;
                int qchunk_1_0 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_0 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_0 * 128) + (unsigned int)((qchunk_1_0 ^ r8) * 16);
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(qaddr_1_0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[3]))
                    : "memory");
                int qhead_2_1 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_1 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_1 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_1 * 128) + (unsigned int)((qchunk_1_1 ^ r8) * 16);
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(qaddr_1_1);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[7]))
                    : "memory");
                int qhead_2_2 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_2 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_2 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_2 * 128) + (unsigned int)((qchunk_1_2 ^ r8) * 16);
                uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(qaddr_1_2);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[11]))
                    : "memory");
                int qhead_2_3 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_3 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_3 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_3 * 128) + (unsigned int)((qchunk_1_3 ^ r8) * 16);
                uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(qaddr_1_3);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[15]))
                    : "memory");
                float dq_1_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_1[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 0 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_1[_lp*2 + 0], dq_1_1[_lp*2+1 + 0]));
                    dq_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_4 = 8 * (m4 / 2) + r8;
                int qchunk_1_4 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_4 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_4 * 128) + (unsigned int)((qchunk_1_4 ^ r8) * 16);
                uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(qaddr_1_4);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[3]))
                    : "memory");
                int qhead_2_5 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_5 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_5 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_5 * 128) + (unsigned int)((qchunk_1_5 ^ r8) * 16);
                uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(qaddr_1_5);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[7]))
                    : "memory");
                int qhead_2_6 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_6 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_6 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_6 * 128) + (unsigned int)((qchunk_1_6 ^ r8) * 16);
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(qaddr_1_6);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[11]))
                    : "memory");
                int qhead_2_7 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_7 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_7 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_7 * 128) + (unsigned int)((qchunk_1_7 ^ r8) * 16);
                uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(qaddr_1_7);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[15]))
                    : "memory");
                float dq_1_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_2[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 64 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_2[_lp*2 + 0], dq_1_2[_lp*2+1 + 0]));
                    dq_bf16_1_r_0[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_8 = 8 * (m4 / 2) + r8;
                int qchunk_1_8 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_8 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_8 * 128) + (unsigned int)((qchunk_1_8 ^ r8) * 16);
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(qaddr_1_8);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[3]))
                    : "memory");
                int qhead_2_9 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_9 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_9 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_9 * 128) + (unsigned int)((qchunk_1_9 ^ r8) * 16);
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(qaddr_1_9);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[7]))
                    : "memory");
                int qhead_2_10 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_10 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_10 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_10 * 128) + (unsigned int)((qchunk_1_10 ^ r8) * 16);
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(qaddr_1_10);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[11]))
                    : "memory");
                int qhead_2_11 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_11 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_11 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_11 * 128) + (unsigned int)((qchunk_1_11 ^ r8) * 16);
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(qaddr_1_11);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[15]))
                    : "memory");
                float dq_1_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_3[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 64 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_3[_lp*2 + 0], dq_1_3[_lp*2+1 + 0]));
                    dq_bf16_1_r_1[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_12 = 8 * (m4 / 2) + r8;
                int qchunk_1_12 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_12 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_12 * 128) + (unsigned int)((qchunk_1_12 ^ r8) * 16);
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(qaddr_1_12);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[3]))
                    : "memory");
                int qhead_2_13 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_13 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_13 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_13 * 128) + (unsigned int)((qchunk_1_13 ^ r8) * 16);
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(qaddr_1_13);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[7]))
                    : "memory");
                int qhead_2_14 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_14 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_14 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_14 * 128) + (unsigned int)((qchunk_1_14 ^ r8) * 16);
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(qaddr_1_14);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[11]))
                    : "memory");
                int qhead_2_15 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_15 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_15 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_15 * 128) + (unsigned int)((qchunk_1_15 ^ r8) * 16);
                uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(qaddr_1_15);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[15]))
                    : "memory");
                float dq_1_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_4[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 128 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_4[_lp*2 + 0], dq_1_4[_lp*2+1 + 0]));
                    dq_bf16_1_r_2[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_16 = 8 * (m4 / 2) + r8;
                int qchunk_1_16 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_16 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_16 * 128) + (unsigned int)((qchunk_1_16 ^ r8) * 16);
                uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(qaddr_1_16);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[3]))
                    : "memory");
                int qhead_2_17 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_17 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_17 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_17 * 128) + (unsigned int)((qchunk_1_17 ^ r8) * 16);
                uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(qaddr_1_17);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[7]))
                    : "memory");
                int qhead_2_18 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_18 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_18 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_18 * 128) + (unsigned int)((qchunk_1_18 ^ r8) * 16);
                uint32_t _stmatrix_addr_58 = static_cast<uint32_t>(qaddr_1_18);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_58), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[11]))
                    : "memory");
                int qhead_2_19 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_19 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_19 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_19 * 128) + (unsigned int)((qchunk_1_19 ^ r8) * 16);
                uint32_t _stmatrix_addr_59 = static_cast<uint32_t>(qaddr_1_19);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_59), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[15]))
                    : "memory");
                float dq_1_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_5[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 128 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_3[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_5[_lp*2 + 0], dq_1_5[_lp*2+1 + 0]));
                    dq_bf16_1_r_3[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_20 = 8 * (m4 / 2) + r8;
                int qchunk_1_20 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_20 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_20 * 128) + (unsigned int)((qchunk_1_20 ^ r8) * 16);
                uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(qaddr_1_20);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[3]))
                    : "memory");
                int qhead_2_21 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_21 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_21 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_21 * 128) + (unsigned int)((qchunk_1_21 ^ r8) * 16);
                uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(qaddr_1_21);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[7]))
                    : "memory");
                int qhead_2_22 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_22 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_22 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_22 * 128) + (unsigned int)((qchunk_1_22 ^ r8) * 16);
                uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(qaddr_1_22);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[11]))
                    : "memory");
                int qhead_2_23 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_23 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_23 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_23 * 128) + (unsigned int)((qchunk_1_23 ^ r8) * 16);
                uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(qaddr_1_23);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[15]))
                    : "memory");
                float dq_1_6[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_6[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 192 + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_4[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_6[_lp*2 + 0], dq_1_6[_lp*2+1 + 0]));
                    dq_bf16_1_r_4[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_24 = 8 * (m4 / 2) + r8;
                int qchunk_1_24 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_24 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_24 * 128) + (unsigned int)((qchunk_1_24 ^ r8) * 16);
                uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(qaddr_1_24);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[3]))
                    : "memory");
                int qhead_2_25 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_25 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_25 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_25 * 128) + (unsigned int)((qchunk_1_25 ^ r8) * 16);
                uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(qaddr_1_25);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[7]))
                    : "memory");
                int qhead_2_26 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_26 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_26 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_26 * 128) + (unsigned int)((qchunk_1_26 ^ r8) * 16);
                uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(qaddr_1_26);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[11]))
                    : "memory");
                int qhead_2_27 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_27 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_27 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_27 * 128) + (unsigned int)((qchunk_1_27 ^ r8) * 16);
                uint32_t _stmatrix_addr_67 = static_cast<uint32_t>(qaddr_1_27);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_67), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[15]))
                    : "memory");
                float dq_1_7[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1_7[31]))
                    : "r"(tmem_tmem + lane_base + 192 + 192 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dq_bf16_1_r_5[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1_7[_lp*2 + 0], dq_1_7[_lp*2+1 + 0]));
                    dq_bf16_1_r_5[_lp] = *(uint32_t*)&_bf2;
                }
                int qhead_2_28 = 8 * (m4 / 2) + r8;
                int qchunk_1_28 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_28 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_28 * 128) + (unsigned int)((qchunk_1_28 ^ r8) * 16);
                uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(qaddr_1_28);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[3]))
                    : "memory");
                int qhead_2_29 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_29 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_29 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_29 * 128) + (unsigned int)((qchunk_1_29 ^ r8) * 16);
                uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(qaddr_1_29);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[7]))
                    : "memory");
                int qhead_2_30 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_30 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_30 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_30 * 128) + (unsigned int)((qchunk_1_30 ^ r8) * 16);
                uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(qaddr_1_30);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[11]))
                    : "memory");
                int qhead_2_31 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_31 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_31 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_31 * 128) + (unsigned int)((qchunk_1_31 ^ r8) * 16);
                uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(qaddr_1_31);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[15]))
                    : "memory");
                float dqr[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dqr[0])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[1])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[2])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[3])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[4])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[5])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[6])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[7])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[8])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[9])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[10])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[11])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[12])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[13])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[14])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[15])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[16])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[17])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[18])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[19])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[20])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[21])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[22])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[23])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[24])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[25])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[26])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[27])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[28])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[29])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[30])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[31]))
                    : "r"(tmem_tmem + lane_base + 448));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int dqr_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dqr[_lp*2 + 0], dqr[_lp*2+1 + 0]));
                    dqr_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                int rhead_1_0 = 8 * (m4 / 2) + r8;
                int rchunk_1_0 = 2 * w + m4 % 2;
                unsigned int raddr_1_0 = k_blk_addr + 65536 + (unsigned int)(rhead_1_0 * 128) + (unsigned int)((rchunk_1_0 ^ r8) * 16);
                uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(raddr_1_0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[3]))
                    : "memory");
                int rhead_1_1 = 16 + 8 * (m4 / 2) + r8;
                int rchunk_1_1 = 2 * w + m4 % 2;
                unsigned int raddr_1_1 = k_blk_addr + 65536 + (unsigned int)(rhead_1_1 * 128) + (unsigned int)((rchunk_1_1 ^ r8) * 16);
                uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(raddr_1_1);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[7]))
                    : "memory");
                int rhead_1_2 = 32 + 8 * (m4 / 2) + r8;
                int rchunk_1_2 = 2 * w + m4 % 2;
                unsigned int raddr_1_2 = k_blk_addr + 65536 + (unsigned int)(rhead_1_2 * 128) + (unsigned int)((rchunk_1_2 ^ r8) * 16);
                uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(raddr_1_2);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[11]))
                    : "memory");
                int rhead_1_3 = 48 + 8 * (m4 / 2) + r8;
                int rchunk_1_3 = 2 * w + m4 % 2;
                unsigned int raddr_1_3 = k_blk_addr + 65536 + (unsigned int)(rhead_1_3 * 128) + (unsigned int)((rchunk_1_3 ^ r8) * 16);
                uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(raddr_1_3);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int stq = 1;
                if (w == 0) {
                    if (elect_sync()) {
                        if (stq != 0) {
                            tma_store_4d((&dq_latent), 0, 0, 0, token, k_blk_addr);
                            tma_store_4d((&dq_latent), 0, 0, 1, token, k_blk_addr + 8192);
                            tma_store_4d((&dq_latent), 0, 0, 2, token, k_blk_addr + 16384);
                            tma_store_4d((&dq_latent), 0, 0, 3, token, k_blk_addr + 24576);
                            tma_store_4d((&dq_latent), 0, 0, 4, token, k_blk_addr + 32768);
                            tma_store_4d((&dq_latent), 0, 0, 5, token, k_blk_addr + 40960);
                            tma_store_4d((&dq_latent), 0, 0, 6, token, k_blk_addr + 49152);
                            tma_store_4d((&dq_latent), 0, 0, 7, token, k_blk_addr + 57344);
                            tma_store_4d((&dq_rope), 0, 0, 0, token, k_blk_addr + 65536);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
                tord_1 += blocks_1;
            }
        }
    }
    // ---- Role: reduce ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // reduce_main
            int lane_0_1 = lane;
            int w2 = (warp - 8) % 4;
            int wg = (warp - 8) / 4;
            int lane_base_1 = (tid & 96) << 16;
            int lat_col = 32 * w2 + lane_0_1;
            int rope_col = 16 * w2 + lane_0_1 % 16;
            int rope_hi = lane_0_1 / 16 * 16;
            int ntok_2 = 1;
            int tord_2 = 0;
            #pragma unroll 1
            for (int tk_2 = 0; tk_2 < ntok_2; tk_2++) {
                int _parity_blocks_ready_2 = (int)((unsigned int)(int)tk_2 % 2);
                mbarrier_wait(blocks_ready_addr, _parity_blocks_ready_2);
                int blocks_2 = blocks_word[0];
                int n_row = -1;
                int n_el = 127;
                int n_er = 127;
                if (blocks_2 > 0) {
                    int _stage_idx_full_2 = (int)((unsigned int)(tord_2 + 0) % 2);
                    int _parity_idx_full_2 = (int)((unsigned int)(tord_2 + 0) / 2 % 2);
                    mbarrier_wait(idx_full_addr + (_stage_idx_full_2) * 8, _parity_idx_full_2);
                    int key_t = tile_idx[(tord_2 + 0) % 2 * 80 + wg * 32 + lane_0_1];
                    mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord_2 + 0) % 2)) * 8);
                    int row_t = key_t;
                    if (key_t >= 0) {
                        if (dkv_has_map != 0) {
                            row_t = dkv_dst_map[(long long)key_t];
                        }
                    }
                    n_row = row_t;
                    if (n_row >= 0) {
                        n_el = e_lat[(long long)n_row];
                        n_er = e_rope[(long long)n_row];
                    }
                }
                #pragma unroll 1
                for (int i_2 = 0; i_2 < blocks_2; i_2++) {
                    int row = n_row;
                    int el = n_el << 16 >> 16;
                    unsigned int thr = (unsigned int)(127 + (n_el >> 16)) << 23;
                    int crow = row;
                    float s_lat = 0.0f;
                    float s_rope = 0.0f;
                    int roff = 0;
                    int roff_r = 0;
                    if (row >= 0) {
                        s_lat = __uint_as_float((unsigned int)(127 - el) << 23);
                        s_rope = __uint_as_float((unsigned int)(127 - n_er) << 23);
                        roff = crow * dkv_stride;
                        roff_r = crow * dkr_stride + dkr_col0;
                    }
                    unsigned int vmask = __ballot_sync(0xFFFFFFFF, row >= 0);
                    unsigned int thr_w = 0;
                    unsigned int thr_l = 4294967295u;
                    if (row >= 0) {
                        thr_l = thr;
                    }
                    unsigned int _warp_redux_0;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0) : "r"(thr_l));
                    thr_w = _warp_redux_0;
                    int _find_first_set_1 = __ffs((int)vmask);
                    int first = _find_first_set_1 - 1 & 31;
                    int roff_v = __shfl_sync(0xFFFFFFFF, roff, first);
                    int roff_rv = __shfl_sync(0xFFFFFFFF, roff_r, first);
                    if (row < 0) {
                        roff = roff_v;
                        roff_r = roff_rv;
                    }
                    int x_row = -1;
                    if (i_2 + 1 < blocks_2) {
                        int _stage_idx_full_3 = (int)((unsigned int)(tord_2 + i_2 + 1) % 2);
                        int _parity_idx_full_3 = (int)((unsigned int)(tord_2 + i_2 + 1) / 2 % 2);
                        mbarrier_wait(idx_full_addr + (_stage_idx_full_3) * 8, _parity_idx_full_3);
                        int key_t_1 = tile_idx[(tord_2 + i_2 + 1) % 2 * 80 + wg * 32 + lane_0_1];
                        mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord_2 + i_2 + 1) % 2)) * 8);
                        int row_t_1 = key_t_1;
                        if (key_t_1 >= 0) {
                            if (dkv_has_map != 0) {
                                row_t_1 = dkv_dst_map[(long long)key_t_1];
                            }
                        }
                        x_row = row_t_1;
                    }
                    int _parity_dkv_a_full = (int)((unsigned int)(tord_2 + i_2) % 2);
                    mbarrier_wait(dkv_a_full_addr, _parity_dkv_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float a0[32];
                    float a1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(a0[0]), "=f"(a0[1]), "=f"(a0[2]), "=f"(a0[3]), "=f"(a0[4]), "=f"(a0[5]), "=f"(a0[6]), "=f"(a0[7]), "=f"(a0[8]), "=f"(a0[9]), "=f"(a0[10]), "=f"(a0[11]), "=f"(a0[12]), "=f"(a0[13]), "=f"(a0[14]), "=f"(a0[15]), "=f"(a0[16]), "=f"(a0[17]), "=f"(a0[18]), "=f"(a0[19]), "=f"(a0[20]), "=f"(a0[21]), "=f"(a0[22]), "=f"(a0[23]), "=f"(a0[24]), "=f"(a0[25]), "=f"(a0[26]), "=f"(a0[27]), "=f"(a0[28]), "=f"(a0[29]), "=f"(a0[30]), "=f"(a0[31])
                        : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(a1[0]), "=f"(a1[1]), "=f"(a1[2]), "=f"(a1[3]), "=f"(a1[4]), "=f"(a1[5]), "=f"(a1[6]), "=f"(a1[7]), "=f"(a1[8]), "=f"(a1[9]), "=f"(a1[10]), "=f"(a1[11]), "=f"(a1[12]), "=f"(a1[13]), "=f"(a1[14]), "=f"(a1[15]), "=f"(a1[16]), "=f"(a1[17]), "=f"(a1[18]), "=f"(a1[19]), "=f"(a1[20]), "=f"(a1[21]), "=f"(a1[22]), "=f"(a1[23]), "=f"(a1[24]), "=f"(a1[25]), "=f"(a1[26]), "=f"(a1[27]), "=f"(a1[28]), "=f"(a1[29]), "=f"(a1[30]), "=f"(a1[31])
                        : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkv_a_drained_addr);
                    float rk[16];
                    int _parity_dkr_full = (int)((unsigned int)(tord_2 + i_2) % 2);
                    mbarrier_wait(dkr_full_addr, _parity_dkr_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rk[0])), "=r"(*reinterpret_cast<uint32_t*>(&rk[1])), "=r"(*reinterpret_cast<uint32_t*>(&rk[2])), "=r"(*reinterpret_cast<uint32_t*>(&rk[3])), "=r"(*reinterpret_cast<uint32_t*>(&rk[4])), "=r"(*reinterpret_cast<uint32_t*>(&rk[5])), "=r"(*reinterpret_cast<uint32_t*>(&rk[6])), "=r"(*reinterpret_cast<uint32_t*>(&rk[7])), "=r"(*reinterpret_cast<uint32_t*>(&rk[8])), "=r"(*reinterpret_cast<uint32_t*>(&rk[9])), "=r"(*reinterpret_cast<uint32_t*>(&rk[10])), "=r"(*reinterpret_cast<uint32_t*>(&rk[11])), "=r"(*reinterpret_cast<uint32_t*>(&rk[12])), "=r"(*reinterpret_cast<uint32_t*>(&rk[13])), "=r"(*reinterpret_cast<uint32_t*>(&rk[14])), "=r"(*reinterpret_cast<uint32_t*>(&rk[15]))
                        : "r"(tmem_tmem + lane_base_1 + 0 + wg * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkr_drained_addr);
                    unsigned int mx = 0;
                    if (vmask != 0) {
                        int src_c_0 = rope_hi;
                        int rr_0 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_0);
                        float scale_c_0 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_0);
                        float yr_0 = rk[0] * scale_c_0;
                        int nr_0 = __float2int_rn(yr_0);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_0 + rope_col)))), "r"(static_cast<int>(nr_0)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_2;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_2) : "f"(yr_0));
                        rk[0] = _abs_2;
                        int src_c_1 = rope_hi + 1;
                        int rr_1 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_1);
                        float scale_c_1 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_1);
                        float yr_1 = rk[1] * scale_c_1;
                        int nr_1 = __float2int_rn(yr_1);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_1 + rope_col)))), "r"(static_cast<int>(nr_1)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_3;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_3) : "f"(yr_1));
                        rk[1] = _abs_3;
                        int src_c_2 = rope_hi + 2;
                        int rr_2 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_2);
                        float scale_c_2 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_2);
                        float yr_2 = rk[2] * scale_c_2;
                        int nr_2 = __float2int_rn(yr_2);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_2 + rope_col)))), "r"(static_cast<int>(nr_2)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_4;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(yr_2));
                        rk[2] = _abs_4;
                        int src_c_3 = rope_hi + 3;
                        int rr_3 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_3);
                        float scale_c_3 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_3);
                        float yr_3 = rk[3] * scale_c_3;
                        int nr_3 = __float2int_rn(yr_3);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_3 + rope_col)))), "r"(static_cast<int>(nr_3)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_5;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(yr_3));
                        rk[3] = _abs_5;
                        int src_c_4 = rope_hi + 4;
                        int rr_4 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_4);
                        float scale_c_4 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_4);
                        float yr_4 = rk[4] * scale_c_4;
                        int nr_4 = __float2int_rn(yr_4);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_4 + rope_col)))), "r"(static_cast<int>(nr_4)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_6;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(yr_4));
                        rk[4] = _abs_6;
                        int src_c_5 = rope_hi + 5;
                        int rr_5 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_5);
                        float scale_c_5 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_5);
                        float yr_5 = rk[5] * scale_c_5;
                        int nr_5 = __float2int_rn(yr_5);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_5 + rope_col)))), "r"(static_cast<int>(nr_5)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_7;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(yr_5));
                        rk[5] = _abs_7;
                        int src_c_6 = rope_hi + 6;
                        int rr_6 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_6);
                        float scale_c_6 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_6);
                        float yr_6 = rk[6] * scale_c_6;
                        int nr_6 = __float2int_rn(yr_6);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_6 + rope_col)))), "r"(static_cast<int>(nr_6)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_8;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_8) : "f"(yr_6));
                        rk[6] = _abs_8;
                        int src_c_7 = rope_hi + 7;
                        int rr_7 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_7);
                        float scale_c_7 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_7);
                        float yr_7 = rk[7] * scale_c_7;
                        int nr_7 = __float2int_rn(yr_7);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_7 + rope_col)))), "r"(static_cast<int>(nr_7)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_9;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(yr_7));
                        rk[7] = _abs_9;
                        int src_c_8 = rope_hi + 8;
                        int rr_8 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_8);
                        float scale_c_8 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_8);
                        float yr_8 = rk[8] * scale_c_8;
                        int nr_8 = __float2int_rn(yr_8);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_8 + rope_col)))), "r"(static_cast<int>(nr_8)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_10;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(yr_8));
                        rk[8] = _abs_10;
                        int src_c_9 = rope_hi + 9;
                        int rr_9 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_9);
                        float scale_c_9 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_9);
                        float yr_9 = rk[9] * scale_c_9;
                        int nr_9 = __float2int_rn(yr_9);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_9 + rope_col)))), "r"(static_cast<int>(nr_9)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_11;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(yr_9));
                        rk[9] = _abs_11;
                        int src_c_10 = rope_hi + 10;
                        int rr_10 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_10);
                        float scale_c_10 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_10);
                        float yr_10 = rk[10] * scale_c_10;
                        int nr_10 = __float2int_rn(yr_10);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_10 + rope_col)))), "r"(static_cast<int>(nr_10)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_12;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_12) : "f"(yr_10));
                        rk[10] = _abs_12;
                        int src_c_11 = rope_hi + 11;
                        int rr_11 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_11);
                        float scale_c_11 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_11);
                        float yr_11 = rk[11] * scale_c_11;
                        int nr_11 = __float2int_rn(yr_11);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_11 + rope_col)))), "r"(static_cast<int>(nr_11)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_13;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(yr_11));
                        rk[11] = _abs_13;
                        int src_c_12 = rope_hi + 12;
                        int rr_12 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_12);
                        float scale_c_12 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_12);
                        float yr_12 = rk[12] * scale_c_12;
                        int nr_12 = __float2int_rn(yr_12);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_12 + rope_col)))), "r"(static_cast<int>(nr_12)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_14;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(yr_12));
                        rk[12] = _abs_14;
                        int src_c_13 = rope_hi + 13;
                        int rr_13 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_13);
                        float scale_c_13 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_13);
                        float yr_13 = rk[13] * scale_c_13;
                        int nr_13 = __float2int_rn(yr_13);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_13 + rope_col)))), "r"(static_cast<int>(nr_13)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_15;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(yr_13));
                        rk[13] = _abs_15;
                        int src_c_14 = rope_hi + 14;
                        int rr_14 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_14);
                        float scale_c_14 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_14);
                        float yr_14 = rk[14] * scale_c_14;
                        int nr_14 = __float2int_rn(yr_14);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_14 + rope_col)))), "r"(static_cast<int>(nr_14)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_16;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_16) : "f"(yr_14));
                        rk[14] = _abs_16;
                        int src_c_15 = rope_hi + 15;
                        int rr_15 = __shfl_sync(0xFFFFFFFF, roff_r, src_c_15);
                        float scale_c_15 = __shfl_sync(0xFFFFFFFF, s_rope, src_c_15);
                        float yr_15 = rk[15] * scale_c_15;
                        int nr_15 = __float2int_rn(yr_15);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkr_acc) + ((long long)(rr_15 + rope_col)))), "r"(static_cast<int>(nr_15)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_17;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(yr_15));
                        rk[15] = _abs_17;
                        float lm = 0.0f;
                        int rj_0 = __shfl_sync(0xFFFFFFFF, roff, 0);
                        float scale_j_0 = __shfl_sync(0xFFFFFFFF, s_lat, 0);
                        long long idx_j_0 = (long long)(rj_0 + lat_col);
                        float y0_0 = a0[0] * scale_j_0;
                        float y1_0 = a1[0] * scale_j_0;
                        int n0_0 = __float2int_rn(y0_0);
                        int n1_0 = __float2int_rn(y1_0);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_0))), "r"(static_cast<int>(n0_0)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_0 + 128ll))), "r"(static_cast<int>(n1_0)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_18;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(y0_0));
                        float _abs_19;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(y1_0));
                        float m2_0;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_0) : "f"(_abs_18), "f"(_abs_19));
                        float _max_20;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_20) : "f"(m2_0), "f"(((lane_0_1 < 16) ? rk[0] : 0.0f)));
                        m2_0 = _max_20;
                        a0[0] = m2_0;
                        float _max_21;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_21) : "f"(lm), "f"(m2_0));
                        lm = _max_21;
                        int rj_1 = __shfl_sync(0xFFFFFFFF, roff, 1);
                        float scale_j_1 = __shfl_sync(0xFFFFFFFF, s_lat, 1);
                        long long idx_j_1 = (long long)(rj_1 + lat_col);
                        float y0_1 = a0[1] * scale_j_1;
                        float y1_1 = a1[1] * scale_j_1;
                        int n0_1 = __float2int_rn(y0_1);
                        int n1_1 = __float2int_rn(y1_1);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_1))), "r"(static_cast<int>(n0_1)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_1 + 128ll))), "r"(static_cast<int>(n1_1)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_22;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(y0_1));
                        float _abs_23;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(y1_1));
                        float m2_1;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_1) : "f"(_abs_22), "f"(_abs_23));
                        float _max_24;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_24) : "f"(m2_1), "f"(((lane_0_1 < 16) ? rk[1] : 0.0f)));
                        m2_1 = _max_24;
                        a0[1] = m2_1;
                        float _max_25;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(lm), "f"(m2_1));
                        lm = _max_25;
                        int rj_2 = __shfl_sync(0xFFFFFFFF, roff, 2);
                        float scale_j_2 = __shfl_sync(0xFFFFFFFF, s_lat, 2);
                        long long idx_j_2 = (long long)(rj_2 + lat_col);
                        float y0_2 = a0[2] * scale_j_2;
                        float y1_2 = a1[2] * scale_j_2;
                        int n0_2 = __float2int_rn(y0_2);
                        int n1_2 = __float2int_rn(y1_2);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_2))), "r"(static_cast<int>(n0_2)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_2 + 128ll))), "r"(static_cast<int>(n1_2)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_26;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(y0_2));
                        float _abs_27;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(y1_2));
                        float m2_2;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_2) : "f"(_abs_26), "f"(_abs_27));
                        float _max_28;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(m2_2), "f"(((lane_0_1 < 16) ? rk[2] : 0.0f)));
                        m2_2 = _max_28;
                        a0[2] = m2_2;
                        float _max_29;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(lm), "f"(m2_2));
                        lm = _max_29;
                        int rj_3 = __shfl_sync(0xFFFFFFFF, roff, 3);
                        float scale_j_3 = __shfl_sync(0xFFFFFFFF, s_lat, 3);
                        long long idx_j_3 = (long long)(rj_3 + lat_col);
                        float y0_3 = a0[3] * scale_j_3;
                        float y1_3 = a1[3] * scale_j_3;
                        int n0_3 = __float2int_rn(y0_3);
                        int n1_3 = __float2int_rn(y1_3);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_3))), "r"(static_cast<int>(n0_3)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_3 + 128ll))), "r"(static_cast<int>(n1_3)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_30;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(y0_3));
                        float _abs_31;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(y1_3));
                        float m2_3;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_3) : "f"(_abs_30), "f"(_abs_31));
                        float _max_32;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(m2_3), "f"(((lane_0_1 < 16) ? rk[3] : 0.0f)));
                        m2_3 = _max_32;
                        a0[3] = m2_3;
                        float _max_33;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_33) : "f"(lm), "f"(m2_3));
                        lm = _max_33;
                        int rj_4 = __shfl_sync(0xFFFFFFFF, roff, 4);
                        float scale_j_4 = __shfl_sync(0xFFFFFFFF, s_lat, 4);
                        long long idx_j_4 = (long long)(rj_4 + lat_col);
                        float y0_4 = a0[4] * scale_j_4;
                        float y1_4 = a1[4] * scale_j_4;
                        int n0_4 = __float2int_rn(y0_4);
                        int n1_4 = __float2int_rn(y1_4);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_4))), "r"(static_cast<int>(n0_4)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_4 + 128ll))), "r"(static_cast<int>(n1_4)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_34;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(y0_4));
                        float _abs_35;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(y1_4));
                        float m2_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_4) : "f"(_abs_34), "f"(_abs_35));
                        float _max_36;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(m2_4), "f"(((lane_0_1 < 16) ? rk[4] : 0.0f)));
                        m2_4 = _max_36;
                        a0[4] = m2_4;
                        float _max_37;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(lm), "f"(m2_4));
                        lm = _max_37;
                        int rj_5 = __shfl_sync(0xFFFFFFFF, roff, 5);
                        float scale_j_5 = __shfl_sync(0xFFFFFFFF, s_lat, 5);
                        long long idx_j_5 = (long long)(rj_5 + lat_col);
                        float y0_5 = a0[5] * scale_j_5;
                        float y1_5 = a1[5] * scale_j_5;
                        int n0_5 = __float2int_rn(y0_5);
                        int n1_5 = __float2int_rn(y1_5);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_5))), "r"(static_cast<int>(n0_5)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_5 + 128ll))), "r"(static_cast<int>(n1_5)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_38;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(y0_5));
                        float _abs_39;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(y1_5));
                        float m2_5;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_5) : "f"(_abs_38), "f"(_abs_39));
                        float _max_40;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(m2_5), "f"(((lane_0_1 < 16) ? rk[5] : 0.0f)));
                        m2_5 = _max_40;
                        a0[5] = m2_5;
                        float _max_41;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_41) : "f"(lm), "f"(m2_5));
                        lm = _max_41;
                        int rj_6 = __shfl_sync(0xFFFFFFFF, roff, 6);
                        float scale_j_6 = __shfl_sync(0xFFFFFFFF, s_lat, 6);
                        long long idx_j_6 = (long long)(rj_6 + lat_col);
                        float y0_6 = a0[6] * scale_j_6;
                        float y1_6 = a1[6] * scale_j_6;
                        int n0_6 = __float2int_rn(y0_6);
                        int n1_6 = __float2int_rn(y1_6);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_6))), "r"(static_cast<int>(n0_6)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_6 + 128ll))), "r"(static_cast<int>(n1_6)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_42;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(y0_6));
                        float _abs_43;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(y1_6));
                        float m2_6;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_6) : "f"(_abs_42), "f"(_abs_43));
                        float _max_44;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(m2_6), "f"(((lane_0_1 < 16) ? rk[6] : 0.0f)));
                        m2_6 = _max_44;
                        a0[6] = m2_6;
                        float _max_45;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(lm), "f"(m2_6));
                        lm = _max_45;
                        int rj_7 = __shfl_sync(0xFFFFFFFF, roff, 7);
                        float scale_j_7 = __shfl_sync(0xFFFFFFFF, s_lat, 7);
                        long long idx_j_7 = (long long)(rj_7 + lat_col);
                        float y0_7 = a0[7] * scale_j_7;
                        float y1_7 = a1[7] * scale_j_7;
                        int n0_7 = __float2int_rn(y0_7);
                        int n1_7 = __float2int_rn(y1_7);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_7))), "r"(static_cast<int>(n0_7)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_7 + 128ll))), "r"(static_cast<int>(n1_7)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_46;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(y0_7));
                        float _abs_47;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(y1_7));
                        float m2_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_7) : "f"(_abs_46), "f"(_abs_47));
                        float _max_48;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(m2_7), "f"(((lane_0_1 < 16) ? rk[7] : 0.0f)));
                        m2_7 = _max_48;
                        a0[7] = m2_7;
                        float _max_49;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(lm), "f"(m2_7));
                        lm = _max_49;
                        int rj_8 = __shfl_sync(0xFFFFFFFF, roff, 8);
                        float scale_j_8 = __shfl_sync(0xFFFFFFFF, s_lat, 8);
                        long long idx_j_8 = (long long)(rj_8 + lat_col);
                        float y0_8 = a0[8] * scale_j_8;
                        float y1_8 = a1[8] * scale_j_8;
                        int n0_8 = __float2int_rn(y0_8);
                        int n1_8 = __float2int_rn(y1_8);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_8))), "r"(static_cast<int>(n0_8)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_8 + 128ll))), "r"(static_cast<int>(n1_8)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_50;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(y0_8));
                        float _abs_51;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(y1_8));
                        float m2_8;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_8) : "f"(_abs_50), "f"(_abs_51));
                        float _max_52;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(m2_8), "f"(((lane_0_1 < 16) ? rk[8] : 0.0f)));
                        m2_8 = _max_52;
                        a0[8] = m2_8;
                        float _max_53;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_53) : "f"(lm), "f"(m2_8));
                        lm = _max_53;
                        int rj_9 = __shfl_sync(0xFFFFFFFF, roff, 9);
                        float scale_j_9 = __shfl_sync(0xFFFFFFFF, s_lat, 9);
                        long long idx_j_9 = (long long)(rj_9 + lat_col);
                        float y0_9 = a0[9] * scale_j_9;
                        float y1_9 = a1[9] * scale_j_9;
                        int n0_9 = __float2int_rn(y0_9);
                        int n1_9 = __float2int_rn(y1_9);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_9))), "r"(static_cast<int>(n0_9)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_9 + 128ll))), "r"(static_cast<int>(n1_9)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_54;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_54) : "f"(y0_9));
                        float _abs_55;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(y1_9));
                        float m2_9;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_9) : "f"(_abs_54), "f"(_abs_55));
                        float _max_56;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(m2_9), "f"(((lane_0_1 < 16) ? rk[9] : 0.0f)));
                        m2_9 = _max_56;
                        a0[9] = m2_9;
                        float _max_57;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(lm), "f"(m2_9));
                        lm = _max_57;
                        int rj_10 = __shfl_sync(0xFFFFFFFF, roff, 10);
                        float scale_j_10 = __shfl_sync(0xFFFFFFFF, s_lat, 10);
                        long long idx_j_10 = (long long)(rj_10 + lat_col);
                        float y0_10 = a0[10] * scale_j_10;
                        float y1_10 = a1[10] * scale_j_10;
                        int n0_10 = __float2int_rn(y0_10);
                        int n1_10 = __float2int_rn(y1_10);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_10))), "r"(static_cast<int>(n0_10)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_10 + 128ll))), "r"(static_cast<int>(n1_10)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_58;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(y0_10));
                        float _abs_59;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(y1_10));
                        float m2_10;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_10) : "f"(_abs_58), "f"(_abs_59));
                        float _max_60;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(m2_10), "f"(((lane_0_1 < 16) ? rk[10] : 0.0f)));
                        m2_10 = _max_60;
                        a0[10] = m2_10;
                        float _max_61;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(lm), "f"(m2_10));
                        lm = _max_61;
                        int rj_11 = __shfl_sync(0xFFFFFFFF, roff, 11);
                        float scale_j_11 = __shfl_sync(0xFFFFFFFF, s_lat, 11);
                        long long idx_j_11 = (long long)(rj_11 + lat_col);
                        float y0_11 = a0[11] * scale_j_11;
                        float y1_11 = a1[11] * scale_j_11;
                        int n0_11 = __float2int_rn(y0_11);
                        int n1_11 = __float2int_rn(y1_11);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_11))), "r"(static_cast<int>(n0_11)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_11 + 128ll))), "r"(static_cast<int>(n1_11)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_62;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_62) : "f"(y0_11));
                        float _abs_63;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(y1_11));
                        float m2_11;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_11) : "f"(_abs_62), "f"(_abs_63));
                        float _max_64;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(m2_11), "f"(((lane_0_1 < 16) ? rk[11] : 0.0f)));
                        m2_11 = _max_64;
                        a0[11] = m2_11;
                        float _max_65;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(lm), "f"(m2_11));
                        lm = _max_65;
                        int rj_12 = __shfl_sync(0xFFFFFFFF, roff, 12);
                        float scale_j_12 = __shfl_sync(0xFFFFFFFF, s_lat, 12);
                        long long idx_j_12 = (long long)(rj_12 + lat_col);
                        float y0_12 = a0[12] * scale_j_12;
                        float y1_12 = a1[12] * scale_j_12;
                        int n0_12 = __float2int_rn(y0_12);
                        int n1_12 = __float2int_rn(y1_12);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_12))), "r"(static_cast<int>(n0_12)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_12 + 128ll))), "r"(static_cast<int>(n1_12)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_66;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(y0_12));
                        float _abs_67;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(y1_12));
                        float m2_12;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_12) : "f"(_abs_66), "f"(_abs_67));
                        float _max_68;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_68) : "f"(m2_12), "f"(((lane_0_1 < 16) ? rk[12] : 0.0f)));
                        m2_12 = _max_68;
                        a0[12] = m2_12;
                        float _max_69;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_69) : "f"(lm), "f"(m2_12));
                        lm = _max_69;
                        int rj_13 = __shfl_sync(0xFFFFFFFF, roff, 13);
                        float scale_j_13 = __shfl_sync(0xFFFFFFFF, s_lat, 13);
                        long long idx_j_13 = (long long)(rj_13 + lat_col);
                        float y0_13 = a0[13] * scale_j_13;
                        float y1_13 = a1[13] * scale_j_13;
                        int n0_13 = __float2int_rn(y0_13);
                        int n1_13 = __float2int_rn(y1_13);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_13))), "r"(static_cast<int>(n0_13)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_13 + 128ll))), "r"(static_cast<int>(n1_13)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_70;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(y0_13));
                        float _abs_71;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(y1_13));
                        float m2_13;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_13) : "f"(_abs_70), "f"(_abs_71));
                        float _max_72;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_72) : "f"(m2_13), "f"(((lane_0_1 < 16) ? rk[13] : 0.0f)));
                        m2_13 = _max_72;
                        a0[13] = m2_13;
                        float _max_73;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(lm), "f"(m2_13));
                        lm = _max_73;
                        int rj_14 = __shfl_sync(0xFFFFFFFF, roff, 14);
                        float scale_j_14 = __shfl_sync(0xFFFFFFFF, s_lat, 14);
                        long long idx_j_14 = (long long)(rj_14 + lat_col);
                        float y0_14 = a0[14] * scale_j_14;
                        float y1_14 = a1[14] * scale_j_14;
                        int n0_14 = __float2int_rn(y0_14);
                        int n1_14 = __float2int_rn(y1_14);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_14))), "r"(static_cast<int>(n0_14)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_14 + 128ll))), "r"(static_cast<int>(n1_14)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_74;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(y0_14));
                        float _abs_75;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(y1_14));
                        float m2_14;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_14) : "f"(_abs_74), "f"(_abs_75));
                        float _max_76;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(m2_14), "f"(((lane_0_1 < 16) ? rk[14] : 0.0f)));
                        m2_14 = _max_76;
                        a0[14] = m2_14;
                        float _max_77;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_77) : "f"(lm), "f"(m2_14));
                        lm = _max_77;
                        int rj_15 = __shfl_sync(0xFFFFFFFF, roff, 15);
                        float scale_j_15 = __shfl_sync(0xFFFFFFFF, s_lat, 15);
                        long long idx_j_15 = (long long)(rj_15 + lat_col);
                        float y0_15 = a0[15] * scale_j_15;
                        float y1_15 = a1[15] * scale_j_15;
                        int n0_15 = __float2int_rn(y0_15);
                        int n1_15 = __float2int_rn(y1_15);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_15))), "r"(static_cast<int>(n0_15)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_15 + 128ll))), "r"(static_cast<int>(n1_15)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_78;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(y0_15));
                        float _abs_79;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(y1_15));
                        float m2_15;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_15) : "f"(_abs_78), "f"(_abs_79));
                        float _max_80;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_80) : "f"(m2_15), "f"(((lane_0_1 < 16) ? rk[15] : 0.0f)));
                        m2_15 = _max_80;
                        a0[15] = m2_15;
                        float _max_81;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_81) : "f"(lm), "f"(m2_15));
                        lm = _max_81;
                        int rj_16 = __shfl_sync(0xFFFFFFFF, roff, 16);
                        float scale_j_16 = __shfl_sync(0xFFFFFFFF, s_lat, 16);
                        long long idx_j_16 = (long long)(rj_16 + lat_col);
                        float y0_16 = a0[16] * scale_j_16;
                        float y1_16 = a1[16] * scale_j_16;
                        int n0_16 = __float2int_rn(y0_16);
                        int n1_16 = __float2int_rn(y1_16);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_16))), "r"(static_cast<int>(n0_16)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_16 + 128ll))), "r"(static_cast<int>(n1_16)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_82;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(y0_16));
                        float _abs_83;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(y1_16));
                        float m2_16;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_16) : "f"(_abs_82), "f"(_abs_83));
                        float _max_84;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_84) : "f"(m2_16), "f"(((lane_0_1 < 16) ? 0.0f : rk[0])));
                        m2_16 = _max_84;
                        a0[16] = m2_16;
                        float _max_85;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_85) : "f"(lm), "f"(m2_16));
                        lm = _max_85;
                        int rj_17 = __shfl_sync(0xFFFFFFFF, roff, 17);
                        float scale_j_17 = __shfl_sync(0xFFFFFFFF, s_lat, 17);
                        long long idx_j_17 = (long long)(rj_17 + lat_col);
                        float y0_17 = a0[17] * scale_j_17;
                        float y1_17 = a1[17] * scale_j_17;
                        int n0_17 = __float2int_rn(y0_17);
                        int n1_17 = __float2int_rn(y1_17);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_17))), "r"(static_cast<int>(n0_17)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_17 + 128ll))), "r"(static_cast<int>(n1_17)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_86;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_86) : "f"(y0_17));
                        float _abs_87;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_87) : "f"(y1_17));
                        float m2_17;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_17) : "f"(_abs_86), "f"(_abs_87));
                        float _max_88;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_88) : "f"(m2_17), "f"(((lane_0_1 < 16) ? 0.0f : rk[1])));
                        m2_17 = _max_88;
                        a0[17] = m2_17;
                        float _max_89;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_89) : "f"(lm), "f"(m2_17));
                        lm = _max_89;
                        int rj_18 = __shfl_sync(0xFFFFFFFF, roff, 18);
                        float scale_j_18 = __shfl_sync(0xFFFFFFFF, s_lat, 18);
                        long long idx_j_18 = (long long)(rj_18 + lat_col);
                        float y0_18 = a0[18] * scale_j_18;
                        float y1_18 = a1[18] * scale_j_18;
                        int n0_18 = __float2int_rn(y0_18);
                        int n1_18 = __float2int_rn(y1_18);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_18))), "r"(static_cast<int>(n0_18)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_18 + 128ll))), "r"(static_cast<int>(n1_18)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_90;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(y0_18));
                        float _abs_91;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(y1_18));
                        float m2_18;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_18) : "f"(_abs_90), "f"(_abs_91));
                        float _max_92;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_92) : "f"(m2_18), "f"(((lane_0_1 < 16) ? 0.0f : rk[2])));
                        m2_18 = _max_92;
                        a0[18] = m2_18;
                        float _max_93;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_93) : "f"(lm), "f"(m2_18));
                        lm = _max_93;
                        int rj_19 = __shfl_sync(0xFFFFFFFF, roff, 19);
                        float scale_j_19 = __shfl_sync(0xFFFFFFFF, s_lat, 19);
                        long long idx_j_19 = (long long)(rj_19 + lat_col);
                        float y0_19 = a0[19] * scale_j_19;
                        float y1_19 = a1[19] * scale_j_19;
                        int n0_19 = __float2int_rn(y0_19);
                        int n1_19 = __float2int_rn(y1_19);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_19))), "r"(static_cast<int>(n0_19)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_19 + 128ll))), "r"(static_cast<int>(n1_19)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_94;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(y0_19));
                        float _abs_95;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_95) : "f"(y1_19));
                        float m2_19;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_19) : "f"(_abs_94), "f"(_abs_95));
                        float _max_96;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_96) : "f"(m2_19), "f"(((lane_0_1 < 16) ? 0.0f : rk[3])));
                        m2_19 = _max_96;
                        a0[19] = m2_19;
                        float _max_97;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(lm), "f"(m2_19));
                        lm = _max_97;
                        int rj_20 = __shfl_sync(0xFFFFFFFF, roff, 20);
                        float scale_j_20 = __shfl_sync(0xFFFFFFFF, s_lat, 20);
                        long long idx_j_20 = (long long)(rj_20 + lat_col);
                        float y0_20 = a0[20] * scale_j_20;
                        float y1_20 = a1[20] * scale_j_20;
                        int n0_20 = __float2int_rn(y0_20);
                        int n1_20 = __float2int_rn(y1_20);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_20))), "r"(static_cast<int>(n0_20)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_20 + 128ll))), "r"(static_cast<int>(n1_20)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_98;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_98) : "f"(y0_20));
                        float _abs_99;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(y1_20));
                        float m2_20;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_20) : "f"(_abs_98), "f"(_abs_99));
                        float _max_100;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_100) : "f"(m2_20), "f"(((lane_0_1 < 16) ? 0.0f : rk[4])));
                        m2_20 = _max_100;
                        a0[20] = m2_20;
                        float _max_101;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_101) : "f"(lm), "f"(m2_20));
                        lm = _max_101;
                        int rj_21 = __shfl_sync(0xFFFFFFFF, roff, 21);
                        float scale_j_21 = __shfl_sync(0xFFFFFFFF, s_lat, 21);
                        long long idx_j_21 = (long long)(rj_21 + lat_col);
                        float y0_21 = a0[21] * scale_j_21;
                        float y1_21 = a1[21] * scale_j_21;
                        int n0_21 = __float2int_rn(y0_21);
                        int n1_21 = __float2int_rn(y1_21);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_21))), "r"(static_cast<int>(n0_21)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_21 + 128ll))), "r"(static_cast<int>(n1_21)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_102;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(y0_21));
                        float _abs_103;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_103) : "f"(y1_21));
                        float m2_21;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_21) : "f"(_abs_102), "f"(_abs_103));
                        float _max_104;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_104) : "f"(m2_21), "f"(((lane_0_1 < 16) ? 0.0f : rk[5])));
                        m2_21 = _max_104;
                        a0[21] = m2_21;
                        float _max_105;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_105) : "f"(lm), "f"(m2_21));
                        lm = _max_105;
                        int rj_22 = __shfl_sync(0xFFFFFFFF, roff, 22);
                        float scale_j_22 = __shfl_sync(0xFFFFFFFF, s_lat, 22);
                        long long idx_j_22 = (long long)(rj_22 + lat_col);
                        float y0_22 = a0[22] * scale_j_22;
                        float y1_22 = a1[22] * scale_j_22;
                        int n0_22 = __float2int_rn(y0_22);
                        int n1_22 = __float2int_rn(y1_22);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_22))), "r"(static_cast<int>(n0_22)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_22 + 128ll))), "r"(static_cast<int>(n1_22)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_106;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_106) : "f"(y0_22));
                        float _abs_107;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_107) : "f"(y1_22));
                        float m2_22;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_22) : "f"(_abs_106), "f"(_abs_107));
                        float _max_108;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_108) : "f"(m2_22), "f"(((lane_0_1 < 16) ? 0.0f : rk[6])));
                        m2_22 = _max_108;
                        a0[22] = m2_22;
                        float _max_109;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(lm), "f"(m2_22));
                        lm = _max_109;
                        int rj_23 = __shfl_sync(0xFFFFFFFF, roff, 23);
                        float scale_j_23 = __shfl_sync(0xFFFFFFFF, s_lat, 23);
                        long long idx_j_23 = (long long)(rj_23 + lat_col);
                        float y0_23 = a0[23] * scale_j_23;
                        float y1_23 = a1[23] * scale_j_23;
                        int n0_23 = __float2int_rn(y0_23);
                        int n1_23 = __float2int_rn(y1_23);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_23))), "r"(static_cast<int>(n0_23)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_23 + 128ll))), "r"(static_cast<int>(n1_23)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_110;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_110) : "f"(y0_23));
                        float _abs_111;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(y1_23));
                        float m2_23;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_23) : "f"(_abs_110), "f"(_abs_111));
                        float _max_112;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_112) : "f"(m2_23), "f"(((lane_0_1 < 16) ? 0.0f : rk[7])));
                        m2_23 = _max_112;
                        a0[23] = m2_23;
                        float _max_113;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_113) : "f"(lm), "f"(m2_23));
                        lm = _max_113;
                        int rj_24 = __shfl_sync(0xFFFFFFFF, roff, 24);
                        float scale_j_24 = __shfl_sync(0xFFFFFFFF, s_lat, 24);
                        long long idx_j_24 = (long long)(rj_24 + lat_col);
                        float y0_24 = a0[24] * scale_j_24;
                        float y1_24 = a1[24] * scale_j_24;
                        int n0_24 = __float2int_rn(y0_24);
                        int n1_24 = __float2int_rn(y1_24);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_24))), "r"(static_cast<int>(n0_24)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_24 + 128ll))), "r"(static_cast<int>(n1_24)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_114;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(y0_24));
                        float _abs_115;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_115) : "f"(y1_24));
                        float m2_24;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_24) : "f"(_abs_114), "f"(_abs_115));
                        float _max_116;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_116) : "f"(m2_24), "f"(((lane_0_1 < 16) ? 0.0f : rk[8])));
                        m2_24 = _max_116;
                        a0[24] = m2_24;
                        float _max_117;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_117) : "f"(lm), "f"(m2_24));
                        lm = _max_117;
                        int rj_25 = __shfl_sync(0xFFFFFFFF, roff, 25);
                        float scale_j_25 = __shfl_sync(0xFFFFFFFF, s_lat, 25);
                        long long idx_j_25 = (long long)(rj_25 + lat_col);
                        float y0_25 = a0[25] * scale_j_25;
                        float y1_25 = a1[25] * scale_j_25;
                        int n0_25 = __float2int_rn(y0_25);
                        int n1_25 = __float2int_rn(y1_25);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_25))), "r"(static_cast<int>(n0_25)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_25 + 128ll))), "r"(static_cast<int>(n1_25)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_118;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(y0_25));
                        float _abs_119;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(y1_25));
                        float m2_25;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_25) : "f"(_abs_118), "f"(_abs_119));
                        float _max_120;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_120) : "f"(m2_25), "f"(((lane_0_1 < 16) ? 0.0f : rk[9])));
                        m2_25 = _max_120;
                        a0[25] = m2_25;
                        float _max_121;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(lm), "f"(m2_25));
                        lm = _max_121;
                        int rj_26 = __shfl_sync(0xFFFFFFFF, roff, 26);
                        float scale_j_26 = __shfl_sync(0xFFFFFFFF, s_lat, 26);
                        long long idx_j_26 = (long long)(rj_26 + lat_col);
                        float y0_26 = a0[26] * scale_j_26;
                        float y1_26 = a1[26] * scale_j_26;
                        int n0_26 = __float2int_rn(y0_26);
                        int n1_26 = __float2int_rn(y1_26);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_26))), "r"(static_cast<int>(n0_26)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_26 + 128ll))), "r"(static_cast<int>(n1_26)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_122;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_122) : "f"(y0_26));
                        float _abs_123;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(y1_26));
                        float m2_26;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_26) : "f"(_abs_122), "f"(_abs_123));
                        float _max_124;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_124) : "f"(m2_26), "f"(((lane_0_1 < 16) ? 0.0f : rk[10])));
                        m2_26 = _max_124;
                        a0[26] = m2_26;
                        float _max_125;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_125) : "f"(lm), "f"(m2_26));
                        lm = _max_125;
                        int rj_27 = __shfl_sync(0xFFFFFFFF, roff, 27);
                        float scale_j_27 = __shfl_sync(0xFFFFFFFF, s_lat, 27);
                        long long idx_j_27 = (long long)(rj_27 + lat_col);
                        float y0_27 = a0[27] * scale_j_27;
                        float y1_27 = a1[27] * scale_j_27;
                        int n0_27 = __float2int_rn(y0_27);
                        int n1_27 = __float2int_rn(y1_27);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_27))), "r"(static_cast<int>(n0_27)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_27 + 128ll))), "r"(static_cast<int>(n1_27)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_126;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(y0_27));
                        float _abs_127;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_127) : "f"(y1_27));
                        float m2_27;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_27) : "f"(_abs_126), "f"(_abs_127));
                        float _max_128;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_128) : "f"(m2_27), "f"(((lane_0_1 < 16) ? 0.0f : rk[11])));
                        m2_27 = _max_128;
                        a0[27] = m2_27;
                        float _max_129;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_129) : "f"(lm), "f"(m2_27));
                        lm = _max_129;
                        int rj_28 = __shfl_sync(0xFFFFFFFF, roff, 28);
                        float scale_j_28 = __shfl_sync(0xFFFFFFFF, s_lat, 28);
                        long long idx_j_28 = (long long)(rj_28 + lat_col);
                        float y0_28 = a0[28] * scale_j_28;
                        float y1_28 = a1[28] * scale_j_28;
                        int n0_28 = __float2int_rn(y0_28);
                        int n1_28 = __float2int_rn(y1_28);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_28))), "r"(static_cast<int>(n0_28)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_28 + 128ll))), "r"(static_cast<int>(n1_28)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_130;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_130) : "f"(y0_28));
                        float _abs_131;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(y1_28));
                        float m2_28;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_28) : "f"(_abs_130), "f"(_abs_131));
                        float _max_132;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_132) : "f"(m2_28), "f"(((lane_0_1 < 16) ? 0.0f : rk[12])));
                        m2_28 = _max_132;
                        a0[28] = m2_28;
                        float _max_133;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_133) : "f"(lm), "f"(m2_28));
                        lm = _max_133;
                        int rj_29 = __shfl_sync(0xFFFFFFFF, roff, 29);
                        float scale_j_29 = __shfl_sync(0xFFFFFFFF, s_lat, 29);
                        long long idx_j_29 = (long long)(rj_29 + lat_col);
                        float y0_29 = a0[29] * scale_j_29;
                        float y1_29 = a1[29] * scale_j_29;
                        int n0_29 = __float2int_rn(y0_29);
                        int n1_29 = __float2int_rn(y1_29);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_29))), "r"(static_cast<int>(n0_29)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_29 + 128ll))), "r"(static_cast<int>(n1_29)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_134;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_134) : "f"(y0_29));
                        float _abs_135;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_135) : "f"(y1_29));
                        float m2_29;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_29) : "f"(_abs_134), "f"(_abs_135));
                        float _max_136;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_136) : "f"(m2_29), "f"(((lane_0_1 < 16) ? 0.0f : rk[13])));
                        m2_29 = _max_136;
                        a0[29] = m2_29;
                        float _max_137;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_137) : "f"(lm), "f"(m2_29));
                        lm = _max_137;
                        int rj_30 = __shfl_sync(0xFFFFFFFF, roff, 30);
                        float scale_j_30 = __shfl_sync(0xFFFFFFFF, s_lat, 30);
                        long long idx_j_30 = (long long)(rj_30 + lat_col);
                        float y0_30 = a0[30] * scale_j_30;
                        float y1_30 = a1[30] * scale_j_30;
                        int n0_30 = __float2int_rn(y0_30);
                        int n1_30 = __float2int_rn(y1_30);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_30))), "r"(static_cast<int>(n0_30)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_30 + 128ll))), "r"(static_cast<int>(n1_30)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_138;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_138) : "f"(y0_30));
                        float _abs_139;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_139) : "f"(y1_30));
                        float m2_30;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_30) : "f"(_abs_138), "f"(_abs_139));
                        float _max_140;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_140) : "f"(m2_30), "f"(((lane_0_1 < 16) ? 0.0f : rk[14])));
                        m2_30 = _max_140;
                        a0[30] = m2_30;
                        float _max_141;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_141) : "f"(lm), "f"(m2_30));
                        lm = _max_141;
                        int rj_31 = __shfl_sync(0xFFFFFFFF, roff, 31);
                        float scale_j_31 = __shfl_sync(0xFFFFFFFF, s_lat, 31);
                        long long idx_j_31 = (long long)(rj_31 + lat_col);
                        float y0_31 = a0[31] * scale_j_31;
                        float y1_31 = a1[31] * scale_j_31;
                        int n0_31 = __float2int_rn(y0_31);
                        int n1_31 = __float2int_rn(y1_31);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_31))), "r"(static_cast<int>(n0_31)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_31 + 128ll))), "r"(static_cast<int>(n1_31)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_142;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_142) : "f"(y0_31));
                        float _abs_143;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_143) : "f"(y1_31));
                        float m2_31;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_31) : "f"(_abs_142), "f"(_abs_143));
                        float _max_144;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_144) : "f"(m2_31), "f"(((lane_0_1 < 16) ? 0.0f : rk[15])));
                        m2_31 = _max_144;
                        a0[31] = m2_31;
                        float _max_145;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_145) : "f"(lm), "f"(m2_31));
                        lm = _max_145;
                        unsigned int _warp_redux_146;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_146) : "r"(__as_u32(lm)));
                        if (_warp_redux_146 >= thr_w) {
                            unsigned int m_j_0;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_0) : "r"(__as_u32(a0[0])));
                            if (lane_0_1 == 0) {
                                unsigned int _max_147 = ((mx) > (m_j_0) ? (mx) : (m_j_0));
                                mx = _max_147;
                            }
                            unsigned int m_j_1;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_1) : "r"(__as_u32(a0[1])));
                            if (lane_0_1 == 1) {
                                unsigned int _max_148 = ((mx) > (m_j_1) ? (mx) : (m_j_1));
                                mx = _max_148;
                            }
                            unsigned int m_j_2;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_2) : "r"(__as_u32(a0[2])));
                            if (lane_0_1 == 2) {
                                unsigned int _max_149 = ((mx) > (m_j_2) ? (mx) : (m_j_2));
                                mx = _max_149;
                            }
                            unsigned int m_j_3;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_3) : "r"(__as_u32(a0[3])));
                            if (lane_0_1 == 3) {
                                unsigned int _max_150 = ((mx) > (m_j_3) ? (mx) : (m_j_3));
                                mx = _max_150;
                            }
                            unsigned int m_j_4;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_4) : "r"(__as_u32(a0[4])));
                            if (lane_0_1 == 4) {
                                unsigned int _max_151 = ((mx) > (m_j_4) ? (mx) : (m_j_4));
                                mx = _max_151;
                            }
                            unsigned int m_j_5;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_5) : "r"(__as_u32(a0[5])));
                            if (lane_0_1 == 5) {
                                unsigned int _max_152 = ((mx) > (m_j_5) ? (mx) : (m_j_5));
                                mx = _max_152;
                            }
                            unsigned int m_j_6;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_6) : "r"(__as_u32(a0[6])));
                            if (lane_0_1 == 6) {
                                unsigned int _max_153 = ((mx) > (m_j_6) ? (mx) : (m_j_6));
                                mx = _max_153;
                            }
                            unsigned int m_j_7;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_7) : "r"(__as_u32(a0[7])));
                            if (lane_0_1 == 7) {
                                unsigned int _max_154 = ((mx) > (m_j_7) ? (mx) : (m_j_7));
                                mx = _max_154;
                            }
                            unsigned int m_j_8;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_8) : "r"(__as_u32(a0[8])));
                            if (lane_0_1 == 8) {
                                unsigned int _max_155 = ((mx) > (m_j_8) ? (mx) : (m_j_8));
                                mx = _max_155;
                            }
                            unsigned int m_j_9;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_9) : "r"(__as_u32(a0[9])));
                            if (lane_0_1 == 9) {
                                unsigned int _max_156 = ((mx) > (m_j_9) ? (mx) : (m_j_9));
                                mx = _max_156;
                            }
                            unsigned int m_j_10;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_10) : "r"(__as_u32(a0[10])));
                            if (lane_0_1 == 10) {
                                unsigned int _max_157 = ((mx) > (m_j_10) ? (mx) : (m_j_10));
                                mx = _max_157;
                            }
                            unsigned int m_j_11;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_11) : "r"(__as_u32(a0[11])));
                            if (lane_0_1 == 11) {
                                unsigned int _max_158 = ((mx) > (m_j_11) ? (mx) : (m_j_11));
                                mx = _max_158;
                            }
                            unsigned int m_j_12;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_12) : "r"(__as_u32(a0[12])));
                            if (lane_0_1 == 12) {
                                unsigned int _max_159 = ((mx) > (m_j_12) ? (mx) : (m_j_12));
                                mx = _max_159;
                            }
                            unsigned int m_j_13;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_13) : "r"(__as_u32(a0[13])));
                            if (lane_0_1 == 13) {
                                unsigned int _max_160 = ((mx) > (m_j_13) ? (mx) : (m_j_13));
                                mx = _max_160;
                            }
                            unsigned int m_j_14;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_14) : "r"(__as_u32(a0[14])));
                            if (lane_0_1 == 14) {
                                unsigned int _max_161 = ((mx) > (m_j_14) ? (mx) : (m_j_14));
                                mx = _max_161;
                            }
                            unsigned int m_j_15;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_15) : "r"(__as_u32(a0[15])));
                            if (lane_0_1 == 15) {
                                unsigned int _max_162 = ((mx) > (m_j_15) ? (mx) : (m_j_15));
                                mx = _max_162;
                            }
                            unsigned int m_j_16;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_16) : "r"(__as_u32(a0[16])));
                            if (lane_0_1 == 16) {
                                unsigned int _max_163 = ((mx) > (m_j_16) ? (mx) : (m_j_16));
                                mx = _max_163;
                            }
                            unsigned int m_j_17;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_17) : "r"(__as_u32(a0[17])));
                            if (lane_0_1 == 17) {
                                unsigned int _max_164 = ((mx) > (m_j_17) ? (mx) : (m_j_17));
                                mx = _max_164;
                            }
                            unsigned int m_j_18;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_18) : "r"(__as_u32(a0[18])));
                            if (lane_0_1 == 18) {
                                unsigned int _max_165 = ((mx) > (m_j_18) ? (mx) : (m_j_18));
                                mx = _max_165;
                            }
                            unsigned int m_j_19;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_19) : "r"(__as_u32(a0[19])));
                            if (lane_0_1 == 19) {
                                unsigned int _max_166 = ((mx) > (m_j_19) ? (mx) : (m_j_19));
                                mx = _max_166;
                            }
                            unsigned int m_j_20;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_20) : "r"(__as_u32(a0[20])));
                            if (lane_0_1 == 20) {
                                unsigned int _max_167 = ((mx) > (m_j_20) ? (mx) : (m_j_20));
                                mx = _max_167;
                            }
                            unsigned int m_j_21;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_21) : "r"(__as_u32(a0[21])));
                            if (lane_0_1 == 21) {
                                unsigned int _max_168 = ((mx) > (m_j_21) ? (mx) : (m_j_21));
                                mx = _max_168;
                            }
                            unsigned int m_j_22;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_22) : "r"(__as_u32(a0[22])));
                            if (lane_0_1 == 22) {
                                unsigned int _max_169 = ((mx) > (m_j_22) ? (mx) : (m_j_22));
                                mx = _max_169;
                            }
                            unsigned int m_j_23;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_23) : "r"(__as_u32(a0[23])));
                            if (lane_0_1 == 23) {
                                unsigned int _max_170 = ((mx) > (m_j_23) ? (mx) : (m_j_23));
                                mx = _max_170;
                            }
                            unsigned int m_j_24;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_24) : "r"(__as_u32(a0[24])));
                            if (lane_0_1 == 24) {
                                unsigned int _max_171 = ((mx) > (m_j_24) ? (mx) : (m_j_24));
                                mx = _max_171;
                            }
                            unsigned int m_j_25;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_25) : "r"(__as_u32(a0[25])));
                            if (lane_0_1 == 25) {
                                unsigned int _max_172 = ((mx) > (m_j_25) ? (mx) : (m_j_25));
                                mx = _max_172;
                            }
                            unsigned int m_j_26;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_26) : "r"(__as_u32(a0[26])));
                            if (lane_0_1 == 26) {
                                unsigned int _max_173 = ((mx) > (m_j_26) ? (mx) : (m_j_26));
                                mx = _max_173;
                            }
                            unsigned int m_j_27;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_27) : "r"(__as_u32(a0[27])));
                            if (lane_0_1 == 27) {
                                unsigned int _max_174 = ((mx) > (m_j_27) ? (mx) : (m_j_27));
                                mx = _max_174;
                            }
                            unsigned int m_j_28;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_28) : "r"(__as_u32(a0[28])));
                            if (lane_0_1 == 28) {
                                unsigned int _max_175 = ((mx) > (m_j_28) ? (mx) : (m_j_28));
                                mx = _max_175;
                            }
                            unsigned int m_j_29;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_29) : "r"(__as_u32(a0[29])));
                            if (lane_0_1 == 29) {
                                unsigned int _max_176 = ((mx) > (m_j_29) ? (mx) : (m_j_29));
                                mx = _max_176;
                            }
                            unsigned int m_j_30;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_30) : "r"(__as_u32(a0[30])));
                            if (lane_0_1 == 30) {
                                unsigned int _max_177 = ((mx) > (m_j_30) ? (mx) : (m_j_30));
                                mx = _max_177;
                            }
                            unsigned int m_j_31;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_31) : "r"(__as_u32(a0[31])));
                            if (lane_0_1 == 31) {
                                unsigned int _max_178 = ((mx) > (m_j_31) ? (mx) : (m_j_31));
                                mx = _max_178;
                            }
                        }
                        mx = mx;
                    }
                    n_row = x_row;
                    if (x_row >= 0) {
                        n_el = e_lat[(long long)x_row];
                        n_er = e_rope[(long long)x_row];
                    }
                    int _parity_dkv_b_full = (int)((unsigned int)(tord_2 + i_2) % 2);
                    mbarrier_wait(dkv_b_full_addr, _parity_dkv_b_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(a0[0]), "=f"(a0[1]), "=f"(a0[2]), "=f"(a0[3]), "=f"(a0[4]), "=f"(a0[5]), "=f"(a0[6]), "=f"(a0[7]), "=f"(a0[8]), "=f"(a0[9]), "=f"(a0[10]), "=f"(a0[11]), "=f"(a0[12]), "=f"(a0[13]), "=f"(a0[14]), "=f"(a0[15]), "=f"(a0[16]), "=f"(a0[17]), "=f"(a0[18]), "=f"(a0[19]), "=f"(a0[20]), "=f"(a0[21]), "=f"(a0[22]), "=f"(a0[23]), "=f"(a0[24]), "=f"(a0[25]), "=f"(a0[26]), "=f"(a0[27]), "=f"(a0[28]), "=f"(a0[29]), "=f"(a0[30]), "=f"(a0[31])
                        : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(a1[0]), "=f"(a1[1]), "=f"(a1[2]), "=f"(a1[3]), "=f"(a1[4]), "=f"(a1[5]), "=f"(a1[6]), "=f"(a1[7]), "=f"(a1[8]), "=f"(a1[9]), "=f"(a1[10]), "=f"(a1[11]), "=f"(a1[12]), "=f"(a1[13]), "=f"(a1[14]), "=f"(a1[15]), "=f"(a1[16]), "=f"(a1[17]), "=f"(a1[18]), "=f"(a1[19]), "=f"(a1[20]), "=f"(a1[21]), "=f"(a1[22]), "=f"(a1[23]), "=f"(a1[24]), "=f"(a1[25]), "=f"(a1[26]), "=f"(a1[27]), "=f"(a1[28]), "=f"(a1[29]), "=f"(a1[30]), "=f"(a1[31])
                        : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkv_b_drained_addr);
                    if (vmask != 0) {
                        float lm_1 = 0.0f;
                        int rj_32 = __shfl_sync(0xFFFFFFFF, roff, 0);
                        float scale_j_32 = __shfl_sync(0xFFFFFFFF, s_lat, 0);
                        long long idx_j_32 = (long long)(rj_32 + lat_col);
                        float y0_32 = a0[0] * scale_j_32;
                        float y1_32 = a1[0] * scale_j_32;
                        int n0_32 = __float2int_rn(y0_32);
                        int n1_32 = __float2int_rn(y1_32);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_32 + 256ll))), "r"(static_cast<int>(n0_32)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_32 + 384ll))), "r"(static_cast<int>(n1_32)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_179;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(y0_32));
                        float _abs_180;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_180) : "f"(y1_32));
                        float m2_32;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_32) : "f"(_abs_179), "f"(_abs_180));
                        a0[0] = m2_32;
                        float _max_181;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_181) : "f"(lm_1), "f"(m2_32));
                        lm_1 = _max_181;
                        int rj_33 = __shfl_sync(0xFFFFFFFF, roff, 1);
                        float scale_j_33 = __shfl_sync(0xFFFFFFFF, s_lat, 1);
                        long long idx_j_33 = (long long)(rj_33 + lat_col);
                        float y0_33 = a0[1] * scale_j_33;
                        float y1_33 = a1[1] * scale_j_33;
                        int n0_33 = __float2int_rn(y0_33);
                        int n1_33 = __float2int_rn(y1_33);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_33 + 256ll))), "r"(static_cast<int>(n0_33)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_33 + 384ll))), "r"(static_cast<int>(n1_33)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_182;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_182) : "f"(y0_33));
                        float _abs_183;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_183) : "f"(y1_33));
                        float m2_33;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_33) : "f"(_abs_182), "f"(_abs_183));
                        a0[1] = m2_33;
                        float _max_184;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_184) : "f"(lm_1), "f"(m2_33));
                        lm_1 = _max_184;
                        int rj_34 = __shfl_sync(0xFFFFFFFF, roff, 2);
                        float scale_j_34 = __shfl_sync(0xFFFFFFFF, s_lat, 2);
                        long long idx_j_34 = (long long)(rj_34 + lat_col);
                        float y0_34 = a0[2] * scale_j_34;
                        float y1_34 = a1[2] * scale_j_34;
                        int n0_34 = __float2int_rn(y0_34);
                        int n1_34 = __float2int_rn(y1_34);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_34 + 256ll))), "r"(static_cast<int>(n0_34)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_34 + 384ll))), "r"(static_cast<int>(n1_34)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_185;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(y0_34));
                        float _abs_186;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_186) : "f"(y1_34));
                        float m2_34;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_34) : "f"(_abs_185), "f"(_abs_186));
                        a0[2] = m2_34;
                        float _max_187;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_187) : "f"(lm_1), "f"(m2_34));
                        lm_1 = _max_187;
                        int rj_35 = __shfl_sync(0xFFFFFFFF, roff, 3);
                        float scale_j_35 = __shfl_sync(0xFFFFFFFF, s_lat, 3);
                        long long idx_j_35 = (long long)(rj_35 + lat_col);
                        float y0_35 = a0[3] * scale_j_35;
                        float y1_35 = a1[3] * scale_j_35;
                        int n0_35 = __float2int_rn(y0_35);
                        int n1_35 = __float2int_rn(y1_35);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_35 + 256ll))), "r"(static_cast<int>(n0_35)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_35 + 384ll))), "r"(static_cast<int>(n1_35)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_188;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_188) : "f"(y0_35));
                        float _abs_189;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_189) : "f"(y1_35));
                        float m2_35;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_35) : "f"(_abs_188), "f"(_abs_189));
                        a0[3] = m2_35;
                        float _max_190;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_190) : "f"(lm_1), "f"(m2_35));
                        lm_1 = _max_190;
                        int rj_36 = __shfl_sync(0xFFFFFFFF, roff, 4);
                        float scale_j_36 = __shfl_sync(0xFFFFFFFF, s_lat, 4);
                        long long idx_j_36 = (long long)(rj_36 + lat_col);
                        float y0_36 = a0[4] * scale_j_36;
                        float y1_36 = a1[4] * scale_j_36;
                        int n0_36 = __float2int_rn(y0_36);
                        int n1_36 = __float2int_rn(y1_36);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_36 + 256ll))), "r"(static_cast<int>(n0_36)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_36 + 384ll))), "r"(static_cast<int>(n1_36)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_191;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(y0_36));
                        float _abs_192;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(y1_36));
                        float m2_36;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_36) : "f"(_abs_191), "f"(_abs_192));
                        a0[4] = m2_36;
                        float _max_193;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_193) : "f"(lm_1), "f"(m2_36));
                        lm_1 = _max_193;
                        int rj_37 = __shfl_sync(0xFFFFFFFF, roff, 5);
                        float scale_j_37 = __shfl_sync(0xFFFFFFFF, s_lat, 5);
                        long long idx_j_37 = (long long)(rj_37 + lat_col);
                        float y0_37 = a0[5] * scale_j_37;
                        float y1_37 = a1[5] * scale_j_37;
                        int n0_37 = __float2int_rn(y0_37);
                        int n1_37 = __float2int_rn(y1_37);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_37 + 256ll))), "r"(static_cast<int>(n0_37)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_37 + 384ll))), "r"(static_cast<int>(n1_37)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_194;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_194) : "f"(y0_37));
                        float _abs_195;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_195) : "f"(y1_37));
                        float m2_37;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_37) : "f"(_abs_194), "f"(_abs_195));
                        a0[5] = m2_37;
                        float _max_196;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_196) : "f"(lm_1), "f"(m2_37));
                        lm_1 = _max_196;
                        int rj_38 = __shfl_sync(0xFFFFFFFF, roff, 6);
                        float scale_j_38 = __shfl_sync(0xFFFFFFFF, s_lat, 6);
                        long long idx_j_38 = (long long)(rj_38 + lat_col);
                        float y0_38 = a0[6] * scale_j_38;
                        float y1_38 = a1[6] * scale_j_38;
                        int n0_38 = __float2int_rn(y0_38);
                        int n1_38 = __float2int_rn(y1_38);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_38 + 256ll))), "r"(static_cast<int>(n0_38)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_38 + 384ll))), "r"(static_cast<int>(n1_38)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_197;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_197) : "f"(y0_38));
                        float _abs_198;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_198) : "f"(y1_38));
                        float m2_38;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_38) : "f"(_abs_197), "f"(_abs_198));
                        a0[6] = m2_38;
                        float _max_199;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_199) : "f"(lm_1), "f"(m2_38));
                        lm_1 = _max_199;
                        int rj_39 = __shfl_sync(0xFFFFFFFF, roff, 7);
                        float scale_j_39 = __shfl_sync(0xFFFFFFFF, s_lat, 7);
                        long long idx_j_39 = (long long)(rj_39 + lat_col);
                        float y0_39 = a0[7] * scale_j_39;
                        float y1_39 = a1[7] * scale_j_39;
                        int n0_39 = __float2int_rn(y0_39);
                        int n1_39 = __float2int_rn(y1_39);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_39 + 256ll))), "r"(static_cast<int>(n0_39)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_39 + 384ll))), "r"(static_cast<int>(n1_39)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_200;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_200) : "f"(y0_39));
                        float _abs_201;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(y1_39));
                        float m2_39;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_39) : "f"(_abs_200), "f"(_abs_201));
                        a0[7] = m2_39;
                        float _max_202;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_202) : "f"(lm_1), "f"(m2_39));
                        lm_1 = _max_202;
                        int rj_40 = __shfl_sync(0xFFFFFFFF, roff, 8);
                        float scale_j_40 = __shfl_sync(0xFFFFFFFF, s_lat, 8);
                        long long idx_j_40 = (long long)(rj_40 + lat_col);
                        float y0_40 = a0[8] * scale_j_40;
                        float y1_40 = a1[8] * scale_j_40;
                        int n0_40 = __float2int_rn(y0_40);
                        int n1_40 = __float2int_rn(y1_40);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_40 + 256ll))), "r"(static_cast<int>(n0_40)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_40 + 384ll))), "r"(static_cast<int>(n1_40)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_203;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(y0_40));
                        float _abs_204;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_204) : "f"(y1_40));
                        float m2_40;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_40) : "f"(_abs_203), "f"(_abs_204));
                        a0[8] = m2_40;
                        float _max_205;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_205) : "f"(lm_1), "f"(m2_40));
                        lm_1 = _max_205;
                        int rj_41 = __shfl_sync(0xFFFFFFFF, roff, 9);
                        float scale_j_41 = __shfl_sync(0xFFFFFFFF, s_lat, 9);
                        long long idx_j_41 = (long long)(rj_41 + lat_col);
                        float y0_41 = a0[9] * scale_j_41;
                        float y1_41 = a1[9] * scale_j_41;
                        int n0_41 = __float2int_rn(y0_41);
                        int n1_41 = __float2int_rn(y1_41);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_41 + 256ll))), "r"(static_cast<int>(n0_41)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_41 + 384ll))), "r"(static_cast<int>(n1_41)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_206;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_206) : "f"(y0_41));
                        float _abs_207;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_207) : "f"(y1_41));
                        float m2_41;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_41) : "f"(_abs_206), "f"(_abs_207));
                        a0[9] = m2_41;
                        float _max_208;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_208) : "f"(lm_1), "f"(m2_41));
                        lm_1 = _max_208;
                        int rj_42 = __shfl_sync(0xFFFFFFFF, roff, 10);
                        float scale_j_42 = __shfl_sync(0xFFFFFFFF, s_lat, 10);
                        long long idx_j_42 = (long long)(rj_42 + lat_col);
                        float y0_42 = a0[10] * scale_j_42;
                        float y1_42 = a1[10] * scale_j_42;
                        int n0_42 = __float2int_rn(y0_42);
                        int n1_42 = __float2int_rn(y1_42);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_42 + 256ll))), "r"(static_cast<int>(n0_42)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_42 + 384ll))), "r"(static_cast<int>(n1_42)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_209;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(y0_42));
                        float _abs_210;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_210) : "f"(y1_42));
                        float m2_42;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_42) : "f"(_abs_209), "f"(_abs_210));
                        a0[10] = m2_42;
                        float _max_211;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_211) : "f"(lm_1), "f"(m2_42));
                        lm_1 = _max_211;
                        int rj_43 = __shfl_sync(0xFFFFFFFF, roff, 11);
                        float scale_j_43 = __shfl_sync(0xFFFFFFFF, s_lat, 11);
                        long long idx_j_43 = (long long)(rj_43 + lat_col);
                        float y0_43 = a0[11] * scale_j_43;
                        float y1_43 = a1[11] * scale_j_43;
                        int n0_43 = __float2int_rn(y0_43);
                        int n1_43 = __float2int_rn(y1_43);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_43 + 256ll))), "r"(static_cast<int>(n0_43)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_43 + 384ll))), "r"(static_cast<int>(n1_43)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_212;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_212) : "f"(y0_43));
                        float _abs_213;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_213) : "f"(y1_43));
                        float m2_43;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_43) : "f"(_abs_212), "f"(_abs_213));
                        a0[11] = m2_43;
                        float _max_214;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_214) : "f"(lm_1), "f"(m2_43));
                        lm_1 = _max_214;
                        int rj_44 = __shfl_sync(0xFFFFFFFF, roff, 12);
                        float scale_j_44 = __shfl_sync(0xFFFFFFFF, s_lat, 12);
                        long long idx_j_44 = (long long)(rj_44 + lat_col);
                        float y0_44 = a0[12] * scale_j_44;
                        float y1_44 = a1[12] * scale_j_44;
                        int n0_44 = __float2int_rn(y0_44);
                        int n1_44 = __float2int_rn(y1_44);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_44 + 256ll))), "r"(static_cast<int>(n0_44)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_44 + 384ll))), "r"(static_cast<int>(n1_44)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_215;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_215) : "f"(y0_44));
                        float _abs_216;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_216) : "f"(y1_44));
                        float m2_44;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_44) : "f"(_abs_215), "f"(_abs_216));
                        a0[12] = m2_44;
                        float _max_217;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_217) : "f"(lm_1), "f"(m2_44));
                        lm_1 = _max_217;
                        int rj_45 = __shfl_sync(0xFFFFFFFF, roff, 13);
                        float scale_j_45 = __shfl_sync(0xFFFFFFFF, s_lat, 13);
                        long long idx_j_45 = (long long)(rj_45 + lat_col);
                        float y0_45 = a0[13] * scale_j_45;
                        float y1_45 = a1[13] * scale_j_45;
                        int n0_45 = __float2int_rn(y0_45);
                        int n1_45 = __float2int_rn(y1_45);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_45 + 256ll))), "r"(static_cast<int>(n0_45)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_45 + 384ll))), "r"(static_cast<int>(n1_45)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_218;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_218) : "f"(y0_45));
                        float _abs_219;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_219) : "f"(y1_45));
                        float m2_45;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_45) : "f"(_abs_218), "f"(_abs_219));
                        a0[13] = m2_45;
                        float _max_220;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_220) : "f"(lm_1), "f"(m2_45));
                        lm_1 = _max_220;
                        int rj_46 = __shfl_sync(0xFFFFFFFF, roff, 14);
                        float scale_j_46 = __shfl_sync(0xFFFFFFFF, s_lat, 14);
                        long long idx_j_46 = (long long)(rj_46 + lat_col);
                        float y0_46 = a0[14] * scale_j_46;
                        float y1_46 = a1[14] * scale_j_46;
                        int n0_46 = __float2int_rn(y0_46);
                        int n1_46 = __float2int_rn(y1_46);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_46 + 256ll))), "r"(static_cast<int>(n0_46)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_46 + 384ll))), "r"(static_cast<int>(n1_46)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_221;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_221) : "f"(y0_46));
                        float _abs_222;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_222) : "f"(y1_46));
                        float m2_46;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_46) : "f"(_abs_221), "f"(_abs_222));
                        a0[14] = m2_46;
                        float _max_223;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_223) : "f"(lm_1), "f"(m2_46));
                        lm_1 = _max_223;
                        int rj_47 = __shfl_sync(0xFFFFFFFF, roff, 15);
                        float scale_j_47 = __shfl_sync(0xFFFFFFFF, s_lat, 15);
                        long long idx_j_47 = (long long)(rj_47 + lat_col);
                        float y0_47 = a0[15] * scale_j_47;
                        float y1_47 = a1[15] * scale_j_47;
                        int n0_47 = __float2int_rn(y0_47);
                        int n1_47 = __float2int_rn(y1_47);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_47 + 256ll))), "r"(static_cast<int>(n0_47)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_47 + 384ll))), "r"(static_cast<int>(n1_47)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_224;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_224) : "f"(y0_47));
                        float _abs_225;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(y1_47));
                        float m2_47;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_47) : "f"(_abs_224), "f"(_abs_225));
                        a0[15] = m2_47;
                        float _max_226;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_226) : "f"(lm_1), "f"(m2_47));
                        lm_1 = _max_226;
                        int rj_48 = __shfl_sync(0xFFFFFFFF, roff, 16);
                        float scale_j_48 = __shfl_sync(0xFFFFFFFF, s_lat, 16);
                        long long idx_j_48 = (long long)(rj_48 + lat_col);
                        float y0_48 = a0[16] * scale_j_48;
                        float y1_48 = a1[16] * scale_j_48;
                        int n0_48 = __float2int_rn(y0_48);
                        int n1_48 = __float2int_rn(y1_48);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_48 + 256ll))), "r"(static_cast<int>(n0_48)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_48 + 384ll))), "r"(static_cast<int>(n1_48)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_227;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_227) : "f"(y0_48));
                        float _abs_228;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(y1_48));
                        float m2_48;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_48) : "f"(_abs_227), "f"(_abs_228));
                        a0[16] = m2_48;
                        float _max_229;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_229) : "f"(lm_1), "f"(m2_48));
                        lm_1 = _max_229;
                        int rj_49 = __shfl_sync(0xFFFFFFFF, roff, 17);
                        float scale_j_49 = __shfl_sync(0xFFFFFFFF, s_lat, 17);
                        long long idx_j_49 = (long long)(rj_49 + lat_col);
                        float y0_49 = a0[17] * scale_j_49;
                        float y1_49 = a1[17] * scale_j_49;
                        int n0_49 = __float2int_rn(y0_49);
                        int n1_49 = __float2int_rn(y1_49);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_49 + 256ll))), "r"(static_cast<int>(n0_49)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_49 + 384ll))), "r"(static_cast<int>(n1_49)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_230;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_230) : "f"(y0_49));
                        float _abs_231;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_231) : "f"(y1_49));
                        float m2_49;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_49) : "f"(_abs_230), "f"(_abs_231));
                        a0[17] = m2_49;
                        float _max_232;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_232) : "f"(lm_1), "f"(m2_49));
                        lm_1 = _max_232;
                        int rj_50 = __shfl_sync(0xFFFFFFFF, roff, 18);
                        float scale_j_50 = __shfl_sync(0xFFFFFFFF, s_lat, 18);
                        long long idx_j_50 = (long long)(rj_50 + lat_col);
                        float y0_50 = a0[18] * scale_j_50;
                        float y1_50 = a1[18] * scale_j_50;
                        int n0_50 = __float2int_rn(y0_50);
                        int n1_50 = __float2int_rn(y1_50);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_50 + 256ll))), "r"(static_cast<int>(n0_50)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_50 + 384ll))), "r"(static_cast<int>(n1_50)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_233;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_233) : "f"(y0_50));
                        float _abs_234;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_234) : "f"(y1_50));
                        float m2_50;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_50) : "f"(_abs_233), "f"(_abs_234));
                        a0[18] = m2_50;
                        float _max_235;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_235) : "f"(lm_1), "f"(m2_50));
                        lm_1 = _max_235;
                        int rj_51 = __shfl_sync(0xFFFFFFFF, roff, 19);
                        float scale_j_51 = __shfl_sync(0xFFFFFFFF, s_lat, 19);
                        long long idx_j_51 = (long long)(rj_51 + lat_col);
                        float y0_51 = a0[19] * scale_j_51;
                        float y1_51 = a1[19] * scale_j_51;
                        int n0_51 = __float2int_rn(y0_51);
                        int n1_51 = __float2int_rn(y1_51);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_51 + 256ll))), "r"(static_cast<int>(n0_51)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_51 + 384ll))), "r"(static_cast<int>(n1_51)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_236;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(y0_51));
                        float _abs_237;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_237) : "f"(y1_51));
                        float m2_51;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_51) : "f"(_abs_236), "f"(_abs_237));
                        a0[19] = m2_51;
                        float _max_238;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_238) : "f"(lm_1), "f"(m2_51));
                        lm_1 = _max_238;
                        int rj_52 = __shfl_sync(0xFFFFFFFF, roff, 20);
                        float scale_j_52 = __shfl_sync(0xFFFFFFFF, s_lat, 20);
                        long long idx_j_52 = (long long)(rj_52 + lat_col);
                        float y0_52 = a0[20] * scale_j_52;
                        float y1_52 = a1[20] * scale_j_52;
                        int n0_52 = __float2int_rn(y0_52);
                        int n1_52 = __float2int_rn(y1_52);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_52 + 256ll))), "r"(static_cast<int>(n0_52)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_52 + 384ll))), "r"(static_cast<int>(n1_52)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_239;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_239) : "f"(y0_52));
                        float _abs_240;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_240) : "f"(y1_52));
                        float m2_52;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_52) : "f"(_abs_239), "f"(_abs_240));
                        a0[20] = m2_52;
                        float _max_241;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_241) : "f"(lm_1), "f"(m2_52));
                        lm_1 = _max_241;
                        int rj_53 = __shfl_sync(0xFFFFFFFF, roff, 21);
                        float scale_j_53 = __shfl_sync(0xFFFFFFFF, s_lat, 21);
                        long long idx_j_53 = (long long)(rj_53 + lat_col);
                        float y0_53 = a0[21] * scale_j_53;
                        float y1_53 = a1[21] * scale_j_53;
                        int n0_53 = __float2int_rn(y0_53);
                        int n1_53 = __float2int_rn(y1_53);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_53 + 256ll))), "r"(static_cast<int>(n0_53)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_53 + 384ll))), "r"(static_cast<int>(n1_53)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_242;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_242) : "f"(y0_53));
                        float _abs_243;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(y1_53));
                        float m2_53;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_53) : "f"(_abs_242), "f"(_abs_243));
                        a0[21] = m2_53;
                        float _max_244;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_244) : "f"(lm_1), "f"(m2_53));
                        lm_1 = _max_244;
                        int rj_54 = __shfl_sync(0xFFFFFFFF, roff, 22);
                        float scale_j_54 = __shfl_sync(0xFFFFFFFF, s_lat, 22);
                        long long idx_j_54 = (long long)(rj_54 + lat_col);
                        float y0_54 = a0[22] * scale_j_54;
                        float y1_54 = a1[22] * scale_j_54;
                        int n0_54 = __float2int_rn(y0_54);
                        int n1_54 = __float2int_rn(y1_54);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_54 + 256ll))), "r"(static_cast<int>(n0_54)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_54 + 384ll))), "r"(static_cast<int>(n1_54)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_245;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_245) : "f"(y0_54));
                        float _abs_246;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_246) : "f"(y1_54));
                        float m2_54;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_54) : "f"(_abs_245), "f"(_abs_246));
                        a0[22] = m2_54;
                        float _max_247;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_247) : "f"(lm_1), "f"(m2_54));
                        lm_1 = _max_247;
                        int rj_55 = __shfl_sync(0xFFFFFFFF, roff, 23);
                        float scale_j_55 = __shfl_sync(0xFFFFFFFF, s_lat, 23);
                        long long idx_j_55 = (long long)(rj_55 + lat_col);
                        float y0_55 = a0[23] * scale_j_55;
                        float y1_55 = a1[23] * scale_j_55;
                        int n0_55 = __float2int_rn(y0_55);
                        int n1_55 = __float2int_rn(y1_55);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_55 + 256ll))), "r"(static_cast<int>(n0_55)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_55 + 384ll))), "r"(static_cast<int>(n1_55)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_248;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_248) : "f"(y0_55));
                        float _abs_249;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_249) : "f"(y1_55));
                        float m2_55;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_55) : "f"(_abs_248), "f"(_abs_249));
                        a0[23] = m2_55;
                        float _max_250;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_250) : "f"(lm_1), "f"(m2_55));
                        lm_1 = _max_250;
                        int rj_56 = __shfl_sync(0xFFFFFFFF, roff, 24);
                        float scale_j_56 = __shfl_sync(0xFFFFFFFF, s_lat, 24);
                        long long idx_j_56 = (long long)(rj_56 + lat_col);
                        float y0_56 = a0[24] * scale_j_56;
                        float y1_56 = a1[24] * scale_j_56;
                        int n0_56 = __float2int_rn(y0_56);
                        int n1_56 = __float2int_rn(y1_56);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_56 + 256ll))), "r"(static_cast<int>(n0_56)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_56 + 384ll))), "r"(static_cast<int>(n1_56)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_251;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_251) : "f"(y0_56));
                        float _abs_252;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_252) : "f"(y1_56));
                        float m2_56;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_56) : "f"(_abs_251), "f"(_abs_252));
                        a0[24] = m2_56;
                        float _max_253;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_253) : "f"(lm_1), "f"(m2_56));
                        lm_1 = _max_253;
                        int rj_57 = __shfl_sync(0xFFFFFFFF, roff, 25);
                        float scale_j_57 = __shfl_sync(0xFFFFFFFF, s_lat, 25);
                        long long idx_j_57 = (long long)(rj_57 + lat_col);
                        float y0_57 = a0[25] * scale_j_57;
                        float y1_57 = a1[25] * scale_j_57;
                        int n0_57 = __float2int_rn(y0_57);
                        int n1_57 = __float2int_rn(y1_57);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_57 + 256ll))), "r"(static_cast<int>(n0_57)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_57 + 384ll))), "r"(static_cast<int>(n1_57)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_254;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(y0_57));
                        float _abs_255;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_255) : "f"(y1_57));
                        float m2_57;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_57) : "f"(_abs_254), "f"(_abs_255));
                        a0[25] = m2_57;
                        float _max_256;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_256) : "f"(lm_1), "f"(m2_57));
                        lm_1 = _max_256;
                        int rj_58 = __shfl_sync(0xFFFFFFFF, roff, 26);
                        float scale_j_58 = __shfl_sync(0xFFFFFFFF, s_lat, 26);
                        long long idx_j_58 = (long long)(rj_58 + lat_col);
                        float y0_58 = a0[26] * scale_j_58;
                        float y1_58 = a1[26] * scale_j_58;
                        int n0_58 = __float2int_rn(y0_58);
                        int n1_58 = __float2int_rn(y1_58);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_58 + 256ll))), "r"(static_cast<int>(n0_58)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_58 + 384ll))), "r"(static_cast<int>(n1_58)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_257;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_257) : "f"(y0_58));
                        float _abs_258;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_258) : "f"(y1_58));
                        float m2_58;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_58) : "f"(_abs_257), "f"(_abs_258));
                        a0[26] = m2_58;
                        float _max_259;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_259) : "f"(lm_1), "f"(m2_58));
                        lm_1 = _max_259;
                        int rj_59 = __shfl_sync(0xFFFFFFFF, roff, 27);
                        float scale_j_59 = __shfl_sync(0xFFFFFFFF, s_lat, 27);
                        long long idx_j_59 = (long long)(rj_59 + lat_col);
                        float y0_59 = a0[27] * scale_j_59;
                        float y1_59 = a1[27] * scale_j_59;
                        int n0_59 = __float2int_rn(y0_59);
                        int n1_59 = __float2int_rn(y1_59);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_59 + 256ll))), "r"(static_cast<int>(n0_59)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_59 + 384ll))), "r"(static_cast<int>(n1_59)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_260;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_260) : "f"(y0_59));
                        float _abs_261;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(y1_59));
                        float m2_59;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_59) : "f"(_abs_260), "f"(_abs_261));
                        a0[27] = m2_59;
                        float _max_262;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_262) : "f"(lm_1), "f"(m2_59));
                        lm_1 = _max_262;
                        int rj_60 = __shfl_sync(0xFFFFFFFF, roff, 28);
                        float scale_j_60 = __shfl_sync(0xFFFFFFFF, s_lat, 28);
                        long long idx_j_60 = (long long)(rj_60 + lat_col);
                        float y0_60 = a0[28] * scale_j_60;
                        float y1_60 = a1[28] * scale_j_60;
                        int n0_60 = __float2int_rn(y0_60);
                        int n1_60 = __float2int_rn(y1_60);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_60 + 256ll))), "r"(static_cast<int>(n0_60)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_60 + 384ll))), "r"(static_cast<int>(n1_60)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_263;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_263) : "f"(y0_60));
                        float _abs_264;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_264) : "f"(y1_60));
                        float m2_60;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_60) : "f"(_abs_263), "f"(_abs_264));
                        a0[28] = m2_60;
                        float _max_265;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_265) : "f"(lm_1), "f"(m2_60));
                        lm_1 = _max_265;
                        int rj_61 = __shfl_sync(0xFFFFFFFF, roff, 29);
                        float scale_j_61 = __shfl_sync(0xFFFFFFFF, s_lat, 29);
                        long long idx_j_61 = (long long)(rj_61 + lat_col);
                        float y0_61 = a0[29] * scale_j_61;
                        float y1_61 = a1[29] * scale_j_61;
                        int n0_61 = __float2int_rn(y0_61);
                        int n1_61 = __float2int_rn(y1_61);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_61 + 256ll))), "r"(static_cast<int>(n0_61)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_61 + 384ll))), "r"(static_cast<int>(n1_61)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_266;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_266) : "f"(y0_61));
                        float _abs_267;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_267) : "f"(y1_61));
                        float m2_61;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_61) : "f"(_abs_266), "f"(_abs_267));
                        a0[29] = m2_61;
                        float _max_268;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_268) : "f"(lm_1), "f"(m2_61));
                        lm_1 = _max_268;
                        int rj_62 = __shfl_sync(0xFFFFFFFF, roff, 30);
                        float scale_j_62 = __shfl_sync(0xFFFFFFFF, s_lat, 30);
                        long long idx_j_62 = (long long)(rj_62 + lat_col);
                        float y0_62 = a0[30] * scale_j_62;
                        float y1_62 = a1[30] * scale_j_62;
                        int n0_62 = __float2int_rn(y0_62);
                        int n1_62 = __float2int_rn(y1_62);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_62 + 256ll))), "r"(static_cast<int>(n0_62)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_62 + 384ll))), "r"(static_cast<int>(n1_62)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_269;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_269) : "f"(y0_62));
                        float _abs_270;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_270) : "f"(y1_62));
                        float m2_62;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_62) : "f"(_abs_269), "f"(_abs_270));
                        a0[30] = m2_62;
                        float _max_271;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_271) : "f"(lm_1), "f"(m2_62));
                        lm_1 = _max_271;
                        int rj_63 = __shfl_sync(0xFFFFFFFF, roff, 31);
                        float scale_j_63 = __shfl_sync(0xFFFFFFFF, s_lat, 31);
                        long long idx_j_63 = (long long)(rj_63 + lat_col);
                        float y0_63 = a0[31] * scale_j_63;
                        float y1_63 = a1[31] * scale_j_63;
                        int n0_63 = __float2int_rn(y0_63);
                        int n1_63 = __float2int_rn(y1_63);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_63 + 256ll))), "r"(static_cast<int>(n0_63)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_j_63 + 384ll))), "r"(static_cast<int>(n1_63)), "l"(0x14F0000000000000ULL) : "memory");
                        float _abs_272;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_272) : "f"(y0_63));
                        float _abs_273;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_273) : "f"(y1_63));
                        float m2_63;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(m2_63) : "f"(_abs_272), "f"(_abs_273));
                        a0[31] = m2_63;
                        float _max_274;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_274) : "f"(lm_1), "f"(m2_63));
                        lm_1 = _max_274;
                        unsigned int _warp_redux_275;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_275) : "r"(__as_u32(lm_1)));
                        if (_warp_redux_275 >= thr_w) {
                            unsigned int m_j_32;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_32) : "r"(__as_u32(a0[0])));
                            if (lane_0_1 == 0) {
                                unsigned int _max_276 = ((mx) > (m_j_32) ? (mx) : (m_j_32));
                                mx = _max_276;
                            }
                            unsigned int m_j_33;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_33) : "r"(__as_u32(a0[1])));
                            if (lane_0_1 == 1) {
                                unsigned int _max_277 = ((mx) > (m_j_33) ? (mx) : (m_j_33));
                                mx = _max_277;
                            }
                            unsigned int m_j_34;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_34) : "r"(__as_u32(a0[2])));
                            if (lane_0_1 == 2) {
                                unsigned int _max_278 = ((mx) > (m_j_34) ? (mx) : (m_j_34));
                                mx = _max_278;
                            }
                            unsigned int m_j_35;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_35) : "r"(__as_u32(a0[3])));
                            if (lane_0_1 == 3) {
                                unsigned int _max_279 = ((mx) > (m_j_35) ? (mx) : (m_j_35));
                                mx = _max_279;
                            }
                            unsigned int m_j_36;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_36) : "r"(__as_u32(a0[4])));
                            if (lane_0_1 == 4) {
                                unsigned int _max_280 = ((mx) > (m_j_36) ? (mx) : (m_j_36));
                                mx = _max_280;
                            }
                            unsigned int m_j_37;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_37) : "r"(__as_u32(a0[5])));
                            if (lane_0_1 == 5) {
                                unsigned int _max_281 = ((mx) > (m_j_37) ? (mx) : (m_j_37));
                                mx = _max_281;
                            }
                            unsigned int m_j_38;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_38) : "r"(__as_u32(a0[6])));
                            if (lane_0_1 == 6) {
                                unsigned int _max_282 = ((mx) > (m_j_38) ? (mx) : (m_j_38));
                                mx = _max_282;
                            }
                            unsigned int m_j_39;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_39) : "r"(__as_u32(a0[7])));
                            if (lane_0_1 == 7) {
                                unsigned int _max_283 = ((mx) > (m_j_39) ? (mx) : (m_j_39));
                                mx = _max_283;
                            }
                            unsigned int m_j_40;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_40) : "r"(__as_u32(a0[8])));
                            if (lane_0_1 == 8) {
                                unsigned int _max_284 = ((mx) > (m_j_40) ? (mx) : (m_j_40));
                                mx = _max_284;
                            }
                            unsigned int m_j_41;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_41) : "r"(__as_u32(a0[9])));
                            if (lane_0_1 == 9) {
                                unsigned int _max_285 = ((mx) > (m_j_41) ? (mx) : (m_j_41));
                                mx = _max_285;
                            }
                            unsigned int m_j_42;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_42) : "r"(__as_u32(a0[10])));
                            if (lane_0_1 == 10) {
                                unsigned int _max_286 = ((mx) > (m_j_42) ? (mx) : (m_j_42));
                                mx = _max_286;
                            }
                            unsigned int m_j_43;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_43) : "r"(__as_u32(a0[11])));
                            if (lane_0_1 == 11) {
                                unsigned int _max_287 = ((mx) > (m_j_43) ? (mx) : (m_j_43));
                                mx = _max_287;
                            }
                            unsigned int m_j_44;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_44) : "r"(__as_u32(a0[12])));
                            if (lane_0_1 == 12) {
                                unsigned int _max_288 = ((mx) > (m_j_44) ? (mx) : (m_j_44));
                                mx = _max_288;
                            }
                            unsigned int m_j_45;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_45) : "r"(__as_u32(a0[13])));
                            if (lane_0_1 == 13) {
                                unsigned int _max_289 = ((mx) > (m_j_45) ? (mx) : (m_j_45));
                                mx = _max_289;
                            }
                            unsigned int m_j_46;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_46) : "r"(__as_u32(a0[14])));
                            if (lane_0_1 == 14) {
                                unsigned int _max_290 = ((mx) > (m_j_46) ? (mx) : (m_j_46));
                                mx = _max_290;
                            }
                            unsigned int m_j_47;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_47) : "r"(__as_u32(a0[15])));
                            if (lane_0_1 == 15) {
                                unsigned int _max_291 = ((mx) > (m_j_47) ? (mx) : (m_j_47));
                                mx = _max_291;
                            }
                            unsigned int m_j_48;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_48) : "r"(__as_u32(a0[16])));
                            if (lane_0_1 == 16) {
                                unsigned int _max_292 = ((mx) > (m_j_48) ? (mx) : (m_j_48));
                                mx = _max_292;
                            }
                            unsigned int m_j_49;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_49) : "r"(__as_u32(a0[17])));
                            if (lane_0_1 == 17) {
                                unsigned int _max_293 = ((mx) > (m_j_49) ? (mx) : (m_j_49));
                                mx = _max_293;
                            }
                            unsigned int m_j_50;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_50) : "r"(__as_u32(a0[18])));
                            if (lane_0_1 == 18) {
                                unsigned int _max_294 = ((mx) > (m_j_50) ? (mx) : (m_j_50));
                                mx = _max_294;
                            }
                            unsigned int m_j_51;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_51) : "r"(__as_u32(a0[19])));
                            if (lane_0_1 == 19) {
                                unsigned int _max_295 = ((mx) > (m_j_51) ? (mx) : (m_j_51));
                                mx = _max_295;
                            }
                            unsigned int m_j_52;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_52) : "r"(__as_u32(a0[20])));
                            if (lane_0_1 == 20) {
                                unsigned int _max_296 = ((mx) > (m_j_52) ? (mx) : (m_j_52));
                                mx = _max_296;
                            }
                            unsigned int m_j_53;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_53) : "r"(__as_u32(a0[21])));
                            if (lane_0_1 == 21) {
                                unsigned int _max_297 = ((mx) > (m_j_53) ? (mx) : (m_j_53));
                                mx = _max_297;
                            }
                            unsigned int m_j_54;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_54) : "r"(__as_u32(a0[22])));
                            if (lane_0_1 == 22) {
                                unsigned int _max_298 = ((mx) > (m_j_54) ? (mx) : (m_j_54));
                                mx = _max_298;
                            }
                            unsigned int m_j_55;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_55) : "r"(__as_u32(a0[23])));
                            if (lane_0_1 == 23) {
                                unsigned int _max_299 = ((mx) > (m_j_55) ? (mx) : (m_j_55));
                                mx = _max_299;
                            }
                            unsigned int m_j_56;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_56) : "r"(__as_u32(a0[24])));
                            if (lane_0_1 == 24) {
                                unsigned int _max_300 = ((mx) > (m_j_56) ? (mx) : (m_j_56));
                                mx = _max_300;
                            }
                            unsigned int m_j_57;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_57) : "r"(__as_u32(a0[25])));
                            if (lane_0_1 == 25) {
                                unsigned int _max_301 = ((mx) > (m_j_57) ? (mx) : (m_j_57));
                                mx = _max_301;
                            }
                            unsigned int m_j_58;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_58) : "r"(__as_u32(a0[26])));
                            if (lane_0_1 == 26) {
                                unsigned int _max_302 = ((mx) > (m_j_58) ? (mx) : (m_j_58));
                                mx = _max_302;
                            }
                            unsigned int m_j_59;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_59) : "r"(__as_u32(a0[27])));
                            if (lane_0_1 == 27) {
                                unsigned int _max_303 = ((mx) > (m_j_59) ? (mx) : (m_j_59));
                                mx = _max_303;
                            }
                            unsigned int m_j_60;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_60) : "r"(__as_u32(a0[28])));
                            if (lane_0_1 == 28) {
                                unsigned int _max_304 = ((mx) > (m_j_60) ? (mx) : (m_j_60));
                                mx = _max_304;
                            }
                            unsigned int m_j_61;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_61) : "r"(__as_u32(a0[29])));
                            if (lane_0_1 == 29) {
                                unsigned int _max_305 = ((mx) > (m_j_61) ? (mx) : (m_j_61));
                                mx = _max_305;
                            }
                            unsigned int m_j_62;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_62) : "r"(__as_u32(a0[30])));
                            if (lane_0_1 == 30) {
                                unsigned int _max_306 = ((mx) > (m_j_62) ? (mx) : (m_j_62));
                                mx = _max_306;
                            }
                            unsigned int m_j_63;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_j_63) : "r"(__as_u32(a0[31])));
                            if (lane_0_1 == 31) {
                                unsigned int _max_307 = ((mx) > (m_j_63) ? (mx) : (m_j_63));
                                mx = _max_307;
                            }
                        }
                        mx = mx;
                    }
                    if (row >= 0) {
                        if (mx >= thr) {
                            unsigned long long t_wide = 2147483648ull;
                            if (mx < 1325400064) {
                                t_wide = (unsigned long long)(unsigned int)__uint_as_float(mx) + 1ull;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(cnt64) + ((long long)crow * 4ll + (long long)w2))), "l"(static_cast<unsigned long long>(t_wide)) : "memory");
                        }
                    }
                }
                tord_2 += blocks_2;
            }
        }
    }
    // ---- Warpgroup: 4 ----
    if (warp >= 16 && warp <= 19) {
#if __CUDA_ARCH__ == 1070
        asm volatile("setmaxnreg.inc.sync.aligned.u32 104;");
#else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
#endif
        // ---- Role: mma ----
        if (warp == 16) {
            { // mma_main
                int ntok_3 = 1;
                int tord_3 = 0;
                #pragma unroll 1
                for (int tk_3 = 0; tk_3 < ntok_3; tk_3++) {
                    int _parity_blocks_ready_3 = (int)((unsigned int)(int)tk_3 % 2);
                    mbarrier_wait(blocks_ready_addr, _parity_blocks_ready_3);
                    int blocks_3 = blocks_word[0];
                    int _parity_qdo_full = (int)((unsigned int)(int)tk_3 % 2);
                    mbarrier_wait(qdo_full_addr, _parity_qdo_full);
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int i_3 = 0; i_3 < blocks_3; i_3++) {
                            if (i_3 == 0) {
                                int _parity_k_full = (int)((unsigned int)tord_3 % 2);
                                mbarrier_wait(k_full_addr, _parity_k_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_0 = ((q_k_addr) >> 4) & 0x7FFF;
                                int _smem_desc_lo_1 = ((k_k_addr) >> 4) & 0x7FFF;
#else
                                int _smem_desc_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
                                int _smem_desc_lo_1 = ((k_k_addr) >> 4) & 0x3FFF;
#endif
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
                        :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_tmem), "r"(0) : "memory");
                                tcgen05_commit(s_full_addr);
                            }
                            int _parity_dp_free = (int)((unsigned int)(tord_3 + i_3) % 2) ^ 1;
                            mbarrier_wait(dp_free_addr, _parity_dp_free);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_2 = ((do_k_addr) >> 4) & 0x7FFF;
                            int _smem_desc_lo_3 = ((v_k_addr) >> 4) & 0x7FFF;
#else
                            int _smem_desc_lo_2 = ((do_k_addr) >> 4) & 0x3FFF;
                            int _smem_desc_lo_3 = ((v_k_addr) >> 4) & 0x3FFF;
#endif
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
                        :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_tmem + (1048576))), "r"(0) : "memory");
                            tcgen05_commit(dp_full_addr);
                            int _parity_p_full = (int)((unsigned int)(tord_3 + i_3) % 2);
                            mbarrier_wait(p_full_addr, _parity_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (i_3 > 0) {
                                int _parity_dkv_b_drained = (int)((unsigned int)(int)(unsigned int)(tord_3 + i_3 - 1) % 2);
                                mbarrier_wait(dkv_b_drained_addr, _parity_dkv_b_drained);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_0 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_0 = (((p_k_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                            int _mma_a_lo_0 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_0 = (((p_k_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                        :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem + (64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_4 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (1) * 1024;
#else
                            int _smem_desc_lo_4 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_4), "r"(_mma_b_lo_0), "r"((tmem_tmem + (128))), "r"(0) : "memory");
                            int _parity_ds_full = (int)((unsigned int)(tord_3 + i_3) % 2);
                            mbarrier_wait(ds_full_addr, _parity_ds_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_1 = ((((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_1 = (((ds_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                            int _mma_a_lo_1 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_1 = (((ds_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                        :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (192))), "r"(((i_3 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_5 = ((((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (1) * 1024;
#else
                            int _smem_desc_lo_5 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_5), "r"(_mma_b_lo_1), "r"((tmem_tmem + (256))), "r"(((i_3 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_6 = ((((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (2) * 1024;
#else
                            int _smem_desc_lo_6 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_6), "r"(_mma_b_lo_1), "r"((tmem_tmem + (320))), "r"(((i_3 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_7 = ((((k_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (3) * 1024;
#else
                            int _smem_desc_lo_7 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_7), "r"(_mma_b_lo_1), "r"((tmem_tmem + (384))), "r"(((i_3 == 0) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_8 = (((kr_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                            int _smem_desc_lo_8 = (((kr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                        :: "r"(_smem_desc_lo_8), "r"(_mma_b_lo_1), "r"((tmem_tmem + (448))), "r"(((i_3 == 0) ? 0 : 1)) : "memory");
                            tcgen05_commit(k_free_addr);
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_2 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_2 = (((ds_k_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                            int _mma_a_lo_2 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                            int _mma_b_lo_2 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                        :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_9 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (1) * 1024;
#else
                            int _smem_desc_lo_9 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_9), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(1) : "memory");
                            tcgen05_commit(dkv_a_full_addr);
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_10 = (((qr_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                            int _smem_desc_lo_10 = (((qr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                        :: "r"(_smem_desc_lo_10), "r"(_mma_b_lo_2), "r"(tmem_tmem), "r"(0) : "memory");
                            tcgen05_commit(dkr_full_addr);
                            int _parity_dkv_a_drained = (int)((unsigned int)(tord_3 + i_3) % 2);
                            mbarrier_wait(dkv_a_drained_addr, _parity_dkv_a_drained);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_11 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (2) * 1024;
#else
                            int _smem_desc_lo_11 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_11), "r"(_mma_b_lo_0), "r"((tmem_tmem + (64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_12 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (3) * 1024;
#else
                            int _smem_desc_lo_12 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_12), "r"(_mma_b_lo_0), "r"((tmem_tmem + (128))), "r"(0) : "memory");
                            tcgen05_commit(p_free_addr);
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_13 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (2) * 1024;
#else
                            int _smem_desc_lo_13 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_13), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_14 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (3) * 1024;
#else
                            int _smem_desc_lo_14 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
#endif
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
                        :: "r"(_smem_desc_lo_14), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(1) : "memory");
                            tcgen05_commit(dkv_b_full_addr);
                            tcgen05_commit(ds_free_addr);
                            if (blocks_3 > i_3 + 1) {
                                int _parity_dkr_drained = (int)((unsigned int)(tord_3 + i_3) % 2);
                                mbarrier_wait(dkr_drained_addr, _parity_dkr_drained);
                                int _parity_k_full_1 = (int)((unsigned int)(int)(unsigned int)(tord_3 + i_3 + 1) % 2);
                                mbarrier_wait(k_full_addr, _parity_k_full_1);
                                int _parity_s_free = (int)((unsigned int)(int)(unsigned int)(tord_3 + i_3 + 1) % 2) ^ 1;
                                mbarrier_wait(s_free_addr, _parity_s_free);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_15 = ((q_k_addr) >> 4) & 0x7FFF;
                                int _smem_desc_lo_16 = ((k_k_addr) >> 4) & 0x7FFF;
#else
                                int _smem_desc_lo_15 = ((q_k_addr) >> 4) & 0x3FFF;
                                int _smem_desc_lo_16 = ((k_k_addr) >> 4) & 0x3FFF;
#endif
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
                        :: "r"(_smem_desc_lo_15), "r"(_smem_desc_lo_16), "r"(tmem_tmem), "r"(0) : "memory");
                                tcgen05_commit(s_full_addr);
                            }
                        }
                        tcgen05_commit(dq_done_addr);
                    }
                    tord_3 += blocks_3;
                }
            }
        // ---- Role: load ----
        } else if (warp == 17) {
            { // load_main
                int ntok_4 = 1;
                #pragma unroll 1
                for (int tk_4 = 0; tk_4 < ntok_4; tk_4++) {
                    int tok_1 = blockIdx.x + tk_4 * gridDim.x;
                    int token_1 = token_base + token_step * tok_1;
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
                    stats[lane_0_2] = lse[token_1 * 64 + lane_0_2] * 1.442695f;
                    stats[32 + lane_0_2] = lse[token_1 * 64 + 32 + lane_0_2] * 1.442695f;
                    stats[64 + lane_0_2] = delta[token_1 * 64 + lane_0_2];
                    stats[96 + lane_0_2] = delta[token_1 * 64 + 32 + lane_0_2];
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(stats_full_addr);
                }
            }
        // ---- Role: metadata ----
        } else if (warp == 18) {
            { // metadata_main
                int lane_0_3 = lane;
                int ntok_5 = 1;
                int tord_4 = 0;
                #pragma unroll 1
                for (int tk_5 = 0; tk_5 < ntok_5; tk_5++) {
                    int tok_2 = blockIdx.x + tk_5 * gridDim.x;
                    int token_2 = token_base + token_step * tok_2;
                    int active = topk;
                    if (has_topk_length != 0) {
                        int _max_1 = ((topk_length[token_2]) > (0) ? (topk_length[token_2]) : (0));
                        int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
                        active = _min_0;
                    }
                    long long row_base = (long long)indices_offset + (long long)token_2 * (long long)idx_stride;
                    int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
                    int last_valid = -1;
                    for (int blk = (active + 127) / 128 - 1; blk >= 0; blk--) {
                        int pos = blk * 128 + lane_0_3 * 4;
                        int best = -1;
                        if (pos < active) {
                            if (aligned4 != 0) {
                                int _vec_load_0[4];
                                {
                                    const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos) + 0);
                                    int4 _ivld_2;
                                    _ivld_2 = *_ivptr_2;
                                    _vec_load_0[0 + 0] = _ivld_2.x;
                                    _vec_load_0[0 + 1] = _ivld_2.y;
                                    _vec_load_0[0 + 2] = _ivld_2.z;
                                    _vec_load_0[0 + 3] = _ivld_2.w;
                                }
                                if (_vec_load_0[0] >= 0 && _vec_load_0[0] < num_kv && active > pos) {
                                    int m = 1;
                                    if (m != 0) {
                                        best = pos;
                                    }
                                }
                                if (_vec_load_0[1] >= 0 && _vec_load_0[1] < num_kv && active > pos + 1) {
                                    int m_1 = 1;
                                    if (m_1 != 0) {
                                        best = pos + 1;
                                    }
                                }
                                if (_vec_load_0[2] >= 0 && _vec_load_0[2] < num_kv && active > pos + 2) {
                                    int m_2 = 1;
                                    if (m_2 != 0) {
                                        best = pos + 2;
                                    }
                                }
                                if (_vec_load_0[3] >= 0 && _vec_load_0[3] < num_kv && active > pos + 3) {
                                    int m_3 = 1;
                                    if (m_3 != 0) {
                                        best = pos + 3;
                                    }
                                }
                            } else {
                                int sv_3_0 = indices[row_base + (long long)pos];
                                if (sv_3_0 >= 0 && sv_3_0 < num_kv && active > pos) {
                                    int m_4 = 1;
                                    if (m_4 != 0) {
                                        best = pos;
                                    }
                                }
                                int sv_3_1 = indices[row_base + (long long)(pos + 1)];
                                if (sv_3_1 >= 0 && sv_3_1 < num_kv && active > pos + 1) {
                                    int m_5 = 1;
                                    if (m_5 != 0) {
                                        best = pos + 1;
                                    }
                                }
                                int sv_3_2 = indices[row_base + (long long)(pos + 2)];
                                if (sv_3_2 >= 0 && sv_3_2 < num_kv && active > pos + 2) {
                                    int m_6 = 1;
                                    if (m_6 != 0) {
                                        best = pos + 2;
                                    }
                                }
                                int sv_3_3 = indices[row_base + (long long)(pos + 3)];
                                if (sv_3_3 >= 0 && sv_3_3 < num_kv && active > pos + 3) {
                                    int m_7 = 1;
                                    if (m_7 != 0) {
                                        best = pos + 3;
                                    }
                                }
                            }
                        }
                        int _warp_redux_3;
                        asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(best));
                        best = _warp_redux_3;
                        if (best >= 0) {
                            last_valid = best;
                            break;
                        }
                    }
                    int blocks_4 = (((last_valid + 64) / 64) > (1) ? ((last_valid + 64) / 64) : (1));
                    if (lane_0_3 == 0) {
                        blocks_word[0] = blocks_4;
                        mbarrier_arrive(blocks_ready_addr);
                    }
                    if (lane_0_3 < 8) {
                        #pragma unroll 1
                        for (int i_4 = 0; i_4 < blocks_4; i_4++) {
                            int n = blocks_4 - 1 - i_4;
                            int slot_2 = (tord_4 + i_4) % 2;
                            unsigned int mask = 0;
                            int position = n * 64 + lane_0_3 * 8;
                            int values[8];
                            int whole_tile = (int)(active >= n * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0);
                            if (whole_tile != 0) {
                                int _vec_load_1[8];
                                {
                                    uint32_t _iv_4_0;
                                    uint32_t _iv_4_1;
                                    uint32_t _iv_4_2;
                                    uint32_t _iv_4_3;
                                    uint32_t _iv_4_4;
                                    uint32_t _iv_4_5;
                                    uint32_t _iv_4_6;
                                    uint32_t _iv_4_7;
                                    asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(indices + (row_base + (long long)position) + (0))) : "memory");
                                    _vec_load_1[0 + 0] = (int32_t)_iv_4_0;
                                    _vec_load_1[0 + 1] = (int32_t)_iv_4_1;
                                    _vec_load_1[0 + 2] = (int32_t)_iv_4_2;
                                    _vec_load_1[0 + 3] = (int32_t)_iv_4_3;
                                    _vec_load_1[0 + 4] = (int32_t)_iv_4_4;
                                    _vec_load_1[0 + 5] = (int32_t)_iv_4_5;
                                    _vec_load_1[0 + 6] = (int32_t)_iv_4_6;
                                    _vec_load_1[0 + 7] = (int32_t)_iv_4_7;
                                }
                                values[0] = _vec_load_1[0];
                                values[1] = _vec_load_1[1];
                                values[2] = _vec_load_1[2];
                                values[3] = _vec_load_1[3];
                                values[4] = _vec_load_1[4];
                                values[5] = _vec_load_1[5];
                                values[6] = _vec_load_1[6];
                                values[7] = _vec_load_1[7];
                            } else {
                                int clamped_1_0 = ((position) < (topk - 1) ? (position) : (topk - 1));
                                int value_2_0 = indices[row_base + (long long)clamped_1_0];
                                if (active <= position) {
                                    value_2_0 = -1;
                                }
                                values[0] = value_2_0;
                                int clamped_1_1 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                                int value_2_1 = indices[row_base + (long long)clamped_1_1];
                                if (active <= position + 1) {
                                    value_2_1 = -1;
                                }
                                values[1] = value_2_1;
                                int clamped_1_2 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                                int value_2_2 = indices[row_base + (long long)clamped_1_2];
                                if (active <= position + 2) {
                                    value_2_2 = -1;
                                }
                                values[2] = value_2_2;
                                int clamped_1_3 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                                int value_2_3 = indices[row_base + (long long)clamped_1_3];
                                if (active <= position + 3) {
                                    value_2_3 = -1;
                                }
                                values[3] = value_2_3;
                                int clamped_1_4 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                                int value_2_4 = indices[row_base + (long long)clamped_1_4];
                                if (active <= position + 4) {
                                    value_2_4 = -1;
                                }
                                values[4] = value_2_4;
                                int clamped_1_5 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                                int value_2_5 = indices[row_base + (long long)clamped_1_5];
                                if (active <= position + 5) {
                                    value_2_5 = -1;
                                }
                                values[5] = value_2_5;
                                int clamped_1_6 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                                int value_2_6 = indices[row_base + (long long)clamped_1_6];
                                if (active <= position + 6) {
                                    value_2_6 = -1;
                                }
                                values[6] = value_2_6;
                                int clamped_1_7 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                                int value_2_7 = indices[row_base + (long long)clamped_1_7];
                                if (active <= position + 7) {
                                    value_2_7 = -1;
                                }
                                values[7] = value_2_7;
                            }
                            int _stage_idx_free = (int)((unsigned int)(tord_4 + i_4) % 2);
                            int _parity_idx_free = (int)((unsigned int)(tord_4 + i_4) / 2 % 2) ^ 1;
                            mbarrier_wait(idx_free_addr + (_stage_idx_free) * 8, _parity_idx_free);
                            int ok_7_0 = (int)(values[0] >= 0 && values[0] < num_kv);
                            if (ok_7_0 != 0) {
                                mask = mask | 1;
                            } else {
                                values[0] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8] = values[0];
                            int ok_7_1 = (int)(values[1] >= 0 && values[1] < num_kv);
                            if (ok_7_1 != 0) {
                                mask = mask | 2;
                            } else {
                                values[1] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 1] = values[1];
                            int ok_7_2 = (int)(values[2] >= 0 && values[2] < num_kv);
                            if (ok_7_2 != 0) {
                                mask = mask | 4;
                            } else {
                                values[2] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 2] = values[2];
                            int ok_7_3 = (int)(values[3] >= 0 && values[3] < num_kv);
                            if (ok_7_3 != 0) {
                                mask = mask | 8;
                            } else {
                                values[3] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 3] = values[3];
                            int ok_7_4 = (int)(values[4] >= 0 && values[4] < num_kv);
                            if (ok_7_4 != 0) {
                                mask = mask | 16;
                            } else {
                                values[4] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 4] = values[4];
                            int ok_7_5 = (int)(values[5] >= 0 && values[5] < num_kv);
                            if (ok_7_5 != 0) {
                                mask = mask | 32;
                            } else {
                                values[5] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 5] = values[5];
                            int ok_7_6 = (int)(values[6] >= 0 && values[6] < num_kv);
                            if (ok_7_6 != 0) {
                                mask = mask | 64;
                            } else {
                                values[6] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 6] = values[6];
                            int ok_7_7 = (int)(values[7] >= 0 && values[7] < num_kv);
                            if (ok_7_7 != 0) {
                                mask = mask | 128;
                            } else {
                                values[7] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 7] = values[7];
                            validity8[slot_2 * 320 + 256 + lane_0_3] = mask;
                            mbarrier_arrive(idx_full_addr + ((int)((unsigned int)(tord_4 + i_4) % 2)) * 8);
                        }
                    }
                    __syncwarp();
                    int nb = blocks_word[0];
                    tord_4 += nb;
                }
            }
        // ---- Role: spare ----
        } else if (warp == 19) {
            // idle — no tasks assigned
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
