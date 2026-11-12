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
kernel_cake_dsa_h64_det_nope_train_bfbe0a61e0a693c742d7(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap dout, const __grid_constant__ CUtensorMap dq_latent, const __grid_constant__ CUtensorMap kv_latent, float* __restrict__ lse, float* __restrict__ delta, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, int* __restrict__ indices, int* __restrict__ topk_length, int* __restrict__ dkv_acc, int dkv_stride, int* __restrict__ dkv_dst_map, int dkv_has_map, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int pass_lo, int pass_hi, int dq_mode, float* __restrict__ dq_partial, int* __restrict__ key_scratch, int* __restrict__ pass_counts, int* __restrict__ e_dkv, unsigned int* __restrict__ cnt, unsigned long long* __restrict__ cnt64, int* __restrict__ redo_mark, int* __restrict__ redo_count)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1030
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#else
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#endif

    const int mbar_base = smem;
    #define qdo_full_addr (mbar_base + 0)
    #define do_full_addr (mbar_base + 8)
    #define do_free_addr (mbar_base + 16)
    #define dq_drained_addr (mbar_base + 24)
    #define stats_full_addr (mbar_base + 32)
    #define blocks_ready_addr (mbar_base + 40)
    #define blocks_free_addr (mbar_base + 56)
    #define dq_done_addr (mbar_base + 72)
    #define idx_full_addr (mbar_base + 80)
    #define idx_free_addr (mbar_base + 96)
    #define k_full_addr (mbar_base + 112)
    #define k_free_addr (mbar_base + 120)
    #define s_full_addr (mbar_base + 128)
    #define s_free_addr (mbar_base + 136)
    #define dp_full_addr (mbar_base + 144)
    #define dp_free_addr (mbar_base + 152)
    #define p_full_addr (mbar_base + 160)
    #define p_free_addr (mbar_base + 168)
    #define ds_full_addr (mbar_base + 176)
    #define ds_free_addr (mbar_base + 184)
    #define dkv_a_full_addr (mbar_base + 192)
    #define dkv_a_drained_addr (mbar_base + 200)
    #define dkv_b_full_addr (mbar_base + 208)
    #define dkv_b_drained_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_mn_addr = smem + 1024;
    __nv_bfloat16* do_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int do_k_addr = smem + 66560;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int do_mn_addr = smem + 66560;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int k_k_addr = smem + 132096;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int v_k_addr = smem + 132096;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int k_mn_addr = smem + 132096;
    __nv_bfloat16* k_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int k_blk_addr = smem + 132096;
    __nv_bfloat16* do_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int do_blk_addr = smem + 66560;
    __nv_bfloat16* p_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int p_k_addr = smem + 197632;
    __nv_bfloat16* ds_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int ds_k_addr = smem + 205824;
    __nv_bfloat16* ds_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int ds_mn_addr = smem + 205824;
    float* stats = reinterpret_cast<float*>(smem_raw + 214016);
    const int stats_addr = smem + 214016;
    int* tile_idx = reinterpret_cast<int*>(smem_raw + 214528);
    const int tile_idx_addr = smem + 214528;
    uint8_t* validity8 = reinterpret_cast<uint8_t*>(smem_raw + 214528);
    const int validity8_addr = smem + 214528;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 214528);
    const int validity32_addr = smem + 214528;
    int* blocks_word = reinterpret_cast<int*>(smem_raw + 215168);
    const int blocks_word_addr = smem + 215168;
    unsigned int* rtab = reinterpret_cast<unsigned int*>(smem_raw + 215184);
    const int rtab_addr = smem + 215184;

    // Kernel setup ops
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= redo_count[0]) return;
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dout))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_latent))) : "memory"); }

    // Mbarrier init (28 barriers at smem_raw[0..224))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // qdo_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // do_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // do_free: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // dq_drained: 1 barriers, init_count=128
            mbarrier_init(smem + 24, 128);
            // stats_full: 1 barriers, init_count=32
            mbarrier_init(smem + 32, 32);
            // blocks_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // blocks_free: 2 barriers, init_count=544
            mbarrier_init(smem + 56, 544);
            mbarrier_init(smem + 64, 544);
            // dq_done: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // idx_full: 2 barriers, init_count=8
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // idx_free: 2 barriers, init_count=388
            mbarrier_init(smem + 96, 388);
            mbarrier_init(smem + 104, 388);
            // k_full: 1 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            // k_free: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // s_free: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // dp_free: 1 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // p_free: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // ds_full: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // ds_free: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // dkv_a_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // dkv_a_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // dkv_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // dkv_b_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 4) {
        int _tmem_hold = smem + 224;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // gather_main
            int ntok = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int tord = 0;
            #pragma unroll 1
            for (int tk = 0; tk < ntok; tk++) {
                int tok = blockIdx.x + tk * gridDim.x;
                int pw = warp;
                int bslot = tk % 2;
                int _stage_blocks_ready = (int)((unsigned int)tk % 2);
                int _parity_blocks_ready = (int)((unsigned int)tk / 2 % 2);
                mbarrier_wait(blocks_ready_addr + (_stage_blocks_ready) * 8, _parity_blocks_ready);
                int blocks = blocks_word[bslot];
                mbarrier_arrive(blocks_free_addr + ((int)((unsigned int)tk % 2)) * 8);
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int j = 0; j < blocks; j++) {
                        int slot = (tord + j) % 2;
                        int _stage_idx_full = (int)((unsigned int)(tord + j) % 2);
                        int _parity_idx_full = (int)((unsigned int)(tord + j) / 2 % 2);
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
                        mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord + j) % 2)) * 8);
                        int _parity_k_free = (int)((unsigned int)(tord + j) % 2) ^ 1;
                        mbarrier_wait(k_free_addr, _parity_k_free);
                        mbarrier_arrive_expect_tx(k_full_addr, 16384);
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
            int ntok_1 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int tord_1 = 0;
            #pragma unroll 1
            for (int tk_1 = 0; tk_1 < ntok_1; tk_1++) {
                int tok_1 = blockIdx.x + tk_1 * gridDim.x;
                int token = token_base + token_step * tok_1;
                int lane_0 = lane;
                int w = warp - 4;
                int lane_base = (tid & 96) << 16;
                int bslot_1 = tk_1 % 2;
                int _stage_blocks_ready_1 = (int)((unsigned int)tk_1 % 2);
                int _parity_blocks_ready_1 = (int)((unsigned int)tk_1 / 2 % 2);
                mbarrier_wait(blocks_ready_addr + (_stage_blocks_ready_1) * 8, _parity_blocks_ready_1);
                int blocks_1 = blocks_word[bslot_1];
                mbarrier_arrive(blocks_free_addr + ((int)((unsigned int)tk_1 % 2)) * 8);
                int _parity_stats_full = (int)((unsigned int)tk_1 % 2);
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
                for (int j_1 = 0; j_1 < blocks_1; j_1++) {
                    int slot_1 = (tord_1 + j_1) % 2;
                    int _stage_idx_full_1 = (int)((unsigned int)(tord_1 + j_1) % 2);
                    int _parity_idx_full_1 = (int)((unsigned int)(tord_1 + j_1) / 2 % 2);
                    mbarrier_wait(idx_full_addr + (_stage_idx_full_1) * 8, _parity_idx_full_1);
                    unsigned int v0 = validity32[slot_1 * 80 + 64];
                    unsigned int v1 = validity32[slot_1 * 80 + 65];
                    mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord_1 + j_1) % 2)) * 8);
                    int _parity_s_full = (int)((unsigned int)(tord_1 + j_1) % 2);
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
                    int _parity_p_free = (int)((unsigned int)(tord_1 + j_1) % 2) ^ 1;
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
                    int _parity_dp_full = (int)((unsigned int)(tord_1 + j_1) % 2);
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
                    int _parity_ds_free = (int)((unsigned int)(tord_1 + j_1) % 2) ^ 1;
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
                int _parity_dq_done = (int)((unsigned int)tk_1 % 2);
                mbarrier_wait(dq_done_addr, _parity_dq_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
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
                unsigned int qaddr_1_0 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_0 * 128) + (unsigned int)((qchunk_1_0 ^ r8) * 16);
                uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(qaddr_1_0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[3]))
                    : "memory");
                int qhead_2_1 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_1 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_1 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_1 * 128) + (unsigned int)((qchunk_1_1 ^ r8) * 16);
                uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(qaddr_1_1);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[7]))
                    : "memory");
                int qhead_2_2 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_2 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_2 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_2 * 128) + (unsigned int)((qchunk_1_2 ^ r8) * 16);
                uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(qaddr_1_2);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_0[11]))
                    : "memory");
                int qhead_2_3 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_3 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_3 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_3 * 128) + (unsigned int)((qchunk_1_3 ^ r8) * 16);
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
                unsigned int qaddr_1_4 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_4 * 128) + (unsigned int)((qchunk_1_4 ^ r8) * 16);
                uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(qaddr_1_4);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[3]))
                    : "memory");
                int qhead_2_5 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_5 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_5 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_5 * 128) + (unsigned int)((qchunk_1_5 ^ r8) * 16);
                uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(qaddr_1_5);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[7]))
                    : "memory");
                int qhead_2_6 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_6 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_6 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_6 * 128) + (unsigned int)((qchunk_1_6 ^ r8) * 16);
                uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(qaddr_1_6);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_1[11]))
                    : "memory");
                int qhead_2_7 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_7 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_7 = do_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_2_7 * 128) + (unsigned int)((qchunk_1_7 ^ r8) * 16);
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
                unsigned int qaddr_1_8 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_8 * 128) + (unsigned int)((qchunk_1_8 ^ r8) * 16);
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(qaddr_1_8);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[3]))
                    : "memory");
                int qhead_2_9 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_9 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_9 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_9 * 128) + (unsigned int)((qchunk_1_9 ^ r8) * 16);
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(qaddr_1_9);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[7]))
                    : "memory");
                int qhead_2_10 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_10 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_10 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_10 * 128) + (unsigned int)((qchunk_1_10 ^ r8) * 16);
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(qaddr_1_10);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_0[11]))
                    : "memory");
                int qhead_2_11 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_11 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_11 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_11 * 128) + (unsigned int)((qchunk_1_11 ^ r8) * 16);
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
                unsigned int qaddr_1_12 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_12 * 128) + (unsigned int)((qchunk_1_12 ^ r8) * 16);
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(qaddr_1_12);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[3]))
                    : "memory");
                int qhead_2_13 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_13 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_13 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_13 * 128) + (unsigned int)((qchunk_1_13 ^ r8) * 16);
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(qaddr_1_13);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[7]))
                    : "memory");
                int qhead_2_14 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_14 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_14 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_14 * 128) + (unsigned int)((qchunk_1_14 ^ r8) * 16);
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(qaddr_1_14);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_1[11]))
                    : "memory");
                int qhead_2_15 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_15 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_15 = do_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_2_15 * 128) + (unsigned int)((qchunk_1_15 ^ r8) * 16);
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
                unsigned int qaddr_1_16 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_16 * 128) + (unsigned int)((qchunk_1_16 ^ r8) * 16);
                uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(qaddr_1_16);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[3]))
                    : "memory");
                int qhead_2_17 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_17 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_17 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_17 * 128) + (unsigned int)((qchunk_1_17 ^ r8) * 16);
                uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(qaddr_1_17);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[7]))
                    : "memory");
                int qhead_2_18 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_18 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_18 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_18 * 128) + (unsigned int)((qchunk_1_18 ^ r8) * 16);
                uint32_t _stmatrix_addr_58 = static_cast<uint32_t>(qaddr_1_18);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_58), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_2[11]))
                    : "memory");
                int qhead_2_19 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_19 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_19 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_19 * 128) + (unsigned int)((qchunk_1_19 ^ r8) * 16);
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
                unsigned int qaddr_1_20 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_20 * 128) + (unsigned int)((qchunk_1_20 ^ r8) * 16);
                uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(qaddr_1_20);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[3]))
                    : "memory");
                int qhead_2_21 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_21 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_21 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_21 * 128) + (unsigned int)((qchunk_1_21 ^ r8) * 16);
                uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(qaddr_1_21);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[7]))
                    : "memory");
                int qhead_2_22 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_22 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_22 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_22 * 128) + (unsigned int)((qchunk_1_22 ^ r8) * 16);
                uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(qaddr_1_22);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_3[11]))
                    : "memory");
                int qhead_2_23 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_23 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_23 = do_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_2_23 * 128) + (unsigned int)((qchunk_1_23 ^ r8) * 16);
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
                unsigned int qaddr_1_24 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_24 * 128) + (unsigned int)((qchunk_1_24 ^ r8) * 16);
                uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(qaddr_1_24);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[3]))
                    : "memory");
                int qhead_2_25 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_25 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_25 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_25 * 128) + (unsigned int)((qchunk_1_25 ^ r8) * 16);
                uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(qaddr_1_25);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[7]))
                    : "memory");
                int qhead_2_26 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_26 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_26 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_26 * 128) + (unsigned int)((qchunk_1_26 ^ r8) * 16);
                uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(qaddr_1_26);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_4[11]))
                    : "memory");
                int qhead_2_27 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_27 = 4 * (w % 2) + m4 % 2;
                unsigned int qaddr_1_27 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_27 * 128) + (unsigned int)((qchunk_1_27 ^ r8) * 16);
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
                unsigned int qaddr_1_28 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_28 * 128) + (unsigned int)((qchunk_1_28 ^ r8) * 16);
                uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(qaddr_1_28);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[3]))
                    : "memory");
                int qhead_2_29 = 16 + 8 * (m4 / 2) + r8;
                int qchunk_1_29 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_29 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_29 * 128) + (unsigned int)((qchunk_1_29 ^ r8) * 16);
                uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(qaddr_1_29);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[7]))
                    : "memory");
                int qhead_2_30 = 32 + 8 * (m4 / 2) + r8;
                int qchunk_1_30 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_30 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_30 * 128) + (unsigned int)((qchunk_1_30 ^ r8) * 16);
                uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(qaddr_1_30);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[11]))
                    : "memory");
                int qhead_2_31 = 48 + 8 * (m4 / 2) + r8;
                int qchunk_1_31 = 4 * (w % 2) + 2 + m4 % 2;
                unsigned int qaddr_1_31 = do_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_2_31 * 128) + (unsigned int)((qchunk_1_31 ^ r8) * 16);
                uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(qaddr_1_31);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16_1_r_5[15]))
                    : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dq_drained_addr);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (w == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                }
                if (w == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(do_free_addr);
                    }
                }
                tord_1 += blocks_1;
            }
        }
    }
    // ---- Role: reduce ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 104;");
        { // reduce_main
            int ntok_2 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
            int tord_2 = 0;
            int lane_0_1 = lane;
            int w2 = (warp - 8) % 4;
            int wg = (warp - 8) / 4;
            int lane_base_1 = (tid & 96) << 16;
            int lat_col = 32 * w2 + lane_0_1;
            unsigned int tab_w = rtab_addr + (unsigned int)((warp - 8) * 256);
            #pragma unroll 1
            for (int tk_2 = 0; tk_2 < ntok_2; tk_2++) {
                int bslot_2 = tk_2 % 2;
                int _stage_blocks_ready_2 = (int)((unsigned int)tk_2 % 2);
                int _parity_blocks_ready_2 = (int)((unsigned int)tk_2 / 2 % 2);
                mbarrier_wait(blocks_ready_addr + (_stage_blocks_ready_2) * 8, _parity_blocks_ready_2);
                int blocks_2 = blocks_word[bslot_2];
                mbarrier_arrive(blocks_free_addr + ((int)((unsigned int)tk_2 % 2)) * 8);
                int n_row = -1;
                int n_e = 127;
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
                        n_e = e_dkv[(long long)n_row];
                    }
                }
                #pragma unroll 1
                for (int j_2 = 0; j_2 < blocks_2; j_2++) {
                    int row = n_row;
                    int ev = n_e << 16 >> 16;
                    unsigned int thr = (unsigned int)(127 + (n_e >> 16)) << 23;
                    float s_dkv = 0.0f;
                    int roff = 0;
                    if (row >= 0) {
                        s_dkv = __uint_as_float((unsigned int)(127 - ev) << 23);
                        roff = row * dkv_stride;
                    }
                    unsigned int vmask = __ballot_sync(0xFFFFFFFF, row >= 0);
                    unsigned int thr_w = 0;
                    unsigned int thr_l = 4294967295u;
                    if (row >= 0) {
#if __CUDA_ARCH__ == 1000
                        thr_l = thr;
#else
                        int _min_0 = ((127 + (n_e >> 16) + ev) < (255) ? (127 + (n_e >> 16) + ev) : (255));
                        int be = ((_min_0) > (0) ? (_min_0) : (0));
                        thr_l = (unsigned int)be << 23;
#endif
                    }
#if __CUDA_ARCH__ == 1000
                    unsigned int _warp_redux_0;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_0) : "r"(thr_l));
                    thr_w = _warp_redux_0;
                    int _find_first_set_1 = __ffs((int)vmask);
                    int roff_v = __shfl_sync(0xFFFFFFFF, roff, _find_first_set_1 - 1 & 31);
#else
                    unsigned int _warp_redux_1;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_1) : "r"(thr_l));
                    thr_w = _warp_redux_1;
                    int _find_first_set_2 = __ffs((int)vmask);
                    int roff_v = __shfl_sync(0xFFFFFFFF, roff, _find_first_set_2 - 1 & 31);
#endif
                    if (row < 0) {
                        roff = roff_v;
#if __CUDA_ARCH__ == 1000
#else
                    }
                    if (vmask != 0) {
                        __syncwarp();
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(tab_w + (unsigned int)(lane_0_1 * 8)), "r"(roff), "r"(__as_u32(s_dkv)) : "memory");
                        __syncwarp();
#endif
                    }
                    int x_row = -1;
                    if (j_2 + 1 < blocks_2) {
                        int _stage_idx_full_3 = (int)((unsigned int)(tord_2 + j_2 + 1) % 2);
                        int _parity_idx_full_3 = (int)((unsigned int)(tord_2 + j_2 + 1) / 2 % 2);
                        mbarrier_wait(idx_full_addr + (_stage_idx_full_3) * 8, _parity_idx_full_3);
                        int key_t_1 = tile_idx[(tord_2 + j_2 + 1) % 2 * 80 + wg * 32 + lane_0_1];
                        mbarrier_arrive(idx_free_addr + ((int)((unsigned int)(tord_2 + j_2 + 1) % 2)) * 8);
                        int row_t_1 = key_t_1;
                        if (key_t_1 >= 0) {
                            if (dkv_has_map != 0) {
                                row_t_1 = dkv_dst_map[(long long)key_t_1];
                            }
                        }
                        x_row = row_t_1;
                    }
                    int _parity_dkv_a_full = (int)((unsigned int)(tord_2 + j_2) % 2);
                    mbarrier_wait(dkv_a_full_addr, _parity_dkv_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float a0[32];
                    float a1[32];
                    float ra0 = 0.0f;
                    float ra1 = 0.0f;
#if __CUDA_ARCH__ == 1000
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
#else
                    if (vmask != 0) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.abs.NaN.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(a0[0]), "=f"(a0[1]), "=f"(a0[2]), "=f"(a0[3]), "=f"(a0[4]), "=f"(a0[5]), "=f"(a0[6]), "=f"(a0[7]), "=f"(a0[8]), "=f"(a0[9]), "=f"(a0[10]), "=f"(a0[11]), "=f"(a0[12]), "=f"(a0[13]), "=f"(a0[14]), "=f"(a0[15]), "=f"(a0[16]), "=f"(a0[17]), "=f"(a0[18]), "=f"(a0[19]), "=f"(a0[20]), "=f"(a0[21]), "=f"(a0[22]), "=f"(a0[23]), "=f"(a0[24]), "=f"(a0[25]), "=f"(a0[26]), "=f"(a0[27]), "=f"(a0[28]), "=f"(a0[29]), "=f"(a0[30]), "=f"(a0[31]), "=f"(ra0)
                            : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.abs.NaN.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(a1[0]), "=f"(a1[1]), "=f"(a1[2]), "=f"(a1[3]), "=f"(a1[4]), "=f"(a1[5]), "=f"(a1[6]), "=f"(a1[7]), "=f"(a1[8]), "=f"(a1[9]), "=f"(a1[10]), "=f"(a1[11]), "=f"(a1[12]), "=f"(a1[13]), "=f"(a1[14]), "=f"(a1[15]), "=f"(a1[16]), "=f"(a1[17]), "=f"(a1[18]), "=f"(a1[19]), "=f"(a1[20]), "=f"(a1[21]), "=f"(a1[22]), "=f"(a1[23]), "=f"(a1[24]), "=f"(a1[25]), "=f"(a1[26]), "=f"(a1[27]), "=f"(a1[28]), "=f"(a1[29]), "=f"(a1[30]), "=f"(a1[31]), "=f"(ra1)
                            : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                    }
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkv_a_drained_addr);
                    unsigned int mx = 0;
                    if (vmask != 0) {
                        float lm = 0.0f;
                        unsigned int t4[4];
                        int rc_0 = 0;
                        float scale_c_0 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, roff, 0);
                        rc_0 = _shfl_sync_2;
                        float _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, s_dkv, 0);
                        scale_c_0 = _shfl_sync_3;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 0) : "memory");
                        rc_0 = (int)t4[0];
                        scale_c_0 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_0 = (long long)(rc_0 + lat_col);
                        float y0_0 = a0[0] * scale_c_0;
                        float y1_0 = a1[0] * scale_c_0;
                        int n0_0 = __float2int_rn(y0_0);
                        int n1_0 = __float2int_rn(y1_0);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_0))), "r"(static_cast<int>(n0_0)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_0 + 128ll))), "r"(static_cast<int>(n1_0)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_5;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(y0_0));
                        float _abs_6;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(y1_0));
                        float _max_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(_abs_5), "f"(_abs_6));
                        a0[0] = _max_4;
                        float _max_7;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(lm), "f"(a0[0]));
                        lm = _max_7;
#else
#endif
                        int rc_1 = 0;
                        float scale_c_1 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, roff, 1);
                        rc_1 = _shfl_sync_8;
                        float _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, s_dkv, 1);
                        scale_c_1 = _shfl_sync_9;
#else
                        rc_1 = (int)t4[2];
                        scale_c_1 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_1 = (long long)(rc_1 + lat_col);
                        float y0_1 = a0[1] * scale_c_1;
                        float y1_1 = a1[1] * scale_c_1;
                        int n0_1 = __float2int_rn(y0_1);
                        int n1_1 = __float2int_rn(y1_1);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_1))), "r"(static_cast<int>(n0_1)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_1 + 128ll))), "r"(static_cast<int>(n1_1)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_11;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(y0_1));
                        float _abs_12;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_12) : "f"(y1_1));
                        float _max_10;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(_abs_11), "f"(_abs_12));
                        a0[1] = _max_10;
                        float _max_13;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(lm), "f"(a0[1]));
                        lm = _max_13;
#else
#endif
                        int rc_2 = 0;
                        float scale_c_2 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, roff, 2);
                        rc_2 = _shfl_sync_14;
                        float _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, s_dkv, 2);
                        scale_c_2 = _shfl_sync_15;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 16) : "memory");
                        rc_2 = (int)t4[0];
                        scale_c_2 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_2 = (long long)(rc_2 + lat_col);
                        float y0_2 = a0[2] * scale_c_2;
                        float y1_2 = a1[2] * scale_c_2;
                        int n0_2 = __float2int_rn(y0_2);
                        int n1_2 = __float2int_rn(y1_2);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_2))), "r"(static_cast<int>(n0_2)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_2 + 128ll))), "r"(static_cast<int>(n1_2)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_17;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(y0_2));
                        float _abs_18;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(y1_2));
                        float _max_16;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(_abs_17), "f"(_abs_18));
                        a0[2] = _max_16;
                        float _max_19;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_19) : "f"(lm), "f"(a0[2]));
                        lm = _max_19;
#else
#endif
                        int rc_3 = 0;
                        float scale_c_3 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, roff, 3);
                        rc_3 = _shfl_sync_20;
                        float _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, s_dkv, 3);
                        scale_c_3 = _shfl_sync_21;
#else
                        rc_3 = (int)t4[2];
                        scale_c_3 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_3 = (long long)(rc_3 + lat_col);
                        float y0_3 = a0[3] * scale_c_3;
                        float y1_3 = a1[3] * scale_c_3;
                        int n0_3 = __float2int_rn(y0_3);
                        int n1_3 = __float2int_rn(y1_3);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_3))), "r"(static_cast<int>(n0_3)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_3 + 128ll))), "r"(static_cast<int>(n1_3)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_23;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(y0_3));
                        float _abs_24;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_24) : "f"(y1_3));
                        float _max_22;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_22) : "f"(_abs_23), "f"(_abs_24));
                        a0[3] = _max_22;
                        float _max_25;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(lm), "f"(a0[3]));
                        lm = _max_25;
#else
#endif
                        int rc_4 = 0;
                        float scale_c_4 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_26 = __shfl_sync(0xFFFFFFFF, roff, 4);
                        rc_4 = _shfl_sync_26;
                        float _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, s_dkv, 4);
                        scale_c_4 = _shfl_sync_27;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 32) : "memory");
                        rc_4 = (int)t4[0];
                        scale_c_4 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_4 = (long long)(rc_4 + lat_col);
                        float y0_4 = a0[4] * scale_c_4;
                        float y1_4 = a1[4] * scale_c_4;
                        int n0_4 = __float2int_rn(y0_4);
                        int n1_4 = __float2int_rn(y1_4);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_4))), "r"(static_cast<int>(n0_4)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_4 + 128ll))), "r"(static_cast<int>(n1_4)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_29;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(y0_4));
                        float _abs_30;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(y1_4));
                        float _max_28;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(_abs_29), "f"(_abs_30));
                        a0[4] = _max_28;
                        float _max_31;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_31) : "f"(lm), "f"(a0[4]));
                        lm = _max_31;
#else
#endif
                        int rc_5 = 0;
                        float scale_c_5 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, roff, 5);
                        rc_5 = _shfl_sync_32;
                        float _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, s_dkv, 5);
                        scale_c_5 = _shfl_sync_33;
#else
                        rc_5 = (int)t4[2];
                        scale_c_5 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_5 = (long long)(rc_5 + lat_col);
                        float y0_5 = a0[5] * scale_c_5;
                        float y1_5 = a1[5] * scale_c_5;
                        int n0_5 = __float2int_rn(y0_5);
                        int n1_5 = __float2int_rn(y1_5);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_5))), "r"(static_cast<int>(n0_5)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_5 + 128ll))), "r"(static_cast<int>(n1_5)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_35;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(y0_5));
                        float _abs_36;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_36) : "f"(y1_5));
                        float _max_34;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(_abs_35), "f"(_abs_36));
                        a0[5] = _max_34;
                        float _max_37;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(lm), "f"(a0[5]));
                        lm = _max_37;
#else
#endif
                        int rc_6 = 0;
                        float scale_c_6 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, roff, 6);
                        rc_6 = _shfl_sync_38;
                        float _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, s_dkv, 6);
                        scale_c_6 = _shfl_sync_39;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 48) : "memory");
                        rc_6 = (int)t4[0];
                        scale_c_6 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_6 = (long long)(rc_6 + lat_col);
                        float y0_6 = a0[6] * scale_c_6;
                        float y1_6 = a1[6] * scale_c_6;
                        int n0_6 = __float2int_rn(y0_6);
                        int n1_6 = __float2int_rn(y1_6);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_6))), "r"(static_cast<int>(n0_6)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_6 + 128ll))), "r"(static_cast<int>(n1_6)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_41;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(y0_6));
                        float _abs_42;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(y1_6));
                        float _max_40;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(_abs_41), "f"(_abs_42));
                        a0[6] = _max_40;
                        float _max_43;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(lm), "f"(a0[6]));
                        lm = _max_43;
#else
#endif
                        int rc_7 = 0;
                        float scale_c_7 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_44 = __shfl_sync(0xFFFFFFFF, roff, 7);
                        rc_7 = _shfl_sync_44;
                        float _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, s_dkv, 7);
                        scale_c_7 = _shfl_sync_45;
#else
                        rc_7 = (int)t4[2];
                        scale_c_7 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_7 = (long long)(rc_7 + lat_col);
                        float y0_7 = a0[7] * scale_c_7;
                        float y1_7 = a1[7] * scale_c_7;
                        int n0_7 = __float2int_rn(y0_7);
                        int n1_7 = __float2int_rn(y1_7);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_7))), "r"(static_cast<int>(n0_7)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_7 + 128ll))), "r"(static_cast<int>(n1_7)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_47;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(y0_7));
                        float _abs_48;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(y1_7));
                        float _max_46;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(_abs_47), "f"(_abs_48));
                        a0[7] = _max_46;
                        float _max_49;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(lm), "f"(a0[7]));
                        lm = _max_49;
#else
#endif
                        int rc_8 = 0;
                        float scale_c_8 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, roff, 8);
                        rc_8 = _shfl_sync_50;
                        float _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, s_dkv, 8);
                        scale_c_8 = _shfl_sync_51;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 64) : "memory");
                        rc_8 = (int)t4[0];
                        scale_c_8 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_8 = (long long)(rc_8 + lat_col);
                        float y0_8 = a0[8] * scale_c_8;
                        float y1_8 = a1[8] * scale_c_8;
                        int n0_8 = __float2int_rn(y0_8);
                        int n1_8 = __float2int_rn(y1_8);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_8))), "r"(static_cast<int>(n0_8)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_8 + 128ll))), "r"(static_cast<int>(n1_8)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_53;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(y0_8));
                        float _abs_54;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_54) : "f"(y1_8));
                        float _max_52;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(_abs_53), "f"(_abs_54));
                        a0[8] = _max_52;
                        float _max_55;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_55) : "f"(lm), "f"(a0[8]));
                        lm = _max_55;
#else
#endif
                        int rc_9 = 0;
                        float scale_c_9 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_56 = __shfl_sync(0xFFFFFFFF, roff, 9);
                        rc_9 = _shfl_sync_56;
                        float _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, s_dkv, 9);
                        scale_c_9 = _shfl_sync_57;
#else
                        rc_9 = (int)t4[2];
                        scale_c_9 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_9 = (long long)(rc_9 + lat_col);
                        float y0_9 = a0[9] * scale_c_9;
                        float y1_9 = a1[9] * scale_c_9;
                        int n0_9 = __float2int_rn(y0_9);
                        int n1_9 = __float2int_rn(y1_9);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_9))), "r"(static_cast<int>(n0_9)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_9 + 128ll))), "r"(static_cast<int>(n1_9)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_59;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(y0_9));
                        float _abs_60;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_60) : "f"(y1_9));
                        float _max_58;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(_abs_59), "f"(_abs_60));
                        a0[9] = _max_58;
                        float _max_61;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(lm), "f"(a0[9]));
                        lm = _max_61;
#else
#endif
                        int rc_10 = 0;
                        float scale_c_10 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_62 = __shfl_sync(0xFFFFFFFF, roff, 10);
                        rc_10 = _shfl_sync_62;
                        float _shfl_sync_63 = __shfl_sync(0xFFFFFFFF, s_dkv, 10);
                        scale_c_10 = _shfl_sync_63;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 80) : "memory");
                        rc_10 = (int)t4[0];
                        scale_c_10 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_10 = (long long)(rc_10 + lat_col);
                        float y0_10 = a0[10] * scale_c_10;
                        float y1_10 = a1[10] * scale_c_10;
                        int n0_10 = __float2int_rn(y0_10);
                        int n1_10 = __float2int_rn(y1_10);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_10))), "r"(static_cast<int>(n0_10)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_10 + 128ll))), "r"(static_cast<int>(n1_10)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_65;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_65) : "f"(y0_10));
                        float _abs_66;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(y1_10));
                        float _max_64;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(_abs_65), "f"(_abs_66));
                        a0[10] = _max_64;
                        float _max_67;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_67) : "f"(lm), "f"(a0[10]));
                        lm = _max_67;
#else
#endif
                        int rc_11 = 0;
                        float scale_c_11 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_68 = __shfl_sync(0xFFFFFFFF, roff, 11);
                        rc_11 = _shfl_sync_68;
                        float _shfl_sync_69 = __shfl_sync(0xFFFFFFFF, s_dkv, 11);
                        scale_c_11 = _shfl_sync_69;
#else
                        rc_11 = (int)t4[2];
                        scale_c_11 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_11 = (long long)(rc_11 + lat_col);
                        float y0_11 = a0[11] * scale_c_11;
                        float y1_11 = a1[11] * scale_c_11;
                        int n0_11 = __float2int_rn(y0_11);
                        int n1_11 = __float2int_rn(y1_11);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_11))), "r"(static_cast<int>(n0_11)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_11 + 128ll))), "r"(static_cast<int>(n1_11)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_71;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(y0_11));
                        float _abs_72;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_72) : "f"(y1_11));
                        float _max_70;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(_abs_71), "f"(_abs_72));
                        a0[11] = _max_70;
                        float _max_73;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(lm), "f"(a0[11]));
                        lm = _max_73;
#else
#endif
                        int rc_12 = 0;
                        float scale_c_12 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_74 = __shfl_sync(0xFFFFFFFF, roff, 12);
                        rc_12 = _shfl_sync_74;
                        float _shfl_sync_75 = __shfl_sync(0xFFFFFFFF, s_dkv, 12);
                        scale_c_12 = _shfl_sync_75;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 96) : "memory");
                        rc_12 = (int)t4[0];
                        scale_c_12 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_12 = (long long)(rc_12 + lat_col);
                        float y0_12 = a0[12] * scale_c_12;
                        float y1_12 = a1[12] * scale_c_12;
                        int n0_12 = __float2int_rn(y0_12);
                        int n1_12 = __float2int_rn(y1_12);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_12))), "r"(static_cast<int>(n0_12)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_12 + 128ll))), "r"(static_cast<int>(n1_12)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_77;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_77) : "f"(y0_12));
                        float _abs_78;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(y1_12));
                        float _max_76;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(_abs_77), "f"(_abs_78));
                        a0[12] = _max_76;
                        float _max_79;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_79) : "f"(lm), "f"(a0[12]));
                        lm = _max_79;
#else
#endif
                        int rc_13 = 0;
                        float scale_c_13 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_80 = __shfl_sync(0xFFFFFFFF, roff, 13);
                        rc_13 = _shfl_sync_80;
                        float _shfl_sync_81 = __shfl_sync(0xFFFFFFFF, s_dkv, 13);
                        scale_c_13 = _shfl_sync_81;
#else
                        rc_13 = (int)t4[2];
                        scale_c_13 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_13 = (long long)(rc_13 + lat_col);
                        float y0_13 = a0[13] * scale_c_13;
                        float y1_13 = a1[13] * scale_c_13;
                        int n0_13 = __float2int_rn(y0_13);
                        int n1_13 = __float2int_rn(y1_13);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_13))), "r"(static_cast<int>(n0_13)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_13 + 128ll))), "r"(static_cast<int>(n1_13)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_83;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(y0_13));
                        float _abs_84;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_84) : "f"(y1_13));
                        float _max_82;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_82) : "f"(_abs_83), "f"(_abs_84));
                        a0[13] = _max_82;
                        float _max_85;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_85) : "f"(lm), "f"(a0[13]));
                        lm = _max_85;
#else
#endif
                        int rc_14 = 0;
                        float scale_c_14 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_86 = __shfl_sync(0xFFFFFFFF, roff, 14);
                        rc_14 = _shfl_sync_86;
                        float _shfl_sync_87 = __shfl_sync(0xFFFFFFFF, s_dkv, 14);
                        scale_c_14 = _shfl_sync_87;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 112) : "memory");
                        rc_14 = (int)t4[0];
                        scale_c_14 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_14 = (long long)(rc_14 + lat_col);
                        float y0_14 = a0[14] * scale_c_14;
                        float y1_14 = a1[14] * scale_c_14;
                        int n0_14 = __float2int_rn(y0_14);
                        int n1_14 = __float2int_rn(y1_14);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_14))), "r"(static_cast<int>(n0_14)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_14 + 128ll))), "r"(static_cast<int>(n1_14)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_89;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_89) : "f"(y0_14));
                        float _abs_90;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(y1_14));
                        float _max_88;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_88) : "f"(_abs_89), "f"(_abs_90));
                        a0[14] = _max_88;
                        float _max_91;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_91) : "f"(lm), "f"(a0[14]));
                        lm = _max_91;
#else
#endif
                        int rc_15 = 0;
                        float scale_c_15 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_92 = __shfl_sync(0xFFFFFFFF, roff, 15);
                        rc_15 = _shfl_sync_92;
                        float _shfl_sync_93 = __shfl_sync(0xFFFFFFFF, s_dkv, 15);
                        scale_c_15 = _shfl_sync_93;
#else
                        rc_15 = (int)t4[2];
                        scale_c_15 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_15 = (long long)(rc_15 + lat_col);
                        float y0_15 = a0[15] * scale_c_15;
                        float y1_15 = a1[15] * scale_c_15;
                        int n0_15 = __float2int_rn(y0_15);
                        int n1_15 = __float2int_rn(y1_15);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_15))), "r"(static_cast<int>(n0_15)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_15 + 128ll))), "r"(static_cast<int>(n1_15)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_95;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_95) : "f"(y0_15));
                        float _abs_96;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_96) : "f"(y1_15));
                        float _max_94;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_94) : "f"(_abs_95), "f"(_abs_96));
                        a0[15] = _max_94;
                        float _max_97;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(lm), "f"(a0[15]));
                        lm = _max_97;
#else
#endif
                        int rc_16 = 0;
                        float scale_c_16 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_98 = __shfl_sync(0xFFFFFFFF, roff, 16);
                        rc_16 = _shfl_sync_98;
                        float _shfl_sync_99 = __shfl_sync(0xFFFFFFFF, s_dkv, 16);
                        scale_c_16 = _shfl_sync_99;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 128) : "memory");
                        rc_16 = (int)t4[0];
                        scale_c_16 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_16 = (long long)(rc_16 + lat_col);
                        float y0_16 = a0[16] * scale_c_16;
                        float y1_16 = a1[16] * scale_c_16;
                        int n0_16 = __float2int_rn(y0_16);
                        int n1_16 = __float2int_rn(y1_16);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_16))), "r"(static_cast<int>(n0_16)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_16 + 128ll))), "r"(static_cast<int>(n1_16)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_101;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_101) : "f"(y0_16));
                        float _abs_102;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(y1_16));
                        float _max_100;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_100) : "f"(_abs_101), "f"(_abs_102));
                        a0[16] = _max_100;
                        float _max_103;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_103) : "f"(lm), "f"(a0[16]));
                        lm = _max_103;
#else
#endif
                        int rc_17 = 0;
                        float scale_c_17 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_104 = __shfl_sync(0xFFFFFFFF, roff, 17);
                        rc_17 = _shfl_sync_104;
                        float _shfl_sync_105 = __shfl_sync(0xFFFFFFFF, s_dkv, 17);
                        scale_c_17 = _shfl_sync_105;
#else
                        rc_17 = (int)t4[2];
                        scale_c_17 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_17 = (long long)(rc_17 + lat_col);
                        float y0_17 = a0[17] * scale_c_17;
                        float y1_17 = a1[17] * scale_c_17;
                        int n0_17 = __float2int_rn(y0_17);
                        int n1_17 = __float2int_rn(y1_17);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_17))), "r"(static_cast<int>(n0_17)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_17 + 128ll))), "r"(static_cast<int>(n1_17)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_107;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_107) : "f"(y0_17));
                        float _abs_108;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_108) : "f"(y1_17));
                        float _max_106;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_106) : "f"(_abs_107), "f"(_abs_108));
                        a0[17] = _max_106;
                        float _max_109;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(lm), "f"(a0[17]));
                        lm = _max_109;
#else
#endif
                        int rc_18 = 0;
                        float scale_c_18 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_110 = __shfl_sync(0xFFFFFFFF, roff, 18);
                        rc_18 = _shfl_sync_110;
                        float _shfl_sync_111 = __shfl_sync(0xFFFFFFFF, s_dkv, 18);
                        scale_c_18 = _shfl_sync_111;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 144) : "memory");
                        rc_18 = (int)t4[0];
                        scale_c_18 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_18 = (long long)(rc_18 + lat_col);
                        float y0_18 = a0[18] * scale_c_18;
                        float y1_18 = a1[18] * scale_c_18;
                        int n0_18 = __float2int_rn(y0_18);
                        int n1_18 = __float2int_rn(y1_18);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_18))), "r"(static_cast<int>(n0_18)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_18 + 128ll))), "r"(static_cast<int>(n1_18)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_113;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_113) : "f"(y0_18));
                        float _abs_114;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(y1_18));
                        float _max_112;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_112) : "f"(_abs_113), "f"(_abs_114));
                        a0[18] = _max_112;
                        float _max_115;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_115) : "f"(lm), "f"(a0[18]));
                        lm = _max_115;
#else
#endif
                        int rc_19 = 0;
                        float scale_c_19 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_116 = __shfl_sync(0xFFFFFFFF, roff, 19);
                        rc_19 = _shfl_sync_116;
                        float _shfl_sync_117 = __shfl_sync(0xFFFFFFFF, s_dkv, 19);
                        scale_c_19 = _shfl_sync_117;
#else
                        rc_19 = (int)t4[2];
                        scale_c_19 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_19 = (long long)(rc_19 + lat_col);
                        float y0_19 = a0[19] * scale_c_19;
                        float y1_19 = a1[19] * scale_c_19;
                        int n0_19 = __float2int_rn(y0_19);
                        int n1_19 = __float2int_rn(y1_19);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_19))), "r"(static_cast<int>(n0_19)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_19 + 128ll))), "r"(static_cast<int>(n1_19)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_119;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(y0_19));
                        float _abs_120;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_120) : "f"(y1_19));
                        float _max_118;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_118) : "f"(_abs_119), "f"(_abs_120));
                        a0[19] = _max_118;
                        float _max_121;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(lm), "f"(a0[19]));
                        lm = _max_121;
#else
#endif
                        int rc_20 = 0;
                        float scale_c_20 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_122 = __shfl_sync(0xFFFFFFFF, roff, 20);
                        rc_20 = _shfl_sync_122;
                        float _shfl_sync_123 = __shfl_sync(0xFFFFFFFF, s_dkv, 20);
                        scale_c_20 = _shfl_sync_123;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 160) : "memory");
                        rc_20 = (int)t4[0];
                        scale_c_20 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_20 = (long long)(rc_20 + lat_col);
                        float y0_20 = a0[20] * scale_c_20;
                        float y1_20 = a1[20] * scale_c_20;
                        int n0_20 = __float2int_rn(y0_20);
                        int n1_20 = __float2int_rn(y1_20);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_20))), "r"(static_cast<int>(n0_20)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_20 + 128ll))), "r"(static_cast<int>(n1_20)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_125;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_125) : "f"(y0_20));
                        float _abs_126;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(y1_20));
                        float _max_124;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_124) : "f"(_abs_125), "f"(_abs_126));
                        a0[20] = _max_124;
                        float _max_127;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_127) : "f"(lm), "f"(a0[20]));
                        lm = _max_127;
#else
#endif
                        int rc_21 = 0;
                        float scale_c_21 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_128 = __shfl_sync(0xFFFFFFFF, roff, 21);
                        rc_21 = _shfl_sync_128;
                        float _shfl_sync_129 = __shfl_sync(0xFFFFFFFF, s_dkv, 21);
                        scale_c_21 = _shfl_sync_129;
#else
                        rc_21 = (int)t4[2];
                        scale_c_21 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_21 = (long long)(rc_21 + lat_col);
                        float y0_21 = a0[21] * scale_c_21;
                        float y1_21 = a1[21] * scale_c_21;
                        int n0_21 = __float2int_rn(y0_21);
                        int n1_21 = __float2int_rn(y1_21);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_21))), "r"(static_cast<int>(n0_21)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_21 + 128ll))), "r"(static_cast<int>(n1_21)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_131;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(y0_21));
                        float _abs_132;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_132) : "f"(y1_21));
                        float _max_130;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_130) : "f"(_abs_131), "f"(_abs_132));
                        a0[21] = _max_130;
                        float _max_133;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_133) : "f"(lm), "f"(a0[21]));
                        lm = _max_133;
#else
#endif
                        int rc_22 = 0;
                        float scale_c_22 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_134 = __shfl_sync(0xFFFFFFFF, roff, 22);
                        rc_22 = _shfl_sync_134;
                        float _shfl_sync_135 = __shfl_sync(0xFFFFFFFF, s_dkv, 22);
                        scale_c_22 = _shfl_sync_135;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 176) : "memory");
                        rc_22 = (int)t4[0];
                        scale_c_22 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_22 = (long long)(rc_22 + lat_col);
                        float y0_22 = a0[22] * scale_c_22;
                        float y1_22 = a1[22] * scale_c_22;
                        int n0_22 = __float2int_rn(y0_22);
                        int n1_22 = __float2int_rn(y1_22);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_22))), "r"(static_cast<int>(n0_22)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_22 + 128ll))), "r"(static_cast<int>(n1_22)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_137;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(y0_22));
                        float _abs_138;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_138) : "f"(y1_22));
                        float _max_136;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_136) : "f"(_abs_137), "f"(_abs_138));
                        a0[22] = _max_136;
                        float _max_139;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_139) : "f"(lm), "f"(a0[22]));
                        lm = _max_139;
#else
#endif
                        int rc_23 = 0;
                        float scale_c_23 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_140 = __shfl_sync(0xFFFFFFFF, roff, 23);
                        rc_23 = _shfl_sync_140;
                        float _shfl_sync_141 = __shfl_sync(0xFFFFFFFF, s_dkv, 23);
                        scale_c_23 = _shfl_sync_141;
#else
                        rc_23 = (int)t4[2];
                        scale_c_23 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_23 = (long long)(rc_23 + lat_col);
                        float y0_23 = a0[23] * scale_c_23;
                        float y1_23 = a1[23] * scale_c_23;
                        int n0_23 = __float2int_rn(y0_23);
                        int n1_23 = __float2int_rn(y1_23);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_23))), "r"(static_cast<int>(n0_23)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_23 + 128ll))), "r"(static_cast<int>(n1_23)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_143;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_143) : "f"(y0_23));
                        float _abs_144;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(y1_23));
                        float _max_142;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_142) : "f"(_abs_143), "f"(_abs_144));
                        a0[23] = _max_142;
                        float _max_145;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_145) : "f"(lm), "f"(a0[23]));
                        lm = _max_145;
#else
#endif
                        int rc_24 = 0;
                        float scale_c_24 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_146 = __shfl_sync(0xFFFFFFFF, roff, 24);
                        rc_24 = _shfl_sync_146;
                        float _shfl_sync_147 = __shfl_sync(0xFFFFFFFF, s_dkv, 24);
                        scale_c_24 = _shfl_sync_147;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 192) : "memory");
                        rc_24 = (int)t4[0];
                        scale_c_24 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_24 = (long long)(rc_24 + lat_col);
                        float y0_24 = a0[24] * scale_c_24;
                        float y1_24 = a1[24] * scale_c_24;
                        int n0_24 = __float2int_rn(y0_24);
                        int n1_24 = __float2int_rn(y1_24);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_24))), "r"(static_cast<int>(n0_24)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_24 + 128ll))), "r"(static_cast<int>(n1_24)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_149;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(y0_24));
                        float _abs_150;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(y1_24));
                        float _max_148;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_148) : "f"(_abs_149), "f"(_abs_150));
                        a0[24] = _max_148;
                        float _max_151;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_151) : "f"(lm), "f"(a0[24]));
                        lm = _max_151;
#else
#endif
                        int rc_25 = 0;
                        float scale_c_25 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_152 = __shfl_sync(0xFFFFFFFF, roff, 25);
                        rc_25 = _shfl_sync_152;
                        float _shfl_sync_153 = __shfl_sync(0xFFFFFFFF, s_dkv, 25);
                        scale_c_25 = _shfl_sync_153;
#else
                        rc_25 = (int)t4[2];
                        scale_c_25 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_25 = (long long)(rc_25 + lat_col);
                        float y0_25 = a0[25] * scale_c_25;
                        float y1_25 = a1[25] * scale_c_25;
                        int n0_25 = __float2int_rn(y0_25);
                        int n1_25 = __float2int_rn(y1_25);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_25))), "r"(static_cast<int>(n0_25)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_25 + 128ll))), "r"(static_cast<int>(n1_25)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_155;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(y0_25));
                        float _abs_156;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_156) : "f"(y1_25));
                        float _max_154;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_154) : "f"(_abs_155), "f"(_abs_156));
                        a0[25] = _max_154;
                        float _max_157;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_157) : "f"(lm), "f"(a0[25]));
                        lm = _max_157;
#else
#endif
                        int rc_26 = 0;
                        float scale_c_26 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_158 = __shfl_sync(0xFFFFFFFF, roff, 26);
                        rc_26 = _shfl_sync_158;
                        float _shfl_sync_159 = __shfl_sync(0xFFFFFFFF, s_dkv, 26);
                        scale_c_26 = _shfl_sync_159;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 208) : "memory");
                        rc_26 = (int)t4[0];
                        scale_c_26 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_26 = (long long)(rc_26 + lat_col);
                        float y0_26 = a0[26] * scale_c_26;
                        float y1_26 = a1[26] * scale_c_26;
                        int n0_26 = __float2int_rn(y0_26);
                        int n1_26 = __float2int_rn(y1_26);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_26))), "r"(static_cast<int>(n0_26)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_26 + 128ll))), "r"(static_cast<int>(n1_26)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_161;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(y0_26));
                        float _abs_162;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_162) : "f"(y1_26));
                        float _max_160;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_160) : "f"(_abs_161), "f"(_abs_162));
                        a0[26] = _max_160;
                        float _max_163;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_163) : "f"(lm), "f"(a0[26]));
                        lm = _max_163;
#else
#endif
                        int rc_27 = 0;
                        float scale_c_27 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_164 = __shfl_sync(0xFFFFFFFF, roff, 27);
                        rc_27 = _shfl_sync_164;
                        float _shfl_sync_165 = __shfl_sync(0xFFFFFFFF, s_dkv, 27);
                        scale_c_27 = _shfl_sync_165;
#else
                        rc_27 = (int)t4[2];
                        scale_c_27 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_27 = (long long)(rc_27 + lat_col);
                        float y0_27 = a0[27] * scale_c_27;
                        float y1_27 = a1[27] * scale_c_27;
                        int n0_27 = __float2int_rn(y0_27);
                        int n1_27 = __float2int_rn(y1_27);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_27))), "r"(static_cast<int>(n0_27)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_27 + 128ll))), "r"(static_cast<int>(n1_27)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_167;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_167) : "f"(y0_27));
                        float _abs_168;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_168) : "f"(y1_27));
                        float _max_166;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_166) : "f"(_abs_167), "f"(_abs_168));
                        a0[27] = _max_166;
                        float _max_169;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_169) : "f"(lm), "f"(a0[27]));
                        lm = _max_169;
#else
#endif
                        int rc_28 = 0;
                        float scale_c_28 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_170 = __shfl_sync(0xFFFFFFFF, roff, 28);
                        rc_28 = _shfl_sync_170;
                        float _shfl_sync_171 = __shfl_sync(0xFFFFFFFF, s_dkv, 28);
                        scale_c_28 = _shfl_sync_171;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 224) : "memory");
                        rc_28 = (int)t4[0];
                        scale_c_28 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_28 = (long long)(rc_28 + lat_col);
                        float y0_28 = a0[28] * scale_c_28;
                        float y1_28 = a1[28] * scale_c_28;
                        int n0_28 = __float2int_rn(y0_28);
                        int n1_28 = __float2int_rn(y1_28);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_28))), "r"(static_cast<int>(n0_28)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_28 + 128ll))), "r"(static_cast<int>(n1_28)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_173;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(y0_28));
                        float _abs_174;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_174) : "f"(y1_28));
                        float _max_172;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_172) : "f"(_abs_173), "f"(_abs_174));
                        a0[28] = _max_172;
                        float _max_175;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_175) : "f"(lm), "f"(a0[28]));
                        lm = _max_175;
#else
#endif
                        int rc_29 = 0;
                        float scale_c_29 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_176 = __shfl_sync(0xFFFFFFFF, roff, 29);
                        rc_29 = _shfl_sync_176;
                        float _shfl_sync_177 = __shfl_sync(0xFFFFFFFF, s_dkv, 29);
                        scale_c_29 = _shfl_sync_177;
#else
                        rc_29 = (int)t4[2];
                        scale_c_29 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_29 = (long long)(rc_29 + lat_col);
                        float y0_29 = a0[29] * scale_c_29;
                        float y1_29 = a1[29] * scale_c_29;
                        int n0_29 = __float2int_rn(y0_29);
                        int n1_29 = __float2int_rn(y1_29);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_29))), "r"(static_cast<int>(n0_29)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_29 + 128ll))), "r"(static_cast<int>(n1_29)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_179;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(y0_29));
                        float _abs_180;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_180) : "f"(y1_29));
                        float _max_178;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_178) : "f"(_abs_179), "f"(_abs_180));
                        a0[29] = _max_178;
                        float _max_181;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_181) : "f"(lm), "f"(a0[29]));
                        lm = _max_181;
#else
#endif
                        int rc_30 = 0;
                        float scale_c_30 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_182 = __shfl_sync(0xFFFFFFFF, roff, 30);
                        rc_30 = _shfl_sync_182;
                        float _shfl_sync_183 = __shfl_sync(0xFFFFFFFF, s_dkv, 30);
                        scale_c_30 = _shfl_sync_183;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4[(0) + 3]))
                            : "r"(tab_w + 240) : "memory");
                        rc_30 = (int)t4[0];
                        scale_c_30 = __uint_as_float(t4[1]);
#endif
                        long long idx_c_30 = (long long)(rc_30 + lat_col);
                        float y0_30 = a0[30] * scale_c_30;
                        float y1_30 = a1[30] * scale_c_30;
                        int n0_30 = __float2int_rn(y0_30);
                        int n1_30 = __float2int_rn(y1_30);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_30))), "r"(static_cast<int>(n0_30)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_30 + 128ll))), "r"(static_cast<int>(n1_30)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_185;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(y0_30));
                        float _abs_186;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_186) : "f"(y1_30));
                        float _max_184;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_184) : "f"(_abs_185), "f"(_abs_186));
                        a0[30] = _max_184;
                        float _max_187;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_187) : "f"(lm), "f"(a0[30]));
                        lm = _max_187;
#else
#endif
                        int rc_31 = 0;
                        float scale_c_31 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_188 = __shfl_sync(0xFFFFFFFF, roff, 31);
                        rc_31 = _shfl_sync_188;
                        float _shfl_sync_189 = __shfl_sync(0xFFFFFFFF, s_dkv, 31);
                        scale_c_31 = _shfl_sync_189;
#else
                        rc_31 = (int)t4[2];
                        scale_c_31 = __uint_as_float(t4[3]);
#endif
                        long long idx_c_31 = (long long)(rc_31 + lat_col);
                        float y0_31 = a0[31] * scale_c_31;
                        float y1_31 = a1[31] * scale_c_31;
                        int n0_31 = __float2int_rn(y0_31);
                        int n1_31 = __float2int_rn(y1_31);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_31))), "r"(static_cast<int>(n0_31)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_31 + 128ll))), "r"(static_cast<int>(n1_31)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_191;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(y0_31));
                        float _abs_192;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(y1_31));
                        float _max_190;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_190) : "f"(_abs_191), "f"(_abs_192));
                        a0[31] = _max_190;
                        float _max_193;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_193) : "f"(lm), "f"(a0[31]));
                        lm = _max_193;
                        unsigned int _warp_redux_194;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_194) : "r"(__as_u32(lm)));
                        if (_warp_redux_194 >= thr_w) {
                            unsigned int m_c_0;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_0) : "r"(__as_u32(a0[0])));
#else
                        float _max_4;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(ra0), "f"(ra1));
                        unsigned int _warp_redux_3;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(__as_u32(_max_4)));
                        if (_warp_redux_3 >= thr_w) {
                            float _abs_6;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_6) : "f"(a0[0]));
                            float _abs_7;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(a1[0]));
                            float _max_5;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(_abs_6), "f"(_abs_7));
                            unsigned int m_a_0;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_0) : "r"(__as_u32(_max_5)));
#endif
                            if (lane_0_1 == 0) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_195 = ((mx) > (m_c_0) ? (mx) : (m_c_0));
                                mx = _max_195;
#else
                                unsigned int _max_8 = ((mx) > (m_a_0) ? (mx) : (m_a_0));
                                mx = _max_8;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_1;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_1) : "r"(__as_u32(a0[1])));
#else
                            float _abs_10;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_10) : "f"(a0[1]));
                            float _abs_11;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(a1[1]));
                            float _max_9;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_abs_10), "f"(_abs_11));
                            unsigned int m_a_1;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_1) : "r"(__as_u32(_max_9)));
#endif
                            if (lane_0_1 == 1) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_196 = ((mx) > (m_c_1) ? (mx) : (m_c_1));
                                mx = _max_196;
#else
                                unsigned int _max_12 = ((mx) > (m_a_1) ? (mx) : (m_a_1));
                                mx = _max_12;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_2;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_2) : "r"(__as_u32(a0[2])));
#else
                            float _abs_14;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_14) : "f"(a0[2]));
                            float _abs_15;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(a1[2]));
                            float _max_13;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(_abs_14), "f"(_abs_15));
                            unsigned int m_a_2;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_2) : "r"(__as_u32(_max_13)));
#endif
                            if (lane_0_1 == 2) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_197 = ((mx) > (m_c_2) ? (mx) : (m_c_2));
                                mx = _max_197;
#else
                                unsigned int _max_16 = ((mx) > (m_a_2) ? (mx) : (m_a_2));
                                mx = _max_16;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_3;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_3) : "r"(__as_u32(a0[3])));
#else
                            float _abs_18;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_18) : "f"(a0[3]));
                            float _abs_19;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(a1[3]));
                            float _max_17;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_17) : "f"(_abs_18), "f"(_abs_19));
                            unsigned int m_a_3;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_3) : "r"(__as_u32(_max_17)));
#endif
                            if (lane_0_1 == 3) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_198 = ((mx) > (m_c_3) ? (mx) : (m_c_3));
                                mx = _max_198;
#else
                                unsigned int _max_20 = ((mx) > (m_a_3) ? (mx) : (m_a_3));
                                mx = _max_20;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_4;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_4) : "r"(__as_u32(a0[4])));
#else
                            float _abs_22;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_22) : "f"(a0[4]));
                            float _abs_23;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(a1[4]));
                            float _max_21;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_21) : "f"(_abs_22), "f"(_abs_23));
                            unsigned int m_a_4;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_4) : "r"(__as_u32(_max_21)));
#endif
                            if (lane_0_1 == 4) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_199 = ((mx) > (m_c_4) ? (mx) : (m_c_4));
                                mx = _max_199;
#else
                                unsigned int _max_24 = ((mx) > (m_a_4) ? (mx) : (m_a_4));
                                mx = _max_24;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_5;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_5) : "r"(__as_u32(a0[5])));
#else
                            float _abs_26;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_26) : "f"(a0[5]));
                            float _abs_27;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(a1[5]));
                            float _max_25;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(_abs_26), "f"(_abs_27));
                            unsigned int m_a_5;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_5) : "r"(__as_u32(_max_25)));
#endif
                            if (lane_0_1 == 5) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_200 = ((mx) > (m_c_5) ? (mx) : (m_c_5));
                                mx = _max_200;
#else
                                unsigned int _max_28 = ((mx) > (m_a_5) ? (mx) : (m_a_5));
                                mx = _max_28;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_6;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_6) : "r"(__as_u32(a0[6])));
#else
                            float _abs_30;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_30) : "f"(a0[6]));
                            float _abs_31;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(a1[6]));
                            float _max_29;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(_abs_30), "f"(_abs_31));
                            unsigned int m_a_6;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_6) : "r"(__as_u32(_max_29)));
#endif
                            if (lane_0_1 == 6) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_201 = ((mx) > (m_c_6) ? (mx) : (m_c_6));
                                mx = _max_201;
#else
                                unsigned int _max_32 = ((mx) > (m_a_6) ? (mx) : (m_a_6));
                                mx = _max_32;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_7;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_7) : "r"(__as_u32(a0[7])));
#else
                            float _abs_34;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_34) : "f"(a0[7]));
                            float _abs_35;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(a1[7]));
                            float _max_33;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_33) : "f"(_abs_34), "f"(_abs_35));
                            unsigned int m_a_7;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_7) : "r"(__as_u32(_max_33)));
#endif
                            if (lane_0_1 == 7) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_202 = ((mx) > (m_c_7) ? (mx) : (m_c_7));
                                mx = _max_202;
#else
                                unsigned int _max_36 = ((mx) > (m_a_7) ? (mx) : (m_a_7));
                                mx = _max_36;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_8;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_8) : "r"(__as_u32(a0[8])));
#else
                            float _abs_38;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_38) : "f"(a0[8]));
                            float _abs_39;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_39) : "f"(a1[8]));
                            float _max_37;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_abs_38), "f"(_abs_39));
                            unsigned int m_a_8;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_8) : "r"(__as_u32(_max_37)));
#endif
                            if (lane_0_1 == 8) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_203 = ((mx) > (m_c_8) ? (mx) : (m_c_8));
                                mx = _max_203;
#else
                                unsigned int _max_40 = ((mx) > (m_a_8) ? (mx) : (m_a_8));
                                mx = _max_40;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_9;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_9) : "r"(__as_u32(a0[9])));
#else
                            float _abs_42;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(a0[9]));
                            float _abs_43;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(a1[9]));
                            float _max_41;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_41) : "f"(_abs_42), "f"(_abs_43));
                            unsigned int m_a_9;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_9) : "r"(__as_u32(_max_41)));
#endif
                            if (lane_0_1 == 9) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_204 = ((mx) > (m_c_9) ? (mx) : (m_c_9));
                                mx = _max_204;
#else
                                unsigned int _max_44 = ((mx) > (m_a_9) ? (mx) : (m_a_9));
                                mx = _max_44;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_10;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_10) : "r"(__as_u32(a0[10])));
#else
                            float _abs_46;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_46) : "f"(a0[10]));
                            float _abs_47;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(a1[10]));
                            float _max_45;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(_abs_46), "f"(_abs_47));
                            unsigned int m_a_10;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_10) : "r"(__as_u32(_max_45)));
#endif
                            if (lane_0_1 == 10) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_205 = ((mx) > (m_c_10) ? (mx) : (m_c_10));
                                mx = _max_205;
#else
                                unsigned int _max_48 = ((mx) > (m_a_10) ? (mx) : (m_a_10));
                                mx = _max_48;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_11;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_11) : "r"(__as_u32(a0[11])));
#else
                            float _abs_50;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(a0[11]));
                            float _abs_51;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(a1[11]));
                            float _max_49;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(_abs_50), "f"(_abs_51));
                            unsigned int m_a_11;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_11) : "r"(__as_u32(_max_49)));
#endif
                            if (lane_0_1 == 11) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_206 = ((mx) > (m_c_11) ? (mx) : (m_c_11));
                                mx = _max_206;
#else
                                unsigned int _max_52 = ((mx) > (m_a_11) ? (mx) : (m_a_11));
                                mx = _max_52;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_12;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_12) : "r"(__as_u32(a0[12])));
#else
                            float _abs_54;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_54) : "f"(a0[12]));
                            float _abs_55;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(a1[12]));
                            float _max_53;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_53) : "f"(_abs_54), "f"(_abs_55));
                            unsigned int m_a_12;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_12) : "r"(__as_u32(_max_53)));
#endif
                            if (lane_0_1 == 12) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_207 = ((mx) > (m_c_12) ? (mx) : (m_c_12));
                                mx = _max_207;
#else
                                unsigned int _max_56 = ((mx) > (m_a_12) ? (mx) : (m_a_12));
                                mx = _max_56;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_13;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_13) : "r"(__as_u32(a0[13])));
#else
                            float _abs_58;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_58) : "f"(a0[13]));
                            float _abs_59;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(a1[13]));
                            float _max_57;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(_abs_58), "f"(_abs_59));
                            unsigned int m_a_13;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_13) : "r"(__as_u32(_max_57)));
#endif
                            if (lane_0_1 == 13) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_208 = ((mx) > (m_c_13) ? (mx) : (m_c_13));
                                mx = _max_208;
#else
                                unsigned int _max_60 = ((mx) > (m_a_13) ? (mx) : (m_a_13));
                                mx = _max_60;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_14;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_14) : "r"(__as_u32(a0[14])));
#else
                            float _abs_62;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_62) : "f"(a0[14]));
                            float _abs_63;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(a1[14]));
                            float _max_61;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(_abs_62), "f"(_abs_63));
                            unsigned int m_a_14;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_14) : "r"(__as_u32(_max_61)));
#endif
                            if (lane_0_1 == 14) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_209 = ((mx) > (m_c_14) ? (mx) : (m_c_14));
                                mx = _max_209;
#else
                                unsigned int _max_64 = ((mx) > (m_a_14) ? (mx) : (m_a_14));
                                mx = _max_64;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_15;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_15) : "r"(__as_u32(a0[15])));
#else
                            float _abs_66;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_66) : "f"(a0[15]));
                            float _abs_67;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(a1[15]));
                            float _max_65;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(_abs_66), "f"(_abs_67));
                            unsigned int m_a_15;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_15) : "r"(__as_u32(_max_65)));
#endif
                            if (lane_0_1 == 15) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_210 = ((mx) > (m_c_15) ? (mx) : (m_c_15));
                                mx = _max_210;
#else
                                unsigned int _max_68 = ((mx) > (m_a_15) ? (mx) : (m_a_15));
                                mx = _max_68;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_16;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_16) : "r"(__as_u32(a0[16])));
#else
                            float _abs_70;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_70) : "f"(a0[16]));
                            float _abs_71;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(a1[16]));
                            float _max_69;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_69) : "f"(_abs_70), "f"(_abs_71));
                            unsigned int m_a_16;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_16) : "r"(__as_u32(_max_69)));
#endif
                            if (lane_0_1 == 16) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_211 = ((mx) > (m_c_16) ? (mx) : (m_c_16));
                                mx = _max_211;
#else
                                unsigned int _max_72 = ((mx) > (m_a_16) ? (mx) : (m_a_16));
                                mx = _max_72;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_17;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_17) : "r"(__as_u32(a0[17])));
#else
                            float _abs_74;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_74) : "f"(a0[17]));
                            float _abs_75;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_75) : "f"(a1[17]));
                            float _max_73;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(_abs_74), "f"(_abs_75));
                            unsigned int m_a_17;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_17) : "r"(__as_u32(_max_73)));
#endif
                            if (lane_0_1 == 17) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_212 = ((mx) > (m_c_17) ? (mx) : (m_c_17));
                                mx = _max_212;
#else
                                unsigned int _max_76 = ((mx) > (m_a_17) ? (mx) : (m_a_17));
                                mx = _max_76;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_18;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_18) : "r"(__as_u32(a0[18])));
#else
                            float _abs_78;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_78) : "f"(a0[18]));
                            float _abs_79;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_79) : "f"(a1[18]));
                            float _max_77;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_77) : "f"(_abs_78), "f"(_abs_79));
                            unsigned int m_a_18;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_18) : "r"(__as_u32(_max_77)));
#endif
                            if (lane_0_1 == 18) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_213 = ((mx) > (m_c_18) ? (mx) : (m_c_18));
                                mx = _max_213;
#else
                                unsigned int _max_80 = ((mx) > (m_a_18) ? (mx) : (m_a_18));
                                mx = _max_80;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_19;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_19) : "r"(__as_u32(a0[19])));
#else
                            float _abs_82;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_82) : "f"(a0[19]));
                            float _abs_83;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_83) : "f"(a1[19]));
                            float _max_81;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_81) : "f"(_abs_82), "f"(_abs_83));
                            unsigned int m_a_19;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_19) : "r"(__as_u32(_max_81)));
#endif
                            if (lane_0_1 == 19) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_214 = ((mx) > (m_c_19) ? (mx) : (m_c_19));
                                mx = _max_214;
#else
                                unsigned int _max_84 = ((mx) > (m_a_19) ? (mx) : (m_a_19));
                                mx = _max_84;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_20;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_20) : "r"(__as_u32(a0[20])));
#else
                            float _abs_86;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_86) : "f"(a0[20]));
                            float _abs_87;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_87) : "f"(a1[20]));
                            float _max_85;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_85) : "f"(_abs_86), "f"(_abs_87));
                            unsigned int m_a_20;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_20) : "r"(__as_u32(_max_85)));
#endif
                            if (lane_0_1 == 20) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_215 = ((mx) > (m_c_20) ? (mx) : (m_c_20));
                                mx = _max_215;
#else
                                unsigned int _max_88 = ((mx) > (m_a_20) ? (mx) : (m_a_20));
                                mx = _max_88;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_21;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_21) : "r"(__as_u32(a0[21])));
#else
                            float _abs_90;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_90) : "f"(a0[21]));
                            float _abs_91;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_91) : "f"(a1[21]));
                            float _max_89;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_89) : "f"(_abs_90), "f"(_abs_91));
                            unsigned int m_a_21;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_21) : "r"(__as_u32(_max_89)));
#endif
                            if (lane_0_1 == 21) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_216 = ((mx) > (m_c_21) ? (mx) : (m_c_21));
                                mx = _max_216;
#else
                                unsigned int _max_92 = ((mx) > (m_a_21) ? (mx) : (m_a_21));
                                mx = _max_92;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_22;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_22) : "r"(__as_u32(a0[22])));
#else
                            float _abs_94;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_94) : "f"(a0[22]));
                            float _abs_95;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_95) : "f"(a1[22]));
                            float _max_93;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_93) : "f"(_abs_94), "f"(_abs_95));
                            unsigned int m_a_22;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_22) : "r"(__as_u32(_max_93)));
#endif
                            if (lane_0_1 == 22) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_217 = ((mx) > (m_c_22) ? (mx) : (m_c_22));
                                mx = _max_217;
#else
                                unsigned int _max_96 = ((mx) > (m_a_22) ? (mx) : (m_a_22));
                                mx = _max_96;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_23;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_23) : "r"(__as_u32(a0[23])));
#else
                            float _abs_98;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_98) : "f"(a0[23]));
                            float _abs_99;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_99) : "f"(a1[23]));
                            float _max_97;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(_abs_98), "f"(_abs_99));
                            unsigned int m_a_23;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_23) : "r"(__as_u32(_max_97)));
#endif
                            if (lane_0_1 == 23) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_218 = ((mx) > (m_c_23) ? (mx) : (m_c_23));
                                mx = _max_218;
#else
                                unsigned int _max_100 = ((mx) > (m_a_23) ? (mx) : (m_a_23));
                                mx = _max_100;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_24;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_24) : "r"(__as_u32(a0[24])));
#else
                            float _abs_102;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_102) : "f"(a0[24]));
                            float _abs_103;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_103) : "f"(a1[24]));
                            float _max_101;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_101) : "f"(_abs_102), "f"(_abs_103));
                            unsigned int m_a_24;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_24) : "r"(__as_u32(_max_101)));
#endif
                            if (lane_0_1 == 24) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_219 = ((mx) > (m_c_24) ? (mx) : (m_c_24));
                                mx = _max_219;
#else
                                unsigned int _max_104 = ((mx) > (m_a_24) ? (mx) : (m_a_24));
                                mx = _max_104;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_25;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_25) : "r"(__as_u32(a0[25])));
#else
                            float _abs_106;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_106) : "f"(a0[25]));
                            float _abs_107;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_107) : "f"(a1[25]));
                            float _max_105;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_105) : "f"(_abs_106), "f"(_abs_107));
                            unsigned int m_a_25;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_25) : "r"(__as_u32(_max_105)));
#endif
                            if (lane_0_1 == 25) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_220 = ((mx) > (m_c_25) ? (mx) : (m_c_25));
                                mx = _max_220;
#else
                                unsigned int _max_108 = ((mx) > (m_a_25) ? (mx) : (m_a_25));
                                mx = _max_108;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_26;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_26) : "r"(__as_u32(a0[26])));
#else
                            float _abs_110;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_110) : "f"(a0[26]));
                            float _abs_111;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_111) : "f"(a1[26]));
                            float _max_109;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(_abs_110), "f"(_abs_111));
                            unsigned int m_a_26;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_26) : "r"(__as_u32(_max_109)));
#endif
                            if (lane_0_1 == 26) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_221 = ((mx) > (m_c_26) ? (mx) : (m_c_26));
                                mx = _max_221;
#else
                                unsigned int _max_112 = ((mx) > (m_a_26) ? (mx) : (m_a_26));
                                mx = _max_112;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_27;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_27) : "r"(__as_u32(a0[27])));
#else
                            float _abs_114;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_114) : "f"(a0[27]));
                            float _abs_115;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_115) : "f"(a1[27]));
                            float _max_113;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_113) : "f"(_abs_114), "f"(_abs_115));
                            unsigned int m_a_27;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_27) : "r"(__as_u32(_max_113)));
#endif
                            if (lane_0_1 == 27) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_222 = ((mx) > (m_c_27) ? (mx) : (m_c_27));
                                mx = _max_222;
#else
                                unsigned int _max_116 = ((mx) > (m_a_27) ? (mx) : (m_a_27));
                                mx = _max_116;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_28;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_28) : "r"(__as_u32(a0[28])));
#else
                            float _abs_118;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_118) : "f"(a0[28]));
                            float _abs_119;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(a1[28]));
                            float _max_117;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_117) : "f"(_abs_118), "f"(_abs_119));
                            unsigned int m_a_28;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_28) : "r"(__as_u32(_max_117)));
#endif
                            if (lane_0_1 == 28) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_223 = ((mx) > (m_c_28) ? (mx) : (m_c_28));
                                mx = _max_223;
#else
                                unsigned int _max_120 = ((mx) > (m_a_28) ? (mx) : (m_a_28));
                                mx = _max_120;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_29;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_29) : "r"(__as_u32(a0[29])));
#else
                            float _abs_122;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_122) : "f"(a0[29]));
                            float _abs_123;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(a1[29]));
                            float _max_121;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(_abs_122), "f"(_abs_123));
                            unsigned int m_a_29;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_29) : "r"(__as_u32(_max_121)));
#endif
                            if (lane_0_1 == 29) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_224 = ((mx) > (m_c_29) ? (mx) : (m_c_29));
                                mx = _max_224;
#else
                                unsigned int _max_124 = ((mx) > (m_a_29) ? (mx) : (m_a_29));
                                mx = _max_124;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_30;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_30) : "r"(__as_u32(a0[30])));
#else
                            float _abs_126;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_126) : "f"(a0[30]));
                            float _abs_127;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_127) : "f"(a1[30]));
                            float _max_125;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_125) : "f"(_abs_126), "f"(_abs_127));
                            unsigned int m_a_30;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_30) : "r"(__as_u32(_max_125)));
#endif
                            if (lane_0_1 == 30) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_225 = ((mx) > (m_c_30) ? (mx) : (m_c_30));
                                mx = _max_225;
#else
                                unsigned int _max_128 = ((mx) > (m_a_30) ? (mx) : (m_a_30));
                                mx = _max_128;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_31;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_31) : "r"(__as_u32(a0[31])));
#else
                            float _abs_130;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_130) : "f"(a0[31]));
                            float _abs_131;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(a1[31]));
                            float _max_129;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_129) : "f"(_abs_130), "f"(_abs_131));
                            unsigned int m_a_31;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_31) : "r"(__as_u32(_max_129)));
#endif
                            if (lane_0_1 == 31) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_226 = ((mx) > (m_c_31) ? (mx) : (m_c_31));
                                mx = _max_226;
#else
                                unsigned int _max_132 = ((mx) > (m_a_31) ? (mx) : (m_a_31));
                                mx = _max_132;
#endif
                            }
                        }
                        mx = mx;
                    }
                    n_row = x_row;
                    if (x_row >= 0) {
                        n_e = e_dkv[(long long)x_row];
                    }
                    int _parity_dkv_b_full = (int)((unsigned int)(tord_2 + j_2) % 2);
                    mbarrier_wait(dkv_b_full_addr, _parity_dkv_b_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1000
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
#else
                    if (vmask != 0) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.abs.NaN.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(a0[0]), "=f"(a0[1]), "=f"(a0[2]), "=f"(a0[3]), "=f"(a0[4]), "=f"(a0[5]), "=f"(a0[6]), "=f"(a0[7]), "=f"(a0[8]), "=f"(a0[9]), "=f"(a0[10]), "=f"(a0[11]), "=f"(a0[12]), "=f"(a0[13]), "=f"(a0[14]), "=f"(a0[15]), "=f"(a0[16]), "=f"(a0[17]), "=f"(a0[18]), "=f"(a0[19]), "=f"(a0[20]), "=f"(a0[21]), "=f"(a0[22]), "=f"(a0[23]), "=f"(a0[24]), "=f"(a0[25]), "=f"(a0[26]), "=f"(a0[27]), "=f"(a0[28]), "=f"(a0[29]), "=f"(a0[30]), "=f"(a0[31]), "=f"(ra0)
                            : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.abs.NaN.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(a1[0]), "=f"(a1[1]), "=f"(a1[2]), "=f"(a1[3]), "=f"(a1[4]), "=f"(a1[5]), "=f"(a1[6]), "=f"(a1[7]), "=f"(a1[8]), "=f"(a1[9]), "=f"(a1[10]), "=f"(a1[11]), "=f"(a1[12]), "=f"(a1[13]), "=f"(a1[14]), "=f"(a1[15]), "=f"(a1[16]), "=f"(a1[17]), "=f"(a1[18]), "=f"(a1[19]), "=f"(a1[20]), "=f"(a1[21]), "=f"(a1[22]), "=f"(a1[23]), "=f"(a1[24]), "=f"(a1[25]), "=f"(a1[26]), "=f"(a1[27]), "=f"(a1[28]), "=f"(a1[29]), "=f"(a1[30]), "=f"(a1[31]), "=f"(ra1)
                            : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                    }
#endif
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkv_b_drained_addr);
                    if (vmask != 0) {
                        float lm_1 = 0.0f;
                        unsigned int t4_1[4];
                        int rc_32 = 0;
                        float scale_c_32 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_227 = __shfl_sync(0xFFFFFFFF, roff, 0);
                        rc_32 = _shfl_sync_227;
                        float _shfl_sync_228 = __shfl_sync(0xFFFFFFFF, s_dkv, 0);
                        scale_c_32 = _shfl_sync_228;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 0) : "memory");
                        rc_32 = (int)t4_1[0];
                        scale_c_32 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_32 = (long long)(rc_32 + lat_col);
                        float y0_32 = a0[0] * scale_c_32;
                        float y1_32 = a1[0] * scale_c_32;
                        int n0_32 = __float2int_rn(y0_32);
                        int n1_32 = __float2int_rn(y1_32);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_32 + 256ll))), "r"(static_cast<int>(n0_32)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_32 + 384ll))), "r"(static_cast<int>(n1_32)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_230;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_230) : "f"(y0_32));
                        float _abs_231;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_231) : "f"(y1_32));
                        float _max_229;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_229) : "f"(_abs_230), "f"(_abs_231));
                        a0[0] = _max_229;
                        float _max_232;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_232) : "f"(lm_1), "f"(a0[0]));
                        lm_1 = _max_232;
#else
#endif
                        int rc_33 = 0;
                        float scale_c_33 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_233 = __shfl_sync(0xFFFFFFFF, roff, 1);
                        rc_33 = _shfl_sync_233;
                        float _shfl_sync_234 = __shfl_sync(0xFFFFFFFF, s_dkv, 1);
                        scale_c_33 = _shfl_sync_234;
#else
                        rc_33 = (int)t4_1[2];
                        scale_c_33 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_33 = (long long)(rc_33 + lat_col);
                        float y0_33 = a0[1] * scale_c_33;
                        float y1_33 = a1[1] * scale_c_33;
                        int n0_33 = __float2int_rn(y0_33);
                        int n1_33 = __float2int_rn(y1_33);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_33 + 256ll))), "r"(static_cast<int>(n0_33)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_33 + 384ll))), "r"(static_cast<int>(n1_33)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_236;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(y0_33));
                        float _abs_237;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_237) : "f"(y1_33));
                        float _max_235;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_235) : "f"(_abs_236), "f"(_abs_237));
                        a0[1] = _max_235;
                        float _max_238;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_238) : "f"(lm_1), "f"(a0[1]));
                        lm_1 = _max_238;
#else
#endif
                        int rc_34 = 0;
                        float scale_c_34 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_239 = __shfl_sync(0xFFFFFFFF, roff, 2);
                        rc_34 = _shfl_sync_239;
                        float _shfl_sync_240 = __shfl_sync(0xFFFFFFFF, s_dkv, 2);
                        scale_c_34 = _shfl_sync_240;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 16) : "memory");
                        rc_34 = (int)t4_1[0];
                        scale_c_34 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_34 = (long long)(rc_34 + lat_col);
                        float y0_34 = a0[2] * scale_c_34;
                        float y1_34 = a1[2] * scale_c_34;
                        int n0_34 = __float2int_rn(y0_34);
                        int n1_34 = __float2int_rn(y1_34);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_34 + 256ll))), "r"(static_cast<int>(n0_34)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_34 + 384ll))), "r"(static_cast<int>(n1_34)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_242;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_242) : "f"(y0_34));
                        float _abs_243;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_243) : "f"(y1_34));
                        float _max_241;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_241) : "f"(_abs_242), "f"(_abs_243));
                        a0[2] = _max_241;
                        float _max_244;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_244) : "f"(lm_1), "f"(a0[2]));
                        lm_1 = _max_244;
#else
#endif
                        int rc_35 = 0;
                        float scale_c_35 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_245 = __shfl_sync(0xFFFFFFFF, roff, 3);
                        rc_35 = _shfl_sync_245;
                        float _shfl_sync_246 = __shfl_sync(0xFFFFFFFF, s_dkv, 3);
                        scale_c_35 = _shfl_sync_246;
#else
                        rc_35 = (int)t4_1[2];
                        scale_c_35 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_35 = (long long)(rc_35 + lat_col);
                        float y0_35 = a0[3] * scale_c_35;
                        float y1_35 = a1[3] * scale_c_35;
                        int n0_35 = __float2int_rn(y0_35);
                        int n1_35 = __float2int_rn(y1_35);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_35 + 256ll))), "r"(static_cast<int>(n0_35)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_35 + 384ll))), "r"(static_cast<int>(n1_35)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_248;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_248) : "f"(y0_35));
                        float _abs_249;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_249) : "f"(y1_35));
                        float _max_247;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_247) : "f"(_abs_248), "f"(_abs_249));
                        a0[3] = _max_247;
                        float _max_250;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_250) : "f"(lm_1), "f"(a0[3]));
                        lm_1 = _max_250;
#else
#endif
                        int rc_36 = 0;
                        float scale_c_36 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_251 = __shfl_sync(0xFFFFFFFF, roff, 4);
                        rc_36 = _shfl_sync_251;
                        float _shfl_sync_252 = __shfl_sync(0xFFFFFFFF, s_dkv, 4);
                        scale_c_36 = _shfl_sync_252;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 32) : "memory");
                        rc_36 = (int)t4_1[0];
                        scale_c_36 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_36 = (long long)(rc_36 + lat_col);
                        float y0_36 = a0[4] * scale_c_36;
                        float y1_36 = a1[4] * scale_c_36;
                        int n0_36 = __float2int_rn(y0_36);
                        int n1_36 = __float2int_rn(y1_36);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_36 + 256ll))), "r"(static_cast<int>(n0_36)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_36 + 384ll))), "r"(static_cast<int>(n1_36)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_254;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_254) : "f"(y0_36));
                        float _abs_255;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_255) : "f"(y1_36));
                        float _max_253;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_253) : "f"(_abs_254), "f"(_abs_255));
                        a0[4] = _max_253;
                        float _max_256;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_256) : "f"(lm_1), "f"(a0[4]));
                        lm_1 = _max_256;
#else
#endif
                        int rc_37 = 0;
                        float scale_c_37 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_257 = __shfl_sync(0xFFFFFFFF, roff, 5);
                        rc_37 = _shfl_sync_257;
                        float _shfl_sync_258 = __shfl_sync(0xFFFFFFFF, s_dkv, 5);
                        scale_c_37 = _shfl_sync_258;
#else
                        rc_37 = (int)t4_1[2];
                        scale_c_37 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_37 = (long long)(rc_37 + lat_col);
                        float y0_37 = a0[5] * scale_c_37;
                        float y1_37 = a1[5] * scale_c_37;
                        int n0_37 = __float2int_rn(y0_37);
                        int n1_37 = __float2int_rn(y1_37);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_37 + 256ll))), "r"(static_cast<int>(n0_37)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_37 + 384ll))), "r"(static_cast<int>(n1_37)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_260;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_260) : "f"(y0_37));
                        float _abs_261;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(y1_37));
                        float _max_259;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_259) : "f"(_abs_260), "f"(_abs_261));
                        a0[5] = _max_259;
                        float _max_262;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_262) : "f"(lm_1), "f"(a0[5]));
                        lm_1 = _max_262;
#else
#endif
                        int rc_38 = 0;
                        float scale_c_38 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_263 = __shfl_sync(0xFFFFFFFF, roff, 6);
                        rc_38 = _shfl_sync_263;
                        float _shfl_sync_264 = __shfl_sync(0xFFFFFFFF, s_dkv, 6);
                        scale_c_38 = _shfl_sync_264;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 48) : "memory");
                        rc_38 = (int)t4_1[0];
                        scale_c_38 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_38 = (long long)(rc_38 + lat_col);
                        float y0_38 = a0[6] * scale_c_38;
                        float y1_38 = a1[6] * scale_c_38;
                        int n0_38 = __float2int_rn(y0_38);
                        int n1_38 = __float2int_rn(y1_38);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_38 + 256ll))), "r"(static_cast<int>(n0_38)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_38 + 384ll))), "r"(static_cast<int>(n1_38)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_266;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_266) : "f"(y0_38));
                        float _abs_267;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_267) : "f"(y1_38));
                        float _max_265;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_265) : "f"(_abs_266), "f"(_abs_267));
                        a0[6] = _max_265;
                        float _max_268;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_268) : "f"(lm_1), "f"(a0[6]));
                        lm_1 = _max_268;
#else
#endif
                        int rc_39 = 0;
                        float scale_c_39 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_269 = __shfl_sync(0xFFFFFFFF, roff, 7);
                        rc_39 = _shfl_sync_269;
                        float _shfl_sync_270 = __shfl_sync(0xFFFFFFFF, s_dkv, 7);
                        scale_c_39 = _shfl_sync_270;
#else
                        rc_39 = (int)t4_1[2];
                        scale_c_39 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_39 = (long long)(rc_39 + lat_col);
                        float y0_39 = a0[7] * scale_c_39;
                        float y1_39 = a1[7] * scale_c_39;
                        int n0_39 = __float2int_rn(y0_39);
                        int n1_39 = __float2int_rn(y1_39);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_39 + 256ll))), "r"(static_cast<int>(n0_39)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_39 + 384ll))), "r"(static_cast<int>(n1_39)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_272;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_272) : "f"(y0_39));
                        float _abs_273;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_273) : "f"(y1_39));
                        float _max_271;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_271) : "f"(_abs_272), "f"(_abs_273));
                        a0[7] = _max_271;
                        float _max_274;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_274) : "f"(lm_1), "f"(a0[7]));
                        lm_1 = _max_274;
#else
#endif
                        int rc_40 = 0;
                        float scale_c_40 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_275 = __shfl_sync(0xFFFFFFFF, roff, 8);
                        rc_40 = _shfl_sync_275;
                        float _shfl_sync_276 = __shfl_sync(0xFFFFFFFF, s_dkv, 8);
                        scale_c_40 = _shfl_sync_276;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 64) : "memory");
                        rc_40 = (int)t4_1[0];
                        scale_c_40 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_40 = (long long)(rc_40 + lat_col);
                        float y0_40 = a0[8] * scale_c_40;
                        float y1_40 = a1[8] * scale_c_40;
                        int n0_40 = __float2int_rn(y0_40);
                        int n1_40 = __float2int_rn(y1_40);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_40 + 256ll))), "r"(static_cast<int>(n0_40)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_40 + 384ll))), "r"(static_cast<int>(n1_40)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_278;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_278) : "f"(y0_40));
                        float _abs_279;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_279) : "f"(y1_40));
                        float _max_277;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_277) : "f"(_abs_278), "f"(_abs_279));
                        a0[8] = _max_277;
                        float _max_280;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_280) : "f"(lm_1), "f"(a0[8]));
                        lm_1 = _max_280;
#else
#endif
                        int rc_41 = 0;
                        float scale_c_41 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_281 = __shfl_sync(0xFFFFFFFF, roff, 9);
                        rc_41 = _shfl_sync_281;
                        float _shfl_sync_282 = __shfl_sync(0xFFFFFFFF, s_dkv, 9);
                        scale_c_41 = _shfl_sync_282;
#else
                        rc_41 = (int)t4_1[2];
                        scale_c_41 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_41 = (long long)(rc_41 + lat_col);
                        float y0_41 = a0[9] * scale_c_41;
                        float y1_41 = a1[9] * scale_c_41;
                        int n0_41 = __float2int_rn(y0_41);
                        int n1_41 = __float2int_rn(y1_41);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_41 + 256ll))), "r"(static_cast<int>(n0_41)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_41 + 384ll))), "r"(static_cast<int>(n1_41)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_284;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_284) : "f"(y0_41));
                        float _abs_285;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_285) : "f"(y1_41));
                        float _max_283;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_283) : "f"(_abs_284), "f"(_abs_285));
                        a0[9] = _max_283;
                        float _max_286;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_286) : "f"(lm_1), "f"(a0[9]));
                        lm_1 = _max_286;
#else
#endif
                        int rc_42 = 0;
                        float scale_c_42 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_287 = __shfl_sync(0xFFFFFFFF, roff, 10);
                        rc_42 = _shfl_sync_287;
                        float _shfl_sync_288 = __shfl_sync(0xFFFFFFFF, s_dkv, 10);
                        scale_c_42 = _shfl_sync_288;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 80) : "memory");
                        rc_42 = (int)t4_1[0];
                        scale_c_42 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_42 = (long long)(rc_42 + lat_col);
                        float y0_42 = a0[10] * scale_c_42;
                        float y1_42 = a1[10] * scale_c_42;
                        int n0_42 = __float2int_rn(y0_42);
                        int n1_42 = __float2int_rn(y1_42);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_42 + 256ll))), "r"(static_cast<int>(n0_42)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_42 + 384ll))), "r"(static_cast<int>(n1_42)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_290;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_290) : "f"(y0_42));
                        float _abs_291;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_291) : "f"(y1_42));
                        float _max_289;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_289) : "f"(_abs_290), "f"(_abs_291));
                        a0[10] = _max_289;
                        float _max_292;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_292) : "f"(lm_1), "f"(a0[10]));
                        lm_1 = _max_292;
#else
#endif
                        int rc_43 = 0;
                        float scale_c_43 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_293 = __shfl_sync(0xFFFFFFFF, roff, 11);
                        rc_43 = _shfl_sync_293;
                        float _shfl_sync_294 = __shfl_sync(0xFFFFFFFF, s_dkv, 11);
                        scale_c_43 = _shfl_sync_294;
#else
                        rc_43 = (int)t4_1[2];
                        scale_c_43 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_43 = (long long)(rc_43 + lat_col);
                        float y0_43 = a0[11] * scale_c_43;
                        float y1_43 = a1[11] * scale_c_43;
                        int n0_43 = __float2int_rn(y0_43);
                        int n1_43 = __float2int_rn(y1_43);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_43 + 256ll))), "r"(static_cast<int>(n0_43)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_43 + 384ll))), "r"(static_cast<int>(n1_43)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_296;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_296) : "f"(y0_43));
                        float _abs_297;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_297) : "f"(y1_43));
                        float _max_295;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_295) : "f"(_abs_296), "f"(_abs_297));
                        a0[11] = _max_295;
                        float _max_298;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_298) : "f"(lm_1), "f"(a0[11]));
                        lm_1 = _max_298;
#else
#endif
                        int rc_44 = 0;
                        float scale_c_44 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_299 = __shfl_sync(0xFFFFFFFF, roff, 12);
                        rc_44 = _shfl_sync_299;
                        float _shfl_sync_300 = __shfl_sync(0xFFFFFFFF, s_dkv, 12);
                        scale_c_44 = _shfl_sync_300;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 96) : "memory");
                        rc_44 = (int)t4_1[0];
                        scale_c_44 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_44 = (long long)(rc_44 + lat_col);
                        float y0_44 = a0[12] * scale_c_44;
                        float y1_44 = a1[12] * scale_c_44;
                        int n0_44 = __float2int_rn(y0_44);
                        int n1_44 = __float2int_rn(y1_44);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_44 + 256ll))), "r"(static_cast<int>(n0_44)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_44 + 384ll))), "r"(static_cast<int>(n1_44)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_302;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_302) : "f"(y0_44));
                        float _abs_303;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_303) : "f"(y1_44));
                        float _max_301;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_301) : "f"(_abs_302), "f"(_abs_303));
                        a0[12] = _max_301;
                        float _max_304;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_304) : "f"(lm_1), "f"(a0[12]));
                        lm_1 = _max_304;
#else
#endif
                        int rc_45 = 0;
                        float scale_c_45 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_305 = __shfl_sync(0xFFFFFFFF, roff, 13);
                        rc_45 = _shfl_sync_305;
                        float _shfl_sync_306 = __shfl_sync(0xFFFFFFFF, s_dkv, 13);
                        scale_c_45 = _shfl_sync_306;
#else
                        rc_45 = (int)t4_1[2];
                        scale_c_45 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_45 = (long long)(rc_45 + lat_col);
                        float y0_45 = a0[13] * scale_c_45;
                        float y1_45 = a1[13] * scale_c_45;
                        int n0_45 = __float2int_rn(y0_45);
                        int n1_45 = __float2int_rn(y1_45);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_45 + 256ll))), "r"(static_cast<int>(n0_45)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_45 + 384ll))), "r"(static_cast<int>(n1_45)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_308;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_308) : "f"(y0_45));
                        float _abs_309;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_309) : "f"(y1_45));
                        float _max_307;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_307) : "f"(_abs_308), "f"(_abs_309));
                        a0[13] = _max_307;
                        float _max_310;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_310) : "f"(lm_1), "f"(a0[13]));
                        lm_1 = _max_310;
#else
#endif
                        int rc_46 = 0;
                        float scale_c_46 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_311 = __shfl_sync(0xFFFFFFFF, roff, 14);
                        rc_46 = _shfl_sync_311;
                        float _shfl_sync_312 = __shfl_sync(0xFFFFFFFF, s_dkv, 14);
                        scale_c_46 = _shfl_sync_312;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 112) : "memory");
                        rc_46 = (int)t4_1[0];
                        scale_c_46 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_46 = (long long)(rc_46 + lat_col);
                        float y0_46 = a0[14] * scale_c_46;
                        float y1_46 = a1[14] * scale_c_46;
                        int n0_46 = __float2int_rn(y0_46);
                        int n1_46 = __float2int_rn(y1_46);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_46 + 256ll))), "r"(static_cast<int>(n0_46)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_46 + 384ll))), "r"(static_cast<int>(n1_46)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_314;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_314) : "f"(y0_46));
                        float _abs_315;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_315) : "f"(y1_46));
                        float _max_313;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_313) : "f"(_abs_314), "f"(_abs_315));
                        a0[14] = _max_313;
                        float _max_316;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_316) : "f"(lm_1), "f"(a0[14]));
                        lm_1 = _max_316;
#else
#endif
                        int rc_47 = 0;
                        float scale_c_47 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_317 = __shfl_sync(0xFFFFFFFF, roff, 15);
                        rc_47 = _shfl_sync_317;
                        float _shfl_sync_318 = __shfl_sync(0xFFFFFFFF, s_dkv, 15);
                        scale_c_47 = _shfl_sync_318;
#else
                        rc_47 = (int)t4_1[2];
                        scale_c_47 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_47 = (long long)(rc_47 + lat_col);
                        float y0_47 = a0[15] * scale_c_47;
                        float y1_47 = a1[15] * scale_c_47;
                        int n0_47 = __float2int_rn(y0_47);
                        int n1_47 = __float2int_rn(y1_47);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_47 + 256ll))), "r"(static_cast<int>(n0_47)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_47 + 384ll))), "r"(static_cast<int>(n1_47)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_320;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_320) : "f"(y0_47));
                        float _abs_321;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_321) : "f"(y1_47));
                        float _max_319;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_319) : "f"(_abs_320), "f"(_abs_321));
                        a0[15] = _max_319;
                        float _max_322;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_322) : "f"(lm_1), "f"(a0[15]));
                        lm_1 = _max_322;
#else
#endif
                        int rc_48 = 0;
                        float scale_c_48 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_323 = __shfl_sync(0xFFFFFFFF, roff, 16);
                        rc_48 = _shfl_sync_323;
                        float _shfl_sync_324 = __shfl_sync(0xFFFFFFFF, s_dkv, 16);
                        scale_c_48 = _shfl_sync_324;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 128) : "memory");
                        rc_48 = (int)t4_1[0];
                        scale_c_48 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_48 = (long long)(rc_48 + lat_col);
                        float y0_48 = a0[16] * scale_c_48;
                        float y1_48 = a1[16] * scale_c_48;
                        int n0_48 = __float2int_rn(y0_48);
                        int n1_48 = __float2int_rn(y1_48);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_48 + 256ll))), "r"(static_cast<int>(n0_48)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_48 + 384ll))), "r"(static_cast<int>(n1_48)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_326;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_326) : "f"(y0_48));
                        float _abs_327;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_327) : "f"(y1_48));
                        float _max_325;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_325) : "f"(_abs_326), "f"(_abs_327));
                        a0[16] = _max_325;
                        float _max_328;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_328) : "f"(lm_1), "f"(a0[16]));
                        lm_1 = _max_328;
#else
#endif
                        int rc_49 = 0;
                        float scale_c_49 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_329 = __shfl_sync(0xFFFFFFFF, roff, 17);
                        rc_49 = _shfl_sync_329;
                        float _shfl_sync_330 = __shfl_sync(0xFFFFFFFF, s_dkv, 17);
                        scale_c_49 = _shfl_sync_330;
#else
                        rc_49 = (int)t4_1[2];
                        scale_c_49 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_49 = (long long)(rc_49 + lat_col);
                        float y0_49 = a0[17] * scale_c_49;
                        float y1_49 = a1[17] * scale_c_49;
                        int n0_49 = __float2int_rn(y0_49);
                        int n1_49 = __float2int_rn(y1_49);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_49 + 256ll))), "r"(static_cast<int>(n0_49)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_49 + 384ll))), "r"(static_cast<int>(n1_49)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_332;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_332) : "f"(y0_49));
                        float _abs_333;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_333) : "f"(y1_49));
                        float _max_331;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_331) : "f"(_abs_332), "f"(_abs_333));
                        a0[17] = _max_331;
                        float _max_334;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_334) : "f"(lm_1), "f"(a0[17]));
                        lm_1 = _max_334;
#else
#endif
                        int rc_50 = 0;
                        float scale_c_50 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_335 = __shfl_sync(0xFFFFFFFF, roff, 18);
                        rc_50 = _shfl_sync_335;
                        float _shfl_sync_336 = __shfl_sync(0xFFFFFFFF, s_dkv, 18);
                        scale_c_50 = _shfl_sync_336;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 144) : "memory");
                        rc_50 = (int)t4_1[0];
                        scale_c_50 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_50 = (long long)(rc_50 + lat_col);
                        float y0_50 = a0[18] * scale_c_50;
                        float y1_50 = a1[18] * scale_c_50;
                        int n0_50 = __float2int_rn(y0_50);
                        int n1_50 = __float2int_rn(y1_50);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_50 + 256ll))), "r"(static_cast<int>(n0_50)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_50 + 384ll))), "r"(static_cast<int>(n1_50)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_338;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_338) : "f"(y0_50));
                        float _abs_339;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_339) : "f"(y1_50));
                        float _max_337;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_337) : "f"(_abs_338), "f"(_abs_339));
                        a0[18] = _max_337;
                        float _max_340;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_340) : "f"(lm_1), "f"(a0[18]));
                        lm_1 = _max_340;
#else
#endif
                        int rc_51 = 0;
                        float scale_c_51 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_341 = __shfl_sync(0xFFFFFFFF, roff, 19);
                        rc_51 = _shfl_sync_341;
                        float _shfl_sync_342 = __shfl_sync(0xFFFFFFFF, s_dkv, 19);
                        scale_c_51 = _shfl_sync_342;
#else
                        rc_51 = (int)t4_1[2];
                        scale_c_51 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_51 = (long long)(rc_51 + lat_col);
                        float y0_51 = a0[19] * scale_c_51;
                        float y1_51 = a1[19] * scale_c_51;
                        int n0_51 = __float2int_rn(y0_51);
                        int n1_51 = __float2int_rn(y1_51);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_51 + 256ll))), "r"(static_cast<int>(n0_51)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_51 + 384ll))), "r"(static_cast<int>(n1_51)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_344;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_344) : "f"(y0_51));
                        float _abs_345;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_345) : "f"(y1_51));
                        float _max_343;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_343) : "f"(_abs_344), "f"(_abs_345));
                        a0[19] = _max_343;
                        float _max_346;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_346) : "f"(lm_1), "f"(a0[19]));
                        lm_1 = _max_346;
#else
#endif
                        int rc_52 = 0;
                        float scale_c_52 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_347 = __shfl_sync(0xFFFFFFFF, roff, 20);
                        rc_52 = _shfl_sync_347;
                        float _shfl_sync_348 = __shfl_sync(0xFFFFFFFF, s_dkv, 20);
                        scale_c_52 = _shfl_sync_348;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 160) : "memory");
                        rc_52 = (int)t4_1[0];
                        scale_c_52 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_52 = (long long)(rc_52 + lat_col);
                        float y0_52 = a0[20] * scale_c_52;
                        float y1_52 = a1[20] * scale_c_52;
                        int n0_52 = __float2int_rn(y0_52);
                        int n1_52 = __float2int_rn(y1_52);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_52 + 256ll))), "r"(static_cast<int>(n0_52)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_52 + 384ll))), "r"(static_cast<int>(n1_52)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_350;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_350) : "f"(y0_52));
                        float _abs_351;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_351) : "f"(y1_52));
                        float _max_349;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_349) : "f"(_abs_350), "f"(_abs_351));
                        a0[20] = _max_349;
                        float _max_352;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_352) : "f"(lm_1), "f"(a0[20]));
                        lm_1 = _max_352;
#else
#endif
                        int rc_53 = 0;
                        float scale_c_53 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_353 = __shfl_sync(0xFFFFFFFF, roff, 21);
                        rc_53 = _shfl_sync_353;
                        float _shfl_sync_354 = __shfl_sync(0xFFFFFFFF, s_dkv, 21);
                        scale_c_53 = _shfl_sync_354;
#else
                        rc_53 = (int)t4_1[2];
                        scale_c_53 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_53 = (long long)(rc_53 + lat_col);
                        float y0_53 = a0[21] * scale_c_53;
                        float y1_53 = a1[21] * scale_c_53;
                        int n0_53 = __float2int_rn(y0_53);
                        int n1_53 = __float2int_rn(y1_53);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_53 + 256ll))), "r"(static_cast<int>(n0_53)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_53 + 384ll))), "r"(static_cast<int>(n1_53)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_356;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_356) : "f"(y0_53));
                        float _abs_357;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_357) : "f"(y1_53));
                        float _max_355;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_355) : "f"(_abs_356), "f"(_abs_357));
                        a0[21] = _max_355;
                        float _max_358;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_358) : "f"(lm_1), "f"(a0[21]));
                        lm_1 = _max_358;
#else
#endif
                        int rc_54 = 0;
                        float scale_c_54 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_359 = __shfl_sync(0xFFFFFFFF, roff, 22);
                        rc_54 = _shfl_sync_359;
                        float _shfl_sync_360 = __shfl_sync(0xFFFFFFFF, s_dkv, 22);
                        scale_c_54 = _shfl_sync_360;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 176) : "memory");
                        rc_54 = (int)t4_1[0];
                        scale_c_54 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_54 = (long long)(rc_54 + lat_col);
                        float y0_54 = a0[22] * scale_c_54;
                        float y1_54 = a1[22] * scale_c_54;
                        int n0_54 = __float2int_rn(y0_54);
                        int n1_54 = __float2int_rn(y1_54);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_54 + 256ll))), "r"(static_cast<int>(n0_54)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_54 + 384ll))), "r"(static_cast<int>(n1_54)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_362;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_362) : "f"(y0_54));
                        float _abs_363;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_363) : "f"(y1_54));
                        float _max_361;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_361) : "f"(_abs_362), "f"(_abs_363));
                        a0[22] = _max_361;
                        float _max_364;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_364) : "f"(lm_1), "f"(a0[22]));
                        lm_1 = _max_364;
#else
#endif
                        int rc_55 = 0;
                        float scale_c_55 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_365 = __shfl_sync(0xFFFFFFFF, roff, 23);
                        rc_55 = _shfl_sync_365;
                        float _shfl_sync_366 = __shfl_sync(0xFFFFFFFF, s_dkv, 23);
                        scale_c_55 = _shfl_sync_366;
#else
                        rc_55 = (int)t4_1[2];
                        scale_c_55 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_55 = (long long)(rc_55 + lat_col);
                        float y0_55 = a0[23] * scale_c_55;
                        float y1_55 = a1[23] * scale_c_55;
                        int n0_55 = __float2int_rn(y0_55);
                        int n1_55 = __float2int_rn(y1_55);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_55 + 256ll))), "r"(static_cast<int>(n0_55)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_55 + 384ll))), "r"(static_cast<int>(n1_55)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_368;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_368) : "f"(y0_55));
                        float _abs_369;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_369) : "f"(y1_55));
                        float _max_367;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_367) : "f"(_abs_368), "f"(_abs_369));
                        a0[23] = _max_367;
                        float _max_370;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_370) : "f"(lm_1), "f"(a0[23]));
                        lm_1 = _max_370;
#else
#endif
                        int rc_56 = 0;
                        float scale_c_56 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_371 = __shfl_sync(0xFFFFFFFF, roff, 24);
                        rc_56 = _shfl_sync_371;
                        float _shfl_sync_372 = __shfl_sync(0xFFFFFFFF, s_dkv, 24);
                        scale_c_56 = _shfl_sync_372;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 192) : "memory");
                        rc_56 = (int)t4_1[0];
                        scale_c_56 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_56 = (long long)(rc_56 + lat_col);
                        float y0_56 = a0[24] * scale_c_56;
                        float y1_56 = a1[24] * scale_c_56;
                        int n0_56 = __float2int_rn(y0_56);
                        int n1_56 = __float2int_rn(y1_56);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_56 + 256ll))), "r"(static_cast<int>(n0_56)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_56 + 384ll))), "r"(static_cast<int>(n1_56)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_374;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_374) : "f"(y0_56));
                        float _abs_375;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_375) : "f"(y1_56));
                        float _max_373;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_373) : "f"(_abs_374), "f"(_abs_375));
                        a0[24] = _max_373;
                        float _max_376;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_376) : "f"(lm_1), "f"(a0[24]));
                        lm_1 = _max_376;
#else
#endif
                        int rc_57 = 0;
                        float scale_c_57 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_377 = __shfl_sync(0xFFFFFFFF, roff, 25);
                        rc_57 = _shfl_sync_377;
                        float _shfl_sync_378 = __shfl_sync(0xFFFFFFFF, s_dkv, 25);
                        scale_c_57 = _shfl_sync_378;
#else
                        rc_57 = (int)t4_1[2];
                        scale_c_57 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_57 = (long long)(rc_57 + lat_col);
                        float y0_57 = a0[25] * scale_c_57;
                        float y1_57 = a1[25] * scale_c_57;
                        int n0_57 = __float2int_rn(y0_57);
                        int n1_57 = __float2int_rn(y1_57);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_57 + 256ll))), "r"(static_cast<int>(n0_57)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_57 + 384ll))), "r"(static_cast<int>(n1_57)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_380;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_380) : "f"(y0_57));
                        float _abs_381;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_381) : "f"(y1_57));
                        float _max_379;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_379) : "f"(_abs_380), "f"(_abs_381));
                        a0[25] = _max_379;
                        float _max_382;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_382) : "f"(lm_1), "f"(a0[25]));
                        lm_1 = _max_382;
#else
#endif
                        int rc_58 = 0;
                        float scale_c_58 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_383 = __shfl_sync(0xFFFFFFFF, roff, 26);
                        rc_58 = _shfl_sync_383;
                        float _shfl_sync_384 = __shfl_sync(0xFFFFFFFF, s_dkv, 26);
                        scale_c_58 = _shfl_sync_384;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 208) : "memory");
                        rc_58 = (int)t4_1[0];
                        scale_c_58 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_58 = (long long)(rc_58 + lat_col);
                        float y0_58 = a0[26] * scale_c_58;
                        float y1_58 = a1[26] * scale_c_58;
                        int n0_58 = __float2int_rn(y0_58);
                        int n1_58 = __float2int_rn(y1_58);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_58 + 256ll))), "r"(static_cast<int>(n0_58)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_58 + 384ll))), "r"(static_cast<int>(n1_58)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_386;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_386) : "f"(y0_58));
                        float _abs_387;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_387) : "f"(y1_58));
                        float _max_385;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_385) : "f"(_abs_386), "f"(_abs_387));
                        a0[26] = _max_385;
                        float _max_388;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_388) : "f"(lm_1), "f"(a0[26]));
                        lm_1 = _max_388;
#else
#endif
                        int rc_59 = 0;
                        float scale_c_59 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_389 = __shfl_sync(0xFFFFFFFF, roff, 27);
                        rc_59 = _shfl_sync_389;
                        float _shfl_sync_390 = __shfl_sync(0xFFFFFFFF, s_dkv, 27);
                        scale_c_59 = _shfl_sync_390;
#else
                        rc_59 = (int)t4_1[2];
                        scale_c_59 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_59 = (long long)(rc_59 + lat_col);
                        float y0_59 = a0[27] * scale_c_59;
                        float y1_59 = a1[27] * scale_c_59;
                        int n0_59 = __float2int_rn(y0_59);
                        int n1_59 = __float2int_rn(y1_59);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_59 + 256ll))), "r"(static_cast<int>(n0_59)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_59 + 384ll))), "r"(static_cast<int>(n1_59)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_392;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_392) : "f"(y0_59));
                        float _abs_393;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_393) : "f"(y1_59));
                        float _max_391;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_391) : "f"(_abs_392), "f"(_abs_393));
                        a0[27] = _max_391;
                        float _max_394;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_394) : "f"(lm_1), "f"(a0[27]));
                        lm_1 = _max_394;
#else
#endif
                        int rc_60 = 0;
                        float scale_c_60 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_395 = __shfl_sync(0xFFFFFFFF, roff, 28);
                        rc_60 = _shfl_sync_395;
                        float _shfl_sync_396 = __shfl_sync(0xFFFFFFFF, s_dkv, 28);
                        scale_c_60 = _shfl_sync_396;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 224) : "memory");
                        rc_60 = (int)t4_1[0];
                        scale_c_60 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_60 = (long long)(rc_60 + lat_col);
                        float y0_60 = a0[28] * scale_c_60;
                        float y1_60 = a1[28] * scale_c_60;
                        int n0_60 = __float2int_rn(y0_60);
                        int n1_60 = __float2int_rn(y1_60);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_60 + 256ll))), "r"(static_cast<int>(n0_60)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_60 + 384ll))), "r"(static_cast<int>(n1_60)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_398;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_398) : "f"(y0_60));
                        float _abs_399;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_399) : "f"(y1_60));
                        float _max_397;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_397) : "f"(_abs_398), "f"(_abs_399));
                        a0[28] = _max_397;
                        float _max_400;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_400) : "f"(lm_1), "f"(a0[28]));
                        lm_1 = _max_400;
#else
#endif
                        int rc_61 = 0;
                        float scale_c_61 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_401 = __shfl_sync(0xFFFFFFFF, roff, 29);
                        rc_61 = _shfl_sync_401;
                        float _shfl_sync_402 = __shfl_sync(0xFFFFFFFF, s_dkv, 29);
                        scale_c_61 = _shfl_sync_402;
#else
                        rc_61 = (int)t4_1[2];
                        scale_c_61 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_61 = (long long)(rc_61 + lat_col);
                        float y0_61 = a0[29] * scale_c_61;
                        float y1_61 = a1[29] * scale_c_61;
                        int n0_61 = __float2int_rn(y0_61);
                        int n1_61 = __float2int_rn(y1_61);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_61 + 256ll))), "r"(static_cast<int>(n0_61)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_61 + 384ll))), "r"(static_cast<int>(n1_61)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_404;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_404) : "f"(y0_61));
                        float _abs_405;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_405) : "f"(y1_61));
                        float _max_403;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_403) : "f"(_abs_404), "f"(_abs_405));
                        a0[29] = _max_403;
                        float _max_406;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_406) : "f"(lm_1), "f"(a0[29]));
                        lm_1 = _max_406;
#else
#endif
                        int rc_62 = 0;
                        float scale_c_62 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_407 = __shfl_sync(0xFFFFFFFF, roff, 30);
                        rc_62 = _shfl_sync_407;
                        float _shfl_sync_408 = __shfl_sync(0xFFFFFFFF, s_dkv, 30);
                        scale_c_62 = _shfl_sync_408;
#else
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t4_1[(0) + 3]))
                            : "r"(tab_w + 240) : "memory");
                        rc_62 = (int)t4_1[0];
                        scale_c_62 = __uint_as_float(t4_1[1]);
#endif
                        long long idx_c_62 = (long long)(rc_62 + lat_col);
                        float y0_62 = a0[30] * scale_c_62;
                        float y1_62 = a1[30] * scale_c_62;
                        int n0_62 = __float2int_rn(y0_62);
                        int n1_62 = __float2int_rn(y1_62);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_62 + 256ll))), "r"(static_cast<int>(n0_62)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_62 + 384ll))), "r"(static_cast<int>(n1_62)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_410;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_410) : "f"(y0_62));
                        float _abs_411;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_411) : "f"(y1_62));
                        float _max_409;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_409) : "f"(_abs_410), "f"(_abs_411));
                        a0[30] = _max_409;
                        float _max_412;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_412) : "f"(lm_1), "f"(a0[30]));
                        lm_1 = _max_412;
#else
#endif
                        int rc_63 = 0;
                        float scale_c_63 = 0.0f;
#if __CUDA_ARCH__ == 1000
                        int _shfl_sync_413 = __shfl_sync(0xFFFFFFFF, roff, 31);
                        rc_63 = _shfl_sync_413;
                        float _shfl_sync_414 = __shfl_sync(0xFFFFFFFF, s_dkv, 31);
                        scale_c_63 = _shfl_sync_414;
#else
                        rc_63 = (int)t4_1[2];
                        scale_c_63 = __uint_as_float(t4_1[3]);
#endif
                        long long idx_c_63 = (long long)(rc_63 + lat_col);
                        float y0_63 = a0[31] * scale_c_63;
                        float y1_63 = a1[31] * scale_c_63;
                        int n0_63 = __float2int_rn(y0_63);
                        int n1_63 = __float2int_rn(y1_63);
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_63 + 256ll))), "r"(static_cast<int>(n0_63)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile("red.relaxed.gpu.global.add.L2::cache_hint.s32 [%0], %1, %2;" :: "l"((reinterpret_cast<int*>(dkv_acc) + (idx_c_63 + 384ll))), "r"(static_cast<int>(n1_63)), "l"(0x14F0000000000000ULL) : "memory");
#if __CUDA_ARCH__ == 1000
                        float _abs_416;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_416) : "f"(y0_63));
                        float _abs_417;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_417) : "f"(y1_63));
                        float _max_415;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_415) : "f"(_abs_416), "f"(_abs_417));
                        a0[31] = _max_415;
                        float _max_418;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_418) : "f"(lm_1), "f"(a0[31]));
                        lm_1 = _max_418;
                        unsigned int _warp_redux_419;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_419) : "r"(__as_u32(lm_1)));
                        if (_warp_redux_419 >= thr_w) {
                            unsigned int m_c_32;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_32) : "r"(__as_u32(a0[0])));
#else
                        float _max_134;
                        asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_134) : "f"(ra0), "f"(ra1));
                        unsigned int _warp_redux_133;
                        asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_133) : "r"(__as_u32(_max_134)));
                        if (_warp_redux_133 >= thr_w) {
                            float _abs_136;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_136) : "f"(a0[0]));
                            float _abs_137;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(a1[0]));
                            float _max_135;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_135) : "f"(_abs_136), "f"(_abs_137));
                            unsigned int m_a_32;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_32) : "r"(__as_u32(_max_135)));
#endif
                            if (lane_0_1 == 0) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_420 = ((mx) > (m_c_32) ? (mx) : (m_c_32));
                                mx = _max_420;
#else
                                unsigned int _max_138 = ((mx) > (m_a_32) ? (mx) : (m_a_32));
                                mx = _max_138;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_33;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_33) : "r"(__as_u32(a0[1])));
#else
                            float _abs_140;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_140) : "f"(a0[1]));
                            float _abs_141;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_141) : "f"(a1[1]));
                            float _max_139;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_139) : "f"(_abs_140), "f"(_abs_141));
                            unsigned int m_a_33;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_33) : "r"(__as_u32(_max_139)));
#endif
                            if (lane_0_1 == 1) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_421 = ((mx) > (m_c_33) ? (mx) : (m_c_33));
                                mx = _max_421;
#else
                                unsigned int _max_142 = ((mx) > (m_a_33) ? (mx) : (m_a_33));
                                mx = _max_142;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_34;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_34) : "r"(__as_u32(a0[2])));
#else
                            float _abs_144;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(a0[2]));
                            float _abs_145;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_145) : "f"(a1[2]));
                            float _max_143;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_143) : "f"(_abs_144), "f"(_abs_145));
                            unsigned int m_a_34;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_34) : "r"(__as_u32(_max_143)));
#endif
                            if (lane_0_1 == 2) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_422 = ((mx) > (m_c_34) ? (mx) : (m_c_34));
                                mx = _max_422;
#else
                                unsigned int _max_146 = ((mx) > (m_a_34) ? (mx) : (m_a_34));
                                mx = _max_146;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_35;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_35) : "r"(__as_u32(a0[3])));
#else
                            float _abs_148;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_148) : "f"(a0[3]));
                            float _abs_149;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(a1[3]));
                            float _max_147;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_147) : "f"(_abs_148), "f"(_abs_149));
                            unsigned int m_a_35;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_35) : "r"(__as_u32(_max_147)));
#endif
                            if (lane_0_1 == 3) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_423 = ((mx) > (m_c_35) ? (mx) : (m_c_35));
                                mx = _max_423;
#else
                                unsigned int _max_150 = ((mx) > (m_a_35) ? (mx) : (m_a_35));
                                mx = _max_150;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_36;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_36) : "r"(__as_u32(a0[4])));
#else
                            float _abs_152;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(a0[4]));
                            float _abs_153;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(a1[4]));
                            float _max_151;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_151) : "f"(_abs_152), "f"(_abs_153));
                            unsigned int m_a_36;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_36) : "r"(__as_u32(_max_151)));
#endif
                            if (lane_0_1 == 4) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_424 = ((mx) > (m_c_36) ? (mx) : (m_c_36));
                                mx = _max_424;
#else
                                unsigned int _max_154 = ((mx) > (m_a_36) ? (mx) : (m_a_36));
                                mx = _max_154;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_37;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_37) : "r"(__as_u32(a0[5])));
#else
                            float _abs_156;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_156) : "f"(a0[5]));
                            float _abs_157;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_157) : "f"(a1[5]));
                            float _max_155;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_155) : "f"(_abs_156), "f"(_abs_157));
                            unsigned int m_a_37;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_37) : "r"(__as_u32(_max_155)));
#endif
                            if (lane_0_1 == 5) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_425 = ((mx) > (m_c_37) ? (mx) : (m_c_37));
                                mx = _max_425;
#else
                                unsigned int _max_158 = ((mx) > (m_a_37) ? (mx) : (m_a_37));
                                mx = _max_158;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_38;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_38) : "r"(__as_u32(a0[6])));
#else
                            float _abs_160;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_160) : "f"(a0[6]));
                            float _abs_161;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(a1[6]));
                            float _max_159;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_159) : "f"(_abs_160), "f"(_abs_161));
                            unsigned int m_a_38;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_38) : "r"(__as_u32(_max_159)));
#endif
                            if (lane_0_1 == 6) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_426 = ((mx) > (m_c_38) ? (mx) : (m_c_38));
                                mx = _max_426;
#else
                                unsigned int _max_162 = ((mx) > (m_a_38) ? (mx) : (m_a_38));
                                mx = _max_162;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_39;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_39) : "r"(__as_u32(a0[7])));
#else
                            float _abs_164;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_164) : "f"(a0[7]));
                            float _abs_165;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_165) : "f"(a1[7]));
                            float _max_163;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_163) : "f"(_abs_164), "f"(_abs_165));
                            unsigned int m_a_39;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_39) : "r"(__as_u32(_max_163)));
#endif
                            if (lane_0_1 == 7) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_427 = ((mx) > (m_c_39) ? (mx) : (m_c_39));
                                mx = _max_427;
#else
                                unsigned int _max_166 = ((mx) > (m_a_39) ? (mx) : (m_a_39));
                                mx = _max_166;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_40;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_40) : "r"(__as_u32(a0[8])));
#else
                            float _abs_168;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_168) : "f"(a0[8]));
                            float _abs_169;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_169) : "f"(a1[8]));
                            float _max_167;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_167) : "f"(_abs_168), "f"(_abs_169));
                            unsigned int m_a_40;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_40) : "r"(__as_u32(_max_167)));
#endif
                            if (lane_0_1 == 8) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_428 = ((mx) > (m_c_40) ? (mx) : (m_c_40));
                                mx = _max_428;
#else
                                unsigned int _max_170 = ((mx) > (m_a_40) ? (mx) : (m_a_40));
                                mx = _max_170;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_41;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_41) : "r"(__as_u32(a0[9])));
#else
                            float _abs_172;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_172) : "f"(a0[9]));
                            float _abs_173;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(a1[9]));
                            float _max_171;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_171) : "f"(_abs_172), "f"(_abs_173));
                            unsigned int m_a_41;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_41) : "r"(__as_u32(_max_171)));
#endif
                            if (lane_0_1 == 9) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_429 = ((mx) > (m_c_41) ? (mx) : (m_c_41));
                                mx = _max_429;
#else
                                unsigned int _max_174 = ((mx) > (m_a_41) ? (mx) : (m_a_41));
                                mx = _max_174;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_42;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_42) : "r"(__as_u32(a0[10])));
#else
                            float _abs_176;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_176) : "f"(a0[10]));
                            float _abs_177;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(a1[10]));
                            float _max_175;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_175) : "f"(_abs_176), "f"(_abs_177));
                            unsigned int m_a_42;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_42) : "r"(__as_u32(_max_175)));
#endif
                            if (lane_0_1 == 10) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_430 = ((mx) > (m_c_42) ? (mx) : (m_c_42));
                                mx = _max_430;
#else
                                unsigned int _max_178 = ((mx) > (m_a_42) ? (mx) : (m_a_42));
                                mx = _max_178;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_43;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_43) : "r"(__as_u32(a0[11])));
#else
                            float _abs_180;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_180) : "f"(a0[11]));
                            float _abs_181;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_181) : "f"(a1[11]));
                            float _max_179;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_179) : "f"(_abs_180), "f"(_abs_181));
                            unsigned int m_a_43;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_43) : "r"(__as_u32(_max_179)));
#endif
                            if (lane_0_1 == 11) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_431 = ((mx) > (m_c_43) ? (mx) : (m_c_43));
                                mx = _max_431;
#else
                                unsigned int _max_182 = ((mx) > (m_a_43) ? (mx) : (m_a_43));
                                mx = _max_182;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_44;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_44) : "r"(__as_u32(a0[12])));
#else
                            float _abs_184;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_184) : "f"(a0[12]));
                            float _abs_185;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(a1[12]));
                            float _max_183;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_183) : "f"(_abs_184), "f"(_abs_185));
                            unsigned int m_a_44;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_44) : "r"(__as_u32(_max_183)));
#endif
                            if (lane_0_1 == 12) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_432 = ((mx) > (m_c_44) ? (mx) : (m_c_44));
                                mx = _max_432;
#else
                                unsigned int _max_186 = ((mx) > (m_a_44) ? (mx) : (m_a_44));
                                mx = _max_186;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_45;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_45) : "r"(__as_u32(a0[13])));
#else
                            float _abs_188;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_188) : "f"(a0[13]));
                            float _abs_189;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_189) : "f"(a1[13]));
                            float _max_187;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_187) : "f"(_abs_188), "f"(_abs_189));
                            unsigned int m_a_45;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_45) : "r"(__as_u32(_max_187)));
#endif
                            if (lane_0_1 == 13) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_433 = ((mx) > (m_c_45) ? (mx) : (m_c_45));
                                mx = _max_433;
#else
                                unsigned int _max_190 = ((mx) > (m_a_45) ? (mx) : (m_a_45));
                                mx = _max_190;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_46;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_46) : "r"(__as_u32(a0[14])));
#else
                            float _abs_192;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_192) : "f"(a0[14]));
                            float _abs_193;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(a1[14]));
                            float _max_191;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_191) : "f"(_abs_192), "f"(_abs_193));
                            unsigned int m_a_46;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_46) : "r"(__as_u32(_max_191)));
#endif
                            if (lane_0_1 == 14) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_434 = ((mx) > (m_c_46) ? (mx) : (m_c_46));
                                mx = _max_434;
#else
                                unsigned int _max_194 = ((mx) > (m_a_46) ? (mx) : (m_a_46));
                                mx = _max_194;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_47;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_47) : "r"(__as_u32(a0[15])));
#else
                            float _abs_196;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_196) : "f"(a0[15]));
                            float _abs_197;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_197) : "f"(a1[15]));
                            float _max_195;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_195) : "f"(_abs_196), "f"(_abs_197));
                            unsigned int m_a_47;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_47) : "r"(__as_u32(_max_195)));
#endif
                            if (lane_0_1 == 15) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_435 = ((mx) > (m_c_47) ? (mx) : (m_c_47));
                                mx = _max_435;
#else
                                unsigned int _max_198 = ((mx) > (m_a_47) ? (mx) : (m_a_47));
                                mx = _max_198;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_48;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_48) : "r"(__as_u32(a0[16])));
#else
                            float _abs_200;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_200) : "f"(a0[16]));
                            float _abs_201;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(a1[16]));
                            float _max_199;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_199) : "f"(_abs_200), "f"(_abs_201));
                            unsigned int m_a_48;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_48) : "r"(__as_u32(_max_199)));
#endif
                            if (lane_0_1 == 16) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_436 = ((mx) > (m_c_48) ? (mx) : (m_c_48));
                                mx = _max_436;
#else
                                unsigned int _max_202 = ((mx) > (m_a_48) ? (mx) : (m_a_48));
                                mx = _max_202;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_49;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_49) : "r"(__as_u32(a0[17])));
#else
                            float _abs_204;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_204) : "f"(a0[17]));
                            float _abs_205;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_205) : "f"(a1[17]));
                            float _max_203;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_203) : "f"(_abs_204), "f"(_abs_205));
                            unsigned int m_a_49;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_49) : "r"(__as_u32(_max_203)));
#endif
                            if (lane_0_1 == 17) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_437 = ((mx) > (m_c_49) ? (mx) : (m_c_49));
                                mx = _max_437;
#else
                                unsigned int _max_206 = ((mx) > (m_a_49) ? (mx) : (m_a_49));
                                mx = _max_206;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_50;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_50) : "r"(__as_u32(a0[18])));
#else
                            float _abs_208;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_208) : "f"(a0[18]));
                            float _abs_209;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(a1[18]));
                            float _max_207;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_207) : "f"(_abs_208), "f"(_abs_209));
                            unsigned int m_a_50;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_50) : "r"(__as_u32(_max_207)));
#endif
                            if (lane_0_1 == 18) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_438 = ((mx) > (m_c_50) ? (mx) : (m_c_50));
                                mx = _max_438;
#else
                                unsigned int _max_210 = ((mx) > (m_a_50) ? (mx) : (m_a_50));
                                mx = _max_210;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_51;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_51) : "r"(__as_u32(a0[19])));
#else
                            float _abs_212;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_212) : "f"(a0[19]));
                            float _abs_213;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_213) : "f"(a1[19]));
                            float _max_211;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_211) : "f"(_abs_212), "f"(_abs_213));
                            unsigned int m_a_51;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_51) : "r"(__as_u32(_max_211)));
#endif
                            if (lane_0_1 == 19) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_439 = ((mx) > (m_c_51) ? (mx) : (m_c_51));
                                mx = _max_439;
#else
                                unsigned int _max_214 = ((mx) > (m_a_51) ? (mx) : (m_a_51));
                                mx = _max_214;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_52;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_52) : "r"(__as_u32(a0[20])));
#else
                            float _abs_216;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_216) : "f"(a0[20]));
                            float _abs_217;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_217) : "f"(a1[20]));
                            float _max_215;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_215) : "f"(_abs_216), "f"(_abs_217));
                            unsigned int m_a_52;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_52) : "r"(__as_u32(_max_215)));
#endif
                            if (lane_0_1 == 20) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_440 = ((mx) > (m_c_52) ? (mx) : (m_c_52));
                                mx = _max_440;
#else
                                unsigned int _max_218 = ((mx) > (m_a_52) ? (mx) : (m_a_52));
                                mx = _max_218;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_53;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_53) : "r"(__as_u32(a0[21])));
#else
                            float _abs_220;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_220) : "f"(a0[21]));
                            float _abs_221;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_221) : "f"(a1[21]));
                            float _max_219;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_219) : "f"(_abs_220), "f"(_abs_221));
                            unsigned int m_a_53;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_53) : "r"(__as_u32(_max_219)));
#endif
                            if (lane_0_1 == 21) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_441 = ((mx) > (m_c_53) ? (mx) : (m_c_53));
                                mx = _max_441;
#else
                                unsigned int _max_222 = ((mx) > (m_a_53) ? (mx) : (m_a_53));
                                mx = _max_222;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_54;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_54) : "r"(__as_u32(a0[22])));
#else
                            float _abs_224;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_224) : "f"(a0[22]));
                            float _abs_225;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_225) : "f"(a1[22]));
                            float _max_223;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_223) : "f"(_abs_224), "f"(_abs_225));
                            unsigned int m_a_54;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_54) : "r"(__as_u32(_max_223)));
#endif
                            if (lane_0_1 == 22) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_442 = ((mx) > (m_c_54) ? (mx) : (m_c_54));
                                mx = _max_442;
#else
                                unsigned int _max_226 = ((mx) > (m_a_54) ? (mx) : (m_a_54));
                                mx = _max_226;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_55;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_55) : "r"(__as_u32(a0[23])));
#else
                            float _abs_228;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_228) : "f"(a0[23]));
                            float _abs_229;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_229) : "f"(a1[23]));
                            float _max_227;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_227) : "f"(_abs_228), "f"(_abs_229));
                            unsigned int m_a_55;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_55) : "r"(__as_u32(_max_227)));
#endif
                            if (lane_0_1 == 23) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_443 = ((mx) > (m_c_55) ? (mx) : (m_c_55));
                                mx = _max_443;
#else
                                unsigned int _max_230 = ((mx) > (m_a_55) ? (mx) : (m_a_55));
                                mx = _max_230;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_56;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_56) : "r"(__as_u32(a0[24])));
#else
                            float _abs_232;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_232) : "f"(a0[24]));
                            float _abs_233;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_233) : "f"(a1[24]));
                            float _max_231;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_231) : "f"(_abs_232), "f"(_abs_233));
                            unsigned int m_a_56;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_56) : "r"(__as_u32(_max_231)));
#endif
                            if (lane_0_1 == 24) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_444 = ((mx) > (m_c_56) ? (mx) : (m_c_56));
                                mx = _max_444;
#else
                                unsigned int _max_234 = ((mx) > (m_a_56) ? (mx) : (m_a_56));
                                mx = _max_234;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_57;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_57) : "r"(__as_u32(a0[25])));
#else
                            float _abs_236;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_236) : "f"(a0[25]));
                            float _abs_237;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_237) : "f"(a1[25]));
                            float _max_235;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_235) : "f"(_abs_236), "f"(_abs_237));
                            unsigned int m_a_57;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_57) : "r"(__as_u32(_max_235)));
#endif
                            if (lane_0_1 == 25) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_445 = ((mx) > (m_c_57) ? (mx) : (m_c_57));
                                mx = _max_445;
#else
                                unsigned int _max_238 = ((mx) > (m_a_57) ? (mx) : (m_a_57));
                                mx = _max_238;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_58;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_58) : "r"(__as_u32(a0[26])));
#else
                            float _abs_240;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_240) : "f"(a0[26]));
                            float _abs_241;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_241) : "f"(a1[26]));
                            float _max_239;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_239) : "f"(_abs_240), "f"(_abs_241));
                            unsigned int m_a_58;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_58) : "r"(__as_u32(_max_239)));
#endif
                            if (lane_0_1 == 26) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_446 = ((mx) > (m_c_58) ? (mx) : (m_c_58));
                                mx = _max_446;
#else
                                unsigned int _max_242 = ((mx) > (m_a_58) ? (mx) : (m_a_58));
                                mx = _max_242;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_59;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_59) : "r"(__as_u32(a0[27])));
#else
                            float _abs_244;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_244) : "f"(a0[27]));
                            float _abs_245;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_245) : "f"(a1[27]));
                            float _max_243;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_243) : "f"(_abs_244), "f"(_abs_245));
                            unsigned int m_a_59;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_59) : "r"(__as_u32(_max_243)));
#endif
                            if (lane_0_1 == 27) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_447 = ((mx) > (m_c_59) ? (mx) : (m_c_59));
                                mx = _max_447;
#else
                                unsigned int _max_246 = ((mx) > (m_a_59) ? (mx) : (m_a_59));
                                mx = _max_246;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_60;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_60) : "r"(__as_u32(a0[28])));
#else
                            float _abs_248;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_248) : "f"(a0[28]));
                            float _abs_249;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_249) : "f"(a1[28]));
                            float _max_247;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_247) : "f"(_abs_248), "f"(_abs_249));
                            unsigned int m_a_60;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_60) : "r"(__as_u32(_max_247)));
#endif
                            if (lane_0_1 == 28) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_448 = ((mx) > (m_c_60) ? (mx) : (m_c_60));
                                mx = _max_448;
#else
                                unsigned int _max_250 = ((mx) > (m_a_60) ? (mx) : (m_a_60));
                                mx = _max_250;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_61;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_61) : "r"(__as_u32(a0[29])));
#else
                            float _abs_252;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_252) : "f"(a0[29]));
                            float _abs_253;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_253) : "f"(a1[29]));
                            float _max_251;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_251) : "f"(_abs_252), "f"(_abs_253));
                            unsigned int m_a_61;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_61) : "r"(__as_u32(_max_251)));
#endif
                            if (lane_0_1 == 29) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_449 = ((mx) > (m_c_61) ? (mx) : (m_c_61));
                                mx = _max_449;
#else
                                unsigned int _max_254 = ((mx) > (m_a_61) ? (mx) : (m_a_61));
                                mx = _max_254;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_62;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_62) : "r"(__as_u32(a0[30])));
#else
                            float _abs_256;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_256) : "f"(a0[30]));
                            float _abs_257;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_257) : "f"(a1[30]));
                            float _max_255;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_255) : "f"(_abs_256), "f"(_abs_257));
                            unsigned int m_a_62;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_62) : "r"(__as_u32(_max_255)));
#endif
                            if (lane_0_1 == 30) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_450 = ((mx) > (m_c_62) ? (mx) : (m_c_62));
                                mx = _max_450;
#else
                                unsigned int _max_258 = ((mx) > (m_a_62) ? (mx) : (m_a_62));
                                mx = _max_258;
#endif
                            }
#if __CUDA_ARCH__ == 1000
                            unsigned int m_c_63;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_c_63) : "r"(__as_u32(a0[31])));
#else
                            float _abs_260;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_260) : "f"(a0[31]));
                            float _abs_261;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_261) : "f"(a1[31]));
                            float _max_259;
                            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_259) : "f"(_abs_260), "f"(_abs_261));
                            unsigned int m_a_63;
                            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(m_a_63) : "r"(__as_u32(_max_259)));
#endif
                            if (lane_0_1 == 31) {
#if __CUDA_ARCH__ == 1000
                                unsigned int _max_451 = ((mx) > (m_c_63) ? (mx) : (m_c_63));
                                mx = _max_451;
#else
                                unsigned int _max_262 = ((mx) > (m_a_63) ? (mx) : (m_a_63));
                                mx = _max_262;
#endif
                            }
                        }
                        mx = mx;
                    }
#if __CUDA_ARCH__ == 1000
#else
                    mx = __as_u32(__uint_as_float(mx) * s_dkv);
#endif
                    if (row >= 0) {
                        if (mx >= thr) {
                            unsigned long long t_wide = 2147483648ull;
                            if (mx < 1325400064) {
                                t_wide = (unsigned long long)(unsigned int)__uint_as_float(mx) + 1ull;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(cnt64) + ((long long)row * 4ll + (long long)w2))), "l"(static_cast<unsigned long long>(t_wide)) : "memory");
                        }
                    }
                }
                tord_2 += blocks_2;
            }
        }
    }
    // ---- Warpgroup: 4 ----
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        // ---- Role: mma ----
        if (warp == 16) {
            { // mma_main
                int ntok_3 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
                int tord_3 = 0;
                #pragma unroll 1
                for (int tk_3 = 0; tk_3 < ntok_3; tk_3++) {
                    int tok_2 = blockIdx.x + tk_3 * gridDim.x;
                    int bslot_3 = tk_3 % 2;
                    int _stage_blocks_ready_3 = (int)((unsigned int)tk_3 % 2);
                    int _parity_blocks_ready_3 = (int)((unsigned int)tk_3 / 2 % 2);
                    mbarrier_wait(blocks_ready_addr + (_stage_blocks_ready_3) * 8, _parity_blocks_ready_3);
                    int blocks_3 = blocks_word[bslot_3];
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(blocks_free_addr + ((int)((unsigned int)tk_3 % 2)) * 8);
                    int _parity_qdo_full = (int)((unsigned int)tk_3 % 2);
                    mbarrier_wait(qdo_full_addr, _parity_qdo_full);
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int i = 0; i < blocks_3; i++) {
                            if (i == 0) {
                                int _parity_k_full = (int)((unsigned int)(tord_3 + i) % 2);
                                mbarrier_wait(k_full_addr, _parity_k_full);
                                if (tord_3 > 0) {
                                    int _parity_s_free = (int)((unsigned int)(tord_3 + i) % 2) ^ 1;
                                    mbarrier_wait(s_free_addr, _parity_s_free);
                                }
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
                        "}\n"
                        :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_tmem), "r"(0) : "memory");
                                tcgen05_commit(s_full_addr);
                            }
                            if (i == 0) {
                                int _parity_do_full = (int)((unsigned int)tk_3 % 2);
                                mbarrier_wait(do_full_addr, _parity_do_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            int _parity_dp_free = (int)((unsigned int)(tord_3 + i) % 2) ^ 1;
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
                            int _parity_p_full = (int)((unsigned int)(tord_3 + i) % 2);
                            mbarrier_wait(p_full_addr, _parity_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (tord_3 + i > 0) {
                                int _parity_dkv_b_drained = (int)((unsigned int)(tord_3 + i) % 2) ^ 1;
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
                            int _parity_ds_full = (int)((unsigned int)(tord_3 + i) % 2);
                            mbarrier_wait(ds_full_addr, _parity_ds_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (i == 0) {
                                int _parity_dq_drained = (int)((unsigned int)tk_3 % 2) ^ 1;
                                mbarrier_wait(dq_drained_addr, _parity_dq_drained);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
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
                        :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (192))), "r"(((i == 0) ? 0 : 1)) : "memory");
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
                        :: "r"(_smem_desc_lo_5), "r"(_mma_b_lo_1), "r"((tmem_tmem + (256))), "r"(((i == 0) ? 0 : 1)) : "memory");
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
                        :: "r"(_smem_desc_lo_6), "r"(_mma_b_lo_1), "r"((tmem_tmem + (320))), "r"(((i == 0) ? 0 : 1)) : "memory");
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
                        :: "r"(_smem_desc_lo_7), "r"(_mma_b_lo_1), "r"((tmem_tmem + (384))), "r"(((i == 0) ? 0 : 1)) : "memory");
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
                            int _smem_desc_lo_8 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (1) * 1024;
#else
                            int _smem_desc_lo_8 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                        :: "r"(_smem_desc_lo_8), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(1) : "memory");
                            tcgen05_commit(dkv_a_full_addr);
                            int _parity_dkv_a_drained = (int)((unsigned int)(tord_3 + i) % 2);
                            mbarrier_wait(dkv_a_drained_addr, _parity_dkv_a_drained);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_9 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (2) * 1024;
#else
                            int _smem_desc_lo_9 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                        :: "r"(_smem_desc_lo_9), "r"(_mma_b_lo_0), "r"((tmem_tmem + (64))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_10 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (3) * 1024;
#else
                            int _smem_desc_lo_10 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                        :: "r"(_smem_desc_lo_10), "r"(_mma_b_lo_0), "r"((tmem_tmem + (128))), "r"(0) : "memory");
                            tcgen05_commit(p_free_addr);
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_11 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (2) * 1024;
#else
                            int _smem_desc_lo_11 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                        :: "r"(_smem_desc_lo_11), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_12 = ((((q_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (3) * 1024;
#else
                            int _smem_desc_lo_12 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                        :: "r"(_smem_desc_lo_12), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(1) : "memory");
                            tcgen05_commit(dkv_b_full_addr);
                            tcgen05_commit(ds_free_addr);
                            if (blocks_3 > i + 1) {
                                int _parity_k_full_1 = (int)((unsigned int)(tord_3 + i) % 2) ^ 1;
                                mbarrier_wait(k_full_addr, _parity_k_full_1);
                                int _parity_s_free_1 = (int)((unsigned int)(tord_3 + i) % 2);
                                mbarrier_wait(s_free_addr, _parity_s_free_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                                int _smem_desc_lo_13 = ((q_k_addr) >> 4) & 0x7FFF;
                                int _smem_desc_lo_14 = ((k_k_addr) >> 4) & 0x7FFF;
#else
                                int _smem_desc_lo_13 = ((q_k_addr) >> 4) & 0x3FFF;
                                int _smem_desc_lo_14 = ((k_k_addr) >> 4) & 0x3FFF;
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
                        :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"(tmem_tmem), "r"(0) : "memory");
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
                int ntok_4 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
                int tord_4 = 0;
                #pragma unroll 1
                for (int tk_4 = 0; tk_4 < ntok_4; tk_4++) {
                    int tok_3 = blockIdx.x + tk_4 * gridDim.x;
                    int token_1 = token_base + token_step * tok_3;
                    int lane_0_2 = lane;
                    int _parity_dq_done_1 = (int)((unsigned int)tk_4 % 2) ^ 1;
                    mbarrier_wait(dq_done_addr, _parity_dq_done_1);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(q_k_addr, (&q_latent), 0, 0, 0, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 8192, (&q_latent), 0, 0, 1, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 16384, (&q_latent), 0, 0, 2, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 24576, (&q_latent), 0, 0, 3, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 32768, (&q_latent), 0, 0, 4, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 40960, (&q_latent), 0, 0, 5, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 49152, (&q_latent), 0, 0, 6, token_1, qdo_full_addr);
                        tma_4d_gmem2smem(q_k_addr + 57344, (&q_latent), 0, 0, 7, token_1, qdo_full_addr);
                        mbarrier_arrive_expect_tx(qdo_full_addr, 65536);
                    }
                    int _parity_do_free = (int)((unsigned int)tk_4 % 2) ^ 1;
                    mbarrier_wait(do_free_addr, _parity_do_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(do_k_addr, (&dout), 0, 0, 0, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 8192, (&dout), 0, 0, 1, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 16384, (&dout), 0, 0, 2, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 24576, (&dout), 0, 0, 3, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 32768, (&dout), 0, 0, 4, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 40960, (&dout), 0, 0, 5, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 49152, (&dout), 0, 0, 6, token_1, do_full_addr);
                        tma_4d_gmem2smem(do_k_addr + 57344, (&dout), 0, 0, 7, token_1, do_full_addr);
                        mbarrier_arrive_expect_tx(do_full_addr, 65536);
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
                int ntok_5 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
                int tord_5 = 0;
                #pragma unroll 1
                for (int tk_5 = 0; tk_5 < ntok_5; tk_5++) {
                    int tok_4 = blockIdx.x + tk_5 * gridDim.x;
                    int token_2 = token_base + token_step * tok_4;
                    int lane_0_3 = lane;
                    int active = topk;
                    if (has_topk_length != 0) {
                        int _max_1 = ((topk_length[token_2]) > (0) ? (topk_length[token_2]) : (0));
#if __CUDA_ARCH__ == 1000
                        int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
                        active = _min_0;
#else
                        int _min_0_1 = ((_max_1) < (topk) ? (_max_1) : (topk));
                        active = _min_0_1;
#endif
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
                                    int r = _vec_load_0[0];
                                    if (dkv_has_map != 0) {
                                        r = dkv_dst_map[(long long)_vec_load_0[0]];
                                    }
                                    m = (int)(redo_mark[(long long)r] == 1);
                                    if (m != 0) {
                                        best = pos;
                                    }
                                }
                                if (_vec_load_0[1] >= 0 && _vec_load_0[1] < num_kv && active > pos + 1) {
                                    int m_1 = 1;
                                    int r_1 = _vec_load_0[1];
                                    if (dkv_has_map != 0) {
                                        r_1 = dkv_dst_map[(long long)_vec_load_0[1]];
                                    }
                                    m_1 = (int)(redo_mark[(long long)r_1] == 1);
                                    if (m_1 != 0) {
                                        best = pos + 1;
                                    }
                                }
                                if (_vec_load_0[2] >= 0 && _vec_load_0[2] < num_kv && active > pos + 2) {
                                    int m_2 = 1;
                                    int r_2 = _vec_load_0[2];
                                    if (dkv_has_map != 0) {
                                        r_2 = dkv_dst_map[(long long)_vec_load_0[2]];
                                    }
                                    m_2 = (int)(redo_mark[(long long)r_2] == 1);
                                    if (m_2 != 0) {
                                        best = pos + 2;
                                    }
                                }
                                if (_vec_load_0[3] >= 0 && _vec_load_0[3] < num_kv && active > pos + 3) {
                                    int m_3 = 1;
                                    int r_3 = _vec_load_0[3];
                                    if (dkv_has_map != 0) {
                                        r_3 = dkv_dst_map[(long long)_vec_load_0[3]];
                                    }
                                    m_3 = (int)(redo_mark[(long long)r_3] == 1);
                                    if (m_3 != 0) {
                                        best = pos + 3;
                                    }
                                }
                            } else {
                                if (active > pos) {
                                    int sv_1_0 = indices[row_base + (long long)pos];
                                    if (sv_1_0 >= 0 && sv_1_0 < num_kv) {
                                        int m_4 = 1;
                                        int r_4 = sv_1_0;
                                        if (dkv_has_map != 0) {
                                            r_4 = dkv_dst_map[(long long)sv_1_0];
                                        }
                                        m_4 = (int)(redo_mark[(long long)r_4] == 1);
                                        if (m_4 != 0) {
                                            best = pos;
                                        }
                                    }
                                }
                                if (active > pos + 1) {
                                    int sv_1_1 = indices[row_base + (long long)(pos + 1)];
                                    if (sv_1_1 >= 0 && sv_1_1 < num_kv) {
                                        int m_5 = 1;
                                        int r_5 = sv_1_1;
                                        if (dkv_has_map != 0) {
                                            r_5 = dkv_dst_map[(long long)sv_1_1];
                                        }
                                        m_5 = (int)(redo_mark[(long long)r_5] == 1);
                                        if (m_5 != 0) {
                                            best = pos + 1;
                                        }
                                    }
                                }
                                if (active > pos + 2) {
                                    int sv_1_2 = indices[row_base + (long long)(pos + 2)];
                                    if (sv_1_2 >= 0 && sv_1_2 < num_kv) {
                                        int m_6 = 1;
                                        int r_6 = sv_1_2;
                                        if (dkv_has_map != 0) {
                                            r_6 = dkv_dst_map[(long long)sv_1_2];
                                        }
                                        m_6 = (int)(redo_mark[(long long)r_6] == 1);
                                        if (m_6 != 0) {
                                            best = pos + 2;
                                        }
                                    }
                                }
                                if (active > pos + 3) {
                                    int sv_1_3 = indices[row_base + (long long)(pos + 3)];
                                    if (sv_1_3 >= 0 && sv_1_3 < num_kv) {
                                        int m_7 = 1;
                                        int r_7 = sv_1_3;
                                        if (dkv_has_map != 0) {
                                            r_7 = dkv_dst_map[(long long)sv_1_3];
                                        }
                                        m_7 = (int)(redo_mark[(long long)r_7] == 1);
                                        if (m_7 != 0) {
                                            best = pos + 3;
                                        }
                                    }
                                }
                            }
                        }
#if __CUDA_ARCH__ == 1000
                        int _warp_redux_3;
                        asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(best));
                        best = _warp_redux_3;
#else
                        int _warp_redux_3_1;
                        asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3_1) : "r"(best));
                        best = _warp_redux_3_1;
#endif
                        if (best >= 0) {
                            last_valid = best;
                            break;
                        }
                    }
                    int blocks_4 = (((last_valid + 64) / 64) > (1) ? ((last_valid + 64) / 64) : (1));
                    int bfree_slot = tk_5 % 2;
                    int _stage_blocks_free = (int)((unsigned int)tk_5 % 2);
                    int _parity_blocks_free = (int)((unsigned int)tk_5 / 2 % 2) ^ 1;
                    mbarrier_wait(blocks_free_addr + (_stage_blocks_free) * 8, _parity_blocks_free);
                    if (lane_0_3 == 0) {
                        blocks_word[bfree_slot] = blocks_4;
                        mbarrier_arrive(blocks_ready_addr + ((int)((unsigned int)tk_5 % 2)) * 8);
                    }
                    if (lane_0_3 < 8) {
                        #pragma unroll 1
                        for (int i_1 = 0; i_1 < blocks_4; i_1++) {
                            int n = blocks_4 - 1 - i_1;
                            int slot_2 = (tord_5 + i_1) % 2;
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
                            int _stage_idx_free = (int)((unsigned int)(tord_5 + i_1) % 2);
                            int _parity_idx_free = (int)((unsigned int)(tord_5 + i_1) / 2 % 2) ^ 1;
                            mbarrier_wait(idx_free_addr + (_stage_idx_free) * 8, _parity_idx_free);
                            int ok_7_0 = (int)(values[0] >= 0 && values[0] < num_kv);
                            if (ok_7_0 != 0) {
                                int m_8 = 1;
                                int r_8 = values[0];
                                if (dkv_has_map != 0) {
                                    r_8 = dkv_dst_map[(long long)values[0]];
                                }
                                m_8 = (int)(redo_mark[(long long)r_8] == 1);
                                ok_7_0 = m_8;
                            }
                            if (ok_7_0 != 0) {
                                mask = mask | 1;
                            } else {
                                values[0] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8] = values[0];
                            int ok_7_1 = (int)(values[1] >= 0 && values[1] < num_kv);
                            if (ok_7_1 != 0) {
                                int m_9 = 1;
                                int r_9 = values[1];
                                if (dkv_has_map != 0) {
                                    r_9 = dkv_dst_map[(long long)values[1]];
                                }
                                m_9 = (int)(redo_mark[(long long)r_9] == 1);
                                ok_7_1 = m_9;
                            }
                            if (ok_7_1 != 0) {
                                mask = mask | 2;
                            } else {
                                values[1] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 1] = values[1];
                            int ok_7_2 = (int)(values[2] >= 0 && values[2] < num_kv);
                            if (ok_7_2 != 0) {
                                int m_10 = 1;
                                int r_10 = values[2];
                                if (dkv_has_map != 0) {
                                    r_10 = dkv_dst_map[(long long)values[2]];
                                }
                                m_10 = (int)(redo_mark[(long long)r_10] == 1);
                                ok_7_2 = m_10;
                            }
                            if (ok_7_2 != 0) {
                                mask = mask | 4;
                            } else {
                                values[2] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 2] = values[2];
                            int ok_7_3 = (int)(values[3] >= 0 && values[3] < num_kv);
                            if (ok_7_3 != 0) {
                                int m_11 = 1;
                                int r_11 = values[3];
                                if (dkv_has_map != 0) {
                                    r_11 = dkv_dst_map[(long long)values[3]];
                                }
                                m_11 = (int)(redo_mark[(long long)r_11] == 1);
                                ok_7_3 = m_11;
                            }
                            if (ok_7_3 != 0) {
                                mask = mask | 8;
                            } else {
                                values[3] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 3] = values[3];
                            int ok_7_4 = (int)(values[4] >= 0 && values[4] < num_kv);
                            if (ok_7_4 != 0) {
                                int m_12 = 1;
                                int r_12 = values[4];
                                if (dkv_has_map != 0) {
                                    r_12 = dkv_dst_map[(long long)values[4]];
                                }
                                m_12 = (int)(redo_mark[(long long)r_12] == 1);
                                ok_7_4 = m_12;
                            }
                            if (ok_7_4 != 0) {
                                mask = mask | 16;
                            } else {
                                values[4] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 4] = values[4];
                            int ok_7_5 = (int)(values[5] >= 0 && values[5] < num_kv);
                            if (ok_7_5 != 0) {
                                int m_13 = 1;
                                int r_13 = values[5];
                                if (dkv_has_map != 0) {
                                    r_13 = dkv_dst_map[(long long)values[5]];
                                }
                                m_13 = (int)(redo_mark[(long long)r_13] == 1);
                                ok_7_5 = m_13;
                            }
                            if (ok_7_5 != 0) {
                                mask = mask | 32;
                            } else {
                                values[5] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 5] = values[5];
                            int ok_7_6 = (int)(values[6] >= 0 && values[6] < num_kv);
                            if (ok_7_6 != 0) {
                                int m_14 = 1;
                                int r_14 = values[6];
                                if (dkv_has_map != 0) {
                                    r_14 = dkv_dst_map[(long long)values[6]];
                                }
                                m_14 = (int)(redo_mark[(long long)r_14] == 1);
                                ok_7_6 = m_14;
                            }
                            if (ok_7_6 != 0) {
                                mask = mask | 64;
                            } else {
                                values[6] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 6] = values[6];
                            int ok_7_7 = (int)(values[7] >= 0 && values[7] < num_kv);
                            if (ok_7_7 != 0) {
                                int m_15 = 1;
                                int r_15 = values[7];
                                if (dkv_has_map != 0) {
                                    r_15 = dkv_dst_map[(long long)values[7]];
                                }
                                m_15 = (int)(redo_mark[(long long)r_15] == 1);
                                ok_7_7 = m_15;
                            }
                            if (ok_7_7 != 0) {
                                mask = mask | 128;
                            } else {
                                values[7] = -1;
                            }
                            tile_idx[slot_2 * 80 + lane_0_3 * 8 + 7] = values[7];
                            validity8[slot_2 * 320 + 256 + lane_0_3] = mask;
                            mbarrier_arrive(idx_full_addr + ((int)((unsigned int)(tord_5 + i_1) % 2)) * 8);
                        }
                    }
                    __syncwarp();
                    int nb = blocks_word[bfree_slot];
                    tord_5 += nb;
                }
            }
        // ---- Role: spare ----
        } else if (warp == 19) {
            { // spare_main
                int ntok_6 = (num_queries - blockIdx.x + gridDim.x - 1) / gridDim.x;
                int tord_6 = 0;
                #pragma unroll 1
                for (int tk_6 = 0; tk_6 < ntok_6; tk_6++) {
                    int tok_5 = blockIdx.x + tk_6 * gridDim.x;
                }
            }
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
