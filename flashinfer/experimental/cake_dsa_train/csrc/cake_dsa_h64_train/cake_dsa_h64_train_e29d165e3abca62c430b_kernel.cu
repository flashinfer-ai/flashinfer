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

#define CAKE_DSA_H64_TRAIN_NEED_APPROX_EXP2
#define CAKE_DSA_H64_TRAIN_NEED_ELECT_SYNC
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_ARRIVE
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_ARRIVE_EXPECT_TX
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_INIT
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_WAIT
#define CAKE_DSA_H64_TRAIN_NEED_TCGEN05_COMMIT
#define CAKE_DSA_H64_TRAIN_NEED_TMA_4D_GMEM2SMEM
#define CAKE_DSA_H64_TRAIN_NEED_TMA_STORE_4D
#include "cake_dsa_h64_train_device.cuh"

#define LAUNCH_MIN_BLOCKS 1

#ifndef DKV_NATURAL
#define DKV_NATURAL 0
#endif
#if DKV_NATURAL != 0 && DKV_NATURAL != 1
#error "DKV_NATURAL must be 0 or 1"
#endif
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_dsa_h64_train_e29d165e3abca62c430b
#endif

extern "C" {

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) void
CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap dout, const __grid_constant__ CUtensorMap dq_latent, const __grid_constant__ CUtensorMap dq_rope, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap k_rope, float* __restrict__ lse, float* __restrict__ delta, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, int* __restrict__ indices, int* __restrict__ topk_length, float* __restrict__ dkv_f32, float* __restrict__ dkr_f32, int dkv_stride, int dkr_stride, int dkr_col0, int* __restrict__ dkv_dst_map, int dkv_has_map, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int pass_lo, int pass_hi, int dq_mode, float* __restrict__ dq_partial, int* __restrict__ key_scratch, int* __restrict__ pass_counts)
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

    // Kernel setup ops
    const int q_k_addr = smem + 1024;
    const int q_mn_addr = smem + 1024;
    const int qr_mn_addr = smem + 66560;
    const int do_k_addr = smem + 74752;
    const int do_mn_addr = smem + 74752;
    const int k_k_addr = smem + 140288;
    const int v_k_addr = smem + 140288;
    const int k_mn_addr = smem + 140288;
    const int kr_mn_addr = smem + 205824;
    const int k_blk_addr = smem + 140288;
    const int p_k_addr = smem + 214016;
    const int ds_k_addr = smem + 222208;
    const int ds_mn_addr = smem + 222208;
    float* stats = reinterpret_cast<float*>(smem_raw + 230400);
    int* tile_idx = reinterpret_cast<int*>(smem_raw + 230912);
    uint8_t* validity8 = reinterpret_cast<uint8_t*>(smem_raw + 230912);
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 230912);
    int* blocks_word = reinterpret_cast<int*>(smem_raw + 231552);
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
                    int slot = (int)((unsigned int)i % 2);
                    int _phase_slot = (int)((unsigned int)i / 2 % 2);
                    mbarrier_wait(idx_full_addr + (slot) * 8, _phase_slot);
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
                    int _parity_k_free = (int)((unsigned int)i % 2) ^ 1;
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
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // compute_main
            int token = token_base + token_step * blockIdx.x;
            int lane_0 = lane;
            int w = warp - 4;
            int lane_base = (tid & 96) << 16;
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
                int slot_1 = (int)((unsigned int)i_1 % 2);
                int _phase_slot_1 = (int)((unsigned int)i_1 / 2 % 2);
                mbarrier_wait(idx_full_addr + (slot_1) * 8, _phase_slot_1);
                unsigned int v0 = validity32[slot_1 * 80 + 64];
                unsigned int v1 = validity32[slot_1 * 80 + 65];
                if (lane_0 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_1) * 8);
                }
                int _parity_s_full = (int)((unsigned int)i_1 % 2);
                mbarrier_wait(s_full_addr, _parity_s_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float s[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s[0])), "=r"(*reinterpret_cast<uint32_t*>(&s[1])), "=r"(*reinterpret_cast<uint32_t*>(&s[2])), "=r"(*reinterpret_cast<uint32_t*>(&s[3])), "=r"(*reinterpret_cast<uint32_t*>(&s[4])), "=r"(*reinterpret_cast<uint32_t*>(&s[5])), "=r"(*reinterpret_cast<uint32_t*>(&s[6])), "=r"(*reinterpret_cast<uint32_t*>(&s[7])), "=r"(*reinterpret_cast<uint32_t*>(&s[8])), "=r"(*reinterpret_cast<uint32_t*>(&s[9])), "=r"(*reinterpret_cast<uint32_t*>(&s[10])), "=r"(*reinterpret_cast<uint32_t*>(&s[11])), "=r"(*reinterpret_cast<uint32_t*>(&s[12])), "=r"(*reinterpret_cast<uint32_t*>(&s[13])), "=r"(*reinterpret_cast<uint32_t*>(&s[14])), "=r"(*reinterpret_cast<uint32_t*>(&s[15])), "=r"(*reinterpret_cast<uint32_t*>(&s[16])), "=r"(*reinterpret_cast<uint32_t*>(&s[17])), "=r"(*reinterpret_cast<uint32_t*>(&s[18])), "=r"(*reinterpret_cast<uint32_t*>(&s[19])), "=r"(*reinterpret_cast<uint32_t*>(&s[20])), "=r"(*reinterpret_cast<uint32_t*>(&s[21])), "=r"(*reinterpret_cast<uint32_t*>(&s[22])), "=r"(*reinterpret_cast<uint32_t*>(&s[23])), "=r"(*reinterpret_cast<uint32_t*>(&s[24])), "=r"(*reinterpret_cast<uint32_t*>(&s[25])), "=r"(*reinterpret_cast<uint32_t*>(&s[26])), "=r"(*reinterpret_cast<uint32_t*>(&s[27])), "=r"(*reinterpret_cast<uint32_t*>(&s[28])), "=r"(*reinterpret_cast<uint32_t*>(&s[29])), "=r"(*reinterpret_cast<uint32_t*>(&s[30])), "=r"(*reinterpret_cast<uint32_t*>(&s[31]))
                    : "r"(tmem_tmem + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float p[32];
                float _exp2_0 = approx_exp2(__fmaf_rn(s[0], scale_log2, -lse0));
                p[0] = _exp2_0;
                float _exp2_1 = approx_exp2(__fmaf_rn(s[2], scale_log2, -lse1));
                p[2] = _exp2_1;
                unsigned int bit = v0 >> (unsigned int)kb & 1;
                if (bit == 0) {
                    p[0] = 0.0f;
                    p[2] = 0.0f;
                }
                float _exp2_2 = approx_exp2(__fmaf_rn(s[1], scale_log2, -lse0));
                p[1] = _exp2_2;
                float _exp2_3 = approx_exp2(__fmaf_rn(s[3], scale_log2, -lse1));
                p[3] = _exp2_3;
                unsigned int bit_0 = v0 >> (unsigned int)(1 + kb) & 1;
                if (bit_0 == 0) {
                    p[1] = 0.0f;
                    p[3] = 0.0f;
                }
                float _exp2_4 = approx_exp2(__fmaf_rn(s[4], scale_log2, -lse0));
                p[4] = _exp2_4;
                float _exp2_5 = approx_exp2(__fmaf_rn(s[6], scale_log2, -lse1));
                p[6] = _exp2_5;
                unsigned int bit_1 = v0 >> (unsigned int)(8 + kb) & 1;
                if (bit_1 == 0) {
                    p[4] = 0.0f;
                    p[6] = 0.0f;
                }
                float _exp2_6 = approx_exp2(__fmaf_rn(s[5], scale_log2, -lse0));
                p[5] = _exp2_6;
                float _exp2_7 = approx_exp2(__fmaf_rn(s[7], scale_log2, -lse1));
                p[7] = _exp2_7;
                unsigned int bit_2 = v0 >> (unsigned int)(9 + kb) & 1;
                if (bit_2 == 0) {
                    p[5] = 0.0f;
                    p[7] = 0.0f;
                }
                float _exp2_8 = approx_exp2(__fmaf_rn(s[8], scale_log2, -lse0));
                p[8] = _exp2_8;
                float _exp2_9 = approx_exp2(__fmaf_rn(s[10], scale_log2, -lse1));
                p[10] = _exp2_9;
                unsigned int bit_3 = v0 >> (unsigned int)(16 + kb) & 1;
                if (bit_3 == 0) {
                    p[8] = 0.0f;
                    p[10] = 0.0f;
                }
                float _exp2_10 = approx_exp2(__fmaf_rn(s[9], scale_log2, -lse0));
                p[9] = _exp2_10;
                float _exp2_11 = approx_exp2(__fmaf_rn(s[11], scale_log2, -lse1));
                p[11] = _exp2_11;
                unsigned int bit_4 = v0 >> (unsigned int)(17 + kb) & 1;
                if (bit_4 == 0) {
                    p[9] = 0.0f;
                    p[11] = 0.0f;
                }
                float _exp2_12 = approx_exp2(__fmaf_rn(s[12], scale_log2, -lse0));
                p[12] = _exp2_12;
                float _exp2_13 = approx_exp2(__fmaf_rn(s[14], scale_log2, -lse1));
                p[14] = _exp2_13;
                unsigned int bit_5 = v0 >> (unsigned int)(24 + kb) & 1;
                if (bit_5 == 0) {
                    p[12] = 0.0f;
                    p[14] = 0.0f;
                }
                float _exp2_14 = approx_exp2(__fmaf_rn(s[13], scale_log2, -lse0));
                p[13] = _exp2_14;
                float _exp2_15 = approx_exp2(__fmaf_rn(s[15], scale_log2, -lse1));
                p[15] = _exp2_15;
                unsigned int bit_6 = v0 >> (unsigned int)(25 + kb) & 1;
                if (bit_6 == 0) {
                    p[13] = 0.0f;
                    p[15] = 0.0f;
                }
                float _exp2_16 = approx_exp2(__fmaf_rn(s[16], scale_log2, -lse0));
                p[16] = _exp2_16;
                float _exp2_17 = approx_exp2(__fmaf_rn(s[18], scale_log2, -lse1));
                p[18] = _exp2_17;
                unsigned int bit_7 = v1 >> (unsigned int)kb & 1;
                if (bit_7 == 0) {
                    p[16] = 0.0f;
                    p[18] = 0.0f;
                }
                float _exp2_18 = approx_exp2(__fmaf_rn(s[17], scale_log2, -lse0));
                p[17] = _exp2_18;
                float _exp2_19 = approx_exp2(__fmaf_rn(s[19], scale_log2, -lse1));
                p[19] = _exp2_19;
                unsigned int bit_8 = v1 >> (unsigned int)(1 + kb) & 1;
                if (bit_8 == 0) {
                    p[17] = 0.0f;
                    p[19] = 0.0f;
                }
                float _exp2_20 = approx_exp2(__fmaf_rn(s[20], scale_log2, -lse0));
                p[20] = _exp2_20;
                float _exp2_21 = approx_exp2(__fmaf_rn(s[22], scale_log2, -lse1));
                p[22] = _exp2_21;
                unsigned int bit_9 = v1 >> (unsigned int)(8 + kb) & 1;
                if (bit_9 == 0) {
                    p[20] = 0.0f;
                    p[22] = 0.0f;
                }
                float _exp2_22 = approx_exp2(__fmaf_rn(s[21], scale_log2, -lse0));
                p[21] = _exp2_22;
                float _exp2_23 = approx_exp2(__fmaf_rn(s[23], scale_log2, -lse1));
                p[23] = _exp2_23;
                unsigned int bit_10 = v1 >> (unsigned int)(9 + kb) & 1;
                if (bit_10 == 0) {
                    p[21] = 0.0f;
                    p[23] = 0.0f;
                }
                float _exp2_24 = approx_exp2(__fmaf_rn(s[24], scale_log2, -lse0));
                p[24] = _exp2_24;
                float _exp2_25 = approx_exp2(__fmaf_rn(s[26], scale_log2, -lse1));
                p[26] = _exp2_25;
                unsigned int bit_11 = v1 >> (unsigned int)(16 + kb) & 1;
                if (bit_11 == 0) {
                    p[24] = 0.0f;
                    p[26] = 0.0f;
                }
                float _exp2_26 = approx_exp2(__fmaf_rn(s[25], scale_log2, -lse0));
                p[25] = _exp2_26;
                float _exp2_27 = approx_exp2(__fmaf_rn(s[27], scale_log2, -lse1));
                p[27] = _exp2_27;
                unsigned int bit_12 = v1 >> (unsigned int)(17 + kb) & 1;
                if (bit_12 == 0) {
                    p[25] = 0.0f;
                    p[27] = 0.0f;
                }
                float _exp2_28 = approx_exp2(__fmaf_rn(s[28], scale_log2, -lse0));
                p[28] = _exp2_28;
                float _exp2_29 = approx_exp2(__fmaf_rn(s[30], scale_log2, -lse1));
                p[30] = _exp2_29;
                unsigned int bit_13 = v1 >> (unsigned int)(24 + kb) & 1;
                if (bit_13 == 0) {
                    p[28] = 0.0f;
                    p[30] = 0.0f;
                }
                float _exp2_30 = approx_exp2(__fmaf_rn(s[29], scale_log2, -lse0));
                p[29] = _exp2_30;
                float _exp2_31 = approx_exp2(__fmaf_rn(s[31], scale_log2, -lse1));
                p[31] = _exp2_31;
                unsigned int bit_14 = v1 >> (unsigned int)(25 + kb) & 1;
                if (bit_14 == 0) {
                    p[29] = 0.0f;
                    p[31] = 0.0f;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _parity_p_free = (int)((unsigned int)i_1 % 2) ^ 1;
                mbarrier_wait(p_free_addr, _parity_p_free);
                unsigned int p_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                    p_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int paddr = p_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(paddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[3]))
                    : "memory");
                unsigned int paddr_15 = p_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(paddr_15);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[7]))
                    : "memory");
                unsigned int paddr_16 = p_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(paddr_16);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[11]))
                    : "memory");
                unsigned int paddr_17 = p_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(paddr_17);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                int _parity_dp_full = (int)((unsigned int)i_1 % 2);
                mbarrier_wait(dp_full_addr, _parity_dp_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float dp[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[1])), "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[3])), "=r"(*reinterpret_cast<uint32_t*>(&dp[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp[5])), "=r"(*reinterpret_cast<uint32_t*>(&dp[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp[7])), "=r"(*reinterpret_cast<uint32_t*>(&dp[8])), "=r"(*reinterpret_cast<uint32_t*>(&dp[9])), "=r"(*reinterpret_cast<uint32_t*>(&dp[10])), "=r"(*reinterpret_cast<uint32_t*>(&dp[11])), "=r"(*reinterpret_cast<uint32_t*>(&dp[12])), "=r"(*reinterpret_cast<uint32_t*>(&dp[13])), "=r"(*reinterpret_cast<uint32_t*>(&dp[14])), "=r"(*reinterpret_cast<uint32_t*>(&dp[15])), "=r"(*reinterpret_cast<uint32_t*>(&dp[16])), "=r"(*reinterpret_cast<uint32_t*>(&dp[17])), "=r"(*reinterpret_cast<uint32_t*>(&dp[18])), "=r"(*reinterpret_cast<uint32_t*>(&dp[19])), "=r"(*reinterpret_cast<uint32_t*>(&dp[20])), "=r"(*reinterpret_cast<uint32_t*>(&dp[21])), "=r"(*reinterpret_cast<uint32_t*>(&dp[22])), "=r"(*reinterpret_cast<uint32_t*>(&dp[23])), "=r"(*reinterpret_cast<uint32_t*>(&dp[24])), "=r"(*reinterpret_cast<uint32_t*>(&dp[25])), "=r"(*reinterpret_cast<uint32_t*>(&dp[26])), "=r"(*reinterpret_cast<uint32_t*>(&dp[27])), "=r"(*reinterpret_cast<uint32_t*>(&dp[28])), "=r"(*reinterpret_cast<uint32_t*>(&dp[29])), "=r"(*reinterpret_cast<uint32_t*>(&dp[30])), "=r"(*reinterpret_cast<uint32_t*>(&dp[31]))
                    : "r"(tmem_tmem + lane_base + 1048576));
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
                int _parity_ds_free = (int)((unsigned int)i_1 % 2) ^ 1;
                mbarrier_wait(ds_free_addr, _parity_ds_free);
                unsigned int ds_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ds[_lp*2 + 0], ds[_lp*2+1 + 0]));
                    ds_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int daddr = ds_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(daddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[3]))
                    : "memory");
                unsigned int daddr_18 = ds_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(daddr_18);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[7]))
                    : "memory");
                unsigned int daddr_19 = ds_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(daddr_19);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[11]))
                    : "memory");
                unsigned int daddr_20 = ds_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(daddr_20);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(ds_full_addr);
            }
            mbarrier_wait(dq_done_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float dq[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq[31]))
                : "r"(tmem_tmem + lane_base + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq[_lp*2 + 0], dq[_lp*2+1 + 0]));
                dq_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead = 8 * (m4 / 2) + r8;
            int qchunk = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead * 128) + (unsigned int)((qchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(qaddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[3]))
                : "memory");
            int qhead_1 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_2 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_3 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_1 * 128) + (unsigned int)((qchunk_2 ^ r8) * 16);
            uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(qaddr_3);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[7]))
                : "memory");
            int qhead_4 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_5 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_6 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_4 * 128) + (unsigned int)((qchunk_5 ^ r8) * 16);
            uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(qaddr_6);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[11]))
                : "memory");
            int qhead_7 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_8 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_9 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_7 * 128) + (unsigned int)((qchunk_8 ^ r8) * 16);
            uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(qaddr_9);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[15]))
                : "memory");
            float dq_10[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[31]))
                : "r"(tmem_tmem + lane_base + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_10_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_10[_lp*2 + 0], dq_10[_lp*2+1 + 0]));
                dq_10_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_11 = 8 * (m4 / 2) + r8;
            int qchunk_12 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_13 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_11 * 128) + (unsigned int)((qchunk_12 ^ r8) * 16);
            uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(qaddr_13);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[3]))
                : "memory");
            int qhead_14 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_15 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_16 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_14 * 128) + (unsigned int)((qchunk_15 ^ r8) * 16);
            uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(qaddr_16);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[7]))
                : "memory");
            int qhead_17 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_18 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_19 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_17 * 128) + (unsigned int)((qchunk_18 ^ r8) * 16);
            uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(qaddr_19);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[11]))
                : "memory");
            int qhead_20 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_21 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_22 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_20 * 128) + (unsigned int)((qchunk_21 ^ r8) * 16);
            uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(qaddr_22);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[15]))
                : "memory");
            float dq_23[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[31]))
                : "r"(tmem_tmem + lane_base + 192 + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_23_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_23[_lp*2 + 0], dq_23[_lp*2+1 + 0]));
                dq_23_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_24 = 8 * (m4 / 2) + r8;
            int qchunk_25 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_26 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_24 * 128) + (unsigned int)((qchunk_25 ^ r8) * 16);
            uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(qaddr_26);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[3]))
                : "memory");
            int qhead_27 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_28 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_29 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_27 * 128) + (unsigned int)((qchunk_28 ^ r8) * 16);
            uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(qaddr_29);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[7]))
                : "memory");
            int qhead_30 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_31 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_32 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_30 * 128) + (unsigned int)((qchunk_31 ^ r8) * 16);
            uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(qaddr_32);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[11]))
                : "memory");
            int qhead_33 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_34 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_35 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_33 * 128) + (unsigned int)((qchunk_34 ^ r8) * 16);
            uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(qaddr_35);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[15]))
                : "memory");
            float dq_36[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[31]))
                : "r"(tmem_tmem + lane_base + 192 + 64 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_36_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_36[_lp*2 + 0], dq_36[_lp*2+1 + 0]));
                dq_36_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_37 = 8 * (m4 / 2) + r8;
            int qchunk_38 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_39 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_37 * 128) + (unsigned int)((qchunk_38 ^ r8) * 16);
            uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(qaddr_39);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[3]))
                : "memory");
            int qhead_40 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_41 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_42 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_40 * 128) + (unsigned int)((qchunk_41 ^ r8) * 16);
            uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(qaddr_42);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[7]))
                : "memory");
            int qhead_43 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_44 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_45 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_43 * 128) + (unsigned int)((qchunk_44 ^ r8) * 16);
            uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(qaddr_45);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[11]))
                : "memory");
            int qhead_46 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_47 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_48 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_46 * 128) + (unsigned int)((qchunk_47 ^ r8) * 16);
            uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(qaddr_48);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[15]))
                : "memory");
            float dq_49[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[31]))
                : "r"(tmem_tmem + lane_base + 192 + 128));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_49_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_49[_lp*2 + 0], dq_49[_lp*2+1 + 0]));
                dq_49_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_50 = 8 * (m4 / 2) + r8;
            int qchunk_51 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_52 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_50 * 128) + (unsigned int)((qchunk_51 ^ r8) * 16);
            uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(qaddr_52);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[3]))
                : "memory");
            int qhead_53 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_54 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_55 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_53 * 128) + (unsigned int)((qchunk_54 ^ r8) * 16);
            uint32_t _stmatrix_addr_57 = static_cast<uint32_t>(qaddr_55);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_57), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[7]))
                : "memory");
            int qhead_56 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_57 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_58 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_56 * 128) + (unsigned int)((qchunk_57 ^ r8) * 16);
            uint32_t _stmatrix_addr_58 = static_cast<uint32_t>(qaddr_58);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_58), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[11]))
                : "memory");
            int qhead_59 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_60 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_61 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_59 * 128) + (unsigned int)((qchunk_60 ^ r8) * 16);
            uint32_t _stmatrix_addr_59 = static_cast<uint32_t>(qaddr_61);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_59), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[15]))
                : "memory");
            float dq_62[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[31]))
                : "r"(tmem_tmem + lane_base + 192 + 128 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_62_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_62[_lp*2 + 0], dq_62[_lp*2+1 + 0]));
                dq_62_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_63 = 8 * (m4 / 2) + r8;
            int qchunk_64 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_65 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_63 * 128) + (unsigned int)((qchunk_64 ^ r8) * 16);
            uint32_t _stmatrix_addr_60 = static_cast<uint32_t>(qaddr_65);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_60), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[3]))
                : "memory");
            int qhead_66 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_67 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_68 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_66 * 128) + (unsigned int)((qchunk_67 ^ r8) * 16);
            uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(qaddr_68);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[7]))
                : "memory");
            int qhead_69 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_70 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_71 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_69 * 128) + (unsigned int)((qchunk_70 ^ r8) * 16);
            uint32_t _stmatrix_addr_62 = static_cast<uint32_t>(qaddr_71);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_62), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[11]))
                : "memory");
            int qhead_72 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_73 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_74 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_72 * 128) + (unsigned int)((qchunk_73 ^ r8) * 16);
            uint32_t _stmatrix_addr_63 = static_cast<uint32_t>(qaddr_74);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_63), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[15]))
                : "memory");
            float dq_75[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_75[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[31]))
                : "r"(tmem_tmem + lane_base + 192 + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_75_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_75[_lp*2 + 0], dq_75[_lp*2+1 + 0]));
                dq_75_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_76 = 8 * (m4 / 2) + r8;
            int qchunk_77 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_78 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_76 * 128) + (unsigned int)((qchunk_77 ^ r8) * 16);
            uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(qaddr_78);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[3]))
                : "memory");
            int qhead_79 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_80 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_81 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_79 * 128) + (unsigned int)((qchunk_80 ^ r8) * 16);
            uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(qaddr_81);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[7]))
                : "memory");
            int qhead_82 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_83 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_84 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_82 * 128) + (unsigned int)((qchunk_83 ^ r8) * 16);
            uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(qaddr_84);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[11]))
                : "memory");
            int qhead_85 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_86 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_87 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_85 * 128) + (unsigned int)((qchunk_86 ^ r8) * 16);
            uint32_t _stmatrix_addr_67 = static_cast<uint32_t>(qaddr_87);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_67), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[15]))
                : "memory");
            float dq_88[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[31]))
                : "r"(tmem_tmem + lane_base + 192 + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            unsigned int dq_88_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_88[_lp*2 + 0], dq_88[_lp*2+1 + 0]));
                dq_88_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_89 = 8 * (m4 / 2) + r8;
            int qchunk_90 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_91 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_89 * 128) + (unsigned int)((qchunk_90 ^ r8) * 16);
            uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(qaddr_91);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[3]))
                : "memory");
            int qhead_92 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_93 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_94 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_92 * 128) + (unsigned int)((qchunk_93 ^ r8) * 16);
            uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(qaddr_94);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[7]))
                : "memory");
            int qhead_95 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_96 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_97 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_95 * 128) + (unsigned int)((qchunk_96 ^ r8) * 16);
            uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(qaddr_97);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[11]))
                : "memory");
            int qhead_98 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_99 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_100 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_98 * 128) + (unsigned int)((qchunk_99 ^ r8) * 16);
            uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(qaddr_100);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[15]))
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
            int rhead = 8 * (m4 / 2) + r8;
            int rchunk = 2 * w + m4 % 2;
            unsigned int raddr = k_blk_addr + 65536 + (unsigned int)(rhead * 128) + (unsigned int)((rchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(raddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[3]))
                : "memory");
            int rhead_101 = 16 + 8 * (m4 / 2) + r8;
            int rchunk_102 = 2 * w + m4 % 2;
            unsigned int raddr_103 = k_blk_addr + 65536 + (unsigned int)(rhead_101 * 128) + (unsigned int)((rchunk_102 ^ r8) * 16);
            uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(raddr_103);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[7]))
                : "memory");
            int rhead_104 = 32 + 8 * (m4 / 2) + r8;
            int rchunk_105 = 2 * w + m4 % 2;
            unsigned int raddr_106 = k_blk_addr + 65536 + (unsigned int)(rhead_104 * 128) + (unsigned int)((rchunk_105 ^ r8) * 16);
            uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(raddr_106);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[11]))
                : "memory");
            int rhead_107 = 48 + 8 * (m4 / 2) + r8;
            int rchunk_108 = 2 * w + m4 % 2;
            unsigned int raddr_109 = k_blk_addr + 65536 + (unsigned int)(rhead_107 * 128) + (unsigned int)((rchunk_108 ^ r8) * 16);
            uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(raddr_109);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[15]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w == 0) {
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
            int lane_base_1 = (tid & 96) << 16;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_2 = blocks_word[0];
            int lpos = 32 * w2 + 4 * (lane_0_1 / 4);
            int rpos = 16 * w2 + 2 * (lane_0_1 / 4);
#if DKV_NATURAL == 1
            int npos = 32 * w2 + 8 * (lane_0_1 >> 2 & 3) + 4 * (lane_0_1 >> 4);
            int nrpos = 16 * w2 + (lane_0_1 >> 2 & 6) + 8 * (lane_0_1 >> 2 & 1);
#endif
            #pragma unroll 1
            for (int i_2 = 0; i_2 < blocks_2; i_2++) {
                int slot_2 = (int)((unsigned int)i_2 % 2);
                int _phase_slot_2 = (int)((unsigned int)i_2 / 2 % 2);
                mbarrier_wait(idx_full_addr + (slot_2) * 8, _phase_slot_2);
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
#if DKV_NATURAL == 1
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
#endif
                }
                int _parity_dkv_a_full = (int)((unsigned int)i_2 % 2);
                mbarrier_wait(dkv_a_full_addr, _parity_dkv_a_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float a0[32];
                float a1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[0])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[1])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[2])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[3])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[4])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[5])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[6])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[7])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[8])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[9])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[10])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[11])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[12])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[13])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[14])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[15]))
                    : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[0])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[1])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[2])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[3])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[4])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[5])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[6])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[7])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[8])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[9])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[10])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[11])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[12])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[13])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[14])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[15]))
                    : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_a_drained_addr);
                float rk[16];
                int _parity_dkr_full = (int)((unsigned int)i_2 % 2);
                mbarrier_wait(dkr_full_addr, _parity_dkr_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rk[0])), "=r"(*reinterpret_cast<uint32_t*>(&rk[1])), "=r"(*reinterpret_cast<uint32_t*>(&rk[2])), "=r"(*reinterpret_cast<uint32_t*>(&rk[3])), "=r"(*reinterpret_cast<uint32_t*>(&rk[4])), "=r"(*reinterpret_cast<uint32_t*>(&rk[5])), "=r"(*reinterpret_cast<uint32_t*>(&rk[6])), "=r"(*reinterpret_cast<uint32_t*>(&rk[7])), "=r"(*reinterpret_cast<uint32_t*>(&rk[8])), "=r"(*reinterpret_cast<uint32_t*>(&rk[9])), "=r"(*reinterpret_cast<uint32_t*>(&rk[10])), "=r"(*reinterpret_cast<uint32_t*>(&rk[11])), "=r"(*reinterpret_cast<uint32_t*>(&rk[12])), "=r"(*reinterpret_cast<uint32_t*>(&rk[13])), "=r"(*reinterpret_cast<uint32_t*>(&rk[14])), "=r"(*reinterpret_cast<uint32_t*>(&rk[15]))
                    : "r"(tmem_tmem + lane_base_1 + wg * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkr_drained_addr);
#if DKV_NATURAL == 1
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16])), 8);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1)), 8);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16])), 8);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5)), 8);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[0] : rk[2]), 4);
#endif
                int key = keys[0];
                if (key >= 0) {
#if DKV_NATURAL == 1
                    int row = rows_1[0];
                    long long base = (long long)row * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_2 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_3 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16]) : _shfl_xor_2)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1) : _shfl_xor_3)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_6 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_7 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16]) : _shfl_xor_6)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5) : _shfl_xor_7)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_8 : rk[0])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[2] : _shfl_xor_8)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base = (long long)key * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base])), "f"(a0[0]), "f"(a0[2]), "f"(a0[16]), "f"(a0[18]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base + 128])), "f"(a1[0]), "f"(a1[2]), "f"(a1[16]), "f"(a1[18]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key * 64 + (long long)rpos])), "f"(rk[0]), "f"(rk[2]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17])), 8);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10)), 8);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17])), 8);
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14)), 8);
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[1] : rk[3]), 4);
#endif
                int key_0 = keys[1];
                if (key_0 >= 0) {
#if DKV_NATURAL == 1
                    int row_1 = rows_1[1];
                    long long base_1 = (long long)row_1 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_11 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_12 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17]) : _shfl_xor_11)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10) : _shfl_xor_12)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_15 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_16 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17]) : _shfl_xor_15)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14) : _shfl_xor_16)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_1 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_17 : rk[1])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[3] : _shfl_xor_17)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_1 = (long long)key_0 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1])), "f"(a0[1]), "f"(a0[3]), "f"(a0[17]), "f"(a0[19]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1 + 128])), "f"(a1[1]), "f"(a1[3]), "f"(a1[17]), "f"(a1[19]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_0 * 64 + (long long)rpos])), "f"(rk[1]), "f"(rk[3]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20])), 8);
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19)), 8);
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20])), 8);
                float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23)), 8);
                float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[4] : rk[6]), 4);
#endif
                int key_1 = keys[2];
                if (key_1 >= 0) {
#if DKV_NATURAL == 1
                    int row_2 = rows_1[2];
                    long long base_2 = (long long)row_2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_20 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_21 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20]) : _shfl_xor_20)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19) : _shfl_xor_21)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_24 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_25 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20]) : _shfl_xor_24)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23) : _shfl_xor_25)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_2 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_26 : rk[4])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[6] : _shfl_xor_26)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_2 = (long long)key_1 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2])), "f"(a0[4]), "f"(a0[6]), "f"(a0[20]), "f"(a0[22]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2 + 128])), "f"(a1[4]), "f"(a1[6]), "f"(a1[20]), "f"(a1[22]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_1 * 64 + (long long)rpos])), "f"(rk[4]), "f"(rk[6]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21])), 8);
                float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28)), 8);
                float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21])), 8);
                float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32)), 8);
                float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[5] : rk[7]), 4);
#endif
                int key_2 = keys[3];
                if (key_2 >= 0) {
#if DKV_NATURAL == 1
                    int row_3 = rows_1[3];
                    long long base_3 = (long long)row_3 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_29 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_30 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21]) : _shfl_xor_29)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28) : _shfl_xor_30)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_33 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_34 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21]) : _shfl_xor_33)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32) : _shfl_xor_34)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_3 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_35 : rk[5])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[7] : _shfl_xor_35)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_3 = (long long)key_2 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3])), "f"(a0[5]), "f"(a0[7]), "f"(a0[21]), "f"(a0[23]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3 + 128])), "f"(a1[5]), "f"(a1[7]), "f"(a1[21]), "f"(a1[23]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_2 * 64 + (long long)rpos])), "f"(rk[5]), "f"(rk[7]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24])), 8);
                float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37)), 8);
                float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24])), 8);
                float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41)), 8);
                float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[8] : rk[10]), 4);
#endif
                int key_3 = keys[4];
                if (key_3 >= 0) {
#if DKV_NATURAL == 1
                    int row_4 = rows_1[4];
                    long long base_4 = (long long)row_4 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_38 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_39 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24]) : _shfl_xor_38)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37) : _shfl_xor_39)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_42 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_43 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24]) : _shfl_xor_42)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41) : _shfl_xor_43)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_4 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_44 : rk[8])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[10] : _shfl_xor_44)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_4 = (long long)key_3 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4])), "f"(a0[8]), "f"(a0[10]), "f"(a0[24]), "f"(a0[26]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4 + 128])), "f"(a1[8]), "f"(a1[10]), "f"(a1[24]), "f"(a1[26]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_3 * 64 + (long long)rpos])), "f"(rk[8]), "f"(rk[10]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25])), 8);
                float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46)), 8);
                float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25])), 8);
                float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50)), 8);
                float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[9] : rk[11]), 4);
#endif
                int key_4 = keys[5];
                if (key_4 >= 0) {
#if DKV_NATURAL == 1
                    int row_5 = rows_1[5];
                    long long base_5 = (long long)row_5 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_47 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_48 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25]) : _shfl_xor_47)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46) : _shfl_xor_48)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_51 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_52 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25]) : _shfl_xor_51)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50) : _shfl_xor_52)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_5 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_53 : rk[9])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[11] : _shfl_xor_53)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_5 = (long long)key_4 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5])), "f"(a0[9]), "f"(a0[11]), "f"(a0[25]), "f"(a0[27]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5 + 128])), "f"(a1[9]), "f"(a1[11]), "f"(a1[25]), "f"(a1[27]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_4 * 64 + (long long)rpos])), "f"(rk[9]), "f"(rk[11]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28])), 8);
                float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55)), 8);
                float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28])), 8);
                float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59)), 8);
                float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[12] : rk[14]), 4);
#endif
                int key_5 = keys[6];
                if (key_5 >= 0) {
#if DKV_NATURAL == 1
                    int row_6 = rows_1[6];
                    long long base_6 = (long long)row_6 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_56 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_57 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28]) : _shfl_xor_56)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55) : _shfl_xor_57)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_60 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_61 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28]) : _shfl_xor_60)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59) : _shfl_xor_61)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_6 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_62 : rk[12])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[14] : _shfl_xor_62)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_6 = (long long)key_5 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6])), "f"(a0[12]), "f"(a0[14]), "f"(a0[28]), "f"(a0[30]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6 + 128])), "f"(a1[12]), "f"(a1[14]), "f"(a1[28]), "f"(a1[30]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_5 * 64 + (long long)rpos])), "f"(rk[12]), "f"(rk[14]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29])), 8);
                float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64)), 8);
                float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29])), 8);
                float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68)), 8);
                float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[13] : rk[15]), 4);
#endif
                int key_6 = keys[7];
                if (key_6 >= 0) {
#if DKV_NATURAL == 1
                    int row_7 = rows_1[7];
                    long long base_7 = (long long)row_7 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_65 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_66 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29]) : _shfl_xor_65)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64) : _shfl_xor_66)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7 + 128])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_69 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_70 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29]) : _shfl_xor_69)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68) : _shfl_xor_70)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)row_7 * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos])), "f"((((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_71 : rk[13])), "f"((((lane_0_1 >> 2 & 1) != 0) ? rk[15] : _shfl_xor_71)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base_7 = (long long)key_6 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7])), "f"(a0[13]), "f"(a0[15]), "f"(a0[29]), "f"(a0[31]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7 + 128])), "f"(a1[13]), "f"(a1[15]), "f"(a1[29]), "f"(a1[31]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v2.f32 [%0], {%1, %2}, %3;" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_6 * 64 + (long long)rpos])), "f"(rk[13]), "f"(rk[15]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
                int _parity_dkv_b_full = (int)((unsigned int)i_2 % 2);
                mbarrier_wait(dkv_b_full_addr, _parity_dkv_b_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[0])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[1])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[2])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[3])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[4])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[5])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[6])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[7])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[8])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[9])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[10])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[11])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[12])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[13])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[14])), "=r"(*reinterpret_cast<uint32_t*>(&(a0 + 16)[15]))
                    : "r"(tmem_tmem + lane_base_1 + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[0])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[1])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[2])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[3])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[4])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[5])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[6])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[7])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[8])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[9])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[10])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[11])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[12])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[13])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[14])), "=r"(*reinterpret_cast<uint32_t*>(&(a1 + 16)[15]))
                    : "r"(tmem_tmem + lane_base_1 + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_b_drained_addr);
#if DKV_NATURAL == 1
                float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16])), 8);
                float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73)), 8);
                float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16])), 8);
                float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77)), 8);
#endif
                int key2 = keys[0];
                if (key2 >= 0) {
#if DKV_NATURAL == 1
                    int row2 = rows_1[0];
                    long long base2 = (long long)row2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_74 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_75 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16]) : _shfl_xor_74)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73) : _shfl_xor_75)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_78 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_79 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16]) : _shfl_xor_78)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77) : _shfl_xor_79)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2 = (long long)key2 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 256])), "f"(a0[0]), "f"(a0[2]), "f"(a0[16]), "f"(a0[18]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 384])), "f"(a1[0]), "f"(a1[2]), "f"(a1[16]), "f"(a1[18]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17])), 8);
                float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81)), 8);
                float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17])), 8);
                float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85)), 8);
#endif
                int key2_7 = keys[1];
                if (key2_7 >= 0) {
#if DKV_NATURAL == 1
                    int row2_1 = rows_1[1];
                    long long base2_1 = (long long)row2_1 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_82 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_83 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17]) : _shfl_xor_82)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81) : _shfl_xor_83)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_86 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_87 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17]) : _shfl_xor_86)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85) : _shfl_xor_87)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_1 = (long long)key2_7 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 256])), "f"(a0[1]), "f"(a0[3]), "f"(a0[17]), "f"(a0[19]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 384])), "f"(a1[1]), "f"(a1[3]), "f"(a1[17]), "f"(a1[19]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20])), 8);
                float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89)), 8);
                float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20])), 8);
                float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93)), 8);
#endif
                int key2_8 = keys[2];
                if (key2_8 >= 0) {
#if DKV_NATURAL == 1
                    int row2_2 = rows_1[2];
                    long long base2_2 = (long long)row2_2 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_90 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_91 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20]) : _shfl_xor_90)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89) : _shfl_xor_91)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_94 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_95 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20]) : _shfl_xor_94)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93) : _shfl_xor_95)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_2 = (long long)key2_8 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 256])), "f"(a0[4]), "f"(a0[6]), "f"(a0[20]), "f"(a0[22]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 384])), "f"(a1[4]), "f"(a1[6]), "f"(a1[20]), "f"(a1[22]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21])), 8);
                float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97)), 8);
                float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21])), 8);
                float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101)), 8);
#endif
                int key2_9 = keys[3];
                if (key2_9 >= 0) {
#if DKV_NATURAL == 1
                    int row2_3 = rows_1[3];
                    long long base2_3 = (long long)row2_3 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_98 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_99 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21]) : _shfl_xor_98)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97) : _shfl_xor_99)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_102 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_103 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21]) : _shfl_xor_102)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101) : _shfl_xor_103)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_3 = (long long)key2_9 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 256])), "f"(a0[5]), "f"(a0[7]), "f"(a0[21]), "f"(a0[23]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 384])), "f"(a1[5]), "f"(a1[7]), "f"(a1[21]), "f"(a1[23]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24])), 8);
                float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105)), 8);
                float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24])), 8);
                float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109)), 8);
#endif
                int key2_10 = keys[4];
                if (key2_10 >= 0) {
#if DKV_NATURAL == 1
                    int row2_4 = rows_1[4];
                    long long base2_4 = (long long)row2_4 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_106 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_107 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24]) : _shfl_xor_106)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105) : _shfl_xor_107)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_110 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_111 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24]) : _shfl_xor_110)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109) : _shfl_xor_111)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_4 = (long long)key2_10 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 256])), "f"(a0[8]), "f"(a0[10]), "f"(a0[24]), "f"(a0[26]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 384])), "f"(a1[8]), "f"(a1[10]), "f"(a1[24]), "f"(a1[26]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25])), 8);
                float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113)), 8);
                float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25])), 8);
                float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117)), 8);
#endif
                int key2_11 = keys[5];
                if (key2_11 >= 0) {
#if DKV_NATURAL == 1
                    int row2_5 = rows_1[5];
                    long long base2_5 = (long long)row2_5 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_114 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_115 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25]) : _shfl_xor_114)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113) : _shfl_xor_115)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_118 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_119 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25]) : _shfl_xor_118)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117) : _shfl_xor_119)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_5 = (long long)key2_11 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 256])), "f"(a0[9]), "f"(a0[11]), "f"(a0[25]), "f"(a0[27]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 384])), "f"(a1[9]), "f"(a1[11]), "f"(a1[25]), "f"(a1[27]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28])), 8);
                float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121)), 8);
                float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28])), 8);
                float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125)), 8);
#endif
                int key2_12 = keys[6];
                if (key2_12 >= 0) {
#if DKV_NATURAL == 1
                    int row2_6 = rows_1[6];
                    long long base2_6 = (long long)row2_6 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_122 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_123 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28]) : _shfl_xor_122)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121) : _shfl_xor_123)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_126 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_127 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28]) : _shfl_xor_126)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125) : _shfl_xor_127)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_6 = (long long)key2_12 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 256])), "f"(a0[12]), "f"(a0[14]), "f"(a0[28]), "f"(a0[30]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 384])), "f"(a1[12]), "f"(a1[14]), "f"(a1[28]), "f"(a1[30]), "l"(0x14F0000000000000ULL) : "memory");
#endif
                }
#if DKV_NATURAL == 1
                float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29])), 8);
                float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129)), 8);
                float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29])), 8);
                float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133)), 8);
#endif
                int key2_13 = keys[7];
                if (key2_13 >= 0) {
#if DKV_NATURAL == 1
                    int row2_7 = rows_1[7];
                    long long base2_7 = (long long)row2_7 * (long long)dkv_stride + (long long)npos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 256])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_130 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_131 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29]) : _shfl_xor_130)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129) : _shfl_xor_131)), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 384])), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_134 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]))), "f"((((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_135 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132))), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29]) : _shfl_xor_134)), "f"((((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133) : _shfl_xor_135)), "l"(0x14F0000000000000ULL) : "memory");
#else
                    long long base2_7 = (long long)key2_13 * 512 + (long long)lpos;
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 256])), "f"(a0[13]), "f"(a0[15]), "f"(a0[29]), "f"(a0[31]), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile("red.global.add.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 384])), "f"(a1[13]), "f"(a1[15]), "f"(a1[29]), "f"(a1[31]), "l"(0x14F0000000000000ULL) : "memory");
#endif
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
                    if (i_3 == 0) {
                        mbarrier_wait(k_full_addr, 0);
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
                    int _parity_dp_free = (int)((unsigned int)i_3 % 2) ^ 1;
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
                    int _parity_p_full = (int)((unsigned int)i_3 % 2);
                    mbarrier_wait(p_full_addr, _parity_p_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (i_3 > 0) {
                        unsigned int _parity_dkv_b_drained = (unsigned int)(i_3 - 1) % 2;
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
                    int _parity_ds_full = (int)((unsigned int)i_3 % 2);
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
                    int _parity_dkv_a_drained = (int)((unsigned int)i_3 % 2);
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
                        int _parity_dkr_drained = (int)((unsigned int)i_3 % 2);
                        mbarrier_wait(dkr_drained_addr, _parity_dkr_drained);
                        unsigned int _parity_k_full = (unsigned int)(i_3 + 1) % 2;
                        mbarrier_wait(k_full_addr, _parity_k_full);
                        unsigned int _parity_s_free = (unsigned int)(i_3 + 1) % 2 ^ 1;
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
            stats[lane_0_2] = lse[token_1 * 64 + lane_0_2] * 1.442695f;
            stats[32 + lane_0_2] = lse[token_1 * 64 + 32 + lane_0_2] * 1.442695f;
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
                            best = pos;
                        }
                        if (_vec_load_0[1] >= 0 && _vec_load_0[1] < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        if (_vec_load_0[2] >= 0 && _vec_load_0[2] < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        if (_vec_load_0[3] >= 0 && _vec_load_0[3] < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    } else {
                        int sv = indices[row_base + (long long)pos];
                        if (sv >= 0 && sv < num_kv && active > pos) {
                            best = pos;
                        }
                        int sv_0 = indices[row_base + (long long)(pos + 1)];
                        if (sv_0 >= 0 && sv_0 < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        int sv_1 = indices[row_base + (long long)(pos + 2)];
                        if (sv_1 >= 0 && sv_1 < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        int sv_2 = indices[row_base + (long long)(pos + 3)];
                        if (sv_2 >= 0 && sv_2 < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    }
                }
                int _warp_redux_3;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_3) : "r"(best));
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
                    int slot_3 = i_4 % 2;
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
                        int clamped = ((position) < (topk - 1) ? (position) : (topk - 1));
                        int value = indices[row_base + (long long)clamped];
                        if (active <= position) {
                            value = -1;
                        }
                        values[0] = value;
                        int clamped_0 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                        int value_1 = indices[row_base + (long long)clamped_0];
                        if (active <= position + 1) {
                            value_1 = -1;
                        }
                        values[1] = value_1;
                        int clamped_2 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                        int value_3 = indices[row_base + (long long)clamped_2];
                        if (active <= position + 2) {
                            value_3 = -1;
                        }
                        values[2] = value_3;
                        int clamped_4 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                        int value_5 = indices[row_base + (long long)clamped_4];
                        if (active <= position + 3) {
                            value_5 = -1;
                        }
                        values[3] = value_5;
                        int clamped_6 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                        int value_7 = indices[row_base + (long long)clamped_6];
                        if (active <= position + 4) {
                            value_7 = -1;
                        }
                        values[4] = value_7;
                        int clamped_8 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                        int value_9 = indices[row_base + (long long)clamped_8];
                        if (active <= position + 5) {
                            value_9 = -1;
                        }
                        values[5] = value_9;
                        int clamped_10 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                        int value_11 = indices[row_base + (long long)clamped_10];
                        if (active <= position + 6) {
                            value_11 = -1;
                        }
                        values[6] = value_11;
                        int clamped_12 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                        int value_13 = indices[row_base + (long long)clamped_12];
                        if (active <= position + 7) {
                            value_13 = -1;
                        }
                        values[7] = value_13;
                    }
                    int _stage_idx_free = (int)((unsigned int)i_4 % 2);
                    int _parity_idx_free = (int)((unsigned int)i_4 / 2 % 2) ^ 1;
                    mbarrier_wait(idx_free_addr + (_stage_idx_free) * 8, _parity_idx_free);
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
        // Task spare_main: no body
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
