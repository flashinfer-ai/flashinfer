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

#define CAKE_DSA_H64_TRAIN_NEED_ADD_F32X2
#define CAKE_DSA_H64_TRAIN_NEED_APPROX_EXP2
#define CAKE_DSA_H64_TRAIN_NEED_ELECT_SYNC
#define CAKE_DSA_H64_TRAIN_NEED_FMA_F32X2
#define CAKE_DSA_H64_TRAIN_NEED_MAX_NOFTZ
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_ARRIVE
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_ARRIVE_EXPECT_TX
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_INIT
#define CAKE_DSA_H64_TRAIN_NEED_MBARRIER_WAIT
#define CAKE_DSA_H64_TRAIN_NEED_MUL_F32X2_NOFTZ_INPLACE
#define CAKE_DSA_H64_TRAIN_NEED_TCGEN05_COMMIT
#define CAKE_DSA_H64_TRAIN_NEED_TMA_STORE_4D
#define CAKE_DSA_H64_TRAIN_NEED_TMEM_ST_X32_F32
#include "cake_dsa_h64_train_device.cuh"

#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_46cdc056c891876ed545(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int derive_length, float scale_log2, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ indices_out, int num_docs, int search_iters, int causal, int convert)
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
    #define q_full_addr (mbar_base + 0)
    #define q_copied_addr (mbar_base + 8)
    #define clc_full_addr (mbar_base + 16)
    #define clc_empty_addr (mbar_base + 24)
    #define len_ready_addr (mbar_base + 32)
    #define qk_done_addr (mbar_base + 48)
    #define sv_done_addr (mbar_base + 72)
    #define kv_ready_addr (mbar_base + 96)
    #define valid_ready_addr (mbar_base + 144)
    #define valid_free_addr (mbar_base + 168)
    #define p_free_addr (mbar_base + 192)
    #define so_ready_addr (mbar_base + 200)
    #define o_written_addr (mbar_base + 208)
    #define o_written_waited_addr (mbar_base + 216)

    // Kernel setup ops
    const int qko_addr = smem + 1024;
    const int k_dual_addr = smem + 1024;
    const int v_addr = smem + 1024;
    float* exchange = reinterpret_cast<float*>(smem_raw + 197632);
    const int exchange_addr = smem + 197632;
    const int probability_addr = smem + 214016;
    uint8_t* validity = reinterpret_cast<uint8_t*>(smem_raw + 222208);
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    float* stats = reinterpret_cast<float*>(smem_raw + 222336);
    const int clc_payload_addr = smem + 223360;
    int* len_cells = reinterpret_cast<int*>(smem_raw + 223408);
    int* bound_cells = reinterpret_cast<int*>(smem_raw + 223416);
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'clc_pipeline' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_copied: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // clc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // clc_empty: 1 barriers, init_count=282
            mbarrier_init(smem + 24, 282);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // len_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'score_pipe' ---
            // qk_done: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sv_done: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // kv_ready: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // valid_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            mbarrier_init(smem + 160, 8);
            // valid_free: 3 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // p_free: 1 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            // so_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'clc_pipeline' ---
            // o_written: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // o_written_waited: 1 barriers, init_count=4
            mbarrier_init(smem + 216, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 2) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: softmax ----
    if (warp <= 3) {
        { // softmax_main
            int query = blockIdx.x;
            unsigned int outer_phase = 0;
            int cursor = 0;
            #pragma unroll 1
            for (int outer = 0; outer < num_queries; outer++) {
                int active = topk;
                if (derive_length != 0) {
                    int len_slot = outer & 1;
                    unsigned int len_phase = outer >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot) * 8, len_phase);
                    int _max_1 = ((len_cells[len_slot]) > (0) ? (len_cells[len_slot]) : (0));
                    int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
                    active = _min_0;
                } else if (has_topk_length != 0) {
                    int _max_3 = ((topk_length[query]) > (0) ? (topk_length[query]) : (0));
                    int _min_2 = ((_max_3) < (topk) ? (_max_3) : (topk));
                    active = _min_2;
                }
                int active_0 = active;
                int blocks = (((active_0 + 63) / 64) > (2) ? ((active_0 + 63) / 64) : (2));
                int row = warp * 32 + lane;
                unsigned int tmrow = tmem_tmem;
                float mi = -1e+30f;
                float li = 0.0f;
                float real_mi = -CUDART_INF_F;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                mbarrier_arrive(o_written_addr);
                #pragma unroll 1
                for (int block = 0; block < blocks; block++) {
                    int slot = cursor % 3;
                    unsigned int phase = cursor / 3 & 1;
                    asm volatile("barrier.sync %0, 64;" :: "r"(1 + (warp & 1)) : "memory");
                    mbarrier_wait(qk_done_addr + (slot) * 8, phase);
                    mbarrier_wait(valid_ready_addr + (slot) * 8, phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float p[32];
                    float peer[32];
                    if (warp < 2) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31])
                            : "r"(tmrow + 384));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 416));
                    } else {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(peer[0]), "=f"(peer[1]), "=f"(peer[2]), "=f"(peer[3]), "=f"(peer[4]), "=f"(peer[5]), "=f"(peer[6]), "=f"(peer[7]), "=f"(peer[8]), "=f"(peer[9]), "=f"(peer[10]), "=f"(peer[11]), "=f"(peer[12]), "=f"(peer[13]), "=f"(peer[14]), "=f"(peer[15]), "=f"(peer[16]), "=f"(peer[17]), "=f"(peer[18]), "=f"(peer[19]), "=f"(peer[20]), "=f"(peer[21]), "=f"(peer[22]), "=f"(peer[23]), "=f"(peer[24]), "=f"(peer[25]), "=f"(peer[26]), "=f"(peer[27]), "=f"(peer[28]), "=f"(peer[29]), "=f"(peer[30]), "=f"(peer[31])
                            : "r"(tmrow + 384));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]), "=f"(p[4]), "=f"(p[5]), "=f"(p[6]), "=f"(p[7]), "=f"(p[8]), "=f"(p[9]), "=f"(p[10]), "=f"(p[11]), "=f"(p[12]), "=f"(p[13]), "=f"(p[14]), "=f"(p[15]), "=f"(p[16]), "=f"(p[17]), "=f"(p[18]), "=f"(p[19]), "=f"(p[20]), "=f"(p[21]), "=f"(p[22]), "=f"(p[23]), "=f"(p[24]), "=f"(p[25]), "=f"(p[26]), "=f"(p[27]), "=f"(p[28]), "=f"(p[29]), "=f"(p[30]), "=f"(p[31])
                            : "r"(tmrow + 416));
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_free_addr);
                    unsigned int p_mask = validity32[slot * 2 + warp / 2];
                    if ((p_mask & 1) == 0) {
                        p[0] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 1 & 1) == 0) {
                        p[1] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 2 & 1) == 0) {
                        p[2] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 3 & 1) == 0) {
                        p[3] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 4 & 1) == 0) {
                        p[4] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 5 & 1) == 0) {
                        p[5] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 6 & 1) == 0) {
                        p[6] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 7 & 1) == 0) {
                        p[7] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 8 & 1) == 0) {
                        p[8] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 9 & 1) == 0) {
                        p[9] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 10 & 1) == 0) {
                        p[10] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 11 & 1) == 0) {
                        p[11] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 12 & 1) == 0) {
                        p[12] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 13 & 1) == 0) {
                        p[13] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 14 & 1) == 0) {
                        p[14] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 15 & 1) == 0) {
                        p[15] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 16 & 1) == 0) {
                        p[16] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 17 & 1) == 0) {
                        p[17] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 18 & 1) == 0) {
                        p[18] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 19 & 1) == 0) {
                        p[19] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 20 & 1) == 0) {
                        p[20] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 21 & 1) == 0) {
                        p[21] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 22 & 1) == 0) {
                        p[22] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 23 & 1) == 0) {
                        p[23] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 24 & 1) == 0) {
                        p[24] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 25 & 1) == 0) {
                        p[25] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 26 & 1) == 0) {
                        p[26] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 27 & 1) == 0) {
                        p[27] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 28 & 1) == 0) {
                        p[28] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 29 & 1) == 0) {
                        p[29] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 30 & 1) == 0) {
                        p[30] = -CUDART_INF_F;
                    }
                    if ((p_mask >> 31 & 1) == 0) {
                        p[31] = -CUDART_INF_F;
                    }
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + lane * 4) * 4)), "f"(peer[0]), "f"(peer[1]), "f"(peer[2]), "f"(peer[3]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 128 + lane * 4) * 4)), "f"(peer[4]), "f"(peer[5]), "f"(peer[6]), "f"(peer[7]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 256 + lane * 4) * 4)), "f"(peer[8]), "f"(peer[9]), "f"(peer[10]), "f"(peer[11]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 384 + lane * 4) * 4)), "f"(peer[12]), "f"(peer[13]), "f"(peer[14]), "f"(peer[15]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 512 + lane * 4) * 4)), "f"(peer[16]), "f"(peer[17]), "f"(peer[18]), "f"(peer[19]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 640 + lane * 4) * 4)), "f"(peer[20]), "f"(peer[21]), "f"(peer[22]), "f"(peer[23]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 768 + lane * 4) * 4)), "f"(peer[24]), "f"(peer[25]), "f"(peer[26]), "f"(peer[27]) : "memory");
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(exchange_addr + (unsigned int)(((warp ^ 2) * 1024 + 896 + lane * 4) * 4)), "f"(peer[28]), "f"(peer[29]), "f"(peer[30]), "f"(peer[31]) : "memory");
                    asm volatile("barrier.sync %0, 64;" :: "r"(1 + (warp & 1)) : "memory");
                    float smem_regs[4];
                    #pragma unroll
                    for (int _lr = 0; _lr < 4; _lr++) {
                        smem_regs[_lr] = exchange[warp * 1024 + lane * 4 + _lr];
                    }
                    float2 pair_sum = add_f32x2(make_float2(p[0], p[1]), make_float2(smem_regs[0], smem_regs[1]));
                    p[0] = pair_sum.x;
                    p[1] = pair_sum.y;
                    float2 pair_sum_0 = add_f32x2(make_float2(p[2], p[3]), make_float2(smem_regs[2], smem_regs[3]));
                    p[2] = pair_sum_0.x;
                    p[3] = pair_sum_0.y;
                    float smem_regs_1[4];
                    #pragma unroll
                    for (int _lr_1 = 0; _lr_1 < 4; _lr_1++) {
                        smem_regs_1[_lr_1] = exchange[warp * 1024 + 128 + lane * 4 + _lr_1];
                    }
                    float2 pair_sum_2 = add_f32x2(make_float2(p[4], p[5]), make_float2(smem_regs_1[0], smem_regs_1[1]));
                    p[4] = pair_sum_2.x;
                    p[5] = pair_sum_2.y;
                    float2 pair_sum_3 = add_f32x2(make_float2(p[6], p[7]), make_float2(smem_regs_1[2], smem_regs_1[3]));
                    p[6] = pair_sum_3.x;
                    p[7] = pair_sum_3.y;
                    float smem_regs_4[4];
                    #pragma unroll
                    for (int _lr_2 = 0; _lr_2 < 4; _lr_2++) {
                        smem_regs_4[_lr_2] = exchange[warp * 1024 + 256 + lane * 4 + _lr_2];
                    }
                    float2 pair_sum_5 = add_f32x2(make_float2(p[8], p[9]), make_float2(smem_regs_4[0], smem_regs_4[1]));
                    p[8] = pair_sum_5.x;
                    p[9] = pair_sum_5.y;
                    float2 pair_sum_6 = add_f32x2(make_float2(p[10], p[11]), make_float2(smem_regs_4[2], smem_regs_4[3]));
                    p[10] = pair_sum_6.x;
                    p[11] = pair_sum_6.y;
                    float smem_regs_7[4];
                    #pragma unroll
                    for (int _lr_3 = 0; _lr_3 < 4; _lr_3++) {
                        smem_regs_7[_lr_3] = exchange[warp * 1024 + 384 + lane * 4 + _lr_3];
                    }
                    float2 pair_sum_8 = add_f32x2(make_float2(p[12], p[13]), make_float2(smem_regs_7[0], smem_regs_7[1]));
                    p[12] = pair_sum_8.x;
                    p[13] = pair_sum_8.y;
                    float2 pair_sum_9 = add_f32x2(make_float2(p[14], p[15]), make_float2(smem_regs_7[2], smem_regs_7[3]));
                    p[14] = pair_sum_9.x;
                    p[15] = pair_sum_9.y;
                    float smem_regs_10[4];
                    #pragma unroll
                    for (int _lr_4 = 0; _lr_4 < 4; _lr_4++) {
                        smem_regs_10[_lr_4] = exchange[warp * 1024 + 512 + lane * 4 + _lr_4];
                    }
                    float2 pair_sum_11 = add_f32x2(make_float2(p[16], p[17]), make_float2(smem_regs_10[0], smem_regs_10[1]));
                    p[16] = pair_sum_11.x;
                    p[17] = pair_sum_11.y;
                    float2 pair_sum_12 = add_f32x2(make_float2(p[18], p[19]), make_float2(smem_regs_10[2], smem_regs_10[3]));
                    p[18] = pair_sum_12.x;
                    p[19] = pair_sum_12.y;
                    float smem_regs_13[4];
                    #pragma unroll
                    for (int _lr_5 = 0; _lr_5 < 4; _lr_5++) {
                        smem_regs_13[_lr_5] = exchange[warp * 1024 + 640 + lane * 4 + _lr_5];
                    }
                    float2 pair_sum_14 = add_f32x2(make_float2(p[20], p[21]), make_float2(smem_regs_13[0], smem_regs_13[1]));
                    p[20] = pair_sum_14.x;
                    p[21] = pair_sum_14.y;
                    float2 pair_sum_15 = add_f32x2(make_float2(p[22], p[23]), make_float2(smem_regs_13[2], smem_regs_13[3]));
                    p[22] = pair_sum_15.x;
                    p[23] = pair_sum_15.y;
                    float smem_regs_16[4];
                    #pragma unroll
                    for (int _lr_6 = 0; _lr_6 < 4; _lr_6++) {
                        smem_regs_16[_lr_6] = exchange[warp * 1024 + 768 + lane * 4 + _lr_6];
                    }
                    float2 pair_sum_17 = add_f32x2(make_float2(p[24], p[25]), make_float2(smem_regs_16[0], smem_regs_16[1]));
                    p[24] = pair_sum_17.x;
                    p[25] = pair_sum_17.y;
                    float2 pair_sum_18 = add_f32x2(make_float2(p[26], p[27]), make_float2(smem_regs_16[2], smem_regs_16[3]));
                    p[26] = pair_sum_18.x;
                    p[27] = pair_sum_18.y;
                    float smem_regs_19[4];
                    #pragma unroll
                    for (int _lr_7 = 0; _lr_7 < 4; _lr_7++) {
                        smem_regs_19[_lr_7] = exchange[warp * 1024 + 896 + lane * 4 + _lr_7];
                    }
                    float2 pair_sum_20 = add_f32x2(make_float2(p[28], p[29]), make_float2(smem_regs_19[0], smem_regs_19[1]));
                    p[28] = pair_sum_20.x;
                    p[29] = pair_sum_20.y;
                    float2 pair_sum_21 = add_f32x2(make_float2(p[30], p[31]), make_float2(smem_regs_19[2], smem_regs_19[3]));
                    p[30] = pair_sum_21.x;
                    p[31] = pair_sum_21.y;
                    mbarrier_arrive(valid_free_addr + (slot) * 8);
                    float local_max0 = -CUDART_INF_F;
                    float local_max1 = -CUDART_INF_F;
                    local_max0 = max_noftz(local_max0, p[0]);
                    local_max0 = max_noftz(local_max0, p[1]);
                    local_max0 = max_noftz(local_max0, p[2]);
                    local_max0 = max_noftz(local_max0, p[3]);
                    local_max0 = max_noftz(local_max0, p[4]);
                    local_max0 = max_noftz(local_max0, p[5]);
                    local_max0 = max_noftz(local_max0, p[6]);
                    local_max0 = max_noftz(local_max0, p[7]);
                    local_max0 = max_noftz(local_max0, p[8]);
                    local_max0 = max_noftz(local_max0, p[9]);
                    local_max0 = max_noftz(local_max0, p[10]);
                    local_max0 = max_noftz(local_max0, p[11]);
                    local_max0 = max_noftz(local_max0, p[12]);
                    local_max0 = max_noftz(local_max0, p[13]);
                    local_max0 = max_noftz(local_max0, p[14]);
                    local_max0 = max_noftz(local_max0, p[15]);
                    local_max1 = max_noftz(local_max1, p[16]);
                    local_max1 = max_noftz(local_max1, p[17]);
                    local_max1 = max_noftz(local_max1, p[18]);
                    local_max1 = max_noftz(local_max1, p[19]);
                    local_max1 = max_noftz(local_max1, p[20]);
                    local_max1 = max_noftz(local_max1, p[21]);
                    local_max1 = max_noftz(local_max1, p[22]);
                    local_max1 = max_noftz(local_max1, p[23]);
                    local_max1 = max_noftz(local_max1, p[24]);
                    local_max1 = max_noftz(local_max1, p[25]);
                    local_max1 = max_noftz(local_max1, p[26]);
                    local_max1 = max_noftz(local_max1, p[27]);
                    local_max1 = max_noftz(local_max1, p[28]);
                    local_max1 = max_noftz(local_max1, p[29]);
                    local_max1 = max_noftz(local_max1, p[30]);
                    local_max1 = max_noftz(local_max1, p[31]);
                    float cur_max = max_noftz(local_max0, local_max1) * scale_log2;
                    int rescale = 0;
                    stats[row] = cur_max;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    cur_max = max_noftz(cur_max, stats[row ^ 64]);
                    real_mi = max_noftz(real_mi, cur_max);
                    int _warp_vote_4 = __any_sync(0xFFFFFFFF, cur_max - mi > 6.0f);
                    rescale = _warp_vote_4;
                    float factor = 1.0f;
                    float next_max = mi;
                    if (rescale != 0) {
                        next_max = max_noftz(cur_max, mi);
                        float _exp2_5 = approx_exp2(mi - next_max);
                        factor = _exp2_5;
                    }
                    mi = next_max;
                    const float2 _fma_b2_6 = {scale_log2, scale_log2};
                    const float2 _fma_c2_7 = {-mi, -mi};
                    float2 _fma_pair_8 = fma_f32x2(make_float2(p[0], p[1]), _fma_b2_6, _fma_c2_7);
                    p[0] = _fma_pair_8.x;
                    p[1] = _fma_pair_8.y;
                    float2 _fma_pair_9 = fma_f32x2(make_float2(p[2], p[3]), _fma_b2_6, _fma_c2_7);
                    p[2] = _fma_pair_9.x;
                    p[3] = _fma_pair_9.y;
                    float2 _fma_pair_10 = fma_f32x2(make_float2(p[4], p[5]), _fma_b2_6, _fma_c2_7);
                    p[4] = _fma_pair_10.x;
                    p[5] = _fma_pair_10.y;
                    float2 _fma_pair_11 = fma_f32x2(make_float2(p[6], p[7]), _fma_b2_6, _fma_c2_7);
                    p[6] = _fma_pair_11.x;
                    p[7] = _fma_pair_11.y;
                    float2 _fma_pair_12 = fma_f32x2(make_float2(p[8], p[9]), _fma_b2_6, _fma_c2_7);
                    p[8] = _fma_pair_12.x;
                    p[9] = _fma_pair_12.y;
                    float2 _fma_pair_13 = fma_f32x2(make_float2(p[10], p[11]), _fma_b2_6, _fma_c2_7);
                    p[10] = _fma_pair_13.x;
                    p[11] = _fma_pair_13.y;
                    float2 _fma_pair_14 = fma_f32x2(make_float2(p[12], p[13]), _fma_b2_6, _fma_c2_7);
                    p[12] = _fma_pair_14.x;
                    p[13] = _fma_pair_14.y;
                    float2 _fma_pair_15 = fma_f32x2(make_float2(p[14], p[15]), _fma_b2_6, _fma_c2_7);
                    p[14] = _fma_pair_15.x;
                    p[15] = _fma_pair_15.y;
                    float2 _fma_pair_16 = fma_f32x2(make_float2(p[16], p[17]), _fma_b2_6, _fma_c2_7);
                    p[16] = _fma_pair_16.x;
                    p[17] = _fma_pair_16.y;
                    float2 _fma_pair_17 = fma_f32x2(make_float2(p[18], p[19]), _fma_b2_6, _fma_c2_7);
                    p[18] = _fma_pair_17.x;
                    p[19] = _fma_pair_17.y;
                    float2 _fma_pair_18 = fma_f32x2(make_float2(p[20], p[21]), _fma_b2_6, _fma_c2_7);
                    p[20] = _fma_pair_18.x;
                    p[21] = _fma_pair_18.y;
                    float2 _fma_pair_19 = fma_f32x2(make_float2(p[22], p[23]), _fma_b2_6, _fma_c2_7);
                    p[22] = _fma_pair_19.x;
                    p[23] = _fma_pair_19.y;
                    float2 _fma_pair_20 = fma_f32x2(make_float2(p[24], p[25]), _fma_b2_6, _fma_c2_7);
                    p[24] = _fma_pair_20.x;
                    p[25] = _fma_pair_20.y;
                    float2 _fma_pair_21 = fma_f32x2(make_float2(p[26], p[27]), _fma_b2_6, _fma_c2_7);
                    p[26] = _fma_pair_21.x;
                    p[27] = _fma_pair_21.y;
                    float2 _fma_pair_22 = fma_f32x2(make_float2(p[28], p[29]), _fma_b2_6, _fma_c2_7);
                    p[28] = _fma_pair_22.x;
                    p[29] = _fma_pair_22.y;
                    float2 _fma_pair_23 = fma_f32x2(make_float2(p[30], p[31]), _fma_b2_6, _fma_c2_7);
                    p[30] = _fma_pair_23.x;
                    p[31] = _fma_pair_23.y;
                    #pragma unroll
                    for (int _le = 0; _le < 32; _le++) {
                        p[_le] = approx_exp2(p[_le]);
                    }
                    float2 sum_pair = make_float2(0.0f, 0.0f);
                    sum_pair = add_f32x2(sum_pair, make_float2(p[0], p[1]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[2], p[3]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[4], p[5]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[6], p[7]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[8], p[9]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[10], p[11]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[12], p[13]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[14], p[15]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[16], p[17]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[18], p[19]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[20], p[21]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[22], p[23]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[24], p[25]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[26], p[27]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[28], p[29]));
                    sum_pair = add_f32x2(sum_pair, make_float2(p[30], p[31]));
                    float cur_sum = sum_pair.x + sum_pair.y;
                    li = __fmaf_rn(li, factor, cur_sum);
                    if (block > 0) {
                        int prev_stage = (cursor - 1) % 3;
                        int prev_phase = (cursor - 1) / 3 & 1;
                        mbarrier_wait(sv_done_addr + (prev_stage) * 8, prev_phase);
                    }
                    unsigned int p_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                        p_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048) * 2)), "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]) : "memory");
                    unsigned int p_bf16_22[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 8], p[_lp*2+1 + 8]));
                        p_bf16_22[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 512) * 2)), "r"(p_bf16_22[0]), "r"(p_bf16_22[1]), "r"(p_bf16_22[2]), "r"(p_bf16_22[3]) : "memory");
                    unsigned int p_bf16_23[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 16], p[_lp*2+1 + 16]));
                        p_bf16_23[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1024) * 2)), "r"(p_bf16_23[0]), "r"(p_bf16_23[1]), "r"(p_bf16_23[2]), "r"(p_bf16_23[3]) : "memory");
                    unsigned int p_bf16_24[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 24], p[_lp*2+1 + 24]));
                        p_bf16_24[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(probability_addr + (unsigned int)((lane * 8 + (warp & 1) * 256 + warp / 2 * 2048 + 1536) * 2)), "r"(p_bf16_24[0]), "r"(p_bf16_24[1]), "r"(p_bf16_24[2]), "r"(p_bf16_24[3]) : "memory");
                    if (block > 0 && rescale != 0) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(tmrow));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_24 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_24);
                        tmem_st_x32_f32(tmrow, _tmem_load_0);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(tmrow + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_25 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_25);
                        tmem_st_x32_f32(tmrow + 32, _tmem_load_1);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(tmrow + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_26 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_26);
                        tmem_st_x32_f32(tmrow + 64, _tmem_load_2);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(tmrow + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_27 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_27);
                        tmem_st_x32_f32(tmrow + 96, _tmem_load_3);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                            : "r"(tmrow + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_28 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_28);
                        tmem_st_x32_f32(tmrow + 128, _tmem_load_4);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(tmrow + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_29 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_29);
                        tmem_st_x32_f32(tmrow + 160, _tmem_load_5);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(tmrow + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_30 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_30);
                        tmem_st_x32_f32(tmrow + 192, _tmem_load_6);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                            : "r"(tmrow + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_31 = {factor, factor};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_31);
                        tmem_st_x32_f32(tmrow + 224, _tmem_load_7);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(so_ready_addr);
                    cursor = cursor + 1;
                }
                mbarrier_wait(o_written_waited_addr, outer_phase);
                if (real_mi == -CUDART_INF_F) {
                    li = 0.0f;
                    mi = -CUDART_INF_F;
                }
                stats[128 + row] = li;
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                li = li + stats[128 + (row ^ 64)];
                if (row < 64) {
                    float _log_32 = logf(li);
                    float cur_lse = __fmaf_rn(mi, 0.6931472f, _log_32);
                    if (real_mi == -CUDART_INF_F) {
                        cur_lse = -CUDART_INF_F;
                    }
                    lse[query * 64 + row] = cur_lse;
                }
                float output_scale = 1.0f / li;
                int have_valid = __any_sync(0xFFFFFFFF, li != 0.0f);
                if (have_valid == 0) {
                    output_scale = 1.0f;
                }
                mbarrier_wait(sv_done_addr + ((cursor - 1) % 3) * 8, (cursor - 1) / 3 & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int o_slot = (cursor + 2) % 3;
                int head = row % 64;
                int query_head = query * 64 + head;
                long long o_row_base = (long long)query_head * 512;
                float ov[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31]), "=f"(ov[32]), "=f"(ov[33]), "=f"(ov[34]), "=f"(ov[35]), "=f"(ov[36]), "=f"(ov[37]), "=f"(ov[38]), "=f"(ov[39]), "=f"(ov[40]), "=f"(ov[41]), "=f"(ov[42]), "=f"(ov[43]), "=f"(ov[44]), "=f"(ov[45]), "=f"(ov[46]), "=f"(ov[47]), "=f"(ov[48]), "=f"(ov[49]), "=f"(ov[50]), "=f"(ov[51]), "=f"(ov[52]), "=f"(ov[53]), "=f"(ov[54]), "=f"(ov[55]), "=f"(ov[56]), "=f"(ov[57]), "=f"(ov[58]), "=f"(ov[59]), "=f"(ov[60]), "=f"(ov[61]), "=f"(ov[62]), "=f"(ov[63])
                        : "r"(tmrow));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov[0] = 0.0f;
                    ov[1] = 0.0f;
                    ov[2] = 0.0f;
                    ov[3] = 0.0f;
                    ov[4] = 0.0f;
                    ov[5] = 0.0f;
                    ov[6] = 0.0f;
                    ov[7] = 0.0f;
                    ov[8] = 0.0f;
                    ov[9] = 0.0f;
                    ov[10] = 0.0f;
                    ov[11] = 0.0f;
                    ov[12] = 0.0f;
                    ov[13] = 0.0f;
                    ov[14] = 0.0f;
                    ov[15] = 0.0f;
                    ov[16] = 0.0f;
                    ov[17] = 0.0f;
                    ov[18] = 0.0f;
                    ov[19] = 0.0f;
                    ov[20] = 0.0f;
                    ov[21] = 0.0f;
                    ov[22] = 0.0f;
                    ov[23] = 0.0f;
                    ov[24] = 0.0f;
                    ov[25] = 0.0f;
                    ov[26] = 0.0f;
                    ov[27] = 0.0f;
                    ov[28] = 0.0f;
                    ov[29] = 0.0f;
                    ov[30] = 0.0f;
                    ov[31] = 0.0f;
                    ov[32] = 0.0f;
                    ov[33] = 0.0f;
                    ov[34] = 0.0f;
                    ov[35] = 0.0f;
                    ov[36] = 0.0f;
                    ov[37] = 0.0f;
                    ov[38] = 0.0f;
                    ov[39] = 0.0f;
                    ov[40] = 0.0f;
                    ov[41] = 0.0f;
                    ov[42] = 0.0f;
                    ov[43] = 0.0f;
                    ov[44] = 0.0f;
                    ov[45] = 0.0f;
                    ov[46] = 0.0f;
                    ov[47] = 0.0f;
                    ov[48] = 0.0f;
                    ov[49] = 0.0f;
                    ov[50] = 0.0f;
                    ov[51] = 0.0f;
                    ov[52] = 0.0f;
                    ov[53] = 0.0f;
                    ov[54] = 0.0f;
                    ov[55] = 0.0f;
                    ov[56] = 0.0f;
                    ov[57] = 0.0f;
                    ov[58] = 0.0f;
                    ov[59] = 0.0f;
                    ov[60] = 0.0f;
                    ov[61] = 0.0f;
                    ov[62] = 0.0f;
                    ov[63] = 0.0f;
                }
                const float2 _scale2_33 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov)[_ls], _scale2_33);
                int out_col = row / 64 * 128;
                float residual[16];
                unsigned int ov_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 0], ov[_lp*2+1 + 0]));
                    ov_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16[_pair]) & 0xffff0000u);
                }
                residual[0] = ov[0] - _unpack_bf16_0[0];
                residual[1] = ov[1] - _unpack_bf16_0[1];
                residual[2] = ov[2] - _unpack_bf16_0[2];
                residual[3] = ov[3] - _unpack_bf16_0[3];
                residual[4] = ov[4] - _unpack_bf16_0[4];
                residual[5] = ov[5] - _unpack_bf16_0[5];
                residual[6] = ov[6] - _unpack_bf16_0[6];
                residual[7] = ov[7] - _unpack_bf16_0[7];
                int byte_offset = o_slot * 65536 + out_col * 128 + row % 64 * 128;
                int swizzled_offset = byte_offset ^ (byte_offset >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset), "r"(ov_bf16[0]), "r"(ov_bf16[1]), "r"(ov_bf16[2]), "r"(ov_bf16[3]) : "memory");
                unsigned int ov_bf16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 8], ov[_lp*2+1 + 8]));
                    ov_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_1[_pair]) & 0xffff0000u);
                }
                residual[8] = ov[8] - _unpack_bf16_1[0];
                residual[9] = ov[9] - _unpack_bf16_1[1];
                residual[10] = ov[10] - _unpack_bf16_1[2];
                residual[11] = ov[11] - _unpack_bf16_1[3];
                residual[12] = ov[12] - _unpack_bf16_1[4];
                residual[13] = ov[13] - _unpack_bf16_1[5];
                residual[14] = ov[14] - _unpack_bf16_1[6];
                residual[15] = ov[15] - _unpack_bf16_1[7];
                int byte_offset_2 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 16;
                int swizzled_offset_3 = byte_offset_2 ^ (byte_offset_2 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_3), "r"(ov_bf16_1[0]), "r"(ov_bf16_1[1]), "r"(ov_bf16_1[2]), "r"(ov_bf16_1[3]) : "memory");
                unsigned int _packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 0], residual[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 2], residual[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 4], residual[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 6], residual[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 8], residual[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 10], residual[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 12], residual[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 14], residual[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_34 = reinterpret_cast<const unsigned*>(_packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col))), "r"(_raw_stv8_34[0]), "r"(_raw_stv8_34[1]), "r"(_raw_stv8_34[2]), "r"(_raw_stv8_34[3]), "r"(_raw_stv8_34[4]), "r"(_raw_stv8_34[5]), "r"(_raw_stv8_34[6]), "r"(_raw_stv8_34[7]) : "memory");
                }
                float residual_4[16];
                unsigned int ov_bf16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 16], ov[_lp*2+1 + 16]));
                    ov_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_5[_pair]) & 0xffff0000u);
                }
                residual_4[0] = ov[16] - _unpack_bf16_2[0];
                residual_4[1] = ov[17] - _unpack_bf16_2[1];
                residual_4[2] = ov[18] - _unpack_bf16_2[2];
                residual_4[3] = ov[19] - _unpack_bf16_2[3];
                residual_4[4] = ov[20] - _unpack_bf16_2[4];
                residual_4[5] = ov[21] - _unpack_bf16_2[5];
                residual_4[6] = ov[22] - _unpack_bf16_2[6];
                residual_4[7] = ov[23] - _unpack_bf16_2[7];
                int byte_offset_6 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 32;
                int swizzled_offset_7 = byte_offset_6 ^ (byte_offset_6 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_7), "r"(ov_bf16_5[0]), "r"(ov_bf16_5[1]), "r"(ov_bf16_5[2]), "r"(ov_bf16_5[3]) : "memory");
                unsigned int ov_bf16_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 24], ov[_lp*2+1 + 24]));
                    ov_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_8[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_8[_pair]) & 0xffff0000u);
                }
                residual_4[8] = ov[24] - _unpack_bf16_3[0];
                residual_4[9] = ov[25] - _unpack_bf16_3[1];
                residual_4[10] = ov[26] - _unpack_bf16_3[2];
                residual_4[11] = ov[27] - _unpack_bf16_3[3];
                residual_4[12] = ov[28] - _unpack_bf16_3[4];
                residual_4[13] = ov[29] - _unpack_bf16_3[5];
                residual_4[14] = ov[30] - _unpack_bf16_3[6];
                residual_4[15] = ov[31] - _unpack_bf16_3[7];
                int byte_offset_9 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 48;
                int swizzled_offset_10 = byte_offset_9 ^ (byte_offset_9 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_10), "r"(ov_bf16_8[0]), "r"(ov_bf16_8[1]), "r"(ov_bf16_8[2]), "r"(ov_bf16_8[3]) : "memory");
                unsigned int _packed_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 0], residual_4[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 2], residual_4[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 4], residual_4[_lp*2+1 + 4]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 6], residual_4[_lp*2+1 + 6]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 8], residual_4[_lp*2+1 + 8]));
                    _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 10], residual_4[_lp*2+1 + 10]));
                    _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 12], residual_4[_lp*2+1 + 12]));
                    _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_4[_lp*2 + 14], residual_4[_lp*2+1 + 14]));
                    _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_35 = reinterpret_cast<const unsigned*>(_packed_1);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 16))), "r"(_raw_stv8_35[0]), "r"(_raw_stv8_35[1]), "r"(_raw_stv8_35[2]), "r"(_raw_stv8_35[3]), "r"(_raw_stv8_35[4]), "r"(_raw_stv8_35[5]), "r"(_raw_stv8_35[6]), "r"(_raw_stv8_35[7]) : "memory");
                }
                float residual_11[16];
                unsigned int ov_bf16_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 32], ov[_lp*2+1 + 32]));
                    ov_bf16_12[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_12[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_12[_pair]) & 0xffff0000u);
                }
                residual_11[0] = ov[32] - _unpack_bf16_4[0];
                residual_11[1] = ov[33] - _unpack_bf16_4[1];
                residual_11[2] = ov[34] - _unpack_bf16_4[2];
                residual_11[3] = ov[35] - _unpack_bf16_4[3];
                residual_11[4] = ov[36] - _unpack_bf16_4[4];
                residual_11[5] = ov[37] - _unpack_bf16_4[5];
                residual_11[6] = ov[38] - _unpack_bf16_4[6];
                residual_11[7] = ov[39] - _unpack_bf16_4[7];
                int byte_offset_13 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 64;
                int swizzled_offset_14 = byte_offset_13 ^ (byte_offset_13 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_14), "r"(ov_bf16_12[0]), "r"(ov_bf16_12[1]), "r"(ov_bf16_12[2]), "r"(ov_bf16_12[3]) : "memory");
                unsigned int ov_bf16_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 40], ov[_lp*2+1 + 40]));
                    ov_bf16_15[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_15[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_15[_pair]) & 0xffff0000u);
                }
                residual_11[8] = ov[40] - _unpack_bf16_5[0];
                residual_11[9] = ov[41] - _unpack_bf16_5[1];
                residual_11[10] = ov[42] - _unpack_bf16_5[2];
                residual_11[11] = ov[43] - _unpack_bf16_5[3];
                residual_11[12] = ov[44] - _unpack_bf16_5[4];
                residual_11[13] = ov[45] - _unpack_bf16_5[5];
                residual_11[14] = ov[46] - _unpack_bf16_5[6];
                residual_11[15] = ov[47] - _unpack_bf16_5[7];
                int byte_offset_16 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 80;
                int swizzled_offset_17 = byte_offset_16 ^ (byte_offset_16 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_17), "r"(ov_bf16_15[0]), "r"(ov_bf16_15[1]), "r"(ov_bf16_15[2]), "r"(ov_bf16_15[3]) : "memory");
                unsigned int _packed_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 0], residual_11[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 2], residual_11[_lp*2+1 + 2]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 4], residual_11[_lp*2+1 + 4]));
                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 6], residual_11[_lp*2+1 + 6]));
                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 8], residual_11[_lp*2+1 + 8]));
                    _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 10], residual_11[_lp*2+1 + 10]));
                    _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 12], residual_11[_lp*2+1 + 12]));
                    _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_11[_lp*2 + 14], residual_11[_lp*2+1 + 14]));
                    _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_36 = reinterpret_cast<const unsigned*>(_packed_2);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 32))), "r"(_raw_stv8_36[0]), "r"(_raw_stv8_36[1]), "r"(_raw_stv8_36[2]), "r"(_raw_stv8_36[3]), "r"(_raw_stv8_36[4]), "r"(_raw_stv8_36[5]), "r"(_raw_stv8_36[6]), "r"(_raw_stv8_36[7]) : "memory");
                }
                float residual_18[16];
                unsigned int ov_bf16_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 48], ov[_lp*2+1 + 48]));
                    ov_bf16_19[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_19[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_19[_pair]) & 0xffff0000u);
                }
                residual_18[0] = ov[48] - _unpack_bf16_6[0];
                residual_18[1] = ov[49] - _unpack_bf16_6[1];
                residual_18[2] = ov[50] - _unpack_bf16_6[2];
                residual_18[3] = ov[51] - _unpack_bf16_6[3];
                residual_18[4] = ov[52] - _unpack_bf16_6[4];
                residual_18[5] = ov[53] - _unpack_bf16_6[5];
                residual_18[6] = ov[54] - _unpack_bf16_6[6];
                residual_18[7] = ov[55] - _unpack_bf16_6[7];
                int byte_offset_20 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 96;
                int swizzled_offset_21 = byte_offset_20 ^ (byte_offset_20 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_21), "r"(ov_bf16_19[0]), "r"(ov_bf16_19[1]), "r"(ov_bf16_19[2]), "r"(ov_bf16_19[3]) : "memory");
                unsigned int ov_bf16_22[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 56], ov[_lp*2+1 + 56]));
                    ov_bf16_22[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_bf16_22[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_bf16_22[_pair]) & 0xffff0000u);
                }
                residual_18[8] = ov[56] - _unpack_bf16_7[0];
                residual_18[9] = ov[57] - _unpack_bf16_7[1];
                residual_18[10] = ov[58] - _unpack_bf16_7[2];
                residual_18[11] = ov[59] - _unpack_bf16_7[3];
                residual_18[12] = ov[60] - _unpack_bf16_7[4];
                residual_18[13] = ov[61] - _unpack_bf16_7[5];
                residual_18[14] = ov[62] - _unpack_bf16_7[6];
                residual_18[15] = ov[63] - _unpack_bf16_7[7];
                int byte_offset_23 = o_slot * 65536 + out_col * 128 + row % 64 * 128 + 112;
                int swizzled_offset_24 = byte_offset_23 ^ (byte_offset_23 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_24), "r"(ov_bf16_22[0]), "r"(ov_bf16_22[1]), "r"(ov_bf16_22[2]), "r"(ov_bf16_22[3]) : "memory");
                unsigned int _packed_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 0], residual_18[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 2], residual_18[_lp*2+1 + 2]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 4], residual_18[_lp*2+1 + 4]));
                    _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 6], residual_18[_lp*2+1 + 6]));
                    _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 8], residual_18[_lp*2+1 + 8]));
                    _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 10], residual_18[_lp*2+1 + 10]));
                    _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 12], residual_18[_lp*2+1 + 12]));
                    _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_18[_lp*2 + 14], residual_18[_lp*2+1 + 14]));
                    _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_37 = reinterpret_cast<const unsigned*>(_packed_3);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col + 48))), "r"(_raw_stv8_37[0]), "r"(_raw_stv8_37[1]), "r"(_raw_stv8_37[2]), "r"(_raw_stv8_37[3]), "r"(_raw_stv8_37[4]), "r"(_raw_stv8_37[5]), "r"(_raw_stv8_37[6]), "r"(_raw_stv8_37[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk = warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk * 8192));
                    }
                }
                float ov_25[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_25[0]), "=f"(ov_25[1]), "=f"(ov_25[2]), "=f"(ov_25[3]), "=f"(ov_25[4]), "=f"(ov_25[5]), "=f"(ov_25[6]), "=f"(ov_25[7]), "=f"(ov_25[8]), "=f"(ov_25[9]), "=f"(ov_25[10]), "=f"(ov_25[11]), "=f"(ov_25[12]), "=f"(ov_25[13]), "=f"(ov_25[14]), "=f"(ov_25[15]), "=f"(ov_25[16]), "=f"(ov_25[17]), "=f"(ov_25[18]), "=f"(ov_25[19]), "=f"(ov_25[20]), "=f"(ov_25[21]), "=f"(ov_25[22]), "=f"(ov_25[23]), "=f"(ov_25[24]), "=f"(ov_25[25]), "=f"(ov_25[26]), "=f"(ov_25[27]), "=f"(ov_25[28]), "=f"(ov_25[29]), "=f"(ov_25[30]), "=f"(ov_25[31]), "=f"(ov_25[32]), "=f"(ov_25[33]), "=f"(ov_25[34]), "=f"(ov_25[35]), "=f"(ov_25[36]), "=f"(ov_25[37]), "=f"(ov_25[38]), "=f"(ov_25[39]), "=f"(ov_25[40]), "=f"(ov_25[41]), "=f"(ov_25[42]), "=f"(ov_25[43]), "=f"(ov_25[44]), "=f"(ov_25[45]), "=f"(ov_25[46]), "=f"(ov_25[47]), "=f"(ov_25[48]), "=f"(ov_25[49]), "=f"(ov_25[50]), "=f"(ov_25[51]), "=f"(ov_25[52]), "=f"(ov_25[53]), "=f"(ov_25[54]), "=f"(ov_25[55]), "=f"(ov_25[56]), "=f"(ov_25[57]), "=f"(ov_25[58]), "=f"(ov_25[59]), "=f"(ov_25[60]), "=f"(ov_25[61]), "=f"(ov_25[62]), "=f"(ov_25[63])
                        : "r"(tmrow + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_25[0] = 0.0f;
                    ov_25[1] = 0.0f;
                    ov_25[2] = 0.0f;
                    ov_25[3] = 0.0f;
                    ov_25[4] = 0.0f;
                    ov_25[5] = 0.0f;
                    ov_25[6] = 0.0f;
                    ov_25[7] = 0.0f;
                    ov_25[8] = 0.0f;
                    ov_25[9] = 0.0f;
                    ov_25[10] = 0.0f;
                    ov_25[11] = 0.0f;
                    ov_25[12] = 0.0f;
                    ov_25[13] = 0.0f;
                    ov_25[14] = 0.0f;
                    ov_25[15] = 0.0f;
                    ov_25[16] = 0.0f;
                    ov_25[17] = 0.0f;
                    ov_25[18] = 0.0f;
                    ov_25[19] = 0.0f;
                    ov_25[20] = 0.0f;
                    ov_25[21] = 0.0f;
                    ov_25[22] = 0.0f;
                    ov_25[23] = 0.0f;
                    ov_25[24] = 0.0f;
                    ov_25[25] = 0.0f;
                    ov_25[26] = 0.0f;
                    ov_25[27] = 0.0f;
                    ov_25[28] = 0.0f;
                    ov_25[29] = 0.0f;
                    ov_25[30] = 0.0f;
                    ov_25[31] = 0.0f;
                    ov_25[32] = 0.0f;
                    ov_25[33] = 0.0f;
                    ov_25[34] = 0.0f;
                    ov_25[35] = 0.0f;
                    ov_25[36] = 0.0f;
                    ov_25[37] = 0.0f;
                    ov_25[38] = 0.0f;
                    ov_25[39] = 0.0f;
                    ov_25[40] = 0.0f;
                    ov_25[41] = 0.0f;
                    ov_25[42] = 0.0f;
                    ov_25[43] = 0.0f;
                    ov_25[44] = 0.0f;
                    ov_25[45] = 0.0f;
                    ov_25[46] = 0.0f;
                    ov_25[47] = 0.0f;
                    ov_25[48] = 0.0f;
                    ov_25[49] = 0.0f;
                    ov_25[50] = 0.0f;
                    ov_25[51] = 0.0f;
                    ov_25[52] = 0.0f;
                    ov_25[53] = 0.0f;
                    ov_25[54] = 0.0f;
                    ov_25[55] = 0.0f;
                    ov_25[56] = 0.0f;
                    ov_25[57] = 0.0f;
                    ov_25[58] = 0.0f;
                    ov_25[59] = 0.0f;
                    ov_25[60] = 0.0f;
                    ov_25[61] = 0.0f;
                    ov_25[62] = 0.0f;
                    ov_25[63] = 0.0f;
                }
                const float2 _scale2_38 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_25)[_ls], _scale2_38);
                int out_col_26 = row / 64 * 128 + 64;
                float residual_27[16];
                unsigned int ov_25_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 0], ov_25[_lp*2+1 + 0]));
                    ov_25_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16[_pair]) & 0xffff0000u);
                }
                residual_27[0] = ov_25[0] - _unpack_bf16_8[0];
                residual_27[1] = ov_25[1] - _unpack_bf16_8[1];
                residual_27[2] = ov_25[2] - _unpack_bf16_8[2];
                residual_27[3] = ov_25[3] - _unpack_bf16_8[3];
                residual_27[4] = ov_25[4] - _unpack_bf16_8[4];
                residual_27[5] = ov_25[5] - _unpack_bf16_8[5];
                residual_27[6] = ov_25[6] - _unpack_bf16_8[6];
                residual_27[7] = ov_25[7] - _unpack_bf16_8[7];
                int byte_offset_28 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128;
                int swizzled_offset_29 = byte_offset_28 ^ (byte_offset_28 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_29), "r"(ov_25_bf16[0]), "r"(ov_25_bf16[1]), "r"(ov_25_bf16[2]), "r"(ov_25_bf16[3]) : "memory");
                unsigned int ov_25_bf16_30[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 8], ov_25[_lp*2+1 + 8]));
                    ov_25_bf16_30[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_30[_pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_30[_pair]) & 0xffff0000u);
                }
                residual_27[8] = ov_25[8] - _unpack_bf16_9[0];
                residual_27[9] = ov_25[9] - _unpack_bf16_9[1];
                residual_27[10] = ov_25[10] - _unpack_bf16_9[2];
                residual_27[11] = ov_25[11] - _unpack_bf16_9[3];
                residual_27[12] = ov_25[12] - _unpack_bf16_9[4];
                residual_27[13] = ov_25[13] - _unpack_bf16_9[5];
                residual_27[14] = ov_25[14] - _unpack_bf16_9[6];
                residual_27[15] = ov_25[15] - _unpack_bf16_9[7];
                int byte_offset_31 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_32 = byte_offset_31 ^ (byte_offset_31 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_32), "r"(ov_25_bf16_30[0]), "r"(ov_25_bf16_30[1]), "r"(ov_25_bf16_30[2]), "r"(ov_25_bf16_30[3]) : "memory");
                unsigned int _packed_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 0], residual_27[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 2], residual_27[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 4], residual_27[_lp*2+1 + 4]));
                    _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 6], residual_27[_lp*2+1 + 6]));
                    _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 8], residual_27[_lp*2+1 + 8]));
                    _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 10], residual_27[_lp*2+1 + 10]));
                    _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 12], residual_27[_lp*2+1 + 12]));
                    _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_27[_lp*2 + 14], residual_27[_lp*2+1 + 14]));
                    _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_39 = reinterpret_cast<const unsigned*>(_packed_4);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_26))), "r"(_raw_stv8_39[0]), "r"(_raw_stv8_39[1]), "r"(_raw_stv8_39[2]), "r"(_raw_stv8_39[3]), "r"(_raw_stv8_39[4]), "r"(_raw_stv8_39[5]), "r"(_raw_stv8_39[6]), "r"(_raw_stv8_39[7]) : "memory");
                }
                float residual_33[16];
                unsigned int ov_25_bf16_34[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 16], ov_25[_lp*2+1 + 16]));
                    ov_25_bf16_34[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_34[_pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_34[_pair]) & 0xffff0000u);
                }
                residual_33[0] = ov_25[16] - _unpack_bf16_10[0];
                residual_33[1] = ov_25[17] - _unpack_bf16_10[1];
                residual_33[2] = ov_25[18] - _unpack_bf16_10[2];
                residual_33[3] = ov_25[19] - _unpack_bf16_10[3];
                residual_33[4] = ov_25[20] - _unpack_bf16_10[4];
                residual_33[5] = ov_25[21] - _unpack_bf16_10[5];
                residual_33[6] = ov_25[22] - _unpack_bf16_10[6];
                residual_33[7] = ov_25[23] - _unpack_bf16_10[7];
                int byte_offset_35 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_36 = byte_offset_35 ^ (byte_offset_35 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_36), "r"(ov_25_bf16_34[0]), "r"(ov_25_bf16_34[1]), "r"(ov_25_bf16_34[2]), "r"(ov_25_bf16_34[3]) : "memory");
                unsigned int ov_25_bf16_37[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 24], ov_25[_lp*2+1 + 24]));
                    ov_25_bf16_37[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_37[_pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_37[_pair]) & 0xffff0000u);
                }
                residual_33[8] = ov_25[24] - _unpack_bf16_11[0];
                residual_33[9] = ov_25[25] - _unpack_bf16_11[1];
                residual_33[10] = ov_25[26] - _unpack_bf16_11[2];
                residual_33[11] = ov_25[27] - _unpack_bf16_11[3];
                residual_33[12] = ov_25[28] - _unpack_bf16_11[4];
                residual_33[13] = ov_25[29] - _unpack_bf16_11[5];
                residual_33[14] = ov_25[30] - _unpack_bf16_11[6];
                residual_33[15] = ov_25[31] - _unpack_bf16_11[7];
                int byte_offset_38 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_39 = byte_offset_38 ^ (byte_offset_38 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_39), "r"(ov_25_bf16_37[0]), "r"(ov_25_bf16_37[1]), "r"(ov_25_bf16_37[2]), "r"(ov_25_bf16_37[3]) : "memory");
                unsigned int _packed_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 0], residual_33[_lp*2+1 + 0]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 2], residual_33[_lp*2+1 + 2]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 4], residual_33[_lp*2+1 + 4]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 6], residual_33[_lp*2+1 + 6]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 8], residual_33[_lp*2+1 + 8]));
                    _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 10], residual_33[_lp*2+1 + 10]));
                    _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 12], residual_33[_lp*2+1 + 12]));
                    _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_33[_lp*2 + 14], residual_33[_lp*2+1 + 14]));
                    _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_40 = reinterpret_cast<const unsigned*>(_packed_5);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_26 + 16))), "r"(_raw_stv8_40[0]), "r"(_raw_stv8_40[1]), "r"(_raw_stv8_40[2]), "r"(_raw_stv8_40[3]), "r"(_raw_stv8_40[4]), "r"(_raw_stv8_40[5]), "r"(_raw_stv8_40[6]), "r"(_raw_stv8_40[7]) : "memory");
                }
                float residual_40[16];
                unsigned int ov_25_bf16_41[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 32], ov_25[_lp*2+1 + 32]));
                    ov_25_bf16_41[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_41[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_41[_pair]) & 0xffff0000u);
                }
                residual_40[0] = ov_25[32] - _unpack_bf16_12[0];
                residual_40[1] = ov_25[33] - _unpack_bf16_12[1];
                residual_40[2] = ov_25[34] - _unpack_bf16_12[2];
                residual_40[3] = ov_25[35] - _unpack_bf16_12[3];
                residual_40[4] = ov_25[36] - _unpack_bf16_12[4];
                residual_40[5] = ov_25[37] - _unpack_bf16_12[5];
                residual_40[6] = ov_25[38] - _unpack_bf16_12[6];
                residual_40[7] = ov_25[39] - _unpack_bf16_12[7];
                int byte_offset_42 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_43 = byte_offset_42 ^ (byte_offset_42 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_43), "r"(ov_25_bf16_41[0]), "r"(ov_25_bf16_41[1]), "r"(ov_25_bf16_41[2]), "r"(ov_25_bf16_41[3]) : "memory");
                unsigned int ov_25_bf16_44[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 40], ov_25[_lp*2+1 + 40]));
                    ov_25_bf16_44[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_44[_pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_44[_pair]) & 0xffff0000u);
                }
                residual_40[8] = ov_25[40] - _unpack_bf16_13[0];
                residual_40[9] = ov_25[41] - _unpack_bf16_13[1];
                residual_40[10] = ov_25[42] - _unpack_bf16_13[2];
                residual_40[11] = ov_25[43] - _unpack_bf16_13[3];
                residual_40[12] = ov_25[44] - _unpack_bf16_13[4];
                residual_40[13] = ov_25[45] - _unpack_bf16_13[5];
                residual_40[14] = ov_25[46] - _unpack_bf16_13[6];
                residual_40[15] = ov_25[47] - _unpack_bf16_13[7];
                int byte_offset_45 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_46 = byte_offset_45 ^ (byte_offset_45 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_46), "r"(ov_25_bf16_44[0]), "r"(ov_25_bf16_44[1]), "r"(ov_25_bf16_44[2]), "r"(ov_25_bf16_44[3]) : "memory");
                unsigned int _packed_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 0], residual_40[_lp*2+1 + 0]));
                    _packed_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 2], residual_40[_lp*2+1 + 2]));
                    _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 4], residual_40[_lp*2+1 + 4]));
                    _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 6], residual_40[_lp*2+1 + 6]));
                    _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 8], residual_40[_lp*2+1 + 8]));
                    _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 10], residual_40[_lp*2+1 + 10]));
                    _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 12], residual_40[_lp*2+1 + 12]));
                    _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_40[_lp*2 + 14], residual_40[_lp*2+1 + 14]));
                    _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_41 = reinterpret_cast<const unsigned*>(_packed_6);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_26 + 32))), "r"(_raw_stv8_41[0]), "r"(_raw_stv8_41[1]), "r"(_raw_stv8_41[2]), "r"(_raw_stv8_41[3]), "r"(_raw_stv8_41[4]), "r"(_raw_stv8_41[5]), "r"(_raw_stv8_41[6]), "r"(_raw_stv8_41[7]) : "memory");
                }
                float residual_47[16];
                unsigned int ov_25_bf16_48[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 48], ov_25[_lp*2+1 + 48]));
                    ov_25_bf16_48[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_48[_pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_48[_pair]) & 0xffff0000u);
                }
                residual_47[0] = ov_25[48] - _unpack_bf16_14[0];
                residual_47[1] = ov_25[49] - _unpack_bf16_14[1];
                residual_47[2] = ov_25[50] - _unpack_bf16_14[2];
                residual_47[3] = ov_25[51] - _unpack_bf16_14[3];
                residual_47[4] = ov_25[52] - _unpack_bf16_14[4];
                residual_47[5] = ov_25[53] - _unpack_bf16_14[5];
                residual_47[6] = ov_25[54] - _unpack_bf16_14[6];
                residual_47[7] = ov_25[55] - _unpack_bf16_14[7];
                int byte_offset_49 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_50 = byte_offset_49 ^ (byte_offset_49 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_50), "r"(ov_25_bf16_48[0]), "r"(ov_25_bf16_48[1]), "r"(ov_25_bf16_48[2]), "r"(ov_25_bf16_48[3]) : "memory");
                unsigned int ov_25_bf16_51[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_25[_lp*2 + 56], ov_25[_lp*2+1 + 56]));
                    ov_25_bf16_51[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_51[_pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_25_bf16_51[_pair]) & 0xffff0000u);
                }
                residual_47[8] = ov_25[56] - _unpack_bf16_15[0];
                residual_47[9] = ov_25[57] - _unpack_bf16_15[1];
                residual_47[10] = ov_25[58] - _unpack_bf16_15[2];
                residual_47[11] = ov_25[59] - _unpack_bf16_15[3];
                residual_47[12] = ov_25[60] - _unpack_bf16_15[4];
                residual_47[13] = ov_25[61] - _unpack_bf16_15[5];
                residual_47[14] = ov_25[62] - _unpack_bf16_15[6];
                residual_47[15] = ov_25[63] - _unpack_bf16_15[7];
                int byte_offset_52 = o_slot * 65536 + out_col_26 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_53 = byte_offset_52 ^ (byte_offset_52 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_53), "r"(ov_25_bf16_51[0]), "r"(ov_25_bf16_51[1]), "r"(ov_25_bf16_51[2]), "r"(ov_25_bf16_51[3]) : "memory");
                unsigned int _packed_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 0], residual_47[_lp*2+1 + 0]));
                    _packed_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 2], residual_47[_lp*2+1 + 2]));
                    _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 4], residual_47[_lp*2+1 + 4]));
                    _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 6], residual_47[_lp*2+1 + 6]));
                    _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 8], residual_47[_lp*2+1 + 8]));
                    _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 10], residual_47[_lp*2+1 + 10]));
                    _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 12], residual_47[_lp*2+1 + 12]));
                    _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_47[_lp*2 + 14], residual_47[_lp*2+1 + 14]));
                    _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_42 = reinterpret_cast<const unsigned*>(_packed_7);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_26 + 48))), "r"(_raw_stv8_42[0]), "r"(_raw_stv8_42[1]), "r"(_raw_stv8_42[2]), "r"(_raw_stv8_42[3]), "r"(_raw_stv8_42[4]), "r"(_raw_stv8_42[5]), "r"(_raw_stv8_42[6]), "r"(_raw_stv8_42[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_1 = warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_1, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_1 * 8192));
                    }
                }
                float ov_54[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_54[0]), "=f"(ov_54[1]), "=f"(ov_54[2]), "=f"(ov_54[3]), "=f"(ov_54[4]), "=f"(ov_54[5]), "=f"(ov_54[6]), "=f"(ov_54[7]), "=f"(ov_54[8]), "=f"(ov_54[9]), "=f"(ov_54[10]), "=f"(ov_54[11]), "=f"(ov_54[12]), "=f"(ov_54[13]), "=f"(ov_54[14]), "=f"(ov_54[15]), "=f"(ov_54[16]), "=f"(ov_54[17]), "=f"(ov_54[18]), "=f"(ov_54[19]), "=f"(ov_54[20]), "=f"(ov_54[21]), "=f"(ov_54[22]), "=f"(ov_54[23]), "=f"(ov_54[24]), "=f"(ov_54[25]), "=f"(ov_54[26]), "=f"(ov_54[27]), "=f"(ov_54[28]), "=f"(ov_54[29]), "=f"(ov_54[30]), "=f"(ov_54[31]), "=f"(ov_54[32]), "=f"(ov_54[33]), "=f"(ov_54[34]), "=f"(ov_54[35]), "=f"(ov_54[36]), "=f"(ov_54[37]), "=f"(ov_54[38]), "=f"(ov_54[39]), "=f"(ov_54[40]), "=f"(ov_54[41]), "=f"(ov_54[42]), "=f"(ov_54[43]), "=f"(ov_54[44]), "=f"(ov_54[45]), "=f"(ov_54[46]), "=f"(ov_54[47]), "=f"(ov_54[48]), "=f"(ov_54[49]), "=f"(ov_54[50]), "=f"(ov_54[51]), "=f"(ov_54[52]), "=f"(ov_54[53]), "=f"(ov_54[54]), "=f"(ov_54[55]), "=f"(ov_54[56]), "=f"(ov_54[57]), "=f"(ov_54[58]), "=f"(ov_54[59]), "=f"(ov_54[60]), "=f"(ov_54[61]), "=f"(ov_54[62]), "=f"(ov_54[63])
                        : "r"(tmrow + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_54[0] = 0.0f;
                    ov_54[1] = 0.0f;
                    ov_54[2] = 0.0f;
                    ov_54[3] = 0.0f;
                    ov_54[4] = 0.0f;
                    ov_54[5] = 0.0f;
                    ov_54[6] = 0.0f;
                    ov_54[7] = 0.0f;
                    ov_54[8] = 0.0f;
                    ov_54[9] = 0.0f;
                    ov_54[10] = 0.0f;
                    ov_54[11] = 0.0f;
                    ov_54[12] = 0.0f;
                    ov_54[13] = 0.0f;
                    ov_54[14] = 0.0f;
                    ov_54[15] = 0.0f;
                    ov_54[16] = 0.0f;
                    ov_54[17] = 0.0f;
                    ov_54[18] = 0.0f;
                    ov_54[19] = 0.0f;
                    ov_54[20] = 0.0f;
                    ov_54[21] = 0.0f;
                    ov_54[22] = 0.0f;
                    ov_54[23] = 0.0f;
                    ov_54[24] = 0.0f;
                    ov_54[25] = 0.0f;
                    ov_54[26] = 0.0f;
                    ov_54[27] = 0.0f;
                    ov_54[28] = 0.0f;
                    ov_54[29] = 0.0f;
                    ov_54[30] = 0.0f;
                    ov_54[31] = 0.0f;
                    ov_54[32] = 0.0f;
                    ov_54[33] = 0.0f;
                    ov_54[34] = 0.0f;
                    ov_54[35] = 0.0f;
                    ov_54[36] = 0.0f;
                    ov_54[37] = 0.0f;
                    ov_54[38] = 0.0f;
                    ov_54[39] = 0.0f;
                    ov_54[40] = 0.0f;
                    ov_54[41] = 0.0f;
                    ov_54[42] = 0.0f;
                    ov_54[43] = 0.0f;
                    ov_54[44] = 0.0f;
                    ov_54[45] = 0.0f;
                    ov_54[46] = 0.0f;
                    ov_54[47] = 0.0f;
                    ov_54[48] = 0.0f;
                    ov_54[49] = 0.0f;
                    ov_54[50] = 0.0f;
                    ov_54[51] = 0.0f;
                    ov_54[52] = 0.0f;
                    ov_54[53] = 0.0f;
                    ov_54[54] = 0.0f;
                    ov_54[55] = 0.0f;
                    ov_54[56] = 0.0f;
                    ov_54[57] = 0.0f;
                    ov_54[58] = 0.0f;
                    ov_54[59] = 0.0f;
                    ov_54[60] = 0.0f;
                    ov_54[61] = 0.0f;
                    ov_54[62] = 0.0f;
                    ov_54[63] = 0.0f;
                }
                const float2 _scale2_43 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_54)[_ls], _scale2_43);
                int out_col_55 = 256 + row / 64 * 128;
                float residual_56[16];
                unsigned int ov_54_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 0], ov_54[_lp*2+1 + 0]));
                    ov_54_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16[_pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16[_pair]) & 0xffff0000u);
                }
                residual_56[0] = ov_54[0] - _unpack_bf16_16[0];
                residual_56[1] = ov_54[1] - _unpack_bf16_16[1];
                residual_56[2] = ov_54[2] - _unpack_bf16_16[2];
                residual_56[3] = ov_54[3] - _unpack_bf16_16[3];
                residual_56[4] = ov_54[4] - _unpack_bf16_16[4];
                residual_56[5] = ov_54[5] - _unpack_bf16_16[5];
                residual_56[6] = ov_54[6] - _unpack_bf16_16[6];
                residual_56[7] = ov_54[7] - _unpack_bf16_16[7];
                int byte_offset_57 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128;
                int swizzled_offset_58 = byte_offset_57 ^ (byte_offset_57 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_58), "r"(ov_54_bf16[0]), "r"(ov_54_bf16[1]), "r"(ov_54_bf16[2]), "r"(ov_54_bf16[3]) : "memory");
                unsigned int ov_54_bf16_59[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 8], ov_54[_lp*2+1 + 8]));
                    ov_54_bf16_59[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_17[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_59[_pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_59[_pair]) & 0xffff0000u);
                }
                residual_56[8] = ov_54[8] - _unpack_bf16_17[0];
                residual_56[9] = ov_54[9] - _unpack_bf16_17[1];
                residual_56[10] = ov_54[10] - _unpack_bf16_17[2];
                residual_56[11] = ov_54[11] - _unpack_bf16_17[3];
                residual_56[12] = ov_54[12] - _unpack_bf16_17[4];
                residual_56[13] = ov_54[13] - _unpack_bf16_17[5];
                residual_56[14] = ov_54[14] - _unpack_bf16_17[6];
                residual_56[15] = ov_54[15] - _unpack_bf16_17[7];
                int byte_offset_60 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_61 = byte_offset_60 ^ (byte_offset_60 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_61), "r"(ov_54_bf16_59[0]), "r"(ov_54_bf16_59[1]), "r"(ov_54_bf16_59[2]), "r"(ov_54_bf16_59[3]) : "memory");
                unsigned int _packed_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 0], residual_56[_lp*2+1 + 0]));
                    _packed_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 2], residual_56[_lp*2+1 + 2]));
                    _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 4], residual_56[_lp*2+1 + 4]));
                    _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 6], residual_56[_lp*2+1 + 6]));
                    _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 8], residual_56[_lp*2+1 + 8]));
                    _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 10], residual_56[_lp*2+1 + 10]));
                    _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 12], residual_56[_lp*2+1 + 12]));
                    _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_56[_lp*2 + 14], residual_56[_lp*2+1 + 14]));
                    _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_44 = reinterpret_cast<const unsigned*>(_packed_8);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_55))), "r"(_raw_stv8_44[0]), "r"(_raw_stv8_44[1]), "r"(_raw_stv8_44[2]), "r"(_raw_stv8_44[3]), "r"(_raw_stv8_44[4]), "r"(_raw_stv8_44[5]), "r"(_raw_stv8_44[6]), "r"(_raw_stv8_44[7]) : "memory");
                }
                float residual_62[16];
                unsigned int ov_54_bf16_63[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 16], ov_54[_lp*2+1 + 16]));
                    ov_54_bf16_63[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_18[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_63[_pair]) << 16);
                    (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_63[_pair]) & 0xffff0000u);
                }
                residual_62[0] = ov_54[16] - _unpack_bf16_18[0];
                residual_62[1] = ov_54[17] - _unpack_bf16_18[1];
                residual_62[2] = ov_54[18] - _unpack_bf16_18[2];
                residual_62[3] = ov_54[19] - _unpack_bf16_18[3];
                residual_62[4] = ov_54[20] - _unpack_bf16_18[4];
                residual_62[5] = ov_54[21] - _unpack_bf16_18[5];
                residual_62[6] = ov_54[22] - _unpack_bf16_18[6];
                residual_62[7] = ov_54[23] - _unpack_bf16_18[7];
                int byte_offset_64 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_65 = byte_offset_64 ^ (byte_offset_64 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_65), "r"(ov_54_bf16_63[0]), "r"(ov_54_bf16_63[1]), "r"(ov_54_bf16_63[2]), "r"(ov_54_bf16_63[3]) : "memory");
                unsigned int ov_54_bf16_66[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 24], ov_54[_lp*2+1 + 24]));
                    ov_54_bf16_66[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_19[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_66[_pair]) << 16);
                    (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_66[_pair]) & 0xffff0000u);
                }
                residual_62[8] = ov_54[24] - _unpack_bf16_19[0];
                residual_62[9] = ov_54[25] - _unpack_bf16_19[1];
                residual_62[10] = ov_54[26] - _unpack_bf16_19[2];
                residual_62[11] = ov_54[27] - _unpack_bf16_19[3];
                residual_62[12] = ov_54[28] - _unpack_bf16_19[4];
                residual_62[13] = ov_54[29] - _unpack_bf16_19[5];
                residual_62[14] = ov_54[30] - _unpack_bf16_19[6];
                residual_62[15] = ov_54[31] - _unpack_bf16_19[7];
                int byte_offset_67 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_68 = byte_offset_67 ^ (byte_offset_67 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_68), "r"(ov_54_bf16_66[0]), "r"(ov_54_bf16_66[1]), "r"(ov_54_bf16_66[2]), "r"(ov_54_bf16_66[3]) : "memory");
                unsigned int _packed_9[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 0], residual_62[_lp*2+1 + 0]));
                    _packed_9[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 2], residual_62[_lp*2+1 + 2]));
                    _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 4], residual_62[_lp*2+1 + 4]));
                    _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 6], residual_62[_lp*2+1 + 6]));
                    _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 8], residual_62[_lp*2+1 + 8]));
                    _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 10], residual_62[_lp*2+1 + 10]));
                    _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 12], residual_62[_lp*2+1 + 12]));
                    _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_62[_lp*2 + 14], residual_62[_lp*2+1 + 14]));
                    _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_45 = reinterpret_cast<const unsigned*>(_packed_9);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_55 + 16))), "r"(_raw_stv8_45[0]), "r"(_raw_stv8_45[1]), "r"(_raw_stv8_45[2]), "r"(_raw_stv8_45[3]), "r"(_raw_stv8_45[4]), "r"(_raw_stv8_45[5]), "r"(_raw_stv8_45[6]), "r"(_raw_stv8_45[7]) : "memory");
                }
                float residual_69[16];
                unsigned int ov_54_bf16_70[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 32], ov_54[_lp*2+1 + 32]));
                    ov_54_bf16_70[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_20[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_70[_pair]) << 16);
                    (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_70[_pair]) & 0xffff0000u);
                }
                residual_69[0] = ov_54[32] - _unpack_bf16_20[0];
                residual_69[1] = ov_54[33] - _unpack_bf16_20[1];
                residual_69[2] = ov_54[34] - _unpack_bf16_20[2];
                residual_69[3] = ov_54[35] - _unpack_bf16_20[3];
                residual_69[4] = ov_54[36] - _unpack_bf16_20[4];
                residual_69[5] = ov_54[37] - _unpack_bf16_20[5];
                residual_69[6] = ov_54[38] - _unpack_bf16_20[6];
                residual_69[7] = ov_54[39] - _unpack_bf16_20[7];
                int byte_offset_71 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_72 = byte_offset_71 ^ (byte_offset_71 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_72), "r"(ov_54_bf16_70[0]), "r"(ov_54_bf16_70[1]), "r"(ov_54_bf16_70[2]), "r"(ov_54_bf16_70[3]) : "memory");
                unsigned int ov_54_bf16_73[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 40], ov_54[_lp*2+1 + 40]));
                    ov_54_bf16_73[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_21[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_73[_pair]) << 16);
                    (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_73[_pair]) & 0xffff0000u);
                }
                residual_69[8] = ov_54[40] - _unpack_bf16_21[0];
                residual_69[9] = ov_54[41] - _unpack_bf16_21[1];
                residual_69[10] = ov_54[42] - _unpack_bf16_21[2];
                residual_69[11] = ov_54[43] - _unpack_bf16_21[3];
                residual_69[12] = ov_54[44] - _unpack_bf16_21[4];
                residual_69[13] = ov_54[45] - _unpack_bf16_21[5];
                residual_69[14] = ov_54[46] - _unpack_bf16_21[6];
                residual_69[15] = ov_54[47] - _unpack_bf16_21[7];
                int byte_offset_74 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_75 = byte_offset_74 ^ (byte_offset_74 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_75), "r"(ov_54_bf16_73[0]), "r"(ov_54_bf16_73[1]), "r"(ov_54_bf16_73[2]), "r"(ov_54_bf16_73[3]) : "memory");
                unsigned int _packed_10[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 0], residual_69[_lp*2+1 + 0]));
                    _packed_10[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 2], residual_69[_lp*2+1 + 2]));
                    _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 4], residual_69[_lp*2+1 + 4]));
                    _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 6], residual_69[_lp*2+1 + 6]));
                    _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 8], residual_69[_lp*2+1 + 8]));
                    _packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 10], residual_69[_lp*2+1 + 10]));
                    _packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 12], residual_69[_lp*2+1 + 12]));
                    _packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_69[_lp*2 + 14], residual_69[_lp*2+1 + 14]));
                    _packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_46 = reinterpret_cast<const unsigned*>(_packed_10);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_55 + 32))), "r"(_raw_stv8_46[0]), "r"(_raw_stv8_46[1]), "r"(_raw_stv8_46[2]), "r"(_raw_stv8_46[3]), "r"(_raw_stv8_46[4]), "r"(_raw_stv8_46[5]), "r"(_raw_stv8_46[6]), "r"(_raw_stv8_46[7]) : "memory");
                }
                float residual_76[16];
                unsigned int ov_54_bf16_77[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 48], ov_54[_lp*2+1 + 48]));
                    ov_54_bf16_77[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_22[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_77[_pair]) << 16);
                    (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_77[_pair]) & 0xffff0000u);
                }
                residual_76[0] = ov_54[48] - _unpack_bf16_22[0];
                residual_76[1] = ov_54[49] - _unpack_bf16_22[1];
                residual_76[2] = ov_54[50] - _unpack_bf16_22[2];
                residual_76[3] = ov_54[51] - _unpack_bf16_22[3];
                residual_76[4] = ov_54[52] - _unpack_bf16_22[4];
                residual_76[5] = ov_54[53] - _unpack_bf16_22[5];
                residual_76[6] = ov_54[54] - _unpack_bf16_22[6];
                residual_76[7] = ov_54[55] - _unpack_bf16_22[7];
                int byte_offset_78 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_79 = byte_offset_78 ^ (byte_offset_78 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_79), "r"(ov_54_bf16_77[0]), "r"(ov_54_bf16_77[1]), "r"(ov_54_bf16_77[2]), "r"(ov_54_bf16_77[3]) : "memory");
                unsigned int ov_54_bf16_80[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_54[_lp*2 + 56], ov_54[_lp*2+1 + 56]));
                    ov_54_bf16_80[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_23[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_80[_pair]) << 16);
                    (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_54_bf16_80[_pair]) & 0xffff0000u);
                }
                residual_76[8] = ov_54[56] - _unpack_bf16_23[0];
                residual_76[9] = ov_54[57] - _unpack_bf16_23[1];
                residual_76[10] = ov_54[58] - _unpack_bf16_23[2];
                residual_76[11] = ov_54[59] - _unpack_bf16_23[3];
                residual_76[12] = ov_54[60] - _unpack_bf16_23[4];
                residual_76[13] = ov_54[61] - _unpack_bf16_23[5];
                residual_76[14] = ov_54[62] - _unpack_bf16_23[6];
                residual_76[15] = ov_54[63] - _unpack_bf16_23[7];
                int byte_offset_81 = o_slot * 65536 + out_col_55 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_82 = byte_offset_81 ^ (byte_offset_81 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_82), "r"(ov_54_bf16_80[0]), "r"(ov_54_bf16_80[1]), "r"(ov_54_bf16_80[2]), "r"(ov_54_bf16_80[3]) : "memory");
                unsigned int _packed_11[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 0], residual_76[_lp*2+1 + 0]));
                    _packed_11[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 2], residual_76[_lp*2+1 + 2]));
                    _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 4], residual_76[_lp*2+1 + 4]));
                    _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 6], residual_76[_lp*2+1 + 6]));
                    _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 8], residual_76[_lp*2+1 + 8]));
                    _packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 10], residual_76[_lp*2+1 + 10]));
                    _packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 12], residual_76[_lp*2+1 + 12]));
                    _packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_76[_lp*2 + 14], residual_76[_lp*2+1 + 14]));
                    _packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_47 = reinterpret_cast<const unsigned*>(_packed_11);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_55 + 48))), "r"(_raw_stv8_47[0]), "r"(_raw_stv8_47[1]), "r"(_raw_stv8_47[2]), "r"(_raw_stv8_47[3]), "r"(_raw_stv8_47[4]), "r"(_raw_stv8_47[5]), "r"(_raw_stv8_47[6]), "r"(_raw_stv8_47[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_2 = 4 + warp * 2;
                        tma_store_4d((&out), 0, 0, output_chunk_2, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_2 * 8192));
                    }
                }
                float ov_83[64];
                if (have_valid != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(ov_83[0]), "=f"(ov_83[1]), "=f"(ov_83[2]), "=f"(ov_83[3]), "=f"(ov_83[4]), "=f"(ov_83[5]), "=f"(ov_83[6]), "=f"(ov_83[7]), "=f"(ov_83[8]), "=f"(ov_83[9]), "=f"(ov_83[10]), "=f"(ov_83[11]), "=f"(ov_83[12]), "=f"(ov_83[13]), "=f"(ov_83[14]), "=f"(ov_83[15]), "=f"(ov_83[16]), "=f"(ov_83[17]), "=f"(ov_83[18]), "=f"(ov_83[19]), "=f"(ov_83[20]), "=f"(ov_83[21]), "=f"(ov_83[22]), "=f"(ov_83[23]), "=f"(ov_83[24]), "=f"(ov_83[25]), "=f"(ov_83[26]), "=f"(ov_83[27]), "=f"(ov_83[28]), "=f"(ov_83[29]), "=f"(ov_83[30]), "=f"(ov_83[31]), "=f"(ov_83[32]), "=f"(ov_83[33]), "=f"(ov_83[34]), "=f"(ov_83[35]), "=f"(ov_83[36]), "=f"(ov_83[37]), "=f"(ov_83[38]), "=f"(ov_83[39]), "=f"(ov_83[40]), "=f"(ov_83[41]), "=f"(ov_83[42]), "=f"(ov_83[43]), "=f"(ov_83[44]), "=f"(ov_83[45]), "=f"(ov_83[46]), "=f"(ov_83[47]), "=f"(ov_83[48]), "=f"(ov_83[49]), "=f"(ov_83[50]), "=f"(ov_83[51]), "=f"(ov_83[52]), "=f"(ov_83[53]), "=f"(ov_83[54]), "=f"(ov_83[55]), "=f"(ov_83[56]), "=f"(ov_83[57]), "=f"(ov_83[58]), "=f"(ov_83[59]), "=f"(ov_83[60]), "=f"(ov_83[61]), "=f"(ov_83[62]), "=f"(ov_83[63])
                        : "r"(tmrow + 128 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                } else {
                    ov_83[0] = 0.0f;
                    ov_83[1] = 0.0f;
                    ov_83[2] = 0.0f;
                    ov_83[3] = 0.0f;
                    ov_83[4] = 0.0f;
                    ov_83[5] = 0.0f;
                    ov_83[6] = 0.0f;
                    ov_83[7] = 0.0f;
                    ov_83[8] = 0.0f;
                    ov_83[9] = 0.0f;
                    ov_83[10] = 0.0f;
                    ov_83[11] = 0.0f;
                    ov_83[12] = 0.0f;
                    ov_83[13] = 0.0f;
                    ov_83[14] = 0.0f;
                    ov_83[15] = 0.0f;
                    ov_83[16] = 0.0f;
                    ov_83[17] = 0.0f;
                    ov_83[18] = 0.0f;
                    ov_83[19] = 0.0f;
                    ov_83[20] = 0.0f;
                    ov_83[21] = 0.0f;
                    ov_83[22] = 0.0f;
                    ov_83[23] = 0.0f;
                    ov_83[24] = 0.0f;
                    ov_83[25] = 0.0f;
                    ov_83[26] = 0.0f;
                    ov_83[27] = 0.0f;
                    ov_83[28] = 0.0f;
                    ov_83[29] = 0.0f;
                    ov_83[30] = 0.0f;
                    ov_83[31] = 0.0f;
                    ov_83[32] = 0.0f;
                    ov_83[33] = 0.0f;
                    ov_83[34] = 0.0f;
                    ov_83[35] = 0.0f;
                    ov_83[36] = 0.0f;
                    ov_83[37] = 0.0f;
                    ov_83[38] = 0.0f;
                    ov_83[39] = 0.0f;
                    ov_83[40] = 0.0f;
                    ov_83[41] = 0.0f;
                    ov_83[42] = 0.0f;
                    ov_83[43] = 0.0f;
                    ov_83[44] = 0.0f;
                    ov_83[45] = 0.0f;
                    ov_83[46] = 0.0f;
                    ov_83[47] = 0.0f;
                    ov_83[48] = 0.0f;
                    ov_83[49] = 0.0f;
                    ov_83[50] = 0.0f;
                    ov_83[51] = 0.0f;
                    ov_83[52] = 0.0f;
                    ov_83[53] = 0.0f;
                    ov_83[54] = 0.0f;
                    ov_83[55] = 0.0f;
                    ov_83[56] = 0.0f;
                    ov_83[57] = 0.0f;
                    ov_83[58] = 0.0f;
                    ov_83[59] = 0.0f;
                    ov_83[60] = 0.0f;
                    ov_83[61] = 0.0f;
                    ov_83[62] = 0.0f;
                    ov_83[63] = 0.0f;
                }
                const float2 _scale2_48 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov_83)[_ls], _scale2_48);
                int out_col_84 = 256 + row / 64 * 128 + 64;
                float residual_85[16];
                unsigned int ov_83_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 0], ov_83[_lp*2+1 + 0]));
                    ov_83_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_24[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16[_pair]) << 16);
                    (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16[_pair]) & 0xffff0000u);
                }
                residual_85[0] = ov_83[0] - _unpack_bf16_24[0];
                residual_85[1] = ov_83[1] - _unpack_bf16_24[1];
                residual_85[2] = ov_83[2] - _unpack_bf16_24[2];
                residual_85[3] = ov_83[3] - _unpack_bf16_24[3];
                residual_85[4] = ov_83[4] - _unpack_bf16_24[4];
                residual_85[5] = ov_83[5] - _unpack_bf16_24[5];
                residual_85[6] = ov_83[6] - _unpack_bf16_24[6];
                residual_85[7] = ov_83[7] - _unpack_bf16_24[7];
                int byte_offset_86 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128;
                int swizzled_offset_87 = byte_offset_86 ^ (byte_offset_86 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_87), "r"(ov_83_bf16[0]), "r"(ov_83_bf16[1]), "r"(ov_83_bf16[2]), "r"(ov_83_bf16[3]) : "memory");
                unsigned int ov_83_bf16_88[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 8], ov_83[_lp*2+1 + 8]));
                    ov_83_bf16_88[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_25[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_88[_pair]) << 16);
                    (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_88[_pair]) & 0xffff0000u);
                }
                residual_85[8] = ov_83[8] - _unpack_bf16_25[0];
                residual_85[9] = ov_83[9] - _unpack_bf16_25[1];
                residual_85[10] = ov_83[10] - _unpack_bf16_25[2];
                residual_85[11] = ov_83[11] - _unpack_bf16_25[3];
                residual_85[12] = ov_83[12] - _unpack_bf16_25[4];
                residual_85[13] = ov_83[13] - _unpack_bf16_25[5];
                residual_85[14] = ov_83[14] - _unpack_bf16_25[6];
                residual_85[15] = ov_83[15] - _unpack_bf16_25[7];
                int byte_offset_89 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 16;
                int swizzled_offset_90 = byte_offset_89 ^ (byte_offset_89 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_90), "r"(ov_83_bf16_88[0]), "r"(ov_83_bf16_88[1]), "r"(ov_83_bf16_88[2]), "r"(ov_83_bf16_88[3]) : "memory");
                unsigned int _packed_12[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 0], residual_85[_lp*2+1 + 0]));
                    _packed_12[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 2], residual_85[_lp*2+1 + 2]));
                    _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 4], residual_85[_lp*2+1 + 4]));
                    _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 6], residual_85[_lp*2+1 + 6]));
                    _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 8], residual_85[_lp*2+1 + 8]));
                    _packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 10], residual_85[_lp*2+1 + 10]));
                    _packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 12], residual_85[_lp*2+1 + 12]));
                    _packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_85[_lp*2 + 14], residual_85[_lp*2+1 + 14]));
                    _packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_49 = reinterpret_cast<const unsigned*>(_packed_12);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_84))), "r"(_raw_stv8_49[0]), "r"(_raw_stv8_49[1]), "r"(_raw_stv8_49[2]), "r"(_raw_stv8_49[3]), "r"(_raw_stv8_49[4]), "r"(_raw_stv8_49[5]), "r"(_raw_stv8_49[6]), "r"(_raw_stv8_49[7]) : "memory");
                }
                float residual_91[16];
                unsigned int ov_83_bf16_92[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 16], ov_83[_lp*2+1 + 16]));
                    ov_83_bf16_92[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_26[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_92[_pair]) << 16);
                    (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_92[_pair]) & 0xffff0000u);
                }
                residual_91[0] = ov_83[16] - _unpack_bf16_26[0];
                residual_91[1] = ov_83[17] - _unpack_bf16_26[1];
                residual_91[2] = ov_83[18] - _unpack_bf16_26[2];
                residual_91[3] = ov_83[19] - _unpack_bf16_26[3];
                residual_91[4] = ov_83[20] - _unpack_bf16_26[4];
                residual_91[5] = ov_83[21] - _unpack_bf16_26[5];
                residual_91[6] = ov_83[22] - _unpack_bf16_26[6];
                residual_91[7] = ov_83[23] - _unpack_bf16_26[7];
                int byte_offset_93 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 32;
                int swizzled_offset_94 = byte_offset_93 ^ (byte_offset_93 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_94), "r"(ov_83_bf16_92[0]), "r"(ov_83_bf16_92[1]), "r"(ov_83_bf16_92[2]), "r"(ov_83_bf16_92[3]) : "memory");
                unsigned int ov_83_bf16_95[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 24], ov_83[_lp*2+1 + 24]));
                    ov_83_bf16_95[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_27[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_95[_pair]) << 16);
                    (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_95[_pair]) & 0xffff0000u);
                }
                residual_91[8] = ov_83[24] - _unpack_bf16_27[0];
                residual_91[9] = ov_83[25] - _unpack_bf16_27[1];
                residual_91[10] = ov_83[26] - _unpack_bf16_27[2];
                residual_91[11] = ov_83[27] - _unpack_bf16_27[3];
                residual_91[12] = ov_83[28] - _unpack_bf16_27[4];
                residual_91[13] = ov_83[29] - _unpack_bf16_27[5];
                residual_91[14] = ov_83[30] - _unpack_bf16_27[6];
                residual_91[15] = ov_83[31] - _unpack_bf16_27[7];
                int byte_offset_96 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 48;
                int swizzled_offset_97 = byte_offset_96 ^ (byte_offset_96 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_97), "r"(ov_83_bf16_95[0]), "r"(ov_83_bf16_95[1]), "r"(ov_83_bf16_95[2]), "r"(ov_83_bf16_95[3]) : "memory");
                unsigned int _packed_13[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 0], residual_91[_lp*2+1 + 0]));
                    _packed_13[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 2], residual_91[_lp*2+1 + 2]));
                    _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 4], residual_91[_lp*2+1 + 4]));
                    _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 6], residual_91[_lp*2+1 + 6]));
                    _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 8], residual_91[_lp*2+1 + 8]));
                    _packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 10], residual_91[_lp*2+1 + 10]));
                    _packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 12], residual_91[_lp*2+1 + 12]));
                    _packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_91[_lp*2 + 14], residual_91[_lp*2+1 + 14]));
                    _packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_50 = reinterpret_cast<const unsigned*>(_packed_13);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_84 + 16))), "r"(_raw_stv8_50[0]), "r"(_raw_stv8_50[1]), "r"(_raw_stv8_50[2]), "r"(_raw_stv8_50[3]), "r"(_raw_stv8_50[4]), "r"(_raw_stv8_50[5]), "r"(_raw_stv8_50[6]), "r"(_raw_stv8_50[7]) : "memory");
                }
                float residual_98[16];
                unsigned int ov_83_bf16_99[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 32], ov_83[_lp*2+1 + 32]));
                    ov_83_bf16_99[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_28[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_99[_pair]) << 16);
                    (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_99[_pair]) & 0xffff0000u);
                }
                residual_98[0] = ov_83[32] - _unpack_bf16_28[0];
                residual_98[1] = ov_83[33] - _unpack_bf16_28[1];
                residual_98[2] = ov_83[34] - _unpack_bf16_28[2];
                residual_98[3] = ov_83[35] - _unpack_bf16_28[3];
                residual_98[4] = ov_83[36] - _unpack_bf16_28[4];
                residual_98[5] = ov_83[37] - _unpack_bf16_28[5];
                residual_98[6] = ov_83[38] - _unpack_bf16_28[6];
                residual_98[7] = ov_83[39] - _unpack_bf16_28[7];
                int byte_offset_100 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 64;
                int swizzled_offset_101 = byte_offset_100 ^ (byte_offset_100 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_101), "r"(ov_83_bf16_99[0]), "r"(ov_83_bf16_99[1]), "r"(ov_83_bf16_99[2]), "r"(ov_83_bf16_99[3]) : "memory");
                unsigned int ov_83_bf16_102[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 40], ov_83[_lp*2+1 + 40]));
                    ov_83_bf16_102[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_29[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_102[_pair]) << 16);
                    (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_102[_pair]) & 0xffff0000u);
                }
                residual_98[8] = ov_83[40] - _unpack_bf16_29[0];
                residual_98[9] = ov_83[41] - _unpack_bf16_29[1];
                residual_98[10] = ov_83[42] - _unpack_bf16_29[2];
                residual_98[11] = ov_83[43] - _unpack_bf16_29[3];
                residual_98[12] = ov_83[44] - _unpack_bf16_29[4];
                residual_98[13] = ov_83[45] - _unpack_bf16_29[5];
                residual_98[14] = ov_83[46] - _unpack_bf16_29[6];
                residual_98[15] = ov_83[47] - _unpack_bf16_29[7];
                int byte_offset_103 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 80;
                int swizzled_offset_104 = byte_offset_103 ^ (byte_offset_103 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_104), "r"(ov_83_bf16_102[0]), "r"(ov_83_bf16_102[1]), "r"(ov_83_bf16_102[2]), "r"(ov_83_bf16_102[3]) : "memory");
                unsigned int _packed_14[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 0], residual_98[_lp*2+1 + 0]));
                    _packed_14[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 2], residual_98[_lp*2+1 + 2]));
                    _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 4], residual_98[_lp*2+1 + 4]));
                    _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 6], residual_98[_lp*2+1 + 6]));
                    _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 8], residual_98[_lp*2+1 + 8]));
                    _packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 10], residual_98[_lp*2+1 + 10]));
                    _packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 12], residual_98[_lp*2+1 + 12]));
                    _packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_98[_lp*2 + 14], residual_98[_lp*2+1 + 14]));
                    _packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_51 = reinterpret_cast<const unsigned*>(_packed_14);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_84 + 32))), "r"(_raw_stv8_51[0]), "r"(_raw_stv8_51[1]), "r"(_raw_stv8_51[2]), "r"(_raw_stv8_51[3]), "r"(_raw_stv8_51[4]), "r"(_raw_stv8_51[5]), "r"(_raw_stv8_51[6]), "r"(_raw_stv8_51[7]) : "memory");
                }
                float residual_105[16];
                unsigned int ov_83_bf16_106[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 48], ov_83[_lp*2+1 + 48]));
                    ov_83_bf16_106[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_30[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_106[_pair]) << 16);
                    (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_106[_pair]) & 0xffff0000u);
                }
                residual_105[0] = ov_83[48] - _unpack_bf16_30[0];
                residual_105[1] = ov_83[49] - _unpack_bf16_30[1];
                residual_105[2] = ov_83[50] - _unpack_bf16_30[2];
                residual_105[3] = ov_83[51] - _unpack_bf16_30[3];
                residual_105[4] = ov_83[52] - _unpack_bf16_30[4];
                residual_105[5] = ov_83[53] - _unpack_bf16_30[5];
                residual_105[6] = ov_83[54] - _unpack_bf16_30[6];
                residual_105[7] = ov_83[55] - _unpack_bf16_30[7];
                int byte_offset_107 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 96;
                int swizzled_offset_108 = byte_offset_107 ^ (byte_offset_107 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_108), "r"(ov_83_bf16_106[0]), "r"(ov_83_bf16_106[1]), "r"(ov_83_bf16_106[2]), "r"(ov_83_bf16_106[3]) : "memory");
                unsigned int ov_83_bf16_109[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_83[_lp*2 + 56], ov_83[_lp*2+1 + 56]));
                    ov_83_bf16_109[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_31[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_109[_pair]) << 16);
                    (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ov_83_bf16_109[_pair]) & 0xffff0000u);
                }
                residual_105[8] = ov_83[56] - _unpack_bf16_31[0];
                residual_105[9] = ov_83[57] - _unpack_bf16_31[1];
                residual_105[10] = ov_83[58] - _unpack_bf16_31[2];
                residual_105[11] = ov_83[59] - _unpack_bf16_31[3];
                residual_105[12] = ov_83[60] - _unpack_bf16_31[4];
                residual_105[13] = ov_83[61] - _unpack_bf16_31[5];
                residual_105[14] = ov_83[62] - _unpack_bf16_31[6];
                residual_105[15] = ov_83[63] - _unpack_bf16_31[7];
                int byte_offset_110 = o_slot * 65536 + out_col_84 * 128 + row % 64 * 128 + 112;
                int swizzled_offset_111 = byte_offset_110 ^ (byte_offset_110 >> 7 & 7) << 4;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(qko_addr + (unsigned int)swizzled_offset_111), "r"(ov_83_bf16_109[0]), "r"(ov_83_bf16_109[1]), "r"(ov_83_bf16_109[2]), "r"(ov_83_bf16_109[3]) : "memory");
                unsigned int _packed_15[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 0], residual_105[_lp*2+1 + 0]));
                    _packed_15[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 2], residual_105[_lp*2+1 + 2]));
                    _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 4], residual_105[_lp*2+1 + 4]));
                    _packed_15[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 6], residual_105[_lp*2+1 + 6]));
                    _packed_15[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 8], residual_105[_lp*2+1 + 8]));
                    _packed_15[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 10], residual_105[_lp*2+1 + 10]));
                    _packed_15[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 12], residual_105[_lp*2+1 + 12]));
                    _packed_15[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_105[_lp*2 + 14], residual_105[_lp*2+1 + 14]));
                    _packed_15[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_52 = reinterpret_cast<const unsigned*>(_packed_15);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(o_lo + (o_row_base + (long long)out_col_84 + 48))), "r"(_raw_stv8_52[0]), "r"(_raw_stv8_52[1]), "r"(_raw_stv8_52[2]), "r"(_raw_stv8_52[3]), "r"(_raw_stv8_52[4]), "r"(_raw_stv8_52[5]), "r"(_raw_stv8_52[6]), "r"(_raw_stv8_52[7]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp < 2) {
                    if (elect_sync()) {
                        int output_chunk_3 = 4 + warp * 2 + 1;
                        tma_store_4d((&out), 0, 0, output_chunk_3, query, qko_addr + (unsigned int)(o_slot * 65536) + (unsigned int)(output_chunk_3 * 8192));
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                mbarrier_wait(clc_full_addr, outer_phase);
                uint32_t _clc_valid_0 = 0;
                uint32_t _clc_ctaid_x_0;
                uint32_t _clc_ctaid_y_0;
                uint32_t _clc_ctaid_z_0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_0 == 0) {
                    break;
                }
                query = _clc_ctaid_x_0;
                outer_phase ^= 1;
            }
            asm volatile("cp.async.bulk.wait_group 0;");
            if (warp == 3) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 4 && warp <= 7) {
        { // producer_main
            int query_1 = blockIdx.x;
            unsigned int outer_phase_1 = 0;
            int cursor_1 = 0;
            int producer_warp = warp - 4;
            #pragma unroll 1
            for (int outer_1 = 0; outer_1 < num_queries; outer_1++) {
                int active_1 = topk;
                if (derive_length != 0) {
                    int len_slot_1 = outer_1 & 1;
                    unsigned int len_phase_1 = outer_1 >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot_1) * 8, len_phase_1);
                    int _max_1_1 = ((len_cells[len_slot_1]) > (0) ? (len_cells[len_slot_1]) : (0));
                    int _min_0_1 = ((_max_1_1) < (topk) ? (_max_1_1) : (topk));
                    active_1 = _min_0_1;
                } else if (has_topk_length != 0) {
                    int _max_3_1 = ((topk_length[query_1]) > (0) ? (topk_length[query_1]) : (0));
                    int _min_2_1 = ((_max_3_1) < (topk) ? (_max_3_1) : (topk));
                    active_1 = _min_2_1;
                }
                int active_0_1 = active_1;
                int blocks_1 = (((active_0_1 + 63) / 64) > (2) ? ((active_0_1 + 63) / 64) : (2));
                long long row_base = (long long)indices_offset + (long long)query_1 * (long long)idx_stride;
                long long row_out = (long long)query_1 * (long long)topk;
                int nxt[16];
                if (elect_sync()) {
                    if (convert != 0) {
                        int nb = 0;
                        int whole_out = active_0_1 >= nb * 64 + 64 && (topk & 3) == 0;
                        if (whole_out != 0) {
                            int off = nb * 64 + producer_warp * 4;
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off) + 0);
                                int4 _ivld_4;
                                _ivld_4 = *_ivptr_4;
                                _vec_load_0[0 + 0] = _ivld_4.x;
                                _vec_load_0[0 + 1] = _ivld_4.y;
                                _vec_load_0[0 + 2] = _ivld_4.z;
                                _vec_load_0[0 + 3] = _ivld_4.w;
                            }
                            nxt[0] = _vec_load_0[0];
                            nxt[1] = _vec_load_0[1];
                            nxt[2] = _vec_load_0[2];
                            nxt[3] = _vec_load_0[3];
                            int off_0 = nb * 64 + 16 + producer_warp * 4;
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_5 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_0) + 0);
                                int4 _ivld_5;
                                _ivld_5 = *_ivptr_5;
                                _vec_load_1[0 + 0] = _ivld_5.x;
                                _vec_load_1[0 + 1] = _ivld_5.y;
                                _vec_load_1[0 + 2] = _ivld_5.z;
                                _vec_load_1[0 + 3] = _ivld_5.w;
                            }
                            nxt[4] = _vec_load_1[0];
                            nxt[5] = _vec_load_1[1];
                            nxt[6] = _vec_load_1[2];
                            nxt[7] = _vec_load_1[3];
                            int off_1 = nb * 64 + 32 + producer_warp * 4;
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_6 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_1) + 0);
                                int4 _ivld_6;
                                _ivld_6 = *_ivptr_6;
                                _vec_load_2[0 + 0] = _ivld_6.x;
                                _vec_load_2[0 + 1] = _ivld_6.y;
                                _vec_load_2[0 + 2] = _ivld_6.z;
                                _vec_load_2[0 + 3] = _ivld_6.w;
                            }
                            nxt[8] = _vec_load_2[0];
                            nxt[9] = _vec_load_2[1];
                            nxt[10] = _vec_load_2[2];
                            nxt[11] = _vec_load_2[3];
                            int off_2 = nb * 64 + 48 + producer_warp * 4;
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_7 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_2) + 0);
                                int4 _ivld_7;
                                _ivld_7 = *_ivptr_7;
                                _vec_load_3[0 + 0] = _ivld_7.x;
                                _vec_load_3[0 + 1] = _ivld_7.y;
                                _vec_load_3[0 + 2] = _ivld_7.z;
                                _vec_load_3[0 + 3] = _ivld_7.w;
                            }
                            nxt[12] = _vec_load_3[0];
                            nxt[13] = _vec_load_3[1];
                            nxt[14] = _vec_load_3[2];
                            nxt[15] = _vec_load_3[3];
                        } else {
                            int off_3 = nb * 64 + producer_warp * 4;
                            int position = off_3;
                            int clamped = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int value = indices_out[row_out + (long long)clamped];
                            if (position >= active_0_1) {
                                value = -1;
                            }
                            nxt[0] = value;
                            int position_0 = off_3 + 1;
                            int clamped_1 = ((position_0) < (topk - 1) ? (position_0) : (topk - 1));
                            int value_2 = indices_out[row_out + (long long)clamped_1];
                            if (position_0 >= active_0_1) {
                                value_2 = -1;
                            }
                            nxt[1] = value_2;
                            int position_3 = off_3 + 2;
                            int clamped_4 = ((position_3) < (topk - 1) ? (position_3) : (topk - 1));
                            int value_5 = indices_out[row_out + (long long)clamped_4];
                            if (position_3 >= active_0_1) {
                                value_5 = -1;
                            }
                            nxt[2] = value_5;
                            int position_6 = off_3 + 3;
                            int clamped_7 = ((position_6) < (topk - 1) ? (position_6) : (topk - 1));
                            int value_8 = indices_out[row_out + (long long)clamped_7];
                            if (position_6 >= active_0_1) {
                                value_8 = -1;
                            }
                            nxt[3] = value_8;
                            int off_9 = nb * 64 + 16 + producer_warp * 4;
                            int position_10 = off_9;
                            int clamped_11 = ((position_10) < (topk - 1) ? (position_10) : (topk - 1));
                            int value_12 = indices_out[row_out + (long long)clamped_11];
                            if (position_10 >= active_0_1) {
                                value_12 = -1;
                            }
                            nxt[4] = value_12;
                            int position_13 = off_9 + 1;
                            int clamped_14 = ((position_13) < (topk - 1) ? (position_13) : (topk - 1));
                            int value_15 = indices_out[row_out + (long long)clamped_14];
                            if (position_13 >= active_0_1) {
                                value_15 = -1;
                            }
                            nxt[5] = value_15;
                            int position_16 = off_9 + 2;
                            int clamped_17 = ((position_16) < (topk - 1) ? (position_16) : (topk - 1));
                            int value_18 = indices_out[row_out + (long long)clamped_17];
                            if (position_16 >= active_0_1) {
                                value_18 = -1;
                            }
                            nxt[6] = value_18;
                            int position_19 = off_9 + 3;
                            int clamped_20 = ((position_19) < (topk - 1) ? (position_19) : (topk - 1));
                            int value_21 = indices_out[row_out + (long long)clamped_20];
                            if (position_19 >= active_0_1) {
                                value_21 = -1;
                            }
                            nxt[7] = value_21;
                            int off_22 = nb * 64 + 32 + producer_warp * 4;
                            int position_23 = off_22;
                            int clamped_24 = ((position_23) < (topk - 1) ? (position_23) : (topk - 1));
                            int value_25 = indices_out[row_out + (long long)clamped_24];
                            if (position_23 >= active_0_1) {
                                value_25 = -1;
                            }
                            nxt[8] = value_25;
                            int position_26 = off_22 + 1;
                            int clamped_27 = ((position_26) < (topk - 1) ? (position_26) : (topk - 1));
                            int value_28 = indices_out[row_out + (long long)clamped_27];
                            if (position_26 >= active_0_1) {
                                value_28 = -1;
                            }
                            nxt[9] = value_28;
                            int position_29 = off_22 + 2;
                            int clamped_30 = ((position_29) < (topk - 1) ? (position_29) : (topk - 1));
                            int value_31 = indices_out[row_out + (long long)clamped_30];
                            if (position_29 >= active_0_1) {
                                value_31 = -1;
                            }
                            nxt[10] = value_31;
                            int position_32 = off_22 + 3;
                            int clamped_33 = ((position_32) < (topk - 1) ? (position_32) : (topk - 1));
                            int value_34 = indices_out[row_out + (long long)clamped_33];
                            if (position_32 >= active_0_1) {
                                value_34 = -1;
                            }
                            nxt[11] = value_34;
                            int off_35 = nb * 64 + 48 + producer_warp * 4;
                            int position_36 = off_35;
                            int clamped_37 = ((position_36) < (topk - 1) ? (position_36) : (topk - 1));
                            int value_38 = indices_out[row_out + (long long)clamped_37];
                            if (position_36 >= active_0_1) {
                                value_38 = -1;
                            }
                            nxt[12] = value_38;
                            int position_39 = off_35 + 1;
                            int clamped_40 = ((position_39) < (topk - 1) ? (position_39) : (topk - 1));
                            int value_41 = indices_out[row_out + (long long)clamped_40];
                            if (position_39 >= active_0_1) {
                                value_41 = -1;
                            }
                            nxt[13] = value_41;
                            int position_42 = off_35 + 2;
                            int clamped_43 = ((position_42) < (topk - 1) ? (position_42) : (topk - 1));
                            int value_44 = indices_out[row_out + (long long)clamped_43];
                            if (position_42 >= active_0_1) {
                                value_44 = -1;
                            }
                            nxt[14] = value_44;
                            int position_45 = off_35 + 3;
                            int clamped_46 = ((position_45) < (topk - 1) ? (position_45) : (topk - 1));
                            int value_47 = indices_out[row_out + (long long)clamped_46];
                            if (position_45 >= active_0_1) {
                                value_47 = -1;
                            }
                            nxt[15] = value_47;
                        }
                    }
                }
                #pragma unroll 1
                for (int block_1 = 0; block_1 < blocks_1; block_1++) {
                    if (elect_sync()) {
                        int idx[16];
                        int smallest = num_kv;
                        int largest = -1;
                        int whole_tile = active_0_1 >= block_1 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (convert != 0) {
                            idx[0] = nxt[0];
                            idx[1] = nxt[1];
                            idx[2] = nxt[2];
                            idx[3] = nxt[3];
                            idx[4] = nxt[4];
                            idx[5] = nxt[5];
                            idx[6] = nxt[6];
                            idx[7] = nxt[7];
                            idx[8] = nxt[8];
                            idx[9] = nxt[9];
                            idx[10] = nxt[10];
                            idx[11] = nxt[11];
                            idx[12] = nxt[12];
                            idx[13] = nxt[13];
                            idx[14] = nxt[14];
                            idx[15] = nxt[15];
                            int _max_10 = ((idx[0]) > (idx[1]) ? (idx[0]) : (idx[1]));
                            int _max_11 = ((idx[2]) > (idx[3]) ? (idx[2]) : (idx[3]));
                            int _max_9 = ((_max_10) > (_max_11) ? (_max_10) : (_max_11));
                            int _max_8 = ((largest) > (_max_9) ? (largest) : (_max_9));
                            largest = _max_8;
                            int _min_14 = ((idx[0]) < (idx[1]) ? (idx[0]) : (idx[1]));
                            int _min_15 = ((idx[2]) < (idx[3]) ? (idx[2]) : (idx[3]));
                            int _min_13 = ((_min_14) < (_min_15) ? (_min_14) : (_min_15));
                            int _min_12 = ((smallest) < (_min_13) ? (smallest) : (_min_13));
                            smallest = _min_12;
                            int _max_18 = ((idx[4]) > (idx[5]) ? (idx[4]) : (idx[5]));
                            int _max_19 = ((idx[6]) > (idx[7]) ? (idx[6]) : (idx[7]));
                            int _max_17 = ((_max_18) > (_max_19) ? (_max_18) : (_max_19));
                            int _max_16 = ((largest) > (_max_17) ? (largest) : (_max_17));
                            largest = _max_16;
                            int _min_22 = ((idx[4]) < (idx[5]) ? (idx[4]) : (idx[5]));
                            int _min_23 = ((idx[6]) < (idx[7]) ? (idx[6]) : (idx[7]));
                            int _min_21 = ((_min_22) < (_min_23) ? (_min_22) : (_min_23));
                            int _min_20 = ((smallest) < (_min_21) ? (smallest) : (_min_21));
                            smallest = _min_20;
                            int _max_26 = ((idx[8]) > (idx[9]) ? (idx[8]) : (idx[9]));
                            int _max_27 = ((idx[10]) > (idx[11]) ? (idx[10]) : (idx[11]));
                            int _max_25 = ((_max_26) > (_max_27) ? (_max_26) : (_max_27));
                            int _max_24 = ((largest) > (_max_25) ? (largest) : (_max_25));
                            largest = _max_24;
                            int _min_30 = ((idx[8]) < (idx[9]) ? (idx[8]) : (idx[9]));
                            int _min_31 = ((idx[10]) < (idx[11]) ? (idx[10]) : (idx[11]));
                            int _min_29 = ((_min_30) < (_min_31) ? (_min_30) : (_min_31));
                            int _min_28 = ((smallest) < (_min_29) ? (smallest) : (_min_29));
                            smallest = _min_28;
                            int _max_34 = ((idx[12]) > (idx[13]) ? (idx[12]) : (idx[13]));
                            int _max_35 = ((idx[14]) > (idx[15]) ? (idx[14]) : (idx[15]));
                            int _max_33 = ((_max_34) > (_max_35) ? (_max_34) : (_max_35));
                            int _max_32 = ((largest) > (_max_33) ? (largest) : (_max_33));
                            largest = _max_32;
                            int _min_38 = ((idx[12]) < (idx[13]) ? (idx[12]) : (idx[13]));
                            int _min_39 = ((idx[14]) < (idx[15]) ? (idx[14]) : (idx[15]));
                            int _min_37 = ((_min_38) < (_min_39) ? (_min_38) : (_min_39));
                            int _min_36 = ((smallest) < (_min_37) ? (smallest) : (_min_37));
                            smallest = _min_36;
                            if (blocks_1 > block_1 + 1) {
                                int nb_1 = block_1 + 1;
                                int whole_out_1 = active_0_1 >= nb_1 * 64 + 64 && (topk & 3) == 0;
                                if (whole_out_1 != 0) {
                                    int off_4 = nb_1 * 64 + producer_warp * 4;
                                    int _vec_load_4[4];
                                    {
                                        const int4* _ivptr_40 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_4) + 0);
                                        int4 _ivld_40;
                                        _ivld_40 = *_ivptr_40;
                                        _vec_load_4[0 + 0] = _ivld_40.x;
                                        _vec_load_4[0 + 1] = _ivld_40.y;
                                        _vec_load_4[0 + 2] = _ivld_40.z;
                                        _vec_load_4[0 + 3] = _ivld_40.w;
                                    }
                                    nxt[0] = _vec_load_4[0];
                                    nxt[1] = _vec_load_4[1];
                                    nxt[2] = _vec_load_4[2];
                                    nxt[3] = _vec_load_4[3];
                                    int off_0_1 = nb_1 * 64 + 16 + producer_warp * 4;
                                    int _vec_load_5[4];
                                    {
                                        const int4* _ivptr_41 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_0_1) + 0);
                                        int4 _ivld_41;
                                        _ivld_41 = *_ivptr_41;
                                        _vec_load_5[0 + 0] = _ivld_41.x;
                                        _vec_load_5[0 + 1] = _ivld_41.y;
                                        _vec_load_5[0 + 2] = _ivld_41.z;
                                        _vec_load_5[0 + 3] = _ivld_41.w;
                                    }
                                    nxt[4] = _vec_load_5[0];
                                    nxt[5] = _vec_load_5[1];
                                    nxt[6] = _vec_load_5[2];
                                    nxt[7] = _vec_load_5[3];
                                    int off_1_1 = nb_1 * 64 + 32 + producer_warp * 4;
                                    int _vec_load_6[4];
                                    {
                                        const int4* _ivptr_42 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_1_1) + 0);
                                        int4 _ivld_42;
                                        _ivld_42 = *_ivptr_42;
                                        _vec_load_6[0 + 0] = _ivld_42.x;
                                        _vec_load_6[0 + 1] = _ivld_42.y;
                                        _vec_load_6[0 + 2] = _ivld_42.z;
                                        _vec_load_6[0 + 3] = _ivld_42.w;
                                    }
                                    nxt[8] = _vec_load_6[0];
                                    nxt[9] = _vec_load_6[1];
                                    nxt[10] = _vec_load_6[2];
                                    nxt[11] = _vec_load_6[3];
                                    int off_2_1 = nb_1 * 64 + 48 + producer_warp * 4;
                                    int _vec_load_7[4];
                                    {
                                        const int4* _ivptr_43 = reinterpret_cast<const int4*>(indices_out + (row_out + (long long)off_2_1) + 0);
                                        int4 _ivld_43;
                                        _ivld_43 = *_ivptr_43;
                                        _vec_load_7[0 + 0] = _ivld_43.x;
                                        _vec_load_7[0 + 1] = _ivld_43.y;
                                        _vec_load_7[0 + 2] = _ivld_43.z;
                                        _vec_load_7[0 + 3] = _ivld_43.w;
                                    }
                                    nxt[12] = _vec_load_7[0];
                                    nxt[13] = _vec_load_7[1];
                                    nxt[14] = _vec_load_7[2];
                                    nxt[15] = _vec_load_7[3];
                                } else {
                                    int off_5 = nb_1 * 64 + producer_warp * 4;
                                    int position_1 = off_5;
                                    int clamped_2 = ((position_1) < (topk - 1) ? (position_1) : (topk - 1));
                                    int value_1 = indices_out[row_out + (long long)clamped_2];
                                    if (position_1 >= active_0_1) {
                                        value_1 = -1;
                                    }
                                    nxt[0] = value_1;
                                    int position_0_1 = off_5 + 1;
                                    int clamped_1_1 = ((position_0_1) < (topk - 1) ? (position_0_1) : (topk - 1));
                                    int value_2_1 = indices_out[row_out + (long long)clamped_1_1];
                                    if (position_0_1 >= active_0_1) {
                                        value_2_1 = -1;
                                    }
                                    nxt[1] = value_2_1;
                                    int position_3_1 = off_5 + 2;
                                    int clamped_4_1 = ((position_3_1) < (topk - 1) ? (position_3_1) : (topk - 1));
                                    int value_5_1 = indices_out[row_out + (long long)clamped_4_1];
                                    if (position_3_1 >= active_0_1) {
                                        value_5_1 = -1;
                                    }
                                    nxt[2] = value_5_1;
                                    int position_6_1 = off_5 + 3;
                                    int clamped_7_1 = ((position_6_1) < (topk - 1) ? (position_6_1) : (topk - 1));
                                    int value_8_1 = indices_out[row_out + (long long)clamped_7_1];
                                    if (position_6_1 >= active_0_1) {
                                        value_8_1 = -1;
                                    }
                                    nxt[3] = value_8_1;
                                    int off_9_1 = nb_1 * 64 + 16 + producer_warp * 4;
                                    int position_10_1 = off_9_1;
                                    int clamped_11_1 = ((position_10_1) < (topk - 1) ? (position_10_1) : (topk - 1));
                                    int value_12_1 = indices_out[row_out + (long long)clamped_11_1];
                                    if (position_10_1 >= active_0_1) {
                                        value_12_1 = -1;
                                    }
                                    nxt[4] = value_12_1;
                                    int position_13_1 = off_9_1 + 1;
                                    int clamped_14_1 = ((position_13_1) < (topk - 1) ? (position_13_1) : (topk - 1));
                                    int value_15_1 = indices_out[row_out + (long long)clamped_14_1];
                                    if (position_13_1 >= active_0_1) {
                                        value_15_1 = -1;
                                    }
                                    nxt[5] = value_15_1;
                                    int position_16_1 = off_9_1 + 2;
                                    int clamped_17_1 = ((position_16_1) < (topk - 1) ? (position_16_1) : (topk - 1));
                                    int value_18_1 = indices_out[row_out + (long long)clamped_17_1];
                                    if (position_16_1 >= active_0_1) {
                                        value_18_1 = -1;
                                    }
                                    nxt[6] = value_18_1;
                                    int position_19_1 = off_9_1 + 3;
                                    int clamped_20_1 = ((position_19_1) < (topk - 1) ? (position_19_1) : (topk - 1));
                                    int value_21_1 = indices_out[row_out + (long long)clamped_20_1];
                                    if (position_19_1 >= active_0_1) {
                                        value_21_1 = -1;
                                    }
                                    nxt[7] = value_21_1;
                                    int off_22_1 = nb_1 * 64 + 32 + producer_warp * 4;
                                    int position_23_1 = off_22_1;
                                    int clamped_24_1 = ((position_23_1) < (topk - 1) ? (position_23_1) : (topk - 1));
                                    int value_25_1 = indices_out[row_out + (long long)clamped_24_1];
                                    if (position_23_1 >= active_0_1) {
                                        value_25_1 = -1;
                                    }
                                    nxt[8] = value_25_1;
                                    int position_26_1 = off_22_1 + 1;
                                    int clamped_27_1 = ((position_26_1) < (topk - 1) ? (position_26_1) : (topk - 1));
                                    int value_28_1 = indices_out[row_out + (long long)clamped_27_1];
                                    if (position_26_1 >= active_0_1) {
                                        value_28_1 = -1;
                                    }
                                    nxt[9] = value_28_1;
                                    int position_29_1 = off_22_1 + 2;
                                    int clamped_30_1 = ((position_29_1) < (topk - 1) ? (position_29_1) : (topk - 1));
                                    int value_31_1 = indices_out[row_out + (long long)clamped_30_1];
                                    if (position_29_1 >= active_0_1) {
                                        value_31_1 = -1;
                                    }
                                    nxt[10] = value_31_1;
                                    int position_32_1 = off_22_1 + 3;
                                    int clamped_33_1 = ((position_32_1) < (topk - 1) ? (position_32_1) : (topk - 1));
                                    int value_34_1 = indices_out[row_out + (long long)clamped_33_1];
                                    if (position_32_1 >= active_0_1) {
                                        value_34_1 = -1;
                                    }
                                    nxt[11] = value_34_1;
                                    int off_35_1 = nb_1 * 64 + 48 + producer_warp * 4;
                                    int position_36_1 = off_35_1;
                                    int clamped_37_1 = ((position_36_1) < (topk - 1) ? (position_36_1) : (topk - 1));
                                    int value_38_1 = indices_out[row_out + (long long)clamped_37_1];
                                    if (position_36_1 >= active_0_1) {
                                        value_38_1 = -1;
                                    }
                                    nxt[12] = value_38_1;
                                    int position_39_1 = off_35_1 + 1;
                                    int clamped_40_1 = ((position_39_1) < (topk - 1) ? (position_39_1) : (topk - 1));
                                    int value_41_1 = indices_out[row_out + (long long)clamped_40_1];
                                    if (position_39_1 >= active_0_1) {
                                        value_41_1 = -1;
                                    }
                                    nxt[13] = value_41_1;
                                    int position_42_1 = off_35_1 + 2;
                                    int clamped_43_1 = ((position_42_1) < (topk - 1) ? (position_42_1) : (topk - 1));
                                    int value_44_1 = indices_out[row_out + (long long)clamped_43_1];
                                    if (position_42_1 >= active_0_1) {
                                        value_44_1 = -1;
                                    }
                                    nxt[14] = value_44_1;
                                    int position_45_1 = off_35_1 + 3;
                                    int clamped_46_1 = ((position_45_1) < (topk - 1) ? (position_45_1) : (topk - 1));
                                    int value_47_1 = indices_out[row_out + (long long)clamped_46_1];
                                    if (position_45_1 >= active_0_1) {
                                        value_47_1 = -1;
                                    }
                                    nxt[15] = value_47_1;
                                }
                            }
                        } else if (whole_tile != 0) {
                            int _vec_load_8[4];
                            {
                                const int4* _ivptr_44 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + producer_warp * 4)) + 0);
                                int4 _ivld_44;
                                _ivld_44 = *_ivptr_44;
                                _vec_load_8[0 + 0] = _ivld_44.x;
                                _vec_load_8[0 + 1] = _ivld_44.y;
                                _vec_load_8[0 + 2] = _ivld_44.z;
                                _vec_load_8[0 + 3] = _ivld_44.w;
                            }
                            idx[0] = _vec_load_8[0];
                            idx[1] = _vec_load_8[1];
                            idx[2] = _vec_load_8[2];
                            idx[3] = _vec_load_8[3];
                            int _max_47 = ((_vec_load_8[0]) > (_vec_load_8[1]) ? (_vec_load_8[0]) : (_vec_load_8[1]));
                            int _max_48 = ((_vec_load_8[2]) > (_vec_load_8[3]) ? (_vec_load_8[2]) : (_vec_load_8[3]));
                            int _max_46 = ((_max_47) > (_max_48) ? (_max_47) : (_max_48));
                            int _max_45 = ((largest) > (_max_46) ? (largest) : (_max_46));
                            largest = _max_45;
                            int _min_51 = ((_vec_load_8[0]) < (_vec_load_8[1]) ? (_vec_load_8[0]) : (_vec_load_8[1]));
                            int _min_52 = ((_vec_load_8[2]) < (_vec_load_8[3]) ? (_vec_load_8[2]) : (_vec_load_8[3]));
                            int _min_50 = ((_min_51) < (_min_52) ? (_min_51) : (_min_52));
                            int _min_49 = ((smallest) < (_min_50) ? (smallest) : (_min_50));
                            smallest = _min_49;
                            int _vec_load_9[4];
                            {
                                const int4* _ivptr_53 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 16 + producer_warp * 4)) + 0);
                                int4 _ivld_53;
                                _ivld_53 = *_ivptr_53;
                                _vec_load_9[0 + 0] = _ivld_53.x;
                                _vec_load_9[0 + 1] = _ivld_53.y;
                                _vec_load_9[0 + 2] = _ivld_53.z;
                                _vec_load_9[0 + 3] = _ivld_53.w;
                            }
                            idx[4] = _vec_load_9[0];
                            idx[5] = _vec_load_9[1];
                            idx[6] = _vec_load_9[2];
                            idx[7] = _vec_load_9[3];
                            int _max_56 = ((_vec_load_9[0]) > (_vec_load_9[1]) ? (_vec_load_9[0]) : (_vec_load_9[1]));
                            int _max_57 = ((_vec_load_9[2]) > (_vec_load_9[3]) ? (_vec_load_9[2]) : (_vec_load_9[3]));
                            int _max_55 = ((_max_56) > (_max_57) ? (_max_56) : (_max_57));
                            int _max_54 = ((largest) > (_max_55) ? (largest) : (_max_55));
                            largest = _max_54;
                            int _min_60 = ((_vec_load_9[0]) < (_vec_load_9[1]) ? (_vec_load_9[0]) : (_vec_load_9[1]));
                            int _min_61 = ((_vec_load_9[2]) < (_vec_load_9[3]) ? (_vec_load_9[2]) : (_vec_load_9[3]));
                            int _min_59 = ((_min_60) < (_min_61) ? (_min_60) : (_min_61));
                            int _min_58 = ((smallest) < (_min_59) ? (smallest) : (_min_59));
                            smallest = _min_58;
                            int _vec_load_10[4];
                            {
                                const int4* _ivptr_62 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 32 + producer_warp * 4)) + 0);
                                int4 _ivld_62;
                                _ivld_62 = *_ivptr_62;
                                _vec_load_10[0 + 0] = _ivld_62.x;
                                _vec_load_10[0 + 1] = _ivld_62.y;
                                _vec_load_10[0 + 2] = _ivld_62.z;
                                _vec_load_10[0 + 3] = _ivld_62.w;
                            }
                            idx[8] = _vec_load_10[0];
                            idx[9] = _vec_load_10[1];
                            idx[10] = _vec_load_10[2];
                            idx[11] = _vec_load_10[3];
                            int _max_65 = ((_vec_load_10[0]) > (_vec_load_10[1]) ? (_vec_load_10[0]) : (_vec_load_10[1]));
                            int _max_66 = ((_vec_load_10[2]) > (_vec_load_10[3]) ? (_vec_load_10[2]) : (_vec_load_10[3]));
                            int _max_64 = ((_max_65) > (_max_66) ? (_max_65) : (_max_66));
                            int _max_63 = ((largest) > (_max_64) ? (largest) : (_max_64));
                            largest = _max_63;
                            int _min_69 = ((_vec_load_10[0]) < (_vec_load_10[1]) ? (_vec_load_10[0]) : (_vec_load_10[1]));
                            int _min_70 = ((_vec_load_10[2]) < (_vec_load_10[3]) ? (_vec_load_10[2]) : (_vec_load_10[3]));
                            int _min_68 = ((_min_69) < (_min_70) ? (_min_69) : (_min_70));
                            int _min_67 = ((smallest) < (_min_68) ? (smallest) : (_min_68));
                            smallest = _min_67;
                            int _vec_load_11[4];
                            {
                                const int4* _ivptr_71 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 64 + 48 + producer_warp * 4)) + 0);
                                int4 _ivld_71;
                                _ivld_71 = *_ivptr_71;
                                _vec_load_11[0 + 0] = _ivld_71.x;
                                _vec_load_11[0 + 1] = _ivld_71.y;
                                _vec_load_11[0 + 2] = _ivld_71.z;
                                _vec_load_11[0 + 3] = _ivld_71.w;
                            }
                            idx[12] = _vec_load_11[0];
                            idx[13] = _vec_load_11[1];
                            idx[14] = _vec_load_11[2];
                            idx[15] = _vec_load_11[3];
                            int _max_74 = ((_vec_load_11[0]) > (_vec_load_11[1]) ? (_vec_load_11[0]) : (_vec_load_11[1]));
                            int _max_75 = ((_vec_load_11[2]) > (_vec_load_11[3]) ? (_vec_load_11[2]) : (_vec_load_11[3]));
                            int _max_73 = ((_max_74) > (_max_75) ? (_max_74) : (_max_75));
                            int _max_72 = ((largest) > (_max_73) ? (largest) : (_max_73));
                            largest = _max_72;
                            int _min_78 = ((_vec_load_11[0]) < (_vec_load_11[1]) ? (_vec_load_11[0]) : (_vec_load_11[1]));
                            int _min_79 = ((_vec_load_11[2]) < (_vec_load_11[3]) ? (_vec_load_11[2]) : (_vec_load_11[3]));
                            int _min_77 = ((_min_78) < (_min_79) ? (_min_78) : (_min_79));
                            int _min_76 = ((smallest) < (_min_77) ? (smallest) : (_min_77));
                            smallest = _min_76;
                        } else {
                            int position_2 = block_1 * 64 + producer_warp * 4;
                            int clamped_3 = ((position_2) < (topk - 1) ? (position_2) : (topk - 1));
                            int value_3 = indices[row_base + (long long)clamped_3];
                            if (position_2 >= active_0_1) {
                                value_3 = -1;
                            }
                            idx[0] = value_3;
                            int _max_80 = ((largest) > (value_3) ? (largest) : (value_3));
                            largest = _max_80;
                            int _min_81 = ((smallest) < (value_3) ? (smallest) : (value_3));
                            smallest = _min_81;
                            int position_0_2 = block_1 * 64 + producer_warp * 4 + 1;
                            int clamped_1_2 = ((position_0_2) < (topk - 1) ? (position_0_2) : (topk - 1));
                            int value_2_2 = indices[row_base + (long long)clamped_1_2];
                            if (position_0_2 >= active_0_1) {
                                value_2_2 = -1;
                            }
                            idx[1] = value_2_2;
                            int _max_82 = ((largest) > (value_2_2) ? (largest) : (value_2_2));
                            largest = _max_82;
                            int _min_83 = ((smallest) < (value_2_2) ? (smallest) : (value_2_2));
                            smallest = _min_83;
                            int position_3_2 = block_1 * 64 + producer_warp * 4 + 2;
                            int clamped_4_2 = ((position_3_2) < (topk - 1) ? (position_3_2) : (topk - 1));
                            int value_5_2 = indices[row_base + (long long)clamped_4_2];
                            if (position_3_2 >= active_0_1) {
                                value_5_2 = -1;
                            }
                            idx[2] = value_5_2;
                            int _max_84 = ((largest) > (value_5_2) ? (largest) : (value_5_2));
                            largest = _max_84;
                            int _min_85 = ((smallest) < (value_5_2) ? (smallest) : (value_5_2));
                            smallest = _min_85;
                            int position_6_2 = block_1 * 64 + producer_warp * 4 + 3;
                            int clamped_7_2 = ((position_6_2) < (topk - 1) ? (position_6_2) : (topk - 1));
                            int value_8_2 = indices[row_base + (long long)clamped_7_2];
                            if (position_6_2 >= active_0_1) {
                                value_8_2 = -1;
                            }
                            idx[3] = value_8_2;
                            int _max_86 = ((largest) > (value_8_2) ? (largest) : (value_8_2));
                            largest = _max_86;
                            int _min_87 = ((smallest) < (value_8_2) ? (smallest) : (value_8_2));
                            smallest = _min_87;
                            int position_9 = block_1 * 64 + 16 + producer_warp * 4;
                            int clamped_10 = ((position_9) < (topk - 1) ? (position_9) : (topk - 1));
                            int value_11 = indices[row_base + (long long)clamped_10];
                            if (position_9 >= active_0_1) {
                                value_11 = -1;
                            }
                            idx[4] = value_11;
                            int _max_88 = ((largest) > (value_11) ? (largest) : (value_11));
                            largest = _max_88;
                            int _min_89 = ((smallest) < (value_11) ? (smallest) : (value_11));
                            smallest = _min_89;
                            int position_12 = block_1 * 64 + 16 + producer_warp * 4 + 1;
                            int clamped_13 = ((position_12) < (topk - 1) ? (position_12) : (topk - 1));
                            int value_14 = indices[row_base + (long long)clamped_13];
                            if (position_12 >= active_0_1) {
                                value_14 = -1;
                            }
                            idx[5] = value_14;
                            int _max_90 = ((largest) > (value_14) ? (largest) : (value_14));
                            largest = _max_90;
                            int _min_91 = ((smallest) < (value_14) ? (smallest) : (value_14));
                            smallest = _min_91;
                            int position_15 = block_1 * 64 + 16 + producer_warp * 4 + 2;
                            int clamped_16 = ((position_15) < (topk - 1) ? (position_15) : (topk - 1));
                            int value_17 = indices[row_base + (long long)clamped_16];
                            if (position_15 >= active_0_1) {
                                value_17 = -1;
                            }
                            idx[6] = value_17;
                            int _max_92 = ((largest) > (value_17) ? (largest) : (value_17));
                            largest = _max_92;
                            int _min_93 = ((smallest) < (value_17) ? (smallest) : (value_17));
                            smallest = _min_93;
                            int position_18 = block_1 * 64 + 16 + producer_warp * 4 + 3;
                            int clamped_19 = ((position_18) < (topk - 1) ? (position_18) : (topk - 1));
                            int value_20 = indices[row_base + (long long)clamped_19];
                            if (position_18 >= active_0_1) {
                                value_20 = -1;
                            }
                            idx[7] = value_20;
                            int _max_94 = ((largest) > (value_20) ? (largest) : (value_20));
                            largest = _max_94;
                            int _min_95 = ((smallest) < (value_20) ? (smallest) : (value_20));
                            smallest = _min_95;
                            int position_21 = block_1 * 64 + 32 + producer_warp * 4;
                            int clamped_22 = ((position_21) < (topk - 1) ? (position_21) : (topk - 1));
                            int value_23 = indices[row_base + (long long)clamped_22];
                            if (position_21 >= active_0_1) {
                                value_23 = -1;
                            }
                            idx[8] = value_23;
                            int _max_96 = ((largest) > (value_23) ? (largest) : (value_23));
                            largest = _max_96;
                            int _min_97 = ((smallest) < (value_23) ? (smallest) : (value_23));
                            smallest = _min_97;
                            int position_24 = block_1 * 64 + 32 + producer_warp * 4 + 1;
                            int clamped_25 = ((position_24) < (topk - 1) ? (position_24) : (topk - 1));
                            int value_26 = indices[row_base + (long long)clamped_25];
                            if (position_24 >= active_0_1) {
                                value_26 = -1;
                            }
                            idx[9] = value_26;
                            int _max_98 = ((largest) > (value_26) ? (largest) : (value_26));
                            largest = _max_98;
                            int _min_99 = ((smallest) < (value_26) ? (smallest) : (value_26));
                            smallest = _min_99;
                            int position_27 = block_1 * 64 + 32 + producer_warp * 4 + 2;
                            int clamped_28 = ((position_27) < (topk - 1) ? (position_27) : (topk - 1));
                            int value_29 = indices[row_base + (long long)clamped_28];
                            if (position_27 >= active_0_1) {
                                value_29 = -1;
                            }
                            idx[10] = value_29;
                            int _max_100 = ((largest) > (value_29) ? (largest) : (value_29));
                            largest = _max_100;
                            int _min_101 = ((smallest) < (value_29) ? (smallest) : (value_29));
                            smallest = _min_101;
                            int position_30 = block_1 * 64 + 32 + producer_warp * 4 + 3;
                            int clamped_31 = ((position_30) < (topk - 1) ? (position_30) : (topk - 1));
                            int value_32 = indices[row_base + (long long)clamped_31];
                            if (position_30 >= active_0_1) {
                                value_32 = -1;
                            }
                            idx[11] = value_32;
                            int _max_102 = ((largest) > (value_32) ? (largest) : (value_32));
                            largest = _max_102;
                            int _min_103 = ((smallest) < (value_32) ? (smallest) : (value_32));
                            smallest = _min_103;
                            int position_33 = block_1 * 64 + 48 + producer_warp * 4;
                            int clamped_34 = ((position_33) < (topk - 1) ? (position_33) : (topk - 1));
                            int value_35 = indices[row_base + (long long)clamped_34];
                            if (position_33 >= active_0_1) {
                                value_35 = -1;
                            }
                            idx[12] = value_35;
                            int _max_104 = ((largest) > (value_35) ? (largest) : (value_35));
                            largest = _max_104;
                            int _min_105 = ((smallest) < (value_35) ? (smallest) : (value_35));
                            smallest = _min_105;
                            int position_36_2 = block_1 * 64 + 48 + producer_warp * 4 + 1;
                            int clamped_37_2 = ((position_36_2) < (topk - 1) ? (position_36_2) : (topk - 1));
                            int value_38_2 = indices[row_base + (long long)clamped_37_2];
                            if (position_36_2 >= active_0_1) {
                                value_38_2 = -1;
                            }
                            idx[13] = value_38_2;
                            int _max_106 = ((largest) > (value_38_2) ? (largest) : (value_38_2));
                            largest = _max_106;
                            int _min_107 = ((smallest) < (value_38_2) ? (smallest) : (value_38_2));
                            smallest = _min_107;
                            int position_39_2 = block_1 * 64 + 48 + producer_warp * 4 + 2;
                            int clamped_40_2 = ((position_39_2) < (topk - 1) ? (position_39_2) : (topk - 1));
                            int value_41_2 = indices[row_base + (long long)clamped_40_2];
                            if (position_39_2 >= active_0_1) {
                                value_41_2 = -1;
                            }
                            idx[14] = value_41_2;
                            int _max_108 = ((largest) > (value_41_2) ? (largest) : (value_41_2));
                            largest = _max_108;
                            int _min_109 = ((smallest) < (value_41_2) ? (smallest) : (value_41_2));
                            smallest = _min_109;
                            int position_42_2 = block_1 * 64 + 48 + producer_warp * 4 + 3;
                            int clamped_43_2 = ((position_42_2) < (topk - 1) ? (position_42_2) : (topk - 1));
                            int value_44_2 = indices[row_base + (long long)clamped_43_2];
                            if (position_42_2 >= active_0_1) {
                                value_44_2 = -1;
                            }
                            idx[15] = value_44_2;
                            int _max_110 = ((largest) > (value_44_2) ? (largest) : (value_44_2));
                            largest = _max_110;
                            int _min_111 = ((smallest) < (value_44_2) ? (smallest) : (value_44_2));
                            smallest = _min_111;
                        }
                        int skip = (smallest == num_kv || largest == -1) && block_1 >= 3;
                        if (block_1 == 1) {
                            mbarrier_wait(q_copied_addr, outer_phase_1);
                        }
                        if (block_1 == 2) {
                            mbarrier_wait(o_written_addr, outer_phase_1);
                        }
                        int slot_1 = cursor_1 % 3;
                        unsigned int phase_1 = cursor_1 / 3 & 1;
                        int free_stage = slot_1;
                        unsigned int free_phase = phase_1 ^ 1;
                        mbarrier_wait(sv_done_addr + (free_stage) * 8, free_phase);
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048), "l"((&kv_latent)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096), "l"((&kv_latent)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144), "l"((&kv_latent)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 8192), "l"((&kv_latent)), "r"(64), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 16384), "l"((&kv_latent)), "r"(128), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 24576), "l"((&kv_latent)), "r"(192), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 2048 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 4096 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 32768), "l"((&kv_latent)), "r"(256), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 40960), "l"((&kv_latent)), "r"(320), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 49152), "l"((&kv_latent)), "r"(384), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(qko_addr + (unsigned int)(slot_1 * 65536) + (unsigned int)(producer_warp * 512) + 6144 + 57344), "l"((&kv_latent)), "r"(448), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (slot_1 * 2) * 8), "r"((uint32_t)(8192)) : "memory");
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (slot_1 * 2 + 1) * 8), "r"((uint32_t)(8192)) : "memory");
                        }
                        cursor_1 = cursor_1 + 1;
                    }
                }
                if (elect_sync()) {
                    if (blocks_1 <= 2) {
                        mbarrier_wait(o_written_addr, outer_phase_1);
                    }
                    mbarrier_arrive(o_written_waited_addr);
                }
                __syncwarp();
                mbarrier_wait(clc_full_addr, outer_phase_1);
                uint32_t _clc_valid_1 = 0;
                uint32_t _clc_ctaid_x_1;
                uint32_t _clc_ctaid_y_1;
                uint32_t _clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_1 == 0) {
                    break;
                }
                query_1 = _clc_ctaid_x_1;
                outer_phase_1 ^= 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int query_2 = blockIdx.x;
            unsigned int outer_phase_2 = 0;
            int cursor_2 = 0;
            int next_valid = 1;
            int next_query = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int outer_2 = 0; outer_2 < num_queries; outer_2++) {
                    int active_2 = topk;
                    if (derive_length != 0) {
                        int len_slot_2 = outer_2 & 1;
                        unsigned int len_phase_2 = outer_2 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_2) * 8, len_phase_2);
                        int _max_1_2 = ((len_cells[len_slot_2]) > (0) ? (len_cells[len_slot_2]) : (0));
                        int _min_0_2 = ((_max_1_2) < (topk) ? (_max_1_2) : (topk));
                        active_2 = _min_0_2;
                    } else if (has_topk_length != 0) {
                        int _max_3_2 = ((topk_length[query_2]) > (0) ? (topk_length[query_2]) : (0));
                        int _min_2_2 = ((_max_3_2) < (topk) ? (_max_3_2) : (topk));
                        active_2 = _min_2_2;
                    }
                    int active_0_2 = active_2;
                    int blocks_2 = (((active_0_2 + 63) / 64) > (2) ? ((active_0_2 + 63) / 64) : (2));
                    int q_slot = (cursor_2 + 1) % 3;
                    if (outer_2 == 0) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(qko_addr + (unsigned int)(q_slot * 65536)), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(query_2),
                               "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive_expect_tx(q_full_addr, 65536);
                    }
                    mbarrier_wait(q_full_addr, outer_phase_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256)), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 8)), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 16)), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 24)), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32)), "l"(_tcgen05_cp_desc_8)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 8)), "l"(_tcgen05_cp_desc_9)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 16)), "l"(_tcgen05_cp_desc_10)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 16384 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 32 + 24)), "l"(_tcgen05_cp_desc_11)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64)), "l"(_tcgen05_cp_desc_12)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 8)), "l"(_tcgen05_cp_desc_13)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 16)), "l"(_tcgen05_cp_desc_14)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 32768 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 64 + 24)), "l"(_tcgen05_cp_desc_15)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_16 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96)), "l"(_tcgen05_cp_desc_16)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_17 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 32)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 8)), "l"(_tcgen05_cp_desc_17)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_18 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 64)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 16)), "l"(_tcgen05_cp_desc_18)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
#if __CUDA_ARCH__ == 1070
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#else
                        uint64_t _tcgen05_cp_desc_19 = ((((uint64_t)(qko_addr + (unsigned int)(q_slot * 65536) + 49152 + 96)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
#endif
                        asm volatile(
                            "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                            :: "r"((uint32_t)(tmem_tmem + 256 + 96 + 24)), "l"(_tcgen05_cp_desc_19)
                            : "memory");
                    }
                    tcgen05_commit(q_copied_addr);
                    #pragma unroll 1
                    for (int block_2 = 0; block_2 < blocks_2 + 1; block_2++) {
                        if (blocks_2 > block_2) {
                            int slot_2 = cursor_2 % 3;
                            unsigned int phase_2 = cursor_2 / 3 & 1;
                            int p_free_phase = cursor_2 & 1 ^ 1;
                            mbarrier_wait(p_free_addr, p_free_phase);
                            if (block_2 == 0) {
                                mbarrier_wait(q_copied_addr, outer_phase_2);
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_addr + (slot_2 * 2) * 8, 32768);
                            int kv_stage = slot_2 * 2;
                            unsigned int kv_phase = phase_2;
                            mbarrier_wait(kv_ready_addr + (kv_stage) * 8, kv_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_20 = (((k_dual_addr) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_20 = (((k_dual_addr) >> 4) & 0x3FFF) + (slot_2) * 4096;
#endif
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (384))), "r"(_smem_desc_lo_20), "r"(tmem_tmem + 256), "r"(0) : "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive_expect_tx(kv_ready_addr + (slot_2 * 2 + 1) * 8, 32768);
                            int kv_stage_0 = slot_2 * 2 + 1;
                            unsigned int kv_phase_1 = phase_2;
                            mbarrier_wait(kv_ready_addr + (kv_stage_0) * 8, kv_phase_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_21 = (((k_dual_addr + 32768) >> 4) & 0x7FFF) + (slot_2) * 4096;
#else
                            int _smem_desc_lo_21 = (((k_dual_addr + 32768) >> 4) & 0x3FFF) + (slot_2) * 4096;
#endif
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem + (384))), "r"(_smem_desc_lo_21), "r"(tmem_tmem + 320), "r"(1) : "memory");
                            tcgen05_commit(qk_done_addr + (slot_2) * 8);
                        }
                        if (block_2 > 0) {
                            int slot_3 = (cursor_2 - 1) % 3;
                            int so_ready_phase = cursor_2 - 1 & 1;
                            mbarrier_wait(so_ready_addr, so_ready_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x7FFF) | 0x400000;
                            int _smem_desc_lo_22 = ((((v_addr) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _mma_a_lo_0 = (((probability_addr) >> 4) & 0x3FFF) | 0x400000;
                            int _smem_desc_lo_22 = ((((v_addr) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_22), "r"(tmem_tmem), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_23 = ((((v_addr + 32768) >> 4) & 0x7FFF) | 0x2000000) + (slot_3) * 4096;
#else
                            int _smem_desc_lo_23 = ((((v_addr + 32768) >> 4) & 0x3FFF) | 0x2000000) + (slot_3) * 4096;
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
                    "mov.b32 adhi, 0x00004008;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_smem_desc_lo_23), "r"((tmem_tmem + (128))), "r"(((block_2 == 1) ? 0 : 1)) : "memory");
                            tcgen05_commit(sv_done_addr + (slot_3) * 8);
                        }
                        if (block_2 == blocks_2 - 1) {
                            mbarrier_wait(clc_full_addr, outer_phase_2);
                            uint32_t _clc_valid_2 = 0;
                            uint32_t _clc_ctaid_x_2;
                            uint32_t _clc_ctaid_y_2;
                            uint32_t _clc_ctaid_z_2;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %3, 1, 0, p1;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                                : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                                : "memory");
                            next_valid = _clc_valid_2;
                            next_query = _clc_ctaid_x_2;
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(clc_empty_addr);
                            if (next_valid != 0) {
                                mbarrier_wait(sv_done_addr + ((cursor_2 - 1) % 3) * 8, (cursor_2 - 1) / 3 & 1);
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(qko_addr + (unsigned int)((cursor_2 + 2) % 3 * 65536)), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(next_query),
                                       "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive_expect_tx(q_full_addr, 65536);
                            }
                        }
                        cursor_2 = cursor_2 + 1;
                    }
                    cursor_2 = cursor_2 - 1;
                    if (next_valid == 0) {
                        break;
                    }
                    query_2 = next_query;
                    outer_phase_2 ^= 1;
                }
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 9) {
        { // metadata_main
            if (lane < 8) {
                int query_3 = blockIdx.x;
                unsigned int outer_phase_3 = 0;
                int cursor_3 = 0;
                #pragma unroll 1
                for (int outer_3 = 0; outer_3 < num_queries; outer_3++) {
                    int active_3 = topk;
                    if (derive_length != 0) {
                        int len_slot_3 = outer_3 & 1;
                        unsigned int len_phase_3 = outer_3 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_3) * 8, len_phase_3);
                        int _max_1_3 = ((len_cells[len_slot_3]) > (0) ? (len_cells[len_slot_3]) : (0));
                        int _min_0_3 = ((_max_1_3) < (topk) ? (_max_1_3) : (topk));
                        active_3 = _min_0_3;
                    } else if (has_topk_length != 0) {
                        int _max_3_3 = ((topk_length[query_3]) > (0) ? (topk_length[query_3]) : (0));
                        int _min_2_3 = ((_max_3_3) < (topk) ? (_max_3_3) : (topk));
                        active_3 = _min_2_3;
                    }
                    int active_0_3 = active_3;
                    int blocks_3 = (((active_0_3 + 63) / 64) > (2) ? ((active_0_3 + 63) / 64) : (2));
                    long long row_base_1 = (long long)indices_offset + (long long)query_3 * (long long)idx_stride;
                    int bound = num_kv - 1;
                    if (derive_length != 0) {
                        bound = bound_cells[outer_3 & 1];
                    }
                    #pragma unroll 1
                    for (int block_3 = 0; block_3 < blocks_3; block_3++) {
                        unsigned int mask = 0;
                        int position_4 = block_3 * 64 + lane * 8;
                        int whole_tile_1 = active_0_3 >= block_3 * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0;
                        if (convert == 1) {
                            if (active_0_3 > position_4) {
                                mask = mask | 1;
                            }
                            if (active_0_3 > position_4 + 1) {
                                mask = mask | 2;
                            }
                            if (active_0_3 > position_4 + 2) {
                                mask = mask | 4;
                            }
                            if (active_0_3 > position_4 + 3) {
                                mask = mask | 8;
                            }
                            if (active_0_3 > position_4 + 4) {
                                mask = mask | 16;
                            }
                            if (active_0_3 > position_4 + 5) {
                                mask = mask | 32;
                            }
                            if (active_0_3 > position_4 + 6) {
                                mask = mask | 64;
                            }
                            if (active_0_3 > position_4 + 7) {
                                mask = mask | 128;
                            }
                        } else if (whole_tile_1 != 0) {
                            int _vec_load_12[8];
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
                                    : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(indices + (row_base_1 + (long long)position_4) + (0))) : "memory");
                                _vec_load_12[0 + 0] = (int32_t)_iv_4_0;
                                _vec_load_12[0 + 1] = (int32_t)_iv_4_1;
                                _vec_load_12[0 + 2] = (int32_t)_iv_4_2;
                                _vec_load_12[0 + 3] = (int32_t)_iv_4_3;
                                _vec_load_12[0 + 4] = (int32_t)_iv_4_4;
                                _vec_load_12[0 + 5] = (int32_t)_iv_4_5;
                                _vec_load_12[0 + 6] = (int32_t)_iv_4_6;
                                _vec_load_12[0 + 7] = (int32_t)_iv_4_7;
                            }
                            if (_vec_load_12[0] >= 0 && _vec_load_12[0] <= bound) {
                                mask = mask | 1;
                            }
                            if (_vec_load_12[1] >= 0 && _vec_load_12[1] <= bound) {
                                mask = mask | 2;
                            }
                            if (_vec_load_12[2] >= 0 && _vec_load_12[2] <= bound) {
                                mask = mask | 4;
                            }
                            if (_vec_load_12[3] >= 0 && _vec_load_12[3] <= bound) {
                                mask = mask | 8;
                            }
                            if (_vec_load_12[4] >= 0 && _vec_load_12[4] <= bound) {
                                mask = mask | 16;
                            }
                            if (_vec_load_12[5] >= 0 && _vec_load_12[5] <= bound) {
                                mask = mask | 32;
                            }
                            if (_vec_load_12[6] >= 0 && _vec_load_12[6] <= bound) {
                                mask = mask | 64;
                            }
                            if (_vec_load_12[7] >= 0 && _vec_load_12[7] <= bound) {
                                mask = mask | 128;
                            }
                        } else {
                            int clamped_5 = ((position_4) < (topk - 1) ? (position_4) : (topk - 1));
                            int value_4 = indices[row_base_1 + (long long)clamped_5];
                            if (value_4 >= 0 && value_4 <= bound && active_0_3 > position_4) {
                                mask = mask | 1;
                            }
                            int clamped_0 = ((position_4 + 1) < (topk - 1) ? (position_4 + 1) : (topk - 1));
                            int value_1_1 = indices[row_base_1 + (long long)clamped_0];
                            if (value_1_1 >= 0 && value_1_1 <= bound && active_0_3 > position_4 + 1) {
                                mask = mask | 2;
                            }
                            int clamped_2_1 = ((position_4 + 2) < (topk - 1) ? (position_4 + 2) : (topk - 1));
                            int value_3_1 = indices[row_base_1 + (long long)clamped_2_1];
                            if (value_3_1 >= 0 && value_3_1 <= bound && active_0_3 > position_4 + 2) {
                                mask = mask | 4;
                            }
                            int clamped_4_3 = ((position_4 + 3) < (topk - 1) ? (position_4 + 3) : (topk - 1));
                            int value_5_3 = indices[row_base_1 + (long long)clamped_4_3];
                            if (value_5_3 >= 0 && value_5_3 <= bound && active_0_3 > position_4 + 3) {
                                mask = mask | 8;
                            }
                            int clamped_6 = ((position_4 + 4) < (topk - 1) ? (position_4 + 4) : (topk - 1));
                            int value_7 = indices[row_base_1 + (long long)clamped_6];
                            if (value_7 >= 0 && value_7 <= bound && active_0_3 > position_4 + 4) {
                                mask = mask | 16;
                            }
                            int clamped_8 = ((position_4 + 5) < (topk - 1) ? (position_4 + 5) : (topk - 1));
                            int value_9 = indices[row_base_1 + (long long)clamped_8];
                            if (value_9 >= 0 && value_9 <= bound && active_0_3 > position_4 + 5) {
                                mask = mask | 32;
                            }
                            int clamped_10_1 = ((position_4 + 6) < (topk - 1) ? (position_4 + 6) : (topk - 1));
                            int value_11_1 = indices[row_base_1 + (long long)clamped_10_1];
                            if (value_11_1 >= 0 && value_11_1 <= bound && active_0_3 > position_4 + 6) {
                                mask = mask | 64;
                            }
                            int clamped_12 = ((position_4 + 7) < (topk - 1) ? (position_4 + 7) : (topk - 1));
                            int value_13 = indices[row_base_1 + (long long)clamped_12];
                            if (value_13 >= 0 && value_13 <= bound && active_0_3 > position_4 + 7) {
                                mask = mask | 128;
                            }
                        }
                        int slot_4 = cursor_3 % 3;
                        unsigned int phase_3 = cursor_3 / 3 & 1;
                        int free_stage_1 = slot_4;
                        unsigned int free_phase_1 = phase_3 ^ 1;
                        mbarrier_wait(valid_free_addr + (free_stage_1) * 8, free_phase_1);
                        validity[slot_4 * 8 + lane] = mask;
                        mbarrier_arrive(valid_ready_addr + (slot_4) * 8);
                        cursor_3 = cursor_3 + 1;
                    }
                    mbarrier_wait(clc_full_addr, outer_phase_3);
                    uint32_t _clc_valid_3 = 0;
                    uint32_t _clc_ctaid_x_3;
                    uint32_t _clc_ctaid_y_3;
                    uint32_t _clc_ctaid_z_3;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_3 == 0) {
                        break;
                    }
                    query_3 = _clc_ctaid_x_3;
                    outer_phase_3 ^= 1;
                }
            } else if (lane == 8) {
                unsigned int outer_phase_4 = 0;
                #pragma unroll 1
                for (int outer_4 = 0; outer_4 < num_queries; outer_4++) {
                    mbarrier_wait(clc_empty_addr, outer_phase_4 ^ 1);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(clc_payload_addr + 0 * 16 + 0 * 16), "r"(clc_full_addr + 0 * 8)
                        : "memory");
                    mbarrier_arrive_expect_tx(clc_full_addr, 16);
                    mbarrier_wait(clc_full_addr, outer_phase_4);
                    uint32_t _clc_valid_4 = 0;
                    uint32_t _clc_ctaid_x_4;
                    uint32_t _clc_ctaid_y_4;
                    uint32_t _clc_ctaid_z_4;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_4 == 0) {
                        break;
                    }
                    outer_phase_4 ^= 1;
                }
            } else {
                if (lane >= 16) {
                    int query_4 = blockIdx.x;
                    unsigned int outer_phase_5 = 0;
                    int lane16 = lane - 16;
                    int scan_aligned = ((idx_stride | indices_offset) & 7) == 0;
                    int out_vec = (topk & 3) == 0;
                    #pragma unroll 1
                    for (int outer_5 = 0; outer_5 < num_queries; outer_5++) {
                        if (derive_length != 0) {
                            long long row_base_2 = (long long)indices_offset + (long long)query_4 * (long long)idx_stride;
                            long long out_base = (long long)query_4 * (long long)topk;
                            int key_base = 0;
                            int bound_1 = num_kv - 1;
                            if (convert != 0) {
                                int lo = 0;
                                int hi = num_docs;
                                #pragma unroll 1
                                for (int it = 0; it < search_iters; it++) {
                                    int mid = (lo + hi) / 2;
                                    int _vec_load_13[1];
                                    {
                                        _vec_load_13[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)mid + 0);
                                    }
                                    int go_lo = (int)(mid > lo && _vec_load_13[0] <= query_4);
                                    int go_hi = (int)(mid > lo && _vec_load_13[0] > query_4);
                                    lo = ((go_lo != 0) ? mid : lo);
                                    hi = ((go_hi != 0) ? mid : hi);
                                }
                                int _vec_load_14[1];
                                {
                                    _vec_load_14[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)lo + 0);
                                }
                                int _vec_load_15[1];
                                {
                                    _vec_load_15[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)(lo + 1) + 0);
                                }
                                key_base = _vec_load_14[0];
                                int k_end = _vec_load_15[0];
                                bound_1 = k_end - key_base - 1;
                                if (causal != 0) {
                                    int _vec_load_16[1];
                                    {
                                        _vec_load_16[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)(lo + 1) + 0);
                                    }
                                    bound_1 = k_end - _vec_load_16[0] + query_4 - key_base;
                                }
                            }
                            int length = 0;
                            int n_full = topk / 128;
                            if (scan_aligned == 0) {
                                n_full = 0;
                            }
                            if (convert == 1) {
                                int base = 0;
                                int fill[4];
                                fill[0] = -1;
                                fill[1] = -1;
                                fill[2] = -1;
                                fill[3] = -1;
                                int n_batch = (n_full + 3) / 4;
                                #pragma unroll 1
                                for (int b = 0; b < n_batch; b++) {
                                    int group = ((b * 4) < (n_full - 1) ? (b * 4) : (n_full - 1));
                                    int spos = group * 128 + lane16 * 8;
                                    int _vec_load_17[8];
                                    {
                                        uint32_t _iv_5_0;
                                        uint32_t _iv_5_1;
                                        uint32_t _iv_5_2;
                                        uint32_t _iv_5_3;
                                        uint32_t _iv_5_4;
                                        uint32_t _iv_5_5;
                                        uint32_t _iv_5_6;
                                        uint32_t _iv_5_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos) + (0))) : "memory");
                                        _vec_load_17[0 + 0] = (int32_t)_iv_5_0;
                                        _vec_load_17[0 + 1] = (int32_t)_iv_5_1;
                                        _vec_load_17[0 + 2] = (int32_t)_iv_5_2;
                                        _vec_load_17[0 + 3] = (int32_t)_iv_5_3;
                                        _vec_load_17[0 + 4] = (int32_t)_iv_5_4;
                                        _vec_load_17[0 + 5] = (int32_t)_iv_5_5;
                                        _vec_load_17[0 + 6] = (int32_t)_iv_5_6;
                                        _vec_load_17[0 + 7] = (int32_t)_iv_5_7;
                                    }
                                    int group_0 = ((b * 4 + 1) < (n_full - 1) ? (b * 4 + 1) : (n_full - 1));
                                    int spos_1 = group_0 * 128 + lane16 * 8;
                                    int _vec_load_18[8];
                                    {
                                        uint32_t _iv_6_0;
                                        uint32_t _iv_6_1;
                                        uint32_t _iv_6_2;
                                        uint32_t _iv_6_3;
                                        uint32_t _iv_6_4;
                                        uint32_t _iv_6_5;
                                        uint32_t _iv_6_6;
                                        uint32_t _iv_6_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_1) + (0))) : "memory");
                                        _vec_load_18[0 + 0] = (int32_t)_iv_6_0;
                                        _vec_load_18[0 + 1] = (int32_t)_iv_6_1;
                                        _vec_load_18[0 + 2] = (int32_t)_iv_6_2;
                                        _vec_load_18[0 + 3] = (int32_t)_iv_6_3;
                                        _vec_load_18[0 + 4] = (int32_t)_iv_6_4;
                                        _vec_load_18[0 + 5] = (int32_t)_iv_6_5;
                                        _vec_load_18[0 + 6] = (int32_t)_iv_6_6;
                                        _vec_load_18[0 + 7] = (int32_t)_iv_6_7;
                                    }
                                    int group_2 = ((b * 4 + 2) < (n_full - 1) ? (b * 4 + 2) : (n_full - 1));
                                    int spos_3 = group_2 * 128 + lane16 * 8;
                                    int _vec_load_19[8];
                                    {
                                        uint32_t _iv_7_0;
                                        uint32_t _iv_7_1;
                                        uint32_t _iv_7_2;
                                        uint32_t _iv_7_3;
                                        uint32_t _iv_7_4;
                                        uint32_t _iv_7_5;
                                        uint32_t _iv_7_6;
                                        uint32_t _iv_7_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_3) + (0))) : "memory");
                                        _vec_load_19[0 + 0] = (int32_t)_iv_7_0;
                                        _vec_load_19[0 + 1] = (int32_t)_iv_7_1;
                                        _vec_load_19[0 + 2] = (int32_t)_iv_7_2;
                                        _vec_load_19[0 + 3] = (int32_t)_iv_7_3;
                                        _vec_load_19[0 + 4] = (int32_t)_iv_7_4;
                                        _vec_load_19[0 + 5] = (int32_t)_iv_7_5;
                                        _vec_load_19[0 + 6] = (int32_t)_iv_7_6;
                                        _vec_load_19[0 + 7] = (int32_t)_iv_7_7;
                                    }
                                    int group_4 = ((b * 4 + 3) < (n_full - 1) ? (b * 4 + 3) : (n_full - 1));
                                    int spos_5 = group_4 * 128 + lane16 * 8;
                                    int _vec_load_20[8];
                                    {
                                        uint32_t _iv_8_0;
                                        uint32_t _iv_8_1;
                                        uint32_t _iv_8_2;
                                        uint32_t _iv_8_3;
                                        uint32_t _iv_8_4;
                                        uint32_t _iv_8_5;
                                        uint32_t _iv_8_6;
                                        uint32_t _iv_8_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_8_0), "=r"(_iv_8_1), "=r"(_iv_8_2), "=r"(_iv_8_3), "=r"(_iv_8_4), "=r"(_iv_8_5), "=r"(_iv_8_6), "=r"(_iv_8_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_5) + (0))) : "memory");
                                        _vec_load_20[0 + 0] = (int32_t)_iv_8_0;
                                        _vec_load_20[0 + 1] = (int32_t)_iv_8_1;
                                        _vec_load_20[0 + 2] = (int32_t)_iv_8_2;
                                        _vec_load_20[0 + 3] = (int32_t)_iv_8_3;
                                        _vec_load_20[0 + 4] = (int32_t)_iv_8_4;
                                        _vec_load_20[0 + 5] = (int32_t)_iv_8_5;
                                        _vec_load_20[0 + 6] = (int32_t)_iv_8_6;
                                        _vec_load_20[0 + 7] = (int32_t)_iv_8_7;
                                    }
                                    if (n_full > b * 4) {
                                        int ok[8];
                                        int outv[8];
                                        unsigned int ballots[8];
                                        int prefix = 0;
                                        int total = 0;
                                        ok[0] = (int)(_vec_load_17[0] >= 0 && _vec_load_17[0] <= bound_1);
                                        outv[0] = -1;
                                        if (ok[0] != 0) {
                                            outv[0] = _vec_load_17[0] + key_base;
                                        }
                                        unsigned int _warp_vote_9 = __ballot_sync(4294901760u, ok[0] != 0);
                                        ballots[0] = _warp_vote_9;
                                        int _rank_in_mask_10 = (int)__popc(ballots[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_10;
                                        total = total + (int)(unsigned int)__popc(ballots[0]);
                                        ok[1] = (int)(_vec_load_17[1] >= 0 && _vec_load_17[1] <= bound_1);
                                        outv[1] = -1;
                                        if (ok[1] != 0) {
                                            outv[1] = _vec_load_17[1] + key_base;
                                        }
                                        unsigned int _warp_vote_11 = __ballot_sync(4294901760u, ok[1] != 0);
                                        ballots[1] = _warp_vote_11;
                                        int _rank_in_mask_12 = (int)__popc(ballots[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_12;
                                        total = total + (int)(unsigned int)__popc(ballots[1]);
                                        ok[2] = (int)(_vec_load_17[2] >= 0 && _vec_load_17[2] <= bound_1);
                                        outv[2] = -1;
                                        if (ok[2] != 0) {
                                            outv[2] = _vec_load_17[2] + key_base;
                                        }
                                        unsigned int _warp_vote_13 = __ballot_sync(4294901760u, ok[2] != 0);
                                        ballots[2] = _warp_vote_13;
                                        int _rank_in_mask_14 = (int)__popc(ballots[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_14;
                                        total = total + (int)(unsigned int)__popc(ballots[2]);
                                        ok[3] = (int)(_vec_load_17[3] >= 0 && _vec_load_17[3] <= bound_1);
                                        outv[3] = -1;
                                        if (ok[3] != 0) {
                                            outv[3] = _vec_load_17[3] + key_base;
                                        }
                                        unsigned int _warp_vote_15 = __ballot_sync(4294901760u, ok[3] != 0);
                                        ballots[3] = _warp_vote_15;
                                        int _rank_in_mask_16 = (int)__popc(ballots[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_16;
                                        total = total + (int)(unsigned int)__popc(ballots[3]);
                                        ok[4] = (int)(_vec_load_17[4] >= 0 && _vec_load_17[4] <= bound_1);
                                        outv[4] = -1;
                                        if (ok[4] != 0) {
                                            outv[4] = _vec_load_17[4] + key_base;
                                        }
                                        unsigned int _warp_vote_17 = __ballot_sync(4294901760u, ok[4] != 0);
                                        ballots[4] = _warp_vote_17;
                                        int _rank_in_mask_18 = (int)__popc(ballots[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_18;
                                        total = total + (int)(unsigned int)__popc(ballots[4]);
                                        ok[5] = (int)(_vec_load_17[5] >= 0 && _vec_load_17[5] <= bound_1);
                                        outv[5] = -1;
                                        if (ok[5] != 0) {
                                            outv[5] = _vec_load_17[5] + key_base;
                                        }
                                        unsigned int _warp_vote_19 = __ballot_sync(4294901760u, ok[5] != 0);
                                        ballots[5] = _warp_vote_19;
                                        int _rank_in_mask_20 = (int)__popc(ballots[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_20;
                                        total = total + (int)(unsigned int)__popc(ballots[5]);
                                        ok[6] = (int)(_vec_load_17[6] >= 0 && _vec_load_17[6] <= bound_1);
                                        outv[6] = -1;
                                        if (ok[6] != 0) {
                                            outv[6] = _vec_load_17[6] + key_base;
                                        }
                                        unsigned int _warp_vote_21 = __ballot_sync(4294901760u, ok[6] != 0);
                                        ballots[6] = _warp_vote_21;
                                        int _rank_in_mask_22 = (int)__popc(ballots[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_22;
                                        total = total + (int)(unsigned int)__popc(ballots[6]);
                                        ok[7] = (int)(_vec_load_17[7] >= 0 && _vec_load_17[7] <= bound_1);
                                        outv[7] = -1;
                                        if (ok[7] != 0) {
                                            outv[7] = _vec_load_17[7] + key_base;
                                        }
                                        unsigned int _warp_vote_23 = __ballot_sync(4294901760u, ok[7] != 0);
                                        ballots[7] = _warp_vote_23;
                                        int _rank_in_mask_24 = (int)__popc(ballots[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix = prefix + _rank_in_mask_24;
                                        total = total + (int)(unsigned int)__popc(ballots[7]);
                                        if (out_vec != 0 && total == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv[0 + 0], outv[0 + 1], outv[0 + 2], outv[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv[4 + 0], outv[4 + 1], outv[4 + 2], outv[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4) = _iv4;
                                            }
                                        } else {
                                            int rank = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[0])), "r"((unsigned int)(ok[0] != 0)) : "memory");
                                            rank = rank + ok[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[1])), "r"((unsigned int)(ok[1] != 0)) : "memory");
                                            rank = rank + ok[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[2])), "r"((unsigned int)(ok[2] != 0)) : "memory");
                                            rank = rank + ok[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[3])), "r"((unsigned int)(ok[3] != 0)) : "memory");
                                            rank = rank + ok[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[4])), "r"((unsigned int)(ok[4] != 0)) : "memory");
                                            rank = rank + ok[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[5])), "r"((unsigned int)(ok[5] != 0)) : "memory");
                                            rank = rank + ok[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[6])), "r"((unsigned int)(ok[6] != 0)) : "memory");
                                            rank = rank + ok[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix + rank)])), "r"((unsigned int)(outv[7])), "r"((unsigned int)(ok[7] != 0)) : "memory");
                                            rank = rank + ok[7];
                                        }
                                        base = base + total;
                                    }
                                    if (n_full > b * 4 + 1) {
                                        int ok_1[8];
                                        int outv_1[8];
                                        unsigned int ballots_1[8];
                                        int prefix_1 = 0;
                                        int total_1 = 0;
                                        ok_1[0] = (int)(_vec_load_18[0] >= 0 && _vec_load_18[0] <= bound_1);
                                        outv_1[0] = -1;
                                        if (ok_1[0] != 0) {
                                            outv_1[0] = _vec_load_18[0] + key_base;
                                        }
                                        unsigned int _warp_vote_25 = __ballot_sync(4294901760u, ok_1[0] != 0);
                                        ballots_1[0] = _warp_vote_25;
                                        int _rank_in_mask_26 = (int)__popc(ballots_1[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_26;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[0]);
                                        ok_1[1] = (int)(_vec_load_18[1] >= 0 && _vec_load_18[1] <= bound_1);
                                        outv_1[1] = -1;
                                        if (ok_1[1] != 0) {
                                            outv_1[1] = _vec_load_18[1] + key_base;
                                        }
                                        unsigned int _warp_vote_27 = __ballot_sync(4294901760u, ok_1[1] != 0);
                                        ballots_1[1] = _warp_vote_27;
                                        int _rank_in_mask_28 = (int)__popc(ballots_1[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_28;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[1]);
                                        ok_1[2] = (int)(_vec_load_18[2] >= 0 && _vec_load_18[2] <= bound_1);
                                        outv_1[2] = -1;
                                        if (ok_1[2] != 0) {
                                            outv_1[2] = _vec_load_18[2] + key_base;
                                        }
                                        unsigned int _warp_vote_29 = __ballot_sync(4294901760u, ok_1[2] != 0);
                                        ballots_1[2] = _warp_vote_29;
                                        int _rank_in_mask_30 = (int)__popc(ballots_1[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_30;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[2]);
                                        ok_1[3] = (int)(_vec_load_18[3] >= 0 && _vec_load_18[3] <= bound_1);
                                        outv_1[3] = -1;
                                        if (ok_1[3] != 0) {
                                            outv_1[3] = _vec_load_18[3] + key_base;
                                        }
                                        unsigned int _warp_vote_31 = __ballot_sync(4294901760u, ok_1[3] != 0);
                                        ballots_1[3] = _warp_vote_31;
                                        int _rank_in_mask_32 = (int)__popc(ballots_1[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_32;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[3]);
                                        ok_1[4] = (int)(_vec_load_18[4] >= 0 && _vec_load_18[4] <= bound_1);
                                        outv_1[4] = -1;
                                        if (ok_1[4] != 0) {
                                            outv_1[4] = _vec_load_18[4] + key_base;
                                        }
                                        unsigned int _warp_vote_33 = __ballot_sync(4294901760u, ok_1[4] != 0);
                                        ballots_1[4] = _warp_vote_33;
                                        int _rank_in_mask_34 = (int)__popc(ballots_1[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_34;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[4]);
                                        ok_1[5] = (int)(_vec_load_18[5] >= 0 && _vec_load_18[5] <= bound_1);
                                        outv_1[5] = -1;
                                        if (ok_1[5] != 0) {
                                            outv_1[5] = _vec_load_18[5] + key_base;
                                        }
                                        unsigned int _warp_vote_35 = __ballot_sync(4294901760u, ok_1[5] != 0);
                                        ballots_1[5] = _warp_vote_35;
                                        int _rank_in_mask_36 = (int)__popc(ballots_1[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_36;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[5]);
                                        ok_1[6] = (int)(_vec_load_18[6] >= 0 && _vec_load_18[6] <= bound_1);
                                        outv_1[6] = -1;
                                        if (ok_1[6] != 0) {
                                            outv_1[6] = _vec_load_18[6] + key_base;
                                        }
                                        unsigned int _warp_vote_37 = __ballot_sync(4294901760u, ok_1[6] != 0);
                                        ballots_1[6] = _warp_vote_37;
                                        int _rank_in_mask_38 = (int)__popc(ballots_1[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_38;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[6]);
                                        ok_1[7] = (int)(_vec_load_18[7] >= 0 && _vec_load_18[7] <= bound_1);
                                        outv_1[7] = -1;
                                        if (ok_1[7] != 0) {
                                            outv_1[7] = _vec_load_18[7] + key_base;
                                        }
                                        unsigned int _warp_vote_39 = __ballot_sync(4294901760u, ok_1[7] != 0);
                                        ballots_1[7] = _warp_vote_39;
                                        int _rank_in_mask_40 = (int)__popc(ballots_1[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_1 = prefix_1 + _rank_in_mask_40;
                                        total_1 = total_1 + (int)(unsigned int)__popc(ballots_1[7]);
                                        if (out_vec != 0 && total_1 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_1[0 + 0], outv_1[0 + 1], outv_1[0 + 2], outv_1[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_1[4 + 0], outv_1[4 + 1], outv_1[4 + 2], outv_1[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4) = _iv4;
                                            }
                                        } else {
                                            int rank_1 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[0])), "r"((unsigned int)(ok_1[0] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[1])), "r"((unsigned int)(ok_1[1] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[2])), "r"((unsigned int)(ok_1[2] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[3])), "r"((unsigned int)(ok_1[3] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[4])), "r"((unsigned int)(ok_1[4] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[5])), "r"((unsigned int)(ok_1[5] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[6])), "r"((unsigned int)(ok_1[6] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_1 + rank_1)])), "r"((unsigned int)(outv_1[7])), "r"((unsigned int)(ok_1[7] != 0)) : "memory");
                                            rank_1 = rank_1 + ok_1[7];
                                        }
                                        base = base + total_1;
                                    }
                                    if (n_full > b * 4 + 2) {
                                        int ok_2[8];
                                        int outv_2[8];
                                        unsigned int ballots_2[8];
                                        int prefix_2 = 0;
                                        int total_2 = 0;
                                        ok_2[0] = (int)(_vec_load_19[0] >= 0 && _vec_load_19[0] <= bound_1);
                                        outv_2[0] = -1;
                                        if (ok_2[0] != 0) {
                                            outv_2[0] = _vec_load_19[0] + key_base;
                                        }
                                        unsigned int _warp_vote_41 = __ballot_sync(4294901760u, ok_2[0] != 0);
                                        ballots_2[0] = _warp_vote_41;
                                        int _rank_in_mask_42 = (int)__popc(ballots_2[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_42;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[0]);
                                        ok_2[1] = (int)(_vec_load_19[1] >= 0 && _vec_load_19[1] <= bound_1);
                                        outv_2[1] = -1;
                                        if (ok_2[1] != 0) {
                                            outv_2[1] = _vec_load_19[1] + key_base;
                                        }
                                        unsigned int _warp_vote_43 = __ballot_sync(4294901760u, ok_2[1] != 0);
                                        ballots_2[1] = _warp_vote_43;
                                        int _rank_in_mask_44 = (int)__popc(ballots_2[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_44;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[1]);
                                        ok_2[2] = (int)(_vec_load_19[2] >= 0 && _vec_load_19[2] <= bound_1);
                                        outv_2[2] = -1;
                                        if (ok_2[2] != 0) {
                                            outv_2[2] = _vec_load_19[2] + key_base;
                                        }
                                        unsigned int _warp_vote_45 = __ballot_sync(4294901760u, ok_2[2] != 0);
                                        ballots_2[2] = _warp_vote_45;
                                        int _rank_in_mask_46 = (int)__popc(ballots_2[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_46;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[2]);
                                        ok_2[3] = (int)(_vec_load_19[3] >= 0 && _vec_load_19[3] <= bound_1);
                                        outv_2[3] = -1;
                                        if (ok_2[3] != 0) {
                                            outv_2[3] = _vec_load_19[3] + key_base;
                                        }
                                        unsigned int _warp_vote_47 = __ballot_sync(4294901760u, ok_2[3] != 0);
                                        ballots_2[3] = _warp_vote_47;
                                        int _rank_in_mask_48 = (int)__popc(ballots_2[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_48;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[3]);
                                        ok_2[4] = (int)(_vec_load_19[4] >= 0 && _vec_load_19[4] <= bound_1);
                                        outv_2[4] = -1;
                                        if (ok_2[4] != 0) {
                                            outv_2[4] = _vec_load_19[4] + key_base;
                                        }
                                        unsigned int _warp_vote_49 = __ballot_sync(4294901760u, ok_2[4] != 0);
                                        ballots_2[4] = _warp_vote_49;
                                        int _rank_in_mask_50 = (int)__popc(ballots_2[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_50;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[4]);
                                        ok_2[5] = (int)(_vec_load_19[5] >= 0 && _vec_load_19[5] <= bound_1);
                                        outv_2[5] = -1;
                                        if (ok_2[5] != 0) {
                                            outv_2[5] = _vec_load_19[5] + key_base;
                                        }
                                        unsigned int _warp_vote_51 = __ballot_sync(4294901760u, ok_2[5] != 0);
                                        ballots_2[5] = _warp_vote_51;
                                        int _rank_in_mask_52 = (int)__popc(ballots_2[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_52;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[5]);
                                        ok_2[6] = (int)(_vec_load_19[6] >= 0 && _vec_load_19[6] <= bound_1);
                                        outv_2[6] = -1;
                                        if (ok_2[6] != 0) {
                                            outv_2[6] = _vec_load_19[6] + key_base;
                                        }
                                        unsigned int _warp_vote_53 = __ballot_sync(4294901760u, ok_2[6] != 0);
                                        ballots_2[6] = _warp_vote_53;
                                        int _rank_in_mask_54 = (int)__popc(ballots_2[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_54;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[6]);
                                        ok_2[7] = (int)(_vec_load_19[7] >= 0 && _vec_load_19[7] <= bound_1);
                                        outv_2[7] = -1;
                                        if (ok_2[7] != 0) {
                                            outv_2[7] = _vec_load_19[7] + key_base;
                                        }
                                        unsigned int _warp_vote_55 = __ballot_sync(4294901760u, ok_2[7] != 0);
                                        ballots_2[7] = _warp_vote_55;
                                        int _rank_in_mask_56 = (int)__popc(ballots_2[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_2 = prefix_2 + _rank_in_mask_56;
                                        total_2 = total_2 + (int)(unsigned int)__popc(ballots_2[7]);
                                        if (out_vec != 0 && total_2 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_2[0 + 0], outv_2[0 + 1], outv_2[0 + 2], outv_2[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_2[4 + 0], outv_2[4 + 1], outv_2[4 + 2], outv_2[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4) = _iv4;
                                            }
                                        } else {
                                            int rank_2 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[0])), "r"((unsigned int)(ok_2[0] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[1])), "r"((unsigned int)(ok_2[1] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[2])), "r"((unsigned int)(ok_2[2] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[3])), "r"((unsigned int)(ok_2[3] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[4])), "r"((unsigned int)(ok_2[4] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[5])), "r"((unsigned int)(ok_2[5] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[6])), "r"((unsigned int)(ok_2[6] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_2 + rank_2)])), "r"((unsigned int)(outv_2[7])), "r"((unsigned int)(ok_2[7] != 0)) : "memory");
                                            rank_2 = rank_2 + ok_2[7];
                                        }
                                        base = base + total_2;
                                    }
                                    if (n_full > b * 4 + 3) {
                                        int ok_3[8];
                                        int outv_3[8];
                                        unsigned int ballots_3[8];
                                        int prefix_3 = 0;
                                        int total_3 = 0;
                                        ok_3[0] = (int)(_vec_load_20[0] >= 0 && _vec_load_20[0] <= bound_1);
                                        outv_3[0] = -1;
                                        if (ok_3[0] != 0) {
                                            outv_3[0] = _vec_load_20[0] + key_base;
                                        }
                                        unsigned int _warp_vote_57 = __ballot_sync(4294901760u, ok_3[0] != 0);
                                        ballots_3[0] = _warp_vote_57;
                                        int _rank_in_mask_58 = (int)__popc(ballots_3[0] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_58;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[0]);
                                        ok_3[1] = (int)(_vec_load_20[1] >= 0 && _vec_load_20[1] <= bound_1);
                                        outv_3[1] = -1;
                                        if (ok_3[1] != 0) {
                                            outv_3[1] = _vec_load_20[1] + key_base;
                                        }
                                        unsigned int _warp_vote_59 = __ballot_sync(4294901760u, ok_3[1] != 0);
                                        ballots_3[1] = _warp_vote_59;
                                        int _rank_in_mask_60 = (int)__popc(ballots_3[1] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_60;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[1]);
                                        ok_3[2] = (int)(_vec_load_20[2] >= 0 && _vec_load_20[2] <= bound_1);
                                        outv_3[2] = -1;
                                        if (ok_3[2] != 0) {
                                            outv_3[2] = _vec_load_20[2] + key_base;
                                        }
                                        unsigned int _warp_vote_61 = __ballot_sync(4294901760u, ok_3[2] != 0);
                                        ballots_3[2] = _warp_vote_61;
                                        int _rank_in_mask_62 = (int)__popc(ballots_3[2] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_62;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[2]);
                                        ok_3[3] = (int)(_vec_load_20[3] >= 0 && _vec_load_20[3] <= bound_1);
                                        outv_3[3] = -1;
                                        if (ok_3[3] != 0) {
                                            outv_3[3] = _vec_load_20[3] + key_base;
                                        }
                                        unsigned int _warp_vote_63 = __ballot_sync(4294901760u, ok_3[3] != 0);
                                        ballots_3[3] = _warp_vote_63;
                                        int _rank_in_mask_64 = (int)__popc(ballots_3[3] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_64;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[3]);
                                        ok_3[4] = (int)(_vec_load_20[4] >= 0 && _vec_load_20[4] <= bound_1);
                                        outv_3[4] = -1;
                                        if (ok_3[4] != 0) {
                                            outv_3[4] = _vec_load_20[4] + key_base;
                                        }
                                        unsigned int _warp_vote_65 = __ballot_sync(4294901760u, ok_3[4] != 0);
                                        ballots_3[4] = _warp_vote_65;
                                        int _rank_in_mask_66 = (int)__popc(ballots_3[4] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_66;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[4]);
                                        ok_3[5] = (int)(_vec_load_20[5] >= 0 && _vec_load_20[5] <= bound_1);
                                        outv_3[5] = -1;
                                        if (ok_3[5] != 0) {
                                            outv_3[5] = _vec_load_20[5] + key_base;
                                        }
                                        unsigned int _warp_vote_67 = __ballot_sync(4294901760u, ok_3[5] != 0);
                                        ballots_3[5] = _warp_vote_67;
                                        int _rank_in_mask_68 = (int)__popc(ballots_3[5] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_68;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[5]);
                                        ok_3[6] = (int)(_vec_load_20[6] >= 0 && _vec_load_20[6] <= bound_1);
                                        outv_3[6] = -1;
                                        if (ok_3[6] != 0) {
                                            outv_3[6] = _vec_load_20[6] + key_base;
                                        }
                                        unsigned int _warp_vote_69 = __ballot_sync(4294901760u, ok_3[6] != 0);
                                        ballots_3[6] = _warp_vote_69;
                                        int _rank_in_mask_70 = (int)__popc(ballots_3[6] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_70;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[6]);
                                        ok_3[7] = (int)(_vec_load_20[7] >= 0 && _vec_load_20[7] <= bound_1);
                                        outv_3[7] = -1;
                                        if (ok_3[7] != 0) {
                                            outv_3[7] = _vec_load_20[7] + key_base;
                                        }
                                        unsigned int _warp_vote_71 = __ballot_sync(4294901760u, ok_3[7] != 0);
                                        ballots_3[7] = _warp_vote_71;
                                        int _rank_in_mask_72 = (int)__popc(ballots_3[7] & ((1u << (unsigned int)lane) - 1u));
                                        prefix_3 = prefix_3 + _rank_in_mask_72;
                                        total_3 = total_3 + (int)(unsigned int)__popc(ballots_3[7]);
                                        if (out_vec != 0 && total_3 == 128 && (base & 3) == 0) {
                                            {
                                                int4 _iv4 = make_int4(outv_3[0 + 0], outv_3[0 + 1], outv_3[0 + 2], outv_3[0 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8)) = _iv4;
                                            }
                                            {
                                                int4 _iv4 = make_int4(outv_3[4 + 0], outv_3[4 + 1], outv_3[4 + 2], outv_3[4 + 3]);
                                                *reinterpret_cast<int4*>(indices_out + out_base + (long long)(base + lane16 * 8) + 4) = _iv4;
                                            }
                                        } else {
                                            int rank_3 = 0;
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[0])), "r"((unsigned int)(ok_3[0] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[0];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[1])), "r"((unsigned int)(ok_3[1] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[1];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[2])), "r"((unsigned int)(ok_3[2] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[2];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[3])), "r"((unsigned int)(ok_3[3] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[3];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[4])), "r"((unsigned int)(ok_3[4] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[4];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[5])), "r"((unsigned int)(ok_3[5] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[5];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[6])), "r"((unsigned int)(ok_3[6] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[6];
                                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + prefix_3 + rank_3)])), "r"((unsigned int)(outv_3[7])), "r"((unsigned int)(ok_3[7] != 0)) : "memory");
                                            rank_3 = rank_3 + ok_3[7];
                                        }
                                        base = base + total_3;
                                    }
                                }
                                int tail0 = n_full * 128 + lane16;
                                int n_tail = (topk - n_full * 128 + 15) / 16;
                                #pragma unroll 1
                                for (int t = 0; t < n_tail; t++) {
                                    int pos = tail0 + t * 16;
                                    int value_6 = -1;
                                    if (pos < topk) {
                                        value_6 = indices[row_base_2 + (long long)pos];
                                    }
                                    int valid = (int)(value_6 >= 0 && value_6 <= bound_1);
                                    int converted = value_6 + key_base;
                                    unsigned int ballot = __ballot_sync(4294901760u, valid != 0);
                                    int _rank_in_mask_73 = (int)__popc(ballot & ((1u << (unsigned int)lane) - 1u));
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)(base + _rank_in_mask_73)])), "r"((unsigned int)(converted)), "r"((unsigned int)(valid != 0)) : "memory");
                                    base = base + (int)(unsigned int)__popc(ballot);
                                }
                                if (out_vec != 0) {
                                    int head_1 = 4 - (base & 3) & 3;
                                    int hpos = base + lane16;
                                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)hpos])), "r"((unsigned int)(-1)), "r"((unsigned int)(lane16 < head_1 && hpos < topk)) : "memory");
                                    int base4 = base + head_1;
                                    int n_vec = (topk - base4) / 4;
                                    #pragma unroll 1
                                    for (int t_1 = 0; t_1 < (n_vec + 15) / 16; t_1++) {
                                        int chunk = t_1 * 16 + lane16;
                                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(indices_out + (out_base + (long long)(base4 + chunk * 4))), "r"(fill[(0) + 0]), "r"(fill[(0) + 1]), "r"(fill[(0) + 2]), "r"(fill[(0) + 3]), "r"((unsigned int)(chunk < n_vec)) : "memory");
                                    }
                                } else {
                                    #pragma unroll 1
                                    for (int t_2 = 0; t_2 < (topk - base + 15) / 16; t_2++) {
                                        int fpos = base + t_2 * 16 + lane16;
                                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&indices_out[out_base + (long long)fpos])), "r"((unsigned int)(-1)), "r"((unsigned int)(fpos < topk)) : "memory");
                                    }
                                }
                                length = base;
                            } else if (convert == 2) {
                                int n_batch_1 = n_full / 4;
                                #pragma unroll 1
                                for (int b_1 = 0; b_1 < n_batch_1; b_1++) {
                                    int spos_2 = b_1 * 4 * 128 + lane16 * 8;
                                    int _vec_load_21[8];
                                    {
                                        uint32_t _iv_74_0;
                                        uint32_t _iv_74_1;
                                        uint32_t _iv_74_2;
                                        uint32_t _iv_74_3;
                                        uint32_t _iv_74_4;
                                        uint32_t _iv_74_5;
                                        uint32_t _iv_74_6;
                                        uint32_t _iv_74_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_74_0), "=r"(_iv_74_1), "=r"(_iv_74_2), "=r"(_iv_74_3), "=r"(_iv_74_4), "=r"(_iv_74_5), "=r"(_iv_74_6), "=r"(_iv_74_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_2) + (0))) : "memory");
                                        _vec_load_21[0 + 0] = (int32_t)_iv_74_0;
                                        _vec_load_21[0 + 1] = (int32_t)_iv_74_1;
                                        _vec_load_21[0 + 2] = (int32_t)_iv_74_2;
                                        _vec_load_21[0 + 3] = (int32_t)_iv_74_3;
                                        _vec_load_21[0 + 4] = (int32_t)_iv_74_4;
                                        _vec_load_21[0 + 5] = (int32_t)_iv_74_5;
                                        _vec_load_21[0 + 6] = (int32_t)_iv_74_6;
                                        _vec_load_21[0 + 7] = (int32_t)_iv_74_7;
                                    }
                                    int spos_0 = (b_1 * 4 + 1) * 128 + lane16 * 8;
                                    int _vec_load_22[8];
                                    {
                                        uint32_t _iv_75_0;
                                        uint32_t _iv_75_1;
                                        uint32_t _iv_75_2;
                                        uint32_t _iv_75_3;
                                        uint32_t _iv_75_4;
                                        uint32_t _iv_75_5;
                                        uint32_t _iv_75_6;
                                        uint32_t _iv_75_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_75_0), "=r"(_iv_75_1), "=r"(_iv_75_2), "=r"(_iv_75_3), "=r"(_iv_75_4), "=r"(_iv_75_5), "=r"(_iv_75_6), "=r"(_iv_75_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_0) + (0))) : "memory");
                                        _vec_load_22[0 + 0] = (int32_t)_iv_75_0;
                                        _vec_load_22[0 + 1] = (int32_t)_iv_75_1;
                                        _vec_load_22[0 + 2] = (int32_t)_iv_75_2;
                                        _vec_load_22[0 + 3] = (int32_t)_iv_75_3;
                                        _vec_load_22[0 + 4] = (int32_t)_iv_75_4;
                                        _vec_load_22[0 + 5] = (int32_t)_iv_75_5;
                                        _vec_load_22[0 + 6] = (int32_t)_iv_75_6;
                                        _vec_load_22[0 + 7] = (int32_t)_iv_75_7;
                                    }
                                    int spos_1_1 = (b_1 * 4 + 2) * 128 + lane16 * 8;
                                    int _vec_load_23[8];
                                    {
                                        uint32_t _iv_76_0;
                                        uint32_t _iv_76_1;
                                        uint32_t _iv_76_2;
                                        uint32_t _iv_76_3;
                                        uint32_t _iv_76_4;
                                        uint32_t _iv_76_5;
                                        uint32_t _iv_76_6;
                                        uint32_t _iv_76_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_76_0), "=r"(_iv_76_1), "=r"(_iv_76_2), "=r"(_iv_76_3), "=r"(_iv_76_4), "=r"(_iv_76_5), "=r"(_iv_76_6), "=r"(_iv_76_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_1_1) + (0))) : "memory");
                                        _vec_load_23[0 + 0] = (int32_t)_iv_76_0;
                                        _vec_load_23[0 + 1] = (int32_t)_iv_76_1;
                                        _vec_load_23[0 + 2] = (int32_t)_iv_76_2;
                                        _vec_load_23[0 + 3] = (int32_t)_iv_76_3;
                                        _vec_load_23[0 + 4] = (int32_t)_iv_76_4;
                                        _vec_load_23[0 + 5] = (int32_t)_iv_76_5;
                                        _vec_load_23[0 + 6] = (int32_t)_iv_76_6;
                                        _vec_load_23[0 + 7] = (int32_t)_iv_76_7;
                                    }
                                    int spos_2_1 = (b_1 * 4 + 3) * 128 + lane16 * 8;
                                    int _vec_load_24[8];
                                    {
                                        uint32_t _iv_77_0;
                                        uint32_t _iv_77_1;
                                        uint32_t _iv_77_2;
                                        uint32_t _iv_77_3;
                                        uint32_t _iv_77_4;
                                        uint32_t _iv_77_5;
                                        uint32_t _iv_77_6;
                                        uint32_t _iv_77_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_77_0), "=r"(_iv_77_1), "=r"(_iv_77_2), "=r"(_iv_77_3), "=r"(_iv_77_4), "=r"(_iv_77_5), "=r"(_iv_77_6), "=r"(_iv_77_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_2_1) + (0))) : "memory");
                                        _vec_load_24[0 + 0] = (int32_t)_iv_77_0;
                                        _vec_load_24[0 + 1] = (int32_t)_iv_77_1;
                                        _vec_load_24[0 + 2] = (int32_t)_iv_77_2;
                                        _vec_load_24[0 + 3] = (int32_t)_iv_77_3;
                                        _vec_load_24[0 + 4] = (int32_t)_iv_77_4;
                                        _vec_load_24[0 + 5] = (int32_t)_iv_77_5;
                                        _vec_load_24[0 + 6] = (int32_t)_iv_77_6;
                                        _vec_load_24[0 + 7] = (int32_t)_iv_77_7;
                                    }
                                    int outv_4[8];
                                    outv_4[0] = -1;
                                    if (_vec_load_21[0] >= 0 && _vec_load_21[0] <= bound_1) {
                                        outv_4[0] = _vec_load_21[0] + key_base;
                                        length = spos_2 + 1;
                                    }
                                    outv_4[1] = -1;
                                    if (_vec_load_21[1] >= 0 && _vec_load_21[1] <= bound_1) {
                                        outv_4[1] = _vec_load_21[1] + key_base;
                                        length = spos_2 + 1 + 1;
                                    }
                                    outv_4[2] = -1;
                                    if (_vec_load_21[2] >= 0 && _vec_load_21[2] <= bound_1) {
                                        outv_4[2] = _vec_load_21[2] + key_base;
                                        length = spos_2 + 2 + 1;
                                    }
                                    outv_4[3] = -1;
                                    if (_vec_load_21[3] >= 0 && _vec_load_21[3] <= bound_1) {
                                        outv_4[3] = _vec_load_21[3] + key_base;
                                        length = spos_2 + 3 + 1;
                                    }
                                    outv_4[4] = -1;
                                    if (_vec_load_21[4] >= 0 && _vec_load_21[4] <= bound_1) {
                                        outv_4[4] = _vec_load_21[4] + key_base;
                                        length = spos_2 + 4 + 1;
                                    }
                                    outv_4[5] = -1;
                                    if (_vec_load_21[5] >= 0 && _vec_load_21[5] <= bound_1) {
                                        outv_4[5] = _vec_load_21[5] + key_base;
                                        length = spos_2 + 5 + 1;
                                    }
                                    outv_4[6] = -1;
                                    if (_vec_load_21[6] >= 0 && _vec_load_21[6] <= bound_1) {
                                        outv_4[6] = _vec_load_21[6] + key_base;
                                        length = spos_2 + 6 + 1;
                                    }
                                    outv_4[7] = -1;
                                    if (_vec_load_21[7] >= 0 && _vec_load_21[7] <= bound_1) {
                                        outv_4[7] = _vec_load_21[7] + key_base;
                                        length = spos_2 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_4[0 + 0], outv_4[0 + 1], outv_4[0 + 2], outv_4[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_4[4 + 0], outv_4[4 + 1], outv_4[4 + 2], outv_4[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2 + 4) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_2] = outv_4[0];
                                        indices_out[out_base + (long long)(spos_2 + 1)] = outv_4[1];
                                        indices_out[out_base + (long long)(spos_2 + 2)] = outv_4[2];
                                        indices_out[out_base + (long long)(spos_2 + 3)] = outv_4[3];
                                        indices_out[out_base + (long long)(spos_2 + 4)] = outv_4[4];
                                        indices_out[out_base + (long long)(spos_2 + 5)] = outv_4[5];
                                        indices_out[out_base + (long long)(spos_2 + 6)] = outv_4[6];
                                        indices_out[out_base + (long long)(spos_2 + 7)] = outv_4[7];
                                    }
                                    int outv_3_1[8];
                                    outv_3_1[0] = -1;
                                    if (_vec_load_22[0] >= 0 && _vec_load_22[0] <= bound_1) {
                                        outv_3_1[0] = _vec_load_22[0] + key_base;
                                        length = spos_0 + 1;
                                    }
                                    outv_3_1[1] = -1;
                                    if (_vec_load_22[1] >= 0 && _vec_load_22[1] <= bound_1) {
                                        outv_3_1[1] = _vec_load_22[1] + key_base;
                                        length = spos_0 + 1 + 1;
                                    }
                                    outv_3_1[2] = -1;
                                    if (_vec_load_22[2] >= 0 && _vec_load_22[2] <= bound_1) {
                                        outv_3_1[2] = _vec_load_22[2] + key_base;
                                        length = spos_0 + 2 + 1;
                                    }
                                    outv_3_1[3] = -1;
                                    if (_vec_load_22[3] >= 0 && _vec_load_22[3] <= bound_1) {
                                        outv_3_1[3] = _vec_load_22[3] + key_base;
                                        length = spos_0 + 3 + 1;
                                    }
                                    outv_3_1[4] = -1;
                                    if (_vec_load_22[4] >= 0 && _vec_load_22[4] <= bound_1) {
                                        outv_3_1[4] = _vec_load_22[4] + key_base;
                                        length = spos_0 + 4 + 1;
                                    }
                                    outv_3_1[5] = -1;
                                    if (_vec_load_22[5] >= 0 && _vec_load_22[5] <= bound_1) {
                                        outv_3_1[5] = _vec_load_22[5] + key_base;
                                        length = spos_0 + 5 + 1;
                                    }
                                    outv_3_1[6] = -1;
                                    if (_vec_load_22[6] >= 0 && _vec_load_22[6] <= bound_1) {
                                        outv_3_1[6] = _vec_load_22[6] + key_base;
                                        length = spos_0 + 6 + 1;
                                    }
                                    outv_3_1[7] = -1;
                                    if (_vec_load_22[7] >= 0 && _vec_load_22[7] <= bound_1) {
                                        outv_3_1[7] = _vec_load_22[7] + key_base;
                                        length = spos_0 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_3_1[0 + 0], outv_3_1[0 + 1], outv_3_1[0 + 2], outv_3_1[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_0) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_3_1[4 + 0], outv_3_1[4 + 1], outv_3_1[4 + 2], outv_3_1[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_0 + 4) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_0] = outv_3_1[0];
                                        indices_out[out_base + (long long)(spos_0 + 1)] = outv_3_1[1];
                                        indices_out[out_base + (long long)(spos_0 + 2)] = outv_3_1[2];
                                        indices_out[out_base + (long long)(spos_0 + 3)] = outv_3_1[3];
                                        indices_out[out_base + (long long)(spos_0 + 4)] = outv_3_1[4];
                                        indices_out[out_base + (long long)(spos_0 + 5)] = outv_3_1[5];
                                        indices_out[out_base + (long long)(spos_0 + 6)] = outv_3_1[6];
                                        indices_out[out_base + (long long)(spos_0 + 7)] = outv_3_1[7];
                                    }
                                    int outv_4_1[8];
                                    outv_4_1[0] = -1;
                                    if (_vec_load_23[0] >= 0 && _vec_load_23[0] <= bound_1) {
                                        outv_4_1[0] = _vec_load_23[0] + key_base;
                                        length = spos_1_1 + 1;
                                    }
                                    outv_4_1[1] = -1;
                                    if (_vec_load_23[1] >= 0 && _vec_load_23[1] <= bound_1) {
                                        outv_4_1[1] = _vec_load_23[1] + key_base;
                                        length = spos_1_1 + 1 + 1;
                                    }
                                    outv_4_1[2] = -1;
                                    if (_vec_load_23[2] >= 0 && _vec_load_23[2] <= bound_1) {
                                        outv_4_1[2] = _vec_load_23[2] + key_base;
                                        length = spos_1_1 + 2 + 1;
                                    }
                                    outv_4_1[3] = -1;
                                    if (_vec_load_23[3] >= 0 && _vec_load_23[3] <= bound_1) {
                                        outv_4_1[3] = _vec_load_23[3] + key_base;
                                        length = spos_1_1 + 3 + 1;
                                    }
                                    outv_4_1[4] = -1;
                                    if (_vec_load_23[4] >= 0 && _vec_load_23[4] <= bound_1) {
                                        outv_4_1[4] = _vec_load_23[4] + key_base;
                                        length = spos_1_1 + 4 + 1;
                                    }
                                    outv_4_1[5] = -1;
                                    if (_vec_load_23[5] >= 0 && _vec_load_23[5] <= bound_1) {
                                        outv_4_1[5] = _vec_load_23[5] + key_base;
                                        length = spos_1_1 + 5 + 1;
                                    }
                                    outv_4_1[6] = -1;
                                    if (_vec_load_23[6] >= 0 && _vec_load_23[6] <= bound_1) {
                                        outv_4_1[6] = _vec_load_23[6] + key_base;
                                        length = spos_1_1 + 6 + 1;
                                    }
                                    outv_4_1[7] = -1;
                                    if (_vec_load_23[7] >= 0 && _vec_load_23[7] <= bound_1) {
                                        outv_4_1[7] = _vec_load_23[7] + key_base;
                                        length = spos_1_1 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_4_1[0 + 0], outv_4_1[0 + 1], outv_4_1[0 + 2], outv_4_1[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_1_1) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_4_1[4 + 0], outv_4_1[4 + 1], outv_4_1[4 + 2], outv_4_1[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_1_1 + 4) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_1_1] = outv_4_1[0];
                                        indices_out[out_base + (long long)(spos_1_1 + 1)] = outv_4_1[1];
                                        indices_out[out_base + (long long)(spos_1_1 + 2)] = outv_4_1[2];
                                        indices_out[out_base + (long long)(spos_1_1 + 3)] = outv_4_1[3];
                                        indices_out[out_base + (long long)(spos_1_1 + 4)] = outv_4_1[4];
                                        indices_out[out_base + (long long)(spos_1_1 + 5)] = outv_4_1[5];
                                        indices_out[out_base + (long long)(spos_1_1 + 6)] = outv_4_1[6];
                                        indices_out[out_base + (long long)(spos_1_1 + 7)] = outv_4_1[7];
                                    }
                                    int outv_5[8];
                                    outv_5[0] = -1;
                                    if (_vec_load_24[0] >= 0 && _vec_load_24[0] <= bound_1) {
                                        outv_5[0] = _vec_load_24[0] + key_base;
                                        length = spos_2_1 + 1;
                                    }
                                    outv_5[1] = -1;
                                    if (_vec_load_24[1] >= 0 && _vec_load_24[1] <= bound_1) {
                                        outv_5[1] = _vec_load_24[1] + key_base;
                                        length = spos_2_1 + 1 + 1;
                                    }
                                    outv_5[2] = -1;
                                    if (_vec_load_24[2] >= 0 && _vec_load_24[2] <= bound_1) {
                                        outv_5[2] = _vec_load_24[2] + key_base;
                                        length = spos_2_1 + 2 + 1;
                                    }
                                    outv_5[3] = -1;
                                    if (_vec_load_24[3] >= 0 && _vec_load_24[3] <= bound_1) {
                                        outv_5[3] = _vec_load_24[3] + key_base;
                                        length = spos_2_1 + 3 + 1;
                                    }
                                    outv_5[4] = -1;
                                    if (_vec_load_24[4] >= 0 && _vec_load_24[4] <= bound_1) {
                                        outv_5[4] = _vec_load_24[4] + key_base;
                                        length = spos_2_1 + 4 + 1;
                                    }
                                    outv_5[5] = -1;
                                    if (_vec_load_24[5] >= 0 && _vec_load_24[5] <= bound_1) {
                                        outv_5[5] = _vec_load_24[5] + key_base;
                                        length = spos_2_1 + 5 + 1;
                                    }
                                    outv_5[6] = -1;
                                    if (_vec_load_24[6] >= 0 && _vec_load_24[6] <= bound_1) {
                                        outv_5[6] = _vec_load_24[6] + key_base;
                                        length = spos_2_1 + 6 + 1;
                                    }
                                    outv_5[7] = -1;
                                    if (_vec_load_24[7] >= 0 && _vec_load_24[7] <= bound_1) {
                                        outv_5[7] = _vec_load_24[7] + key_base;
                                        length = spos_2_1 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_5[0 + 0], outv_5[0 + 1], outv_5[0 + 2], outv_5[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2_1) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_5[4 + 0], outv_5[4 + 1], outv_5[4 + 2], outv_5[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_2_1 + 4) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_2_1] = outv_5[0];
                                        indices_out[out_base + (long long)(spos_2_1 + 1)] = outv_5[1];
                                        indices_out[out_base + (long long)(spos_2_1 + 2)] = outv_5[2];
                                        indices_out[out_base + (long long)(spos_2_1 + 3)] = outv_5[3];
                                        indices_out[out_base + (long long)(spos_2_1 + 4)] = outv_5[4];
                                        indices_out[out_base + (long long)(spos_2_1 + 5)] = outv_5[5];
                                        indices_out[out_base + (long long)(spos_2_1 + 6)] = outv_5[6];
                                        indices_out[out_base + (long long)(spos_2_1 + 7)] = outv_5[7];
                                    }
                                }
                                #pragma unroll 1
                                for (int r = 0; r < n_full - n_batch_1 * 4; r++) {
                                    int spos_4 = (n_batch_1 * 4 + r) * 128 + lane16 * 8;
                                    int _vec_load_25[8];
                                    {
                                        uint32_t _iv_78_0;
                                        uint32_t _iv_78_1;
                                        uint32_t _iv_78_2;
                                        uint32_t _iv_78_3;
                                        uint32_t _iv_78_4;
                                        uint32_t _iv_78_5;
                                        uint32_t _iv_78_6;
                                        uint32_t _iv_78_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_78_0), "=r"(_iv_78_1), "=r"(_iv_78_2), "=r"(_iv_78_3), "=r"(_iv_78_4), "=r"(_iv_78_5), "=r"(_iv_78_6), "=r"(_iv_78_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_4) + (0))) : "memory");
                                        _vec_load_25[0 + 0] = (int32_t)_iv_78_0;
                                        _vec_load_25[0 + 1] = (int32_t)_iv_78_1;
                                        _vec_load_25[0 + 2] = (int32_t)_iv_78_2;
                                        _vec_load_25[0 + 3] = (int32_t)_iv_78_3;
                                        _vec_load_25[0 + 4] = (int32_t)_iv_78_4;
                                        _vec_load_25[0 + 5] = (int32_t)_iv_78_5;
                                        _vec_load_25[0 + 6] = (int32_t)_iv_78_6;
                                        _vec_load_25[0 + 7] = (int32_t)_iv_78_7;
                                    }
                                    int outv_6[8];
                                    outv_6[0] = -1;
                                    if (_vec_load_25[0] >= 0 && _vec_load_25[0] <= bound_1) {
                                        outv_6[0] = _vec_load_25[0] + key_base;
                                        length = spos_4 + 1;
                                    }
                                    outv_6[1] = -1;
                                    if (_vec_load_25[1] >= 0 && _vec_load_25[1] <= bound_1) {
                                        outv_6[1] = _vec_load_25[1] + key_base;
                                        length = spos_4 + 1 + 1;
                                    }
                                    outv_6[2] = -1;
                                    if (_vec_load_25[2] >= 0 && _vec_load_25[2] <= bound_1) {
                                        outv_6[2] = _vec_load_25[2] + key_base;
                                        length = spos_4 + 2 + 1;
                                    }
                                    outv_6[3] = -1;
                                    if (_vec_load_25[3] >= 0 && _vec_load_25[3] <= bound_1) {
                                        outv_6[3] = _vec_load_25[3] + key_base;
                                        length = spos_4 + 3 + 1;
                                    }
                                    outv_6[4] = -1;
                                    if (_vec_load_25[4] >= 0 && _vec_load_25[4] <= bound_1) {
                                        outv_6[4] = _vec_load_25[4] + key_base;
                                        length = spos_4 + 4 + 1;
                                    }
                                    outv_6[5] = -1;
                                    if (_vec_load_25[5] >= 0 && _vec_load_25[5] <= bound_1) {
                                        outv_6[5] = _vec_load_25[5] + key_base;
                                        length = spos_4 + 5 + 1;
                                    }
                                    outv_6[6] = -1;
                                    if (_vec_load_25[6] >= 0 && _vec_load_25[6] <= bound_1) {
                                        outv_6[6] = _vec_load_25[6] + key_base;
                                        length = spos_4 + 6 + 1;
                                    }
                                    outv_6[7] = -1;
                                    if (_vec_load_25[7] >= 0 && _vec_load_25[7] <= bound_1) {
                                        outv_6[7] = _vec_load_25[7] + key_base;
                                        length = spos_4 + 7 + 1;
                                    }
                                    if (out_vec != 0) {
                                        {
                                            int4 _iv4 = make_int4(outv_6[0 + 0], outv_6[0 + 1], outv_6[0 + 2], outv_6[0 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_4) = _iv4;
                                        }
                                        {
                                            int4 _iv4 = make_int4(outv_6[4 + 0], outv_6[4 + 1], outv_6[4 + 2], outv_6[4 + 3]);
                                            *reinterpret_cast<int4*>(indices_out + out_base + (long long)spos_4 + 4) = _iv4;
                                        }
                                    } else {
                                        indices_out[out_base + (long long)spos_4] = outv_6[0];
                                        indices_out[out_base + (long long)(spos_4 + 1)] = outv_6[1];
                                        indices_out[out_base + (long long)(spos_4 + 2)] = outv_6[2];
                                        indices_out[out_base + (long long)(spos_4 + 3)] = outv_6[3];
                                        indices_out[out_base + (long long)(spos_4 + 4)] = outv_6[4];
                                        indices_out[out_base + (long long)(spos_4 + 5)] = outv_6[5];
                                        indices_out[out_base + (long long)(spos_4 + 6)] = outv_6[6];
                                        indices_out[out_base + (long long)(spos_4 + 7)] = outv_6[7];
                                    }
                                }
                                int tail0_1 = n_full * 128 + lane16;
                                int n_tail_1 = (topk - n_full * 128 - lane16 + 15) / 16;
                                #pragma unroll 1
                                for (int t_3 = 0; t_3 < n_tail_1; t_3++) {
                                    int pos_1 = tail0_1 + t_3 * 16;
                                    int value_10 = indices[row_base_2 + (long long)pos_1];
                                    int converted_1 = -1;
                                    if (value_10 >= 0 && value_10 <= bound_1) {
                                        converted_1 = value_10 + key_base;
                                        length = pos_1 + 1;
                                    }
                                    indices_out[out_base + (long long)pos_1] = converted_1;
                                }
                            } else {
                                #pragma unroll 4
                                for (int g = 0; g < n_full; g++) {
                                    int spos_6 = g * 128 + lane16 * 8;
                                    int _vec_load_26[8];
                                    {
                                        uint32_t _iv_79_0;
                                        uint32_t _iv_79_1;
                                        uint32_t _iv_79_2;
                                        uint32_t _iv_79_3;
                                        uint32_t _iv_79_4;
                                        uint32_t _iv_79_5;
                                        uint32_t _iv_79_6;
                                        uint32_t _iv_79_7;
                                        asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                            : "=r"(_iv_79_0), "=r"(_iv_79_1), "=r"(_iv_79_2), "=r"(_iv_79_3), "=r"(_iv_79_4), "=r"(_iv_79_5), "=r"(_iv_79_6), "=r"(_iv_79_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos_6) + (0))) : "memory");
                                        _vec_load_26[0 + 0] = (int32_t)_iv_79_0;
                                        _vec_load_26[0 + 1] = (int32_t)_iv_79_1;
                                        _vec_load_26[0 + 2] = (int32_t)_iv_79_2;
                                        _vec_load_26[0 + 3] = (int32_t)_iv_79_3;
                                        _vec_load_26[0 + 4] = (int32_t)_iv_79_4;
                                        _vec_load_26[0 + 5] = (int32_t)_iv_79_5;
                                        _vec_load_26[0 + 6] = (int32_t)_iv_79_6;
                                        _vec_load_26[0 + 7] = (int32_t)_iv_79_7;
                                    }
                                    if (_vec_load_26[0] >= 0 && _vec_load_26[0] <= bound_1) {
                                        length = spos_6 + 1;
                                    }
                                    if (_vec_load_26[1] >= 0 && _vec_load_26[1] <= bound_1) {
                                        length = spos_6 + 1 + 1;
                                    }
                                    if (_vec_load_26[2] >= 0 && _vec_load_26[2] <= bound_1) {
                                        length = spos_6 + 2 + 1;
                                    }
                                    if (_vec_load_26[3] >= 0 && _vec_load_26[3] <= bound_1) {
                                        length = spos_6 + 3 + 1;
                                    }
                                    if (_vec_load_26[4] >= 0 && _vec_load_26[4] <= bound_1) {
                                        length = spos_6 + 4 + 1;
                                    }
                                    if (_vec_load_26[5] >= 0 && _vec_load_26[5] <= bound_1) {
                                        length = spos_6 + 5 + 1;
                                    }
                                    if (_vec_load_26[6] >= 0 && _vec_load_26[6] <= bound_1) {
                                        length = spos_6 + 6 + 1;
                                    }
                                    if (_vec_load_26[7] >= 0 && _vec_load_26[7] <= bound_1) {
                                        length = spos_6 + 7 + 1;
                                    }
                                }
                                int tail0_2 = n_full * 128 + lane16;
                                int n_tail_2 = (topk - n_full * 128 - lane16 + 15) / 16;
                                #pragma unroll 1
                                for (int t_4 = 0; t_4 < n_tail_2; t_4++) {
                                    int pos_2 = tail0_2 + t_4 * 16;
                                    int value_16 = indices[row_base_2 + (long long)pos_2];
                                    if (value_16 >= 0 && value_16 <= bound_1) {
                                        length = pos_2 + 1;
                                    }
                                }
                            }
                            int part = length;
                            if (convert != 1) {
                                int peer_1 = __shfl_xor_sync(4294901760u, part, 8);
                                int _max_80_1 = ((part) > (peer_1) ? (part) : (peer_1));
                                part = _max_80_1;
                                int peer_0 = __shfl_xor_sync(4294901760u, part, 4);
                                int _max_81 = ((part) > (peer_0) ? (part) : (peer_0));
                                part = _max_81;
                                int peer_1_1 = __shfl_xor_sync(4294901760u, part, 2);
                                int _max_82_1 = ((part) > (peer_1_1) ? (part) : (peer_1_1));
                                part = _max_82_1;
                                int peer_2 = __shfl_xor_sync(4294901760u, part, 1);
                                int _max_83 = ((part) > (peer_2) ? (part) : (peer_2));
                                part = _max_83;
                            }
                            if (convert != 0) {
                                __syncwarp(4294901760u);
                            }
                            if (lane == 16) {
                                len_cells[outer_5 & 1] = part;
                                bound_cells[outer_5 & 1] = bound_1;
                                topk_length[(long long)query_4] = part;
                                mbarrier_arrive(len_ready_addr + (outer_5 & 1) * 8);
                            }
                        }
                        mbarrier_wait(clc_full_addr, outer_phase_5);
                        uint32_t _clc_valid_5 = 0;
                        uint32_t _clc_ctaid_x_5;
                        uint32_t _clc_ctaid_y_5;
                        uint32_t _clc_ctaid_z_5;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                            : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                            : "memory");
                        mbarrier_arrive(clc_empty_addr);
                        if (_clc_valid_5 == 0) {
                            break;
                        }
                        query_4 = _clc_ctaid_x_5;
                        outer_phase_5 ^= 1;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
