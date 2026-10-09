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
 *
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
 * DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_mega_gate_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 11
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_SX_OFF 1024
#define SMEM_SX_STAGE_BYTES 2048
#define SMEM_SX_STRIDE 2048
#define SMEM_SW_OFF 23552
#define SMEM_SW_STAGE_BYTES 16384
#define SMEM_SW_STRIDE 16384
#define SMEM_METADATA_OFF 203776
#define SMEM_METADATA_STAGE_BYTES 3072
#define SMEM_METADATA_STRIDE 3072
#define SMEM_REDUCE_PARTIALS_OFF 206848
#define SMEM_REDUCE_PARTIALS_STAGE_BYTES 8192
#define SMEM_REDUCE_PARTIALS_STRIDE 8192
#define SMEM_CACHED_COUNTS_OFF 205312
#define SMEM_CACHED_COUNTS_STAGE_BYTES 1536
#define SMEM_CACHED_COUNTS_STRIDE 1536
#define SMEM_TOUCH_SINK_OFF 215040
#define SMEM_TOUCH_SINK_STAGE_BYTES 128
#define SMEM_TOUCH_SINK_STRIDE 128
#define SMEM_TOTAL 215168
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(16,1,1) void
kernel_cake_deepgemm_mega_gate_8be2bbbc2c42bd4ebb76(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap W, float* __restrict__ bias, float* __restrict__ image_bias, uint8_t* __restrict__ image_mask, uint8_t* __restrict__ mask, int* __restrict__ physical_map, int* __restrict__ logical_count, long long* __restrict__ topk_idx, long long* __restrict__ unmapped_idx, float* __restrict__ topk_weights, float* __restrict__ scratch, unsigned long long* __restrict__ score_barriers, uint8_t* __restrict__ fixed_mask, uint8_t* __restrict__ random_mask, int num_tokens, int num_shared, int map_width, unsigned int ep_rank, float routed_scale, long long unmapped_stride, int num_workers, int route_flags, int num_split_k)
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
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 88)
    #define tmem_full_addr (mbar_base + 176)
    #define tmem_empty_addr (mbar_base + 192)
    #define metadata_ready_addr (mbar_base + 208)
    #define reduce_full_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sx_addr = smem + 1024;
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 23552);
    const int sw_addr = smem + 23552;
    float* metadata = reinterpret_cast<float*>(smem_raw + 203776);
    const int metadata_addr = smem + 203776;
    float* reduce_partials = reinterpret_cast<float*>(smem_raw + 206848);
    const int reduce_partials_addr = smem + 206848;
    int* cached_counts = reinterpret_cast<int*>(smem_raw + 205312);
    const int cached_counts_addr = smem + 205312;
    int* touch_sink = reinterpret_cast<int*>(smem_raw + 215040);
    const int touch_sink_addr = smem + 215040;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (0, 4, 8), init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 64, 1);
            // empty: stages (0, 4, 8), init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'epi_pipe' ---
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // tmem_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // metadata_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 208, 32);
            // reduce_full: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (1, 5, 9), init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 72, 1);
            // empty: stages (1, 5, 9), init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 160, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (2, 6, 10), init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 80, 1);
            // empty: stages (2, 6, 10), init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 168, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 3) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (3, 7), init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 56, 1);
            // empty: stages (3, 7), init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 144, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 2) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int ls = 0;
            int logical = bid % 48;
            int expert_group = logical % 3;
            int split = logical / 3;
            split = logical % 16;
            expert_group = logical / 16;
            int num_blocks = (num_tokens + 15) / 16;
            int tail_n = (num_tokens - (num_blocks - 1) * 16 + 15) / 16 * 16;
            unsigned int _phase_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile = bid / 48; tile < num_blocks; tile += num_workers) {
                    int _max_0 = ((tail_n) > (16) ? (tail_n) : (16));
                    int effective_n = ((num_blocks == tile + 1) ? _max_0 : 16);
                    #pragma unroll 4
                    for (int kb = 0; kb < 5; kb++) {
                        mbarrier_wait(empty_addr + (ls) * 8, _phase_empty);
                        int ko = split * 5 * 64 + kb * 64;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sx_addr + ls * 2048), "l"((&X)), "r"(0), "r"(tile * 16), "r"(ko / 64),
                               "r"(full_addr + (ls) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem(sw_addr + ls * 16384, (&W), 0, expert_group * 128, ko / 64, full_addr + (ls) * 8);
                        mbarrier_arrive_expect_tx(full_addr + (ls) * 8, 18432);
                        ls += 1;
                        if (ls == 11) { ls = 0; _phase_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int ms = 0;
            unsigned int me = 0;
            int num_blocks_1 = (num_tokens + 15) / 16;
            int tail_n_1 = (num_tokens - (num_blocks_1 - 1) * 16 + 15) / 16 * 16;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid / 48; tile_1 < num_blocks_1; tile_1 += num_workers) {
                int _max_1 = ((tail_n_1) > (16) ? (tail_n_1) : (16));
                int effective_n_1 = ((num_blocks_1 == tile_1 + 1) ? _max_1 : 16);
                mbarrier_wait(tmem_empty_addr + (me) * 8, _phase_tmem_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int kb_1 = 0; kb_1 < 5; kb_1++) {
                    mbarrier_wait(full_addr + (ms) * 8, _phase_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init = ((kb_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((sw_addr) >> 4) & 0x3FFF) + (ms) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((sx_addr) >> 4) & 0x3FFF) + (ms) * 128);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        const uint32_t _mma_ss_idesc_0 = ((0x8040490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(effective_n_1) >> 3) << 17));
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (me * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, _mma_ss_idesc_0, ((init) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (me * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, _mma_ss_idesc_0, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (me * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, _mma_ss_idesc_0, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (me * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, _mma_ss_idesc_0, 1);
                        }
                    }
                    __syncwarp();
                    if (kb_1 + 1 == 5) {
                        elect_commit(tmem_full_addr + (me) * 8);
                    }
                    __syncwarp();
                    elect_commit(empty_addr + (ms) * 8);
                    ms += 1;
                    if (ms == 11) { ms = 0; _phase_full ^= 1; }
                }
                me += 1;
                if (me == 2) { me = 0; _phase_tmem_empty ^= 1; }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 2) {
        { // idle_main
            if ((route_flags & 1) != 0) {
                int touched_sum = 0;
                #pragma unroll 16
                for (int sector = lane * 8; sector < 384 * map_width; sector += 256) {
                    touched_sum = touched_sum + physical_map[sector];
                }
                touch_sink[lane] = touched_sum;
            }
        }
    }
    // ---- Role: cache ----
    if (warp == 3) {
        { // cache_main
            #pragma unroll
            for (int wave = 0; wave < 12; wave++) {
                int expert = wave * 32 + lane;
                metadata[expert] = ((expert < 384) ? bias[expert] : 0.0f);
                cached_counts[expert] = (((route_flags & 1) != 0 && expert < 384) ? logical_count[expert] : 0);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            mbarrier_arrive(metadata_ready_addr);
        }
    }
    // ---- Role: gate ----
    if (warp >= 4 && warp <= 11) {
        { // gate_main
            unsigned int ge = 0;
            int gw = warp - 4;
            int logical_1 = bid % 48;
            int expert_group_1 = logical_1 % 3;
            int split_1 = logical_1 / 3;
            split_1 = logical_1 % 16;
            expert_group_1 = logical_1 / 16;
            uint64_t _grid_id_0;
            asm volatile("mov.u64 %0, %%gridid;" : "=l"(_grid_id_0));
            unsigned long long epoch = (_grid_id_0 + 1) * 64;
            mbarrier_wait(metadata_ready_addr, 0);
            if (split_1 == 0) {
                if (gw == 0 && lane == 0) {
                    mbarrier_arrive_expect_tx(reduce_full_addr, 7680);
                }
            }
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(reduce_partials_addr), "r"(0));
            unsigned int remote_partials = _mapa_0;
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(reduce_full_addr), "r"(0));
            unsigned int remote_reduce_full = _mapa_1;
            int num_blocks_2 = (num_tokens + 15) / 16;
            int tail_n_2 = (num_tokens - (num_blocks_2 - 1) * 16 + 15) / 16 * 16;
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid / 48; tile_2 < num_blocks_2; tile_2 += num_workers) {
                int _max_2 = ((tail_n_2) > (16) ? (tail_n_2) : (16));
                int effective_n_2 = ((num_blocks_2 == tile_2 + 1) ? _max_2 : 16);
                int _min_0 = ((num_tokens - tile_2 * 16) < (16) ? (num_tokens - tile_2 * 16) : (16));
                int valid_tokens = _min_0;
                if (gw == 7 && lane == 31) {
                    if (logical_1 == 0) {
                        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(epoch)) : "memory");
                    } else if (split_1 == 0) {
                        {
                        unsigned long long _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                        } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch)) >= static_cast<unsigned long long>(3));
                        }
                    }
                }
                __syncwarp();
                mbarrier_wait(tmem_full_addr + (ge) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int subpartition = gw % 4;
                int num_columns = ((0) ? effective_n_2 / 2 : effective_n_2);
                int token_base = ((0) ? subpartition / 2 * num_columns : 0);
                int expert_atom = ((0) ? subpartition % 2 : subpartition);
                int expert_1 = expert_group_1 * 128 + expert_atom * 32 + lane;
                int expert_local = expert_atom * 32 + lane;
                #pragma unroll 1
                for (int col = gw / 4 * 8; col < num_columns; col += 16) {
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + ge * 16 + (unsigned int)(subpartition * 32 << 16) + (unsigned int)col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int item = 0; item < 8; item++) {
                        float stored = _tmem_load_0[item];
                        int partial_token = token_base + col + item;
                        if (partial_token < 1) {
                            if (split_1 == 0) {
                                reduce_partials[partial_token * 128 + expert_local] = stored;
                            } else {
                                asm volatile(
                                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                    :: "r"(remote_partials + (unsigned int)(((split_1 + partial_token) * 128 + expert_local) * 4)), "f"(stored), "r"(remote_reduce_full) : "memory");
                            }
                        }
                    }
                }
                asm volatile("bar.sync 8, 256;" ::: "memory");
                if (split_1 == 0) {
                    mbarrier_wait(reduce_full_addr, 0);
                    if (gw < 4) {
                        float lp[16];
                        #pragma unroll
                        for (int sp = 0; sp < 16; sp++) {
                            lp[sp] = reduce_partials[sp * 128 + expert_local];
                        }
                        float ls1[8];
                        float lb1[8];
                        float le1[8];
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            ls1[i] = lp[2 * i] + lp[2 * i + 1];
                            lb1[i] = ls1[i] - lp[2 * i];
                            le1[i] = lp[2 * i] - (ls1[i] - lb1[i]) + (lp[2 * i + 1] - lb1[i]);
                        }
                        float ls2[4];
                        float lb2[4];
                        float le2[4];
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            ls2[i_1] = ls1[2 * i_1] + ls1[2 * i_1 + 1];
                            lb2[i_1] = ls2[i_1] - ls1[2 * i_1];
                            le2[i_1] = ls1[2 * i_1] - (ls2[i_1] - lb2[i_1]) + (ls1[2 * i_1 + 1] - lb2[i_1]);
                            le1[i_1] = le1[2 * i_1] + le1[2 * i_1 + 1];
                        }
                        float ls3[2];
                        float lb3[2];
                        float le3[2];
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 2; i_2++) {
                            ls3[i_2] = ls2[2 * i_2] + ls2[2 * i_2 + 1];
                            lb3[i_2] = ls3[i_2] - ls2[2 * i_2];
                            le3[i_2] = ls2[2 * i_2] - (ls3[i_2] - lb3[i_2]) + (ls2[2 * i_2 + 1] - lb3[i_2]);
                            le1[i_2] = le1[2 * i_2] + le1[2 * i_2 + 1];
                            le2[i_2] = le2[2 * i_2] + le2[2 * i_2 + 1];
                        }
                        float leader_sum4 = ls3[0] + ls3[1];
                        float leader_bp4 = leader_sum4 - ls3[0];
                        float leader_err4 = ls3[0] - (leader_sum4 - leader_bp4) + (ls3[1] - leader_bp4);
                        float leader_err = le1[0] + le1[1] + (le2[0] + le2[1]) + (le3[0] + le3[1] + leader_err4);
                        float leader_total = leader_sum4 + leader_err;
                        float leader_unbiased = leader_total;
                        float result = leader_total;
                        float _exp_0 = expf(leader_total);
                        float _log1p_0 = log1pf(_exp_0);
                        float softplus = _log1p_0;
                        float _sqrt_0;
                        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(((leader_total > 20.0f) ? leader_total : softplus)));
                        result = _sqrt_0;
                        leader_unbiased = result;
                        float leader_bias = 0.0f;
                        leader_bias = metadata[expert_1];
                        float leader_ranking = ((expert_1 < 384) ? leader_unbiased + leader_bias : -CUDART_INF_F);
                        unsigned int leader_id = (unsigned int)expert_1;
                        float leader_record[4];
                        leader_record[1] = leader_unbiased;
                        leader_record[2] = (float)expert_1;
                        leader_record[3] = 0.0f;
                        float pad_record[4];
                        pad_record[0] = -CUDART_INF_F;
                        pad_record[1] = 0.0f;
                        pad_record[2] = -1.0f;
                        pad_record[3] = 0.0f;
                        #pragma unroll
                        for (int oi = 0; oi < 6; oi++) {
                            float _warp_redux_f32_0;
                            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(leader_ranking));
                            unsigned int leader_tied = ((leader_ranking == _warp_redux_f32_0 && leader_ranking > -CUDART_INF_F) ? leader_id : (unsigned int)4294967295);
                            unsigned int _warp_redux_u32_0;
                            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(leader_tied));
                            if (_warp_redux_u32_0 == (unsigned int)4294967295) {
                                if (lane == oi) {
                                    {
                                        float4 _v4 = make_float4(pad_record[0 + 0], pad_record[0 + 1], pad_record[0 + 2], pad_record[0 + 3]);
                                        *reinterpret_cast<float4*>((scratch + (tile_2 * 16 * 16 * 384 + expert_group_1 * 128 + (gw * 6 + oi) * 4)) + 0) = _v4;
                                    }
                                }
                            } else if (leader_tied == _warp_redux_u32_0) {
                                leader_record[0] = leader_ranking;
                                {
                                    float4 _v4 = make_float4(leader_record[0 + 0], leader_record[0 + 1], leader_record[0 + 2], leader_record[0 + 3]);
                                    *reinterpret_cast<float4*>((scratch + (tile_2 * 16 * 16 * 384 + expert_group_1 * 128 + (gw * 6 + oi) * 4)) + 0) = _v4;
                                }
                                leader_ranking = -CUDART_INF_F;
                            }
                        }
                    }
                    asm volatile("bar.sync 8, 256;" ::: "memory");
                }
                if (gw == 7 && lane == 31) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tmem_empty_addr + (ge) * 8);
                    if (split_1 == 0) {
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                        asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                        #elif defined(__CUDA_ARCH__)
                        #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                        #endif
                    }
                }
                if (gw == 0 && lane == 0) {
                    if (logical_1 == 0) {
                        {
                        unsigned long long _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                        } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch + 3)) >= static_cast<unsigned long long>(1));
                        }
                    }
                }
                asm volatile("bar.sync 8, 256;" ::: "memory");
                #pragma unroll 1
                for (int token_local = gw * 48 + logical_1; token_local < valid_tokens; token_local += 384) {
                    int token = tile_2 * 16 + token_local;
                    int handled = 0;
                    if (handled == 0) {
                        float cand_ranking[3];
                        float cand_unbiased[3];
                        unsigned int cand_id[3];
                        #pragma unroll
                        for (int cand_slot = 0; cand_slot < 3; cand_slot++) {
                            cand_ranking[cand_slot] = -CUDART_INF_F;
                            cand_unbiased[cand_slot] = 0.0f;
                            cand_id[cand_slot] = (unsigned int)4294967295;
                            int cand_index = cand_slot * 32 + lane;
                            if (cand_index < 72) {
                                float _vec_load_0[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + ((tile_2 * 16 * 16 + token_local) * 384 + cand_index / 24 * 128 + cand_index % 24 * 4) + 0);
                                    _vec_load_0[0 + 0] = _v4.x;
                                    _vec_load_0[0 + 1] = _v4.y;
                                    _vec_load_0[0 + 2] = _v4.z;
                                    _vec_load_0[0 + 3] = _v4.w;
                                }
                                cand_ranking[cand_slot] = _vec_load_0[0];
                                cand_unbiased[cand_slot] = _vec_load_0[1];
                                cand_id[cand_slot] = (unsigned int)(int)_vec_load_0[2];
                            }
                        }
                        if (cand_ranking[1] > cand_ranking[0] || cand_ranking[1] == cand_ranking[0] && cand_id[1] < cand_id[0]) {
                            float swapped_ranking = cand_ranking[0];
                            float swapped_unbiased = cand_unbiased[0];
                            unsigned int swapped_id = cand_id[0];
                            cand_ranking[0] = cand_ranking[1];
                            cand_unbiased[0] = cand_unbiased[1];
                            cand_id[0] = cand_id[1];
                            cand_ranking[1] = swapped_ranking;
                            cand_unbiased[1] = swapped_unbiased;
                            cand_id[1] = swapped_id;
                        }
                        if (cand_ranking[2] > cand_ranking[1] || cand_ranking[2] == cand_ranking[1] && cand_id[2] < cand_id[1]) {
                            float swapped_ranking_1 = cand_ranking[1];
                            float swapped_unbiased_1 = cand_unbiased[1];
                            unsigned int swapped_id_1 = cand_id[1];
                            cand_ranking[1] = cand_ranking[2];
                            cand_unbiased[1] = cand_unbiased[2];
                            cand_id[1] = cand_id[2];
                            cand_ranking[2] = swapped_ranking_1;
                            cand_unbiased[2] = swapped_unbiased_1;
                            cand_id[2] = swapped_id_1;
                        }
                        if (cand_ranking[1] > cand_ranking[0] || cand_ranking[1] == cand_ranking[0] && cand_id[1] < cand_id[0]) {
                            float swapped_ranking_2 = cand_ranking[0];
                            float swapped_unbiased_2 = cand_unbiased[0];
                            unsigned int swapped_id_2 = cand_id[0];
                            cand_ranking[0] = cand_ranking[1];
                            cand_unbiased[0] = cand_unbiased[1];
                            cand_id[0] = cand_id[1];
                            cand_ranking[1] = swapped_ranking_2;
                            cand_unbiased[1] = swapped_unbiased_2;
                            cand_id[1] = swapped_id_2;
                        }
                        int merge_cursor = 0;
                        int selected = -1;
                        float selected_score = 0.0f;
                        #pragma unroll
                        for (int oi_1 = 0; oi_1 < 6; oi_1++) {
                            float proposed_ranking = -CUDART_INF_F;
                            float proposed_unbiased = 0.0f;
                            unsigned int proposed_id = (unsigned int)4294967295;
                            #pragma unroll
                            for (int cand_slot_1 = 0; cand_slot_1 < 3; cand_slot_1++) {
                                if (merge_cursor == cand_slot_1) {
                                    proposed_ranking = cand_ranking[cand_slot_1];
                                    proposed_unbiased = cand_unbiased[cand_slot_1];
                                    proposed_id = cand_id[cand_slot_1];
                                }
                            }
                            float _warp_redux_f32_1;
                            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_1) : "f"(proposed_ranking));
                            unsigned int merge_tied = ((proposed_ranking == _warp_redux_f32_1 && proposed_ranking > -CUDART_INF_F) ? proposed_id : (unsigned int)4294967295);
                            unsigned int _warp_redux_u32_1;
                            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(merge_tied));
                            float _warp_redux_f32_2;
                            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_2) : "f"(((merge_tied == _warp_redux_u32_1) ? proposed_unbiased : -CUDART_INF_F)));
                            if (_warp_redux_u32_1 != (unsigned int)4294967295) {
                                if (merge_tied == _warp_redux_u32_1) {
                                    merge_cursor = merge_cursor + 1;
                                }
                                if (lane == oi_1) {
                                    selected_score = _warp_redux_f32_2;
                                }
                            }
                            if (lane == oi_1) {
                                selected = (int)_warp_redux_u32_1;
                            }
                        }
                        int chosen = selected;
                        if (lane >= 6 && lane < 6 + num_shared) {
                            chosen = lane + 384 - 6;
                        }
                        int physical = chosen;
                        if ((route_flags & 1) != 0) {
                            if (lane < 6 + num_shared) {
                                unsigned int duplicates = 0;
                                if (lane < 6) {
                                    duplicates = (unsigned int)cached_counts[chosen];
                                } else {
                                    duplicates = (unsigned int)logical_count[chosen];
                                }
                                unsigned int duplicate = (ep_rank + (unsigned int)token * 23333) % duplicates;
                                physical = physical_map[(unsigned int)(chosen * map_width) + duplicate];
                            }
                        }
                        float total = selected_score;
                        #pragma unroll
                        for (int delta = 0; delta < 5; delta++) {
                            if (16 >> delta < 8) {
                                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, total, 16 >> delta);
                                total = total + _shfl_xor_0;
                            }
                        }
                        float result_1 = selected_score;
                        if (lane < 6) {
                            result_1 = selected_score / (total + 1e-20f) * routed_scale;
                            if ((route_flags & 2) != 0) {
                                unmapped_idx[(long long)token * unmapped_stride + (long long)lane] = chosen;
                            }
                        } else if (lane < 6 + num_shared) {
                            result_1 = 1.0f;
                        }
                        if (lane < 6 + num_shared) {
                            topk_idx[token * (6 + num_shared) + lane] = (long long)physical;
                            topk_weights[token * (6 + num_shared) + lane] = result_1;
                        }
                    }
                }
                ge += 1;
                if (ge == 2) { ge = 0; _phase_tmem_full ^= 1; }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    __syncwarp();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(32));
    }

    // Cleanup
}

} // extern "C"
