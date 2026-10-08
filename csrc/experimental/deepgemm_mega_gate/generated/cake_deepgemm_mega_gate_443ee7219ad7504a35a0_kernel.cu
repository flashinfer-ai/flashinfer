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
#define TMEM_NCOLS 352
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 11
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_SX_OFF 1024
#define SMEM_SX_STAGE_BYTES 11264
#define SMEM_SX_STRIDE 11264
#define SMEM_SW_OFF 124928
#define SMEM_SW_STAGE_BYTES 8192
#define SMEM_SW_STRIDE 8192
#define SMEM_METADATA_OFF 215040
#define SMEM_METADATA_STAGE_BYTES 3072
#define SMEM_METADATA_STRIDE 3072
#define SMEM_CACHED_COUNTS_OFF 216576
#define SMEM_CACHED_COUNTS_STAGE_BYTES 1536
#define SMEM_CACHED_COUNTS_STRIDE 1536
#define SMEM_TOTAL 218112
#define LAUNCH_MIN_BLOCKS 1

template <int kNumSplitK>
__global__ __launch_bounds__(896, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_mega_gate_443ee7219ad7504a35a0(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap W, float* __restrict__ bias, float* __restrict__ image_bias, uint8_t* __restrict__ image_mask, uint8_t* __restrict__ mask, int* __restrict__ physical_map, int* __restrict__ logical_count, long long* __restrict__ topk_idx, long long* __restrict__ unmapped_idx, float* __restrict__ topk_weights, float* __restrict__ scratch, unsigned long long* __restrict__ score_barriers, uint8_t* __restrict__ fixed_mask, uint8_t* __restrict__ random_mask, int num_tokens, int num_shared, int map_width, unsigned int ep_rank, float routed_scale, long long unmapped_stride, int num_workers, int route_flags, int num_split_k)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sx_addr = smem + 1024;
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 124928);
    const int sw_addr = smem + 124928;
    float* metadata = reinterpret_cast<float*>(smem_raw + 215040);
    const int metadata_addr = smem + 215040;
    int* cached_counts = reinterpret_cast<int*>(smem_raw + 216576);
    const int cached_counts_addr = smem + 216576;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (0, 4, 8), init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 64, 2);
            // empty: stages (0, 4, 8), init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'epi_pipe' ---
            // tmem_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // tmem_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 192, 2);
            mbarrier_init(smem + 200, 2);
            // metadata_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 208, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // full: stages (1, 5, 9), init_count=2
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 72, 2);
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
            // full: stages (2, 6, 10), init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 80, 2);
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
            // full: stages (3, 7), init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 56, 2);
            // empty: stages (3, 7), init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 144, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 352 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 2) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    // Partial post-allocation TMEM rendezvous (864 threads on named barrier 9)
    if (warp >= 1 && warp <= 27) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(9), "r"(864) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp >= 1 && warp <= 27) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int ls = 0;
            int logical = bid % (6 * kNumSplitK);
            int expert_group = logical / 2 % 3;
            int split = logical / 6;
            int num_blocks = (num_tokens + 175) / 176;
            int tail_n = (num_tokens - (num_blocks - 1) * 176 + 15) / 16 * 16;
            unsigned int _phase_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile = bid / (6 * kNumSplitK); tile < num_blocks; tile += num_workers) {
                    int _max_0 = ((tail_n) > (32) ? (tail_n) : (32));
                    int effective_n = ((num_blocks == tile + 1) ? _max_0 : 176);
                    #pragma unroll 4
                    for (int kb = 0; kb < (80 / kNumSplitK); kb++) {
                        mbarrier_wait(empty_addr + (ls) * 8, _phase_empty);
                        int ko = split * (80 / kNumSplitK) * 64 + kb * 64;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(sx_addr + ls * 11264), "l"((&X)), "r"(0), "r"(tile * 176 + cta_rank * (effective_n / 2)), "r"(ko / 64),
                               "r"(((full_addr + (ls) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem_cta2(sw_addr + ls * 8192, (&W), 0, expert_group * 128 + cta_rank * 64, ko / 64, ((full_addr + (ls) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (ls) * 8, 38912);
                        } else {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(full_addr + ls * 8), "r"(0) : "memory");
                        }
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
            int num_blocks_1 = (num_tokens + 175) / 176;
            int tail_n_1 = (num_tokens - (num_blocks_1 - 1) * 176 + 15) / 16 * 16;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int tile_1 = bid / (6 * kNumSplitK); tile_1 < num_blocks_1; tile_1 += num_workers) {
                    int _max_1 = ((tail_n_1) > (32) ? (tail_n_1) : (32));
                    int effective_n_1 = ((num_blocks_1 == tile_1 + 1) ? _max_1 : 176);
                    mbarrier_wait(tmem_empty_addr + (me) * 8, _phase_tmem_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int kb_1 = 0; kb_1 < (80 / kNumSplitK); kb_1++) {
                        mbarrier_wait(full_addr + (ms) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init = ((kb_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((sw_addr) >> 4) & 0x3FFF) + (ms) * 512;
                        int _mma_b_lo_0 = (((sx_addr) >> 4) & 0x3FFF) + (ms) * 704;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, %4;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (me * 176))), "r"(((init) ? 0 : 1)), "r"(((0x82c0490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(effective_n_1) >> 3) << 17))));
                        __syncwarp();
                        if (kb_1 + 1 == (80 / kNumSplitK)) {
                            elect_commit_cg2_multicast(tmem_full_addr + (me) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (ms) * 8, (uint16_t)(3));
                        ms += 1;
                        if (ms == 11) { ms = 0; _phase_full ^= 1; }
                    }
                    me += 1;
                    if (me == 2) { me = 0; _phase_tmem_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 2) {
        // idle — no tasks assigned
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
    if (warp >= 4 && warp <= 27) {
        { // gate_main
            unsigned int ge = 0;
            int gw = warp - 4;
            int logical_1 = bid % (6 * kNumSplitK);
            int expert_group_1 = logical_1 / 2 % 3;
            int split_1 = logical_1 / 6;
            uint64_t _grid_id_0;
            asm volatile("mov.u64 %0, %%gridid;" : "=l"(_grid_id_0));
            unsigned long long epoch = (_grid_id_0 + 1) * 64;
            mbarrier_wait(metadata_ready_addr, 0);
            int num_blocks_2 = (num_tokens + 175) / 176;
            int tail_n_2 = (num_tokens - (num_blocks_2 - 1) * 176 + 15) / 16 * 16;
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (int tile_2 = bid / (6 * kNumSplitK); tile_2 < num_blocks_2; tile_2 += num_workers) {
                int _max_2 = ((tail_n_2) > (32) ? (tail_n_2) : (32));
                int effective_n_2 = ((num_blocks_2 == tile_2 + 1) ? _max_2 : 176);
                int _min_0 = ((num_tokens - tile_2 * 176) < (176) ? (num_tokens - tile_2 * 176) : (176));
                int valid_tokens = _min_0;
                if (gw == 23 && lane == 31) {
                    if (logical_1 == 0) {
                        asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(epoch)) : "memory");
                    } else {
                        {
                        unsigned long long _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                        } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch)) >= static_cast<unsigned long long>((6 * kNumSplitK)));
                        }
                    }
                }
                __syncwarp();
                mbarrier_wait(tmem_full_addr + (ge) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int subpartition = gw % 4;
                int num_columns = ((1) ? effective_n_2 / 2 : effective_n_2);
                int token_base = ((1) ? subpartition / 2 * num_columns : 0);
                int expert_atom = ((1) ? subpartition % 2 : subpartition);
                int expert_1 = expert_group_1 * 128 + cta_rank * 64 + expert_atom * 32 + lane;
                int expert_local = expert_atom * 32 + lane;
                #pragma unroll 1
                for (int col = gw / 4 * 8; col < num_columns; col += 48) {
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + ge * 176 + (unsigned int)(subpartition * 32 << 16) + (unsigned int)col);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int item = 0; item < 8; item++) {
                        float stored = _tmem_load_0[item];
                        scratch[((tile_2 * kNumSplitK + split_1) * 176 + token_base + col + item) * 384 + expert_1] = stored;
                    }
                }
                asm volatile("bar.sync 8, 768;" ::: "memory");
                if (gw == 23 && lane == 31) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((tmem_empty_addr + (ge) * 8) & 0xFEFFFFFF) : "memory");
                    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
                    asm volatile("red.async.release.gpu.global.add.u64 [%0], %1;" :: "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))), "l"(static_cast<unsigned long long>(1)) : "memory");
                    #elif defined(__CUDA_ARCH__)
                    #error "GlobalRedAsyncReleaseAdd requires SM100 or newer"
                    #endif
                }
                if (gw == 0 && lane == 0) {
                    {
                    unsigned long long _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_acquire_observed) : "l"((reinterpret_cast<unsigned long long*>(score_barriers) + (tile_2 * 16))) : "memory");
                    } while (static_cast<unsigned long long>(_acquire_observed - static_cast<unsigned long long>(epoch + (6 * kNumSplitK))) >= static_cast<unsigned long long>(1));
                    }
                }
                asm volatile("bar.sync 8, 768;" ::: "memory");
                #pragma unroll 1
                for (int token_local = gw * (6 * kNumSplitK) + logical_1; token_local < valid_tokens; token_local += (144 * kNumSplitK)) {
                    int token = tile_2 * 176 + token_local;
                    int handled = 0;
                    if (handled == 0) {
                        float rankings[12];
                        float unbiased_values[12];
                        unsigned int permutations[3];
                        int is_image = 0;
                        #pragma unroll
                        for (int wave_1 = 0; wave_1 < 3; wave_1++) {
                            int expert_base = wave_1 * 128 + lane * 4;
                            float _vec_load_0[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(scratch + ((tile_2 * kNumSplitK * 176 + token_local) * 384 + expert_base) + 0);
                                _vec_load_0[0 + 0] = _v4.x;
                                _vec_load_0[0 + 1] = _v4.y;
                                _vec_load_0[0 + 2] = _v4.z;
                                _vec_load_0[0 + 3] = _v4.w;
                            }
                            #pragma unroll
                            for (int sp = 1; sp < kNumSplitK; sp++) {
                                float _vec_load_1[4];
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(scratch + (((tile_2 * kNumSplitK + sp) * 176 + token_local) * 384 + expert_base) + 0);
                                    _vec_load_1[0 + 0] = _v4.x;
                                    _vec_load_1[0 + 1] = _v4.y;
                                    _vec_load_1[0 + 2] = _v4.z;
                                    _vec_load_1[0 + 3] = _v4.w;
                                }
                                #pragma unroll
                                for (int value = 0; value < 4; value++) {
                                    _vec_load_0[value] = _vec_load_0[value] + _vec_load_1[value];
                                }
                            }
                            #pragma unroll
                            for (int value_1 = 0; value_1 < 4; value_1++) {
                                rankings[wave_1 * 4 + value_1] = _vec_load_0[value_1];
                            }
                        }
                        #pragma unroll
                        for (int wave_2 = 0; wave_2 < 3; wave_2++) {
                            float exponentials[4];
                            float softplus_values[4];
                            #pragma unroll
                            for (int value_2 = 0; value_2 < 4; value_2++) {
                                float _exp_0 = expf(rankings[wave_2 * 4 + value_2]);
                                exponentials[value_2] = _exp_0;
                            }
                            #pragma unroll
                            for (int value_3 = 0; value_3 < 4; value_3++) {
                                float _log1p_0 = log1pf(exponentials[value_3]);
                                softplus_values[value_3] = _log1p_0;
                            }
                            #pragma unroll
                            for (int value_4 = 0; value_4 < 4; value_4++) {
                                float raw_score = rankings[wave_2 * 4 + value_4];
                                float _sqrt_0;
                                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(((raw_score > 20.0f) ? raw_score : softplus_values[value_4])));
                                rankings[wave_2 * 4 + value_4] = _sqrt_0;
                            }
                            #pragma unroll
                            for (int value_5 = 0; value_5 < 4; value_5++) {
                                int expert_id = wave_2 * 128 + lane * 4 + value_5;
                                float raw = rankings[wave_2 * 4 + value_5];
                                unbiased_values[wave_2 * 4 + value_5] = raw;
                                float ranking_bias = 0.0f;
                                ranking_bias = metadata[expert_id];
                                rankings[wave_2 * 4 + value_5] = ((expert_id < 384) ? raw + ranking_bias : -CUDART_INF_F);
                            }
                        }
                        #pragma unroll
                        for (int wave_3 = 0; wave_3 < 3; wave_3++) {
                            int local_indices[4];
                            #pragma unroll
                            for (int value_6 = 0; value_6 < 4; value_6++) {
                                local_indices[value_6] = value_6;
                            }
                            if (rankings[wave_3 * 4 + 1] > rankings[wave_3 * 4] || rankings[wave_3 * 4 + 1] == rankings[wave_3 * 4] && local_indices[1] < local_indices[0]) {
                                float saved_score = rankings[wave_3 * 4];
                                int saved_idx = local_indices[0];
                                rankings[wave_3 * 4] = rankings[wave_3 * 4 + 1];
                                local_indices[0] = local_indices[1];
                                rankings[wave_3 * 4 + 1] = saved_score;
                                local_indices[1] = saved_idx;
                            }
                            if (rankings[wave_3 * 4 + 3] > rankings[wave_3 * 4 + 2] || rankings[wave_3 * 4 + 3] == rankings[wave_3 * 4 + 2] && local_indices[3] < local_indices[2]) {
                                float saved_score_1 = rankings[wave_3 * 4 + 2];
                                int saved_idx_1 = local_indices[2];
                                rankings[wave_3 * 4 + 2] = rankings[wave_3 * 4 + 3];
                                local_indices[2] = local_indices[3];
                                rankings[wave_3 * 4 + 3] = saved_score_1;
                                local_indices[3] = saved_idx_1;
                            }
                            if (rankings[wave_3 * 4 + 2] > rankings[wave_3 * 4] || rankings[wave_3 * 4 + 2] == rankings[wave_3 * 4] && local_indices[2] < local_indices[0]) {
                                float saved_score_2 = rankings[wave_3 * 4];
                                int saved_idx_2 = local_indices[0];
                                rankings[wave_3 * 4] = rankings[wave_3 * 4 + 2];
                                local_indices[0] = local_indices[2];
                                rankings[wave_3 * 4 + 2] = saved_score_2;
                                local_indices[2] = saved_idx_2;
                            }
                            if (rankings[wave_3 * 4 + 3] > rankings[wave_3 * 4 + 1] || rankings[wave_3 * 4 + 3] == rankings[wave_3 * 4 + 1] && local_indices[3] < local_indices[1]) {
                                float saved_score_3 = rankings[wave_3 * 4 + 1];
                                int saved_idx_3 = local_indices[1];
                                rankings[wave_3 * 4 + 1] = rankings[wave_3 * 4 + 3];
                                local_indices[1] = local_indices[3];
                                rankings[wave_3 * 4 + 3] = saved_score_3;
                                local_indices[3] = saved_idx_3;
                            }
                            if (rankings[wave_3 * 4 + 2] > rankings[wave_3 * 4 + 1] || rankings[wave_3 * 4 + 2] == rankings[wave_3 * 4 + 1] && local_indices[2] < local_indices[1]) {
                                float saved_score_4 = rankings[wave_3 * 4 + 1];
                                int saved_idx_4 = local_indices[1];
                                rankings[wave_3 * 4 + 1] = rankings[wave_3 * 4 + 2];
                                local_indices[1] = local_indices[2];
                                rankings[wave_3 * 4 + 2] = saved_score_4;
                                local_indices[2] = saved_idx_4;
                            }
                            permutations[wave_3] = (unsigned int)(local_indices[0] | local_indices[1] << 2 | local_indices[2] << 4 | local_indices[3] << 6);
                        }
                        unsigned int cursors = 0;
                        int selected = -1;
                        #pragma unroll
                        for (int oi = 0; oi < 6; oi++) {
                            float best_score = -CUDART_INF_F;
                            int best_expert = -1;
                            unsigned int best_wave = 0;
                            #pragma unroll
                            for (int wave_4 = 0; wave_4 < 3; wave_4++) {
                                unsigned int cursor = cursors >> (unsigned int)(wave_4 * 3) & 7;
                                float candidate_score = ((cursor == 4) ? -CUDART_INF_F : (((cursor & 2) != 0) ? (((cursor & 1) != 0) ? rankings[wave_4 * 4 + 3] : rankings[wave_4 * 4 + 2]) : (((cursor & 1) != 0) ? rankings[wave_4 * 4 + 1] : rankings[wave_4 * 4])));
                                unsigned int candidate_offset = permutations[wave_4] >> cursor * 2 & 3;
                                if (candidate_score > best_score) {
                                    best_score = candidate_score;
                                    best_expert = wave_4 * 128 + lane * 4 + (int)candidate_offset;
                                    best_wave = wave_4;
                                }
                            }
                            float _warp_redux_f32_0;
                            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(best_score));
                            unsigned int tied = ((best_score == _warp_redux_f32_0 && best_expert >= 0) ? (unsigned int)best_expert : (unsigned int)4294967295);
                            unsigned int _warp_redux_u32_0;
                            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(tied));
                            if ((unsigned int)best_expert == _warp_redux_u32_0) {
                                cursors = cursors + (1 << best_wave * 3);
                            }
                            if (lane == oi) {
                                selected = (int)_warp_redux_u32_0;
                            }
                        }
                        float selected_score = 0.0f;
                        int selected_value = ((selected >= 0) ? selected / 128 * 4 + selected % 4 : -1);
                        int source_lane = ((selected >= 0) ? selected % 128 / 4 : 0);
                        #pragma unroll
                        for (int vi = 0; vi < 12; vi++) {
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, unbiased_values[vi], source_lane);
                            if (selected_value == vi) {
                                selected_score = _shfl_0;
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
                        float result = selected_score;
                        if (lane < 6) {
                            result = selected_score / (total + 1e-20f) * routed_scale;
                            if ((route_flags & 2) != 0) {
                                unmapped_idx[(long long)token * unmapped_stride + (long long)lane] = chosen;
                            }
                        } else if (lane < 6 + num_shared) {
                            result = 1.0f;
                        }
                        if (lane < 6 + num_shared) {
                            topk_idx[token * (6 + num_shared) + lane] = (long long)physical;
                            topk_weights[token * (6 + num_shared) + lane] = result;
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
    }

    // Cleanup
}

template __global__ void kernel_cake_deepgemm_mega_gate_443ee7219ad7504a35a0<2>(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap W, float* __restrict__ bias, float* __restrict__ image_bias, uint8_t* __restrict__ image_mask, uint8_t* __restrict__ mask, int* __restrict__ physical_map, int* __restrict__ logical_count, long long* __restrict__ topk_idx, long long* __restrict__ unmapped_idx, float* __restrict__ topk_weights, float* __restrict__ scratch, unsigned long long* __restrict__ score_barriers, uint8_t* __restrict__ fixed_mask, uint8_t* __restrict__ random_mask, int num_tokens, int num_shared, int map_width, unsigned int ep_rank, float routed_scale, long long unmapped_stride, int num_workers, int route_flags, int num_split_k);
template __global__ void kernel_cake_deepgemm_mega_gate_443ee7219ad7504a35a0<4>(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap W, float* __restrict__ bias, float* __restrict__ image_bias, uint8_t* __restrict__ image_mask, uint8_t* __restrict__ mask, int* __restrict__ physical_map, int* __restrict__ logical_count, long long* __restrict__ topk_idx, long long* __restrict__ unmapped_idx, float* __restrict__ topk_weights, float* __restrict__ scratch, unsigned long long* __restrict__ score_barriers, uint8_t* __restrict__ fixed_mask, uint8_t* __restrict__ random_mask, int num_tokens, int num_shared, int map_width, unsigned int ep_rank, float routed_scale, long long unmapped_stride, int num_workers, int route_flags, int num_split_k);
