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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_msa_nvfp4_decode_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 384
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_K16_OFFSET 128
#define TMEM_TMEM_O_OFFSET 256
#define NUM_RAW_RING_STAGES 8
#define NUM_S_RING_STAGES 2
#define NUM_P_RING_STAGES 2
#define NUM_ITEM_RING_STAGES 2
#define SMEM_SMEM_F32_OFF 1024
#define SMEM_SMEM_F32_STAGE_BYTES 1024
#define SMEM_SMEM_F32_STRIDE 1024
#define SMEM_SMEM_PAGE_OFF 2048
#define SMEM_SMEM_PAGE_STAGE_BYTES 256
#define SMEM_SMEM_PAGE_STRIDE 256
#define SMEM_SMEM_BSTART_OFF 2304
#define SMEM_SMEM_BSTART_STAGE_BYTES 256
#define SMEM_SMEM_BSTART_STRIDE 256
#define SMEM_SMEM_MASK_OFF 2560
#define SMEM_SMEM_MASK_STAGE_BYTES 256
#define SMEM_SMEM_MASK_STRIDE 256
#define SMEM_SMEM_NPAIRS_OFF 2816
#define SMEM_SMEM_NPAIRS_STAGE_BYTES 8
#define SMEM_SMEM_NPAIRS_STRIDE 8
#define SMEM_SMEM_KVLEN_OFF 2824
#define SMEM_SMEM_KVLEN_STAGE_BYTES 8
#define SMEM_SMEM_KVLEN_STRIDE 8
#define SMEM_SMEM_QU_OFF 6144
#define SMEM_SMEM_QU_STAGE_BYTES 4096
#define SMEM_SMEM_QU_STRIDE 4096
#define SMEM_SMEM_QTOK_OFF 6144
#define SMEM_SMEM_QTOK_STAGE_BYTES 2048
#define SMEM_SMEM_QTOK_STRIDE 2048
#define SMEM_SMEM_P_OFF 22528
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_SCALE_OFF 55296
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 1024
#define SMEM_SMEM_RAW_OFF 63488
#define SMEM_SMEM_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_RAW_STRIDE 8192
#define SMEM_SMEM_V_OFF 129024
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_V_MN_OFF 129024
#define SMEM_SMEM_V_MN_STAGE_BYTES 32768
#define SMEM_SMEM_V_MN_STRIDE 32768
#define SMEM_SMEM_V13_OFF 63488
#define SMEM_SMEM_V13_STAGE_BYTES 16384
#define SMEM_SMEM_V13_STRIDE 16384
#define SMEM_SMEM_V14_OFF 2832
#define SMEM_SMEM_V14_STAGE_BYTES 256
#define SMEM_SMEM_V14_STRIDE 256
#define SMEM_TOTAL 227328
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_msa_nvfp4_decode_af2e689ccab0037b5cda(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ task_kind, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages, int q2k_token_stride, int q2k_head_stride)
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
    #define q_empty_addr (mbar_base + 16)
    #define meta_full_addr (mbar_base + 32)
    #define kv_raw_full_addr (mbar_base + 48)
    #define kv_raw_empty_addr (mbar_base + 304)
    #define k16_full_addr (mbar_base + 368)
    #define k16_empty_addr (mbar_base + 384)
    #define v_full_addr (mbar_base + 400)
    #define v_empty_addr (mbar_base + 424)
    #define s_full_addr (mbar_base + 448)
    #define s_empty_addr (mbar_base + 464)
    #define p_full_addr (mbar_base + 480)
    #define pv_done_addr (mbar_base + 496)
    #define decode_done_addr (mbar_base + 624)
    #define meta_empty_addr (mbar_base + 632)
    #define tmem_dead_addr (mbar_base + 648)
    #define part_ready_addr (mbar_base + 656)
    #define stat_ready_addr (mbar_base + 664)
    #define zone_dead_addr (mbar_base + 672)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* smem_f32 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_F32_OFF);
    const int smem_f32_addr = smem + SMEM_SMEM_F32_OFF;
    int* smem_page = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_PAGE_OFF);
    const int smem_page_addr = smem + SMEM_SMEM_PAGE_OFF;
    int* smem_bstart = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_BSTART_OFF);
    const int smem_bstart_addr = smem + SMEM_SMEM_BSTART_OFF;
    unsigned int* smem_mask = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_MASK_OFF);
    const int smem_mask_addr = smem + SMEM_SMEM_MASK_OFF;
    int* smem_npairs = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_NPAIRS_OFF);
    const int smem_npairs_addr = smem + SMEM_SMEM_NPAIRS_OFF;
    int* smem_kvlen = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_KVLEN_OFF);
    const int smem_kvlen_addr = smem + SMEM_SMEM_KVLEN_OFF;
    __nv_bfloat16* smem_qu = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QU_OFF);
    const int smem_qu_addr = smem + SMEM_SMEM_QU_OFF;
    __nv_bfloat16* smem_qtok = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QTOK_OFF);
    const int smem_qtok_addr = smem + SMEM_SMEM_QTOK_OFF;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P_OFF);
    const int smem_p_addr = smem + SMEM_SMEM_P_OFF;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SCALE_OFF);
    const int smem_scale_addr = smem + SMEM_SMEM_SCALE_OFF;
    uint8_t* smem_raw_1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_RAW_OFF);
    const int smem_raw_addr = smem + SMEM_SMEM_RAW_OFF;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    __nv_bfloat16* smem_v_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_MN_OFF);
    const int smem_v_mn_addr = smem + SMEM_SMEM_V_MN_OFF;
    float* smem_v13 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_V13_OFF);
    const int smem_v13_addr = smem + SMEM_SMEM_V13_OFF;
    float* smem_v14 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_V14_OFF);
    const int smem_v14_addr = smem + SMEM_SMEM_V14_OFF;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 85 barriers)
    // Mbarriers at smem_raw[0..680)

    if (warp == 0) {
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // meta_full: 2 barriers, init_count=1
        // kv_raw_full: 32 barriers, init_count=1
        // --- pipeline 'raw_ring' ---
        // kv_raw_empty: 8 barriers, init_count=4
        // k16_full: 2 barriers, init_count=4
        // k16_empty: 2 barriers, init_count=1
        // v_full: 3 barriers, init_count=4
        // v_empty: 3 barriers, init_count=1
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=4
        // p_full: 2 barriers, init_count=4
        // pv_done: 16 barriers, init_count=1
        // decode_done: 1 barriers, init_count=4
        // meta_empty: 2 barriers, init_count=4
        // tmem_dead: 1 barriers, init_count=4
        // part_ready: 1 barriers, init_count=1
        // stat_ready: 1 barriers, init_count=1
        // zone_dead: 1 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 0 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(30), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(26), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(21), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        uint32_t _mbarrier_init_count_0_64 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(18), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(14), "r"((uint32_t)(1)));
        if (lane < 21) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_0_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 680);
    if (warp == 0) {
        int _tmem_hold = smem + 680;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    // barrier.cluster.wait deferred (WarpConfig.cluster_init_wait_warps): role entries, or the schedule's ClusterSyncWait when the tuple is empty

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_k16 = taddr + 128;
    const int tmem_tmem_o = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // softmax_main
            unsigned int total_units_sm = total_q / seqlen_q * num_kv_heads;
            const int warp_in_role = warp % 4;
            const int token = warp_in_role * 32 + lane;
            const int tmem_lane_base = warp % 4 * 32 << 16;
            int item_stage = 0;
            int item_phase = 0;
            int s_stage = 0;
            int s_phase = 0;
            int p_stage = 0;
            float sv0[16];
            float sv1[16];
            #pragma unroll 1
            for (unsigned int work_idx = blockIdx.x / 4; work_idx < total_units_sm; work_idx += gridDim.x / 4) {
                int unit = work_idx;
                int request = unit / num_kv_heads;
                int kv_head = unit % num_kv_heads;
                int group_size = num_q_heads / num_kv_heads;
                mbarrier_wait(meta_full_addr + (item_stage) * 8, item_phase);
                int npairs = smem_npairs[item_stage];
                if (warp_in_role == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(part_ready_addr, 16384);
                        mbarrier_arrive_expect_tx(stat_ready_addr, 256);
                    }
                }
                int kv_len = smem_kvlen[item_stage];
                int meta_base = item_stage * 32;
                int item_par = item_stage;
                item_stage += 1;
                if (item_stage == 2) { item_stage = 0; item_phase ^= 1; }
                float origin[2];
                float neg_ms[2];
                int pending = 3;
                origin[0] = -CAKE_INF;
                neg_ms[0] = 0.0f;
                origin[1] = -CAKE_INF;
                neg_ms[1] = 0.0f;
                float lane_sum[32];
                lane_sum[0] = 0.0f;
                lane_sum[1] = 0.0f;
                lane_sum[2] = 0.0f;
                lane_sum[3] = 0.0f;
                lane_sum[4] = 0.0f;
                lane_sum[5] = 0.0f;
                lane_sum[6] = 0.0f;
                lane_sum[7] = 0.0f;
                lane_sum[8] = 0.0f;
                lane_sum[9] = 0.0f;
                lane_sum[10] = 0.0f;
                lane_sum[11] = 0.0f;
                lane_sum[12] = 0.0f;
                lane_sum[13] = 0.0f;
                lane_sum[14] = 0.0f;
                lane_sum[15] = 0.0f;
                lane_sum[16] = 0.0f;
                lane_sum[17] = 0.0f;
                lane_sum[18] = 0.0f;
                lane_sum[19] = 0.0f;
                lane_sum[20] = 0.0f;
                lane_sum[21] = 0.0f;
                lane_sum[22] = 0.0f;
                lane_sum[23] = 0.0f;
                lane_sum[24] = 0.0f;
                lane_sum[25] = 0.0f;
                lane_sum[26] = 0.0f;
                lane_sum[27] = 0.0f;
                lane_sum[28] = 0.0f;
                lane_sum[29] = 0.0f;
                lane_sum[30] = 0.0f;
                lane_sum[31] = 0.0f;
                int cq_base = kv_len - seqlen_q + 1;
                #pragma unroll 1
                for (int p = 0; p < 16; p++) {
                    if (npairs <= p) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage) * 8, s_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)(s_stage * 64) + (unsigned int)tmem_lane_base;
                    tmem_ld_x16(&sv0[0], s_addr);
                    tmem_ld_x16(&sv1[0], s_addr + 32);
                    int bs0 = smem_bstart[meta_base + 2 * p];
                    int bs1 = smem_bstart[meta_base + 2 * p + 1];
                    unsigned int mw0 = smem_mask[meta_base + 2 * p];
                    unsigned int mw1 = smem_mask[meta_base + 2 * p + 1];
                    int lim0 = kv_len - bs0;
                    if (lim0 > 128) {
                        lim0 = 128;
                    }
                    int lim1 = kv_len - bs1;
                    if (lim1 > 128) {
                        lim1 = 128;
                    }
                    int cq0 = cq_base - bs0;
                    int cq1 = cq_base - bs1;
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int v0 = lim0;
                    int c0 = cq0;
                    if (v0 > c0) {
                        v0 = c0;
                    }
                    if ((mw0 & 1) == 0) {
                        v0 = 0;
                    }
                    if (token >= v0) {
                        sv0[0] = -CAKE_INF;
                        sv0[1] = -CAKE_INF;
                        sv0[2] = -CAKE_INF;
                        sv0[3] = -CAKE_INF;
                        sv0[4] = -CAKE_INF;
                        sv0[5] = -CAKE_INF;
                        sv0[6] = -CAKE_INF;
                        sv0[7] = -CAKE_INF;
                        sv0[8] = -CAKE_INF;
                        sv0[9] = -CAKE_INF;
                        sv0[10] = -CAKE_INF;
                        sv0[11] = -CAKE_INF;
                        sv0[12] = -CAKE_INF;
                        sv0[13] = -CAKE_INF;
                        sv0[14] = -CAKE_INF;
                        sv0[15] = -CAKE_INF;
                    }
                    int v1 = lim1;
                    int c1 = cq1;
                    if (v1 > c1) {
                        v1 = c1;
                    }
                    if ((mw1 & 1) == 0) {
                        v1 = 0;
                    }
                    if (token >= v1) {
                        sv1[0] = -CAKE_INF;
                        sv1[1] = -CAKE_INF;
                        sv1[2] = -CAKE_INF;
                        sv1[3] = -CAKE_INF;
                        sv1[4] = -CAKE_INF;
                        sv1[5] = -CAKE_INF;
                        sv1[6] = -CAKE_INF;
                        sv1[7] = -CAKE_INF;
                        sv1[8] = -CAKE_INF;
                        sv1[9] = -CAKE_INF;
                        sv1[10] = -CAKE_INF;
                        sv1[11] = -CAKE_INF;
                        sv1[12] = -CAKE_INF;
                        sv1[13] = -CAKE_INF;
                        sv1[14] = -CAKE_INF;
                        sv1[15] = -CAKE_INF;
                    }
                    if ((pending & 1) == 1) {
                        float m0 = sv0[0];
                        float m1 = sv1[0];
                        float _max_0 = max_noftz(m0, sv0[1]);
                        m0 = _max_0;
                        float _max_1 = max_noftz(m1, sv1[1]);
                        m1 = _max_1;
                        float _max_2 = max_noftz(m0, sv0[2]);
                        m0 = _max_2;
                        float _max_3 = max_noftz(m1, sv1[2]);
                        m1 = _max_3;
                        float _max_4 = max_noftz(m0, sv0[3]);
                        m0 = _max_4;
                        float _max_5 = max_noftz(m1, sv1[3]);
                        m1 = _max_5;
                        float _max_6 = max_noftz(m0, sv0[4]);
                        m0 = _max_6;
                        float _max_7 = max_noftz(m1, sv1[4]);
                        m1 = _max_7;
                        float _max_8 = max_noftz(m0, sv0[5]);
                        m0 = _max_8;
                        float _max_9 = max_noftz(m1, sv1[5]);
                        m1 = _max_9;
                        float _max_10 = max_noftz(m0, sv0[6]);
                        m0 = _max_10;
                        float _max_11 = max_noftz(m1, sv1[6]);
                        m1 = _max_11;
                        float _max_12 = max_noftz(m0, sv0[7]);
                        m0 = _max_12;
                        float _max_13 = max_noftz(m1, sv1[7]);
                        m1 = _max_13;
                        float _max_14 = max_noftz(m0, sv0[8]);
                        m0 = _max_14;
                        float _max_15 = max_noftz(m1, sv1[8]);
                        m1 = _max_15;
                        float _max_16 = max_noftz(m0, sv0[9]);
                        m0 = _max_16;
                        float _max_17 = max_noftz(m1, sv1[9]);
                        m1 = _max_17;
                        float _max_18 = max_noftz(m0, sv0[10]);
                        m0 = _max_18;
                        float _max_19 = max_noftz(m1, sv1[10]);
                        m1 = _max_19;
                        float _max_20 = max_noftz(m0, sv0[11]);
                        m0 = _max_20;
                        float _max_21 = max_noftz(m1, sv1[11]);
                        m1 = _max_21;
                        float _max_22 = max_noftz(m0, sv0[12]);
                        m0 = _max_22;
                        float _max_23 = max_noftz(m1, sv1[12]);
                        m1 = _max_23;
                        float _max_24 = max_noftz(m0, sv0[13]);
                        m0 = _max_24;
                        float _max_25 = max_noftz(m1, sv1[13]);
                        m1 = _max_25;
                        float _max_26 = max_noftz(m0, sv0[14]);
                        m0 = _max_26;
                        float _max_27 = max_noftz(m1, sv1[14]);
                        m1 = _max_27;
                        float _max_28 = max_noftz(m0, sv0[15]);
                        m0 = _max_28;
                        float _max_29 = max_noftz(m1, sv1[15]);
                        m1 = _max_29;
                        float _warp_redux_f32_0;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(m0));
                        float w0 = _warp_redux_f32_0;
                        float _warp_redux_f32_1;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_1) : "f"(m1));
                        float w1 = _warp_redux_f32_1;
                        float cand = ((w0 > -CAKE_INF) ? w0 : w1);
                        if (cand > -CAKE_INF) {
                            origin[0] = cand;
                            neg_ms[0] = (-cand) * softmax_scale_log2;
                            pending = pending - 1;
                        }
                    }
                    if (p >= 2) {
                        mbarrier_wait(pv_done_addr + (p - 2) * 8, item_par);
                    }
                    int p0_base = smem_p_addr + (unsigned int)(p_stage * 2 * 8192);
                    int p1_base = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 8192);
                    unsigned int live0 = mw0 & 1;
                    if (live0 != 0) {
                        float _fma_0 = __fmaf_rn(sv0[0], softmax_scale_log2, neg_ms[0]);
                        float _exp2_0 = approx_exp2(_fma_0);
                        sv0[0] = _exp2_0;
                        float _fma_1 = __fmaf_rn(sv0[1], softmax_scale_log2, neg_ms[0]);
                        float _exp2_1 = approx_exp2(_fma_1);
                        sv0[1] = _exp2_1;
                        float _fma_2 = __fmaf_rn(sv0[2], softmax_scale_log2, neg_ms[0]);
                        float _exp2_2 = approx_exp2(_fma_2);
                        sv0[2] = _exp2_2;
                        float _fma_3 = __fmaf_rn(sv0[3], softmax_scale_log2, neg_ms[0]);
                        float _exp2_3 = approx_exp2(_fma_3);
                        sv0[3] = _exp2_3;
                        float _fma_4 = __fmaf_rn(sv0[4], softmax_scale_log2, neg_ms[0]);
                        float _exp2_4 = approx_exp2(_fma_4);
                        sv0[4] = _exp2_4;
                        float _fma_5 = __fmaf_rn(sv0[5], softmax_scale_log2, neg_ms[0]);
                        float _exp2_5 = approx_exp2(_fma_5);
                        sv0[5] = _exp2_5;
                        float _fma_6 = __fmaf_rn(sv0[6], softmax_scale_log2, neg_ms[0]);
                        float _exp2_6 = approx_exp2(_fma_6);
                        sv0[6] = _exp2_6;
                        float _fma_7 = __fmaf_rn(sv0[7], softmax_scale_log2, neg_ms[0]);
                        float _exp2_7 = approx_exp2(_fma_7);
                        sv0[7] = _exp2_7;
                        float _fma_8 = __fmaf_rn(sv0[8], softmax_scale_log2, neg_ms[0]);
                        float _exp2_8 = approx_exp2(_fma_8);
                        sv0[8] = _exp2_8;
                        float _fma_9 = __fmaf_rn(sv0[9], softmax_scale_log2, neg_ms[0]);
                        float _exp2_9 = approx_exp2(_fma_9);
                        sv0[9] = _exp2_9;
                        float _fma_10 = __fmaf_rn(sv0[10], softmax_scale_log2, neg_ms[0]);
                        float _exp2_10 = approx_exp2(_fma_10);
                        sv0[10] = _exp2_10;
                        float _fma_11 = __fmaf_rn(sv0[11], softmax_scale_log2, neg_ms[0]);
                        float _exp2_11 = approx_exp2(_fma_11);
                        sv0[11] = _exp2_11;
                        float _fma_12 = __fmaf_rn(sv0[12], softmax_scale_log2, neg_ms[0]);
                        float _exp2_12 = approx_exp2(_fma_12);
                        sv0[12] = _exp2_12;
                        float _fma_13 = __fmaf_rn(sv0[13], softmax_scale_log2, neg_ms[0]);
                        float _exp2_13 = approx_exp2(_fma_13);
                        sv0[13] = _exp2_13;
                        float _fma_14 = __fmaf_rn(sv0[14], softmax_scale_log2, neg_ms[0]);
                        float _exp2_14 = approx_exp2(_fma_14);
                        sv0[14] = _exp2_14;
                        float _fma_15 = __fmaf_rn(sv0[15], softmax_scale_log2, neg_ms[0]);
                        float _exp2_15 = approx_exp2(_fma_15);
                        sv0[15] = _exp2_15;
                        unsigned int x[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv0[_lp*2 + 0], sv0[_lp*2+1 + 0]));
                            x[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_0[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(x[0]), "f"(lane_sum[0]), "f"(lane_sum[1]));
                        lane_sum[0] = _bf16x2_add_f32_0[0];
                        lane_sum[1] = _bf16x2_add_f32_0[1];
                        float _bf16x2_add_f32_1[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(x[1]), "f"(lane_sum[2]), "f"(lane_sum[3]));
                        lane_sum[2] = _bf16x2_add_f32_1[0];
                        lane_sum[3] = _bf16x2_add_f32_1[1];
                        float _bf16x2_add_f32_2[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_2[0]), "=&f"(_bf16x2_add_f32_2[1]) : "r"(x[2]), "f"(lane_sum[4]), "f"(lane_sum[5]));
                        lane_sum[4] = _bf16x2_add_f32_2[0];
                        lane_sum[5] = _bf16x2_add_f32_2[1];
                        float _bf16x2_add_f32_3[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_3[0]), "=&f"(_bf16x2_add_f32_3[1]) : "r"(x[3]), "f"(lane_sum[6]), "f"(lane_sum[7]));
                        lane_sum[6] = _bf16x2_add_f32_3[0];
                        lane_sum[7] = _bf16x2_add_f32_3[1];
                        float _bf16x2_add_f32_4[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_4[0]), "=&f"(_bf16x2_add_f32_4[1]) : "r"(x[4]), "f"(lane_sum[8]), "f"(lane_sum[9]));
                        lane_sum[8] = _bf16x2_add_f32_4[0];
                        lane_sum[9] = _bf16x2_add_f32_4[1];
                        float _bf16x2_add_f32_5[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_5[0]), "=&f"(_bf16x2_add_f32_5[1]) : "r"(x[5]), "f"(lane_sum[10]), "f"(lane_sum[11]));
                        lane_sum[10] = _bf16x2_add_f32_5[0];
                        lane_sum[11] = _bf16x2_add_f32_5[1];
                        float _bf16x2_add_f32_6[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_6[0]), "=&f"(_bf16x2_add_f32_6[1]) : "r"(x[6]), "f"(lane_sum[12]), "f"(lane_sum[13]));
                        lane_sum[12] = _bf16x2_add_f32_6[0];
                        lane_sum[13] = _bf16x2_add_f32_6[1];
                        float _bf16x2_add_f32_7[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_7[0]), "=&f"(_bf16x2_add_f32_7[1]) : "r"(x[7]), "f"(lane_sum[14]), "f"(lane_sum[15]));
                        lane_sum[14] = _bf16x2_add_f32_7[0];
                        lane_sum[15] = _bf16x2_add_f32_7[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 64 ^ (token * 64 >> 7 & 3) << 4))), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 64 + 16 ^ (token * 64 + 16 >> 7 & 3) << 4))), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]) : "memory");
                    } else {
                        unsigned int zero = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 64 ^ (token * 64 >> 7 & 3) << 4))), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 64 + 16 ^ (token * 64 + 16 >> 7 & 3) << 4))), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                    }
                    unsigned int live1 = mw1 & 1;
                    if (live1 != 0) {
                        float _fma_16 = __fmaf_rn(sv1[0], softmax_scale_log2, neg_ms[0]);
                        float _exp2_16 = approx_exp2(_fma_16);
                        sv1[0] = _exp2_16;
                        float _fma_17 = __fmaf_rn(sv1[1], softmax_scale_log2, neg_ms[0]);
                        float _exp2_17 = approx_exp2(_fma_17);
                        sv1[1] = _exp2_17;
                        float _fma_18 = __fmaf_rn(sv1[2], softmax_scale_log2, neg_ms[0]);
                        float _exp2_18 = approx_exp2(_fma_18);
                        sv1[2] = _exp2_18;
                        float _fma_19 = __fmaf_rn(sv1[3], softmax_scale_log2, neg_ms[0]);
                        float _exp2_19 = approx_exp2(_fma_19);
                        sv1[3] = _exp2_19;
                        float _fma_20 = __fmaf_rn(sv1[4], softmax_scale_log2, neg_ms[0]);
                        float _exp2_20 = approx_exp2(_fma_20);
                        sv1[4] = _exp2_20;
                        float _fma_21 = __fmaf_rn(sv1[5], softmax_scale_log2, neg_ms[0]);
                        float _exp2_21 = approx_exp2(_fma_21);
                        sv1[5] = _exp2_21;
                        float _fma_22 = __fmaf_rn(sv1[6], softmax_scale_log2, neg_ms[0]);
                        float _exp2_22 = approx_exp2(_fma_22);
                        sv1[6] = _exp2_22;
                        float _fma_23 = __fmaf_rn(sv1[7], softmax_scale_log2, neg_ms[0]);
                        float _exp2_23 = approx_exp2(_fma_23);
                        sv1[7] = _exp2_23;
                        float _fma_24 = __fmaf_rn(sv1[8], softmax_scale_log2, neg_ms[0]);
                        float _exp2_24 = approx_exp2(_fma_24);
                        sv1[8] = _exp2_24;
                        float _fma_25 = __fmaf_rn(sv1[9], softmax_scale_log2, neg_ms[0]);
                        float _exp2_25 = approx_exp2(_fma_25);
                        sv1[9] = _exp2_25;
                        float _fma_26 = __fmaf_rn(sv1[10], softmax_scale_log2, neg_ms[0]);
                        float _exp2_26 = approx_exp2(_fma_26);
                        sv1[10] = _exp2_26;
                        float _fma_27 = __fmaf_rn(sv1[11], softmax_scale_log2, neg_ms[0]);
                        float _exp2_27 = approx_exp2(_fma_27);
                        sv1[11] = _exp2_27;
                        float _fma_28 = __fmaf_rn(sv1[12], softmax_scale_log2, neg_ms[0]);
                        float _exp2_28 = approx_exp2(_fma_28);
                        sv1[12] = _exp2_28;
                        float _fma_29 = __fmaf_rn(sv1[13], softmax_scale_log2, neg_ms[0]);
                        float _exp2_29 = approx_exp2(_fma_29);
                        sv1[13] = _exp2_29;
                        float _fma_30 = __fmaf_rn(sv1[14], softmax_scale_log2, neg_ms[0]);
                        float _exp2_30 = approx_exp2(_fma_30);
                        sv1[14] = _exp2_30;
                        float _fma_31 = __fmaf_rn(sv1[15], softmax_scale_log2, neg_ms[0]);
                        float _exp2_31 = approx_exp2(_fma_31);
                        sv1[15] = _exp2_31;
                        unsigned int x_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv1[_lp*2 + 0], sv1[_lp*2+1 + 0]));
                            x_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_8[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_8[0]), "=&f"(_bf16x2_add_f32_8[1]) : "r"(x_1[0]), "f"(lane_sum[0]), "f"(lane_sum[1]));
                        lane_sum[0] = _bf16x2_add_f32_8[0];
                        lane_sum[1] = _bf16x2_add_f32_8[1];
                        float _bf16x2_add_f32_9[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_9[0]), "=&f"(_bf16x2_add_f32_9[1]) : "r"(x_1[1]), "f"(lane_sum[2]), "f"(lane_sum[3]));
                        lane_sum[2] = _bf16x2_add_f32_9[0];
                        lane_sum[3] = _bf16x2_add_f32_9[1];
                        float _bf16x2_add_f32_10[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_10[0]), "=&f"(_bf16x2_add_f32_10[1]) : "r"(x_1[2]), "f"(lane_sum[4]), "f"(lane_sum[5]));
                        lane_sum[4] = _bf16x2_add_f32_10[0];
                        lane_sum[5] = _bf16x2_add_f32_10[1];
                        float _bf16x2_add_f32_11[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_11[0]), "=&f"(_bf16x2_add_f32_11[1]) : "r"(x_1[3]), "f"(lane_sum[6]), "f"(lane_sum[7]));
                        lane_sum[6] = _bf16x2_add_f32_11[0];
                        lane_sum[7] = _bf16x2_add_f32_11[1];
                        float _bf16x2_add_f32_12[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_12[0]), "=&f"(_bf16x2_add_f32_12[1]) : "r"(x_1[4]), "f"(lane_sum[8]), "f"(lane_sum[9]));
                        lane_sum[8] = _bf16x2_add_f32_12[0];
                        lane_sum[9] = _bf16x2_add_f32_12[1];
                        float _bf16x2_add_f32_13[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_13[0]), "=&f"(_bf16x2_add_f32_13[1]) : "r"(x_1[5]), "f"(lane_sum[10]), "f"(lane_sum[11]));
                        lane_sum[10] = _bf16x2_add_f32_13[0];
                        lane_sum[11] = _bf16x2_add_f32_13[1];
                        float _bf16x2_add_f32_14[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_14[0]), "=&f"(_bf16x2_add_f32_14[1]) : "r"(x_1[6]), "f"(lane_sum[12]), "f"(lane_sum[13]));
                        lane_sum[12] = _bf16x2_add_f32_14[0];
                        lane_sum[13] = _bf16x2_add_f32_14[1];
                        float _bf16x2_add_f32_15[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_15[0]), "=&f"(_bf16x2_add_f32_15[1]) : "r"(x_1[7]), "f"(lane_sum[14]), "f"(lane_sum[15]));
                        lane_sum[14] = _bf16x2_add_f32_15[0];
                        lane_sum[15] = _bf16x2_add_f32_15[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 64 ^ (token * 64 >> 7 & 3) << 4))), "r"(x_1[0]), "r"(x_1[1]), "r"(x_1[2]), "r"(x_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 64 + 16 ^ (token * 64 + 16 >> 7 & 3) << 4))), "r"(x_1[4]), "r"(x_1[5]), "r"(x_1[6]), "r"(x_1[7]) : "memory");
                    } else {
                        unsigned int zero_1 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 64 ^ (token * 64 >> 7 & 3) << 4))), "r"(zero_1), "r"(zero_1), "r"(zero_1), "r"(zero_1) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 64 + 16 ^ (token * 64 + 16 >> 7 & 3) << 4))), "r"(zero_1), "r"(zero_1), "r"(zero_1), "r"(zero_1) : "memory");
                    }
                    tmem_ld_x16(&sv0[0], s_addr + 16);
                    tmem_ld_x16(&sv1[0], s_addr + 32 + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                    }
                    s_stage += 1;
                    if (s_stage == 2) { s_stage = 0; s_phase ^= 1; }
                    int v0_0 = lim0;
                    int c0_1 = cq0 + 1;
                    if (v0_0 > c0_1) {
                        v0_0 = c0_1;
                    }
                    if ((mw0 >> 1 & 1) == 0) {
                        v0_0 = 0;
                    }
                    if (token >= v0_0) {
                        sv0[0] = -CAKE_INF;
                        sv0[1] = -CAKE_INF;
                        sv0[2] = -CAKE_INF;
                        sv0[3] = -CAKE_INF;
                        sv0[4] = -CAKE_INF;
                        sv0[5] = -CAKE_INF;
                        sv0[6] = -CAKE_INF;
                        sv0[7] = -CAKE_INF;
                        sv0[8] = -CAKE_INF;
                        sv0[9] = -CAKE_INF;
                        sv0[10] = -CAKE_INF;
                        sv0[11] = -CAKE_INF;
                        sv0[12] = -CAKE_INF;
                        sv0[13] = -CAKE_INF;
                        sv0[14] = -CAKE_INF;
                        sv0[15] = -CAKE_INF;
                    }
                    int v1_2 = lim1;
                    int c1_3 = cq1 + 1;
                    if (v1_2 > c1_3) {
                        v1_2 = c1_3;
                    }
                    if ((mw1 >> 1 & 1) == 0) {
                        v1_2 = 0;
                    }
                    if (token >= v1_2) {
                        sv1[0] = -CAKE_INF;
                        sv1[1] = -CAKE_INF;
                        sv1[2] = -CAKE_INF;
                        sv1[3] = -CAKE_INF;
                        sv1[4] = -CAKE_INF;
                        sv1[5] = -CAKE_INF;
                        sv1[6] = -CAKE_INF;
                        sv1[7] = -CAKE_INF;
                        sv1[8] = -CAKE_INF;
                        sv1[9] = -CAKE_INF;
                        sv1[10] = -CAKE_INF;
                        sv1[11] = -CAKE_INF;
                        sv1[12] = -CAKE_INF;
                        sv1[13] = -CAKE_INF;
                        sv1[14] = -CAKE_INF;
                        sv1[15] = -CAKE_INF;
                    }
                    if ((pending >> 1 & 1) == 1) {
                        float m0_1 = sv0[0];
                        float m1_1 = sv1[0];
                        float _max_30 = max_noftz(m0_1, sv0[1]);
                        m0_1 = _max_30;
                        float _max_31 = max_noftz(m1_1, sv1[1]);
                        m1_1 = _max_31;
                        float _max_32 = max_noftz(m0_1, sv0[2]);
                        m0_1 = _max_32;
                        float _max_33 = max_noftz(m1_1, sv1[2]);
                        m1_1 = _max_33;
                        float _max_34 = max_noftz(m0_1, sv0[3]);
                        m0_1 = _max_34;
                        float _max_35 = max_noftz(m1_1, sv1[3]);
                        m1_1 = _max_35;
                        float _max_36 = max_noftz(m0_1, sv0[4]);
                        m0_1 = _max_36;
                        float _max_37 = max_noftz(m1_1, sv1[4]);
                        m1_1 = _max_37;
                        float _max_38 = max_noftz(m0_1, sv0[5]);
                        m0_1 = _max_38;
                        float _max_39 = max_noftz(m1_1, sv1[5]);
                        m1_1 = _max_39;
                        float _max_40 = max_noftz(m0_1, sv0[6]);
                        m0_1 = _max_40;
                        float _max_41 = max_noftz(m1_1, sv1[6]);
                        m1_1 = _max_41;
                        float _max_42 = max_noftz(m0_1, sv0[7]);
                        m0_1 = _max_42;
                        float _max_43 = max_noftz(m1_1, sv1[7]);
                        m1_1 = _max_43;
                        float _max_44 = max_noftz(m0_1, sv0[8]);
                        m0_1 = _max_44;
                        float _max_45 = max_noftz(m1_1, sv1[8]);
                        m1_1 = _max_45;
                        float _max_46 = max_noftz(m0_1, sv0[9]);
                        m0_1 = _max_46;
                        float _max_47 = max_noftz(m1_1, sv1[9]);
                        m1_1 = _max_47;
                        float _max_48 = max_noftz(m0_1, sv0[10]);
                        m0_1 = _max_48;
                        float _max_49 = max_noftz(m1_1, sv1[10]);
                        m1_1 = _max_49;
                        float _max_50 = max_noftz(m0_1, sv0[11]);
                        m0_1 = _max_50;
                        float _max_51 = max_noftz(m1_1, sv1[11]);
                        m1_1 = _max_51;
                        float _max_52 = max_noftz(m0_1, sv0[12]);
                        m0_1 = _max_52;
                        float _max_53 = max_noftz(m1_1, sv1[12]);
                        m1_1 = _max_53;
                        float _max_54 = max_noftz(m0_1, sv0[13]);
                        m0_1 = _max_54;
                        float _max_55 = max_noftz(m1_1, sv1[13]);
                        m1_1 = _max_55;
                        float _max_56 = max_noftz(m0_1, sv0[14]);
                        m0_1 = _max_56;
                        float _max_57 = max_noftz(m1_1, sv1[14]);
                        m1_1 = _max_57;
                        float _max_58 = max_noftz(m0_1, sv0[15]);
                        m0_1 = _max_58;
                        float _max_59 = max_noftz(m1_1, sv1[15]);
                        m1_1 = _max_59;
                        float _warp_redux_f32_2;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_2) : "f"(m0_1));
                        float w0_1 = _warp_redux_f32_2;
                        float _warp_redux_f32_3;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_3) : "f"(m1_1));
                        float w1_1 = _warp_redux_f32_3;
                        float cand_1 = ((w0_1 > -CAKE_INF) ? w0_1 : w1_1);
                        if (cand_1 > -CAKE_INF) {
                            origin[1] = cand_1;
                            neg_ms[1] = (-cand_1) * softmax_scale_log2;
                            pending = pending - 2;
                        }
                    }
                    int p0_base_4 = smem_p_addr + (unsigned int)(p_stage * 2 * 8192);
                    int p1_base_5 = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 8192);
                    unsigned int live0_6 = mw0 >> 1 & 1;
                    if (live0_6 != 0) {
                        float _fma_32 = __fmaf_rn(sv0[0], softmax_scale_log2, neg_ms[1]);
                        float _exp2_32 = approx_exp2(_fma_32);
                        sv0[0] = _exp2_32;
                        float _fma_33 = __fmaf_rn(sv0[1], softmax_scale_log2, neg_ms[1]);
                        float _exp2_33 = approx_exp2(_fma_33);
                        sv0[1] = _exp2_33;
                        float _fma_34 = __fmaf_rn(sv0[2], softmax_scale_log2, neg_ms[1]);
                        float _exp2_34 = approx_exp2(_fma_34);
                        sv0[2] = _exp2_34;
                        float _fma_35 = __fmaf_rn(sv0[3], softmax_scale_log2, neg_ms[1]);
                        float _exp2_35 = approx_exp2(_fma_35);
                        sv0[3] = _exp2_35;
                        float _fma_36 = __fmaf_rn(sv0[4], softmax_scale_log2, neg_ms[1]);
                        float _exp2_36 = approx_exp2(_fma_36);
                        sv0[4] = _exp2_36;
                        float _fma_37 = __fmaf_rn(sv0[5], softmax_scale_log2, neg_ms[1]);
                        float _exp2_37 = approx_exp2(_fma_37);
                        sv0[5] = _exp2_37;
                        float _fma_38 = __fmaf_rn(sv0[6], softmax_scale_log2, neg_ms[1]);
                        float _exp2_38 = approx_exp2(_fma_38);
                        sv0[6] = _exp2_38;
                        float _fma_39 = __fmaf_rn(sv0[7], softmax_scale_log2, neg_ms[1]);
                        float _exp2_39 = approx_exp2(_fma_39);
                        sv0[7] = _exp2_39;
                        float _fma_40 = __fmaf_rn(sv0[8], softmax_scale_log2, neg_ms[1]);
                        float _exp2_40 = approx_exp2(_fma_40);
                        sv0[8] = _exp2_40;
                        float _fma_41 = __fmaf_rn(sv0[9], softmax_scale_log2, neg_ms[1]);
                        float _exp2_41 = approx_exp2(_fma_41);
                        sv0[9] = _exp2_41;
                        float _fma_42 = __fmaf_rn(sv0[10], softmax_scale_log2, neg_ms[1]);
                        float _exp2_42 = approx_exp2(_fma_42);
                        sv0[10] = _exp2_42;
                        float _fma_43 = __fmaf_rn(sv0[11], softmax_scale_log2, neg_ms[1]);
                        float _exp2_43 = approx_exp2(_fma_43);
                        sv0[11] = _exp2_43;
                        float _fma_44 = __fmaf_rn(sv0[12], softmax_scale_log2, neg_ms[1]);
                        float _exp2_44 = approx_exp2(_fma_44);
                        sv0[12] = _exp2_44;
                        float _fma_45 = __fmaf_rn(sv0[13], softmax_scale_log2, neg_ms[1]);
                        float _exp2_45 = approx_exp2(_fma_45);
                        sv0[13] = _exp2_45;
                        float _fma_46 = __fmaf_rn(sv0[14], softmax_scale_log2, neg_ms[1]);
                        float _exp2_46 = approx_exp2(_fma_46);
                        sv0[14] = _exp2_46;
                        float _fma_47 = __fmaf_rn(sv0[15], softmax_scale_log2, neg_ms[1]);
                        float _exp2_47 = approx_exp2(_fma_47);
                        sv0[15] = _exp2_47;
                        unsigned int x_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv0[_lp*2 + 0], sv0[_lp*2+1 + 0]));
                            x_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_16[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_16[0]), "=&f"(_bf16x2_add_f32_16[1]) : "r"(x_2[0]), "f"(lane_sum[16]), "f"(lane_sum[17]));
                        lane_sum[16] = _bf16x2_add_f32_16[0];
                        lane_sum[17] = _bf16x2_add_f32_16[1];
                        float _bf16x2_add_f32_17[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_17[0]), "=&f"(_bf16x2_add_f32_17[1]) : "r"(x_2[1]), "f"(lane_sum[18]), "f"(lane_sum[19]));
                        lane_sum[18] = _bf16x2_add_f32_17[0];
                        lane_sum[19] = _bf16x2_add_f32_17[1];
                        float _bf16x2_add_f32_18[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_18[0]), "=&f"(_bf16x2_add_f32_18[1]) : "r"(x_2[2]), "f"(lane_sum[20]), "f"(lane_sum[21]));
                        lane_sum[20] = _bf16x2_add_f32_18[0];
                        lane_sum[21] = _bf16x2_add_f32_18[1];
                        float _bf16x2_add_f32_19[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_19[0]), "=&f"(_bf16x2_add_f32_19[1]) : "r"(x_2[3]), "f"(lane_sum[22]), "f"(lane_sum[23]));
                        lane_sum[22] = _bf16x2_add_f32_19[0];
                        lane_sum[23] = _bf16x2_add_f32_19[1];
                        float _bf16x2_add_f32_20[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_20[0]), "=&f"(_bf16x2_add_f32_20[1]) : "r"(x_2[4]), "f"(lane_sum[24]), "f"(lane_sum[25]));
                        lane_sum[24] = _bf16x2_add_f32_20[0];
                        lane_sum[25] = _bf16x2_add_f32_20[1];
                        float _bf16x2_add_f32_21[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_21[0]), "=&f"(_bf16x2_add_f32_21[1]) : "r"(x_2[5]), "f"(lane_sum[26]), "f"(lane_sum[27]));
                        lane_sum[26] = _bf16x2_add_f32_21[0];
                        lane_sum[27] = _bf16x2_add_f32_21[1];
                        float _bf16x2_add_f32_22[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_22[0]), "=&f"(_bf16x2_add_f32_22[1]) : "r"(x_2[6]), "f"(lane_sum[28]), "f"(lane_sum[29]));
                        lane_sum[28] = _bf16x2_add_f32_22[0];
                        lane_sum[29] = _bf16x2_add_f32_22[1];
                        float _bf16x2_add_f32_23[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_23[0]), "=&f"(_bf16x2_add_f32_23[1]) : "r"(x_2[7]), "f"(lane_sum[30]), "f"(lane_sum[31]));
                        lane_sum[30] = _bf16x2_add_f32_23[0];
                        lane_sum[31] = _bf16x2_add_f32_23[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 64 + 32 ^ (token * 64 + 32 >> 7 & 3) << 4))), "r"(x_2[0]), "r"(x_2[1]), "r"(x_2[2]), "r"(x_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 64 + 48 ^ (token * 64 + 48 >> 7 & 3) << 4))), "r"(x_2[4]), "r"(x_2[5]), "r"(x_2[6]), "r"(x_2[7]) : "memory");
                    } else {
                        unsigned int zero_2 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 64 + 32 ^ (token * 64 + 32 >> 7 & 3) << 4))), "r"(zero_2), "r"(zero_2), "r"(zero_2), "r"(zero_2) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 64 + 48 ^ (token * 64 + 48 >> 7 & 3) << 4))), "r"(zero_2), "r"(zero_2), "r"(zero_2), "r"(zero_2) : "memory");
                    }
                    unsigned int live1_7 = mw1 >> 1 & 1;
                    if (live1_7 != 0) {
                        float _fma_48 = __fmaf_rn(sv1[0], softmax_scale_log2, neg_ms[1]);
                        float _exp2_48 = approx_exp2(_fma_48);
                        sv1[0] = _exp2_48;
                        float _fma_49 = __fmaf_rn(sv1[1], softmax_scale_log2, neg_ms[1]);
                        float _exp2_49 = approx_exp2(_fma_49);
                        sv1[1] = _exp2_49;
                        float _fma_50 = __fmaf_rn(sv1[2], softmax_scale_log2, neg_ms[1]);
                        float _exp2_50 = approx_exp2(_fma_50);
                        sv1[2] = _exp2_50;
                        float _fma_51 = __fmaf_rn(sv1[3], softmax_scale_log2, neg_ms[1]);
                        float _exp2_51 = approx_exp2(_fma_51);
                        sv1[3] = _exp2_51;
                        float _fma_52 = __fmaf_rn(sv1[4], softmax_scale_log2, neg_ms[1]);
                        float _exp2_52 = approx_exp2(_fma_52);
                        sv1[4] = _exp2_52;
                        float _fma_53 = __fmaf_rn(sv1[5], softmax_scale_log2, neg_ms[1]);
                        float _exp2_53 = approx_exp2(_fma_53);
                        sv1[5] = _exp2_53;
                        float _fma_54 = __fmaf_rn(sv1[6], softmax_scale_log2, neg_ms[1]);
                        float _exp2_54 = approx_exp2(_fma_54);
                        sv1[6] = _exp2_54;
                        float _fma_55 = __fmaf_rn(sv1[7], softmax_scale_log2, neg_ms[1]);
                        float _exp2_55 = approx_exp2(_fma_55);
                        sv1[7] = _exp2_55;
                        float _fma_56 = __fmaf_rn(sv1[8], softmax_scale_log2, neg_ms[1]);
                        float _exp2_56 = approx_exp2(_fma_56);
                        sv1[8] = _exp2_56;
                        float _fma_57 = __fmaf_rn(sv1[9], softmax_scale_log2, neg_ms[1]);
                        float _exp2_57 = approx_exp2(_fma_57);
                        sv1[9] = _exp2_57;
                        float _fma_58 = __fmaf_rn(sv1[10], softmax_scale_log2, neg_ms[1]);
                        float _exp2_58 = approx_exp2(_fma_58);
                        sv1[10] = _exp2_58;
                        float _fma_59 = __fmaf_rn(sv1[11], softmax_scale_log2, neg_ms[1]);
                        float _exp2_59 = approx_exp2(_fma_59);
                        sv1[11] = _exp2_59;
                        float _fma_60 = __fmaf_rn(sv1[12], softmax_scale_log2, neg_ms[1]);
                        float _exp2_60 = approx_exp2(_fma_60);
                        sv1[12] = _exp2_60;
                        float _fma_61 = __fmaf_rn(sv1[13], softmax_scale_log2, neg_ms[1]);
                        float _exp2_61 = approx_exp2(_fma_61);
                        sv1[13] = _exp2_61;
                        float _fma_62 = __fmaf_rn(sv1[14], softmax_scale_log2, neg_ms[1]);
                        float _exp2_62 = approx_exp2(_fma_62);
                        sv1[14] = _exp2_62;
                        float _fma_63 = __fmaf_rn(sv1[15], softmax_scale_log2, neg_ms[1]);
                        float _exp2_63 = approx_exp2(_fma_63);
                        sv1[15] = _exp2_63;
                        unsigned int x_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv1[_lp*2 + 0], sv1[_lp*2+1 + 0]));
                            x_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_24[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_24[0]), "=&f"(_bf16x2_add_f32_24[1]) : "r"(x_3[0]), "f"(lane_sum[16]), "f"(lane_sum[17]));
                        lane_sum[16] = _bf16x2_add_f32_24[0];
                        lane_sum[17] = _bf16x2_add_f32_24[1];
                        float _bf16x2_add_f32_25[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_25[0]), "=&f"(_bf16x2_add_f32_25[1]) : "r"(x_3[1]), "f"(lane_sum[18]), "f"(lane_sum[19]));
                        lane_sum[18] = _bf16x2_add_f32_25[0];
                        lane_sum[19] = _bf16x2_add_f32_25[1];
                        float _bf16x2_add_f32_26[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_26[0]), "=&f"(_bf16x2_add_f32_26[1]) : "r"(x_3[2]), "f"(lane_sum[20]), "f"(lane_sum[21]));
                        lane_sum[20] = _bf16x2_add_f32_26[0];
                        lane_sum[21] = _bf16x2_add_f32_26[1];
                        float _bf16x2_add_f32_27[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_27[0]), "=&f"(_bf16x2_add_f32_27[1]) : "r"(x_3[3]), "f"(lane_sum[22]), "f"(lane_sum[23]));
                        lane_sum[22] = _bf16x2_add_f32_27[0];
                        lane_sum[23] = _bf16x2_add_f32_27[1];
                        float _bf16x2_add_f32_28[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_28[0]), "=&f"(_bf16x2_add_f32_28[1]) : "r"(x_3[4]), "f"(lane_sum[24]), "f"(lane_sum[25]));
                        lane_sum[24] = _bf16x2_add_f32_28[0];
                        lane_sum[25] = _bf16x2_add_f32_28[1];
                        float _bf16x2_add_f32_29[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_29[0]), "=&f"(_bf16x2_add_f32_29[1]) : "r"(x_3[5]), "f"(lane_sum[26]), "f"(lane_sum[27]));
                        lane_sum[26] = _bf16x2_add_f32_29[0];
                        lane_sum[27] = _bf16x2_add_f32_29[1];
                        float _bf16x2_add_f32_30[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_30[0]), "=&f"(_bf16x2_add_f32_30[1]) : "r"(x_3[6]), "f"(lane_sum[28]), "f"(lane_sum[29]));
                        lane_sum[28] = _bf16x2_add_f32_30[0];
                        lane_sum[29] = _bf16x2_add_f32_30[1];
                        float _bf16x2_add_f32_31[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_31[0]), "=&f"(_bf16x2_add_f32_31[1]) : "r"(x_3[7]), "f"(lane_sum[30]), "f"(lane_sum[31]));
                        lane_sum[30] = _bf16x2_add_f32_31[0];
                        lane_sum[31] = _bf16x2_add_f32_31[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 64 + 32 ^ (token * 64 + 32 >> 7 & 3) << 4))), "r"(x_3[0]), "r"(x_3[1]), "r"(x_3[2]), "r"(x_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 64 + 48 ^ (token * 64 + 48 >> 7 & 3) << 4))), "r"(x_3[4]), "r"(x_3[5]), "r"(x_3[6]), "r"(x_3[7]) : "memory");
                    } else {
                        unsigned int zero_3 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 64 + 32 ^ (token * 64 + 32 >> 7 & 3) << 4))), "r"(zero_3), "r"(zero_3), "r"(zero_3), "r"(zero_3) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 64 + 48 ^ (token * 64 + 48 >> 7 & 3) << 4))), "r"(zero_3), "r"(zero_3), "r"(zero_3), "r"(zero_3) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    }
                    p_stage += 1;
                    if (p_stage == 2) { p_stage = 0; }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + (item_par) * 8);
                }
                int b4 = lane / 16 % 2;
                int b3 = lane / 8 % 2;
                int b2 = lane / 4 % 2;
                int b1 = lane / 2 % 2;
                float r8[8];
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    float snd8 = ((b4 == 0) ? lane_sum[c + 8] : lane_sum[c]);
                    float kp8 = ((b4 == 0) ? lane_sum[c] : lane_sum[c + 8]);
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, snd8, 16);
                    r8[c] = kp8 + _shfl_xor_0;
                }
                float r4[4];
                #pragma unroll
                for (int c_1 = 0; c_1 < 4; c_1++) {
                    float snd4 = ((b3 == 0) ? r8[c_1 + 4] : r8[c_1]);
                    float kp4 = ((b3 == 0) ? r8[c_1] : r8[c_1 + 4]);
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, snd4, 8);
                    r4[c_1] = kp4 + _shfl_xor_1;
                }
                float r2[2];
                #pragma unroll
                for (int c_2 = 0; c_2 < 2; c_2++) {
                    float snd2 = ((b2 == 0) ? r4[c_2 + 2] : r4[c_2]);
                    float kp2 = ((b2 == 0) ? r4[c_2] : r4[c_2 + 2]);
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, snd2, 4);
                    r2[c_2] = kp2 + _shfl_xor_2;
                }
                float snd1 = ((b1 == 0) ? r2[1] : r2[0]);
                float kp1 = ((b1 == 0) ? r2[0] : r2[1]);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, snd1, 2);
                float r1 = kp1 + _shfl_xor_3;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, r1, 1);
                r1 = r1 + _shfl_xor_4;
                float head_sum_w = r1;
                int slot_base = warp_in_role * 2 * 32;
                if (lane % 2 == 0) {
                    smem_f32[slot_base + lane / 2] = head_sum_w;
                }
                if (lane == 0) {
                    smem_f32[slot_base + 16] = origin[0];
                }
                int b4_0 = lane / 16 % 2;
                int b3_1 = lane / 8 % 2;
                int b2_2 = lane / 4 % 2;
                int b1_3 = lane / 2 % 2;
                float r8_4[8];
                #pragma unroll
                for (int c_3 = 0; c_3 < 8; c_3++) {
                    float snd8_1 = ((b4_0 == 0) ? lane_sum[16 + (c_3 + 8)] : lane_sum[16 + c_3]);
                    float kp8_1 = ((b4_0 == 0) ? lane_sum[16 + c_3] : lane_sum[16 + (c_3 + 8)]);
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, snd8_1, 16);
                    r8_4[c_3] = kp8_1 + _shfl_xor_5;
                }
                float r4_5[4];
                #pragma unroll
                for (int c_4 = 0; c_4 < 4; c_4++) {
                    float snd4_1 = ((b3_1 == 0) ? r8_4[c_4 + 4] : r8_4[c_4]);
                    float kp4_1 = ((b3_1 == 0) ? r8_4[c_4] : r8_4[c_4 + 4]);
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, snd4_1, 8);
                    r4_5[c_4] = kp4_1 + _shfl_xor_6;
                }
                float r2_6[2];
                #pragma unroll
                for (int c_5 = 0; c_5 < 2; c_5++) {
                    float snd2_1 = ((b2_2 == 0) ? r4_5[c_5 + 2] : r4_5[c_5]);
                    float kp2_1 = ((b2_2 == 0) ? r4_5[c_5] : r4_5[c_5 + 2]);
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, snd2_1, 4);
                    r2_6[c_5] = kp2_1 + _shfl_xor_7;
                }
                float snd1_7 = ((b1_3 == 0) ? r2_6[1] : r2_6[0]);
                float kp1_8 = ((b1_3 == 0) ? r2_6[0] : r2_6[1]);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, snd1_7, 2);
                float r1_9 = kp1_8 + _shfl_xor_8;
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, r1_9, 1);
                r1_9 = r1_9 + _shfl_xor_9;
                float head_sum_w_10 = r1_9;
                int slot_base_11 = (warp_in_role * 2 + 1) * 32;
                if (lane % 2 == 0) {
                    smem_f32[slot_base_11 + lane / 2] = head_sum_w_10;
                }
                if (lane == 0) {
                    smem_f32[slot_base_11 + 16] = origin[1];
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                float m_ab[2];
                float head_sum[2];
                float wgt[8];
                float og0 = smem_f32[16];
                float og1 = smem_f32[80];
                float og2 = smem_f32[144];
                float og3 = smem_f32[208];
                float _max_60 = max_noftz(og0, og1);
                float _max_61 = max_noftz(og2, og3);
                float _max_62 = max_noftz(_max_60, _max_61);
                m_ab[0] = _max_62;
                float _exp2_64 = approx_exp2((og0 - m_ab[0]) * softmax_scale_log2);
                wgt[0] = ((og0 > -CAKE_INF) ? _exp2_64 : 0.0f);
                float _exp2_65 = approx_exp2((og1 - m_ab[0]) * softmax_scale_log2);
                wgt[1] = ((og1 > -CAKE_INF) ? _exp2_65 : 0.0f);
                float _exp2_66 = approx_exp2((og2 - m_ab[0]) * softmax_scale_log2);
                wgt[2] = ((og2 > -CAKE_INF) ? _exp2_66 : 0.0f);
                float _exp2_67 = approx_exp2((og3 - m_ab[0]) * softmax_scale_log2);
                wgt[3] = ((og3 > -CAKE_INF) ? _exp2_67 : 0.0f);
                head_sum[0] = 0.0f;
                if (lane < 16) {
                    head_sum[0] = smem_f32[lane] * wgt[0] + smem_f32[64 + lane] * wgt[1] + smem_f32[128 + lane] * wgt[2] + smem_f32[192 + lane] * wgt[3];
                }
                float og0_12 = smem_f32[48];
                float og1_13 = smem_f32[112];
                float og2_14 = smem_f32[176];
                float og3_15 = smem_f32[240];
                float _max_63 = max_noftz(og0_12, og1_13);
                float _max_64 = max_noftz(og2_14, og3_15);
                float _max_65 = max_noftz(_max_63, _max_64);
                m_ab[1] = _max_65;
                float _exp2_68 = approx_exp2((og0_12 - m_ab[1]) * softmax_scale_log2);
                wgt[4] = ((og0_12 > -CAKE_INF) ? _exp2_68 : 0.0f);
                float _exp2_69 = approx_exp2((og1_13 - m_ab[1]) * softmax_scale_log2);
                wgt[5] = ((og1_13 > -CAKE_INF) ? _exp2_69 : 0.0f);
                float _exp2_70 = approx_exp2((og2_14 - m_ab[1]) * softmax_scale_log2);
                wgt[6] = ((og2_14 > -CAKE_INF) ? _exp2_70 : 0.0f);
                float _exp2_71 = approx_exp2((og3_15 - m_ab[1]) * softmax_scale_log2);
                wgt[7] = ((og3_15 > -CAKE_INF) ? _exp2_71 : 0.0f);
                head_sum[1] = 0.0f;
                if (lane < 16) {
                    head_sum[1] = smem_f32[32 + lane] * wgt[4] + smem_f32[96 + lane] * wgt[5] + smem_f32[160 + lane] * wgt[6] + smem_f32[224 + lane] * wgt[7];
                }
                if (warp_in_role == 0) {
                    if (lane < 16) {
                        int rank_st = cta_rank;
                        int head_st = lane;
                        int o_st = head_st / 8;
                        int hl_st = head_st % 8;
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(smem_v14_addr + (unsigned int)(4 * ((rank_st * 8 + hl_st) * 2))), "r"(o_st));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(stat_ready_addr), "r"(o_st));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(_mapa_0), "f"(m_ab[0]), "r"(_mapa_1) : "memory");
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(_mapa_0 + 4), "f"(head_sum[0]), "r"(_mapa_1) : "memory");
                        int head_st_0 = 16 + lane;
                        int o_st_1 = head_st_0 / 8;
                        int hl_st_2 = head_st_0 % 8;
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(smem_v14_addr + (unsigned int)(4 * ((rank_st * 8 + hl_st_2) * 2))), "r"(o_st_1));
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(stat_ready_addr), "r"(o_st_1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(_mapa_2), "f"(m_ab[1]), "r"(_mapa_3) : "memory");
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                            :: "r"(_mapa_2 + 4), "f"(head_sum[1]), "r"(_mapa_3) : "memory");
                    }
                }
                if (npairs > 0) {
                    mbarrier_wait(pv_done_addr + (npairs - 1) * 8, item_par);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                if (warp_in_role == 0) {
                    if (elect_sync()) {
                        uint32_t _mapa_4;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_4) : "r"(zone_dead_addr), "r"(0));
                        asm volatile(
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                            :: "r"(_mapa_4) : "memory");
                        uint32_t _mapa_5;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_5) : "r"(zone_dead_addr), "r"(1));
                        asm volatile(
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                            :: "r"(_mapa_5) : "memory");
                        uint32_t _mapa_6;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_6) : "r"(zone_dead_addr), "r"(2));
                        asm volatile(
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                            :: "r"(_mapa_6) : "memory");
                        uint32_t _mapa_7;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_7) : "r"(zone_dead_addr), "r"(3));
                        asm volatile(
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                            :: "r"(_mapa_7) : "memory");
                    }
                }
                float o_epi[32];
                o_epi[0] = 0.0f;
                o_epi[1] = 0.0f;
                o_epi[2] = 0.0f;
                o_epi[3] = 0.0f;
                o_epi[4] = 0.0f;
                o_epi[5] = 0.0f;
                o_epi[6] = 0.0f;
                o_epi[7] = 0.0f;
                o_epi[8] = 0.0f;
                o_epi[9] = 0.0f;
                o_epi[10] = 0.0f;
                o_epi[11] = 0.0f;
                o_epi[12] = 0.0f;
                o_epi[13] = 0.0f;
                o_epi[14] = 0.0f;
                o_epi[15] = 0.0f;
                o_epi[16] = 0.0f;
                o_epi[17] = 0.0f;
                o_epi[18] = 0.0f;
                o_epi[19] = 0.0f;
                o_epi[20] = 0.0f;
                o_epi[21] = 0.0f;
                o_epi[22] = 0.0f;
                o_epi[23] = 0.0f;
                o_epi[24] = 0.0f;
                o_epi[25] = 0.0f;
                o_epi[26] = 0.0f;
                o_epi[27] = 0.0f;
                o_epi[28] = 0.0f;
                o_epi[29] = 0.0f;
                o_epi[30] = 0.0f;
                o_epi[31] = 0.0f;
                if (npairs > 0) {
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + 256 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_64 = __fmaf_rn(_tmem_load_0[0], wgt[0], o_epi[0]);
                    o_epi[0] = _fma_64;
                    float _fma_65 = __fmaf_rn(_tmem_load_0[1], wgt[0], o_epi[1]);
                    o_epi[1] = _fma_65;
                    float _fma_66 = __fmaf_rn(_tmem_load_0[2], wgt[0], o_epi[2]);
                    o_epi[2] = _fma_66;
                    float _fma_67 = __fmaf_rn(_tmem_load_0[3], wgt[0], o_epi[3]);
                    o_epi[3] = _fma_67;
                    float _fma_68 = __fmaf_rn(_tmem_load_0[4], wgt[0], o_epi[4]);
                    o_epi[4] = _fma_68;
                    float _fma_69 = __fmaf_rn(_tmem_load_0[5], wgt[0], o_epi[5]);
                    o_epi[5] = _fma_69;
                    float _fma_70 = __fmaf_rn(_tmem_load_0[6], wgt[0], o_epi[6]);
                    o_epi[6] = _fma_70;
                    float _fma_71 = __fmaf_rn(_tmem_load_0[7], wgt[0], o_epi[7]);
                    o_epi[7] = _fma_71;
                    float _fma_72 = __fmaf_rn(_tmem_load_0[8], wgt[0], o_epi[8]);
                    o_epi[8] = _fma_72;
                    float _fma_73 = __fmaf_rn(_tmem_load_0[9], wgt[0], o_epi[9]);
                    o_epi[9] = _fma_73;
                    float _fma_74 = __fmaf_rn(_tmem_load_0[10], wgt[0], o_epi[10]);
                    o_epi[10] = _fma_74;
                    float _fma_75 = __fmaf_rn(_tmem_load_0[11], wgt[0], o_epi[11]);
                    o_epi[11] = _fma_75;
                    float _fma_76 = __fmaf_rn(_tmem_load_0[12], wgt[0], o_epi[12]);
                    o_epi[12] = _fma_76;
                    float _fma_77 = __fmaf_rn(_tmem_load_0[13], wgt[0], o_epi[13]);
                    o_epi[13] = _fma_77;
                    float _fma_78 = __fmaf_rn(_tmem_load_0[14], wgt[0], o_epi[14]);
                    o_epi[14] = _fma_78;
                    float _fma_79 = __fmaf_rn(_tmem_load_0[15], wgt[0], o_epi[15]);
                    o_epi[15] = _fma_79;
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_80 = __fmaf_rn(_tmem_load_1[0], wgt[4], o_epi[16]);
                    o_epi[16] = _fma_80;
                    float _fma_81 = __fmaf_rn(_tmem_load_1[1], wgt[4], o_epi[17]);
                    o_epi[17] = _fma_81;
                    float _fma_82 = __fmaf_rn(_tmem_load_1[2], wgt[4], o_epi[18]);
                    o_epi[18] = _fma_82;
                    float _fma_83 = __fmaf_rn(_tmem_load_1[3], wgt[4], o_epi[19]);
                    o_epi[19] = _fma_83;
                    float _fma_84 = __fmaf_rn(_tmem_load_1[4], wgt[4], o_epi[20]);
                    o_epi[20] = _fma_84;
                    float _fma_85 = __fmaf_rn(_tmem_load_1[5], wgt[4], o_epi[21]);
                    o_epi[21] = _fma_85;
                    float _fma_86 = __fmaf_rn(_tmem_load_1[6], wgt[4], o_epi[22]);
                    o_epi[22] = _fma_86;
                    float _fma_87 = __fmaf_rn(_tmem_load_1[7], wgt[4], o_epi[23]);
                    o_epi[23] = _fma_87;
                    float _fma_88 = __fmaf_rn(_tmem_load_1[8], wgt[4], o_epi[24]);
                    o_epi[24] = _fma_88;
                    float _fma_89 = __fmaf_rn(_tmem_load_1[9], wgt[4], o_epi[25]);
                    o_epi[25] = _fma_89;
                    float _fma_90 = __fmaf_rn(_tmem_load_1[10], wgt[4], o_epi[26]);
                    o_epi[26] = _fma_90;
                    float _fma_91 = __fmaf_rn(_tmem_load_1[11], wgt[4], o_epi[27]);
                    o_epi[27] = _fma_91;
                    float _fma_92 = __fmaf_rn(_tmem_load_1[12], wgt[4], o_epi[28]);
                    o_epi[28] = _fma_92;
                    float _fma_93 = __fmaf_rn(_tmem_load_1[13], wgt[4], o_epi[29]);
                    o_epi[29] = _fma_93;
                    float _fma_94 = __fmaf_rn(_tmem_load_1[14], wgt[4], o_epi[30]);
                    o_epi[30] = _fma_94;
                    float _fma_95 = __fmaf_rn(_tmem_load_1[15], wgt[4], o_epi[31]);
                    o_epi[31] = _fma_95;
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 256 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_96 = __fmaf_rn(_tmem_load_2[0], wgt[1], o_epi[0]);
                    o_epi[0] = _fma_96;
                    float _fma_97 = __fmaf_rn(_tmem_load_2[1], wgt[1], o_epi[1]);
                    o_epi[1] = _fma_97;
                    float _fma_98 = __fmaf_rn(_tmem_load_2[2], wgt[1], o_epi[2]);
                    o_epi[2] = _fma_98;
                    float _fma_99 = __fmaf_rn(_tmem_load_2[3], wgt[1], o_epi[3]);
                    o_epi[3] = _fma_99;
                    float _fma_100 = __fmaf_rn(_tmem_load_2[4], wgt[1], o_epi[4]);
                    o_epi[4] = _fma_100;
                    float _fma_101 = __fmaf_rn(_tmem_load_2[5], wgt[1], o_epi[5]);
                    o_epi[5] = _fma_101;
                    float _fma_102 = __fmaf_rn(_tmem_load_2[6], wgt[1], o_epi[6]);
                    o_epi[6] = _fma_102;
                    float _fma_103 = __fmaf_rn(_tmem_load_2[7], wgt[1], o_epi[7]);
                    o_epi[7] = _fma_103;
                    float _fma_104 = __fmaf_rn(_tmem_load_2[8], wgt[1], o_epi[8]);
                    o_epi[8] = _fma_104;
                    float _fma_105 = __fmaf_rn(_tmem_load_2[9], wgt[1], o_epi[9]);
                    o_epi[9] = _fma_105;
                    float _fma_106 = __fmaf_rn(_tmem_load_2[10], wgt[1], o_epi[10]);
                    o_epi[10] = _fma_106;
                    float _fma_107 = __fmaf_rn(_tmem_load_2[11], wgt[1], o_epi[11]);
                    o_epi[11] = _fma_107;
                    float _fma_108 = __fmaf_rn(_tmem_load_2[12], wgt[1], o_epi[12]);
                    o_epi[12] = _fma_108;
                    float _fma_109 = __fmaf_rn(_tmem_load_2[13], wgt[1], o_epi[13]);
                    o_epi[13] = _fma_109;
                    float _fma_110 = __fmaf_rn(_tmem_load_2[14], wgt[1], o_epi[14]);
                    o_epi[14] = _fma_110;
                    float _fma_111 = __fmaf_rn(_tmem_load_2[15], wgt[1], o_epi[15]);
                    o_epi[15] = _fma_111;
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 256 + 32 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_112 = __fmaf_rn(_tmem_load_3[0], wgt[5], o_epi[16]);
                    o_epi[16] = _fma_112;
                    float _fma_113 = __fmaf_rn(_tmem_load_3[1], wgt[5], o_epi[17]);
                    o_epi[17] = _fma_113;
                    float _fma_114 = __fmaf_rn(_tmem_load_3[2], wgt[5], o_epi[18]);
                    o_epi[18] = _fma_114;
                    float _fma_115 = __fmaf_rn(_tmem_load_3[3], wgt[5], o_epi[19]);
                    o_epi[19] = _fma_115;
                    float _fma_116 = __fmaf_rn(_tmem_load_3[4], wgt[5], o_epi[20]);
                    o_epi[20] = _fma_116;
                    float _fma_117 = __fmaf_rn(_tmem_load_3[5], wgt[5], o_epi[21]);
                    o_epi[21] = _fma_117;
                    float _fma_118 = __fmaf_rn(_tmem_load_3[6], wgt[5], o_epi[22]);
                    o_epi[22] = _fma_118;
                    float _fma_119 = __fmaf_rn(_tmem_load_3[7], wgt[5], o_epi[23]);
                    o_epi[23] = _fma_119;
                    float _fma_120 = __fmaf_rn(_tmem_load_3[8], wgt[5], o_epi[24]);
                    o_epi[24] = _fma_120;
                    float _fma_121 = __fmaf_rn(_tmem_load_3[9], wgt[5], o_epi[25]);
                    o_epi[25] = _fma_121;
                    float _fma_122 = __fmaf_rn(_tmem_load_3[10], wgt[5], o_epi[26]);
                    o_epi[26] = _fma_122;
                    float _fma_123 = __fmaf_rn(_tmem_load_3[11], wgt[5], o_epi[27]);
                    o_epi[27] = _fma_123;
                    float _fma_124 = __fmaf_rn(_tmem_load_3[12], wgt[5], o_epi[28]);
                    o_epi[28] = _fma_124;
                    float _fma_125 = __fmaf_rn(_tmem_load_3[13], wgt[5], o_epi[29]);
                    o_epi[29] = _fma_125;
                    float _fma_126 = __fmaf_rn(_tmem_load_3[14], wgt[5], o_epi[30]);
                    o_epi[30] = _fma_126;
                    float _fma_127 = __fmaf_rn(_tmem_load_3[15], wgt[5], o_epi[31]);
                    o_epi[31] = _fma_127;
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 256 + 64 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_128 = __fmaf_rn(_tmem_load_4[0], wgt[2], o_epi[0]);
                    o_epi[0] = _fma_128;
                    float _fma_129 = __fmaf_rn(_tmem_load_4[1], wgt[2], o_epi[1]);
                    o_epi[1] = _fma_129;
                    float _fma_130 = __fmaf_rn(_tmem_load_4[2], wgt[2], o_epi[2]);
                    o_epi[2] = _fma_130;
                    float _fma_131 = __fmaf_rn(_tmem_load_4[3], wgt[2], o_epi[3]);
                    o_epi[3] = _fma_131;
                    float _fma_132 = __fmaf_rn(_tmem_load_4[4], wgt[2], o_epi[4]);
                    o_epi[4] = _fma_132;
                    float _fma_133 = __fmaf_rn(_tmem_load_4[5], wgt[2], o_epi[5]);
                    o_epi[5] = _fma_133;
                    float _fma_134 = __fmaf_rn(_tmem_load_4[6], wgt[2], o_epi[6]);
                    o_epi[6] = _fma_134;
                    float _fma_135 = __fmaf_rn(_tmem_load_4[7], wgt[2], o_epi[7]);
                    o_epi[7] = _fma_135;
                    float _fma_136 = __fmaf_rn(_tmem_load_4[8], wgt[2], o_epi[8]);
                    o_epi[8] = _fma_136;
                    float _fma_137 = __fmaf_rn(_tmem_load_4[9], wgt[2], o_epi[9]);
                    o_epi[9] = _fma_137;
                    float _fma_138 = __fmaf_rn(_tmem_load_4[10], wgt[2], o_epi[10]);
                    o_epi[10] = _fma_138;
                    float _fma_139 = __fmaf_rn(_tmem_load_4[11], wgt[2], o_epi[11]);
                    o_epi[11] = _fma_139;
                    float _fma_140 = __fmaf_rn(_tmem_load_4[12], wgt[2], o_epi[12]);
                    o_epi[12] = _fma_140;
                    float _fma_141 = __fmaf_rn(_tmem_load_4[13], wgt[2], o_epi[13]);
                    o_epi[13] = _fma_141;
                    float _fma_142 = __fmaf_rn(_tmem_load_4[14], wgt[2], o_epi[14]);
                    o_epi[14] = _fma_142;
                    float _fma_143 = __fmaf_rn(_tmem_load_4[15], wgt[2], o_epi[15]);
                    o_epi[15] = _fma_143;
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 256 + 64 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_144 = __fmaf_rn(_tmem_load_5[0], wgt[6], o_epi[16]);
                    o_epi[16] = _fma_144;
                    float _fma_145 = __fmaf_rn(_tmem_load_5[1], wgt[6], o_epi[17]);
                    o_epi[17] = _fma_145;
                    float _fma_146 = __fmaf_rn(_tmem_load_5[2], wgt[6], o_epi[18]);
                    o_epi[18] = _fma_146;
                    float _fma_147 = __fmaf_rn(_tmem_load_5[3], wgt[6], o_epi[19]);
                    o_epi[19] = _fma_147;
                    float _fma_148 = __fmaf_rn(_tmem_load_5[4], wgt[6], o_epi[20]);
                    o_epi[20] = _fma_148;
                    float _fma_149 = __fmaf_rn(_tmem_load_5[5], wgt[6], o_epi[21]);
                    o_epi[21] = _fma_149;
                    float _fma_150 = __fmaf_rn(_tmem_load_5[6], wgt[6], o_epi[22]);
                    o_epi[22] = _fma_150;
                    float _fma_151 = __fmaf_rn(_tmem_load_5[7], wgt[6], o_epi[23]);
                    o_epi[23] = _fma_151;
                    float _fma_152 = __fmaf_rn(_tmem_load_5[8], wgt[6], o_epi[24]);
                    o_epi[24] = _fma_152;
                    float _fma_153 = __fmaf_rn(_tmem_load_5[9], wgt[6], o_epi[25]);
                    o_epi[25] = _fma_153;
                    float _fma_154 = __fmaf_rn(_tmem_load_5[10], wgt[6], o_epi[26]);
                    o_epi[26] = _fma_154;
                    float _fma_155 = __fmaf_rn(_tmem_load_5[11], wgt[6], o_epi[27]);
                    o_epi[27] = _fma_155;
                    float _fma_156 = __fmaf_rn(_tmem_load_5[12], wgt[6], o_epi[28]);
                    o_epi[28] = _fma_156;
                    float _fma_157 = __fmaf_rn(_tmem_load_5[13], wgt[6], o_epi[29]);
                    o_epi[29] = _fma_157;
                    float _fma_158 = __fmaf_rn(_tmem_load_5[14], wgt[6], o_epi[30]);
                    o_epi[30] = _fma_158;
                    float _fma_159 = __fmaf_rn(_tmem_load_5[15], wgt[6], o_epi[31]);
                    o_epi[31] = _fma_159;
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 256 + 96 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_160 = __fmaf_rn(_tmem_load_6[0], wgt[3], o_epi[0]);
                    o_epi[0] = _fma_160;
                    float _fma_161 = __fmaf_rn(_tmem_load_6[1], wgt[3], o_epi[1]);
                    o_epi[1] = _fma_161;
                    float _fma_162 = __fmaf_rn(_tmem_load_6[2], wgt[3], o_epi[2]);
                    o_epi[2] = _fma_162;
                    float _fma_163 = __fmaf_rn(_tmem_load_6[3], wgt[3], o_epi[3]);
                    o_epi[3] = _fma_163;
                    float _fma_164 = __fmaf_rn(_tmem_load_6[4], wgt[3], o_epi[4]);
                    o_epi[4] = _fma_164;
                    float _fma_165 = __fmaf_rn(_tmem_load_6[5], wgt[3], o_epi[5]);
                    o_epi[5] = _fma_165;
                    float _fma_166 = __fmaf_rn(_tmem_load_6[6], wgt[3], o_epi[6]);
                    o_epi[6] = _fma_166;
                    float _fma_167 = __fmaf_rn(_tmem_load_6[7], wgt[3], o_epi[7]);
                    o_epi[7] = _fma_167;
                    float _fma_168 = __fmaf_rn(_tmem_load_6[8], wgt[3], o_epi[8]);
                    o_epi[8] = _fma_168;
                    float _fma_169 = __fmaf_rn(_tmem_load_6[9], wgt[3], o_epi[9]);
                    o_epi[9] = _fma_169;
                    float _fma_170 = __fmaf_rn(_tmem_load_6[10], wgt[3], o_epi[10]);
                    o_epi[10] = _fma_170;
                    float _fma_171 = __fmaf_rn(_tmem_load_6[11], wgt[3], o_epi[11]);
                    o_epi[11] = _fma_171;
                    float _fma_172 = __fmaf_rn(_tmem_load_6[12], wgt[3], o_epi[12]);
                    o_epi[12] = _fma_172;
                    float _fma_173 = __fmaf_rn(_tmem_load_6[13], wgt[3], o_epi[13]);
                    o_epi[13] = _fma_173;
                    float _fma_174 = __fmaf_rn(_tmem_load_6[14], wgt[3], o_epi[14]);
                    o_epi[14] = _fma_174;
                    float _fma_175 = __fmaf_rn(_tmem_load_6[15], wgt[3], o_epi[15]);
                    o_epi[15] = _fma_175;
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 256 + 96 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_176 = __fmaf_rn(_tmem_load_7[0], wgt[7], o_epi[16]);
                    o_epi[16] = _fma_176;
                    float _fma_177 = __fmaf_rn(_tmem_load_7[1], wgt[7], o_epi[17]);
                    o_epi[17] = _fma_177;
                    float _fma_178 = __fmaf_rn(_tmem_load_7[2], wgt[7], o_epi[18]);
                    o_epi[18] = _fma_178;
                    float _fma_179 = __fmaf_rn(_tmem_load_7[3], wgt[7], o_epi[19]);
                    o_epi[19] = _fma_179;
                    float _fma_180 = __fmaf_rn(_tmem_load_7[4], wgt[7], o_epi[20]);
                    o_epi[20] = _fma_180;
                    float _fma_181 = __fmaf_rn(_tmem_load_7[5], wgt[7], o_epi[21]);
                    o_epi[21] = _fma_181;
                    float _fma_182 = __fmaf_rn(_tmem_load_7[6], wgt[7], o_epi[22]);
                    o_epi[22] = _fma_182;
                    float _fma_183 = __fmaf_rn(_tmem_load_7[7], wgt[7], o_epi[23]);
                    o_epi[23] = _fma_183;
                    float _fma_184 = __fmaf_rn(_tmem_load_7[8], wgt[7], o_epi[24]);
                    o_epi[24] = _fma_184;
                    float _fma_185 = __fmaf_rn(_tmem_load_7[9], wgt[7], o_epi[25]);
                    o_epi[25] = _fma_185;
                    float _fma_186 = __fmaf_rn(_tmem_load_7[10], wgt[7], o_epi[26]);
                    o_epi[26] = _fma_186;
                    float _fma_187 = __fmaf_rn(_tmem_load_7[11], wgt[7], o_epi[27]);
                    o_epi[27] = _fma_187;
                    float _fma_188 = __fmaf_rn(_tmem_load_7[12], wgt[7], o_epi[28]);
                    o_epi[28] = _fma_188;
                    float _fma_189 = __fmaf_rn(_tmem_load_7[13], wgt[7], o_epi[29]);
                    o_epi[29] = _fma_189;
                    float _fma_190 = __fmaf_rn(_tmem_load_7[14], wgt[7], o_epi[30]);
                    o_epi[30] = _fma_190;
                    float _fma_191 = __fmaf_rn(_tmem_load_7[15], wgt[7], o_epi[31]);
                    o_epi[31] = _fma_191;
                }
                if (total_units_sm <= work_idx + (unsigned int)(gridDim.x / 4)) {
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int rank_c = cta_rank;
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
                }
                mbarrier_wait_cluster_hint(zone_dead_addr, 0, 10000000);
                uint32_t _mapa_8;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_8) : "r"(smem_v13_addr + (unsigned int)(4 * ((rank_c * 128 + token) * 8))), "r"(0));
                uint32_t _mapa_9;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_9) : "r"(part_ready_addr), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_8), "r"(__float_as_uint(o_epi[0])), "r"(__float_as_uint(o_epi[1])), "r"(__float_as_uint(o_epi[2])), "r"(__float_as_uint(o_epi[3])), "r"(_mapa_9) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_8 + 16), "r"(__float_as_uint(o_epi[4])), "r"(__float_as_uint(o_epi[5])), "r"(__float_as_uint(o_epi[6])), "r"(__float_as_uint(o_epi[7])), "r"(_mapa_9) : "memory");
                uint32_t _mapa_10;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_10) : "r"(smem_v13_addr + (unsigned int)(4 * ((rank_c * 128 + token) * 8))), "r"(1));
                uint32_t _mapa_11;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_11) : "r"(part_ready_addr), "r"(1));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_10), "r"(__float_as_uint(o_epi[8])), "r"(__float_as_uint(o_epi[9])), "r"(__float_as_uint(o_epi[10])), "r"(__float_as_uint(o_epi[11])), "r"(_mapa_11) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_10 + 16), "r"(__float_as_uint(o_epi[12])), "r"(__float_as_uint(o_epi[13])), "r"(__float_as_uint(o_epi[14])), "r"(__float_as_uint(o_epi[15])), "r"(_mapa_11) : "memory");
                uint32_t _mapa_12;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_12) : "r"(smem_v13_addr + (unsigned int)(4 * ((rank_c * 128 + token) * 8))), "r"(2));
                uint32_t _mapa_13;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_13) : "r"(part_ready_addr), "r"(2));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_12), "r"(__float_as_uint(o_epi[16])), "r"(__float_as_uint(o_epi[17])), "r"(__float_as_uint(o_epi[18])), "r"(__float_as_uint(o_epi[19])), "r"(_mapa_13) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_12 + 16), "r"(__float_as_uint(o_epi[20])), "r"(__float_as_uint(o_epi[21])), "r"(__float_as_uint(o_epi[22])), "r"(__float_as_uint(o_epi[23])), "r"(_mapa_13) : "memory");
                uint32_t _mapa_14;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_14) : "r"(smem_v13_addr + (unsigned int)(4 * ((rank_c * 128 + token) * 8))), "r"(3));
                uint32_t _mapa_15;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_15) : "r"(part_ready_addr), "r"(3));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_14), "r"(__float_as_uint(o_epi[24])), "r"(__float_as_uint(o_epi[25])), "r"(__float_as_uint(o_epi[26])), "r"(__float_as_uint(o_epi[27])), "r"(_mapa_15) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_14 + 16), "r"(__float_as_uint(o_epi[28])), "r"(__float_as_uint(o_epi[29])), "r"(__float_as_uint(o_epi[30])), "r"(__float_as_uint(o_epi[31])), "r"(_mapa_15) : "memory");
                int tid128 = warp_in_role * 32 + lane;
                int hl = tid128 % 8;
                int g = tid128 / 8 * 2;
                int m_head = rank_c * 8 + hl;
                int m_tok = m_head / 16;
                int m_h = m_head % 16;
                mbarrier_wait_cluster_hint(stat_ready_addr, 0, 10000000);
                float st_m[4];
                float st_d[4];
                float merge_m = -CAKE_INF;
                st_m[0] = smem_v14[hl * 2];
                st_d[0] = smem_v14[hl * 2 + 1];
                float _max_66 = max_noftz(merge_m, st_m[0]);
                merge_m = _max_66;
                st_m[1] = smem_v14[(8 + hl) * 2];
                st_d[1] = smem_v14[(8 + hl) * 2 + 1];
                float _max_67 = max_noftz(merge_m, st_m[1]);
                merge_m = _max_67;
                st_m[2] = smem_v14[(16 + hl) * 2];
                st_d[2] = smem_v14[(16 + hl) * 2 + 1];
                float _max_68 = max_noftz(merge_m, st_m[2]);
                merge_m = _max_68;
                st_m[3] = smem_v14[(24 + hl) * 2];
                st_d[3] = smem_v14[(24 + hl) * 2 + 1];
                float _max_69 = max_noftz(merge_m, st_m[3]);
                merge_m = _max_69;
                float merge_d = 0.0f;
                float merge_w[4];
                float _exp2_72 = approx_exp2(softmax_scale_log2 * (st_m[0] - merge_m));
                merge_w[0] = ((st_m[0] == -CAKE_INF) ? 0.0f : _exp2_72);
                float _fma_192 = __fmaf_rn(st_d[0], merge_w[0], merge_d);
                merge_d = _fma_192;
                float _exp2_73 = approx_exp2(softmax_scale_log2 * (st_m[1] - merge_m));
                merge_w[1] = ((st_m[1] == -CAKE_INF) ? 0.0f : _exp2_73);
                float _fma_193 = __fmaf_rn(st_d[1], merge_w[1], merge_d);
                merge_d = _fma_193;
                float _exp2_74 = approx_exp2(softmax_scale_log2 * (st_m[2] - merge_m));
                merge_w[2] = ((st_m[2] == -CAKE_INF) ? 0.0f : _exp2_74);
                float _fma_194 = __fmaf_rn(st_d[2], merge_w[2], merge_d);
                merge_d = _fma_194;
                float _exp2_75 = approx_exp2(softmax_scale_log2 * (st_m[3] - merge_m));
                merge_w[3] = ((st_m[3] == -CAKE_INF) ? 0.0f : _exp2_75);
                float _fma_195 = __fmaf_rn(st_d[3], merge_w[3], merge_d);
                merge_d = _fma_195;
                float _rcp_0 = approx_rcp(merge_d);
                float merge_inv = ((merge_d > 0.0f) ? output_scale * _rcp_0 : 0.0f);
                mbarrier_wait_cluster_hint(part_ready_addr, 0, 10000000);
                if (m_h < group_size) {
                    if (m_tok < seqlen_q) {
                        int m_row = (request * seqlen_q + m_tok) * num_q_heads + kv_head * group_size + m_h;
                        float merge_o[8];
                        merge_o[0] = 0.0f;
                        merge_o[1] = 0.0f;
                        merge_o[2] = 0.0f;
                        merge_o[3] = 0.0f;
                        merge_o[4] = 0.0f;
                        merge_o[5] = 0.0f;
                        merge_o[6] = 0.0f;
                        merge_o[7] = 0.0f;
                        float _fma_196 = __fmaf_rn(smem_v13[g * 8 + hl], merge_w[0], merge_o[0]);
                        merge_o[0] = _fma_196;
                        float _fma_197 = __fmaf_rn(smem_v13[(g + 1) * 8 + hl], merge_w[0], merge_o[1]);
                        merge_o[1] = _fma_197;
                        float _fma_198 = __fmaf_rn(smem_v13[(g + 32) * 8 + hl], merge_w[0], merge_o[2]);
                        merge_o[2] = _fma_198;
                        float _fma_199 = __fmaf_rn(smem_v13[(g + 32 + 1) * 8 + hl], merge_w[0], merge_o[3]);
                        merge_o[3] = _fma_199;
                        float _fma_200 = __fmaf_rn(smem_v13[(g + 64) * 8 + hl], merge_w[0], merge_o[4]);
                        merge_o[4] = _fma_200;
                        float _fma_201 = __fmaf_rn(smem_v13[(g + 64 + 1) * 8 + hl], merge_w[0], merge_o[5]);
                        merge_o[5] = _fma_201;
                        float _fma_202 = __fmaf_rn(smem_v13[(g + 96) * 8 + hl], merge_w[0], merge_o[6]);
                        merge_o[6] = _fma_202;
                        float _fma_203 = __fmaf_rn(smem_v13[(g + 96 + 1) * 8 + hl], merge_w[0], merge_o[7]);
                        merge_o[7] = _fma_203;
                        float _fma_204 = __fmaf_rn(smem_v13[(128 + g) * 8 + hl], merge_w[1], merge_o[0]);
                        merge_o[0] = _fma_204;
                        float _fma_205 = __fmaf_rn(smem_v13[(128 + g + 1) * 8 + hl], merge_w[1], merge_o[1]);
                        merge_o[1] = _fma_205;
                        float _fma_206 = __fmaf_rn(smem_v13[(128 + g + 32) * 8 + hl], merge_w[1], merge_o[2]);
                        merge_o[2] = _fma_206;
                        float _fma_207 = __fmaf_rn(smem_v13[(128 + g + 32 + 1) * 8 + hl], merge_w[1], merge_o[3]);
                        merge_o[3] = _fma_207;
                        float _fma_208 = __fmaf_rn(smem_v13[(128 + g + 64) * 8 + hl], merge_w[1], merge_o[4]);
                        merge_o[4] = _fma_208;
                        float _fma_209 = __fmaf_rn(smem_v13[(128 + g + 64 + 1) * 8 + hl], merge_w[1], merge_o[5]);
                        merge_o[5] = _fma_209;
                        float _fma_210 = __fmaf_rn(smem_v13[(128 + g + 96) * 8 + hl], merge_w[1], merge_o[6]);
                        merge_o[6] = _fma_210;
                        float _fma_211 = __fmaf_rn(smem_v13[(128 + g + 96 + 1) * 8 + hl], merge_w[1], merge_o[7]);
                        merge_o[7] = _fma_211;
                        float _fma_212 = __fmaf_rn(smem_v13[(256 + g) * 8 + hl], merge_w[2], merge_o[0]);
                        merge_o[0] = _fma_212;
                        float _fma_213 = __fmaf_rn(smem_v13[(256 + g + 1) * 8 + hl], merge_w[2], merge_o[1]);
                        merge_o[1] = _fma_213;
                        float _fma_214 = __fmaf_rn(smem_v13[(256 + g + 32) * 8 + hl], merge_w[2], merge_o[2]);
                        merge_o[2] = _fma_214;
                        float _fma_215 = __fmaf_rn(smem_v13[(256 + g + 32 + 1) * 8 + hl], merge_w[2], merge_o[3]);
                        merge_o[3] = _fma_215;
                        float _fma_216 = __fmaf_rn(smem_v13[(256 + g + 64) * 8 + hl], merge_w[2], merge_o[4]);
                        merge_o[4] = _fma_216;
                        float _fma_217 = __fmaf_rn(smem_v13[(256 + g + 64 + 1) * 8 + hl], merge_w[2], merge_o[5]);
                        merge_o[5] = _fma_217;
                        float _fma_218 = __fmaf_rn(smem_v13[(256 + g + 96) * 8 + hl], merge_w[2], merge_o[6]);
                        merge_o[6] = _fma_218;
                        float _fma_219 = __fmaf_rn(smem_v13[(256 + g + 96 + 1) * 8 + hl], merge_w[2], merge_o[7]);
                        merge_o[7] = _fma_219;
                        float _fma_220 = __fmaf_rn(smem_v13[(384 + g) * 8 + hl], merge_w[3], merge_o[0]);
                        merge_o[0] = _fma_220;
                        float _fma_221 = __fmaf_rn(smem_v13[(384 + g + 1) * 8 + hl], merge_w[3], merge_o[1]);
                        merge_o[1] = _fma_221;
                        float _fma_222 = __fmaf_rn(smem_v13[(384 + g + 32) * 8 + hl], merge_w[3], merge_o[2]);
                        merge_o[2] = _fma_222;
                        float _fma_223 = __fmaf_rn(smem_v13[(384 + g + 32 + 1) * 8 + hl], merge_w[3], merge_o[3]);
                        merge_o[3] = _fma_223;
                        float _fma_224 = __fmaf_rn(smem_v13[(384 + g + 64) * 8 + hl], merge_w[3], merge_o[4]);
                        merge_o[4] = _fma_224;
                        float _fma_225 = __fmaf_rn(smem_v13[(384 + g + 64 + 1) * 8 + hl], merge_w[3], merge_o[5]);
                        merge_o[5] = _fma_225;
                        float _fma_226 = __fmaf_rn(smem_v13[(384 + g + 96) * 8 + hl], merge_w[3], merge_o[6]);
                        merge_o[6] = _fma_226;
                        float _fma_227 = __fmaf_rn(smem_v13[(384 + g + 96 + 1) * 8 + hl], merge_w[3], merge_o[7]);
                        merge_o[7] = _fma_227;
                        {
                            const float2 _prescale2_0 = {merge_inv, merge_inv};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&merge_o[0])[_ps], _prescale2_0);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                merge_o[0 + _ps] *= merge_inv;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(merge_o[0 + 0], merge_o[0 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (m_row * 128 + g)))[0]) = _pk;
                        }
                        {
                            const float2 _prescale2_1 = {merge_inv, merge_inv};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&merge_o[2])[_ps], _prescale2_1);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                merge_o[2 + _ps] *= merge_inv;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(merge_o[2 + 0], merge_o[2 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (m_row * 128 + g + 32)))[0]) = _pk;
                        }
                        {
                            const float2 _prescale2_2 = {merge_inv, merge_inv};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&merge_o[4])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                merge_o[4 + _ps] *= merge_inv;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(merge_o[4 + 0], merge_o[4 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (m_row * 128 + g + 64)))[0]) = _pk;
                        }
                        {
                            const float2 _prescale2_3 = {merge_inv, merge_inv};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 1; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&merge_o[6])[_ps], _prescale2_3);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 2; _ps++)
                                merge_o[6 + _ps] *= merge_inv;
                            #endif
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(merge_o[6 + 0], merge_o[6 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (m_row * 128 + g + 96)))[0]) = _pk;
                        }
                        if (g == 0) {
                            float merge_lse = -CAKE_INF;
                            if (merge_d > 0.0f) {
                                float _log2_0;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(merge_d));
                                merge_lse = merge_m * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + m_row) + (0)) = merge_lse;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: transform_k ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // transform_k_main
            unsigned int total_units_k = total_q / seqlen_q * num_kv_heads;
            int role_tid = warp % 4 * 32 + lane;
            int item_stage_k = 0;
            int item_phase_k = 0;
            int kt_k = 0;
            int tile_base_k = 0;
            #pragma unroll 1
            for (unsigned int work_idx_k = blockIdx.x / 4; work_idx_k < total_units_k; work_idx_k += gridDim.x / 4) {
                mbarrier_wait(meta_full_addr + (item_stage_k) * 8, item_phase_k);
                int npairs_k = smem_npairs[item_stage_k];
                item_stage_k += 1;
                if (item_stage_k == 2) { item_stage_k = 0; item_phase_k ^= 1; }
                #pragma unroll 1
                for (int sp = 0; sp < 16; sp++) {
                    if (npairs_k <= sp) {
                        break;
                    }
                    int kl_k = 2 * sp;
                    if (sp >= 2) {
                        kl_k = 4 * sp - 4;
                    }
                    #pragma unroll 1
                    for (int jv = 0; jv < 2; jv++) {
                        int st = (tile_base_k + kl_k + jv) % 8;
                        int fst = kt_k % 16;
                        int fph = kt_k / 16 & 1;
                        int slot = kt_k % 2;
                        int use = kt_k / 2 & 1;
                        mbarrier_wait(kv_raw_full_addr + (fst) * 8, fph);
                        mbarrier_wait(k16_empty_addr + (slot) * 8, 1 - use);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int t = role_tid;
                        int row_base = smem_raw_addr + (unsigned int)(st * 8192) + (unsigned int)(t * 64);
                        int swz = t >> 1 & 3;
                        const int lane_base = warp % 4 * 32 << 16;
                        int dst_base = taddr + 128 + (unsigned int)(slot * 64) + (unsigned int)lane_base;
                        unsigned int sw[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw[(0) + 1]))
                            : "r"(smem_scale_addr + (unsigned int)(st * 1024) + (unsigned int)(t * 8)));
                        unsigned int sc[8];
                        unsigned int pair = sw[0] & 65535;
                        uint32_t _fp8_bf16x2_0;
                        {
                        _fp8_bf16x2_0 = cvt_bf16x2_e4m3x2_v((uint16_t)(pair));
                        }
                        unsigned int sf = _fp8_bf16x2_0;
                        uint32_t _prmt_b32_0;
                        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_0) : "r"(sf), "r"(sf));
                        sc[0] = _prmt_b32_0;
                        uint32_t _prmt_b32_1;
                        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_1) : "r"(sf), "r"(sf));
                        sc[1] = _prmt_b32_1;
                        unsigned int pair_0 = sw[0] >> 16 & 65535;
                        uint32_t _fp8_bf16x2_1;
                        {
                        _fp8_bf16x2_1 = cvt_bf16x2_e4m3x2_v((uint16_t)(pair_0));
                        }
                        unsigned int sf_1 = _fp8_bf16x2_1;
                        uint32_t _prmt_b32_2;
                        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_2) : "r"(sf_1), "r"(sf_1));
                        sc[2] = _prmt_b32_2;
                        uint32_t _prmt_b32_3;
                        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_3) : "r"(sf_1), "r"(sf_1));
                        sc[3] = _prmt_b32_3;
                        unsigned int pair_2 = sw[1] & 65535;
                        uint32_t _fp8_bf16x2_2;
                        {
                        _fp8_bf16x2_2 = cvt_bf16x2_e4m3x2_v((uint16_t)(pair_2));
                        }
                        unsigned int sf_3 = _fp8_bf16x2_2;
                        uint32_t _prmt_b32_4;
                        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_4) : "r"(sf_3), "r"(sf_3));
                        sc[4] = _prmt_b32_4;
                        uint32_t _prmt_b32_5;
                        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_5) : "r"(sf_3), "r"(sf_3));
                        sc[5] = _prmt_b32_5;
                        unsigned int pair_4 = sw[1] >> 16 & 65535;
                        uint32_t _fp8_bf16x2_3;
                        {
                        _fp8_bf16x2_3 = cvt_bf16x2_e4m3x2_v((uint16_t)(pair_4));
                        }
                        unsigned int sf_5 = _fp8_bf16x2_3;
                        uint32_t _prmt_b32_6;
                        asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_6) : "r"(sf_5), "r"(sf_5));
                        sc[6] = _prmt_b32_6;
                        uint32_t _prmt_b32_7;
                        asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_7) : "r"(sf_5), "r"(sf_5));
                        sc[7] = _prmt_b32_7;
                        unsigned int raw[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                            : "r"(row_base + ((0 ^ swz) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 3]))
                            : "r"(row_base + ((1 ^ swz) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 3]))
                            : "r"(row_base + ((2 ^ swz) << 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 3]))
                            : "r"(row_base + ((3 ^ swz) << 4)));
                        unsigned int w[8];
                        uint32_t _fp4_bf16x8_0[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[0])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[0])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_0[0], hi = (uint16_t)(_fp4_bf16x8_0[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_0[1], hi = (uint16_t)(_fp4_bf16x8_0[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_0[2], hi = (uint16_t)(_fp4_bf16x8_0[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_0[3], hi = (uint16_t)(_fp4_bf16x8_0[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_0;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_fp4_bf16x8_0[0]), "r"(sc[0]));
                        w[0] = _bf16x2_mul_0;
                        uint32_t _bf16x2_mul_1;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_fp4_bf16x8_0[1]), "r"(sc[0]));
                        w[1] = _bf16x2_mul_1;
                        uint32_t _bf16x2_mul_2;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(_fp4_bf16x8_0[2]), "r"(sc[0]));
                        w[2] = _bf16x2_mul_2;
                        uint32_t _bf16x2_mul_3;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(_fp4_bf16x8_0[3]), "r"(sc[0]));
                        w[3] = _bf16x2_mul_3;
                        uint32_t _fp4_bf16x8_1[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw[1])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw[1])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[0], hi = (uint16_t)(_fp4_bf16x8_1[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[1], hi = (uint16_t)(_fp4_bf16x8_1[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[2], hi = (uint16_t)(_fp4_bf16x8_1[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[3], hi = (uint16_t)(_fp4_bf16x8_1[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_4;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(_fp4_bf16x8_1[0]), "r"(sc[0]));
                        w[4] = _bf16x2_mul_4;
                        uint32_t _bf16x2_mul_5;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(_fp4_bf16x8_1[1]), "r"(sc[0]));
                        w[5] = _bf16x2_mul_5;
                        uint32_t _bf16x2_mul_6;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(_fp4_bf16x8_1[2]), "r"(sc[0]));
                        w[6] = _bf16x2_mul_6;
                        uint32_t _bf16x2_mul_7;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(_fp4_bf16x8_1[3]), "r"(sc[0]));
                        w[7] = _bf16x2_mul_7;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base), "r"(*reinterpret_cast<const uint32_t*>(&w[0])), "r"(*reinterpret_cast<const uint32_t*>(&w[1])), "r"(*reinterpret_cast<const uint32_t*>(&w[2])), "r"(*reinterpret_cast<const uint32_t*>(&w[3])), "r"(*reinterpret_cast<const uint32_t*>(&w[4])), "r"(*reinterpret_cast<const uint32_t*>(&w[5])), "r"(*reinterpret_cast<const uint32_t*>(&w[6])), "r"(*reinterpret_cast<const uint32_t*>(&w[7])));
                        unsigned int w_6[8];
                        uint32_t _fp4_bf16x8_2[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw[2])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_2[0]), "=r"(_fp4_bf16x8_2[1]), "=r"(_fp4_bf16x8_2[2]), "=r"(_fp4_bf16x8_2[3]) : "r"((uint32_t)(raw[2])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_2[0], hi = (uint16_t)(_fp4_bf16x8_2[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_2[1], hi = (uint16_t)(_fp4_bf16x8_2[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_2[2], hi = (uint16_t)(_fp4_bf16x8_2[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_2[3], hi = (uint16_t)(_fp4_bf16x8_2[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_2[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_8;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(_fp4_bf16x8_2[0]), "r"(sc[1]));
                        w_6[0] = _bf16x2_mul_8;
                        uint32_t _bf16x2_mul_9;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(_fp4_bf16x8_2[1]), "r"(sc[1]));
                        w_6[1] = _bf16x2_mul_9;
                        uint32_t _bf16x2_mul_10;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(_fp4_bf16x8_2[2]), "r"(sc[1]));
                        w_6[2] = _bf16x2_mul_10;
                        uint32_t _bf16x2_mul_11;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(_fp4_bf16x8_2[3]), "r"(sc[1]));
                        w_6[3] = _bf16x2_mul_11;
                        uint32_t _fp4_bf16x8_3[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw[3])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_3[0]), "=r"(_fp4_bf16x8_3[1]), "=r"(_fp4_bf16x8_3[2]), "=r"(_fp4_bf16x8_3[3]) : "r"((uint32_t)(raw[3])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_3[0], hi = (uint16_t)(_fp4_bf16x8_3[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_3[1], hi = (uint16_t)(_fp4_bf16x8_3[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_3[2], hi = (uint16_t)(_fp4_bf16x8_3[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_3[3], hi = (uint16_t)(_fp4_bf16x8_3[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_3[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_12;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(_fp4_bf16x8_3[0]), "r"(sc[1]));
                        w_6[4] = _bf16x2_mul_12;
                        uint32_t _bf16x2_mul_13;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(_fp4_bf16x8_3[1]), "r"(sc[1]));
                        w_6[5] = _bf16x2_mul_13;
                        uint32_t _bf16x2_mul_14;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(_fp4_bf16x8_3[2]), "r"(sc[1]));
                        w_6[6] = _bf16x2_mul_14;
                        uint32_t _bf16x2_mul_15;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(_fp4_bf16x8_3[3]), "r"(sc[1]));
                        w_6[7] = _bf16x2_mul_15;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 8), "r"(*reinterpret_cast<const uint32_t*>(&w_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_6[7])));
                        unsigned int w_7[8];
                        uint32_t _fp4_bf16x8_4[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw[4])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_4[0]), "=r"(_fp4_bf16x8_4[1]), "=r"(_fp4_bf16x8_4[2]), "=r"(_fp4_bf16x8_4[3]) : "r"((uint32_t)(raw[4])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_4[0], hi = (uint16_t)(_fp4_bf16x8_4[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_4[1], hi = (uint16_t)(_fp4_bf16x8_4[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_4[2], hi = (uint16_t)(_fp4_bf16x8_4[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_4[3], hi = (uint16_t)(_fp4_bf16x8_4[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_4[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_16;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(_fp4_bf16x8_4[0]), "r"(sc[2]));
                        w_7[0] = _bf16x2_mul_16;
                        uint32_t _bf16x2_mul_17;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(_fp4_bf16x8_4[1]), "r"(sc[2]));
                        w_7[1] = _bf16x2_mul_17;
                        uint32_t _bf16x2_mul_18;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(_fp4_bf16x8_4[2]), "r"(sc[2]));
                        w_7[2] = _bf16x2_mul_18;
                        uint32_t _bf16x2_mul_19;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(_fp4_bf16x8_4[3]), "r"(sc[2]));
                        w_7[3] = _bf16x2_mul_19;
                        uint32_t _fp4_bf16x8_5[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw[5])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_5[0]), "=r"(_fp4_bf16x8_5[1]), "=r"(_fp4_bf16x8_5[2]), "=r"(_fp4_bf16x8_5[3]) : "r"((uint32_t)(raw[5])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_5[0], hi = (uint16_t)(_fp4_bf16x8_5[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_5[1], hi = (uint16_t)(_fp4_bf16x8_5[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_5[2], hi = (uint16_t)(_fp4_bf16x8_5[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_5[3], hi = (uint16_t)(_fp4_bf16x8_5[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_5[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_20;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(_fp4_bf16x8_5[0]), "r"(sc[2]));
                        w_7[4] = _bf16x2_mul_20;
                        uint32_t _bf16x2_mul_21;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(_fp4_bf16x8_5[1]), "r"(sc[2]));
                        w_7[5] = _bf16x2_mul_21;
                        uint32_t _bf16x2_mul_22;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(_fp4_bf16x8_5[2]), "r"(sc[2]));
                        w_7[6] = _bf16x2_mul_22;
                        uint32_t _bf16x2_mul_23;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(_fp4_bf16x8_5[3]), "r"(sc[2]));
                        w_7[7] = _bf16x2_mul_23;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 16), "r"(*reinterpret_cast<const uint32_t*>(&w_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_7[7])));
                        unsigned int w_8[8];
                        uint32_t _fp4_bf16x8_6[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw[6])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_6[0]), "=r"(_fp4_bf16x8_6[1]), "=r"(_fp4_bf16x8_6[2]), "=r"(_fp4_bf16x8_6[3]) : "r"((uint32_t)(raw[6])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_6[0], hi = (uint16_t)(_fp4_bf16x8_6[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_6[1], hi = (uint16_t)(_fp4_bf16x8_6[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_6[2], hi = (uint16_t)(_fp4_bf16x8_6[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_6[3], hi = (uint16_t)(_fp4_bf16x8_6[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_6[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_24;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(_fp4_bf16x8_6[0]), "r"(sc[3]));
                        w_8[0] = _bf16x2_mul_24;
                        uint32_t _bf16x2_mul_25;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(_fp4_bf16x8_6[1]), "r"(sc[3]));
                        w_8[1] = _bf16x2_mul_25;
                        uint32_t _bf16x2_mul_26;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(_fp4_bf16x8_6[2]), "r"(sc[3]));
                        w_8[2] = _bf16x2_mul_26;
                        uint32_t _bf16x2_mul_27;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(_fp4_bf16x8_6[3]), "r"(sc[3]));
                        w_8[3] = _bf16x2_mul_27;
                        uint32_t _fp4_bf16x8_7[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw[7])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_7[0]), "=r"(_fp4_bf16x8_7[1]), "=r"(_fp4_bf16x8_7[2]), "=r"(_fp4_bf16x8_7[3]) : "r"((uint32_t)(raw[7])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_7[0], hi = (uint16_t)(_fp4_bf16x8_7[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_7[1], hi = (uint16_t)(_fp4_bf16x8_7[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_7[2], hi = (uint16_t)(_fp4_bf16x8_7[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_7[3], hi = (uint16_t)(_fp4_bf16x8_7[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_7[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_28;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(_fp4_bf16x8_7[0]), "r"(sc[3]));
                        w_8[4] = _bf16x2_mul_28;
                        uint32_t _bf16x2_mul_29;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(_fp4_bf16x8_7[1]), "r"(sc[3]));
                        w_8[5] = _bf16x2_mul_29;
                        uint32_t _bf16x2_mul_30;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(_fp4_bf16x8_7[2]), "r"(sc[3]));
                        w_8[6] = _bf16x2_mul_30;
                        uint32_t _bf16x2_mul_31;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(_fp4_bf16x8_7[3]), "r"(sc[3]));
                        w_8[7] = _bf16x2_mul_31;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 16 + 8), "r"(*reinterpret_cast<const uint32_t*>(&w_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_8[7])));
                        unsigned int w_9[8];
                        uint32_t _fp4_bf16x8_8[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw[8])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_8[0]), "=r"(_fp4_bf16x8_8[1]), "=r"(_fp4_bf16x8_8[2]), "=r"(_fp4_bf16x8_8[3]) : "r"((uint32_t)(raw[8])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_8[0], hi = (uint16_t)(_fp4_bf16x8_8[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_8[1], hi = (uint16_t)(_fp4_bf16x8_8[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_8[2], hi = (uint16_t)(_fp4_bf16x8_8[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_8[3], hi = (uint16_t)(_fp4_bf16x8_8[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_8[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_32;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(_fp4_bf16x8_8[0]), "r"(sc[4]));
                        w_9[0] = _bf16x2_mul_32;
                        uint32_t _bf16x2_mul_33;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(_fp4_bf16x8_8[1]), "r"(sc[4]));
                        w_9[1] = _bf16x2_mul_33;
                        uint32_t _bf16x2_mul_34;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(_fp4_bf16x8_8[2]), "r"(sc[4]));
                        w_9[2] = _bf16x2_mul_34;
                        uint32_t _bf16x2_mul_35;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(_fp4_bf16x8_8[3]), "r"(sc[4]));
                        w_9[3] = _bf16x2_mul_35;
                        uint32_t _fp4_bf16x8_9[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw[9])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_9[0]), "=r"(_fp4_bf16x8_9[1]), "=r"(_fp4_bf16x8_9[2]), "=r"(_fp4_bf16x8_9[3]) : "r"((uint32_t)(raw[9])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_9[0], hi = (uint16_t)(_fp4_bf16x8_9[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_9[1], hi = (uint16_t)(_fp4_bf16x8_9[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_9[2], hi = (uint16_t)(_fp4_bf16x8_9[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_9[3], hi = (uint16_t)(_fp4_bf16x8_9[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_9[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_36;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(_fp4_bf16x8_9[0]), "r"(sc[4]));
                        w_9[4] = _bf16x2_mul_36;
                        uint32_t _bf16x2_mul_37;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(_fp4_bf16x8_9[1]), "r"(sc[4]));
                        w_9[5] = _bf16x2_mul_37;
                        uint32_t _bf16x2_mul_38;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(_fp4_bf16x8_9[2]), "r"(sc[4]));
                        w_9[6] = _bf16x2_mul_38;
                        uint32_t _bf16x2_mul_39;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(_fp4_bf16x8_9[3]), "r"(sc[4]));
                        w_9[7] = _bf16x2_mul_39;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 32), "r"(*reinterpret_cast<const uint32_t*>(&w_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_9[7])));
                        unsigned int w_10[8];
                        uint32_t _fp4_bf16x8_10[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw[10])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_10[0]), "=r"(_fp4_bf16x8_10[1]), "=r"(_fp4_bf16x8_10[2]), "=r"(_fp4_bf16x8_10[3]) : "r"((uint32_t)(raw[10])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_10[0], hi = (uint16_t)(_fp4_bf16x8_10[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_10[1], hi = (uint16_t)(_fp4_bf16x8_10[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_10[2], hi = (uint16_t)(_fp4_bf16x8_10[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_10[3], hi = (uint16_t)(_fp4_bf16x8_10[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_10[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_40;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(_fp4_bf16x8_10[0]), "r"(sc[5]));
                        w_10[0] = _bf16x2_mul_40;
                        uint32_t _bf16x2_mul_41;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(_fp4_bf16x8_10[1]), "r"(sc[5]));
                        w_10[1] = _bf16x2_mul_41;
                        uint32_t _bf16x2_mul_42;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(_fp4_bf16x8_10[2]), "r"(sc[5]));
                        w_10[2] = _bf16x2_mul_42;
                        uint32_t _bf16x2_mul_43;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(_fp4_bf16x8_10[3]), "r"(sc[5]));
                        w_10[3] = _bf16x2_mul_43;
                        uint32_t _fp4_bf16x8_11[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw[11])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_11[0]), "=r"(_fp4_bf16x8_11[1]), "=r"(_fp4_bf16x8_11[2]), "=r"(_fp4_bf16x8_11[3]) : "r"((uint32_t)(raw[11])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_11[0], hi = (uint16_t)(_fp4_bf16x8_11[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_11[1], hi = (uint16_t)(_fp4_bf16x8_11[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_11[2], hi = (uint16_t)(_fp4_bf16x8_11[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_11[3], hi = (uint16_t)(_fp4_bf16x8_11[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_11[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_44;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(_fp4_bf16x8_11[0]), "r"(sc[5]));
                        w_10[4] = _bf16x2_mul_44;
                        uint32_t _bf16x2_mul_45;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(_fp4_bf16x8_11[1]), "r"(sc[5]));
                        w_10[5] = _bf16x2_mul_45;
                        uint32_t _bf16x2_mul_46;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(_fp4_bf16x8_11[2]), "r"(sc[5]));
                        w_10[6] = _bf16x2_mul_46;
                        uint32_t _bf16x2_mul_47;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(_fp4_bf16x8_11[3]), "r"(sc[5]));
                        w_10[7] = _bf16x2_mul_47;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 32 + 8), "r"(*reinterpret_cast<const uint32_t*>(&w_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_10[7])));
                        unsigned int w_11[8];
                        uint32_t _fp4_bf16x8_12[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw[12])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_12[0]), "=r"(_fp4_bf16x8_12[1]), "=r"(_fp4_bf16x8_12[2]), "=r"(_fp4_bf16x8_12[3]) : "r"((uint32_t)(raw[12])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_12[0], hi = (uint16_t)(_fp4_bf16x8_12[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_12[1], hi = (uint16_t)(_fp4_bf16x8_12[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_12[2], hi = (uint16_t)(_fp4_bf16x8_12[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_12[3], hi = (uint16_t)(_fp4_bf16x8_12[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_12[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_48;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(_fp4_bf16x8_12[0]), "r"(sc[6]));
                        w_11[0] = _bf16x2_mul_48;
                        uint32_t _bf16x2_mul_49;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(_fp4_bf16x8_12[1]), "r"(sc[6]));
                        w_11[1] = _bf16x2_mul_49;
                        uint32_t _bf16x2_mul_50;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(_fp4_bf16x8_12[2]), "r"(sc[6]));
                        w_11[2] = _bf16x2_mul_50;
                        uint32_t _bf16x2_mul_51;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(_fp4_bf16x8_12[3]), "r"(sc[6]));
                        w_11[3] = _bf16x2_mul_51;
                        uint32_t _fp4_bf16x8_13[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw[13])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_13[0]), "=r"(_fp4_bf16x8_13[1]), "=r"(_fp4_bf16x8_13[2]), "=r"(_fp4_bf16x8_13[3]) : "r"((uint32_t)(raw[13])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_13[0], hi = (uint16_t)(_fp4_bf16x8_13[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_13[1], hi = (uint16_t)(_fp4_bf16x8_13[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_13[2], hi = (uint16_t)(_fp4_bf16x8_13[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_13[3], hi = (uint16_t)(_fp4_bf16x8_13[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_13[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_52;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(_fp4_bf16x8_13[0]), "r"(sc[6]));
                        w_11[4] = _bf16x2_mul_52;
                        uint32_t _bf16x2_mul_53;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(_fp4_bf16x8_13[1]), "r"(sc[6]));
                        w_11[5] = _bf16x2_mul_53;
                        uint32_t _bf16x2_mul_54;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(_fp4_bf16x8_13[2]), "r"(sc[6]));
                        w_11[6] = _bf16x2_mul_54;
                        uint32_t _bf16x2_mul_55;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(_fp4_bf16x8_13[3]), "r"(sc[6]));
                        w_11[7] = _bf16x2_mul_55;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 48), "r"(*reinterpret_cast<const uint32_t*>(&w_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_11[7])));
                        unsigned int w_12[8];
                        uint32_t _fp4_bf16x8_14[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw[14])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_14[0]), "=r"(_fp4_bf16x8_14[1]), "=r"(_fp4_bf16x8_14[2]), "=r"(_fp4_bf16x8_14[3]) : "r"((uint32_t)(raw[14])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_14[0], hi = (uint16_t)(_fp4_bf16x8_14[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_14[1], hi = (uint16_t)(_fp4_bf16x8_14[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_14[2], hi = (uint16_t)(_fp4_bf16x8_14[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_14[3], hi = (uint16_t)(_fp4_bf16x8_14[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_14[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_56;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(_fp4_bf16x8_14[0]), "r"(sc[7]));
                        w_12[0] = _bf16x2_mul_56;
                        uint32_t _bf16x2_mul_57;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(_fp4_bf16x8_14[1]), "r"(sc[7]));
                        w_12[1] = _bf16x2_mul_57;
                        uint32_t _bf16x2_mul_58;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(_fp4_bf16x8_14[2]), "r"(sc[7]));
                        w_12[2] = _bf16x2_mul_58;
                        uint32_t _bf16x2_mul_59;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(_fp4_bf16x8_14[3]), "r"(sc[7]));
                        w_12[3] = _bf16x2_mul_59;
                        uint32_t _fp4_bf16x8_15[4];
                        {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw[15])));
                        #else
                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                            "}"
                            : "=r"(_fp4_bf16x8_15[0]), "=r"(_fp4_bf16x8_15[1]), "=r"(_fp4_bf16x8_15[2]), "=r"(_fp4_bf16x8_15[3]) : "r"((uint32_t)(raw[15])));
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_15[0], hi = (uint16_t)(_fp4_bf16x8_15[0] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[0]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_15[1], hi = (uint16_t)(_fp4_bf16x8_15[1] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[1]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_15[2], hi = (uint16_t)(_fp4_bf16x8_15[2] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[2]) : "f"(f1), "f"(f0));
                        }
                        {
                        uint16_t lo = (uint16_t)_fp4_bf16x8_15[3], hi = (uint16_t)(_fp4_bf16x8_15[3] >> 16);
                        float f0, f1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_15[3]) : "f"(f1), "f"(f0));
                        }
                        #endif
                        }
                        uint32_t _bf16x2_mul_60;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(_fp4_bf16x8_15[0]), "r"(sc[7]));
                        w_12[4] = _bf16x2_mul_60;
                        uint32_t _bf16x2_mul_61;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(_fp4_bf16x8_15[1]), "r"(sc[7]));
                        w_12[5] = _bf16x2_mul_61;
                        uint32_t _bf16x2_mul_62;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(_fp4_bf16x8_15[2]), "r"(sc[7]));
                        w_12[6] = _bf16x2_mul_62;
                        uint32_t _bf16x2_mul_63;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(_fp4_bf16x8_15[3]), "r"(sc[7]));
                        w_12[7] = _bf16x2_mul_63;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "r"(dst_base + 48 + 8), "r"(*reinterpret_cast<const uint32_t*>(&w_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_12[7])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(k16_full_addr + (slot) * 8);
                            mbarrier_arrive(kv_raw_empty_addr + (st) * 8);
                        }
                        kt_k = kt_k + 1;
                    }
                }
                tile_base_k = tile_base_k + 4 * npairs_k;
            }
        }
    }
    // ---- Role: transform_v ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // transform_v_main
            unsigned int total_units_t = total_q / seqlen_q * num_kv_heads;
            int role_tid_v = warp % 4 * 32 + lane;
            int item_stage_t = 0;
            int item_phase_t = 0;
            int tile_n = 0;
            int vtile_t = 0;
            #pragma unroll 1
            for (unsigned int work_idx_t = blockIdx.x / 4; work_idx_t < total_units_t; work_idx_t += gridDim.x / 4) {
                mbarrier_wait(meta_full_addr + (item_stage_t) * 8, item_phase_t);
                int npairs_t = smem_npairs[item_stage_t];
                item_stage_t += 1;
                if (item_stage_t == 2) { item_stage_t = 0; item_phase_t ^= 1; }
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 16; pair_1++) {
                    if (npairs_t <= pair_1) {
                        break;
                    }
                    if (pair_1 == 0) {
                        tile_n = tile_n + 2;
                        if (npairs_t > 1) {
                            tile_n = tile_n + 2;
                        }
                    }
                    if (npairs_t > pair_1 + 2) {
                        tile_n = tile_n + 2;
                    }
                    #pragma unroll 1
                    for (int v_tile_r = 0; v_tile_r < 2; v_tile_r++) {
                        int st_1 = tile_n % 8;
                        int fst_1 = 16 + vtile_t % 16;
                        int fph_1 = vtile_t / 16 & 1;
                        int vs = vtile_t % 3;
                        int vph = 1 - (vtile_t / 3 & 1);
                        mbarrier_wait(kv_raw_full_addr + (fst_1) * 8, fph_1);
                        mbarrier_wait(v_empty_addr + (vs) * 8, vph);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int t_1 = role_tid_v;
                        int row_base_1 = smem_raw_addr + (unsigned int)(st_1 * 8192) + (unsigned int)(t_1 * 64);
                        int swz_1 = t_1 >> 1 & 3;
                        int tm7 = t_1 & 7;
                        int dst_row = smem_v_addr + (unsigned int)(vs * 32768) + (unsigned int)((t_1 >> 3) * 1024) + (unsigned int)(tm7 * 128);
                        unsigned int sel = (t_1 & 3) * 17 + 64;
                        int scale_row = smem_scale_addr + (unsigned int)(st_1 * 1024) + (unsigned int)((t_1 >> 2) * 32);
                        {
                            unsigned int sw_1[8];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 3]))
                                : "r"(scale_row));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 3]))
                                : "r"(scale_row + 16));
                            unsigned int sc_1[8];
                            #pragma unroll
                            for (int q = 0; q < 4; q++) {
                                uint32_t _prmt_b32_8;
                                asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_8) : "r"(sw_1[2 * q]), "r"(sw_1[2 * q + 1]), "r"(sel));
                                unsigned int pair_0_1 = _prmt_b32_8;
                                uint32_t _fp8_bf16x2_4;
                                {
                                _fp8_bf16x2_4 = cvt_bf16x2_e4m3x2_v((uint16_t)(pair_0_1));
                                }
                                unsigned int sf_2 = _fp8_bf16x2_4;
                                uint32_t _prmt_b32_9;
                                asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_9) : "r"(sf_2), "r"(sf_2));
                                sc_1[2 * q] = _prmt_b32_9;
                                uint32_t _prmt_b32_10;
                                asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_10) : "r"(sf_2), "r"(sf_2));
                                sc_1[2 * q + 1] = _prmt_b32_10;
                            }
                            unsigned int raw_1[16];
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 4; q_1++) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_1) + 3]))
                                    : "r"(row_base_1 + ((q_1 ^ swz_1) << 4)));
                            }
                            #pragma unroll
                            for (int q_2 = 0; q_2 < 4; q_2++) {
                                int plane = dst_row + (q_2 >> 1) * 16384;
                                #pragma unroll
                                for (int i = 0; i < 4; i++) {
                                    uint32_t _fp4_bf16x8_16[4];
                                    {
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                        "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                                        "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                                        "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                                        "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                                        "}"
                                        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_1[4 * q_2 + i])));
                                    #else
                                    asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                        "cvt.rn.f16x2.e2m1x2 %0, b0;"
                                        "cvt.rn.f16x2.e2m1x2 %1, b1;"
                                        "cvt.rn.f16x2.e2m1x2 %2, b2;"
                                        "cvt.rn.f16x2.e2m1x2 %3, b3;"
                                        "}"
                                        : "=r"(_fp4_bf16x8_16[0]), "=r"(_fp4_bf16x8_16[1]), "=r"(_fp4_bf16x8_16[2]), "=r"(_fp4_bf16x8_16[3]) : "r"((uint32_t)(raw_1[4 * q_2 + i])));
                                    {
                                    uint16_t lo = (uint16_t)_fp4_bf16x8_16[0], hi = (uint16_t)(_fp4_bf16x8_16[0] >> 16);
                                    float f0, f1;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[0]) : "f"(f1), "f"(f0));
                                    }
                                    {
                                    uint16_t lo = (uint16_t)_fp4_bf16x8_16[1], hi = (uint16_t)(_fp4_bf16x8_16[1] >> 16);
                                    float f0, f1;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[1]) : "f"(f1), "f"(f0));
                                    }
                                    {
                                    uint16_t lo = (uint16_t)_fp4_bf16x8_16[2], hi = (uint16_t)(_fp4_bf16x8_16[2] >> 16);
                                    float f0, f1;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[2]) : "f"(f1), "f"(f0));
                                    }
                                    {
                                    uint16_t lo = (uint16_t)_fp4_bf16x8_16[3], hi = (uint16_t)(_fp4_bf16x8_16[3] >> 16);
                                    float f0, f1;
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_16[3]) : "f"(f1), "f"(f0));
                                    }
                                    #endif
                                    }
                                    unsigned int w_1[4];
                                    #pragma unroll
                                    for (int j = 0; j < 4; j++) {
                                        uint32_t _bf16x2_mul_64;
                                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(_fp4_bf16x8_16[j]), "r"(sc_1[2 * q_2 + (i >> 1)]));
                                        w_1[j] = _bf16x2_mul_64;
                                    }
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                        "r"(plane + ((4 * (q_2 & 1) + i ^ tm7) << 4)), "r"(*reinterpret_cast<uint32_t*>(&w_1[0])), "r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 3])));
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(v_full_addr + (vs) * 8);
                            mbarrier_arrive(kv_raw_empty_addr + (st_1) * 8);
                        }
                        tile_n = tile_n + 1;
                        vtile_t = vtile_t + 1;
                    }
                }
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 12) {
        { // producer_main
            unsigned int total_units_l = total_q / seqlen_q * num_kv_heads;
            int load_stage = 0;
            int load_phase = 1;
            int kt_l = 0;
            int vt_l = 0;
            int item_par_1 = 0;
            int me_phase = 1;
            unsigned int kb[1];
            int ki[1];
            unsigned int first_idx = blockIdx.x / 4;
            unsigned int zero_u = 0;
            unsigned int unit_sel = ((first_idx < total_units_l) ? first_idx : zero_u);
            int unit_f = unit_sel;
            int request_f = unit_f / num_kv_heads;
            int kv_head_f = unit_f % num_kv_heads;
            int lane_0 = lane;
            int jpos = lane_0 & 15;
            int thalf = lane_0 >> 4;
            int tok_e = thalf;
            int tok_c = ((tok_e < seqlen_q) ? tok_e : seqlen_q - 1);
            int id_raw = task_kind[(request_f * seqlen_q + tok_c) * q2k_token_stride + kv_head_f * q2k_head_stride + jpos];
            int id_tok = ((tok_e < seqlen_q) ? id_raw : -1);
            int id_c = ((id_tok >= 0) ? id_tok : 0);
            int page_raw = kv_indices[request_f * msa_max_pages + id_c];
            int page_e = ((id_tok >= 0) ? page_raw : -1);
            int id_e = ((page_e < 0) ? -1 : id_tok);
            kb[0] = (unsigned int)(-1 - id_e);
            ki[0] = ((id_e < 0) ? 0 : page_e << 3 | tok_e);
            unsigned int ck = kb[0];
            int ci = ki[0];
            unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, ck, 1);
            unsigned int pb = _shfl_xor_10;
            int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, ci, 1);
            int pc = _shfl_xor_11;
            int r = (int)(pb > ck || pb == ck && pc < ci);
            int po = r;
            int lower = (int)((lane_0 & 1) == 0);
            int desc = (int)((lane_0 & 2) == 0);
            int wmax = desc * lower + (1 - desc) * (1 - lower);
            int tk = ((wmax != 0) ? po : 1 - po);
            kb[0] = ((tk != 0) ? pb : ck);
            ki[0] = ((tk != 0) ? pc : ci);
            unsigned int ck_1 = kb[0];
            int ci_2 = ki[0];
            unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, ck_1, 2);
            unsigned int pb_3 = _shfl_xor_12;
            int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, ci_2, 2);
            int pc_4 = _shfl_xor_13;
            int r_5 = (int)(pb_3 > ck_1 || pb_3 == ck_1 && pc_4 < ci_2);
            int po_6 = r_5;
            int lower_7 = (int)((lane_0 & 2) == 0);
            int desc_8 = (int)((lane_0 & 4) == 0);
            int wmax_9 = desc_8 * lower_7 + (1 - desc_8) * (1 - lower_7);
            int tk_10 = ((wmax_9 != 0) ? po_6 : 1 - po_6);
            kb[0] = ((tk_10 != 0) ? pb_3 : ck_1);
            ki[0] = ((tk_10 != 0) ? pc_4 : ci_2);
            unsigned int ck_11 = kb[0];
            int ci_12 = ki[0];
            unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, ck_11, 1);
            unsigned int pb_13 = _shfl_xor_14;
            int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, ci_12, 1);
            int pc_14 = _shfl_xor_15;
            int r_15 = (int)(pb_13 > ck_11 || pb_13 == ck_11 && pc_14 < ci_12);
            int po_16 = r_15;
            int lower_17 = (int)((lane_0 & 1) == 0);
            int desc_18 = (int)((lane_0 & 4) == 0);
            int wmax_19 = desc_18 * lower_17 + (1 - desc_18) * (1 - lower_17);
            int tk_20 = ((wmax_19 != 0) ? po_16 : 1 - po_16);
            kb[0] = ((tk_20 != 0) ? pb_13 : ck_11);
            ki[0] = ((tk_20 != 0) ? pc_14 : ci_12);
            unsigned int ck_21 = kb[0];
            int ci_22 = ki[0];
            unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, ck_21, 4);
            unsigned int pb_23 = _shfl_xor_16;
            int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, ci_22, 4);
            int pc_24 = _shfl_xor_17;
            int r_25 = (int)(pb_23 > ck_21 || pb_23 == ck_21 && pc_24 < ci_22);
            int po_26 = r_25;
            int lower_27 = (int)((lane_0 & 4) == 0);
            int desc_28 = (int)((lane_0 & 8) == 0);
            int wmax_29 = desc_28 * lower_27 + (1 - desc_28) * (1 - lower_27);
            int tk_30 = ((wmax_29 != 0) ? po_26 : 1 - po_26);
            kb[0] = ((tk_30 != 0) ? pb_23 : ck_21);
            ki[0] = ((tk_30 != 0) ? pc_24 : ci_22);
            unsigned int ck_31 = kb[0];
            int ci_32 = ki[0];
            unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, ck_31, 2);
            unsigned int pb_33 = _shfl_xor_18;
            int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, ci_32, 2);
            int pc_34 = _shfl_xor_19;
            int r_35 = (int)(pb_33 > ck_31 || pb_33 == ck_31 && pc_34 < ci_32);
            int po_36 = r_35;
            int lower_37 = (int)((lane_0 & 2) == 0);
            int desc_38 = (int)((lane_0 & 8) == 0);
            int wmax_39 = desc_38 * lower_37 + (1 - desc_38) * (1 - lower_37);
            int tk_40 = ((wmax_39 != 0) ? po_36 : 1 - po_36);
            kb[0] = ((tk_40 != 0) ? pb_33 : ck_31);
            ki[0] = ((tk_40 != 0) ? pc_34 : ci_32);
            unsigned int ck_41 = kb[0];
            int ci_42 = ki[0];
            unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, ck_41, 1);
            unsigned int pb_43 = _shfl_xor_20;
            int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, ci_42, 1);
            int pc_44 = _shfl_xor_21;
            int r_45 = (int)(pb_43 > ck_41 || pb_43 == ck_41 && pc_44 < ci_42);
            int po_46 = r_45;
            int lower_47 = (int)((lane_0 & 1) == 0);
            int desc_48 = (int)((lane_0 & 8) == 0);
            int wmax_49 = desc_48 * lower_47 + (1 - desc_48) * (1 - lower_47);
            int tk_50 = ((wmax_49 != 0) ? po_46 : 1 - po_46);
            kb[0] = ((tk_50 != 0) ? pb_43 : ck_41);
            ki[0] = ((tk_50 != 0) ? pc_44 : ci_42);
            unsigned int ck_51 = kb[0];
            int ci_52 = ki[0];
            unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, ck_51, 8);
            unsigned int pb_53 = _shfl_xor_22;
            int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, ci_52, 8);
            int pc_54 = _shfl_xor_23;
            int r_55 = (int)(pb_53 > ck_51 || pb_53 == ck_51 && pc_54 < ci_52);
            int po_56 = r_55;
            int lower_57 = (int)((lane_0 & 8) == 0);
            int desc_58 = (int)((lane_0 & 16) == 0);
            int wmax_59 = desc_58 * lower_57 + (1 - desc_58) * (1 - lower_57);
            int tk_60 = ((wmax_59 != 0) ? po_56 : 1 - po_56);
            kb[0] = ((tk_60 != 0) ? pb_53 : ck_51);
            ki[0] = ((tk_60 != 0) ? pc_54 : ci_52);
            unsigned int ck_61 = kb[0];
            int ci_62 = ki[0];
            unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, ck_61, 4);
            unsigned int pb_63 = _shfl_xor_24;
            int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, ci_62, 4);
            int pc_64 = _shfl_xor_25;
            int r_65 = (int)(pb_63 > ck_61 || pb_63 == ck_61 && pc_64 < ci_62);
            int po_66 = r_65;
            int lower_67 = (int)((lane_0 & 4) == 0);
            int desc_68 = (int)((lane_0 & 16) == 0);
            int wmax_69 = desc_68 * lower_67 + (1 - desc_68) * (1 - lower_67);
            int tk_70 = ((wmax_69 != 0) ? po_66 : 1 - po_66);
            kb[0] = ((tk_70 != 0) ? pb_63 : ck_61);
            ki[0] = ((tk_70 != 0) ? pc_64 : ci_62);
            unsigned int ck_71 = kb[0];
            int ci_72 = ki[0];
            unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, ck_71, 2);
            unsigned int pb_73 = _shfl_xor_26;
            int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, ci_72, 2);
            int pc_74 = _shfl_xor_27;
            int r_75 = (int)(pb_73 > ck_71 || pb_73 == ck_71 && pc_74 < ci_72);
            int po_76 = r_75;
            int lower_77 = (int)((lane_0 & 2) == 0);
            int desc_78 = (int)((lane_0 & 16) == 0);
            int wmax_79 = desc_78 * lower_77 + (1 - desc_78) * (1 - lower_77);
            int tk_80 = ((wmax_79 != 0) ? po_76 : 1 - po_76);
            kb[0] = ((tk_80 != 0) ? pb_73 : ck_71);
            ki[0] = ((tk_80 != 0) ? pc_74 : ci_72);
            unsigned int ck_81 = kb[0];
            int ci_82 = ki[0];
            unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, ck_81, 1);
            unsigned int pb_83 = _shfl_xor_28;
            int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, ci_82, 1);
            int pc_84 = _shfl_xor_29;
            int r_85 = (int)(pb_83 > ck_81 || pb_83 == ck_81 && pc_84 < ci_82);
            int po_86 = r_85;
            int lower_87 = (int)((lane_0 & 1) == 0);
            int desc_88 = (int)((lane_0 & 16) == 0);
            int wmax_89 = desc_88 * lower_87 + (1 - desc_88) * (1 - lower_87);
            int tk_90 = ((wmax_89 != 0) ? po_86 : 1 - po_86);
            kb[0] = ((tk_90 != 0) ? pb_83 : ck_81);
            ki[0] = ((tk_90 != 0) ? pc_84 : ci_82);
            unsigned int ck_91 = kb[0];
            int ci_92 = ki[0];
            unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, ck_91, 16);
            unsigned int pb_93 = _shfl_xor_30;
            int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, ci_92, 16);
            int pc_94 = _shfl_xor_31;
            int r_95 = (int)(pb_93 > ck_91 || pb_93 == ck_91 && pc_94 < ci_92);
            int po_96 = r_95;
            int lower_97 = (int)((lane_0 & 16) == 0);
            int desc_98 = (int)((lane_0 & 0) == 0);
            int wmax_99 = desc_98 * lower_97 + (1 - desc_98) * (1 - lower_97);
            int tk_100 = ((wmax_99 != 0) ? po_96 : 1 - po_96);
            kb[0] = ((tk_100 != 0) ? pb_93 : ck_91);
            ki[0] = ((tk_100 != 0) ? pc_94 : ci_92);
            unsigned int ck_101 = kb[0];
            int ci_102 = ki[0];
            unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, ck_101, 8);
            unsigned int pb_103 = _shfl_xor_32;
            int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, ci_102, 8);
            int pc_104 = _shfl_xor_33;
            int r_105 = (int)(pb_103 > ck_101 || pb_103 == ck_101 && pc_104 < ci_102);
            int po_106 = r_105;
            int lower_107 = (int)((lane_0 & 8) == 0);
            int desc_108 = (int)((lane_0 & 0) == 0);
            int wmax_109 = desc_108 * lower_107 + (1 - desc_108) * (1 - lower_107);
            int tk_110 = ((wmax_109 != 0) ? po_106 : 1 - po_106);
            kb[0] = ((tk_110 != 0) ? pb_103 : ck_101);
            ki[0] = ((tk_110 != 0) ? pc_104 : ci_102);
            unsigned int ck_111 = kb[0];
            int ci_112 = ki[0];
            unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, ck_111, 4);
            unsigned int pb_113 = _shfl_xor_34;
            int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, ci_112, 4);
            int pc_114 = _shfl_xor_35;
            int r_115 = (int)(pb_113 > ck_111 || pb_113 == ck_111 && pc_114 < ci_112);
            int po_116 = r_115;
            int lower_117 = (int)((lane_0 & 4) == 0);
            int desc_118 = (int)((lane_0 & 0) == 0);
            int wmax_119 = desc_118 * lower_117 + (1 - desc_118) * (1 - lower_117);
            int tk_120 = ((wmax_119 != 0) ? po_116 : 1 - po_116);
            kb[0] = ((tk_120 != 0) ? pb_113 : ck_111);
            ki[0] = ((tk_120 != 0) ? pc_114 : ci_112);
            unsigned int ck_121 = kb[0];
            int ci_122 = ki[0];
            unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, ck_121, 2);
            unsigned int pb_123 = _shfl_xor_36;
            int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, ci_122, 2);
            int pc_124 = _shfl_xor_37;
            int r_125 = (int)(pb_123 > ck_121 || pb_123 == ck_121 && pc_124 < ci_122);
            int po_126 = r_125;
            int lower_127 = (int)((lane_0 & 2) == 0);
            int desc_128 = (int)((lane_0 & 0) == 0);
            int wmax_129 = desc_128 * lower_127 + (1 - desc_128) * (1 - lower_127);
            int tk_130 = ((wmax_129 != 0) ? po_126 : 1 - po_126);
            kb[0] = ((tk_130 != 0) ? pb_123 : ck_121);
            ki[0] = ((tk_130 != 0) ? pc_124 : ci_122);
            unsigned int ck_131 = kb[0];
            int ci_132 = ki[0];
            unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, ck_131, 1);
            unsigned int pb_133 = _shfl_xor_38;
            int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, ci_132, 1);
            int pc_134 = _shfl_xor_39;
            int r_135 = (int)(pb_133 > ck_131 || pb_133 == ck_131 && pc_134 < ci_132);
            int po_136 = r_135;
            int lower_137 = (int)((lane_0 & 1) == 0);
            int desc_138 = (int)((lane_0 & 0) == 0);
            int wmax_139 = desc_138 * lower_137 + (1 - desc_138) * (1 - lower_137);
            int tk_140 = ((wmax_139 != 0) ? po_136 : 1 - po_136);
            kb[0] = ((tk_140 != 0) ? pb_133 : ck_131);
            ki[0] = ((tk_140 != 0) ? pc_134 : ci_132);
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x / 4; work_idx_l < total_units_l; work_idx_l += gridDim.x / 4) {
                int unit_l = work_idx_l;
                int request_l = unit_l / num_kv_heads;
                int kv_head_l = unit_l % num_kv_heads;
                int group_size_l = num_q_heads / num_kv_heads;
                int kv_len_l = task_kv_head[request_l];
                __syncwarp();
                mbarrier_wait(meta_empty_addr + (item_par_1) * 8, me_phase);
                int meta_base_l = item_par_1 * 32;
                int lane_1 = lane;
                unsigned int lane_one = 1;
                unsigned int below = (lane_one << (unsigned int)lane_1) - 1;
                unsigned int incl = below | lane_one << (unsigned int)lane_1;
                smem_mask[meta_base_l + lane_1] = 0;
                __syncwarp();
                int base_prev = 0;
                unsigned int k_i = kb[0];
                int p_i = ki[0];
                unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, k_i, 1, 32);
                unsigned int pk_up = _shfl_up_0;
                int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, p_i, 1, 32);
                int pp_up = _shfl_up_1;
                unsigned int pk = pk_up;
                int pp = pp_up;
                int first = 0;
                first = (int)(lane_1 == 0);
                int valid = (int)(k_i != 0);
                int diff_id = (int)(k_i != pk) | first;
                int diff_tok = (int)((p_i & 7) != (pp & 7));
                int newseg = valid & diff_id;
                int contrib = valid & (diff_id | diff_tok);
                int rank_p = cta_rank;
                int id_cls = -1 - (int)k_i;
                int keep = (int)((id_cls & 3) == rank_p);
                newseg = newseg & keep;
                contrib = contrib & keep;
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, newseg != 0);
                unsigned int fl = _vote_0;
                int _popc_0 = __popc(fl & incl);
                int slot_1 = base_prev + _popc_0 - 1;
                if (newseg != 0) {
                    int id_i = -1 - (int)k_i;
                    smem_page[meta_base_l + slot_1] = (p_i >> 3) * num_kv_heads + kv_head_l;
                    smem_bstart[meta_base_l + slot_1] = id_i * 128;
                }
                if (contrib != 0) {
                    uint32_t _shared_atomic_old_0;
                    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((smem_mask_addr + 4 * (meta_base_l + slot_1)))), "r"(static_cast<uint32_t>(lane_one << (unsigned int)(p_i & 7))) : "memory");
                }
                int _popc_1 = __popc(fl);
                base_prev = base_prev + _popc_1;
                int count = base_prev;
                __syncwarp();
                if (lane_1 == 0) {
                    if (count % 2 == 1) {
                        smem_page[meta_base_l + count] = smem_page[meta_base_l + count - 1];
                        smem_bstart[meta_base_l + count] = smem_bstart[meta_base_l + count - 1];
                    }
                }
                int npairs_1 = (count + count % 2) / 2;
                __syncwarp();
                int npairs_l = npairs_1;
                if (lane == 0) {
                    smem_kvlen[item_par_1] = kv_len_l;
                }
                __syncwarp();
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        #pragma unroll
                        for (int k_tile = 0; k_tile < 2; k_tile++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                            int physical_page = smem_page[meta_base_l + k_tile] / num_kv_heads;
                            int kv_head_1 = smem_page[meta_base_l + k_tile] - physical_page * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_1, physical_page, kv_raw_full_addr + (kt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_1, physical_page, kv_raw_full_addr + (kt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                mbarrier_wait(q_empty_addr + (item_par_1) * 8, me_phase);
                if (elect_sync()) {
                    int q_row0_l = request_l * seqlen_q * num_q_heads + kv_head_l * group_size_l;
                    mbarrier_arrive_expect_tx(q_full_addr + (item_par_1) * 8, 8192);
                    int q_row_t = q_row0_l;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(2 * item_par_1 * 2 * 2048), (&Q), 0, q_row_t, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 + 1) * 2 * 2048), (&Q), 0, q_row_t, 1, q_full_addr + (item_par_1) * 8);
                    int q_row_t_0 = q_row0_l + num_q_heads;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 * 2 + 1) * 2048), (&Q), 0, q_row_t_0, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(((2 * item_par_1 + 1) * 2 + 1) * 2048), (&Q), 0, q_row_t_0, 1, q_full_addr + (item_par_1) * 8);
                }
                if (elect_sync()) {
                    if (npairs_l > 1) {
                        #pragma unroll
                        for (int k_tile_1 = 0; k_tile_1 < 2; k_tile_1++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                            int physical_page_1 = smem_page[meta_base_l + 2 + k_tile_1] / num_kv_heads;
                            int kv_head_2 = smem_page[meta_base_l + 2 + k_tile_1] - physical_page_1 * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_2, physical_page_1, kv_raw_full_addr + (kt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_2, physical_page_1, kv_raw_full_addr + (kt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                if (elect_sync()) {
                    smem_npairs[item_par_1] = npairs_l;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(meta_full_addr + (item_par_1) * 8);
                }
                unsigned int nxt_idx = work_idx_l + (unsigned int)(gridDim.x / 4);
                if (nxt_idx < total_units_l) {
                    int unit_n = nxt_idx;
                    int request_n = unit_n / num_kv_heads;
                    int kv_head_n = unit_n % num_kv_heads;
                    int lane_2 = lane;
                    int jpos_3 = lane_2 & 15;
                    int thalf_4 = lane_2 >> 4;
                    int tok_e_5 = thalf_4;
                    int tok_c_6 = ((tok_e_5 < seqlen_q) ? tok_e_5 : seqlen_q - 1);
                    int id_raw_7 = task_kind[(request_n * seqlen_q + tok_c_6) * q2k_token_stride + kv_head_n * q2k_head_stride + jpos_3];
                    int id_tok_8 = ((tok_e_5 < seqlen_q) ? id_raw_7 : -1);
                    int id_c_9 = ((id_tok_8 >= 0) ? id_tok_8 : 0);
                    int page_raw_10 = kv_indices[request_n * msa_max_pages + id_c_9];
                    int page_e_11 = ((id_tok_8 >= 0) ? page_raw_10 : -1);
                    int id_e_12 = ((page_e_11 < 0) ? -1 : id_tok_8);
                    kb[0] = (unsigned int)(-1 - id_e_12);
                    ki[0] = ((id_e_12 < 0) ? 0 : page_e_11 << 3 | tok_e_5);
                    unsigned int ck_13 = kb[0];
                    int ci_14 = ki[0];
                    unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, ck_13, 1);
                    unsigned int pb_15 = _shfl_xor_40;
                    int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, ci_14, 1);
                    int pc_16 = _shfl_xor_41;
                    int r_17 = (int)(pb_15 > ck_13 || pb_15 == ck_13 && pc_16 < ci_14);
                    int po_18 = r_17;
                    int lower_19 = (int)((lane_2 & 1) == 0);
                    int desc_20 = (int)((lane_2 & 2) == 0);
                    int wmax_21 = desc_20 * lower_19 + (1 - desc_20) * (1 - lower_19);
                    int tk_22 = ((wmax_21 != 0) ? po_18 : 1 - po_18);
                    kb[0] = ((tk_22 != 0) ? pb_15 : ck_13);
                    ki[0] = ((tk_22 != 0) ? pc_16 : ci_14);
                    unsigned int ck_23 = kb[0];
                    int ci_24 = ki[0];
                    unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, ck_23, 2);
                    unsigned int pb_25 = _shfl_xor_42;
                    int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, ci_24, 2);
                    int pc_26 = _shfl_xor_43;
                    int r_27 = (int)(pb_25 > ck_23 || pb_25 == ck_23 && pc_26 < ci_24);
                    int po_28 = r_27;
                    int lower_29 = (int)((lane_2 & 2) == 0);
                    int desc_30 = (int)((lane_2 & 4) == 0);
                    int wmax_31 = desc_30 * lower_29 + (1 - desc_30) * (1 - lower_29);
                    int tk_32 = ((wmax_31 != 0) ? po_28 : 1 - po_28);
                    kb[0] = ((tk_32 != 0) ? pb_25 : ck_23);
                    ki[0] = ((tk_32 != 0) ? pc_26 : ci_24);
                    unsigned int ck_33 = kb[0];
                    int ci_34 = ki[0];
                    unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, ck_33, 1);
                    unsigned int pb_35 = _shfl_xor_44;
                    int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, ci_34, 1);
                    int pc_36 = _shfl_xor_45;
                    int r_37 = (int)(pb_35 > ck_33 || pb_35 == ck_33 && pc_36 < ci_34);
                    int po_38 = r_37;
                    int lower_39 = (int)((lane_2 & 1) == 0);
                    int desc_40 = (int)((lane_2 & 4) == 0);
                    int wmax_41 = desc_40 * lower_39 + (1 - desc_40) * (1 - lower_39);
                    int tk_42 = ((wmax_41 != 0) ? po_38 : 1 - po_38);
                    kb[0] = ((tk_42 != 0) ? pb_35 : ck_33);
                    ki[0] = ((tk_42 != 0) ? pc_36 : ci_34);
                    unsigned int ck_43 = kb[0];
                    int ci_44 = ki[0];
                    unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, ck_43, 4);
                    unsigned int pb_45 = _shfl_xor_46;
                    int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, ci_44, 4);
                    int pc_46 = _shfl_xor_47;
                    int r_47 = (int)(pb_45 > ck_43 || pb_45 == ck_43 && pc_46 < ci_44);
                    int po_48 = r_47;
                    int lower_49 = (int)((lane_2 & 4) == 0);
                    int desc_50 = (int)((lane_2 & 8) == 0);
                    int wmax_51 = desc_50 * lower_49 + (1 - desc_50) * (1 - lower_49);
                    int tk_52 = ((wmax_51 != 0) ? po_48 : 1 - po_48);
                    kb[0] = ((tk_52 != 0) ? pb_45 : ck_43);
                    ki[0] = ((tk_52 != 0) ? pc_46 : ci_44);
                    unsigned int ck_53 = kb[0];
                    int ci_54 = ki[0];
                    unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, ck_53, 2);
                    unsigned int pb_55 = _shfl_xor_48;
                    int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, ci_54, 2);
                    int pc_56 = _shfl_xor_49;
                    int r_57 = (int)(pb_55 > ck_53 || pb_55 == ck_53 && pc_56 < ci_54);
                    int po_58 = r_57;
                    int lower_59 = (int)((lane_2 & 2) == 0);
                    int desc_60 = (int)((lane_2 & 8) == 0);
                    int wmax_61 = desc_60 * lower_59 + (1 - desc_60) * (1 - lower_59);
                    int tk_62 = ((wmax_61 != 0) ? po_58 : 1 - po_58);
                    kb[0] = ((tk_62 != 0) ? pb_55 : ck_53);
                    ki[0] = ((tk_62 != 0) ? pc_56 : ci_54);
                    unsigned int ck_63 = kb[0];
                    int ci_64 = ki[0];
                    unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, ck_63, 1);
                    unsigned int pb_65 = _shfl_xor_50;
                    int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, ci_64, 1);
                    int pc_66 = _shfl_xor_51;
                    int r_67 = (int)(pb_65 > ck_63 || pb_65 == ck_63 && pc_66 < ci_64);
                    int po_68 = r_67;
                    int lower_69 = (int)((lane_2 & 1) == 0);
                    int desc_70 = (int)((lane_2 & 8) == 0);
                    int wmax_71 = desc_70 * lower_69 + (1 - desc_70) * (1 - lower_69);
                    int tk_72 = ((wmax_71 != 0) ? po_68 : 1 - po_68);
                    kb[0] = ((tk_72 != 0) ? pb_65 : ck_63);
                    ki[0] = ((tk_72 != 0) ? pc_66 : ci_64);
                    unsigned int ck_73 = kb[0];
                    int ci_74 = ki[0];
                    unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, ck_73, 8);
                    unsigned int pb_75 = _shfl_xor_52;
                    int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, ci_74, 8);
                    int pc_76 = _shfl_xor_53;
                    int r_77 = (int)(pb_75 > ck_73 || pb_75 == ck_73 && pc_76 < ci_74);
                    int po_78 = r_77;
                    int lower_79 = (int)((lane_2 & 8) == 0);
                    int desc_80 = (int)((lane_2 & 16) == 0);
                    int wmax_81 = desc_80 * lower_79 + (1 - desc_80) * (1 - lower_79);
                    int tk_82 = ((wmax_81 != 0) ? po_78 : 1 - po_78);
                    kb[0] = ((tk_82 != 0) ? pb_75 : ck_73);
                    ki[0] = ((tk_82 != 0) ? pc_76 : ci_74);
                    unsigned int ck_83 = kb[0];
                    int ci_84 = ki[0];
                    unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, ck_83, 4);
                    unsigned int pb_85 = _shfl_xor_54;
                    int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, ci_84, 4);
                    int pc_86 = _shfl_xor_55;
                    int r_87 = (int)(pb_85 > ck_83 || pb_85 == ck_83 && pc_86 < ci_84);
                    int po_88 = r_87;
                    int lower_89 = (int)((lane_2 & 4) == 0);
                    int desc_90 = (int)((lane_2 & 16) == 0);
                    int wmax_91 = desc_90 * lower_89 + (1 - desc_90) * (1 - lower_89);
                    int tk_92 = ((wmax_91 != 0) ? po_88 : 1 - po_88);
                    kb[0] = ((tk_92 != 0) ? pb_85 : ck_83);
                    ki[0] = ((tk_92 != 0) ? pc_86 : ci_84);
                    unsigned int ck_93 = kb[0];
                    int ci_94 = ki[0];
                    unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, ck_93, 2);
                    unsigned int pb_95 = _shfl_xor_56;
                    int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, ci_94, 2);
                    int pc_96 = _shfl_xor_57;
                    int r_97 = (int)(pb_95 > ck_93 || pb_95 == ck_93 && pc_96 < ci_94);
                    int po_98 = r_97;
                    int lower_99 = (int)((lane_2 & 2) == 0);
                    int desc_100 = (int)((lane_2 & 16) == 0);
                    int wmax_101 = desc_100 * lower_99 + (1 - desc_100) * (1 - lower_99);
                    int tk_102 = ((wmax_101 != 0) ? po_98 : 1 - po_98);
                    kb[0] = ((tk_102 != 0) ? pb_95 : ck_93);
                    ki[0] = ((tk_102 != 0) ? pc_96 : ci_94);
                    unsigned int ck_103 = kb[0];
                    int ci_104 = ki[0];
                    unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, ck_103, 1);
                    unsigned int pb_105 = _shfl_xor_58;
                    int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, ci_104, 1);
                    int pc_106 = _shfl_xor_59;
                    int r_107 = (int)(pb_105 > ck_103 || pb_105 == ck_103 && pc_106 < ci_104);
                    int po_108 = r_107;
                    int lower_109 = (int)((lane_2 & 1) == 0);
                    int desc_110 = (int)((lane_2 & 16) == 0);
                    int wmax_111 = desc_110 * lower_109 + (1 - desc_110) * (1 - lower_109);
                    int tk_112 = ((wmax_111 != 0) ? po_108 : 1 - po_108);
                    kb[0] = ((tk_112 != 0) ? pb_105 : ck_103);
                    ki[0] = ((tk_112 != 0) ? pc_106 : ci_104);
                    unsigned int ck_113 = kb[0];
                    int ci_114 = ki[0];
                    unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, ck_113, 16);
                    unsigned int pb_115 = _shfl_xor_60;
                    int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, ci_114, 16);
                    int pc_116 = _shfl_xor_61;
                    int r_117 = (int)(pb_115 > ck_113 || pb_115 == ck_113 && pc_116 < ci_114);
                    int po_118 = r_117;
                    int lower_119 = (int)((lane_2 & 16) == 0);
                    int desc_120 = (int)((lane_2 & 0) == 0);
                    int wmax_121 = desc_120 * lower_119 + (1 - desc_120) * (1 - lower_119);
                    int tk_122 = ((wmax_121 != 0) ? po_118 : 1 - po_118);
                    kb[0] = ((tk_122 != 0) ? pb_115 : ck_113);
                    ki[0] = ((tk_122 != 0) ? pc_116 : ci_114);
                    unsigned int ck_123 = kb[0];
                    int ci_124 = ki[0];
                    unsigned int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, ck_123, 8);
                    unsigned int pb_125 = _shfl_xor_62;
                    int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, ci_124, 8);
                    int pc_126 = _shfl_xor_63;
                    int r_127 = (int)(pb_125 > ck_123 || pb_125 == ck_123 && pc_126 < ci_124);
                    int po_128 = r_127;
                    int lower_129 = (int)((lane_2 & 8) == 0);
                    int desc_130 = (int)((lane_2 & 0) == 0);
                    int wmax_131 = desc_130 * lower_129 + (1 - desc_130) * (1 - lower_129);
                    int tk_132 = ((wmax_131 != 0) ? po_128 : 1 - po_128);
                    kb[0] = ((tk_132 != 0) ? pb_125 : ck_123);
                    ki[0] = ((tk_132 != 0) ? pc_126 : ci_124);
                    unsigned int ck_133 = kb[0];
                    int ci_134 = ki[0];
                    unsigned int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, ck_133, 4);
                    unsigned int pb_135 = _shfl_xor_64;
                    int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, ci_134, 4);
                    int pc_136 = _shfl_xor_65;
                    int r_137 = (int)(pb_135 > ck_133 || pb_135 == ck_133 && pc_136 < ci_134);
                    int po_138 = r_137;
                    int lower_139 = (int)((lane_2 & 4) == 0);
                    int desc_140 = (int)((lane_2 & 0) == 0);
                    int wmax_141 = desc_140 * lower_139 + (1 - desc_140) * (1 - lower_139);
                    int tk_142 = ((wmax_141 != 0) ? po_138 : 1 - po_138);
                    kb[0] = ((tk_142 != 0) ? pb_135 : ck_133);
                    ki[0] = ((tk_142 != 0) ? pc_136 : ci_134);
                    unsigned int ck_143 = kb[0];
                    int ci_144 = ki[0];
                    unsigned int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, ck_143, 2);
                    unsigned int pb_145 = _shfl_xor_66;
                    int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, ci_144, 2);
                    int pc_146 = _shfl_xor_67;
                    int r_147 = (int)(pb_145 > ck_143 || pb_145 == ck_143 && pc_146 < ci_144);
                    int po_148 = r_147;
                    int lower_149 = (int)((lane_2 & 2) == 0);
                    int desc_150 = (int)((lane_2 & 0) == 0);
                    int wmax_151 = desc_150 * lower_149 + (1 - desc_150) * (1 - lower_149);
                    int tk_152 = ((wmax_151 != 0) ? po_148 : 1 - po_148);
                    kb[0] = ((tk_152 != 0) ? pb_145 : ck_143);
                    ki[0] = ((tk_152 != 0) ? pc_146 : ci_144);
                    unsigned int ck_153 = kb[0];
                    int ci_154 = ki[0];
                    unsigned int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, ck_153, 1);
                    unsigned int pb_155 = _shfl_xor_68;
                    int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, ci_154, 1);
                    int pc_156 = _shfl_xor_69;
                    int r_157 = (int)(pb_155 > ck_153 || pb_155 == ck_153 && pc_156 < ci_154);
                    int po_158 = r_157;
                    int lower_159 = (int)((lane_2 & 1) == 0);
                    int desc_160 = (int)((lane_2 & 0) == 0);
                    int wmax_161 = desc_160 * lower_159 + (1 - desc_160) * (1 - lower_159);
                    int tk_162 = ((wmax_161 != 0) ? po_158 : 1 - po_158);
                    kb[0] = ((tk_162 != 0) ? pb_155 : ck_153);
                    ki[0] = ((tk_162 != 0) ? pc_156 : ci_154);
                }
                __syncwarp();
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pair_3 = 0; pair_3 < 16; pair_3++) {
                        if (npairs_l <= pair_3) {
                            break;
                        }
                        if (npairs_l > pair_3 + 2) {
                            #pragma unroll
                            for (int k_tile_2 = 0; k_tile_2 < 2; k_tile_2++) {
                                mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                                mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                                int physical_page_2 = smem_page[meta_base_l + pair_3 * 2 + 4 + k_tile_2] / num_kv_heads;
                                int kv_head_3 = smem_page[meta_base_l + pair_3 * 2 + 4 + k_tile_2] - physical_page_2 * num_kv_heads;
                                tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_3, physical_page_2, kv_raw_full_addr + (kt_l % 16) * 8);
                                tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_3, physical_page_2, kv_raw_full_addr + (kt_l % 16) * 8);
                                load_stage += 1;
                                if (load_stage == 8) { load_stage = 0; load_phase ^= 1; }
                                kt_l = kt_l + 1;
                            }
                        }
                        #pragma unroll
                        for (int v_tile = 0; v_tile < 2; v_tile++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (16 + vt_l % 16) * 8, 9216);
                            int physical_page_3 = smem_page[meta_base_l + pair_3 * 2 + v_tile] / num_kv_heads;
                            int kv_head_4 = smem_page[meta_base_l + pair_3 * 2 + v_tile] - physical_page_3 * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&V), 0, 0, kv_head_4, physical_page_3, kv_raw_full_addr + (16 + vt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&V_scale), 0, 0, kv_head_4, physical_page_3, kv_raw_full_addr + (16 + vt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 8) { load_stage = 0; load_phase ^= 1; }
                            vt_l = vt_l + 1;
                        }
                    }
                }
                item_par_1 = 1 - item_par_1;
                if (item_par_1 == 0) {
                    me_phase = 1 - me_phase;
                }
            }
        }
    }
    // ---- Role: mma_s ----
    if (warp == 13) {
        { // mma_s_main
            unsigned int total_units_s = total_q / seqlen_q * num_kv_heads;
            int item_stage_s = 0;
            int item_phase_s = 0;
            int s_stage_m = 0;
            int s_phase_m = 1;
            int ktile_n = 0;
            #pragma unroll 1
            for (unsigned int work_idx_s = blockIdx.x / 4; work_idx_s < total_units_s; work_idx_s += gridDim.x / 4) {
                mbarrier_wait(meta_full_addr + (item_stage_s) * 8, item_phase_s);
                int npairs_s = smem_npairs[item_stage_s];
                int q_buf_s = item_stage_s;
                int q_phase_s = item_phase_s;
                item_stage_s += 1;
                if (item_stage_s == 2) { item_stage_s = 0; item_phase_s ^= 1; }
                mbarrier_wait(q_full_addr + (q_buf_s) * 8, q_phase_s);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (npairs_s == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(q_empty_addr + (q_buf_s) * 8);
                    }
                }
                #pragma unroll 1
                for (int sp_1 = 0; sp_1 < 16; sp_1++) {
                    if (npairs_s <= sp_1) {
                        break;
                    }
                    mbarrier_wait(s_empty_addr + (s_stage_m) * 8, s_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int slot_2 = ktile_n % 2;
                    int use_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_2) * 8, use_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_0 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 256);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 64))), "r"(_mma_b_lo_0), "r"(tmem_tmem_k16 + 2 * slot_2 * 32), "r"(0));
                    int _mma_b_lo_1 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 256);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 64))), "r"(_mma_b_lo_1), "r"(tmem_tmem_k16 + (2 * slot_2 + 1) * 32), "r"(1));
                    elect_commit(k16_empty_addr + (slot_2) * 8);
                    ktile_n = ktile_n + 1;
                    int slot_0 = ktile_n % 2;
                    int use_1_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_0) * 8, use_1_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 256);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 64 + 32))), "r"(_mma_b_lo_2), "r"(tmem_tmem_k16 + 2 * slot_0 * 32), "r"(0));
                    int _mma_b_lo_3 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 256);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 64 + 32))), "r"(_mma_b_lo_3), "r"(tmem_tmem_k16 + (2 * slot_0 + 1) * 32), "r"(1));
                    elect_commit(k16_empty_addr + (slot_0) * 8);
                    ktile_n = ktile_n + 1;
                    if (sp_1 == npairs_s - 1) {
                        elect_commit(q_empty_addr + (q_buf_s) * 8);
                    }
                    elect_commit(s_full_addr + (s_stage_m) * 8);
                    s_stage_m += 1;
                    if (s_stage_m == 2) { s_stage_m = 0; s_phase_m ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 14) {
        { // mma_pv_main
            unsigned int total_units_v = total_q / seqlen_q * num_kv_heads;
            int item_stage_v = 0;
            int item_phase_v = 0;
            int pf_stage_v = 0;
            int pf_phase_v = 0;
            int vtile_n = 0;
            int have_prev_v = 0;
            unsigned int _phase_decode_done_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_v = blockIdx.x / 4; work_idx_v < total_units_v; work_idx_v += gridDim.x / 4) {
                mbarrier_wait(meta_full_addr + (item_stage_v) * 8, item_phase_v);
                int npairs_v = smem_npairs[item_stage_v];
                item_stage_v += 1;
                if (item_stage_v == 2) { item_stage_v = 0; item_phase_v ^= 1; }
                int first_pv = 1;
                int dd_waited = 0;
                #pragma unroll 1
                for (int pv = 0; pv < 16; pv++) {
                    if (npairs_v <= pv) {
                        break;
                    }
                    mbarrier_wait(p_full_addr + (pf_stage_v) * 8, pf_phase_v);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (dd_waited == 0) {
                        if (have_prev_v != 0) {
                            mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                            _phase_decode_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        dd_waited = 1;
                    }
                    int vt = vtile_n;
                    int vs_1 = vt % 3;
                    int vph_1 = vt / 3 & 1;
                    mbarrier_wait(v_full_addr + (vs_1) * 8, vph_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j = ((1) ? first_pv : 0);
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o), "r"(((init_j) ? 0 : 1)));
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_o + (32))), "r"(((init_j) ? 0 : 1)));
                    int _mma_a_lo_6 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_o + (64))), "r"(((init_j) ? 0 : 1)));
                    int _mma_a_lo_7 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 768;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_tmem_o + (96))), "r"(((init_j) ? 0 : 1)));
                    elect_commit(v_empty_addr + (vs_1) * 8);
                    int vt_0 = vtile_n + 1;
                    int vs_1_1 = vt_0 % 3;
                    int vph_2 = vt_0 / 3 & 1;
                    mbarrier_wait(v_full_addr + (vs_1_1) * 8, vph_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j_3 = ((0) ? first_pv : 0);
                    int _mma_a_lo_8 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_8 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 1) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_tmem_o), "r"(((init_j_3) ? 0 : 1)));
                    int _mma_a_lo_9 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 1) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem_o + (32))), "r"(((init_j_3) ? 0 : 1)));
                    int _mma_a_lo_10 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 1) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem_o + (64))), "r"(((init_j_3) ? 0 : 1)));
                    int _mma_a_lo_11 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_11 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 1) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "add.u32 alo, %0, 768;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem_o + (96))), "r"(((init_j_3) ? 0 : 1)));
                    elect_commit(v_empty_addr + (vs_1_1) * 8);
                    first_pv = 0;
                    vtile_n = vtile_n + 2;
                    elect_commit(pv_done_addr + (pv) * 8);
                    pf_stage_v += 1;
                    if (pf_stage_v == 2) { pf_stage_v = 0; pf_phase_v ^= 1; }
                }
                if (dd_waited == 0) {
                    if (have_prev_v != 0) {
                        mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                        _phase_decode_done_0 ^= 1;
                    }
                }
                #pragma unroll 1
                for (int spare = npairs_v; spare < 16; spare++) {
                    if (elect_sync()) {
                        mbarrier_arrive(pv_done_addr + (spare) * 8);
                    }
                }
                have_prev_v = 1;
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        { // idle_main
            unsigned int _phase_tmem_dead_0 = 0;
            mbarrier_wait(tmem_dead_addr, _phase_tmem_dead_0);
            _phase_tmem_dead_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }

    // Cleanup
}

} // extern "C"
