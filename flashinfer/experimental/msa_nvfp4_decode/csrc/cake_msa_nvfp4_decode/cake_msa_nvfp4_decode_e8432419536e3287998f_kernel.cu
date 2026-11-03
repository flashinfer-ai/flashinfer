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
#define TMEM_NCOLS 512
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_K16_OFFSET 128
#define TMEM_TMEM_O_OFFSET 256
#define NUM_RAW_RING_STAGES 10
#define NUM_S_RING_STAGES 1
#define NUM_P_RING_STAGES 1
#define NUM_ITEM_RING_STAGES 2
#define SMEM_SMEM_F32_OFF 1024
#define SMEM_SMEM_F32_STAGE_BYTES 2048
#define SMEM_SMEM_F32_STRIDE 2048
#define SMEM_SMEM_PAGE_OFF 3072
#define SMEM_SMEM_PAGE_STAGE_BYTES 512
#define SMEM_SMEM_PAGE_STRIDE 512
#define SMEM_SMEM_BSTART_OFF 3584
#define SMEM_SMEM_BSTART_STAGE_BYTES 512
#define SMEM_SMEM_BSTART_STRIDE 512
#define SMEM_SMEM_MASK_OFF 4096
#define SMEM_SMEM_MASK_STAGE_BYTES 512
#define SMEM_SMEM_MASK_STRIDE 512
#define SMEM_SMEM_NPAIRS_OFF 4608
#define SMEM_SMEM_NPAIRS_STAGE_BYTES 8
#define SMEM_SMEM_NPAIRS_STRIDE 8
#define SMEM_SMEM_KVLEN_OFF 4616
#define SMEM_SMEM_KVLEN_STAGE_BYTES 8
#define SMEM_SMEM_KVLEN_STRIDE 8
#define SMEM_SMEM_QU_OFF 5120
#define SMEM_SMEM_QU_STAGE_BYTES 8192
#define SMEM_SMEM_QU_STRIDE 8192
#define SMEM_SMEM_QTOK_OFF 5120
#define SMEM_SMEM_QTOK_STAGE_BYTES 2048
#define SMEM_SMEM_QTOK_STRIDE 2048
#define SMEM_SMEM_P_OFF 37888
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_SMEM_SCALE_OFF 70656
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 1024
#define SMEM_SMEM_RAW_OFF 80896
#define SMEM_SMEM_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_RAW_STRIDE 8192
#define SMEM_SMEM_V_OFF 162816
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_V_MN_OFF 162816
#define SMEM_SMEM_V_MN_STAGE_BYTES 32768
#define SMEM_SMEM_V_MN_STRIDE 32768
#define SMEM_TOTAL 228352
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_msa_nvfp4_decode_e8432419536e3287998f(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ task_kind, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages, int q2k_token_stride, int q2k_head_stride)
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
    #define k16_full_addr (mbar_base + 384)
    #define k16_empty_addr (mbar_base + 400)
    #define v_full_addr (mbar_base + 416)
    #define v_empty_addr (mbar_base + 432)
    #define s_full_addr (mbar_base + 448)
    #define s_empty_addr (mbar_base + 456)
    #define p_full_addr (mbar_base + 464)
    #define pv_done_addr (mbar_base + 472)
    #define decode_done_addr (mbar_base + 728)
    #define meta_empty_addr (mbar_base + 736)
    #define tmem_dead_addr (mbar_base + 752)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

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

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 95 barriers)
    // Mbarriers at smem_raw[0..760)

    if (warp == 0) {
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // meta_full: 2 barriers, init_count=1
        // kv_raw_full: 32 barriers, init_count=1
        // --- pipeline 'raw_ring' ---
        // kv_raw_empty: 10 barriers, init_count=4
        // k16_full: 2 barriers, init_count=4
        // k16_empty: 2 barriers, init_count=1
        // v_full: 2 barriers, init_count=4
        // v_empty: 2 barriers, init_count=1
        // s_full: 1 barriers, init_count=1
        // s_empty: 1 barriers, init_count=4
        // p_full: 1 barriers, init_count=4
        // pv_done: 32 barriers, init_count=1
        // decode_done: 1 barriers, init_count=4
        // meta_empty: 2 barriers, init_count=4
        // tmem_dead: 1 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 0 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(27), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(25), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(22), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(18), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        uint32_t _mbarrier_init_count_0_64 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(27), "r"((uint32_t)(1)));
        if (lane < 31) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_0_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 760);
    if (warp == 0) {
        int _tmem_hold = smem + 760;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

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
            for (unsigned int work_idx = blockIdx.x; work_idx < total_units_sm; work_idx += gridDim.x) {
                int unit = work_idx;
                int request = unit / num_kv_heads;
                int kv_head = unit % num_kv_heads;
                int group_size = num_q_heads / num_kv_heads;
                mbarrier_wait(meta_full_addr + (item_stage) * 8, item_phase);
                int npairs = smem_npairs[item_stage];
                int kv_len = smem_kvlen[item_stage];
                int meta_base = item_stage * 64;
                int item_par = item_stage;
                item_stage += 1;
                if (item_stage == 2) { item_stage = 0; item_phase ^= 1; }
                float origin[4];
                float neg_ms[4];
                int pending = 15;
                origin[0] = -CAKE_INF;
                neg_ms[0] = 0.0f;
                origin[1] = -CAKE_INF;
                neg_ms[1] = 0.0f;
                origin[2] = -CAKE_INF;
                neg_ms[2] = 0.0f;
                origin[3] = -CAKE_INF;
                neg_ms[3] = 0.0f;
                float lane_sum[64];
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
                lane_sum[32] = 0.0f;
                lane_sum[33] = 0.0f;
                lane_sum[34] = 0.0f;
                lane_sum[35] = 0.0f;
                lane_sum[36] = 0.0f;
                lane_sum[37] = 0.0f;
                lane_sum[38] = 0.0f;
                lane_sum[39] = 0.0f;
                lane_sum[40] = 0.0f;
                lane_sum[41] = 0.0f;
                lane_sum[42] = 0.0f;
                lane_sum[43] = 0.0f;
                lane_sum[44] = 0.0f;
                lane_sum[45] = 0.0f;
                lane_sum[46] = 0.0f;
                lane_sum[47] = 0.0f;
                lane_sum[48] = 0.0f;
                lane_sum[49] = 0.0f;
                lane_sum[50] = 0.0f;
                lane_sum[51] = 0.0f;
                lane_sum[52] = 0.0f;
                lane_sum[53] = 0.0f;
                lane_sum[54] = 0.0f;
                lane_sum[55] = 0.0f;
                lane_sum[56] = 0.0f;
                lane_sum[57] = 0.0f;
                lane_sum[58] = 0.0f;
                lane_sum[59] = 0.0f;
                lane_sum[60] = 0.0f;
                lane_sum[61] = 0.0f;
                lane_sum[62] = 0.0f;
                lane_sum[63] = 0.0f;
                int cq_base = kv_len - seqlen_q + 1;
                #pragma unroll 1
                for (int p = 0; p < 32; p++) {
                    if (npairs <= p) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage) * 8, s_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)(s_stage * 128) + (unsigned int)tmem_lane_base;
                    tmem_ld_x16(&sv0[0], s_addr);
                    tmem_ld_x16(&sv1[0], s_addr + 64);
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
                    if (p >= 1) {
                        mbarrier_wait(pv_done_addr + (p - 1) * 8, item_par);
                    }
                    int p0_base = smem_p_addr + (unsigned int)(p_stage * 2 * 16384);
                    int p1_base = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 16384);
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
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 128 ^ (token * 128 >> 7 & 7) << 4))), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 128 + 16 ^ (token * 128 + 16 >> 7 & 7) << 4))), "r"(x[4]), "r"(x[5]), "r"(x[6]), "r"(x[7]) : "memory");
                    } else {
                        unsigned int zero = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 128 ^ (token * 128 >> 7 & 7) << 4))), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 128 + 16 ^ (token * 128 + 16 >> 7 & 7) << 4))), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
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
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 128 ^ (token * 128 >> 7 & 7) << 4))), "r"(x_1[0]), "r"(x_1[1]), "r"(x_1[2]), "r"(x_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 128 + 16 ^ (token * 128 + 16 >> 7 & 7) << 4))), "r"(x_1[4]), "r"(x_1[5]), "r"(x_1[6]), "r"(x_1[7]) : "memory");
                    } else {
                        unsigned int zero_1 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 128 ^ (token * 128 >> 7 & 7) << 4))), "r"(zero_1), "r"(zero_1), "r"(zero_1), "r"(zero_1) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 128 + 16 ^ (token * 128 + 16 >> 7 & 7) << 4))), "r"(zero_1), "r"(zero_1), "r"(zero_1), "r"(zero_1) : "memory");
                    }
                    tmem_ld_x16(&sv0[0], s_addr + 16);
                    tmem_ld_x16(&sv1[0], s_addr + 64 + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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
                    int p0_base_4 = smem_p_addr + (unsigned int)(p_stage * 2 * 16384);
                    int p1_base_5 = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 16384);
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
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 128 + 32 ^ (token * 128 + 32 >> 7 & 7) << 4))), "r"(x_2[0]), "r"(x_2[1]), "r"(x_2[2]), "r"(x_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 128 + 48 ^ (token * 128 + 48 >> 7 & 7) << 4))), "r"(x_2[4]), "r"(x_2[5]), "r"(x_2[6]), "r"(x_2[7]) : "memory");
                    } else {
                        unsigned int zero_2 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 128 + 32 ^ (token * 128 + 32 >> 7 & 7) << 4))), "r"(zero_2), "r"(zero_2), "r"(zero_2), "r"(zero_2) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_4 + (token * 128 + 48 ^ (token * 128 + 48 >> 7 & 7) << 4))), "r"(zero_2), "r"(zero_2), "r"(zero_2), "r"(zero_2) : "memory");
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
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 128 + 32 ^ (token * 128 + 32 >> 7 & 7) << 4))), "r"(x_3[0]), "r"(x_3[1]), "r"(x_3[2]), "r"(x_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 128 + 48 ^ (token * 128 + 48 >> 7 & 7) << 4))), "r"(x_3[4]), "r"(x_3[5]), "r"(x_3[6]), "r"(x_3[7]) : "memory");
                    } else {
                        unsigned int zero_3 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 128 + 32 ^ (token * 128 + 32 >> 7 & 7) << 4))), "r"(zero_3), "r"(zero_3), "r"(zero_3), "r"(zero_3) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_5 + (token * 128 + 48 ^ (token * 128 + 48 >> 7 & 7) << 4))), "r"(zero_3), "r"(zero_3), "r"(zero_3), "r"(zero_3) : "memory");
                    }
                    tmem_ld_x16(&sv0[0], s_addr + 32);
                    tmem_ld_x16(&sv1[0], s_addr + 64 + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int v0_8 = lim0;
                    int c0_9 = cq0 + 2;
                    if (v0_8 > c0_9) {
                        v0_8 = c0_9;
                    }
                    if ((mw0 >> 2 & 1) == 0) {
                        v0_8 = 0;
                    }
                    if (token >= v0_8) {
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
                    int v1_10 = lim1;
                    int c1_11 = cq1 + 2;
                    if (v1_10 > c1_11) {
                        v1_10 = c1_11;
                    }
                    if ((mw1 >> 2 & 1) == 0) {
                        v1_10 = 0;
                    }
                    if (token >= v1_10) {
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
                    if ((pending >> 2 & 1) == 1) {
                        float m0_2 = sv0[0];
                        float m1_2 = sv1[0];
                        float _max_60 = max_noftz(m0_2, sv0[1]);
                        m0_2 = _max_60;
                        float _max_61 = max_noftz(m1_2, sv1[1]);
                        m1_2 = _max_61;
                        float _max_62 = max_noftz(m0_2, sv0[2]);
                        m0_2 = _max_62;
                        float _max_63 = max_noftz(m1_2, sv1[2]);
                        m1_2 = _max_63;
                        float _max_64 = max_noftz(m0_2, sv0[3]);
                        m0_2 = _max_64;
                        float _max_65 = max_noftz(m1_2, sv1[3]);
                        m1_2 = _max_65;
                        float _max_66 = max_noftz(m0_2, sv0[4]);
                        m0_2 = _max_66;
                        float _max_67 = max_noftz(m1_2, sv1[4]);
                        m1_2 = _max_67;
                        float _max_68 = max_noftz(m0_2, sv0[5]);
                        m0_2 = _max_68;
                        float _max_69 = max_noftz(m1_2, sv1[5]);
                        m1_2 = _max_69;
                        float _max_70 = max_noftz(m0_2, sv0[6]);
                        m0_2 = _max_70;
                        float _max_71 = max_noftz(m1_2, sv1[6]);
                        m1_2 = _max_71;
                        float _max_72 = max_noftz(m0_2, sv0[7]);
                        m0_2 = _max_72;
                        float _max_73 = max_noftz(m1_2, sv1[7]);
                        m1_2 = _max_73;
                        float _max_74 = max_noftz(m0_2, sv0[8]);
                        m0_2 = _max_74;
                        float _max_75 = max_noftz(m1_2, sv1[8]);
                        m1_2 = _max_75;
                        float _max_76 = max_noftz(m0_2, sv0[9]);
                        m0_2 = _max_76;
                        float _max_77 = max_noftz(m1_2, sv1[9]);
                        m1_2 = _max_77;
                        float _max_78 = max_noftz(m0_2, sv0[10]);
                        m0_2 = _max_78;
                        float _max_79 = max_noftz(m1_2, sv1[10]);
                        m1_2 = _max_79;
                        float _max_80 = max_noftz(m0_2, sv0[11]);
                        m0_2 = _max_80;
                        float _max_81 = max_noftz(m1_2, sv1[11]);
                        m1_2 = _max_81;
                        float _max_82 = max_noftz(m0_2, sv0[12]);
                        m0_2 = _max_82;
                        float _max_83 = max_noftz(m1_2, sv1[12]);
                        m1_2 = _max_83;
                        float _max_84 = max_noftz(m0_2, sv0[13]);
                        m0_2 = _max_84;
                        float _max_85 = max_noftz(m1_2, sv1[13]);
                        m1_2 = _max_85;
                        float _max_86 = max_noftz(m0_2, sv0[14]);
                        m0_2 = _max_86;
                        float _max_87 = max_noftz(m1_2, sv1[14]);
                        m1_2 = _max_87;
                        float _max_88 = max_noftz(m0_2, sv0[15]);
                        m0_2 = _max_88;
                        float _max_89 = max_noftz(m1_2, sv1[15]);
                        m1_2 = _max_89;
                        float _warp_redux_f32_4;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_4) : "f"(m0_2));
                        float w0_2 = _warp_redux_f32_4;
                        float _warp_redux_f32_5;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_5) : "f"(m1_2));
                        float w1_2 = _warp_redux_f32_5;
                        float cand_2 = ((w0_2 > -CAKE_INF) ? w0_2 : w1_2);
                        if (cand_2 > -CAKE_INF) {
                            origin[2] = cand_2;
                            neg_ms[2] = (-cand_2) * softmax_scale_log2;
                            pending = pending - 4;
                        }
                    }
                    int p0_base_12 = smem_p_addr + (unsigned int)(p_stage * 2 * 16384);
                    int p1_base_13 = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 16384);
                    unsigned int live0_14 = mw0 >> 2 & 1;
                    if (live0_14 != 0) {
                        float _fma_64 = __fmaf_rn(sv0[0], softmax_scale_log2, neg_ms[2]);
                        float _exp2_64 = approx_exp2(_fma_64);
                        sv0[0] = _exp2_64;
                        float _fma_65 = __fmaf_rn(sv0[1], softmax_scale_log2, neg_ms[2]);
                        float _exp2_65 = approx_exp2(_fma_65);
                        sv0[1] = _exp2_65;
                        float _fma_66 = __fmaf_rn(sv0[2], softmax_scale_log2, neg_ms[2]);
                        float _exp2_66 = approx_exp2(_fma_66);
                        sv0[2] = _exp2_66;
                        float _fma_67 = __fmaf_rn(sv0[3], softmax_scale_log2, neg_ms[2]);
                        float _exp2_67 = approx_exp2(_fma_67);
                        sv0[3] = _exp2_67;
                        float _fma_68 = __fmaf_rn(sv0[4], softmax_scale_log2, neg_ms[2]);
                        float _exp2_68 = approx_exp2(_fma_68);
                        sv0[4] = _exp2_68;
                        float _fma_69 = __fmaf_rn(sv0[5], softmax_scale_log2, neg_ms[2]);
                        float _exp2_69 = approx_exp2(_fma_69);
                        sv0[5] = _exp2_69;
                        float _fma_70 = __fmaf_rn(sv0[6], softmax_scale_log2, neg_ms[2]);
                        float _exp2_70 = approx_exp2(_fma_70);
                        sv0[6] = _exp2_70;
                        float _fma_71 = __fmaf_rn(sv0[7], softmax_scale_log2, neg_ms[2]);
                        float _exp2_71 = approx_exp2(_fma_71);
                        sv0[7] = _exp2_71;
                        float _fma_72 = __fmaf_rn(sv0[8], softmax_scale_log2, neg_ms[2]);
                        float _exp2_72 = approx_exp2(_fma_72);
                        sv0[8] = _exp2_72;
                        float _fma_73 = __fmaf_rn(sv0[9], softmax_scale_log2, neg_ms[2]);
                        float _exp2_73 = approx_exp2(_fma_73);
                        sv0[9] = _exp2_73;
                        float _fma_74 = __fmaf_rn(sv0[10], softmax_scale_log2, neg_ms[2]);
                        float _exp2_74 = approx_exp2(_fma_74);
                        sv0[10] = _exp2_74;
                        float _fma_75 = __fmaf_rn(sv0[11], softmax_scale_log2, neg_ms[2]);
                        float _exp2_75 = approx_exp2(_fma_75);
                        sv0[11] = _exp2_75;
                        float _fma_76 = __fmaf_rn(sv0[12], softmax_scale_log2, neg_ms[2]);
                        float _exp2_76 = approx_exp2(_fma_76);
                        sv0[12] = _exp2_76;
                        float _fma_77 = __fmaf_rn(sv0[13], softmax_scale_log2, neg_ms[2]);
                        float _exp2_77 = approx_exp2(_fma_77);
                        sv0[13] = _exp2_77;
                        float _fma_78 = __fmaf_rn(sv0[14], softmax_scale_log2, neg_ms[2]);
                        float _exp2_78 = approx_exp2(_fma_78);
                        sv0[14] = _exp2_78;
                        float _fma_79 = __fmaf_rn(sv0[15], softmax_scale_log2, neg_ms[2]);
                        float _exp2_79 = approx_exp2(_fma_79);
                        sv0[15] = _exp2_79;
                        unsigned int x_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv0[_lp*2 + 0], sv0[_lp*2+1 + 0]));
                            x_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_32[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_32[0]), "=&f"(_bf16x2_add_f32_32[1]) : "r"(x_4[0]), "f"(lane_sum[32]), "f"(lane_sum[33]));
                        lane_sum[32] = _bf16x2_add_f32_32[0];
                        lane_sum[33] = _bf16x2_add_f32_32[1];
                        float _bf16x2_add_f32_33[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_33[0]), "=&f"(_bf16x2_add_f32_33[1]) : "r"(x_4[1]), "f"(lane_sum[34]), "f"(lane_sum[35]));
                        lane_sum[34] = _bf16x2_add_f32_33[0];
                        lane_sum[35] = _bf16x2_add_f32_33[1];
                        float _bf16x2_add_f32_34[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_34[0]), "=&f"(_bf16x2_add_f32_34[1]) : "r"(x_4[2]), "f"(lane_sum[36]), "f"(lane_sum[37]));
                        lane_sum[36] = _bf16x2_add_f32_34[0];
                        lane_sum[37] = _bf16x2_add_f32_34[1];
                        float _bf16x2_add_f32_35[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_35[0]), "=&f"(_bf16x2_add_f32_35[1]) : "r"(x_4[3]), "f"(lane_sum[38]), "f"(lane_sum[39]));
                        lane_sum[38] = _bf16x2_add_f32_35[0];
                        lane_sum[39] = _bf16x2_add_f32_35[1];
                        float _bf16x2_add_f32_36[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_36[0]), "=&f"(_bf16x2_add_f32_36[1]) : "r"(x_4[4]), "f"(lane_sum[40]), "f"(lane_sum[41]));
                        lane_sum[40] = _bf16x2_add_f32_36[0];
                        lane_sum[41] = _bf16x2_add_f32_36[1];
                        float _bf16x2_add_f32_37[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_37[0]), "=&f"(_bf16x2_add_f32_37[1]) : "r"(x_4[5]), "f"(lane_sum[42]), "f"(lane_sum[43]));
                        lane_sum[42] = _bf16x2_add_f32_37[0];
                        lane_sum[43] = _bf16x2_add_f32_37[1];
                        float _bf16x2_add_f32_38[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_38[0]), "=&f"(_bf16x2_add_f32_38[1]) : "r"(x_4[6]), "f"(lane_sum[44]), "f"(lane_sum[45]));
                        lane_sum[44] = _bf16x2_add_f32_38[0];
                        lane_sum[45] = _bf16x2_add_f32_38[1];
                        float _bf16x2_add_f32_39[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_39[0]), "=&f"(_bf16x2_add_f32_39[1]) : "r"(x_4[7]), "f"(lane_sum[46]), "f"(lane_sum[47]));
                        lane_sum[46] = _bf16x2_add_f32_39[0];
                        lane_sum[47] = _bf16x2_add_f32_39[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_12 + (token * 128 + 64 ^ (token * 128 + 64 >> 7 & 7) << 4))), "r"(x_4[0]), "r"(x_4[1]), "r"(x_4[2]), "r"(x_4[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_12 + (token * 128 + 80 ^ (token * 128 + 80 >> 7 & 7) << 4))), "r"(x_4[4]), "r"(x_4[5]), "r"(x_4[6]), "r"(x_4[7]) : "memory");
                    } else {
                        unsigned int zero_4 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_12 + (token * 128 + 64 ^ (token * 128 + 64 >> 7 & 7) << 4))), "r"(zero_4), "r"(zero_4), "r"(zero_4), "r"(zero_4) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_12 + (token * 128 + 80 ^ (token * 128 + 80 >> 7 & 7) << 4))), "r"(zero_4), "r"(zero_4), "r"(zero_4), "r"(zero_4) : "memory");
                    }
                    unsigned int live1_15 = mw1 >> 2 & 1;
                    if (live1_15 != 0) {
                        float _fma_80 = __fmaf_rn(sv1[0], softmax_scale_log2, neg_ms[2]);
                        float _exp2_80 = approx_exp2(_fma_80);
                        sv1[0] = _exp2_80;
                        float _fma_81 = __fmaf_rn(sv1[1], softmax_scale_log2, neg_ms[2]);
                        float _exp2_81 = approx_exp2(_fma_81);
                        sv1[1] = _exp2_81;
                        float _fma_82 = __fmaf_rn(sv1[2], softmax_scale_log2, neg_ms[2]);
                        float _exp2_82 = approx_exp2(_fma_82);
                        sv1[2] = _exp2_82;
                        float _fma_83 = __fmaf_rn(sv1[3], softmax_scale_log2, neg_ms[2]);
                        float _exp2_83 = approx_exp2(_fma_83);
                        sv1[3] = _exp2_83;
                        float _fma_84 = __fmaf_rn(sv1[4], softmax_scale_log2, neg_ms[2]);
                        float _exp2_84 = approx_exp2(_fma_84);
                        sv1[4] = _exp2_84;
                        float _fma_85 = __fmaf_rn(sv1[5], softmax_scale_log2, neg_ms[2]);
                        float _exp2_85 = approx_exp2(_fma_85);
                        sv1[5] = _exp2_85;
                        float _fma_86 = __fmaf_rn(sv1[6], softmax_scale_log2, neg_ms[2]);
                        float _exp2_86 = approx_exp2(_fma_86);
                        sv1[6] = _exp2_86;
                        float _fma_87 = __fmaf_rn(sv1[7], softmax_scale_log2, neg_ms[2]);
                        float _exp2_87 = approx_exp2(_fma_87);
                        sv1[7] = _exp2_87;
                        float _fma_88 = __fmaf_rn(sv1[8], softmax_scale_log2, neg_ms[2]);
                        float _exp2_88 = approx_exp2(_fma_88);
                        sv1[8] = _exp2_88;
                        float _fma_89 = __fmaf_rn(sv1[9], softmax_scale_log2, neg_ms[2]);
                        float _exp2_89 = approx_exp2(_fma_89);
                        sv1[9] = _exp2_89;
                        float _fma_90 = __fmaf_rn(sv1[10], softmax_scale_log2, neg_ms[2]);
                        float _exp2_90 = approx_exp2(_fma_90);
                        sv1[10] = _exp2_90;
                        float _fma_91 = __fmaf_rn(sv1[11], softmax_scale_log2, neg_ms[2]);
                        float _exp2_91 = approx_exp2(_fma_91);
                        sv1[11] = _exp2_91;
                        float _fma_92 = __fmaf_rn(sv1[12], softmax_scale_log2, neg_ms[2]);
                        float _exp2_92 = approx_exp2(_fma_92);
                        sv1[12] = _exp2_92;
                        float _fma_93 = __fmaf_rn(sv1[13], softmax_scale_log2, neg_ms[2]);
                        float _exp2_93 = approx_exp2(_fma_93);
                        sv1[13] = _exp2_93;
                        float _fma_94 = __fmaf_rn(sv1[14], softmax_scale_log2, neg_ms[2]);
                        float _exp2_94 = approx_exp2(_fma_94);
                        sv1[14] = _exp2_94;
                        float _fma_95 = __fmaf_rn(sv1[15], softmax_scale_log2, neg_ms[2]);
                        float _exp2_95 = approx_exp2(_fma_95);
                        sv1[15] = _exp2_95;
                        unsigned int x_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv1[_lp*2 + 0], sv1[_lp*2+1 + 0]));
                            x_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_40[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_40[0]), "=&f"(_bf16x2_add_f32_40[1]) : "r"(x_5[0]), "f"(lane_sum[32]), "f"(lane_sum[33]));
                        lane_sum[32] = _bf16x2_add_f32_40[0];
                        lane_sum[33] = _bf16x2_add_f32_40[1];
                        float _bf16x2_add_f32_41[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_41[0]), "=&f"(_bf16x2_add_f32_41[1]) : "r"(x_5[1]), "f"(lane_sum[34]), "f"(lane_sum[35]));
                        lane_sum[34] = _bf16x2_add_f32_41[0];
                        lane_sum[35] = _bf16x2_add_f32_41[1];
                        float _bf16x2_add_f32_42[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_42[0]), "=&f"(_bf16x2_add_f32_42[1]) : "r"(x_5[2]), "f"(lane_sum[36]), "f"(lane_sum[37]));
                        lane_sum[36] = _bf16x2_add_f32_42[0];
                        lane_sum[37] = _bf16x2_add_f32_42[1];
                        float _bf16x2_add_f32_43[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_43[0]), "=&f"(_bf16x2_add_f32_43[1]) : "r"(x_5[3]), "f"(lane_sum[38]), "f"(lane_sum[39]));
                        lane_sum[38] = _bf16x2_add_f32_43[0];
                        lane_sum[39] = _bf16x2_add_f32_43[1];
                        float _bf16x2_add_f32_44[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_44[0]), "=&f"(_bf16x2_add_f32_44[1]) : "r"(x_5[4]), "f"(lane_sum[40]), "f"(lane_sum[41]));
                        lane_sum[40] = _bf16x2_add_f32_44[0];
                        lane_sum[41] = _bf16x2_add_f32_44[1];
                        float _bf16x2_add_f32_45[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_45[0]), "=&f"(_bf16x2_add_f32_45[1]) : "r"(x_5[5]), "f"(lane_sum[42]), "f"(lane_sum[43]));
                        lane_sum[42] = _bf16x2_add_f32_45[0];
                        lane_sum[43] = _bf16x2_add_f32_45[1];
                        float _bf16x2_add_f32_46[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_46[0]), "=&f"(_bf16x2_add_f32_46[1]) : "r"(x_5[6]), "f"(lane_sum[44]), "f"(lane_sum[45]));
                        lane_sum[44] = _bf16x2_add_f32_46[0];
                        lane_sum[45] = _bf16x2_add_f32_46[1];
                        float _bf16x2_add_f32_47[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_47[0]), "=&f"(_bf16x2_add_f32_47[1]) : "r"(x_5[7]), "f"(lane_sum[46]), "f"(lane_sum[47]));
                        lane_sum[46] = _bf16x2_add_f32_47[0];
                        lane_sum[47] = _bf16x2_add_f32_47[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_13 + (token * 128 + 64 ^ (token * 128 + 64 >> 7 & 7) << 4))), "r"(x_5[0]), "r"(x_5[1]), "r"(x_5[2]), "r"(x_5[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_13 + (token * 128 + 80 ^ (token * 128 + 80 >> 7 & 7) << 4))), "r"(x_5[4]), "r"(x_5[5]), "r"(x_5[6]), "r"(x_5[7]) : "memory");
                    } else {
                        unsigned int zero_5 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_13 + (token * 128 + 64 ^ (token * 128 + 64 >> 7 & 7) << 4))), "r"(zero_5), "r"(zero_5), "r"(zero_5), "r"(zero_5) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_13 + (token * 128 + 80 ^ (token * 128 + 80 >> 7 & 7) << 4))), "r"(zero_5), "r"(zero_5), "r"(zero_5), "r"(zero_5) : "memory");
                    }
                    tmem_ld_x16(&sv0[0], s_addr + 48);
                    tmem_ld_x16(&sv1[0], s_addr + 64 + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                    }
                    s_phase ^= 1;
                    int v0_16 = lim0;
                    int c0_17 = cq0 + 3;
                    if (v0_16 > c0_17) {
                        v0_16 = c0_17;
                    }
                    if ((mw0 >> 3 & 1) == 0) {
                        v0_16 = 0;
                    }
                    if (token >= v0_16) {
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
                    int v1_18 = lim1;
                    int c1_19 = cq1 + 3;
                    if (v1_18 > c1_19) {
                        v1_18 = c1_19;
                    }
                    if ((mw1 >> 3 & 1) == 0) {
                        v1_18 = 0;
                    }
                    if (token >= v1_18) {
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
                    if ((pending >> 3 & 1) == 1) {
                        float m0_3 = sv0[0];
                        float m1_3 = sv1[0];
                        float _max_90 = max_noftz(m0_3, sv0[1]);
                        m0_3 = _max_90;
                        float _max_91 = max_noftz(m1_3, sv1[1]);
                        m1_3 = _max_91;
                        float _max_92 = max_noftz(m0_3, sv0[2]);
                        m0_3 = _max_92;
                        float _max_93 = max_noftz(m1_3, sv1[2]);
                        m1_3 = _max_93;
                        float _max_94 = max_noftz(m0_3, sv0[3]);
                        m0_3 = _max_94;
                        float _max_95 = max_noftz(m1_3, sv1[3]);
                        m1_3 = _max_95;
                        float _max_96 = max_noftz(m0_3, sv0[4]);
                        m0_3 = _max_96;
                        float _max_97 = max_noftz(m1_3, sv1[4]);
                        m1_3 = _max_97;
                        float _max_98 = max_noftz(m0_3, sv0[5]);
                        m0_3 = _max_98;
                        float _max_99 = max_noftz(m1_3, sv1[5]);
                        m1_3 = _max_99;
                        float _max_100 = max_noftz(m0_3, sv0[6]);
                        m0_3 = _max_100;
                        float _max_101 = max_noftz(m1_3, sv1[6]);
                        m1_3 = _max_101;
                        float _max_102 = max_noftz(m0_3, sv0[7]);
                        m0_3 = _max_102;
                        float _max_103 = max_noftz(m1_3, sv1[7]);
                        m1_3 = _max_103;
                        float _max_104 = max_noftz(m0_3, sv0[8]);
                        m0_3 = _max_104;
                        float _max_105 = max_noftz(m1_3, sv1[8]);
                        m1_3 = _max_105;
                        float _max_106 = max_noftz(m0_3, sv0[9]);
                        m0_3 = _max_106;
                        float _max_107 = max_noftz(m1_3, sv1[9]);
                        m1_3 = _max_107;
                        float _max_108 = max_noftz(m0_3, sv0[10]);
                        m0_3 = _max_108;
                        float _max_109 = max_noftz(m1_3, sv1[10]);
                        m1_3 = _max_109;
                        float _max_110 = max_noftz(m0_3, sv0[11]);
                        m0_3 = _max_110;
                        float _max_111 = max_noftz(m1_3, sv1[11]);
                        m1_3 = _max_111;
                        float _max_112 = max_noftz(m0_3, sv0[12]);
                        m0_3 = _max_112;
                        float _max_113 = max_noftz(m1_3, sv1[12]);
                        m1_3 = _max_113;
                        float _max_114 = max_noftz(m0_3, sv0[13]);
                        m0_3 = _max_114;
                        float _max_115 = max_noftz(m1_3, sv1[13]);
                        m1_3 = _max_115;
                        float _max_116 = max_noftz(m0_3, sv0[14]);
                        m0_3 = _max_116;
                        float _max_117 = max_noftz(m1_3, sv1[14]);
                        m1_3 = _max_117;
                        float _max_118 = max_noftz(m0_3, sv0[15]);
                        m0_3 = _max_118;
                        float _max_119 = max_noftz(m1_3, sv1[15]);
                        m1_3 = _max_119;
                        float _warp_redux_f32_6;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_6) : "f"(m0_3));
                        float w0_3 = _warp_redux_f32_6;
                        float _warp_redux_f32_7;
                        asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_7) : "f"(m1_3));
                        float w1_3 = _warp_redux_f32_7;
                        float cand_3 = ((w0_3 > -CAKE_INF) ? w0_3 : w1_3);
                        if (cand_3 > -CAKE_INF) {
                            origin[3] = cand_3;
                            neg_ms[3] = (-cand_3) * softmax_scale_log2;
                            pending = pending - 8;
                        }
                    }
                    int p0_base_20 = smem_p_addr + (unsigned int)(p_stage * 2 * 16384);
                    int p1_base_21 = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 16384);
                    unsigned int live0_22 = mw0 >> 3 & 1;
                    if (live0_22 != 0) {
                        float _fma_96 = __fmaf_rn(sv0[0], softmax_scale_log2, neg_ms[3]);
                        float _exp2_96 = approx_exp2(_fma_96);
                        sv0[0] = _exp2_96;
                        float _fma_97 = __fmaf_rn(sv0[1], softmax_scale_log2, neg_ms[3]);
                        float _exp2_97 = approx_exp2(_fma_97);
                        sv0[1] = _exp2_97;
                        float _fma_98 = __fmaf_rn(sv0[2], softmax_scale_log2, neg_ms[3]);
                        float _exp2_98 = approx_exp2(_fma_98);
                        sv0[2] = _exp2_98;
                        float _fma_99 = __fmaf_rn(sv0[3], softmax_scale_log2, neg_ms[3]);
                        float _exp2_99 = approx_exp2(_fma_99);
                        sv0[3] = _exp2_99;
                        float _fma_100 = __fmaf_rn(sv0[4], softmax_scale_log2, neg_ms[3]);
                        float _exp2_100 = approx_exp2(_fma_100);
                        sv0[4] = _exp2_100;
                        float _fma_101 = __fmaf_rn(sv0[5], softmax_scale_log2, neg_ms[3]);
                        float _exp2_101 = approx_exp2(_fma_101);
                        sv0[5] = _exp2_101;
                        float _fma_102 = __fmaf_rn(sv0[6], softmax_scale_log2, neg_ms[3]);
                        float _exp2_102 = approx_exp2(_fma_102);
                        sv0[6] = _exp2_102;
                        float _fma_103 = __fmaf_rn(sv0[7], softmax_scale_log2, neg_ms[3]);
                        float _exp2_103 = approx_exp2(_fma_103);
                        sv0[7] = _exp2_103;
                        float _fma_104 = __fmaf_rn(sv0[8], softmax_scale_log2, neg_ms[3]);
                        float _exp2_104 = approx_exp2(_fma_104);
                        sv0[8] = _exp2_104;
                        float _fma_105 = __fmaf_rn(sv0[9], softmax_scale_log2, neg_ms[3]);
                        float _exp2_105 = approx_exp2(_fma_105);
                        sv0[9] = _exp2_105;
                        float _fma_106 = __fmaf_rn(sv0[10], softmax_scale_log2, neg_ms[3]);
                        float _exp2_106 = approx_exp2(_fma_106);
                        sv0[10] = _exp2_106;
                        float _fma_107 = __fmaf_rn(sv0[11], softmax_scale_log2, neg_ms[3]);
                        float _exp2_107 = approx_exp2(_fma_107);
                        sv0[11] = _exp2_107;
                        float _fma_108 = __fmaf_rn(sv0[12], softmax_scale_log2, neg_ms[3]);
                        float _exp2_108 = approx_exp2(_fma_108);
                        sv0[12] = _exp2_108;
                        float _fma_109 = __fmaf_rn(sv0[13], softmax_scale_log2, neg_ms[3]);
                        float _exp2_109 = approx_exp2(_fma_109);
                        sv0[13] = _exp2_109;
                        float _fma_110 = __fmaf_rn(sv0[14], softmax_scale_log2, neg_ms[3]);
                        float _exp2_110 = approx_exp2(_fma_110);
                        sv0[14] = _exp2_110;
                        float _fma_111 = __fmaf_rn(sv0[15], softmax_scale_log2, neg_ms[3]);
                        float _exp2_111 = approx_exp2(_fma_111);
                        sv0[15] = _exp2_111;
                        unsigned int x_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv0[_lp*2 + 0], sv0[_lp*2+1 + 0]));
                            x_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_48[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_48[0]), "=&f"(_bf16x2_add_f32_48[1]) : "r"(x_6[0]), "f"(lane_sum[48]), "f"(lane_sum[49]));
                        lane_sum[48] = _bf16x2_add_f32_48[0];
                        lane_sum[49] = _bf16x2_add_f32_48[1];
                        float _bf16x2_add_f32_49[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_49[0]), "=&f"(_bf16x2_add_f32_49[1]) : "r"(x_6[1]), "f"(lane_sum[50]), "f"(lane_sum[51]));
                        lane_sum[50] = _bf16x2_add_f32_49[0];
                        lane_sum[51] = _bf16x2_add_f32_49[1];
                        float _bf16x2_add_f32_50[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_50[0]), "=&f"(_bf16x2_add_f32_50[1]) : "r"(x_6[2]), "f"(lane_sum[52]), "f"(lane_sum[53]));
                        lane_sum[52] = _bf16x2_add_f32_50[0];
                        lane_sum[53] = _bf16x2_add_f32_50[1];
                        float _bf16x2_add_f32_51[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_51[0]), "=&f"(_bf16x2_add_f32_51[1]) : "r"(x_6[3]), "f"(lane_sum[54]), "f"(lane_sum[55]));
                        lane_sum[54] = _bf16x2_add_f32_51[0];
                        lane_sum[55] = _bf16x2_add_f32_51[1];
                        float _bf16x2_add_f32_52[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_52[0]), "=&f"(_bf16x2_add_f32_52[1]) : "r"(x_6[4]), "f"(lane_sum[56]), "f"(lane_sum[57]));
                        lane_sum[56] = _bf16x2_add_f32_52[0];
                        lane_sum[57] = _bf16x2_add_f32_52[1];
                        float _bf16x2_add_f32_53[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_53[0]), "=&f"(_bf16x2_add_f32_53[1]) : "r"(x_6[5]), "f"(lane_sum[58]), "f"(lane_sum[59]));
                        lane_sum[58] = _bf16x2_add_f32_53[0];
                        lane_sum[59] = _bf16x2_add_f32_53[1];
                        float _bf16x2_add_f32_54[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_54[0]), "=&f"(_bf16x2_add_f32_54[1]) : "r"(x_6[6]), "f"(lane_sum[60]), "f"(lane_sum[61]));
                        lane_sum[60] = _bf16x2_add_f32_54[0];
                        lane_sum[61] = _bf16x2_add_f32_54[1];
                        float _bf16x2_add_f32_55[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_55[0]), "=&f"(_bf16x2_add_f32_55[1]) : "r"(x_6[7]), "f"(lane_sum[62]), "f"(lane_sum[63]));
                        lane_sum[62] = _bf16x2_add_f32_55[0];
                        lane_sum[63] = _bf16x2_add_f32_55[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_20 + (token * 128 + 96 ^ (token * 128 + 96 >> 7 & 7) << 4))), "r"(x_6[0]), "r"(x_6[1]), "r"(x_6[2]), "r"(x_6[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_20 + (token * 128 + 112 ^ (token * 128 + 112 >> 7 & 7) << 4))), "r"(x_6[4]), "r"(x_6[5]), "r"(x_6[6]), "r"(x_6[7]) : "memory");
                    } else {
                        unsigned int zero_6 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_20 + (token * 128 + 96 ^ (token * 128 + 96 >> 7 & 7) << 4))), "r"(zero_6), "r"(zero_6), "r"(zero_6), "r"(zero_6) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base_20 + (token * 128 + 112 ^ (token * 128 + 112 >> 7 & 7) << 4))), "r"(zero_6), "r"(zero_6), "r"(zero_6), "r"(zero_6) : "memory");
                    }
                    unsigned int live1_23 = mw1 >> 3 & 1;
                    if (live1_23 != 0) {
                        float _fma_112 = __fmaf_rn(sv1[0], softmax_scale_log2, neg_ms[3]);
                        float _exp2_112 = approx_exp2(_fma_112);
                        sv1[0] = _exp2_112;
                        float _fma_113 = __fmaf_rn(sv1[1], softmax_scale_log2, neg_ms[3]);
                        float _exp2_113 = approx_exp2(_fma_113);
                        sv1[1] = _exp2_113;
                        float _fma_114 = __fmaf_rn(sv1[2], softmax_scale_log2, neg_ms[3]);
                        float _exp2_114 = approx_exp2(_fma_114);
                        sv1[2] = _exp2_114;
                        float _fma_115 = __fmaf_rn(sv1[3], softmax_scale_log2, neg_ms[3]);
                        float _exp2_115 = approx_exp2(_fma_115);
                        sv1[3] = _exp2_115;
                        float _fma_116 = __fmaf_rn(sv1[4], softmax_scale_log2, neg_ms[3]);
                        float _exp2_116 = approx_exp2(_fma_116);
                        sv1[4] = _exp2_116;
                        float _fma_117 = __fmaf_rn(sv1[5], softmax_scale_log2, neg_ms[3]);
                        float _exp2_117 = approx_exp2(_fma_117);
                        sv1[5] = _exp2_117;
                        float _fma_118 = __fmaf_rn(sv1[6], softmax_scale_log2, neg_ms[3]);
                        float _exp2_118 = approx_exp2(_fma_118);
                        sv1[6] = _exp2_118;
                        float _fma_119 = __fmaf_rn(sv1[7], softmax_scale_log2, neg_ms[3]);
                        float _exp2_119 = approx_exp2(_fma_119);
                        sv1[7] = _exp2_119;
                        float _fma_120 = __fmaf_rn(sv1[8], softmax_scale_log2, neg_ms[3]);
                        float _exp2_120 = approx_exp2(_fma_120);
                        sv1[8] = _exp2_120;
                        float _fma_121 = __fmaf_rn(sv1[9], softmax_scale_log2, neg_ms[3]);
                        float _exp2_121 = approx_exp2(_fma_121);
                        sv1[9] = _exp2_121;
                        float _fma_122 = __fmaf_rn(sv1[10], softmax_scale_log2, neg_ms[3]);
                        float _exp2_122 = approx_exp2(_fma_122);
                        sv1[10] = _exp2_122;
                        float _fma_123 = __fmaf_rn(sv1[11], softmax_scale_log2, neg_ms[3]);
                        float _exp2_123 = approx_exp2(_fma_123);
                        sv1[11] = _exp2_123;
                        float _fma_124 = __fmaf_rn(sv1[12], softmax_scale_log2, neg_ms[3]);
                        float _exp2_124 = approx_exp2(_fma_124);
                        sv1[12] = _exp2_124;
                        float _fma_125 = __fmaf_rn(sv1[13], softmax_scale_log2, neg_ms[3]);
                        float _exp2_125 = approx_exp2(_fma_125);
                        sv1[13] = _exp2_125;
                        float _fma_126 = __fmaf_rn(sv1[14], softmax_scale_log2, neg_ms[3]);
                        float _exp2_126 = approx_exp2(_fma_126);
                        sv1[14] = _exp2_126;
                        float _fma_127 = __fmaf_rn(sv1[15], softmax_scale_log2, neg_ms[3]);
                        float _exp2_127 = approx_exp2(_fma_127);
                        sv1[15] = _exp2_127;
                        unsigned int x_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv1[_lp*2 + 0], sv1[_lp*2+1 + 0]));
                            x_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        float _bf16x2_add_f32_56[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_56[0]), "=&f"(_bf16x2_add_f32_56[1]) : "r"(x_7[0]), "f"(lane_sum[48]), "f"(lane_sum[49]));
                        lane_sum[48] = _bf16x2_add_f32_56[0];
                        lane_sum[49] = _bf16x2_add_f32_56[1];
                        float _bf16x2_add_f32_57[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_57[0]), "=&f"(_bf16x2_add_f32_57[1]) : "r"(x_7[1]), "f"(lane_sum[50]), "f"(lane_sum[51]));
                        lane_sum[50] = _bf16x2_add_f32_57[0];
                        lane_sum[51] = _bf16x2_add_f32_57[1];
                        float _bf16x2_add_f32_58[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_58[0]), "=&f"(_bf16x2_add_f32_58[1]) : "r"(x_7[2]), "f"(lane_sum[52]), "f"(lane_sum[53]));
                        lane_sum[52] = _bf16x2_add_f32_58[0];
                        lane_sum[53] = _bf16x2_add_f32_58[1];
                        float _bf16x2_add_f32_59[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_59[0]), "=&f"(_bf16x2_add_f32_59[1]) : "r"(x_7[3]), "f"(lane_sum[54]), "f"(lane_sum[55]));
                        lane_sum[54] = _bf16x2_add_f32_59[0];
                        lane_sum[55] = _bf16x2_add_f32_59[1];
                        float _bf16x2_add_f32_60[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_60[0]), "=&f"(_bf16x2_add_f32_60[1]) : "r"(x_7[4]), "f"(lane_sum[56]), "f"(lane_sum[57]));
                        lane_sum[56] = _bf16x2_add_f32_60[0];
                        lane_sum[57] = _bf16x2_add_f32_60[1];
                        float _bf16x2_add_f32_61[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_61[0]), "=&f"(_bf16x2_add_f32_61[1]) : "r"(x_7[5]), "f"(lane_sum[58]), "f"(lane_sum[59]));
                        lane_sum[58] = _bf16x2_add_f32_61[0];
                        lane_sum[59] = _bf16x2_add_f32_61[1];
                        float _bf16x2_add_f32_62[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_62[0]), "=&f"(_bf16x2_add_f32_62[1]) : "r"(x_7[6]), "f"(lane_sum[60]), "f"(lane_sum[61]));
                        lane_sum[60] = _bf16x2_add_f32_62[0];
                        lane_sum[61] = _bf16x2_add_f32_62[1];
                        float _bf16x2_add_f32_63[2];
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 lo, hi;\n\t"
                            "mov.b32 {lo, hi}, %2;\n\t"
                            "add.rn.f32.bf16 %0, lo, %3;\n\t"
                            "add.rn.f32.bf16 %1, hi, %4;\n\t"
                            "}\n"
                            : "=&f"(_bf16x2_add_f32_63[0]), "=&f"(_bf16x2_add_f32_63[1]) : "r"(x_7[7]), "f"(lane_sum[62]), "f"(lane_sum[63]));
                        lane_sum[62] = _bf16x2_add_f32_63[0];
                        lane_sum[63] = _bf16x2_add_f32_63[1];
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_21 + (token * 128 + 96 ^ (token * 128 + 96 >> 7 & 7) << 4))), "r"(x_7[0]), "r"(x_7[1]), "r"(x_7[2]), "r"(x_7[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_21 + (token * 128 + 112 ^ (token * 128 + 112 >> 7 & 7) << 4))), "r"(x_7[4]), "r"(x_7[5]), "r"(x_7[6]), "r"(x_7[7]) : "memory");
                    } else {
                        unsigned int zero_7 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_21 + (token * 128 + 96 ^ (token * 128 + 96 >> 7 & 7) << 4))), "r"(zero_7), "r"(zero_7), "r"(zero_7), "r"(zero_7) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base_21 + (token * 128 + 112 ^ (token * 128 + 112 >> 7 & 7) << 4))), "r"(zero_7), "r"(zero_7), "r"(zero_7), "r"(zero_7) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    }
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
                int slot_base = warp_in_role * 4 * 32;
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
                int slot_base_11 = (warp_in_role * 4 + 1) * 32;
                if (lane % 2 == 0) {
                    smem_f32[slot_base_11 + lane / 2] = head_sum_w_10;
                }
                if (lane == 0) {
                    smem_f32[slot_base_11 + 16] = origin[1];
                }
                int b4_12 = lane / 16 % 2;
                int b3_13 = lane / 8 % 2;
                int b2_14 = lane / 4 % 2;
                int b1_15 = lane / 2 % 2;
                float r8_16[8];
                #pragma unroll
                for (int c_6 = 0; c_6 < 8; c_6++) {
                    float snd8_2 = ((b4_12 == 0) ? lane_sum[32 + (c_6 + 8)] : lane_sum[32 + c_6]);
                    float kp8_2 = ((b4_12 == 0) ? lane_sum[32 + c_6] : lane_sum[32 + (c_6 + 8)]);
                    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, snd8_2, 16);
                    r8_16[c_6] = kp8_2 + _shfl_xor_10;
                }
                float r4_17[4];
                #pragma unroll
                for (int c_7 = 0; c_7 < 4; c_7++) {
                    float snd4_2 = ((b3_13 == 0) ? r8_16[c_7 + 4] : r8_16[c_7]);
                    float kp4_2 = ((b3_13 == 0) ? r8_16[c_7] : r8_16[c_7 + 4]);
                    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, snd4_2, 8);
                    r4_17[c_7] = kp4_2 + _shfl_xor_11;
                }
                float r2_18[2];
                #pragma unroll
                for (int c_8 = 0; c_8 < 2; c_8++) {
                    float snd2_2 = ((b2_14 == 0) ? r4_17[c_8 + 2] : r4_17[c_8]);
                    float kp2_2 = ((b2_14 == 0) ? r4_17[c_8] : r4_17[c_8 + 2]);
                    float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, snd2_2, 4);
                    r2_18[c_8] = kp2_2 + _shfl_xor_12;
                }
                float snd1_19 = ((b1_15 == 0) ? r2_18[1] : r2_18[0]);
                float kp1_20 = ((b1_15 == 0) ? r2_18[0] : r2_18[1]);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, snd1_19, 2);
                float r1_21 = kp1_20 + _shfl_xor_13;
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, r1_21, 1);
                r1_21 = r1_21 + _shfl_xor_14;
                float head_sum_w_22 = r1_21;
                int slot_base_23 = (warp_in_role * 4 + 2) * 32;
                if (lane % 2 == 0) {
                    smem_f32[slot_base_23 + lane / 2] = head_sum_w_22;
                }
                if (lane == 0) {
                    smem_f32[slot_base_23 + 16] = origin[2];
                }
                int b4_24 = lane / 16 % 2;
                int b3_25 = lane / 8 % 2;
                int b2_26 = lane / 4 % 2;
                int b1_27 = lane / 2 % 2;
                float r8_28[8];
                #pragma unroll
                for (int c_9 = 0; c_9 < 8; c_9++) {
                    float snd8_3 = ((b4_24 == 0) ? lane_sum[48 + (c_9 + 8)] : lane_sum[48 + c_9]);
                    float kp8_3 = ((b4_24 == 0) ? lane_sum[48 + c_9] : lane_sum[48 + (c_9 + 8)]);
                    float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, snd8_3, 16);
                    r8_28[c_9] = kp8_3 + _shfl_xor_15;
                }
                float r4_29[4];
                #pragma unroll
                for (int c_10 = 0; c_10 < 4; c_10++) {
                    float snd4_3 = ((b3_25 == 0) ? r8_28[c_10 + 4] : r8_28[c_10]);
                    float kp4_3 = ((b3_25 == 0) ? r8_28[c_10] : r8_28[c_10 + 4]);
                    float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, snd4_3, 8);
                    r4_29[c_10] = kp4_3 + _shfl_xor_16;
                }
                float r2_30[2];
                #pragma unroll
                for (int c_11 = 0; c_11 < 2; c_11++) {
                    float snd2_3 = ((b2_26 == 0) ? r4_29[c_11 + 2] : r4_29[c_11]);
                    float kp2_3 = ((b2_26 == 0) ? r4_29[c_11] : r4_29[c_11 + 2]);
                    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, snd2_3, 4);
                    r2_30[c_11] = kp2_3 + _shfl_xor_17;
                }
                float snd1_31 = ((b1_27 == 0) ? r2_30[1] : r2_30[0]);
                float kp1_32 = ((b1_27 == 0) ? r2_30[0] : r2_30[1]);
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, snd1_31, 2);
                float r1_33 = kp1_32 + _shfl_xor_18;
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, r1_33, 1);
                r1_33 = r1_33 + _shfl_xor_19;
                float head_sum_w_34 = r1_33;
                int slot_base_35 = (warp_in_role * 4 + 3) * 32;
                if (lane % 2 == 0) {
                    smem_f32[slot_base_35 + lane / 2] = head_sum_w_34;
                }
                if (lane == 0) {
                    smem_f32[slot_base_35 + 16] = origin[3];
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                float m_ab[4];
                float head_sum[4];
                float wgt[16];
                float og0 = smem_f32[16];
                float og1 = smem_f32[144];
                float og2 = smem_f32[272];
                float og3 = smem_f32[400];
                float _max_120 = max_noftz(og0, og1);
                float _max_121 = max_noftz(og2, og3);
                float _max_122 = max_noftz(_max_120, _max_121);
                m_ab[0] = _max_122;
                float _exp2_128 = approx_exp2((og0 - m_ab[0]) * softmax_scale_log2);
                wgt[0] = ((og0 > -CAKE_INF) ? _exp2_128 : 0.0f);
                float _exp2_129 = approx_exp2((og1 - m_ab[0]) * softmax_scale_log2);
                wgt[1] = ((og1 > -CAKE_INF) ? _exp2_129 : 0.0f);
                float _exp2_130 = approx_exp2((og2 - m_ab[0]) * softmax_scale_log2);
                wgt[2] = ((og2 > -CAKE_INF) ? _exp2_130 : 0.0f);
                float _exp2_131 = approx_exp2((og3 - m_ab[0]) * softmax_scale_log2);
                wgt[3] = ((og3 > -CAKE_INF) ? _exp2_131 : 0.0f);
                head_sum[0] = 0.0f;
                if (lane < 16) {
                    head_sum[0] = smem_f32[lane] * wgt[0] + smem_f32[128 + lane] * wgt[1] + smem_f32[256 + lane] * wgt[2] + smem_f32[384 + lane] * wgt[3];
                }
                float og0_36 = smem_f32[48];
                float og1_37 = smem_f32[176];
                float og2_38 = smem_f32[304];
                float og3_39 = smem_f32[432];
                float _max_123 = max_noftz(og0_36, og1_37);
                float _max_124 = max_noftz(og2_38, og3_39);
                float _max_125 = max_noftz(_max_123, _max_124);
                m_ab[1] = _max_125;
                float _exp2_132 = approx_exp2((og0_36 - m_ab[1]) * softmax_scale_log2);
                wgt[4] = ((og0_36 > -CAKE_INF) ? _exp2_132 : 0.0f);
                float _exp2_133 = approx_exp2((og1_37 - m_ab[1]) * softmax_scale_log2);
                wgt[5] = ((og1_37 > -CAKE_INF) ? _exp2_133 : 0.0f);
                float _exp2_134 = approx_exp2((og2_38 - m_ab[1]) * softmax_scale_log2);
                wgt[6] = ((og2_38 > -CAKE_INF) ? _exp2_134 : 0.0f);
                float _exp2_135 = approx_exp2((og3_39 - m_ab[1]) * softmax_scale_log2);
                wgt[7] = ((og3_39 > -CAKE_INF) ? _exp2_135 : 0.0f);
                head_sum[1] = 0.0f;
                if (lane < 16) {
                    head_sum[1] = smem_f32[32 + lane] * wgt[4] + smem_f32[160 + lane] * wgt[5] + smem_f32[288 + lane] * wgt[6] + smem_f32[416 + lane] * wgt[7];
                }
                float og0_40 = smem_f32[80];
                float og1_41 = smem_f32[208];
                float og2_42 = smem_f32[336];
                float og3_43 = smem_f32[464];
                float _max_126 = max_noftz(og0_40, og1_41);
                float _max_127 = max_noftz(og2_42, og3_43);
                float _max_128 = max_noftz(_max_126, _max_127);
                m_ab[2] = _max_128;
                float _exp2_136 = approx_exp2((og0_40 - m_ab[2]) * softmax_scale_log2);
                wgt[8] = ((og0_40 > -CAKE_INF) ? _exp2_136 : 0.0f);
                float _exp2_137 = approx_exp2((og1_41 - m_ab[2]) * softmax_scale_log2);
                wgt[9] = ((og1_41 > -CAKE_INF) ? _exp2_137 : 0.0f);
                float _exp2_138 = approx_exp2((og2_42 - m_ab[2]) * softmax_scale_log2);
                wgt[10] = ((og2_42 > -CAKE_INF) ? _exp2_138 : 0.0f);
                float _exp2_139 = approx_exp2((og3_43 - m_ab[2]) * softmax_scale_log2);
                wgt[11] = ((og3_43 > -CAKE_INF) ? _exp2_139 : 0.0f);
                head_sum[2] = 0.0f;
                if (lane < 16) {
                    head_sum[2] = smem_f32[64 + lane] * wgt[8] + smem_f32[192 + lane] * wgt[9] + smem_f32[320 + lane] * wgt[10] + smem_f32[448 + lane] * wgt[11];
                }
                float og0_44 = smem_f32[112];
                float og1_45 = smem_f32[240];
                float og2_46 = smem_f32[368];
                float og3_47 = smem_f32[496];
                float _max_129 = max_noftz(og0_44, og1_45);
                float _max_130 = max_noftz(og2_46, og3_47);
                float _max_131 = max_noftz(_max_129, _max_130);
                m_ab[3] = _max_131;
                float _exp2_140 = approx_exp2((og0_44 - m_ab[3]) * softmax_scale_log2);
                wgt[12] = ((og0_44 > -CAKE_INF) ? _exp2_140 : 0.0f);
                float _exp2_141 = approx_exp2((og1_45 - m_ab[3]) * softmax_scale_log2);
                wgt[13] = ((og1_45 > -CAKE_INF) ? _exp2_141 : 0.0f);
                float _exp2_142 = approx_exp2((og2_46 - m_ab[3]) * softmax_scale_log2);
                wgt[14] = ((og2_46 > -CAKE_INF) ? _exp2_142 : 0.0f);
                float _exp2_143 = approx_exp2((og3_47 - m_ab[3]) * softmax_scale_log2);
                wgt[15] = ((og3_47 > -CAKE_INF) ? _exp2_143 : 0.0f);
                head_sum[3] = 0.0f;
                if (lane < 16) {
                    head_sum[3] = smem_f32[96 + lane] * wgt[12] + smem_f32[224 + lane] * wgt[13] + smem_f32[352 + lane] * wgt[14] + smem_f32[480 + lane] * wgt[15];
                }
                if (npairs > 0) {
                    mbarrier_wait(pv_done_addr + (npairs - 1) * 8, item_par);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi[64];
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
                o_epi[32] = 0.0f;
                o_epi[33] = 0.0f;
                o_epi[34] = 0.0f;
                o_epi[35] = 0.0f;
                o_epi[36] = 0.0f;
                o_epi[37] = 0.0f;
                o_epi[38] = 0.0f;
                o_epi[39] = 0.0f;
                o_epi[40] = 0.0f;
                o_epi[41] = 0.0f;
                o_epi[42] = 0.0f;
                o_epi[43] = 0.0f;
                o_epi[44] = 0.0f;
                o_epi[45] = 0.0f;
                o_epi[46] = 0.0f;
                o_epi[47] = 0.0f;
                o_epi[48] = 0.0f;
                o_epi[49] = 0.0f;
                o_epi[50] = 0.0f;
                o_epi[51] = 0.0f;
                o_epi[52] = 0.0f;
                o_epi[53] = 0.0f;
                o_epi[54] = 0.0f;
                o_epi[55] = 0.0f;
                o_epi[56] = 0.0f;
                o_epi[57] = 0.0f;
                o_epi[58] = 0.0f;
                o_epi[59] = 0.0f;
                o_epi[60] = 0.0f;
                o_epi[61] = 0.0f;
                o_epi[62] = 0.0f;
                o_epi[63] = 0.0f;
                if (npairs > 0) {
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], taddr + 256 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_128 = __fmaf_rn(_tmem_load_0[0], wgt[0], o_epi[0]);
                    o_epi[0] = _fma_128;
                    float _fma_129 = __fmaf_rn(_tmem_load_0[1], wgt[0], o_epi[1]);
                    o_epi[1] = _fma_129;
                    float _fma_130 = __fmaf_rn(_tmem_load_0[2], wgt[0], o_epi[2]);
                    o_epi[2] = _fma_130;
                    float _fma_131 = __fmaf_rn(_tmem_load_0[3], wgt[0], o_epi[3]);
                    o_epi[3] = _fma_131;
                    float _fma_132 = __fmaf_rn(_tmem_load_0[4], wgt[0], o_epi[4]);
                    o_epi[4] = _fma_132;
                    float _fma_133 = __fmaf_rn(_tmem_load_0[5], wgt[0], o_epi[5]);
                    o_epi[5] = _fma_133;
                    float _fma_134 = __fmaf_rn(_tmem_load_0[6], wgt[0], o_epi[6]);
                    o_epi[6] = _fma_134;
                    float _fma_135 = __fmaf_rn(_tmem_load_0[7], wgt[0], o_epi[7]);
                    o_epi[7] = _fma_135;
                    float _fma_136 = __fmaf_rn(_tmem_load_0[8], wgt[0], o_epi[8]);
                    o_epi[8] = _fma_136;
                    float _fma_137 = __fmaf_rn(_tmem_load_0[9], wgt[0], o_epi[9]);
                    o_epi[9] = _fma_137;
                    float _fma_138 = __fmaf_rn(_tmem_load_0[10], wgt[0], o_epi[10]);
                    o_epi[10] = _fma_138;
                    float _fma_139 = __fmaf_rn(_tmem_load_0[11], wgt[0], o_epi[11]);
                    o_epi[11] = _fma_139;
                    float _fma_140 = __fmaf_rn(_tmem_load_0[12], wgt[0], o_epi[12]);
                    o_epi[12] = _fma_140;
                    float _fma_141 = __fmaf_rn(_tmem_load_0[13], wgt[0], o_epi[13]);
                    o_epi[13] = _fma_141;
                    float _fma_142 = __fmaf_rn(_tmem_load_0[14], wgt[0], o_epi[14]);
                    o_epi[14] = _fma_142;
                    float _fma_143 = __fmaf_rn(_tmem_load_0[15], wgt[0], o_epi[15]);
                    o_epi[15] = _fma_143;
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_144 = __fmaf_rn(_tmem_load_1[0], wgt[4], o_epi[16]);
                    o_epi[16] = _fma_144;
                    float _fma_145 = __fmaf_rn(_tmem_load_1[1], wgt[4], o_epi[17]);
                    o_epi[17] = _fma_145;
                    float _fma_146 = __fmaf_rn(_tmem_load_1[2], wgt[4], o_epi[18]);
                    o_epi[18] = _fma_146;
                    float _fma_147 = __fmaf_rn(_tmem_load_1[3], wgt[4], o_epi[19]);
                    o_epi[19] = _fma_147;
                    float _fma_148 = __fmaf_rn(_tmem_load_1[4], wgt[4], o_epi[20]);
                    o_epi[20] = _fma_148;
                    float _fma_149 = __fmaf_rn(_tmem_load_1[5], wgt[4], o_epi[21]);
                    o_epi[21] = _fma_149;
                    float _fma_150 = __fmaf_rn(_tmem_load_1[6], wgt[4], o_epi[22]);
                    o_epi[22] = _fma_150;
                    float _fma_151 = __fmaf_rn(_tmem_load_1[7], wgt[4], o_epi[23]);
                    o_epi[23] = _fma_151;
                    float _fma_152 = __fmaf_rn(_tmem_load_1[8], wgt[4], o_epi[24]);
                    o_epi[24] = _fma_152;
                    float _fma_153 = __fmaf_rn(_tmem_load_1[9], wgt[4], o_epi[25]);
                    o_epi[25] = _fma_153;
                    float _fma_154 = __fmaf_rn(_tmem_load_1[10], wgt[4], o_epi[26]);
                    o_epi[26] = _fma_154;
                    float _fma_155 = __fmaf_rn(_tmem_load_1[11], wgt[4], o_epi[27]);
                    o_epi[27] = _fma_155;
                    float _fma_156 = __fmaf_rn(_tmem_load_1[12], wgt[4], o_epi[28]);
                    o_epi[28] = _fma_156;
                    float _fma_157 = __fmaf_rn(_tmem_load_1[13], wgt[4], o_epi[29]);
                    o_epi[29] = _fma_157;
                    float _fma_158 = __fmaf_rn(_tmem_load_1[14], wgt[4], o_epi[30]);
                    o_epi[30] = _fma_158;
                    float _fma_159 = __fmaf_rn(_tmem_load_1[15], wgt[4], o_epi[31]);
                    o_epi[31] = _fma_159;
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 256 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_160 = __fmaf_rn(_tmem_load_2[0], wgt[8], o_epi[32]);
                    o_epi[32] = _fma_160;
                    float _fma_161 = __fmaf_rn(_tmem_load_2[1], wgt[8], o_epi[33]);
                    o_epi[33] = _fma_161;
                    float _fma_162 = __fmaf_rn(_tmem_load_2[2], wgt[8], o_epi[34]);
                    o_epi[34] = _fma_162;
                    float _fma_163 = __fmaf_rn(_tmem_load_2[3], wgt[8], o_epi[35]);
                    o_epi[35] = _fma_163;
                    float _fma_164 = __fmaf_rn(_tmem_load_2[4], wgt[8], o_epi[36]);
                    o_epi[36] = _fma_164;
                    float _fma_165 = __fmaf_rn(_tmem_load_2[5], wgt[8], o_epi[37]);
                    o_epi[37] = _fma_165;
                    float _fma_166 = __fmaf_rn(_tmem_load_2[6], wgt[8], o_epi[38]);
                    o_epi[38] = _fma_166;
                    float _fma_167 = __fmaf_rn(_tmem_load_2[7], wgt[8], o_epi[39]);
                    o_epi[39] = _fma_167;
                    float _fma_168 = __fmaf_rn(_tmem_load_2[8], wgt[8], o_epi[40]);
                    o_epi[40] = _fma_168;
                    float _fma_169 = __fmaf_rn(_tmem_load_2[9], wgt[8], o_epi[41]);
                    o_epi[41] = _fma_169;
                    float _fma_170 = __fmaf_rn(_tmem_load_2[10], wgt[8], o_epi[42]);
                    o_epi[42] = _fma_170;
                    float _fma_171 = __fmaf_rn(_tmem_load_2[11], wgt[8], o_epi[43]);
                    o_epi[43] = _fma_171;
                    float _fma_172 = __fmaf_rn(_tmem_load_2[12], wgt[8], o_epi[44]);
                    o_epi[44] = _fma_172;
                    float _fma_173 = __fmaf_rn(_tmem_load_2[13], wgt[8], o_epi[45]);
                    o_epi[45] = _fma_173;
                    float _fma_174 = __fmaf_rn(_tmem_load_2[14], wgt[8], o_epi[46]);
                    o_epi[46] = _fma_174;
                    float _fma_175 = __fmaf_rn(_tmem_load_2[15], wgt[8], o_epi[47]);
                    o_epi[47] = _fma_175;
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 256 + 48 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_176 = __fmaf_rn(_tmem_load_3[0], wgt[12], o_epi[48]);
                    o_epi[48] = _fma_176;
                    float _fma_177 = __fmaf_rn(_tmem_load_3[1], wgt[12], o_epi[49]);
                    o_epi[49] = _fma_177;
                    float _fma_178 = __fmaf_rn(_tmem_load_3[2], wgt[12], o_epi[50]);
                    o_epi[50] = _fma_178;
                    float _fma_179 = __fmaf_rn(_tmem_load_3[3], wgt[12], o_epi[51]);
                    o_epi[51] = _fma_179;
                    float _fma_180 = __fmaf_rn(_tmem_load_3[4], wgt[12], o_epi[52]);
                    o_epi[52] = _fma_180;
                    float _fma_181 = __fmaf_rn(_tmem_load_3[5], wgt[12], o_epi[53]);
                    o_epi[53] = _fma_181;
                    float _fma_182 = __fmaf_rn(_tmem_load_3[6], wgt[12], o_epi[54]);
                    o_epi[54] = _fma_182;
                    float _fma_183 = __fmaf_rn(_tmem_load_3[7], wgt[12], o_epi[55]);
                    o_epi[55] = _fma_183;
                    float _fma_184 = __fmaf_rn(_tmem_load_3[8], wgt[12], o_epi[56]);
                    o_epi[56] = _fma_184;
                    float _fma_185 = __fmaf_rn(_tmem_load_3[9], wgt[12], o_epi[57]);
                    o_epi[57] = _fma_185;
                    float _fma_186 = __fmaf_rn(_tmem_load_3[10], wgt[12], o_epi[58]);
                    o_epi[58] = _fma_186;
                    float _fma_187 = __fmaf_rn(_tmem_load_3[11], wgt[12], o_epi[59]);
                    o_epi[59] = _fma_187;
                    float _fma_188 = __fmaf_rn(_tmem_load_3[12], wgt[12], o_epi[60]);
                    o_epi[60] = _fma_188;
                    float _fma_189 = __fmaf_rn(_tmem_load_3[13], wgt[12], o_epi[61]);
                    o_epi[61] = _fma_189;
                    float _fma_190 = __fmaf_rn(_tmem_load_3[14], wgt[12], o_epi[62]);
                    o_epi[62] = _fma_190;
                    float _fma_191 = __fmaf_rn(_tmem_load_3[15], wgt[12], o_epi[63]);
                    o_epi[63] = _fma_191;
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 256 + 64 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_192 = __fmaf_rn(_tmem_load_4[0], wgt[1], o_epi[0]);
                    o_epi[0] = _fma_192;
                    float _fma_193 = __fmaf_rn(_tmem_load_4[1], wgt[1], o_epi[1]);
                    o_epi[1] = _fma_193;
                    float _fma_194 = __fmaf_rn(_tmem_load_4[2], wgt[1], o_epi[2]);
                    o_epi[2] = _fma_194;
                    float _fma_195 = __fmaf_rn(_tmem_load_4[3], wgt[1], o_epi[3]);
                    o_epi[3] = _fma_195;
                    float _fma_196 = __fmaf_rn(_tmem_load_4[4], wgt[1], o_epi[4]);
                    o_epi[4] = _fma_196;
                    float _fma_197 = __fmaf_rn(_tmem_load_4[5], wgt[1], o_epi[5]);
                    o_epi[5] = _fma_197;
                    float _fma_198 = __fmaf_rn(_tmem_load_4[6], wgt[1], o_epi[6]);
                    o_epi[6] = _fma_198;
                    float _fma_199 = __fmaf_rn(_tmem_load_4[7], wgt[1], o_epi[7]);
                    o_epi[7] = _fma_199;
                    float _fma_200 = __fmaf_rn(_tmem_load_4[8], wgt[1], o_epi[8]);
                    o_epi[8] = _fma_200;
                    float _fma_201 = __fmaf_rn(_tmem_load_4[9], wgt[1], o_epi[9]);
                    o_epi[9] = _fma_201;
                    float _fma_202 = __fmaf_rn(_tmem_load_4[10], wgt[1], o_epi[10]);
                    o_epi[10] = _fma_202;
                    float _fma_203 = __fmaf_rn(_tmem_load_4[11], wgt[1], o_epi[11]);
                    o_epi[11] = _fma_203;
                    float _fma_204 = __fmaf_rn(_tmem_load_4[12], wgt[1], o_epi[12]);
                    o_epi[12] = _fma_204;
                    float _fma_205 = __fmaf_rn(_tmem_load_4[13], wgt[1], o_epi[13]);
                    o_epi[13] = _fma_205;
                    float _fma_206 = __fmaf_rn(_tmem_load_4[14], wgt[1], o_epi[14]);
                    o_epi[14] = _fma_206;
                    float _fma_207 = __fmaf_rn(_tmem_load_4[15], wgt[1], o_epi[15]);
                    o_epi[15] = _fma_207;
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 256 + 64 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_208 = __fmaf_rn(_tmem_load_5[0], wgt[5], o_epi[16]);
                    o_epi[16] = _fma_208;
                    float _fma_209 = __fmaf_rn(_tmem_load_5[1], wgt[5], o_epi[17]);
                    o_epi[17] = _fma_209;
                    float _fma_210 = __fmaf_rn(_tmem_load_5[2], wgt[5], o_epi[18]);
                    o_epi[18] = _fma_210;
                    float _fma_211 = __fmaf_rn(_tmem_load_5[3], wgt[5], o_epi[19]);
                    o_epi[19] = _fma_211;
                    float _fma_212 = __fmaf_rn(_tmem_load_5[4], wgt[5], o_epi[20]);
                    o_epi[20] = _fma_212;
                    float _fma_213 = __fmaf_rn(_tmem_load_5[5], wgt[5], o_epi[21]);
                    o_epi[21] = _fma_213;
                    float _fma_214 = __fmaf_rn(_tmem_load_5[6], wgt[5], o_epi[22]);
                    o_epi[22] = _fma_214;
                    float _fma_215 = __fmaf_rn(_tmem_load_5[7], wgt[5], o_epi[23]);
                    o_epi[23] = _fma_215;
                    float _fma_216 = __fmaf_rn(_tmem_load_5[8], wgt[5], o_epi[24]);
                    o_epi[24] = _fma_216;
                    float _fma_217 = __fmaf_rn(_tmem_load_5[9], wgt[5], o_epi[25]);
                    o_epi[25] = _fma_217;
                    float _fma_218 = __fmaf_rn(_tmem_load_5[10], wgt[5], o_epi[26]);
                    o_epi[26] = _fma_218;
                    float _fma_219 = __fmaf_rn(_tmem_load_5[11], wgt[5], o_epi[27]);
                    o_epi[27] = _fma_219;
                    float _fma_220 = __fmaf_rn(_tmem_load_5[12], wgt[5], o_epi[28]);
                    o_epi[28] = _fma_220;
                    float _fma_221 = __fmaf_rn(_tmem_load_5[13], wgt[5], o_epi[29]);
                    o_epi[29] = _fma_221;
                    float _fma_222 = __fmaf_rn(_tmem_load_5[14], wgt[5], o_epi[30]);
                    o_epi[30] = _fma_222;
                    float _fma_223 = __fmaf_rn(_tmem_load_5[15], wgt[5], o_epi[31]);
                    o_epi[31] = _fma_223;
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 256 + 64 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_224 = __fmaf_rn(_tmem_load_6[0], wgt[9], o_epi[32]);
                    o_epi[32] = _fma_224;
                    float _fma_225 = __fmaf_rn(_tmem_load_6[1], wgt[9], o_epi[33]);
                    o_epi[33] = _fma_225;
                    float _fma_226 = __fmaf_rn(_tmem_load_6[2], wgt[9], o_epi[34]);
                    o_epi[34] = _fma_226;
                    float _fma_227 = __fmaf_rn(_tmem_load_6[3], wgt[9], o_epi[35]);
                    o_epi[35] = _fma_227;
                    float _fma_228 = __fmaf_rn(_tmem_load_6[4], wgt[9], o_epi[36]);
                    o_epi[36] = _fma_228;
                    float _fma_229 = __fmaf_rn(_tmem_load_6[5], wgt[9], o_epi[37]);
                    o_epi[37] = _fma_229;
                    float _fma_230 = __fmaf_rn(_tmem_load_6[6], wgt[9], o_epi[38]);
                    o_epi[38] = _fma_230;
                    float _fma_231 = __fmaf_rn(_tmem_load_6[7], wgt[9], o_epi[39]);
                    o_epi[39] = _fma_231;
                    float _fma_232 = __fmaf_rn(_tmem_load_6[8], wgt[9], o_epi[40]);
                    o_epi[40] = _fma_232;
                    float _fma_233 = __fmaf_rn(_tmem_load_6[9], wgt[9], o_epi[41]);
                    o_epi[41] = _fma_233;
                    float _fma_234 = __fmaf_rn(_tmem_load_6[10], wgt[9], o_epi[42]);
                    o_epi[42] = _fma_234;
                    float _fma_235 = __fmaf_rn(_tmem_load_6[11], wgt[9], o_epi[43]);
                    o_epi[43] = _fma_235;
                    float _fma_236 = __fmaf_rn(_tmem_load_6[12], wgt[9], o_epi[44]);
                    o_epi[44] = _fma_236;
                    float _fma_237 = __fmaf_rn(_tmem_load_6[13], wgt[9], o_epi[45]);
                    o_epi[45] = _fma_237;
                    float _fma_238 = __fmaf_rn(_tmem_load_6[14], wgt[9], o_epi[46]);
                    o_epi[46] = _fma_238;
                    float _fma_239 = __fmaf_rn(_tmem_load_6[15], wgt[9], o_epi[47]);
                    o_epi[47] = _fma_239;
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 256 + 64 + 48 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_240 = __fmaf_rn(_tmem_load_7[0], wgt[13], o_epi[48]);
                    o_epi[48] = _fma_240;
                    float _fma_241 = __fmaf_rn(_tmem_load_7[1], wgt[13], o_epi[49]);
                    o_epi[49] = _fma_241;
                    float _fma_242 = __fmaf_rn(_tmem_load_7[2], wgt[13], o_epi[50]);
                    o_epi[50] = _fma_242;
                    float _fma_243 = __fmaf_rn(_tmem_load_7[3], wgt[13], o_epi[51]);
                    o_epi[51] = _fma_243;
                    float _fma_244 = __fmaf_rn(_tmem_load_7[4], wgt[13], o_epi[52]);
                    o_epi[52] = _fma_244;
                    float _fma_245 = __fmaf_rn(_tmem_load_7[5], wgt[13], o_epi[53]);
                    o_epi[53] = _fma_245;
                    float _fma_246 = __fmaf_rn(_tmem_load_7[6], wgt[13], o_epi[54]);
                    o_epi[54] = _fma_246;
                    float _fma_247 = __fmaf_rn(_tmem_load_7[7], wgt[13], o_epi[55]);
                    o_epi[55] = _fma_247;
                    float _fma_248 = __fmaf_rn(_tmem_load_7[8], wgt[13], o_epi[56]);
                    o_epi[56] = _fma_248;
                    float _fma_249 = __fmaf_rn(_tmem_load_7[9], wgt[13], o_epi[57]);
                    o_epi[57] = _fma_249;
                    float _fma_250 = __fmaf_rn(_tmem_load_7[10], wgt[13], o_epi[58]);
                    o_epi[58] = _fma_250;
                    float _fma_251 = __fmaf_rn(_tmem_load_7[11], wgt[13], o_epi[59]);
                    o_epi[59] = _fma_251;
                    float _fma_252 = __fmaf_rn(_tmem_load_7[12], wgt[13], o_epi[60]);
                    o_epi[60] = _fma_252;
                    float _fma_253 = __fmaf_rn(_tmem_load_7[13], wgt[13], o_epi[61]);
                    o_epi[61] = _fma_253;
                    float _fma_254 = __fmaf_rn(_tmem_load_7[14], wgt[13], o_epi[62]);
                    o_epi[62] = _fma_254;
                    float _fma_255 = __fmaf_rn(_tmem_load_7[15], wgt[13], o_epi[63]);
                    o_epi[63] = _fma_255;
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 256 + 128 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_256 = __fmaf_rn(_tmem_load_8[0], wgt[2], o_epi[0]);
                    o_epi[0] = _fma_256;
                    float _fma_257 = __fmaf_rn(_tmem_load_8[1], wgt[2], o_epi[1]);
                    o_epi[1] = _fma_257;
                    float _fma_258 = __fmaf_rn(_tmem_load_8[2], wgt[2], o_epi[2]);
                    o_epi[2] = _fma_258;
                    float _fma_259 = __fmaf_rn(_tmem_load_8[3], wgt[2], o_epi[3]);
                    o_epi[3] = _fma_259;
                    float _fma_260 = __fmaf_rn(_tmem_load_8[4], wgt[2], o_epi[4]);
                    o_epi[4] = _fma_260;
                    float _fma_261 = __fmaf_rn(_tmem_load_8[5], wgt[2], o_epi[5]);
                    o_epi[5] = _fma_261;
                    float _fma_262 = __fmaf_rn(_tmem_load_8[6], wgt[2], o_epi[6]);
                    o_epi[6] = _fma_262;
                    float _fma_263 = __fmaf_rn(_tmem_load_8[7], wgt[2], o_epi[7]);
                    o_epi[7] = _fma_263;
                    float _fma_264 = __fmaf_rn(_tmem_load_8[8], wgt[2], o_epi[8]);
                    o_epi[8] = _fma_264;
                    float _fma_265 = __fmaf_rn(_tmem_load_8[9], wgt[2], o_epi[9]);
                    o_epi[9] = _fma_265;
                    float _fma_266 = __fmaf_rn(_tmem_load_8[10], wgt[2], o_epi[10]);
                    o_epi[10] = _fma_266;
                    float _fma_267 = __fmaf_rn(_tmem_load_8[11], wgt[2], o_epi[11]);
                    o_epi[11] = _fma_267;
                    float _fma_268 = __fmaf_rn(_tmem_load_8[12], wgt[2], o_epi[12]);
                    o_epi[12] = _fma_268;
                    float _fma_269 = __fmaf_rn(_tmem_load_8[13], wgt[2], o_epi[13]);
                    o_epi[13] = _fma_269;
                    float _fma_270 = __fmaf_rn(_tmem_load_8[14], wgt[2], o_epi[14]);
                    o_epi[14] = _fma_270;
                    float _fma_271 = __fmaf_rn(_tmem_load_8[15], wgt[2], o_epi[15]);
                    o_epi[15] = _fma_271;
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 256 + 128 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_272 = __fmaf_rn(_tmem_load_9[0], wgt[6], o_epi[16]);
                    o_epi[16] = _fma_272;
                    float _fma_273 = __fmaf_rn(_tmem_load_9[1], wgt[6], o_epi[17]);
                    o_epi[17] = _fma_273;
                    float _fma_274 = __fmaf_rn(_tmem_load_9[2], wgt[6], o_epi[18]);
                    o_epi[18] = _fma_274;
                    float _fma_275 = __fmaf_rn(_tmem_load_9[3], wgt[6], o_epi[19]);
                    o_epi[19] = _fma_275;
                    float _fma_276 = __fmaf_rn(_tmem_load_9[4], wgt[6], o_epi[20]);
                    o_epi[20] = _fma_276;
                    float _fma_277 = __fmaf_rn(_tmem_load_9[5], wgt[6], o_epi[21]);
                    o_epi[21] = _fma_277;
                    float _fma_278 = __fmaf_rn(_tmem_load_9[6], wgt[6], o_epi[22]);
                    o_epi[22] = _fma_278;
                    float _fma_279 = __fmaf_rn(_tmem_load_9[7], wgt[6], o_epi[23]);
                    o_epi[23] = _fma_279;
                    float _fma_280 = __fmaf_rn(_tmem_load_9[8], wgt[6], o_epi[24]);
                    o_epi[24] = _fma_280;
                    float _fma_281 = __fmaf_rn(_tmem_load_9[9], wgt[6], o_epi[25]);
                    o_epi[25] = _fma_281;
                    float _fma_282 = __fmaf_rn(_tmem_load_9[10], wgt[6], o_epi[26]);
                    o_epi[26] = _fma_282;
                    float _fma_283 = __fmaf_rn(_tmem_load_9[11], wgt[6], o_epi[27]);
                    o_epi[27] = _fma_283;
                    float _fma_284 = __fmaf_rn(_tmem_load_9[12], wgt[6], o_epi[28]);
                    o_epi[28] = _fma_284;
                    float _fma_285 = __fmaf_rn(_tmem_load_9[13], wgt[6], o_epi[29]);
                    o_epi[29] = _fma_285;
                    float _fma_286 = __fmaf_rn(_tmem_load_9[14], wgt[6], o_epi[30]);
                    o_epi[30] = _fma_286;
                    float _fma_287 = __fmaf_rn(_tmem_load_9[15], wgt[6], o_epi[31]);
                    o_epi[31] = _fma_287;
                    float _tmem_load_10[16];
                    tmem_ld_x16(&_tmem_load_10[0], taddr + 256 + 128 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_288 = __fmaf_rn(_tmem_load_10[0], wgt[10], o_epi[32]);
                    o_epi[32] = _fma_288;
                    float _fma_289 = __fmaf_rn(_tmem_load_10[1], wgt[10], o_epi[33]);
                    o_epi[33] = _fma_289;
                    float _fma_290 = __fmaf_rn(_tmem_load_10[2], wgt[10], o_epi[34]);
                    o_epi[34] = _fma_290;
                    float _fma_291 = __fmaf_rn(_tmem_load_10[3], wgt[10], o_epi[35]);
                    o_epi[35] = _fma_291;
                    float _fma_292 = __fmaf_rn(_tmem_load_10[4], wgt[10], o_epi[36]);
                    o_epi[36] = _fma_292;
                    float _fma_293 = __fmaf_rn(_tmem_load_10[5], wgt[10], o_epi[37]);
                    o_epi[37] = _fma_293;
                    float _fma_294 = __fmaf_rn(_tmem_load_10[6], wgt[10], o_epi[38]);
                    o_epi[38] = _fma_294;
                    float _fma_295 = __fmaf_rn(_tmem_load_10[7], wgt[10], o_epi[39]);
                    o_epi[39] = _fma_295;
                    float _fma_296 = __fmaf_rn(_tmem_load_10[8], wgt[10], o_epi[40]);
                    o_epi[40] = _fma_296;
                    float _fma_297 = __fmaf_rn(_tmem_load_10[9], wgt[10], o_epi[41]);
                    o_epi[41] = _fma_297;
                    float _fma_298 = __fmaf_rn(_tmem_load_10[10], wgt[10], o_epi[42]);
                    o_epi[42] = _fma_298;
                    float _fma_299 = __fmaf_rn(_tmem_load_10[11], wgt[10], o_epi[43]);
                    o_epi[43] = _fma_299;
                    float _fma_300 = __fmaf_rn(_tmem_load_10[12], wgt[10], o_epi[44]);
                    o_epi[44] = _fma_300;
                    float _fma_301 = __fmaf_rn(_tmem_load_10[13], wgt[10], o_epi[45]);
                    o_epi[45] = _fma_301;
                    float _fma_302 = __fmaf_rn(_tmem_load_10[14], wgt[10], o_epi[46]);
                    o_epi[46] = _fma_302;
                    float _fma_303 = __fmaf_rn(_tmem_load_10[15], wgt[10], o_epi[47]);
                    o_epi[47] = _fma_303;
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 256 + 128 + 48 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_304 = __fmaf_rn(_tmem_load_11[0], wgt[14], o_epi[48]);
                    o_epi[48] = _fma_304;
                    float _fma_305 = __fmaf_rn(_tmem_load_11[1], wgt[14], o_epi[49]);
                    o_epi[49] = _fma_305;
                    float _fma_306 = __fmaf_rn(_tmem_load_11[2], wgt[14], o_epi[50]);
                    o_epi[50] = _fma_306;
                    float _fma_307 = __fmaf_rn(_tmem_load_11[3], wgt[14], o_epi[51]);
                    o_epi[51] = _fma_307;
                    float _fma_308 = __fmaf_rn(_tmem_load_11[4], wgt[14], o_epi[52]);
                    o_epi[52] = _fma_308;
                    float _fma_309 = __fmaf_rn(_tmem_load_11[5], wgt[14], o_epi[53]);
                    o_epi[53] = _fma_309;
                    float _fma_310 = __fmaf_rn(_tmem_load_11[6], wgt[14], o_epi[54]);
                    o_epi[54] = _fma_310;
                    float _fma_311 = __fmaf_rn(_tmem_load_11[7], wgt[14], o_epi[55]);
                    o_epi[55] = _fma_311;
                    float _fma_312 = __fmaf_rn(_tmem_load_11[8], wgt[14], o_epi[56]);
                    o_epi[56] = _fma_312;
                    float _fma_313 = __fmaf_rn(_tmem_load_11[9], wgt[14], o_epi[57]);
                    o_epi[57] = _fma_313;
                    float _fma_314 = __fmaf_rn(_tmem_load_11[10], wgt[14], o_epi[58]);
                    o_epi[58] = _fma_314;
                    float _fma_315 = __fmaf_rn(_tmem_load_11[11], wgt[14], o_epi[59]);
                    o_epi[59] = _fma_315;
                    float _fma_316 = __fmaf_rn(_tmem_load_11[12], wgt[14], o_epi[60]);
                    o_epi[60] = _fma_316;
                    float _fma_317 = __fmaf_rn(_tmem_load_11[13], wgt[14], o_epi[61]);
                    o_epi[61] = _fma_317;
                    float _fma_318 = __fmaf_rn(_tmem_load_11[14], wgt[14], o_epi[62]);
                    o_epi[62] = _fma_318;
                    float _fma_319 = __fmaf_rn(_tmem_load_11[15], wgt[14], o_epi[63]);
                    o_epi[63] = _fma_319;
                    float _tmem_load_12[16];
                    tmem_ld_x16(&_tmem_load_12[0], taddr + 256 + 192 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_320 = __fmaf_rn(_tmem_load_12[0], wgt[3], o_epi[0]);
                    o_epi[0] = _fma_320;
                    float _fma_321 = __fmaf_rn(_tmem_load_12[1], wgt[3], o_epi[1]);
                    o_epi[1] = _fma_321;
                    float _fma_322 = __fmaf_rn(_tmem_load_12[2], wgt[3], o_epi[2]);
                    o_epi[2] = _fma_322;
                    float _fma_323 = __fmaf_rn(_tmem_load_12[3], wgt[3], o_epi[3]);
                    o_epi[3] = _fma_323;
                    float _fma_324 = __fmaf_rn(_tmem_load_12[4], wgt[3], o_epi[4]);
                    o_epi[4] = _fma_324;
                    float _fma_325 = __fmaf_rn(_tmem_load_12[5], wgt[3], o_epi[5]);
                    o_epi[5] = _fma_325;
                    float _fma_326 = __fmaf_rn(_tmem_load_12[6], wgt[3], o_epi[6]);
                    o_epi[6] = _fma_326;
                    float _fma_327 = __fmaf_rn(_tmem_load_12[7], wgt[3], o_epi[7]);
                    o_epi[7] = _fma_327;
                    float _fma_328 = __fmaf_rn(_tmem_load_12[8], wgt[3], o_epi[8]);
                    o_epi[8] = _fma_328;
                    float _fma_329 = __fmaf_rn(_tmem_load_12[9], wgt[3], o_epi[9]);
                    o_epi[9] = _fma_329;
                    float _fma_330 = __fmaf_rn(_tmem_load_12[10], wgt[3], o_epi[10]);
                    o_epi[10] = _fma_330;
                    float _fma_331 = __fmaf_rn(_tmem_load_12[11], wgt[3], o_epi[11]);
                    o_epi[11] = _fma_331;
                    float _fma_332 = __fmaf_rn(_tmem_load_12[12], wgt[3], o_epi[12]);
                    o_epi[12] = _fma_332;
                    float _fma_333 = __fmaf_rn(_tmem_load_12[13], wgt[3], o_epi[13]);
                    o_epi[13] = _fma_333;
                    float _fma_334 = __fmaf_rn(_tmem_load_12[14], wgt[3], o_epi[14]);
                    o_epi[14] = _fma_334;
                    float _fma_335 = __fmaf_rn(_tmem_load_12[15], wgt[3], o_epi[15]);
                    o_epi[15] = _fma_335;
                    float _tmem_load_13[16];
                    tmem_ld_x16(&_tmem_load_13[0], taddr + 256 + 192 + 16 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_336 = __fmaf_rn(_tmem_load_13[0], wgt[7], o_epi[16]);
                    o_epi[16] = _fma_336;
                    float _fma_337 = __fmaf_rn(_tmem_load_13[1], wgt[7], o_epi[17]);
                    o_epi[17] = _fma_337;
                    float _fma_338 = __fmaf_rn(_tmem_load_13[2], wgt[7], o_epi[18]);
                    o_epi[18] = _fma_338;
                    float _fma_339 = __fmaf_rn(_tmem_load_13[3], wgt[7], o_epi[19]);
                    o_epi[19] = _fma_339;
                    float _fma_340 = __fmaf_rn(_tmem_load_13[4], wgt[7], o_epi[20]);
                    o_epi[20] = _fma_340;
                    float _fma_341 = __fmaf_rn(_tmem_load_13[5], wgt[7], o_epi[21]);
                    o_epi[21] = _fma_341;
                    float _fma_342 = __fmaf_rn(_tmem_load_13[6], wgt[7], o_epi[22]);
                    o_epi[22] = _fma_342;
                    float _fma_343 = __fmaf_rn(_tmem_load_13[7], wgt[7], o_epi[23]);
                    o_epi[23] = _fma_343;
                    float _fma_344 = __fmaf_rn(_tmem_load_13[8], wgt[7], o_epi[24]);
                    o_epi[24] = _fma_344;
                    float _fma_345 = __fmaf_rn(_tmem_load_13[9], wgt[7], o_epi[25]);
                    o_epi[25] = _fma_345;
                    float _fma_346 = __fmaf_rn(_tmem_load_13[10], wgt[7], o_epi[26]);
                    o_epi[26] = _fma_346;
                    float _fma_347 = __fmaf_rn(_tmem_load_13[11], wgt[7], o_epi[27]);
                    o_epi[27] = _fma_347;
                    float _fma_348 = __fmaf_rn(_tmem_load_13[12], wgt[7], o_epi[28]);
                    o_epi[28] = _fma_348;
                    float _fma_349 = __fmaf_rn(_tmem_load_13[13], wgt[7], o_epi[29]);
                    o_epi[29] = _fma_349;
                    float _fma_350 = __fmaf_rn(_tmem_load_13[14], wgt[7], o_epi[30]);
                    o_epi[30] = _fma_350;
                    float _fma_351 = __fmaf_rn(_tmem_load_13[15], wgt[7], o_epi[31]);
                    o_epi[31] = _fma_351;
                    float _tmem_load_14[16];
                    tmem_ld_x16(&_tmem_load_14[0], taddr + 256 + 192 + 32 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_352 = __fmaf_rn(_tmem_load_14[0], wgt[11], o_epi[32]);
                    o_epi[32] = _fma_352;
                    float _fma_353 = __fmaf_rn(_tmem_load_14[1], wgt[11], o_epi[33]);
                    o_epi[33] = _fma_353;
                    float _fma_354 = __fmaf_rn(_tmem_load_14[2], wgt[11], o_epi[34]);
                    o_epi[34] = _fma_354;
                    float _fma_355 = __fmaf_rn(_tmem_load_14[3], wgt[11], o_epi[35]);
                    o_epi[35] = _fma_355;
                    float _fma_356 = __fmaf_rn(_tmem_load_14[4], wgt[11], o_epi[36]);
                    o_epi[36] = _fma_356;
                    float _fma_357 = __fmaf_rn(_tmem_load_14[5], wgt[11], o_epi[37]);
                    o_epi[37] = _fma_357;
                    float _fma_358 = __fmaf_rn(_tmem_load_14[6], wgt[11], o_epi[38]);
                    o_epi[38] = _fma_358;
                    float _fma_359 = __fmaf_rn(_tmem_load_14[7], wgt[11], o_epi[39]);
                    o_epi[39] = _fma_359;
                    float _fma_360 = __fmaf_rn(_tmem_load_14[8], wgt[11], o_epi[40]);
                    o_epi[40] = _fma_360;
                    float _fma_361 = __fmaf_rn(_tmem_load_14[9], wgt[11], o_epi[41]);
                    o_epi[41] = _fma_361;
                    float _fma_362 = __fmaf_rn(_tmem_load_14[10], wgt[11], o_epi[42]);
                    o_epi[42] = _fma_362;
                    float _fma_363 = __fmaf_rn(_tmem_load_14[11], wgt[11], o_epi[43]);
                    o_epi[43] = _fma_363;
                    float _fma_364 = __fmaf_rn(_tmem_load_14[12], wgt[11], o_epi[44]);
                    o_epi[44] = _fma_364;
                    float _fma_365 = __fmaf_rn(_tmem_load_14[13], wgt[11], o_epi[45]);
                    o_epi[45] = _fma_365;
                    float _fma_366 = __fmaf_rn(_tmem_load_14[14], wgt[11], o_epi[46]);
                    o_epi[46] = _fma_366;
                    float _fma_367 = __fmaf_rn(_tmem_load_14[15], wgt[11], o_epi[47]);
                    o_epi[47] = _fma_367;
                    float _tmem_load_15[16];
                    tmem_ld_x16(&_tmem_load_15[0], taddr + 256 + 192 + 48 + (unsigned int)tmem_lane_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _fma_368 = __fmaf_rn(_tmem_load_15[0], wgt[15], o_epi[48]);
                    o_epi[48] = _fma_368;
                    float _fma_369 = __fmaf_rn(_tmem_load_15[1], wgt[15], o_epi[49]);
                    o_epi[49] = _fma_369;
                    float _fma_370 = __fmaf_rn(_tmem_load_15[2], wgt[15], o_epi[50]);
                    o_epi[50] = _fma_370;
                    float _fma_371 = __fmaf_rn(_tmem_load_15[3], wgt[15], o_epi[51]);
                    o_epi[51] = _fma_371;
                    float _fma_372 = __fmaf_rn(_tmem_load_15[4], wgt[15], o_epi[52]);
                    o_epi[52] = _fma_372;
                    float _fma_373 = __fmaf_rn(_tmem_load_15[5], wgt[15], o_epi[53]);
                    o_epi[53] = _fma_373;
                    float _fma_374 = __fmaf_rn(_tmem_load_15[6], wgt[15], o_epi[54]);
                    o_epi[54] = _fma_374;
                    float _fma_375 = __fmaf_rn(_tmem_load_15[7], wgt[15], o_epi[55]);
                    o_epi[55] = _fma_375;
                    float _fma_376 = __fmaf_rn(_tmem_load_15[8], wgt[15], o_epi[56]);
                    o_epi[56] = _fma_376;
                    float _fma_377 = __fmaf_rn(_tmem_load_15[9], wgt[15], o_epi[57]);
                    o_epi[57] = _fma_377;
                    float _fma_378 = __fmaf_rn(_tmem_load_15[10], wgt[15], o_epi[58]);
                    o_epi[58] = _fma_378;
                    float _fma_379 = __fmaf_rn(_tmem_load_15[11], wgt[15], o_epi[59]);
                    o_epi[59] = _fma_379;
                    float _fma_380 = __fmaf_rn(_tmem_load_15[12], wgt[15], o_epi[60]);
                    o_epi[60] = _fma_380;
                    float _fma_381 = __fmaf_rn(_tmem_load_15[13], wgt[15], o_epi[61]);
                    o_epi[61] = _fma_381;
                    float _fma_382 = __fmaf_rn(_tmem_load_15[14], wgt[15], o_epi[62]);
                    o_epi[62] = _fma_382;
                    float _fma_383 = __fmaf_rn(_tmem_load_15[15], wgt[15], o_epi[63]);
                    o_epi[63] = _fma_383;
                }
                if (total_units_sm <= work_idx + (unsigned int)gridDim.x) {
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int q_row0 = request * seqlen_q * num_q_heads + kv_head * group_size;
                float _rcp_0 = approx_rcp(head_sum[0]);
                float inv_lane = ((head_sum[0] > 0.0f) ? output_scale * _rcp_0 : 0.0f);
                if (warp_in_role == 0) {
                    if (group_size > lane) {
                        float lse_lane = -CAKE_INF;
                        if (head_sum[0] > 0.0f) {
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(head_sum[0]));
                            lse_lane = m_ab[0] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                        }
                        *(reinterpret_cast<float*>(msa_lse + (q_row0 + lane)) + (0)) = lse_lane;
                    }
                }
                float inv_all[16];
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_lane, 0);
                inv_all[0] = _shfl_0;
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, inv_lane, 1);
                inv_all[1] = _shfl_1;
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, inv_lane, 2);
                inv_all[2] = _shfl_2;
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, inv_lane, 3);
                inv_all[3] = _shfl_3;
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, inv_lane, 4);
                inv_all[4] = _shfl_4;
                float _shfl_5 = __shfl_sync(0xFFFFFFFF, inv_lane, 5);
                inv_all[5] = _shfl_5;
                float _shfl_6 = __shfl_sync(0xFFFFFFFF, inv_lane, 6);
                inv_all[6] = _shfl_6;
                float _shfl_7 = __shfl_sync(0xFFFFFFFF, inv_lane, 7);
                inv_all[7] = _shfl_7;
                float _shfl_8 = __shfl_sync(0xFFFFFFFF, inv_lane, 8);
                inv_all[8] = _shfl_8;
                float _shfl_9 = __shfl_sync(0xFFFFFFFF, inv_lane, 9);
                inv_all[9] = _shfl_9;
                float _shfl_10 = __shfl_sync(0xFFFFFFFF, inv_lane, 10);
                inv_all[10] = _shfl_10;
                float _shfl_11 = __shfl_sync(0xFFFFFFFF, inv_lane, 11);
                inv_all[11] = _shfl_11;
                float _shfl_12 = __shfl_sync(0xFFFFFFFF, inv_lane, 12);
                inv_all[12] = _shfl_12;
                float _shfl_13 = __shfl_sync(0xFFFFFFFF, inv_lane, 13);
                inv_all[13] = _shfl_13;
                float _shfl_14 = __shfl_sync(0xFFFFFFFF, inv_lane, 14);
                inv_all[14] = _shfl_14;
                float _shfl_15 = __shfl_sync(0xFFFFFFFF, inv_lane, 15);
                inv_all[15] = _shfl_15;
                if (group_size > 0) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + (q_row0 * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[0] * inv_all[0]);
                }
                if (group_size > 1) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 1) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[1] * inv_all[1]);
                }
                if (group_size > 2) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 2) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[2] * inv_all[2]);
                }
                if (group_size > 3) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 3) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[3] * inv_all[3]);
                }
                if (group_size > 4) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 4) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[4] * inv_all[4]);
                }
                if (group_size > 5) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 5) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[5] * inv_all[5]);
                }
                if (group_size > 6) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 6) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[6] * inv_all[6]);
                }
                if (group_size > 7) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 7) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[7] * inv_all[7]);
                }
                if (group_size > 8) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 8) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[8] * inv_all[8]);
                }
                if (group_size > 9) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 9) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[9] * inv_all[9]);
                }
                if (group_size > 10) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 10) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[10] * inv_all[10]);
                }
                if (group_size > 11) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 11) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[11] * inv_all[11]);
                }
                if (group_size > 12) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 12) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[12] * inv_all[12]);
                }
                if (group_size > 13) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 13) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[13] * inv_all[13]);
                }
                if (group_size > 14) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 14) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[14] * inv_all[14]);
                }
                if (group_size > 15) {
                    *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0 + 15) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[15] * inv_all[15]);
                }
                if (seqlen_q > 1) {
                    int q_row0_0 = (request * seqlen_q + 1) * num_q_heads + kv_head * group_size;
                    float _rcp_1 = approx_rcp(head_sum[1]);
                    float inv_lane_1 = ((head_sum[1] > 0.0f) ? output_scale * _rcp_1 : 0.0f);
                    if (warp_in_role == 0) {
                        if (group_size > lane) {
                            float lse_lane_1 = -CAKE_INF;
                            if (head_sum[1] > 0.0f) {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(head_sum[1]));
                                lse_lane_1 = m_ab[1] * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + (q_row0_0 + lane)) + (0)) = lse_lane_1;
                        }
                    }
                    float inv_all_2[16];
                    float _shfl_16 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 0);
                    inv_all_2[0] = _shfl_16;
                    float _shfl_17 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 1);
                    inv_all_2[1] = _shfl_17;
                    float _shfl_18 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 2);
                    inv_all_2[2] = _shfl_18;
                    float _shfl_19 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 3);
                    inv_all_2[3] = _shfl_19;
                    float _shfl_20 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 4);
                    inv_all_2[4] = _shfl_20;
                    float _shfl_21 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 5);
                    inv_all_2[5] = _shfl_21;
                    float _shfl_22 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 6);
                    inv_all_2[6] = _shfl_22;
                    float _shfl_23 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 7);
                    inv_all_2[7] = _shfl_23;
                    float _shfl_24 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 8);
                    inv_all_2[8] = _shfl_24;
                    float _shfl_25 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 9);
                    inv_all_2[9] = _shfl_25;
                    float _shfl_26 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 10);
                    inv_all_2[10] = _shfl_26;
                    float _shfl_27 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 11);
                    inv_all_2[11] = _shfl_27;
                    float _shfl_28 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 12);
                    inv_all_2[12] = _shfl_28;
                    float _shfl_29 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 13);
                    inv_all_2[13] = _shfl_29;
                    float _shfl_30 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 14);
                    inv_all_2[14] = _shfl_30;
                    float _shfl_31 = __shfl_sync(0xFFFFFFFF, inv_lane_1, 15);
                    inv_all_2[15] = _shfl_31;
                    if (group_size > 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + (q_row0_0 * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[16] * inv_all_2[0]);
                    }
                    if (group_size > 1) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 1) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[17] * inv_all_2[1]);
                    }
                    if (group_size > 2) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 2) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[18] * inv_all_2[2]);
                    }
                    if (group_size > 3) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 3) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[19] * inv_all_2[3]);
                    }
                    if (group_size > 4) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 4) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[20] * inv_all_2[4]);
                    }
                    if (group_size > 5) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 5) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[21] * inv_all_2[5]);
                    }
                    if (group_size > 6) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 6) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[22] * inv_all_2[6]);
                    }
                    if (group_size > 7) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 7) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[23] * inv_all_2[7]);
                    }
                    if (group_size > 8) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 8) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[24] * inv_all_2[8]);
                    }
                    if (group_size > 9) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 9) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[25] * inv_all_2[9]);
                    }
                    if (group_size > 10) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 10) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[26] * inv_all_2[10]);
                    }
                    if (group_size > 11) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 11) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[27] * inv_all_2[11]);
                    }
                    if (group_size > 12) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 12) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[28] * inv_all_2[12]);
                    }
                    if (group_size > 13) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 13) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[29] * inv_all_2[13]);
                    }
                    if (group_size > 14) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 14) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[30] * inv_all_2[14]);
                    }
                    if (group_size > 15) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0 + 15) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[31] * inv_all_2[15]);
                    }
                }
                if (seqlen_q > 2) {
                    int q_row0_0_1 = (request * seqlen_q + 2) * num_q_heads + kv_head * group_size;
                    float _rcp_2 = approx_rcp(head_sum[2]);
                    float inv_lane_1_1 = ((head_sum[2] > 0.0f) ? output_scale * _rcp_2 : 0.0f);
                    if (warp_in_role == 0) {
                        if (group_size > lane) {
                            float lse_lane_2 = -CAKE_INF;
                            if (head_sum[2] > 0.0f) {
                                float _log2_2;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(head_sum[2]));
                                lse_lane_2 = m_ab[2] * softmax_scale_log2 * 0.6931471805599453f + _log2_2 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + (q_row0_0_1 + lane)) + (0)) = lse_lane_2;
                        }
                    }
                    float inv_all_2_1[16];
                    float _shfl_32 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 0);
                    inv_all_2_1[0] = _shfl_32;
                    float _shfl_33 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 1);
                    inv_all_2_1[1] = _shfl_33;
                    float _shfl_34 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 2);
                    inv_all_2_1[2] = _shfl_34;
                    float _shfl_35 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 3);
                    inv_all_2_1[3] = _shfl_35;
                    float _shfl_36 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 4);
                    inv_all_2_1[4] = _shfl_36;
                    float _shfl_37 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 5);
                    inv_all_2_1[5] = _shfl_37;
                    float _shfl_38 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 6);
                    inv_all_2_1[6] = _shfl_38;
                    float _shfl_39 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 7);
                    inv_all_2_1[7] = _shfl_39;
                    float _shfl_40 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 8);
                    inv_all_2_1[8] = _shfl_40;
                    float _shfl_41 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 9);
                    inv_all_2_1[9] = _shfl_41;
                    float _shfl_42 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 10);
                    inv_all_2_1[10] = _shfl_42;
                    float _shfl_43 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 11);
                    inv_all_2_1[11] = _shfl_43;
                    float _shfl_44 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 12);
                    inv_all_2_1[12] = _shfl_44;
                    float _shfl_45 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 13);
                    inv_all_2_1[13] = _shfl_45;
                    float _shfl_46 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 14);
                    inv_all_2_1[14] = _shfl_46;
                    float _shfl_47 = __shfl_sync(0xFFFFFFFF, inv_lane_1_1, 15);
                    inv_all_2_1[15] = _shfl_47;
                    if (group_size > 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + (q_row0_0_1 * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[32] * inv_all_2_1[0]);
                    }
                    if (group_size > 1) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 1) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[33] * inv_all_2_1[1]);
                    }
                    if (group_size > 2) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 2) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[34] * inv_all_2_1[2]);
                    }
                    if (group_size > 3) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 3) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[35] * inv_all_2_1[3]);
                    }
                    if (group_size > 4) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 4) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[36] * inv_all_2_1[4]);
                    }
                    if (group_size > 5) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 5) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[37] * inv_all_2_1[5]);
                    }
                    if (group_size > 6) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 6) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[38] * inv_all_2_1[6]);
                    }
                    if (group_size > 7) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 7) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[39] * inv_all_2_1[7]);
                    }
                    if (group_size > 8) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 8) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[40] * inv_all_2_1[8]);
                    }
                    if (group_size > 9) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 9) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[41] * inv_all_2_1[9]);
                    }
                    if (group_size > 10) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 10) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[42] * inv_all_2_1[10]);
                    }
                    if (group_size > 11) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 11) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[43] * inv_all_2_1[11]);
                    }
                    if (group_size > 12) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 12) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[44] * inv_all_2_1[12]);
                    }
                    if (group_size > 13) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 13) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[45] * inv_all_2_1[13]);
                    }
                    if (group_size > 14) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 14) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[46] * inv_all_2_1[14]);
                    }
                    if (group_size > 15) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_1 + 15) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[47] * inv_all_2_1[15]);
                    }
                }
                if (seqlen_q > 3) {
                    int q_row0_0_2 = (request * seqlen_q + 3) * num_q_heads + kv_head * group_size;
                    float _rcp_3 = approx_rcp(head_sum[3]);
                    float inv_lane_1_2 = ((head_sum[3] > 0.0f) ? output_scale * _rcp_3 : 0.0f);
                    if (warp_in_role == 0) {
                        if (group_size > lane) {
                            float lse_lane_3 = -CAKE_INF;
                            if (head_sum[3] > 0.0f) {
                                float _log2_3;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(head_sum[3]));
                                lse_lane_3 = m_ab[3] * softmax_scale_log2 * 0.6931471805599453f + _log2_3 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + (q_row0_0_2 + lane)) + (0)) = lse_lane_3;
                        }
                    }
                    float inv_all_2_2[16];
                    float _shfl_48 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 0);
                    inv_all_2_2[0] = _shfl_48;
                    float _shfl_49 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 1);
                    inv_all_2_2[1] = _shfl_49;
                    float _shfl_50 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 2);
                    inv_all_2_2[2] = _shfl_50;
                    float _shfl_51 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 3);
                    inv_all_2_2[3] = _shfl_51;
                    float _shfl_52 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 4);
                    inv_all_2_2[4] = _shfl_52;
                    float _shfl_53 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 5);
                    inv_all_2_2[5] = _shfl_53;
                    float _shfl_54 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 6);
                    inv_all_2_2[6] = _shfl_54;
                    float _shfl_55 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 7);
                    inv_all_2_2[7] = _shfl_55;
                    float _shfl_56 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 8);
                    inv_all_2_2[8] = _shfl_56;
                    float _shfl_57 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 9);
                    inv_all_2_2[9] = _shfl_57;
                    float _shfl_58 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 10);
                    inv_all_2_2[10] = _shfl_58;
                    float _shfl_59 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 11);
                    inv_all_2_2[11] = _shfl_59;
                    float _shfl_60 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 12);
                    inv_all_2_2[12] = _shfl_60;
                    float _shfl_61 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 13);
                    inv_all_2_2[13] = _shfl_61;
                    float _shfl_62 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 14);
                    inv_all_2_2[14] = _shfl_62;
                    float _shfl_63 = __shfl_sync(0xFFFFFFFF, inv_lane_1_2, 15);
                    inv_all_2_2[15] = _shfl_63;
                    if (group_size > 0) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + (q_row0_0_2 * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[48] * inv_all_2_2[0]);
                    }
                    if (group_size > 1) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 1) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[49] * inv_all_2_2[1]);
                    }
                    if (group_size > 2) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 2) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[50] * inv_all_2_2[2]);
                    }
                    if (group_size > 3) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 3) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[51] * inv_all_2_2[3]);
                    }
                    if (group_size > 4) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 4) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[52] * inv_all_2_2[4]);
                    }
                    if (group_size > 5) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 5) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[53] * inv_all_2_2[5]);
                    }
                    if (group_size > 6) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 6) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[54] * inv_all_2_2[6]);
                    }
                    if (group_size > 7) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 7) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[55] * inv_all_2_2[7]);
                    }
                    if (group_size > 8) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 8) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[56] * inv_all_2_2[8]);
                    }
                    if (group_size > 9) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 9) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[57] * inv_all_2_2[9]);
                    }
                    if (group_size > 10) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 10) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[58] * inv_all_2_2[10]);
                    }
                    if (group_size > 11) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 11) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[59] * inv_all_2_2[11]);
                    }
                    if (group_size > 12) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 12) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[60] * inv_all_2_2[12]);
                    }
                    if (group_size > 13) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 13) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[61] * inv_all_2_2[13]);
                    }
                    if (group_size > 14) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 14) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[62] * inv_all_2_2[14]);
                    }
                    if (group_size > 15) {
                        *(reinterpret_cast<__nv_bfloat16*>(O + ((q_row0_0_2 + 15) * 128 + token)) + (0)) = __float2bfloat16_rn(o_epi[63] * inv_all_2_2[15]);
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
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
            for (unsigned int work_idx_k = blockIdx.x; work_idx_k < total_units_k; work_idx_k += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_k) * 8, item_phase_k);
                int npairs_k = smem_npairs[item_stage_k];
                item_stage_k += 1;
                if (item_stage_k == 2) { item_stage_k = 0; item_phase_k ^= 1; }
                #pragma unroll 1
                for (int sp = 0; sp < 32; sp++) {
                    if (npairs_k <= sp) {
                        break;
                    }
                    int kl_k = 2 * sp;
                    if (sp >= 2) {
                        kl_k = 4 * sp - 4;
                    }
                    #pragma unroll 1
                    for (int jv = 0; jv < 2; jv++) {
                        int st = (tile_base_k + kl_k + jv) % 10;
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
            for (unsigned int work_idx_t = blockIdx.x; work_idx_t < total_units_t; work_idx_t += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_t) * 8, item_phase_t);
                int npairs_t = smem_npairs[item_stage_t];
                item_stage_t += 1;
                if (item_stage_t == 2) { item_stage_t = 0; item_phase_t ^= 1; }
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 32; pair_1++) {
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
                        int st_1 = tile_n % 10;
                        int fst_1 = 16 + vtile_t % 16;
                        int fph_1 = vtile_t / 16 & 1;
                        int vs = vtile_t % 2;
                        int vph = 1 - (vtile_t / 2 & 1);
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
            unsigned int kb[2];
            int ki[2];
            unsigned int first_idx = blockIdx.x;
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
            int tok_e_1 = thalf + 2;
            int tok_c_2 = ((tok_e_1 < seqlen_q) ? tok_e_1 : seqlen_q - 1);
            int id_raw_3 = task_kind[(request_f * seqlen_q + tok_c_2) * q2k_token_stride + kv_head_f * q2k_head_stride + jpos];
            int id_tok_4 = ((tok_e_1 < seqlen_q) ? id_raw_3 : -1);
            int id_c_5 = ((id_tok_4 >= 0) ? id_tok_4 : 0);
            int page_raw_6 = kv_indices[request_f * msa_max_pages + id_c_5];
            int page_e_7 = ((id_tok_4 >= 0) ? page_raw_6 : -1);
            int id_e_8 = ((page_e_7 < 0) ? -1 : id_tok_4);
            kb[1] = (unsigned int)(-1 - id_e_8);
            ki[1] = ((id_e_8 < 0) ? 0 : page_e_7 << 3 | tok_e_1);
            unsigned int ck = kb[0];
            int ci = ki[0];
            unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, ck, 1);
            unsigned int pb = _shfl_xor_20;
            int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, ci, 1);
            int pc = _shfl_xor_21;
            int r = (int)(pb > ck || pb == ck && pc < ci);
            int po = r;
            int lower = (int)((lane_0 & 1) == 0);
            int desc = (int)((lane_0 & 2) == 0);
            int wmax = desc * lower + (1 - desc) * (1 - lower);
            int tk = ((wmax != 0) ? po : 1 - po);
            kb[0] = ((tk != 0) ? pb : ck);
            ki[0] = ((tk != 0) ? pc : ci);
            unsigned int ck_9 = kb[1];
            int ci_10 = ki[1];
            unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, ck_9, 1);
            unsigned int pb_11 = _shfl_xor_22;
            int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, ci_10, 1);
            int pc_12 = _shfl_xor_23;
            int r_13 = (int)(pb_11 > ck_9 || pb_11 == ck_9 && pc_12 < ci_10);
            int po_14 = r_13;
            int lower_15 = (int)((lane_0 & 1) == 0);
            int desc_16 = (int)((lane_0 & 2) == 0);
            int wmax_17 = desc_16 * lower_15 + (1 - desc_16) * (1 - lower_15);
            int tk_18 = ((wmax_17 != 0) ? po_14 : 1 - po_14);
            kb[1] = ((tk_18 != 0) ? pb_11 : ck_9);
            ki[1] = ((tk_18 != 0) ? pc_12 : ci_10);
            unsigned int ck_19 = kb[0];
            int ci_20 = ki[0];
            unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, ck_19, 2);
            unsigned int pb_21 = _shfl_xor_24;
            int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, ci_20, 2);
            int pc_22 = _shfl_xor_25;
            int r_23 = (int)(pb_21 > ck_19 || pb_21 == ck_19 && pc_22 < ci_20);
            int po_24 = r_23;
            int lower_25 = (int)((lane_0 & 2) == 0);
            int desc_26 = (int)((lane_0 & 4) == 0);
            int wmax_27 = desc_26 * lower_25 + (1 - desc_26) * (1 - lower_25);
            int tk_28 = ((wmax_27 != 0) ? po_24 : 1 - po_24);
            kb[0] = ((tk_28 != 0) ? pb_21 : ck_19);
            ki[0] = ((tk_28 != 0) ? pc_22 : ci_20);
            unsigned int ck_29 = kb[1];
            int ci_30 = ki[1];
            unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, ck_29, 2);
            unsigned int pb_31 = _shfl_xor_26;
            int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, ci_30, 2);
            int pc_32 = _shfl_xor_27;
            int r_33 = (int)(pb_31 > ck_29 || pb_31 == ck_29 && pc_32 < ci_30);
            int po_34 = r_33;
            int lower_35 = (int)((lane_0 & 2) == 0);
            int desc_36 = (int)((lane_0 & 4) == 0);
            int wmax_37 = desc_36 * lower_35 + (1 - desc_36) * (1 - lower_35);
            int tk_38 = ((wmax_37 != 0) ? po_34 : 1 - po_34);
            kb[1] = ((tk_38 != 0) ? pb_31 : ck_29);
            ki[1] = ((tk_38 != 0) ? pc_32 : ci_30);
            unsigned int ck_39 = kb[0];
            int ci_40 = ki[0];
            unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, ck_39, 1);
            unsigned int pb_41 = _shfl_xor_28;
            int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, ci_40, 1);
            int pc_42 = _shfl_xor_29;
            int r_43 = (int)(pb_41 > ck_39 || pb_41 == ck_39 && pc_42 < ci_40);
            int po_44 = r_43;
            int lower_45 = (int)((lane_0 & 1) == 0);
            int desc_46 = (int)((lane_0 & 4) == 0);
            int wmax_47 = desc_46 * lower_45 + (1 - desc_46) * (1 - lower_45);
            int tk_48 = ((wmax_47 != 0) ? po_44 : 1 - po_44);
            kb[0] = ((tk_48 != 0) ? pb_41 : ck_39);
            ki[0] = ((tk_48 != 0) ? pc_42 : ci_40);
            unsigned int ck_49 = kb[1];
            int ci_50 = ki[1];
            unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, ck_49, 1);
            unsigned int pb_51 = _shfl_xor_30;
            int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, ci_50, 1);
            int pc_52 = _shfl_xor_31;
            int r_53 = (int)(pb_51 > ck_49 || pb_51 == ck_49 && pc_52 < ci_50);
            int po_54 = r_53;
            int lower_55 = (int)((lane_0 & 1) == 0);
            int desc_56 = (int)((lane_0 & 4) == 0);
            int wmax_57 = desc_56 * lower_55 + (1 - desc_56) * (1 - lower_55);
            int tk_58 = ((wmax_57 != 0) ? po_54 : 1 - po_54);
            kb[1] = ((tk_58 != 0) ? pb_51 : ck_49);
            ki[1] = ((tk_58 != 0) ? pc_52 : ci_50);
            unsigned int ck_59 = kb[0];
            int ci_60 = ki[0];
            unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, ck_59, 4);
            unsigned int pb_61 = _shfl_xor_32;
            int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, ci_60, 4);
            int pc_62 = _shfl_xor_33;
            int r_63 = (int)(pb_61 > ck_59 || pb_61 == ck_59 && pc_62 < ci_60);
            int po_64 = r_63;
            int lower_65 = (int)((lane_0 & 4) == 0);
            int desc_66 = (int)((lane_0 & 8) == 0);
            int wmax_67 = desc_66 * lower_65 + (1 - desc_66) * (1 - lower_65);
            int tk_68 = ((wmax_67 != 0) ? po_64 : 1 - po_64);
            kb[0] = ((tk_68 != 0) ? pb_61 : ck_59);
            ki[0] = ((tk_68 != 0) ? pc_62 : ci_60);
            unsigned int ck_69 = kb[1];
            int ci_70 = ki[1];
            unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, ck_69, 4);
            unsigned int pb_71 = _shfl_xor_34;
            int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, ci_70, 4);
            int pc_72 = _shfl_xor_35;
            int r_73 = (int)(pb_71 > ck_69 || pb_71 == ck_69 && pc_72 < ci_70);
            int po_74 = r_73;
            int lower_75 = (int)((lane_0 & 4) == 0);
            int desc_76 = (int)((lane_0 & 8) == 0);
            int wmax_77 = desc_76 * lower_75 + (1 - desc_76) * (1 - lower_75);
            int tk_78 = ((wmax_77 != 0) ? po_74 : 1 - po_74);
            kb[1] = ((tk_78 != 0) ? pb_71 : ck_69);
            ki[1] = ((tk_78 != 0) ? pc_72 : ci_70);
            unsigned int ck_79 = kb[0];
            int ci_80 = ki[0];
            unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, ck_79, 2);
            unsigned int pb_81 = _shfl_xor_36;
            int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, ci_80, 2);
            int pc_82 = _shfl_xor_37;
            int r_83 = (int)(pb_81 > ck_79 || pb_81 == ck_79 && pc_82 < ci_80);
            int po_84 = r_83;
            int lower_85 = (int)((lane_0 & 2) == 0);
            int desc_86 = (int)((lane_0 & 8) == 0);
            int wmax_87 = desc_86 * lower_85 + (1 - desc_86) * (1 - lower_85);
            int tk_88 = ((wmax_87 != 0) ? po_84 : 1 - po_84);
            kb[0] = ((tk_88 != 0) ? pb_81 : ck_79);
            ki[0] = ((tk_88 != 0) ? pc_82 : ci_80);
            unsigned int ck_89 = kb[1];
            int ci_90 = ki[1];
            unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, ck_89, 2);
            unsigned int pb_91 = _shfl_xor_38;
            int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, ci_90, 2);
            int pc_92 = _shfl_xor_39;
            int r_93 = (int)(pb_91 > ck_89 || pb_91 == ck_89 && pc_92 < ci_90);
            int po_94 = r_93;
            int lower_95 = (int)((lane_0 & 2) == 0);
            int desc_96 = (int)((lane_0 & 8) == 0);
            int wmax_97 = desc_96 * lower_95 + (1 - desc_96) * (1 - lower_95);
            int tk_98 = ((wmax_97 != 0) ? po_94 : 1 - po_94);
            kb[1] = ((tk_98 != 0) ? pb_91 : ck_89);
            ki[1] = ((tk_98 != 0) ? pc_92 : ci_90);
            unsigned int ck_99 = kb[0];
            int ci_100 = ki[0];
            unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, ck_99, 1);
            unsigned int pb_101 = _shfl_xor_40;
            int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, ci_100, 1);
            int pc_102 = _shfl_xor_41;
            int r_103 = (int)(pb_101 > ck_99 || pb_101 == ck_99 && pc_102 < ci_100);
            int po_104 = r_103;
            int lower_105 = (int)((lane_0 & 1) == 0);
            int desc_106 = (int)((lane_0 & 8) == 0);
            int wmax_107 = desc_106 * lower_105 + (1 - desc_106) * (1 - lower_105);
            int tk_108 = ((wmax_107 != 0) ? po_104 : 1 - po_104);
            kb[0] = ((tk_108 != 0) ? pb_101 : ck_99);
            ki[0] = ((tk_108 != 0) ? pc_102 : ci_100);
            unsigned int ck_109 = kb[1];
            int ci_110 = ki[1];
            unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, ck_109, 1);
            unsigned int pb_111 = _shfl_xor_42;
            int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, ci_110, 1);
            int pc_112 = _shfl_xor_43;
            int r_113 = (int)(pb_111 > ck_109 || pb_111 == ck_109 && pc_112 < ci_110);
            int po_114 = r_113;
            int lower_115 = (int)((lane_0 & 1) == 0);
            int desc_116 = (int)((lane_0 & 8) == 0);
            int wmax_117 = desc_116 * lower_115 + (1 - desc_116) * (1 - lower_115);
            int tk_118 = ((wmax_117 != 0) ? po_114 : 1 - po_114);
            kb[1] = ((tk_118 != 0) ? pb_111 : ck_109);
            ki[1] = ((tk_118 != 0) ? pc_112 : ci_110);
            unsigned int ck_119 = kb[0];
            int ci_120 = ki[0];
            unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, ck_119, 8);
            unsigned int pb_121 = _shfl_xor_44;
            int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, ci_120, 8);
            int pc_122 = _shfl_xor_45;
            int r_123 = (int)(pb_121 > ck_119 || pb_121 == ck_119 && pc_122 < ci_120);
            int po_124 = r_123;
            int lower_125 = (int)((lane_0 & 8) == 0);
            int desc_126 = (int)((lane_0 & 16) == 0);
            int wmax_127 = desc_126 * lower_125 + (1 - desc_126) * (1 - lower_125);
            int tk_128 = ((wmax_127 != 0) ? po_124 : 1 - po_124);
            kb[0] = ((tk_128 != 0) ? pb_121 : ck_119);
            ki[0] = ((tk_128 != 0) ? pc_122 : ci_120);
            unsigned int ck_129 = kb[1];
            int ci_130 = ki[1];
            unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, ck_129, 8);
            unsigned int pb_131 = _shfl_xor_46;
            int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, ci_130, 8);
            int pc_132 = _shfl_xor_47;
            int r_133 = (int)(pb_131 > ck_129 || pb_131 == ck_129 && pc_132 < ci_130);
            int po_134 = r_133;
            int lower_135 = (int)((lane_0 & 8) == 0);
            int desc_136 = (int)((lane_0 & 16) == 0);
            int wmax_137 = desc_136 * lower_135 + (1 - desc_136) * (1 - lower_135);
            int tk_138 = ((wmax_137 != 0) ? po_134 : 1 - po_134);
            kb[1] = ((tk_138 != 0) ? pb_131 : ck_129);
            ki[1] = ((tk_138 != 0) ? pc_132 : ci_130);
            unsigned int ck_139 = kb[0];
            int ci_140 = ki[0];
            unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, ck_139, 4);
            unsigned int pb_141 = _shfl_xor_48;
            int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, ci_140, 4);
            int pc_142 = _shfl_xor_49;
            int r_143 = (int)(pb_141 > ck_139 || pb_141 == ck_139 && pc_142 < ci_140);
            int po_144 = r_143;
            int lower_145 = (int)((lane_0 & 4) == 0);
            int desc_146 = (int)((lane_0 & 16) == 0);
            int wmax_147 = desc_146 * lower_145 + (1 - desc_146) * (1 - lower_145);
            int tk_148 = ((wmax_147 != 0) ? po_144 : 1 - po_144);
            kb[0] = ((tk_148 != 0) ? pb_141 : ck_139);
            ki[0] = ((tk_148 != 0) ? pc_142 : ci_140);
            unsigned int ck_149 = kb[1];
            int ci_150 = ki[1];
            unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, ck_149, 4);
            unsigned int pb_151 = _shfl_xor_50;
            int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, ci_150, 4);
            int pc_152 = _shfl_xor_51;
            int r_153 = (int)(pb_151 > ck_149 || pb_151 == ck_149 && pc_152 < ci_150);
            int po_154 = r_153;
            int lower_155 = (int)((lane_0 & 4) == 0);
            int desc_156 = (int)((lane_0 & 16) == 0);
            int wmax_157 = desc_156 * lower_155 + (1 - desc_156) * (1 - lower_155);
            int tk_158 = ((wmax_157 != 0) ? po_154 : 1 - po_154);
            kb[1] = ((tk_158 != 0) ? pb_151 : ck_149);
            ki[1] = ((tk_158 != 0) ? pc_152 : ci_150);
            unsigned int ck_159 = kb[0];
            int ci_160 = ki[0];
            unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, ck_159, 2);
            unsigned int pb_161 = _shfl_xor_52;
            int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, ci_160, 2);
            int pc_162 = _shfl_xor_53;
            int r_163 = (int)(pb_161 > ck_159 || pb_161 == ck_159 && pc_162 < ci_160);
            int po_164 = r_163;
            int lower_165 = (int)((lane_0 & 2) == 0);
            int desc_166 = (int)((lane_0 & 16) == 0);
            int wmax_167 = desc_166 * lower_165 + (1 - desc_166) * (1 - lower_165);
            int tk_168 = ((wmax_167 != 0) ? po_164 : 1 - po_164);
            kb[0] = ((tk_168 != 0) ? pb_161 : ck_159);
            ki[0] = ((tk_168 != 0) ? pc_162 : ci_160);
            unsigned int ck_169 = kb[1];
            int ci_170 = ki[1];
            unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, ck_169, 2);
            unsigned int pb_171 = _shfl_xor_54;
            int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, ci_170, 2);
            int pc_172 = _shfl_xor_55;
            int r_173 = (int)(pb_171 > ck_169 || pb_171 == ck_169 && pc_172 < ci_170);
            int po_174 = r_173;
            int lower_175 = (int)((lane_0 & 2) == 0);
            int desc_176 = (int)((lane_0 & 16) == 0);
            int wmax_177 = desc_176 * lower_175 + (1 - desc_176) * (1 - lower_175);
            int tk_178 = ((wmax_177 != 0) ? po_174 : 1 - po_174);
            kb[1] = ((tk_178 != 0) ? pb_171 : ck_169);
            ki[1] = ((tk_178 != 0) ? pc_172 : ci_170);
            unsigned int ck_179 = kb[0];
            int ci_180 = ki[0];
            unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, ck_179, 1);
            unsigned int pb_181 = _shfl_xor_56;
            int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, ci_180, 1);
            int pc_182 = _shfl_xor_57;
            int r_183 = (int)(pb_181 > ck_179 || pb_181 == ck_179 && pc_182 < ci_180);
            int po_184 = r_183;
            int lower_185 = (int)((lane_0 & 1) == 0);
            int desc_186 = (int)((lane_0 & 16) == 0);
            int wmax_187 = desc_186 * lower_185 + (1 - desc_186) * (1 - lower_185);
            int tk_188 = ((wmax_187 != 0) ? po_184 : 1 - po_184);
            kb[0] = ((tk_188 != 0) ? pb_181 : ck_179);
            ki[0] = ((tk_188 != 0) ? pc_182 : ci_180);
            unsigned int ck_189 = kb[1];
            int ci_190 = ki[1];
            unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, ck_189, 1);
            unsigned int pb_191 = _shfl_xor_58;
            int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, ci_190, 1);
            int pc_192 = _shfl_xor_59;
            int r_193 = (int)(pb_191 > ck_189 || pb_191 == ck_189 && pc_192 < ci_190);
            int po_194 = r_193;
            int lower_195 = (int)((lane_0 & 1) == 0);
            int desc_196 = (int)((lane_0 & 16) == 0);
            int wmax_197 = desc_196 * lower_195 + (1 - desc_196) * (1 - lower_195);
            int tk_198 = ((wmax_197 != 0) ? po_194 : 1 - po_194);
            kb[1] = ((tk_198 != 0) ? pb_191 : ck_189);
            ki[1] = ((tk_198 != 0) ? pc_192 : ci_190);
            unsigned int ck_199 = kb[0];
            int ci_200 = ki[0];
            unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, ck_199, 16);
            unsigned int pb_201 = _shfl_xor_60;
            int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, ci_200, 16);
            int pc_202 = _shfl_xor_61;
            int r_203 = (int)(pb_201 > ck_199 || pb_201 == ck_199 && pc_202 < ci_200);
            int po_204 = r_203;
            int lower_205 = (int)((lane_0 & 16) == 0);
            int desc_206 = (int)((lane_0 & 0) == 0);
            int wmax_207 = desc_206 * lower_205 + (1 - desc_206) * (1 - lower_205);
            int tk_208 = ((wmax_207 != 0) ? po_204 : 1 - po_204);
            kb[0] = ((tk_208 != 0) ? pb_201 : ck_199);
            ki[0] = ((tk_208 != 0) ? pc_202 : ci_200);
            unsigned int ck_209 = kb[1];
            int ci_210 = ki[1];
            unsigned int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, ck_209, 16);
            unsigned int pb_211 = _shfl_xor_62;
            int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, ci_210, 16);
            int pc_212 = _shfl_xor_63;
            int r_213 = (int)(pb_211 > ck_209 || pb_211 == ck_209 && pc_212 < ci_210);
            int po_214 = r_213;
            int lower_215 = (int)((lane_0 & 16) == 0);
            int desc_216 = 0;
            int wmax_217 = desc_216 * lower_215 + (1 - desc_216) * (1 - lower_215);
            int tk_218 = ((wmax_217 != 0) ? po_214 : 1 - po_214);
            kb[1] = ((tk_218 != 0) ? pb_211 : ck_209);
            ki[1] = ((tk_218 != 0) ? pc_212 : ci_210);
            unsigned int ck_219 = kb[0];
            int ci_220 = ki[0];
            unsigned int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, ck_219, 8);
            unsigned int pb_221 = _shfl_xor_64;
            int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, ci_220, 8);
            int pc_222 = _shfl_xor_65;
            int r_223 = (int)(pb_221 > ck_219 || pb_221 == ck_219 && pc_222 < ci_220);
            int po_224 = r_223;
            int lower_225 = (int)((lane_0 & 8) == 0);
            int desc_226 = (int)((lane_0 & 0) == 0);
            int wmax_227 = desc_226 * lower_225 + (1 - desc_226) * (1 - lower_225);
            int tk_228 = ((wmax_227 != 0) ? po_224 : 1 - po_224);
            kb[0] = ((tk_228 != 0) ? pb_221 : ck_219);
            ki[0] = ((tk_228 != 0) ? pc_222 : ci_220);
            unsigned int ck_229 = kb[1];
            int ci_230 = ki[1];
            unsigned int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, ck_229, 8);
            unsigned int pb_231 = _shfl_xor_66;
            int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, ci_230, 8);
            int pc_232 = _shfl_xor_67;
            int r_233 = (int)(pb_231 > ck_229 || pb_231 == ck_229 && pc_232 < ci_230);
            int po_234 = r_233;
            int lower_235 = (int)((lane_0 & 8) == 0);
            int desc_236 = 0;
            int wmax_237 = desc_236 * lower_235 + (1 - desc_236) * (1 - lower_235);
            int tk_238 = ((wmax_237 != 0) ? po_234 : 1 - po_234);
            kb[1] = ((tk_238 != 0) ? pb_231 : ck_229);
            ki[1] = ((tk_238 != 0) ? pc_232 : ci_230);
            unsigned int ck_239 = kb[0];
            int ci_240 = ki[0];
            unsigned int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, ck_239, 4);
            unsigned int pb_241 = _shfl_xor_68;
            int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, ci_240, 4);
            int pc_242 = _shfl_xor_69;
            int r_243 = (int)(pb_241 > ck_239 || pb_241 == ck_239 && pc_242 < ci_240);
            int po_244 = r_243;
            int lower_245 = (int)((lane_0 & 4) == 0);
            int desc_246 = (int)((lane_0 & 0) == 0);
            int wmax_247 = desc_246 * lower_245 + (1 - desc_246) * (1 - lower_245);
            int tk_248 = ((wmax_247 != 0) ? po_244 : 1 - po_244);
            kb[0] = ((tk_248 != 0) ? pb_241 : ck_239);
            ki[0] = ((tk_248 != 0) ? pc_242 : ci_240);
            unsigned int ck_249 = kb[1];
            int ci_250 = ki[1];
            unsigned int _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, ck_249, 4);
            unsigned int pb_251 = _shfl_xor_70;
            int _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, ci_250, 4);
            int pc_252 = _shfl_xor_71;
            int r_253 = (int)(pb_251 > ck_249 || pb_251 == ck_249 && pc_252 < ci_250);
            int po_254 = r_253;
            int lower_255 = (int)((lane_0 & 4) == 0);
            int desc_256 = 0;
            int wmax_257 = desc_256 * lower_255 + (1 - desc_256) * (1 - lower_255);
            int tk_258 = ((wmax_257 != 0) ? po_254 : 1 - po_254);
            kb[1] = ((tk_258 != 0) ? pb_251 : ck_249);
            ki[1] = ((tk_258 != 0) ? pc_252 : ci_250);
            unsigned int ck_259 = kb[0];
            int ci_260 = ki[0];
            unsigned int _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, ck_259, 2);
            unsigned int pb_261 = _shfl_xor_72;
            int _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, ci_260, 2);
            int pc_262 = _shfl_xor_73;
            int r_263 = (int)(pb_261 > ck_259 || pb_261 == ck_259 && pc_262 < ci_260);
            int po_264 = r_263;
            int lower_265 = (int)((lane_0 & 2) == 0);
            int desc_266 = (int)((lane_0 & 0) == 0);
            int wmax_267 = desc_266 * lower_265 + (1 - desc_266) * (1 - lower_265);
            int tk_268 = ((wmax_267 != 0) ? po_264 : 1 - po_264);
            kb[0] = ((tk_268 != 0) ? pb_261 : ck_259);
            ki[0] = ((tk_268 != 0) ? pc_262 : ci_260);
            unsigned int ck_269 = kb[1];
            int ci_270 = ki[1];
            unsigned int _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, ck_269, 2);
            unsigned int pb_271 = _shfl_xor_74;
            int _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, ci_270, 2);
            int pc_272 = _shfl_xor_75;
            int r_273 = (int)(pb_271 > ck_269 || pb_271 == ck_269 && pc_272 < ci_270);
            int po_274 = r_273;
            int lower_275 = (int)((lane_0 & 2) == 0);
            int desc_276 = 0;
            int wmax_277 = desc_276 * lower_275 + (1 - desc_276) * (1 - lower_275);
            int tk_278 = ((wmax_277 != 0) ? po_274 : 1 - po_274);
            kb[1] = ((tk_278 != 0) ? pb_271 : ck_269);
            ki[1] = ((tk_278 != 0) ? pc_272 : ci_270);
            unsigned int ck_279 = kb[0];
            int ci_280 = ki[0];
            unsigned int _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, ck_279, 1);
            unsigned int pb_281 = _shfl_xor_76;
            int _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, ci_280, 1);
            int pc_282 = _shfl_xor_77;
            int r_283 = (int)(pb_281 > ck_279 || pb_281 == ck_279 && pc_282 < ci_280);
            int po_284 = r_283;
            int lower_285 = (int)((lane_0 & 1) == 0);
            int desc_286 = (int)((lane_0 & 0) == 0);
            int wmax_287 = desc_286 * lower_285 + (1 - desc_286) * (1 - lower_285);
            int tk_288 = ((wmax_287 != 0) ? po_284 : 1 - po_284);
            kb[0] = ((tk_288 != 0) ? pb_281 : ck_279);
            ki[0] = ((tk_288 != 0) ? pc_282 : ci_280);
            unsigned int ck_289 = kb[1];
            int ci_290 = ki[1];
            unsigned int _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, ck_289, 1);
            unsigned int pb_291 = _shfl_xor_78;
            int _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, ci_290, 1);
            int pc_292 = _shfl_xor_79;
            int r_293 = (int)(pb_291 > ck_289 || pb_291 == ck_289 && pc_292 < ci_290);
            int po_294 = r_293;
            int lower_295 = (int)((lane_0 & 1) == 0);
            int desc_296 = 0;
            int wmax_297 = desc_296 * lower_295 + (1 - desc_296) * (1 - lower_295);
            int tk_298 = ((wmax_297 != 0) ? po_294 : 1 - po_294);
            kb[1] = ((tk_298 != 0) ? pb_291 : ck_289);
            ki[1] = ((tk_298 != 0) ? pc_292 : ci_290);
            unsigned int xk = kb[0];
            int xi = ki[0];
            unsigned int yk = kb[1];
            int yi = ki[1];
            int take = (int)(yk > xk || yk == xk && yi < xi);
            kb[0] = ((take != 0) ? yk : xk);
            ki[0] = ((take != 0) ? yi : xi);
            kb[1] = ((take != 0) ? xk : yk);
            ki[1] = ((take != 0) ? xi : yi);
            unsigned int ck_299 = kb[0];
            int ci_300 = ki[0];
            unsigned int _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, ck_299, 16);
            unsigned int pb_301 = _shfl_xor_80;
            int _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, ci_300, 16);
            int pc_302 = _shfl_xor_81;
            int r_303 = (int)(pb_301 > ck_299 || pb_301 == ck_299 && pc_302 < ci_300);
            int po_304 = r_303;
            int lower_305 = (int)((lane_0 & 16) == 0);
            int desc_306 = (int)((lane_0 & 0) == 0);
            int wmax_307 = desc_306 * lower_305 + (1 - desc_306) * (1 - lower_305);
            int tk_308 = ((wmax_307 != 0) ? po_304 : 1 - po_304);
            kb[0] = ((tk_308 != 0) ? pb_301 : ck_299);
            ki[0] = ((tk_308 != 0) ? pc_302 : ci_300);
            unsigned int ck_309 = kb[1];
            int ci_310 = ki[1];
            unsigned int _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, ck_309, 16);
            unsigned int pb_311 = _shfl_xor_82;
            int _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, ci_310, 16);
            int pc_312 = _shfl_xor_83;
            int r_313 = (int)(pb_311 > ck_309 || pb_311 == ck_309 && pc_312 < ci_310);
            int po_314 = r_313;
            int lower_315 = (int)((lane_0 & 16) == 0);
            int desc_316 = (int)((lane_0 & 0) == 0);
            int wmax_317 = desc_316 * lower_315 + (1 - desc_316) * (1 - lower_315);
            int tk_318 = ((wmax_317 != 0) ? po_314 : 1 - po_314);
            kb[1] = ((tk_318 != 0) ? pb_311 : ck_309);
            ki[1] = ((tk_318 != 0) ? pc_312 : ci_310);
            unsigned int ck_319 = kb[0];
            int ci_320 = ki[0];
            unsigned int _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, ck_319, 8);
            unsigned int pb_321 = _shfl_xor_84;
            int _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, ci_320, 8);
            int pc_322 = _shfl_xor_85;
            int r_323 = (int)(pb_321 > ck_319 || pb_321 == ck_319 && pc_322 < ci_320);
            int po_324 = r_323;
            int lower_325 = (int)((lane_0 & 8) == 0);
            int desc_326 = (int)((lane_0 & 0) == 0);
            int wmax_327 = desc_326 * lower_325 + (1 - desc_326) * (1 - lower_325);
            int tk_328 = ((wmax_327 != 0) ? po_324 : 1 - po_324);
            kb[0] = ((tk_328 != 0) ? pb_321 : ck_319);
            ki[0] = ((tk_328 != 0) ? pc_322 : ci_320);
            unsigned int ck_329 = kb[1];
            int ci_330 = ki[1];
            unsigned int _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, ck_329, 8);
            unsigned int pb_331 = _shfl_xor_86;
            int _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, ci_330, 8);
            int pc_332 = _shfl_xor_87;
            int r_333 = (int)(pb_331 > ck_329 || pb_331 == ck_329 && pc_332 < ci_330);
            int po_334 = r_333;
            int lower_335 = (int)((lane_0 & 8) == 0);
            int desc_336 = (int)((lane_0 & 0) == 0);
            int wmax_337 = desc_336 * lower_335 + (1 - desc_336) * (1 - lower_335);
            int tk_338 = ((wmax_337 != 0) ? po_334 : 1 - po_334);
            kb[1] = ((tk_338 != 0) ? pb_331 : ck_329);
            ki[1] = ((tk_338 != 0) ? pc_332 : ci_330);
            unsigned int ck_339 = kb[0];
            int ci_340 = ki[0];
            unsigned int _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, ck_339, 4);
            unsigned int pb_341 = _shfl_xor_88;
            int _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, ci_340, 4);
            int pc_342 = _shfl_xor_89;
            int r_343 = (int)(pb_341 > ck_339 || pb_341 == ck_339 && pc_342 < ci_340);
            int po_344 = r_343;
            int lower_345 = (int)((lane_0 & 4) == 0);
            int desc_346 = (int)((lane_0 & 0) == 0);
            int wmax_347 = desc_346 * lower_345 + (1 - desc_346) * (1 - lower_345);
            int tk_348 = ((wmax_347 != 0) ? po_344 : 1 - po_344);
            kb[0] = ((tk_348 != 0) ? pb_341 : ck_339);
            ki[0] = ((tk_348 != 0) ? pc_342 : ci_340);
            unsigned int ck_349 = kb[1];
            int ci_350 = ki[1];
            unsigned int _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, ck_349, 4);
            unsigned int pb_351 = _shfl_xor_90;
            int _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, ci_350, 4);
            int pc_352 = _shfl_xor_91;
            int r_353 = (int)(pb_351 > ck_349 || pb_351 == ck_349 && pc_352 < ci_350);
            int po_354 = r_353;
            int lower_355 = (int)((lane_0 & 4) == 0);
            int desc_356 = (int)((lane_0 & 0) == 0);
            int wmax_357 = desc_356 * lower_355 + (1 - desc_356) * (1 - lower_355);
            int tk_358 = ((wmax_357 != 0) ? po_354 : 1 - po_354);
            kb[1] = ((tk_358 != 0) ? pb_351 : ck_349);
            ki[1] = ((tk_358 != 0) ? pc_352 : ci_350);
            unsigned int ck_359 = kb[0];
            int ci_360 = ki[0];
            unsigned int _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, ck_359, 2);
            unsigned int pb_361 = _shfl_xor_92;
            int _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, ci_360, 2);
            int pc_362 = _shfl_xor_93;
            int r_363 = (int)(pb_361 > ck_359 || pb_361 == ck_359 && pc_362 < ci_360);
            int po_364 = r_363;
            int lower_365 = (int)((lane_0 & 2) == 0);
            int desc_366 = (int)((lane_0 & 0) == 0);
            int wmax_367 = desc_366 * lower_365 + (1 - desc_366) * (1 - lower_365);
            int tk_368 = ((wmax_367 != 0) ? po_364 : 1 - po_364);
            kb[0] = ((tk_368 != 0) ? pb_361 : ck_359);
            ki[0] = ((tk_368 != 0) ? pc_362 : ci_360);
            unsigned int ck_369 = kb[1];
            int ci_370 = ki[1];
            unsigned int _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, ck_369, 2);
            unsigned int pb_371 = _shfl_xor_94;
            int _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, ci_370, 2);
            int pc_372 = _shfl_xor_95;
            int r_373 = (int)(pb_371 > ck_369 || pb_371 == ck_369 && pc_372 < ci_370);
            int po_374 = r_373;
            int lower_375 = (int)((lane_0 & 2) == 0);
            int desc_376 = (int)((lane_0 & 0) == 0);
            int wmax_377 = desc_376 * lower_375 + (1 - desc_376) * (1 - lower_375);
            int tk_378 = ((wmax_377 != 0) ? po_374 : 1 - po_374);
            kb[1] = ((tk_378 != 0) ? pb_371 : ck_369);
            ki[1] = ((tk_378 != 0) ? pc_372 : ci_370);
            unsigned int ck_379 = kb[0];
            int ci_380 = ki[0];
            unsigned int _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, ck_379, 1);
            unsigned int pb_381 = _shfl_xor_96;
            int _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, ci_380, 1);
            int pc_382 = _shfl_xor_97;
            int r_383 = (int)(pb_381 > ck_379 || pb_381 == ck_379 && pc_382 < ci_380);
            int po_384 = r_383;
            int lower_385 = (int)((lane_0 & 1) == 0);
            int desc_386 = (int)((lane_0 & 0) == 0);
            int wmax_387 = desc_386 * lower_385 + (1 - desc_386) * (1 - lower_385);
            int tk_388 = ((wmax_387 != 0) ? po_384 : 1 - po_384);
            kb[0] = ((tk_388 != 0) ? pb_381 : ck_379);
            ki[0] = ((tk_388 != 0) ? pc_382 : ci_380);
            unsigned int ck_389 = kb[1];
            int ci_390 = ki[1];
            unsigned int _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, ck_389, 1);
            unsigned int pb_391 = _shfl_xor_98;
            int _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, ci_390, 1);
            int pc_392 = _shfl_xor_99;
            int r_393 = (int)(pb_391 > ck_389 || pb_391 == ck_389 && pc_392 < ci_390);
            int po_394 = r_393;
            int lower_395 = (int)((lane_0 & 1) == 0);
            int desc_396 = (int)((lane_0 & 0) == 0);
            int wmax_397 = desc_396 * lower_395 + (1 - desc_396) * (1 - lower_395);
            int tk_398 = ((wmax_397 != 0) ? po_394 : 1 - po_394);
            kb[1] = ((tk_398 != 0) ? pb_391 : ck_389);
            ki[1] = ((tk_398 != 0) ? pc_392 : ci_390);
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x; work_idx_l < total_units_l; work_idx_l += gridDim.x) {
                int unit_l = work_idx_l;
                int request_l = unit_l / num_kv_heads;
                int kv_head_l = unit_l % num_kv_heads;
                int group_size_l = num_q_heads / num_kv_heads;
                int kv_len_l = task_kv_head[request_l];
                __syncwarp();
                mbarrier_wait(meta_empty_addr + (item_par_1) * 8, me_phase);
                int meta_base_l = item_par_1 * 64;
                int lane_1 = lane;
                unsigned int lane_one = 1;
                unsigned int below = (lane_one << (unsigned int)lane_1) - 1;
                unsigned int incl = below | lane_one << (unsigned int)lane_1;
                smem_mask[meta_base_l + lane_1] = 0;
                smem_mask[meta_base_l + 32 + lane_1] = 0;
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
                unsigned int k_i_2 = kb[1];
                int p_i_3 = ki[1];
                unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, k_i_2, 1, 32);
                unsigned int pk_up_4 = _shfl_up_2;
                int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, p_i_3, 1, 32);
                int pp_up_5 = _shfl_up_3;
                unsigned int pk_6 = pk_up_4;
                int pp_7 = pp_up_5;
                int first_8 = 0;
                unsigned int k_prev = kb[0];
                int p_prev = ki[0];
                unsigned int _shfl_64 = __shfl_sync(0xFFFFFFFF, k_prev, 31);
                unsigned int k31 = _shfl_64;
                int _shfl_65 = __shfl_sync(0xFFFFFFFF, p_prev, 31);
                int p31 = _shfl_65;
                pk_6 = ((lane_1 == 0) ? k31 : pk_up_4);
                pp_7 = ((lane_1 == 0) ? p31 : pp_up_5);
                int valid_9 = (int)(k_i_2 != 0);
                int diff_id_10 = (int)(k_i_2 != pk_6) | first_8;
                int diff_tok_11 = (int)((p_i_3 & 7) != (pp_7 & 7));
                int newseg_12 = valid_9 & diff_id_10;
                int contrib_13 = valid_9 & (diff_id_10 | diff_tok_11);
                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, newseg_12 != 0);
                unsigned int fl_14 = _vote_1;
                int _popc_2 = __popc(fl_14 & incl);
                int slot_15 = base_prev + _popc_2 - 1;
                if (newseg_12 != 0) {
                    int id_i_1 = -1 - (int)k_i_2;
                    smem_page[meta_base_l + slot_15] = (p_i_3 >> 3) * num_kv_heads + kv_head_l;
                    smem_bstart[meta_base_l + slot_15] = id_i_1 * 128;
                }
                if (contrib_13 != 0) {
                    uint32_t _shared_atomic_old_1;
                    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_1) : "r"(static_cast<uint32_t>((smem_mask_addr + 4 * (meta_base_l + slot_15)))), "r"(static_cast<uint32_t>(lane_one << (unsigned int)(p_i_3 & 7))) : "memory");
                }
                int _popc_3 = __popc(fl_14);
                base_prev = base_prev + _popc_3;
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
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_raw_addr + (unsigned int)(load_stage * 8192)), "l"((&K)), "r"(0), "r"(0), "r"(kv_head_1), "r"(physical_page),
                                   "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_scale_addr + (unsigned int)(load_stage * 1024)), "l"((&K_scale)), "r"(0), "r"(0), "r"(kv_head_1), "r"(physical_page),
                                   "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            load_stage += 1;
                            if (load_stage == 10) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                mbarrier_wait(q_empty_addr + (item_par_1) * 8, me_phase);
                if (elect_sync()) {
                    int q_row0_l = request_l * seqlen_q * num_q_heads + kv_head_l * group_size_l;
                    mbarrier_arrive_expect_tx(q_full_addr + (item_par_1) * 8, 16384);
                    int q_row_t = q_row0_l;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(2 * item_par_1 * 4 * 2048), (&Q), 0, q_row_t, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 + 1) * 4 * 2048), (&Q), 0, q_row_t, 1, q_full_addr + (item_par_1) * 8);
                    int q_row_t_0 = q_row0_l + num_q_heads;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 * 4 + 1) * 2048), (&Q), 0, q_row_t_0, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(((2 * item_par_1 + 1) * 4 + 1) * 2048), (&Q), 0, q_row_t_0, 1, q_full_addr + (item_par_1) * 8);
                    int q_row_t_1 = q_row0_l + 2 * num_q_heads;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 * 4 + 2) * 2048), (&Q), 0, q_row_t_1, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(((2 * item_par_1 + 1) * 4 + 2) * 2048), (&Q), 0, q_row_t_1, 1, q_full_addr + (item_par_1) * 8);
                    int q_row_t_2 = q_row0_l + 3 * num_q_heads;
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)((2 * item_par_1 * 4 + 3) * 2048), (&Q), 0, q_row_t_2, 0, q_full_addr + (item_par_1) * 8);
                    tma_3d_gmem2smem(smem_qtok_addr + (unsigned int)(((2 * item_par_1 + 1) * 4 + 3) * 2048), (&Q), 0, q_row_t_2, 1, q_full_addr + (item_par_1) * 8);
                }
                if (elect_sync()) {
                    if (npairs_l > 1) {
                        #pragma unroll
                        for (int k_tile_1 = 0; k_tile_1 < 2; k_tile_1++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                            int physical_page_1 = smem_page[meta_base_l + 2 + k_tile_1] / num_kv_heads;
                            int kv_head_2 = smem_page[meta_base_l + 2 + k_tile_1] - physical_page_1 * num_kv_heads;
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_raw_addr + (unsigned int)(load_stage * 8192)), "l"((&K)), "r"(0), "r"(0), "r"(kv_head_2), "r"(physical_page_1),
                                   "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_scale_addr + (unsigned int)(load_stage * 1024)), "l"((&K_scale)), "r"(0), "r"(0), "r"(kv_head_2), "r"(physical_page_1),
                                   "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            load_stage += 1;
                            if (load_stage == 10) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                if (elect_sync()) {
                    smem_npairs[item_par_1] = npairs_l;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(meta_full_addr + (item_par_1) * 8);
                }
                unsigned int nxt_idx = work_idx_l + (unsigned int)gridDim.x;
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
                    int tok_e_13 = thalf_4 + 2;
                    int tok_c_14 = ((tok_e_13 < seqlen_q) ? tok_e_13 : seqlen_q - 1);
                    int id_raw_15 = task_kind[(request_n * seqlen_q + tok_c_14) * q2k_token_stride + kv_head_n * q2k_head_stride + jpos_3];
                    int id_tok_16 = ((tok_e_13 < seqlen_q) ? id_raw_15 : -1);
                    int id_c_17 = ((id_tok_16 >= 0) ? id_tok_16 : 0);
                    int page_raw_18 = kv_indices[request_n * msa_max_pages + id_c_17];
                    int page_e_19 = ((id_tok_16 >= 0) ? page_raw_18 : -1);
                    int id_e_20 = ((page_e_19 < 0) ? -1 : id_tok_16);
                    kb[1] = (unsigned int)(-1 - id_e_20);
                    ki[1] = ((id_e_20 < 0) ? 0 : page_e_19 << 3 | tok_e_13);
                    unsigned int ck_21 = kb[0];
                    int ci_22 = ki[0];
                    unsigned int _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, ck_21, 1);
                    unsigned int pb_23 = _shfl_xor_100;
                    int _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, ci_22, 1);
                    int pc_24 = _shfl_xor_101;
                    int r_25 = (int)(pb_23 > ck_21 || pb_23 == ck_21 && pc_24 < ci_22);
                    int po_26 = r_25;
                    int lower_27 = (int)((lane_2 & 1) == 0);
                    int desc_28 = (int)((lane_2 & 2) == 0);
                    int wmax_29 = desc_28 * lower_27 + (1 - desc_28) * (1 - lower_27);
                    int tk_30 = ((wmax_29 != 0) ? po_26 : 1 - po_26);
                    kb[0] = ((tk_30 != 0) ? pb_23 : ck_21);
                    ki[0] = ((tk_30 != 0) ? pc_24 : ci_22);
                    unsigned int ck_31 = kb[1];
                    int ci_32 = ki[1];
                    unsigned int _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, ck_31, 1);
                    unsigned int pb_33 = _shfl_xor_102;
                    int _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, ci_32, 1);
                    int pc_34 = _shfl_xor_103;
                    int r_35 = (int)(pb_33 > ck_31 || pb_33 == ck_31 && pc_34 < ci_32);
                    int po_36 = r_35;
                    int lower_37 = (int)((lane_2 & 1) == 0);
                    int desc_38 = (int)((lane_2 & 2) == 0);
                    int wmax_39 = desc_38 * lower_37 + (1 - desc_38) * (1 - lower_37);
                    int tk_40 = ((wmax_39 != 0) ? po_36 : 1 - po_36);
                    kb[1] = ((tk_40 != 0) ? pb_33 : ck_31);
                    ki[1] = ((tk_40 != 0) ? pc_34 : ci_32);
                    unsigned int ck_41 = kb[0];
                    int ci_42 = ki[0];
                    unsigned int _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, ck_41, 2);
                    unsigned int pb_43 = _shfl_xor_104;
                    int _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, ci_42, 2);
                    int pc_44 = _shfl_xor_105;
                    int r_45 = (int)(pb_43 > ck_41 || pb_43 == ck_41 && pc_44 < ci_42);
                    int po_46 = r_45;
                    int lower_47 = (int)((lane_2 & 2) == 0);
                    int desc_48 = (int)((lane_2 & 4) == 0);
                    int wmax_49 = desc_48 * lower_47 + (1 - desc_48) * (1 - lower_47);
                    int tk_50 = ((wmax_49 != 0) ? po_46 : 1 - po_46);
                    kb[0] = ((tk_50 != 0) ? pb_43 : ck_41);
                    ki[0] = ((tk_50 != 0) ? pc_44 : ci_42);
                    unsigned int ck_51 = kb[1];
                    int ci_52 = ki[1];
                    unsigned int _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, ck_51, 2);
                    unsigned int pb_53 = _shfl_xor_106;
                    int _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, ci_52, 2);
                    int pc_54 = _shfl_xor_107;
                    int r_55 = (int)(pb_53 > ck_51 || pb_53 == ck_51 && pc_54 < ci_52);
                    int po_56 = r_55;
                    int lower_57 = (int)((lane_2 & 2) == 0);
                    int desc_58 = (int)((lane_2 & 4) == 0);
                    int wmax_59 = desc_58 * lower_57 + (1 - desc_58) * (1 - lower_57);
                    int tk_60 = ((wmax_59 != 0) ? po_56 : 1 - po_56);
                    kb[1] = ((tk_60 != 0) ? pb_53 : ck_51);
                    ki[1] = ((tk_60 != 0) ? pc_54 : ci_52);
                    unsigned int ck_61 = kb[0];
                    int ci_62 = ki[0];
                    unsigned int _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, ck_61, 1);
                    unsigned int pb_63 = _shfl_xor_108;
                    int _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, ci_62, 1);
                    int pc_64 = _shfl_xor_109;
                    int r_65 = (int)(pb_63 > ck_61 || pb_63 == ck_61 && pc_64 < ci_62);
                    int po_66 = r_65;
                    int lower_67 = (int)((lane_2 & 1) == 0);
                    int desc_68 = (int)((lane_2 & 4) == 0);
                    int wmax_69 = desc_68 * lower_67 + (1 - desc_68) * (1 - lower_67);
                    int tk_70 = ((wmax_69 != 0) ? po_66 : 1 - po_66);
                    kb[0] = ((tk_70 != 0) ? pb_63 : ck_61);
                    ki[0] = ((tk_70 != 0) ? pc_64 : ci_62);
                    unsigned int ck_71 = kb[1];
                    int ci_72 = ki[1];
                    unsigned int _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, ck_71, 1);
                    unsigned int pb_73 = _shfl_xor_110;
                    int _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, ci_72, 1);
                    int pc_74 = _shfl_xor_111;
                    int r_75 = (int)(pb_73 > ck_71 || pb_73 == ck_71 && pc_74 < ci_72);
                    int po_76 = r_75;
                    int lower_77 = (int)((lane_2 & 1) == 0);
                    int desc_78 = (int)((lane_2 & 4) == 0);
                    int wmax_79 = desc_78 * lower_77 + (1 - desc_78) * (1 - lower_77);
                    int tk_80 = ((wmax_79 != 0) ? po_76 : 1 - po_76);
                    kb[1] = ((tk_80 != 0) ? pb_73 : ck_71);
                    ki[1] = ((tk_80 != 0) ? pc_74 : ci_72);
                    unsigned int ck_81 = kb[0];
                    int ci_82 = ki[0];
                    unsigned int _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, ck_81, 4);
                    unsigned int pb_83 = _shfl_xor_112;
                    int _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, ci_82, 4);
                    int pc_84 = _shfl_xor_113;
                    int r_85 = (int)(pb_83 > ck_81 || pb_83 == ck_81 && pc_84 < ci_82);
                    int po_86 = r_85;
                    int lower_87 = (int)((lane_2 & 4) == 0);
                    int desc_88 = (int)((lane_2 & 8) == 0);
                    int wmax_89 = desc_88 * lower_87 + (1 - desc_88) * (1 - lower_87);
                    int tk_90 = ((wmax_89 != 0) ? po_86 : 1 - po_86);
                    kb[0] = ((tk_90 != 0) ? pb_83 : ck_81);
                    ki[0] = ((tk_90 != 0) ? pc_84 : ci_82);
                    unsigned int ck_91 = kb[1];
                    int ci_92 = ki[1];
                    unsigned int _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, ck_91, 4);
                    unsigned int pb_93 = _shfl_xor_114;
                    int _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, ci_92, 4);
                    int pc_94 = _shfl_xor_115;
                    int r_95 = (int)(pb_93 > ck_91 || pb_93 == ck_91 && pc_94 < ci_92);
                    int po_96 = r_95;
                    int lower_97 = (int)((lane_2 & 4) == 0);
                    int desc_98 = (int)((lane_2 & 8) == 0);
                    int wmax_99 = desc_98 * lower_97 + (1 - desc_98) * (1 - lower_97);
                    int tk_100 = ((wmax_99 != 0) ? po_96 : 1 - po_96);
                    kb[1] = ((tk_100 != 0) ? pb_93 : ck_91);
                    ki[1] = ((tk_100 != 0) ? pc_94 : ci_92);
                    unsigned int ck_101 = kb[0];
                    int ci_102 = ki[0];
                    unsigned int _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, ck_101, 2);
                    unsigned int pb_103 = _shfl_xor_116;
                    int _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, ci_102, 2);
                    int pc_104 = _shfl_xor_117;
                    int r_105 = (int)(pb_103 > ck_101 || pb_103 == ck_101 && pc_104 < ci_102);
                    int po_106 = r_105;
                    int lower_107 = (int)((lane_2 & 2) == 0);
                    int desc_108 = (int)((lane_2 & 8) == 0);
                    int wmax_109 = desc_108 * lower_107 + (1 - desc_108) * (1 - lower_107);
                    int tk_110 = ((wmax_109 != 0) ? po_106 : 1 - po_106);
                    kb[0] = ((tk_110 != 0) ? pb_103 : ck_101);
                    ki[0] = ((tk_110 != 0) ? pc_104 : ci_102);
                    unsigned int ck_111 = kb[1];
                    int ci_112 = ki[1];
                    unsigned int _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, ck_111, 2);
                    unsigned int pb_113 = _shfl_xor_118;
                    int _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, ci_112, 2);
                    int pc_114 = _shfl_xor_119;
                    int r_115 = (int)(pb_113 > ck_111 || pb_113 == ck_111 && pc_114 < ci_112);
                    int po_116 = r_115;
                    int lower_117 = (int)((lane_2 & 2) == 0);
                    int desc_118 = (int)((lane_2 & 8) == 0);
                    int wmax_119 = desc_118 * lower_117 + (1 - desc_118) * (1 - lower_117);
                    int tk_120 = ((wmax_119 != 0) ? po_116 : 1 - po_116);
                    kb[1] = ((tk_120 != 0) ? pb_113 : ck_111);
                    ki[1] = ((tk_120 != 0) ? pc_114 : ci_112);
                    unsigned int ck_121 = kb[0];
                    int ci_122 = ki[0];
                    unsigned int _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, ck_121, 1);
                    unsigned int pb_123 = _shfl_xor_120;
                    int _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, ci_122, 1);
                    int pc_124 = _shfl_xor_121;
                    int r_125 = (int)(pb_123 > ck_121 || pb_123 == ck_121 && pc_124 < ci_122);
                    int po_126 = r_125;
                    int lower_127 = (int)((lane_2 & 1) == 0);
                    int desc_128 = (int)((lane_2 & 8) == 0);
                    int wmax_129 = desc_128 * lower_127 + (1 - desc_128) * (1 - lower_127);
                    int tk_130 = ((wmax_129 != 0) ? po_126 : 1 - po_126);
                    kb[0] = ((tk_130 != 0) ? pb_123 : ck_121);
                    ki[0] = ((tk_130 != 0) ? pc_124 : ci_122);
                    unsigned int ck_131 = kb[1];
                    int ci_132 = ki[1];
                    unsigned int _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, ck_131, 1);
                    unsigned int pb_133 = _shfl_xor_122;
                    int _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, ci_132, 1);
                    int pc_134 = _shfl_xor_123;
                    int r_135 = (int)(pb_133 > ck_131 || pb_133 == ck_131 && pc_134 < ci_132);
                    int po_136 = r_135;
                    int lower_137 = (int)((lane_2 & 1) == 0);
                    int desc_138 = (int)((lane_2 & 8) == 0);
                    int wmax_139 = desc_138 * lower_137 + (1 - desc_138) * (1 - lower_137);
                    int tk_140 = ((wmax_139 != 0) ? po_136 : 1 - po_136);
                    kb[1] = ((tk_140 != 0) ? pb_133 : ck_131);
                    ki[1] = ((tk_140 != 0) ? pc_134 : ci_132);
                    unsigned int ck_141 = kb[0];
                    int ci_142 = ki[0];
                    unsigned int _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, ck_141, 8);
                    unsigned int pb_143 = _shfl_xor_124;
                    int _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, ci_142, 8);
                    int pc_144 = _shfl_xor_125;
                    int r_145 = (int)(pb_143 > ck_141 || pb_143 == ck_141 && pc_144 < ci_142);
                    int po_146 = r_145;
                    int lower_147 = (int)((lane_2 & 8) == 0);
                    int desc_148 = (int)((lane_2 & 16) == 0);
                    int wmax_149 = desc_148 * lower_147 + (1 - desc_148) * (1 - lower_147);
                    int tk_150 = ((wmax_149 != 0) ? po_146 : 1 - po_146);
                    kb[0] = ((tk_150 != 0) ? pb_143 : ck_141);
                    ki[0] = ((tk_150 != 0) ? pc_144 : ci_142);
                    unsigned int ck_151 = kb[1];
                    int ci_152 = ki[1];
                    unsigned int _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, ck_151, 8);
                    unsigned int pb_153 = _shfl_xor_126;
                    int _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, ci_152, 8);
                    int pc_154 = _shfl_xor_127;
                    int r_155 = (int)(pb_153 > ck_151 || pb_153 == ck_151 && pc_154 < ci_152);
                    int po_156 = r_155;
                    int lower_157 = (int)((lane_2 & 8) == 0);
                    int desc_158 = (int)((lane_2 & 16) == 0);
                    int wmax_159 = desc_158 * lower_157 + (1 - desc_158) * (1 - lower_157);
                    int tk_160 = ((wmax_159 != 0) ? po_156 : 1 - po_156);
                    kb[1] = ((tk_160 != 0) ? pb_153 : ck_151);
                    ki[1] = ((tk_160 != 0) ? pc_154 : ci_152);
                    unsigned int ck_161 = kb[0];
                    int ci_162 = ki[0];
                    unsigned int _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, ck_161, 4);
                    unsigned int pb_163 = _shfl_xor_128;
                    int _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, ci_162, 4);
                    int pc_164 = _shfl_xor_129;
                    int r_165 = (int)(pb_163 > ck_161 || pb_163 == ck_161 && pc_164 < ci_162);
                    int po_166 = r_165;
                    int lower_167 = (int)((lane_2 & 4) == 0);
                    int desc_168 = (int)((lane_2 & 16) == 0);
                    int wmax_169 = desc_168 * lower_167 + (1 - desc_168) * (1 - lower_167);
                    int tk_170 = ((wmax_169 != 0) ? po_166 : 1 - po_166);
                    kb[0] = ((tk_170 != 0) ? pb_163 : ck_161);
                    ki[0] = ((tk_170 != 0) ? pc_164 : ci_162);
                    unsigned int ck_171 = kb[1];
                    int ci_172 = ki[1];
                    unsigned int _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, ck_171, 4);
                    unsigned int pb_173 = _shfl_xor_130;
                    int _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, ci_172, 4);
                    int pc_174 = _shfl_xor_131;
                    int r_175 = (int)(pb_173 > ck_171 || pb_173 == ck_171 && pc_174 < ci_172);
                    int po_176 = r_175;
                    int lower_177 = (int)((lane_2 & 4) == 0);
                    int desc_178 = (int)((lane_2 & 16) == 0);
                    int wmax_179 = desc_178 * lower_177 + (1 - desc_178) * (1 - lower_177);
                    int tk_180 = ((wmax_179 != 0) ? po_176 : 1 - po_176);
                    kb[1] = ((tk_180 != 0) ? pb_173 : ck_171);
                    ki[1] = ((tk_180 != 0) ? pc_174 : ci_172);
                    unsigned int ck_181 = kb[0];
                    int ci_182 = ki[0];
                    unsigned int _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, ck_181, 2);
                    unsigned int pb_183 = _shfl_xor_132;
                    int _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, ci_182, 2);
                    int pc_184 = _shfl_xor_133;
                    int r_185 = (int)(pb_183 > ck_181 || pb_183 == ck_181 && pc_184 < ci_182);
                    int po_186 = r_185;
                    int lower_187 = (int)((lane_2 & 2) == 0);
                    int desc_188 = (int)((lane_2 & 16) == 0);
                    int wmax_189 = desc_188 * lower_187 + (1 - desc_188) * (1 - lower_187);
                    int tk_190 = ((wmax_189 != 0) ? po_186 : 1 - po_186);
                    kb[0] = ((tk_190 != 0) ? pb_183 : ck_181);
                    ki[0] = ((tk_190 != 0) ? pc_184 : ci_182);
                    unsigned int ck_191 = kb[1];
                    int ci_192 = ki[1];
                    unsigned int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, ck_191, 2);
                    unsigned int pb_193 = _shfl_xor_134;
                    int _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, ci_192, 2);
                    int pc_194 = _shfl_xor_135;
                    int r_195 = (int)(pb_193 > ck_191 || pb_193 == ck_191 && pc_194 < ci_192);
                    int po_196 = r_195;
                    int lower_197 = (int)((lane_2 & 2) == 0);
                    int desc_198 = (int)((lane_2 & 16) == 0);
                    int wmax_199 = desc_198 * lower_197 + (1 - desc_198) * (1 - lower_197);
                    int tk_200 = ((wmax_199 != 0) ? po_196 : 1 - po_196);
                    kb[1] = ((tk_200 != 0) ? pb_193 : ck_191);
                    ki[1] = ((tk_200 != 0) ? pc_194 : ci_192);
                    unsigned int ck_201 = kb[0];
                    int ci_202 = ki[0];
                    unsigned int _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, ck_201, 1);
                    unsigned int pb_203 = _shfl_xor_136;
                    int _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, ci_202, 1);
                    int pc_204 = _shfl_xor_137;
                    int r_205 = (int)(pb_203 > ck_201 || pb_203 == ck_201 && pc_204 < ci_202);
                    int po_206 = r_205;
                    int lower_207 = (int)((lane_2 & 1) == 0);
                    int desc_208 = (int)((lane_2 & 16) == 0);
                    int wmax_209 = desc_208 * lower_207 + (1 - desc_208) * (1 - lower_207);
                    int tk_210 = ((wmax_209 != 0) ? po_206 : 1 - po_206);
                    kb[0] = ((tk_210 != 0) ? pb_203 : ck_201);
                    ki[0] = ((tk_210 != 0) ? pc_204 : ci_202);
                    unsigned int ck_211 = kb[1];
                    int ci_212 = ki[1];
                    unsigned int _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, ck_211, 1);
                    unsigned int pb_213 = _shfl_xor_138;
                    int _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, ci_212, 1);
                    int pc_214 = _shfl_xor_139;
                    int r_215 = (int)(pb_213 > ck_211 || pb_213 == ck_211 && pc_214 < ci_212);
                    int po_216 = r_215;
                    int lower_217 = (int)((lane_2 & 1) == 0);
                    int desc_218 = (int)((lane_2 & 16) == 0);
                    int wmax_219 = desc_218 * lower_217 + (1 - desc_218) * (1 - lower_217);
                    int tk_220 = ((wmax_219 != 0) ? po_216 : 1 - po_216);
                    kb[1] = ((tk_220 != 0) ? pb_213 : ck_211);
                    ki[1] = ((tk_220 != 0) ? pc_214 : ci_212);
                    unsigned int ck_221 = kb[0];
                    int ci_222 = ki[0];
                    unsigned int _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, ck_221, 16);
                    unsigned int pb_223 = _shfl_xor_140;
                    int _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, ci_222, 16);
                    int pc_224 = _shfl_xor_141;
                    int r_225 = (int)(pb_223 > ck_221 || pb_223 == ck_221 && pc_224 < ci_222);
                    int po_226 = r_225;
                    int lower_227 = (int)((lane_2 & 16) == 0);
                    int desc_228 = (int)((lane_2 & 0) == 0);
                    int wmax_229 = desc_228 * lower_227 + (1 - desc_228) * (1 - lower_227);
                    int tk_230 = ((wmax_229 != 0) ? po_226 : 1 - po_226);
                    kb[0] = ((tk_230 != 0) ? pb_223 : ck_221);
                    ki[0] = ((tk_230 != 0) ? pc_224 : ci_222);
                    unsigned int ck_231 = kb[1];
                    int ci_232 = ki[1];
                    unsigned int _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, ck_231, 16);
                    unsigned int pb_233 = _shfl_xor_142;
                    int _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, ci_232, 16);
                    int pc_234 = _shfl_xor_143;
                    int r_235 = (int)(pb_233 > ck_231 || pb_233 == ck_231 && pc_234 < ci_232);
                    int po_236 = r_235;
                    int lower_237 = (int)((lane_2 & 16) == 0);
                    int desc_238 = 0;
                    int wmax_239 = desc_238 * lower_237 + (1 - desc_238) * (1 - lower_237);
                    int tk_240 = ((wmax_239 != 0) ? po_236 : 1 - po_236);
                    kb[1] = ((tk_240 != 0) ? pb_233 : ck_231);
                    ki[1] = ((tk_240 != 0) ? pc_234 : ci_232);
                    unsigned int ck_241 = kb[0];
                    int ci_242 = ki[0];
                    unsigned int _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, ck_241, 8);
                    unsigned int pb_243 = _shfl_xor_144;
                    int _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, ci_242, 8);
                    int pc_244 = _shfl_xor_145;
                    int r_245 = (int)(pb_243 > ck_241 || pb_243 == ck_241 && pc_244 < ci_242);
                    int po_246 = r_245;
                    int lower_247 = (int)((lane_2 & 8) == 0);
                    int desc_248 = (int)((lane_2 & 0) == 0);
                    int wmax_249 = desc_248 * lower_247 + (1 - desc_248) * (1 - lower_247);
                    int tk_250 = ((wmax_249 != 0) ? po_246 : 1 - po_246);
                    kb[0] = ((tk_250 != 0) ? pb_243 : ck_241);
                    ki[0] = ((tk_250 != 0) ? pc_244 : ci_242);
                    unsigned int ck_251 = kb[1];
                    int ci_252 = ki[1];
                    unsigned int _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, ck_251, 8);
                    unsigned int pb_253 = _shfl_xor_146;
                    int _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, ci_252, 8);
                    int pc_254 = _shfl_xor_147;
                    int r_255 = (int)(pb_253 > ck_251 || pb_253 == ck_251 && pc_254 < ci_252);
                    int po_256 = r_255;
                    int lower_257 = (int)((lane_2 & 8) == 0);
                    int desc_258 = 0;
                    int wmax_259 = desc_258 * lower_257 + (1 - desc_258) * (1 - lower_257);
                    int tk_260 = ((wmax_259 != 0) ? po_256 : 1 - po_256);
                    kb[1] = ((tk_260 != 0) ? pb_253 : ck_251);
                    ki[1] = ((tk_260 != 0) ? pc_254 : ci_252);
                    unsigned int ck_261 = kb[0];
                    int ci_262 = ki[0];
                    unsigned int _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, ck_261, 4);
                    unsigned int pb_263 = _shfl_xor_148;
                    int _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, ci_262, 4);
                    int pc_264 = _shfl_xor_149;
                    int r_265 = (int)(pb_263 > ck_261 || pb_263 == ck_261 && pc_264 < ci_262);
                    int po_266 = r_265;
                    int lower_267 = (int)((lane_2 & 4) == 0);
                    int desc_268 = (int)((lane_2 & 0) == 0);
                    int wmax_269 = desc_268 * lower_267 + (1 - desc_268) * (1 - lower_267);
                    int tk_270 = ((wmax_269 != 0) ? po_266 : 1 - po_266);
                    kb[0] = ((tk_270 != 0) ? pb_263 : ck_261);
                    ki[0] = ((tk_270 != 0) ? pc_264 : ci_262);
                    unsigned int ck_271 = kb[1];
                    int ci_272 = ki[1];
                    unsigned int _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, ck_271, 4);
                    unsigned int pb_273 = _shfl_xor_150;
                    int _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, ci_272, 4);
                    int pc_274 = _shfl_xor_151;
                    int r_275 = (int)(pb_273 > ck_271 || pb_273 == ck_271 && pc_274 < ci_272);
                    int po_276 = r_275;
                    int lower_277 = (int)((lane_2 & 4) == 0);
                    int desc_278 = 0;
                    int wmax_279 = desc_278 * lower_277 + (1 - desc_278) * (1 - lower_277);
                    int tk_280 = ((wmax_279 != 0) ? po_276 : 1 - po_276);
                    kb[1] = ((tk_280 != 0) ? pb_273 : ck_271);
                    ki[1] = ((tk_280 != 0) ? pc_274 : ci_272);
                    unsigned int ck_281 = kb[0];
                    int ci_282 = ki[0];
                    unsigned int _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, ck_281, 2);
                    unsigned int pb_283 = _shfl_xor_152;
                    int _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, ci_282, 2);
                    int pc_284 = _shfl_xor_153;
                    int r_285 = (int)(pb_283 > ck_281 || pb_283 == ck_281 && pc_284 < ci_282);
                    int po_286 = r_285;
                    int lower_287 = (int)((lane_2 & 2) == 0);
                    int desc_288 = (int)((lane_2 & 0) == 0);
                    int wmax_289 = desc_288 * lower_287 + (1 - desc_288) * (1 - lower_287);
                    int tk_290 = ((wmax_289 != 0) ? po_286 : 1 - po_286);
                    kb[0] = ((tk_290 != 0) ? pb_283 : ck_281);
                    ki[0] = ((tk_290 != 0) ? pc_284 : ci_282);
                    unsigned int ck_291 = kb[1];
                    int ci_292 = ki[1];
                    unsigned int _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, ck_291, 2);
                    unsigned int pb_293 = _shfl_xor_154;
                    int _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, ci_292, 2);
                    int pc_294 = _shfl_xor_155;
                    int r_295 = (int)(pb_293 > ck_291 || pb_293 == ck_291 && pc_294 < ci_292);
                    int po_296 = r_295;
                    int lower_297 = (int)((lane_2 & 2) == 0);
                    int desc_298 = 0;
                    int wmax_299 = desc_298 * lower_297 + (1 - desc_298) * (1 - lower_297);
                    int tk_300 = ((wmax_299 != 0) ? po_296 : 1 - po_296);
                    kb[1] = ((tk_300 != 0) ? pb_293 : ck_291);
                    ki[1] = ((tk_300 != 0) ? pc_294 : ci_292);
                    unsigned int ck_301 = kb[0];
                    int ci_302 = ki[0];
                    unsigned int _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, ck_301, 1);
                    unsigned int pb_303 = _shfl_xor_156;
                    int _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, ci_302, 1);
                    int pc_304 = _shfl_xor_157;
                    int r_305 = (int)(pb_303 > ck_301 || pb_303 == ck_301 && pc_304 < ci_302);
                    int po_306 = r_305;
                    int lower_307 = (int)((lane_2 & 1) == 0);
                    int desc_308 = (int)((lane_2 & 0) == 0);
                    int wmax_309 = desc_308 * lower_307 + (1 - desc_308) * (1 - lower_307);
                    int tk_310 = ((wmax_309 != 0) ? po_306 : 1 - po_306);
                    kb[0] = ((tk_310 != 0) ? pb_303 : ck_301);
                    ki[0] = ((tk_310 != 0) ? pc_304 : ci_302);
                    unsigned int ck_311 = kb[1];
                    int ci_312 = ki[1];
                    unsigned int _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, ck_311, 1);
                    unsigned int pb_313 = _shfl_xor_158;
                    int _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, ci_312, 1);
                    int pc_314 = _shfl_xor_159;
                    int r_315 = (int)(pb_313 > ck_311 || pb_313 == ck_311 && pc_314 < ci_312);
                    int po_316 = r_315;
                    int lower_317 = (int)((lane_2 & 1) == 0);
                    int desc_318 = 0;
                    int wmax_319 = desc_318 * lower_317 + (1 - desc_318) * (1 - lower_317);
                    int tk_320 = ((wmax_319 != 0) ? po_316 : 1 - po_316);
                    kb[1] = ((tk_320 != 0) ? pb_313 : ck_311);
                    ki[1] = ((tk_320 != 0) ? pc_314 : ci_312);
                    unsigned int xk_321 = kb[0];
                    int xi_322 = ki[0];
                    unsigned int yk_323 = kb[1];
                    int yi_324 = ki[1];
                    int take_325 = (int)(yk_323 > xk_321 || yk_323 == xk_321 && yi_324 < xi_322);
                    kb[0] = ((take_325 != 0) ? yk_323 : xk_321);
                    ki[0] = ((take_325 != 0) ? yi_324 : xi_322);
                    kb[1] = ((take_325 != 0) ? xk_321 : yk_323);
                    ki[1] = ((take_325 != 0) ? xi_322 : yi_324);
                    unsigned int ck_326 = kb[0];
                    int ci_327 = ki[0];
                    unsigned int _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, ck_326, 16);
                    unsigned int pb_328 = _shfl_xor_160;
                    int _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, ci_327, 16);
                    int pc_329 = _shfl_xor_161;
                    int r_330 = (int)(pb_328 > ck_326 || pb_328 == ck_326 && pc_329 < ci_327);
                    int po_331 = r_330;
                    int lower_332 = (int)((lane_2 & 16) == 0);
                    int desc_333 = (int)((lane_2 & 0) == 0);
                    int wmax_334 = desc_333 * lower_332 + (1 - desc_333) * (1 - lower_332);
                    int tk_335 = ((wmax_334 != 0) ? po_331 : 1 - po_331);
                    kb[0] = ((tk_335 != 0) ? pb_328 : ck_326);
                    ki[0] = ((tk_335 != 0) ? pc_329 : ci_327);
                    unsigned int ck_336 = kb[1];
                    int ci_337 = ki[1];
                    unsigned int _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, ck_336, 16);
                    unsigned int pb_338 = _shfl_xor_162;
                    int _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, ci_337, 16);
                    int pc_339 = _shfl_xor_163;
                    int r_340 = (int)(pb_338 > ck_336 || pb_338 == ck_336 && pc_339 < ci_337);
                    int po_341 = r_340;
                    int lower_342 = (int)((lane_2 & 16) == 0);
                    int desc_343 = (int)((lane_2 & 0) == 0);
                    int wmax_344 = desc_343 * lower_342 + (1 - desc_343) * (1 - lower_342);
                    int tk_345 = ((wmax_344 != 0) ? po_341 : 1 - po_341);
                    kb[1] = ((tk_345 != 0) ? pb_338 : ck_336);
                    ki[1] = ((tk_345 != 0) ? pc_339 : ci_337);
                    unsigned int ck_346 = kb[0];
                    int ci_347 = ki[0];
                    unsigned int _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, ck_346, 8);
                    unsigned int pb_348 = _shfl_xor_164;
                    int _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, ci_347, 8);
                    int pc_349 = _shfl_xor_165;
                    int r_350 = (int)(pb_348 > ck_346 || pb_348 == ck_346 && pc_349 < ci_347);
                    int po_351 = r_350;
                    int lower_352 = (int)((lane_2 & 8) == 0);
                    int desc_353 = (int)((lane_2 & 0) == 0);
                    int wmax_354 = desc_353 * lower_352 + (1 - desc_353) * (1 - lower_352);
                    int tk_355 = ((wmax_354 != 0) ? po_351 : 1 - po_351);
                    kb[0] = ((tk_355 != 0) ? pb_348 : ck_346);
                    ki[0] = ((tk_355 != 0) ? pc_349 : ci_347);
                    unsigned int ck_356 = kb[1];
                    int ci_357 = ki[1];
                    unsigned int _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, ck_356, 8);
                    unsigned int pb_358 = _shfl_xor_166;
                    int _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, ci_357, 8);
                    int pc_359 = _shfl_xor_167;
                    int r_360 = (int)(pb_358 > ck_356 || pb_358 == ck_356 && pc_359 < ci_357);
                    int po_361 = r_360;
                    int lower_362 = (int)((lane_2 & 8) == 0);
                    int desc_363 = (int)((lane_2 & 0) == 0);
                    int wmax_364 = desc_363 * lower_362 + (1 - desc_363) * (1 - lower_362);
                    int tk_365 = ((wmax_364 != 0) ? po_361 : 1 - po_361);
                    kb[1] = ((tk_365 != 0) ? pb_358 : ck_356);
                    ki[1] = ((tk_365 != 0) ? pc_359 : ci_357);
                    unsigned int ck_366 = kb[0];
                    int ci_367 = ki[0];
                    unsigned int _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, ck_366, 4);
                    unsigned int pb_368 = _shfl_xor_168;
                    int _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, ci_367, 4);
                    int pc_369 = _shfl_xor_169;
                    int r_370 = (int)(pb_368 > ck_366 || pb_368 == ck_366 && pc_369 < ci_367);
                    int po_371 = r_370;
                    int lower_372 = (int)((lane_2 & 4) == 0);
                    int desc_373 = (int)((lane_2 & 0) == 0);
                    int wmax_374 = desc_373 * lower_372 + (1 - desc_373) * (1 - lower_372);
                    int tk_375 = ((wmax_374 != 0) ? po_371 : 1 - po_371);
                    kb[0] = ((tk_375 != 0) ? pb_368 : ck_366);
                    ki[0] = ((tk_375 != 0) ? pc_369 : ci_367);
                    unsigned int ck_376 = kb[1];
                    int ci_377 = ki[1];
                    unsigned int _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, ck_376, 4);
                    unsigned int pb_378 = _shfl_xor_170;
                    int _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, ci_377, 4);
                    int pc_379 = _shfl_xor_171;
                    int r_380 = (int)(pb_378 > ck_376 || pb_378 == ck_376 && pc_379 < ci_377);
                    int po_381 = r_380;
                    int lower_382 = (int)((lane_2 & 4) == 0);
                    int desc_383 = (int)((lane_2 & 0) == 0);
                    int wmax_384 = desc_383 * lower_382 + (1 - desc_383) * (1 - lower_382);
                    int tk_385 = ((wmax_384 != 0) ? po_381 : 1 - po_381);
                    kb[1] = ((tk_385 != 0) ? pb_378 : ck_376);
                    ki[1] = ((tk_385 != 0) ? pc_379 : ci_377);
                    unsigned int ck_386 = kb[0];
                    int ci_387 = ki[0];
                    unsigned int _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, ck_386, 2);
                    unsigned int pb_388 = _shfl_xor_172;
                    int _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, ci_387, 2);
                    int pc_389 = _shfl_xor_173;
                    int r_390 = (int)(pb_388 > ck_386 || pb_388 == ck_386 && pc_389 < ci_387);
                    int po_391 = r_390;
                    int lower_392 = (int)((lane_2 & 2) == 0);
                    int desc_393 = (int)((lane_2 & 0) == 0);
                    int wmax_394 = desc_393 * lower_392 + (1 - desc_393) * (1 - lower_392);
                    int tk_395 = ((wmax_394 != 0) ? po_391 : 1 - po_391);
                    kb[0] = ((tk_395 != 0) ? pb_388 : ck_386);
                    ki[0] = ((tk_395 != 0) ? pc_389 : ci_387);
                    unsigned int ck_396 = kb[1];
                    int ci_397 = ki[1];
                    unsigned int _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, ck_396, 2);
                    unsigned int pb_398 = _shfl_xor_174;
                    int _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, ci_397, 2);
                    int pc_399 = _shfl_xor_175;
                    int r_400 = (int)(pb_398 > ck_396 || pb_398 == ck_396 && pc_399 < ci_397);
                    int po_401 = r_400;
                    int lower_402 = (int)((lane_2 & 2) == 0);
                    int desc_403 = (int)((lane_2 & 0) == 0);
                    int wmax_404 = desc_403 * lower_402 + (1 - desc_403) * (1 - lower_402);
                    int tk_405 = ((wmax_404 != 0) ? po_401 : 1 - po_401);
                    kb[1] = ((tk_405 != 0) ? pb_398 : ck_396);
                    ki[1] = ((tk_405 != 0) ? pc_399 : ci_397);
                    unsigned int ck_406 = kb[0];
                    int ci_407 = ki[0];
                    unsigned int _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, ck_406, 1);
                    unsigned int pb_408 = _shfl_xor_176;
                    int _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, ci_407, 1);
                    int pc_409 = _shfl_xor_177;
                    int r_410 = (int)(pb_408 > ck_406 || pb_408 == ck_406 && pc_409 < ci_407);
                    int po_411 = r_410;
                    int lower_412 = (int)((lane_2 & 1) == 0);
                    int desc_413 = (int)((lane_2 & 0) == 0);
                    int wmax_414 = desc_413 * lower_412 + (1 - desc_413) * (1 - lower_412);
                    int tk_415 = ((wmax_414 != 0) ? po_411 : 1 - po_411);
                    kb[0] = ((tk_415 != 0) ? pb_408 : ck_406);
                    ki[0] = ((tk_415 != 0) ? pc_409 : ci_407);
                    unsigned int ck_416 = kb[1];
                    int ci_417 = ki[1];
                    unsigned int _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, ck_416, 1);
                    unsigned int pb_418 = _shfl_xor_178;
                    int _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, ci_417, 1);
                    int pc_419 = _shfl_xor_179;
                    int r_420 = (int)(pb_418 > ck_416 || pb_418 == ck_416 && pc_419 < ci_417);
                    int po_421 = r_420;
                    int lower_422 = (int)((lane_2 & 1) == 0);
                    int desc_423 = (int)((lane_2 & 0) == 0);
                    int wmax_424 = desc_423 * lower_422 + (1 - desc_423) * (1 - lower_422);
                    int tk_425 = ((wmax_424 != 0) ? po_421 : 1 - po_421);
                    kb[1] = ((tk_425 != 0) ? pb_418 : ck_416);
                    ki[1] = ((tk_425 != 0) ? pc_419 : ci_417);
                }
                __syncwarp();
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pair_3 = 0; pair_3 < 32; pair_3++) {
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
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_raw_addr + (unsigned int)(load_stage * 8192)), "l"((&K)), "r"(0), "r"(0), "r"(kv_head_3), "r"(physical_page_2),
                                       "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_scale_addr + (unsigned int)(load_stage * 1024)), "l"((&K_scale)), "r"(0), "r"(0), "r"(kv_head_3), "r"(physical_page_2),
                                       "r"(kv_raw_full_addr + (kt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                load_stage += 1;
                                if (load_stage == 10) { load_stage = 0; load_phase ^= 1; }
                                kt_l = kt_l + 1;
                            }
                        }
                        #pragma unroll
                        for (int v_tile = 0; v_tile < 2; v_tile++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (16 + vt_l % 16) * 8, 9216);
                            int physical_page_3 = smem_page[meta_base_l + pair_3 * 2 + v_tile] / num_kv_heads;
                            int kv_head_4 = smem_page[meta_base_l + pair_3 * 2 + v_tile] - physical_page_3 * num_kv_heads;
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_raw_addr + (unsigned int)(load_stage * 8192)), "l"((&V)), "r"(0), "r"(0), "r"(kv_head_4), "r"(physical_page_3),
                                   "r"(kv_raw_full_addr + (16 + vt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_scale_addr + (unsigned int)(load_stage * 1024)), "l"((&V_scale)), "r"(0), "r"(0), "r"(kv_head_4), "r"(physical_page_3),
                                   "r"(kv_raw_full_addr + (16 + vt_l % 16) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            load_stage += 1;
                            if (load_stage == 10) { load_stage = 0; load_phase ^= 1; }
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
            for (unsigned int work_idx_s = blockIdx.x; work_idx_s < total_units_s; work_idx_s += gridDim.x) {
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
                for (int sp_1 = 0; sp_1 < 32; sp_1++) {
                    if (npairs_s <= sp_1) {
                        break;
                    }
                    mbarrier_wait(s_empty_addr + (s_stage_m) * 8, s_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int slot_2 = ktile_n % 2;
                    int use_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_2) * 8, use_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_0 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"((tmem_tmem_s + (s_stage_m * 128))), "r"(_mma_b_lo_0), "r"(tmem_tmem_k16 + 2 * slot_2 * 32), "r"(0));
                    int _mma_b_lo_1 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"((tmem_tmem_s + (s_stage_m * 128))), "r"(_mma_b_lo_1), "r"(tmem_tmem_k16 + (2 * slot_2 + 1) * 32), "r"(1));
                    elect_commit(k16_empty_addr + (slot_2) * 8);
                    ktile_n = ktile_n + 1;
                    int slot_0 = ktile_n % 2;
                    int use_1_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_0) * 8, use_1_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"((tmem_tmem_s + (s_stage_m * 128 + 64))), "r"(_mma_b_lo_2), "r"(tmem_tmem_k16 + 2 * slot_0 * 32), "r"(0));
                    int _mma_b_lo_3 = make_warp_uniform((((smem_qu_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"((tmem_tmem_s + (s_stage_m * 128 + 64))), "r"(_mma_b_lo_3), "r"(tmem_tmem_k16 + (2 * slot_0 + 1) * 32), "r"(1));
                    elect_commit(k16_empty_addr + (slot_0) * 8);
                    ktile_n = ktile_n + 1;
                    if (sp_1 == npairs_s - 1) {
                        elect_commit(q_empty_addr + (q_buf_s) * 8);
                    }
                    elect_commit(s_full_addr + (s_stage_m) * 8);
                    s_phase_m ^= 1;
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
            for (unsigned int work_idx_v = blockIdx.x; work_idx_v < total_units_v; work_idx_v += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_v) * 8, item_phase_v);
                int npairs_v = smem_npairs[item_stage_v];
                item_stage_v += 1;
                if (item_stage_v == 2) { item_stage_v = 0; item_phase_v ^= 1; }
                int first_pv = 1;
                int dd_waited = 0;
                #pragma unroll 1
                for (int pv = 0; pv < 32; pv++) {
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
                    int vs_1 = vt % 2;
                    int vph_1 = vt / 2 & 1;
                    mbarrier_wait(v_full_addr + (vs_1) * 8, vph_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j = ((1) ? first_pv : 0);
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135365776, ((init_j) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_5) + 256U);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_5) + 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, ((init_j) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_6 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_6) + 512U);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_6) + 512U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135365776, ((init_j) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_7 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_7) + 768U);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_7) + 768U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (192)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135365776, ((init_j) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (192)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 135365776, 1);
                        }
                    }
                    elect_commit(v_empty_addr + (vs_1) * 8);
                    int vt_0 = vtile_n + 1;
                    int vs_1_1 = vt_0 % 2;
                    int vph_2 = vt_0 / 2 & 1;
                    mbarrier_wait(v_full_addr + (vs_1_1) * 8, vph_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j_3 = ((0) ? first_pv : 0);
                    int _mma_a_lo_8 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_8 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2 + 1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135365776, ((init_j_3) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_9 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2 + 1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_9) + 256U);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_9) + 256U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (64)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135365776, ((init_j_3) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (64)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_10 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2 + 1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_10) + 512U);
                        uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_10) + 512U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 135365776, ((init_j_3) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 135365776, 1);
                        }
                    }
                    int _mma_a_lo_11 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                    int _mma_b_lo_11 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x4000000) + (pf_stage_v * 2 + 1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_a_lo_11) + 768U);
                        uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>((_mma_b_lo_11) + 768U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (192)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 135365776, ((init_j_3) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_o + (192)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 135365776, 1);
                        }
                    }
                    elect_commit(v_empty_addr + (vs_1_1) * 8);
                    first_pv = 0;
                    vtile_n = vtile_n + 2;
                    elect_commit(pv_done_addr + (pv) * 8);
                    pf_phase_v ^= 1;
                }
                if (dd_waited == 0) {
                    if (have_prev_v != 0) {
                        mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                        _phase_decode_done_0 ^= 1;
                    }
                }
                #pragma unroll 1
                for (int spare = npairs_v; spare < 32; spare++) {
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
