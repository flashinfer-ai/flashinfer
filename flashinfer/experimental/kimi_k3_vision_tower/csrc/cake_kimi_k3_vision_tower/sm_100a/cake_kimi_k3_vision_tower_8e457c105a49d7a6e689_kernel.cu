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
#include "cake_kimi_k3_vision_tower_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBS_OFFSET 64
#define TMEM_OUTPUT_0_OFFSET 384
#define NUM_KV_STAGES 7
#define NUM_QP_STAGES 2
#define SMEM_SCALES_OFF 1024
#define SMEM_SCALES_STAGE_BYTES 2048
#define SMEM_SCALES_STRIDE 2048
#define SMEM_SCALE_X_OFF 183296
#define SMEM_SCALE_X_STAGE_BYTES 1024
#define SMEM_SCALE_X_STRIDE 1024
#define SMEM_MREF_X_OFF 184320
#define SMEM_MREF_X_STAGE_BYTES 2048
#define SMEM_MREF_X_STRIDE 2048
#define SMEM_SMEM_QA_OFF 3072
#define SMEM_SMEM_QA_STAGE_BYTES 32768
#define SMEM_SMEM_QA_STRIDE 32768
#define SMEM_SMEM_KV_OFF 68608
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 68608
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_O_OFF 3072
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_TOTAL 186368
#define USE_TMEM_LD_RED 0
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_vision_tower_8e457c105a49d7a6e689(CakeTensorMap const* Q, __nv_bfloat16* __restrict__ Q_raw, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ O, CakeTensorMap const* O_tma, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ unit_table, unsigned long long* __restrict__ probe, unsigned int total_tiles, int num_heads, float softmax_scale_log2, float* __restrict__ partial_O, float* __restrict__ partial_ML)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 168)
    #define p_full_2_addr (mbar_base + 192)
    #define o_full_addr (mbar_base + 216)
    #define mref_full_addr (mbar_base + 240)
    #define corr_done_addr (mbar_base + 256)
    #define scale_full_addr (mbar_base + 272)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(O_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* scales = reinterpret_cast<float*>(smem_raw + 1024);
    const int scales_addr = smem + 1024;
    float* scale_x = reinterpret_cast<float*>(smem_raw + 183296);
    const int scale_x_addr = smem + 183296;
    float* mref_x = reinterpret_cast<float*>(smem_raw + 184320);
    const int mref_x_addr = smem + 184320;
    __nv_bfloat16* smem_qa = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_qa_addr = smem + 3072;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_kv_addr = smem + 68608;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_v_addr = smem + 68608;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_o_addr = smem + 3072;

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[0..288)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'qp' ---
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'kv' ---
            // kv_full: 7 barriers, init_count=2
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            // kv_empty: 7 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_full: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // p_full: 3 barriers, init_count=512
            mbarrier_init(smem + 168, 512);
            mbarrier_init(smem + 176, 512);
            mbarrier_init(smem + 184, 512);
            // p_full_2: 3 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            mbarrier_init(smem + 208, 256);
            // o_full: 3 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // mref_full: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            // scale_full: 2 barriers, init_count=128
            mbarrier_init(smem + 272, 128);
            mbarrier_init(smem + 280, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 0) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probs = taddr + 64;
    const int tmem_output_0 = taddr + 384;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            unsigned int stage = make_warp_uniform(warp / 4);
            uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
            int sync_group = make_warp_uniform(warp_group_idx);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int q_slot_s = 0;
            unsigned int other_stage = make_warp_uniform(1 - stage);
            unsigned int sbase = 0;
            unsigned int unit_par = 0;
            int nx_seg = 0;
            int nx_packed = 0;
            int nx_begin = 0;
            int nx_len = 0;
            int nx_rng = 0;
            int nx_slot = -1;
            unsigned int nx_tile = cluster_id;
            if (nx_tile < total_tiles) {
                int nrec = nx_tile * 4;
                int w0 = unit_table[nrec];
                int w1 = unit_table[nrec + 1];
                int w2 = unit_table[nrec + 2];
                int w3 = unit_table[nrec + 3];
                nx_seg = w0;
                nx_packed = w1;
                nx_begin = w2;
                nx_len = w3;
                nx_rng = 0;
                nx_slot = -1;
            }
            unsigned int _phase_mref_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int head = nx_packed >> 16;
                int c = nx_packed & 65535;
                int u_begin = nx_begin;
                int u_len = nx_len;
                int m_block = c * 2 + cta_rank;
                int num_n_blocks = (u_len + BLOCK_N - 1) / BLOCK_N;
                int n_count = (num_n_blocks + 1) / 2;
                nx_tile = tile_idx + num_clusters;
                if (nx_tile < total_tiles) {
                    int nrec_1 = nx_tile * 4;
                    int w0_1 = unit_table[nrec_1];
                    int w1_1 = unit_table[nrec_1 + 1];
                    int w2_1 = unit_table[nrec_1 + 2];
                    int w3_1 = unit_table[nrec_1 + 3];
                    nx_seg = w0_1;
                    nx_packed = w1_1;
                    nx_begin = w2_1;
                    nx_len = w3_1;
                    nx_rng = 0;
                    nx_slot = -1;
                }
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                int mref_wr_off = unit_par * 256 + stage * (unsigned int)BLOCK_M;
                int mref_rd_off = unit_par * 256 + other_stage * (unsigned int)BLOCK_M;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < n_count; n_iter++) {
                    int n_block = (unsigned int)(2 * n_count - 1) - stage - 2 * n_iter;
                    unsigned int kk = 2 * n_iter + stage;
                    unsigned int sbuf = kk % 3;
                    mbarrier_wait(s_full_addr + (sbuf) * 8, (kk / 3 ^ sbase >> sbuf) & 1);
                    int s_addr = taddr + sbuf * 128 + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CAKE_INF;
                    {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                            : "r"(s_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                            : "r"(s_addr + 32));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95])
                            : "r"(s_addr + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127])
                            : "r"(s_addr + 96));
                        float2 _reg_reduce_max2_0 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_0);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_0);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_0);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_0);
                        float sv_max = row_max_reduce(_reg_reduce_max2_0);
                        tile_max = sv_max;
                    }
                    int tail_valid = u_len - n_block * BLOCK_N;
                    if (tail_valid < BLOCK_N) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_1 = tail_valid;
                            if (_lim_1 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_2 = tail_valid - 32;
                            if (_lim_2 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_2));
                            }
                        }
                        if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -CAKE_INF;
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_3 = tail_valid - 64;
                            if (_lim_3 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_3));
                            }
                        }
                        if (!(_slice_lo_mask_2 & (1u << 0))) sv[64] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 1))) sv[65] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 2))) sv[66] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 3))) sv[67] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 4))) sv[68] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 5))) sv[69] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 6))) sv[70] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 7))) sv[71] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 8))) sv[72] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 9))) sv[73] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 10))) sv[74] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 11))) sv[75] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 12))) sv[76] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 13))) sv[77] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 14))) sv[78] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 15))) sv[79] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 16))) sv[80] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 17))) sv[81] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 18))) sv[82] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 19))) sv[83] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 20))) sv[84] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 21))) sv[85] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 22))) sv[86] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 23))) sv[87] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 24))) sv[88] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 25))) sv[89] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 26))) sv[90] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 27))) sv[91] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 28))) sv[92] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 29))) sv[93] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 30))) sv[94] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 31))) sv[95] = -CAKE_INF;
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_4 = tail_valid - 96;
                            if (_lim_4 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_4 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_4));
                            }
                        }
                        if (!(_slice_lo_mask_3 & (1u << 0))) sv[96] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 1))) sv[97] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 2))) sv[98] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 3))) sv[99] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 4))) sv[100] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 5))) sv[101] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 6))) sv[102] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 7))) sv[103] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 8))) sv[104] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 9))) sv[105] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 10))) sv[106] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 11))) sv[107] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 12))) sv[108] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 13))) sv[109] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 14))) sv[110] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 15))) sv[111] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 16))) sv[112] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 17))) sv[113] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 18))) sv[114] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 19))) sv[115] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 20))) sv[116] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 21))) sv[117] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 22))) sv[118] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 23))) sv[119] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 24))) sv[120] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 25))) sv[121] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 26))) sv[122] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 27))) sv[123] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 28))) sv[124] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 29))) sv[125] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 30))) sv[126] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 31))) sv[127] = -CAKE_INF;
                        float2 _reg_reduce_max2_5 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_5);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_5);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_5);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_5);
                        float sv_max_1 = row_max_reduce(_reg_reduce_max2_5);
                        tile_max = sv_max_1;
                    }
                    float m_prev = -CAKE_INF;
                    if (kk != 0) {
                        mbarrier_wait(mref_full_addr + (other_stage) * 8, _phase_mref_full);
                        _phase_mref_full ^= 1;
                        m_prev = mref_x[warp % 4 * 32 + lane + mref_rd_off];
                    }
                    float _max_1 = max_noftz(tile_max, m_prev);
                    float new_max = _max_1;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(m_prev, softmax_scale_log2, -new_max_scaled);
                    float acc_scale_log2 = _fma_0;
                    float acc_scale;
                    float m_ref;
                    if (acc_scale_log2 >= -8.0f) {
                        acc_scale = 1.0f;
                        m_ref = m_prev;
                        new_max_scaled = m_prev * softmax_scale_log2;
                    } else {
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((m_prev > -CAKE_INF) ? _exp2_0 : 1.0f);
                        m_ref = new_max;
                    }
                    if (stage == 0 || n_iter != (unsigned int)(n_count - 1)) {
                        mref_x[warp % 4 * 32 + lane + mref_wr_off] = m_ref;
                        mbarrier_arrive(mref_full_addr + (stage) * 8);
                    }
                    float sum_scale = 1.0f;
                    if (row_max != m_ref) {
                        if (row_max > -CAKE_INF) {
                            float _fma_1 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                            float _exp2_1 = approx_exp2(_fma_1);
                            sum_scale = _exp2_1;
                        }
                    }
                    row_max = m_ref;
                    const float2 _fma_b2_6 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_7 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_6, _fma_c2_7);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_8 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                            sv[_le*2] = _exp2_pair_8.x;
                            sv[_le*2 + 1] = _exp2_pair_8.y;
                        } else {
                            sv[_le*2] = approx_exp2(sv[_le*2]);
                            sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                        }
                    }
                    sv[2] = approx_exp2(sv[2]);
                    scale_x[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    mbarrier_arrive(scale_full_addr + (stage) * 8);
                    #pragma unroll
                    for (int _le = 0; _le < 6; _le++) {
                        if (1 && _le >= 4) {
                            float2 _exp2_pair_9 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 3], sv[_le*2 + 1 + 3]));
                            sv[_le*2 + 3] = _exp2_pair_9.x;
                            sv[_le*2 + 1 + 3] = _exp2_pair_9.y;
                        } else {
                            sv[_le*2 + 3] = approx_exp2(sv[_le*2 + 3]);
                            sv[_le*2 + 1 + 3] = approx_exp2(sv[_le*2 + 1 + 3]);
                        }
                    }
                    sv[15] = approx_exp2(sv[15]);
                    const float2 _fma_b2_10 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_11 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 56; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_10, _fma_c2_11);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (1 && _le >= 12) {
                            float2 _exp2_pair_12 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 16], sv[_le*2 + 1 + 16]));
                            sv[_le*2 + 16] = _exp2_pair_12.x;
                            sv[_le*2 + 1 + 16] = _exp2_pair_12.y;
                        } else {
                            sv[_le*2 + 16] = approx_exp2(sv[_le*2 + 16]);
                            sv[_le*2 + 1 + 16] = approx_exp2(sv[_le*2 + 1 + 16]);
                        }
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (1 && _le >= 12) {
                            float2 _exp2_pair_13 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 48], sv[_le*2 + 1 + 48]));
                            sv[_le*2 + 48] = _exp2_pair_13.x;
                            sv[_le*2 + 1 + 48] = _exp2_pair_13.y;
                        } else {
                            sv[_le*2 + 48] = approx_exp2(sv[_le*2 + 48]);
                            sv[_le*2 + 1 + 48] = approx_exp2(sv[_le*2 + 1 + 48]);
                        }
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        if (1 && _le >= 12) {
                            float2 _exp2_pair_14 = ex2_emulation_f32x2_value(make_float2(sv[_le*2 + 80], sv[_le*2 + 1 + 80]));
                            sv[_le*2 + 80] = _exp2_pair_14.x;
                            sv[_le*2 + 1 + 80] = _exp2_pair_14.y;
                        } else {
                            sv[_le*2 + 80] = approx_exp2(sv[_le*2 + 80]);
                            sv[_le*2 + 1 + 80] = approx_exp2(sv[_le*2 + 1 + 80]);
                        }
                    }
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 112] = approx_exp2(sv[_le + 112]);
                    }
                    int p_addr = s_addr + 64;
                    float2 _f2_0 = make_float2(sv[0], sv[1]);
                    float2 partial = _f2_0;
                    #pragma unroll
                    for (int pair = 2; pair < 32; pair += 2) {
                        float2 _f2_1 = make_float2(sv[pair], sv[pair + 1]);
                        partial = add_f32x2(partial, _f2_1);
                    }
                    uint32_t sv_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                        sv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[15])));
                    float2 _f2_2 = make_float2(sv[32], sv[33]);
                    float2 partial_0 = _f2_2;
                    #pragma unroll
                    for (int pair_1 = 34; pair_1 < 64; pair_1 += 2) {
                        float2 _f2_3 = make_float2(sv[pair_1], sv[pair_1 + 1]);
                        partial_0 = add_f32x2(partial_0, _f2_3);
                    }
                    uint32_t sv_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 32], sv[_lp*2+1 + 32]));
                        sv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (sbuf) * 8) & 0xFEFFFFFF) : "memory");
                    float2 _f2_4 = make_float2(sv[64], sv[65]);
                    float2 partial_2 = _f2_4;
                    #pragma unroll
                    for (int pair_2 = 66; pair_2 < 96; pair_2 += 2) {
                        float2 _f2_5 = make_float2(sv[pair_2], sv[pair_2 + 1]);
                        partial_2 = add_f32x2(partial_2, _f2_5);
                    }
                    uint32_t sv_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 64], sv[_lp*2+1 + 64]));
                        sv_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[15])));
                    float2 _f2_6 = make_float2(sv[96], sv[97]);
                    float2 partial_4 = _f2_6;
                    #pragma unroll
                    for (int pair_3 = 98; pair_3 < 128; pair_3 += 2) {
                        float2 _f2_7 = make_float2(sv[pair_3], sv[pair_3 + 1]);
                        partial_4 = add_f32x2(partial_4, _f2_7);
                    }
                    uint32_t sv_bf16_5[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 96], sv[_lp*2+1 + 96]));
                        sv_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 48), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (sbuf) * 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    float2 sum01 = add_f32x2(partial, partial_0);
                    float2 sum23 = add_f32x2(partial_2, partial_4);
                    float2 total_sum = add_f32x2(sum01, sum23);
                    row_sum = row_sum * sum_scale + total_sum.x + total_sum.y;
                }
                scales[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum;
                scales[warp % 4 * 32 + lane + scale_off] = row_max;
                unit_par = unit_par ^ 1;
                unsigned int nb = 2 * n_count;
                sbase = sbase ^ ((nb + 2) / 3 & 1 | ((nb + 1) / 3 & 1) << 1 | (nb / 3 & 1) << 2);
                if (sync_group == 0) {
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            unsigned int oph = 0;
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
            unsigned int q_slot_c = 0;
            unsigned int _phase_scale_full_0 = 0;
            unsigned int _phase_scale_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles; tile_idx_1 += num_clusters) {
                int rec = tile_idx_1 * 4;
                int seg = unit_table[rec];
                int packed = unit_table[rec + 1];
                int head_1 = packed >> 16;
                int c_1 = packed & 65535;
                int doc_begin = seg_begin[seg];
                int doc_len = seg_len[seg];
                int m_block_1 = c_1 * 2 + cta_rank;
                int num_n_blocks_1 = (doc_len + BLOCK_N - 1) / BLOCK_N;
                int n_count_1 = (num_n_blocks_1 + 1) / 2;
                mbarrier_wait(scale_full_addr, _phase_scale_full_0);
                _phase_scale_full_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                _phase_scale_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scale_k = scale_x[warp % 4 * 32 + lane + BLOCK_M];
                int _vote_0 = __all_sync(0xFFFFFFFF, scale_k == 1.0f);
                int skip_k = _vote_0;
                mbarrier_arrive(corr_done_addr + 8);
                if (skip_k == 0) {
                    mbarrier_wait(o_full_addr, oph & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int col = 0; col < HEAD_DIM / 16; col++) {
                        int addr_k = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], addr_k);
                        const float2 _scale2_0 = {scale_k, scale_k};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                        tmem_st_x16_f32(addr_k, _tmem_load_0);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                } else {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(o_full_addr, oph & 1);
                }
                oph = oph ^ 1;
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < n_count_1; n_iter_1++) {
                    unsigned int k0 = 2 * n_iter_1;
                    unsigned int b0 = k0 % 3;
                    unsigned int pb0 = (k0 + 2) % 3;
                    mbarrier_wait(scale_full_addr, _phase_scale_full_0);
                    _phase_scale_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale_k_0 = scale_x[warp % 4 * 32 + lane];
                    int _vote_1 = __all_sync(0xFFFFFFFF, scale_k_0 == 1.0f);
                    int skip_k_1 = _vote_1;
                    mbarrier_arrive(corr_done_addr);
                    if (skip_k_1 == 0) {
                        mbarrier_wait(o_full_addr + (pb0) * 8, oph >> pb0 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col_1 = 0; col_1 < HEAD_DIM / 16; col_1++) {
                            int addr_k_1 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_1 * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr_k_1);
                            const float2 _scale2_1 = {scale_k_0, scale_k_0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                            tmem_st_x16_f32(addr_k_1, _tmem_load_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (b0) * 8) & 0xFEFFFFFF) : "memory");
                    } else {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (b0) * 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_wait(o_full_addr + (pb0) * 8, oph >> pb0 & 1);
                    }
                    oph = oph ^ pb0 + 1 + (pb0 >> 1);
                    unsigned int b1 = (k0 + 1) % 3;
                    mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                    _phase_scale_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float scale_k_2 = scale_x[warp % 4 * 32 + lane + BLOCK_M];
                    int _vote_2 = __all_sync(0xFFFFFFFF, scale_k_2 == 1.0f);
                    int skip_k_3 = _vote_2;
                    mbarrier_arrive(corr_done_addr + 8);
                    if (skip_k_3 == 0) {
                        mbarrier_wait(o_full_addr + (b0) * 8, oph >> b0 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col_2 = 0; col_2 < HEAD_DIM / 16; col_2++) {
                            int addr_k_2 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                            float _tmem_load_2[16];
                            tmem_ld_x16(&_tmem_load_2[0], addr_k_2);
                            const float2 _scale2_2 = {scale_k_2, scale_k_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_2);
                            tmem_st_x16_f32(addr_k_2, _tmem_load_2);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (b1) * 8) & 0xFEFFFFFF) : "memory");
                    } else {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (b1) * 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_wait(o_full_addr + (b0) * 8, oph >> b0 & 1);
                    }
                    oph = oph ^ b0 + 1 + (b0 >> 1);
                }
                unsigned int lb = (2 * n_count_1 - 1) % 3;
                mbarrier_wait(o_full_addr + (lb) * 8, oph >> lb & 1);
                oph = oph ^ lb + 1 + (lb >> 1);
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                float m0 = scales[warp % 4 * 32 + lane];
                float m1 = scales[warp % 4 * 32 + lane + BLOCK_M];
                float l0 = scales[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float l1 = scales[warp % 4 * 32 + lane + 3 * BLOCK_M];
                float _max_2 = max_noftz(m0, m1);
                float m_ref_1 = _max_2;
                float safe_ref = ((m_ref_1 == -CAKE_INF) ? 0.0f : m_ref_1);
                float ref_scaled = safe_ref * softmax_scale_log2;
                float a0 = 0.0f;
                float a1 = 0.0f;
                if (m0 > -CAKE_INF) {
                    float _fma_2 = __fmaf_rn(m0, softmax_scale_log2, -ref_scaled);
                    float _exp2_2 = approx_exp2(_fma_2);
                    a0 = _exp2_2;
                }
                if (m1 > -CAKE_INF) {
                    float _fma_3 = __fmaf_rn(m1, softmax_scale_log2, -ref_scaled);
                    float _exp2_3 = approx_exp2(_fma_3);
                    a1 = _exp2_3;
                }
                float final_sum = l0 * a0 + l1 * a1;
                float inv_sum;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_0 = approx_rcp(final_sum);
                    inv_sum = _rcp_0;
                } else {
                    inv_sum = 0.0f;
                }
                float inv_o = inv_sum;
                int tile_row = m_block_1 * BLOCK_M;
                int tile_valid = doc_len - tile_row;
                unsigned int slot0 = q_slot_c;
                if (tile_valid >= BLOCK_M) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    #pragma unroll
                    for (int col_3 = 0; col_3 < HEAD_DIM / 16; col_3++) {
                        int addr = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_3 * 16);
                        float _tmem_load_3[16];
                        tmem_ld_x16(&_tmem_load_3[0], addr);
                        const float2 _scale2_3 = {inv_o, inv_o};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_3);
                        uint32_t _tmem_load_3_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                            _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int srow = (unsigned int)(col_3 / 4 * 128) + slot0 * 256 + (unsigned int)(warp % 4 * 32 + lane);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o_addr + (unsigned int)(srow * 128 + col_3 % 4 * 32 ^ (srow * 128 + col_3 % 4 * 32 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o_addr + (unsigned int)(srow * 128 + (col_3 % 4 * 32 + 16) ^ (srow * 128 + (col_3 % 4 * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            tma_store_4d(O_tma, 0, doc_begin + tile_row, head_1, 0, smem_o_addr + 2 * slot0 * 16384);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                } else {
                    int local_row = m_block_1 * BLOCK_M + (warp % 4 * 32 + lane);
                    int out_row = (doc_begin + local_row) * num_heads + head_1;
                    #pragma unroll
                    for (int col_4 = 0; col_4 < HEAD_DIM / 16; col_4++) {
                        int addr_o = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_4 * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], addr_o);
                        if (local_row < doc_len) {
                            {
                                const float2 _prescale2_4 = {inv_o, inv_o};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[0])[_ps], _prescale2_4);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 16; _ps++)
                                    _tmem_load_4[0 + _ps] *= inv_o;
                                #endif
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_4[0 + 0], _tmem_load_4[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_4[0 + 2], _tmem_load_4[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_4[0 + 4], _tmem_load_4[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_4[0 + 6], _tmem_load_4[0 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_4[0 + 8], _tmem_load_4[0 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_4[0 + 10], _tmem_load_4[0 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_4[0 + 12], _tmem_load_4[0 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_4[0 + 14], _tmem_load_4[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_row * HEAD_DIM + col_4 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_row * HEAD_DIM + col_4 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                }
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(q_empty_addr + (q_slot_c) * 8);
                    }
                }
                q_slot_c += 1;
                if (q_slot_c == 2) { q_slot_c = 0; }
            }
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            if (cta_rank == 0) {
                unsigned int kv_stage = 0;
                unsigned int kv_phase = 0;
                unsigned int pph = 0;
                unsigned int q_stage = 0;
                unsigned int q_phase = 0;
                int nx_seg_1 = 0;
                int nx_packed_1 = 0;
                int nx_begin_1 = 0;
                int nx_len_1 = 0;
                int nx_rng_1 = 0;
                int nx_slot_1 = -1;
                unsigned int nx_tile_1 = cluster_id;
                if (nx_tile_1 < total_tiles) {
                    int nrec_2 = nx_tile_1 * 4;
                    int w0_2 = unit_table[nrec_2];
                    int w1_2 = unit_table[nrec_2 + 1];
                    int w2_2 = unit_table[nrec_2 + 2];
                    int w3_2 = unit_table[nrec_2 + 3];
                    nx_seg_1 = w0_2;
                    nx_packed_1 = w1_2;
                    nx_begin_1 = w2_2;
                    nx_len_1 = w3_2;
                    nx_rng_1 = 0;
                    nx_slot_1 = -1;
                }
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles; tile_idx_2 += num_clusters) {
                    int head_2 = nx_packed_1 >> 16;
                    int c_2 = nx_packed_1 & 65535;
                    int u_begin_1 = nx_begin_1;
                    int u_len_1 = nx_len_1;
                    int m_block_2 = c_2 * 2 + cta_rank;
                    int num_n_blocks_2 = (u_len_1 + BLOCK_N - 1) / BLOCK_N;
                    int n_count_2 = (num_n_blocks_2 + 1) / 2;
                    nx_tile_1 = tile_idx_2 + num_clusters;
                    if (nx_tile_1 < total_tiles) {
                        int nrec_3 = nx_tile_1 * 4;
                        int w0_3 = unit_table[nrec_3];
                        int w1_3 = unit_table[nrec_3 + 1];
                        int w2_3 = unit_table[nrec_3 + 2];
                        int w3_3 = unit_table[nrec_3 + 3];
                        nx_seg_1 = w0_3;
                        nx_packed_1 = w1_3;
                        nx_begin_1 = w2_3;
                        nx_len_1 = w3_3;
                        nx_rng_1 = 0;
                        nx_slot_1 = -1;
                    }
                    mbarrier_wait_cluster_hint(q_full_addr + (q_stage) * 8, q_phase, 10000000);
                    unsigned int n_blocks = 2 * n_count_2;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                    int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(kv_empty_addr + (kv_stage) * 8, (uint16_t)(3));
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                    int _mma_b_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(kv_empty_addr + (kv_stage) * 8, (uint16_t)(3));
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    if (n_count_2 > 1) {
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                        int _mma_b_lo_2 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_scores + (256))), "r"(0));
                        elect_commit_cg2_multicast(s_full_addr + 16, (uint16_t)(3));
                        elect_commit_cg2_multicast(kv_empty_addr + (kv_stage) * 8, (uint16_t)(3));
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int kb = 0; kb < n_blocks; kb++) {
                        unsigned int rb = kb % 3;
                        int first_pv_flag = first_pv;
                        unsigned int v_stage = kv_stage;
                        unsigned int v_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                        mbarrier_wait(p_full_addr + (rb) * 8, pph >> rb & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"((unsigned int)tmem_probs + rb * 128), "r"(((first_pv_flag) ? 0 : 1)));
                        mbarrier_wait(p_full_2_addr + (rb) * 8, pph >> rb & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_4), "r"((unsigned int)tmem_probs + rb * 128), "r"(1));
                        pph = pph ^ rb + 1 + (rb >> 1);
                        elect_commit_cg2_multicast(o_full_addr + (rb) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(kv_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        if (n_blocks > kb + 3) {
                            unsigned int k_stage = kv_stage;
                            unsigned int k_phase = kv_phase;
                            kv_stage += 1;
                            if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                            mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_5 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                            int _mma_b_lo_5 = (((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_scores + (rb * 128))), "r"(0));
                            elect_commit_cg2_multicast(s_full_addr + (rb) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(kv_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        }
                        first_pv = 0;
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; q_phase ^= 1; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int q_load_stage = 0;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Q)) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(O_tma)) : "memory");
            }
            int nx_seg_2 = 0;
            int nx_packed_2 = 0;
            int nx_begin_2 = 0;
            int nx_len_2 = 0;
            int nx_rng_2 = 0;
            int nx_slot_2 = -1;
            unsigned int nx_tile_2 = cluster_id;
            if (nx_tile_2 < total_tiles) {
                int nrec_4 = nx_tile_2 * 4;
                int w0_4 = unit_table[nrec_4];
                int w1_4 = unit_table[nrec_4 + 1];
                int w2_4 = unit_table[nrec_4 + 2];
                int w3_4 = unit_table[nrec_4 + 3];
                nx_seg_2 = w0_4;
                nx_packed_2 = w1_4;
                nx_begin_2 = w2_4;
                nx_len_2 = w3_4;
                nx_rng_2 = 0;
                nx_slot_2 = -1;
            }
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles; tile_idx_3 += num_clusters) {
                int head_3 = nx_packed_2 >> 16;
                int c_3 = nx_packed_2 & 65535;
                int u_begin_2 = nx_begin_2;
                int u_len_2 = nx_len_2;
                int m_block_3 = c_3 * 2 + cta_rank;
                int num_n_blocks_3 = (u_len_2 + BLOCK_N - 1) / BLOCK_N;
                int n_count_3 = (num_n_blocks_3 + 1) / 2;
                nx_tile_2 = tile_idx_3 + num_clusters;
                if (nx_tile_2 < total_tiles) {
                    int nrec_5 = nx_tile_2 * 4;
                    int w0_5 = unit_table[nrec_5];
                    int w1_5 = unit_table[nrec_5 + 1];
                    int w2_5 = unit_table[nrec_5 + 2];
                    int w3_5 = unit_table[nrec_5 + 3];
                    nx_seg_2 = w0_5;
                    nx_packed_2 = w1_5;
                    nx_begin_2 = w2_5;
                    nx_len_2 = w3_5;
                    nx_rng_2 = 0;
                    nx_slot_2 = -1;
                }
                int q_local = m_block_3 * BLOCK_M;
                int q_row = u_begin_2 + q_local;
                int q_remaining = u_len_2 - q_local;
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + (q_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_qa_addr + q_load_stage * 32768, Q, 0, q_row, head_3, 0, ((q_full_addr + (q_load_stage) * 8) & 0xFEFFFFFF));
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                int top = 2 * n_count_3 - 1;
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, K, 0, u_begin_2 + top * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, K, 0, u_begin_2 + (top - 1) * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                if (n_count_3 > 1) {
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, K, 0, u_begin_2 + (top - 2) * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                #pragma unroll 1
                for (unsigned int kb_1 = 0; kb_1 < 2 * n_count_3; kb_1++) {
                    int blk = (unsigned int)top - kb_1;
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, V, 0, u_begin_2 + blk * BLOCK_N, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                    if (kb_1 + 3 < (unsigned int)(2 * n_count_3)) {
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, K, 0, u_begin_2 + (blk - 3) * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        }
                        load_stage += 1;
                        if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
