// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
#include "cake_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_STATE_OFFSET 0
#define TMEM_TMEM_Q_STATE_OFFSET 128
#define TMEM_TMEM_STATE_INP_OFFSET 192
#define TMEM_TMEM_CG0_SHARED_ACC_OFFSET 256
#define TMEM_TMEM_CG1_SHARED_ACC_OFFSET 384
#define TMEM_TMEM_SHARED_INP_OFFSET 448
#define NUM_K_PIPE_STAGES 4
#define NUM_Q_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 3
#define NUM_GATE_PIPE_STAGES 5
#define NUM_ONE_STAGE_STAGES 1
#define NUM_CG0_ACC_PIPE_STAGES 2
#define NUM_AINV_PIPE_STAGES 3
#define NUM_QK_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 33792
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_K_TRANS_MMA_OFF 33792
#define SMEM_SMEM_K_TRANS_MMA_STAGE_BYTES 16384
#define SMEM_SMEM_K_TRANS_MMA_STRIDE 16384
#define SMEM_SMEM_V_OFF 99328
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_V_MMA_OFF 99328
#define SMEM_SMEM_V_MMA_STAGE_BYTES 8192
#define SMEM_SMEM_V_MMA_STRIDE 16384
#define SMEM_SMEM_AINV_OFF 148480
#define SMEM_SMEM_AINV_STAGE_BYTES 8192
#define SMEM_SMEM_AINV_STRIDE 8192
#define SMEM_SMEM_AINV_RM_OFF 148480
#define SMEM_SMEM_AINV_RM_STAGE_BYTES 8192
#define SMEM_SMEM_AINV_RM_STRIDE 8192
#define SMEM_SMEM_QK_OFF 173056
#define SMEM_SMEM_QK_STAGE_BYTES 8192
#define SMEM_SMEM_QK_STRIDE 8192
#define SMEM_SMEM_O_OFF 189440
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_SMEM_CUMSUMLOG_OFF 222208
#define SMEM_SMEM_CUMSUMLOG_STAGE_BYTES 256
#define SMEM_SMEM_CUMSUMLOG_STRIDE 256
#define SMEM_SMEM_CUMPROD_OFF 223488
#define SMEM_SMEM_CUMPROD_STAGE_BYTES 256
#define SMEM_SMEM_CUMPROD_STRIDE 256
#define SMEM_SMEM_BETA_OFF 224768
#define SMEM_SMEM_BETA_STAGE_BYTES 256
#define SMEM_SMEM_BETA_STRIDE 256
#define SMEM_TOTAL 226048
#define THREADS 384
#define NUM_O_HEADS_LOG2 4
#define HEAD_GROUP_LOG2 0
#define SINGLE_CHUNK_NO_STATE 0
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define ENABLE_CHECKPOINTS 0
#define IS_GQA 1
#define USE_STATE_INDICES 1



extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial_bf16state(CakeTensorMap const* Q, CakeTensorMap const* K, CakeTensorMap const* V, CakeTensorMap const* O, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ state_indices, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ output_state, __nv_bfloat16* __restrict__ checkpoint_state, int* __restrict__ cu_checkpoints, uint8_t* __restrict__ tensormap_workspace, long long initial_state_stride_slot, long long output_state_stride_slot, int checkpoint_every_n_tokens, float scale, int num_seqs, int num_q_heads, int num_v_heads, int total_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(O)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_k_addr = smem + 33792;
    __nv_bfloat16* smem_k_trans_mma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_k_trans_mma_addr = smem + 33792;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_v_addr = smem + 99328;
    __nv_bfloat16* smem_v_mma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_v_mma_addr = smem + 99328;
    __nv_bfloat16* smem_ainv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_ainv_addr = smem + 148480;
    __nv_bfloat16* smem_ainv_rm = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_ainv_rm_addr = smem + 148480;
    __nv_bfloat16* smem_qk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_qk_addr = smem + 173056;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int smem_o_addr = smem + 189440;
    float* smem_cumsumlog = reinterpret_cast<float*>(smem_raw + 222208);
    const int smem_cumsumlog_addr = smem + 222208;
    float* smem_cumprod = reinterpret_cast<float*>(smem_raw + 223488);
    const int smem_cumprod_addr = smem + 223488;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 224768);
    const int smem_beta_addr = smem + 224768;

    // Mbarrier init (30 groups, 72 barriers)
    // Mbarriers at smem_raw[0..576)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // load_k_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // load_k_empty: 4 barriers, init_count=2
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // --- pipeline 'q_pipe' ---
            // load_q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'v_pipe' ---
            // load_v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // --- pipeline 'gate_pipe' ---
            // load_gate_full: 5 barriers, init_count=32
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            // load_beta_full: 5 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            // q_state_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // q_state_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            // kv_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // kv_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // initial_state_loaded: 1 barriers, init_count=4
            mbarrier_init(smem + 216, 4);
            // --- pipeline 'cg0_acc_pipe' ---
            // cg0_shared_acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // cg0_shared_acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // --- pipeline 'one_stage' ---
            // cg1_shared_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            // cg1_shared_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 264, 128);
            // --- pipeline 'ainv_pipe' ---
            // ainv_ready: 3 barriers, init_count=128
            mbarrier_init(smem + 272, 128);
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // --- pipeline 'qk_pipe' ---
            // qk_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            mbarrier_init(smem + 304, 128);
            // state_inp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 312, 128);
            // vks_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            // nv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            // decay_v_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            // --- pipeline 'o_pipe' ---
            // o_store_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // --- pipeline 'gate_pipe' ---
            // gate_cg0_empty: 5 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            mbarrier_init(smem + 384, 128);
            mbarrier_init(smem + 392, 128);
            // gate_cg1_empty: 5 barriers, init_count=128
            mbarrier_init(smem + 400, 128);
            mbarrier_init(smem + 408, 128);
            mbarrier_init(smem + 416, 128);
            mbarrier_init(smem + 424, 128);
            mbarrier_init(smem + 432, 128);
            // beta_smem_empty: 5 barriers, init_count=128
            mbarrier_init(smem + 440, 128);
            mbarrier_init(smem + 448, 128);
            mbarrier_init(smem + 456, 128);
            mbarrier_init(smem + 464, 128);
            mbarrier_init(smem + 472, 128);
            // --- pipeline 'qk_pipe' ---
            // qk_smem_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 480, 1);
            mbarrier_init(smem + 488, 1);
            // --- pipeline 'ainv_pipe' ---
            // ainv_smem_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 496, 1);
            mbarrier_init(smem + 504, 1);
            mbarrier_init(smem + 512, 1);
            // --- pipeline 'q_pipe' ---
            // q_smem_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 520, 2);
            mbarrier_init(smem + 528, 2);
            // --- pipeline 'v_pipe' ---
            // v_smem_empty: 3 barriers, init_count=4
            mbarrier_init(smem + 536, 4);
            mbarrier_init(smem + 544, 4);
            mbarrier_init(smem + 552, 4);
            // --- pipeline 'o_pipe' ---
            // o_smem_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 560, 32);
            mbarrier_init(smem + 568, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 576);
    if (warp == 4) {
        int _tmem_hold = smem + 576;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define load_k_full_addr (mbar_base + 0)
    #define load_k_empty_addr (mbar_base + 32)
    #define load_q_full_addr (mbar_base + 64)
    #define load_v_full_addr (mbar_base + 80)
    #define load_gate_full_addr (mbar_base + 104)
    #define load_beta_full_addr (mbar_base + 144)
    #define q_state_acc_full_addr (mbar_base + 184)
    #define q_state_acc_empty_addr (mbar_base + 192)
    #define kv_acc_full_addr (mbar_base + 200)
    #define kv_acc_empty_addr (mbar_base + 208)
    #define initial_state_loaded_addr (mbar_base + 216)
    #define cg0_shared_acc_full_addr (mbar_base + 224)
    #define cg0_shared_acc_empty_addr (mbar_base + 240)
    #define cg1_shared_acc_full_addr (mbar_base + 256)
    #define cg1_shared_acc_empty_addr (mbar_base + 264)
    #define ainv_ready_addr (mbar_base + 272)
    #define qk_ready_addr (mbar_base + 296)
    #define state_inp_ready_addr (mbar_base + 312)
    #define vks_ready_addr (mbar_base + 320)
    #define nv_ready_addr (mbar_base + 328)
    #define decay_v_ready_addr (mbar_base + 336)
    #define o_store_ready_addr (mbar_base + 344)
    #define gate_cg0_empty_addr (mbar_base + 360)
    #define gate_cg1_empty_addr (mbar_base + 400)
    #define beta_smem_empty_addr (mbar_base + 440)
    #define qk_smem_empty_addr (mbar_base + 480)
    #define ainv_smem_empty_addr (mbar_base + 496)
    #define q_smem_empty_addr (mbar_base + 520)
    #define v_smem_empty_addr (mbar_base + 536)
    #define o_smem_empty_addr (mbar_base + 560)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr;
    const int tmem_tmem_q_state = taddr + 128;
    const int tmem_tmem_state_inp = taddr + 192;
    const int tmem_tmem_cg0_shared_acc = taddr + 256;
    const int tmem_tmem_cg1_shared_acc = taddr + 384;
    const int tmem_tmem_shared_inp = taddr + 448;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: compute_group_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // compute_group_0_main
            unsigned int gate_cg0_stage = 0;
            unsigned int gate_cg0_phase = 0;
            unsigned int beta_cg0_stage = 0;
            unsigned int beta_cg0_phase = 0;
            unsigned int ainv_cg0_stage = 0;
            unsigned int ainv_cg0_phase = 1;
            unsigned int qk_cg0_stage = 0;
            unsigned int qk_cg0_phase = 1;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
                int num_o_heads = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx = tile / 2;
                int value_split_idx = tile % 2;
                int batch_idx = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx >> NUM_O_HEADS_LOG2 : base_tile_idx / num_o_heads);
                int head_idx = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx & num_o_heads - 1 : base_tile_idx % num_o_heads);
                int qk_head_idx = ((IS_GQA != 0) ? head_idx : ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_v_heads / num_q_heads)));
                int v_head_idx = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_q_heads / num_v_heads)) : head_idx);
                int batch_start = cu_seqlens[batch_idx];
                int batch_end = cu_seqlens[batch_idx + 1];
                int seqlen_b = batch_end - batch_start;
                int num_pairs_b = (seqlen_b + 128 - 1) / 128;
                int num_chunks_b = num_pairs_b * 2;
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks_b; chunk_idx += 2) {
                    int chunk_offset = batch_start + chunk_idx * 64;
                    int _cg0_marker = batch_idx + head_idx + chunk_offset + batch_end;
                    int warp_id_in_role = (warp - 0);
                    int warp_id_in_role_cg0 = warp_id_in_role;
                    int row_cg0 = warp_id_in_role_cg0 * 32 + lane;
                    int lane_quad_cg0 = lane & 3;
                    int lane_row_cg0 = lane / 4;
                    int qk_warp_row_base_cg0 = warp_id_in_role_cg0 * 16;
                    int qk_tmem_row_base_cg0 = warp_id_in_role_cg0 * 32 << 16;
                    unsigned int gate0_stage = gate_cg0_stage;
                    mbarrier_wait(load_gate_full_addr + (gate0_stage) * 8, gate_cg0_phase);
                    gate_cg0_stage += 1;
                    if (gate_cg0_stage == 5) { gate_cg0_stage = 0; gate_cg0_phase ^= 1; }
                    unsigned int gate1_stage = ((0) ? (gate0_stage + 1) % 5 : gate_cg0_stage);
                    {
                        mbarrier_wait(load_gate_full_addr + (gate1_stage) * 8, gate_cg0_phase);
                        gate_cg0_stage += 1;
                        if (gate_cg0_stage == 5) { gate_cg0_stage = 0; gate_cg0_phase ^= 1; }
                    }
                    int gate0_elem_base = gate0_stage * 64;
                    int gate1_elem_base = gate1_stage * 64;
                    float qk_transfer0_cg0[32];
                    float qk_transfer1_cg0[32];
                    #pragma unroll
                    for (int qk_j_cg0 = 0; qk_j_cg0 < 32; qk_j_cg0++) {
                        int qk_repeat_cg0 = qk_j_cg0 / 4;
                        int qk_reg_cg0 = qk_j_cg0 & 3;
                        int qk_row_cg0 = qk_warp_row_base_cg0 + lane_row_cg0 + qk_reg_cg0 / 2 * 8;
                        int qk_col_cg0 = qk_repeat_cg0 * 8 + lane_quad_cg0 * 2 + (qk_reg_cg0 & 1);
                        float qk_row_cumsumlog0_cg0 = smem_cumsumlog[gate0_elem_base + qk_row_cg0];
                        float _exp2_0 = approx_exp2(qk_row_cumsumlog0_cg0 - smem_cumsumlog[gate0_elem_base + qk_col_cg0]);
                        qk_transfer0_cg0[qk_j_cg0] = ((qk_row_cg0 >= qk_col_cg0) ? _exp2_0 : 0.0f);
                    }
                    {
                        #pragma unroll
                        for (int qk_j_cg0_1 = 0; qk_j_cg0_1 < 32; qk_j_cg0_1++) {
                            int qk_repeat_cg0_1 = qk_j_cg0_1 / 4;
                            int qk_reg_cg0_1 = qk_j_cg0_1 & 3;
                            int qk_row_cg0_1 = qk_warp_row_base_cg0 + lane_row_cg0 + qk_reg_cg0_1 / 2 * 8;
                            int qk_col_cg0_1 = qk_repeat_cg0_1 * 8 + lane_quad_cg0 * 2 + (qk_reg_cg0_1 & 1);
                            float qk_row_cumsumlog1_cg0 = smem_cumsumlog[gate1_elem_base + qk_row_cg0_1];
                            float _exp2_1 = approx_exp2(qk_row_cumsumlog1_cg0 - smem_cumsumlog[gate1_elem_base + qk_col_cg0_1]);
                            qk_transfer1_cg0[qk_j_cg0_1] = ((qk_row_cg0_1 >= qk_col_cg0_1) ? _exp2_1 : 0.0f);
                        }
                    }
                    mbarrier_arrive(gate_cg0_empty_addr + (gate0_stage) * 8);
                    {
                        mbarrier_arrive(gate_cg0_empty_addr + (gate1_stage) * 8);
                    }
                    unsigned int beta0_stage = beta_cg0_stage;
                    mbarrier_wait(load_beta_full_addr + (beta0_stage) * 8, beta_cg0_phase);
                    beta_cg0_stage += 1;
                    if (beta_cg0_stage == 5) { beta_cg0_stage = 0; beta_cg0_phase ^= 1; }
                    unsigned int beta1_stage = ((0) ? (beta0_stage + 1) % 5 : beta_cg0_stage);
                    {
                        mbarrier_wait(load_beta_full_addr + (beta1_stage) * 8, beta_cg0_phase);
                        beta_cg0_stage += 1;
                        if (beta_cg0_stage == 5) { beta_cg0_stage = 0; beta_cg0_phase ^= 1; }
                    }
                    int beta0_elem_base = beta0_stage * 64;
                    int beta1_elem_base = beta1_stage * 64;
                    unsigned int ainv0_stage = ainv_cg0_stage;
                    mbarrier_wait(ainv_smem_empty_addr + (ainv0_stage) * 8, ainv_cg0_phase);
                    ainv_cg0_stage += 1;
                    if (ainv_cg0_stage == 3) { ainv_cg0_stage = 0; ainv_cg0_phase ^= 1; }
                    unsigned int ainv1_stage = ((0) ? (ainv0_stage + 1) % 3 : ainv_cg0_stage);
                    {
                        mbarrier_wait(ainv_smem_empty_addr + (ainv1_stage) * 8, ainv_cg0_phase);
                        ainv_cg0_stage += 1;
                        if (ainv_cg0_stage == 3) { ainv_cg0_stage = 0; ainv_cg0_phase ^= 1; }
                        mbarrier_wait(cg0_shared_acc_full_addr, 0);
                        int kk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                            : "r"(kk_addr)
                            : "memory");
                        int kk_beta_row0_cg0 = qk_warp_row_base_cg0 + lane_row_cg0;
                        int kk_beta_row1_cg0 = kk_beta_row0_cg0 + 8;
                        float kk_beta0_cg0 = smem_beta[beta0_elem_base + kk_beta_row0_cg0];
                        float kk_beta1_cg0 = smem_beta[beta0_elem_base + kk_beta_row1_cg0];
                        int kk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                        int kk_stsm_col_lane_cg0 = (lane & 16) / 2;
                        #pragma unroll
                        for (int qk_repeat_cg0_2 = 0; qk_repeat_cg0_2 < 4; qk_repeat_cg0_2++) {
                            const int qk_j0_cg0 = qk_repeat_cg0_2 * 8;
                            float _t0[8];
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                reinterpret_cast<float2*>(_t0)[_ls] = mul_f32x2(reinterpret_cast<float2*>((_tmem_load_0 + qk_j0_cg0))[_ls], reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0))[_ls]);
                            const float2 _scale2_0 = {kk_beta0_cg0, kk_beta0_cg0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 0))[_ls], _scale2_0);
                            const float2 _scale2_1 = {kk_beta1_cg0, kk_beta1_cg0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 2))[_ls], _scale2_1);
                            const float2 _scale2_2 = {kk_beta0_cg0, kk_beta0_cg0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 4))[_ls], _scale2_2);
                            const float2 _scale2_3 = {kk_beta1_cg0, kk_beta1_cg0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t0 + 6))[_ls], _scale2_3);
                            uint32_t _t0_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t0[_lp*2 + 0], _t0[_lp*2+1 + 0]));
                                _t0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int kk_stsm_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)(kk_stsm_row_cg0 * 128 + (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0) * 2 ^ (kk_stsm_row_cg0 * 128 + (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0) * 2 >> 7 & 7) << 4));
                            uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t0_bf16[3]))
                                : "memory");
                        }
                        mbarrier_arrive(cg0_shared_acc_empty_addr);
                        mbarrier_wait(cg0_shared_acc_full_addr + 8, 0);
                        int kk_addr_0 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(kk_addr_0)
                            : "memory");
                        int kk_beta_row0_cg0_1 = qk_warp_row_base_cg0 + lane_row_cg0;
                        int kk_beta_row1_cg0_2 = kk_beta_row0_cg0_1 + 8;
                        float kk_beta0_cg0_3 = smem_beta[beta1_elem_base + kk_beta_row0_cg0_1];
                        float kk_beta1_cg0_4 = smem_beta[beta1_elem_base + kk_beta_row1_cg0_2];
                        int kk_stsm_row_cg0_5 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                        int kk_stsm_col_lane_cg0_6 = (lane & 16) / 2;
                        #pragma unroll
                        for (int qk_repeat_cg0_3 = 0; qk_repeat_cg0_3 < 4; qk_repeat_cg0_3++) {
                            const int qk_j0_cg0_1 = qk_repeat_cg0_3 * 8;
                            float _t1[8];
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                reinterpret_cast<float2*>(_t1)[_ls] = mul_f32x2(reinterpret_cast<float2*>((_tmem_load_1 + qk_j0_cg0_1))[_ls], reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_1))[_ls]);
                            const float2 _scale2_5 = {kk_beta0_cg0_3, kk_beta0_cg0_3};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 0))[_ls], _scale2_5);
                            const float2 _scale2_6 = {kk_beta1_cg0_4, kk_beta1_cg0_4};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 2))[_ls], _scale2_6);
                            const float2 _scale2_7 = {kk_beta0_cg0_3, kk_beta0_cg0_3};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 4))[_ls], _scale2_7);
                            const float2 _scale2_8 = {kk_beta1_cg0_4, kk_beta1_cg0_4};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_t1 + 6))[_ls], _scale2_8);
                            uint32_t _t1_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t1[_lp*2 + 0], _t1[_lp*2+1 + 0]));
                                _t1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int kk_stsm_addr_cg0_1 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)(kk_stsm_row_cg0_5 * 128 + (qk_repeat_cg0_3 * 16 + kk_stsm_col_lane_cg0_6) * 2 ^ (kk_stsm_row_cg0_5 * 128 + (qk_repeat_cg0_3 * 16 + kk_stsm_col_lane_cg0_6) * 2 >> 7 & 7) << 4));
                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0_1);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t1_bf16[3]))
                                : "memory");
                        }
                        mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    int inverse_group_cg0 = warp_id_in_role_cg0 / 2;
                    int inverse_local_warp_cg0 = warp_id_in_role_cg0 & 1;
                    unsigned int inverse_stage_cg0 = ((inverse_group_cg0 == 1) ? ainv1_stage : ainv0_stage);
                    int inverse_row_cg0 = ((0) ? warp_id_in_role_cg0 * 32 + lane : inverse_local_warp_cg0 * 32 + lane);
                    int diag_block_cg0 = inverse_row_cg0 / 8;
                    int lane_in_diag_cg0 = lane & 7;
                    int diag_col_base_cg0 = diag_block_cg0 * 8;
                    float inv_row_cg0[8];
                    if (inverse_row_cg0 < 64) {
                        int inv_diag_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)(inverse_row_cg0 * 128 + diag_col_base_cg0 * 2 ^ (inverse_row_cg0 * 128 + diag_col_base_cg0 * 2 >> 7 & 7) << 4));
                        unsigned int inv_row_packed_cg0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&inv_row_packed_cg0[0])), "=r"(*reinterpret_cast<uint32_t*>(&inv_row_packed_cg0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv_row_packed_cg0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv_row_packed_cg0[(0) + 3]))
                            : "r"(inv_diag_addr_cg0));
                        float inv_row_packed_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&inv_row_packed_cg0_f32[_pair * 2])[0]), "=f"((&inv_row_packed_cg0_f32[_pair * 2])[1])
                                : "r"(inv_row_packed_cg0[_pair]));
                        }
                        #pragma unroll
                        for (int inv_j = 0; inv_j < 8; inv_j++) {
                            inv_row_cg0[inv_j] = inv_row_packed_cg0_f32[inv_j];
                            if (lane_in_diag_cg0 == inv_j) {
                                inv_row_cg0[inv_j] = 1.0f;
                            }
                        }
                        int diag_group_base_cg0 = lane - lane_in_diag_cg0;
                        #pragma unroll
                        for (int src_row_cg0 = 0; src_row_cg0 < 7; src_row_cg0++) {
                            float row_scale_cg0 = -1.0f * inv_row_cg0[src_row_cg0];
                            #pragma unroll
                            for (int prev_col_cg0 = 0; prev_col_cg0 < src_row_cg0; prev_col_cg0++) {
                                int pivot_lane_cg0 = diag_group_base_cg0 + src_row_cg0;
                                float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_row_cg0[prev_col_cg0], pivot_lane_cg0);
                                float shfl_val_cg0 = _shfl_0;
                                if (lane_in_diag_cg0 > src_row_cg0) {
                                    inv_row_cg0[prev_col_cg0] = inv_row_cg0[prev_col_cg0] + row_scale_cg0 * shfl_val_cg0;
                                }
                            }
                            if (lane_in_diag_cg0 > src_row_cg0) {
                                inv_row_cg0[src_row_cg0] = row_scale_cg0;
                            }
                        }
                        uint32_t inv_row_cg0_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row_cg0[_lp*2 + 0], inv_row_cg0[_lp*2+1 + 0]));
                            inv_row_cg0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int inv_store_j_cg0 = 0; inv_store_j_cg0 < 4; inv_store_j_cg0++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(inv_diag_addr_cg0 + inv_store_j_cg0 * 4), "r"((inv_row_cg0_bf16[inv_store_j_cg0])));
                        }
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    int inv16_tile_base_cg0 = row_cg0 / 32 * 16;
                    int inv16_lane_row_cg0 = lane & 7;
                    int inv16_d_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + (inv16_tile_base_cg0 + 8) * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + (inv16_tile_base_cg0 + 8) * 2 >> 7 & 7) << 4));
                    int inv16_c_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    int inv16_a_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    int inv16_o_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    unsigned int inv16_d_frag_cg0[2];
                    unsigned int inv16_c_frag_cg0[1];
                    float inv16_dc_acc_cg0[4];
                    unsigned int inv16_dc_bf16_cg0[2];
                    unsigned int inv16_a_frag_cg0[1];
                    float inv16_o_acc_cg0[4];
                    unsigned int inv16_o_bf16_cg0[2];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_d_frag_cg0[0])
                        : "r"(inv16_d_addr_cg0)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_d_frag_cg0[1])
                        : "r"(inv16_d_addr_cg0)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_c_frag_cg0[0])
                        : "r"(inv16_c_addr_cg0)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(inv16_dc_acc_cg0[0]), "=f"(inv16_dc_acc_cg0[1]), "=f"(inv16_dc_acc_cg0[2]), "=f"(inv16_dc_acc_cg0[3])
                        : "r"(inv16_d_frag_cg0[0]), "r"(inv16_d_frag_cg0[1]), "r"(inv16_c_frag_cg0[0]));
                    #pragma unroll
                    for (int inv16_i_cg0 = 0; inv16_i_cg0 < 4; inv16_i_cg0++) {
                        inv16_dc_acc_cg0[inv16_i_cg0] = inv16_dc_acc_cg0[inv16_i_cg0] * -1.0f;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_dc_acc_cg0[_lp*2 + 0], inv16_dc_acc_cg0[_lp*2+1 + 0]));
                        inv16_dc_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_a_frag_cg0[0])
                        : "r"(inv16_a_addr_cg0)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                        : "=f"(inv16_o_acc_cg0[0]), "=f"(inv16_o_acc_cg0[1]), "=f"(inv16_o_acc_cg0[2]), "=f"(inv16_o_acc_cg0[3])
                        : "r"(inv16_dc_bf16_cg0[0]), "r"(inv16_dc_bf16_cg0[1]), "r"(inv16_a_frag_cg0[0]));
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_o_acc_cg0[_lp*2 + 0], inv16_o_acc_cg0[_lp*2+1 + 0]));
                        inv16_o_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)inv16_o_addr_cg0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&inv16_o_bf16_cg0[0]))
                        : "memory");
                    {
                        int inv16_lane_row_cg0_0 = lane & 7;
                        int inv16_d_addr_cg0_1 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + (inv16_tile_base_cg0 + 8) * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + (inv16_tile_base_cg0 + 8) * 2 >> 7 & 7) << 4));
                        int inv16_c_addr_cg0_2 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                        int inv16_a_addr_cg0_3 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                        int inv16_o_addr_cg0_4 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                        unsigned int inv16_d_frag_cg0_5[2];
                        unsigned int inv16_c_frag_cg0_6[1];
                        float inv16_dc_acc_cg0_7[4];
                        unsigned int inv16_dc_bf16_cg0_8[2];
                        unsigned int inv16_a_frag_cg0_9[1];
                        float inv16_o_acc_cg0_10[4];
                        unsigned int inv16_o_bf16_cg0_11[2];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv16_d_frag_cg0_5[0])
                            : "r"(inv16_d_addr_cg0_1)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv16_d_frag_cg0_5[1])
                            : "r"(inv16_d_addr_cg0_1)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv16_c_frag_cg0_6[0])
                            : "r"(inv16_c_addr_cg0_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv16_dc_acc_cg0_7[0]), "=f"(inv16_dc_acc_cg0_7[1]), "=f"(inv16_dc_acc_cg0_7[2]), "=f"(inv16_dc_acc_cg0_7[3])
                            : "r"(inv16_d_frag_cg0_5[0]), "r"(inv16_d_frag_cg0_5[1]), "r"(inv16_c_frag_cg0_6[0]));
                        #pragma unroll
                        for (int inv16_i_cg0_1 = 0; inv16_i_cg0_1 < 4; inv16_i_cg0_1++) {
                            inv16_dc_acc_cg0_7[inv16_i_cg0_1] = inv16_dc_acc_cg0_7[inv16_i_cg0_1] * -1.0f;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_dc_acc_cg0_7[_lp*2 + 0], inv16_dc_acc_cg0_7[_lp*2+1 + 0]));
                            inv16_dc_bf16_cg0_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(inv16_a_frag_cg0_9[0])
                            : "r"(inv16_a_addr_cg0_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv16_o_acc_cg0_10[0]), "=f"(inv16_o_acc_cg0_10[1]), "=f"(inv16_o_acc_cg0_10[2]), "=f"(inv16_o_acc_cg0_10[3])
                            : "r"(inv16_dc_bf16_cg0_8[0]), "r"(inv16_dc_bf16_cg0_8[1]), "r"(inv16_a_frag_cg0_9[0]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_o_acc_cg0_10[_lp*2 + 0], inv16_o_acc_cg0_10[_lp*2+1 + 0]));
                            inv16_o_bf16_cg0_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>((unsigned long long)inv16_o_addr_cg0_4);
                        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&inv16_o_bf16_cg0_11[0]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (inverse_row_cg0 < 64) {
                        int inv32_tile_base_cg0 = inverse_local_warp_cg0 * 32;
                        int inv32_lane_row_cg0 = lane % 16;
                        int inv32_lane_col_cg0 = lane / 16 * 8;
                        int inv32_d_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + 16 + inv32_lane_col_cg0) * 2 ^ ((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + 16 + inv32_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv32_c_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 ^ ((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv32_a_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((inv32_tile_base_cg0 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 ^ ((inv32_tile_base_cg0 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv32_o_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 ^ ((inv32_tile_base_cg0 + 16 + inv32_lane_row_cg0) * 128 + (inv32_tile_base_cg0 + inv32_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        unsigned int inv32_d_frag_cg0[4];
                        unsigned int inv32_c_frag_cg0[4];
                        float inv32_dc_acc_cg0[8];
                        unsigned int inv32_dc_bf16_cg0[4];
                        unsigned int inv32_a_frag_cg0[4];
                        float inv32_o_acc_cg0[8];
                        unsigned int inv32_o_bf16_cg0[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv32_d_frag_cg0[0]), "=r"(inv32_d_frag_cg0[1]), "=r"(inv32_d_frag_cg0[2]), "=r"(inv32_d_frag_cg0[3])
                            : "r"(inv32_d_addr_cg0)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv32_c_frag_cg0[0]), "=r"(inv32_c_frag_cg0[1]), "=r"(inv32_c_frag_cg0[2]), "=r"(inv32_c_frag_cg0[3])
                            : "r"(inv32_c_addr_cg0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv32_dc_acc_cg0[0]), "=f"(inv32_dc_acc_cg0[1]), "=f"(inv32_dc_acc_cg0[2]), "=f"(inv32_dc_acc_cg0[3])
                            : "r"(inv32_d_frag_cg0[0]), "r"(inv32_d_frag_cg0[1]), "r"(inv32_d_frag_cg0[2]), "r"(inv32_d_frag_cg0[3]), "r"(inv32_c_frag_cg0[0]), "r"(inv32_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv32_dc_acc_cg0[4]), "=f"(inv32_dc_acc_cg0[(4) + 1]), "=f"(inv32_dc_acc_cg0[(4) + 2]), "=f"(inv32_dc_acc_cg0[(4) + 3])
                            : "r"(inv32_d_frag_cg0[0]), "r"(inv32_d_frag_cg0[1]), "r"(inv32_d_frag_cg0[2]), "r"(inv32_d_frag_cg0[3]), "r"(inv32_c_frag_cg0[2]), "r"(inv32_c_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int inv32_i_cg0 = 0; inv32_i_cg0 < 8; inv32_i_cg0++) {
                            inv32_dc_acc_cg0[inv32_i_cg0] = inv32_dc_acc_cg0[inv32_i_cg0] * -1.0f;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv32_dc_acc_cg0[_lp*2 + 0], inv32_dc_acc_cg0[_lp*2+1 + 0]));
                            inv32_dc_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv32_a_frag_cg0[0]), "=r"(inv32_a_frag_cg0[1]), "=r"(inv32_a_frag_cg0[2]), "=r"(inv32_a_frag_cg0[3])
                            : "r"(inv32_a_addr_cg0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv32_o_acc_cg0[0]), "=f"(inv32_o_acc_cg0[1]), "=f"(inv32_o_acc_cg0[2]), "=f"(inv32_o_acc_cg0[3])
                            : "r"(inv32_dc_bf16_cg0[0]), "r"(inv32_dc_bf16_cg0[1]), "r"(inv32_dc_bf16_cg0[2]), "r"(inv32_dc_bf16_cg0[3]), "r"(inv32_a_frag_cg0[0]), "r"(inv32_a_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                            : "=f"(inv32_o_acc_cg0[4]), "=f"(inv32_o_acc_cg0[(4) + 1]), "=f"(inv32_o_acc_cg0[(4) + 2]), "=f"(inv32_o_acc_cg0[(4) + 3])
                            : "r"(inv32_dc_bf16_cg0[0]), "r"(inv32_dc_bf16_cg0[1]), "r"(inv32_dc_bf16_cg0[2]), "r"(inv32_dc_bf16_cg0[3]), "r"(inv32_a_frag_cg0[2]), "r"(inv32_a_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv32_o_acc_cg0[_lp*2 + 0], inv32_o_acc_cg0[_lp*2+1 + 0]));
                            inv32_o_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_12 = static_cast<uint32_t>((unsigned long long)inv32_o_addr_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[3]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    int inv64_warp_y_cg0 = inverse_local_warp_cg0;
                    int inv64_lane_row_cg0 = lane % 16;
                    int inv64_lane_col_cg0 = lane / 16 * 8;
                    unsigned int inv64_o_bf16_0_cg0[4];
                    unsigned int inv64_o_bf16_1_cg0[4];
                    if (inverse_row_cg0 < 64) {
                        unsigned int inv64_d_frag_cg0[4];
                        unsigned int inv64_c_frag_cg0[4];
                        float inv64_dc_acc_cg0[8];
                        unsigned int inv64_dc_bf16_0_cg0[4];
                        unsigned int inv64_dc_bf16_1_cg0[4];
                        #pragma unroll
                        for (int inv64_zero_i_cg0 = 0; inv64_zero_i_cg0 < 8; inv64_zero_i_cg0++) {
                            inv64_dc_acc_cg0[inv64_zero_i_cg0] = 0.0f;
                        }
                        int inv64_d_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((32 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        int inv64_d_addr_cg0_0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_1 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((48 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((48 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_0)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int inv64_i_cg0 = 0; inv64_i_cg0 < 8; inv64_i_cg0++) {
                            inv64_dc_acc_cg0[inv64_i_cg0] = inv64_dc_acc_cg0[inv64_i_cg0] * -1.0f;
                        }
                        {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_dc_acc_cg0[_lp*2 + 0], inv64_dc_acc_cg0[_lp*2+1 + 0]));
                                inv64_dc_bf16_0_cg0[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int inv64_zero_i_cg0_1 = 0; inv64_zero_i_cg0_1 < 8; inv64_zero_i_cg0_1++) {
                            inv64_dc_acc_cg0[inv64_zero_i_cg0_1] = 0.0f;
                        }
                        int inv64_d_addr_cg0_2 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_3 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_2)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        int inv64_d_addr_cg0_4 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_5 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((48 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((48 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_4)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0_5)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int inv64_i_cg0_1 = 0; inv64_i_cg0_1 < 8; inv64_i_cg0_1++) {
                            inv64_dc_acc_cg0[inv64_i_cg0_1] = inv64_dc_acc_cg0[inv64_i_cg0_1] * -1.0f;
                        }
                        {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_dc_acc_cg0[_lp*2 + 0], inv64_dc_acc_cg0[_lp*2+1 + 0]));
                                inv64_dc_bf16_1_cg0[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        unsigned int inv64_a_frag_cg0[4];
                        float inv64_o_acc_cg0[8];
                        #pragma unroll
                        for (int inv64_zero_i_cg0_2 = 0; inv64_zero_i_cg0_2 < 8; inv64_zero_i_cg0_2++) {
                            inv64_o_acc_cg0[inv64_zero_i_cg0_2] = 0.0f;
                        }
                        int inv64_a_addr_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)(inv64_lane_row_cg0 * 128 + inv64_lane_col_cg0 * 2 ^ (inv64_lane_row_cg0 * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0)
                            : "memory");
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                                : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                                : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        }
                        int inv64_a_addr_cg0_6 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_6)
                            : "memory");
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                                : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                                : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        }
                        {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_o_acc_cg0[_lp*2 + 0], inv64_o_acc_cg0[_lp*2+1 + 0]));
                                inv64_o_bf16_0_cg0[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int inv64_zero_i_cg0_3 = 0; inv64_zero_i_cg0_3 < 8; inv64_zero_i_cg0_3++) {
                            inv64_o_acc_cg0[inv64_zero_i_cg0_3] = 0.0f;
                        }
                        int inv64_a_addr_cg0_7 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)(inv64_lane_row_cg0 * 128 + (16 + inv64_lane_col_cg0) * 2 ^ (inv64_lane_row_cg0 * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_7)
                            : "memory");
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                                : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                                : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        }
                        int inv64_a_addr_cg0_8 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_8)
                            : "memory");
                        {
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                                : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                                : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        }
                        {
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_o_acc_cg0[_lp*2 + 0], inv64_o_acc_cg0[_lp*2+1 + 0]));
                                inv64_o_bf16_1_cg0[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (inverse_row_cg0 < 64) {
                        int inv64_o_addr_0_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        int inv64_o_addr_1_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)inv64_o_addr_0_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[3]))
                            : "memory");
                        uint32_t _stmatrix_addr_14 = static_cast<uint32_t>((unsigned long long)inv64_o_addr_1_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[3]))
                            : "memory");
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (warp_id_in_role_cg0 < 4) {
                        unsigned int ainv_beta_ld_bits_cg0[4];
                        int ainv_beta_tile_row_cg0 = warp_id_in_role_cg0 * 16 + (lane & 7) + (lane & 8);
                        int ainv_beta_tile_col_lane_cg0 = (lane & 16) / 2;
                        int ainv_beta_pair_col_cg0 = (lane & 3) * 2;
                        #pragma unroll 1
                        for (int beta_col_tile_cg0 = 0; beta_col_tile_cg0 < 4; beta_col_tile_cg0++) {
                            int ainv_beta_tile_col_cg0 = beta_col_tile_cg0 * 16 + ainv_beta_tile_col_lane_cg0;
                            int ainv_beta_ld_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 ^ (ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 >> 7 & 7) << 4));
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(ainv_beta_ld_bits_cg0[0]), "=r"(ainv_beta_ld_bits_cg0[1]), "=r"(ainv_beta_ld_bits_cg0[2]), "=r"(ainv_beta_ld_bits_cg0[3])
                                : "r"(ainv_beta_ld_addr_cg0)
                                : "memory");
                            int ainv_beta_col0_cg0 = beta_col_tile_cg0 * 16 + ainv_beta_pair_col_cg0;
                            int ainv_beta_col8_cg0 = ainv_beta_col0_cg0 + 8;
                            float ainv_beta_scale0_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0];
                            float ainv_beta_scale0_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0 + 1];
                            float ainv_beta_scale8_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0];
                            float ainv_beta_scale8_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0 + 1];
                            uint32_t _bf16x2_scale_0;
                            {
                                uint32_t _bf16x2_pair_15 = ainv_beta_ld_bits_cg0[0];
                                float _bf16x2_lo_15;
                                float _bf16x2_hi_15;
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_15) : "h"((uint16_t)(_bf16x2_pair_15 & 0xFFFFu)));
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_15) : "h"((uint16_t)(_bf16x2_pair_15 >> 16)));
                                _bf16x2_lo_15 *= ainv_beta_scale0_lo_cg0;
                                _bf16x2_hi_15 *= ainv_beta_scale0_hi_cg0;
                                uint32_t _bf16x2_out_15;
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_15) : "f"(_bf16x2_hi_15), "f"(_bf16x2_lo_15));
                                _bf16x2_scale_0 = _bf16x2_out_15;
                            }
                            uint32_t _bf16x2_scale_1;
                            {
                                uint32_t _bf16x2_pair_16 = ainv_beta_ld_bits_cg0[1];
                                float _bf16x2_lo_16;
                                float _bf16x2_hi_16;
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_16) : "h"((uint16_t)(_bf16x2_pair_16 & 0xFFFFu)));
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_16) : "h"((uint16_t)(_bf16x2_pair_16 >> 16)));
                                _bf16x2_lo_16 *= ainv_beta_scale0_lo_cg0;
                                _bf16x2_hi_16 *= ainv_beta_scale0_hi_cg0;
                                uint32_t _bf16x2_out_16;
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_16) : "f"(_bf16x2_hi_16), "f"(_bf16x2_lo_16));
                                _bf16x2_scale_1 = _bf16x2_out_16;
                            }
                            uint32_t _bf16x2_scale_2;
                            {
                                uint32_t _bf16x2_pair_17 = ainv_beta_ld_bits_cg0[2];
                                float _bf16x2_lo_17;
                                float _bf16x2_hi_17;
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_17) : "h"((uint16_t)(_bf16x2_pair_17 & 0xFFFFu)));
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_17) : "h"((uint16_t)(_bf16x2_pair_17 >> 16)));
                                _bf16x2_lo_17 *= ainv_beta_scale8_lo_cg0;
                                _bf16x2_hi_17 *= ainv_beta_scale8_hi_cg0;
                                uint32_t _bf16x2_out_17;
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_17) : "f"(_bf16x2_hi_17), "f"(_bf16x2_lo_17));
                                _bf16x2_scale_2 = _bf16x2_out_17;
                            }
                            uint32_t _bf16x2_scale_3;
                            {
                                uint32_t _bf16x2_pair_18 = ainv_beta_ld_bits_cg0[3];
                                float _bf16x2_lo_18;
                                float _bf16x2_hi_18;
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_18) : "h"((uint16_t)(_bf16x2_pair_18 & 0xFFFFu)));
                                asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_18) : "h"((uint16_t)(_bf16x2_pair_18 >> 16)));
                                _bf16x2_lo_18 *= ainv_beta_scale8_lo_cg0;
                                _bf16x2_hi_18 *= ainv_beta_scale8_hi_cg0;
                                uint32_t _bf16x2_out_18;
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_18) : "f"(_bf16x2_hi_18), "f"(_bf16x2_lo_18));
                                _bf16x2_scale_3 = _bf16x2_out_18;
                            }
                            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>((unsigned long long)(smem_ainv_addr + ainv0_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 ^ (ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_0)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_1)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_2)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_3))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(ainv_ready_addr + (ainv0_stage) * 8);
                    mbarrier_arrive(beta_smem_empty_addr + (beta0_stage) * 8);
                    {
                        if (warp_id_in_role_cg0 < 4) {
                            unsigned int ainv_beta_ld_bits_cg0_1[4];
                            int ainv_beta_tile_row_cg0_1 = warp_id_in_role_cg0 * 16 + (lane & 7) + (lane & 8);
                            int ainv_beta_tile_col_lane_cg0_1 = (lane & 16) / 2;
                            int ainv_beta_pair_col_cg0_1 = (lane & 3) * 2;
                            #pragma unroll 1
                            for (int beta_col_tile_cg0_1 = 0; beta_col_tile_cg0_1 < 4; beta_col_tile_cg0_1++) {
                                int ainv_beta_tile_col_cg0_1 = beta_col_tile_cg0_1 * 16 + ainv_beta_tile_col_lane_cg0_1;
                                int ainv_beta_ld_addr_cg0_1 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 ^ (ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 >> 7 & 7) << 4));
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(ainv_beta_ld_bits_cg0_1[0]), "=r"(ainv_beta_ld_bits_cg0_1[1]), "=r"(ainv_beta_ld_bits_cg0_1[2]), "=r"(ainv_beta_ld_bits_cg0_1[3])
                                    : "r"(ainv_beta_ld_addr_cg0_1)
                                    : "memory");
                                int ainv_beta_col0_cg0_1 = beta_col_tile_cg0_1 * 16 + ainv_beta_pair_col_cg0_1;
                                int ainv_beta_col8_cg0_1 = ainv_beta_col0_cg0_1 + 8;
                                float ainv_beta_scale0_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1];
                                float ainv_beta_scale0_hi_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1 + 1];
                                float ainv_beta_scale8_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1];
                                float ainv_beta_scale8_hi_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1 + 1];
                                uint32_t _bf16x2_scale_4;
                                {
                                    uint32_t _bf16x2_pair_20 = ainv_beta_ld_bits_cg0_1[0];
                                    float _bf16x2_lo_20;
                                    float _bf16x2_hi_20;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_20) : "h"((uint16_t)(_bf16x2_pair_20 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_20) : "h"((uint16_t)(_bf16x2_pair_20 >> 16)));
                                    _bf16x2_lo_20 *= ainv_beta_scale0_lo_cg0_1;
                                    _bf16x2_hi_20 *= ainv_beta_scale0_hi_cg0_1;
                                    uint32_t _bf16x2_out_20;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_20) : "f"(_bf16x2_hi_20), "f"(_bf16x2_lo_20));
                                    _bf16x2_scale_4 = _bf16x2_out_20;
                                }
                                uint32_t _bf16x2_scale_5;
                                {
                                    uint32_t _bf16x2_pair_21 = ainv_beta_ld_bits_cg0_1[1];
                                    float _bf16x2_lo_21;
                                    float _bf16x2_hi_21;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_21) : "h"((uint16_t)(_bf16x2_pair_21 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_21) : "h"((uint16_t)(_bf16x2_pair_21 >> 16)));
                                    _bf16x2_lo_21 *= ainv_beta_scale0_lo_cg0_1;
                                    _bf16x2_hi_21 *= ainv_beta_scale0_hi_cg0_1;
                                    uint32_t _bf16x2_out_21;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_21) : "f"(_bf16x2_hi_21), "f"(_bf16x2_lo_21));
                                    _bf16x2_scale_5 = _bf16x2_out_21;
                                }
                                uint32_t _bf16x2_scale_6;
                                {
                                    uint32_t _bf16x2_pair_22 = ainv_beta_ld_bits_cg0_1[2];
                                    float _bf16x2_lo_22;
                                    float _bf16x2_hi_22;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_22) : "h"((uint16_t)(_bf16x2_pair_22 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_22) : "h"((uint16_t)(_bf16x2_pair_22 >> 16)));
                                    _bf16x2_lo_22 *= ainv_beta_scale8_lo_cg0_1;
                                    _bf16x2_hi_22 *= ainv_beta_scale8_hi_cg0_1;
                                    uint32_t _bf16x2_out_22;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_22) : "f"(_bf16x2_hi_22), "f"(_bf16x2_lo_22));
                                    _bf16x2_scale_6 = _bf16x2_out_22;
                                }
                                uint32_t _bf16x2_scale_7;
                                {
                                    uint32_t _bf16x2_pair_23 = ainv_beta_ld_bits_cg0_1[3];
                                    float _bf16x2_lo_23;
                                    float _bf16x2_hi_23;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_23) : "h"((uint16_t)(_bf16x2_pair_23 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_23) : "h"((uint16_t)(_bf16x2_pair_23 >> 16)));
                                    _bf16x2_lo_23 *= ainv_beta_scale8_lo_cg0_1;
                                    _bf16x2_hi_23 *= ainv_beta_scale8_hi_cg0_1;
                                    uint32_t _bf16x2_out_23;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_23) : "f"(_bf16x2_hi_23), "f"(_bf16x2_lo_23));
                                    _bf16x2_scale_7 = _bf16x2_out_23;
                                }
                                uint32_t _stmatrix_addr_24 = static_cast<uint32_t>((unsigned long long)(smem_ainv_addr + ainv1_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 ^ (ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_4)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_5)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_6)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_7))
                                    : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(ainv_ready_addr + (ainv1_stage) * 8);
                        mbarrier_arrive(beta_smem_empty_addr + (beta1_stage) * 8);
                    }
                    unsigned int qk0_stage = qk_cg0_stage;
                    mbarrier_wait(qk_smem_empty_addr + (qk0_stage) * 8, qk_cg0_phase);
                    qk_cg0_stage += 1;
                    if (qk_cg0_stage == 2) { qk_cg0_stage = 0; qk_cg0_phase ^= 1; }
                    unsigned int qk1_stage = ((0) ? (qk0_stage + 1) % 2 : qk_cg0_stage);
                    {
                        mbarrier_wait(qk_smem_empty_addr + (qk1_stage) * 8, qk_cg0_phase);
                        qk_cg0_stage += 1;
                        if (qk_cg0_stage == 2) { qk_cg0_stage = 0; qk_cg0_phase ^= 1; }
                    }
                    mbarrier_wait(cg0_shared_acc_full_addr, 1);
                    int qk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                        : "r"(qk_addr)
                        : "memory");
                    int qk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                    int qk_stsm_col_lane_cg0 = (lane & 16) / 2;
                    #pragma unroll
                    for (int qk_repeat_cg0_4 = 0; qk_repeat_cg0_4 < 4; qk_repeat_cg0_4++) {
                        const int qk_j0_cg0_2 = qk_repeat_cg0_4 * 8;
                        float _t3[8];
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(_t3)[_ls] = mul_f32x2(reinterpret_cast<float2*>((_tmem_load_3 + qk_j0_cg0_2))[_ls], reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0_2))[_ls]);
                        const float2 _scale2_25 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_t3)[_ls], _scale2_25);
                        uint32_t _t3_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t3[_lp*2 + 0], _t3[_lp*2+1 + 0]));
                            _t3_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_26 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk0_stage * 8192 + (unsigned int)(qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_4 * 16 + qk_stsm_col_lane_cg0) * 2 ^ (qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_4 * 16 + qk_stsm_col_lane_cg0) * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t3_bf16[3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(cg0_shared_acc_empty_addr);
                    mbarrier_arrive(qk_ready_addr + (qk0_stage) * 8);
                    {
                        mbarrier_wait(cg0_shared_acc_full_addr + 8, 1);
                        int qk_addr_0 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                            : "r"(qk_addr_0)
                            : "memory");
                        int qk_stsm_row_cg0_1 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                        int qk_stsm_col_lane_cg0_2 = (lane & 16) / 2;
                        #pragma unroll
                        for (int qk_repeat_cg0_5 = 0; qk_repeat_cg0_5 < 4; qk_repeat_cg0_5++) {
                            const int qk_j0_cg0_3 = qk_repeat_cg0_5 * 8;
                            float _t4[8];
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                reinterpret_cast<float2*>(_t4)[_ls] = mul_f32x2(reinterpret_cast<float2*>((_tmem_load_4 + qk_j0_cg0_3))[_ls], reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_3))[_ls]);
                            const float2 _scale2_27 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_t4)[_ls], _scale2_27);
                            uint32_t _t4_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t4[_lp*2 + 0], _t4[_lp*2+1 + 0]));
                                _t4_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t _stmatrix_addr_28 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk1_stage * 8192 + (unsigned int)(qk_stsm_row_cg0_1 * 128 + (qk_repeat_cg0_5 * 16 + qk_stsm_col_lane_cg0_2) * 2 ^ (qk_stsm_row_cg0_1 * 128 + (qk_repeat_cg0_5 * 16 + qk_stsm_col_lane_cg0_2) * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&_t4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t4_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t4_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t4_bf16[3]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                        mbarrier_arrive(qk_ready_addr + (qk1_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: compute_group_1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        { // compute_group_1_main
            unsigned int gate_cg1_stage = 0;
            unsigned int gate_cg1_phase = 0;
            unsigned int v_cg1_stage = 0;
            unsigned int v_cg1_phase = 0;
            unsigned int o_cg1_stage = 0;
            unsigned int o_cg1_phase = 1;
            unsigned int _phase_initial_state_loaded_0 = 0;
            unsigned int _phase_kv_acc_full_0 = 0;
            unsigned int _phase_cg1_shared_acc_full_0 = 0;
            unsigned int _phase_q_state_acc_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int num_o_heads_1 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx_1 = tile_1 / 2;
                int value_split_idx_1 = tile_1 % 2;
                int batch_idx_1 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_1 >> NUM_O_HEADS_LOG2 : base_tile_idx_1 / num_o_heads_1);
                int head_idx_1 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_1 & num_o_heads_1 - 1 : base_tile_idx_1 % num_o_heads_1);
                int qk_head_idx_1 = ((IS_GQA != 0) ? head_idx_1 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_v_heads / num_q_heads)));
                int v_head_idx_1 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_q_heads / num_v_heads)) : head_idx_1);
                int batch_start_1 = cu_seqlens[batch_idx_1];
                int batch_end_1 = cu_seqlens[batch_idx_1 + 1];
                int seqlen_b_1 = batch_end_1 - batch_start_1;
                int num_pairs_b_1 = (seqlen_b_1 + 128 - 1) / 128;
                int num_chunks_b_1 = num_pairs_b_1 * 2;
                int state_slot_cg1 = batch_idx_1;
                {
                    state_slot_cg1 = state_indices[batch_idx_1];
                }
                long long state_head_offset_cg1 = (long long)head_idx_1 * 16384;
                long long initial_state_head_base_cg1 = (long long)state_slot_cg1 * initial_state_stride_slot + state_head_offset_cg1;
                long long output_state_head_base_cg1 = (long long)state_slot_cg1 * output_state_stride_slot + state_head_offset_cg1;
                if (USE_INITIAL_STATE != 0 && num_chunks_b_1 > 0) {
                    int warp_in_wg = warp % 4;
                    int state_tmem_row_base_init = warp_in_wg * 32 << 16;
                    int warp_id_in_role_1 = (warp - 4);
                    int state_warp_init = warp_id_in_role_1;
                    int state_row_top_init = state_warp_init * 16 + lane / 4;
                    int state_row_bot_init = state_row_top_init + 8;
                    long long state_base_top_init = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_top_init) * 128;
                    long long state_base_bot_init = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_bot_init) * 128;
                    if (state_warp_init < 4) {
                        #pragma unroll
                        for (int state_col_half_init = 0; state_col_half_init < 2; state_col_half_init++) {
                            float state_init_frag[32];
                            #pragma unroll
                            for (int state_col_group_init = 0; state_col_group_init < 8; state_col_group_init++) {
                                int state_col_pair_init = state_col_half_init * 64 + state_col_group_init * 8 + (lane & 3) * 2;
                                const int state_reg_base_init = state_col_group_init * 4;
                                {
                                    {
                                        uint32_t _bf16x2_bits_0;
                                        _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(initial_state + state_base_top_init + (long long)state_col_pair_init);
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&state_init_frag[state_reg_base_init])[0]), "=f"((&state_init_frag[state_reg_base_init])[1])
                                            : "r"(_bf16x2_bits_0));
                                    }
                                    {
                                        uint32_t _bf16x2_bits_1;
                                        _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(initial_state + state_base_bot_init + (long long)state_col_pair_init);
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&state_init_frag[state_reg_base_init + 2])[0]), "=f"((&state_init_frag[state_reg_base_init + 2])[1])
                                            : "r"(_bf16x2_bits_1));
                                    }
                                }
                            }
                            int state_init_addr = taddr + (unsigned int)state_tmem_row_base_init + (unsigned int)(state_col_half_init * 64);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_init_addr), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[3])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[4])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[5])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[6])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[7])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[8])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[9])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[10])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[11])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[12])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[13])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[14])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[15])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[16])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[17])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[18])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[19])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[20])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[21])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[22])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[23])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[24])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[25])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[26])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[27])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[28])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[29])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[30])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[31]))
                                : "memory");
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(initial_state_loaded_addr);
                    }
                    mbarrier_wait(initial_state_loaded_addr, _phase_initial_state_loaded_0);
                    _phase_initial_state_loaded_0 ^= 1;
                    if (state_warp_init == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(kv_acc_full_addr);
                        }
                    }
                }
                int active_chunks_b = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_1 > 0)) != 0) ? 1 : num_chunks_b_1);
                if (active_chunks_b > 0) {
                    {
                        int chunk_offset_1 = batch_start_1;
                        int _cg1_marker = batch_idx_1 + head_idx_1 + chunk_offset_1 + batch_end_1 + checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE + ENABLE_CHECKPOINTS;
                        mbarrier_wait(load_gate_full_addr + (gate_cg1_stage) * 8, gate_cg1_phase);
                        int gate_cg1_elem_base = gate_cg1_stage * 64;
                        int warp_in_wg_1 = warp % 4;
                        int tmem_row_base_v = warp_in_wg_1 * 32 << 16;
                        int warp_id_in_role_2 = (warp - 4);
                        int warp_id_in_role_cg1 = warp_id_in_role_2;
                        int lane_quad_cg1 = lane & 3;
                        float chunk_decay_cg1 = smem_cumprod[gate_cg1_elem_base + 64 - 1];
                        {
                            mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                            _phase_kv_acc_full_0 ^= 1;
                            if (warp_id_in_role_cg1 < 4) {
                                int state_addr_0 = taddr + (unsigned int)tmem_row_base_v;
                                int state_addr_1 = state_addr_0 + 64;
                                float _tmem_load_5[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                                    : "r"(state_addr_0)
                                    : "memory");
                                float _tmem_load_6[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                                    : "r"(state_addr_1)
                                    : "memory");
                                int state_inp_addr_0 = taddr + 192 + (unsigned int)tmem_row_base_v;
                                int state_inp_addr_1 = state_inp_addr_0 + 32;
                                uint32_t _tmem_load_5_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(state_inp_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5_bf16[15]))
                                    : "memory");
                                uint32_t _tmem_load_6_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(state_inp_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6_bf16[15]))
                                    : "memory");
                                const float2 _scale2_2 = {chunk_decay_cg1, chunk_decay_cg1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_2);
                                const float2 _scale2_3 = {chunk_decay_cg1, chunk_decay_cg1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_3);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                    :: "r"(state_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_5[31]))
                                    : "memory");
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                    :: "r"(state_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_6[31]))
                                    : "memory");
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            mbarrier_arrive(state_inp_ready_addr);
                            mbarrier_arrive(kv_acc_empty_addr);
                        }
                        float cg1_cumprod_frag[32];
                        float cg1_decay_scale_frag[32];
                        float last_cumsumlog_cg1 = smem_cumsumlog[gate_cg1_elem_base + 64 - 1];
                        #pragma unroll
                        for (int cg1_scale_j = 0; cg1_scale_j < 32; cg1_scale_j++) {
                            int cg1_scale_repeat = cg1_scale_j / 4;
                            int cg1_scale_reg = cg1_scale_j & 3;
                            int cg1_scale_col = cg1_scale_repeat * 8 + lane_quad_cg1 * 2 + (cg1_scale_reg & 1);
                            float cg1_scale_log = smem_cumsumlog[gate_cg1_elem_base + cg1_scale_col];
                            cg1_cumprod_frag[cg1_scale_j] = smem_cumprod[gate_cg1_elem_base + cg1_scale_col];
                            float _exp2_2 = approx_exp2(last_cumsumlog_cg1 - cg1_scale_log);
                            cg1_decay_scale_frag[cg1_scale_j] = _exp2_2;
                        }
                        mbarrier_arrive(gate_cg1_empty_addr + (gate_cg1_stage) * 8);
                        gate_cg1_stage += 1;
                        if (gate_cg1_stage == 5) { gate_cg1_stage = 0; gate_cg1_phase ^= 1; }
                        mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, v_cg1_phase);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            int v_stage_addr_cg1 = smem_v_mma_addr + v_cg1_stage * 16384;
                            float v_frag_cg1[32];
                            if (warp_id_in_role_cg1 < 4) {
                                unsigned int v_ld_bits_cg1[2];
                                #pragma unroll
                                for (int v_frag_repeat_cg1 = 0; v_frag_repeat_cg1 < 8; v_frag_repeat_cg1++) {
                                    int v_ld_mtx_cg1 = lane / 8 & 1;
                                    int v_ld_token_cg1 = v_frag_repeat_cg1 * 8 + (lane & 7);
                                    int v_ld_dv_cg1 = warp_id_in_role_cg1 * 16 + v_ld_mtx_cg1 * 8;
                                    int v_t0_cg1 = v_ld_token_cg1 & 15;
                                    int v_t1_cg1 = v_ld_token_cg1 / 16;
                                    int v_ld_linear_byte_off_cg1 = (v_ld_dv_cg1 + v_t0_cg1 * 64 + v_t1_cg1 * 1024) * 2;
                                    int v_ld_byte_off_cg1 = v_ld_linear_byte_off_cg1 ^ (v_ld_linear_byte_off_cg1 >> 7 & 7) << 4;
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                        : "=r"(v_ld_bits_cg1[0]), "=r"(v_ld_bits_cg1[1])
                                        : "r"(v_stage_addr_cg1 + v_ld_byte_off_cg1)
                                        : "memory");
                                    float v_ld_bits_cg1_f32[4];
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 2; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            "shl.b32 %0, %2, 16;\n\t"
                                            "and.b32 %1, %2, 0xffff0000;\n\t"
                                            "}\n"
                                            : "=f"((&v_ld_bits_cg1_f32[_pair * 2])[0]), "=f"((&v_ld_bits_cg1_f32[_pair * 2])[1])
                                            : "r"(v_ld_bits_cg1[_pair]));
                                    }
                                    const int v_frag_j0_cg1 = v_frag_repeat_cg1 * 4;
                                    #pragma unroll
                                    for (int v_frag_sub_cg1 = 0; v_frag_sub_cg1 < 4; v_frag_sub_cg1++) {
                                        v_frag_cg1[v_frag_j0_cg1 + v_frag_sub_cg1] = v_ld_bits_cg1_f32[v_frag_sub_cg1];
                                    }
                                }
                            }
                            mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                            _phase_cg1_shared_acc_full_0 ^= 1;
                            float ks_frag[32];
                            if (warp_id_in_role_cg1 < 4) {
                                int ks_addr = taddr + 384 + (unsigned int)tmem_row_base_v;
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[3])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[4])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[5])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[6])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[7])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[8])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[9])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[10])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[11])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[12])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[13])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[14])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[15])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[16])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[17])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[18])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[19])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[20])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[21])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[22])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[23])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[24])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[25])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[26])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[27])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[28])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[29])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[30])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag[31]))
                                    : "r"(ks_addr)
                                    : "memory");
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(ks_frag)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                            }
                            mbarrier_arrive(cg1_shared_acc_empty_addr);
                            if (warp_id_in_role_cg1 < 4) {
                                float _t5[32];
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    reinterpret_cast<float2*>(_t5)[_ls] = sub_f32x2(reinterpret_cast<float2*>(v_frag_cg1)[_ls], reinterpret_cast<const float2*>(ks_frag)[_ls]);
                                uint32_t _t5_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t5[_lp*2 + 0], _t5[_lp*2+1 + 0]));
                                    _t5_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int vks_addr = taddr + 448 + (unsigned int)tmem_row_base_v;
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(vks_addr), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_t5_bf16[15]))
                                    : "memory");
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        mbarrier_arrive(vks_ready_addr);
                        {
                            mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                            _phase_q_state_acc_full_0 ^= 1;
                            if (warp_id_in_role_cg1 < 4) {
                                int qs_addr = taddr + 128 + (unsigned int)tmem_row_base_v;
                                float _tmem_load_7[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                                    : "r"(qs_addr)
                                    : "memory");
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                                const float2 _scale2_4 = {scale, scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_4);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                    :: "r"(qs_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_7[31]))
                                    : "memory");
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            mbarrier_arrive(q_state_acc_empty_addr);
                        }
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        if (elect_sync()) {
                            mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                        }
                        v_cg1_stage += 1;
                        if (v_cg1_stage == 3) { v_cg1_stage = 0; v_cg1_phase ^= 1; }
                        float nv_frag_cg1[32];
                        unsigned int nv_packed_cg1[16];
                        if (warp_id_in_role_cg1 < 4) {
                            int nv_src_addr_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[31]))
                                : "r"(nv_src_addr_cg1)
                                : "memory");
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1[_lp*2 + 0], nv_frag_cg1[_lp*2+1 + 0]));
                                nv_packed_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        if (warp_id_in_role_cg1 < 4) {
                            int nv_dst_addr_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(nv_dst_addr_cg1), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[15]))
                                : "memory");
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            mbarrier_arrive(nv_ready_addr);
                            {
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(nv_frag_cg1)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag)[_ls]);
                                uint32_t nv_frag_cg1_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1[_lp*2 + 0], nv_frag_cg1[_lp*2+1 + 0]));
                                    nv_frag_cg1_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int decay_dst_addr_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v;
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(decay_dst_addr_cg1), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16[15]))
                                    : "memory");
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                mbarrier_arrive(decay_v_ready_addr);
                            }
                        } else {
                            mbarrier_arrive(nv_ready_addr);
                            {
                                mbarrier_arrive(decay_v_ready_addr);
                            }
                        }
                        mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, o_cg1_phase);
                        int o_stage_addr_cg1 = smem_o_addr + o_cg1_stage * 16384;
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        if (warp_id_in_role_cg1 < 4) {
                            int q_state_addr = taddr + 128 + (unsigned int)tmem_row_base_v;
                            float _tmem_load_8[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[31]))
                                : "r"(q_state_addr)
                                : "memory");
                            uint32_t _tmem_load_8_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int token_group_cg1 = 0; token_group_cg1 < 4; token_group_cg1++) {
                                int o_mtx_idx_cg1 = lane / 8;
                                int o_row_addr_cg1 = lane & 7;
                                int o_dim_base_cg1 = warp_id_in_role_cg1 * 16 + (o_mtx_idx_cg1 & 1) * 8;
                                int o_token_base_cg1 = token_group_cg1 * 16 + o_mtx_idx_cg1 / 2 * 8;
                                int o_token_addr_cg1 = o_token_base_cg1 + o_row_addr_cg1;
                                int o_token_pair_cg1 = o_token_addr_cg1 / 2;
                                int o_token_parity_cg1 = o_token_addr_cg1 & 1;
                                int o_raw_row_cg1 = o_token_pair_cg1;
                                int o_raw_col_cg1 = (o_dim_base_cg1 & 63 ^ (o_token_pair_cg1 & 3) << 4 ^ o_token_parity_cg1 << 3) + o_token_parity_cg1 * 64;
                                int o_stsm_offset_cg1 = (o_raw_row_cg1 * 128 + o_raw_col_cg1) * 2;
                                const int o_pack_base_cg1 = token_group_cg1 * 4;
                                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1 + o_stsm_offset_cg1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[o_pack_base_cg1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[o_pack_base_cg1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[o_pack_base_cg1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_8_bf16[o_pack_base_cg1 + 3]))
                                    : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(q_state_acc_empty_addr);
                        mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                        o_cg1_stage += 1;
                        if (o_cg1_stage == 2) { o_cg1_stage = 0; o_cg1_phase ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_1 = 1; chunk_idx_1 < active_chunks_b; chunk_idx_1++) {
                    int chunk_offset_2 = batch_start_1 + chunk_idx_1 * 64;
                    int _cg1_marker_1 = batch_idx_1 + head_idx_1 + chunk_offset_2 + batch_end_1 + checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE + ENABLE_CHECKPOINTS;
                    mbarrier_wait(load_gate_full_addr + (gate_cg1_stage) * 8, gate_cg1_phase);
                    int gate_cg1_elem_base_1 = gate_cg1_stage * 64;
                    int warp_in_wg_2 = warp % 4;
                    int tmem_row_base_v_1 = warp_in_wg_2 * 32 << 16;
                    int warp_id_in_role_3 = (warp - 4);
                    int warp_id_in_role_cg1_1 = warp_id_in_role_3;
                    int lane_quad_cg1_1 = lane & 3;
                    float chunk_decay_cg1_1 = smem_cumprod[gate_cg1_elem_base_1 + 64 - 1];
                    {
                        mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                        _phase_kv_acc_full_0 ^= 1;
                        if (warp_id_in_role_cg1_1 < 4) {
                            int state_addr_0_1 = taddr + (unsigned int)tmem_row_base_v_1;
                            int state_addr_1_1 = state_addr_0_1 + 64;
                            float _tmem_load_13[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[31]))
                                : "r"(state_addr_0_1)
                                : "memory");
                            float _tmem_load_14[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[31]))
                                : "r"(state_addr_1_1)
                                : "memory");
                            int state_inp_addr_0_1 = taddr + 192 + (unsigned int)tmem_row_base_v_1;
                            int state_inp_addr_1_1 = state_inp_addr_0_1 + 32;
                            uint32_t _tmem_load_13_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                                _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(state_inp_addr_0_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13_bf16[15]))
                                : "memory");
                            uint32_t _tmem_load_14_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                                _tmem_load_14_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(state_inp_addr_1_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14_bf16[15]))
                                : "memory");
                            const float2 _scale2_6 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_6);
                            const float2 _scale2_7 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_14)[_ls], _scale2_7);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_addr_0_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_13[31]))
                                : "memory");
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_addr_1_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_14[31]))
                                : "memory");
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(state_inp_ready_addr);
                        mbarrier_arrive(kv_acc_empty_addr);
                    }
                    float cg1_cumprod_frag_1[32];
                    float cg1_decay_scale_frag_1[32];
                    float last_cumsumlog_cg1_1 = smem_cumsumlog[gate_cg1_elem_base_1 + 64 - 1];
                    #pragma unroll
                    for (int cg1_scale_j_1 = 0; cg1_scale_j_1 < 32; cg1_scale_j_1++) {
                        int cg1_scale_repeat_1 = cg1_scale_j_1 / 4;
                        int cg1_scale_reg_1 = cg1_scale_j_1 & 3;
                        int cg1_scale_col_1 = cg1_scale_repeat_1 * 8 + lane_quad_cg1_1 * 2 + (cg1_scale_reg_1 & 1);
                        float cg1_scale_log_1 = smem_cumsumlog[gate_cg1_elem_base_1 + cg1_scale_col_1];
                        cg1_cumprod_frag_1[cg1_scale_j_1] = smem_cumprod[gate_cg1_elem_base_1 + cg1_scale_col_1];
                        float _exp2_4 = approx_exp2(last_cumsumlog_cg1_1 - cg1_scale_log_1);
                        cg1_decay_scale_frag_1[cg1_scale_j_1] = _exp2_4;
                    }
                    mbarrier_arrive(gate_cg1_empty_addr + (gate_cg1_stage) * 8);
                    gate_cg1_stage += 1;
                    if (gate_cg1_stage == 5) { gate_cg1_stage = 0; gate_cg1_phase ^= 1; }
                    mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, v_cg1_phase);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        int v_stage_addr_cg1_1 = smem_v_mma_addr + v_cg1_stage * 16384;
                        float v_frag_cg1_1[32];
                        if (warp_id_in_role_cg1_1 < 4) {
                            unsigned int v_ld_bits_cg1_1[2];
                            #pragma unroll
                            for (int v_frag_repeat_cg1_1 = 0; v_frag_repeat_cg1_1 < 8; v_frag_repeat_cg1_1++) {
                                int v_ld_mtx_cg1_1 = lane / 8 & 1;
                                int v_ld_token_cg1_1 = v_frag_repeat_cg1_1 * 8 + (lane & 7);
                                int v_ld_dv_cg1_1 = warp_id_in_role_cg1_1 * 16 + v_ld_mtx_cg1_1 * 8;
                                int v_t0_cg1_1 = v_ld_token_cg1_1 & 15;
                                int v_t1_cg1_1 = v_ld_token_cg1_1 / 16;
                                int v_ld_linear_byte_off_cg1_1 = (v_ld_dv_cg1_1 + v_t0_cg1_1 * 64 + v_t1_cg1_1 * 1024) * 2;
                                int v_ld_byte_off_cg1_1 = v_ld_linear_byte_off_cg1_1 ^ (v_ld_linear_byte_off_cg1_1 >> 7 & 7) << 4;
                                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                    : "=r"(v_ld_bits_cg1_1[0]), "=r"(v_ld_bits_cg1_1[1])
                                    : "r"(v_stage_addr_cg1_1 + v_ld_byte_off_cg1_1)
                                    : "memory");
                                float v_ld_bits_cg1_f32_1[4];
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&v_ld_bits_cg1_f32_1[_pair * 2])[0]), "=f"((&v_ld_bits_cg1_f32_1[_pair * 2])[1])
                                        : "r"(v_ld_bits_cg1_1[_pair]));
                                }
                                const int v_frag_j0_cg1_1 = v_frag_repeat_cg1_1 * 4;
                                #pragma unroll
                                for (int v_frag_sub_cg1_1 = 0; v_frag_sub_cg1_1 < 4; v_frag_sub_cg1_1++) {
                                    v_frag_cg1_1[v_frag_j0_cg1_1 + v_frag_sub_cg1_1] = v_ld_bits_cg1_f32_1[v_frag_sub_cg1_1];
                                }
                            }
                        }
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        float ks_frag_1[32];
                        if (warp_id_in_role_cg1_1 < 4) {
                            int ks_addr_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_1[31]))
                                : "r"(ks_addr_1)
                                : "memory");
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(ks_frag_1)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                        }
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        if (warp_id_in_role_cg1_1 < 4) {
                            float _t7[32];
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>(_t7)[_ls] = sub_f32x2(reinterpret_cast<float2*>(v_frag_cg1_1)[_ls], reinterpret_cast<const float2*>(ks_frag_1)[_ls]);
                            uint32_t _t7_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_t7[_lp*2 + 0], _t7[_lp*2+1 + 0]));
                                _t7_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int vks_addr_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(vks_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_t7_bf16[15]))
                                : "memory");
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                    }
                    mbarrier_arrive(vks_ready_addr);
                    {
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        if (warp_id_in_role_cg1_1 < 4) {
                            int qs_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1;
                            float _tmem_load_15[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[31]))
                                : "r"(qs_addr_1)
                                : "memory");
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_15)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                            const float2 _scale2_8 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_15)[_ls], _scale2_8);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(qs_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_15[31]))
                                : "memory");
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(q_state_acc_empty_addr);
                    }
                    mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                    _phase_cg1_shared_acc_full_0 ^= 1;
                    if (elect_sync()) {
                        mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                    }
                    v_cg1_stage += 1;
                    if (v_cg1_stage == 3) { v_cg1_stage = 0; v_cg1_phase ^= 1; }
                    float nv_frag_cg1_1[32];
                    unsigned int nv_packed_cg1_1[16];
                    if (warp_id_in_role_cg1_1 < 4) {
                        int nv_src_addr_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[31]))
                            : "r"(nv_src_addr_cg1_1)
                            : "memory");
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1_1[_lp*2 + 0], nv_frag_cg1_1[_lp*2+1 + 0]));
                            nv_packed_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                    }
                    mbarrier_arrive(cg1_shared_acc_empty_addr);
                    if (warp_id_in_role_cg1_1 < 4) {
                        int nv_dst_addr_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(nv_dst_addr_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[15]))
                            : "memory");
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(nv_ready_addr);
                        {
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(nv_frag_cg1_1)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag_1)[_ls]);
                            uint32_t nv_frag_cg1_bf16_1[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1_1[_lp*2 + 0], nv_frag_cg1_1[_lp*2+1 + 0]));
                                nv_frag_cg1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            int decay_dst_addr_cg1_1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(decay_dst_addr_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_frag_cg1_bf16_1[15]))
                                : "memory");
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            mbarrier_arrive(decay_v_ready_addr);
                        }
                    } else {
                        mbarrier_arrive(nv_ready_addr);
                        {
                            mbarrier_arrive(decay_v_ready_addr);
                        }
                    }
                    mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, o_cg1_phase);
                    int o_stage_addr_cg1_1 = smem_o_addr + o_cg1_stage * 16384;
                    mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                    _phase_q_state_acc_full_0 ^= 1;
                    if (warp_id_in_role_cg1_1 < 4) {
                        int q_state_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1;
                        float _tmem_load_16[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[31]))
                            : "r"(q_state_addr_1)
                            : "memory");
                        uint32_t _tmem_load_16_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 0], _tmem_load_16[_lp*2+1 + 0]));
                            _tmem_load_16_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int token_group_cg1_1 = 0; token_group_cg1_1 < 4; token_group_cg1_1++) {
                            int o_mtx_idx_cg1_1 = lane / 8;
                            int o_row_addr_cg1_1 = lane & 7;
                            int o_dim_base_cg1_1 = warp_id_in_role_cg1_1 * 16 + (o_mtx_idx_cg1_1 & 1) * 8;
                            int o_token_base_cg1_1 = token_group_cg1_1 * 16 + o_mtx_idx_cg1_1 / 2 * 8;
                            int o_token_addr_cg1_1 = o_token_base_cg1_1 + o_row_addr_cg1_1;
                            int o_token_pair_cg1_1 = o_token_addr_cg1_1 / 2;
                            int o_token_parity_cg1_1 = o_token_addr_cg1_1 & 1;
                            int o_raw_row_cg1_1 = o_token_pair_cg1_1;
                            int o_raw_col_cg1_1 = (o_dim_base_cg1_1 & 63 ^ (o_token_pair_cg1_1 & 3) << 4 ^ o_token_parity_cg1_1 << 3) + o_token_parity_cg1_1 * 64;
                            int o_stsm_offset_cg1_1 = (o_raw_row_cg1_1 * 128 + o_raw_col_cg1_1) * 2;
                            const int o_pack_base_cg1_1 = token_group_cg1_1 * 4;
                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1_1 + o_stsm_offset_cg1_1));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_16_bf16[o_pack_base_cg1_1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_16_bf16[o_pack_base_cg1_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_16_bf16[o_pack_base_cg1_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_16_bf16[o_pack_base_cg1_1 + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_state_acc_empty_addr);
                    mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                    o_cg1_stage += 1;
                    if (o_cg1_stage == 2) { o_cg1_stage = 0; o_cg1_phase ^= 1; }
                }
                if ((STORE_FINAL_STATE != 0 || ENABLE_CHECKPOINTS != 0) && num_chunks_b_1 > 0) {
                    mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                    _phase_kv_acc_full_0 ^= 1;
                    int warp_in_wg_3 = warp % 4;
                    int state_tmem_row_base_cg1 = warp_in_wg_3 * 32 << 16;
                    int warp_id_in_role_4 = (warp - 4);
                    int state_warp_id_in_role_cg1 = warp_id_in_role_4;
                    int state_row_top_cg1 = state_warp_id_in_role_cg1 * 16 + lane / 4;
                    int state_row_bot_cg1 = state_row_top_cg1 + 8;
                    long long state_base_idx_top_cg1 = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_top_cg1) * 128;
                    long long state_base_idx_bot_cg1 = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_bot_cg1) * 128;
                    if (state_warp_id_in_role_cg1 < 4) {
                        #pragma unroll
                        for (int state_col_half_cg1 = 0; state_col_half_cg1 < 2; state_col_half_cg1++) {
                            int final_state_addr_cg1 = taddr + (unsigned int)state_tmem_row_base_cg1 + (unsigned int)(state_col_half_cg1 * 64);
                            float _tmem_load_17[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[31]))
                                : "r"(final_state_addr_cg1)
                                : "memory");
                            #pragma unroll
                            for (int state_col_group_cg1 = 0; state_col_group_cg1 < 8; state_col_group_cg1++) {
                                int state_col_pair_cg1 = state_col_half_cg1 * 64 + state_col_group_cg1 * 8 + (lane & 3) * 2;
                                const int state_reg_base_cg1 = state_col_group_cg1 * 4;
                                {
                                    *(reinterpret_cast<__nv_bfloat16*>(output_state + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1)) + (0)) = __float2bfloat16_rn(_tmem_load_17[state_reg_base_cg1]);
                                    *(reinterpret_cast<__nv_bfloat16*>(output_state + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1 + 1)) + (0)) = __float2bfloat16_rn(_tmem_load_17[state_reg_base_cg1 + 1]);
                                    *(reinterpret_cast<__nv_bfloat16*>(output_state + (state_base_idx_bot_cg1 + (long long)state_col_pair_cg1)) + (0)) = __float2bfloat16_rn(_tmem_load_17[state_reg_base_cg1 + 2]);
                                    *(reinterpret_cast<__nv_bfloat16*>(output_state + (state_base_idx_bot_cg1 + (long long)state_col_pair_cg1 + 1)) + (0)) = __float2bfloat16_rn(_tmem_load_17[state_reg_base_cg1 + 3]);
                                }
                            }
                        }
                    }
                    mbarrier_arrive(kv_acc_empty_addr);
                }
                if (STORE_FINAL_STATE != 0 && USE_INITIAL_STATE != 0 && num_chunks_b_1 == 0) {
                    int warp_id_in_role_5 = (warp - 4);
                    int empty_state_warp = warp_id_in_role_5;
                    int empty_state_row_top = empty_state_warp * 16 + lane / 4;
                    int empty_state_row_bot = empty_state_row_top + 8;
                    int empty_state_col_lane = (lane & 3) * 4;
                    if (empty_state_warp < 4) {
                        #pragma unroll
                        for (int empty_col_group = 0; empty_col_group < 8; empty_col_group++) {
                            int empty_state_col = empty_col_group * 16 + empty_state_col_lane;
                            long long empty_initial_top_base = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_top) * 128;
                            long long empty_initial_bot_base = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_bot) * 128;
                            long long empty_output_top_base = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_top) * 128;
                            long long empty_output_bot_base = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_bot) * 128;
                            float _vec_load_0[4];
                            {
                                uint2 _vld_10;
                                _vld_10 = *reinterpret_cast<const uint2*>(initial_state + empty_initial_top_base + (long long)empty_state_col);
                                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _pair * 2])[1])
                                        : "r"(_vpairs_10[_pair]));
                                }
                            }
                            float _vec_load_1[4];
                            {
                                uint2 _vld_11;
                                _vld_11 = *reinterpret_cast<const uint2*>(initial_state + empty_initial_bot_base + (long long)empty_state_col);
                                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _pair * 2])[1])
                                        : "r"(_vpairs_11[_pair]));
                                }
                            }
                            {
                                uint2 _pk2;
                                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(output_state + (empty_output_top_base + (long long)empty_state_col)))[0]) = _pk2;
                            }
                            {
                                uint2 _pk2;
                                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                _pk[0] = __floats2bfloat162_rn(_vec_load_1[0 + 0], _vec_load_1[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_vec_load_1[0 + 2], _vec_load_1[0 + 3]);
                                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(output_state + (empty_output_bot_base + (long long)empty_state_col)))[0]) = _pk2;
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Q)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(O)) : "memory");
            unsigned int k_cg0_stage = 0;
            unsigned int k_cg0_phase = 0;
            unsigned int q_cg0_stage = 0;
            unsigned int q_cg0_phase = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < total_tiles; tile_2 += num_bids) {
                int num_o_heads_2 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx_2 = tile_2 / 2;
                int value_split_idx_2 = tile_2 % 2;
                int batch_idx_2 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_2 >> NUM_O_HEADS_LOG2 : base_tile_idx_2 / num_o_heads_2);
                int head_idx_2 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_2 & num_o_heads_2 - 1 : base_tile_idx_2 % num_o_heads_2);
                int qk_head_idx_2 = ((IS_GQA != 0) ? head_idx_2 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_v_heads / num_q_heads)));
                int v_head_idx_2 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_q_heads / num_v_heads)) : head_idx_2);
                int batch_start_2 = cu_seqlens[batch_idx_2];
                int batch_end_2 = cu_seqlens[batch_idx_2 + 1];
                int seqlen_b_2 = batch_end_2 - batch_start_2;
                int num_pairs_b_2 = (seqlen_b_2 + 128 - 1) / 128;
                int num_chunks_b_2 = num_pairs_b_2 * 2;
                int num_pairs_b_0 = num_chunks_b_2 / 2;
                #pragma unroll 1
                for (int _pair_idx = 0; _pair_idx < num_pairs_b_0; _pair_idx++) {
                    unsigned int k0_stage = k_cg0_stage;
                    mbarrier_wait(load_k_full_addr + (k0_stage) * 8, k_cg0_phase);
                    k_cg0_stage += 1;
                    if (k_cg0_stage == 4) { k_cg0_stage = 0; k_cg0_phase ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr, 1);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k0_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k0_stage) * 1024);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem_cg0_shared_acc), "r"(0));
                    elect_commit(cg0_shared_acc_full_addr);
                    unsigned int k1_stage = ((0) ? (k0_stage + 1) % 4 : k_cg0_stage);
                    {
                        mbarrier_wait(load_k_full_addr + (k1_stage) * 8, k_cg0_phase);
                        k_cg0_stage += 1;
                        if (k_cg0_stage == 4) { k_cg0_stage = 0; k_cg0_phase ^= 1; }
                        mbarrier_wait(cg0_shared_acc_empty_addr + 8, 1);
                        int _mma_a_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k1_stage) * 1024);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k1_stage) * 1024);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_cg0_shared_acc + (64))), "r"(0));
                        elect_commit(cg0_shared_acc_full_addr + 8);
                    }
                    unsigned int q0_stage = q_cg0_stage;
                    mbarrier_wait(load_q_full_addr + (q0_stage) * 8, q_cg0_phase);
                    q_cg0_stage += 1;
                    if (q_cg0_stage == 2) { q_cg0_stage = 0; q_cg0_phase ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr, 0);
                    int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q0_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k0_stage) * 1024);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_tmem_cg0_shared_acc), "r"(0));
                    elect_commit(cg0_shared_acc_full_addr);
                    unsigned int q1_stage = ((0) ? (q0_stage + 1) % 2 : q_cg0_stage);
                    {
                        mbarrier_wait(load_q_full_addr + (q1_stage) * 8, q_cg0_phase);
                        q_cg0_stage += 1;
                        if (q_cg0_stage == 2) { q_cg0_stage = 0; q_cg0_phase ^= 1; }
                        mbarrier_wait(cg0_shared_acc_empty_addr + 8, 0);
                        int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q1_stage) * 1024);
                        int _mma_b_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k1_stage) * 1024);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_cg0_shared_acc + (64))), "r"(0));
                        elect_commit(cg0_shared_acc_full_addr + 8);
                    }
                    elect_commit(load_k_empty_addr + (k0_stage) * 8);
                    {
                        elect_commit(load_k_empty_addr + (k1_stage) * 8);
                    }
                    elect_commit(q_smem_empty_addr + (q0_stage) * 8);
                    {
                        elect_commit(q_smem_empty_addr + (q1_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: tma_qkv ----
    if (warp == 9) {
        { // tma_qkv_main
            int cta_slot_base = bid * 512;
            if (elect_sync()) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(Q);
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + cta_slot_base));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(K);
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base + 128)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(V);
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base + 256)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            unsigned int k_stage = 0;
            unsigned int k_empty_phase_tma = 1;
            unsigned int q_stage = 0;
            unsigned int q_empty_phase_tma = 1;
            unsigned int v_stage = 0;
            unsigned int v_empty_phase_tma = 1;
            #pragma unroll 1
            for (unsigned int tile_3 = bid; tile_3 < total_tiles; tile_3 += num_bids) {
                int num_o_heads_3 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx_3 = tile_3 / 2;
                int value_split_idx_3 = tile_3 % 2;
                int batch_idx_3 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_3 >> NUM_O_HEADS_LOG2 : base_tile_idx_3 / num_o_heads_3);
                int head_idx_3 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_3 & num_o_heads_3 - 1 : base_tile_idx_3 % num_o_heads_3);
                int qk_head_idx_3 = ((IS_GQA != 0) ? head_idx_3 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_v_heads / num_q_heads)));
                int v_head_idx_3 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_q_heads / num_v_heads)) : head_idx_3);
                int batch_start_3 = cu_seqlens[batch_idx_3];
                int batch_end_3 = cu_seqlens[batch_idx_3 + 1];
                int seqlen_b_3 = batch_end_3 - batch_start_3;
                int num_pairs_b_3 = (seqlen_b_3 + 128 - 1) / 128;
                int num_chunks_b_3 = num_pairs_b_3 * 2;
                int k_head_idx = ((num_q_heads >= num_v_heads) ? v_head_idx_3 : qk_head_idx_3);
                int value_split_offset = value_split_idx_3 * 64;
                if (elect_sync()) {
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base)), "r"((uint32_t)(batch_end_3)) : "memory");
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 128))), "r"((uint32_t)(batch_end_3)) : "memory");
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 256))), "r"((uint32_t)(batch_end_3)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 128))) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 256))) : "memory");
                }
                int num_pairs_tma = num_chunks_b_3 / 2;
                #pragma unroll 1
                for (int pair_idx_tma = 0; pair_idx_tma < num_pairs_tma; pair_idx_tma++) {
                    #pragma unroll
                    for (int chunk_in_pair = 0; chunk_in_pair < ((0) ? 1 : 2); chunk_in_pair++) {
                        int chunk_idx_2 = pair_idx_tma * 2 + chunk_in_pair;
                        int chunk_offset_3 = batch_start_3 + chunk_idx_2 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(load_k_empty_addr + (k_stage) * 8, k_empty_phase_tma);
                            mbarrier_arrive_expect_tx(load_k_full_addr + (k_stage) * 8, 16384);
                            #pragma unroll
                            for (int dim_half = 0; dim_half < 2; dim_half++) {
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + (unsigned int)(dim_half * 8192), tensormap_workspace + (cta_slot_base + 128), dim_half * 64, chunk_offset_3, k_head_idx, load_k_full_addr + (k_stage) * 8);
                            }
                        }
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; k_empty_phase_tma ^= 1; }
                    }
                    #pragma unroll
                    for (int chunk_in_pair_1 = 0; chunk_in_pair_1 < ((0) ? 1 : 2); chunk_in_pair_1++) {
                        int chunk_idx_3 = pair_idx_tma * 2 + chunk_in_pair_1;
                        int chunk_offset_4 = batch_start_3 + chunk_idx_3 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(q_smem_empty_addr + (q_stage) * 8, q_empty_phase_tma);
                            mbarrier_arrive_expect_tx(load_q_full_addr + (q_stage) * 8, 16384);
                            #pragma unroll
                            for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                tma_3d_gmem2smem(smem_q_addr + q_stage * 16384 + (unsigned int)(dim_half_1 * 8192), tensormap_workspace + cta_slot_base, dim_half_1 * 64, chunk_offset_4, qk_head_idx_3, load_q_full_addr + (q_stage) * 8);
                            }
                        }
                        q_stage += 1;
                        if (q_stage == 2) { q_stage = 0; q_empty_phase_tma ^= 1; }
                    }
                    #pragma unroll
                    for (int chunk_in_pair_2 = 0; chunk_in_pair_2 < ((0) ? 1 : 2); chunk_in_pair_2++) {
                        int chunk_idx_4 = pair_idx_tma * 2 + chunk_in_pair_2;
                        int chunk_offset_5 = batch_start_3 + chunk_idx_4 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(v_smem_empty_addr + (v_stage) * 8, v_empty_phase_tma);
                            mbarrier_arrive_expect_tx(load_v_full_addr + (v_stage) * 8, 8192);
                            tma_3d_gmem2smem(smem_v_mma_addr + v_stage * 16384, tensormap_workspace + (cta_slot_base + 256), value_split_offset, chunk_offset_5, v_head_idx_3, load_v_full_addr + (v_stage) * 8);
                        }
                        v_stage += 1;
                        if (v_stage == 3) { v_stage = 0; v_empty_phase_tma ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma_cg1 ----
    if (warp == 10) {
        { // mma_cg1_main
            unsigned int k_stage_1 = 0;
            unsigned int k_phase_mma = 0;
            unsigned int q_stage_1 = 0;
            unsigned int q_phase_mma = 0;
            unsigned int v_stage_mma = 0;
            unsigned int v_phase_mma = 0;
            unsigned int ainv_stage_mma = 0;
            unsigned int ainv_phase_mma = 0;
            unsigned int qk_stage_mma = 0;
            unsigned int qk_phase_mma = 0;
            unsigned int q_state_acc_mma_stage = 0;
            unsigned int q_state_acc_mma_phase = 1;
            unsigned int kv_acc_mma_stage = 0;
            unsigned int kv_acc_mma_phase = 1;
            unsigned int _phase_state_inp_ready_0 = 0;
            unsigned int _phase_cg1_shared_acc_empty_0 = 1;
            unsigned int _phase_vks_ready_0 = 0;
            unsigned int _phase_nv_ready_0 = 0;
            unsigned int _phase_decay_v_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_4 = bid; tile_4 < total_tiles; tile_4 += num_bids) {
                int num_o_heads_4 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx_4 = tile_4 / 2;
                int value_split_idx_4 = tile_4 % 2;
                int batch_idx_4 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_4 >> NUM_O_HEADS_LOG2 : base_tile_idx_4 / num_o_heads_4);
                int head_idx_4 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_4 & num_o_heads_4 - 1 : base_tile_idx_4 % num_o_heads_4);
                int qk_head_idx_4 = ((IS_GQA != 0) ? head_idx_4 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_v_heads / num_q_heads)));
                int v_head_idx_4 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_q_heads / num_v_heads)) : head_idx_4);
                int batch_start_4 = cu_seqlens[batch_idx_4];
                int batch_end_4 = cu_seqlens[batch_idx_4 + 1];
                int seqlen_b_4 = batch_end_4 - batch_start_4;
                int num_pairs_b_4 = (seqlen_b_4 + 128 - 1) / 128;
                int num_chunks_b_4 = num_pairs_b_4 * 2;
                int active_chunks_b_1 = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_4 > 0)) != 0) ? 1 : num_chunks_b_4);
                if (active_chunks_b_1 > 0) {
                    {
                        kv_acc_mma_stage += 1;
                        if (kv_acc_mma_stage == 1) { kv_acc_mma_stage = 0; kv_acc_mma_phase ^= 1; }
                        int chunk_offset_6 = batch_start_4;
                        int _mma_marker = batch_idx_4 + head_idx_4 + chunk_offset_6 + batch_end_4 + 512;
                        {
                            mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, k_phase_mma);
                        }
                        mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, q_phase_mma);
                        {
                            mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                            _phase_state_inp_ready_0 ^= 1;
                            mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                            _phase_cg1_shared_acc_empty_0 ^= 1;
                            int _mma_b_lo_4 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(_mma_b_lo_4), "r"(tmem_tmem_state_inp), "r"(0));
                            elect_commit(cg1_shared_acc_full_addr);
                            mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8, q_state_acc_mma_phase);
                            int _mma_b_lo_5 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_5), "r"(tmem_tmem_state_inp), "r"(0));
                            elect_commit(q_state_acc_full_addr);
                            q_state_acc_mma_stage += 1;
                            if (q_state_acc_mma_stage == 1) { q_state_acc_mma_stage = 0; q_state_acc_mma_phase ^= 1; }
                        }
                        if (elect_sync()) {
                            {
                                tcgen05_commit(q_smem_empty_addr + (q_stage_1) * 8);
                            }
                        }
                        mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
                        _phase_vks_ready_0 ^= 1;
                        mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, ainv_phase_mma);
                        mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                        _phase_cg1_shared_acc_empty_0 ^= 1;
                        {
                            int _mma_b_lo_6 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(_mma_b_lo_6), "r"(tmem_tmem_shared_inp), "r"(0));
                        }
                        elect_commit(cg1_shared_acc_full_addr);
                        elect_commit(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
                        ainv_stage_mma += 1;
                        if (ainv_stage_mma == 3) { ainv_stage_mma = 0; ainv_phase_mma ^= 1; }
                        mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, qk_phase_mma);
                        mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
                        _phase_nv_ready_0 ^= 1;
                        mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8, q_state_acc_mma_phase);
                        int _mma_b_lo_8 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_8), "r"(tmem_tmem_shared_inp), "r"(((!1) ? 0 : 1)));
                        elect_commit(q_state_acc_full_addr);
                        q_state_acc_mma_stage += 1;
                        if (q_state_acc_mma_stage == 1) { q_state_acc_mma_stage = 0; q_state_acc_mma_phase ^= 1; }
                        elect_commit(qk_smem_empty_addr + (qk_stage_mma) * 8);
                        qk_stage_mma += 1;
                        if (qk_stage_mma == 2) { qk_stage_mma = 0; qk_phase_mma ^= 1; }
                        {
                            mbarrier_wait(kv_acc_empty_addr + (kv_acc_mma_stage) * 8, kv_acc_mma_phase);
                            mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                            _phase_decay_v_ready_0 ^= 1;
                            int _mma_b_lo_9 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (k_stage_1) * 1024);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state), "r"(_mma_b_lo_9), "r"(tmem_tmem_shared_inp + 32), "r"(((!1) ? 0 : 1)));
                            elect_commit(kv_acc_full_addr);
                            elect_commit(load_k_empty_addr + (k_stage_1) * 8);
                            kv_acc_mma_stage += 1;
                            if (kv_acc_mma_stage == 1) { kv_acc_mma_stage = 0; kv_acc_mma_phase ^= 1; }
                        }
                        k_stage_1 += 1;
                        if (k_stage_1 == 4) { k_stage_1 = 0; k_phase_mma ^= 1; }
                        q_stage_1 += 1;
                        if (q_stage_1 == 2) { q_stage_1 = 0; q_phase_mma ^= 1; }
                        v_stage_mma += 1;
                        if (v_stage_mma == 3) { v_stage_mma = 0; v_phase_mma ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_5 = 1; chunk_idx_5 < active_chunks_b_1; chunk_idx_5++) {
                    int chunk_offset_7 = batch_start_4 + chunk_idx_5 * 64;
                    int _mma_marker_1 = batch_idx_4 + head_idx_4 + chunk_offset_7 + batch_end_4 + 512;
                    {
                        mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, k_phase_mma);
                    }
                    mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, q_phase_mma);
                    {
                        mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                        _phase_state_inp_ready_0 ^= 1;
                        mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                        _phase_cg1_shared_acc_empty_0 ^= 1;
                        int _mma_b_lo_16 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(_mma_b_lo_16), "r"(tmem_tmem_state_inp), "r"(0));
                        elect_commit(cg1_shared_acc_full_addr);
                        mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8, q_state_acc_mma_phase);
                        int _mma_b_lo_17 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 56], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_17), "r"(tmem_tmem_state_inp), "r"(0));
                        elect_commit(q_state_acc_full_addr);
                        q_state_acc_mma_stage += 1;
                        if (q_state_acc_mma_stage == 1) { q_state_acc_mma_stage = 0; q_state_acc_mma_phase ^= 1; }
                    }
                    if (elect_sync()) {
                        {
                            tcgen05_commit(q_smem_empty_addr + (q_stage_1) * 8);
                        }
                    }
                    mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
                    _phase_vks_ready_0 ^= 1;
                    mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, ainv_phase_mma);
                    mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                    _phase_cg1_shared_acc_empty_0 ^= 1;
                    {
                        int _mma_b_lo_18 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(_mma_b_lo_18), "r"(tmem_tmem_shared_inp), "r"(0));
                    }
                    elect_commit(cg1_shared_acc_full_addr);
                    elect_commit(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
                    ainv_stage_mma += 1;
                    if (ainv_stage_mma == 3) { ainv_stage_mma = 0; ainv_phase_mma ^= 1; }
                    mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, qk_phase_mma);
                    mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
                    _phase_nv_ready_0 ^= 1;
                    mbarrier_wait(q_state_acc_empty_addr + (q_state_acc_mma_stage) * 8, q_state_acc_mma_phase);
                    int _mma_b_lo_20 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_20), "r"(tmem_tmem_shared_inp), "r"(((!1) ? 0 : 1)));
                    elect_commit(q_state_acc_full_addr);
                    q_state_acc_mma_stage += 1;
                    if (q_state_acc_mma_stage == 1) { q_state_acc_mma_stage = 0; q_state_acc_mma_phase ^= 1; }
                    elect_commit(qk_smem_empty_addr + (qk_stage_mma) * 8);
                    qk_stage_mma += 1;
                    if (qk_stage_mma == 2) { qk_stage_mma = 0; qk_phase_mma ^= 1; }
                    {
                        mbarrier_wait(kv_acc_empty_addr + (kv_acc_mma_stage) * 8, kv_acc_mma_phase);
                        mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                        _phase_decay_v_ready_0 ^= 1;
                        int _mma_b_lo_21 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (k_stage_1) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2], db, id, p0;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%2 + 24], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state), "r"(_mma_b_lo_21), "r"(tmem_tmem_shared_inp + 32), "r"(((!1) ? 0 : 1)));
                        elect_commit(kv_acc_full_addr);
                        elect_commit(load_k_empty_addr + (k_stage_1) * 8);
                        kv_acc_mma_stage += 1;
                        if (kv_acc_mma_stage == 1) { kv_acc_mma_stage = 0; kv_acc_mma_phase ^= 1; }
                    }
                    k_stage_1 += 1;
                    if (k_stage_1 == 4) { k_stage_1 = 0; k_phase_mma ^= 1; }
                    q_stage_1 += 1;
                    if (q_stage_1 == 2) { q_stage_1 = 0; q_phase_mma ^= 1; }
                    v_stage_mma += 1;
                    if (v_stage_mma == 3) { v_stage_mma = 0; v_phase_mma ^= 1; }
                }
            }
        }
    }
    // ---- Role: output_epilogue ----
    if (warp == 11) {
        { // output_epilogue_main
            int cta_slot_base_epi = bid * 512;
            unsigned int gate_prod_stage = 0;
            unsigned int gate_prod_phase = 1;
            unsigned int o_epi_stage = 0;
            unsigned int o_epi_phase = 0;
            unsigned int o_release_stage = 0;
            unsigned int o_release_phase = 1;
            if (lane == 0) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>(O);
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            #pragma unroll 1
            for (unsigned int tile_5 = bid; tile_5 < total_tiles; tile_5 += num_bids) {
                int num_o_heads_5 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int base_tile_idx_5 = tile_5 / 2;
                int value_split_idx_5 = tile_5 % 2;
                int batch_idx_5 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_5 >> NUM_O_HEADS_LOG2 : base_tile_idx_5 / num_o_heads_5);
                int head_idx_5 = ((NUM_O_HEADS_LOG2 >= 0) ? base_tile_idx_5 & num_o_heads_5 - 1 : base_tile_idx_5 % num_o_heads_5);
                int qk_head_idx_5 = ((IS_GQA != 0) ? head_idx_5 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_v_heads / num_q_heads)));
                int v_head_idx_5 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_q_heads / num_v_heads)) : head_idx_5);
                int batch_start_5 = cu_seqlens[batch_idx_5];
                int batch_end_5 = cu_seqlens[batch_idx_5 + 1];
                int seqlen_b_5 = batch_end_5 - batch_start_5;
                int num_pairs_b_5 = (seqlen_b_5 + 128 - 1) / 128;
                int num_chunks_b_5 = num_pairs_b_5 * 2;
                int value_split_offset_1 = value_split_idx_5 * 64;
                if (lane == 0) {
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))), "r"((uint32_t)(batch_end_5)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                }
                int num_valid_chunks_b = (batch_end_5 - batch_start_5 + 64 - 1) / 64;
                if (num_chunks_b_5 > 0) {
                    #pragma unroll
                    for (int prefetch_idx = 0; prefetch_idx < ((0) ? 1 : 2); prefetch_idx++) {
                        int prefetch_offset = batch_start_5 + prefetch_idx * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        int gb_lane = lane;
                        int gate_elem_base = gate_prod_stage * 64;
                        float gate_log_0 = 0.0f;
                        float gate_log_1 = 0.0f;
                        int token_0 = prefetch_offset + gb_lane;
                        int token_1 = token_0 + 32;
                        int gate_idx_0 = token_0 * num_o_heads_5 + head_idx_5;
                        int gate_idx_1 = token_1 * num_o_heads_5 + head_idx_5;
                        int beta_idx_0 = gate_idx_0;
                        int beta_idx_1 = gate_idx_1;
                        float gate_val_0 = 1.0f;
                        float gate_val_1 = 1.0f;
                        if (prefetch_idx >= num_valid_chunks_b - 1) {
                            if (token_0 < batch_end_5) {
                                gate_val_0 = gate[gate_idx_0];
                            }
                            if (token_1 < batch_end_5) {
                                gate_val_1 = gate[gate_idx_1];
                            }
                        } else {
                            gate_val_0 = gate[gate_idx_0];
                            gate_val_1 = gate[gate_idx_1];
                        }
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(gate_val_0 + 1e-10f));
                        gate_log_0 = _log2_0;
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(gate_val_1 + 1e-10f));
                        gate_log_1 = _log2_1;
                        float _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 1, 32);
                        if (gb_lane >= 1) {
                            gate_log_0 = gate_log_0 + _shfl_up_0;
                        }
                        float _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 1, 32);
                        if (gb_lane >= 1) {
                            gate_log_1 = gate_log_1 + _shfl_up_1;
                        }
                        float _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 2, 32);
                        if (gb_lane >= 2) {
                            gate_log_0 = gate_log_0 + _shfl_up_2;
                        }
                        float _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 2, 32);
                        if (gb_lane >= 2) {
                            gate_log_1 = gate_log_1 + _shfl_up_3;
                        }
                        float _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 4, 32);
                        if (gb_lane >= 4) {
                            gate_log_0 = gate_log_0 + _shfl_up_4;
                        }
                        float _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 4, 32);
                        if (gb_lane >= 4) {
                            gate_log_1 = gate_log_1 + _shfl_up_5;
                        }
                        float _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 8, 32);
                        if (gb_lane >= 8) {
                            gate_log_0 = gate_log_0 + _shfl_up_6;
                        }
                        float _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 8, 32);
                        if (gb_lane >= 8) {
                            gate_log_1 = gate_log_1 + _shfl_up_7;
                        }
                        float _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 16, 32);
                        if (gb_lane >= 16) {
                            gate_log_0 = gate_log_0 + _shfl_up_8;
                        }
                        float _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 16, 32);
                        if (gb_lane >= 16) {
                            gate_log_1 = gate_log_1 + _shfl_up_9;
                        }
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, gate_log_0, 31);
                        gate_log_1 = gate_log_1 + _shfl_1;
                        smem_cumsumlog[gate_elem_base + gb_lane] = gate_log_0;
                        smem_cumsumlog[gate_elem_base + gb_lane + 32] = gate_log_1;
                        float _exp2_5 = approx_exp2(gate_log_0);
                        smem_cumprod[gate_elem_base + gb_lane] = _exp2_5;
                        float _exp2_6 = approx_exp2(gate_log_1);
                        smem_cumprod[gate_elem_base + gb_lane + 32] = _exp2_6;
                        mbarrier_arrive(load_gate_full_addr + (gate_prod_stage) * 8);
                        int beta_stage_addr = smem_beta_addr + gate_prod_stage * 256;
                        int beta_dst_0 = beta_stage_addr + gb_lane * 4;
                        int beta_dst_1 = beta_stage_addr + (gb_lane + 32) * 4;
                        if (prefetch_idx >= num_valid_chunks_b - 1) {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_0), "l"(beta + beta_idx_0), "r"((token_0 < batch_end_5) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1), "l"(beta + beta_idx_1), "r"((token_1 < batch_end_5) ? 4 : 0));
                        } else {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_0), "l"(beta + beta_idx_0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_1), "l"(beta + beta_idx_1));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(load_beta_full_addr + (gate_prod_stage) * 8) : "memory");
                        gate_prod_stage += 1;
                        if (gate_prod_stage == 5) { gate_prod_stage = 0; gate_prod_phase ^= 1; }
                    }
                    if (num_chunks_b_5 > 2) {
                        #pragma unroll
                        for (int prefetch_idx_1 = 2; prefetch_idx_1 < 4; prefetch_idx_1++) {
                            int prefetch_offset_1 = batch_start_5 + prefetch_idx_1 * 64;
                            mbarrier_wait(gate_cg0_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                            mbarrier_wait(gate_cg1_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                            mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                            int gb_lane_1 = lane;
                            int gate_elem_base_1 = gate_prod_stage * 64;
                            float gate_log_0_1 = 0.0f;
                            float gate_log_1_1 = 0.0f;
                            int token_0_1 = prefetch_offset_1 + gb_lane_1;
                            int token_1_1 = token_0_1 + 32;
                            int gate_idx_0_1 = token_0_1 * num_o_heads_5 + head_idx_5;
                            int gate_idx_1_1 = token_1_1 * num_o_heads_5 + head_idx_5;
                            int beta_idx_0_1 = gate_idx_0_1;
                            int beta_idx_1_1 = gate_idx_1_1;
                            float gate_val_0_1 = 1.0f;
                            float gate_val_1_1 = 1.0f;
                            if (prefetch_idx_1 >= num_valid_chunks_b - 1) {
                                if (token_0_1 < batch_end_5) {
                                    gate_val_0_1 = gate[gate_idx_0_1];
                                }
                                if (token_1_1 < batch_end_5) {
                                    gate_val_1_1 = gate[gate_idx_1_1];
                                }
                            } else {
                                gate_val_0_1 = gate[gate_idx_0_1];
                                gate_val_1_1 = gate[gate_idx_1_1];
                            }
                            float _log2_2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(gate_val_0_1 + 1e-10f));
                            gate_log_0_1 = _log2_2;
                            float _log2_3;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(gate_val_1_1 + 1e-10f));
                            gate_log_1_1 = _log2_3;
                            float _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 1, 32);
                            if (gb_lane_1 >= 1) {
                                gate_log_0_1 = gate_log_0_1 + _shfl_up_10;
                            }
                            float _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 1, 32);
                            if (gb_lane_1 >= 1) {
                                gate_log_1_1 = gate_log_1_1 + _shfl_up_11;
                            }
                            float _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 2, 32);
                            if (gb_lane_1 >= 2) {
                                gate_log_0_1 = gate_log_0_1 + _shfl_up_12;
                            }
                            float _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 2, 32);
                            if (gb_lane_1 >= 2) {
                                gate_log_1_1 = gate_log_1_1 + _shfl_up_13;
                            }
                            float _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 4, 32);
                            if (gb_lane_1 >= 4) {
                                gate_log_0_1 = gate_log_0_1 + _shfl_up_14;
                            }
                            float _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 4, 32);
                            if (gb_lane_1 >= 4) {
                                gate_log_1_1 = gate_log_1_1 + _shfl_up_15;
                            }
                            float _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 8, 32);
                            if (gb_lane_1 >= 8) {
                                gate_log_0_1 = gate_log_0_1 + _shfl_up_16;
                            }
                            float _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 8, 32);
                            if (gb_lane_1 >= 8) {
                                gate_log_1_1 = gate_log_1_1 + _shfl_up_17;
                            }
                            float _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 16, 32);
                            if (gb_lane_1 >= 16) {
                                gate_log_0_1 = gate_log_0_1 + _shfl_up_18;
                            }
                            float _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 16, 32);
                            if (gb_lane_1 >= 16) {
                                gate_log_1_1 = gate_log_1_1 + _shfl_up_19;
                            }
                            float _shfl_2 = __shfl_sync(0xFFFFFFFF, gate_log_0_1, 31);
                            gate_log_1_1 = gate_log_1_1 + _shfl_2;
                            smem_cumsumlog[gate_elem_base_1 + gb_lane_1] = gate_log_0_1;
                            smem_cumsumlog[gate_elem_base_1 + gb_lane_1 + 32] = gate_log_1_1;
                            float _exp2_7 = approx_exp2(gate_log_0_1);
                            smem_cumprod[gate_elem_base_1 + gb_lane_1] = _exp2_7;
                            float _exp2_8 = approx_exp2(gate_log_1_1);
                            smem_cumprod[gate_elem_base_1 + gb_lane_1 + 32] = _exp2_8;
                            mbarrier_arrive(load_gate_full_addr + (gate_prod_stage) * 8);
                            int beta_stage_addr_1 = smem_beta_addr + gate_prod_stage * 256;
                            int beta_dst_0_1 = beta_stage_addr_1 + gb_lane_1 * 4;
                            int beta_dst_1_1 = beta_stage_addr_1 + (gb_lane_1 + 32) * 4;
                            if (prefetch_idx_1 >= num_valid_chunks_b - 1) {
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                    :: "r"(beta_dst_0_1), "l"(beta + beta_idx_0_1), "r"((token_0_1 < batch_end_5) ? 4 : 0));
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                    :: "r"(beta_dst_1_1), "l"(beta + beta_idx_1_1), "r"((token_1_1 < batch_end_5) ? 4 : 0));
                            } else {
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                    :: "r"(beta_dst_0_1), "l"(beta + beta_idx_0_1));
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                    :: "r"(beta_dst_1_1), "l"(beta + beta_idx_1_1));
                            }
                            asm volatile("cp.async.commit_group;");
                            asm volatile(
                                "{\n\t"
                                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                                "}"
                                :: "r"(load_beta_full_addr + (gate_prod_stage) * 8) : "memory");
                            gate_prod_stage += 1;
                            if (gate_prod_stage == 5) { gate_prod_stage = 0; gate_prod_phase ^= 1; }
                        }
                    }
                }
                int active_output_chunks_b = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_5 > 0)) != 0) ? 1 : num_chunks_b_5);
                #pragma unroll 1
                for (int chunk_idx_6 = 0; chunk_idx_6 < active_output_chunks_b; chunk_idx_6++) {
                    int chunk_offset_8 = batch_start_5 + chunk_idx_6 * 64;
                    int prefetch_idx_2 = chunk_idx_6 + 4;
                    if (prefetch_idx_2 < num_chunks_b_5) {
                        int prefetch_offset_2 = batch_start_5 + prefetch_idx_2 * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        mbarrier_wait(beta_smem_empty_addr + (gate_prod_stage) * 8, gate_prod_phase);
                        int gb_lane_2 = lane;
                        int gate_elem_base_2 = gate_prod_stage * 64;
                        float gate_log_0_2 = 0.0f;
                        float gate_log_1_2 = 0.0f;
                        int token_0_2 = prefetch_offset_2 + gb_lane_2;
                        int token_1_2 = token_0_2 + 32;
                        int gate_idx_0_2 = token_0_2 * num_o_heads_5 + head_idx_5;
                        int gate_idx_1_2 = token_1_2 * num_o_heads_5 + head_idx_5;
                        int beta_idx_0_2 = gate_idx_0_2;
                        int beta_idx_1_2 = gate_idx_1_2;
                        float gate_val_0_2 = 1.0f;
                        float gate_val_1_2 = 1.0f;
                        if (prefetch_idx_2 >= num_valid_chunks_b - 1) {
                            if (token_0_2 < batch_end_5) {
                                gate_val_0_2 = gate[gate_idx_0_2];
                            }
                            if (token_1_2 < batch_end_5) {
                                gate_val_1_2 = gate[gate_idx_1_2];
                            }
                        } else {
                            gate_val_0_2 = gate[gate_idx_0_2];
                            gate_val_1_2 = gate[gate_idx_1_2];
                        }
                        float _log2_4;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(gate_val_0_2 + 1e-10f));
                        gate_log_0_2 = _log2_4;
                        float _log2_5;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(gate_val_1_2 + 1e-10f));
                        gate_log_1_2 = _log2_5;
                        float _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 1, 32);
                        if (gb_lane_2 >= 1) {
                            gate_log_0_2 = gate_log_0_2 + _shfl_up_20;
                        }
                        float _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 1, 32);
                        if (gb_lane_2 >= 1) {
                            gate_log_1_2 = gate_log_1_2 + _shfl_up_21;
                        }
                        float _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 2, 32);
                        if (gb_lane_2 >= 2) {
                            gate_log_0_2 = gate_log_0_2 + _shfl_up_22;
                        }
                        float _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 2, 32);
                        if (gb_lane_2 >= 2) {
                            gate_log_1_2 = gate_log_1_2 + _shfl_up_23;
                        }
                        float _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 4, 32);
                        if (gb_lane_2 >= 4) {
                            gate_log_0_2 = gate_log_0_2 + _shfl_up_24;
                        }
                        float _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 4, 32);
                        if (gb_lane_2 >= 4) {
                            gate_log_1_2 = gate_log_1_2 + _shfl_up_25;
                        }
                        float _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 8, 32);
                        if (gb_lane_2 >= 8) {
                            gate_log_0_2 = gate_log_0_2 + _shfl_up_26;
                        }
                        float _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 8, 32);
                        if (gb_lane_2 >= 8) {
                            gate_log_1_2 = gate_log_1_2 + _shfl_up_27;
                        }
                        float _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 16, 32);
                        if (gb_lane_2 >= 16) {
                            gate_log_0_2 = gate_log_0_2 + _shfl_up_28;
                        }
                        float _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 16, 32);
                        if (gb_lane_2 >= 16) {
                            gate_log_1_2 = gate_log_1_2 + _shfl_up_29;
                        }
                        float _shfl_3 = __shfl_sync(0xFFFFFFFF, gate_log_0_2, 31);
                        gate_log_1_2 = gate_log_1_2 + _shfl_3;
                        smem_cumsumlog[gate_elem_base_2 + gb_lane_2] = gate_log_0_2;
                        smem_cumsumlog[gate_elem_base_2 + gb_lane_2 + 32] = gate_log_1_2;
                        float _exp2_9 = approx_exp2(gate_log_0_2);
                        smem_cumprod[gate_elem_base_2 + gb_lane_2] = _exp2_9;
                        float _exp2_10 = approx_exp2(gate_log_1_2);
                        smem_cumprod[gate_elem_base_2 + gb_lane_2 + 32] = _exp2_10;
                        mbarrier_arrive(load_gate_full_addr + (gate_prod_stage) * 8);
                        int beta_stage_addr_2 = smem_beta_addr + gate_prod_stage * 256;
                        int beta_dst_0_2 = beta_stage_addr_2 + gb_lane_2 * 4;
                        int beta_dst_1_2 = beta_stage_addr_2 + (gb_lane_2 + 32) * 4;
                        if (prefetch_idx_2 >= num_valid_chunks_b - 1) {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_0_2), "l"(beta + beta_idx_0_2), "r"((token_0_2 < batch_end_5) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1_2), "l"(beta + beta_idx_1_2), "r"((token_1_2 < batch_end_5) ? 4 : 0));
                        } else {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_0_2), "l"(beta + beta_idx_0_2));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_1_2), "l"(beta + beta_idx_1_2));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(load_beta_full_addr + (gate_prod_stage) * 8) : "memory");
                        gate_prod_stage += 1;
                        if (gate_prod_stage == 5) { gate_prod_stage = 0; gate_prod_phase ^= 1; }
                    }
                    mbarrier_wait(o_store_ready_addr + (o_epi_stage) * 8, o_epi_phase);
                    if (lane == 0) {
                        if (chunk_idx_6 == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_3d(tensormap_workspace + (cta_slot_base_epi + 384), value_split_offset_1, chunk_offset_8, head_idx_5, smem_o_addr + o_epi_stage * 16384);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    o_epi_stage += 1;
                    if (o_epi_stage == 2) { o_epi_stage = 0; o_epi_phase ^= 1; }
                    if (chunk_idx_6 > 0) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                        mbarrier_arrive(o_smem_empty_addr + (o_release_stage) * 8);
                        o_release_stage += 1;
                        if (o_release_stage == 2) { o_release_stage = 0; o_release_phase ^= 1; }
                    }
                }
                if (active_output_chunks_b > 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    mbarrier_arrive(o_smem_empty_addr + (o_release_stage) * 8);
                    o_release_stage += 1;
                    if (o_release_stage == 2) { o_release_stage = 0; o_release_phase ^= 1; }
                }
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

#undef ENABLE_CHECKPOINTS
#undef HEAD_GROUP_LOG2
#undef IS_GQA
#undef CAKE_INF
#undef NUM_AINV_PIPE_STAGES
#undef NUM_CG0_ACC_PIPE_STAGES
#undef NUM_GATE_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_ONE_STAGE_STAGES
#undef NUM_O_HEADS_LOG2
#undef NUM_O_PIPE_STAGES
#undef NUM_QK_PIPE_STAGES
#undef NUM_Q_PIPE_STAGES
#undef NUM_V_PIPE_STAGES
#undef SINGLE_CHUNK_NO_STATE
#undef SMEM_SMEM_AINV_OFF
#undef SMEM_SMEM_AINV_RM_OFF
#undef SMEM_SMEM_AINV_RM_STAGE_BYTES
#undef SMEM_SMEM_AINV_RM_STRIDE
#undef SMEM_SMEM_AINV_STAGE_BYTES
#undef SMEM_SMEM_AINV_STRIDE
#undef SMEM_SMEM_BETA_OFF
#undef SMEM_SMEM_BETA_STAGE_BYTES
#undef SMEM_SMEM_BETA_STRIDE
#undef SMEM_SMEM_CUMPROD_OFF
#undef SMEM_SMEM_CUMPROD_STAGE_BYTES
#undef SMEM_SMEM_CUMPROD_STRIDE
#undef SMEM_SMEM_CUMSUMLOG_OFF
#undef SMEM_SMEM_CUMSUMLOG_STAGE_BYTES
#undef SMEM_SMEM_CUMSUMLOG_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_SMEM_K_TRANS_MMA_OFF
#undef SMEM_SMEM_K_TRANS_MMA_STAGE_BYTES
#undef SMEM_SMEM_K_TRANS_MMA_STRIDE
#undef SMEM_SMEM_O_OFF
#undef SMEM_SMEM_O_STAGE_BYTES
#undef SMEM_SMEM_O_STRIDE
#undef SMEM_SMEM_QK_OFF
#undef SMEM_SMEM_QK_STAGE_BYTES
#undef SMEM_SMEM_QK_STRIDE
#undef SMEM_SMEM_Q_OFF
#undef SMEM_SMEM_Q_STAGE_BYTES
#undef SMEM_SMEM_Q_STRIDE
#undef SMEM_SMEM_V_MMA_OFF
#undef SMEM_SMEM_V_MMA_STAGE_BYTES
#undef SMEM_SMEM_V_MMA_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_TOTAL
#undef STORE_FINAL_STATE
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_CG0_SHARED_ACC_OFFSET
#undef TMEM_TMEM_CG1_SHARED_ACC_OFFSET
#undef TMEM_TMEM_Q_STATE_OFFSET
#undef TMEM_TMEM_SHARED_INP_OFFSET
#undef TMEM_TMEM_STATE_INP_OFFSET
#undef TMEM_TMEM_STATE_OFFSET
#undef USE_INITIAL_STATE
#undef USE_STATE_INDICES
#undef ainv_ready_addr
#undef ainv_smem_empty_addr
#undef beta_smem_empty_addr
#undef cg0_shared_acc_empty_addr
#undef cg0_shared_acc_full_addr
#undef cg1_shared_acc_empty_addr
#undef cg1_shared_acc_full_addr
#undef decay_v_ready_addr
#undef gate_cg0_empty_addr
#undef gate_cg1_empty_addr
#undef initial_state_loaded_addr
#undef kv_acc_empty_addr
#undef kv_acc_full_addr
#undef load_beta_full_addr
#undef load_gate_full_addr
#undef load_k_empty_addr
#undef load_k_full_addr
#undef load_q_full_addr
#undef load_v_full_addr
#undef nv_ready_addr
#undef o_smem_empty_addr
#undef o_store_ready_addr
#undef q_smem_empty_addr
#undef q_state_acc_empty_addr
#undef q_state_acc_full_addr
#undef qk_ready_addr
#undef qk_smem_empty_addr
#undef smem_ainv_addr
#undef smem_ainv_rm_addr
#undef smem_beta_addr
#undef smem_cumprod_addr
#undef smem_cumsumlog_addr
#undef smem_k_addr
#undef smem_k_trans_mma_addr
#undef smem_o_addr
#undef smem_q_addr
#undef smem_qk_addr
#undef smem_v_addr
#undef smem_v_mma_addr
#undef state_inp_ready_addr
#undef v_smem_empty_addr
#undef vks_ready_addr
// clang-format on
