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
#define TMEM_TMEM_STATE_INPUT_OFFSET 192
#define TMEM_TMEM_CG0_ACC_OFFSET 256
#define TMEM_TMEM_CG1_ACC_OFFSET 384
#define TMEM_TMEM_SHARED_INPUT_OFFSET 448
#define NUM_Q_PIPE_STAGES 2
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define NUM_T_PIPE_STAGES 4
#define NUM_GATE_PIPE_STAGES 5
#define NUM_AINV_PIPE_STAGES 3
#define NUM_QK_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define NUM_ONE_STAGE_STAGES 1
#define NUM_CG0_ACC_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 33792
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_V_OFF 82944
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_T_OFF 115712
#define SMEM_SMEM_T_STAGE_BYTES 8192
#define SMEM_SMEM_T_STRIDE 8192
#define SMEM_SMEM_AINV_OFF 148480
#define SMEM_SMEM_AINV_STAGE_BYTES 8192
#define SMEM_SMEM_AINV_STRIDE 8192
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
#define SMEM_SMEM_K_TRANS_OFF 33792
#define SMEM_SMEM_K_TRANS_STAGE_BYTES 16384
#define SMEM_SMEM_K_TRANS_STRIDE 16384
#define SMEM_TOTAL 224768
#define THREADS 384

__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_flashinfer_blackwell_gdn_cp_prefill_final_equal_head_h32_v1(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap T, const __grid_constant__ CUtensorMap O, float* __restrict__ alpha, long long* __restrict__ cu_seqlens, float* __restrict__ fixed_state, float* __restrict__ initial_state_workspace, uint8_t* __restrict__ tensormap_workspace, int cp_chunk_len, int source_cp_chunk_len, int num_q_heads, int num_k_heads, int num_v_heads, int num_sab_heads, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __half* smem_q = reinterpret_cast<__half*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __half* smem_k = reinterpret_cast<__half*>(smem_raw + 33792);
    const int smem_k_addr = smem + 33792;
    __half* smem_v = reinterpret_cast<__half*>(smem_raw + 82944);
    const int smem_v_addr = smem + 82944;
    __half* smem_t = reinterpret_cast<__half*>(smem_raw + 115712);
    const int smem_t_addr = smem + 115712;
    __half* smem_ainv = reinterpret_cast<__half*>(smem_raw + 148480);
    const int smem_ainv_addr = smem + 148480;
    __half* smem_qk = reinterpret_cast<__half*>(smem_raw + 173056);
    const int smem_qk_addr = smem + 173056;
    __half* smem_o = reinterpret_cast<__half*>(smem_raw + 189440);
    const int smem_o_addr = smem + 189440;
    float* smem_cumsumlog = reinterpret_cast<float*>(smem_raw + 222208);
    const int smem_cumsumlog_addr = smem + 222208;
    float* smem_cumprod = reinterpret_cast<float*>(smem_raw + 223488);
    const int smem_cumprod_addr = smem + 223488;
    __half* smem_k_trans = reinterpret_cast<__half*>(smem_raw + 33792);
    const int smem_k_trans_addr = smem + 33792;

    // Mbarrier init (29 groups, 61 barriers)
    // Mbarriers at smem_raw[0..488)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // load_q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // load_q_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // --- pipeline 'k_pipe' ---
            // load_k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // load_k_empty: 3 barriers, init_count=2
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            // --- pipeline 'v_pipe' ---
            // load_v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // load_v_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            // --- pipeline 't_pipe' ---
            // load_t_full: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // load_t_empty: 4 barriers, init_count=4
            mbarrier_init(smem + 144, 4);
            mbarrier_init(smem + 152, 4);
            mbarrier_init(smem + 160, 4);
            mbarrier_init(smem + 168, 4);
            // --- pipeline 'gate_pipe' ---
            // load_gate_full: 5 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            mbarrier_init(smem + 200, 32);
            mbarrier_init(smem + 208, 32);
            // load_gate_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            mbarrier_init(smem + 240, 256);
            mbarrier_init(smem + 248, 256);
            // --- pipeline 'one_stage' ---
            // q_state_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            // q_state_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 264, 128);
            // kv_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // kv_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            // --- pipeline 'cg0_acc_pipe' ---
            // cg0_acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // cg0_acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            mbarrier_init(smem + 312, 128);
            // --- pipeline 'one_stage' ---
            // cg1_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            // cg1_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            // --- pipeline 'ainv_pipe' ---
            // ainv_ready: 3 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            mbarrier_init(smem + 344, 128);
            mbarrier_init(smem + 352, 128);
            // ainv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            // --- pipeline 'qk_pipe' ---
            // qk_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            mbarrier_init(smem + 392, 128);
            // qk_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            // --- pipeline 'one_stage' ---
            // state_input_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 416, 128);
            // state_input_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 424, 1);
            // vks_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 432, 128);
            // nv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 440, 128);
            // decay_v_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 448, 128);
            // --- pipeline 'o_pipe' ---
            // o_store_ready: 2 barriers, init_count=128
            mbarrier_init(smem + 456, 128);
            mbarrier_init(smem + 464, 128);
            // o_store_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 472, 32);
            mbarrier_init(smem + 480, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 488);
    if (warp == 4) {
        int _tmem_hold = smem + 488;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define load_q_full_addr (mbar_base + 0)
    #define load_q_empty_addr (mbar_base + 16)
    #define load_k_full_addr (mbar_base + 32)
    #define load_k_empty_addr (mbar_base + 56)
    #define load_v_full_addr (mbar_base + 80)
    #define load_v_empty_addr (mbar_base + 96)
    #define load_t_full_addr (mbar_base + 112)
    #define load_t_empty_addr (mbar_base + 144)
    #define load_gate_full_addr (mbar_base + 176)
    #define load_gate_empty_addr (mbar_base + 216)
    #define q_state_acc_full_addr (mbar_base + 256)
    #define q_state_acc_empty_addr (mbar_base + 264)
    #define kv_acc_full_addr (mbar_base + 272)
    #define kv_acc_empty_addr (mbar_base + 280)
    #define cg0_acc_full_addr (mbar_base + 288)
    #define cg0_acc_empty_addr (mbar_base + 304)
    #define cg1_acc_full_addr (mbar_base + 320)
    #define cg1_acc_empty_addr (mbar_base + 328)
    #define ainv_ready_addr (mbar_base + 336)
    #define ainv_empty_addr (mbar_base + 360)
    #define qk_ready_addr (mbar_base + 384)
    #define qk_empty_addr (mbar_base + 400)
    #define state_input_ready_addr (mbar_base + 416)
    #define state_input_empty_addr (mbar_base + 424)
    #define vks_ready_addr (mbar_base + 432)
    #define nv_ready_addr (mbar_base + 440)
    #define decay_v_ready_addr (mbar_base + 448)
    #define o_store_ready_addr (mbar_base + 456)
    #define o_store_empty_addr (mbar_base + 472)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr;
    const int tmem_tmem_q_state = taddr + 128;
    const int tmem_tmem_state_input = taddr + 192;
    const int tmem_tmem_cg0_acc = taddr + 256;
    const int tmem_tmem_cg1_acc = taddr + 384;
    const int tmem_tmem_shared_input = taddr + 448;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: compute_group_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // compute_group_0_main
            int sab_head = blockIdx.x % num_sab_heads;
            int chunk_in_seq = blockIdx.x / num_sab_heads;
            int seq_idx = blockIdx.y;
            int seq_start = (int)cu_seqlens[seq_idx];
            int seq_end = (int)cu_seqlens[seq_idx + 1];
            int seq_len = seq_end - seq_start;
            int num_cp_chunks = (seq_len + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining = seq_len - chunk_in_seq * cp_chunk_len;
                chunk_len = ((remaining < cp_chunk_len) ? remaining : cp_chunk_len);
            }
            int num_valid_blocks = (chunk_len + 64 - 1) / 64;
            int num_padded_blocks = (chunk_len + 128 - 1) / 128 * 2;
            int chunk_start = seq_start + chunk_in_seq * cp_chunk_len;
            int chunk_end = chunk_start + chunk_len;
            int prefix_items = ((seq_idx < seq_start) ? seq_idx : seq_start);
            int cp_chunk = prefix_items + (seq_start - prefix_items) / cp_chunk_len + chunk_in_seq;
            int t_blocks_per_chunk = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items = ((seq_idx < seq_start) ? seq_idx : seq_start);
            int t_block_start = t_prefix_items + (seq_start - t_prefix_items) / 64 + chunk_in_seq * t_blocks_per_chunk;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                unsigned int gate_stage_cg0 = 0;
                unsigned int gate_phase_cg0 = 0;
                unsigned int t_stage_cg0 = 0;
                unsigned int t_phase_cg0 = 0;
                unsigned int ainv_stage_cg0 = 0;
                unsigned int ainv_empty_phase_cg0 = 1;
                unsigned int qk_stage_cg0 = 0;
                unsigned int qk_empty_phase_cg0 = 1;
                unsigned int acc_stage_cg0 = 0;
                unsigned int acc_full_phase_cg0 = 0;
                int warp_id_in_role = (warp - 0);
                int warp_cg0 = warp_id_in_role;
                int lane_row_cg0 = lane / 4;
                int lane_quad_cg0 = lane & 3;
                int t_ld_row_cg0 = warp_cg0 * 16 + (lane & 7) + (lane & 8);
                int t_ld_col_lane_cg0 = (lane & 16) / 2;
                int qk_tmem_row_cg0 = warp_cg0 * 32 << 16;
                int qk_logical_row_base_cg0 = warp_cg0 * 16;
                #pragma unroll 2
                for (int block_cg0 = 0; block_cg0 < num_padded_blocks; block_cg0++) {
                    int valid_tokens_cg0 = 64;
                    {
                    }
                    int source_block_end_cg0 = chunk_start - (int)cu_seqlens[blockIdx.y] + block_cg0 * 64 + valid_tokens_cg0;
                    int source_chunk_end_cg0 = (source_block_end_cg0 + source_cp_chunk_len - 1) / source_cp_chunk_len * source_cp_chunk_len;
                    int source_final_block_cg0 = ((source_block_end_cg0 == source_chunk_end_cg0 || source_block_end_cg0 >= (int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y]) ? 1 : 0);
                    {
                    }
                    mbarrier_wait(load_gate_full_addr + (gate_stage_cg0) * 8, gate_phase_cg0);
                    mbarrier_wait(load_t_full_addr + (t_stage_cg0) * 8, t_phase_cg0);
                    mbarrier_wait(ainv_empty_addr + (ainv_stage_cg0) * 8, ainv_empty_phase_cg0);
                    mbarrier_wait(qk_empty_addr + (qk_stage_cg0) * 8, qk_empty_phase_cg0);
                    float t_row_logs_cg0[2];
                    {
                        {
                            int t_hoisted_gate_base_cg0 = gate_stage_cg0 * 64;
                            #pragma unroll
                            for (int t_row_group_cg0 = 0; t_row_group_cg0 < 2; t_row_group_cg0++) {
                                int t_hoisted_row_cg0 = warp_cg0 * 16 + lane_row_cg0 + t_row_group_cg0 * 8;
                                t_row_logs_cg0[t_row_group_cg0] = smem_cumsumlog[t_hoisted_gate_base_cg0 + t_hoisted_row_cg0];
                            }
                        }
                    }
                    unsigned int t_gamma_words_cg0[16];
                    float t_col_logs_for_qk_cg0[16];
                    float qk_scales_cg0[32];
                    #pragma unroll
                    for (int t_col_tile_cg0 = 0; t_col_tile_cg0 < 4; t_col_tile_cg0++) {
                        unsigned int t_bits_cg0[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(t_bits_cg0[0]), "=r"(t_bits_cg0[1]), "=r"(t_bits_cg0[2]), "=r"(t_bits_cg0[3])
                            : "r"((smem_t_addr + t_stage_cg0 * 8192 + (unsigned int)((t_col_tile_cg0 * 16 + t_ld_col_lane_cg0) / 64 * 8192 + t_ld_row_cg0 * 128 + (t_col_tile_cg0 * 16 + t_ld_col_lane_cg0) % 64 * 2 ^ ((t_col_tile_cg0 * 16 + t_ld_col_lane_cg0) / 64 * 8192 + t_ld_row_cg0 * 128 + (t_col_tile_cg0 * 16 + t_ld_col_lane_cg0) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        float t_bits_cg0_f32[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&t_bits_cg0_f32[_pair * 2]))
                                : "r"(t_bits_cg0[_pair]));
                        }
                        {
                            {
                                {
                                    {
                                        {
                                            int t_gate_base_cg0 = gate_stage_cg0 * 64;
                                            float t_col_logs_cg0[4];
                                            #pragma unroll
                                            for (int t_col_group_cg0 = 0; t_col_group_cg0 < 2; t_col_group_cg0++) {
                                                int t_cached_col_lo_cg0 = t_col_tile_cg0 * 16 + lane_quad_cg0 * 2 + t_col_group_cg0 * 8;
                                                float2 _f2_6 = make_float2(smem_cumsumlog[t_gate_base_cg0 + t_cached_col_lo_cg0], smem_cumsumlog[t_gate_base_cg0 + t_cached_col_lo_cg0 + 1]);
                                                float2 t_col_log_pair_cg0 = _f2_6;
                                                t_col_logs_cg0[t_col_group_cg0 * 2] = t_col_log_pair_cg0.x;
                                                t_col_logs_cg0[t_col_group_cg0 * 2 + 1] = t_col_log_pair_cg0.y;
                                            }
                                            #pragma unroll
                                            for (int t_item_cg0 = 0; t_item_cg0 < 8; t_item_cg0++) {
                                                int t_quad_cg0 = t_item_cg0 & 3;
                                                int t_row_cg0 = warp_cg0 * 16 + lane_row_cg0 + t_quad_cg0 / 2 * 8;
                                                int t_col_cg0 = t_col_tile_cg0 * 16 + lane_quad_cg0 * 2 + (t_quad_cg0 & 1) + t_item_cg0 / 4 * 8;
                                                int t_row_log_group_cg0 = t_quad_cg0 / 2;
                                                int t_col_log_item_cg0 = t_item_cg0 / 4 * 2 + (t_quad_cg0 & 1);
                                                int t_valid_cg0 = 0;
                                                float gamma_cg0 = 0.0f;
                                                float symmetric_scale_cg0 = 0.0f;
                                                t_valid_cg0 = ((t_col_cg0 >= t_row_cg0) ? 1 : 0);
                                                if (source_final_block_cg0 != 0) {
                                                    int final_t_threshold_cg0 = valid_tokens_cg0 - 33;
                                                    if (final_t_threshold_cg0 < 0) {
                                                        final_t_threshold_cg0 = 0;
                                                    }
                                                    if (t_row_cg0 != t_col_cg0 && t_row_cg0 < final_t_threshold_cg0) {
                                                        t_valid_cg0 = 0;
                                                        gamma_cg0 = 0.0f;
                                                    }
                                                }
                                                if (t_valid_cg0 != 0) {
                                                    float _exp2_15 = approx_exp2(t_col_logs_cg0[t_col_log_item_cg0] - t_row_logs_cg0[t_row_log_group_cg0]);
                                                    gamma_cg0 = _exp2_15;
                                                }
                                                t_bits_cg0_f32[t_item_cg0] = (-gamma_cg0) * t_bits_cg0_f32[t_item_cg0];
                                                int qk_valid_cg0 = 0;
                                                float qk_scale_cg0 = 0.0f;
                                                qk_valid_cg0 = ((t_row_cg0 >= t_col_cg0) ? 1 : 0);
                                                if (source_final_block_cg0 != 0) {
                                                    int final_qk_threshold_cg0 = valid_tokens_cg0 - 33;
                                                    if (final_qk_threshold_cg0 < 0) {
                                                        final_qk_threshold_cg0 = 0;
                                                    }
                                                    if (t_row_cg0 != t_col_cg0 && t_col_cg0 < final_qk_threshold_cg0) {
                                                        qk_valid_cg0 = 0;
                                                        qk_scale_cg0 = 0.0f;
                                                    }
                                                }
                                                if (qk_valid_cg0 != 0) {
                                                    float _exp2_16 = approx_exp2(t_row_logs_cg0[t_row_log_group_cg0] - t_col_logs_cg0[t_col_log_item_cg0]);
                                                    qk_scale_cg0 = _exp2_16;
                                                }
                                                const int qk_scale_item_cg0 = t_col_tile_cg0 * 8 + t_item_cg0;
                                                qk_scales_cg0[qk_scale_item_cg0] = qk_scale_cg0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        unsigned int t_signed_bits_cg0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(t_bits_cg0_f32[_lp*2 + 0], t_bits_cg0_f32[_lp*2+1 + 0]));
                            t_signed_bits_cg0[_lp] = *(uint32_t*)&_h2;
                        }
                        int t_store_row_cg0 = t_col_tile_cg0 * 16 + (lane & 7) + (lane & 16) / 2;
                        int t_store_col_lane_cg0 = warp_cg0 * 16 + (lane & 8);
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_ainv_addr + ainv_stage_cg0 * 8192 + (unsigned int)(t_store_col_lane_cg0 / 64 * 8192 + t_store_row_cg0 * 128 + t_store_col_lane_cg0 % 64 * 2 ^ (t_store_col_lane_cg0 / 64 * 8192 + t_store_row_cg0 * 128 + t_store_col_lane_cg0 % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&t_signed_bits_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&t_signed_bits_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&t_signed_bits_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&t_signed_bits_cg0[3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    unsigned int qk_gamma_words_cg0[16];
                    if (elect_sync()) {
                        mbarrier_arrive(load_t_empty_addr + (t_stage_cg0) * 8);
                    }
                    mbarrier_arrive(ainv_ready_addr + (ainv_stage_cg0) * 8);
                    mbarrier_wait(cg0_acc_full_addr + (acc_stage_cg0) * 8, acc_full_phase_cg0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(taddr + 256 + acc_stage_cg0 * 64 + (unsigned int)qk_tmem_row_cg0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    {
                        {
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], reinterpret_cast<const float2*>(qk_scales_cg0)[_ls]);
                        const float2 _scale2_1 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_1);
                    }
                    unsigned int qk_bits_cg0[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        qk_bits_cg0[_lp] = *(uint32_t*)&_h2;
                    }
                    int qk_store_row_cg0 = qk_logical_row_base_cg0 + (lane & 7) + (lane & 8);
                    int qk_store_col_lane_cg0 = (lane & 16) / 2;
                    #pragma unroll
                    for (int qk_store_repeat_cg0 = 0; qk_store_repeat_cg0 < 4; qk_store_repeat_cg0++) {
                        const int qk_word_base_cg0 = qk_store_repeat_cg0 * 4;
                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk_stage_cg0 * 8192 + (unsigned int)((qk_store_repeat_cg0 * 16 + qk_store_col_lane_cg0) / 64 * 8192 + qk_store_row_cg0 * 128 + (qk_store_repeat_cg0 * 16 + qk_store_col_lane_cg0) % 64 * 2 ^ ((qk_store_repeat_cg0 * 16 + qk_store_col_lane_cg0) / 64 * 8192 + qk_store_row_cg0 * 128 + (qk_store_repeat_cg0 * 16 + qk_store_col_lane_cg0) % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&qk_bits_cg0[qk_word_base_cg0])), "r"(*reinterpret_cast<const uint32_t*>(&qk_bits_cg0[qk_word_base_cg0 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&qk_bits_cg0[qk_word_base_cg0 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&qk_bits_cg0[qk_word_base_cg0 + 3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        mbarrier_arrive(cg0_acc_empty_addr + (acc_stage_cg0) * 8);
                        mbarrier_arrive(qk_ready_addr + (qk_stage_cg0) * 8);
                    }
                    mbarrier_arrive(load_gate_empty_addr + (gate_stage_cg0) * 8);
                    gate_stage_cg0 += 1;
                    if (gate_stage_cg0 == 5) { gate_stage_cg0 = 0; gate_phase_cg0 ^= 1; }
                    t_stage_cg0 += 1;
                    if (t_stage_cg0 == 4) { t_stage_cg0 = 0; t_phase_cg0 ^= 1; }
                    ainv_stage_cg0 += 1;
                    if (ainv_stage_cg0 == 3) { ainv_stage_cg0 = 0; ainv_empty_phase_cg0 ^= 1; }
                    qk_stage_cg0 += 1;
                    if (qk_stage_cg0 == 2) { qk_stage_cg0 = 0; qk_empty_phase_cg0 ^= 1; }
                    acc_stage_cg0 += 1;
                    if (acc_stage_cg0 == 2) { acc_stage_cg0 = 0; acc_full_phase_cg0 ^= 1; }
                }
                #pragma unroll
                for (int _ = 0; _ < 3; _++) {
                    mbarrier_wait(ainv_empty_addr + (ainv_stage_cg0) * 8, ainv_empty_phase_cg0);
                    ainv_stage_cg0 += 1;
                    if (ainv_stage_cg0 == 3) { ainv_stage_cg0 = 0; ainv_empty_phase_cg0 ^= 1; }
                }
                #pragma unroll
                for (int __1 = 0; __1 < 2; __1++) {
                    mbarrier_wait(qk_empty_addr + (qk_stage_cg0) * 8, qk_empty_phase_cg0);
                    qk_stage_cg0 += 1;
                    if (qk_stage_cg0 == 2) { qk_stage_cg0 = 0; qk_empty_phase_cg0 ^= 1; }
                }
            }
        }
    // ---- Role: compute_group_1 ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        { // compute_group_1_main
            int sab_head_1 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_1 = blockIdx.x / num_sab_heads;
            int seq_idx_1 = blockIdx.y;
            int seq_start_1 = (int)cu_seqlens[seq_idx_1];
            int seq_end_1 = (int)cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = seq_end_1 - seq_start_1;
            int num_cp_chunks_1 = (seq_len_1 + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len_1 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining_1 = seq_len_1 - chunk_in_seq_1 * cp_chunk_len;
                chunk_len_1 = ((remaining_1 < cp_chunk_len) ? remaining_1 : cp_chunk_len);
            }
            int num_valid_blocks_1 = (chunk_len_1 + 64 - 1) / 64;
            int num_padded_blocks_1 = (chunk_len_1 + 128 - 1) / 128 * 2;
            int chunk_start_1 = seq_start_1 + chunk_in_seq_1 * cp_chunk_len;
            int chunk_end_1 = chunk_start_1 + chunk_len_1;
            int prefix_items_1 = ((seq_idx_1 < seq_start_1) ? seq_idx_1 : seq_start_1);
            int cp_chunk_1 = prefix_items_1 + (seq_start_1 - prefix_items_1) / cp_chunk_len + chunk_in_seq_1;
            int t_blocks_per_chunk_1 = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items_1 = ((seq_idx_1 < seq_start_1) ? seq_idx_1 : seq_start_1);
            int t_block_start_1 = t_prefix_items_1 + (seq_start_1 - t_prefix_items_1) / 64 + chunk_in_seq_1 * t_blocks_per_chunk_1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int warp_id_in_role_1 = (warp - 4);
                int warp_cg1 = warp_id_in_role_1;
                int tmem_row_base_cg1 = warp_cg1 * 32 << 16;
                int lane_quad_cg1 = lane & 3;
                int state_slot_cg1 = ((chunk_in_seq_1 > 0) ? (cp_chunk_1 - 1) * num_sab_heads + sab_head_1 : blockIdx.y * num_sab_heads + sab_head_1);
                int state_row_cg1 = warp_cg1 * 32 + lane;
                long long state_base_cg1 = (long long)state_slot_cg1 * 16384 + (long long)state_row_cg1 * 128;
                #pragma unroll
                for (int state_col_block_cg1 = 0; state_col_block_cg1 < 4; state_col_block_cg1++) {
                    float state_seed_cg1[32];
                    if (chunk_in_seq_1 > 0) {
                        #pragma unroll
                        for (int state_vec_cg1 = 0; state_vec_cg1 < 8; state_vec_cg1++) {
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(fixed_state + state_base_cg1 + (long long)(state_col_block_cg1 * 32) + (long long)(state_vec_cg1 * 4));
                                state_seed_cg1[state_vec_cg1 * 4 + 0] = _v4.x;
                                state_seed_cg1[state_vec_cg1 * 4 + 1] = _v4.y;
                                state_seed_cg1[state_vec_cg1 * 4 + 2] = _v4.z;
                                state_seed_cg1[state_vec_cg1 * 4 + 3] = _v4.w;
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int state_vec_cg1_1 = 0; state_vec_cg1_1 < 8; state_vec_cg1_1++) {
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(initial_state_workspace + state_base_cg1 + (long long)(state_col_block_cg1 * 32) + (long long)(state_vec_cg1_1 * 4));
                                state_seed_cg1[state_vec_cg1_1 * 4 + 0] = _v4.x;
                                state_seed_cg1[state_vec_cg1_1 * 4 + 1] = _v4.y;
                                state_seed_cg1[state_vec_cg1_1 * 4 + 2] = _v4.z;
                                state_seed_cg1[state_vec_cg1_1 * 4 + 3] = _v4.w;
                            }
                        }
                    }
                    tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base_cg1 + (unsigned int)(state_col_block_cg1 * 32), state_seed_cg1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (warp_cg1 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(kv_acc_full_addr);
                    }
                }
                unsigned int gate_stage_cg1 = 0;
                unsigned int gate_phase_cg1 = 0;
                unsigned int v_stage_cg1 = 0;
                unsigned int v_phase_cg1 = 0;
                unsigned int kv_stage_cg1 = 0;
                unsigned int kv_full_phase_cg1 = 0;
                unsigned int cg1_acc_stage_cg1 = 0;
                unsigned int cg1_acc_full_phase_cg1 = 0;
                unsigned int q_state_stage_cg1 = 0;
                unsigned int q_state_full_phase_cg1 = 0;
                unsigned int o_stage_cg1 = 0;
                unsigned int o_empty_phase_cg1 = 1;
                unsigned int state_input_stage_cg1 = 0;
                unsigned int state_input_empty_phase_cg1 = 1;
                #pragma unroll 1
                for (int block_cg1 = 0; block_cg1 < num_padded_blocks_1; block_cg1++) {
                    int valid_tokens_cg1 = chunk_end_1 - (chunk_start_1 + block_cg1 * 64);
                    if (valid_tokens_cg1 > 64) {
                        valid_tokens_cg1 = 64;
                    }
                    if (valid_tokens_cg1 < 0) {
                        valid_tokens_cg1 = 0;
                    }
                    int source_block_end_cg1 = chunk_start_1 - (int)cu_seqlens[blockIdx.y] + block_cg1 * 64 + valid_tokens_cg1;
                    int source_chunk_end_cg1 = (source_block_end_cg1 + source_cp_chunk_len - 1) / source_cp_chunk_len * source_cp_chunk_len;
                    int source_final_block_cg1 = ((valid_tokens_cg1 == 64 && (source_block_end_cg1 == source_chunk_end_cg1 || source_block_end_cg1 >= (int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y])) ? 1 : 0);
                    {
                    }
                    mbarrier_wait(load_gate_full_addr + (gate_stage_cg1) * 8, gate_phase_cg1);
                    int gate_base_cg1 = gate_stage_cg1 * 64;
                    mbarrier_wait(kv_acc_full_addr + (kv_stage_cg1) * 8, kv_full_phase_cg1);
                    mbarrier_wait(state_input_empty_addr + (state_input_stage_cg1) * 8, state_input_empty_phase_cg1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float state_decay_cg1 = smem_cumprod[gate_base_cg1 + 64 - 1];
                    float state_values_cg1[128];
                    tmem_ld_x32(&state_values_cg1[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x32(&state_values_cg1[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x32(&state_values_cg1[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x32(&state_values_cg1[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int state_input_bits_cg1[64];
                    #pragma unroll
                    for (int _lp = 0; _lp < 64; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(state_values_cg1[_lp*2 + 0], state_values_cg1[_lp*2+1 + 0]));
                        state_input_bits_cg1[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int state_col_block_cg1_1 = 0; state_col_block_cg1_1 < 4; state_col_block_cg1_1++) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 192 + (unsigned int)tmem_row_base_cg1 + (unsigned int)(state_col_block_cg1_1 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[0])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[1])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[2])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[3])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[4])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[5])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[6])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[7])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[8])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[9])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[10])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[11])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[12])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[13])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[14])), "r"(*reinterpret_cast<const uint32_t*>(&(state_input_bits_cg1 + state_col_block_cg1_1 * 16)[15])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_input_ready_addr + (state_input_stage_cg1) * 8);
                    state_input_stage_cg1 += 1;
                    if (state_input_stage_cg1 == 1) { state_input_stage_cg1 = 0; state_input_empty_phase_cg1 ^= 1; }
                    const float2 _scale2_2 = {state_decay_cg1, state_decay_cg1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(state_values_cg1)[_ls], _scale2_2);
                    #pragma unroll
                    for (int state_col_block_cg1_2 = 0; state_col_block_cg1_2 < 4; state_col_block_cg1_2++) {
                        tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base_cg1 + (unsigned int)(state_col_block_cg1_2 * 32), (state_values_cg1 + state_col_block_cg1_2 * 32));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(kv_acc_empty_addr + (kv_stage_cg1) * 8);
                    float cumprod_values_cg1[32];
                    float decay_scales_cg1[32];
                    float last_log_cg1 = smem_cumsumlog[gate_base_cg1 + 64 - 1];
                    #pragma unroll
                    for (int gate_pair_cg1 = 0; gate_pair_cg1 < 16; gate_pair_cg1++) {
                        int gate_repeat_cg1 = gate_pair_cg1 / 2;
                        int gate_token_lo_cg1 = gate_repeat_cg1 * 8 + lane_quad_cg1 * 2;
                        int gate_token_hi_cg1 = gate_token_lo_cg1 + 1;
                        float2 _f2_13 = make_float2(smem_cumsumlog[gate_base_cg1 + gate_token_lo_cg1], smem_cumsumlog[gate_base_cg1 + gate_token_hi_cg1]);
                        float2 gate_logs_cg1 = _f2_13;
                        float2 _f2_14 = make_float2(last_log_cg1, last_log_cg1);
                        float2 decay_diff_cg1 = sub_f32x2(_f2_14, gate_logs_cg1);
                        const int gate_item_lo_cg1 = gate_pair_cg1 * 2;
                        const int gate_item_hi_cg1 = gate_item_lo_cg1 + 1;
                        cumprod_values_cg1[gate_item_lo_cg1] = smem_cumprod[gate_base_cg1 + gate_token_lo_cg1];
                        cumprod_values_cg1[gate_item_hi_cg1] = smem_cumprod[gate_base_cg1 + gate_token_hi_cg1];
                        float _exp2_31 = approx_exp2(decay_diff_cg1.x);
                        decay_scales_cg1[gate_item_lo_cg1] = _exp2_31;
                        float _exp2_32 = approx_exp2(decay_diff_cg1.y);
                        decay_scales_cg1[gate_item_hi_cg1] = _exp2_32;
                    }
                    mbarrier_arrive(load_gate_empty_addr + (gate_stage_cg1) * 8);
                    float ks_frag_lo_cg1[32];
                    float ks_frag_hi_cg1[32];
                    {
                        mbarrier_wait(cg1_acc_full_addr + (cg1_acc_stage_cg1) * 8, cg1_acc_full_phase_cg1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int ks_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_cg1;
                        int ks_addr_hi_cg1 = ks_addr_lo_cg1 + 1048576;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_lo_cg1[31]))
                            : "r"(ks_addr_lo_cg1));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&ks_frag_hi_cg1[31]))
                            : "r"(ks_addr_hi_cg1));
                    }
                    mbarrier_wait(load_v_full_addr + (v_stage_cg1) * 8, v_phase_cg1);
                    unsigned int v_frag_lo_cg1[16];
                    unsigned int v_frag_hi_cg1[16];
                    #pragma unroll
                    for (int v_repeat_cg1 = 0; v_repeat_cg1 < 8; v_repeat_cg1++) {
                        unsigned int v_bits_cg1[4];
                        int v_matrix_cg1 = lane / 8;
                        int v_token_cg1 = v_repeat_cg1 * 8 + (lane & 7);
                        int v_dim_cg1 = warp_cg1 * 32 + v_matrix_cg1 * 8;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(v_bits_cg1[0]), "=r"(v_bits_cg1[1]), "=r"(v_bits_cg1[2]), "=r"(v_bits_cg1[3])
                            : "r"((smem_v_addr + v_stage_cg1 * 16384 + (unsigned int)((v_token_cg1 + v_dim_cg1 / 64 * 64) * 128 + (v_dim_cg1 & 63) * 2 ^ ((v_token_cg1 + v_dim_cg1 / 64 * 64) * 128 + (v_dim_cg1 & 63) * 2 >> 7 & 7) << 4)))
                            : "memory");
                        const int v_word_cg1 = v_repeat_cg1 * 2;
                        v_frag_lo_cg1[v_word_cg1] = v_bits_cg1[0];
                        v_frag_lo_cg1[v_word_cg1 + 1] = v_bits_cg1[1];
                        v_frag_hi_cg1[v_word_cg1] = v_bits_cg1[2];
                        v_frag_hi_cg1[v_word_cg1 + 1] = v_bits_cg1[3];
                    }
                    {
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(cg1_acc_empty_addr + (cg1_acc_stage_cg1) * 8);
                    cg1_acc_stage_cg1 += 1;
                    if (cg1_acc_stage_cg1 == 1) { cg1_acc_stage_cg1 = 0; cg1_acc_full_phase_cg1 ^= 1; }
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(ks_frag_lo_cg1)[_ls], reinterpret_cast<const float2*>(cumprod_values_cg1)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(ks_frag_hi_cg1)[_ls], reinterpret_cast<const float2*>(cumprod_values_cg1)[_ls]);
                    unsigned int ks_frag_lo_cg1_f16[16];
                    unsigned int ks_frag_hi_cg1_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(ks_frag_lo_cg1[_lp*2 + 0], ks_frag_lo_cg1[_lp*2+1 + 0]));
                        ks_frag_lo_cg1_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(ks_frag_hi_cg1[_lp*2 + 0], ks_frag_hi_cg1[_lp*2+1 + 0]));
                        ks_frag_hi_cg1_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    float qs_frag_early_cg1[32];
                    int qs_addr_early_cg1 = taddr + 128 + (unsigned int)tmem_row_base_cg1;
                    {
                        mbarrier_wait(q_state_acc_full_addr + (q_state_stage_cg1) * 8, q_state_full_phase_cg1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&qs_frag_early_cg1[31]))
                            : "r"(qs_addr_early_cg1));
                        __syncwarp();
                    }
                    unsigned int vks_bits_lo_cg1[16];
                    unsigned int vks_bits_hi_cg1[16];
                    #pragma unroll
                    for (int vks_pair_cg1 = 0; vks_pair_cg1 < 16; vks_pair_cg1++) {
                        {
                            if (source_final_block_cg1 != 0) {
                                vks_bits_lo_cg1[vks_pair_cg1] = v_frag_lo_cg1[vks_pair_cg1];
                                vks_bits_hi_cg1[vks_pair_cg1] = v_frag_hi_cg1[vks_pair_cg1];
                            } else {
                                uint32_t _f16x2_sub_0;
                                asm volatile("sub.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_sub_0) : "r"(v_frag_lo_cg1[vks_pair_cg1]), "r"(ks_frag_lo_cg1_f16[vks_pair_cg1]));
                                vks_bits_lo_cg1[vks_pair_cg1] = _f16x2_sub_0;
                                uint32_t _f16x2_sub_1;
                                asm volatile("sub.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_sub_1) : "r"(v_frag_hi_cg1[vks_pair_cg1]), "r"(ks_frag_hi_cg1_f16[vks_pair_cg1]));
                                vks_bits_hi_cg1[vks_pair_cg1] = _f16x2_sub_1;
                            }
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_lo_cg1[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_bits_hi_cg1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(vks_ready_addr);
                    {
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        {
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qs_frag_early_cg1)[_ls], reinterpret_cast<const float2*>(cumprod_values_cg1)[_ls]);
                            const float2 _scale2_3 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(qs_frag_early_cg1)[_ls], _scale2_3);
                        }
                        if (source_final_block_cg1 != 0) {
                            qs_frag_early_cg1[0] = 0.0f;
                            qs_frag_early_cg1[1] = 0.0f;
                            qs_frag_early_cg1[2] = 0.0f;
                            qs_frag_early_cg1[3] = 0.0f;
                            qs_frag_early_cg1[4] = 0.0f;
                            qs_frag_early_cg1[5] = 0.0f;
                            qs_frag_early_cg1[6] = 0.0f;
                            qs_frag_early_cg1[7] = 0.0f;
                            qs_frag_early_cg1[8] = 0.0f;
                            qs_frag_early_cg1[9] = 0.0f;
                            qs_frag_early_cg1[10] = 0.0f;
                            qs_frag_early_cg1[11] = 0.0f;
                            qs_frag_early_cg1[12] = 0.0f;
                            qs_frag_early_cg1[13] = 0.0f;
                            qs_frag_early_cg1[14] = 0.0f;
                            qs_frag_early_cg1[15] = 0.0f;
                            qs_frag_early_cg1[16] = 0.0f;
                            qs_frag_early_cg1[17] = 0.0f;
                            qs_frag_early_cg1[18] = 0.0f;
                            qs_frag_early_cg1[19] = 0.0f;
                            qs_frag_early_cg1[20] = 0.0f;
                            qs_frag_early_cg1[21] = 0.0f;
                            qs_frag_early_cg1[22] = 0.0f;
                            qs_frag_early_cg1[23] = 0.0f;
                            qs_frag_early_cg1[24] = 0.0f;
                            qs_frag_early_cg1[25] = 0.0f;
                            qs_frag_early_cg1[26] = 0.0f;
                            qs_frag_early_cg1[27] = 0.0f;
                            qs_frag_early_cg1[28] = 0.0f;
                            qs_frag_early_cg1[29] = 0.0f;
                            qs_frag_early_cg1[30] = 0.0f;
                            qs_frag_early_cg1[31] = 0.0f;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(qs_addr_early_cg1), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[15])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[16])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[17])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[18])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[19])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[20])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[21])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[22])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[23])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[24])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[25])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[26])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[27])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[28])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[29])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[30])), "r"(*reinterpret_cast<const uint32_t*>(&qs_frag_early_cg1[31])));
                        int qs_addr_hi_cg1 = qs_addr_early_cg1 + 1048576;
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(qs_addr_hi_cg1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        {
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], reinterpret_cast<const float2*>(cumprod_values_cg1)[_ls]);
                            const float2 _scale2_4 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_4);
                        }
                        if (source_final_block_cg1 != 0) {
                            _tmem_load_1[0] = 0.0f;
                            _tmem_load_1[1] = 0.0f;
                            _tmem_load_1[2] = 0.0f;
                            _tmem_load_1[3] = 0.0f;
                            _tmem_load_1[4] = 0.0f;
                            _tmem_load_1[5] = 0.0f;
                            _tmem_load_1[6] = 0.0f;
                            _tmem_load_1[7] = 0.0f;
                            _tmem_load_1[8] = 0.0f;
                            _tmem_load_1[9] = 0.0f;
                            _tmem_load_1[10] = 0.0f;
                            _tmem_load_1[11] = 0.0f;
                            _tmem_load_1[12] = 0.0f;
                            _tmem_load_1[13] = 0.0f;
                            _tmem_load_1[14] = 0.0f;
                            _tmem_load_1[15] = 0.0f;
                            _tmem_load_1[16] = 0.0f;
                            _tmem_load_1[17] = 0.0f;
                            _tmem_load_1[18] = 0.0f;
                            _tmem_load_1[19] = 0.0f;
                            _tmem_load_1[20] = 0.0f;
                            _tmem_load_1[21] = 0.0f;
                            _tmem_load_1[22] = 0.0f;
                            _tmem_load_1[23] = 0.0f;
                            _tmem_load_1[24] = 0.0f;
                            _tmem_load_1[25] = 0.0f;
                            _tmem_load_1[26] = 0.0f;
                            _tmem_load_1[27] = 0.0f;
                            _tmem_load_1[28] = 0.0f;
                            _tmem_load_1[29] = 0.0f;
                            _tmem_load_1[30] = 0.0f;
                            _tmem_load_1[31] = 0.0f;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(qs_addr_hi_cg1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(q_state_acc_empty_addr + (q_state_stage_cg1) * 8);
                    q_state_stage_cg1 += 1;
                    if (q_state_stage_cg1 == 1) { q_state_stage_cg1 = 0; q_state_full_phase_cg1 ^= 1; }
                    mbarrier_wait(cg1_acc_full_addr + (cg1_acc_stage_cg1) * 8, cg1_acc_full_phase_cg1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(load_v_empty_addr + (v_stage_cg1) * 8);
                    }
                    int nv_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_cg1;
                    int nv_addr_hi_cg1 = nv_addr_lo_cg1 + 1048576;
                    float nv_frag_lo_cg1[32];
                    float nv_frag_hi_cg1[32];
                    {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_lo_cg1[31]))
                            : "r"(nv_addr_lo_cg1));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_hi_cg1[31]))
                            : "r"(nv_addr_hi_cg1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                    mbarrier_arrive(cg1_acc_empty_addr + (cg1_acc_stage_cg1) * 8);
                    cg1_acc_stage_cg1 += 1;
                    if (cg1_acc_stage_cg1 == 1) { cg1_acc_stage_cg1 = 0; cg1_acc_full_phase_cg1 ^= 1; }
                    unsigned int nv_bits_lo_cg1[16];
                    unsigned int nv_bits_hi_cg1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(nv_frag_lo_cg1[_lp*2 + 0], nv_frag_lo_cg1[_lp*2+1 + 0]));
                        nv_bits_lo_cg1[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(nv_frag_hi_cg1[_lp*2 + 0], nv_frag_hi_cg1[_lp*2+1 + 0]));
                        nv_bits_hi_cg1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_lo_cg1[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_bits_hi_cg1[15])));
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(nv_frag_lo_cg1)[_ls], reinterpret_cast<const float2*>(decay_scales_cg1)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(nv_frag_hi_cg1)[_ls], reinterpret_cast<const float2*>(decay_scales_cg1)[_ls]);
                    unsigned int decay_bits_lo_cg1[16];
                    unsigned int decay_bits_hi_cg1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(nv_frag_lo_cg1[_lp*2 + 0], nv_frag_lo_cg1[_lp*2+1 + 0]));
                        decay_bits_lo_cg1[_lp] = *(uint32_t*)&_h2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(nv_frag_hi_cg1[_lp*2 + 0], nv_frag_hi_cg1[_lp*2+1 + 0]));
                        decay_bits_hi_cg1[_lp] = *(uint32_t*)&_h2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + 32 + (unsigned int)tmem_row_base_cg1), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_lo_cg1[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + 448 + 32 + (unsigned int)tmem_row_base_cg1 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&decay_bits_hi_cg1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    {
                        mbarrier_arrive(nv_ready_addr);
                        mbarrier_arrive(decay_v_ready_addr);
                    }
                    mbarrier_wait(o_store_empty_addr + (o_stage_cg1) * 8, o_empty_phase_cg1);
                    mbarrier_wait(q_state_acc_full_addr + (q_state_stage_cg1) * 8, q_state_full_phase_cg1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int o_stage_addr_cg1 = smem_o_addr + o_stage_cg1 * 16384;
                    {
                        #pragma unroll
                        for (int o_half_cg1 = 0; o_half_cg1 < 2; o_half_cg1++) {
                            int o_addr_cg1 = taddr + 128 + (unsigned int)tmem_row_base_cg1 + (unsigned int)(o_half_cg1 * 16 << 16);
                            float _tmem_load_3[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                                : "r"(o_addr_cg1));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            unsigned int o_bits_cg1[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                                o_bits_cg1[_lp] = *(uint32_t*)&_h2;
                            }
                            #pragma unroll
                            for (int o_group_cg1 = 0; o_group_cg1 < 4; o_group_cg1++) {
                                int o_matrix_cg1 = lane / 8;
                                int o_row_addr_cg1 = lane & 7;
                                int o_dim_base_cg1 = warp_cg1 * 32 + o_half_cg1 * 16 + (o_matrix_cg1 & 1) * 8;
                                int o_token_base_cg1 = o_group_cg1 * 16 + o_matrix_cg1 / 2 * 8;
                                int o_token_addr_cg1 = o_token_base_cg1 + o_row_addr_cg1;
                                int o_token_pair_cg1 = o_token_addr_cg1 / 2;
                                int o_token_parity_cg1 = o_token_addr_cg1 & 1;
                                int o_raw_row_cg1 = o_token_pair_cg1 + o_dim_base_cg1 / 64 * 32;
                                int o_raw_col_cg1 = (o_dim_base_cg1 & 63 ^ (o_token_pair_cg1 & 3) << 4 ^ o_token_parity_cg1 << 3) + o_token_parity_cg1 * 64;
                                int o_offset_cg1 = (o_raw_row_cg1 * 128 + o_raw_col_cg1) * 2;
                                const int o_word_cg1 = o_group_cg1 * 4;
                                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1 + o_offset_cg1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&o_bits_cg1[o_word_cg1])), "r"(*reinterpret_cast<const uint32_t*>(&o_bits_cg1[o_word_cg1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&o_bits_cg1[o_word_cg1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&o_bits_cg1[o_word_cg1 + 3]))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        mbarrier_arrive(q_state_acc_empty_addr + (q_state_stage_cg1) * 8);
                        q_state_stage_cg1 += 1;
                        if (q_state_stage_cg1 == 1) { q_state_stage_cg1 = 0; q_state_full_phase_cg1 ^= 1; }
                    }
                    mbarrier_arrive(o_store_ready_addr + (o_stage_cg1) * 8);
                    kv_stage_cg1 += 1;
                    if (kv_stage_cg1 == 1) { kv_stage_cg1 = 0; kv_full_phase_cg1 ^= 1; }
                    gate_stage_cg1 += 1;
                    if (gate_stage_cg1 == 5) { gate_stage_cg1 = 0; gate_phase_cg1 ^= 1; }
                    v_stage_cg1 += 1;
                    if (v_stage_cg1 == 2) { v_stage_cg1 = 0; v_phase_cg1 ^= 1; }
                    o_stage_cg1 += 1;
                    if (o_stage_cg1 == 2) { o_stage_cg1 = 0; o_empty_phase_cg1 ^= 1; }
                }
                mbarrier_wait(kv_acc_full_addr + (kv_stage_cg1) * 8, kv_full_phase_cg1);
                kv_stage_cg1 += 1;
                if (kv_stage_cg1 == 1) { kv_stage_cg1 = 0; kv_full_phase_cg1 ^= 1; }
                if (chunk_in_seq_1 == num_cp_chunks_1 - 1) {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_state_values_cg1[128];
                    tmem_ld_x32(&final_state_values_cg1[0], taddr + (unsigned int)tmem_row_base_cg1);
                    tmem_ld_x32(&final_state_values_cg1[32], taddr + (unsigned int)tmem_row_base_cg1 + 32);
                    tmem_ld_x32(&final_state_values_cg1[64], taddr + (unsigned int)tmem_row_base_cg1 + 64);
                    tmem_ld_x32(&final_state_values_cg1[96], taddr + (unsigned int)tmem_row_base_cg1 + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                mbarrier_arrive(kv_acc_empty_addr + (kv_stage_cg1) * 8);
                #pragma unroll
                for (int __2 = 0; __2 < 2; __2++) {
                    mbarrier_wait(o_store_empty_addr + (o_stage_cg1) * 8, o_empty_phase_cg1);
                    o_stage_cg1 += 1;
                    if (o_stage_cg1 == 2) { o_stage_cg1 = 0; o_empty_phase_cg1 ^= 1; }
                }
                mbarrier_wait(state_input_empty_addr + (state_input_stage_cg1) * 8, state_input_empty_phase_cg1);
                state_input_stage_cg1 += 1;
                if (state_input_stage_cg1 == 1) { state_input_stage_cg1 = 0; state_input_empty_phase_cg1 ^= 1; }
            }
        }
    // ---- Role: qk_mma ----
    } else if (warp == 8) {
        { // qk_mma_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&T))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&O))) : "memory");
            int sab_head_2 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_2 = blockIdx.x / num_sab_heads;
            int seq_idx_2 = blockIdx.y;
            int seq_start_2 = (int)cu_seqlens[seq_idx_2];
            int seq_end_2 = (int)cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = seq_end_2 - seq_start_2;
            int num_cp_chunks_2 = (seq_len_2 + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len_2 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining_2 = seq_len_2 - chunk_in_seq_2 * cp_chunk_len;
                chunk_len_2 = ((remaining_2 < cp_chunk_len) ? remaining_2 : cp_chunk_len);
            }
            int num_valid_blocks_2 = (chunk_len_2 + 64 - 1) / 64;
            int num_padded_blocks_2 = (chunk_len_2 + 128 - 1) / 128 * 2;
            int chunk_start_2 = seq_start_2 + chunk_in_seq_2 * cp_chunk_len;
            int chunk_end_2 = chunk_start_2 + chunk_len_2;
            int prefix_items_2 = ((seq_idx_2 < seq_start_2) ? seq_idx_2 : seq_start_2);
            int cp_chunk_2 = prefix_items_2 + (seq_start_2 - prefix_items_2) / cp_chunk_len + chunk_in_seq_2;
            int t_blocks_per_chunk_2 = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items_2 = ((seq_idx_2 < seq_start_2) ? seq_idx_2 : seq_start_2);
            int t_block_start_2 = t_prefix_items_2 + (seq_start_2 - t_prefix_items_2) / 64 + chunk_in_seq_2 * t_blocks_per_chunk_2;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                unsigned int q_stage_qk = 0;
                unsigned int q_phase_qk = 0;
                unsigned int k_stage_qk = 0;
                unsigned int k_phase_qk = 0;
                unsigned int acc_stage_qk = 0;
                unsigned int acc_phase_qk = 1;
                #pragma unroll 2
                for (int __3 = 0; __3 < num_padded_blocks_2; __3++) {
                    {
                        mbarrier_wait(load_q_full_addr + (q_stage_qk) * 8, q_phase_qk);
                        mbarrier_wait(load_k_full_addr + (k_stage_qk) * 8, k_phase_qk);
                    }
                    mbarrier_wait(cg0_acc_empty_addr + (acc_stage_qk) * 8, acc_phase_qk);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_qk) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_qk) * 1024);
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
                    "mov.b32 id, 68157456;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_cg0_acc + (acc_stage_qk * 64))), "r"(0));
                    elect_commit(cg0_acc_full_addr + (acc_stage_qk) * 8);
                    elect_commit(load_q_empty_addr + (q_stage_qk) * 8);
                    elect_commit(load_k_empty_addr + (k_stage_qk) * 8);
                    q_stage_qk += 1;
                    if (q_stage_qk == 2) { q_stage_qk = 0; q_phase_qk ^= 1; }
                    k_stage_qk += 1;
                    if (k_stage_qk == 3) { k_stage_qk = 0; k_phase_qk ^= 1; }
                    acc_stage_qk += 1;
                    if (acc_stage_qk == 2) { acc_stage_qk = 0; acc_phase_qk ^= 1; }
                }
                acc_stage_qk += 1;
                if (acc_stage_qk == 2) { acc_stage_qk = 0; acc_phase_qk ^= 1; }
                mbarrier_wait(cg0_acc_empty_addr + (acc_stage_qk) * 8, acc_phase_qk);
            }
        }
    // ---- Role: tma_qkvt ----
    } else if (warp == 9) {
        { // tma_qkvt_main
            int sab_head_3 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_3 = blockIdx.x / num_sab_heads;
            int seq_idx_3 = blockIdx.y;
            int seq_start_3 = (int)cu_seqlens[seq_idx_3];
            int seq_end_3 = (int)cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = seq_end_3 - seq_start_3;
            int num_cp_chunks_3 = (seq_len_3 + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len_3 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining_3 = seq_len_3 - chunk_in_seq_3 * cp_chunk_len;
                chunk_len_3 = ((remaining_3 < cp_chunk_len) ? remaining_3 : cp_chunk_len);
            }
            int num_valid_blocks_3 = (chunk_len_3 + 64 - 1) / 64;
            int num_padded_blocks_3 = (chunk_len_3 + 128 - 1) / 128 * 2;
            int chunk_start_3 = seq_start_3 + chunk_in_seq_3 * cp_chunk_len;
            int chunk_end_3 = chunk_start_3 + chunk_len_3;
            int prefix_items_3 = ((seq_idx_3 < seq_start_3) ? seq_idx_3 : seq_start_3);
            int cp_chunk_3 = prefix_items_3 + (seq_start_3 - prefix_items_3) / cp_chunk_len + chunk_in_seq_3;
            int t_blocks_per_chunk_3 = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items_3 = ((seq_idx_3 < seq_start_3) ? seq_idx_3 : seq_start_3);
            int t_block_start_3 = t_prefix_items_3 + (seq_start_3 - t_prefix_items_3) / 64 + chunk_in_seq_3 * t_blocks_per_chunk_3;
            int cta_slot_base_tma = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * 640;
            if (elect_sync()) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&Q));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + cta_slot_base_tma));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&K));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 128)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&V));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 256)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            if (elect_sync()) {
                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base_tma)), "r"((uint32_t)(chunk_end_3)) : "memory");
                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 128))), "r"((uint32_t)(chunk_end_3)) : "memory");
                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 256))), "r"((uint32_t)(chunk_end_3)) : "memory");
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            unsigned int _phase_load_k_empty = 1;
            unsigned int _phase_load_q_empty = 1;
            unsigned int _phase_load_t_empty = 1;
            unsigned int _phase_load_v_empty = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int q_head = sab_head_3;
                int k_head = sab_head_3;
                int v_head = sab_head_3;
                unsigned int q_stage = 0;
                unsigned int k_stage = 0;
                unsigned int v_stage = 0;
                unsigned int t_stage = 0;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int block_idx = 0; block_idx < num_padded_blocks_3; block_idx++) {
                        int block_offset = chunk_start_3 + block_idx * 64;
                        mbarrier_wait(load_k_empty_addr + (k_stage) * 8, _phase_load_k_empty);
                        if (block_idx == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 128))) : "memory");
                        }
                        mbarrier_arrive_expect_tx(load_k_full_addr + (k_stage) * 8, 16384);
                        tma_4d_gmem2smem(smem_k_addr + k_stage * 16384, tensormap_workspace + (cta_slot_base_tma + 128), 0, block_offset, 0, k_head, load_k_full_addr + (k_stage) * 8);
                        k_stage += 1;
                        if (k_stage == 3) { k_stage = 0; _phase_load_k_empty ^= 1; }
                        mbarrier_wait(load_q_empty_addr + (q_stage) * 8, _phase_load_q_empty);
                        if (block_idx == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base_tma)) : "memory");
                        }
                        mbarrier_arrive_expect_tx(load_q_full_addr + (q_stage) * 8, 16384);
                        tma_4d_gmem2smem(smem_q_addr + q_stage * 16384, tensormap_workspace + cta_slot_base_tma, 0, block_offset, 0, q_head, load_q_full_addr + (q_stage) * 8);
                        q_stage += 1;
                        if (q_stage == 2) { q_stage = 0; _phase_load_q_empty ^= 1; }
                        int t_block = block_idx;
                        {
                        }
                        mbarrier_wait(load_t_empty_addr + (t_stage) * 8, _phase_load_t_empty);
                        mbarrier_arrive_expect_tx(load_t_full_addr + (t_stage) * 8, 8192);
                        tma_4d_gmem2smem(smem_t_addr + t_stage * 8192, (&T), 0, 0, sab_head_3, t_block_start_3 + t_block, load_t_full_addr + (t_stage) * 8);
                        t_stage += 1;
                        if (t_stage == 4) { t_stage = 0; _phase_load_t_empty ^= 1; }
                        mbarrier_wait(load_v_empty_addr + (v_stage) * 8, _phase_load_v_empty);
                        if (block_idx == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_tma + 256))) : "memory");
                        }
                        mbarrier_arrive_expect_tx(load_v_full_addr + (v_stage) * 8, 16384);
                        tma_4d_gmem2smem(smem_v_addr + v_stage * 16384, tensormap_workspace + (cta_slot_base_tma + 256), 0, block_offset, 0, v_head, load_v_full_addr + (v_stage) * 8);
                        v_stage += 1;
                        if (v_stage == 2) { v_stage = 0; _phase_load_v_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int __4 = 0; __4 < 2; __4++) {
                        mbarrier_wait(load_q_empty_addr + (q_stage) * 8, _phase_load_q_empty);
                        q_stage += 1;
                        if (q_stage == 2) { q_stage = 0; _phase_load_q_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int __5 = 0; __5 < 3; __5++) {
                        mbarrier_wait(load_k_empty_addr + (k_stage) * 8, _phase_load_k_empty);
                        k_stage += 1;
                        if (k_stage == 3) { k_stage = 0; _phase_load_k_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int __6 = 0; __6 < 2; __6++) {
                        mbarrier_wait(load_v_empty_addr + (v_stage) * 8, _phase_load_v_empty);
                        v_stage += 1;
                        if (v_stage == 2) { v_stage = 0; _phase_load_v_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int __7 = 0; __7 < 4; __7++) {
                        mbarrier_wait(load_t_empty_addr + (t_stage) * 8, _phase_load_t_empty);
                        t_stage += 1;
                        if (t_stage == 4) { t_stage = 0; _phase_load_t_empty ^= 1; }
                    }
                }
            }
        }
    // ---- Role: state_mma ----
    } else if (warp == 10) {
        { // state_mma_main
            int sab_head_4 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_4 = blockIdx.x / num_sab_heads;
            int seq_idx_4 = blockIdx.y;
            int seq_start_4 = (int)cu_seqlens[seq_idx_4];
            int seq_end_4 = (int)cu_seqlens[seq_idx_4 + 1];
            int seq_len_4 = seq_end_4 - seq_start_4;
            int num_cp_chunks_4 = (seq_len_4 + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len_4 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining_4 = seq_len_4 - chunk_in_seq_4 * cp_chunk_len;
                chunk_len_4 = ((remaining_4 < cp_chunk_len) ? remaining_4 : cp_chunk_len);
            }
            int num_valid_blocks_4 = (chunk_len_4 + 64 - 1) / 64;
            int num_padded_blocks_4 = (chunk_len_4 + 128 - 1) / 128 * 2;
            int chunk_start_4 = seq_start_4 + chunk_in_seq_4 * cp_chunk_len;
            int chunk_end_4 = chunk_start_4 + chunk_len_4;
            int prefix_items_4 = ((seq_idx_4 < seq_start_4) ? seq_idx_4 : seq_start_4);
            int cp_chunk_4 = prefix_items_4 + (seq_start_4 - prefix_items_4) / cp_chunk_len + chunk_in_seq_4;
            int t_blocks_per_chunk_4 = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items_4 = ((seq_idx_4 < seq_start_4) ? seq_idx_4 : seq_start_4);
            int t_block_start_4 = t_prefix_items_4 + (seq_start_4 - t_prefix_items_4) / 64 + chunk_in_seq_4 * t_blocks_per_chunk_4;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                unsigned int q_stage_state = 0;
                unsigned int q_phase_state = 0;
                unsigned int k_stage_state = 0;
                unsigned int k_phase_state = 0;
                unsigned int ainv_stage_state = 0;
                unsigned int ainv_ready_phase_state = 0;
                unsigned int qk_stage_state = 0;
                unsigned int qk_ready_phase_state = 0;
                unsigned int q_state_stage_state = 0;
                unsigned int q_state_empty_phase_state = 1;
                unsigned int kv_stage_state = 0;
                unsigned int kv_empty_phase_state = 1;
                unsigned int state_input_stage_state = 0;
                unsigned int state_input_phase_state = 0;
                unsigned int vks_stage_state = 0;
                unsigned int vks_phase_state = 0;
                unsigned int nv_stage_state = 0;
                unsigned int nv_phase_state = 0;
                unsigned int decay_v_stage_state = 0;
                unsigned int decay_v_phase_state = 0;
                unsigned int cg1_acc_stage_state = 0;
                unsigned int cg1_acc_empty_phase_state = 1;
                kv_stage_state += 1;
                if (kv_stage_state == 1) { kv_stage_state = 0; kv_empty_phase_state ^= 1; }
                #pragma unroll 1
                for (int __8 = 0; __8 < num_padded_blocks_4; __8++) {
                    {
                        mbarrier_wait(load_q_full_addr + (q_stage_state) * 8, q_phase_state);
                        mbarrier_wait(load_k_full_addr + (k_stage_state) * 8, k_phase_state);
                    }
                    {
                        mbarrier_wait(cg1_acc_empty_addr, cg1_acc_empty_phase_state);
                        mbarrier_wait(state_input_ready_addr, state_input_phase_state);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_state) * 1024);
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_cg1_acc), "r"(_mma_b_lo_1), "r"(tmem_tmem_state_input), "r"(0));
                    elect_commit(cg1_acc_full_addr);
                    cg1_acc_stage_state += 1;
                    if (cg1_acc_stage_state == 1) { cg1_acc_stage_state = 0; cg1_acc_empty_phase_state ^= 1; }
                    mbarrier_wait(q_state_acc_empty_addr + (q_state_stage_state) * 8, q_state_empty_phase_state);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_state) * 1024);
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_2), "r"(tmem_tmem_state_input), "r"(0));
                    elect_commit(q_state_acc_full_addr);
                    q_state_stage_state += 1;
                    if (q_state_stage_state == 1) { q_state_stage_state = 0; q_state_empty_phase_state ^= 1; }
                    elect_commit(state_input_empty_addr);
                    elect_commit(load_q_empty_addr + (q_stage_state) * 8);
                    {
                        mbarrier_wait(cg1_acc_empty_addr, cg1_acc_empty_phase_state);
                        mbarrier_wait(vks_ready_addr, vks_phase_state);
                        mbarrier_wait(ainv_ready_addr + (ainv_stage_state) * 8, ainv_ready_phase_state);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_3 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_state) * 512);
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
                    "mov.b32 id, 135266320;\n\t"
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
                    :: "r"(tmem_tmem_cg1_acc), "r"(_mma_b_lo_3), "r"(tmem_tmem_shared_input), "r"(0));
                    elect_commit(cg1_acc_full_addr);
                    cg1_acc_stage_state += 1;
                    if (cg1_acc_stage_state == 1) { cg1_acc_stage_state = 0; cg1_acc_empty_phase_state ^= 1; }
                    elect_commit(ainv_empty_addr + (ainv_stage_state) * 8);
                    {
                        mbarrier_wait(q_state_acc_empty_addr + (q_state_stage_state) * 8, q_state_empty_phase_state);
                        mbarrier_wait(qk_ready_addr + (qk_stage_state) * 8, qk_ready_phase_state);
                        mbarrier_wait(nv_ready_addr, nv_phase_state);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_4 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_state) * 512);
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
                    "mov.b32 id, 135266320;\n\t"
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
                    :: "r"(tmem_tmem_q_state), "r"(_mma_b_lo_4), "r"(tmem_tmem_shared_input), "r"(1));
                    elect_commit(q_state_acc_full_addr);
                    elect_commit(qk_empty_addr + (qk_stage_state) * 8);
                    mbarrier_wait(kv_acc_empty_addr + (kv_stage_state) * 8, kv_empty_phase_state);
                    mbarrier_wait(decay_v_ready_addr, decay_v_phase_state);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_k_trans_addr) >> 4) & 0x3FFF) | 0x2000000) + (k_stage_state) * 1024);
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state), "r"(_mma_b_lo_5), "r"(tmem_tmem_shared_input + 32), "r"(1));
                    elect_commit(kv_acc_full_addr);
                    elect_commit(load_k_empty_addr + (k_stage_state) * 8);
                    q_stage_state += 1;
                    if (q_stage_state == 2) { q_stage_state = 0; q_phase_state ^= 1; }
                    k_stage_state += 1;
                    if (k_stage_state == 3) { k_stage_state = 0; k_phase_state ^= 1; }
                    ainv_stage_state += 1;
                    if (ainv_stage_state == 3) { ainv_stage_state = 0; ainv_ready_phase_state ^= 1; }
                    qk_stage_state += 1;
                    if (qk_stage_state == 2) { qk_stage_state = 0; qk_ready_phase_state ^= 1; }
                    q_state_stage_state += 1;
                    if (q_state_stage_state == 1) { q_state_stage_state = 0; q_state_empty_phase_state ^= 1; }
                    kv_stage_state += 1;
                    if (kv_stage_state == 1) { kv_stage_state = 0; kv_empty_phase_state ^= 1; }
                    state_input_stage_state += 1;
                    if (state_input_stage_state == 1) { state_input_stage_state = 0; state_input_phase_state ^= 1; }
                    vks_stage_state += 1;
                    if (vks_stage_state == 1) { vks_stage_state = 0; vks_phase_state ^= 1; }
                    nv_stage_state += 1;
                    if (nv_stage_state == 1) { nv_stage_state = 0; nv_phase_state ^= 1; }
                    decay_v_stage_state += 1;
                    if (decay_v_stage_state == 1) { decay_v_stage_state = 0; decay_v_phase_state ^= 1; }
                }
                mbarrier_wait(cg1_acc_empty_addr, cg1_acc_empty_phase_state);
                mbarrier_wait(q_state_acc_empty_addr + (q_state_stage_state) * 8, q_state_empty_phase_state);
                mbarrier_wait(kv_acc_empty_addr + (kv_stage_state) * 8, kv_empty_phase_state);
            }
        }
    // ---- Role: gate_epilogue ----
    } else if (warp == 11) {
        { // gate_epilogue_main
            int sab_head_5 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_5 = blockIdx.x / num_sab_heads;
            int seq_idx_5 = blockIdx.y;
            int seq_start_5 = (int)cu_seqlens[seq_idx_5];
            int seq_end_5 = (int)cu_seqlens[seq_idx_5 + 1];
            int seq_len_5 = seq_end_5 - seq_start_5;
            int num_cp_chunks_5 = (seq_len_5 + cp_chunk_len - 1) / cp_chunk_len;
            int chunk_len_5 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                int remaining_5 = seq_len_5 - chunk_in_seq_5 * cp_chunk_len;
                chunk_len_5 = ((remaining_5 < cp_chunk_len) ? remaining_5 : cp_chunk_len);
            }
            int num_valid_blocks_5 = (chunk_len_5 + 64 - 1) / 64;
            int num_padded_blocks_5 = (chunk_len_5 + 128 - 1) / 128 * 2;
            int chunk_start_5 = seq_start_5 + chunk_in_seq_5 * cp_chunk_len;
            int chunk_end_5 = chunk_start_5 + chunk_len_5;
            int prefix_items_5 = ((seq_idx_5 < seq_start_5) ? seq_idx_5 : seq_start_5);
            int cp_chunk_5 = prefix_items_5 + (seq_start_5 - prefix_items_5) / cp_chunk_len + chunk_in_seq_5;
            int t_blocks_per_chunk_5 = (cp_chunk_len + 64 - 1) / 64;
            int t_prefix_items_5 = ((seq_idx_5 < seq_start_5) ? seq_idx_5 : seq_start_5);
            int t_block_start_5 = t_prefix_items_5 + (seq_start_5 - t_prefix_items_5) / 64 + chunk_in_seq_5 * t_blocks_per_chunk_5;
            int cta_slot_base_epi = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * 640;
            if (elect_sync()) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&O));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 512)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            if (elect_sync()) {
                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 512))), "r"((uint32_t)(chunk_end_5)) : "memory");
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 512))) : "memory");
            }
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + cp_chunk_len - 1) / cp_chunk_len) {
                unsigned int gate_stage = 0;
                unsigned int gate_empty_phase = 1;
                unsigned int o_epi_stage = 0;
                unsigned int o_ready_phase = 0;
                #pragma unroll
                for (int prefetch_gate = 0; prefetch_gate < 2; prefetch_gate++) {
                    mbarrier_wait(load_gate_empty_addr + (gate_stage) * 8, gate_empty_phase);
                    int row0 = lane;
                    int row1 = row0 + 32;
                    int token0 = chunk_start_5 + prefetch_gate * 64 + row0;
                    int token1 = token0 + 32;
                    float gate0 = 1.0f;
                    float gate1 = 1.0f;
                    {
                        gate0 = alpha[token0 * num_sab_heads + sab_head_5];
                        gate1 = alpha[token1 * num_sab_heads + sab_head_5];
                    }
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(gate0 + 1e-10f));
                    float log0 = _log2_0;
                    float _log2_1;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(gate1 + 1e-10f));
                    float log1 = _log2_1;
                    float _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, log0, 1, 32);
                    float prior0 = _shfl_up_0;
                    float _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, log1, 1, 32);
                    float prior1 = _shfl_up_1;
                    if (lane >= 1) {
                        log0 = log0 + prior0;
                        log1 = log1 + prior1;
                    }
                    float _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, log0, 2, 32);
                    float prior0_0 = _shfl_up_2;
                    float _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, log1, 2, 32);
                    float prior1_1 = _shfl_up_3;
                    if (lane >= 2) {
                        log0 = log0 + prior0_0;
                        log1 = log1 + prior1_1;
                    }
                    float _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, log0, 4, 32);
                    float prior0_2 = _shfl_up_4;
                    float _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, log1, 4, 32);
                    float prior1_3 = _shfl_up_5;
                    if (lane >= 4) {
                        log0 = log0 + prior0_2;
                        log1 = log1 + prior1_3;
                    }
                    float _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, log0, 8, 32);
                    float prior0_4 = _shfl_up_6;
                    float _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, log1, 8, 32);
                    float prior1_5 = _shfl_up_7;
                    if (lane >= 8) {
                        log0 = log0 + prior0_4;
                        log1 = log1 + prior1_5;
                    }
                    float _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, log0, 16, 32);
                    float prior0_6 = _shfl_up_8;
                    float _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, log1, 16, 32);
                    float prior1_7 = _shfl_up_9;
                    if (lane >= 16) {
                        log0 = log0 + prior0_6;
                        log1 = log1 + prior1_7;
                    }
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, log0, 31);
                    log1 = log1 + _shfl_0;
                    int stage_base = gate_stage * 64;
                    smem_cumsumlog[stage_base + row0] = log0;
                    smem_cumsumlog[stage_base + row1] = log1;
                    float _exp2_0 = approx_exp2(log0);
                    smem_cumprod[stage_base + row0] = _exp2_0;
                    float _exp2_1 = approx_exp2(log1);
                    smem_cumprod[stage_base + row1] = _exp2_1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(load_gate_full_addr + (gate_stage) * 8);
                    gate_stage += 1;
                    if (gate_stage == 5) { gate_stage = 0; gate_empty_phase ^= 1; }
                }
                if (num_padded_blocks_5 > 2) {
                    #pragma unroll
                    for (int prefetch_gate_1 = 2; prefetch_gate_1 < 4; prefetch_gate_1++) {
                        mbarrier_wait(load_gate_empty_addr + (gate_stage) * 8, gate_empty_phase);
                        int row0_1 = lane;
                        int row1_1 = row0_1 + 32;
                        int token0_1 = chunk_start_5 + prefetch_gate_1 * 64 + row0_1;
                        int token1_1 = token0_1 + 32;
                        float gate0_1 = 1.0f;
                        float gate1_1 = 1.0f;
                        {
                            gate0_1 = alpha[token0_1 * num_sab_heads + sab_head_5];
                            gate1_1 = alpha[token1_1 * num_sab_heads + sab_head_5];
                        }
                        float _log2_2;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(gate0_1 + 1e-10f));
                        float log0_1 = _log2_2;
                        float _log2_3;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(gate1_1 + 1e-10f));
                        float log1_1 = _log2_3;
                        float _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, log0_1, 1, 32);
                        float prior0_1 = _shfl_up_10;
                        float _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, log1_1, 1, 32);
                        float prior1_2 = _shfl_up_11;
                        if (lane >= 1) {
                            log0_1 = log0_1 + prior0_1;
                            log1_1 = log1_1 + prior1_2;
                        }
                        float _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, log0_1, 2, 32);
                        float prior0_0_1 = _shfl_up_12;
                        float _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, log1_1, 2, 32);
                        float prior1_1_1 = _shfl_up_13;
                        if (lane >= 2) {
                            log0_1 = log0_1 + prior0_0_1;
                            log1_1 = log1_1 + prior1_1_1;
                        }
                        float _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, log0_1, 4, 32);
                        float prior0_2_1 = _shfl_up_14;
                        float _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, log1_1, 4, 32);
                        float prior1_3_1 = _shfl_up_15;
                        if (lane >= 4) {
                            log0_1 = log0_1 + prior0_2_1;
                            log1_1 = log1_1 + prior1_3_1;
                        }
                        float _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, log0_1, 8, 32);
                        float prior0_4_1 = _shfl_up_16;
                        float _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, log1_1, 8, 32);
                        float prior1_5_1 = _shfl_up_17;
                        if (lane >= 8) {
                            log0_1 = log0_1 + prior0_4_1;
                            log1_1 = log1_1 + prior1_5_1;
                        }
                        float _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, log0_1, 16, 32);
                        float prior0_6_1 = _shfl_up_18;
                        float _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, log1_1, 16, 32);
                        float prior1_7_1 = _shfl_up_19;
                        if (lane >= 16) {
                            log0_1 = log0_1 + prior0_6_1;
                            log1_1 = log1_1 + prior1_7_1;
                        }
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, log0_1, 31);
                        log1_1 = log1_1 + _shfl_1;
                        int stage_base_1 = gate_stage * 64;
                        smem_cumsumlog[stage_base_1 + row0_1] = log0_1;
                        smem_cumsumlog[stage_base_1 + row1_1] = log1_1;
                        float _exp2_2 = approx_exp2(log0_1);
                        smem_cumprod[stage_base_1 + row0_1] = _exp2_2;
                        float _exp2_3 = approx_exp2(log1_1);
                        smem_cumprod[stage_base_1 + row1_1] = _exp2_3;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(load_gate_full_addr + (gate_stage) * 8);
                        gate_stage += 1;
                        if (gate_stage == 5) { gate_stage = 0; gate_empty_phase ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int block_epi = 0; block_epi < num_padded_blocks_5; block_epi++) {
                    int next_gate = block_epi + 4;
                    if (next_gate < num_padded_blocks_5) {
                        mbarrier_wait(load_gate_empty_addr + (gate_stage) * 8, gate_empty_phase);
                        int row0_2 = lane;
                        int row1_2 = row0_2 + 32;
                        int token0_2 = chunk_start_5 + next_gate * 64 + row0_2;
                        int token1_2 = token0_2 + 32;
                        float gate0_2 = 1.0f;
                        float gate1_2 = 1.0f;
                        {
                            gate0_2 = alpha[token0_2 * num_sab_heads + sab_head_5];
                            gate1_2 = alpha[token1_2 * num_sab_heads + sab_head_5];
                        }
                        float _log2_4;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(gate0_2 + 1e-10f));
                        float log0_2 = _log2_4;
                        float _log2_5;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(gate1_2 + 1e-10f));
                        float log1_2 = _log2_5;
                        float _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, log0_2, 1, 32);
                        float prior0_3 = _shfl_up_20;
                        float _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, log1_2, 1, 32);
                        float prior1_4 = _shfl_up_21;
                        if (lane >= 1) {
                            log0_2 = log0_2 + prior0_3;
                            log1_2 = log1_2 + prior1_4;
                        }
                        float _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, log0_2, 2, 32);
                        float prior0_0_2 = _shfl_up_22;
                        float _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, log1_2, 2, 32);
                        float prior1_1_2 = _shfl_up_23;
                        if (lane >= 2) {
                            log0_2 = log0_2 + prior0_0_2;
                            log1_2 = log1_2 + prior1_1_2;
                        }
                        float _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, log0_2, 4, 32);
                        float prior0_2_2 = _shfl_up_24;
                        float _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, log1_2, 4, 32);
                        float prior1_3_2 = _shfl_up_25;
                        if (lane >= 4) {
                            log0_2 = log0_2 + prior0_2_2;
                            log1_2 = log1_2 + prior1_3_2;
                        }
                        float _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, log0_2, 8, 32);
                        float prior0_4_2 = _shfl_up_26;
                        float _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, log1_2, 8, 32);
                        float prior1_5_2 = _shfl_up_27;
                        if (lane >= 8) {
                            log0_2 = log0_2 + prior0_4_2;
                            log1_2 = log1_2 + prior1_5_2;
                        }
                        float _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, log0_2, 16, 32);
                        float prior0_6_2 = _shfl_up_28;
                        float _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, log1_2, 16, 32);
                        float prior1_7_2 = _shfl_up_29;
                        if (lane >= 16) {
                            log0_2 = log0_2 + prior0_6_2;
                            log1_2 = log1_2 + prior1_7_2;
                        }
                        float _shfl_2 = __shfl_sync(0xFFFFFFFF, log0_2, 31);
                        log1_2 = log1_2 + _shfl_2;
                        int stage_base_2 = gate_stage * 64;
                        smem_cumsumlog[stage_base_2 + row0_2] = log0_2;
                        smem_cumsumlog[stage_base_2 + row1_2] = log1_2;
                        float _exp2_4 = approx_exp2(log0_2);
                        smem_cumprod[stage_base_2 + row0_2] = _exp2_4;
                        float _exp2_5 = approx_exp2(log1_2);
                        smem_cumprod[stage_base_2 + row1_2] = _exp2_5;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(load_gate_full_addr + (gate_stage) * 8);
                        gate_stage += 1;
                        if (gate_stage == 5) { gate_stage = 0; gate_empty_phase ^= 1; }
                    }
                    mbarrier_wait(o_store_ready_addr + (o_epi_stage) * 8, o_ready_phase);
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_4d(tensormap_workspace + (cta_slot_base_epi + 512), 0, chunk_start_5 + block_epi * 64, 0, sab_head_5, smem_o_addr + o_epi_stage * 16384);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    mbarrier_arrive(o_store_empty_addr + (o_epi_stage) * 8);
                    o_epi_stage += 1;
                    if (o_epi_stage == 2) { o_epi_stage = 0; o_ready_phase ^= 1; }
                }
                #pragma unroll
                for (int __9 = 0; __9 < 5; __9++) {
                    mbarrier_wait(load_gate_empty_addr + (gate_stage) * 8, gate_empty_phase);
                    gate_stage += 1;
                    if (gate_stage == 5) { gate_stage = 0; gate_empty_phase ^= 1; }
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

#undef CAKE_INF
#undef NUM_AINV_PIPE_STAGES
#undef NUM_CG0_ACC_PIPE_STAGES
#undef NUM_GATE_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_ONE_STAGE_STAGES
#undef NUM_O_PIPE_STAGES
#undef NUM_QK_PIPE_STAGES
#undef NUM_Q_PIPE_STAGES
#undef NUM_T_PIPE_STAGES
#undef NUM_V_PIPE_STAGES
#undef SMEM_SMEM_AINV_OFF
#undef SMEM_SMEM_AINV_STAGE_BYTES
#undef SMEM_SMEM_AINV_STRIDE
#undef SMEM_SMEM_CUMPROD_OFF
#undef SMEM_SMEM_CUMPROD_STAGE_BYTES
#undef SMEM_SMEM_CUMPROD_STRIDE
#undef SMEM_SMEM_CUMSUMLOG_OFF
#undef SMEM_SMEM_CUMSUMLOG_STAGE_BYTES
#undef SMEM_SMEM_CUMSUMLOG_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_SMEM_K_TRANS_OFF
#undef SMEM_SMEM_K_TRANS_STAGE_BYTES
#undef SMEM_SMEM_K_TRANS_STRIDE
#undef SMEM_SMEM_O_OFF
#undef SMEM_SMEM_O_STAGE_BYTES
#undef SMEM_SMEM_O_STRIDE
#undef SMEM_SMEM_QK_OFF
#undef SMEM_SMEM_QK_STAGE_BYTES
#undef SMEM_SMEM_QK_STRIDE
#undef SMEM_SMEM_Q_OFF
#undef SMEM_SMEM_Q_STAGE_BYTES
#undef SMEM_SMEM_Q_STRIDE
#undef SMEM_SMEM_T_OFF
#undef SMEM_SMEM_T_STAGE_BYTES
#undef SMEM_SMEM_T_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_CG0_ACC_OFFSET
#undef TMEM_TMEM_CG1_ACC_OFFSET
#undef TMEM_TMEM_Q_STATE_OFFSET
#undef TMEM_TMEM_SHARED_INPUT_OFFSET
#undef TMEM_TMEM_STATE_INPUT_OFFSET
#undef TMEM_TMEM_STATE_OFFSET
#undef ainv_empty_addr
#undef ainv_ready_addr
#undef cg0_acc_empty_addr
#undef cg0_acc_full_addr
#undef cg1_acc_empty_addr
#undef cg1_acc_full_addr
#undef decay_v_ready_addr
#undef kv_acc_empty_addr
#undef kv_acc_full_addr
#undef load_gate_empty_addr
#undef load_gate_full_addr
#undef load_k_empty_addr
#undef load_k_full_addr
#undef load_q_empty_addr
#undef load_q_full_addr
#undef load_t_empty_addr
#undef load_t_full_addr
#undef load_v_empty_addr
#undef load_v_full_addr
#undef nv_ready_addr
#undef o_store_empty_addr
#undef o_store_ready_addr
#undef q_state_acc_empty_addr
#undef q_state_acc_full_addr
#undef qk_empty_addr
#undef qk_ready_addr
#undef smem_ainv_addr
#undef smem_cumprod_addr
#undef smem_cumsumlog_addr
#undef smem_k_addr
#undef smem_k_trans_addr
#undef smem_o_addr
#undef smem_q_addr
#undef smem_qk_addr
#undef smem_t_addr
#undef smem_v_addr
#undef state_input_empty_addr
#undef state_input_ready_addr
#undef vks_ready_addr
// clang-format on
