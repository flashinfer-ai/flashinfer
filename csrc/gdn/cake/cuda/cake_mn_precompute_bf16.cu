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
#define TMEM_TMEM_M_OFFSET 0
#define TMEM_TMEM_N_OFFSET 128
#define TMEM_TMEM_SCRATCH_OFFSET 256
#define TMEM_TMEM_M_INPUT_OFFSET 320
#define TMEM_TMEM_N_INPUT_OFFSET 384
#define TMEM_TMEM_XY_ACC_OFFSET 448
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define NUM_T_PIPE_STAGES 3
#define NUM_ALPHA_PIPE_STAGES 4
#define NUM_X_READY_PIPE_STAGES 2
#define NUM_ONE_STAGE_STAGES 1
#define SMEM_SMEM_K_OFF 1024
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_K_TRANS_OFF 1024
#define SMEM_SMEM_K_TRANS_STAGE_BYTES 16384
#define SMEM_SMEM_K_TRANS_STRIDE 16384
#define SMEM_SMEM_V_OFF 50176
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_T_OFF 99328
#define SMEM_SMEM_T_STAGE_BYTES 8192
#define SMEM_SMEM_T_STRIDE 8192
#define SMEM_SMEM_X_OFF 123904
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_ALPHA_OFF 156672
#define SMEM_SMEM_ALPHA_STAGE_BYTES 768
#define SMEM_SMEM_ALPHA_STRIDE 768
#define SMEM_TOTAL 159744
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
kernel_flashinfer_blackwell_gdn_cp_prefill_mn_precompute_bf16_v1(const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap T, float* __restrict__ alpha, float* __restrict__ local_transfer, float* __restrict__ local_state, long long* __restrict__ cu_seqlens, int chunk_len, int num_k_heads, int num_v_heads, int num_sab_heads, int total_cp_chunks, int num_seqs)
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
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    __nv_bfloat16* smem_k_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_k_trans_addr = smem + 1024;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int smem_v_addr = smem + 50176;
    __nv_bfloat16* smem_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_t_addr = smem + 99328;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 123904);
    const int smem_x_addr = smem + 123904;
    float* smem_alpha = reinterpret_cast<float*>(smem_raw + 156672);
    const int smem_alpha_addr = smem + 156672;

    // Mbarrier init (34 groups, 54 barriers)
    // Mbarriers at smem_raw[0..432)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // k_empty: 3 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // v_empty: 3 barriers, init_count=4
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 't_pipe' ---
            // t_full: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // t_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'alpha_pipe' ---
            // alpha_full: 4 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            // alpha_empty: 4 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            // --- pipeline 'one_stage' ---
            // m_init_full: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // m_init_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // n_init_full: 1 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            // n_init_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            // x_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            // x_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 248, 128);
            // --- pipeline 'x_ready_pipe' ---
            // x_ready_full: 2 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            // x_ready_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 272, 2);
            mbarrier_init(smem + 280, 2);
            // --- pipeline 'one_stage' ---
            // m_input_full: 1 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            // m_input_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            // n_input_full: 1 barriers, init_count=128
            mbarrier_init(smem + 304, 128);
            // n_input_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            // z_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            // z_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 328, 128);
            // z_ready_full: 1 barriers, init_count=128
            mbarrier_init(smem + 336, 128);
            // z_ready_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 344, 1);
            // m_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 352, 1);
            // m_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 360, 128);
            // y_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 368, 1);
            // y_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 376, 128);
            // y_ready_full: 1 barriers, init_count=128
            mbarrier_init(smem + 384, 128);
            // y_ready_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 392, 1);
            // n_acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            // n_acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 408, 128);
            // done_full: 1 barriers, init_count=128
            mbarrier_init(smem + 416, 128);
            // done_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 424, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 432);
    if (warp == 4) {
        int _tmem_hold = smem + 432;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define k_full_addr (mbar_base + 0)
    #define k_empty_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 72)
    #define t_full_addr (mbar_base + 96)
    #define t_empty_addr (mbar_base + 120)
    #define alpha_full_addr (mbar_base + 144)
    #define alpha_empty_addr (mbar_base + 176)
    #define m_init_full_addr (mbar_base + 208)
    #define m_init_empty_addr (mbar_base + 216)
    #define n_init_full_addr (mbar_base + 224)
    #define n_init_empty_addr (mbar_base + 232)
    #define x_acc_full_addr (mbar_base + 240)
    #define x_acc_empty_addr (mbar_base + 248)
    #define x_ready_full_addr (mbar_base + 256)
    #define x_ready_empty_addr (mbar_base + 272)
    #define m_input_full_addr (mbar_base + 288)
    #define m_input_empty_addr (mbar_base + 296)
    #define n_input_full_addr (mbar_base + 304)
    #define n_input_empty_addr (mbar_base + 312)
    #define z_acc_full_addr (mbar_base + 320)
    #define z_acc_empty_addr (mbar_base + 328)
    #define z_ready_full_addr (mbar_base + 336)
    #define z_ready_empty_addr (mbar_base + 344)
    #define m_acc_full_addr (mbar_base + 352)
    #define m_acc_empty_addr (mbar_base + 360)
    #define y_acc_full_addr (mbar_base + 368)
    #define y_acc_empty_addr (mbar_base + 376)
    #define y_ready_full_addr (mbar_base + 384)
    #define y_ready_empty_addr (mbar_base + 392)
    #define n_acc_full_addr (mbar_base + 400)
    #define n_acc_empty_addr (mbar_base + 408)
    #define done_full_addr (mbar_base + 416)
    #define done_empty_addr (mbar_base + 424)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_m = taddr;
    const int tmem_tmem_n = taddr + 128;
    const int tmem_tmem_scratch = taddr + 256;
    const int tmem_tmem_m_input = taddr + 320;
    const int tmem_tmem_n_input = taddr + 384;
    const int tmem_tmem_xy_acc = taddr + 448;

    // ---- Register redistribution for WGs split across roles ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
    }

    // ---- Role: compute_group_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // compute_group_0_main
            int sab_head = blockIdx.x % num_sab_heads;
            int chunk_in_seq = blockIdx.x / num_sab_heads;
            int seq_idx = blockIdx.y;
            int seq_start = (int)cu_seqlens[seq_idx];
            int seq_end = (int)cu_seqlens[seq_idx + 1];
            int seq_len = seq_end - seq_start;
            int num_chunks = (seq_len + chunk_len - 1) / chunk_len;
            int remaining = seq_len - chunk_in_seq * chunk_len;
            int valid_len = ((remaining < chunk_len) ? remaining : chunk_len);
            int num_blocks = (valid_len + 64 - 1) / 64;
            int tok_offset = seq_start + chunk_in_seq * chunk_len;
            int prefix_items = ((seq_idx < seq_start) ? seq_idx : seq_start);
            int cp_chunk = prefix_items + (seq_start - prefix_items) / chunk_len + chunk_in_seq;
            int t_blocks_per_chunk = (chunk_len + 64 - 1) / 64;
            int t_prefix_items = ((seq_idx < seq_start) ? seq_idx : seq_start);
            int t_block_start = t_prefix_items + (seq_start - t_prefix_items) / 64 + chunk_in_seq * t_blocks_per_chunk;
            unsigned int _phase_m_init_empty_0 = 1;
            unsigned int _phase_alpha_full = 0;
            unsigned int _phase_x_acc_full_0 = 0;
            unsigned int _phase_x_ready_empty = 1;
            unsigned int _phase_m_input_empty_0 = 1;
            unsigned int _phase_z_acc_full_0 = 0;
            unsigned int _phase_z_ready_empty_0 = 1;
            unsigned int _phase_m_acc_full_0 = 0;
            unsigned int _phase_done_empty_0 = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                int warp_in_wg = warp % 4;
                int row_base = warp_in_wg * 32 << 16;
                int row = warp_in_wg * 32 + lane;
                #pragma unroll
                for (int panel = 0; panel < 4; panel++) {
                    float values[32];
                    #pragma unroll
                    for (int item = 0; item < 32; item++) {
                        int col = panel * 32 + item;
                        values[item] = ((row == col) ? 1.0f : 0.0f);
                    }
                    tmem_st_x32_f32(taddr + (unsigned int)row_base + (unsigned int)(panel * 32), values);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_wait(m_init_empty_addr, _phase_m_init_empty_0);
                _phase_m_init_empty_0 ^= 1;
                mbarrier_arrive(m_init_full_addr);
                unsigned int alpha_stage_m = 0;
                unsigned int x_stage_m = 0;
                #pragma unroll 1
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    mbarrier_wait(alpha_full_addr + (alpha_stage_m) * 8, _phase_alpha_full);
                    mbarrier_wait(x_acc_full_addr, _phase_x_acc_full_0);
                    _phase_x_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(x_ready_empty_addr + (x_stage_m) * 8, _phase_x_ready_empty);
                    int warp_row = warp_in_wg * 32;
                    int matrix_idx = lane / 8;
                    int address_row = lane & 7;
                    #pragma unroll
                    for (int row_half = 0; row_half < 2; row_half++) {
                        int row_base_0 = warp_row + row_half * 16 << 16;
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                            : "r"(taddr + 448 + (unsigned int)row_base_0)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int packed[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int token_group = 0; token_group < 4; token_group++) {
                            int dst_row = warp_row + row_half * 16 + (matrix_idx & 1) * 8;
                            int dst_token = token_group * 16 + matrix_idx / 2 * 8 + address_row;
                            int row_group = dst_row / 64;
                            int row_within = dst_row % 64;
                            int atom_row = row_group * 64 + dst_token;
                            const int reg_base = token_group * 4;
                            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_x_addr + x_stage_m * 16384 + (unsigned int)(atom_row * 128 + row_within * 2 ^ (atom_row * 128 + row_within * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed[reg_base])), "r"(*reinterpret_cast<const uint32_t*>(&packed[reg_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[reg_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[reg_base + 3]))
                                : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(x_acc_empty_addr);
                    mbarrier_arrive(x_ready_full_addr + (x_stage_m) * 8);
                    if (block_idx > 0) {
                        int row_base_0_1 = warp_in_wg * 32 << 16;
                        #pragma unroll
                        for (int panel_1 = 0; panel_1 < 4; panel_1++) {
                            float _tmem_load_1[32];
                            tmem_ld_x32(&_tmem_load_1[0], taddr + (unsigned int)row_base_0_1 + (unsigned int)(panel_1 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            unsigned int packed_1[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                                packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + 320 + (unsigned int)row_base_0_1 + (unsigned int)(panel_1 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[15]))
                                : "memory");
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_wait(m_input_empty_addr, _phase_m_input_empty_0);
                        _phase_m_input_empty_0 ^= 1;
                        mbarrier_arrive(m_input_full_addr);
                        mbarrier_wait(z_acc_full_addr, _phase_z_acc_full_0);
                        _phase_z_acc_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int warp_row_1 = warp_in_wg * 32;
                        #pragma unroll
                        for (int row_half_1 = 0; row_half_1 < 2; row_half_1++) {
                            int row_base_1 = warp_row_1 + row_half_1 * 16 << 16;
                            float _tmem_load_2[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                                : "r"(taddr + 256 + (unsigned int)row_base_1)
                                : "memory");
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            unsigned int packed_2[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                                packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + 320 + (unsigned int)row_base_1), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_2[15]))
                                : "memory");
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(z_acc_empty_addr);
                        mbarrier_wait(z_ready_empty_addr, _phase_z_ready_empty_0);
                        _phase_z_ready_empty_0 ^= 1;
                        mbarrier_arrive(z_ready_full_addr);
                    }
                    float block_coeff_m = smem_alpha[alpha_stage_m * 192 + 64 + 63];
                    mbarrier_wait(m_acc_full_addr, _phase_m_acc_full_0);
                    _phase_m_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base_0_2 = warp_in_wg * 32 << 16;
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 4; panel_2++) {
                        float _tmem_load_3[32];
                        tmem_ld_x32(&_tmem_load_3[0], taddr + (unsigned int)row_base_0_2 + (unsigned int)(panel_2 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_1 = {block_coeff_m, block_coeff_m};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                        tmem_st_x32_f32(taddr + (unsigned int)row_base_0_2 + (unsigned int)(panel_2 * 32), _tmem_load_3);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(m_acc_empty_addr);
                    mbarrier_arrive(alpha_empty_addr + (alpha_stage_m) * 8);
                    alpha_stage_m += 1;
                    if (alpha_stage_m == 4) { alpha_stage_m = 0; _phase_alpha_full ^= 1; }
                    x_stage_m += 1;
                    if (x_stage_m == 2) { x_stage_m = 0; _phase_x_ready_empty ^= 1; }
                }
                long long matrix_base_m = ((long long)cp_chunk * (long long)num_sab_heads + (long long)sab_head) * 128 * 128;
                int row_base_0_3 = warp_in_wg * 32 << 16;
                int row_1 = warp_in_wg * 32 + lane;
                #pragma unroll
                for (int panel_3 = 0; panel_3 < 4; panel_3++) {
                    float _tmem_load_4[32];
                    tmem_ld_x32(&_tmem_load_4[0], taddr + (unsigned int)row_base_0_3 + (unsigned int)(panel_3 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int vector = 0; vector < 8; vector++) {
                        {
                            float4 _v4 = make_float4(_tmem_load_4[vector * 4 + 0], _tmem_load_4[vector * 4 + 1], _tmem_load_4[vector * 4 + 2], _tmem_load_4[vector * 4 + 3]);
                            *reinterpret_cast<float4*>(local_transfer + (matrix_base_m + (long long)row_1 * 128 + (long long)(panel_3 * 32) + (long long)(vector * 4)) + 0) = _v4;
                        }
                    }
                }
                mbarrier_wait(done_empty_addr, _phase_done_empty_0);
                _phase_done_empty_0 ^= 1;
                mbarrier_arrive(done_full_addr);
            }
        }
    }
    // ---- Role: compute_group_1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // compute_group_1_main
            int sab_head_1 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_1 = blockIdx.x / num_sab_heads;
            int seq_idx_1 = blockIdx.y;
            int seq_start_1 = (int)cu_seqlens[seq_idx_1];
            int seq_end_1 = (int)cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = seq_end_1 - seq_start_1;
            int num_chunks_1 = (seq_len_1 + chunk_len - 1) / chunk_len;
            int remaining_1 = seq_len_1 - chunk_in_seq_1 * chunk_len;
            int valid_len_1 = ((remaining_1 < chunk_len) ? remaining_1 : chunk_len);
            int num_blocks_1 = (valid_len_1 + 64 - 1) / 64;
            int tok_offset_1 = seq_start_1 + chunk_in_seq_1 * chunk_len;
            int prefix_items_1 = ((seq_idx_1 < seq_start_1) ? seq_idx_1 : seq_start_1);
            int cp_chunk_1 = prefix_items_1 + (seq_start_1 - prefix_items_1) / chunk_len + chunk_in_seq_1;
            int t_blocks_per_chunk_1 = (chunk_len + 64 - 1) / 64;
            int t_prefix_items_1 = ((seq_idx_1 < seq_start_1) ? seq_idx_1 : seq_start_1);
            int t_block_start_1 = t_prefix_items_1 + (seq_start_1 - t_prefix_items_1) / 64 + chunk_in_seq_1 * t_blocks_per_chunk_1;
            unsigned int _phase_n_init_empty_0 = 1;
            unsigned int _phase_alpha_full_1 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_n_input_empty_0 = 1;
            unsigned int _phase_y_acc_full_0 = 0;
            unsigned int _phase_y_ready_empty_0 = 1;
            unsigned int _phase_n_acc_full_0 = 0;
            unsigned int _phase_done_full_0 = 0;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                int warp_in_wg_1 = warp % 4;
                int row_base_2 = warp_in_wg_1 * 32 << 16;
                float values_1[32];
                values_1[0] = 0.0f;
                values_1[1] = 0.0f;
                values_1[2] = 0.0f;
                values_1[3] = 0.0f;
                values_1[4] = 0.0f;
                values_1[5] = 0.0f;
                values_1[6] = 0.0f;
                values_1[7] = 0.0f;
                values_1[8] = 0.0f;
                values_1[9] = 0.0f;
                values_1[10] = 0.0f;
                values_1[11] = 0.0f;
                values_1[12] = 0.0f;
                values_1[13] = 0.0f;
                values_1[14] = 0.0f;
                values_1[15] = 0.0f;
                values_1[16] = 0.0f;
                values_1[17] = 0.0f;
                values_1[18] = 0.0f;
                values_1[19] = 0.0f;
                values_1[20] = 0.0f;
                values_1[21] = 0.0f;
                values_1[22] = 0.0f;
                values_1[23] = 0.0f;
                values_1[24] = 0.0f;
                values_1[25] = 0.0f;
                values_1[26] = 0.0f;
                values_1[27] = 0.0f;
                values_1[28] = 0.0f;
                values_1[29] = 0.0f;
                values_1[30] = 0.0f;
                values_1[31] = 0.0f;
                #pragma unroll
                for (int panel_4 = 0; panel_4 < 4; panel_4++) {
                    tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base_2 + (unsigned int)(panel_4 * 32), values_1);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_wait(n_init_empty_addr, _phase_n_init_empty_0);
                _phase_n_init_empty_0 ^= 1;
                mbarrier_arrive(n_init_full_addr);
                unsigned int alpha_stage_n = 0;
                unsigned int v_stage_n = 0;
                #pragma unroll 1
                for (int _ = 0; _ < num_blocks_1; _++) {
                    mbarrier_wait(alpha_full_addr + (alpha_stage_n) * 8, _phase_alpha_full_1);
                    mbarrier_wait(v_full_addr + (v_stage_n) * 8, _phase_v_full);
                    int row_base_0_4 = warp_in_wg_1 * 32 << 16;
                    #pragma unroll
                    for (int panel_5 = 0; panel_5 < 4; panel_5++) {
                        float _tmem_load_5[32];
                        tmem_ld_x32(&_tmem_load_5[0], taddr + 128 + (unsigned int)row_base_0_4 + (unsigned int)(panel_5 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int packed_3[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 384 + (unsigned int)row_base_0_4 + (unsigned int)(panel_5 * 16)), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_3[15]))
                            : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_wait(n_input_empty_addr, _phase_n_input_empty_0);
                    _phase_n_input_empty_0 ^= 1;
                    mbarrier_arrive(n_input_full_addr);
                    mbarrier_wait(y_acc_full_addr, _phase_y_acc_full_0);
                    _phase_y_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float block_coeff_n = smem_alpha[alpha_stage_n * 192 + 64 + 63];
                    int warp_row_2 = warp_in_wg_1 * 32;
                    int lane_quad = lane & 3;
                    int matrix_idx_1 = lane / 8;
                    int address_row_1 = lane & 7;
                    #pragma unroll
                    for (int row_half_2 = 0; row_half_2 < 2; row_half_2++) {
                        int row_base_1_1 = warp_row_2 + row_half_2 * 16 << 16;
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                            : "r"(taddr + 448 + (unsigned int)row_base_1_1)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float result[32];
                        #pragma unroll
                        for (int token_group_1 = 0; token_group_1 < 4; token_group_1++) {
                            int src_row = warp_row_2 + row_half_2 * 16 + (matrix_idx_1 & 1) * 8;
                            int src_token = token_group_1 * 16 + matrix_idx_1 / 2 * 8 + address_row_1;
                            int row_group_1 = src_row / 64;
                            int row_within_1 = src_row % 64;
                            int atom_row_1 = row_group_1 * 64 + src_token;
                            unsigned int v_bits[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(v_bits[0]), "=r"(v_bits[1]), "=r"(v_bits[2]), "=r"(v_bits[3])
                                : "r"((smem_v_addr + v_stage_n * 16384 + (unsigned int)(atom_row_1 * 128 + row_within_1 * 2 ^ (atom_row_1 * 128 + row_within_1 * 2 >> 7 & 7) << 4)))
                                : "memory");
                            float v_bits_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&v_bits_f32[_pair * 2])[0]), "=f"((&v_bits_f32[_pair * 2])[1])
                                    : "r"(v_bits[_pair]));
                            }
                            #pragma unroll
                            for (int item_1 = 0; item_1 < 8; item_1++) {
                                const int reg_idx = token_group_1 * 8 + item_1;
                                int token = token_group_1 * 16 + item_1 / 4 * 8 + lane_quad * 2 + (item_1 & 1);
                                float neg_end_rcp = smem_alpha[alpha_stage_n * 192 + 128 + (unsigned int)token];
                                result[reg_idx] = block_coeff_n * _tmem_load_6[reg_idx] + v_bits_f32[item_1] * neg_end_rcp;
                            }
                        }
                        unsigned int packed_4[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                            packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(taddr + 384 + (unsigned int)row_base_1_1), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_4[15]))
                            : "memory");
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(y_acc_empty_addr);
                    int row_base_1_2 = warp_in_wg_1 * 32 << 16;
                    #pragma unroll
                    for (int panel_6 = 0; panel_6 < 4; panel_6++) {
                        float _tmem_load_7[32];
                        tmem_ld_x32(&_tmem_load_7[0], taddr + 128 + (unsigned int)row_base_1_2 + (unsigned int)(panel_6 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_0 = {block_coeff_n, block_coeff_n};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_0);
                        tmem_st_x32_f32(taddr + 128 + (unsigned int)row_base_1_2 + (unsigned int)(panel_6 * 32), _tmem_load_7);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_wait(y_ready_empty_addr, _phase_y_ready_empty_0);
                    _phase_y_ready_empty_0 ^= 1;
                    mbarrier_arrive(y_ready_full_addr);
                    mbarrier_wait(n_acc_full_addr, _phase_n_acc_full_0);
                    _phase_n_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(n_acc_empty_addr);
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (v_stage_n) * 8);
                    }
                    mbarrier_arrive(alpha_empty_addr + (alpha_stage_n) * 8);
                    alpha_stage_n += 1;
                    if (alpha_stage_n == 4) { alpha_stage_n = 0; _phase_alpha_full_1 ^= 1; }
                    v_stage_n += 1;
                    if (v_stage_n == 3) { v_stage_n = 0; _phase_v_full ^= 1; }
                }
                long long matrix_base_n = ((long long)cp_chunk_1 * (long long)num_sab_heads + (long long)sab_head_1) * 128 * 128;
                int row_base_0_5 = warp_in_wg_1 * 32 << 16;
                int row_2 = warp_in_wg_1 * 32 + lane;
                #pragma unroll
                for (int panel_7 = 0; panel_7 < 4; panel_7++) {
                    float _tmem_load_8[32];
                    tmem_ld_x32(&_tmem_load_8[0], taddr + 128 + (unsigned int)row_base_0_5 + (unsigned int)(panel_7 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int vector_1 = 0; vector_1 < 8; vector_1++) {
                        {
                            float4 _v4 = make_float4(_tmem_load_8[vector_1 * 4 + 0], _tmem_load_8[vector_1 * 4 + 1], _tmem_load_8[vector_1 * 4 + 2], _tmem_load_8[vector_1 * 4 + 3]);
                            *reinterpret_cast<float4*>(local_state + (matrix_base_n + (long long)row_2 * 128 + (long long)(panel_7 * 32) + (long long)(vector_1 * 4)) + 0) = _v4;
                        }
                    }
                }
                mbarrier_wait(done_full_addr, _phase_done_full_0);
                _phase_done_full_0 ^= 1;
                mbarrier_arrive(done_empty_addr);
            }
        }
    }
    // ---- Role: transfer_mma ----
    if (warp == 8) {
        { // transfer_mma_main
            int sab_head_2 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_2 = blockIdx.x / num_sab_heads;
            int seq_idx_2 = blockIdx.y;
            int seq_start_2 = (int)cu_seqlens[seq_idx_2];
            int seq_end_2 = (int)cu_seqlens[seq_idx_2 + 1];
            int seq_len_2 = seq_end_2 - seq_start_2;
            int num_chunks_2 = (seq_len_2 + chunk_len - 1) / chunk_len;
            int remaining_2 = seq_len_2 - chunk_in_seq_2 * chunk_len;
            int valid_len_2 = ((remaining_2 < chunk_len) ? remaining_2 : chunk_len);
            int num_blocks_2 = (valid_len_2 + 64 - 1) / 64;
            int tok_offset_2 = seq_start_2 + chunk_in_seq_2 * chunk_len;
            int prefix_items_2 = ((seq_idx_2 < seq_start_2) ? seq_idx_2 : seq_start_2);
            int cp_chunk_2 = prefix_items_2 + (seq_start_2 - prefix_items_2) / chunk_len + chunk_in_seq_2;
            int t_blocks_per_chunk_2 = (chunk_len + 64 - 1) / 64;
            int t_prefix_items_2 = ((seq_idx_2 < seq_start_2) ? seq_idx_2 : seq_start_2);
            int t_block_start_2 = t_prefix_items_2 + (seq_start_2 - t_prefix_items_2) / 64 + chunk_in_seq_2 * t_blocks_per_chunk_2;
            unsigned int _phase_m_init_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_m_input_full_0 = 0;
            unsigned int _phase_z_acc_empty_0 = 1;
            unsigned int _phase_z_ready_full_0 = 0;
            unsigned int _phase_x_ready_full = 0;
            unsigned int _phase_m_acc_empty_0 = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                mbarrier_wait(m_init_full_addr, _phase_m_init_full_0);
                _phase_m_init_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                elect_commit(m_init_empty_addr);
                unsigned int transfer_k_stage = 0;
                unsigned int transfer_x_stage = 0;
                #pragma unroll 1
                for (int block_idx_1 = 0; block_idx_1 < num_blocks_2; block_idx_1++) {
                    mbarrier_wait(k_full_addr + (transfer_k_stage) * 8, _phase_k_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (block_idx_1 > 0) {
                        mbarrier_wait(m_input_full_addr, _phase_m_input_full_0);
                        _phase_m_input_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_0 = make_warp_uniform((((smem_k_trans_addr) >> 4) & 0x3FFF) + (transfer_k_stage) * 1024);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_scratch), "r"(_mma_b_lo_0), "r"(tmem_tmem_m_input), "r"(0));
                        mbarrier_wait(z_acc_empty_addr, _phase_z_acc_empty_0);
                        _phase_z_acc_empty_0 ^= 1;
                        elect_commit(z_acc_full_addr);
                        mbarrier_wait(z_ready_full_addr, _phase_z_ready_full_0);
                        _phase_z_ready_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        elect_commit(m_input_empty_addr);
                        elect_commit(z_ready_empty_addr);
                    }
                    mbarrier_wait(x_ready_full_addr + (transfer_x_stage) * 8, _phase_x_ready_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(m_acc_empty_addr, _phase_m_acc_empty_0);
                    _phase_m_acc_empty_0 ^= 1;
                    if (block_idx_1 == 0) {
                        int _mma_a_lo_1 = make_warp_uniform(((((smem_k_addr) >> 4) & 0x3FFF) | 0x2000000) + (transfer_k_stage) * 1024);
                        int _mma_b_lo_1 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x2000000) + (transfer_x_stage) * 1024);
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
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_tmem_m), "r"(1));
                    } else {
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x2000000) + (transfer_x_stage) * 1024);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_tmem_m), "r"(_mma_b_lo_2), "r"(tmem_tmem_m_input), "r"(1));
                    }
                    elect_commit(m_acc_full_addr);
                    elect_commit(x_ready_empty_addr + (transfer_x_stage) * 8);
                    elect_commit(k_empty_addr + (transfer_k_stage) * 8);
                    transfer_k_stage += 1;
                    if (transfer_k_stage == 3) { transfer_k_stage = 0; _phase_k_full ^= 1; }
                    transfer_x_stage += 1;
                    if (transfer_x_stage == 2) { transfer_x_stage = 0; _phase_x_ready_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: tma_loader ----
    if (warp == 9) {
        { // tma_loader_main
            int sab_head_3 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_3 = blockIdx.x / num_sab_heads;
            int seq_idx_3 = blockIdx.y;
            int seq_start_3 = (int)cu_seqlens[seq_idx_3];
            int seq_end_3 = (int)cu_seqlens[seq_idx_3 + 1];
            int seq_len_3 = seq_end_3 - seq_start_3;
            int num_chunks_3 = (seq_len_3 + chunk_len - 1) / chunk_len;
            int remaining_3 = seq_len_3 - chunk_in_seq_3 * chunk_len;
            int valid_len_3 = ((remaining_3 < chunk_len) ? remaining_3 : chunk_len);
            int num_blocks_3 = (valid_len_3 + 64 - 1) / 64;
            int tok_offset_3 = seq_start_3 + chunk_in_seq_3 * chunk_len;
            int prefix_items_3 = ((seq_idx_3 < seq_start_3) ? seq_idx_3 : seq_start_3);
            int cp_chunk_3 = prefix_items_3 + (seq_start_3 - prefix_items_3) / chunk_len + chunk_in_seq_3;
            int t_blocks_per_chunk_3 = (chunk_len + 64 - 1) / 64;
            int t_prefix_items_3 = ((seq_idx_3 < seq_start_3) ? seq_idx_3 : seq_start_3);
            int t_block_start_3 = t_prefix_items_3 + (seq_start_3 - t_prefix_items_3) / 64 + chunk_in_seq_3 * t_blocks_per_chunk_3;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_empty = 1;
            unsigned int _phase_t_empty = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                int k_head = sab_head_3 * num_k_heads / num_sab_heads;
                int v_head = sab_head_3 * num_v_heads / num_sab_heads;
                unsigned int k_stage = 0;
                unsigned int v_stage = 0;
                unsigned int t_stage = 0;
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&T))) : "memory");
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int block_idx_2 = 0; block_idx_2 < num_blocks_3; block_idx_2++) {
                        mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                        mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, 16384);
                        tma_4d_gmem2smem(smem_k_addr + k_stage * 16384, (&K), 0, tok_offset_3 + block_idx_2 * 64, 0, k_head, k_full_addr + (k_stage) * 8);
                        k_stage += 1;
                        if (k_stage == 3) { k_stage = 0; _phase_k_empty ^= 1; }
                        mbarrier_wait(v_empty_addr + (v_stage) * 8, _phase_v_empty);
                        mbarrier_arrive_expect_tx(v_full_addr + (v_stage) * 8, 16384);
                        tma_4d_gmem2smem(smem_v_addr + v_stage * 16384, (&V), 0, tok_offset_3 + block_idx_2 * 64, 0, v_head, v_full_addr + (v_stage) * 8);
                        v_stage += 1;
                        if (v_stage == 3) { v_stage = 0; _phase_v_empty ^= 1; }
                        mbarrier_wait(t_empty_addr + (t_stage) * 8, _phase_t_empty);
                        mbarrier_arrive_expect_tx(t_full_addr + (t_stage) * 8, 8192);
                        tma_4d_gmem2smem(smem_t_addr + t_stage * 8192, (&T), 0, 0, sab_head_3, t_block_start_3 + block_idx_2, t_full_addr + (t_stage) * 8);
                        t_stage += 1;
                        if (t_stage == 3) { t_stage = 0; _phase_t_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: alpha_loader ----
    if (warp == 10) {
        { // alpha_loader_main
            int sab_head_4 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_4 = blockIdx.x / num_sab_heads;
            int seq_idx_4 = blockIdx.y;
            int seq_start_4 = (int)cu_seqlens[seq_idx_4];
            int seq_end_4 = (int)cu_seqlens[seq_idx_4 + 1];
            int seq_len_4 = seq_end_4 - seq_start_4;
            int num_chunks_4 = (seq_len_4 + chunk_len - 1) / chunk_len;
            int remaining_4 = seq_len_4 - chunk_in_seq_4 * chunk_len;
            int valid_len_4 = ((remaining_4 < chunk_len) ? remaining_4 : chunk_len);
            int num_blocks_4 = (valid_len_4 + 64 - 1) / 64;
            int tok_offset_4 = seq_start_4 + chunk_in_seq_4 * chunk_len;
            int prefix_items_4 = ((seq_idx_4 < seq_start_4) ? seq_idx_4 : seq_start_4);
            int cp_chunk_4 = prefix_items_4 + (seq_start_4 - prefix_items_4) / chunk_len + chunk_in_seq_4;
            int t_blocks_per_chunk_4 = (chunk_len + 64 - 1) / 64;
            int t_prefix_items_4 = ((seq_idx_4 < seq_start_4) ? seq_idx_4 : seq_start_4);
            int t_block_start_4 = t_prefix_items_4 + (seq_start_4 - t_prefix_items_4) / 64 + chunk_in_seq_4 * t_blocks_per_chunk_4;
            unsigned int _phase_alpha_empty = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                unsigned int alpha_stage = 0;
                #pragma unroll 1
                for (int block_idx_3 = 0; block_idx_3 < num_blocks_4; block_idx_3++) {
                    mbarrier_wait(alpha_empty_addr + (alpha_stage) * 8, _phase_alpha_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int row0 = lane;
                    int row1 = row0 + 32;
                    int token0 = block_idx_3 * 64 + row0;
                    int token1 = block_idx_3 * 64 + row1;
                    float gate0 = 1.0f;
                    float gate1 = 1.0f;
                    if (token0 < valid_len_4) {
                        gate0 = alpha[(tok_offset_4 + token0) * num_sab_heads + sab_head_4];
                    }
                    if (token1 < valid_len_4) {
                        gate1 = alpha[(tok_offset_4 + token1) * num_sab_heads + sab_head_4];
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
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, log1, 31);
                    float end_log = _shfl_1;
                    float _exp2_0 = approx_exp2(log0);
                    float cumprod0 = _exp2_0;
                    float _exp2_1 = approx_exp2(log1);
                    float cumprod1 = _exp2_1;
                    float _exp2_2 = approx_exp2(end_log - log0);
                    float neg_end_rcp0 = -_exp2_2;
                    float _exp2_3 = approx_exp2(end_log - log1);
                    float neg_end_rcp1 = -_exp2_3;
                    if (token0 >= valid_len_4) {
                        neg_end_rcp0 = 0.0f;
                    }
                    if (token1 >= valid_len_4) {
                        neg_end_rcp1 = 0.0f;
                    }
                    smem_alpha[alpha_stage * 192 + (unsigned int)row0] = log0;
                    smem_alpha[alpha_stage * 192 + 64 + (unsigned int)row0] = cumprod0;
                    smem_alpha[alpha_stage * 192 + 128 + (unsigned int)row0] = neg_end_rcp0;
                    smem_alpha[alpha_stage * 192 + (unsigned int)row1] = log1;
                    smem_alpha[alpha_stage * 192 + 64 + (unsigned int)row1] = cumprod1;
                    smem_alpha[alpha_stage * 192 + 128 + (unsigned int)row1] = neg_end_rcp1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(alpha_full_addr + (alpha_stage) * 8);
                    alpha_stage += 1;
                    if (alpha_stage == 4) { alpha_stage = 0; _phase_alpha_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: state_mma ----
    if (warp == 11) {
        { // state_mma_main
            int sab_head_5 = blockIdx.x % num_sab_heads;
            int chunk_in_seq_5 = blockIdx.x / num_sab_heads;
            int seq_idx_5 = blockIdx.y;
            int seq_start_5 = (int)cu_seqlens[seq_idx_5];
            int seq_end_5 = (int)cu_seqlens[seq_idx_5 + 1];
            int seq_len_5 = seq_end_5 - seq_start_5;
            int num_chunks_5 = (seq_len_5 + chunk_len - 1) / chunk_len;
            int remaining_5 = seq_len_5 - chunk_in_seq_5 * chunk_len;
            int valid_len_5 = ((remaining_5 < chunk_len) ? remaining_5 : chunk_len);
            int num_blocks_5 = (valid_len_5 + 64 - 1) / 64;
            int tok_offset_5 = seq_start_5 + chunk_in_seq_5 * chunk_len;
            int prefix_items_5 = ((seq_idx_5 < seq_start_5) ? seq_idx_5 : seq_start_5);
            int cp_chunk_5 = prefix_items_5 + (seq_start_5 - prefix_items_5) / chunk_len + chunk_in_seq_5;
            int t_blocks_per_chunk_5 = (chunk_len + 64 - 1) / 64;
            int t_prefix_items_5 = ((seq_idx_5 < seq_start_5) ? seq_idx_5 : seq_start_5);
            int t_block_start_5 = t_prefix_items_5 + (seq_start_5 - t_prefix_items_5) / 64 + chunk_in_seq_5 * t_blocks_per_chunk_5;
            unsigned int _phase_n_init_full_0 = 0;
            unsigned int _phase_k_full_1 = 0;
            unsigned int _phase_t_full = 0;
            unsigned int _phase_x_acc_empty_0 = 1;
            unsigned int _phase_x_ready_full_1 = 0;
            unsigned int _phase_n_input_full_0 = 0;
            unsigned int _phase_y_acc_empty_0 = 1;
            unsigned int _phase_y_ready_full_0 = 0;
            unsigned int _phase_n_acc_empty_0 = 1;
            if (blockIdx.x / num_sab_heads < ((int)cu_seqlens[blockIdx.y + 1] - (int)cu_seqlens[blockIdx.y] + chunk_len - 1) / chunk_len) {
                mbarrier_wait(n_init_full_addr, _phase_n_init_full_0);
                _phase_n_init_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                elect_commit(n_init_empty_addr);
                unsigned int state_k_stage = 0;
                unsigned int state_t_stage = 0;
                unsigned int state_x_stage = 0;
                #pragma unroll 1
                for (int __1 = 0; __1 < num_blocks_5; __1++) {
                    mbarrier_wait(k_full_addr + (state_k_stage) * 8, _phase_k_full_1);
                    mbarrier_wait(t_full_addr + (state_t_stage) * 8, _phase_t_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(x_acc_empty_addr, _phase_x_acc_empty_0);
                    _phase_x_acc_empty_0 ^= 1;
                    int _mma_a_lo_3 = make_warp_uniform(((((smem_k_addr) >> 4) & 0x3FFF) | 0x2000000) + (state_k_stage) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_t_addr) >> 4) & 0x3FFF) | 0x2000000) + (state_t_stage) * 512);
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_tmem_xy_acc), "r"(0));
                    elect_commit(x_acc_full_addr);
                    elect_commit(t_empty_addr + (state_t_stage) * 8);
                    mbarrier_wait(x_ready_full_addr + (state_x_stage) * 8, _phase_x_ready_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(n_input_full_addr, _phase_n_input_full_0);
                    _phase_n_input_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(y_acc_empty_addr, _phase_y_acc_empty_0);
                    _phase_y_acc_empty_0 ^= 1;
                    int _mma_b_lo_4 = make_warp_uniform((((smem_k_trans_addr) >> 4) & 0x3FFF) + (state_k_stage) * 1024);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_xy_acc), "r"(_mma_b_lo_4), "r"(tmem_tmem_n_input), "r"(0));
                    elect_commit(y_acc_full_addr);
                    elect_commit(n_input_empty_addr);
                    elect_commit(k_empty_addr + (state_k_stage) * 8);
                    mbarrier_wait(y_ready_full_addr, _phase_y_ready_full_0);
                    _phase_y_ready_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    elect_commit(y_ready_empty_addr);
                    mbarrier_wait(n_acc_empty_addr, _phase_n_acc_empty_0);
                    _phase_n_acc_empty_0 ^= 1;
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x2000000) + (state_x_stage) * 1024);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_tmem_n), "r"(_mma_b_lo_5), "r"(tmem_tmem_n_input), "r"(1));
                    elect_commit(n_acc_full_addr);
                    elect_commit(x_ready_empty_addr + (state_x_stage) * 8);
                    state_k_stage += 1;
                    if (state_k_stage == 3) { state_k_stage = 0; _phase_k_full_1 ^= 1; }
                    state_t_stage += 1;
                    if (state_t_stage == 3) { state_t_stage = 0; _phase_t_full ^= 1; }
                    state_x_stage += 1;
                    if (state_x_stage == 2) { state_x_stage = 0; _phase_x_ready_full_1 ^= 1; }
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
#undef NUM_ALPHA_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef NUM_ONE_STAGE_STAGES
#undef NUM_T_PIPE_STAGES
#undef NUM_V_PIPE_STAGES
#undef NUM_X_READY_PIPE_STAGES
#undef SMEM_SMEM_ALPHA_OFF
#undef SMEM_SMEM_ALPHA_STAGE_BYTES
#undef SMEM_SMEM_ALPHA_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_SMEM_K_TRANS_OFF
#undef SMEM_SMEM_K_TRANS_STAGE_BYTES
#undef SMEM_SMEM_K_TRANS_STRIDE
#undef SMEM_SMEM_T_OFF
#undef SMEM_SMEM_T_STAGE_BYTES
#undef SMEM_SMEM_T_STRIDE
#undef SMEM_SMEM_V_OFF
#undef SMEM_SMEM_V_STAGE_BYTES
#undef SMEM_SMEM_V_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_M_INPUT_OFFSET
#undef TMEM_TMEM_M_OFFSET
#undef TMEM_TMEM_N_INPUT_OFFSET
#undef TMEM_TMEM_N_OFFSET
#undef TMEM_TMEM_SCRATCH_OFFSET
#undef TMEM_TMEM_XY_ACC_OFFSET
#undef alpha_empty_addr
#undef alpha_full_addr
#undef done_empty_addr
#undef done_full_addr
#undef k_empty_addr
#undef k_full_addr
#undef m_acc_empty_addr
#undef m_acc_full_addr
#undef m_init_empty_addr
#undef m_init_full_addr
#undef m_input_empty_addr
#undef m_input_full_addr
#undef n_acc_empty_addr
#undef n_acc_full_addr
#undef n_init_empty_addr
#undef n_init_full_addr
#undef n_input_empty_addr
#undef n_input_full_addr
#undef smem_alpha_addr
#undef smem_k_addr
#undef smem_k_trans_addr
#undef smem_t_addr
#undef smem_v_addr
#undef smem_x_addr
#undef t_empty_addr
#undef t_full_addr
#undef v_empty_addr
#undef v_full_addr
#undef x_acc_empty_addr
#undef x_acc_full_addr
#undef x_ready_empty_addr
#undef x_ready_full_addr
#undef y_acc_empty_addr
#undef y_acc_full_addr
#undef y_ready_empty_addr
#undef y_ready_full_addr
#undef z_acc_empty_addr
#undef z_acc_full_addr
#undef z_ready_empty_addr
#undef z_ready_full_addr
// clang-format on
