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
#define TMEM_NCOLS 256
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_OPERAND_OFFSET 128
#define NUM_M_PIPE_STAGES 1
#define NUM_N_PIPE_STAGES 1
#define NUM_ONE_STAGE_STAGES 1
#define SMEM_SMEM_M_OFF 1024
#define SMEM_SMEM_M_STAGE_BYTES 65536
#define SMEM_SMEM_M_STRIDE 65536
#define SMEM_SMEM_N_OFF 66560
#define SMEM_SMEM_N_STAGE_BYTES 65536
#define SMEM_SMEM_N_STRIDE 65536
#define SMEM_TOTAL 132096
#define THREADS 256

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
        "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_flashinfer_blackwell_gdn_cp_prefill_fixup_utcmma128_v1(const __grid_constant__ CUtensorMap local_transfer, const __grid_constant__ CUtensorMap local_state, float* __restrict__ initial_state, float* __restrict__ initial_state_workspace, float* __restrict__ fixed_state, float* __restrict__ output_state, long long* __restrict__ cu_seqlens, int chunk_len, int total_cp_chunks, int num_seqs, int num_heads)
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
    float* smem_m = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_m_addr = smem + 1024;
    float* smem_n = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_n_addr = smem + 66560;

    // Mbarrier init (8 groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'm_pipe' ---
            // m_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // m_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'n_pipe' ---
            // n_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // n_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 24, 4);
            // --- pipeline 'one_stage' ---
            // mma_ready_full: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // mma_ready_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // mma_done_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // mma_done_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define m_full_addr (mbar_base + 0)
    #define m_empty_addr (mbar_base + 8)
    #define n_full_addr (mbar_base + 16)
    #define n_empty_addr (mbar_base + 24)
    #define mma_ready_full_addr (mbar_base + 32)
    #define mma_ready_empty_addr (mbar_base + 40)
    #define mma_done_full_addr (mbar_base + 48)
    #define mma_done_empty_addr (mbar_base + 56)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_operand = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        { // compute_main
            int row_cta_idx = 0;
            int head_seq_idx = blockIdx.x;
            int head_idx = head_seq_idx % num_heads;
            int seq_idx = head_seq_idx / num_heads;
            int seq_start = (int)cu_seqlens[seq_idx];
            int seq_end = (int)cu_seqlens[seq_idx + 1];
            int seq_len = seq_end - seq_start;
            int num_chunks = (seq_len + chunk_len - 1) / chunk_len;
            int prefix = ((seq_idx < seq_start) ? seq_idx : seq_start);
            int chunk_start = prefix + (seq_start - prefix) / chunk_len;
            bool active = num_chunks > 0;
            unsigned int _phase_n_full = 0;
            unsigned int _phase_mma_ready_empty_0 = 1;
            unsigned int _phase_mma_done_full_0 = 0;
            if (active) {
                int state_head = seq_idx * num_heads + head_idx;
                long long state_base = (long long)state_head * 128 * 128 + (long long)(row_cta_idx * 128 * 128);
                int warp_in_wg = warp % 4;
                int local_row = warp_in_wg * 32 + lane;
                int tmem_row_base = warp_in_wg * 32 << 16;
                #pragma unroll
                for (int col_tile = 0; col_tile < 4; col_tile++) {
                    float values[32];
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 8; vec_idx++) {
                        int col = col_tile * 32 + vec_idx * 4;
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(initial_state + state_base + (long long)(local_row * 128) + (long long)col);
                            values[vec_idx * 4 + 0] = _v4.x;
                            values[vec_idx * 4 + 1] = _v4.y;
                            values[vec_idx * 4 + 2] = _v4.z;
                            values[vec_idx * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base + (unsigned int)(col_tile * 32), values);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int local_row_0 = warp_in_wg * 32 + lane;
                int tmem_row_base_1 = warp_in_wg * 32 << 16;
                #pragma unroll
                for (int col_tile_1 = 0; col_tile_1 < 4; col_tile_1++) {
                    float _tmem_load_0[32];
                    tmem_ld_x32(&_tmem_load_0[0], taddr + (unsigned int)tmem_row_base_1 + (unsigned int)(col_tile_1 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int vec_idx_1 = 0; vec_idx_1 < 8; vec_idx_1++) {
                        int col_1 = col_tile_1 * 32 + vec_idx_1 * 4;
                        {
                            float4 _v4 = make_float4(_tmem_load_0[vec_idx_1 * 4 + 0], _tmem_load_0[vec_idx_1 * 4 + 1], _tmem_load_0[vec_idx_1 * 4 + 2], _tmem_load_0[vec_idx_1 * 4 + 3]);
                            *reinterpret_cast<float4*>(initial_state_workspace + state_base + (long long)(local_row_0 * 128) + (long long)col_1) = _v4;
                        }
                    }
                }
                unsigned int n_stage_compute = 0;
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
                    int tmem_row_base_0 = warp_in_wg * 32 << 16;
                    #pragma unroll
                    for (int col_tile_2 = 0; col_tile_2 < 4; col_tile_2++) {
                        float _tmem_load_1[32];
                        tmem_ld_x32(&_tmem_load_1[0], taddr + (unsigned int)tmem_row_base_0 + (unsigned int)(col_tile_2 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int tf32_values[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            tf32_values[_lp] = __float_as_uint(_tmem_load_1[_lp + 0]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x32.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(taddr + 128 + (unsigned int)tmem_row_base_0 + (unsigned int)(col_tile_2 * 32)), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[0])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[1])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[2])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[3])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[4])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[5])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[6])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[7])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[8])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[9])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[10])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[11])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[12])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[13])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[14])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[15])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[16])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[17])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[18])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[19])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[20])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[21])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[22])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[23])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[24])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[25])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[26])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[27])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[28])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[29])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[30])), "r"(*reinterpret_cast<const uint32_t*>(&tf32_values[31])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_wait(n_full_addr + (n_stage_compute) * 8, _phase_n_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(mma_ready_empty_addr, _phase_mma_ready_empty_0);
                    _phase_mma_ready_empty_0 ^= 1;
                    int local_row_1 = warp_in_wg * 32 + lane;
                    int tmem_row_base_2 = warp_in_wg * 32 << 16;
                    #pragma unroll
                    for (int col_tile_3 = 0; col_tile_3 < 4; col_tile_3++) {
                        float values_1[32];
                        #pragma unroll
                        for (int vec_idx_2 = 0; vec_idx_2 < 8; vec_idx_2++) {
                            int col_2 = col_tile_3 * 32 + vec_idx_2 * 4;
                            int col_panel = col_2 / 32;
                            int col_within = col_2 % 32;
                            int atom_row = col_panel * 128 + local_row_1;
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&values_1[vec_idx_2 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&values_1[(vec_idx_2 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&values_1[(vec_idx_2 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&values_1[(vec_idx_2 * 4) + 3]))
                                : "r"((smem_n_addr + n_stage_compute * 65536 + (unsigned int)(atom_row * 128 + col_within * 4 ^ (atom_row * 128 + col_within * 4 >> 7 & 7) << 4))));
                        }
                        tmem_st_x32_f32(taddr + (unsigned int)tmem_row_base_2 + (unsigned int)(col_tile_3 * 32), values_1);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(mma_ready_full_addr);
                    if (elect_sync()) {
                        mbarrier_arrive(n_empty_addr + (n_stage_compute) * 8);
                    }
                    mbarrier_wait(mma_done_full_addr, _phase_mma_done_full_0);
                    _phase_mma_done_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int chunk = chunk_start + chunk_idx;
                    long long fixed_base = ((long long)chunk * (long long)num_heads + (long long)head_idx) * 128 * 128 + (long long)(row_cta_idx * 128 * 128);
                    if (chunk_idx == num_chunks - 1) {
                        int local_row_2 = warp_in_wg * 32 + lane;
                        int tmem_row_base_3 = warp_in_wg * 32 << 16;
                        #pragma unroll
                        for (int col_tile_4 = 0; col_tile_4 < 4; col_tile_4++) {
                            float _tmem_load_2[32];
                            tmem_ld_x32(&_tmem_load_2[0], taddr + (unsigned int)tmem_row_base_3 + (unsigned int)(col_tile_4 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #pragma unroll
                            for (int vec_idx_3 = 0; vec_idx_3 < 8; vec_idx_3++) {
                                int col_3 = col_tile_4 * 32 + vec_idx_3 * 4;
                                {
                                    float4 _v4 = make_float4(_tmem_load_2[vec_idx_3 * 4 + 0], _tmem_load_2[vec_idx_3 * 4 + 1], _tmem_load_2[vec_idx_3 * 4 + 2], _tmem_load_2[vec_idx_3 * 4 + 3]);
                                    *reinterpret_cast<float4*>(fixed_state + fixed_base + (long long)(local_row_2 * 128) + (long long)col_3) = _v4;
                                }
                                {
                                    float4 _v4 = make_float4(_tmem_load_2[vec_idx_3 * 4 + 0], _tmem_load_2[vec_idx_3 * 4 + 1], _tmem_load_2[vec_idx_3 * 4 + 2], _tmem_load_2[vec_idx_3 * 4 + 3]);
                                    *reinterpret_cast<float4*>(output_state + state_base + (long long)(local_row_2 * 128) + (long long)col_3) = _v4;
                                }
                            }
                        }
                    } else {
                        int local_row_2_1 = warp_in_wg * 32 + lane;
                        int tmem_row_base_3_1 = warp_in_wg * 32 << 16;
                        #pragma unroll
                        for (int col_tile_5 = 0; col_tile_5 < 4; col_tile_5++) {
                            float _tmem_load_3[32];
                            tmem_ld_x32(&_tmem_load_3[0], taddr + (unsigned int)tmem_row_base_3_1 + (unsigned int)(col_tile_5 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #pragma unroll
                            for (int vec_idx_4 = 0; vec_idx_4 < 8; vec_idx_4++) {
                                int col_4 = col_tile_5 * 32 + vec_idx_4 * 4;
                                {
                                    float4 _v4 = make_float4(_tmem_load_3[vec_idx_4 * 4 + 0], _tmem_load_3[vec_idx_4 * 4 + 1], _tmem_load_3[vec_idx_4 * 4 + 2], _tmem_load_3[vec_idx_4 * 4 + 3]);
                                    *reinterpret_cast<float4*>(fixed_state + fixed_base + (long long)(local_row_2_1 * 128) + (long long)col_4) = _v4;
                                }
                            }
                        }
                    }
                    mbarrier_arrive(mma_done_empty_addr);
                    n_stage_compute += 1;
                    if (n_stage_compute == 1) { n_stage_compute = 0; _phase_n_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: other ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // other_main
            int row_cta_idx_1 = 0;
            int head_seq_idx_1 = blockIdx.x;
            int head_idx_1 = head_seq_idx_1 % num_heads;
            int seq_idx_1 = head_seq_idx_1 / num_heads;
            int seq_start_1 = (int)cu_seqlens[seq_idx_1];
            int seq_end_1 = (int)cu_seqlens[seq_idx_1 + 1];
            int seq_len_1 = seq_end_1 - seq_start_1;
            int num_chunks_1 = (seq_len_1 + chunk_len - 1) / chunk_len;
            int prefix_1 = ((seq_idx_1 < seq_start_1) ? seq_idx_1 : seq_start_1);
            int chunk_start_1 = prefix_1 + (seq_start_1 - prefix_1) / chunk_len;
            bool active_1 = num_chunks_1 > 0;
            unsigned int _phase_m_full = 0;
            unsigned int _phase_mma_ready_full_0 = 0;
            unsigned int _phase_mma_done_empty_0 = 1;
            unsigned int _phase_n_empty = 1;
            unsigned int _phase_m_empty = 1;
            if (active_1) {
                int warp_in_wg_1 = warp % 4;
                if (warp_in_wg_1 == 0) {
                    unsigned int m_stage_mma = 0;
                    #pragma unroll 1
                    for (int _ = 0; _ < num_chunks_1; _++) {
                        mbarrier_wait(m_full_addr + (m_stage_mma) * 8, _phase_m_full);
                        mbarrier_wait(mma_ready_full_addr, _phase_mma_ready_full_0);
                        _phase_mma_ready_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(mma_done_empty_addr, _phase_mma_done_empty_0);
                        _phase_mma_done_empty_0 ^= 1;
                        int _mma_b_lo_0 = make_warp_uniform(((((smem_m_addr) >> 4) & 0x3FFF) | 0x4000000) + (m_stage_mma) * 4096);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, dout, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 dhi, 0x20004020;\n\t"
                    "mov.b32 id, 135334160;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2], db, id, p0;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2048;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2], db, id, p0;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 8], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2112;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 8], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 16], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2176;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 16], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 192;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 24], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2240;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 24], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 32], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2304;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 32], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 320;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 40], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2368;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 40], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 48], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2432;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 48], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 448;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 56], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2496;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 56], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 64], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2560;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 64], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 576;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 72], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2624;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 72], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 640;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 80], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2688;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 80], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 704;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 88], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2752;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 88], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 768;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 96], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2816;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 96], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 832;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 104], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2880;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 104], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 896;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 112], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 2944;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 112], db, id, p1;\n\t"
                    "mov.b32 dout, %0;\n\t"
                    "add.u32 blo, %1, 960;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 120], db, id, p1;\n\t"
                    "add.u32 dout, %0, 64;\n\t"
                    "add.u32 blo, %1, 3008;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [dout], [%2 + 120], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_acc), "r"(_mma_b_lo_0), "r"(tmem_tmem_operand), "r"(1));
                        elect_commit(mma_done_full_addr);
                        elect_commit(mma_ready_empty_addr);
                        elect_commit(m_empty_addr + (m_stage_mma) * 8);
                        m_stage_mma += 1;
                        if (m_stage_mma == 1) { m_stage_mma = 0; _phase_m_full ^= 1; }
                    }
                } else if (warp_in_wg_1 == 1) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&local_transfer))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&local_state))) : "memory");
                    unsigned int m_stage = 0;
                    unsigned int n_stage = 0;
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int chunk_idx_1 = 0; chunk_idx_1 < num_chunks_1; chunk_idx_1++) {
                            int chunk_1 = chunk_start_1 + chunk_idx_1;
                            mbarrier_wait(n_empty_addr + (n_stage) * 8, _phase_n_empty);
                            mbarrier_arrive_expect_tx(n_full_addr + (n_stage) * 8, 65536);
                            tma_5d_gmem2smem(smem_n_addr, (&local_state), 0, row_cta_idx_1 * 128, 0, head_idx_1, chunk_1, n_full_addr + (n_stage) * 8);
                            n_stage += 1;
                            if (n_stage == 1) { n_stage = 0; _phase_n_empty ^= 1; }
                            mbarrier_wait(m_empty_addr + (m_stage) * 8, _phase_m_empty);
                            mbarrier_arrive_expect_tx(m_full_addr + (m_stage) * 8, 65536);
                            tma_5d_gmem2smem(smem_m_addr, (&local_transfer), 0, 0, 0, head_idx_1, chunk_1, m_full_addr + (m_stage) * 8);
                            m_stage += 1;
                            if (m_stage == 1) { m_stage = 0; _phase_m_full ^= 1; _phase_m_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"

#undef CAKE_INF
#undef NUM_M_PIPE_STAGES
#undef NUM_N_PIPE_STAGES
#undef NUM_ONE_STAGE_STAGES
#undef SMEM_SMEM_M_OFF
#undef SMEM_SMEM_M_STAGE_BYTES
#undef SMEM_SMEM_M_STRIDE
#undef SMEM_SMEM_N_OFF
#undef SMEM_SMEM_N_STAGE_BYTES
#undef SMEM_SMEM_N_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_TMEM_ACC_OFFSET
#undef TMEM_TMEM_OPERAND_OFFSET
#undef m_empty_addr
#undef m_full_addr
#undef mma_done_empty_addr
#undef mma_done_full_addr
#undef mma_ready_empty_addr
#undef mma_ready_full_addr
#undef n_empty_addr
#undef n_full_addr
#undef smem_m_addr
#undef smem_n_addr
// clang-format on
