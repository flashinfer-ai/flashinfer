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
#define NUM_K_PIPE_STAGES 1
#define NUM_BETA_PIPE_STAGES 1
#define SMEM_SMEM_K_OFF 1024
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_INVERSE_OFF 17408
#define SMEM_SMEM_INVERSE_STAGE_BYTES 8192
#define SMEM_SMEM_INVERSE_STRIDE 8192
#define SMEM_SMEM_BETA_OFF 25600
#define SMEM_SMEM_BETA_STAGE_BYTES 256
#define SMEM_SMEM_BETA_STRIDE 256
#define SMEM_TOTAL 25856
#define THREADS 128



extern "C" {

__global__ __launch_bounds__(128, 8) void
kernel_flashinfer_blackwell_gdn_cp_prefill_t_precompute_v1(const __grid_constant__ CUtensorMap K, float* __restrict__ beta, __half* __restrict__ t, long long* __restrict__ cu_seqlens, int num_k_heads, int num_sab_heads, int total_t_blocks, int num_seqs)
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
    __half* smem_k = reinterpret_cast<__half*>(smem_raw + 1024);
    const int smem_k_addr = smem + 1024;
    __half* smem_inverse = reinterpret_cast<__half*>(smem_raw + 17408);
    const int smem_inverse_addr = smem + 17408;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_beta_addr = smem + 25600;

    // Mbarrier init (4 groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // k_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 8, 4);
            // --- pipeline 'beta_pipe' ---
            // beta_full: 1 barriers, init_count=32
            mbarrier_init(smem + 16, 32);
            // beta_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 24, 128);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();

    const int mbar_base = smem;
    #define k_full_addr (mbar_base + 0)
    #define k_empty_addr (mbar_base + 8)
    #define beta_full_addr (mbar_base + 16)
    #define beta_empty_addr (mbar_base + 24)

    // === Task calls (dependency order) ===
    int sab_head = blockIdx.x % num_sab_heads;
    int block_in_seq = blockIdx.x / num_sab_heads;
    int seq_idx = blockIdx.y;
    long long seq_start = cu_seqlens[seq_idx];
    long long seq_end = cu_seqlens[seq_idx + 1];
    int seq_len = (int)(seq_end - seq_start);
    int num_blocks = (seq_len + 64 - 1) / 64;
    if (block_in_seq < num_blocks) {
        int k_head = sab_head * num_k_heads / num_sab_heads;
        int token_offset = (int)seq_start + block_in_seq * 64;
        int remaining = seq_len - block_in_seq * 64;
        int valid_len = ((remaining < 64) ? remaining : 64);
        int prefix_items = ((seq_idx < (int)seq_start) ? seq_idx : (int)seq_start);
        int t_block = prefix_items + ((int)seq_start - prefix_items) / 64 + block_in_seq;
        if (warp == 1) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            if (elect_sync()) {
                mbarrier_wait(k_empty_addr, 1);
                mbarrier_arrive_expect_tx(k_full_addr, 16384);
                tma_4d_gmem2smem(smem_k_addr, (&K), 0, token_offset, 0, k_head, k_full_addr);
            }
        } else if (warp == 2) {
            mbarrier_wait(beta_empty_addr, 1);
            int beta_row0 = lane;
            int beta_row1 = beta_row0 + 32;
            float beta0 = 0.0f;
            float beta1 = 0.0f;
            if (beta_row0 < valid_len) {
                beta0 = beta[(token_offset + beta_row0) * num_sab_heads + sab_head];
            }
            if (beta_row1 < valid_len) {
                beta1 = beta[(token_offset + beta_row1) * num_sab_heads + sab_head];
            }
            smem_beta[beta_row0] = beta0;
            smem_beta[beta_row1] = beta1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(beta_full_addr);
        }
        mbarrier_wait(k_full_addr, 0);
        mbarrier_wait(beta_full_addr, 0);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        unsigned int gram_a[32];
        unsigned int gram_b[128];
        float gram_acc[32];
        int gram_row_base = warp * 16;
        #pragma unroll
        for (int mma_d = 0; mma_d < 8; mma_d++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(gram_a[mma_d * 4]), "=r"(gram_a[mma_d * 4 + 1]), "=r"(gram_a[mma_d * 4 + 2]), "=r"(gram_a[mma_d * 4 + 3])
                : "r"((smem_k_addr + (unsigned int)((mma_d * 16 + 8 * (lane / 16)) / 64 * 8192 + (gram_row_base + lane % 16) * 128 + (mma_d * 16 + 8 * (lane / 16)) % 64 * 2) ^ (smem_k_addr + (unsigned int)((mma_d * 16 + 8 * (lane / 16)) / 64 * 8192 + (gram_row_base + lane % 16) * 128 + (mma_d * 16 + 8 * (lane / 16)) % 64 * 2) >> 7 & 7) << 4))
                : "memory");
        }
        #pragma unroll
        for (int gram_n = 0; gram_n < 4; gram_n++) {
            #pragma unroll
            for (int mma_d_1 = 0; mma_d_1 < 8; mma_d_1++) {
                int b_lo_base = gram_n * 16 + mma_d_1 % 4 * 2 + mma_d_1 / 4 * 64;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(gram_b[b_lo_base]), "=r"(gram_b[b_lo_base + 1]), "=r"(gram_b[b_lo_base + 8]), "=r"(gram_b[b_lo_base + 8 + 1])
                    : "r"((smem_k_addr + (unsigned int)((mma_d_1 * 16 + 8 * (lane % 16 / 8)) / 64 * 8192 + (gram_n * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mma_d_1 * 16 + 8 * (lane % 16 / 8)) % 64 * 2) ^ (smem_k_addr + (unsigned int)((mma_d_1 * 16 + 8 * (lane % 16 / 8)) / 64 * 8192 + (gram_n * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mma_d_1 * 16 + 8 * (lane % 16 / 8)) % 64 * 2) >> 7 & 7) << 4))
                    : "memory");
            }
        }
        gram_acc[0] = 0.0f;
        gram_acc[1] = 0.0f;
        gram_acc[2] = 0.0f;
        gram_acc[3] = 0.0f;
        gram_acc[4] = 0.0f;
        gram_acc[5] = 0.0f;
        gram_acc[6] = 0.0f;
        gram_acc[7] = 0.0f;
        gram_acc[8] = 0.0f;
        gram_acc[9] = 0.0f;
        gram_acc[10] = 0.0f;
        gram_acc[11] = 0.0f;
        gram_acc[12] = 0.0f;
        gram_acc[13] = 0.0f;
        gram_acc[14] = 0.0f;
        gram_acc[15] = 0.0f;
        gram_acc[16] = 0.0f;
        gram_acc[17] = 0.0f;
        gram_acc[18] = 0.0f;
        gram_acc[19] = 0.0f;
        gram_acc[20] = 0.0f;
        gram_acc[21] = 0.0f;
        gram_acc[22] = 0.0f;
        gram_acc[23] = 0.0f;
        gram_acc[24] = 0.0f;
        gram_acc[25] = 0.0f;
        gram_acc[26] = 0.0f;
        gram_acc[27] = 0.0f;
        gram_acc[28] = 0.0f;
        gram_acc[29] = 0.0f;
        gram_acc[30] = 0.0f;
        gram_acc[31] = 0.0f;
        #pragma unroll
        for (int gram_n_1 = 0; gram_n_1 < 4; gram_n_1++) {
            #pragma unroll
            for (int mma_d_2 = 0; mma_d_2 < 8; mma_d_2++) {
                int acc_base = gram_n_1 * 8;
                int b_lo_base_1 = gram_n_1 * 16 + mma_d_2 % 4 * 2 + mma_d_2 / 4 * 64;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((gram_acc + acc_base)[0]), "+f"((gram_acc + acc_base)[1]), "+f"((gram_acc + acc_base)[2]), "+f"((gram_acc + acc_base)[3])
                    : "r"((gram_a + mma_d_2 * 4)[0]), "r"((gram_a + mma_d_2 * 4)[1]), "r"((gram_a + mma_d_2 * 4)[2]), "r"((gram_a + mma_d_2 * 4)[3]), "r"((gram_b + b_lo_base_1)[0]), "r"((gram_b + b_lo_base_1)[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((gram_acc + acc_base + 4)[0]), "+f"((gram_acc + acc_base + 4)[1]), "+f"((gram_acc + acc_base + 4)[2]), "+f"((gram_acc + acc_base + 4)[3])
                    : "r"((gram_a + mma_d_2 * 4)[0]), "r"((gram_a + mma_d_2 * 4)[1]), "r"((gram_a + mma_d_2 * 4)[2]), "r"((gram_a + mma_d_2 * 4)[3]), "r"((gram_b + b_lo_base_1 + 8)[0]), "r"((gram_b + b_lo_base_1 + 8)[1]));
            }
        }
        #pragma unroll
        for (int gram_n_2 = 0; gram_n_2 < 4; gram_n_2++) {
            int gram_col_base = gram_n_2 * 16;
            int row0 = gram_row_base + lane / 4;
            int row1 = row0 + 8;
            int col0 = gram_col_base + lane % 4 * 2;
            float beta0_1 = smem_beta[row0];
            float beta1_1 = smem_beta[row1];
            float seed[8];
            seed[0] = 0.0f;
            seed[1] = 0.0f;
            seed[2] = 0.0f;
            seed[3] = 0.0f;
            seed[4] = 0.0f;
            seed[5] = 0.0f;
            seed[6] = 0.0f;
            seed[7] = 0.0f;
            if (row0 > col0) {
                seed[0] = gram_acc[gram_n_2 * 8] * beta0_1;
            }
            if (row0 > col0 + 1) {
                seed[1] = gram_acc[gram_n_2 * 8 + 1] * beta0_1;
            }
            if (row1 > col0) {
                seed[2] = gram_acc[gram_n_2 * 8 + 2] * beta1_1;
            }
            if (row1 > col0 + 1) {
                seed[3] = gram_acc[gram_n_2 * 8 + 3] * beta1_1;
            }
            if (row0 > col0 + 8) {
                seed[4] = gram_acc[gram_n_2 * 8 + 4] * beta0_1;
            }
            if (row0 > col0 + 9) {
                seed[5] = gram_acc[gram_n_2 * 8 + 5] * beta0_1;
            }
            if (row1 > col0 + 8) {
                seed[6] = gram_acc[gram_n_2 * 8 + 6] * beta1_1;
            }
            if (row1 > col0 + 9) {
                seed[7] = gram_acc[gram_n_2 * 8 + 7] * beta1_1;
            }
            unsigned int packed[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                packed[_lp] = *(uint32_t*)&_h2;
            }
            int lane_row = lane % 16;
            int lane_col = lane / 16 * 8;
            uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)((gram_col_base + lane_col) / 8 * 1024 + (gram_row_base + lane_row) * 16 + (gram_col_base + lane_col) % 8 * 2)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3]))
                : "memory");
        }
        if (lane == 0) {
            mbarrier_arrive(k_empty_addr);
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        int inverse_thread = warp * 32 + lane;
        if (inverse_thread < 64) {
            int diag_block = inverse_thread / 8;
            int lane_in_diag = lane & 7;
            int diag_col_base = diag_block * 8;
            float inv_row[8];
            unsigned int words[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"((smem_inverse_addr + (unsigned int)(diag_col_base / 8 * 1024 + inverse_thread * 16 + diag_col_base % 8 * 2))));
            #pragma unroll
            for (int word = 0; word < 4; word++) {
                unsigned int hi_bits = words[word] >> 16;
                __half lo_f16 = 0.0f;
                __half hi_f16 = 0.0f;
                lo_f16 = reinterpret_cast<__half*>(&words[word])[0];
                hi_f16 = reinterpret_cast<__half*>(&hi_bits)[0];
                inv_row[word * 2] = (float)lo_f16;
                inv_row[word * 2 + 1] = (float)hi_f16;
            }
            #pragma unroll
            for (int diag_col = 0; diag_col < 8; diag_col++) {
                if (lane_in_diag == diag_col) {
                    inv_row[diag_col] = 1.0f;
                } else if (lane_in_diag < diag_col) {
                    inv_row[diag_col] = 0.0f;
                }
            }
            int subgroup_base = lane - lane_in_diag;
            #pragma unroll
            for (int src_row = 0; src_row < 7; src_row++) {
                float row_scale = -inv_row[src_row];
                #pragma unroll
                for (int prev_col = 0; prev_col < src_row; prev_col++) {
                    int pivot_lane = subgroup_base + src_row;
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, inv_row[prev_col], pivot_lane);
                    float pivot = _shfl_0;
                    if (lane_in_diag > src_row) {
                        float _fma_0 = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                        inv_row[prev_col] = _fma_0;
                    }
                }
                if (lane_in_diag > src_row) {
                    inv_row[src_row] = row_scale;
                }
            }
            unsigned int words_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                words_0[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_inverse_addr + (unsigned int)(diag_col_base / 8 * 1024 + inverse_thread * 16 + diag_col_base % 8 * 2))), "r"(words_0[0]), "r"(words_0[1]), "r"(words_0[2]), "r"(words_0[3]) : "memory");
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        int lane_row_1 = lane & 7;
        unsigned int d_frag[2];
        unsigned int c_frag[1];
        float dc_acc[4];
        unsigned int dc_f16[2];
        unsigned int a_frag[1];
        float out_acc[4];
        unsigned int out_f16[2];
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
            : "=r"(d_frag[0])
            : "r"((smem_inverse_addr + (unsigned int)((warp * 16 + 8) / 8 * 1024 + (warp * 16 + 8 + lane_row_1) * 16 + (warp * 16 + 8) % 8 * 2)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
            : "=r"(d_frag[1])
            : "r"((smem_inverse_addr + (unsigned int)((warp * 16 + 8) / 8 * 1024 + (warp * 16 + 8 + lane_row_1) * 16 + (warp * 16 + 8) % 8 * 2)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
            : "=r"(c_frag[0])
            : "r"((smem_inverse_addr + (unsigned int)(warp * 16 / 8 * 1024 + (warp * 16 + 8 + lane_row_1) * 16 + warp * 16 % 8 * 2)))
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
            : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        const float2 _scale2_1 = {-1.0f, -1.0f};
        #pragma unroll
        for (int _ls = 0; _ls < 2; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_1);
        #pragma unroll
        for (int _lp = 0; _lp < 2; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
            dc_f16[_lp] = *(uint32_t*)&_h2;
        }
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
            : "=r"(a_frag[0])
            : "r"((smem_inverse_addr + (unsigned int)(warp * 16 / 8 * 1024 + (warp * 16 + lane_row_1) * 16 + warp * 16 % 8 * 2)))
            : "memory");
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            : "=f"(out_acc[0]), "=f"(out_acc[1]), "=f"(out_acc[2]), "=f"(out_acc[3])
            : "r"(dc_f16[0]), "r"(dc_f16[1]), "r"(a_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        #pragma unroll
        for (int _lp = 0; _lp < 2; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(out_acc[_lp*2 + 0], out_acc[_lp*2+1 + 0]));
            out_f16[_lp] = *(uint32_t*)&_h2;
        }
        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)(warp * 16 / 8 * 1024 + (warp * 16 + 8 + lane_row_1) * 16 + warp * 16 % 8 * 2)));
        asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&out_f16[0]))
            : "memory");
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        if (inverse_thread < 64) {
            int lane_row_0 = lane % 16;
            unsigned int d_frag_1[4];
            unsigned int c_frag_2[4];
            float dc_acc_3[8];
            unsigned int dc_f16_4[4];
            unsigned int a_frag_5[4];
            float out_acc_6[8];
            unsigned int out_f16_7[4];
            #pragma unroll
            for (int inverse32_repeat = 0; inverse32_repeat < 2; inverse32_repeat++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(d_frag_1[inverse32_repeat * 2]), "=r"(d_frag_1[inverse32_repeat * 2 + 1])
                    : "r"((smem_inverse_addr + (unsigned int)((inverse_thread / 32 * 32 + 16 + inverse32_repeat * 8) / 8 * 1024 + (inverse_thread / 32 * 32 + 16 + lane_row_0) * 16 + (inverse_thread / 32 * 32 + 16 + inverse32_repeat * 8) % 8 * 2)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(c_frag_2[inverse32_repeat * 2]), "=r"(c_frag_2[inverse32_repeat * 2 + 1])
                    : "r"((smem_inverse_addr + (unsigned int)((inverse_thread / 32 * 32 + inverse32_repeat * 8) / 8 * 1024 + (inverse_thread / 32 * 32 + 16 + lane_row_0) * 16 + (inverse_thread / 32 * 32 + inverse32_repeat * 8) % 8 * 2)))
                    : "memory");
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(dc_acc_3[0]), "=f"(dc_acc_3[1]), "=f"(dc_acc_3[2]), "=f"(dc_acc_3[3])
                : "r"(d_frag_1[0]), "r"(d_frag_1[1]), "r"(d_frag_1[2]), "r"(d_frag_1[3]), "r"(c_frag_2[0]), "r"(c_frag_2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(dc_acc_3[4]), "=f"(dc_acc_3[(4) + 1]), "=f"(dc_acc_3[(4) + 2]), "=f"(dc_acc_3[(4) + 3])
                : "r"(d_frag_1[0]), "r"(d_frag_1[1]), "r"(d_frag_1[2]), "r"(d_frag_1[3]), "r"(c_frag_2[2]), "r"(c_frag_2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            const float2 _scale2_3 = {-1.0f, -1.0f};
            #pragma unroll
            for (int _ls = 0; _ls < 4; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc_3)[_ls], _scale2_3);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(dc_acc_3[_lp*2 + 0], dc_acc_3[_lp*2+1 + 0]));
                dc_f16_4[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int inverse32_repeat_1 = 0; inverse32_repeat_1 < 2; inverse32_repeat_1++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(a_frag_5[inverse32_repeat_1 * 2]), "=r"(a_frag_5[inverse32_repeat_1 * 2 + 1])
                    : "r"((smem_inverse_addr + (unsigned int)((inverse_thread / 32 * 32 + inverse32_repeat_1 * 8) / 8 * 1024 + (inverse_thread / 32 * 32 + lane_row_0) * 16 + (inverse_thread / 32 * 32 + inverse32_repeat_1 * 8) % 8 * 2)))
                    : "memory");
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(out_acc_6[0]), "=f"(out_acc_6[1]), "=f"(out_acc_6[2]), "=f"(out_acc_6[3])
                : "r"(dc_f16_4[0]), "r"(dc_f16_4[1]), "r"(dc_f16_4[2]), "r"(dc_f16_4[3]), "r"(a_frag_5[0]), "r"(a_frag_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(out_acc_6[4]), "=f"(out_acc_6[(4) + 1]), "=f"(out_acc_6[(4) + 2]), "=f"(out_acc_6[(4) + 3])
                : "r"(dc_f16_4[0]), "r"(dc_f16_4[1]), "r"(dc_f16_4[2]), "r"(dc_f16_4[3]), "r"(a_frag_5[2]), "r"(a_frag_5[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(out_acc_6[_lp*2 + 0], out_acc_6[_lp*2+1 + 0]));
                out_f16_7[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int inverse32_repeat_2 = 0; inverse32_repeat_2 < 2; inverse32_repeat_2++) {
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)((inverse_thread / 32 * 32 + inverse32_repeat_2 * 8) / 8 * 1024 + (inverse_thread / 32 * 32 + 16 + lane_row_0) * 16 + (inverse_thread / 32 * 32 + inverse32_repeat_2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7[inverse32_repeat_2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_f16_7[inverse32_repeat_2 * 2 + 1]))
                    : "memory");
            }
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        int inv64_x = warp / 2;
        int inv64_y = warp & 1;
        int inv64_lane_row = lane % 16;
        int inv64_lane_col = lane / 16 * 8;
        unsigned int inv64_d[4];
        unsigned int inv64_c[4];
        float inv64_dc_acc[8];
        inv64_dc_acc[0] = 0.0f;
        inv64_dc_acc[1] = 0.0f;
        inv64_dc_acc[2] = 0.0f;
        inv64_dc_acc[3] = 0.0f;
        inv64_dc_acc[4] = 0.0f;
        inv64_dc_acc[5] = 0.0f;
        inv64_dc_acc[6] = 0.0f;
        inv64_dc_acc[7] = 0.0f;
        #pragma unroll
        for (int inv64_k = 0; inv64_k < 2; inv64_k++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(inv64_d[0]), "=r"(inv64_d[1]), "=r"(inv64_d[2]), "=r"(inv64_d[3])
                : "r"((smem_inverse_addr + (unsigned int)((32 + inv64_k * 16 + inv64_lane_col) / 8 * 1024 + (32 + inv64_y * 16 + inv64_lane_row) * 16 + (32 + inv64_k * 16 + inv64_lane_col) % 8 * 2)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(inv64_c[0]), "=r"(inv64_c[1]), "=r"(inv64_c[2]), "=r"(inv64_c[3])
                : "r"((smem_inverse_addr + (unsigned int)((inv64_x * 16 + inv64_lane_col) / 8 * 1024 + (32 + inv64_k * 16 + inv64_lane_row) * 16 + (inv64_x * 16 + inv64_lane_col) % 8 * 2)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(inv64_dc_acc[0]), "+f"(inv64_dc_acc[1]), "+f"(inv64_dc_acc[2]), "+f"(inv64_dc_acc[3])
                : "r"(inv64_d[0]), "r"(inv64_d[1]), "r"(inv64_d[2]), "r"(inv64_d[3]), "r"(inv64_c[0]), "r"(inv64_c[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(inv64_dc_acc[4]), "+f"(inv64_dc_acc[(4) + 1]), "+f"(inv64_dc_acc[(4) + 2]), "+f"(inv64_dc_acc[(4) + 3])
                : "r"(inv64_d[0]), "r"(inv64_d[1]), "r"(inv64_d[2]), "r"(inv64_d[3]), "r"(inv64_c[2]), "r"(inv64_c[(2) + 1]));
        }
        const float2 _scale2_5 = {-1.0f, -1.0f};
        #pragma unroll
        for (int _ls = 0; _ls < 4; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(inv64_dc_acc)[_ls], _scale2_5);
        unsigned int inv64_dc_f16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(inv64_dc_acc[_lp*2 + 0], inv64_dc_acc[_lp*2+1 + 0]));
            inv64_dc_f16[_lp] = *(uint32_t*)&_h2;
        }
        unsigned int inv64_a[8];
        float inv64_out0_acc[8];
        float inv64_out1_acc[8];
        #pragma unroll
        for (int inv64_a_repeat = 0; inv64_a_repeat < 4; inv64_a_repeat++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(inv64_a[inv64_a_repeat * 2]), "=r"(inv64_a[inv64_a_repeat * 2 + 1])
                : "r"((smem_inverse_addr + (unsigned int)(inv64_a_repeat * 8 / 8 * 1024 + (inv64_x * 16 + inv64_lane_row) * 16 + inv64_a_repeat * 8 % 8 * 2)))
                : "memory");
        }
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(inv64_out0_acc[0]), "=f"(inv64_out0_acc[1]), "=f"(inv64_out0_acc[2]), "=f"(inv64_out0_acc[3])
            : "r"(inv64_dc_f16[0]), "r"(inv64_dc_f16[1]), "r"(inv64_dc_f16[2]), "r"(inv64_dc_f16[3]), "r"(inv64_a[0]), "r"(inv64_a[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(inv64_out0_acc[4]), "=f"(inv64_out0_acc[(4) + 1]), "=f"(inv64_out0_acc[(4) + 2]), "=f"(inv64_out0_acc[(4) + 3])
            : "r"(inv64_dc_f16[0]), "r"(inv64_dc_f16[1]), "r"(inv64_dc_f16[2]), "r"(inv64_dc_f16[3]), "r"(inv64_a[2]), "r"(inv64_a[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(inv64_out1_acc[0]), "=f"(inv64_out1_acc[1]), "=f"(inv64_out1_acc[2]), "=f"(inv64_out1_acc[3])
            : "r"(inv64_dc_f16[0]), "r"(inv64_dc_f16[1]), "r"(inv64_dc_f16[2]), "r"(inv64_dc_f16[3]), "r"(inv64_a[4]), "r"(inv64_a[(4) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(inv64_out1_acc[4]), "=f"(inv64_out1_acc[(4) + 1]), "=f"(inv64_out1_acc[(4) + 2]), "=f"(inv64_out1_acc[(4) + 3])
            : "r"(inv64_dc_f16[0]), "r"(inv64_dc_f16[1]), "r"(inv64_dc_f16[2]), "r"(inv64_dc_f16[3]), "r"(inv64_a[6]), "r"(inv64_a[(6) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
        unsigned int inv64_out0_f16[4];
        unsigned int inv64_out1_f16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(inv64_out0_acc[_lp*2 + 0], inv64_out0_acc[_lp*2+1 + 0]));
            inv64_out0_f16[_lp] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(inv64_out1_acc[_lp*2 + 0], inv64_out1_acc[_lp*2+1 + 0]));
            inv64_out1_f16[_lp] = *(uint32_t*)&_h2;
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        if (inv64_x == 0) {
            int lane_row_0_1 = lane % 16;
            int lane_col_1 = lane / 16 * 8;
            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)(lane_col_1 / 8 * 1024 + (32 + inv64_y * 16 + lane_row_0_1) * 16 + lane_col_1 % 8 * 2)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[3]))
                : "memory");
            int lane_row_1_1 = lane % 16;
            int lane_col_2 = lane / 16 * 8;
            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)((16 + lane_col_2) / 8 * 1024 + (32 + inv64_y * 16 + lane_row_1_1) * 16 + (16 + lane_col_2) % 8 * 2)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[3]))
                : "memory");
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        if (inv64_x == 1) {
            unsigned int inv64_red0[4];
            unsigned int inv64_red1[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(inv64_red0[0]), "=r"(inv64_red0[1]), "=r"(inv64_red0[2]), "=r"(inv64_red0[3])
                : "r"((smem_inverse_addr + (unsigned int)(inv64_lane_col / 8 * 1024 + (32 + inv64_y * 16 + inv64_lane_row) * 16 + inv64_lane_col % 8 * 2)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(inv64_red1[0]), "=r"(inv64_red1[1]), "=r"(inv64_red1[2]), "=r"(inv64_red1[3])
                : "r"((smem_inverse_addr + (unsigned int)((16 + inv64_lane_col) / 8 * 1024 + (32 + inv64_y * 16 + inv64_lane_row) * 16 + (16 + inv64_lane_col) % 8 * 2)))
                : "memory");
            float inv64_own0[8];
            float inv64_own1[8];
            float inv64_peer0[8];
            float inv64_peer1[8];
            #pragma unroll
            for (int word_1 = 0; word_1 < 4; word_1++) {
                unsigned int hi_bits_1 = inv64_out0_f16[word_1] >> 16;
                __half lo_f16_1 = 0.0f;
                __half hi_f16_1 = 0.0f;
                lo_f16_1 = reinterpret_cast<__half*>(&inv64_out0_f16[word_1])[0];
                hi_f16_1 = reinterpret_cast<__half*>(&hi_bits_1)[0];
                inv64_own0[word_1 * 2] = (float)lo_f16_1;
                inv64_own0[word_1 * 2 + 1] = (float)hi_f16_1;
            }
            #pragma unroll
            for (int word_2 = 0; word_2 < 4; word_2++) {
                unsigned int hi_bits_2 = inv64_out1_f16[word_2] >> 16;
                __half lo_f16_2 = 0.0f;
                __half hi_f16_2 = 0.0f;
                lo_f16_2 = reinterpret_cast<__half*>(&inv64_out1_f16[word_2])[0];
                hi_f16_2 = reinterpret_cast<__half*>(&hi_bits_2)[0];
                inv64_own1[word_2 * 2] = (float)lo_f16_2;
                inv64_own1[word_2 * 2 + 1] = (float)hi_f16_2;
            }
            #pragma unroll
            for (int word_3 = 0; word_3 < 4; word_3++) {
                unsigned int hi_bits_3 = inv64_red0[word_3] >> 16;
                __half lo_f16_3 = 0.0f;
                __half hi_f16_3 = 0.0f;
                lo_f16_3 = reinterpret_cast<__half*>(&inv64_red0[word_3])[0];
                hi_f16_3 = reinterpret_cast<__half*>(&hi_bits_3)[0];
                inv64_peer0[word_3 * 2] = (float)lo_f16_3;
                inv64_peer0[word_3 * 2 + 1] = (float)hi_f16_3;
            }
            #pragma unroll
            for (int word_4 = 0; word_4 < 4; word_4++) {
                unsigned int hi_bits_4 = inv64_red1[word_4] >> 16;
                __half lo_f16_4 = 0.0f;
                __half hi_f16_4 = 0.0f;
                lo_f16_4 = reinterpret_cast<__half*>(&inv64_red1[word_4])[0];
                hi_f16_4 = reinterpret_cast<__half*>(&hi_bits_4)[0];
                inv64_peer1[word_4 * 2] = (float)lo_f16_4;
                inv64_peer1[word_4 * 2 + 1] = (float)hi_f16_4;
            }
            #pragma unroll
            for (int inv64_i = 0; inv64_i < 8; inv64_i++) {
                inv64_own0[inv64_i] = inv64_own0[inv64_i] + inv64_peer0[inv64_i];
                inv64_own1[inv64_i] = inv64_own1[inv64_i] + inv64_peer1[inv64_i];
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(inv64_own0[_lp*2 + 0], inv64_own0[_lp*2+1 + 0]));
                inv64_out0_f16[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(inv64_own1[_lp*2 + 0], inv64_own1[_lp*2+1 + 0]));
                inv64_out1_f16[_lp] = *(uint32_t*)&_h2;
            }
            int lane_row_0_2 = lane % 16;
            int lane_col_3 = lane / 16 * 8;
            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)(lane_col_3 / 8 * 1024 + (32 + inv64_y * 16 + lane_row_0_2) * 16 + lane_col_3 % 8 * 2)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out0_f16[3]))
                : "memory");
            int lane_row_1_2 = lane % 16;
            int lane_col_2_1 = lane / 16 * 8;
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(smem_inverse_addr + (unsigned int)((16 + lane_col_2_1) / 8 * 1024 + (32 + inv64_y * 16 + lane_row_1_2) * 16 + (16 + lane_col_2_1) % 8 * 2)));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_out1_f16[3]))
                : "memory");
        }
        asm volatile("barrier.sync 6, 128;" ::: "memory");
        unsigned int final_words[4];
        float final_values[8];
        int final_row_base = warp * 16;
        long long t_head_base = ((long long)t_block * (long long)num_sab_heads + (long long)sab_head) * 4096;
        #pragma unroll
        for (int final_n = 0; final_n < 4; final_n++) {
            int final_col_base = final_n * 16;
            int final_lane_row = lane % 16;
            int final_lane_col = lane / 16 * 8;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(final_words[0]), "=r"(final_words[1]), "=r"(final_words[2]), "=r"(final_words[3])
                : "r"((smem_inverse_addr + (unsigned int)((final_col_base + final_lane_col) / 8 * 1024 + (final_row_base + final_lane_row) * 16 + (final_col_base + final_lane_col) % 8 * 2)))
                : "memory");
            #pragma unroll
            for (int word_5 = 0; word_5 < 4; word_5++) {
                unsigned int hi_bits_5 = final_words[word_5] >> 16;
                __half lo_f16_5 = 0.0f;
                __half hi_f16_5 = 0.0f;
                lo_f16_5 = reinterpret_cast<__half*>(&final_words[word_5])[0];
                hi_f16_5 = reinterpret_cast<__half*>(&hi_bits_5)[0];
                final_values[word_5 * 2] = (float)lo_f16_5;
                final_values[word_5 * 2 + 1] = (float)hi_f16_5;
            }
            int final_r0 = final_row_base + lane / 4;
            int final_r1 = final_r0 + 8;
            int final_c0 = final_col_base + lane % 4 * 2;
            float out0 = 0.0f;
            float out1 = 0.0f;
            float out2 = 0.0f;
            float out3 = 0.0f;
            float out4 = 0.0f;
            float out5 = 0.0f;
            float out6 = 0.0f;
            float out7 = 0.0f;
            if (final_r0 < valid_len && final_c0 < valid_len) {
                out0 = -1.0f * smem_beta[final_c0] * final_values[0];
            }
            if (final_r0 < valid_len && valid_len > final_c0 + 1) {
                out1 = -1.0f * smem_beta[final_c0 + 1] * final_values[1];
            }
            if (final_r1 < valid_len && final_c0 < valid_len) {
                out2 = -1.0f * smem_beta[final_c0] * final_values[2];
            }
            if (final_r1 < valid_len && valid_len > final_c0 + 1) {
                out3 = -1.0f * smem_beta[final_c0 + 1] * final_values[3];
            }
            if (final_r0 < valid_len && valid_len > final_c0 + 8) {
                out4 = -1.0f * smem_beta[final_c0 + 8] * final_values[4];
            }
            if (final_r0 < valid_len && valid_len > final_c0 + 9) {
                out5 = -1.0f * smem_beta[final_c0 + 9] * final_values[5];
            }
            if (final_r1 < valid_len && valid_len > final_c0 + 8) {
                out6 = -1.0f * smem_beta[final_c0 + 8] * final_values[6];
            }
            if (final_r1 < valid_len && valid_len > final_c0 + 9) {
                out7 = -1.0f * smem_beta[final_c0 + 9] * final_values[7];
            }
            t[t_head_base + (long long)final_c0 * 64 + (long long)final_r0] = out0;
            t[t_head_base + (long long)(final_c0 + 1) * 64 + (long long)final_r0] = out1;
            t[t_head_base + (long long)final_c0 * 64 + (long long)final_r1] = out2;
            t[t_head_base + (long long)(final_c0 + 1) * 64 + (long long)final_r1] = out3;
            t[t_head_base + (long long)(final_c0 + 8) * 64 + (long long)final_r0] = out4;
            t[t_head_base + (long long)(final_c0 + 9) * 64 + (long long)final_r0] = out5;
            t[t_head_base + (long long)(final_c0 + 8) * 64 + (long long)final_r1] = out6;
            t[t_head_base + (long long)(final_c0 + 9) * 64 + (long long)final_r1] = out7;
        }
        mbarrier_arrive(beta_empty_addr);
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

#undef CAKE_INF
#undef NUM_BETA_PIPE_STAGES
#undef NUM_K_PIPE_STAGES
#undef SMEM_SMEM_BETA_OFF
#undef SMEM_SMEM_BETA_STAGE_BYTES
#undef SMEM_SMEM_BETA_STRIDE
#undef SMEM_SMEM_INVERSE_OFF
#undef SMEM_SMEM_INVERSE_STAGE_BYTES
#undef SMEM_SMEM_INVERSE_STRIDE
#undef SMEM_SMEM_K_OFF
#undef SMEM_SMEM_K_STAGE_BYTES
#undef SMEM_SMEM_K_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef beta_empty_addr
#undef beta_full_addr
#undef k_empty_addr
#undef k_full_addr
#undef smem_beta_addr
#undef smem_inverse_addr
#undef smem_k_addr
// clang-format on
