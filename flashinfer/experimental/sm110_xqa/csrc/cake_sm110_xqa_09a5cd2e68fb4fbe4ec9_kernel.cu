/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include "cake_sm110_xqa_device_abi.cuh"

#define TMEM_NCOLS 512
#define TMEM_S0_OFFSET 0
#define TMEM_S1_OFFSET 64
#define TMEM_P0_OFFSET 128
#define TMEM_P1_OFFSET 160
#define TMEM_O0_OFFSET 256
#define TMEM_O1_OFFSET 384
#define NUM_RING_STAGES 6
#define NUM_Q_PIPE_STAGES 8
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SQ_OFF 1024
#define SMEM_SQ_STAGE_BYTES 131072
#define SMEM_SQ_STRIDE 131072
#define SMEM_SK_OFF 132096
#define SMEM_SK_STAGE_BYTES 16384
#define SMEM_SK_STRIDE 16384
#define SMEM_SV_OFF 132096
#define SMEM_SV_STAGE_BYTES 16384
#define SMEM_SV_STRIDE 16384
#define SMEM_SSTAGE_OFF 1024
#define SMEM_SSTAGE_STAGE_BYTES 67584
#define SMEM_SSTAGE_STRIDE 67584
#define SMEM_SX_OFF 230400
#define SMEM_SX_STAGE_BYTES 2048
#define SMEM_SX_STRIDE 2048
#define SMEM_TOTAL 232448
#define LAUNCH_MIN_BLOCKS 1

#include "cake_sm110_xqa_kv_cache_list_contig.h"

#include <math_constants.h>
#include "cake_sm110_xqa_device_helpers.cuh"
extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,2,1) void
kernel_cake_sm110_xqa_09a5cd2e68fb4fbe4ec9(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const unsigned int* __restrict__ mask, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define landed_addr (mbar_base + 0)
    #define ready_addr (mbar_base + 48)
    #define free_addr (mbar_base + 96)
    #define q_ready_addr (mbar_base + 144)
    #define s_full_addr (mbar_base + 208)
    #define s_empty_addr (mbar_base + 224)
    #define p_full_addr (mbar_base + 240)
    #define p_empty_addr (mbar_base + 256)
    #define tmem_retire_addr (mbar_base + 272)
    #define o_ready_addr (mbar_base + 280)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * (gridDim.y / 2) + (blockIdx.y / 2)) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * (gridDim.y / 2) * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __half* sq = reinterpret_cast<__half*>(smem_raw + 1024);
    const int sq_addr = smem + 1024;
    __half* sk = reinterpret_cast<__half*>(smem_raw + 132096);
    const int sk_addr = smem + 132096;
    __half* sv = reinterpret_cast<__half*>(smem_raw + 132096);
    const int sv_addr = smem + 132096;
    unsigned int* sstage = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int sstage_addr = smem + 1024;
    float* sx = reinterpret_cast<float*>(smem_raw + 230400);
    const int sx_addr = smem + 230400;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory");
    if (warp == 13) {
        if (elect_sync()) {
            unsigned int blocks_per_head = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head = (unsigned int)blockIdx.y / blocks_per_head;
            unsigned int qpf_row_begin = (unsigned int)blockIdx.y % blocks_per_head * 128;
            unsigned int qpf_token0 = q_seq_len * (unsigned int)blockIdx.z + qpf_row_begin / 8;
            unsigned int qpf_qhead = head * 8;
            unsigned int qpf_odd = (unsigned int)cta_rank & 1;
            #pragma unroll
            for (int pair = 0; pair < 2; pair++) {
                if (qpf_odd == (unsigned int)(pair & 1)) {
                    #pragma unroll
                    for (int slab = 2 * pair; slab < 2 * pair + 2; slab++) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&Q))), "r"((int)((unsigned int)(slab * 64))), "r"((int)(qpf_qhead)), "r"((int)(qpf_token0)) : "memory");
                    }
                }
            }
        }
    }
    int _mma_base_lo_0 = ((sq_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((sk_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_2 = ((sq_addr + 16384) >> 4) & 0x3FFF;
    int _mma_base_lo_3 = ((sk_addr + 8192) >> 4) & 0x3FFF;
    int _mma_base_lo_4 = ((sq_addr + 32768) >> 4) & 0x3FFF;
    int _mma_base_lo_5 = ((sq_addr + 49152) >> 4) & 0x3FFF;
    int _mma_base_lo_6 = ((sq_addr + 65536) >> 4) & 0x3FFF;
    int _mma_base_lo_7 = ((sq_addr + 81920) >> 4) & 0x3FFF;
    int _mma_base_lo_8 = ((sq_addr + 98304) >> 4) & 0x3FFF;
    int _mma_base_lo_9 = ((sq_addr + 114688) >> 4) & 0x3FFF;
    int _mma_base_lo_10 = ((sq_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_11 = ((sk_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_12 = ((sq_addr + 32768) >> 4) & 0x3FFF;
    int _mma_base_lo_13 = ((sq_addr + 65536) >> 4) & 0x3FFF;
    int _mma_base_lo_14 = ((sq_addr + 98304) >> 4) & 0x3FFF;
    int _mma_base_lo_15 = (((sv_addr) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_16 = (((sv_addr + 2048) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_17 = (((sv_addr + 4096) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_18 = (((sv_addr + 6144) >> 4) & 0x3FFF) | 0x2000000;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[0..288)


    // Mbarrier init group 1
    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // --- pipeline 'q_pipe' ---
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // --- pipeline 's_pipe' ---
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            // --- pipeline 'p_pipe' ---
            mbarrier_init(smem + 240, 256);
            mbarrier_init(smem + 248, 256);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 256);
            mbarrier_init(smem + 280, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 12) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp <= 12) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(3), "r"(416) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp <= 12) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_s0 = taddr;
    const int tmem_s1 = taddr + 64;
    const int tmem_p0 = taddr + 128;
    const int tmem_p1 = taddr + 160;
    const int tmem_o0 = taddr + 256;
    const int tmem_o1 = taddr + 384;

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // softmax_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int actual_q = q_seq_len;
            unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset = q_cu_seq_lens[blockIdx.z];
                actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
            }
            unsigned int blocks_per_head_1 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_1 = (unsigned int)blockIdx.y / blocks_per_head_1;
            unsigned int row_begin = (unsigned int)blockIdx.y % blocks_per_head_1 * 128;
            unsigned int output_col = blockIdx.x * 256;
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length = (unsigned int)_vec_load_2[0];
            unsigned int num_tiles = (length + 63) / 64;
            int row_base = warp % 4 * 32 << 16;
            unsigned int my_row = warp % 4 * 32 + lane;
            unsigned int head_token = row_begin + my_row;
            int row_valid = ((head_token < actual_q * 8) ? 1 : 0);
            unsigned int query_token = head_token / 8;
            unsigned int prefix = length - actual_q;
            unsigned int mask_stride = (q_seq_len + 31) / 32;
            unsigned int _min_0 = ((query_token) < (actual_q - 1) ? (query_token) : (actual_q - 1));
            unsigned int mask_row = (request_offset + _min_0) * mask_stride;
            unsigned int _vec_load_3[1];
            {
                _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(mask + mask_row);
            }
            unsigned int mw0 = _vec_load_3[0];
            unsigned int mw1 = 0;
            unsigned int mw2 = 0;
            unsigned int mw3 = 0;
            unsigned int zero32 = 0;
            if (mask_stride > 1) {
                unsigned int _vec_load_4[1];
                {
                    _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 1));
                }
                mw1 = _vec_load_4[0];
            }
            if (mask_stride > 2) {
                unsigned int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 2));
                }
                mw2 = _vec_load_5[0];
            }
            if (mask_stride > 3) {
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 3));
                }
                mw3 = _vec_load_6[0];
            }
            unsigned int pk[16];
            float scale = attention_scale * 1.442695f * k_cache_scale;
            float m_used = -CUDART_INF_F;
            float row_sum = 0.0f;
            unsigned int sm_s_stage = 0;
            unsigned int sm_p_stage = 0;
            unsigned int packed[32];
            unsigned int half = (unsigned int)warp / 4;
            int col0 = (int)half * 32;
            unsigned int other = 1 - half;
            unsigned int x_max_mine = half * 128 + my_row;
            unsigned int x_max_other = other * 128 + my_row;
            unsigned int x_sum_mine = x_max_mine + 256;
            unsigned int x_sum_other = x_max_other + 256;
            unsigned int o_col0 = half * 128;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = 0; tile < num_tiles; tile++) {
                mbarrier_wait_hint(s_full_addr + (sm_s_stage) * 8, _phase_s_full, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int s_addr = taddr + sm_s_stage * 64 + (unsigned int)row_base;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(s_addr + col0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_empty_addr + (sm_s_stage) * 8);
                sm_s_stage += 1;
                if (sm_s_stage == 2) { sm_s_stage = 0; _phase_s_full ^= 1; }
                unsigned int tile_token0 = tile * 64;
                if (row_valid == 0) {
                    if (!(0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                } else {
                    if (length < tile_token0 + 64) {
                        int valid = (int)(length - tile_token0) - col0;
                        if (valid <= 0) {
                            if (!(0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                            if (!(0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                            if (!(0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                            if (!(0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                            if (!(0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                            if (!(0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                            if (!(0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                            if (!(0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                            if (!(0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                            if (!(0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                            if (!(0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                            if (!(0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                            if (!(0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                            if (!(0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                            if (!(0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                            if (!(0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                            if (!(0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                            if (!(0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                            if (!(0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                            if (!(0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                            if (!(0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                            if (!(0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                            if (!(0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                            if (!(0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                            if (!(0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                            if (!(0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                            if (!(0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                            if (!(0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                            if (!(0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                            if (!(0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                            if (!(0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                            if (!(0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                        } else if (valid < 32) {
                            uint32_t _thresh_mask_1;
                            {
                                int _lim_2 = valid;
                                if (_lim_2 <= 0) { _thresh_mask_1 = 0u; }
                                else if (_lim_2 >= 32) { _thresh_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_1) : "r"(_lim_2));
                                }
                            }
                            if (!(_thresh_mask_1 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_1 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                        }
                    }
                    if (prefix < tile_token0 + 64) {
                        int tree0 = tile_token0 - prefix;
                        if (mask_stride <= 4) {
                            unsigned int vis_lo = 4294967295;
                            unsigned int vis_hi = 4294967295;
                            if (tree0 >= 0) {
                                unsigned int widx = (unsigned int)tree0 >> 5;
                                unsigned int shift = (unsigned int)tree0 & 31;
                                unsigned int pj0 = ((widx == 0) ? mw0 : ((widx == 1) ? mw1 : ((widx == 2) ? mw2 : ((widx == 3) ? mw3 : zero32))));
                                unsigned int pj1 = ((widx + 1 == 0) ? mw0 : ((widx + 1 == 1) ? mw1 : ((widx + 1 == 2) ? mw2 : ((widx + 1 == 3) ? mw3 : zero32))));
                                unsigned int pj2 = ((widx + 2 == 0) ? mw0 : ((widx + 2 == 1) ? mw1 : ((widx + 2 == 2) ? mw2 : ((widx + 2 == 3) ? mw3 : zero32))));
                                vis_lo = pj0 >> shift | pj1 << 31 - shift << 1;
                                vis_hi = pj1 >> shift | pj2 << 31 - shift << 1;
                            } else {
                                unsigned int lead = (unsigned int)(-tree0);
                                if (lead < 32) {
                                    vis_lo = mw0 << lead | ((unsigned int)1 << lead) - 1;
                                    vis_hi = mw1 << lead | mw0 >> 32 - lead;
                                } else {
                                    unsigned int lead32 = lead - 32;
                                    vis_hi = mw0 << lead32 | ((unsigned int)1 << lead32) - 1;
                                }
                            }
                            unsigned int vis = ((half == 0) ? vis_lo : vis_hi);
                            _tmem_load_0[0] = (((vis & 1) != 0) ? _tmem_load_0[0] : -CUDART_INF_F);
                            _tmem_load_0[1] = (((vis >> 1 & 1) != 0) ? _tmem_load_0[1] : -CUDART_INF_F);
                            _tmem_load_0[2] = (((vis >> 2 & 1) != 0) ? _tmem_load_0[2] : -CUDART_INF_F);
                            _tmem_load_0[3] = (((vis >> 3 & 1) != 0) ? _tmem_load_0[3] : -CUDART_INF_F);
                            _tmem_load_0[4] = (((vis >> 4 & 1) != 0) ? _tmem_load_0[4] : -CUDART_INF_F);
                            _tmem_load_0[5] = (((vis >> 5 & 1) != 0) ? _tmem_load_0[5] : -CUDART_INF_F);
                            _tmem_load_0[6] = (((vis >> 6 & 1) != 0) ? _tmem_load_0[6] : -CUDART_INF_F);
                            _tmem_load_0[7] = (((vis >> 7 & 1) != 0) ? _tmem_load_0[7] : -CUDART_INF_F);
                            _tmem_load_0[8] = (((vis >> 8 & 1) != 0) ? _tmem_load_0[8] : -CUDART_INF_F);
                            _tmem_load_0[9] = (((vis >> 9 & 1) != 0) ? _tmem_load_0[9] : -CUDART_INF_F);
                            _tmem_load_0[10] = (((vis >> 10 & 1) != 0) ? _tmem_load_0[10] : -CUDART_INF_F);
                            _tmem_load_0[11] = (((vis >> 11 & 1) != 0) ? _tmem_load_0[11] : -CUDART_INF_F);
                            _tmem_load_0[12] = (((vis >> 12 & 1) != 0) ? _tmem_load_0[12] : -CUDART_INF_F);
                            _tmem_load_0[13] = (((vis >> 13 & 1) != 0) ? _tmem_load_0[13] : -CUDART_INF_F);
                            _tmem_load_0[14] = (((vis >> 14 & 1) != 0) ? _tmem_load_0[14] : -CUDART_INF_F);
                            _tmem_load_0[15] = (((vis >> 15 & 1) != 0) ? _tmem_load_0[15] : -CUDART_INF_F);
                            _tmem_load_0[16] = (((vis >> 16 & 1) != 0) ? _tmem_load_0[16] : -CUDART_INF_F);
                            _tmem_load_0[17] = (((vis >> 17 & 1) != 0) ? _tmem_load_0[17] : -CUDART_INF_F);
                            _tmem_load_0[18] = (((vis >> 18 & 1) != 0) ? _tmem_load_0[18] : -CUDART_INF_F);
                            _tmem_load_0[19] = (((vis >> 19 & 1) != 0) ? _tmem_load_0[19] : -CUDART_INF_F);
                            _tmem_load_0[20] = (((vis >> 20 & 1) != 0) ? _tmem_load_0[20] : -CUDART_INF_F);
                            _tmem_load_0[21] = (((vis >> 21 & 1) != 0) ? _tmem_load_0[21] : -CUDART_INF_F);
                            _tmem_load_0[22] = (((vis >> 22 & 1) != 0) ? _tmem_load_0[22] : -CUDART_INF_F);
                            _tmem_load_0[23] = (((vis >> 23 & 1) != 0) ? _tmem_load_0[23] : -CUDART_INF_F);
                            _tmem_load_0[24] = (((vis >> 24 & 1) != 0) ? _tmem_load_0[24] : -CUDART_INF_F);
                            _tmem_load_0[25] = (((vis >> 25 & 1) != 0) ? _tmem_load_0[25] : -CUDART_INF_F);
                            _tmem_load_0[26] = (((vis >> 26 & 1) != 0) ? _tmem_load_0[26] : -CUDART_INF_F);
                            _tmem_load_0[27] = (((vis >> 27 & 1) != 0) ? _tmem_load_0[27] : -CUDART_INF_F);
                            _tmem_load_0[28] = (((vis >> 28 & 1) != 0) ? _tmem_load_0[28] : -CUDART_INF_F);
                            _tmem_load_0[29] = (((vis >> 29 & 1) != 0) ? _tmem_load_0[29] : -CUDART_INF_F);
                            _tmem_load_0[30] = (((vis >> 30 & 1) != 0) ? _tmem_load_0[30] : -CUDART_INF_F);
                            _tmem_load_0[31] = (((vis >> 31 & 1) != 0) ? _tmem_load_0[31] : -CUDART_INF_F);
                        } else {
                            int _max_3 = ((tree0) > (0) ? (tree0) : (0));
                            unsigned int word0 = _max_3 >> 5;
                            unsigned int _vec_load_7[1];
                            {
                                _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0));
                            }
                            unsigned int w0 = _vec_load_7[0];
                            unsigned int w1 = 0;
                            unsigned int w2 = 0;
                            if (mask_stride > word0 + 1) {
                                unsigned int _vec_load_8[1];
                                {
                                    _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 1));
                                }
                                w1 = _vec_load_8[0];
                            }
                            if (mask_stride > word0 + 2) {
                                unsigned int _vec_load_9[1];
                                {
                                    _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 2));
                                }
                                w2 = _vec_load_9[0];
                            }
                            int _max_0 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[0] = ((tree0 + col0 < 0 || ((((tree0 + col0 >> 5) - (_max_0 >> 5) == 0) ? w0 : (((tree0 + col0 >> 5) - (_max_0 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 & 31) & 1) != 0) ? _tmem_load_0[0] : -CUDART_INF_F);
                            int _max_1 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[1] = ((tree0 + col0 + 1 < 0 || ((((tree0 + col0 + 1 >> 5) - (_max_1 >> 5) == 0) ? w0 : (((tree0 + col0 + 1 >> 5) - (_max_1 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 1 & 31) & 1) != 0) ? _tmem_load_0[1] : -CUDART_INF_F);
                            int _max_2 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[2] = ((tree0 + col0 + 2 < 0 || ((((tree0 + col0 + 2 >> 5) - (_max_2 >> 5) == 0) ? w0 : (((tree0 + col0 + 2 >> 5) - (_max_2 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 2 & 31) & 1) != 0) ? _tmem_load_0[2] : -CUDART_INF_F);
                            int _max_3_1 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[3] = ((tree0 + col0 + 3 < 0 || ((((tree0 + col0 + 3 >> 5) - (_max_3_1 >> 5) == 0) ? w0 : (((tree0 + col0 + 3 >> 5) - (_max_3_1 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 3 & 31) & 1) != 0) ? _tmem_load_0[3] : -CUDART_INF_F);
                            int _max_4 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[4] = ((tree0 + col0 + 4 < 0 || ((((tree0 + col0 + 4 >> 5) - (_max_4 >> 5) == 0) ? w0 : (((tree0 + col0 + 4 >> 5) - (_max_4 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 4 & 31) & 1) != 0) ? _tmem_load_0[4] : -CUDART_INF_F);
                            int _max_5 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[5] = ((tree0 + col0 + 5 < 0 || ((((tree0 + col0 + 5 >> 5) - (_max_5 >> 5) == 0) ? w0 : (((tree0 + col0 + 5 >> 5) - (_max_5 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 5 & 31) & 1) != 0) ? _tmem_load_0[5] : -CUDART_INF_F);
                            int _max_6 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[6] = ((tree0 + col0 + 6 < 0 || ((((tree0 + col0 + 6 >> 5) - (_max_6 >> 5) == 0) ? w0 : (((tree0 + col0 + 6 >> 5) - (_max_6 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 6 & 31) & 1) != 0) ? _tmem_load_0[6] : -CUDART_INF_F);
                            int _max_7 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[7] = ((tree0 + col0 + 7 < 0 || ((((tree0 + col0 + 7 >> 5) - (_max_7 >> 5) == 0) ? w0 : (((tree0 + col0 + 7 >> 5) - (_max_7 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 7 & 31) & 1) != 0) ? _tmem_load_0[7] : -CUDART_INF_F);
                            int _max_8 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[8] = ((tree0 + col0 + 8 < 0 || ((((tree0 + col0 + 8 >> 5) - (_max_8 >> 5) == 0) ? w0 : (((tree0 + col0 + 8 >> 5) - (_max_8 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 8 & 31) & 1) != 0) ? _tmem_load_0[8] : -CUDART_INF_F);
                            int _max_9 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[9] = ((tree0 + col0 + 9 < 0 || ((((tree0 + col0 + 9 >> 5) - (_max_9 >> 5) == 0) ? w0 : (((tree0 + col0 + 9 >> 5) - (_max_9 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 9 & 31) & 1) != 0) ? _tmem_load_0[9] : -CUDART_INF_F);
                            int _max_10 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[10] = ((tree0 + col0 + 10 < 0 || ((((tree0 + col0 + 10 >> 5) - (_max_10 >> 5) == 0) ? w0 : (((tree0 + col0 + 10 >> 5) - (_max_10 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 10 & 31) & 1) != 0) ? _tmem_load_0[10] : -CUDART_INF_F);
                            int _max_11 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[11] = ((tree0 + col0 + 11 < 0 || ((((tree0 + col0 + 11 >> 5) - (_max_11 >> 5) == 0) ? w0 : (((tree0 + col0 + 11 >> 5) - (_max_11 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 11 & 31) & 1) != 0) ? _tmem_load_0[11] : -CUDART_INF_F);
                            int _max_12 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[12] = ((tree0 + col0 + 12 < 0 || ((((tree0 + col0 + 12 >> 5) - (_max_12 >> 5) == 0) ? w0 : (((tree0 + col0 + 12 >> 5) - (_max_12 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 12 & 31) & 1) != 0) ? _tmem_load_0[12] : -CUDART_INF_F);
                            int _max_13 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[13] = ((tree0 + col0 + 13 < 0 || ((((tree0 + col0 + 13 >> 5) - (_max_13 >> 5) == 0) ? w0 : (((tree0 + col0 + 13 >> 5) - (_max_13 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 13 & 31) & 1) != 0) ? _tmem_load_0[13] : -CUDART_INF_F);
                            int _max_14 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[14] = ((tree0 + col0 + 14 < 0 || ((((tree0 + col0 + 14 >> 5) - (_max_14 >> 5) == 0) ? w0 : (((tree0 + col0 + 14 >> 5) - (_max_14 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 14 & 31) & 1) != 0) ? _tmem_load_0[14] : -CUDART_INF_F);
                            int _max_15 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[15] = ((tree0 + col0 + 15 < 0 || ((((tree0 + col0 + 15 >> 5) - (_max_15 >> 5) == 0) ? w0 : (((tree0 + col0 + 15 >> 5) - (_max_15 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 15 & 31) & 1) != 0) ? _tmem_load_0[15] : -CUDART_INF_F);
                            int _max_16 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[16] = ((tree0 + col0 + 16 < 0 || ((((tree0 + col0 + 16 >> 5) - (_max_16 >> 5) == 0) ? w0 : (((tree0 + col0 + 16 >> 5) - (_max_16 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 16 & 31) & 1) != 0) ? _tmem_load_0[16] : -CUDART_INF_F);
                            int _max_17 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[17] = ((tree0 + col0 + 17 < 0 || ((((tree0 + col0 + 17 >> 5) - (_max_17 >> 5) == 0) ? w0 : (((tree0 + col0 + 17 >> 5) - (_max_17 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 17 & 31) & 1) != 0) ? _tmem_load_0[17] : -CUDART_INF_F);
                            int _max_18 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[18] = ((tree0 + col0 + 18 < 0 || ((((tree0 + col0 + 18 >> 5) - (_max_18 >> 5) == 0) ? w0 : (((tree0 + col0 + 18 >> 5) - (_max_18 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 18 & 31) & 1) != 0) ? _tmem_load_0[18] : -CUDART_INF_F);
                            int _max_19 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[19] = ((tree0 + col0 + 19 < 0 || ((((tree0 + col0 + 19 >> 5) - (_max_19 >> 5) == 0) ? w0 : (((tree0 + col0 + 19 >> 5) - (_max_19 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 19 & 31) & 1) != 0) ? _tmem_load_0[19] : -CUDART_INF_F);
                            int _max_20 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[20] = ((tree0 + col0 + 20 < 0 || ((((tree0 + col0 + 20 >> 5) - (_max_20 >> 5) == 0) ? w0 : (((tree0 + col0 + 20 >> 5) - (_max_20 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 20 & 31) & 1) != 0) ? _tmem_load_0[20] : -CUDART_INF_F);
                            int _max_21 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[21] = ((tree0 + col0 + 21 < 0 || ((((tree0 + col0 + 21 >> 5) - (_max_21 >> 5) == 0) ? w0 : (((tree0 + col0 + 21 >> 5) - (_max_21 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 21 & 31) & 1) != 0) ? _tmem_load_0[21] : -CUDART_INF_F);
                            int _max_22 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[22] = ((tree0 + col0 + 22 < 0 || ((((tree0 + col0 + 22 >> 5) - (_max_22 >> 5) == 0) ? w0 : (((tree0 + col0 + 22 >> 5) - (_max_22 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 22 & 31) & 1) != 0) ? _tmem_load_0[22] : -CUDART_INF_F);
                            int _max_23 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[23] = ((tree0 + col0 + 23 < 0 || ((((tree0 + col0 + 23 >> 5) - (_max_23 >> 5) == 0) ? w0 : (((tree0 + col0 + 23 >> 5) - (_max_23 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 23 & 31) & 1) != 0) ? _tmem_load_0[23] : -CUDART_INF_F);
                            int _max_24 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[24] = ((tree0 + col0 + 24 < 0 || ((((tree0 + col0 + 24 >> 5) - (_max_24 >> 5) == 0) ? w0 : (((tree0 + col0 + 24 >> 5) - (_max_24 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 24 & 31) & 1) != 0) ? _tmem_load_0[24] : -CUDART_INF_F);
                            int _max_25 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[25] = ((tree0 + col0 + 25 < 0 || ((((tree0 + col0 + 25 >> 5) - (_max_25 >> 5) == 0) ? w0 : (((tree0 + col0 + 25 >> 5) - (_max_25 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 25 & 31) & 1) != 0) ? _tmem_load_0[25] : -CUDART_INF_F);
                            int _max_26 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[26] = ((tree0 + col0 + 26 < 0 || ((((tree0 + col0 + 26 >> 5) - (_max_26 >> 5) == 0) ? w0 : (((tree0 + col0 + 26 >> 5) - (_max_26 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 26 & 31) & 1) != 0) ? _tmem_load_0[26] : -CUDART_INF_F);
                            int _max_27 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[27] = ((tree0 + col0 + 27 < 0 || ((((tree0 + col0 + 27 >> 5) - (_max_27 >> 5) == 0) ? w0 : (((tree0 + col0 + 27 >> 5) - (_max_27 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 27 & 31) & 1) != 0) ? _tmem_load_0[27] : -CUDART_INF_F);
                            int _max_28 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[28] = ((tree0 + col0 + 28 < 0 || ((((tree0 + col0 + 28 >> 5) - (_max_28 >> 5) == 0) ? w0 : (((tree0 + col0 + 28 >> 5) - (_max_28 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 28 & 31) & 1) != 0) ? _tmem_load_0[28] : -CUDART_INF_F);
                            int _max_29 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[29] = ((tree0 + col0 + 29 < 0 || ((((tree0 + col0 + 29 >> 5) - (_max_29 >> 5) == 0) ? w0 : (((tree0 + col0 + 29 >> 5) - (_max_29 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 29 & 31) & 1) != 0) ? _tmem_load_0[29] : -CUDART_INF_F);
                            int _max_30 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[30] = ((tree0 + col0 + 30 < 0 || ((((tree0 + col0 + 30 >> 5) - (_max_30 >> 5) == 0) ? w0 : (((tree0 + col0 + 30 >> 5) - (_max_30 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 30 & 31) & 1) != 0) ? _tmem_load_0[30] : -CUDART_INF_F);
                            int _max_31 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[31] = ((tree0 + col0 + 31 < 0 || ((((tree0 + col0 + 31 >> 5) - (_max_31 >> 5) == 0) ? w0 : (((tree0 + col0 + 31 >> 5) - (_max_31 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + col0 + 31 & 31) & 1) != 0) ? _tmem_load_0[31] : -CUDART_INF_F);
                        }
                    }
                }
                float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
                float my_max = row_max_reduce(_reg_reduce_max2_4);
                sx[x_max_mine] = my_max;
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 4, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float other_max = sx[x_max_other];
                float tile_max;
                asm("max.f32 %0, %1, %2;" : "=f"(tile_max) : "f"(my_max), "f"(other_max));
                float tile_max_scaled = ((tile_max != -CUDART_INF_F) ? tile_max * scale : -CUDART_INF_F);
                float m_new;
                asm("max.f32 %0, %1, %2;" : "=f"(m_new) : "f"(m_used), "f"(tile_max_scaled));
                float need_rescale = ((m_used != -CUDART_INF_F && m_new > m_used + 8.0f) ? 1.0f : 0.0f);
                float alpha = 1.0f;
                if (need_rescale != 0.0f || m_used == -CUDART_INF_F) {
                    float _exp2_5 = approx_exp2(m_used - m_new);
                    alpha = ((m_used != -CUDART_INF_F) ? _exp2_5 : 1.0f);
                    m_used = m_new;
                }
                mbarrier_wait_hint(p_empty_addr + (sm_p_stage) * 8, _phase_p_empty, 10000000);
                float _warp_reduce_6 = need_rescale;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_6 = max_noftz(_warp_reduce_6, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset));
                if (_warp_reduce_6 != 0.0f) {
                    unsigned int lag_stage = tile + 1 & 1;
                    unsigned int lag_phase = (tile + 1 >> 1) + 1 & 1;
                    mbarrier_wait_hint(p_empty_addr + (lag_stage) * 8, lag_phase, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + 256 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_7 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_7);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_1[_ls] = _tmem_load_1[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 256 + o_col0 + (unsigned int)row_base, _tmem_load_1);
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 272 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_8 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_8);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_2[_ls] = _tmem_load_2[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 272 + o_col0 + (unsigned int)row_base, _tmem_load_2);
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 288 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_9 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_9);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_3[_ls] = _tmem_load_3[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 288 + o_col0 + (unsigned int)row_base, _tmem_load_3);
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 304 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_10 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_10);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_4[_ls] = _tmem_load_4[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 304 + o_col0 + (unsigned int)row_base, _tmem_load_4);
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 320 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_11 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_11);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_5[_ls] = _tmem_load_5[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 320 + o_col0 + (unsigned int)row_base, _tmem_load_5);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 336 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_12 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_12);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_6[_ls] = _tmem_load_6[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 336 + o_col0 + (unsigned int)row_base, _tmem_load_6);
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 352 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_13 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_13);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_7[_ls] = _tmem_load_7[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 352 + o_col0 + (unsigned int)row_base, _tmem_load_7);
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 368 + o_col0 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_14 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_14);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++) {
                        _tmem_load_8[_ls] = _tmem_load_8[_ls] * alpha;
                    }
                    #endif
                    tmem_st_x16_f32(taddr + 368 + o_col0 + (unsigned int)row_base, _tmem_load_8);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                }
                float bias = ((m_used != -CUDART_INF_F) ? -m_used : 0.0f);
                const float2 _fma_b2_15 = {scale, scale};
                const float2 _fma_c2_16 = {bias, bias};
                float2 _fma_pair_17 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[0] = _fma_pair_17.x;
                _tmem_load_0[1] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[2] = _fma_pair_18.x;
                _tmem_load_0[3] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[4] = _fma_pair_19.x;
                _tmem_load_0[5] = _fma_pair_19.y;
                float2 _fma_pair_20 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[6] = _fma_pair_20.x;
                _tmem_load_0[7] = _fma_pair_20.y;
                float2 _fma_pair_21 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[8] = _fma_pair_21.x;
                _tmem_load_0[9] = _fma_pair_21.y;
                float2 _fma_pair_22 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[10] = _fma_pair_22.x;
                _tmem_load_0[11] = _fma_pair_22.y;
                float2 _fma_pair_23 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[12] = _fma_pair_23.x;
                _tmem_load_0[13] = _fma_pair_23.y;
                float2 _fma_pair_24 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[14] = _fma_pair_24.x;
                _tmem_load_0[15] = _fma_pair_24.y;
                float2 _fma_pair_25 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[16] = _fma_pair_25.x;
                _tmem_load_0[17] = _fma_pair_25.y;
                float2 _fma_pair_26 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[18] = _fma_pair_26.x;
                _tmem_load_0[19] = _fma_pair_26.y;
                float2 _fma_pair_27 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[20] = _fma_pair_27.x;
                _tmem_load_0[21] = _fma_pair_27.y;
                float2 _fma_pair_28 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[22] = _fma_pair_28.x;
                _tmem_load_0[23] = _fma_pair_28.y;
                float2 _fma_pair_29 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[24] = _fma_pair_29.x;
                _tmem_load_0[25] = _fma_pair_29.y;
                float2 _fma_pair_30 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[26] = _fma_pair_30.x;
                _tmem_load_0[27] = _fma_pair_30.y;
                float2 _fma_pair_31 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[28] = _fma_pair_31.x;
                _tmem_load_0[29] = _fma_pair_31.y;
                float2 _fma_pair_32 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_15, _fma_c2_16);
                _tmem_load_0[30] = _fma_pair_32.x;
                _tmem_load_0[31] = _fma_pair_32.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float2 _reg_reduce_sum2_33 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_sum2_33 = add_f32x2(_reg_reduce_sum2_33, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                float my_sum = _reg_reduce_sum2_33.x + _reg_reduce_sum2_33.y;
                sx[x_sum_mine] = my_sum;
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 4, 256;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float other_sum = sx[x_sum_other];
                float lo_sum = ((half == 0) ? my_sum : other_sum);
                float hi_sum = ((half == 0) ? other_sum : my_sum);
                row_sum = __fmaf_rn(row_sum, alpha, lo_sum) + hi_sum;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_h2;
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int p_addr = taddr + 128 + sm_p_stage * 32 + half * 16 + (unsigned int)row_base;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr + (sm_p_stage) * 8);
                sm_p_stage += 1;
                if (sm_p_stage == 2) { sm_p_stage = 0; _phase_p_empty ^= 1; }
            }
            unsigned int _phase_o_ready_0 = 0;
            mbarrier_wait_hint(o_ready_addr, _phase_o_ready_0, 10000000);
            _phase_o_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _rcp_34 = approx_rcp(row_sum);
            float inverse = ((row_sum > 0.0f) ? _rcp_34 : 0.0f);
            inverse = inverse * v_cache_scale;
            float _tmem_load_9[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                : "r"(taddr + 256 + o_col0 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #if __CUDA_ARCH__ >= 1000
            const float2 _scale2_35 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _scale2_35);
            #else
            #pragma unroll
            for (int _ls = 0; _ls < 32; _ls++) {
                _tmem_load_9[_ls] = _tmem_load_9[_ls] * inverse;
            }
            #endif
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_10[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31])
                : "r"(taddr + 288 + o_col0 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #if __CUDA_ARCH__ >= 1000
            const float2 _scale2_36 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_ls], _scale2_36);
            #else
            #pragma unroll
            for (int _ls = 0; _ls < 32; _ls++) {
                _tmem_load_10[_ls] = _tmem_load_10[_ls] * inverse;
            }
            #endif
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 64)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 64 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 64 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 64 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_11[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31])
                : "r"(taddr + 320 + o_col0 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #if __CUDA_ARCH__ >= 1000
            const float2 _scale2_37 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], _scale2_37);
            #else
            #pragma unroll
            for (int _ls = 0; _ls < 32; _ls++) {
                _tmem_load_11[_ls] = _tmem_load_11[_ls] * inverse;
            }
            #endif
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 128)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 128 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 128 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 128 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_12[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_12[0]), "=f"(_tmem_load_12[1]), "=f"(_tmem_load_12[2]), "=f"(_tmem_load_12[3]), "=f"(_tmem_load_12[4]), "=f"(_tmem_load_12[5]), "=f"(_tmem_load_12[6]), "=f"(_tmem_load_12[7]), "=f"(_tmem_load_12[8]), "=f"(_tmem_load_12[9]), "=f"(_tmem_load_12[10]), "=f"(_tmem_load_12[11]), "=f"(_tmem_load_12[12]), "=f"(_tmem_load_12[13]), "=f"(_tmem_load_12[14]), "=f"(_tmem_load_12[15]), "=f"(_tmem_load_12[16]), "=f"(_tmem_load_12[17]), "=f"(_tmem_load_12[18]), "=f"(_tmem_load_12[19]), "=f"(_tmem_load_12[20]), "=f"(_tmem_load_12[21]), "=f"(_tmem_load_12[22]), "=f"(_tmem_load_12[23]), "=f"(_tmem_load_12[24]), "=f"(_tmem_load_12[25]), "=f"(_tmem_load_12[26]), "=f"(_tmem_load_12[27]), "=f"(_tmem_load_12[28]), "=f"(_tmem_load_12[29]), "=f"(_tmem_load_12[30]), "=f"(_tmem_load_12[31])
                : "r"(taddr + 352 + o_col0 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #if __CUDA_ARCH__ >= 1000
            const float2 _scale2_38 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_38);
            #else
            #pragma unroll
            for (int _ls = 0; _ls < 32; _ls++) {
                _tmem_load_12[_ls] = _tmem_load_12[_ls] * inverse;
            }
            #endif
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 192)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 192 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 192 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + o_col0 * 2 + 192 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_retire_addr);
            asm volatile("barrier.sync 4, 256;" ::: "memory");
            unsigned int valid_rows = actual_q * 8 - row_begin;
            unsigned int q_heads = num_kv_heads * head_group_size;
            #pragma unroll 4
            for (unsigned int i = 0; i < 16; i++) {
                unsigned int r = warp % 4 * 32 + half * 16 + i;
                if (r < valid_rows) {
                    unsigned int ht = row_begin + r;
                    unsigned int orow = (request_offset + ht / 8) * q_heads + head_1 * 8 + ht % 8;
                    uint32_t _sstage_reg_0[4];
                    __int128_t _smem_raw_39;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_39) : "r"(sstage_addr + (r * 132 + lane * 4) * 4));
                    _sstage_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[0];
                    _sstage_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[1];
                    _sstage_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[2];
                    _sstage_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[3];
                    reinterpret_cast<int4*>(output + (orow * 512 + output_col + lane * 8))[0] = reinterpret_cast<int4*>(_sstage_reg_0)[0];
                }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // convert_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int actual_q_1 = q_seq_len;
            unsigned int request_offset_1 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_1 = q_cu_seq_lens[blockIdx.z];
                actual_q_1 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_1;
            }
            unsigned int blocks_per_head_2 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_2 = (unsigned int)blockIdx.y / blocks_per_head_2;
            unsigned int row_begin_1 = (unsigned int)blockIdx.y % blocks_per_head_2 * 128;
            unsigned int output_col_1 = blockIdx.x * 256;
            int _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_1 = (unsigned int)_vec_load_0[0];
            unsigned int num_tiles_1 = (length_1 + 63) / 64;
            int warp_id_in_role = (warp - 8);
            unsigned int tid_1 = (unsigned int)(warp_id_in_role * 32) + lane;
            unsigned int zero = 0;
            unsigned int conv_stage = 0;
            unsigned int _phase_landed = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = 0; tile_1 < num_tiles_1; tile_1++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ready_addr + (conv_stage) * 8);
                    }
                    conv_stage += 1;
                    if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
                }
                if (tile_1 > 0) {
                    #pragma unroll
                    for (int v = 0; v < 2; v++) {
                        mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                        unsigned int valid_1 = ((length_1 - (tile_1 - 1) * 64) < ((unsigned int)64) ? (length_1 - (tile_1 - 1) * 64) : ((unsigned int)64));
                        if (valid_1 < 64) {
                            unsigned int f16_base = conv_stage * 16384;
                            for (unsigned int g = valid_1 * 8 + tid_1; g < 512; g += 128) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base + g * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base + 8192 + g * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(ready_addr + (conv_stage) * 8);
                        }
                        conv_stage += 1;
                        if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                unsigned int valid_2 = ((length_1 - (num_tiles_1 - 1) * 64) < ((unsigned int)64) ? (length_1 - (num_tiles_1 - 1) * 64) : ((unsigned int)64));
                if (valid_2 < 64) {
                    unsigned int f16_base_1 = conv_stage * 16384;
                    for (unsigned int g_1 = valid_2 * 8 + tid_1; g_1 < 512; g_1 += 128) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base_1 + g_1 * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base_1 + 8192 + g_1 * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(ready_addr + (conv_stage) * 8);
                }
                conv_stage += 1;
                if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // mma_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("barrier.sync 2, 32;" ::: "memory");
            unsigned int actual_q_2 = q_seq_len;
            unsigned int request_offset_2 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_2 = q_cu_seq_lens[blockIdx.z];
                actual_q_2 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_2;
            }
            unsigned int blocks_per_head_3 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_3 = (unsigned int)blockIdx.y / blocks_per_head_3;
            unsigned int row_begin_2 = (unsigned int)blockIdx.y % blocks_per_head_3 * 128;
            unsigned int output_col_2 = blockIdx.x * 256;
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_2 = (unsigned int)_vec_load_1[0];
            unsigned int num_tiles_2 = (length_2 + 63) / 64;
            unsigned int mma_stage = 0;
            unsigned int qk_stage = 0;
            unsigned int pv_stage = 0;
            unsigned int _phase_s_empty = 1;
            unsigned int _phase_q_ready_0 = 0;
            unsigned int _phase_ready = 0;
            unsigned int _phase_q_ready_1 = 0;
            unsigned int _phase_q_ready_2 = 0;
            unsigned int _phase_q_ready_3 = 0;
            unsigned int _phase_q_ready_4 = 0;
            unsigned int _phase_q_ready_5 = 0;
            unsigned int _phase_q_ready_6 = 0;
            unsigned int _phase_q_ready_7 = 0;
            unsigned int _phase_p_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = 0; tile_2 < num_tiles_2; tile_2++) {
                mbarrier_wait_hint(s_empty_addr + (qk_stage) * 8, _phase_s_empty, 10000000);
                if (tile_2 == 0) {
                    mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
                    _phase_q_ready_0 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_0 = (_mma_base_lo_0) + (0) * 8192;
                    int _smem_desc_lo_1 = (_mma_base_lo_1) + (mma_stage) * 1024;
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_0);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135266320, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135266320, 1);
                        }
                    }
                    mbarrier_wait_hint(q_ready_addr + 8, _phase_q_ready_1, 10000000);
                    _phase_q_ready_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_3 = elect_sync();
                    int _smem_desc_lo_4 = (_mma_base_lo_2) + (0) * 8192;
                    int _smem_desc_lo_5 = (_mma_base_lo_3) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_s0), "r"(1), "r"(_elected_3) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_3);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 16, _phase_q_ready_2, 10000000);
                    _phase_q_ready_2 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_6 = (_mma_base_lo_4) + (0) * 8192;
                    int _smem_desc_lo_7 = (_mma_base_lo_1) + (mma_stage) * 1024;
                    {
                        uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_6);
                        uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 135266320, 1);
                        }
                    }
                    mbarrier_wait_hint(q_ready_addr + 24, _phase_q_ready_3, 10000000);
                    _phase_q_ready_3 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_9 = elect_sync();
                    int _smem_desc_lo_10 = (_mma_base_lo_5) + (0) * 8192;
                    int _smem_desc_lo_11 = (_mma_base_lo_3) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_s0), "r"(1), "r"(_elected_9) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_9);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 32, _phase_q_ready_4, 10000000);
                    _phase_q_ready_4 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_12 = (_mma_base_lo_6) + (0) * 8192;
                    int _smem_desc_lo_13 = (_mma_base_lo_1) + (mma_stage) * 1024;
                    {
                        uint64_t _mma_ss_a_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_12);
                        uint64_t _mma_ss_b_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_13);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 135266320, 1);
                        }
                    }
                    mbarrier_wait_hint(q_ready_addr + 40, _phase_q_ready_5, 10000000);
                    _phase_q_ready_5 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_15 = elect_sync();
                    int _smem_desc_lo_16 = (_mma_base_lo_7) + (0) * 8192;
                    int _smem_desc_lo_17 = (_mma_base_lo_3) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"(tmem_s0), "r"(1), "r"(_elected_15) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_15);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 48, _phase_q_ready_6, 10000000);
                    _phase_q_ready_6 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_18 = (_mma_base_lo_8) + (0) * 8192;
                    int _smem_desc_lo_19 = (_mma_base_lo_1) + (mma_stage) * 1024;
                    {
                        uint64_t _mma_ss_a_desc_20 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_18);
                        uint64_t _mma_ss_b_desc_20 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_smem_desc_lo_19);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_20, _mma_ss_b_desc_20, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_20, _mma_ss_b_desc_20, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_20, _mma_ss_b_desc_20, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_s0, _mma_ss_a_desc_20, _mma_ss_b_desc_20, 135266320, 1);
                        }
                    }
                    mbarrier_wait_hint(q_ready_addr + 56, _phase_q_ready_7, 10000000);
                    _phase_q_ready_7 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_21 = elect_sync();
                    int _smem_desc_lo_22 = (_mma_base_lo_9) + (0) * 8192;
                    int _smem_desc_lo_23 = (_mma_base_lo_3) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_22), "r"(_smem_desc_lo_23), "r"(tmem_s0), "r"(1), "r"(_elected_21) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_21);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                } else if (qk_stage == 0) {
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_24 = elect_sync();
                    int _smem_desc_lo_25 = (_mma_base_lo_10) + (0) * 8192;
                    int _smem_desc_lo_26 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_25), "r"(_smem_desc_lo_26), "r"(tmem_s0), "r"(0), "r"(_elected_24) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_24);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_27 = elect_sync();
                    int _smem_desc_lo_28 = (_mma_base_lo_12) + (0) * 8192;
                    int _smem_desc_lo_29 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_28), "r"(_smem_desc_lo_29), "r"(tmem_s0), "r"(1), "r"(_elected_27) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_27);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_30 = elect_sync();
                    int _smem_desc_lo_31 = (_mma_base_lo_13) + (0) * 8192;
                    int _smem_desc_lo_32 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_31), "r"(_smem_desc_lo_32), "r"(tmem_s0), "r"(1), "r"(_elected_30) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_30);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_33 = elect_sync();
                    int _smem_desc_lo_34 = (_mma_base_lo_14) + (0) * 8192;
                    int _smem_desc_lo_35 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_34), "r"(_smem_desc_lo_35), "r"(tmem_s0), "r"(1), "r"(_elected_33) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_33);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                } else {
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_36 = elect_sync();
                    int _smem_desc_lo_37 = (_mma_base_lo_10) + (0) * 8192;
                    int _smem_desc_lo_38 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_37), "r"(_smem_desc_lo_38), "r"(tmem_s1), "r"(0), "r"(_elected_36) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_36);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_39 = elect_sync();
                    int _smem_desc_lo_40 = (_mma_base_lo_12) + (0) * 8192;
                    int _smem_desc_lo_41 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_40), "r"(_smem_desc_lo_41), "r"(tmem_s1), "r"(1), "r"(_elected_39) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_39);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_42 = elect_sync();
                    int _smem_desc_lo_43 = (_mma_base_lo_13) + (0) * 8192;
                    int _smem_desc_lo_44 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_43), "r"(_smem_desc_lo_44), "r"(tmem_s1), "r"(1), "r"(_elected_42) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_42);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_45 = elect_sync();
                    int _smem_desc_lo_46 = (_mma_base_lo_14) + (0) * 8192;
                    int _smem_desc_lo_47 = (_mma_base_lo_11) + (mma_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_smem_desc_lo_46), "r"(_smem_desc_lo_47), "r"(tmem_s1), "r"(1), "r"(_elected_45) : "memory");
                    tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_45);
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                }
                elect_commit(s_full_addr + (qk_stage) * 8);
                qk_stage += 1;
                if (qk_stage == 2) { qk_stage = 0; _phase_s_empty ^= 1; }
                if (tile_2 > 0) {
                    mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
                    const bool _selected_wait_cond_48 = pv_stage == 0;
                    mbarrier_wait((_selected_wait_cond_48 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_48 ? (_phase_ready) : (_phase_ready)));
                    if (_selected_wait_cond_48) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_49 = elect_sync();
                        int _smem_desc_lo_50 = (_mma_base_lo_15) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_50), "r"(tmem_p0), "r"(((tile_2 - 1 == 0) ? 0 : 1)), "r"(_elected_49) : "memory");
                        int _smem_desc_lo_51 = (_mma_base_lo_16) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_51), "r"(tmem_p0 + 8), "r"(1), "r"(_elected_49) : "memory");
                        int _smem_desc_lo_52 = (_mma_base_lo_17) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_52), "r"(tmem_p0 + 16), "r"(1), "r"(_elected_49) : "memory");
                        int _smem_desc_lo_53 = (_mma_base_lo_18) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_53), "r"(tmem_p0 + 24), "r"(1), "r"(_elected_49) : "memory");
                        tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_49);
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_54 = elect_sync();
                        int _smem_desc_lo_55 = (_mma_base_lo_15) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_55), "r"(tmem_p0), "r"(((tile_2 - 1 == 0) ? 0 : 1)), "r"(_elected_54) : "memory");
                        int _smem_desc_lo_56 = (_mma_base_lo_16) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_56), "r"(tmem_p0 + 8), "r"(1), "r"(_elected_54) : "memory");
                        int _smem_desc_lo_57 = (_mma_base_lo_17) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_57), "r"(tmem_p0 + 16), "r"(1), "r"(_elected_54) : "memory");
                        int _smem_desc_lo_58 = (_mma_base_lo_18) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_58), "r"(tmem_p0 + 24), "r"(1), "r"(_elected_54) : "memory");
                        tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_54);
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    } else {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_59 = elect_sync();
                        int _smem_desc_lo_60 = (_mma_base_lo_15) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_60), "r"(tmem_p1), "r"(((tile_2 - 1 == 0) ? 0 : 1)), "r"(_elected_59) : "memory");
                        int _smem_desc_lo_61 = (_mma_base_lo_16) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_61), "r"(tmem_p1 + 8), "r"(1), "r"(_elected_59) : "memory");
                        int _smem_desc_lo_62 = (_mma_base_lo_17) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_62), "r"(tmem_p1 + 16), "r"(1), "r"(_elected_59) : "memory");
                        int _smem_desc_lo_63 = (_mma_base_lo_18) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_63), "r"(tmem_p1 + 24), "r"(1), "r"(_elected_59) : "memory");
                        tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_59);
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_64 = elect_sync();
                        int _smem_desc_lo_65 = (_mma_base_lo_15) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_65), "r"(tmem_p1), "r"(((tile_2 - 1 == 0) ? 0 : 1)), "r"(_elected_64) : "memory");
                        int _smem_desc_lo_66 = (_mma_base_lo_16) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_66), "r"(tmem_p1 + 8), "r"(1), "r"(_elected_64) : "memory");
                        int _smem_desc_lo_67 = (_mma_base_lo_17) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_67), "r"(tmem_p1 + 16), "r"(1), "r"(_elected_64) : "memory");
                        int _smem_desc_lo_68 = (_mma_base_lo_18) + (mma_stage) * 1024;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_68), "r"(tmem_p1 + 24), "r"(1), "r"(_elected_64) : "memory");
                        tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_64);
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    }
                    elect_commit(p_empty_addr + (pv_stage) * 8);
                    pv_stage += 1;
                    if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
                }
            }
            mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
            const bool _selected_wait_cond_69 = pv_stage == 0;
            mbarrier_wait((_selected_wait_cond_69 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_69 ? (_phase_ready) : (_phase_ready)));
            if (_selected_wait_cond_69) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_70 = elect_sync();
                int _smem_desc_lo_71 = (_mma_base_lo_15) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_71), "r"(tmem_p0), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)), "r"(_elected_70) : "memory");
                int _smem_desc_lo_72 = (_mma_base_lo_16) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_72), "r"(tmem_p0 + 8), "r"(1), "r"(_elected_70) : "memory");
                int _smem_desc_lo_73 = (_mma_base_lo_17) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_73), "r"(tmem_p0 + 16), "r"(1), "r"(_elected_70) : "memory");
                int _smem_desc_lo_74 = (_mma_base_lo_18) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_74), "r"(tmem_p0 + 24), "r"(1), "r"(_elected_70) : "memory");
                tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_70);
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_75 = elect_sync();
                int _smem_desc_lo_76 = (_mma_base_lo_15) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_76), "r"(tmem_p0), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)), "r"(_elected_75) : "memory");
                int _smem_desc_lo_77 = (_mma_base_lo_16) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_77), "r"(tmem_p0 + 8), "r"(1), "r"(_elected_75) : "memory");
                int _smem_desc_lo_78 = (_mma_base_lo_17) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_78), "r"(tmem_p0 + 16), "r"(1), "r"(_elected_75) : "memory");
                int _smem_desc_lo_79 = (_mma_base_lo_18) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_79), "r"(tmem_p0 + 24), "r"(1), "r"(_elected_75) : "memory");
                tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_75);
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
            } else {
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_80 = elect_sync();
                int _smem_desc_lo_81 = (_mma_base_lo_15) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_81), "r"(tmem_p1), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)), "r"(_elected_80) : "memory");
                int _smem_desc_lo_82 = (_mma_base_lo_16) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_82), "r"(tmem_p1 + 8), "r"(1), "r"(_elected_80) : "memory");
                int _smem_desc_lo_83 = (_mma_base_lo_17) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_83), "r"(tmem_p1 + 16), "r"(1), "r"(_elected_80) : "memory");
                int _smem_desc_lo_84 = (_mma_base_lo_18) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_smem_desc_lo_84), "r"(tmem_p1 + 24), "r"(1), "r"(_elected_80) : "memory");
                tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_80);
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t _elected_85 = elect_sync();
                int _smem_desc_lo_86 = (_mma_base_lo_15) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_86), "r"(tmem_p1), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)), "r"(_elected_85) : "memory");
                int _smem_desc_lo_87 = (_mma_base_lo_16) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_87), "r"(tmem_p1 + 8), "r"(1), "r"(_elected_85) : "memory");
                int _smem_desc_lo_88 = (_mma_base_lo_17) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_88), "r"(tmem_p1 + 16), "r"(1), "r"(_elected_85) : "memory");
                int _smem_desc_lo_89 = (_mma_base_lo_18) + (mma_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_smem_desc_lo_89), "r"(tmem_p1 + 24), "r"(1), "r"(_elected_85) : "memory");
                tcgen05_commit_cg1_multicast_leader(free_addr + (mma_stage) * 8, (uint16_t)(15), _elected_85);
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
            }
            elect_commit(p_empty_addr + (pv_stage) * 8);
            pv_stage += 1;
            if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
            elect_commit(o_ready_addr);
            unsigned int _phase_tmem_retire_0 = 0;
            mbarrier_wait_hint(tmem_retire_addr, _phase_tmem_retire_0, 10000000);
            _phase_tmem_retire_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // load_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int info[4];
            unsigned int zero_1 = 0;
            info[1] = zero_1;
            info[2] = zero_1;
            {
                info[0] = *reinterpret_cast<const unsigned int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            if ((unsigned long long)q_cu_seq_lens != 0) {
                {
                    info[1] = *reinterpret_cast<const unsigned int*>(q_cu_seq_lens + blockIdx.z);
                }
                {
                    info[2] = *reinterpret_cast<const unsigned int*>(q_cu_seq_lens + (blockIdx.z + 1));
                }
            }
            unsigned int rank = (unsigned int)cta_rank;
            unsigned int blocks_per_head_4 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_4 = (unsigned int)blockIdx.y / blocks_per_head_4;
            unsigned int load_stage = 0;
            unsigned int tile0 = 0;
            unsigned int _phase_free = 1;
            #pragma unroll
            for (int c_1 = 0; c_1 < 4; c_1++) {
                unsigned int kv_index = (unsigned int)(blockIdx.z * 2) * num_kv_heads + head_4;
                unsigned int token = tile0 * 64;
                mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        if (rank == (unsigned int)(2 * c_1 + h & 3)) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h * 8192)), "l"((&KV)), "r"(c_1 * 128 + h * 64), "r"(token), "r"(kv_index),
                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(15)) : "memory");
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
            }
            unsigned int actual_q_3 = q_seq_len;
            unsigned int request_offset_3 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_3 = info[1];
                actual_q_3 = info[2] - info[1];
            }
            unsigned int blocks_per_head_0 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int row_begin_3 = (unsigned int)blockIdx.y % blocks_per_head_0 * 128;
            unsigned int output_col_3 = blockIdx.x * 256;
            unsigned int length_3 = info[0];
            unsigned int num_tiles_3 = (length_3 + 63) / 64;
            unsigned int token0 = request_offset_3 + row_begin_3 / 8;
            #pragma unroll
            for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                if (elect_sync()) {
                    #pragma unroll
                    for (int slab_1 = 2 * pair_1; slab_1 < 2 * pair_1 + 2; slab_1++) {
                        mbarrier_arrive_expect_tx(q_ready_addr + (slab_1) * 8, 16384);
                    }
                    if ((rank & 1) == (unsigned int)(pair_1 & 1)) {
                        #pragma unroll
                        for (int slab_2 = 2 * pair_1; slab_2 < 2 * pair_1 + 2; slab_2++) {
                            if (rank >> 1 == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab_2 * 16384)), "l"((&Q)), "r"(slab_2 * 64), "r"(head_4 * 8), "r"(token0),
                                       "r"(q_ready_addr + (slab_2) * 8), "h"((uint16_t)(3)) : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab_2 * 16384)), "l"((&Q)), "r"(slab_2 * 64), "r"(head_4 * 8), "r"(token0),
                                       "r"(q_ready_addr + (slab_2) * 8), "h"((uint16_t)(12)) : "memory");
                            }
                        }
                    }
                }
            }
            unsigned int gate_stage = 3;
            unsigned int gate_phase = 0;
            uint32_t _mbar_token_0 = mbarrier_try_wait(landed_addr + (gate_stage) * 8, gate_phase);
            mbarrier_wait_token(landed_addr + (gate_stage) * 8, gate_phase, _mbar_token_0);
            #pragma unroll 1
            for (unsigned int tile_3 = 1; tile_3 < num_tiles_3; tile_3++) {
                #pragma unroll
                for (int c_2 = 0; c_2 < 4; c_2++) {
                    unsigned int kv_index_1 = (unsigned int)(blockIdx.z * 2) * num_kv_heads + head_4;
                    unsigned int token_1 = tile_3 * 64;
                    mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 2; h_1++) {
                            if (rank == (unsigned int)(2 * c_2 + h_1 & 3)) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_1 * 8192)), "l"((&KV)), "r"(c_2 * 128 + h_1 * 64), "r"(token_1), "r"(kv_index_1),
                                       "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(15)) : "memory");
                            }
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
                }
                if (tile_3 > 0) {
                    #pragma unroll
                    for (int v_2 = 0; v_2 < 2; v_2++) {
                        unsigned int kv_index_2 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_4;
                        unsigned int token_2 = (tile_3 - 1) * 64;
                        mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                            #pragma unroll
                            for (int h_2 = 0; h_2 < 2; h_2++) {
                                if (rank >> 1 == (unsigned int)(v_2 + h_2 & 1)) {
                                    if ((rank & 1) == 0) {
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_2 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_2 * 128) + (unsigned int)(h_2 * 64)), "r"(token_2), "r"(kv_index_2),
                                               "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                                    } else {
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_2 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_2 * 128) + (unsigned int)(h_2 * 64)), "r"(token_2), "r"(kv_index_2),
                                               "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                                    }
                                }
                            }
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                unsigned int kv_index_3 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_4;
                unsigned int token_3 = (num_tiles_3 - 1) * 64;
                mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 2; h_3++) {
                        if (rank >> 1 == (unsigned int)(v_3 + h_3 & 1)) {
                            if ((rank & 1) == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_3 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_3 * 128) + (unsigned int)(h_3 * 64)), "r"(token_3), "r"(kv_index_3),
                                       "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_3 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_3 * 128) + (unsigned int)(h_3 * 64)), "r"(token_3), "r"(kv_index_3),
                                       "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                            }
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: idle ----
    if (warp >= 14 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // idle_main
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            __syncwarp();
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
