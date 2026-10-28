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
#define NUM_RING_STAGES 12
#define NUM_Q_PIPE_STAGES 4
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SQ_OFF 1024
#define SMEM_SQ_STAGE_BYTES 131072
#define SMEM_SQ_STRIDE 131072
#define SMEM_SK_OFF 132096
#define SMEM_SK_STAGE_BYTES 8192
#define SMEM_SK_STRIDE 8192
#define SMEM_SV_OFF 132096
#define SMEM_SV_STAGE_BYTES 8192
#define SMEM_SV_STRIDE 8192
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_sm110_xqa_cd771bf9c866203bae1a(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KVV, unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const unsigned int* __restrict__ mask, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale)
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
    #define ready_addr (mbar_base + 96)
    #define free_addr (mbar_base + 192)
    #define q_ready_addr (mbar_base + 288)
    #define s_full_addr (mbar_base + 320)
    #define s_empty_addr (mbar_base + 336)
    #define p_full_addr (mbar_base + 352)
    #define p_empty_addr (mbar_base + 368)
    #define tmem_retire_addr (mbar_base + 384)
    #define o_ready_addr (mbar_base + 392)
    #define tmem_pair_addr (mbar_base + 400)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

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
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KVV))) : "memory");
    int _mma_base_lo_0 = ((sq_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((sk_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_2 = ((sq_addr + 32768) >> 4) & 0x3FFF;
    int _mma_base_lo_3 = ((sq_addr + 65536) >> 4) & 0x3FFF;
    int _mma_base_lo_4 = ((sq_addr + 98304) >> 4) & 0x3FFF;
    int _mma_base_lo_5 = (((sv_addr) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_6 = (((sv_addr + 2048) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_7 = (((sv_addr + 4096) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_8 = (((sv_addr + 6144) >> 4) & 0x3FFF) | 0x2000000;

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 51 barriers)
    // Mbarriers at smem_raw[0..408)


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
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            mbarrier_init(smem + 160, 8);
            mbarrier_init(smem + 168, 8);
            mbarrier_init(smem + 176, 8);
            mbarrier_init(smem + 184, 8);
            mbarrier_init(smem + 192, 2);
            mbarrier_init(smem + 200, 2);
            mbarrier_init(smem + 208, 2);
            mbarrier_init(smem + 216, 2);
            mbarrier_init(smem + 224, 2);
            mbarrier_init(smem + 232, 2);
            mbarrier_init(smem + 240, 2);
            mbarrier_init(smem + 248, 2);
            mbarrier_init(smem + 256, 2);
            mbarrier_init(smem + 264, 2);
            mbarrier_init(smem + 272, 2);
            mbarrier_init(smem + 280, 2);
            // --- pipeline 'q_pipe' ---
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // --- pipeline 's_pipe' ---
            mbarrier_init(smem + 320, 2);
            mbarrier_init(smem + 328, 2);
            mbarrier_init(smem + 336, 512);
            mbarrier_init(smem + 344, 512);
            // --- pipeline 'p_pipe' ---
            mbarrier_init(smem + 352, 512);
            mbarrier_init(smem + 360, 512);
            mbarrier_init(smem + 368, 2);
            mbarrier_init(smem + 376, 2);
            mbarrier_init(smem + 384, 512);
            mbarrier_init(smem + 392, 2);
            mbarrier_init(smem + 400, 32);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncthreads();
    asm volatile("barrier.sync 1, 512;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 408);
    if (warp == 12) {
        int _tmem_hold = smem + 408;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
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
            unsigned int actual_q = q_seq_len;
            unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset = q_cu_seq_lens[blockIdx.z];
                actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
            }
            unsigned int rank = (unsigned int)cta_rank;
            unsigned int pair_rank = rank & 1;
            unsigned int pair_id = rank >> 1;
            unsigned int pairs_per_head = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head = (unsigned int)blockIdx.y / pairs_per_head;
            unsigned int row_begin = ((unsigned int)blockIdx.y % pairs_per_head * 2 + pair_rank) * 128;
            unsigned int output_col = pair_id * 256;
            int _vec_load_3[1];
            {
                _vec_load_3[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length = (unsigned int)_vec_load_3[0];
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
            unsigned int _vec_load_4[1];
            {
                _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(mask + mask_row);
            }
            unsigned int mw0 = _vec_load_4[0];
            unsigned int mw1 = 0;
            unsigned int mw2 = 0;
            unsigned int mw3 = 0;
            unsigned int zero32 = 0;
            if (mask_stride > 1) {
                unsigned int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 1));
                }
                mw1 = _vec_load_5[0];
            }
            if (mask_stride > 2) {
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 2));
                }
                mw2 = _vec_load_6[0];
            }
            if (mask_stride > 3) {
                unsigned int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 3));
                }
                mw3 = _vec_load_7[0];
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
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((s_empty_addr + (sm_s_stage) * 8) & 0xFEFFFFFF) : "memory");
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
                            unsigned int _vec_load_8[1];
                            {
                                _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0));
                            }
                            unsigned int w0 = _vec_load_8[0];
                            unsigned int w1 = 0;
                            unsigned int w2 = 0;
                            if (mask_stride > word0 + 1) {
                                unsigned int _vec_load_9[1];
                                {
                                    _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 1));
                                }
                                w1 = _vec_load_9[0];
                            }
                            if (mask_stride > word0 + 2) {
                                unsigned int _vec_load_10[1];
                                {
                                    _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 2));
                                }
                                w2 = _vec_load_10[0];
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
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr + (sm_p_stage) * 8) & 0xFEFFFFFF) : "memory");
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
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((tmem_retire_addr) & 0xFEFFFFFF) : "memory");
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.sync 4, 256;" ::: "memory");
            unsigned int valid_rows = actual_q * 8 - row_begin;
            unsigned int q_heads = num_kv_heads * head_group_size;
            #pragma unroll 4
            for (unsigned int i = 0; i < 16; i++) {
                unsigned int r = warp % 4 * 32 + half * 16 + i;
                if (r < valid_rows) {
                    unsigned int ht = row_begin + r;
                    unsigned int orow = (request_offset + ht / 8) * q_heads + head * 8 + ht % 8;
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
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: convert ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        { // convert_main
            unsigned int actual_q_1 = q_seq_len;
            unsigned int request_offset_1 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_1 = q_cu_seq_lens[blockIdx.z];
                actual_q_1 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_1;
            }
            unsigned int rank_1 = (unsigned int)cta_rank;
            unsigned int pair_rank_1 = rank_1 & 1;
            unsigned int pair_id_1 = rank_1 >> 1;
            unsigned int pairs_per_head_1 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_1 = (unsigned int)blockIdx.y / pairs_per_head_1;
            unsigned int row_begin_1 = ((unsigned int)blockIdx.y % pairs_per_head_1 * 2 + pair_rank_1) * 128;
            unsigned int output_col_1 = pair_id_1 * 256;
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_1 = (unsigned int)_vec_load_1[0];
            unsigned int num_tiles_1 = (length_1 + 63) / 64;
            int warp_id_in_role = (warp - 8);
            unsigned int tid_1 = (unsigned int)(warp_id_in_role * 32) + lane;
            unsigned int zero = 0;
            unsigned int conv_stage = 0;
            unsigned int rbuf[8];
            unsigned int krow = tid_1 / 8;
            unsigned int kc16 = tid_1 % 8;
            unsigned int kswz = krow % 8 * 16;
            unsigned int kline0 = kc16 / 4 * 32 + krow;
            unsigned int kdst_a = kline0 * 128 + (kc16 % 4 * 32 ^ kswz);
            unsigned int kdst_b = kline0 * 128 + (kc16 % 4 * 32 + 16 ^ kswz);
            unsigned int kraw = krow * 128 + kc16 * 16;
            unsigned int vrow = tid_1 / 4;
            unsigned int vc4 = tid_1 % 4;
            unsigned int vswz = vrow % 8 * 16;
            unsigned int vdst_a = vrow * 128 + (vc4 * 32 ^ vswz);
            unsigned int vdst_b = vrow * 128 + (vc4 * 32 + 16 ^ vswz);
            unsigned int vraw_half = vrow * 64 + vc4 * 16;
            unsigned int vraw_full = vrow * 128 + pair_rank_1 * 64 + vc4 * 16;
            unsigned int vraw = vraw_full;
            unsigned int _phase_landed = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = 0; tile_1 < num_tiles_1; tile_1++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                    if (tile_1 == 0) {
                        mbarrier_wait_hint(q_ready_addr + (c) * 8, 0, 10000000);
                    }
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[0])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 3]))
                        : "r"(sk_addr + conv_stage * 8192 + 4096 + kraw));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[4])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 3]))
                        : "r"(sk_addr + conv_stage * 8192 + 4096 + kraw + 2048));
                    __syncwarp();
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(rbuf[0])));
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(rbuf[0] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(rbuf[1])));
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(rbuf[1] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(rbuf[2])));
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(rbuf[2] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(rbuf[3])));
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(rbuf[3] >> 16)));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + kdst_a)), "r"(_e4m3x2_to_f16x2_0), "r"(_e4m3x2_to_f16x2_1), "r"(_e4m3x2_to_f16x2_2), "r"(_e4m3x2_to_f16x2_3) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + kdst_b)), "r"(_e4m3x2_to_f16x2_4), "r"(_e4m3x2_to_f16x2_5), "r"(_e4m3x2_to_f16x2_6), "r"(_e4m3x2_to_f16x2_7) : "memory");
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(rbuf[4])));
                    uint32_t _e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(rbuf[4] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(rbuf[5])));
                    uint32_t _e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(rbuf[5] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_12;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)(rbuf[6])));
                    uint32_t _e4m3x2_to_f16x2_13;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)(rbuf[6] >> 16)));
                    uint32_t _e4m3x2_to_f16x2_14;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)(rbuf[7])));
                    uint32_t _e4m3x2_to_f16x2_15;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)(rbuf[7] >> 16)));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 2048 + kdst_a)), "r"(_e4m3x2_to_f16x2_8), "r"(_e4m3x2_to_f16x2_9), "r"(_e4m3x2_to_f16x2_10), "r"(_e4m3x2_to_f16x2_11) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 2048 + kdst_b)), "r"(_e4m3x2_to_f16x2_12), "r"(_e4m3x2_to_f16x2_13), "r"(_e4m3x2_to_f16x2_14), "r"(_e4m3x2_to_f16x2_15) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((ready_addr + (conv_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    conv_stage += 1;
                    if (conv_stage == 12) { conv_stage = 0; _phase_landed ^= 1; }
                }
                if (tile_1 > 0) {
                    #pragma unroll
                    for (int v = 0; v < 2; v++) {
                        mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[0])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 3]))
                            : "r"(sk_addr + conv_stage * 8192 + vraw));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[4])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 3]))
                            : "r"(sk_addr + conv_stage * 8192 + vraw + 4096));
                        asm volatile("barrier.sync 5, 128;" ::: "memory");
                        uint32_t _e4m3x2_to_f16x2_16;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)(rbuf[0])));
                        uint32_t _e4m3x2_to_f16x2_17;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)(rbuf[0] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_18;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_18) : "h"((uint16_t)(rbuf[1])));
                        uint32_t _e4m3x2_to_f16x2_19;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_19) : "h"((uint16_t)(rbuf[1] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_20;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_20) : "h"((uint16_t)(rbuf[2])));
                        uint32_t _e4m3x2_to_f16x2_21;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_21) : "h"((uint16_t)(rbuf[2] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_22;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_22) : "h"((uint16_t)(rbuf[3])));
                        uint32_t _e4m3x2_to_f16x2_23;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_23) : "h"((uint16_t)(rbuf[3] >> 16)));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + vdst_a)), "r"(_e4m3x2_to_f16x2_16), "r"(_e4m3x2_to_f16x2_17), "r"(_e4m3x2_to_f16x2_18), "r"(_e4m3x2_to_f16x2_19) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + vdst_b)), "r"(_e4m3x2_to_f16x2_20), "r"(_e4m3x2_to_f16x2_21), "r"(_e4m3x2_to_f16x2_22), "r"(_e4m3x2_to_f16x2_23) : "memory");
                        uint32_t _e4m3x2_to_f16x2_24;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_24) : "h"((uint16_t)(rbuf[4])));
                        uint32_t _e4m3x2_to_f16x2_25;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_25) : "h"((uint16_t)(rbuf[4] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_26;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_26) : "h"((uint16_t)(rbuf[5])));
                        uint32_t _e4m3x2_to_f16x2_27;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_27) : "h"((uint16_t)(rbuf[5] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_28;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_28) : "h"((uint16_t)(rbuf[6])));
                        uint32_t _e4m3x2_to_f16x2_29;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_29) : "h"((uint16_t)(rbuf[6] >> 16)));
                        uint32_t _e4m3x2_to_f16x2_30;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_30) : "h"((uint16_t)(rbuf[7])));
                        uint32_t _e4m3x2_to_f16x2_31;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_31) : "h"((uint16_t)(rbuf[7] >> 16)));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 4096 + vdst_a)), "r"(_e4m3x2_to_f16x2_24), "r"(_e4m3x2_to_f16x2_25), "r"(_e4m3x2_to_f16x2_26), "r"(_e4m3x2_to_f16x2_27) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 4096 + vdst_b)), "r"(_e4m3x2_to_f16x2_28), "r"(_e4m3x2_to_f16x2_29), "r"(_e4m3x2_to_f16x2_30), "r"(_e4m3x2_to_f16x2_31) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((ready_addr + (conv_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        conv_stage += 1;
                        if (conv_stage == 12) { conv_stage = 0; _phase_landed ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[0])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(0) + 3]))
                    : "r"(sk_addr + conv_stage * 8192 + vraw));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rbuf[4])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rbuf[(4) + 3]))
                    : "r"(sk_addr + conv_stage * 8192 + vraw + 4096));
                asm volatile("barrier.sync 5, 128;" ::: "memory");
                uint32_t _e4m3x2_to_f16x2_32;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_32) : "h"((uint16_t)(rbuf[0])));
                uint32_t _e4m3x2_to_f16x2_33;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_33) : "h"((uint16_t)(rbuf[0] >> 16)));
                uint32_t _e4m3x2_to_f16x2_34;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_34) : "h"((uint16_t)(rbuf[1])));
                uint32_t _e4m3x2_to_f16x2_35;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_35) : "h"((uint16_t)(rbuf[1] >> 16)));
                uint32_t _e4m3x2_to_f16x2_36;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_36) : "h"((uint16_t)(rbuf[2])));
                uint32_t _e4m3x2_to_f16x2_37;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_37) : "h"((uint16_t)(rbuf[2] >> 16)));
                uint32_t _e4m3x2_to_f16x2_38;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_38) : "h"((uint16_t)(rbuf[3])));
                uint32_t _e4m3x2_to_f16x2_39;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_39) : "h"((uint16_t)(rbuf[3] >> 16)));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + vdst_a)), "r"(_e4m3x2_to_f16x2_32), "r"(_e4m3x2_to_f16x2_33), "r"(_e4m3x2_to_f16x2_34), "r"(_e4m3x2_to_f16x2_35) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + vdst_b)), "r"(_e4m3x2_to_f16x2_36), "r"(_e4m3x2_to_f16x2_37), "r"(_e4m3x2_to_f16x2_38), "r"(_e4m3x2_to_f16x2_39) : "memory");
                uint32_t _e4m3x2_to_f16x2_40;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_40) : "h"((uint16_t)(rbuf[4])));
                uint32_t _e4m3x2_to_f16x2_41;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_41) : "h"((uint16_t)(rbuf[4] >> 16)));
                uint32_t _e4m3x2_to_f16x2_42;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_42) : "h"((uint16_t)(rbuf[5])));
                uint32_t _e4m3x2_to_f16x2_43;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_43) : "h"((uint16_t)(rbuf[5] >> 16)));
                uint32_t _e4m3x2_to_f16x2_44;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_44) : "h"((uint16_t)(rbuf[6])));
                uint32_t _e4m3x2_to_f16x2_45;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_45) : "h"((uint16_t)(rbuf[6] >> 16)));
                uint32_t _e4m3x2_to_f16x2_46;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_46) : "h"((uint16_t)(rbuf[7])));
                uint32_t _e4m3x2_to_f16x2_47;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_47) : "h"((uint16_t)(rbuf[7] >> 16)));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 4096 + vdst_a)), "r"(_e4m3x2_to_f16x2_40), "r"(_e4m3x2_to_f16x2_41), "r"(_e4m3x2_to_f16x2_42), "r"(_e4m3x2_to_f16x2_43) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (conv_stage * 8192 + 4096 + vdst_b)), "r"(_e4m3x2_to_f16x2_44), "r"(_e4m3x2_to_f16x2_45), "r"(_e4m3x2_to_f16x2_46), "r"(_e4m3x2_to_f16x2_47) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((ready_addr + (conv_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                conv_stage += 1;
                if (conv_stage == 12) { conv_stage = 0; _phase_landed ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // mma_main
            asm volatile("barrier.sync 2, 32;" ::: "memory");
            unsigned int actual_q_2 = q_seq_len;
            unsigned int request_offset_2 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_2 = q_cu_seq_lens[blockIdx.z];
                actual_q_2 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_2;
            }
            unsigned int rank_2 = (unsigned int)cta_rank;
            unsigned int pair_rank_2 = rank_2 & 1;
            unsigned int pair_id_2 = rank_2 >> 1;
            unsigned int pairs_per_head_2 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_2 = (unsigned int)blockIdx.y / pairs_per_head_2;
            unsigned int row_begin_2 = ((unsigned int)blockIdx.y % pairs_per_head_2 * 2 + pair_rank_2) * 128;
            unsigned int output_col_2 = pair_id_2 * 256;
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_2 = (unsigned int)_vec_load_2[0];
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
            unsigned int _phase_p_full = 0;
            unsigned int _phase_tmem_retire_0 = 0;
            if ((cta_rank & 1) == 0) {
                #pragma unroll 1
                for (unsigned int tile_2 = 0; tile_2 < num_tiles_2; tile_2++) {
                    mbarrier_wait_hint(s_empty_addr + (qk_stage) * 8, _phase_s_empty, 10000000);
                    if (tile_2 == 0) {
                        mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
                        _phase_q_ready_0 ^= 1;
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_0 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_s0), "r"(0) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(q_ready_addr + 8, _phase_q_ready_1, 10000000);
                        _phase_q_ready_1 ^= 1;
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_1 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(q_ready_addr + 16, _phase_q_ready_2, 10000000);
                        _phase_q_ready_2 ^= 1;
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_2 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(q_ready_addr + 24, _phase_q_ready_3, 10000000);
                        _phase_q_ready_3 ^= 1;
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_3 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_3 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                    } else if (qk_stage == 0) {
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_4 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_s0), "r"(0) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_5 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_5 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_6 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_6 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_7 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_7 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_s0), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                    } else {
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_8 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_8 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_s1), "r"(0) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_9 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_9 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"(tmem_s1), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_10 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_10 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_s1), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_11 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_11 = (_mma_base_lo_1) + (mma_stage) * 512;
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
                    "mov.b32 id, 269484048;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"(tmem_s1), "r"(1) : "memory");
                        elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        mma_stage += 1;
                        if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                    }
                    elect_commit_cg2_multicast(s_full_addr + (qk_stage) * 8, (uint16_t)(15));
                    qk_stage += 1;
                    if (qk_stage == 2) { qk_stage = 0; _phase_s_empty ^= 1; }
                    if (tile_2 > 0) {
                        mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
                        const bool _selected_wait_cond_0 = pv_stage == 0;
                        mbarrier_wait((_selected_wait_cond_0 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_0 ? (_phase_ready) : (_phase_ready)));
                        if (_selected_wait_cond_0) {
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_12 = (_mma_base_lo_5) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p0, _mma_b_lo_12, 0x40004040, 270598160, ((tile_2 - 1 == 0) ? 0 : 1));
                            int _mma_b_lo_13 = (_mma_base_lo_6) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p0 + 8, _mma_b_lo_13, 0x40004040, 270598160, 1);
                            int _mma_b_lo_14 = (_mma_base_lo_7) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p0 + 16, _mma_b_lo_14, 0x40004040, 270598160, 1);
                            int _mma_b_lo_15 = (_mma_base_lo_8) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p0 + 24, _mma_b_lo_15, 0x40004040, 270598160, 1);
                            elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                            mma_stage += 1;
                            if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                            mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_16 = (_mma_base_lo_5) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p0, _mma_b_lo_16, 0x40004040, 270598160, ((tile_2 - 1 == 0) ? 0 : 1));
                            int _mma_b_lo_17 = (_mma_base_lo_6) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p0 + 8, _mma_b_lo_17, 0x40004040, 270598160, 1);
                            int _mma_b_lo_18 = (_mma_base_lo_7) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p0 + 16, _mma_b_lo_18, 0x40004040, 270598160, 1);
                            int _mma_b_lo_19 = (_mma_base_lo_8) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p0 + 24, _mma_b_lo_19, 0x40004040, 270598160, 1);
                            elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                            mma_stage += 1;
                            if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        } else {
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_20 = (_mma_base_lo_5) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p1, _mma_b_lo_20, 0x40004040, 270598160, ((tile_2 - 1 == 0) ? 0 : 1));
                            int _mma_b_lo_21 = (_mma_base_lo_6) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p1 + 8, _mma_b_lo_21, 0x40004040, 270598160, 1);
                            int _mma_b_lo_22 = (_mma_base_lo_7) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p1 + 16, _mma_b_lo_22, 0x40004040, 270598160, 1);
                            int _mma_b_lo_23 = (_mma_base_lo_8) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o0, tmem_p1 + 24, _mma_b_lo_23, 0x40004040, 270598160, 1);
                            elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                            mma_stage += 1;
                            if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                            mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_b_lo_24 = (_mma_base_lo_5) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p1, _mma_b_lo_24, 0x40004040, 270598160, ((tile_2 - 1 == 0) ? 0 : 1));
                            int _mma_b_lo_25 = (_mma_base_lo_6) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p1 + 8, _mma_b_lo_25, 0x40004040, 270598160, 1);
                            int _mma_b_lo_26 = (_mma_base_lo_7) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p1 + 16, _mma_b_lo_26, 0x40004040, 270598160, 1);
                            int _mma_b_lo_27 = (_mma_base_lo_8) + (mma_stage) * 512;
                            mma_ts_step_cg2(tmem_o1, tmem_p1 + 24, _mma_b_lo_27, 0x40004040, 270598160, 1);
                            elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                            mma_stage += 1;
                            if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                        }
                        elect_commit_cg2_multicast(p_empty_addr + (pv_stage) * 8, (uint16_t)(15));
                        pv_stage += 1;
                        if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
                    }
                }
                mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
                const bool _selected_wait_cond_1 = pv_stage == 0;
                mbarrier_wait((_selected_wait_cond_1 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_1 ? (_phase_ready) : (_phase_ready)));
                if (_selected_wait_cond_1) {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_28 = (_mma_base_lo_5) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p0, _mma_b_lo_28, 0x40004040, 270598160, ((num_tiles_2 - 1 == 0) ? 0 : 1));
                    int _mma_b_lo_29 = (_mma_base_lo_6) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p0 + 8, _mma_b_lo_29, 0x40004040, 270598160, 1);
                    int _mma_b_lo_30 = (_mma_base_lo_7) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p0 + 16, _mma_b_lo_30, 0x40004040, 270598160, 1);
                    int _mma_b_lo_31 = (_mma_base_lo_8) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p0 + 24, _mma_b_lo_31, 0x40004040, 270598160, 1);
                    elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    mma_stage += 1;
                    if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_32 = (_mma_base_lo_5) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p0, _mma_b_lo_32, 0x40004040, 270598160, ((num_tiles_2 - 1 == 0) ? 0 : 1));
                    int _mma_b_lo_33 = (_mma_base_lo_6) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p0 + 8, _mma_b_lo_33, 0x40004040, 270598160, 1);
                    int _mma_b_lo_34 = (_mma_base_lo_7) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p0 + 16, _mma_b_lo_34, 0x40004040, 270598160, 1);
                    int _mma_b_lo_35 = (_mma_base_lo_8) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p0 + 24, _mma_b_lo_35, 0x40004040, 270598160, 1);
                    elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    mma_stage += 1;
                    if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                } else {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_36 = (_mma_base_lo_5) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p1, _mma_b_lo_36, 0x40004040, 270598160, ((num_tiles_2 - 1 == 0) ? 0 : 1));
                    int _mma_b_lo_37 = (_mma_base_lo_6) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p1 + 8, _mma_b_lo_37, 0x40004040, 270598160, 1);
                    int _mma_b_lo_38 = (_mma_base_lo_7) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p1 + 16, _mma_b_lo_38, 0x40004040, 270598160, 1);
                    int _mma_b_lo_39 = (_mma_base_lo_8) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o0, tmem_p1 + 24, _mma_b_lo_39, 0x40004040, 270598160, 1);
                    elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    mma_stage += 1;
                    if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_40 = (_mma_base_lo_5) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p1, _mma_b_lo_40, 0x40004040, 270598160, ((num_tiles_2 - 1 == 0) ? 0 : 1));
                    int _mma_b_lo_41 = (_mma_base_lo_6) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p1 + 8, _mma_b_lo_41, 0x40004040, 270598160, 1);
                    int _mma_b_lo_42 = (_mma_base_lo_7) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p1 + 16, _mma_b_lo_42, 0x40004040, 270598160, 1);
                    int _mma_b_lo_43 = (_mma_base_lo_8) + (mma_stage) * 512;
                    mma_ts_step_cg2(tmem_o1, tmem_p1 + 24, _mma_b_lo_43, 0x40004040, 270598160, 1);
                    elect_commit_cg2_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    mma_stage += 1;
                    if (mma_stage == 12) { mma_stage = 0; _phase_ready ^= 1; }
                }
                elect_commit_cg2_multicast(p_empty_addr + (pv_stage) * 8, (uint16_t)(15));
                pv_stage += 1;
                if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
                elect_commit_cg2_multicast(o_ready_addr, (uint16_t)(15));
                mbarrier_wait_hint(tmem_retire_addr, _phase_tmem_retire_0, 10000000);
                _phase_tmem_retire_0 ^= 1;
            }
            if ((cta_rank & 1) != 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_pair_addr), "r"(cta_rank ^ 1) : "memory");
            }
            unsigned int _phase_tmem_pair_0 = 0;
            mbarrier_wait_hint(tmem_pair_addr, _phase_tmem_pair_0, 10000000);
            _phase_tmem_pair_0 ^= 1;
            if ((cta_rank & 1) == 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_pair_addr), "r"(cta_rank ^ 1) : "memory");
            }
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // load_main
            unsigned int actual_q_3 = q_seq_len;
            unsigned int request_offset_3 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_3 = q_cu_seq_lens[blockIdx.z];
                actual_q_3 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_3;
            }
            unsigned int rank_3 = (unsigned int)cta_rank;
            unsigned int pair_rank_3 = rank_3 & 1;
            unsigned int pair_id_3 = rank_3 >> 1;
            unsigned int pairs_per_head_3 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_3 = (unsigned int)blockIdx.y / pairs_per_head_3;
            unsigned int row_begin_3 = ((unsigned int)blockIdx.y % pairs_per_head_3 * 2 + pair_rank_3) * 128;
            unsigned int output_col_3 = pair_id_3 * 256;
            int _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_3 = (unsigned int)_vec_load_0[0];
            unsigned int num_tiles_3 = (length_3 + 63) / 64;
            unsigned int token0 = request_offset_3 + row_begin_3 / 8;
            unsigned int tok_half = pair_rank_3 * 32;
            unsigned int col_half = output_col_3 + pair_rank_3 * 64;
            unsigned int zero_off = 0;
            unsigned int load_stage = 0;
            unsigned int lag_stage_1 = 0;
            unsigned int lag_ph = 0;
            unsigned int tile0 = 0;
            unsigned int _phase_free = 1;
            #pragma unroll
            for (int c_1 = 0; c_1 < 4; c_1++) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_ready_addr + (c_1) * 8, 32768);
                    if (pair_id_3 == (unsigned int)(c_1 & 1)) {
                        #pragma unroll
                        for (int slab = 2 * c_1; slab < 2 * c_1 + 2; slab++) {
                            if (pair_rank_3 == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab * 16384)), "l"((&Q)), "r"(slab * 64), "r"(head_3 * 8), "r"(token0),
                                       "r"(q_ready_addr + (c_1) * 8), "h"((uint16_t)(5)) : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab * 16384)), "l"((&Q)), "r"(slab * 64), "r"(head_3 * 8), "r"(token0),
                                       "r"(q_ready_addr + (c_1) * 8), "h"((uint16_t)(10)) : "memory");
                            }
                        }
                    }
                }
                if (c_1 >= 6) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(landed_addr + (lag_stage_1) * 8, lag_ph);
                    mbarrier_wait_token(landed_addr + (lag_stage_1) * 8, lag_ph, _mbar_token_0);
                    lag_stage_1 += 1;
                    if (lag_stage_1 == 12) { lag_stage_1 = 0; lag_ph ^= 1; }
                }
                unsigned int kv_index = (unsigned int)(blockIdx.z * 2) * num_kv_heads + head_3;
                unsigned int token = tile0 * 64 + tok_half;
                mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 4096);
                    if (pair_id_3 == (unsigned int)(c_1 & 1)) {
                        if (pair_rank_3 == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sk_addr + load_stage * 8192 + 4096), "l"((&KV)), "r"(c_1 * 128), "r"(token), "r"(kv_index),
                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sk_addr + load_stage * 8192 + 4096), "l"((&KV)), "r"(c_1 * 128), "r"(token), "r"(kv_index),
                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 12) { load_stage = 0; _phase_free ^= 1; }
            }
            #pragma unroll 1
            for (unsigned int tile_3 = 1; tile_3 < num_tiles_3; tile_3++) {
                #pragma unroll
                for (int c_2 = 0; c_2 < 4; c_2++) {
                    if (tile_3 * 6 + (unsigned int)c_2 >= 6) {
                        uint32_t _mbar_token_1 = mbarrier_try_wait(landed_addr + (lag_stage_1) * 8, lag_ph);
                        mbarrier_wait_token(landed_addr + (lag_stage_1) * 8, lag_ph, _mbar_token_1);
                        lag_stage_1 += 1;
                        if (lag_stage_1 == 12) { lag_stage_1 = 0; lag_ph ^= 1; }
                    }
                    unsigned int kv_index_1 = (unsigned int)(blockIdx.z * 2) * num_kv_heads + head_3;
                    unsigned int token_1 = tile_3 * 64 + tok_half;
                    mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 4096);
                        if (pair_id_3 == (unsigned int)(c_2 & 1)) {
                            if (pair_rank_3 == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sk_addr + load_stage * 8192 + 4096), "l"((&KV)), "r"(c_2 * 128), "r"(token_1), "r"(kv_index_1),
                                       "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sk_addr + load_stage * 8192 + 4096), "l"((&KV)), "r"(c_2 * 128), "r"(token_1), "r"(kv_index_1),
                                       "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                            }
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 12) { load_stage = 0; _phase_free ^= 1; }
                }
                if (tile_3 > 0) {
                    #pragma unroll
                    for (int v_2 = 0; v_2 < 2; v_2++) {
                        if (tile_3 * 6 + 4 + (unsigned int)v_2 >= 6) {
                            uint32_t _mbar_token_2 = mbarrier_try_wait(landed_addr + (lag_stage_1) * 8, lag_ph);
                            mbarrier_wait_token(landed_addr + (lag_stage_1) * 8, lag_ph, _mbar_token_2);
                            lag_stage_1 += 1;
                            if (lag_stage_1 == 12) { lag_stage_1 = 0; lag_ph ^= 1; }
                        }
                        unsigned int kv_index_2 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_3;
                        unsigned int token_2 = (tile_3 - 1) * 64 + zero_off;
                        mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 8192);
                            if (pair_rank_3 == (unsigned int)(v_2 & 1)) {
                                if (pair_id_3 == 0) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(sk_addr + load_stage * 8192), "l"((&KVV)), "r"(output_col_3 + (unsigned int)(v_2 * 128)), "r"(token_2), "r"(kv_index_2),
                                           "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                                } else {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(sk_addr + load_stage * 8192), "l"((&KVV)), "r"(output_col_3 + (unsigned int)(v_2 * 128)), "r"(token_2), "r"(kv_index_2),
                                           "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(12)) : "memory");
                                }
                            }
                        }
                        load_stage += 1;
                        if (load_stage == 12) { load_stage = 0; _phase_free ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                if (num_tiles_3 * 6 + 4 + (unsigned int)v_3 >= 6) {
                    uint32_t _mbar_token_3 = mbarrier_try_wait(landed_addr + (lag_stage_1) * 8, lag_ph);
                    mbarrier_wait_token(landed_addr + (lag_stage_1) * 8, lag_ph, _mbar_token_3);
                    lag_stage_1 += 1;
                    if (lag_stage_1 == 12) { lag_stage_1 = 0; lag_ph ^= 1; }
                }
                unsigned int kv_index_3 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_3;
                unsigned int token_3 = (num_tiles_3 - 1) * 64 + zero_off;
                mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 8192);
                    if (pair_rank_3 == (unsigned int)(v_3 & 1)) {
                        if (pair_id_3 == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sk_addr + load_stage * 8192), "l"((&KVV)), "r"(output_col_3 + (unsigned int)(v_3 * 128)), "r"(token_3), "r"(kv_index_3),
                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(3)) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(sk_addr + load_stage * 8192), "l"((&KVV)), "r"(output_col_3 + (unsigned int)(v_3 * 128)), "r"(token_3), "r"(kv_index_3),
                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(12)) : "memory");
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 12) { load_stage = 0; _phase_free ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: idle ----
    if (warp >= 14 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // idle_main
            __syncwarp();
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
