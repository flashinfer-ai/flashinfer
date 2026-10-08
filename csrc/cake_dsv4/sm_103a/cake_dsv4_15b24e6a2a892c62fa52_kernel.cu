/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "cake_dsv4_device_common.cuh"
// Translation-unit-local helpers declared by the shared preamble (per-program policy).
// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}


#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_V_LO_OFF 115712
#define SMEM_SMEM_V_LO_STAGE_BYTES 16384
#define SMEM_SMEM_V_LO_STRIDE 32768
#define SMEM_SMEM_V_HI_OFF 132096
#define SMEM_SMEM_V_HI_STAGE_BYTES 16384
#define SMEM_SMEM_V_HI_STRIDE 32768
#define SMEM_SMEM_P_OFF 181248
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_INDICES_OFF 214016
#define SMEM_SMEM_INDICES_STAGE_BYTES 4608
#define SMEM_SMEM_INDICES_STRIDE 4608
#define SMEM_SMEM_SOFTMAX_EXCHANGE_OFF 218624
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STRIDE 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_OFF 219136
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STRIDE 512
#define SMEM_SMEM_INDEX_VALID_OFF 219648
#define SMEM_SMEM_INDEX_VALID_STAGE_BYTES 288
#define SMEM_SMEM_INDEX_VALID_STRIDE 288
#define SMEM_TOTAL 220032
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_15b24e6a2a892c62fa52(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_k, const __grid_constant__ CUtensorMap tmap_compressed_k, const __grid_constant__ CUtensorMap tmap_swa_v, const __grid_constant__ CUtensorMap tmap_compressed_v, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int num_query_tokens, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int sparse_topk, int has_sinks, int total_work_items, int ragged_query, int max_q_len, int batch_size)
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
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 96)
    #define s_empty_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define p_empty_addr (mbar_base + 144)
    #define o_empty_addr (mbar_base + 160)
    #define stats_addr (mbar_base + 168)
    #define stats_empty_addr (mbar_base + 184)
    #define o_full_addr (mbar_base + 200)
    #define o_first_slice_seeded_addr (mbar_base + 208)
    #define tmem_dealloc_peer_addr (mbar_base + 216)
    #define index_valid_full_addr (mbar_base + 224)
    #define index_valid_empty_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_K_OFF);
    const int smem_k_addr = smem + SMEM_SMEM_K_OFF;
    __nv_bfloat16* smem_v_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_LO_OFF);
    const int smem_v_lo_addr = smem + SMEM_SMEM_V_LO_OFF;
    __nv_bfloat16* smem_v_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_HI_OFF);
    const int smem_v_hi_addr = smem + SMEM_SMEM_V_HI_OFF;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P_OFF);
    const int smem_p_addr = smem + SMEM_SMEM_P_OFF;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_INDICES_OFF);
    const int smem_indices_addr = smem + SMEM_SMEM_INDICES_OFF;
    float* smem_softmax_exchange = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_SOFTMAX_EXCHANGE_OFF);
    const int smem_softmax_exchange_addr = smem + SMEM_SMEM_SOFTMAX_EXCHANGE_OFF;
    float* smem_epilogue_exchange = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_EPILOGUE_EXCHANGE_OFF);
    const int smem_epilogue_exchange_addr = smem + SMEM_SMEM_EPILOGUE_EXCHANGE_OFF;
    unsigned int* smem_index_valid = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_INDEX_VALID_OFF);
    const int smem_index_valid_addr = smem + SMEM_SMEM_INDEX_VALID_OFF;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            mbarrier_init(smem + 192, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // o_first_slice_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 208, 256);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 216, 32);
            // index_valid_full: 2 barriers, init_count=32
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            // index_valid_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
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
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_wg_main
            const int softmax_dummy = 0;
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int head_row_base = warp % 2 * 32;
            const int score_row_base = warp % 4 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row = head_row_base + lane;
            const int exchange_idx = n_half * 64 + my_row;
            int tile_cursor = 0;
            int valid_iter = 0;
            float seed_zero[4];
            #pragma unroll
            for (int seed_i = 0; seed_i < 4; seed_i++) {
                seed_zero[seed_i] = 0.0f;
            }
            #pragma unroll
            for (int seed_stage = 0; seed_stage < 1; seed_stage++) {
                #pragma unroll
                for (int seed_col = 128 + seed_stage * 128; seed_col < 128 + (seed_stage + 1) * 128; seed_col += 4) {
                    int seed_addr = taddr + (unsigned int)seed_col + (unsigned int)(score_row_base << 16);
                    tmem_st_x4_f32(seed_addr, seed_zero);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(o_first_slice_seeded_addr);
            int peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_first_slice_seeded_addr), "r"(peer_rank) : "memory");
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work_items * 2; work_idx += num_clusters) {
                int split_idx = work_idx % (unsigned int)(((0) ? 3 : 1));
                int query_idx = work_idx / (unsigned int)(((0) ? 3 : 1)) / 2;
                int _max_4 = ((sparse_topk_lens[query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx] + sparse_topk_lens_offset) : (0));
                int _min_0 = ((_max_4) < (sparse_topk) ? (_max_4) : (sparse_topk));
                int active_topk = _min_0;
                int query_batch = query_idx / max_q_len;
                int query_offset = query_idx - query_batch * max_q_len;
                int query_length = max_q_len;
                if (ragged_query != 0) {
                    query_batch = 0;
                    #pragma unroll 2
                    for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                        int lane_entry = chunk * 32 + lane + 1;
                        int _min_1 = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                        int lane_load = _min_1;
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, lane_entry <= batch_size && query_idx >= cum_seq_lens_q[lane_load]);
                        unsigned int started = _vote_0;
                        int _popc_0 = __popc(started);
                        query_batch = query_batch + _popc_0;
                    }
                    int query_begin = cum_seq_lens_q[query_batch];
                    query_length = cum_seq_lens_q[query_batch + 1] - query_begin;
                    query_offset = query_idx - query_begin;
                }
                int visible = seq_lens[query_batch] - query_length + query_offset + 1;
                if (visible < 0) {
                    visible = 0;
                }
                if (visible > 128) {
                    visible = 128;
                }
                int swa_visible = visible;
                int swa_active_topk = active_topk;
                if (swa_active_topk > swa_visible) {
                    swa_active_topk = swa_visible;
                }
                int num_kv_tiles = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                int first_kv_tile = split_idx * num_kv_tiles;
                int head_idx = cta_rank * 64 + my_row;
                float row_max_val = -CAKE_INF;
                float row_sum_val = 0.0f;
                if (has_sinks != 0 && head_idx < num_heads && split_idx == 0) {
                    row_max_val = sinks[head_idx] * 1.4426950408889634f / softmax_scale_log2;
                    if (n_half == 0) {
                        row_sum_val = 1.0f;
                    }
                }
                int valid_slot = valid_iter & 1;
                int valid_full_phase = valid_iter >> 1 & 1;
                mbarrier_wait(index_valid_full_addr + (valid_slot) * 8, valid_full_phase);
                int valid_word_base = valid_slot * 36 + n_half * 2;
                #pragma unroll 1
                for (int tile = 0; tile < num_kv_tiles; tile++) {
                    int pipeline_tile = tile_cursor + tile;
                    int score_phase = pipeline_tile & 1;
                    int s_full_phase = pipeline_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (score_phase) * 8, s_full_phase);
                    int score_offset = ((score_phase != 0) ? 64 : 0);
                    int score_base = taddr + (unsigned int)score_offset + (unsigned int)(score_row_base << 16);
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(score_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(score_base + 32));
                    int tile_bound = active_topk;
                    if (first_kv_tile + tile == 0) {
                        tile_bound = swa_active_topk;
                    }
                    int valid_cols = tile_bound - (first_kv_tile + tile) * 128 - n_half * 64;
                    if (valid_cols < 0) {
                        valid_cols = 0;
                    }
                    if (valid_cols > 64) {
                        valid_cols = 64;
                    }
                    unsigned int keep_lo = smem_index_valid[valid_word_base + tile * 4];
                    unsigned int keep_hi = smem_index_valid[valid_word_base + tile * 4 + 1];
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_0 = valid_cols;
                        if (_lim_0 <= 0) { _thresh_mask_0 = 0u; }
                        else if (_lim_0 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_0) : "r"(_lim_0));
                        }
                    }
                    uint32_t _thresh_mask_1;
                    {
                        int _lim_1 = valid_cols - 32;
                        if (_lim_1 <= 0) { _thresh_mask_1 = 0u; }
                        else if (_lim_1 >= 32) { _thresh_mask_1 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_1) : "r"(_lim_1));
                        }
                    }
                    if (!(_thresh_mask_0 & keep_lo & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                    if (!(_thresh_mask_0 & keep_lo & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                    if (!(_thresh_mask_1 & keep_hi & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                    float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_2);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_2);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_2);
                    float _max_5 = max_noftz(_tmem_load_0_max, row_max_val);
                    float new_max = _max_5;
                    smem_softmax_exchange[exchange_idx] = new_max;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float _max_6 = max_noftz(new_max, smem_softmax_exchange[exchange_idx ^ 64]);
                    new_max = _max_6;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float no_correction = (((new_max - row_max_val) * softmax_scale_log2 <= 0.0f) ? 1.0f : 0.0f);
                    float delta = softmax_scale_log2 * (row_max_val - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    float exp_delta = _exp2_0;
                    float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                    row_max_val = new_max;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_4 = {-max_scaled, -max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 32; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_3, _fma_c2_4);
                    #pragma unroll
                    for (int _le = 0; _le < 64; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_5 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_5);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_5);
                    float _tmem_load_0_sum = _reg_reduce_sum2_5.x + _reg_reduce_sum2_5.y;
                    row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                    int stats_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(stats_empty_addr + (score_phase) * 8, stats_empty_phase);
                    float meta[4];
                    meta[0] = row_sum_val;
                    meta[1] = row_max_val;
                    meta[2] = acc_scale;
                    meta[3] = no_correction;
                    int meta_addr = taddr + 384 + (unsigned int)(score_phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(head_row_base << 16);
                    tmem_st_x4_f32(meta_addr, meta);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(stats_addr + (score_phase) * 8);
                    int p_stage = pipeline_tile % 2;
                    int p_empty_phase = pipeline_tile / 2 & 1 ^ 1;
                    mbarrier_wait(p_empty_addr + (p_stage) * 8, p_empty_phase);
                    uint32_t _tmem_load_0_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int smem_p_stage = p_stage * 2 + n_half;
                    int p_base_smem = smem_p_addr + (unsigned int)(smem_p_stage * 8192);
                    #pragma unroll
                    for (int vec = 0; vec < 8; vec++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_smem + (my_row * 128 + vec * 16 ^ (my_row * 128 + vec * 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[vec * 4]), "r"(_tmem_load_0_bf16[vec * 4 + 1]), "r"(_tmem_load_0_bf16[vec * 4 + 2]), "r"(_tmem_load_0_bf16[vec * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (unsigned int)(p_stage * 8)), "r"(0) : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (unsigned int)(score_phase * 8)), "r"(0) : "memory");
                }
                mbarrier_arrive(index_valid_empty_addr + (valid_slot) * 8);
                valid_iter = valid_iter + 1;
                tile_cursor = tile_cursor + num_kv_tiles;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_wg_main
            const int correction_dummy = 0;
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int head_row_base_1 = warp % 2 * 32;
            const int output_row_base = warp % 4 * 32;
            const int n_half_1 = warp % 4 / 2;
            const int my_row_1 = head_row_base_1 + lane;
            const int corr_row = head_row_base_1 << 16;
            int tile_cursor_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = cluster_id; work_idx_1 < total_work_items * 2; work_idx_1 += num_clusters) {
                int split_idx_1 = work_idx_1 % (unsigned int)(((0) ? 3 : 1));
                int query_idx_1 = work_idx_1 / (unsigned int)(((0) ? 3 : 1)) / 2;
                int v_half = work_idx_1 / (unsigned int)(((0) ? 3 : 1)) % 2;
                int num_kv_tiles_1 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                int v_slice_base = v_half * 4;
                float final_sum = 0.0f;
                float final_max = -CAKE_INF;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile_1 = tile_cursor_1 + tile_1;
                    int score_phase_1 = pipeline_tile_1 & 1;
                    int stats_phase = pipeline_tile_1 >> 1 & 1;
                    mbarrier_wait(stats_addr + (score_phase_1) * 8, stats_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int meta_addr_1 = taddr + 384 + (unsigned int)(score_phase_1 * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row;
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], meta_addr_1);
                    final_sum = _tmem_load_1[0];
                    final_max = _tmem_load_1[1];
                    float acc_scale_1 = _tmem_load_1[2];
                    float no_correction_1 = _tmem_load_1[3];
                    if (tile_1 > 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_1 = __all_sync(0xFFFFFFFF, no_correction_1 == 1.0f);
                        int skip_correction = _vote_1;
                        if (skip_correction == 0) {
                            #pragma unroll
                            for (int local_slice = 0; local_slice < 2; local_slice++) {
                                #pragma unroll
                                for (int acc_stage = 0; acc_stage < 1; acc_stage++) {
                                    int o_base = taddr + 128 + (unsigned int)(acc_stage * 128) + (unsigned int)(local_slice * 64) + (unsigned int)(output_row_base << 16);
                                    #pragma unroll
                                    for (int c = 0; c < 64; c += 16) {
                                        float _tmem_load_2[16];
                                        tmem_ld_x16(&_tmem_load_2[0], o_base + c);
                                        #if __CUDA_ARCH__ >= 1000
                                        const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                                        #else
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 16; _ls++) {
                                            _tmem_load_2[_ls] = _tmem_load_2[_ls] * acc_scale_1;
                                        }
                                        #endif
                                        tmem_st_x16_f32(o_base + c, _tmem_load_2);
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_empty_addr), "r"(0) : "memory");
                    }
                    mbarrier_arrive(stats_empty_addr + (score_phase_1) * 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                const int exchange_idx_1 = n_half_1 * 64 + my_row_1;
                smem_epilogue_exchange[exchange_idx_1] = final_sum;
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                final_sum = final_sum + smem_epilogue_exchange[exchange_idx_1 ^ 64];
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                float _rcp_0 = approx_rcp(final_sum);
                float inv_sum = ((final_sum > 0.0f) ? _rcp_0 : 0.0f);
                int head_idx_1 = cta_rank * 64 + my_row_1;
                int output_offset = ((0) ? ((query_idx_1 * num_heads + head_idx_1) * ((0) ? 3 : 1) + split_idx_1) * 512 : (query_idx_1 * num_heads + head_idx_1) * 512);
                #pragma unroll
                for (int acc_stage_1 = 0; acc_stage_1 < 1; acc_stage_1++) {
                    #pragma unroll
                    for (int local_slice_1 = 0; local_slice_1 < 2; local_slice_1++) {
                        int logical_slice = v_slice_base + acc_stage_1 * 2 * 2 + n_half_1 * 2 + local_slice_1;
                        int o_base_1 = taddr + 128 + (unsigned int)(acc_stage_1 * 128) + (unsigned int)(local_slice_1 * 64) + (unsigned int)(output_row_base << 16);
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 64; c_1 += 32) {
                            float _tmem_load_3[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                                : "r"(o_base_1 + c_1));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            int gmem_base = output_offset + logical_slice * 64 + c_1;
                            #pragma unroll
                            for (int j = 0; j < 32; j += 16) {
                                {
                                    const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[j])[_ps], _prescale2_1);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_3[j + _ps] *= inv_sum * output_scale;
                                    #endif
                                    {
                                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_3[j + 0], _tmem_load_3[j + 1]);
                                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_3[j + 2], _tmem_load_3[j + 3]);
                                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_3[j + 4], _tmem_load_3[j + 5]);
                                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_3[j + 6], _tmem_load_3[j + 7]);
                                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_3[j + 8], _tmem_load_3[j + 9]);
                                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_3[j + 10], _tmem_load_3[j + 11]);
                                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_3[j + 12], _tmem_load_3[j + 13]);
                                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_3[j + 14], _tmem_load_3[j + 15]);
                                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base + j)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr), "r"(0) : "memory");
                tile_cursor_1 = tile_cursor_1 + num_kv_tiles_1;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int mma_dummy = 0;
            unsigned int k_stage = 0;
            unsigned int v_stage = 0;
            int tile_cursor_2 = 0;
            unsigned int _phase_o_first_slice_seeded_0 = 0;
            mbarrier_wait(o_first_slice_seeded_addr, _phase_o_first_slice_seeded_0);
            _phase_o_first_slice_seeded_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = cluster_id; work_idx_2 < total_work_items * 2; work_idx_2 += num_clusters) {
                int num_kv_tiles_2 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                if (cta_rank == 0) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                        int pipeline_tile_2 = tile_cursor_2 + tile_2;
                        int score_phase_2 = pipeline_tile_2 & 1;
                        int s_empty_phase = pipeline_tile_2 >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (score_phase_2) * 8, s_empty_phase);
                        int score_col = ((score_phase_2 != 0) ? 64 : 0);
                        #pragma unroll
                        for (int qk_stage = 0; qk_stage < 4; qk_stage++) {
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2) * 512;
                            int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(((qk_stage == 0) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (qk_stage * 2 + 1) * 512;
                            int _mma_b_lo_1 = (((smem_k_addr + 8192) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 3) { k_stage = 0; _phase_k_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(s_full_addr + (score_phase_2) * 8, (uint16_t)(3));
                        if (tile_2 > 0) {
                            int prev_pipeline_tile = pipeline_tile_2 - 1;
                            int p_stage_1 = prev_pipeline_tile % 2;
                            int p_full_phase = prev_pipeline_tile / 2 & 1;
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                            mbarrier_wait(p_full_addr + (p_stage_1) * 8, p_full_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int ps = 0; ps < 2; ps++) {
                                int smem_p_stage_1 = p_stage_1 * 2 + ps;
                                mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                                #pragma unroll
                                for (int acc_stage_2 = 0; acc_stage_2 < 1; acc_stage_2++) {
                                    int output_col = 128 + acc_stage_2 * 128;
                                    if (acc_stage_2 == 0) {
                                        int _mma_a_lo_2 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_1) * 512;
                                        int _mma_b_lo_2 = ((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 2048;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (output_col))), "r"(((((ps == 0) ? first_pv : 0)) ? 0 : 1)));
                                    } else {
                                        int _mma_a_lo_3 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_1) * 512;
                                        int _mma_b_lo_3 = ((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 2048;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_scratch + (output_col))), "r"(((((ps == 0) ? first_pv : 0)) ? 0 : 1)));
                                    }
                                }
                                elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                                v_stage += 1;
                                if (v_stage == 2) { v_stage = 0; _phase_v_full ^= 1; }
                            }
                            first_pv = 0;
                            elect_commit_cg2_multicast(p_empty_addr + (p_stage_1) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        }
                    }
                    int last_pipeline_tile = tile_cursor_2 + num_kv_tiles_2 - 1;
                    int last_p_stage = last_pipeline_tile % 2;
                    int last_p_phase = last_pipeline_tile / 2 & 1;
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(p_full_addr + (last_p_stage) * 8, last_p_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int ps_1 = 0; ps_1 < 2; ps_1++) {
                        int smem_p_stage_2 = last_p_stage * 2 + ps_1;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                        #pragma unroll
                        for (int acc_stage_3 = 0; acc_stage_3 < 1; acc_stage_3++) {
                            int output_col_1 = 128 + acc_stage_3 * 128;
                            if (acc_stage_3 == 0) {
                                int _mma_a_lo_4 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_2) * 512;
                                int _mma_b_lo_4 = ((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 2048;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (output_col_1))), "r"(((((ps_1 == 0) ? first_pv : 0)) ? 0 : 1)));
                            } else {
                                int _mma_a_lo_5 = (((smem_p_addr) >> 4) & 0x3FFF) + (smem_p_stage_2) * 512;
                                int _mma_b_lo_5 = ((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 2048;
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
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_scratch + (output_col_1))), "r"(((((ps_1 == 0) ? first_pv : 0)) ? 0 : 1)));
                            }
                        }
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 2) { v_stage = 0; _phase_v_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(p_empty_addr + (last_p_stage) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                }
                tile_cursor_2 = tile_cursor_2 + num_kv_tiles_2;
            }
            if (cta_rank == 0) {
                #pragma unroll
                for (int tail_offset = 0; tail_offset < 2; tail_offset++) {
                    int tail_tile = tile_cursor_2 + tail_offset;
                    int tail_stage = tail_tile & 1;
                    int tail_phase = tail_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(s_empty_addr + (tail_stage) * 8, tail_phase);
                }
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            int peer_rank_1 = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_dealloc_peer_addr), "r"(peer_rank_1) : "memory");
            mbarrier_wait(tmem_dealloc_peer_addr, 0);
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_wg ----
    if (warp >= 9 && warp <= 12) {
        { // load_wg_main
            const int load_dummy = 0;
            const int load_warp_rank = warp - 9;
            unsigned int k_stage_1 = 0;
            unsigned int v_stage_1 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int work_idx_3 = cluster_id; work_idx_3 < total_work_items * 2; work_idx_3 += num_clusters) {
                int split_idx_2 = work_idx_3 % (unsigned int)(((0) ? 3 : 1));
                int query_idx_2 = work_idx_3 / (unsigned int)(((0) ? 3 : 1)) / 2;
                int v_half_1 = work_idx_3 / (unsigned int)(((0) ? 3 : 1)) % 2;
                int num_kv_tiles_3 = ((sparse_topk + 128 - 1) / 128 + ((0) ? 3 : 1) - 1) / ((0) ? 3 : 1);
                int first_kv_tile_1 = split_idx_2 * num_kv_tiles_3;
                int v_col_base = v_half_1 * 256;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (load_warp_rank == 0) {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(131072)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        #pragma unroll
                        for (int q_stage = 0; q_stage < 8; q_stage++) {
                            tma_4d_gmem2smem_cta2(smem_q_addr + (unsigned int)(q_stage * 8192), (&tmap_q), 0, cta_rank * 64, q_stage, query_idx_2, ((q_full_addr) & 0xFEFFFFFF));
                        }
                    }
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                #pragma unroll
                for (int qk_stage_1 = 0; qk_stage_1 < 4; qk_stage_1++) {
                    mbarrier_wait(k_empty_addr + (k_stage_1) * 8, _phase_k_empty);
                    if (load_warp_rank == 0) {
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            }
                        }
                    }
                    int first_rows[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&first_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows[(0) + 3]))
                        : "r"(smem_indices_addr));
                    int _max_0 = ((first_rows[0]) > (0) ? (first_rows[0]) : (0));
                    #pragma unroll
                    for (int elected_group = 0; elected_group < 4; elected_group++) {
                        int group = load_warp_rank * 4 + elected_group;
                        int index_offset = cta_rank * 64 + group * 4;
                        int raw_rows[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                            : "r"(smem_indices_addr + (unsigned int)(index_offset * 4)));
                        if (elect_sync()) {
                            if (first_kv_tile_1 == 0) {
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512), (&tmap_swa_k), qk_stage_1 * 128, ((raw_rows[0] >= 0) ? raw_rows[0] : _max_0), ((raw_rows[1] >= 0) ? raw_rows[1] : _max_0), ((raw_rows[2] >= 0) ? raw_rows[2] : _max_0), ((raw_rows[3] >= 0) ? raw_rows[3] : _max_0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512) + 8192, (&tmap_swa_k), qk_stage_1 * 128 + 64, ((raw_rows[0] >= 0) ? raw_rows[0] : _max_0), ((raw_rows[1] >= 0) ? raw_rows[1] : _max_0), ((raw_rows[2] >= 0) ? raw_rows[2] : _max_0), ((raw_rows[3] >= 0) ? raw_rows[3] : _max_0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            } else {
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512), (&tmap_compressed_k), qk_stage_1 * 128, ((raw_rows[0] >= 0) ? raw_rows[0] : _max_0), ((raw_rows[1] >= 0) ? raw_rows[1] : _max_0), ((raw_rows[2] >= 0) ? raw_rows[2] : _max_0), ((raw_rows[3] >= 0) ? raw_rows[3] : _max_0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group * 512) + 8192, (&tmap_compressed_k), qk_stage_1 * 128 + 64, ((raw_rows[0] >= 0) ? raw_rows[0] : _max_0), ((raw_rows[1] >= 0) ? raw_rows[1] : _max_0), ((raw_rows[2] >= 0) ? raw_rows[2] : _max_0), ((raw_rows[3] >= 0) ? raw_rows[3] : _max_0), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            }
                        }
                    }
                    k_stage_1 += 1;
                    if (k_stage_1 == 3) { k_stage_1 = 0; _phase_k_empty ^= 1; }
                }
                #pragma unroll 1
                for (int tile_3 = 1; tile_3 < num_kv_tiles_3; tile_3++) {
                    #pragma unroll
                    for (int qk_stage_2 = 0; qk_stage_2 < 4; qk_stage_2++) {
                        mbarrier_wait(k_empty_addr + (k_stage_1) * 8, _phase_k_empty);
                        if (load_warp_rank == 0) {
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                            }
                        }
                        int first_rows_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_1[(0) + 3]))
                            : "r"(smem_indices_addr + (unsigned int)(tile_3 * 128 * 4)));
                        int _max_1 = ((first_rows_1[0]) > (0) ? (first_rows_1[0]) : (0));
                        #pragma unroll
                        for (int elected_group_1 = 0; elected_group_1 < 4; elected_group_1++) {
                            int group_1 = load_warp_rank * 4 + elected_group_1;
                            int index_offset_1 = tile_3 * 128 + cta_rank * 64 + group_1 * 4;
                            int raw_rows_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_offset_1 * 4)));
                            if (elect_sync()) {
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512), (&tmap_compressed_k), qk_stage_2 * 128, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : _max_1), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : _max_1), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : _max_1), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : _max_1), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                tma_gather4_gmem2smem_mc_cta2(smem_k_addr + k_stage_1 * 16384 + (unsigned int)(group_1 * 512) + 8192, (&tmap_compressed_k), qk_stage_2 * 128 + 64, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : _max_1), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : _max_1), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : _max_1), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : _max_1), ((k_full_addr + (k_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                            }
                        }
                        k_stage_1 += 1;
                        if (k_stage_1 == 3) { k_stage_1 = 0; _phase_k_empty ^= 1; }
                    }
                    int prev_tile = tile_3 - 1;
                    #pragma unroll
                    for (int ps_2 = 0; ps_2 < 2; ps_2++) {
                        mbarrier_wait(v_empty_addr + (v_stage_1) * 8, _phase_v_empty);
                        if (load_warp_rank == 0) {
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                            }
                        }
                        int first_rows_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&first_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_2[(0) + 3]))
                            : "r"(smem_indices_addr + (unsigned int)(prev_tile * 128 * 4)));
                        int _max_2 = ((first_rows_2[0]) > (0) ? (first_rows_2[0]) : (0));
                        if (first_kv_tile_1 + prev_tile == 0) {
                            #pragma unroll
                            for (int elected_group_2 = 0; elected_group_2 < 4; elected_group_2++) {
                                int group_2 = load_warp_rank * 4 + elected_group_2;
                                int index_offset_2 = prev_tile * 128 + ps_2 * 64 + group_2 * 4;
                                int raw_rows_2[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_2[(0) + 3]))
                                    : "r"(smem_indices_addr + (unsigned int)(index_offset_2 * 4)));
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int acc_stage_4 = 0; acc_stage_4 < 1; acc_stage_4++) {
                                        int dst_v = ((acc_stage_4 == 0) ? smem_v_lo_addr + v_stage_1 * 32768 : smem_v_hi_addr + v_stage_1 * 32768);
                                        tma_gather4_gmem2smem_mc_cta2(dst_v + group_2 * 512, (&tmap_swa_v), v_col_base + acc_stage_4 * 128 * 2 + cta_rank * 128, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : _max_2), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : _max_2), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : _max_2), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : _max_2), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(dst_v + group_2 * 512 + 8192, (&tmap_swa_v), v_col_base + acc_stage_4 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_2[0] >= 0) ? raw_rows_2[0] : _max_2), ((raw_rows_2[1] >= 0) ? raw_rows_2[1] : _max_2), ((raw_rows_2[2] >= 0) ? raw_rows_2[2] : _max_2), ((raw_rows_2[3] >= 0) ? raw_rows_2[3] : _max_2), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int elected_group_3 = 0; elected_group_3 < 4; elected_group_3++) {
                                int group_3 = load_warp_rank * 4 + elected_group_3;
                                int index_offset_3 = prev_tile * 128 + ps_2 * 64 + group_3 * 4;
                                int raw_rows_3[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_3[(0) + 3]))
                                    : "r"(smem_indices_addr + (unsigned int)(index_offset_3 * 4)));
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int acc_stage_5 = 0; acc_stage_5 < 1; acc_stage_5++) {
                                        int dst_v_1 = ((acc_stage_5 == 0) ? smem_v_lo_addr + v_stage_1 * 32768 : smem_v_hi_addr + v_stage_1 * 32768);
                                        tma_gather4_gmem2smem_mc_cta2(dst_v_1 + group_3 * 512, (&tmap_compressed_v), v_col_base + acc_stage_5 * 128 * 2 + cta_rank * 128, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : _max_2), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : _max_2), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : _max_2), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : _max_2), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                        tma_gather4_gmem2smem_mc_cta2(dst_v_1 + group_3 * 512 + 8192, (&tmap_compressed_v), v_col_base + acc_stage_5 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_3[0] >= 0) ? raw_rows_3[0] : _max_2), ((raw_rows_3[1] >= 0) ? raw_rows_3[1] : _max_2), ((raw_rows_3[2] >= 0) ? raw_rows_3[2] : _max_2), ((raw_rows_3[3] >= 0) ? raw_rows_3[3] : _max_2), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    }
                                }
                            }
                        }
                        v_stage_1 += 1;
                        if (v_stage_1 == 2) { v_stage_1 = 0; _phase_v_empty ^= 1; }
                    }
                }
                int last_tile = num_kv_tiles_3 - 1;
                #pragma unroll
                for (int ps_3 = 0; ps_3 < 2; ps_3++) {
                    mbarrier_wait(v_empty_addr + (v_stage_1) * 8, _phase_v_empty);
                    if (load_warp_rank == 0) {
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            }
                        }
                    }
                    int first_rows_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&first_rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&first_rows_3[(0) + 3]))
                        : "r"(smem_indices_addr + (unsigned int)(last_tile * 128 * 4)));
                    int _max_3 = ((first_rows_3[0]) > (0) ? (first_rows_3[0]) : (0));
                    if (first_kv_tile_1 + last_tile == 0) {
                        #pragma unroll
                        for (int elected_group_4 = 0; elected_group_4 < 4; elected_group_4++) {
                            int group_4 = load_warp_rank * 4 + elected_group_4;
                            int index_offset_4 = last_tile * 128 + ps_3 * 64 + group_4 * 4;
                            int raw_rows_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_4[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_offset_4 * 4)));
                            if (elect_sync()) {
                                #pragma unroll
                                for (int acc_stage_6 = 0; acc_stage_6 < 1; acc_stage_6++) {
                                    int dst_v_2 = ((acc_stage_6 == 0) ? smem_v_lo_addr + v_stage_1 * 32768 : smem_v_hi_addr + v_stage_1 * 32768);
                                    tma_gather4_gmem2smem_mc_cta2(dst_v_2 + group_4 * 512, (&tmap_swa_v), v_col_base + acc_stage_6 * 128 * 2 + cta_rank * 128, ((raw_rows_4[0] >= 0) ? raw_rows_4[0] : _max_3), ((raw_rows_4[1] >= 0) ? raw_rows_4[1] : _max_3), ((raw_rows_4[2] >= 0) ? raw_rows_4[2] : _max_3), ((raw_rows_4[3] >= 0) ? raw_rows_4[3] : _max_3), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    tma_gather4_gmem2smem_mc_cta2(dst_v_2 + group_4 * 512 + 8192, (&tmap_swa_v), v_col_base + acc_stage_6 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_4[0] >= 0) ? raw_rows_4[0] : _max_3), ((raw_rows_4[1] >= 0) ? raw_rows_4[1] : _max_3), ((raw_rows_4[2] >= 0) ? raw_rows_4[2] : _max_3), ((raw_rows_4[3] >= 0) ? raw_rows_4[3] : _max_3), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int elected_group_5 = 0; elected_group_5 < 4; elected_group_5++) {
                            int group_5 = load_warp_rank * 4 + elected_group_5;
                            int index_offset_5 = last_tile * 128 + ps_3 * 64 + group_5 * 4;
                            int raw_rows_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_5[(0) + 3]))
                                : "r"(smem_indices_addr + (unsigned int)(index_offset_5 * 4)));
                            if (elect_sync()) {
                                #pragma unroll
                                for (int acc_stage_7 = 0; acc_stage_7 < 1; acc_stage_7++) {
                                    int dst_v_3 = ((acc_stage_7 == 0) ? smem_v_lo_addr + v_stage_1 * 32768 : smem_v_hi_addr + v_stage_1 * 32768);
                                    tma_gather4_gmem2smem_mc_cta2(dst_v_3 + group_5 * 512, (&tmap_compressed_v), v_col_base + acc_stage_7 * 128 * 2 + cta_rank * 128, ((raw_rows_5[0] >= 0) ? raw_rows_5[0] : _max_3), ((raw_rows_5[1] >= 0) ? raw_rows_5[1] : _max_3), ((raw_rows_5[2] >= 0) ? raw_rows_5[2] : _max_3), ((raw_rows_5[3] >= 0) ? raw_rows_5[3] : _max_3), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                    tma_gather4_gmem2smem_mc_cta2(dst_v_3 + group_5 * 512 + 8192, (&tmap_compressed_v), v_col_base + acc_stage_7 * 128 * 2 + cta_rank * 128 + 64, ((raw_rows_5[0] >= 0) ? raw_rows_5[0] : _max_3), ((raw_rows_5[1] >= 0) ? raw_rows_5[1] : _max_3), ((raw_rows_5[2] >= 0) ? raw_rows_5[2] : _max_3), ((raw_rows_5[3] >= 0) ? raw_rows_5[3] : _max_3), ((v_full_addr + (v_stage_1) * 8) & 0xFEFFFFFF), 1 << cta_rank);
                                }
                            }
                        }
                    }
                    v_stage_1 += 1;
                    if (v_stage_1 == 2) { v_stage_1 = 0; _phase_v_empty ^= 1; }
                }
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
        }
    }
    // ---- Role: index_warp ----
    if (warp == 13) {
        { // index_warp_main
            const int index_dummy = 0;
            int valid_iter_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_4 = cluster_id; work_idx_4 < total_work_items * 2; work_idx_4 += num_clusters) {
                int split_idx_3 = work_idx_4 % (unsigned int)(((0) ? 3 : 1));
                int query_idx_3 = work_idx_4 / (unsigned int)(((0) ? 3 : 1)) / 2;
                int owner_col_base = split_idx_3 * (9 / ((0) ? 3 : 1)) * 128;
                int* swa_row = swa_indices + (query_idx_3 * swa_index_stride);
                int* compressed_row = compressed_indices + (query_idx_3 * compressed_index_stride);
                int valid_slot_1 = valid_iter_1 & 1;
                int valid_empty_phase = valid_iter_1 >> 1 & 1 ^ 1;
                mbarrier_wait(index_valid_empty_addr + (valid_slot_1) * 8, valid_empty_phase);
                int staged[36];
                #pragma unroll
                for (int load_pass = 0; load_pass < 9 / ((0) ? 3 : 1); load_pass++) {
                    int load_col = owner_col_base + load_pass * 128 + lane * 4;
                    int* load_src = ((load_col < 128) ? (swa_row + load_col) : (compressed_row + (load_col - 128)));
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        staged[load_pass * 4 + i] = -1;
                    }
                    if (load_col + 4 <= sparse_topk) {
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(load_src + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_0[0 + 0] = _ivld_0.x;
                            _vec_load_0[0 + 1] = _ivld_0.y;
                            _vec_load_0[0 + 2] = _ivld_0.z;
                            _vec_load_0[0 + 3] = _ivld_0.w;
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            staged[load_pass * 4 + i_1] = _vec_load_0[i_1];
                        }
                    }
                    if (load_col < sparse_topk && load_col + 4 > sparse_topk) {
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 4; i_2++) {
                            if (load_col + i_2 < sparse_topk) {
                                staged[load_pass * 4 + i_2] = load_src[i_2];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int store_pass = 0; store_pass < 9 / ((0) ? 3 : 1); store_pass++) {
                    int store_offset = store_pass * 128 + lane * 4;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(store_offset * 4)), "r"(staged[store_pass * 4]), "r"(staged[store_pass * 4 + 1]), "r"(staged[store_pass * 4 + 2]), "r"(staged[store_pass * 4 + 3]) : "memory");
                    unsigned int staged_word = (((staged[store_pass * 4] >= 0) ? 1 : 0) | ((staged[store_pass * 4 + 1] >= 0) ? 1 : 0) << 1 | ((staged[store_pass * 4 + 2] >= 0) ? 1 : 0) << 2 | ((staged[store_pass * 4 + 3] >= 0) ? 1 : 0) << 3) << (lane & 7) * 4;
                    unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, staged_word, 1);
                    staged_word = staged_word | _shfl_xor_0;
                    unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, staged_word, 2);
                    staged_word = staged_word | _shfl_xor_1;
                    unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, staged_word, 4);
                    staged_word = staged_word | _shfl_xor_2;
                    if ((lane & 7) == 0) {
                        smem_index_valid[valid_slot_1 * 36 + store_pass * 4 + (lane >> 3)] = staged_word;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(index_valid_full_addr + (valid_slot_1) * 8);
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                asm volatile("barrier.sync 2, 160;" ::: "memory");
                valid_iter_1 = valid_iter_1 + 1;
            }
        }
    }
    // ---- Role: empty_warps ----
    if (warp >= 14 && warp <= 15) {
        { // empty_warps_main
            const int empty_dummy = 0;
            #pragma unroll 1
            for (unsigned int work_idx_5 = cluster_id; work_idx_5 < total_work_items * 2; work_idx_5 += num_clusters) {
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
