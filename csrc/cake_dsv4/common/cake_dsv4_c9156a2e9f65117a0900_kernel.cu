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
#define TMEM_NCOLS 80
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 72
#define TMEM_TMEM_O0_OFFSET 8
#define TMEM_TMEM_O1_OFFSET 16
#define TMEM_TMEM_O2_OFFSET 24
#define TMEM_TMEM_O3_OFFSET 32
#define TMEM_TMEM_O0B_OFFSET 40
#define TMEM_TMEM_O1B_OFFSET 48
#define TMEM_TMEM_O2B_OFFSET 56
#define TMEM_TMEM_O3B_OFFSET 64
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_Q_HALF_OFF 1024
#define SMEM_SMEM_Q_HALF_STAGE_BYTES 4096
#define SMEM_SMEM_Q_HALF_STRIDE 4096
#define SMEM_SMEM_KV_OFF 9216
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 9216
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 140288
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_P_HALF_OFF 140288
#define SMEM_SMEM_P_HALF_STAGE_BYTES 1024
#define SMEM_SMEM_P_HALF_STRIDE 1024
#define SMEM_SMEM_EXCH_OFF 142336
#define SMEM_SMEM_EXCH_STAGE_BYTES 128
#define SMEM_SMEM_EXCH_STRIDE 128
#define SMEM_SMEM_EXCH1_OFF 143040
#define SMEM_SMEM_EXCH1_STAGE_BYTES 128
#define SMEM_SMEM_EXCH1_STRIDE 128
#define SMEM_SMEM_SUM_OFF 142464
#define SMEM_SMEM_SUM_STAGE_BYTES 32
#define SMEM_SMEM_SUM_STRIDE 32
#define SMEM_SMEM_ALPHA_OFF 143168
#define SMEM_SMEM_ALPHA_STAGE_BYTES 32
#define SMEM_SMEM_ALPHA_STRIDE 32
#define SMEM_SMEM_INDICES_OFF 142528
#define SMEM_SMEM_INDICES_STAGE_BYTES 512
#define SMEM_SMEM_INDICES_STRIDE 512
#define SMEM_TOTAL 143232
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_c9156a2e9f65117a0900(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ partition_arrivals, int* __restrict__ swa_indices, int* __restrict__ compressed_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int swa_index_stride, int compressed_index_stride, int sparse_topk_lens_offset, int sparse_topk, int num_splits, int num_head_tiles, int has_sinks, int ragged_query, int max_q_len, int batch_size)
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
    #define kv_full_addr (mbar_base + 8)
    #define s_full_addr (mbar_base + 40)
    #define p_full_addr (mbar_base + 56)
    #define stats_full_addr (mbar_base + 72)
    #define o_full_lo_addr (mbar_base + 80)
    #define o_full_hi_addr (mbar_base + 96)
    #define tmem_dealloc_addr (mbar_base + 112)
    #define partial_done_addr (mbar_base + 120)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    __nv_bfloat16* smem_q_half = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q_HALF_OFF);
    const int smem_q_half_addr = smem + SMEM_SMEM_Q_HALF_OFF;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_KV_OFF);
    const int smem_kv_addr = smem + SMEM_SMEM_KV_OFF;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P_OFF);
    const int smem_p_addr = smem + SMEM_SMEM_P_OFF;
    __nv_bfloat16* smem_p_half = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P_HALF_OFF);
    const int smem_p_half_addr = smem + SMEM_SMEM_P_HALF_OFF;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_EXCH_OFF);
    const int smem_exch_addr = smem + SMEM_SMEM_EXCH_OFF;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_EXCH1_OFF);
    const int smem_exch1_addr = smem + SMEM_SMEM_EXCH1_OFF;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_SUM_OFF);
    const int smem_sum_addr = smem + SMEM_SMEM_SUM_OFF;
    float* smem_alpha = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_ALPHA_OFF);
    const int smem_alpha_addr = smem + SMEM_SMEM_ALPHA_OFF;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_INDICES_OFF);
    const int smem_indices_addr = smem + SMEM_SMEM_INDICES_OFF;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            mbarrier_init(smem + 64, 128);
            // stats_full: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // o_full_lo: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // o_full_hi: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tmem_dealloc: 1 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            // partial_done: 1 barriers, init_count=256
            mbarrier_init(smem + 120, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_s1 = taddr + 72;
    const int tmem_tmem_o0 = taddr + 8;
    const int tmem_tmem_o1 = taddr + 16;
    const int tmem_tmem_o2 = taddr + 24;
    const int tmem_tmem_o3 = taddr + 32;
    const int tmem_tmem_o0b = taddr + 40;
    const int tmem_tmem_o1b = taddr + 48;
    const int tmem_tmem_o2b = taddr + 56;
    const int tmem_tmem_o3b = taddr + 64;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int work_idx = blockIdx.x;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int _max_4 = ((sparse_topk_lens[query_idx] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx] + sparse_topk_lens_offset) : (0));
            int _min_8 = ((_max_4) < (sparse_topk) ? (_max_4) : (sparse_topk));
            int active_topk = _min_8;
            int head_base = head_tile * 8;
            const int key_lane = lane % 16;
            const int head_lo = lane / 16 * 4;
            const int half_pos = warp * 16 + key_lane;
            const int pos0 = half_pos;
            const int pos1 = 64 + half_pos;
            int w120_total_groups = (sparse_topk + 3) / 4;
            int w120_full_splits = w120_total_groups / 32;
            int w120_split32 = split_idx * 32;
            int w120_next32 = w120_split32 + 32;
            int _min_9 = ((w120_split32 / w120_full_splits) < (32) ? (w120_split32 / w120_full_splits) : (32));
            int w120_swa_lo = _min_9;
            int _min_10 = ((w120_next32 / w120_full_splits) < (32) ? (w120_next32 / w120_full_splits) : (32));
            int w120_swa_hi = _min_10;
            int w120_nswa = w120_swa_hi - w120_swa_lo;
            int w120_comp_lo = w120_split32 - w120_swa_lo;
            int w120_comp_left = w120_total_groups - 32 - w120_comp_lo;
            int w120_nswa_left = w120_swa_lo - w120_swa_hi + 32;
            int _min_11 = ((w120_nswa_left) < (w120_comp_left) ? (w120_nswa_left) : (w120_comp_left));
            int w120_ncomp = _min_11;
            int w120_h0_group = pos0 / 4;
            int w120_h0_row = pos0 % 4;
            int global_token0 = ((w120_h0_group < w120_nswa) ? (w120_swa_lo + w120_h0_group) * 4 + w120_h0_row : 128 + (w120_comp_lo + w120_h0_group - w120_nswa) * 4 + w120_h0_row);
            int w120_h1_group = pos1 / 4;
            int w120_h1_row = pos1 % 4;
            int global_token1 = ((w120_h1_group < w120_nswa) ? (w120_swa_lo + w120_h1_group) * 4 + w120_h1_row : 128 + (w120_comp_lo + w120_h1_group - w120_nswa) * 4 + w120_h1_row);
            float w29_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float w29_sinks[4];
            w29_sinks[0] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo < num_heads) {
                w29_sinks[0] = sinks[head_base + head_lo];
            }
            w29_sinks[1] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 1 < num_heads) {
                w29_sinks[1] = sinks[head_base + head_lo + 1];
            }
            w29_sinks[2] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 2 < num_heads) {
                w29_sinks[2] = sinks[head_base + head_lo + 2];
            }
            w29_sinks[3] = 0.0f;
            if (has_sinks != 0 && split_idx == 0 && head_base + head_lo + 3 < num_heads) {
                w29_sinks[3] = sinks[head_base + head_lo + 3];
            }
            int token_valid0 = 1;
            if (global_token0 >= active_topk) {
                token_valid0 = 0;
            }
            int token_valid1 = 1;
            if (global_token1 >= active_topk) {
                token_valid1 = 0;
            }
            int query_batch = query_idx / max_q_len;
            int query_offset = query_idx - query_batch * max_q_len;
            int query_length = max_q_len;
            if (ragged_query != 0) {
                query_batch = 0;
                #pragma unroll 2
                for (int chunk = 0; chunk < (batch_size + 31) / 32; chunk++) {
                    int lane_entry = chunk * 32 + lane + 1;
                    int _min_12 = ((lane_entry) < (batch_size) ? (lane_entry) : (batch_size));
                    int lane_load = _min_12;
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
            if (global_token0 < 128 && global_token0 >= swa_visible) {
                token_valid0 = 0;
            }
            if (global_token1 < 128 && global_token1 >= swa_visible) {
                token_valid1 = 0;
            }
            const int row_addr = warp * 32 << 16;
            unsigned int _phase_s_full_0 = 0;
            mbarrier_wait(s_full_addr, _phase_s_full_0);
            _phase_s_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
                " {%0, %1, %2, %3}, [%4], 4;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr + (unsigned int)row_addr));
            int staged_index0 = smem_indices[pos0];
            if (staged_index0 < 0) {
                token_valid0 = 0;
            }
            if (token_valid0 == 0) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    _tmem_load_0[j] = -CAKE_INF;
                }
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                if (head_base + head_lo + j_1 >= num_heads) {
                    _tmem_load_0[j_1] = -CAKE_INF;
                }
            }
            float partial_max0[4];
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_0[j_2], 8);
                float _max_5 = max_noftz(_tmem_load_0[j_2], _shfl_xor_0);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, _max_5, 4);
                float _max_6 = max_noftz(_max_5, _shfl_xor_1);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, _max_6, 2);
                float _max_7 = max_noftz(_max_6, _shfl_xor_2);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, _max_7, 1);
                float _max_8 = max_noftz(_max_7, _shfl_xor_3);
                partial_max0[j_2] = _max_8;
            }
            if (key_lane == 0) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    smem_exch[warp * 8 + head_lo + j_3] = partial_max0[j_3];
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            float exch_max0 = 0.0f;
            if (lane < 8) {
                int h_lane = lane;
                float _max_9 = max_noftz(smem_exch[h_lane], smem_exch[8 + h_lane]);
                float _max_10 = max_noftz(smem_exch[16 + h_lane], smem_exch[24 + h_lane]);
                float _max_11 = max_noftz(_max_9, _max_10);
                exch_max0 = _max_11;
            }
            float tile_max0[4];
            #pragma unroll
            for (int j_4 = 0; j_4 < 4; j_4++) {
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, exch_max0, head_lo + j_4);
                tile_max0[j_4] = _shfl_0;
            }
            float max0_values[4];
            float max0_scaled[4];
            float exp0[4];
            float warp_sum0[4];
            #pragma unroll
            for (int j_5 = 0; j_5 < 4; j_5++) {
                int global_head = head_base + head_lo + j_5;
                float row_max0 = tile_max0[j_5];
                if (has_sinks != 0 && split_idx == 0 && global_head < num_heads) {
                    float sink_unscaled = w29_sinks[j_5] * 1.4426950408889634f / w29_scale_log2;
                    float _max_12 = max_noftz(row_max0, sink_unscaled);
                    row_max0 = _max_12;
                }
                max0_values[j_5] = row_max0;
                float safe_max0 = ((row_max0 == -CAKE_INF) ? 0.0f : row_max0);
                max0_scaled[j_5] = safe_max0 * w29_scale_log2;
                float _exp2_0 = approx_exp2(_tmem_load_0[j_5] * w29_scale_log2 - max0_scaled[j_5]);
                exp0[j_5] = ((_tmem_load_0[j_5] != -CAKE_INF) ? _exp2_0 : 0.0f);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5], 8);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + _shfl_xor_4, 4);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + _shfl_xor_4 + _shfl_xor_5, 2);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, exp0[j_5] + _shfl_xor_4 + _shfl_xor_5 + _shfl_xor_6, 1);
                warp_sum0[j_5] = exp0[j_5] + _shfl_xor_4 + _shfl_xor_5 + _shfl_xor_6 + _shfl_xor_7;
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 4; j_6++) {
                {
                    __nv_bfloat16 _bval_0 = __float2bfloat16_rn(exp0[j_6]);
                    uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                    uint32_t _addr_0 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(half_pos / 64 * 1024 + (head_lo + j_6) * 128 + half_pos % 64 * 2 ^ (half_pos / 64 * 1024 + (head_lo + j_6) * 128 + half_pos % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                }
            }
            asm volatile("fence.proxy.async;");
            mbarrier_arrive(p_full_addr);
            unsigned int _phase_s_full_1 = 0;
            mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
            _phase_s_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x4.b32"
                " {%0, %1, %2, %3}, [%4], 4;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                : "r"(taddr + 72 + (unsigned int)row_addr));
            int staged_index1 = smem_indices[pos1];
            if (staged_index1 < 0) {
                token_valid1 = 0;
            }
            if (token_valid1 == 0) {
                #pragma unroll
                for (int j_7 = 0; j_7 < 4; j_7++) {
                    _tmem_load_1[j_7] = -CAKE_INF;
                }
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 4; j_8++) {
                if (head_base + head_lo + j_8 >= num_heads) {
                    _tmem_load_1[j_8] = -CAKE_INF;
                }
            }
            float partial_max1[4];
            #pragma unroll
            for (int j_9 = 0; j_9 < 4; j_9++) {
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, _tmem_load_1[j_9], 8);
                float _max_13 = max_noftz(_tmem_load_1[j_9], _shfl_xor_8);
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, _max_13, 4);
                float _max_14 = max_noftz(_max_13, _shfl_xor_9);
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, _max_14, 2);
                float _max_15 = max_noftz(_max_14, _shfl_xor_10);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, _max_15, 1);
                float _max_16 = max_noftz(_max_15, _shfl_xor_11);
                partial_max1[j_9] = _max_16;
            }
            if (key_lane == 0) {
                #pragma unroll
                for (int j_10 = 0; j_10 < 4; j_10++) {
                    smem_exch1[warp * 8 + head_lo + j_10] = partial_max1[j_10];
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            float exch_max1 = 0.0f;
            if (lane < 8) {
                int h_lane_1 = lane;
                float _max_17 = max_noftz(smem_exch1[h_lane_1], smem_exch1[8 + h_lane_1]);
                float _max_18 = max_noftz(smem_exch1[16 + h_lane_1], smem_exch1[24 + h_lane_1]);
                float _max_19 = max_noftz(_max_17, _max_18);
                exch_max1 = _max_19;
            }
            float tile_max1[4];
            #pragma unroll
            for (int j_11 = 0; j_11 < 4; j_11++) {
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, exch_max1, head_lo + j_11);
                tile_max1[j_11] = _shfl_1;
            }
            float row_max_values[4];
            float max_scaled[4];
            float alpha0[4];
            float exp1[4];
            float warp_sum[4];
            #pragma unroll
            for (int j_12 = 0; j_12 < 4; j_12++) {
                float _max_20 = max_noftz(max0_values[j_12], tile_max1[j_12]);
                float row_max = _max_20;
                row_max_values[j_12] = row_max;
                float safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                max_scaled[j_12] = safe_max * w29_scale_log2;
                float _exp2_1 = approx_exp2(max0_scaled[j_12] - max_scaled[j_12]);
                alpha0[j_12] = ((max0_values[j_12] == -CAKE_INF) ? 0.0f : _exp2_1);
                float _exp2_2 = approx_exp2(_tmem_load_1[j_12] * w29_scale_log2 - max_scaled[j_12]);
                exp1[j_12] = ((_tmem_load_1[j_12] != -CAKE_INF) ? _exp2_2 : 0.0f);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12], 8);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + _shfl_xor_12, 4);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + _shfl_xor_12 + _shfl_xor_13, 2);
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, exp1[j_12] + _shfl_xor_12 + _shfl_xor_13 + _shfl_xor_14, 1);
                float warp_sum1 = exp1[j_12] + _shfl_xor_12 + _shfl_xor_13 + _shfl_xor_14 + _shfl_xor_15;
                float _fma_0 = __fmaf_rn(warp_sum0[j_12], alpha0[j_12], warp_sum1);
                warp_sum[j_12] = _fma_0;
            }
            #pragma unroll
            for (int j_13 = 0; j_13 < 4; j_13++) {
                {
                    __nv_bfloat16 _bval_1 = __float2bfloat16_rn(exp1[j_13]);
                    uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                    uint32_t _addr_1 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(half_pos / 64 * 1024 + (8 + head_lo + j_13) * 128 + half_pos % 64 * 2 ^ (half_pos / 64 * 1024 + (8 + head_lo + j_13) * 128 + half_pos % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                }
            }
            asm volatile("fence.proxy.async;");
            mbarrier_arrive(p_full_addr + 8);
            if (key_lane == 0) {
                #pragma unroll
                for (int j_14 = 0; j_14 < 4; j_14++) {
                    smem_exch[warp * 8 + head_lo + j_14] = warp_sum[j_14];
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (warp == 0) {
                if (key_lane < 4) {
                    int h_lane_2 = head_lo + key_lane;
                    int global_head_1 = head_base + h_lane_2;
                    float row_sum = smem_exch[h_lane_2] + smem_exch[8 + h_lane_2] + smem_exch[16 + h_lane_2] + smem_exch[24 + h_lane_2];
                    float row_max_1 = row_max_values[key_lane];
                    if (has_sinks != 0 && split_idx == 0 && global_head_1 < num_heads) {
                        float sink_scaled = w29_sinks[key_lane] * 1.4426950408889634f;
                        float _exp2_3 = approx_exp2(sink_scaled - max_scaled[key_lane]);
                        row_sum = row_sum + _exp2_3;
                    }
                    smem_sum[h_lane_2] = row_sum;
                    smem_alpha[h_lane_2] = alpha0[key_lane];
                    if (global_head_1 < num_heads) {
                        int lse_offset = (query_idx * num_heads + global_head_1) * num_splits + split_idx;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                        partial_lse[lse_offset] = ((row_sum > 0.0f) ? row_max_1 * w29_scale_log2 + _log2_0 : -CAKE_INF);
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            mbarrier_arrive(stats_full_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_lo_main
            int work_idx_1 = blockIdx.x;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 8;
            const int local_warp = warp - 4;
            const int d_lane = local_warp * 32 + lane;
            const int row_addr_1 = local_warp * 32 << 16;
            float output_scale = bmm2_scale[0];
            int merge_group = query_idx_1 * num_head_tiles + head_tile_1;
            int _max_21 = ((sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_1] + sparse_topk_lens_offset) : (0));
            int _min_13 = ((_max_21) < (sparse_topk) ? (_max_21) : (sparse_topk));
            int w29_active = _min_13;
            int w120_total_groups_1 = (sparse_topk + 3) / 4;
            int w120_full_splits_1 = w120_total_groups_1 / 32;
            int w120_split32_1 = split_idx_1 * 32;
            int w120_next32_1 = w120_split32_1 + 32;
            int _min_14 = ((w120_split32_1 / w120_full_splits_1) < (32) ? (w120_split32_1 / w120_full_splits_1) : (32));
            int w120_swa_lo_1 = _min_14;
            int _min_15 = ((w120_next32_1 / w120_full_splits_1) < (32) ? (w120_next32_1 / w120_full_splits_1) : (32));
            int w120_swa_hi_1 = _min_15;
            int w120_nswa_1 = w120_swa_hi_1 - w120_swa_lo_1;
            int w120_comp_lo_1 = w120_split32_1 - w120_swa_lo_1;
            int w120_comp_left_1 = w120_total_groups_1 - 32 - w120_comp_lo_1;
            int w120_nswa_left_1 = w120_swa_lo_1 - w120_swa_hi_1 + 32;
            int _min_16 = ((w120_nswa_left_1) < (w120_comp_left_1) ? (w120_nswa_left_1) : (w120_comp_left_1));
            int w120_ncomp_1 = _min_16;
            int w120_active_group = w29_active / 4;
            int w120_active_rem = w29_active % 4;
            int w120_swa_off = w120_active_group - w120_swa_lo_1;
            int _max_22 = ((w120_swa_off) > (0) ? (w120_swa_off) : (0));
            int _min_17 = ((_max_22) < (w120_nswa_1) ? (_max_22) : (w120_nswa_1));
            int w120_full_swa = _min_17;
            int w120_comp_off = w120_active_group - (w120_comp_lo_1 + 32);
            int _max_23 = ((w120_comp_off) > (0) ? (w120_comp_off) : (0));
            int _min_18 = ((_max_23) < (w120_ncomp_1) ? (_max_23) : (w120_ncomp_1));
            int w120_full_comp = _min_18;
            int w120_full_rows = (w120_full_swa + w120_full_comp) * 4;
            int w29_valid_rows = w120_full_rows + ((w120_swa_off >= 0 && w120_swa_off < w120_nswa_1) ? w120_active_rem : ((w120_comp_off >= 0 && w120_comp_off < w120_ncomp_1) ? w120_active_rem : 0));
            int w29_valid_groups = (w29_valid_rows + 3) / 4;
            asm volatile("barrier.sync 8, 352;" ::: "memory");
            if (local_warp == 0) {
                int w29_u3_quad = lane % 16;
                int w29_u3_quad_pos = w29_u3_quad;
                int w29_u3_atom = 2 + lane / 16;
                if (w29_u3_quad_pos < w29_valid_groups) {
                    int w29_u3_raw0 = smem_indices[w29_u3_quad_pos * 4];
                    int w29_u3_row0 = ((w29_u3_raw0 >= 0) ? w29_u3_raw0 : 0);
                    int w29_u3_raw1 = smem_indices[w29_u3_quad_pos * 4 + 1];
                    int w29_u3_row1 = ((w29_u3_raw1 >= 0) ? w29_u3_raw1 : 0);
                    int w29_u3_raw2 = smem_indices[w29_u3_quad_pos * 4 + 2];
                    int w29_u3_row2 = ((w29_u3_raw2 >= 0) ? w29_u3_raw2 : 0);
                    int w29_u3_raw3 = smem_indices[w29_u3_quad_pos * 4 + 3];
                    int w29_u3_row3 = ((w29_u3_raw3 >= 0) ? w29_u3_raw3 : 0);
                    int w120_group_in_swa_w29_u3 = ((w29_u3_quad_pos < w120_nswa_1) ? 1 : 0);
                    if (w120_group_in_swa_w29_u3 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u3_atom * 8192) + (unsigned int)(w29_u3_quad * 512), (&tmap_swa_kv), 256 + w29_u3_atom * 64, w29_u3_row0, w29_u3_row1, w29_u3_row2, w29_u3_row3, kv_full_addr + 8);
                    }
                    if (w120_group_in_swa_w29_u3 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u3_atom * 8192) + (unsigned int)(w29_u3_quad * 512), (&tmap_compressed_kv), 256 + w29_u3_atom * 64, w29_u3_row0, w29_u3_row1, w29_u3_row2, w29_u3_row3, kv_full_addr + 8);
                    }
                }
            }
            if (local_warp == 1) {
                int w29_u4_quad = lane % 16;
                int w29_u4_quad_pos = 16 + w29_u4_quad;
                int w29_u4_atom = lane / 16;
                if (w29_u4_quad_pos < w29_valid_groups) {
                    int w29_u4_raw0 = smem_indices[w29_u4_quad_pos * 4];
                    int w29_u4_row0 = ((w29_u4_raw0 >= 0) ? w29_u4_raw0 : 0);
                    int w29_u4_raw1 = smem_indices[w29_u4_quad_pos * 4 + 1];
                    int w29_u4_row1 = ((w29_u4_raw1 >= 0) ? w29_u4_raw1 : 0);
                    int w29_u4_raw2 = smem_indices[w29_u4_quad_pos * 4 + 2];
                    int w29_u4_row2 = ((w29_u4_raw2 >= 0) ? w29_u4_raw2 : 0);
                    int w29_u4_raw3 = smem_indices[w29_u4_quad_pos * 4 + 3];
                    int w29_u4_row3 = ((w29_u4_raw3 >= 0) ? w29_u4_raw3 : 0);
                    int w120_group_in_swa_w29_u4 = ((w29_u4_quad_pos < w120_nswa_1) ? 1 : 0);
                    if (w120_group_in_swa_w29_u4 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom * 8192) + (unsigned int)(w29_u4_quad * 512), (&tmap_swa_kv), w29_u4_atom * 64, w29_u4_row0, w29_u4_row1, w29_u4_row2, w29_u4_row3, kv_full_addr + 16);
                    }
                    if (w120_group_in_swa_w29_u4 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u4_atom * 8192) + (unsigned int)(w29_u4_quad * 512), (&tmap_compressed_kv), w29_u4_atom * 64, w29_u4_row0, w29_u4_row1, w29_u4_row2, w29_u4_row3, kv_full_addr + 16);
                    }
                }
            }
            if (local_warp == 2) {
                int w29_u5_quad = lane % 16;
                int w29_u5_quad_pos = 16 + w29_u5_quad;
                int w29_u5_atom = 2 + lane / 16;
                if (w29_u5_quad_pos < w29_valid_groups) {
                    int w29_u5_raw0 = smem_indices[w29_u5_quad_pos * 4];
                    int w29_u5_row0 = ((w29_u5_raw0 >= 0) ? w29_u5_raw0 : 0);
                    int w29_u5_raw1 = smem_indices[w29_u5_quad_pos * 4 + 1];
                    int w29_u5_row1 = ((w29_u5_raw1 >= 0) ? w29_u5_raw1 : 0);
                    int w29_u5_raw2 = smem_indices[w29_u5_quad_pos * 4 + 2];
                    int w29_u5_row2 = ((w29_u5_raw2 >= 0) ? w29_u5_raw2 : 0);
                    int w29_u5_raw3 = smem_indices[w29_u5_quad_pos * 4 + 3];
                    int w29_u5_row3 = ((w29_u5_raw3 >= 0) ? w29_u5_raw3 : 0);
                    int w120_group_in_swa_w29_u5 = ((w29_u5_quad_pos < w120_nswa_1) ? 1 : 0);
                    if (w120_group_in_swa_w29_u5 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u5_atom * 8192) + (unsigned int)(w29_u5_quad * 512), (&tmap_swa_kv), w29_u5_atom * 64, w29_u5_row0, w29_u5_row1, w29_u5_row2, w29_u5_row3, kv_full_addr + 16);
                    }
                    if (w120_group_in_swa_w29_u5 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 65536 + (unsigned int)(w29_u5_atom * 8192) + (unsigned int)(w29_u5_quad * 512), (&tmap_compressed_kv), w29_u5_atom * 64, w29_u5_row0, w29_u5_row1, w29_u5_row2, w29_u5_row3, kv_full_addr + 16);
                    }
                }
            }
            if (local_warp == 3) {
                int w29_u6_quad = lane % 16;
                int w29_u6_quad_pos = 16 + w29_u6_quad;
                int w29_u6_atom = lane / 16;
                if (w29_u6_quad_pos < w29_valid_groups) {
                    int w29_u6_raw0 = smem_indices[w29_u6_quad_pos * 4];
                    int w29_u6_row0 = ((w29_u6_raw0 >= 0) ? w29_u6_raw0 : 0);
                    int w29_u6_raw1 = smem_indices[w29_u6_quad_pos * 4 + 1];
                    int w29_u6_row1 = ((w29_u6_raw1 >= 0) ? w29_u6_raw1 : 0);
                    int w29_u6_raw2 = smem_indices[w29_u6_quad_pos * 4 + 2];
                    int w29_u6_row2 = ((w29_u6_raw2 >= 0) ? w29_u6_raw2 : 0);
                    int w29_u6_raw3 = smem_indices[w29_u6_quad_pos * 4 + 3];
                    int w29_u6_row3 = ((w29_u6_raw3 >= 0) ? w29_u6_raw3 : 0);
                    int w120_group_in_swa_w29_u6 = ((w29_u6_quad_pos < w120_nswa_1) ? 1 : 0);
                    if (w120_group_in_swa_w29_u6 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom * 8192) + (unsigned int)(w29_u6_quad * 512), (&tmap_swa_kv), 256 + w29_u6_atom * 64, w29_u6_row0, w29_u6_row1, w29_u6_row2, w29_u6_row3, kv_full_addr + 24);
                    }
                    if (w120_group_in_swa_w29_u6 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u6_atom * 8192) + (unsigned int)(w29_u6_quad * 512), (&tmap_compressed_kv), 256 + w29_u6_atom * 64, w29_u6_row0, w29_u6_row1, w29_u6_row2, w29_u6_row3, kv_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_stats_full_0 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0);
            _phase_stats_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sums[8];
            float alpha_values[8];
            #pragma unroll
            for (int h = 0; h < 8; h++) {
                float row_sum_1 = smem_sum[h];
                float _rcp_0 = approx_rcp(row_sum_1);
                inv_sums[h] = ((row_sum_1 > 0.0f) ? _rcp_0 : 0.0f);
                alpha_values[h] = smem_alpha[h];
            }
            unsigned int _phase_o_full_lo_0 = 0;
            mbarrier_wait(o_full_lo_addr, _phase_o_full_lo_0);
            _phase_o_full_lo_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + 8 + (unsigned int)row_addr_1);
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + 40 + (unsigned int)row_addr_1);
            const int d_idx0 = d_lane;
            #pragma unroll
            for (int h_1 = 0; h_1 < 8; h_1++) {
                int global_head_2 = head_base_1 + h_1;
                if (global_head_2 < num_heads) {
                    float _fma_1 = __fmaf_rn(_tmem_load_2[h_1], alpha_values[h_1], _tmem_load_3[h_1]);
                    float merged0 = _fma_1;
                    int output_offset = ((query_idx_1 * num_heads + global_head_2) * num_splits + split_idx_1) * 512 + d_idx0;
                    partial_O[output_offset] = merged0 * inv_sums[h_1] * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_lo_1 = 0;
            mbarrier_wait(o_full_lo_addr + 8, _phase_o_full_lo_1);
            _phase_o_full_lo_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], taddr + 16 + (unsigned int)row_addr_1);
            float _tmem_load_5[8];
            tmem_ld_x8(&_tmem_load_5[0], taddr + 48 + (unsigned int)row_addr_1);
            const int d_idx1 = 128 + d_lane;
            #pragma unroll
            for (int h_2 = 0; h_2 < 8; h_2++) {
                int global_head_3 = head_base_1 + h_2;
                if (global_head_3 < num_heads) {
                    float _fma_2 = __fmaf_rn(_tmem_load_4[h_2], alpha_values[h_2], _tmem_load_5[h_2]);
                    float merged1 = _fma_2;
                    int output_offset_1 = ((query_idx_1 * num_heads + global_head_3) * num_splits + split_idx_1) * 512 + d_idx1;
                    partial_O[output_offset_1] = merged1 * inv_sums[h_2] * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_partial_done_0 = 0;
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
                mbarrier_wait(partial_done_addr, _phase_partial_done_0);
                _phase_partial_done_0 ^= 1;
                if (warp == 4) {
                    if (elect_sync()) {
                        unsigned int _atomic_old_0;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_0) : "l"(&partition_arrivals[merge_group]), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int prior_arrivals = _atomic_old_0;
                        smem_exch[0] = __uint_as_float(prior_arrivals);
                    }
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float flag_bits = smem_exch[0];
                int prior_count = __as_u32(flag_bits);
                int last_split = num_splits - 1;
                if (prior_count == last_split) {
                    if (warp == 4) {
                        if (elect_sync()) {
                            {
                                unsigned int* _gcr_p = reinterpret_cast<unsigned int*>(partition_arrivals) + (merge_group);
                                asm volatile("st.release.gpu.global.u32 [%0], %1;" : : "l"(_gcr_p), "r"(0u) : "memory");
                            }
                        }
                    }
                    #pragma unroll
                    for (int local_h = 0; local_h < 2; local_h++) {
                        int global_head_4 = head_base_1 + local_warp * 2 + local_h;
                        if (global_head_4 < num_heads) {
                            int lse_base = (query_idx_1 * num_heads + global_head_4) * num_splits;
                            int partial_base = lse_base * 512;
                            int lane_d = lane * 8;
                            int _min_19 = ((1) < (last_split) ? (1) : (last_split));
                            int split_1 = _min_19;
                            int _min_20 = ((2) < (last_split) ? (2) : (last_split));
                            int split_2 = _min_20;
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (partial_base + lane_d) + 0);
                                uint4 _vld_0[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_0[_blk] = _vptr_0[_blk];
                                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_1[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_1 * 512 + lane_d) + 0);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_1[_blk] = _vptr_1[_blk];
                                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_2[8];
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_2 * 512 + lane_d) + 0);
                                uint4 _vld_2[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_3[8];
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O + (partial_base + 256 + lane_d) + 0);
                                uint4 _vld_3[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_4[8];
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_1 * 512 + 256 + lane_d) + 0);
                                uint4 _vld_4[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float _vec_load_5[8];
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split_2 * 512 + 256 + lane_d) + 0);
                                uint4 _vld_5[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_5[_blk] = _vptr_5[_blk];
                                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                            float local_lse = -CAKE_INF;
                            if (lane < num_splits) {
                                local_lse = partial_lse[lse_base + lane];
                            }
                            float _warp_reduce_0 = local_lse;
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                            float global_max = _warp_reduce_0;
                            float local_weight = 0.0f;
                            if (lane < num_splits) {
                                float _exp2_4 = approx_exp2(local_lse - global_max);
                                local_weight = ((local_lse == -CAKE_INF) ? 0.0f : _exp2_4);
                            }
                            float _warp_reduce_1 = local_weight;
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                            float global_sum = _warp_reduce_1;
                            float _rcp_1 = approx_rcp(global_sum);
                            float normalized_weight = ((global_sum > 0.0f) ? local_weight * _rcp_1 : 0.0f);
                            float _shfl_2 = __shfl_sync(0xFFFFFFFF, normalized_weight, 0);
                            float weight_0 = _shfl_2;
                            float _shfl_3 = __shfl_sync(0xFFFFFFFF, normalized_weight, 1);
                            float weight_1 = _shfl_3;
                            float _shfl_4 = __shfl_sync(0xFFFFFFFF, normalized_weight, 2);
                            float weight_2 = _shfl_4;
                            float merged_0[8];
                            float merged_1[8];
                            #pragma unroll
                            for (int elem = 0; elem < 8; elem++) {
                                merged_0[elem] = weight_0 * _vec_load_0[elem] + weight_1 * _vec_load_1[elem] + weight_2 * _vec_load_2[elem];
                                merged_1[elem] = weight_0 * _vec_load_3[elem] + weight_1 * _vec_load_4[elem] + weight_2 * _vec_load_5[elem];
                            }
                            int output_base = (query_idx_1 * num_heads + global_head_4) * 512 + lane_d;
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(merged_0[0 + 0], merged_0[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(merged_0[0 + 2], merged_0[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(merged_0[0 + 4], merged_0[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(merged_0[0 + 6], merged_0[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + output_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(merged_1[0 + 0], merged_1[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(merged_1[0 + 2], merged_1[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(merged_1[0 + 4], merged_1[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(merged_1[0 + 6], merged_1[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base + 256)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_hi_main
            int work_idx_2 = blockIdx.x;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 8;
            const int local_warp_1 = warp - 8;
            const int d_lane_1 = local_warp_1 * 32 + lane;
            const int row_addr_2 = local_warp_1 * 32 << 16;
            float output_scale_1 = bmm2_scale[0];
            int _max_24 = ((sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_2] + sparse_topk_lens_offset) : (0));
            int _min_21 = ((_max_24) < (sparse_topk) ? (_max_24) : (sparse_topk));
            int w29_active_1 = _min_21;
            int w120_total_groups_2 = (sparse_topk + 3) / 4;
            int w120_full_splits_2 = w120_total_groups_2 / 32;
            int w120_split32_2 = split_idx_2 * 32;
            int w120_next32_2 = w120_split32_2 + 32;
            int _min_22 = ((w120_split32_2 / w120_full_splits_2) < (32) ? (w120_split32_2 / w120_full_splits_2) : (32));
            int w120_swa_lo_2 = _min_22;
            int _min_23 = ((w120_next32_2 / w120_full_splits_2) < (32) ? (w120_next32_2 / w120_full_splits_2) : (32));
            int w120_swa_hi_2 = _min_23;
            int w120_nswa_2 = w120_swa_hi_2 - w120_swa_lo_2;
            int w120_comp_lo_2 = w120_split32_2 - w120_swa_lo_2;
            int w120_comp_left_2 = w120_total_groups_2 - 32 - w120_comp_lo_2;
            int w120_nswa_left_2 = w120_swa_lo_2 - w120_swa_hi_2 + 32;
            int _min_24 = ((w120_nswa_left_2) < (w120_comp_left_2) ? (w120_nswa_left_2) : (w120_comp_left_2));
            int w120_ncomp_2 = _min_24;
            int w120_active_group_1 = w29_active_1 / 4;
            int w120_active_rem_1 = w29_active_1 % 4;
            int w120_swa_off_1 = w120_active_group_1 - w120_swa_lo_2;
            int _max_25 = ((w120_swa_off_1) > (0) ? (w120_swa_off_1) : (0));
            int _min_25 = ((_max_25) < (w120_nswa_2) ? (_max_25) : (w120_nswa_2));
            int w120_full_swa_1 = _min_25;
            int w120_comp_off_1 = w120_active_group_1 - (w120_comp_lo_2 + 32);
            int _max_26 = ((w120_comp_off_1) > (0) ? (w120_comp_off_1) : (0));
            int _min_26 = ((_max_26) < (w120_ncomp_2) ? (_max_26) : (w120_ncomp_2));
            int w120_full_comp_1 = _min_26;
            int w120_full_rows_1 = (w120_full_swa_1 + w120_full_comp_1) * 4;
            int w29_valid_rows_1 = w120_full_rows_1 + ((w120_swa_off_1 >= 0 && w120_swa_off_1 < w120_nswa_2) ? w120_active_rem_1 : ((w120_comp_off_1 >= 0 && w120_comp_off_1 < w120_ncomp_2) ? w120_active_rem_1 : 0));
            int w29_valid_groups_1 = (w29_valid_rows_1 + 3) / 4;
            asm volatile("barrier.sync 8, 352;" ::: "memory");
            if (local_warp_1 == 0) {
                int w29_u7_quad = lane % 16;
                int w29_u7_quad_pos = 16 + w29_u7_quad;
                int w29_u7_atom = 2 + lane / 16;
                if (w29_u7_quad_pos < w29_valid_groups_1) {
                    int w29_u7_raw0 = smem_indices[w29_u7_quad_pos * 4];
                    int w29_u7_row0 = ((w29_u7_raw0 >= 0) ? w29_u7_raw0 : 0);
                    int w29_u7_raw1 = smem_indices[w29_u7_quad_pos * 4 + 1];
                    int w29_u7_row1 = ((w29_u7_raw1 >= 0) ? w29_u7_raw1 : 0);
                    int w29_u7_raw2 = smem_indices[w29_u7_quad_pos * 4 + 2];
                    int w29_u7_row2 = ((w29_u7_raw2 >= 0) ? w29_u7_raw2 : 0);
                    int w29_u7_raw3 = smem_indices[w29_u7_quad_pos * 4 + 3];
                    int w29_u7_row3 = ((w29_u7_raw3 >= 0) ? w29_u7_raw3 : 0);
                    int w120_group_in_swa_w29_u7 = ((w29_u7_quad_pos < w120_nswa_2) ? 1 : 0);
                    if (w120_group_in_swa_w29_u7 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u7_atom * 8192) + (unsigned int)(w29_u7_quad * 512), (&tmap_swa_kv), 256 + w29_u7_atom * 64, w29_u7_row0, w29_u7_row1, w29_u7_row2, w29_u7_row3, kv_full_addr + 24);
                    }
                    if (w120_group_in_swa_w29_u7 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 98304 + (unsigned int)(w29_u7_atom * 8192) + (unsigned int)(w29_u7_quad * 512), (&tmap_compressed_kv), 256 + w29_u7_atom * 64, w29_u7_row0, w29_u7_row1, w29_u7_row2, w29_u7_row3, kv_full_addr + 24);
                    }
                }
            }
            unsigned int _phase_stats_full_0_1 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0_1);
            _phase_stats_full_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sums_1[8];
            float alpha_values_1[8];
            #pragma unroll
            for (int h_3 = 0; h_3 < 8; h_3++) {
                float row_sum_2 = smem_sum[h_3];
                float _rcp_2 = approx_rcp(row_sum_2);
                inv_sums_1[h_3] = ((row_sum_2 > 0.0f) ? _rcp_2 : 0.0f);
                alpha_values_1[h_3] = smem_alpha[h_3];
            }
            unsigned int _phase_o_full_hi_0 = 0;
            mbarrier_wait(o_full_hi_addr, _phase_o_full_hi_0);
            _phase_o_full_hi_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_6[8];
            tmem_ld_x8(&_tmem_load_6[0], taddr + 24 + (unsigned int)row_addr_2);
            float _tmem_load_7[8];
            tmem_ld_x8(&_tmem_load_7[0], taddr + 56 + (unsigned int)row_addr_2);
            const int d_idx2 = 256 + d_lane_1;
            #pragma unroll
            for (int h_4 = 0; h_4 < 8; h_4++) {
                int global_head_5 = head_base_2 + h_4;
                if (global_head_5 < num_heads) {
                    float _fma_3 = __fmaf_rn(_tmem_load_6[h_4], alpha_values_1[h_4], _tmem_load_7[h_4]);
                    float merged2 = _fma_3;
                    int output_offset_2 = ((query_idx_2 * num_heads + global_head_5) * num_splits + split_idx_2) * 512 + d_idx2;
                    partial_O[output_offset_2] = merged2 * inv_sums_1[h_4] * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_hi_1 = 0;
            mbarrier_wait(o_full_hi_addr + 8, _phase_o_full_hi_1);
            _phase_o_full_hi_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_8[8];
            tmem_ld_x8(&_tmem_load_8[0], taddr + 32 + (unsigned int)row_addr_2);
            float _tmem_load_9[8];
            tmem_ld_x8(&_tmem_load_9[0], taddr + 64 + (unsigned int)row_addr_2);
            const int d_idx3 = 384 + d_lane_1;
            #pragma unroll
            for (int h_5 = 0; h_5 < 8; h_5++) {
                int global_head_6 = head_base_2 + h_5;
                if (global_head_6 < num_heads) {
                    float _fma_4 = __fmaf_rn(_tmem_load_8[h_5], alpha_values_1[h_5], _tmem_load_9[h_5]);
                    float merged3 = _fma_4;
                    int output_offset_3 = ((query_idx_2 * num_heads + global_head_6) * num_splits + split_idx_2) * 512 + d_idx3;
                    partial_O[output_offset_3] = merged3 * inv_sums_1[h_5] * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_kv_full_0 = 0;
            mbarrier_wait(kv_full_addr, _phase_kv_full_0);
            _phase_kv_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
            int _mma_b_lo_0 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (0) * 256);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67241104, 1);
                }
            }
            unsigned int _phase_kv_full_1 = 0;
            mbarrier_wait(kv_full_addr + 8, _phase_kv_full_1);
            _phase_kv_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
            int _mma_b_lo_1 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (1) * 256);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67241104, 1);
                }
            }
            elect_commit(s_full_addr);
            unsigned int _phase_kv_full_2 = 0;
            mbarrier_wait(kv_full_addr + 16, _phase_kv_full_2);
            _phase_kv_full_2 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (2) * 2048);
            int _mma_b_lo_2 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (0) * 256);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 67241104, 1);
                }
            }
            unsigned int _phase_kv_full_3 = 0;
            mbarrier_wait(kv_full_addr + 24, _phase_kv_full_3);
            _phase_kv_full_3 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (3) * 2048);
            int _mma_b_lo_3 = make_warp_uniform((((smem_q_half_addr) >> 4) & 0x3FFF) + (1) * 256);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 67241104, 1);
                }
            }
            elect_commit(s_full_addr + 8);
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr, 0);
            int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 2048);
            int _mma_b_lo_4 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
            }
            int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (0) * 2048);
            int _mma_b_lo_5 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                }
            }
            mbarrier_wait(kv_full_addr + 8, 0);
            int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 2048);
            int _mma_b_lo_6 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                }
            }
            int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (1) * 2048);
            int _mma_b_lo_7 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (0) * 64);
            {
                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                }
            }
            unsigned int _phase_p_full_1 = 0;
            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
            _phase_p_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr + 16, 0);
            int _mma_a_lo_8 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 2048);
            int _mma_b_lo_8 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0b, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr);
            int _mma_a_lo_9 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (2) * 2048);
            int _mma_b_lo_9 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_9);
                uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_9);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1b, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr + 8);
            mbarrier_wait(kv_full_addr + 24, 0);
            int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 2048);
            int _mma_b_lo_10 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_10);
                uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_10);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_10, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2b, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr);
            int _mma_a_lo_11 = make_warp_uniform(((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x2000000) + (3) * 2048);
            int _mma_b_lo_11 = make_warp_uniform(((((smem_p_half_addr) >> 4) & 0x3FFF) | 0x400000) + (1) * 64);
            {
                uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_11);
                uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_11);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_11, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3b, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr + 8);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_rank = warp - 13;
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int split_idx_3 = split_work_3 % num_splits;
            int query_idx_3 = split_work_3 / num_splits;
            int w120_total_groups_3 = (sparse_topk + 3) / 4;
            int w120_full_splits_3 = w120_total_groups_3 / 32;
            int w120_split32_3 = split_idx_3 * 32;
            int w120_next32_3 = w120_split32_3 + 32;
            int _min_0 = ((w120_split32_3 / w120_full_splits_3) < (32) ? (w120_split32_3 / w120_full_splits_3) : (32));
            int w120_swa_lo_3 = _min_0;
            int _min_1 = ((w120_next32_3 / w120_full_splits_3) < (32) ? (w120_next32_3 / w120_full_splits_3) : (32));
            int w120_swa_hi_3 = _min_1;
            int w120_nswa_3 = w120_swa_hi_3 - w120_swa_lo_3;
            int w120_comp_lo_3 = w120_split32_3 - w120_swa_lo_3;
            int w120_comp_left_3 = w120_total_groups_3 - 32 - w120_comp_lo_3;
            int w120_nswa_left_3 = w120_swa_lo_3 - w120_swa_hi_3 + 32;
            int _min_2 = ((w120_nswa_left_3) < (w120_comp_left_3) ? (w120_nswa_left_3) : (w120_comp_left_3));
            int w120_ncomp_3 = _min_2;
            int lane_in_swa = ((w120_nswa_3 > lane) ? 1 : 0);
            int lane_col_base = ((lane_in_swa == 1) ? (w120_swa_lo_3 + lane) * 4 : 128 + (w120_comp_lo_3 + lane - w120_nswa_3) * 4);
            int* row_ptr = ((lane_in_swa == 1) ? (swa_indices + (query_idx_3 * swa_index_stride + lane_col_base)) : (compressed_indices + (query_idx_3 * compressed_index_stride + (lane_col_base - 128))));
            int vector_offset = lane * 4;
            if (load_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 8192);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 4; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 2048), (&tmap_q), 0, head_tile_3 * 8, q_stage * 2, query_idx_3, q_full_addr);
                    }
                }
                int row_stride = ((lane_in_swa == 1) ? swa_index_stride : compressed_index_stride);
                int row_vec_misaligned = query_idx_3 * row_stride & 3;
                if (lane_col_base + 3 < sparse_topk && row_vec_misaligned == 0) {
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                        :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "l"(row_ptr));
                } else {
                    int staged_rows[4];
                    #pragma unroll
                    for (int row_i = 0; row_i < 4; row_i++) {
                        staged_rows[row_i] = -1;
                        if (lane_col_base + row_i < sparse_topk) {
                            staged_rows[row_i] = row_ptr[row_i];
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "r"(staged_rows[0]), "r"(staged_rows[1]), "r"(staged_rows[2]), "r"(staged_rows[3]) : "memory");
                }
                asm volatile("cp.async.commit_group;");
            }
            int _max_0 = ((sparse_topk_lens[query_idx_3] + sparse_topk_lens_offset) > (0) ? (sparse_topk_lens[query_idx_3] + sparse_topk_lens_offset) : (0));
            int _min_3 = ((_max_0) < (sparse_topk) ? (_max_0) : (sparse_topk));
            int active_topk_1 = _min_3;
            int w120_active_group_2 = active_topk_1 / 4;
            int w120_active_rem_2 = active_topk_1 % 4;
            int w120_swa_off_2 = w120_active_group_2 - w120_swa_lo_3;
            int _max_1 = ((w120_swa_off_2) > (0) ? (w120_swa_off_2) : (0));
            int _min_4 = ((_max_1) < (w120_nswa_3) ? (_max_1) : (w120_nswa_3));
            int w120_full_swa_2 = _min_4;
            int w120_comp_off_2 = w120_active_group_2 - (w120_comp_lo_3 + 32);
            int _max_2 = ((w120_comp_off_2) > (0) ? (w120_comp_off_2) : (0));
            int _min_5 = ((_max_2) < (w120_ncomp_3) ? (_max_2) : (w120_ncomp_3));
            int w120_full_comp_2 = _min_5;
            int w120_full_rows_2 = (w120_full_swa_2 + w120_full_comp_2) * 4;
            int valid_rows = w120_full_rows_2 + ((w120_swa_off_2 >= 0 && w120_swa_off_2 < w120_nswa_3) ? w120_active_rem_2 : ((w120_comp_off_2 >= 0 && w120_comp_off_2 < w120_ncomp_3) ? w120_active_rem_2 : 0));
            int valid_groups = (valid_rows + 3) / 4;
            int _min_6 = ((valid_groups) < (16) ? (valid_groups) : (16));
            int live_groups0 = _min_6;
            int _max_3 = ((valid_groups - 16) > (0) ? (valid_groups - 16) : (0));
            int _min_7 = ((_max_3) < (16) ? (_max_3) : (16));
            int live_groups1 = _min_7;
            int loader_thread = load_rank * 32 + lane;
            int dead_vecs0 = (16 - live_groups0) * 128;
            int dead_vecs1 = (16 - live_groups1) * 128;
            for (int tail_vec = loader_thread; tail_vec < dead_vecs0; tail_vec += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(tail_vec % 128 / 32 * 8192 + (live_groups0 + tail_vec / 128) * 512 + tail_vec % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_1 = loader_thread; tail_vec_1 < dead_vecs0; tail_vec_1 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(32768 + tail_vec_1 % 128 / 32 * 8192 + (live_groups0 + tail_vec_1 / 128) * 512 + tail_vec_1 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_2 = loader_thread; tail_vec_2 < dead_vecs1; tail_vec_2 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(65536 + tail_vec_2 % 128 / 32 * 8192 + (live_groups1 + tail_vec_2 / 128) * 512 + tail_vec_2 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            for (int tail_vec_3 = loader_thread; tail_vec_3 < dead_vecs1; tail_vec_3 += 96) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(98304 + tail_vec_3 % 128 / 32 * 8192 + (live_groups1 + tail_vec_3 / 128) * 512 + tail_vec_3 % 32 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
            }
            if (load_rank == 0) {
                asm volatile("cp.async.wait_group 0;");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 352;" ::: "memory");
            int gather_tx0 = live_groups0 * 4 * 512;
            int gather_tx1 = live_groups1 * 4 * 512;
            if (0 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr, gather_tx0);
                }
            }
            if (1 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 8, gather_tx0);
                }
            }
            if (2 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 16, gather_tx1);
                }
            }
            if (0 == load_rank) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + 24, gather_tx1);
                }
            }
            if (load_rank == 0) {
                int w29_u0_quad = lane % 16;
                int w29_u0_quad_pos = w29_u0_quad;
                int w29_u0_atom = lane / 16;
                if (w29_u0_quad_pos < valid_groups) {
                    int w29_u0_raw0 = smem_indices[w29_u0_quad_pos * 4];
                    int w29_u0_row0 = ((w29_u0_raw0 >= 0) ? w29_u0_raw0 : 0);
                    int w29_u0_raw1 = smem_indices[w29_u0_quad_pos * 4 + 1];
                    int w29_u0_row1 = ((w29_u0_raw1 >= 0) ? w29_u0_raw1 : 0);
                    int w29_u0_raw2 = smem_indices[w29_u0_quad_pos * 4 + 2];
                    int w29_u0_row2 = ((w29_u0_raw2 >= 0) ? w29_u0_raw2 : 0);
                    int w29_u0_raw3 = smem_indices[w29_u0_quad_pos * 4 + 3];
                    int w29_u0_row3 = ((w29_u0_raw3 >= 0) ? w29_u0_raw3 : 0);
                    int w120_group_in_swa_w29_u0 = ((w29_u0_quad_pos < w120_nswa_3) ? 1 : 0);
                    if (w120_group_in_swa_w29_u0 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom * 8192) + (unsigned int)(w29_u0_quad * 512), (&tmap_swa_kv), w29_u0_atom * 64, w29_u0_row0, w29_u0_row1, w29_u0_row2, w29_u0_row3, kv_full_addr);
                    }
                    if (w120_group_in_swa_w29_u0 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u0_atom * 8192) + (unsigned int)(w29_u0_quad * 512), (&tmap_compressed_kv), w29_u0_atom * 64, w29_u0_row0, w29_u0_row1, w29_u0_row2, w29_u0_row3, kv_full_addr);
                    }
                }
            }
            if (load_rank == 1) {
                int w29_u1_quad = lane % 16;
                int w29_u1_quad_pos = w29_u1_quad;
                int w29_u1_atom = 2 + lane / 16;
                if (w29_u1_quad_pos < valid_groups) {
                    int w29_u1_raw0 = smem_indices[w29_u1_quad_pos * 4];
                    int w29_u1_row0 = ((w29_u1_raw0 >= 0) ? w29_u1_raw0 : 0);
                    int w29_u1_raw1 = smem_indices[w29_u1_quad_pos * 4 + 1];
                    int w29_u1_row1 = ((w29_u1_raw1 >= 0) ? w29_u1_raw1 : 0);
                    int w29_u1_raw2 = smem_indices[w29_u1_quad_pos * 4 + 2];
                    int w29_u1_row2 = ((w29_u1_raw2 >= 0) ? w29_u1_raw2 : 0);
                    int w29_u1_raw3 = smem_indices[w29_u1_quad_pos * 4 + 3];
                    int w29_u1_row3 = ((w29_u1_raw3 >= 0) ? w29_u1_raw3 : 0);
                    int w120_group_in_swa_w29_u1 = ((w29_u1_quad_pos < w120_nswa_3) ? 1 : 0);
                    if (w120_group_in_swa_w29_u1 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u1_atom * 8192) + (unsigned int)(w29_u1_quad * 512), (&tmap_swa_kv), w29_u1_atom * 64, w29_u1_row0, w29_u1_row1, w29_u1_row2, w29_u1_row3, kv_full_addr);
                    }
                    if (w120_group_in_swa_w29_u1 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + (unsigned int)(w29_u1_atom * 8192) + (unsigned int)(w29_u1_quad * 512), (&tmap_compressed_kv), w29_u1_atom * 64, w29_u1_row0, w29_u1_row1, w29_u1_row2, w29_u1_row3, kv_full_addr);
                    }
                }
            }
            if (load_rank == 2) {
                int w29_u2_quad = lane % 16;
                int w29_u2_quad_pos = w29_u2_quad;
                int w29_u2_atom = lane / 16;
                if (w29_u2_quad_pos < valid_groups) {
                    int w29_u2_raw0 = smem_indices[w29_u2_quad_pos * 4];
                    int w29_u2_row0 = ((w29_u2_raw0 >= 0) ? w29_u2_raw0 : 0);
                    int w29_u2_raw1 = smem_indices[w29_u2_quad_pos * 4 + 1];
                    int w29_u2_row1 = ((w29_u2_raw1 >= 0) ? w29_u2_raw1 : 0);
                    int w29_u2_raw2 = smem_indices[w29_u2_quad_pos * 4 + 2];
                    int w29_u2_row2 = ((w29_u2_raw2 >= 0) ? w29_u2_raw2 : 0);
                    int w29_u2_raw3 = smem_indices[w29_u2_quad_pos * 4 + 3];
                    int w29_u2_row3 = ((w29_u2_raw3 >= 0) ? w29_u2_raw3 : 0);
                    int w120_group_in_swa_w29_u2 = ((w29_u2_quad_pos < w120_nswa_3) ? 1 : 0);
                    if (w120_group_in_swa_w29_u2 == 1) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom * 8192) + (unsigned int)(w29_u2_quad * 512), (&tmap_swa_kv), 256 + w29_u2_atom * 64, w29_u2_row0, w29_u2_row1, w29_u2_row2, w29_u2_row3, kv_full_addr + 8);
                    }
                    if (w120_group_in_swa_w29_u2 == 0) {
                        tma_gather4_gmem2smem(smem_kv_addr + 32768 + (unsigned int)(w29_u2_atom * 8192) + (unsigned int)(w29_u2_quad * 512), (&tmap_compressed_kv), 256 + w29_u2_atom * 64, w29_u2_row0, w29_u2_row1, w29_u2_row2, w29_u2_row3, kv_full_addr + 8);
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
