/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_dense_mqa_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_KV_OFF 13312
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_KV_SCALES_OFF 177152
#define SMEM_SMEM_KV_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_KV_SCALES_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 182272
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 128
#define SMEM_SMEM_WEIGHTS_STRIDE 128
#define SMEM_SCHED_PREFIX_OFF 13312
#define SMEM_SCHED_PREFIX_STAGE_BYTES 16384
#define SMEM_SCHED_PREFIX_STRIDE 16384
#define SMEM_SCHED_WARP_SUMS_OFF 29696
#define SMEM_SCHED_WARP_SUMS_STAGE_BYTES 128
#define SMEM_SCHED_WARP_SUMS_STRIDE 128
#define SMEM_SCHED_BOUNDS_OFF 182656
#define SMEM_SCHED_BOUNDS_STAGE_BYTES 16
#define SMEM_SCHED_BOUNDS_STRIDE 16
#define SMEM_TOTAL 182784
#define K_NEXT_N 1
#define K_NEXT_N_ATOM 1
#define K_NUM_NEXT_N_ATOMS 1
#define K_CHUNK_SPLITS 1048576
#define K_NUM_HEADS 32
#define K_UMMA_N 32
#define K_Q_TILE_ROWS 32
#define K_Q_TX_BYTES 4224
#define K_BLOCK_KV 64
#define K_NUM_BLOCKS_PER_SPLIT 4
#define K_SCALE_TAIL_F32 2048
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_edff6f9d8310efe62779(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int block_table_stride, int batch_size, int num_sms, int stride_logits, unsigned int* __restrict__ sched_counters)
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
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 88)
    #define umma_full_addr (mbar_base + 128)
    #define umma_empty_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_OFF);
    const int smem_kv_addr = smem + SMEM_SMEM_KV_OFF;
    float* smem_kv_scales = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_KV_SCALES_OFF);
    const int smem_kv_scales_addr = smem + SMEM_SMEM_KV_SCALES_OFF;
    float* smem_weights = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_WEIGHTS_OFF);
    const int smem_weights_addr = smem + SMEM_SMEM_WEIGHTS_OFF;
    int* sched_prefix = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_PREFIX_OFF);
    const int sched_prefix_addr = smem + SMEM_SCHED_PREFIX_OFF;
    int* sched_warp_sums = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_WARP_SUMS_OFF);
    const int sched_warp_sums_addr = smem + SMEM_SCHED_WARP_SUMS_OFF;
    int* sched_bounds = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_BOUNDS_OFF);
    const int sched_bounds_addr = smem + SMEM_SCHED_BOUNDS_OFF;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV_scales))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 24, 288);
            mbarrier_init(smem + 32, 288);
            mbarrier_init(smem + 40, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 5 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 10) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int math_thread_idx = wg_idx * 128 + (unsigned int)(warp % 4 * 32 + lane);
            unsigned int mt_q_stage = 0;
            unsigned int mt_kv_stage = 0;
            unsigned int mt_tmem_stage = wg_idx;
            int cp_lane = lane;
            int cp_state[7];
            int cp_run = 0;
            int cp_q = 0;
            int cp_s = 0;
            int cp_ctx = 0;
            cp_q = cp_lane * 5;
            cp_s = 0;
            if (cp_q < batch_size) {
                cp_ctx = context_lens[cp_q * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s = (cp_ctx + 256 - 1) / 256;
            }
            cp_run += cp_s;
            cp_state[0] = cp_run;
            cp_q = cp_lane * 5 + 1;
            cp_s = 0;
            if (cp_q < batch_size) {
                cp_ctx = context_lens[cp_q * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s = (cp_ctx + 256 - 1) / 256;
            }
            cp_run += cp_s;
            cp_state[1] = cp_run;
            cp_q = cp_lane * 5 + 2;
            cp_s = 0;
            if (cp_q < batch_size) {
                cp_ctx = context_lens[cp_q * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s = (cp_ctx + 256 - 1) / 256;
            }
            cp_run += cp_s;
            cp_state[2] = cp_run;
            cp_q = cp_lane * 5 + 3;
            cp_s = 0;
            if (cp_q < batch_size) {
                cp_ctx = context_lens[cp_q * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s = (cp_ctx + 256 - 1) / 256;
            }
            cp_run += cp_s;
            cp_state[3] = cp_run;
            cp_q = cp_lane * 5 + 4;
            cp_s = 0;
            if (cp_q < batch_size) {
                cp_ctx = context_lens[cp_q * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s = (cp_ctx + 256 - 1) / 256;
            }
            cp_run += cp_s;
            cp_state[4] = cp_run;
            int cp_incl = cp_run;
            int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, cp_incl, 1, 32);
            int cp_up = _shfl_up_15;
            if (cp_lane >= 1) {
                cp_incl += cp_up;
            }
            int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, cp_incl, 2, 32);
            int cp_up_0 = _shfl_up_16;
            if (cp_lane >= 2) {
                cp_incl += cp_up_0;
            }
            int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, cp_incl, 4, 32);
            int cp_up_1 = _shfl_up_17;
            if (cp_lane >= 4) {
                cp_incl += cp_up_1;
            }
            int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, cp_incl, 8, 32);
            int cp_up_2 = _shfl_up_18;
            if (cp_lane >= 8) {
                cp_incl += cp_up_2;
            }
            int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, cp_incl, 16, 32);
            int cp_up_3 = _shfl_up_19;
            if (cp_lane >= 16) {
                cp_incl += cp_up_3;
            }
            int _shfl_22 = __shfl_sync(0xFFFFFFFF, cp_incl, 31);
            int cp_total = _shfl_22;
            cp_state[5] = cp_incl;
            cp_state[6] = cp_incl - cp_run;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (int u = bid; u < cp_total; u += num_bids) {
                int cu_u = u;
                int cu_incl = cp_state[5];
                int cu_excl = cp_state[6];
                unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, cu_incl > cu_u);
                unsigned int cu_hit = _vote_3;
                int _ffs_3 = __ffs(cu_hit);
                int cu_lane = _ffs_3 - 1;
                int _shfl_23 = __shfl_sync(0xFFFFFFFF, cu_excl, cu_lane);
                int cu_base = _shfl_23;
                int cu_rel = cu_u - cu_base;
                int cu_j = 0;
                int cu_prev = 0;
                int cu_mine = 0;
                int cu_pj = 0;
                cu_mine = cp_state[0];
                int _shfl_24 = __shfl_sync(0xFFFFFFFF, cu_mine, cu_lane);
                cu_pj = _shfl_24;
                if (cu_pj <= cu_rel) {
                    cu_j += 1;
                    cu_prev = cu_pj;
                }
                cu_mine = cp_state[1];
                int _shfl_25 = __shfl_sync(0xFFFFFFFF, cu_mine, cu_lane);
                cu_pj = _shfl_25;
                if (cu_pj <= cu_rel) {
                    cu_j += 1;
                    cu_prev = cu_pj;
                }
                cu_mine = cp_state[2];
                int _shfl_26 = __shfl_sync(0xFFFFFFFF, cu_mine, cu_lane);
                cu_pj = _shfl_26;
                if (cu_pj <= cu_rel) {
                    cu_j += 1;
                    cu_prev = cu_pj;
                }
                cu_mine = cp_state[3];
                int _shfl_27 = __shfl_sync(0xFFFFFFFF, cu_mine, cu_lane);
                cu_pj = _shfl_27;
                if (cu_pj <= cu_rel) {
                    cu_j += 1;
                    cu_prev = cu_pj;
                }
                cu_mine = cp_state[4];
                int _shfl_28 = __shfl_sync(0xFFFFFFFF, cu_mine, cu_lane);
                cu_pj = _shfl_28;
                if (cu_pj <= cu_rel) {
                    cu_j += 1;
                    cu_prev = cu_pj;
                }
                int cu_req = cu_lane * 5 + cu_j;
                int cu_s = cu_rel - cu_prev;
                int ctx_last = context_lens[cu_req * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits = (ctx_last + 256 - 1) / 256;
                if (cu_s < num_kv_splits) {
                    int tok_base = cu_req * K_NEXT_N;
                    mbarrier_wait(q_full_addr + (mt_q_stage) * 8, _phase_q_full);
                    float weights_reg[K_Q_TILE_ROWS];
                    int w_base = mt_q_stage * (unsigned int)K_Q_TILE_ROWS;
                    #pragma unroll
                    for (int wi = 0; wi < K_Q_TILE_ROWS; wi++) {
                        weights_reg[wi] = smem_weights[w_base + wi];
                    }
                    int ctx_qi[K_NEXT_N_ATOM];
                    #pragma unroll
                    for (int qp = 0; qp < K_NEXT_N_ATOM; qp++) {
                        ctx_qi[qp] = context_lens[tok_base + qp];
                    }
                    mbarrier_wait(kv_full_addr + (mt_kv_stage) * 8, _phase_kv_full);
                    int sc_base = mt_kv_stage * 256;
                    float scale_kv = smem_kv_scales[sc_base + math_thread_idx];
                    mbarrier_wait(umma_full_addr + (mt_tmem_stage) * 8, _phase_umma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(kv_empty_addr + (mt_kv_stage) * 8);
                    mt_kv_stage += 1;
                    if (mt_kv_stage == 5) { mt_kv_stage = 0; _phase_kv_full ^= 1; }
                    int kv_pos = cu_s * 256 + math_thread_idx;
                    #pragma unroll
                    for (int qi = 0; qi < K_NEXT_N_ATOM; qi++) {
                        float _tmem_load_0[32];
                        tmem_ld_x16(&_tmem_load_0[0], taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        tmem_ld_x16(&_tmem_load_0[16], taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16) + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (qi == K_NEXT_N_ATOM - 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(umma_empty_addr + (mt_tmem_stage) * 8);
                        }
                        float _relu_wsum_0;
                        {
                            float2 _sum0 = make_float2(0.0f, 0.0f);
                            float2 _sum1 = make_float2(0.0f, 0.0f);
                            #pragma unroll
                            for (int _j = 0; _j < 32; _j += 4) {
                                asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                    "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                    "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                    "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                    "mov.b64 _pb, {%3, %4};\n\t"
                                    "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                    : "+l"(*(unsigned long long*)&_sum0) : "f"(_tmem_load_0[0 + _j + 0]), "f"(_tmem_load_0[0 + _j + 1]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 0]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 1]));
                                asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                    "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                    "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                    "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                    "mov.b64 _pb, {%3, %4};\n\t"
                                    "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                    : "+l"(*(unsigned long long*)&_sum1) : "f"(_tmem_load_0[0 + _j + 2]), "f"(_tmem_load_0[0 + _j + 3]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 2]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 3]));
                            }
                            float2 _sum;
                            asm("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                            _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                        }
                        float scaled = _relu_wsum_0 * scale_kv;
                        float result = ((kv_pos < ctx_qi[qi]) ? scaled : -CAKE_INF);
                        int out_elem = (tok_base + qi) * stride_logits + kv_pos;
                        *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = result;
                    }
                    mt_tmem_stage += 1;
                    if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                    mt_tmem_stage += 1;
                    if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_empty_addr + (mt_q_stage) * 8);
                    mt_q_stage += 1;
                    if (mt_q_stage == 3) { mt_q_stage = 0; _phase_q_full ^= 1; }
                }
            }
        }
    // ---- Role: load_q ----
    } else if (warp == 8) {
        { // load_q_main
            unsigned int lq_stage = 0;
            int cp_lane_1 = lane;
            int cp_state_1[7];
            int cp_run_1 = 0;
            int cp_q_1 = 0;
            int cp_s_1 = 0;
            int cp_ctx_1 = 0;
            cp_q_1 = cp_lane_1 * 5;
            cp_s_1 = 0;
            if (cp_q_1 < batch_size) {
                cp_ctx_1 = context_lens[cp_q_1 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_1 = (cp_ctx_1 + 256 - 1) / 256;
            }
            cp_run_1 += cp_s_1;
            cp_state_1[0] = cp_run_1;
            cp_q_1 = cp_lane_1 * 5 + 1;
            cp_s_1 = 0;
            if (cp_q_1 < batch_size) {
                cp_ctx_1 = context_lens[cp_q_1 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_1 = (cp_ctx_1 + 256 - 1) / 256;
            }
            cp_run_1 += cp_s_1;
            cp_state_1[1] = cp_run_1;
            cp_q_1 = cp_lane_1 * 5 + 2;
            cp_s_1 = 0;
            if (cp_q_1 < batch_size) {
                cp_ctx_1 = context_lens[cp_q_1 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_1 = (cp_ctx_1 + 256 - 1) / 256;
            }
            cp_run_1 += cp_s_1;
            cp_state_1[2] = cp_run_1;
            cp_q_1 = cp_lane_1 * 5 + 3;
            cp_s_1 = 0;
            if (cp_q_1 < batch_size) {
                cp_ctx_1 = context_lens[cp_q_1 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_1 = (cp_ctx_1 + 256 - 1) / 256;
            }
            cp_run_1 += cp_s_1;
            cp_state_1[3] = cp_run_1;
            cp_q_1 = cp_lane_1 * 5 + 4;
            cp_s_1 = 0;
            if (cp_q_1 < batch_size) {
                cp_ctx_1 = context_lens[cp_q_1 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_1 = (cp_ctx_1 + 256 - 1) / 256;
            }
            cp_run_1 += cp_s_1;
            cp_state_1[4] = cp_run_1;
            int cp_incl_1 = cp_run_1;
            int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, cp_incl_1, 1, 32);
            int cp_up_4 = _shfl_up_0;
            if (cp_lane_1 >= 1) {
                cp_incl_1 += cp_up_4;
            }
            int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, cp_incl_1, 2, 32);
            int cp_up_0_1 = _shfl_up_1;
            if (cp_lane_1 >= 2) {
                cp_incl_1 += cp_up_0_1;
            }
            int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, cp_incl_1, 4, 32);
            int cp_up_1_1 = _shfl_up_2;
            if (cp_lane_1 >= 4) {
                cp_incl_1 += cp_up_1_1;
            }
            int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, cp_incl_1, 8, 32);
            int cp_up_2_1 = _shfl_up_3;
            if (cp_lane_1 >= 8) {
                cp_incl_1 += cp_up_2_1;
            }
            int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, cp_incl_1, 16, 32);
            int cp_up_3_1 = _shfl_up_4;
            if (cp_lane_1 >= 16) {
                cp_incl_1 += cp_up_3_1;
            }
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, cp_incl_1, 31);
            int cp_total_1 = _shfl_0;
            cp_state_1[5] = cp_incl_1;
            cp_state_1[6] = cp_incl_1 - cp_run_1;
            unsigned int _phase_q_empty = 1;
            #pragma unroll 1
            for (int u_1 = bid; u_1 < cp_total_1; u_1 += num_bids) {
                int cu_u_1 = u_1;
                int cu_incl_1 = cp_state_1[5];
                int cu_excl_1 = cp_state_1[6];
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, cu_incl_1 > cu_u_1);
                unsigned int cu_hit_1 = _vote_0;
                int _ffs_0 = __ffs(cu_hit_1);
                int cu_lane_1 = _ffs_0 - 1;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, cu_excl_1, cu_lane_1);
                int cu_base_1 = _shfl_1;
                int cu_rel_1 = cu_u_1 - cu_base_1;
                int cu_j_1 = 0;
                int cu_prev_1 = 0;
                int cu_mine_1 = 0;
                int cu_pj_1 = 0;
                cu_mine_1 = cp_state_1[0];
                int _shfl_2 = __shfl_sync(0xFFFFFFFF, cu_mine_1, cu_lane_1);
                cu_pj_1 = _shfl_2;
                if (cu_pj_1 <= cu_rel_1) {
                    cu_j_1 += 1;
                    cu_prev_1 = cu_pj_1;
                }
                cu_mine_1 = cp_state_1[1];
                int _shfl_3 = __shfl_sync(0xFFFFFFFF, cu_mine_1, cu_lane_1);
                cu_pj_1 = _shfl_3;
                if (cu_pj_1 <= cu_rel_1) {
                    cu_j_1 += 1;
                    cu_prev_1 = cu_pj_1;
                }
                cu_mine_1 = cp_state_1[2];
                int _shfl_4 = __shfl_sync(0xFFFFFFFF, cu_mine_1, cu_lane_1);
                cu_pj_1 = _shfl_4;
                if (cu_pj_1 <= cu_rel_1) {
                    cu_j_1 += 1;
                    cu_prev_1 = cu_pj_1;
                }
                cu_mine_1 = cp_state_1[3];
                int _shfl_5 = __shfl_sync(0xFFFFFFFF, cu_mine_1, cu_lane_1);
                cu_pj_1 = _shfl_5;
                if (cu_pj_1 <= cu_rel_1) {
                    cu_j_1 += 1;
                    cu_prev_1 = cu_pj_1;
                }
                cu_mine_1 = cp_state_1[4];
                int _shfl_6 = __shfl_sync(0xFFFFFFFF, cu_mine_1, cu_lane_1);
                cu_pj_1 = _shfl_6;
                if (cu_pj_1 <= cu_rel_1) {
                    cu_j_1 += 1;
                    cu_prev_1 = cu_pj_1;
                }
                int cu_req_1 = cu_lane_1 * 5 + cu_j_1;
                int cu_s_1 = cu_rel_1 - cu_prev_1;
                int ctx_last_1 = context_lens[cu_req_1 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_1 = (ctx_last_1 + 256 - 1) / 256;
                if (cu_s_1 < num_kv_splits_1) {
                    int tok_base_1 = cu_req_1 * K_NEXT_N;
                    mbarrier_wait(q_empty_addr + (lq_stage) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(smem_q_addr + lq_stage * 4096, (&Q), 0, tok_base_1 * K_NUM_HEADS, q_full_addr + (lq_stage) * 8);
                        tma_2d_gmem2smem(smem_weights_addr + lq_stage * 128, (&Weights), 0, tok_base_1, q_full_addr + (lq_stage) * 8);
                        mbarrier_arrive_expect_tx(q_full_addr + (lq_stage) * 8, K_Q_TX_BYTES);
                    }
                    lq_stage += 1;
                    if (lq_stage == 3) { lq_stage = 0; _phase_q_empty ^= 1; }
                }
            }
        }
    // ---- Role: load_kv ----
    } else if (warp == 9) {
        { // load_kv_main
            unsigned int lk_stage = 0;
            int cp_lane_2 = lane;
            int cp_state_2[7];
            int cp_run_2 = 0;
            int cp_q_2 = 0;
            int cp_s_2 = 0;
            int cp_ctx_2 = 0;
            cp_q_2 = cp_lane_2 * 5;
            cp_s_2 = 0;
            if (cp_q_2 < batch_size) {
                cp_ctx_2 = context_lens[cp_q_2 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_2 = (cp_ctx_2 + 256 - 1) / 256;
            }
            cp_run_2 += cp_s_2;
            cp_state_2[0] = cp_run_2;
            cp_q_2 = cp_lane_2 * 5 + 1;
            cp_s_2 = 0;
            if (cp_q_2 < batch_size) {
                cp_ctx_2 = context_lens[cp_q_2 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_2 = (cp_ctx_2 + 256 - 1) / 256;
            }
            cp_run_2 += cp_s_2;
            cp_state_2[1] = cp_run_2;
            cp_q_2 = cp_lane_2 * 5 + 2;
            cp_s_2 = 0;
            if (cp_q_2 < batch_size) {
                cp_ctx_2 = context_lens[cp_q_2 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_2 = (cp_ctx_2 + 256 - 1) / 256;
            }
            cp_run_2 += cp_s_2;
            cp_state_2[2] = cp_run_2;
            cp_q_2 = cp_lane_2 * 5 + 3;
            cp_s_2 = 0;
            if (cp_q_2 < batch_size) {
                cp_ctx_2 = context_lens[cp_q_2 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_2 = (cp_ctx_2 + 256 - 1) / 256;
            }
            cp_run_2 += cp_s_2;
            cp_state_2[3] = cp_run_2;
            cp_q_2 = cp_lane_2 * 5 + 4;
            cp_s_2 = 0;
            if (cp_q_2 < batch_size) {
                cp_ctx_2 = context_lens[cp_q_2 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_2 = (cp_ctx_2 + 256 - 1) / 256;
            }
            cp_run_2 += cp_s_2;
            cp_state_2[4] = cp_run_2;
            int cp_incl_2 = cp_run_2;
            int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, cp_incl_2, 1, 32);
            int cp_up_5 = _shfl_up_5;
            if (cp_lane_2 >= 1) {
                cp_incl_2 += cp_up_5;
            }
            int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, cp_incl_2, 2, 32);
            int cp_up_0_2 = _shfl_up_6;
            if (cp_lane_2 >= 2) {
                cp_incl_2 += cp_up_0_2;
            }
            int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, cp_incl_2, 4, 32);
            int cp_up_1_2 = _shfl_up_7;
            if (cp_lane_2 >= 4) {
                cp_incl_2 += cp_up_1_2;
            }
            int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, cp_incl_2, 8, 32);
            int cp_up_2_2 = _shfl_up_8;
            if (cp_lane_2 >= 8) {
                cp_incl_2 += cp_up_2_2;
            }
            int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, cp_incl_2, 16, 32);
            int cp_up_3_2 = _shfl_up_9;
            if (cp_lane_2 >= 16) {
                cp_incl_2 += cp_up_3_2;
            }
            int _shfl_7 = __shfl_sync(0xFFFFFFFF, cp_incl_2, 31);
            int cp_total_2 = _shfl_7;
            cp_state_2[5] = cp_incl_2;
            cp_state_2[6] = cp_incl_2 - cp_run_2;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int u_2 = bid; u_2 < cp_total_2; u_2 += num_bids) {
                int cu_u_2 = u_2;
                int cu_incl_2 = cp_state_2[5];
                int cu_excl_2 = cp_state_2[6];
                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, cu_incl_2 > cu_u_2);
                unsigned int cu_hit_2 = _vote_1;
                int _ffs_1 = __ffs(cu_hit_2);
                int cu_lane_2 = _ffs_1 - 1;
                int _shfl_8 = __shfl_sync(0xFFFFFFFF, cu_excl_2, cu_lane_2);
                int cu_base_2 = _shfl_8;
                int cu_rel_2 = cu_u_2 - cu_base_2;
                int cu_j_2 = 0;
                int cu_prev_2 = 0;
                int cu_mine_2 = 0;
                int cu_pj_2 = 0;
                cu_mine_2 = cp_state_2[0];
                int _shfl_9 = __shfl_sync(0xFFFFFFFF, cu_mine_2, cu_lane_2);
                cu_pj_2 = _shfl_9;
                if (cu_pj_2 <= cu_rel_2) {
                    cu_j_2 += 1;
                    cu_prev_2 = cu_pj_2;
                }
                cu_mine_2 = cp_state_2[1];
                int _shfl_10 = __shfl_sync(0xFFFFFFFF, cu_mine_2, cu_lane_2);
                cu_pj_2 = _shfl_10;
                if (cu_pj_2 <= cu_rel_2) {
                    cu_j_2 += 1;
                    cu_prev_2 = cu_pj_2;
                }
                cu_mine_2 = cp_state_2[2];
                int _shfl_11 = __shfl_sync(0xFFFFFFFF, cu_mine_2, cu_lane_2);
                cu_pj_2 = _shfl_11;
                if (cu_pj_2 <= cu_rel_2) {
                    cu_j_2 += 1;
                    cu_prev_2 = cu_pj_2;
                }
                cu_mine_2 = cp_state_2[3];
                int _shfl_12 = __shfl_sync(0xFFFFFFFF, cu_mine_2, cu_lane_2);
                cu_pj_2 = _shfl_12;
                if (cu_pj_2 <= cu_rel_2) {
                    cu_j_2 += 1;
                    cu_prev_2 = cu_pj_2;
                }
                cu_mine_2 = cp_state_2[4];
                int _shfl_13 = __shfl_sync(0xFFFFFFFF, cu_mine_2, cu_lane_2);
                cu_pj_2 = _shfl_13;
                if (cu_pj_2 <= cu_rel_2) {
                    cu_j_2 += 1;
                    cu_prev_2 = cu_pj_2;
                }
                int cu_req_2 = cu_lane_2 * 5 + cu_j_2;
                int cu_s_2 = cu_rel_2 - cu_prev_2;
                int ctx_last_2 = context_lens[cu_req_2 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_2 = (ctx_last_2 + 256 - 1) / 256;
                int lk_page_off = cu_s_2 * K_NUM_BLOCKS_PER_SPLIT + lane;
                int lk_word = 0;
                if (lane < K_NUM_BLOCKS_PER_SPLIT) {
                    if (lk_page_off < block_table_stride) {
                        lk_word = block_table[cu_req_2 * block_table_stride + lk_page_off];
                    }
                }
                if (cu_s_2 < num_kv_splits_2) {
                    int lk_num_pages = (ctx_last_2 + K_BLOCK_KV - 1) / K_BLOCK_KV;
                    int lk_coord = ((lk_page_off < lk_num_pages) ? lk_word : 0);
                    int page_idx_reg[K_NUM_BLOCKS_PER_SPLIT];
                    #pragma unroll
                    for (int i = 0; i < K_NUM_BLOCKS_PER_SPLIT; i++) {
                        int _shfl_14 = __shfl_sync(0xFFFFFFFF, lk_coord, i);
                        page_idx_reg[i] = _shfl_14;
                    }
                    mbarrier_wait(kv_empty_addr + (lk_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        #pragma unroll
                        for (int i_1 = 0; i_1 < K_NUM_BLOCKS_PER_SPLIT; i_1++) {
                            tma_3d_gmem2smem(smem_kv_addr + lk_stage * 32768 + (unsigned int)(i_1 * (K_BLOCK_KV * 128)), (&KV), 0, 0, page_idx_reg[i_1], kv_full_addr + (lk_stage) * 8);
                            tma_2d_gmem2smem(smem_kv_scales_addr + lk_stage * 1024 + (unsigned int)(i_1 * (K_BLOCK_KV * 4)), (&KV_scales), K_SCALE_TAIL_F32, page_idx_reg[i_1], kv_full_addr + (lk_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(kv_full_addr + (lk_stage) * 8, 33792);
                    }
                    lk_stage += 1;
                    if (lk_stage == 5) { lk_stage = 0; _phase_kv_empty ^= 1; }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        { // mma_main
            unsigned int mm_q_stage = 0;
            unsigned int mm_kv_stage = 0;
            unsigned int mm_tmem_stage = 0;
            int cp_lane_3 = lane;
            int cp_state_3[7];
            int cp_run_3 = 0;
            int cp_q_3 = 0;
            int cp_s_3 = 0;
            int cp_ctx_3 = 0;
            cp_q_3 = cp_lane_3 * 5;
            cp_s_3 = 0;
            if (cp_q_3 < batch_size) {
                cp_ctx_3 = context_lens[cp_q_3 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_3 = (cp_ctx_3 + 256 - 1) / 256;
            }
            cp_run_3 += cp_s_3;
            cp_state_3[0] = cp_run_3;
            cp_q_3 = cp_lane_3 * 5 + 1;
            cp_s_3 = 0;
            if (cp_q_3 < batch_size) {
                cp_ctx_3 = context_lens[cp_q_3 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_3 = (cp_ctx_3 + 256 - 1) / 256;
            }
            cp_run_3 += cp_s_3;
            cp_state_3[1] = cp_run_3;
            cp_q_3 = cp_lane_3 * 5 + 2;
            cp_s_3 = 0;
            if (cp_q_3 < batch_size) {
                cp_ctx_3 = context_lens[cp_q_3 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_3 = (cp_ctx_3 + 256 - 1) / 256;
            }
            cp_run_3 += cp_s_3;
            cp_state_3[2] = cp_run_3;
            cp_q_3 = cp_lane_3 * 5 + 3;
            cp_s_3 = 0;
            if (cp_q_3 < batch_size) {
                cp_ctx_3 = context_lens[cp_q_3 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_3 = (cp_ctx_3 + 256 - 1) / 256;
            }
            cp_run_3 += cp_s_3;
            cp_state_3[3] = cp_run_3;
            cp_q_3 = cp_lane_3 * 5 + 4;
            cp_s_3 = 0;
            if (cp_q_3 < batch_size) {
                cp_ctx_3 = context_lens[cp_q_3 * K_NEXT_N + (K_NEXT_N - 1)];
                cp_s_3 = (cp_ctx_3 + 256 - 1) / 256;
            }
            cp_run_3 += cp_s_3;
            cp_state_3[4] = cp_run_3;
            int cp_incl_3 = cp_run_3;
            int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, cp_incl_3, 1, 32);
            int cp_up_6 = _shfl_up_10;
            if (cp_lane_3 >= 1) {
                cp_incl_3 += cp_up_6;
            }
            int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, cp_incl_3, 2, 32);
            int cp_up_0_3 = _shfl_up_11;
            if (cp_lane_3 >= 2) {
                cp_incl_3 += cp_up_0_3;
            }
            int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, cp_incl_3, 4, 32);
            int cp_up_1_3 = _shfl_up_12;
            if (cp_lane_3 >= 4) {
                cp_incl_3 += cp_up_1_3;
            }
            int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, cp_incl_3, 8, 32);
            int cp_up_2_3 = _shfl_up_13;
            if (cp_lane_3 >= 8) {
                cp_incl_3 += cp_up_2_3;
            }
            int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, cp_incl_3, 16, 32);
            int cp_up_3_3 = _shfl_up_14;
            if (cp_lane_3 >= 16) {
                cp_incl_3 += cp_up_3_3;
            }
            int _shfl_15 = __shfl_sync(0xFFFFFFFF, cp_incl_3, 31);
            int cp_total_3 = _shfl_15;
            cp_state_3[5] = cp_incl_3;
            cp_state_3[6] = cp_incl_3 - cp_run_3;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (int u_3 = bid; u_3 < cp_total_3; u_3 += num_bids) {
                int cu_u_3 = u_3;
                int cu_incl_3 = cp_state_3[5];
                int cu_excl_3 = cp_state_3[6];
                unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, cu_incl_3 > cu_u_3);
                unsigned int cu_hit_3 = _vote_2;
                int _ffs_2 = __ffs(cu_hit_3);
                int cu_lane_3 = _ffs_2 - 1;
                int _shfl_16 = __shfl_sync(0xFFFFFFFF, cu_excl_3, cu_lane_3);
                int cu_base_3 = _shfl_16;
                int cu_rel_3 = cu_u_3 - cu_base_3;
                int cu_j_3 = 0;
                int cu_prev_3 = 0;
                int cu_mine_3 = 0;
                int cu_pj_3 = 0;
                cu_mine_3 = cp_state_3[0];
                int _shfl_17 = __shfl_sync(0xFFFFFFFF, cu_mine_3, cu_lane_3);
                cu_pj_3 = _shfl_17;
                if (cu_pj_3 <= cu_rel_3) {
                    cu_j_3 += 1;
                    cu_prev_3 = cu_pj_3;
                }
                cu_mine_3 = cp_state_3[1];
                int _shfl_18 = __shfl_sync(0xFFFFFFFF, cu_mine_3, cu_lane_3);
                cu_pj_3 = _shfl_18;
                if (cu_pj_3 <= cu_rel_3) {
                    cu_j_3 += 1;
                    cu_prev_3 = cu_pj_3;
                }
                cu_mine_3 = cp_state_3[2];
                int _shfl_19 = __shfl_sync(0xFFFFFFFF, cu_mine_3, cu_lane_3);
                cu_pj_3 = _shfl_19;
                if (cu_pj_3 <= cu_rel_3) {
                    cu_j_3 += 1;
                    cu_prev_3 = cu_pj_3;
                }
                cu_mine_3 = cp_state_3[3];
                int _shfl_20 = __shfl_sync(0xFFFFFFFF, cu_mine_3, cu_lane_3);
                cu_pj_3 = _shfl_20;
                if (cu_pj_3 <= cu_rel_3) {
                    cu_j_3 += 1;
                    cu_prev_3 = cu_pj_3;
                }
                cu_mine_3 = cp_state_3[4];
                int _shfl_21 = __shfl_sync(0xFFFFFFFF, cu_mine_3, cu_lane_3);
                cu_pj_3 = _shfl_21;
                if (cu_pj_3 <= cu_rel_3) {
                    cu_j_3 += 1;
                    cu_prev_3 = cu_pj_3;
                }
                int cu_req_3 = cu_lane_3 * 5 + cu_j_3;
                int cu_s_3 = cu_rel_3 - cu_prev_3;
                int ctx_last_3 = context_lens[cu_req_3 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_3 = (ctx_last_3 + 256 - 1) / 256;
                if (cu_s_3 < num_kv_splits_3) {
                    mbarrier_wait(q_full_addr + (mm_q_stage) * 8, _phase_q_full_1);
                    mbarrier_wait(kv_full_addr + (mm_kv_stage) * 8, _phase_kv_full_1);
                    if (elect_sync()) {
                        mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                        int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mm_tmem_stage * 32))), "r"(0));
                        tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                        mm_tmem_stage += 1;
                        if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                        mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                        int _mma_b_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 256;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_acc + (mm_tmem_stage * 32))), "r"(0));
                        tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                        mm_tmem_stage += 1;
                        if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                    }
                    __syncwarp();
                    mm_kv_stage += 1;
                    if (mm_kv_stage == 5) { mm_kv_stage = 0; _phase_kv_full_1 ^= 1; }
                    mbarrier_arrive(q_empty_addr + (mm_q_stage) * 8);
                    mm_q_stage += 1;
                    if (mm_q_stage == 3) { mm_q_stage = 0; _phase_q_full_1 ^= 1; }
                }
            }
        }
    // ---- Role: idle ----
    } else if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
