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
#ifndef CAKE_GDN_STATE_VEC_ALIGNED
#error "CAKE_GDN_STATE_VEC_ALIGNED is a downstream specialization of this program; define it on the compile line"
#endif
// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_gdn_device_common.cuh"
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#ifndef NUM_O_HEADS_LOG2
#error "NUM_O_HEADS_LOG2 is a downstream specialization of this program; define it on the compile line"
#elif NUM_O_HEADS_LOG2 != 3 && NUM_O_HEADS_LOG2 != 4 && NUM_O_HEADS_LOG2 != 5
#error "NUM_O_HEADS_LOG2 takes 3, 4, 5 in this program; another value renders other code"
#endif
#ifndef HEAD_GROUP_LOG2
#error "HEAD_GROUP_LOG2 is a downstream specialization of this program; define it on the compile line"
#elif HEAD_GROUP_LOG2 != 1 && HEAD_GROUP_LOG2 != 2
#error "HEAD_GROUP_LOG2 takes 1, 2 in this program; another value renders other code"
#endif
#ifndef SINGLE_CHUNK_NO_STATE
#error "SINGLE_CHUNK_NO_STATE is a downstream specialization of this program; define it on the compile line"
#elif SINGLE_CHUNK_NO_STATE != 0
#error "SINGLE_CHUNK_NO_STATE takes 0 in this program; another value renders other code"
#endif
#ifndef UNIT_GATES
#error "UNIT_GATES is a downstream specialization of this program; define it on the compile line"
#elif UNIT_GATES != 0
#error "UNIT_GATES takes 0 in this program; another value renders other code"
#endif
#ifndef USE_INITIAL_STATE
#error "USE_INITIAL_STATE is a downstream specialization of this program; define it on the compile line"
#elif USE_INITIAL_STATE != 1
#error "USE_INITIAL_STATE takes 1 in this program; another value renders other code"
#endif
#ifndef STORE_FINAL_STATE
#error "STORE_FINAL_STATE is a downstream specialization of this program; define it on the compile line"
#elif STORE_FINAL_STATE != 1
#error "STORE_FINAL_STATE takes 1 in this program; another value renders other code"
#endif
#ifndef ENABLE_CHECKPOINTS
#error "ENABLE_CHECKPOINTS is a downstream specialization of this program; define it on the compile line"
#elif ENABLE_CHECKPOINTS != 0
#error "ENABLE_CHECKPOINTS takes 0 in this program; another value renders other code"
#endif
#ifndef IS_GQA
#error "IS_GQA is a downstream specialization of this program; define it on the compile line"
#elif IS_GQA != 0
#error "IS_GQA takes 0 in this program; another value renders other code"
#endif
#ifndef USE_STATE_INDICES
#error "USE_STATE_INDICES is a downstream specialization of this program; define it on the compile line"
#elif USE_STATE_INDICES != 0
#error "USE_STATE_INDICES takes 0 in this program; another value renders other code"
#endif
#define LAUNCH_MIN_BLOCKS 1

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}

__device__ __forceinline__ uint32_t make_warp_uniform_local(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_453069def52efc7aca98(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ state_indices, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ output_state, __nv_bfloat16* __restrict__ checkpoint_state, int* __restrict__ cu_checkpoints, uint8_t* __restrict__ tensormap_workspace, long long initial_state_stride_slot, long long output_state_stride_slot, int checkpoint_every_n_tokens, float scale, int num_seqs, int total_tokens, int num_q_heads, int num_v_heads, int total_tiles, int metadata_i64_mask)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform_local(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform_local(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

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
    #define kv_state_drained_addr (mbar_base + 576)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
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

    // Mbarrier init (80 barriers at smem_raw[0..640))

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
            // --- pipeline 'one_stage' ---
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
            // --- pipeline 'one_stage' ---
            // kv_state_drained: 8 barriers, init_count=128
            mbarrier_init(smem + 576, 128);
            mbarrier_init(smem + 584, 128);
            mbarrier_init(smem + 592, 128);
            mbarrier_init(smem + 600, 128);
            mbarrier_init(smem + 608, 128);
            mbarrier_init(smem + 616, 128);
            mbarrier_init(smem + 624, 128);
            mbarrier_init(smem + 632, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 640);
    if (warp == 4) {
        int _tmem_hold = smem + 640;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_tmem_state = taddr;
    const unsigned int tmem_tmem_q_state = taddr + 128;
    const unsigned int tmem_tmem_state_inp = taddr + 192;
    const unsigned int tmem_tmem_cg0_shared_acc = taddr + 256;
    const unsigned int tmem_tmem_cg1_shared_acc = taddr + 384;
    const unsigned int tmem_tmem_shared_inp = taddr + 448;

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
    }

    // ---- Role: compute_group_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // compute_group_0_main
            unsigned int gate_k_cg0 = 0;
            int cg0_pairs = 0;
            unsigned int beta_k_cg0 = 0;
            unsigned int ainv_cg0_stage = 0;
            unsigned int qk_cg0_stage = 0;
            unsigned int _phase_load_gate_full = 0;
            unsigned int _phase_load_beta_full = 0;
            unsigned int _phase_ainv_smem_empty = 1;
            unsigned int _phase_qk_smem_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
                int num_o_heads = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx = ((NUM_O_HEADS_LOG2 >= 0) ? tile >> (unsigned int)NUM_O_HEADS_LOG2 : tile / (unsigned int)num_o_heads);
                int head_idx = ((NUM_O_HEADS_LOG2 >= 0) ? tile & (unsigned int)(num_o_heads - 1) : tile % (unsigned int)num_o_heads);
                int qk_head_idx = ((IS_GQA != 0) ? head_idx : ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_v_heads / num_q_heads)));
                int v_head_idx = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_q_heads / num_v_heads)) : head_idx);
                int raw_batch_start = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx] : cu_seqlens[batch_idx]);
                int raw_batch_end = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx + 1)] : cu_seqlens[batch_idx + 1]);
                int batch_start_floor = ((raw_batch_start > 0) ? raw_batch_start : 0);
                int batch_start = ((batch_start_floor < total_tokens) ? batch_start_floor : total_tokens);
                int batch_end_floor = ((raw_batch_end > batch_start) ? raw_batch_end : batch_start);
                int batch_end = ((batch_end_floor < total_tokens) ? batch_end_floor : total_tokens);
                int seqlen_b = batch_end - batch_start;
                int num_pairs_b = (seqlen_b + 128 - 1) / 128;
                int num_chunks_b = num_pairs_b * 2;
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks_b; chunk_idx += 2) {
                    int chunk_offset = batch_start + chunk_idx * 64;
                    int _cg0_marker = batch_idx + head_idx + chunk_offset + batch_end;
                    int warp_id_in_role = warp - 0;
                    int warp_id_in_role_cg0 = warp_id_in_role;
                    int row_cg0 = warp_id_in_role_cg0 * 32 + lane;
                    int lane_quad_cg0 = lane & 3;
                    int lane_row_cg0 = lane / 4;
                    int qk_warp_row_base_cg0 = warp_id_in_role_cg0 * 16;
                    int qk_tmem_row_base_cg0 = warp_id_in_role_cg0 * 32 << 16;
                    unsigned int gate0_stage = gate_k_cg0;
                    {
                        mbarrier_wait(load_gate_full_addr + (gate_k_cg0) * 8, _phase_load_gate_full);
                        gate_k_cg0 += 1;
                        if (gate_k_cg0 == 5) { gate_k_cg0 = 0; _phase_load_gate_full ^= 1; }
                    }
                    unsigned int gate1_stage = gate_k_cg0;
                    {
                        mbarrier_wait(load_gate_full_addr + (gate_k_cg0) * 8, _phase_load_gate_full);
                        gate_k_cg0 += 1;
                        if (gate_k_cg0 == 5) { gate_k_cg0 = 0; _phase_load_gate_full ^= 1; }
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
                        {
                            float qk_row_cumsumlog0_cg0 = smem_cumsumlog[gate0_elem_base + qk_row_cg0];
                            float _exp2_0 = approx_exp2(qk_row_cumsumlog0_cg0 - smem_cumsumlog[gate0_elem_base + qk_col_cg0]);
                            qk_transfer0_cg0[qk_j_cg0] = ((qk_row_cg0 >= qk_col_cg0) ? _exp2_0 : 0.0f);
                        }
                    }
                    #pragma unroll
                    for (int qk_j_cg0_1 = 0; qk_j_cg0_1 < 32; qk_j_cg0_1++) {
                        int qk_repeat_cg0_1 = qk_j_cg0_1 / 4;
                        int qk_reg_cg0_1 = qk_j_cg0_1 & 3;
                        int qk_row_cg0_1 = qk_warp_row_base_cg0 + lane_row_cg0 + qk_reg_cg0_1 / 2 * 8;
                        int qk_col_cg0_1 = qk_repeat_cg0_1 * 8 + lane_quad_cg0 * 2 + (qk_reg_cg0_1 & 1);
                        {
                            float qk_row_cumsumlog1_cg0 = smem_cumsumlog[gate1_elem_base + qk_row_cg0_1];
                            float _exp2_1 = approx_exp2(qk_row_cumsumlog1_cg0 - smem_cumsumlog[gate1_elem_base + qk_col_cg0_1]);
                            qk_transfer1_cg0[qk_j_cg0_1] = ((qk_row_cg0_1 >= qk_col_cg0_1) ? _exp2_1 : 0.0f);
                        }
                    }
                    {
                        mbarrier_arrive(gate_cg0_empty_addr + (gate0_stage) * 8);
                        mbarrier_arrive(gate_cg0_empty_addr + (gate1_stage) * 8);
                    }
                    unsigned int beta0_stage = beta_k_cg0;
                    {
                        mbarrier_wait(load_beta_full_addr + (beta_k_cg0) * 8, _phase_load_beta_full);
                        beta_k_cg0 += 1;
                        if (beta_k_cg0 == 5) { beta_k_cg0 = 0; _phase_load_beta_full ^= 1; }
                    }
                    unsigned int beta1_stage = beta_k_cg0;
                    {
                        mbarrier_wait(load_beta_full_addr + (beta_k_cg0) * 8, _phase_load_beta_full);
                        beta_k_cg0 += 1;
                        if (beta_k_cg0 == 5) { beta_k_cg0 = 0; _phase_load_beta_full ^= 1; }
                    }
                    int beta0_elem_base = beta0_stage * 64;
                    int beta1_elem_base = beta1_stage * 64;
                    unsigned int ainv0_stage = ainv_cg0_stage;
                    mbarrier_wait(ainv_smem_empty_addr + (ainv_cg0_stage) * 8, _phase_ainv_smem_empty);
                    ainv_cg0_stage += 1;
                    if (ainv_cg0_stage == 3) { ainv_cg0_stage = 0; _phase_ainv_smem_empty ^= 1; }
                    unsigned int ainv1_stage = ainv_cg0_stage;
                    mbarrier_wait(ainv_smem_empty_addr + (ainv_cg0_stage) * 8, _phase_ainv_smem_empty);
                    ainv_cg0_stage += 1;
                    if (ainv_cg0_stage == 3) { ainv_cg0_stage = 0; _phase_ainv_smem_empty ^= 1; }
                    mbarrier_wait(cg0_shared_acc_full_addr, 0);
                    int kk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[31]))
                        : "r"(kk_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        int kk_beta_row0_cg0 = qk_warp_row_base_cg0 + lane_row_cg0;
                        int kk_beta_row1_cg0 = kk_beta_row0_cg0 + 8;
                        float kk_beta0_cg0 = smem_beta[beta0_elem_base + kk_beta_row0_cg0];
                        float kk_beta1_cg0 = smem_beta[beta0_elem_base + kk_beta_row1_cg0];
                        int kk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                        int kk_stsm_col_lane_cg0 = (lane & 16) / 2;
                        #pragma unroll
                        for (int qk_repeat_cg0_2 = 0; qk_repeat_cg0_2 < 4; qk_repeat_cg0_2++) {
                            const int qk_j0_cg0 = qk_repeat_cg0_2 * 8;
                            float kk_scaled_cg0[8];
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                reinterpret_cast<float2*>(kk_scaled_cg0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load + qk_j0_cg0))[_ls], reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0))[_ls]);
                            {
                                float2 _pair_scale_even2_2 = make_float2(kk_beta0_cg0, kk_beta0_cg0);
                                float2 _pair_scale_odd2_2 = make_float2(kk_beta1_cg0, kk_beta1_cg0);
                                float2* _pair_scale_src2_2 = reinterpret_cast<float2*>(&kk_scaled_cg0[0]);
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_2));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_2[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_2));
                            }
                            unsigned int kk_scaled_cg0_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kk_scaled_cg0[_lp*2 + 0], kk_scaled_cg0[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int kk_stsm_addr_cg0 = (smem_ainv_rm_addr + ainv0_stage * 8192 + (unsigned int)(kk_stsm_row_cg0 * 128 + (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0) * 2 ^ (kk_stsm_row_cg0 * 128 + (qk_repeat_cg0_2 * 16 + kk_stsm_col_lane_cg0) * 2 >> 7 & 7) << 4));
                            uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16[3]))
                                : "memory");
                        }
                    }
                    mbarrier_arrive(cg0_shared_acc_empty_addr);
                    mbarrier_wait(cg0_shared_acc_full_addr + 8, 0);
                    int kk_addr_1 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[31]))
                        : "r"(kk_addr_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        int kk_beta_row0_cg0_1 = qk_warp_row_base_cg0 + lane_row_cg0;
                        int kk_beta_row1_cg0_1 = kk_beta_row0_cg0_1 + 8;
                        float kk_beta0_cg0_1 = smem_beta[beta1_elem_base + kk_beta_row0_cg0_1];
                        float kk_beta1_cg0_1 = smem_beta[beta1_elem_base + kk_beta_row1_cg0_1];
                        int kk_stsm_row_cg0_1 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                        int kk_stsm_col_lane_cg0_1 = (lane & 16) / 2;
                        #pragma unroll
                        for (int qk_repeat_cg0_3 = 0; qk_repeat_cg0_3 < 4; qk_repeat_cg0_3++) {
                            const int qk_j0_cg0_1 = qk_repeat_cg0_3 * 8;
                            float kk_scaled_cg0_1[8];
                            #pragma unroll
                            for (int _ls = 0; _ls < 4; _ls++)
                                reinterpret_cast<float2*>(kk_scaled_cg0_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_0 + qk_j0_cg0_1))[_ls], reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_1))[_ls]);
                            {
                                float2 _pair_scale_even2_4 = make_float2(kk_beta0_cg0_1, kk_beta0_cg0_1);
                                float2 _pair_scale_odd2_4 = make_float2(kk_beta1_cg0_1, kk_beta1_cg0_1);
                                float2* _pair_scale_src2_4 = reinterpret_cast<float2*>(&kk_scaled_cg0_1[0]);
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_4));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_4));
                                asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_4[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_4));
                            }
                            unsigned int kk_scaled_cg0_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kk_scaled_cg0_1[_lp*2 + 0], kk_scaled_cg0_1[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            int kk_stsm_addr_cg0_1 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)(kk_stsm_row_cg0_1 * 128 + (qk_repeat_cg0_3 * 16 + kk_stsm_col_lane_cg0_1) * 2 ^ (kk_stsm_row_cg0_1 * 128 + (qk_repeat_cg0_3 * 16 + kk_stsm_col_lane_cg0_1) * 2 >> 7 & 7) << 4));
                            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)kk_stsm_addr_cg0_1);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1[3]))
                                : "memory");
                        }
                    }
                    mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    int inverse_group_cg0 = warp_id_in_role_cg0 / 2;
                    int inverse_local_warp_cg0 = warp_id_in_role_cg0 & 1;
                    unsigned int inverse_stage_cg0 = ((inverse_group_cg0 == 1) ? ainv1_stage : ainv0_stage);
                    int inverse_row_cg0 = tid & 63;
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
                        float unpack_bf16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(inv_row_packed_cg0[_pair]) << 16);
                            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(inv_row_packed_cg0[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int inv_j = 0; inv_j < 8; inv_j++) {
                            inv_row_cg0[inv_j] = unpack_bf16[inv_j];
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
                                float shfl_val_cg0;
                                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(shfl_val_cg0) : "f"(inv_row_cg0[prev_col_cg0]), "r"(pivot_lane_cg0));
                                if (lane_in_diag_cg0 > src_row_cg0) {
                                    inv_row_cg0[prev_col_cg0] = __fmaf_rn(row_scale_cg0, shfl_val_cg0, inv_row_cg0[prev_col_cg0]);
                                }
                            }
                            if (lane_in_diag_cg0 > src_row_cg0) {
                                inv_row_cg0[src_row_cg0] = row_scale_cg0;
                            }
                        }
                        unsigned int kk_scaled_cg0_bf16_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row_cg0[_lp*2 + 0], inv_row_cg0[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int inv_store_j_cg0 = 0; inv_store_j_cg0 < 4; inv_store_j_cg0++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(inv_diag_addr_cg0 + inv_store_j_cg0 * 4), "r"(kk_scaled_cg0_bf16_2[inv_store_j_cg0]));
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
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(inv16_dc_acc_cg0[0]), "=f"(inv16_dc_acc_cg0[1]), "=f"(inv16_dc_acc_cg0[2]), "=f"(inv16_dc_acc_cg0[3])
                        : "r"(inv16_d_frag_cg0[0]), "r"(inv16_d_frag_cg0[1]), "r"(inv16_c_frag_cg0[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
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
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(inv16_o_acc_cg0[0]), "=f"(inv16_o_acc_cg0[1]), "=f"(inv16_o_acc_cg0[2]), "=f"(inv16_o_acc_cg0[3])
                        : "r"(inv16_dc_bf16_cg0[0]), "r"(inv16_dc_bf16_cg0[1]), "r"(inv16_a_frag_cg0[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_o_acc_cg0[_lp*2 + 0], inv16_o_acc_cg0[_lp*2+1 + 0]));
                        inv16_o_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                    }
                    __syncwarp();
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)inv16_o_addr_cg0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&inv16_o_bf16_cg0[0]))
                        : "memory");
                    int inv16_lane_row_cg0_0 = lane & 7;
                    int inv16_d_addr_cg0_0 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + (inv16_tile_base_cg0 + 8) * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + (inv16_tile_base_cg0 + 8) * 2 >> 7 & 7) << 4));
                    int inv16_c_addr_cg0_0 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    int inv16_a_addr_cg0_0 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    int inv16_o_addr_cg0_0 = (smem_ainv_rm_addr + ainv1_stage * 8192 + (unsigned int)((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 ^ ((inv16_tile_base_cg0 + 8 + inv16_lane_row_cg0_0) * 128 + inv16_tile_base_cg0 * 2 >> 7 & 7) << 4));
                    unsigned int inv16_d_frag_cg0_0[2];
                    unsigned int inv16_c_frag_cg0_0[1];
                    float inv16_dc_acc_cg0_0[4];
                    unsigned int inv16_dc_bf16_cg0_0[2];
                    unsigned int inv16_a_frag_cg0_0[1];
                    float inv16_o_acc_cg0_0[4];
                    unsigned int inv16_o_bf16_cg0_0[2];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_d_frag_cg0_0[0])
                        : "r"(inv16_d_addr_cg0_0)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_d_frag_cg0_0[1])
                        : "r"(inv16_d_addr_cg0_0)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_c_frag_cg0_0[0])
                        : "r"(inv16_c_addr_cg0_0)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(inv16_dc_acc_cg0_0[0]), "=f"(inv16_dc_acc_cg0_0[1]), "=f"(inv16_dc_acc_cg0_0[2]), "=f"(inv16_dc_acc_cg0_0[3])
                        : "r"(inv16_d_frag_cg0_0[0]), "r"(inv16_d_frag_cg0_0[1]), "r"(inv16_c_frag_cg0_0[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int inv16_i_cg0_1 = 0; inv16_i_cg0_1 < 4; inv16_i_cg0_1++) {
                        inv16_dc_acc_cg0_0[inv16_i_cg0_1] = inv16_dc_acc_cg0_0[inv16_i_cg0_1] * -1.0f;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_dc_acc_cg0_0[_lp*2 + 0], inv16_dc_acc_cg0_0[_lp*2+1 + 0]));
                        inv16_dc_bf16_cg0_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv16_a_frag_cg0_0[0])
                        : "r"(inv16_a_addr_cg0_0)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(inv16_o_acc_cg0_0[0]), "=f"(inv16_o_acc_cg0_0[1]), "=f"(inv16_o_acc_cg0_0[2]), "=f"(inv16_o_acc_cg0_0[3])
                        : "r"(inv16_dc_bf16_cg0_0[0]), "r"(inv16_dc_bf16_cg0_0[1]), "r"(inv16_a_frag_cg0_0[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv16_o_acc_cg0_0[_lp*2 + 0], inv16_o_acc_cg0_0[_lp*2+1 + 0]));
                        inv16_o_bf16_cg0_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    __syncwarp();
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)inv16_o_addr_cg0_0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&inv16_o_bf16_cg0_0[0]))
                        : "memory");
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
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(inv32_dc_acc_cg0[0]), "=f"(inv32_dc_acc_cg0[1]), "=f"(inv32_dc_acc_cg0[2]), "=f"(inv32_dc_acc_cg0[3])
                            : "r"(inv32_d_frag_cg0[0]), "r"(inv32_d_frag_cg0[1]), "r"(inv32_d_frag_cg0[2]), "r"(inv32_d_frag_cg0[3]), "r"(inv32_c_frag_cg0[0]), "r"(inv32_c_frag_cg0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(inv32_dc_acc_cg0[4]), "=f"(inv32_dc_acc_cg0[(4) + 1]), "=f"(inv32_dc_acc_cg0[(4) + 2]), "=f"(inv32_dc_acc_cg0[(4) + 3])
                            : "r"(inv32_d_frag_cg0[0]), "r"(inv32_d_frag_cg0[1]), "r"(inv32_d_frag_cg0[2]), "r"(inv32_d_frag_cg0[3]), "r"(inv32_c_frag_cg0[2]), "r"(inv32_c_frag_cg0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
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
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(inv32_o_acc_cg0[0]), "=f"(inv32_o_acc_cg0[1]), "=f"(inv32_o_acc_cg0[2]), "=f"(inv32_o_acc_cg0[3])
                            : "r"(inv32_dc_bf16_cg0[0]), "r"(inv32_dc_bf16_cg0[1]), "r"(inv32_dc_bf16_cg0[2]), "r"(inv32_dc_bf16_cg0[3]), "r"(inv32_a_frag_cg0[0]), "r"(inv32_a_frag_cg0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(inv32_o_acc_cg0[4]), "=f"(inv32_o_acc_cg0[(4) + 1]), "=f"(inv32_o_acc_cg0[(4) + 2]), "=f"(inv32_o_acc_cg0[(4) + 3])
                            : "r"(inv32_dc_bf16_cg0[0]), "r"(inv32_dc_bf16_cg0[1]), "r"(inv32_dc_bf16_cg0[2]), "r"(inv32_dc_bf16_cg0[3]), "r"(inv32_a_frag_cg0[2]), "r"(inv32_a_frag_cg0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv32_o_acc_cg0[_lp*2 + 0], inv32_o_acc_cg0[_lp*2+1 + 0]));
                            inv32_o_bf16_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        __syncwarp();
                        uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)inv32_o_addr_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv32_o_bf16_cg0[3]))
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
                        for (int inv64_zero_i_cg0_1_0 = 0; inv64_zero_i_cg0_1_0 < 8; inv64_zero_i_cg0_1_0++) {
                            inv64_dc_acc_cg0[inv64_zero_i_cg0_1_0] = 0.0f;
                        }
                        int inv64_d_addr_cg0_0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((32 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_0)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0_0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        int inv64_d_addr_cg0_1 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_1 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((48 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((48 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_1)
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
                        for (int inv64_i_cg0_0 = 0; inv64_i_cg0_0 < 8; inv64_i_cg0_0++) {
                            inv64_dc_acc_cg0[inv64_i_cg0_0] = inv64_dc_acc_cg0[inv64_i_cg0_0] * -1.0f;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_dc_acc_cg0[_lp*2 + 0], inv64_dc_acc_cg0[_lp*2+1 + 0]));
                            inv64_dc_bf16_0_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int inv64_zero_i_cg0_1_1 = 0; inv64_zero_i_cg0_1_1 < 8; inv64_zero_i_cg0_1_1++) {
                            inv64_dc_acc_cg0[inv64_zero_i_cg0_1_1] = 0.0f;
                        }
                        int inv64_d_addr_cg0_2 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (32 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_2 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_2)
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_c_frag_cg0[0]), "=r"(inv64_c_frag_cg0[1]), "=r"(inv64_c_frag_cg0[2]), "=r"(inv64_c_frag_cg0[3])
                            : "r"(inv64_c_addr_cg0_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[0]), "+f"(inv64_dc_acc_cg0[1]), "+f"(inv64_dc_acc_cg0[2]), "+f"(inv64_dc_acc_cg0[3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[0]), "r"(inv64_c_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_dc_acc_cg0[4]), "+f"(inv64_dc_acc_cg0[(4) + 1]), "+f"(inv64_dc_acc_cg0[(4) + 2]), "+f"(inv64_dc_acc_cg0[(4) + 3])
                            : "r"(inv64_d_frag_cg0[0]), "r"(inv64_d_frag_cg0[1]), "r"(inv64_d_frag_cg0[2]), "r"(inv64_d_frag_cg0[3]), "r"(inv64_c_frag_cg0[2]), "r"(inv64_c_frag_cg0[(2) + 1]));
                        int inv64_d_addr_cg0_3 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (48 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        int inv64_c_addr_cg0_3 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((48 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((48 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_d_frag_cg0[0]), "=r"(inv64_d_frag_cg0[1]), "=r"(inv64_d_frag_cg0[2]), "=r"(inv64_d_frag_cg0[3])
                            : "r"(inv64_d_addr_cg0_3)
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
                        #pragma unroll
                        for (int inv64_i_cg0_1 = 0; inv64_i_cg0_1 < 8; inv64_i_cg0_1++) {
                            inv64_dc_acc_cg0[inv64_i_cg0_1] = inv64_dc_acc_cg0[inv64_i_cg0_1] * -1.0f;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_dc_acc_cg0[_lp*2 + 0], inv64_dc_acc_cg0[_lp*2+1 + 0]));
                            inv64_dc_bf16_1_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int inv64_a_frag_cg0[4];
                        float inv64_o_acc_cg0[8];
                        #pragma unroll
                        for (int inv64_zero_i_cg0_1_2 = 0; inv64_zero_i_cg0_1_2 < 8; inv64_zero_i_cg0_1_2++) {
                            inv64_o_acc_cg0[inv64_zero_i_cg0_1_2] = 0.0f;
                        }
                        int inv64_a_addr_cg0_0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)(inv64_lane_row_cg0 * 128 + inv64_lane_col_cg0 * 2 ^ (inv64_lane_row_cg0 * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_0)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                            : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                            : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        int inv64_a_addr_cg0_1 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_1)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                            : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                            : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_o_acc_cg0[_lp*2 + 0], inv64_o_acc_cg0[_lp*2+1 + 0]));
                            inv64_o_bf16_0_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int inv64_zero_i_cg0_1_3 = 0; inv64_zero_i_cg0_1_3 < 8; inv64_zero_i_cg0_1_3++) {
                            inv64_o_acc_cg0[inv64_zero_i_cg0_1_3] = 0.0f;
                        }
                        int inv64_a_addr_cg0_2 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)(inv64_lane_row_cg0 * 128 + (16 + inv64_lane_col_cg0) * 2 ^ (inv64_lane_row_cg0 * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_2)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                            : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                            : "r"(inv64_dc_bf16_0_cg0[0]), "r"(inv64_dc_bf16_0_cg0[1]), "r"(inv64_dc_bf16_0_cg0[2]), "r"(inv64_dc_bf16_0_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        int inv64_a_addr_cg0_3 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(inv64_a_frag_cg0[0]), "=r"(inv64_a_frag_cg0[1]), "=r"(inv64_a_frag_cg0[2]), "=r"(inv64_a_frag_cg0[3])
                            : "r"(inv64_a_addr_cg0_3)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[0]), "+f"(inv64_o_acc_cg0[1]), "+f"(inv64_o_acc_cg0[2]), "+f"(inv64_o_acc_cg0[3])
                            : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[0]), "r"(inv64_a_frag_cg0[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(inv64_o_acc_cg0[4]), "+f"(inv64_o_acc_cg0[(4) + 1]), "+f"(inv64_o_acc_cg0[(4) + 2]), "+f"(inv64_o_acc_cg0[(4) + 3])
                            : "r"(inv64_dc_bf16_1_cg0[0]), "r"(inv64_dc_bf16_1_cg0[1]), "r"(inv64_dc_bf16_1_cg0[2]), "r"(inv64_dc_bf16_1_cg0[3]), "r"(inv64_a_frag_cg0[2]), "r"(inv64_a_frag_cg0[(2) + 1]));
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv64_o_acc_cg0[_lp*2 + 0], inv64_o_acc_cg0[_lp*2+1 + 0]));
                            inv64_o_bf16_1_cg0[_lp] = *(uint32_t*)&_bf2;
                        }
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (inverse_row_cg0 < 64) {
                        int inv64_o_addr_0_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + inv64_lane_col_cg0 * 2 >> 7 & 7) << 4));
                        int inv64_o_addr_1_cg0 = (smem_ainv_rm_addr + inverse_stage_cg0 * 8192 + (unsigned int)((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 ^ ((32 + inv64_warp_y_cg0 * 16 + inv64_lane_row_cg0) * 128 + (16 + inv64_lane_col_cg0) * 2 >> 7 & 7) << 4));
                        uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)inv64_o_addr_0_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_0_cg0[3]))
                            : "memory");
                        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)inv64_o_addr_1_cg0);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[0])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[1])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[2])), "r"(*reinterpret_cast<const uint32_t*>(&inv64_o_bf16_1_cg0[3]))
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
                            {
                                float ainv_beta_scale0_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0];
                                float ainv_beta_scale0_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col0_cg0 + 1];
                                float ainv_beta_scale8_lo_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0];
                                float ainv_beta_scale8_hi_cg0 = smem_beta[beta0_elem_base + ainv_beta_col8_cg0 + 1];
                                uint32_t _bf16x2_scale_pair_11;
                                {
                                    uint32_t _bf16x2_pair_12 = ainv_beta_ld_bits_cg0[0];
                                    float _bf16x2_lo_12;
                                    float _bf16x2_hi_12;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_12) : "h"((uint16_t)(_bf16x2_pair_12 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_12) : "h"((uint16_t)(_bf16x2_pair_12 >> 16)));
                                    _bf16x2_lo_12 *= ainv_beta_scale0_lo_cg0;
                                    _bf16x2_hi_12 *= ainv_beta_scale0_hi_cg0;
                                    uint32_t _bf16x2_out_12;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_12) : "f"(_bf16x2_hi_12), "f"(_bf16x2_lo_12));
                                    _bf16x2_scale_pair_11 = _bf16x2_out_12;
                                }
                                uint32_t _bf16x2_scale_pair_13;
                                {
                                    uint32_t _bf16x2_pair_14 = ainv_beta_ld_bits_cg0[1];
                                    float _bf16x2_lo_14;
                                    float _bf16x2_hi_14;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_14) : "h"((uint16_t)(_bf16x2_pair_14 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_14) : "h"((uint16_t)(_bf16x2_pair_14 >> 16)));
                                    _bf16x2_lo_14 *= ainv_beta_scale0_lo_cg0;
                                    _bf16x2_hi_14 *= ainv_beta_scale0_hi_cg0;
                                    uint32_t _bf16x2_out_14;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_14) : "f"(_bf16x2_hi_14), "f"(_bf16x2_lo_14));
                                    _bf16x2_scale_pair_13 = _bf16x2_out_14;
                                }
                                uint32_t _bf16x2_scale_pair_15;
                                {
                                    uint32_t _bf16x2_pair_16 = ainv_beta_ld_bits_cg0[2];
                                    float _bf16x2_lo_16;
                                    float _bf16x2_hi_16;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_16) : "h"((uint16_t)(_bf16x2_pair_16 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_16) : "h"((uint16_t)(_bf16x2_pair_16 >> 16)));
                                    _bf16x2_lo_16 *= ainv_beta_scale8_lo_cg0;
                                    _bf16x2_hi_16 *= ainv_beta_scale8_hi_cg0;
                                    uint32_t _bf16x2_out_16;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_16) : "f"(_bf16x2_hi_16), "f"(_bf16x2_lo_16));
                                    _bf16x2_scale_pair_15 = _bf16x2_out_16;
                                }
                                uint32_t _bf16x2_scale_pair_17;
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
                                    _bf16x2_scale_pair_17 = _bf16x2_out_18;
                                }
                                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>((unsigned long long)(smem_ainv_addr + ainv0_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 ^ (ainv_beta_tile_row_cg0 * 128 + ainv_beta_tile_col_cg0 * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_11)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_13)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_15)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_17))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(ainv_ready_addr + (ainv0_stage) * 8);
                    {
                        mbarrier_arrive(beta_smem_empty_addr + (beta0_stage) * 8);
                    }
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
                            {
                                float ainv_beta_scale0_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1];
                                float ainv_beta_scale0_hi_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col0_cg0_1 + 1];
                                float ainv_beta_scale8_lo_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1];
                                float ainv_beta_scale8_hi_cg0_1 = smem_beta[beta1_elem_base + ainv_beta_col8_cg0_1 + 1];
                                uint32_t _bf16x2_scale_pair_20;
                                {
                                    uint32_t _bf16x2_pair_21 = ainv_beta_ld_bits_cg0_1[0];
                                    float _bf16x2_lo_21;
                                    float _bf16x2_hi_21;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_21) : "h"((uint16_t)(_bf16x2_pair_21 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_21) : "h"((uint16_t)(_bf16x2_pair_21 >> 16)));
                                    _bf16x2_lo_21 *= ainv_beta_scale0_lo_cg0_1;
                                    _bf16x2_hi_21 *= ainv_beta_scale0_hi_cg0_1;
                                    uint32_t _bf16x2_out_21;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_21) : "f"(_bf16x2_hi_21), "f"(_bf16x2_lo_21));
                                    _bf16x2_scale_pair_20 = _bf16x2_out_21;
                                }
                                uint32_t _bf16x2_scale_pair_22;
                                {
                                    uint32_t _bf16x2_pair_23 = ainv_beta_ld_bits_cg0_1[1];
                                    float _bf16x2_lo_23;
                                    float _bf16x2_hi_23;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_23) : "h"((uint16_t)(_bf16x2_pair_23 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_23) : "h"((uint16_t)(_bf16x2_pair_23 >> 16)));
                                    _bf16x2_lo_23 *= ainv_beta_scale0_lo_cg0_1;
                                    _bf16x2_hi_23 *= ainv_beta_scale0_hi_cg0_1;
                                    uint32_t _bf16x2_out_23;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_23) : "f"(_bf16x2_hi_23), "f"(_bf16x2_lo_23));
                                    _bf16x2_scale_pair_22 = _bf16x2_out_23;
                                }
                                uint32_t _bf16x2_scale_pair_24;
                                {
                                    uint32_t _bf16x2_pair_25 = ainv_beta_ld_bits_cg0_1[2];
                                    float _bf16x2_lo_25;
                                    float _bf16x2_hi_25;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_25) : "h"((uint16_t)(_bf16x2_pair_25 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_25) : "h"((uint16_t)(_bf16x2_pair_25 >> 16)));
                                    _bf16x2_lo_25 *= ainv_beta_scale8_lo_cg0_1;
                                    _bf16x2_hi_25 *= ainv_beta_scale8_hi_cg0_1;
                                    uint32_t _bf16x2_out_25;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_25) : "f"(_bf16x2_hi_25), "f"(_bf16x2_lo_25));
                                    _bf16x2_scale_pair_24 = _bf16x2_out_25;
                                }
                                uint32_t _bf16x2_scale_pair_26;
                                {
                                    uint32_t _bf16x2_pair_27 = ainv_beta_ld_bits_cg0_1[3];
                                    float _bf16x2_lo_27;
                                    float _bf16x2_hi_27;
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_27) : "h"((uint16_t)(_bf16x2_pair_27 & 0xFFFFu)));
                                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_27) : "h"((uint16_t)(_bf16x2_pair_27 >> 16)));
                                    _bf16x2_lo_27 *= ainv_beta_scale8_lo_cg0_1;
                                    _bf16x2_hi_27 *= ainv_beta_scale8_hi_cg0_1;
                                    uint32_t _bf16x2_out_27;
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_27) : "f"(_bf16x2_hi_27), "f"(_bf16x2_lo_27));
                                    _bf16x2_scale_pair_26 = _bf16x2_out_27;
                                }
                                uint32_t _stmatrix_addr_28 = static_cast<uint32_t>((unsigned long long)(smem_ainv_addr + ainv1_stage * 8192 + (unsigned int)(ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 ^ (ainv_beta_tile_row_cg0_1 * 128 + ainv_beta_tile_col_cg0_1 * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_20)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_22)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_24)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_scale_pair_26))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(ainv_ready_addr + (ainv1_stage) * 8);
                    {
                        mbarrier_arrive(beta_smem_empty_addr + (beta1_stage) * 8);
                    }
                    unsigned int qk0_stage = qk_cg0_stage;
                    mbarrier_wait(qk_smem_empty_addr + (qk_cg0_stage) * 8, _phase_qk_smem_empty);
                    qk_cg0_stage += 1;
                    if (qk_cg0_stage == 2) { qk_cg0_stage = 0; _phase_qk_smem_empty ^= 1; }
                    unsigned int qk1_stage = qk_cg0_stage;
                    mbarrier_wait(qk_smem_empty_addr + (qk_cg0_stage) * 8, _phase_qk_smem_empty);
                    qk_cg0_stage += 1;
                    if (qk_cg0_stage == 2) { qk_cg0_stage = 0; _phase_qk_smem_empty ^= 1; }
                    mbarrier_wait(cg0_shared_acc_full_addr, 1);
                    int qk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[31]))
                        : "r"(qk_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int qk_stsm_row_cg0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                    int qk_stsm_col_lane_cg0 = (lane & 16) / 2;
                    #pragma unroll
                    for (int qk_repeat_cg0_4 = 0; qk_repeat_cg0_4 < 4; qk_repeat_cg0_4++) {
                        const int qk_j0_cg0_2 = qk_repeat_cg0_4 * 8;
                        float qk_scaled_cg0[8];
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(qk_scaled_cg0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_1 + qk_j0_cg0_2))[_ls], reinterpret_cast<const float2*>((qk_transfer0_cg0 + qk_j0_cg0_2))[_ls]);
                        const float2 _scale2_29 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qk_scaled_cg0)[_ls], _scale2_29);
                        unsigned int kk_scaled_cg0_bf16_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_scaled_cg0[_lp*2 + 0], qk_scaled_cg0[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_30 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk0_stage * 8192 + (unsigned int)(qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_4 * 16 + qk_stsm_col_lane_cg0) * 2 ^ (qk_stsm_row_cg0 * 128 + (qk_repeat_cg0_4 * 16 + qk_stsm_col_lane_cg0) * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_3[3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(cg0_shared_acc_empty_addr);
                    mbarrier_arrive(qk_ready_addr + (qk0_stage) * 8);
                    mbarrier_wait(cg0_shared_acc_full_addr + 8, 1);
                    int qk_addr_0 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[31]))
                        : "r"(qk_addr_0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int qk_stsm_row_cg0_0 = qk_warp_row_base_cg0 + (lane & 7) + (lane & 8);
                    int qk_stsm_col_lane_cg0_0 = (lane & 16) / 2;
                    #pragma unroll
                    for (int qk_repeat_cg0_5 = 0; qk_repeat_cg0_5 < 4; qk_repeat_cg0_5++) {
                        const int qk_j0_cg0_3 = qk_repeat_cg0_5 * 8;
                        float qk_scaled_cg0_1[8];
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            reinterpret_cast<float2*>(qk_scaled_cg0_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((tmem_load_2 + qk_j0_cg0_3))[_ls], reinterpret_cast<const float2*>((qk_transfer1_cg0 + qk_j0_cg0_3))[_ls]);
                        const float2 _scale2_31 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(qk_scaled_cg0_1)[_ls], _scale2_31);
                        unsigned int kk_scaled_cg0_bf16_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qk_scaled_cg0_1[_lp*2 + 0], qk_scaled_cg0_1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t _stmatrix_addr_32 = static_cast<uint32_t>((unsigned long long)(smem_qk_addr + qk1_stage * 8192 + (unsigned int)(qk_stsm_row_cg0_0 * 128 + (qk_repeat_cg0_5 * 16 + qk_stsm_col_lane_cg0_0) * 2 ^ (qk_stsm_row_cg0_0 * 128 + (qk_repeat_cg0_5 * 16 + qk_stsm_col_lane_cg0_0) * 2 >> 7 & 7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_4[3]))
                            : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                    mbarrier_arrive(qk_ready_addr + (qk1_stage) * 8);
                    cg0_pairs += 1;
                }
            }
        }
    }
    // ---- Role: compute_group_1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 248;");
        { // compute_group_1_main
            unsigned int gate_k_cg1 = 0;
            unsigned int v_cg1_stage = 0;
            unsigned int v_cg1_phase = 0;
            unsigned int o_cg1_stage = 0;
            unsigned int _phase_initial_state_loaded_0 = 0;
            unsigned int _phase_load_gate_full_1 = 0;
            unsigned int _phase_kv_acc_full_0 = 0;
            unsigned int _phase_load_v_full = 0;
            unsigned int _phase_cg1_shared_acc_full_0 = 0;
            unsigned int _phase_q_state_acc_full_0 = 0;
            unsigned int _phase_o_smem_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int num_o_heads_1 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx_1 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_1 >> (unsigned int)NUM_O_HEADS_LOG2 : tile_1 / (unsigned int)num_o_heads_1);
                int head_idx_1 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_1 & (unsigned int)(num_o_heads_1 - 1) : tile_1 % (unsigned int)num_o_heads_1);
                int qk_head_idx_1 = ((IS_GQA != 0) ? head_idx_1 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_v_heads / num_q_heads)));
                int v_head_idx_1 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_q_heads / num_v_heads)) : head_idx_1);
                int raw_batch_start_1 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_1] : cu_seqlens[batch_idx_1]);
                int raw_batch_end_1 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_1 + 1)] : cu_seqlens[batch_idx_1 + 1]);
                int batch_start_floor_1 = ((raw_batch_start_1 > 0) ? raw_batch_start_1 : 0);
                int batch_start_1 = ((batch_start_floor_1 < total_tokens) ? batch_start_floor_1 : total_tokens);
                int batch_end_floor_1 = ((raw_batch_end_1 > batch_start_1) ? raw_batch_end_1 : batch_start_1);
                int batch_end_1 = ((batch_end_floor_1 < total_tokens) ? batch_end_floor_1 : total_tokens);
                int seqlen_b_1 = batch_end_1 - batch_start_1;
                int num_pairs_b_1 = (seqlen_b_1 + 128 - 1) / 128;
                int num_chunks_b_1 = num_pairs_b_1 * 2;
                int value_split_idx_cg1 = 0;
                int state_slot_cg1 = batch_idx_1;
                long long state_head_offset_cg1 = (long long)head_idx_1 * 16384ll;
                long long initial_state_head_base_cg1 = (long long)state_slot_cg1 * initial_state_stride_slot + state_head_offset_cg1;
                long long output_state_head_base_cg1 = (long long)state_slot_cg1 * output_state_stride_slot + state_head_offset_cg1;
                if (USE_INITIAL_STATE != 0 && num_chunks_b_1 > 0) {
                    #if !CAKE_GDN_STATE_VEC_ALIGNED
                    bool init_vec_ok = ((unsigned long long)initial_state + (unsigned long long)initial_state_head_base_cg1 * 2ull & 3ull) == 0ull;
                    #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                    int warp_in_wg = warp % 4;
                    int state_tmem_row_base_init = warp_in_wg * 32 << 16;
                    int warp_id_in_role_1 = warp - 4;
                    int state_warp_init = warp_id_in_role_1;
                    #pragma unroll 1
                    for (int state_lane_half_init = 0; state_lane_half_init < 2; state_lane_half_init++) {
                        int state_row_top_init = state_warp_init * 32 + state_lane_half_init * 16 + lane / 4;
                        long long state_base_top_init = initial_state_head_base_cg1 + (long long)state_row_top_init * 128ll;
                        long long state_base_bot_init = state_base_top_init + 1024ll;
                        #pragma unroll
                        for (int state_col_half_init = 0; state_col_half_init < 2; state_col_half_init++) {
                            float state_init_frag[32];
                            #if CAKE_GDN_STATE_VEC_ALIGNED
                            #pragma unroll
                            for (int state_col_group_init = 0; state_col_group_init < 8; state_col_group_init++) {
                                int state_col_pair_init = state_col_half_init * 64 + state_col_group_init * 8 + (lane & 3) * 2;
                                const int state_reg_base_init = state_col_group_init * 4;
                                {
                                    uint32_t _bf16x2_bits_0;
                                    _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(initial_state + state_base_top_init + (long long)state_col_pair_init);
                                    (&state_init_frag[state_reg_base_init])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) << 16);
                                    (&state_init_frag[state_reg_base_init])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) & 0xffff0000u);
                                }
                                {
                                    uint32_t _bf16x2_bits_1;
                                    _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(initial_state + state_base_bot_init + (long long)state_col_pair_init);
                                    (&state_init_frag[state_reg_base_init + 2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) << 16);
                                    (&state_init_frag[state_reg_base_init + 2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) & 0xffff0000u);
                                }
                            }
                            #else  // CAKE_GDN_STATE_VEC_ALIGNED
                            if (__builtin_expect(init_vec_ok, 1)) {
                                #pragma unroll
                                for (int state_col_group_init = 0; state_col_group_init < 8; state_col_group_init++) {
                                    int state_col_pair_init = state_col_half_init * 64 + state_col_group_init * 8 + (lane & 3) * 2;
                                    const int state_reg_base_init = state_col_group_init * 4;
                                    {
                                        uint32_t _bf16x2_bits_0;
                                        asm volatile("ld.global.nc.b32 %0, [%1];"
                                            : "=r"(_bf16x2_bits_0) : "l"((const void*)(initial_state + (state_base_top_init + (long long)state_col_pair_init))));
                                        (&state_init_frag[state_reg_base_init])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) << 16);
                                        (&state_init_frag[state_reg_base_init])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_0) & 0xffff0000u);
                                    }
                                    {
                                        uint32_t _bf16x2_bits_1;
                                        asm volatile("ld.global.nc.b32 %0, [%1];"
                                            : "=r"(_bf16x2_bits_1) : "l"((const void*)(initial_state + (state_base_bot_init + (long long)state_col_pair_init))));
                                        (&state_init_frag[state_reg_base_init + 2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) << 16);
                                        (&state_init_frag[state_reg_base_init + 2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_1) & 0xffff0000u);
                                    }
                                }
                            } else {
                                #pragma unroll
                                for (int state_col_group_init_1 = 0; state_col_group_init_1 < 8; state_col_group_init_1++) {
                                    int state_col_pair_init_1 = state_col_half_init * 64 + state_col_group_init_1 * 8 + (lane & 3) * 2;
                                    const int state_reg_base_init_1 = state_col_group_init_1 * 4;
                                    {
                                        uint32_t _bf16_bits_2;
                                        asm volatile("ld.global.nc.u16 %0, [%1];"
                                            : "=r"(_bf16_bits_2) : "l"((const void*)(initial_state + (state_base_top_init + (long long)state_col_pair_init_1))));
                                        state_init_frag[state_reg_base_init_1] = __uint_as_float(_bf16_bits_2 << 16);
                                    }
                                    {
                                        uint32_t _bf16_bits_3;
                                        asm volatile("ld.global.nc.u16 %0, [%1];"
                                            : "=r"(_bf16_bits_3) : "l"((const void*)(initial_state + (state_base_top_init + (long long)state_col_pair_init_1 + 1ll))));
                                        state_init_frag[state_reg_base_init_1 + 1] = __uint_as_float(_bf16_bits_3 << 16);
                                    }
                                    {
                                        uint32_t _bf16_bits_4;
                                        asm volatile("ld.global.nc.u16 %0, [%1];"
                                            : "=r"(_bf16_bits_4) : "l"((const void*)(initial_state + (state_base_bot_init + (long long)state_col_pair_init_1))));
                                        state_init_frag[state_reg_base_init_1 + 2] = __uint_as_float(_bf16_bits_4 << 16);
                                    }
                                    {
                                        uint32_t _bf16_bits_5;
                                        asm volatile("ld.global.nc.u16 %0, [%1];"
                                            : "=r"(_bf16_bits_5) : "l"((const void*)(initial_state + (state_base_bot_init + (long long)state_col_pair_init_1 + 1ll))));
                                        state_init_frag[state_reg_base_init_1 + 3] = __uint_as_float(_bf16_bits_5 << 16);
                                    }
                                }
                            }
                            #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                            int state_init_addr = taddr + (unsigned int)(state_tmem_row_base_init + (state_lane_half_init * 16 << 16)) + (unsigned int)(state_col_half_init * 64);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_init_addr), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[3])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[4])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[5])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[6])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[7])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[8])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[9])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[10])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[11])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[12])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[13])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[14])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[15])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[16])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[17])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[18])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[19])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[20])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[21])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[22])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[23])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[24])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[25])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[26])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[27])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[28])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[29])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[30])), "r"(*reinterpret_cast<const uint32_t*>(&state_init_frag[31])));
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
                        {
                            mbarrier_wait(load_gate_full_addr + (gate_k_cg1) * 8, _phase_load_gate_full_1);
                        }
                        int gate_cg1_elem_base = gate_k_cg1 * 64;
                        int warp_in_wg_1 = warp % 4;
                        int tmem_row_base_v = warp_in_wg_1 * 32 << 16;
                        int lane_quad_cg1 = lane & 3;
                        float chunk_decay_cg1 = ((UNIT_GATES != 0) ? 1.0f : smem_cumprod[gate_cg1_elem_base + 64 - 1]);
                        mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                        _phase_kv_acc_full_0 ^= 1;
                        int state_addr_0 = taddr + (unsigned int)tmem_row_base_v;
                        int state_addr_1 = state_addr_0 + 32;
                        int state_addr_2 = state_addr_0 + 64;
                        int state_addr_3 = state_addr_0 + 96;
                        float _tmem_load_17[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_17[0]), "=f"(_tmem_load_17[1]), "=f"(_tmem_load_17[2]), "=f"(_tmem_load_17[3]), "=f"(_tmem_load_17[4]), "=f"(_tmem_load_17[5]), "=f"(_tmem_load_17[6]), "=f"(_tmem_load_17[7]), "=f"(_tmem_load_17[8]), "=f"(_tmem_load_17[9]), "=f"(_tmem_load_17[10]), "=f"(_tmem_load_17[11]), "=f"(_tmem_load_17[12]), "=f"(_tmem_load_17[13]), "=f"(_tmem_load_17[14]), "=f"(_tmem_load_17[15]), "=f"(_tmem_load_17[16]), "=f"(_tmem_load_17[17]), "=f"(_tmem_load_17[18]), "=f"(_tmem_load_17[19]), "=f"(_tmem_load_17[20]), "=f"(_tmem_load_17[21]), "=f"(_tmem_load_17[22]), "=f"(_tmem_load_17[23]), "=f"(_tmem_load_17[24]), "=f"(_tmem_load_17[25]), "=f"(_tmem_load_17[26]), "=f"(_tmem_load_17[27]), "=f"(_tmem_load_17[28]), "=f"(_tmem_load_17[29]), "=f"(_tmem_load_17[30]), "=f"(_tmem_load_17[31])
                            : "r"(state_addr_0));
                        float _tmem_load_18[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_18[0]), "=f"(_tmem_load_18[1]), "=f"(_tmem_load_18[2]), "=f"(_tmem_load_18[3]), "=f"(_tmem_load_18[4]), "=f"(_tmem_load_18[5]), "=f"(_tmem_load_18[6]), "=f"(_tmem_load_18[7]), "=f"(_tmem_load_18[8]), "=f"(_tmem_load_18[9]), "=f"(_tmem_load_18[10]), "=f"(_tmem_load_18[11]), "=f"(_tmem_load_18[12]), "=f"(_tmem_load_18[13]), "=f"(_tmem_load_18[14]), "=f"(_tmem_load_18[15]), "=f"(_tmem_load_18[16]), "=f"(_tmem_load_18[17]), "=f"(_tmem_load_18[18]), "=f"(_tmem_load_18[19]), "=f"(_tmem_load_18[20]), "=f"(_tmem_load_18[21]), "=f"(_tmem_load_18[22]), "=f"(_tmem_load_18[23]), "=f"(_tmem_load_18[24]), "=f"(_tmem_load_18[25]), "=f"(_tmem_load_18[26]), "=f"(_tmem_load_18[27]), "=f"(_tmem_load_18[28]), "=f"(_tmem_load_18[29]), "=f"(_tmem_load_18[30]), "=f"(_tmem_load_18[31])
                            : "r"(state_addr_1));
                        float _tmem_load_19[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_19[0]), "=f"(_tmem_load_19[1]), "=f"(_tmem_load_19[2]), "=f"(_tmem_load_19[3]), "=f"(_tmem_load_19[4]), "=f"(_tmem_load_19[5]), "=f"(_tmem_load_19[6]), "=f"(_tmem_load_19[7]), "=f"(_tmem_load_19[8]), "=f"(_tmem_load_19[9]), "=f"(_tmem_load_19[10]), "=f"(_tmem_load_19[11]), "=f"(_tmem_load_19[12]), "=f"(_tmem_load_19[13]), "=f"(_tmem_load_19[14]), "=f"(_tmem_load_19[15]), "=f"(_tmem_load_19[16]), "=f"(_tmem_load_19[17]), "=f"(_tmem_load_19[18]), "=f"(_tmem_load_19[19]), "=f"(_tmem_load_19[20]), "=f"(_tmem_load_19[21]), "=f"(_tmem_load_19[22]), "=f"(_tmem_load_19[23]), "=f"(_tmem_load_19[24]), "=f"(_tmem_load_19[25]), "=f"(_tmem_load_19[26]), "=f"(_tmem_load_19[27]), "=f"(_tmem_load_19[28]), "=f"(_tmem_load_19[29]), "=f"(_tmem_load_19[30]), "=f"(_tmem_load_19[31])
                            : "r"(state_addr_2));
                        float _tmem_load_20[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_20[0]), "=f"(_tmem_load_20[1]), "=f"(_tmem_load_20[2]), "=f"(_tmem_load_20[3]), "=f"(_tmem_load_20[4]), "=f"(_tmem_load_20[5]), "=f"(_tmem_load_20[6]), "=f"(_tmem_load_20[7]), "=f"(_tmem_load_20[8]), "=f"(_tmem_load_20[9]), "=f"(_tmem_load_20[10]), "=f"(_tmem_load_20[11]), "=f"(_tmem_load_20[12]), "=f"(_tmem_load_20[13]), "=f"(_tmem_load_20[14]), "=f"(_tmem_load_20[15]), "=f"(_tmem_load_20[16]), "=f"(_tmem_load_20[17]), "=f"(_tmem_load_20[18]), "=f"(_tmem_load_20[19]), "=f"(_tmem_load_20[20]), "=f"(_tmem_load_20[21]), "=f"(_tmem_load_20[22]), "=f"(_tmem_load_20[23]), "=f"(_tmem_load_20[24]), "=f"(_tmem_load_20[25]), "=f"(_tmem_load_20[26]), "=f"(_tmem_load_20[27]), "=f"(_tmem_load_20[28]), "=f"(_tmem_load_20[29]), "=f"(_tmem_load_20[30]), "=f"(_tmem_load_20[31])
                            : "r"(state_addr_3));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int state_inp_addr_0 = taddr + 192 + (unsigned int)tmem_row_base_v;
                        int state_inp_addr_1 = state_inp_addr_0 + 16;
                        int state_inp_addr_2 = state_inp_addr_0 + 32;
                        int state_inp_addr_3 = state_inp_addr_0 + 48;
                        unsigned int kk_scaled_cg0_bf16_5[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 0], _tmem_load_17[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_0), "r"(kk_scaled_cg0_bf16_5[0]), "r"(kk_scaled_cg0_bf16_5[1]), "r"(kk_scaled_cg0_bf16_5[2]), "r"(kk_scaled_cg0_bf16_5[3]), "r"(kk_scaled_cg0_bf16_5[4]), "r"(kk_scaled_cg0_bf16_5[5]), "r"(kk_scaled_cg0_bf16_5[6]), "r"(kk_scaled_cg0_bf16_5[7]), "r"(kk_scaled_cg0_bf16_5[8]), "r"(kk_scaled_cg0_bf16_5[9]), "r"(kk_scaled_cg0_bf16_5[10]), "r"(kk_scaled_cg0_bf16_5[11]), "r"(kk_scaled_cg0_bf16_5[12]), "r"(kk_scaled_cg0_bf16_5[13]), "r"(kk_scaled_cg0_bf16_5[14]), "r"(kk_scaled_cg0_bf16_5[15]));
                        unsigned int kk_scaled_cg0_bf16_0[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_1), "r"(kk_scaled_cg0_bf16_0[0]), "r"(kk_scaled_cg0_bf16_0[1]), "r"(kk_scaled_cg0_bf16_0[2]), "r"(kk_scaled_cg0_bf16_0[3]), "r"(kk_scaled_cg0_bf16_0[4]), "r"(kk_scaled_cg0_bf16_0[5]), "r"(kk_scaled_cg0_bf16_0[6]), "r"(kk_scaled_cg0_bf16_0[7]), "r"(kk_scaled_cg0_bf16_0[8]), "r"(kk_scaled_cg0_bf16_0[9]), "r"(kk_scaled_cg0_bf16_0[10]), "r"(kk_scaled_cg0_bf16_0[11]), "r"(kk_scaled_cg0_bf16_0[12]), "r"(kk_scaled_cg0_bf16_0[13]), "r"(kk_scaled_cg0_bf16_0[14]), "r"(kk_scaled_cg0_bf16_0[15]));
                        unsigned int kk_scaled_cg0_bf16_1_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_2), "r"(kk_scaled_cg0_bf16_1_1[0]), "r"(kk_scaled_cg0_bf16_1_1[1]), "r"(kk_scaled_cg0_bf16_1_1[2]), "r"(kk_scaled_cg0_bf16_1_1[3]), "r"(kk_scaled_cg0_bf16_1_1[4]), "r"(kk_scaled_cg0_bf16_1_1[5]), "r"(kk_scaled_cg0_bf16_1_1[6]), "r"(kk_scaled_cg0_bf16_1_1[7]), "r"(kk_scaled_cg0_bf16_1_1[8]), "r"(kk_scaled_cg0_bf16_1_1[9]), "r"(kk_scaled_cg0_bf16_1_1[10]), "r"(kk_scaled_cg0_bf16_1_1[11]), "r"(kk_scaled_cg0_bf16_1_1[12]), "r"(kk_scaled_cg0_bf16_1_1[13]), "r"(kk_scaled_cg0_bf16_1_1[14]), "r"(kk_scaled_cg0_bf16_1_1[15]));
                        unsigned int kk_scaled_cg0_bf16_2_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_20[_lp*2 + 0], _tmem_load_20[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_2_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_3), "r"(kk_scaled_cg0_bf16_2_1[0]), "r"(kk_scaled_cg0_bf16_2_1[1]), "r"(kk_scaled_cg0_bf16_2_1[2]), "r"(kk_scaled_cg0_bf16_2_1[3]), "r"(kk_scaled_cg0_bf16_2_1[4]), "r"(kk_scaled_cg0_bf16_2_1[5]), "r"(kk_scaled_cg0_bf16_2_1[6]), "r"(kk_scaled_cg0_bf16_2_1[7]), "r"(kk_scaled_cg0_bf16_2_1[8]), "r"(kk_scaled_cg0_bf16_2_1[9]), "r"(kk_scaled_cg0_bf16_2_1[10]), "r"(kk_scaled_cg0_bf16_2_1[11]), "r"(kk_scaled_cg0_bf16_2_1[12]), "r"(kk_scaled_cg0_bf16_2_1[13]), "r"(kk_scaled_cg0_bf16_2_1[14]), "r"(kk_scaled_cg0_bf16_2_1[15]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(state_inp_ready_addr);
                        #if CAKE_GDN_STATE_VEC_ALIGNED
                        const float2 _scale2_2 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_17)[_ls], _scale2_2);
                        const float2 _scale2_3 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_18)[_ls], _scale2_3);
                        const float2 _scale2_4 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_19)[_ls], _scale2_4);
                        const float2 _scale2_5 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_20)[_ls], _scale2_5);
                        #else  // CAKE_GDN_STATE_VEC_ALIGNED
                        const float2 _scale2_6 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_17)[_ls], _scale2_6);
                        const float2 _scale2_7 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_18)[_ls], _scale2_7);
                        const float2 _scale2_8 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_19)[_ls], _scale2_8);
                        const float2 _scale2_9 = {chunk_decay_cg1, chunk_decay_cg1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_20)[_ls], _scale2_9);
                        #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                        tmem_st_x32_f32(state_addr_0, _tmem_load_17);
                        tmem_st_x32_f32(state_addr_1, _tmem_load_18);
                        tmem_st_x32_f32(state_addr_2, _tmem_load_19);
                        tmem_st_x32_f32(state_addr_3, _tmem_load_20);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(kv_acc_empty_addr);
                        float cg1_cumprod_frag[32];
                        float cg1_decay_scale_frag[32];
                        {
                            float last_cumsumlog_cg1 = smem_cumsumlog[gate_cg1_elem_base + 64 - 1];
                            #pragma unroll
                            for (int cg1_scale_j = 0; cg1_scale_j < 32; cg1_scale_j++) {
                                int cg1_scale_repeat = cg1_scale_j / 4;
                                int cg1_scale_reg = cg1_scale_j & 3;
                                int cg1_scale_col = cg1_scale_repeat * 8 + lane_quad_cg1 * 2 + (cg1_scale_reg & 1);
                                float cg1_scale_log = smem_cumsumlog[gate_cg1_elem_base + cg1_scale_col];
                                cg1_cumprod_frag[cg1_scale_j] = smem_cumprod[gate_cg1_elem_base + cg1_scale_col];
                                #if CAKE_GDN_STATE_VEC_ALIGNED
                                float _exp2_6 = approx_exp2(last_cumsumlog_cg1 - cg1_scale_log);
                                cg1_decay_scale_frag[cg1_scale_j] = _exp2_6;
                                #else  // CAKE_GDN_STATE_VEC_ALIGNED
                                float _exp2_10 = approx_exp2(last_cumsumlog_cg1 - cg1_scale_log);
                                cg1_decay_scale_frag[cg1_scale_j] = _exp2_10;
                                #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                            }
                        }
                        {
                            mbarrier_arrive(gate_cg1_empty_addr + (gate_k_cg1) * 8);
                            gate_k_cg1 += 1;
                            if (gate_k_cg1 == 5) { gate_k_cg1 = 0; _phase_load_gate_full_1 ^= 1; }
                        }
                        mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, _phase_load_v_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int v_stage_addr_cg1 = smem_v_addr + v_cg1_stage * 16384;
                        unsigned int v_frag_lo_cg1[16];
                        unsigned int v_frag_hi_cg1[16];
                        unsigned int v_ld_bits_cg1[4];
                        #pragma unroll
                        for (int v_frag_repeat_cg1 = 0; v_frag_repeat_cg1 < 8; v_frag_repeat_cg1++) {
                            int v_ld_mtx_cg1 = lane / 8;
                            int v_ld_token_cg1 = v_frag_repeat_cg1 * 8 + (lane & 7);
                            int warp_id_in_role_2 = warp - 4;
                            int v_ld_dv_cg1 = warp_id_in_role_2 * 32 + v_ld_mtx_cg1 * 8;
                            int v_d0_cg1 = v_ld_dv_cg1 & 63;
                            int v_d1_cg1 = v_ld_dv_cg1 / 64;
                            int v_t0_cg1 = v_ld_token_cg1 & 15;
                            int v_t1_cg1 = v_ld_token_cg1 / 16;
                            int v_ld_linear_byte_off_cg1 = (v_d0_cg1 + v_d1_cg1 * 4096 + v_t0_cg1 * 64 + v_t1_cg1 * 1024) * 2;
                            int v_ld_byte_off_cg1 = v_ld_linear_byte_off_cg1 ^ (v_ld_linear_byte_off_cg1 >> 7 & 7) << 4;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(v_ld_bits_cg1[0]), "=r"(v_ld_bits_cg1[1]), "=r"(v_ld_bits_cg1[2]), "=r"(v_ld_bits_cg1[3])
                                : "r"(v_stage_addr_cg1 + v_ld_byte_off_cg1)
                                : "memory");
                            const int v_frag_j0_cg1 = v_frag_repeat_cg1 * 2;
                            v_frag_lo_cg1[v_frag_j0_cg1] = v_ld_bits_cg1[0];
                            v_frag_lo_cg1[v_frag_j0_cg1 + 1] = v_ld_bits_cg1[1];
                            v_frag_hi_cg1[v_frag_j0_cg1] = v_ld_bits_cg1[2];
                            v_frag_hi_cg1[v_frag_j0_cg1 + 1] = v_ld_bits_cg1[3];
                        }
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        int ks_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
                        int ks_addr_hi_cg1 = ks_addr_lo_cg1 + 1048576;
                        float _tmem_load_8[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[31]))
                            : "r"(ks_addr_lo_cg1));
                        float _tmem_load_9[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[31]))
                            : "r"(ks_addr_hi_cg1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                        unsigned int kk_scaled_cg0_bf16_3_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_3_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int kk_scaled_cg0_bf16_4_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_4_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int vks_packed_lo_cg1[16];
                        unsigned int vks_packed_hi_cg1[16];
                        #pragma unroll
                        for (int frag_pair_j = 0; frag_pair_j < 16; frag_pair_j++) {
                            uint32_t bf16x2_sub;
                            asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_sub) : "r"(v_frag_lo_cg1[frag_pair_j]), "r"(kk_scaled_cg0_bf16_3_1[frag_pair_j]));
                            vks_packed_lo_cg1[frag_pair_j] = bf16x2_sub;
                            uint32_t bf16x2_sub_0;
                            asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_sub_0) : "r"(v_frag_hi_cg1[frag_pair_j]), "r"(kk_scaled_cg0_bf16_4_1[frag_pair_j]));
                            vks_packed_hi_cg1[frag_pair_j] = bf16x2_sub_0;
                        }
                        int vks_addr_lo_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
                        int vks_addr_hi_cg1 = vks_addr_lo_cg1 + 1048576;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(vks_addr_lo_cg1), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1[15])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(vks_addr_hi_cg1), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1[15])));
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(vks_ready_addr);
                        #pragma unroll
                        for (int dim_half_qs = 0; dim_half_qs < 2; dim_half_qs++) {
                            int qs_addr = taddr + 128 + (unsigned int)tmem_row_base_v + (unsigned int)(dim_half_qs * 16 << 16);
                            float _tmem_load_23[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23[31]))
                                : "r"(qs_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                            #if CAKE_GDN_STATE_VEC_ALIGNED
                            const float2 _scale2_7 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23)[_ls], _scale2_7);
                            #else  // CAKE_GDN_STATE_VEC_ALIGNED
                            const float2 _scale2_11 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23)[_ls], _scale2_11);
                            #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(qs_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23[31])));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(q_state_acc_empty_addr);
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        if (elect_sync()) {
                            mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                        }
                        v_cg1_stage += 1;
                        if (v_cg1_stage == 3) { v_cg1_stage = 0; _phase_load_v_full ^= 1; }
                        int nv_src_addr_lo_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
                        float _tmem_load_11[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[31]))
                            : "r"(nv_src_addr_lo_cg1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int kk_scaled_cg0_bf16_5_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_5_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        int nv_dst_addr_lo_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(nv_dst_addr_lo_cg1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5_1[15])));
                        int nv_src_addr_hi_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v + 1048576;
                        float _tmem_load_24[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[31]))
                            : "r"(nv_src_addr_hi_cg1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        unsigned int kk_scaled_cg0_bf16_6[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_24[_lp*2 + 0], _tmem_load_24[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        int nv_dst_addr_hi_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v + 1048576;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(nv_dst_addr_hi_cg1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(nv_ready_addr);
                        {
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag)[_ls]);
                            unsigned int kk_scaled_cg0_bf16_7[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            int decay_dst_addr_lo_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(decay_dst_addr_lo_cg1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[15])));
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_24)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag)[_ls]);
                            unsigned int kk_scaled_cg0_bf16_8[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_24[_lp*2 + 0], _tmem_load_24[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_8[_lp] = *(uint32_t*)&_bf2;
                            }
                            int decay_dst_addr_hi_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v + 1048576;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(decay_dst_addr_hi_cg1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            mbarrier_arrive(decay_v_ready_addr);
                        }
                        mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, _phase_o_smem_empty);
                        int o_stage_addr_cg1 = smem_o_addr + o_cg1_stage * 16384;
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        #pragma unroll
                        for (int dim_half_cg1 = 0; dim_half_cg1 < 2; dim_half_cg1++) {
                            int dim_half_row_cg1 = dim_half_cg1 * 16;
                            int q_state_addr = taddr + 128 + (unsigned int)tmem_row_base_v + (unsigned int)(dim_half_row_cg1 << 16);
                            float _tmem_load_26[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[31]))
                                : "r"(q_state_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            unsigned int kk_scaled_cg0_bf16_9[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_26[_lp*2 + 0], _tmem_load_26[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int token_group_cg1 = 0; token_group_cg1 < 4; token_group_cg1++) {
                                int o_mtx_idx_cg1 = lane / 8;
                                int o_row_addr_cg1 = lane & 7;
                                int warp_id_in_role_3 = warp - 4;
                                int o_dim_base_cg1 = warp_id_in_role_3 * 32 + dim_half_row_cg1 + (o_mtx_idx_cg1 & 1) * 8;
                                int o_token_base_cg1 = token_group_cg1 * 16 + o_mtx_idx_cg1 / 2 * 8;
                                int o_token_addr_cg1 = o_token_base_cg1 + o_row_addr_cg1;
                                int o_token_pair_cg1 = o_token_addr_cg1 / 2;
                                int o_token_parity_cg1 = o_token_addr_cg1 & 1;
                                int o_raw_row_cg1 = o_token_pair_cg1 + o_dim_base_cg1 / 64 * 32;
                                int o_raw_col_cg1 = (o_dim_base_cg1 & 63 ^ (o_token_pair_cg1 & 3) << 4 ^ o_token_parity_cg1 << 3) + o_token_parity_cg1 * 64;
                                int o_stsm_offset_cg1 = (o_raw_row_cg1 * 128 + o_raw_col_cg1) * 2;
                                const int o_pack_base_cg1 = token_group_cg1 * 4;
                                #if CAKE_GDN_STATE_VEC_ALIGNED
                                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1 + o_stsm_offset_cg1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 3]))
                                    : "memory");
                                #else  // CAKE_GDN_STATE_VEC_ALIGNED
                                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1 + o_stsm_offset_cg1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[o_pack_base_cg1 + 3]))
                                    : "memory");
                                #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(q_state_acc_empty_addr);
                        mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                        o_cg1_stage += 1;
                        if (o_cg1_stage == 2) { o_cg1_stage = 0; _phase_o_smem_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_1 = 1; chunk_idx_1 < active_chunks_b; chunk_idx_1++) {
                    int chunk_offset_2 = batch_start_1 + chunk_idx_1 * 64;
                    int _cg1_marker_1 = batch_idx_1 + head_idx_1 + chunk_offset_2 + batch_end_1 + checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE + ENABLE_CHECKPOINTS;
                    {
                        mbarrier_wait(load_gate_full_addr + (gate_k_cg1) * 8, _phase_load_gate_full_1);
                    }
                    int gate_cg1_elem_base_1 = gate_k_cg1 * 64;
                    int warp_in_wg_2 = warp % 4;
                    int tmem_row_base_v_1 = warp_in_wg_2 * 32 << 16;
                    int lane_quad_cg1_1 = lane & 3;
                    float chunk_decay_cg1_1 = ((UNIT_GATES != 0) ? 1.0f : smem_cumprod[gate_cg1_elem_base_1 + 64 - 1]);
                    mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                    _phase_kv_acc_full_0 ^= 1;
                    int state_addr_0_1 = taddr + (unsigned int)tmem_row_base_v_1;
                    int state_addr_1_1 = state_addr_0_1 + 32;
                    int state_addr_2_1 = state_addr_0_1 + 64;
                    int state_addr_3_1 = state_addr_0_1 + 96;
                    float _tmem_load_17_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_17_1[0]), "=f"(_tmem_load_17_1[1]), "=f"(_tmem_load_17_1[2]), "=f"(_tmem_load_17_1[3]), "=f"(_tmem_load_17_1[4]), "=f"(_tmem_load_17_1[5]), "=f"(_tmem_load_17_1[6]), "=f"(_tmem_load_17_1[7]), "=f"(_tmem_load_17_1[8]), "=f"(_tmem_load_17_1[9]), "=f"(_tmem_load_17_1[10]), "=f"(_tmem_load_17_1[11]), "=f"(_tmem_load_17_1[12]), "=f"(_tmem_load_17_1[13]), "=f"(_tmem_load_17_1[14]), "=f"(_tmem_load_17_1[15]), "=f"(_tmem_load_17_1[16]), "=f"(_tmem_load_17_1[17]), "=f"(_tmem_load_17_1[18]), "=f"(_tmem_load_17_1[19]), "=f"(_tmem_load_17_1[20]), "=f"(_tmem_load_17_1[21]), "=f"(_tmem_load_17_1[22]), "=f"(_tmem_load_17_1[23]), "=f"(_tmem_load_17_1[24]), "=f"(_tmem_load_17_1[25]), "=f"(_tmem_load_17_1[26]), "=f"(_tmem_load_17_1[27]), "=f"(_tmem_load_17_1[28]), "=f"(_tmem_load_17_1[29]), "=f"(_tmem_load_17_1[30]), "=f"(_tmem_load_17_1[31])
                        : "r"(state_addr_0_1));
                    float _tmem_load_18_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_18_1[0]), "=f"(_tmem_load_18_1[1]), "=f"(_tmem_load_18_1[2]), "=f"(_tmem_load_18_1[3]), "=f"(_tmem_load_18_1[4]), "=f"(_tmem_load_18_1[5]), "=f"(_tmem_load_18_1[6]), "=f"(_tmem_load_18_1[7]), "=f"(_tmem_load_18_1[8]), "=f"(_tmem_load_18_1[9]), "=f"(_tmem_load_18_1[10]), "=f"(_tmem_load_18_1[11]), "=f"(_tmem_load_18_1[12]), "=f"(_tmem_load_18_1[13]), "=f"(_tmem_load_18_1[14]), "=f"(_tmem_load_18_1[15]), "=f"(_tmem_load_18_1[16]), "=f"(_tmem_load_18_1[17]), "=f"(_tmem_load_18_1[18]), "=f"(_tmem_load_18_1[19]), "=f"(_tmem_load_18_1[20]), "=f"(_tmem_load_18_1[21]), "=f"(_tmem_load_18_1[22]), "=f"(_tmem_load_18_1[23]), "=f"(_tmem_load_18_1[24]), "=f"(_tmem_load_18_1[25]), "=f"(_tmem_load_18_1[26]), "=f"(_tmem_load_18_1[27]), "=f"(_tmem_load_18_1[28]), "=f"(_tmem_load_18_1[29]), "=f"(_tmem_load_18_1[30]), "=f"(_tmem_load_18_1[31])
                        : "r"(state_addr_1_1));
                    float _tmem_load_19_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_19_1[0]), "=f"(_tmem_load_19_1[1]), "=f"(_tmem_load_19_1[2]), "=f"(_tmem_load_19_1[3]), "=f"(_tmem_load_19_1[4]), "=f"(_tmem_load_19_1[5]), "=f"(_tmem_load_19_1[6]), "=f"(_tmem_load_19_1[7]), "=f"(_tmem_load_19_1[8]), "=f"(_tmem_load_19_1[9]), "=f"(_tmem_load_19_1[10]), "=f"(_tmem_load_19_1[11]), "=f"(_tmem_load_19_1[12]), "=f"(_tmem_load_19_1[13]), "=f"(_tmem_load_19_1[14]), "=f"(_tmem_load_19_1[15]), "=f"(_tmem_load_19_1[16]), "=f"(_tmem_load_19_1[17]), "=f"(_tmem_load_19_1[18]), "=f"(_tmem_load_19_1[19]), "=f"(_tmem_load_19_1[20]), "=f"(_tmem_load_19_1[21]), "=f"(_tmem_load_19_1[22]), "=f"(_tmem_load_19_1[23]), "=f"(_tmem_load_19_1[24]), "=f"(_tmem_load_19_1[25]), "=f"(_tmem_load_19_1[26]), "=f"(_tmem_load_19_1[27]), "=f"(_tmem_load_19_1[28]), "=f"(_tmem_load_19_1[29]), "=f"(_tmem_load_19_1[30]), "=f"(_tmem_load_19_1[31])
                        : "r"(state_addr_2_1));
                    float _tmem_load_20_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_20_1[0]), "=f"(_tmem_load_20_1[1]), "=f"(_tmem_load_20_1[2]), "=f"(_tmem_load_20_1[3]), "=f"(_tmem_load_20_1[4]), "=f"(_tmem_load_20_1[5]), "=f"(_tmem_load_20_1[6]), "=f"(_tmem_load_20_1[7]), "=f"(_tmem_load_20_1[8]), "=f"(_tmem_load_20_1[9]), "=f"(_tmem_load_20_1[10]), "=f"(_tmem_load_20_1[11]), "=f"(_tmem_load_20_1[12]), "=f"(_tmem_load_20_1[13]), "=f"(_tmem_load_20_1[14]), "=f"(_tmem_load_20_1[15]), "=f"(_tmem_load_20_1[16]), "=f"(_tmem_load_20_1[17]), "=f"(_tmem_load_20_1[18]), "=f"(_tmem_load_20_1[19]), "=f"(_tmem_load_20_1[20]), "=f"(_tmem_load_20_1[21]), "=f"(_tmem_load_20_1[22]), "=f"(_tmem_load_20_1[23]), "=f"(_tmem_load_20_1[24]), "=f"(_tmem_load_20_1[25]), "=f"(_tmem_load_20_1[26]), "=f"(_tmem_load_20_1[27]), "=f"(_tmem_load_20_1[28]), "=f"(_tmem_load_20_1[29]), "=f"(_tmem_load_20_1[30]), "=f"(_tmem_load_20_1[31])
                        : "r"(state_addr_3_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int state_inp_addr_0_1 = taddr + 192 + (unsigned int)tmem_row_base_v_1;
                    int state_inp_addr_1_1 = state_inp_addr_0_1 + 16;
                    int state_inp_addr_2_1 = state_inp_addr_0_1 + 32;
                    int state_inp_addr_3_1 = state_inp_addr_0_1 + 48;
                    unsigned int kk_scaled_cg0_bf16_10[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17_1[_lp*2 + 0], _tmem_load_17_1[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_10[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(state_inp_addr_0_1), "r"(kk_scaled_cg0_bf16_10[0]), "r"(kk_scaled_cg0_bf16_10[1]), "r"(kk_scaled_cg0_bf16_10[2]), "r"(kk_scaled_cg0_bf16_10[3]), "r"(kk_scaled_cg0_bf16_10[4]), "r"(kk_scaled_cg0_bf16_10[5]), "r"(kk_scaled_cg0_bf16_10[6]), "r"(kk_scaled_cg0_bf16_10[7]), "r"(kk_scaled_cg0_bf16_10[8]), "r"(kk_scaled_cg0_bf16_10[9]), "r"(kk_scaled_cg0_bf16_10[10]), "r"(kk_scaled_cg0_bf16_10[11]), "r"(kk_scaled_cg0_bf16_10[12]), "r"(kk_scaled_cg0_bf16_10[13]), "r"(kk_scaled_cg0_bf16_10[14]), "r"(kk_scaled_cg0_bf16_10[15]));
                    unsigned int kk_scaled_cg0_bf16_14[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18_1[_lp*2 + 0], _tmem_load_18_1[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_14[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(state_inp_addr_1_1), "r"(kk_scaled_cg0_bf16_14[0]), "r"(kk_scaled_cg0_bf16_14[1]), "r"(kk_scaled_cg0_bf16_14[2]), "r"(kk_scaled_cg0_bf16_14[3]), "r"(kk_scaled_cg0_bf16_14[4]), "r"(kk_scaled_cg0_bf16_14[5]), "r"(kk_scaled_cg0_bf16_14[6]), "r"(kk_scaled_cg0_bf16_14[7]), "r"(kk_scaled_cg0_bf16_14[8]), "r"(kk_scaled_cg0_bf16_14[9]), "r"(kk_scaled_cg0_bf16_14[10]), "r"(kk_scaled_cg0_bf16_14[11]), "r"(kk_scaled_cg0_bf16_14[12]), "r"(kk_scaled_cg0_bf16_14[13]), "r"(kk_scaled_cg0_bf16_14[14]), "r"(kk_scaled_cg0_bf16_14[15]));
                    unsigned int kk_scaled_cg0_bf16_15[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19_1[_lp*2 + 0], _tmem_load_19_1[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_15[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(state_inp_addr_2_1), "r"(kk_scaled_cg0_bf16_15[0]), "r"(kk_scaled_cg0_bf16_15[1]), "r"(kk_scaled_cg0_bf16_15[2]), "r"(kk_scaled_cg0_bf16_15[3]), "r"(kk_scaled_cg0_bf16_15[4]), "r"(kk_scaled_cg0_bf16_15[5]), "r"(kk_scaled_cg0_bf16_15[6]), "r"(kk_scaled_cg0_bf16_15[7]), "r"(kk_scaled_cg0_bf16_15[8]), "r"(kk_scaled_cg0_bf16_15[9]), "r"(kk_scaled_cg0_bf16_15[10]), "r"(kk_scaled_cg0_bf16_15[11]), "r"(kk_scaled_cg0_bf16_15[12]), "r"(kk_scaled_cg0_bf16_15[13]), "r"(kk_scaled_cg0_bf16_15[14]), "r"(kk_scaled_cg0_bf16_15[15]));
                    unsigned int kk_scaled_cg0_bf16_16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_20_1[_lp*2 + 0], _tmem_load_20_1[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(state_inp_addr_3_1), "r"(kk_scaled_cg0_bf16_16[0]), "r"(kk_scaled_cg0_bf16_16[1]), "r"(kk_scaled_cg0_bf16_16[2]), "r"(kk_scaled_cg0_bf16_16[3]), "r"(kk_scaled_cg0_bf16_16[4]), "r"(kk_scaled_cg0_bf16_16[5]), "r"(kk_scaled_cg0_bf16_16[6]), "r"(kk_scaled_cg0_bf16_16[7]), "r"(kk_scaled_cg0_bf16_16[8]), "r"(kk_scaled_cg0_bf16_16[9]), "r"(kk_scaled_cg0_bf16_16[10]), "r"(kk_scaled_cg0_bf16_16[11]), "r"(kk_scaled_cg0_bf16_16[12]), "r"(kk_scaled_cg0_bf16_16[13]), "r"(kk_scaled_cg0_bf16_16[14]), "r"(kk_scaled_cg0_bf16_16[15]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(state_inp_ready_addr);
                    #if CAKE_GDN_STATE_VEC_ALIGNED
                    const float2 _scale2_9 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_17_1)[_ls], _scale2_9);
                    const float2 _scale2_10 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_18_1)[_ls], _scale2_10);
                    const float2 _scale2_11 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_19_1)[_ls], _scale2_11);
                    const float2 _scale2_12 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_20_1)[_ls], _scale2_12);
                    #else  // CAKE_GDN_STATE_VEC_ALIGNED
                    const float2 _scale2_13 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_17_1)[_ls], _scale2_13);
                    const float2 _scale2_14 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_18_1)[_ls], _scale2_14);
                    const float2 _scale2_15 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_19_1)[_ls], _scale2_15);
                    const float2 _scale2_16 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_20_1)[_ls], _scale2_16);
                    #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                    tmem_st_x32_f32(state_addr_0_1, _tmem_load_17_1);
                    tmem_st_x32_f32(state_addr_1_1, _tmem_load_18_1);
                    tmem_st_x32_f32(state_addr_2_1, _tmem_load_19_1);
                    tmem_st_x32_f32(state_addr_3_1, _tmem_load_20_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(kv_acc_empty_addr);
                    float cg1_cumprod_frag_1[32];
                    float cg1_decay_scale_frag_1[32];
                    {
                        float last_cumsumlog_cg1_1 = smem_cumsumlog[gate_cg1_elem_base_1 + 64 - 1];
                        #pragma unroll
                        for (int cg1_scale_j_1 = 0; cg1_scale_j_1 < 32; cg1_scale_j_1++) {
                            int cg1_scale_repeat_1 = cg1_scale_j_1 / 4;
                            int cg1_scale_reg_1 = cg1_scale_j_1 & 3;
                            int cg1_scale_col_1 = cg1_scale_repeat_1 * 8 + lane_quad_cg1_1 * 2 + (cg1_scale_reg_1 & 1);
                            float cg1_scale_log_1 = smem_cumsumlog[gate_cg1_elem_base_1 + cg1_scale_col_1];
                            cg1_cumprod_frag_1[cg1_scale_j_1] = smem_cumprod[gate_cg1_elem_base_1 + cg1_scale_col_1];
                            #if CAKE_GDN_STATE_VEC_ALIGNED
                            float _exp2_13 = approx_exp2(last_cumsumlog_cg1_1 - cg1_scale_log_1);
                            cg1_decay_scale_frag_1[cg1_scale_j_1] = _exp2_13;
                            #else  // CAKE_GDN_STATE_VEC_ALIGNED
                            float _exp2_17 = approx_exp2(last_cumsumlog_cg1_1 - cg1_scale_log_1);
                            cg1_decay_scale_frag_1[cg1_scale_j_1] = _exp2_17;
                            #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                        }
                    }
                    {
                        mbarrier_arrive(gate_cg1_empty_addr + (gate_k_cg1) * 8);
                        gate_k_cg1 += 1;
                        if (gate_k_cg1 == 5) { gate_k_cg1 = 0; _phase_load_gate_full_1 ^= 1; }
                    }
                    mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, _phase_load_v_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int v_stage_addr_cg1_1 = smem_v_addr + v_cg1_stage * 16384;
                    unsigned int v_frag_lo_cg1_1[16];
                    unsigned int v_frag_hi_cg1_1[16];
                    unsigned int v_ld_bits_cg1_1[4];
                    #pragma unroll
                    for (int v_frag_repeat_cg1_1 = 0; v_frag_repeat_cg1_1 < 8; v_frag_repeat_cg1_1++) {
                        int v_ld_mtx_cg1_1 = lane / 8;
                        int v_ld_token_cg1_1 = v_frag_repeat_cg1_1 * 8 + (lane & 7);
                        int warp_id_in_role_4 = warp - 4;
                        int v_ld_dv_cg1_1 = warp_id_in_role_4 * 32 + v_ld_mtx_cg1_1 * 8;
                        int v_d0_cg1_1 = v_ld_dv_cg1_1 & 63;
                        int v_d1_cg1_1 = v_ld_dv_cg1_1 / 64;
                        int v_t0_cg1_1 = v_ld_token_cg1_1 & 15;
                        int v_t1_cg1_1 = v_ld_token_cg1_1 / 16;
                        int v_ld_linear_byte_off_cg1_1 = (v_d0_cg1_1 + v_d1_cg1_1 * 4096 + v_t0_cg1_1 * 64 + v_t1_cg1_1 * 1024) * 2;
                        int v_ld_byte_off_cg1_1 = v_ld_linear_byte_off_cg1_1 ^ (v_ld_linear_byte_off_cg1_1 >> 7 & 7) << 4;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(v_ld_bits_cg1_1[0]), "=r"(v_ld_bits_cg1_1[1]), "=r"(v_ld_bits_cg1_1[2]), "=r"(v_ld_bits_cg1_1[3])
                            : "r"(v_stage_addr_cg1_1 + v_ld_byte_off_cg1_1)
                            : "memory");
                        const int v_frag_j0_cg1_1 = v_frag_repeat_cg1_1 * 2;
                        v_frag_lo_cg1_1[v_frag_j0_cg1_1] = v_ld_bits_cg1_1[0];
                        v_frag_lo_cg1_1[v_frag_j0_cg1_1 + 1] = v_ld_bits_cg1_1[1];
                        v_frag_hi_cg1_1[v_frag_j0_cg1_1] = v_ld_bits_cg1_1[2];
                        v_frag_hi_cg1_1[v_frag_j0_cg1_1 + 1] = v_ld_bits_cg1_1[3];
                    }
                    mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                    _phase_cg1_shared_acc_full_0 ^= 1;
                    int ks_addr_lo_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                    int ks_addr_hi_cg1_1 = ks_addr_lo_cg1_1 + 1048576;
                    float _tmem_load_21[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_21[31]))
                        : "r"(ks_addr_lo_cg1_1));
                    float _tmem_load_22[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_22[31]))
                        : "r"(ks_addr_hi_cg1_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(cg1_shared_acc_empty_addr);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_21)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_22)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                    unsigned int kk_scaled_cg0_bf16_17[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_21[_lp*2 + 0], _tmem_load_21[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_17[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int kk_scaled_cg0_bf16_18[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_22[_lp*2 + 0], _tmem_load_22[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_18[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int vks_packed_lo_cg1_1[16];
                    unsigned int vks_packed_hi_cg1_1[16];
                    #pragma unroll
                    for (int frag_pair_j_1 = 0; frag_pair_j_1 < 16; frag_pair_j_1++) {
                        uint32_t bf16x2_sub_1;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_sub_1) : "r"(v_frag_lo_cg1_1[frag_pair_j_1]), "r"(kk_scaled_cg0_bf16_17[frag_pair_j_1]));
                        vks_packed_lo_cg1_1[frag_pair_j_1] = bf16x2_sub_1;
                        uint32_t bf16x2_sub_1_1;
                        asm volatile("sub.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_sub_1_1) : "r"(v_frag_hi_cg1_1[frag_pair_j_1]), "r"(kk_scaled_cg0_bf16_18[frag_pair_j_1]));
                        vks_packed_hi_cg1_1[frag_pair_j_1] = bf16x2_sub_1_1;
                    }
                    int vks_addr_lo_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                    int vks_addr_hi_cg1_1 = vks_addr_lo_cg1_1 + 1048576;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(vks_addr_lo_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_lo_cg1_1[15])));
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(vks_addr_hi_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&vks_packed_hi_cg1_1[15])));
                    mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                    _phase_q_state_acc_full_0 ^= 1;
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(vks_ready_addr);
                    #pragma unroll
                    for (int dim_half_qs_1 = 0; dim_half_qs_1 < 2; dim_half_qs_1++) {
                        int qs_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1 + (unsigned int)(dim_half_qs_1 * 16 << 16);
                        float _tmem_load_23_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_23_1[31]))
                            : "r"(qs_addr_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23_1)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                        #if CAKE_GDN_STATE_VEC_ALIGNED
                        const float2 _scale2_14 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23_1)[_ls], _scale2_14);
                        #else  // CAKE_GDN_STATE_VEC_ALIGNED
                        const float2 _scale2_18 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23_1)[_ls], _scale2_18);
                        #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(qs_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_23_1[31])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(q_state_acc_empty_addr);
                    mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                    _phase_cg1_shared_acc_full_0 ^= 1;
                    if (elect_sync()) {
                        mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                    }
                    v_cg1_stage += 1;
                    if (v_cg1_stage == 3) { v_cg1_stage = 0; _phase_load_v_full ^= 1; }
                    int nv_src_addr_lo_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                    float _tmem_load_24_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24_1[31]))
                        : "r"(nv_src_addr_lo_cg1_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int kk_scaled_cg0_bf16_19[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_24_1[_lp*2 + 0], _tmem_load_24_1[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_19[_lp] = *(uint32_t*)&_bf2;
                    }
                    int nv_dst_addr_lo_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(nv_dst_addr_lo_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_19[15])));
                    int nv_src_addr_hi_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1 + 1048576;
                    float _tmem_load_25[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[31]))
                        : "r"(nv_src_addr_hi_cg1_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(cg1_shared_acc_empty_addr);
                    unsigned int kk_scaled_cg0_bf16_20[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_25[_lp*2 + 0], _tmem_load_25[_lp*2+1 + 0]));
                        kk_scaled_cg0_bf16_20[_lp] = *(uint32_t*)&_bf2;
                    }
                    int nv_dst_addr_hi_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1 + 1048576;
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x128b.x8.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(nv_dst_addr_hi_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_20[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(nv_ready_addr);
                    {
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_24_1)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag_1)[_ls]);
                        unsigned int kk_scaled_cg0_bf16_21[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_24_1[_lp*2 + 0], _tmem_load_24_1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_21[_lp] = *(uint32_t*)&_bf2;
                        }
                        int decay_dst_addr_lo_cg1_1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(decay_dst_addr_lo_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_21[15])));
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_25)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag_1)[_ls]);
                        unsigned int kk_scaled_cg0_bf16_22[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_25[_lp*2 + 0], _tmem_load_25[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_22[_lp] = *(uint32_t*)&_bf2;
                        }
                        int decay_dst_addr_hi_cg1_1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1 + 1048576;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(decay_dst_addr_hi_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_22[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(decay_v_ready_addr);
                    }
                    mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, _phase_o_smem_empty);
                    int o_stage_addr_cg1_1 = smem_o_addr + o_cg1_stage * 16384;
                    mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                    _phase_q_state_acc_full_0 ^= 1;
                    #pragma unroll
                    for (int dim_half_cg1_1 = 0; dim_half_cg1_1 < 2; dim_half_cg1_1++) {
                        int dim_half_row_cg1_1 = dim_half_cg1_1 * 16;
                        int q_state_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1 + (unsigned int)(dim_half_row_cg1_1 << 16);
                        float _tmem_load_26_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26_1[31]))
                            : "r"(q_state_addr_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        unsigned int kk_scaled_cg0_bf16_23[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_26_1[_lp*2 + 0], _tmem_load_26_1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_23[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int token_group_cg1_1 = 0; token_group_cg1_1 < 4; token_group_cg1_1++) {
                            int o_mtx_idx_cg1_1 = lane / 8;
                            int o_row_addr_cg1_1 = lane & 7;
                            int warp_id_in_role_5 = warp - 4;
                            int o_dim_base_cg1_1 = warp_id_in_role_5 * 32 + dim_half_row_cg1_1 + (o_mtx_idx_cg1_1 & 1) * 8;
                            int o_token_base_cg1_1 = token_group_cg1_1 * 16 + o_mtx_idx_cg1_1 / 2 * 8;
                            int o_token_addr_cg1_1 = o_token_base_cg1_1 + o_row_addr_cg1_1;
                            int o_token_pair_cg1_1 = o_token_addr_cg1_1 / 2;
                            int o_token_parity_cg1_1 = o_token_addr_cg1_1 & 1;
                            int o_raw_row_cg1_1 = o_token_pair_cg1_1 + o_dim_base_cg1_1 / 64 * 32;
                            int o_raw_col_cg1_1 = (o_dim_base_cg1_1 & 63 ^ (o_token_pair_cg1_1 & 3) << 4 ^ o_token_parity_cg1_1 << 3) + o_token_parity_cg1_1 * 64;
                            int o_stsm_offset_cg1_1 = (o_raw_row_cg1_1 * 128 + o_raw_col_cg1_1) * 2;
                            const int o_pack_base_cg1_1 = token_group_cg1_1 * 4;
                            #if CAKE_GDN_STATE_VEC_ALIGNED
                            uint32_t _stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1_1 + o_stsm_offset_cg1_1));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 3]))
                                : "memory");
                            #else  // CAKE_GDN_STATE_VEC_ALIGNED
                            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1_1 + o_stsm_offset_cg1_1));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_23[o_pack_base_cg1_1 + 3]))
                                : "memory");
                            #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(q_state_acc_empty_addr);
                    mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                    o_cg1_stage += 1;
                    if (o_cg1_stage == 2) { o_cg1_stage = 0; _phase_o_smem_empty ^= 1; }
                }
                if ((STORE_FINAL_STATE != 0 || ENABLE_CHECKPOINTS != 0) && num_chunks_b_1 > 0) {
                    mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                    _phase_kv_acc_full_0 ^= 1;
                    int warp_in_wg_3 = warp % 4;
                    int state_tmem_row_base_cg1 = warp_in_wg_3 * 32 << 16;
                    #if CAKE_GDN_STATE_VEC_ALIGNED
                    #pragma unroll
                    for (int state_lane_half_cg1 = 0; state_lane_half_cg1 < 2; state_lane_half_cg1++) {
                        int warp_id_in_role_6 = warp - 4;
                        int state_row_top_cg1 = warp_id_in_role_6 * 32 + state_lane_half_cg1 * 16 + lane / 4;
                        long long state_base_idx_top_cg1 = output_state_head_base_cg1 + (long long)state_row_top_cg1 * 128ll;
                        #pragma unroll
                        for (int state_col_half_cg1 = 0; state_col_half_cg1 < 2; state_col_half_cg1++) {
                            int final_state_addr_cg1 = taddr + (unsigned int)(state_tmem_row_base_cg1 + (state_lane_half_cg1 * 16 << 16)) + (unsigned int)(state_col_half_cg1 * 64);
                            float _tmem_load[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[31]))
                                : "r"(final_state_addr_cg1));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #pragma unroll
                            for (int state_col_group_cg1 = 0; state_col_group_cg1 < 8; state_col_group_cg1++) {
                                int state_col_pair_cg1 = state_col_half_cg1 * 64 + state_col_group_cg1 * 8 + (lane & 3) * 2;
                                const int state_reg_base_cg1 = state_col_group_cg1 * 4;
                                {
                                    unsigned int packed_0[1];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load + state_reg_base_cg1)[_lp*2 + 0], (_tmem_load + state_reg_base_cg1)[_lp*2+1 + 0]));
                                        packed_0[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<unsigned*>(output_state + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1))[0] = reinterpret_cast<unsigned*>(packed_0)[0];
                                    unsigned int packed_1[1];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load + (state_reg_base_cg1 + 2))[_lp*2 + 0], (_tmem_load + (state_reg_base_cg1 + 2))[_lp*2+1 + 0]));
                                        packed_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<unsigned*>(output_state + (state_base_idx_top_cg1 + 1024ll + (long long)state_col_pair_cg1))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
                                }
                            }
                        }
                    }
                    #else  // CAKE_GDN_STATE_VEC_ALIGNED
                    if (__builtin_expect((STORE_FINAL_STATE == 0 || ((unsigned long long)output_state + (unsigned long long)output_state_head_base_cg1 * 2ull & 3ull) == 0ull) && (ENABLE_CHECKPOINTS == 0 || ((unsigned long long)checkpoint_state & 3ull) == 0ull), 1)) {
                        #pragma unroll
                        for (int state_lane_half_cg1 = 0; state_lane_half_cg1 < 2; state_lane_half_cg1++) {
                            int warp_id_in_role_6 = warp - 4;
                            int state_row_top_cg1 = warp_id_in_role_6 * 32 + state_lane_half_cg1 * 16 + lane / 4;
                            long long state_base_idx_top_cg1 = output_state_head_base_cg1 + (long long)state_row_top_cg1 * 128ll;
                            #pragma unroll
                            for (int state_col_half_cg1 = 0; state_col_half_cg1 < 2; state_col_half_cg1++) {
                                int final_state_addr_cg1 = taddr + (unsigned int)(state_tmem_row_base_cg1 + (state_lane_half_cg1 * 16 << 16)) + (unsigned int)(state_col_half_cg1 * 64);
                                float _tmem_load[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[31]))
                                    : "r"(final_state_addr_cg1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int state_col_group_cg1 = 0; state_col_group_cg1 < 8; state_col_group_cg1++) {
                                    int state_col_pair_cg1 = state_col_half_cg1 * 64 + state_col_group_cg1 * 8 + (lane & 3) * 2;
                                    const int state_reg_base_cg1 = state_col_group_cg1 * 4;
                                    {
                                        unsigned int packed_0[1];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load + state_reg_base_cg1)[_lp*2 + 0], (_tmem_load + state_reg_base_cg1)[_lp*2+1 + 0]));
                                            packed_0[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<unsigned*>(output_state + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1))[0] = reinterpret_cast<unsigned*>(packed_0)[0];
                                        unsigned int packed_1[1];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load + (state_reg_base_cg1 + 2))[_lp*2 + 0], (_tmem_load + (state_reg_base_cg1 + 2))[_lp*2+1 + 0]));
                                            packed_1[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<unsigned*>(output_state + (state_base_idx_top_cg1 + 1024ll + (long long)state_col_pair_cg1))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
                                    }
                                }
                            }
                        }
                    } else {
                        #pragma unroll 1
                        for (int state_lane_half_cg1_1 = 0; state_lane_half_cg1_1 < 2; state_lane_half_cg1_1++) {
                            int warp_id_in_role_7 = warp - 4;
                            int state_row_top_cg1_1 = warp_id_in_role_7 * 32 + state_lane_half_cg1_1 * 16 + lane / 4;
                            long long state_base_idx_top_cg1_1 = output_state_head_base_cg1 + (long long)state_row_top_cg1_1 * 128ll;
                            #pragma unroll 1
                            for (int state_col_half_cg1_1 = 0; state_col_half_cg1_1 < 2; state_col_half_cg1_1++) {
                                int final_state_addr_cg1_1 = taddr + (unsigned int)(state_tmem_row_base_cg1 + (state_lane_half_cg1_1 * 16 << 16)) + (unsigned int)(state_col_half_cg1_1 * 64);
                                float _tmem_load_1[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                                    : "r"(final_state_addr_cg1_1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                #pragma unroll
                                for (int state_col_group_cg1_1 = 0; state_col_group_cg1_1 < 8; state_col_group_cg1_1++) {
                                    int state_col_pair_cg1_1 = state_col_half_cg1_1 * 64 + state_col_group_cg1_1 * 8 + (lane & 3) * 2;
                                    const int state_reg_base_cg1_1 = state_col_group_cg1_1 * 4;
                                    {
                                        unsigned int packed_4[1];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + state_reg_base_cg1_1)[_lp*2 + 0], (_tmem_load_1 + state_reg_base_cg1_1)[_lp*2+1 + 0]));
                                            packed_4[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state) + (state_base_idx_top_cg1_1 + (long long)state_col_pair_cg1_1)) = (unsigned short)((uint16_t)packed_4[0]);
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state) + (state_base_idx_top_cg1_1 + (long long)state_col_pair_cg1_1 + 1ll)) = (unsigned short)((uint16_t)(packed_4[0] >> 16));
                                        unsigned int packed_5[1];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 1; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_1 + (state_reg_base_cg1_1 + 2))[_lp*2 + 0], (_tmem_load_1 + (state_reg_base_cg1_1 + 2))[_lp*2+1 + 0]));
                                            packed_5[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state) + (state_base_idx_top_cg1_1 + 1024ll + (long long)state_col_pair_cg1_1)) = (unsigned short)((uint16_t)packed_5[0]);
                                        *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state) + (state_base_idx_top_cg1_1 + 1024ll + (long long)state_col_pair_cg1_1 + 1ll)) = (unsigned short)((uint16_t)(packed_5[0] >> 16));
                                    }
                                }
                            }
                        }
                    }
                    #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                    mbarrier_arrive(kv_state_drained_addr);
                }
                if (STORE_FINAL_STATE != 0 && USE_INITIAL_STATE != 0 && num_chunks_b_1 == 0) {
                    #if CAKE_GDN_STATE_VEC_ALIGNED
                    int warp_id_in_role_7 = warp - 4;
                    int empty_state_row = warp_id_in_role_7 * 32 + lane;
                    long long empty_initial_base = initial_state_head_base_cg1 + (long long)empty_state_row * 128ll;
                    long long empty_output_base = output_state_head_base_cg1 + (long long)empty_state_row * 128ll;
                    #pragma unroll
                    for (int empty_state_vec = 0; empty_state_vec < 32; empty_state_vec++) {
                        int empty_state_col = empty_state_vec * 4;
                        float _vec_load[4];
                        {
                            uint2 _vld_16;
                            _vld_16 = *reinterpret_cast<const uint2*>(initial_state + empty_initial_base + (long long)empty_state_col);
                            uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
                            #pragma unroll
                            for (int _pair = 0; _pair < 2; _pair++) {
                                (&_vec_load[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                                (&_vec_load[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                            }
                        }
                        unsigned int packed[2];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load[_lp*2 + 0], _vec_load[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load[_lp*2 + 2], _vec_load[_lp*2+1 + 2]));
                            packed[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int2*>(output_state + (empty_output_base + (long long)empty_state_col))[0] = reinterpret_cast<int2*>(packed)[0];
                    }
                    #else  // CAKE_GDN_STATE_VEC_ALIGNED
                    bool empty_vec_ok = ((unsigned long long)initial_state + (unsigned long long)initial_state_head_base_cg1 * 2ull & 7ull) == 0ull && ((unsigned long long)output_state + (unsigned long long)output_state_head_base_cg1 * 2ull & 7ull) == 0ull;
                    int warp_id_in_role_8 = warp - 4;
                    int empty_state_row = warp_id_in_role_8 * 32 + lane;
                    long long empty_initial_base = initial_state_head_base_cg1 + (long long)empty_state_row * 128ll;
                    long long empty_output_base = output_state_head_base_cg1 + (long long)empty_state_row * 128ll;
                    if (__builtin_expect(empty_vec_ok, 1)) {
                        #pragma unroll
                        for (int empty_state_vec = 0; empty_state_vec < 32; empty_state_vec++) {
                            int empty_state_col = empty_state_vec * 4;
                            float _vec_load[4];
                            {
                                uint2 _vld_20;
                                asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
                                    : "=r"(_vld_20.x), "=r"(_vld_20.y) : "l"((const void*)(initial_state + empty_initial_base + (long long)empty_state_col)));
                                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
                                #pragma unroll
                                for (int _pair = 0; _pair < 2; _pair++) {
                                    (&_vec_load[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                                    (&_vec_load[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                                }
                            }
                            unsigned int packed[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load[_lp*2 + 0], _vec_load[_lp*2+1 + 0]));
                                packed[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load[_lp*2 + 2], _vec_load[_lp*2+1 + 2]));
                                packed[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int2*>(output_state + (empty_output_base + (long long)empty_state_col))[0] = reinterpret_cast<int2*>(packed)[0];
                        }
                    } else {
                        #pragma unroll 1
                        for (int state_copy_step = 0; state_copy_step < 32; state_copy_step++) {
                            float state_copy_vals[4];
                            {
                                uint32_t _bf16_bits_21;
                                asm volatile("ld.global.nc.u16 %0, [%1];"
                                    : "=r"(_bf16_bits_21) : "l"((const void*)(initial_state + (empty_initial_base + (long long)(state_copy_step * 4)))));
                                state_copy_vals[0] = __uint_as_float(_bf16_bits_21 << 16);
                            }
                            {
                                uint32_t _bf16_bits_22;
                                asm volatile("ld.global.nc.u16 %0, [%1];"
                                    : "=r"(_bf16_bits_22) : "l"((const void*)(initial_state + (empty_initial_base + (long long)(state_copy_step * 4 + 1)))));
                                state_copy_vals[1] = __uint_as_float(_bf16_bits_22 << 16);
                            }
                            {
                                uint32_t _bf16_bits_23;
                                asm volatile("ld.global.nc.u16 %0, [%1];"
                                    : "=r"(_bf16_bits_23) : "l"((const void*)(initial_state + (empty_initial_base + (long long)(state_copy_step * 4 + 2)))));
                                state_copy_vals[2] = __uint_as_float(_bf16_bits_23 << 16);
                            }
                            {
                                uint32_t _bf16_bits_24;
                                asm volatile("ld.global.nc.u16 %0, [%1];"
                                    : "=r"(_bf16_bits_24) : "l"((const void*)(initial_state + (empty_initial_base + (long long)(state_copy_step * 4 + 3)))));
                                state_copy_vals[3] = __uint_as_float(_bf16_bits_24 << 16);
                            }
                            unsigned int packed_2[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_copy_vals[_lp*2 + 0], state_copy_vals[_lp*2+1 + 0]));
                                packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_copy_vals[_lp*2 + 2], state_copy_vals[_lp*2+1 + 2]));
                                packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state + (empty_output_base + (long long)(state_copy_step * 4))) + 0) = (unsigned short)((uint16_t)packed_2[0]);
                            *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state + (empty_output_base + (long long)(state_copy_step * 4))) + 1) = (unsigned short)((uint16_t)(packed_2[0] >> 16));
                            *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state + (empty_output_base + (long long)(state_copy_step * 4))) + 2) = (unsigned short)((uint16_t)packed_2[1]);
                            *reinterpret_cast<unsigned short*>(reinterpret_cast<uint16_t*>(output_state + (empty_output_base + (long long)(state_copy_step * 4))) + 3) = (unsigned short)((uint16_t)(packed_2[1] >> 16));
                        }
                    }
                    #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&O))) : "memory");
            unsigned int k_cg0_stage = 0;
            unsigned int q_cg0_stage = 0;
            int mma_cg0_pairs = 0;
            unsigned int _phase_load_k_full = 0;
            unsigned int _phase_load_q_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < total_tiles; tile_2 += num_bids) {
                int num_o_heads_2 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx_2 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_2 >> (unsigned int)NUM_O_HEADS_LOG2 : tile_2 / (unsigned int)num_o_heads_2);
                int head_idx_2 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_2 & (unsigned int)(num_o_heads_2 - 1) : tile_2 % (unsigned int)num_o_heads_2);
                int qk_head_idx_2 = ((IS_GQA != 0) ? head_idx_2 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_v_heads / num_q_heads)));
                int v_head_idx_2 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_q_heads / num_v_heads)) : head_idx_2);
                int raw_batch_start_2 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_2] : cu_seqlens[batch_idx_2]);
                int raw_batch_end_2 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_2 + 1)] : cu_seqlens[batch_idx_2 + 1]);
                int batch_start_floor_2 = ((raw_batch_start_2 > 0) ? raw_batch_start_2 : 0);
                int batch_start_2 = ((batch_start_floor_2 < total_tokens) ? batch_start_floor_2 : total_tokens);
                int batch_end_floor_2 = ((raw_batch_end_2 > batch_start_2) ? raw_batch_end_2 : batch_start_2);
                int batch_end_2 = ((batch_end_floor_2 < total_tokens) ? batch_end_floor_2 : total_tokens);
                int seqlen_b_2 = batch_end_2 - batch_start_2;
                int num_pairs_b_2 = (seqlen_b_2 + 128 - 1) / 128;
                int num_chunks_b_2 = num_pairs_b_2 * 2;
                int num_pairs_b_0 = num_chunks_b_2 / 2;
                #pragma unroll 1
                for (int _pair_idx = 0; _pair_idx < num_pairs_b_0; _pair_idx++) {
                    unsigned int stage_0 = k_cg0_stage;
                    mbarrier_wait(load_k_full_addr + (k_cg0_stage) * 8, _phase_load_k_full);
                    k_cg0_stage += 1;
                    if (k_cg0_stage == 4) { k_cg0_stage = 0; _phase_load_k_full ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr, 1);
                    int _mma_a_lo_0 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                        }
                    }
                    elect_commit(cg0_shared_acc_full_addr);
                    unsigned int stage_1 = k_cg0_stage;
                    mbarrier_wait(load_k_full_addr + (k_cg0_stage) * 8, _phase_load_k_full);
                    k_cg0_stage += 1;
                    if (k_cg0_stage == 4) { k_cg0_stage = 0; _phase_load_k_full ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr + 8, 1);
                    int _mma_a_lo_1 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 68158608, 1);
                        }
                    }
                    elect_commit(cg0_shared_acc_full_addr + 8);
                    unsigned int stage_2 = q_cg0_stage;
                    mbarrier_wait(load_q_full_addr + (q_cg0_stage) * 8, _phase_load_q_full);
                    q_cg0_stage += 1;
                    if (q_cg0_stage == 2) { q_cg0_stage = 0; _phase_load_q_full ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr, 0);
                    int _mma_a_lo_2 = make_warp_uniform_local((((smem_q_addr) >> 4) & 0x7FFF) + (stage_2) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_cg0_shared_acc, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 68158608, 1);
                        }
                    }
                    elect_commit(cg0_shared_acc_full_addr);
                    unsigned int stage_3 = q_cg0_stage;
                    mbarrier_wait(load_q_full_addr + (q_cg0_stage) * 8, _phase_load_q_full);
                    q_cg0_stage += 1;
                    if (q_cg0_stage == 2) { q_cg0_stage = 0; _phase_load_q_full ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr + 8, 0);
                    int _mma_a_lo_3 = make_warp_uniform_local((((smem_q_addr) >> 4) & 0x7FFF) + (stage_3) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_cg0_shared_acc + (64)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 68158608, 1);
                        }
                    }
                    elect_commit(cg0_shared_acc_full_addr + 8);
                    elect_commit(load_k_empty_addr + (stage_0) * 8);
                    elect_commit(load_k_empty_addr + (stage_1) * 8);
                    elect_commit(q_smem_empty_addr + (stage_2) * 8);
                    elect_commit(q_smem_empty_addr + (stage_3) * 8);
                    mma_cg0_pairs += 1;
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
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&Q));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + cta_slot_base));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&K));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base + 128)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&V));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base + 256)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            unsigned int k_stage = 0;
            unsigned int q_stage = 0;
            unsigned int v_stage = 0;
            unsigned int _phase_load_k_empty = 1;
            unsigned int _phase_q_smem_empty = 1;
            unsigned int _phase_v_smem_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_3 = bid; tile_3 < total_tiles; tile_3 += num_bids) {
                int num_o_heads_3 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx_3 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_3 >> (unsigned int)NUM_O_HEADS_LOG2 : tile_3 / (unsigned int)num_o_heads_3);
                int head_idx_3 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_3 & (unsigned int)(num_o_heads_3 - 1) : tile_3 % (unsigned int)num_o_heads_3);
                int qk_head_idx_3 = ((IS_GQA != 0) ? head_idx_3 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_v_heads / num_q_heads)));
                int v_head_idx_3 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_q_heads / num_v_heads)) : head_idx_3);
                int raw_batch_start_3 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_3] : cu_seqlens[batch_idx_3]);
                int raw_batch_end_3 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_3 + 1)] : cu_seqlens[batch_idx_3 + 1]);
                int batch_start_floor_3 = ((raw_batch_start_3 > 0) ? raw_batch_start_3 : 0);
                int batch_start_3 = ((batch_start_floor_3 < total_tokens) ? batch_start_floor_3 : total_tokens);
                int batch_end_floor_3 = ((raw_batch_end_3 > batch_start_3) ? raw_batch_end_3 : batch_start_3);
                int batch_end_3 = ((batch_end_floor_3 < total_tokens) ? batch_end_floor_3 : total_tokens);
                int seqlen_b_3 = batch_end_3 - batch_start_3;
                int num_pairs_b_3 = (seqlen_b_3 + 128 - 1) / 128;
                int num_chunks_b_3 = num_pairs_b_3 * 2;
                int k_head_idx = ((num_q_heads >= num_v_heads) ? v_head_idx_3 : qk_head_idx_3);
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
                    for (int chunk_in_pair = 0; chunk_in_pair < 2; chunk_in_pair++) {
                        int chunk_idx_2 = pair_idx_tma * 2 + chunk_in_pair;
                        int chunk_offset_3 = batch_start_3 + chunk_idx_2 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(load_k_empty_addr + (k_stage) * 8, _phase_load_k_empty);
                            mbarrier_arrive_expect_tx(load_k_full_addr + (k_stage) * 8, 16384);
                            #pragma unroll
                            for (int dim_half = 0; dim_half < 2; dim_half++) {
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + (unsigned int)(dim_half * 8192), tensormap_workspace + (cta_slot_base + 128), dim_half * 64, chunk_offset_3, k_head_idx, load_k_full_addr + (k_stage) * 8);
                            }
                        }
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_load_k_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int chunk_in_pair_1 = 0; chunk_in_pair_1 < 2; chunk_in_pair_1++) {
                        int chunk_idx_3 = pair_idx_tma * 2 + chunk_in_pair_1;
                        int chunk_offset_4 = batch_start_3 + chunk_idx_3 * 64;
                        mbarrier_wait(q_smem_empty_addr + (q_stage) * 8, _phase_q_smem_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(load_q_full_addr + (q_stage) * 8, 16384);
                            #pragma unroll
                            for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                tma_3d_gmem2smem(smem_q_addr + q_stage * 16384 + (unsigned int)(dim_half_1 * 8192), tensormap_workspace + cta_slot_base, dim_half_1 * 64, chunk_offset_4, qk_head_idx_3, load_q_full_addr + (q_stage) * 8);
                            }
                        }
                        q_stage += 1;
                        if (q_stage == 2) { q_stage = 0; _phase_q_smem_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int chunk_in_pair_2 = 0; chunk_in_pair_2 < 2; chunk_in_pair_2++) {
                        int chunk_idx_4 = pair_idx_tma * 2 + chunk_in_pair_2;
                        int chunk_offset_5 = batch_start_3 + chunk_idx_4 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(v_smem_empty_addr + (v_stage) * 8, _phase_v_smem_empty);
                            mbarrier_arrive_expect_tx(load_v_full_addr + (v_stage) * 8, 16384);
                            #pragma unroll
                            for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
                                tma_3d_gmem2smem(smem_v_addr + v_stage * 16384 + (unsigned int)(dim_half_2 * 8192), tensormap_workspace + (cta_slot_base + 256), dim_half_2 * 64, chunk_offset_5, v_head_idx_3, load_v_full_addr + (v_stage) * 8);
                            }
                        }
                        v_stage += 1;
                        if (v_stage == 3) { v_stage = 0; _phase_v_smem_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma_cg1 ----
    if (warp == 10) {
        { // mma_cg1_main
            unsigned int k_stage_1 = 0;
            unsigned int q_stage_1 = 0;
            unsigned int v_stage_mma = 0;
            unsigned int ainv_stage_mma = 0;
            unsigned int qk_stage_mma = 0;
            int kv_acc_release_pending = 1;
            unsigned int _phase_kv_state_drained = 1;
            unsigned int _phase_load_k_full_1 = 0;
            unsigned int _phase_load_q_full_1 = 0;
            unsigned int _phase_state_inp_ready_0 = 0;
            unsigned int _phase_cg1_shared_acc_empty_0 = 1;
            unsigned int _phase_q_state_acc_empty = 1;
            unsigned int _phase_vks_ready_0 = 0;
            unsigned int _phase_ainv_ready = 0;
            unsigned int _phase_qk_ready = 0;
            unsigned int _phase_nv_ready_0 = 0;
            unsigned int _phase_decay_v_ready_0 = 0;
            unsigned int _phase_load_v_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_4 = bid; tile_4 < total_tiles; tile_4 += num_bids) {
                int num_o_heads_4 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx_4 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_4 >> (unsigned int)NUM_O_HEADS_LOG2 : tile_4 / (unsigned int)num_o_heads_4);
                int head_idx_4 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_4 & (unsigned int)(num_o_heads_4 - 1) : tile_4 % (unsigned int)num_o_heads_4);
                int qk_head_idx_4 = ((IS_GQA != 0) ? head_idx_4 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_v_heads / num_q_heads)));
                int v_head_idx_4 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_q_heads / num_v_heads)) : head_idx_4);
                int raw_batch_start_4 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_4] : cu_seqlens[batch_idx_4]);
                int raw_batch_end_4 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_4 + 1)] : cu_seqlens[batch_idx_4 + 1]);
                int batch_start_floor_4 = ((raw_batch_start_4 > 0) ? raw_batch_start_4 : 0);
                int batch_start_4 = ((batch_start_floor_4 < total_tokens) ? batch_start_floor_4 : total_tokens);
                int batch_end_floor_4 = ((raw_batch_end_4 > batch_start_4) ? raw_batch_end_4 : batch_start_4);
                int batch_end_4 = ((batch_end_floor_4 < total_tokens) ? batch_end_floor_4 : total_tokens);
                int seqlen_b_4 = batch_end_4 - batch_start_4;
                int num_pairs_b_4 = (seqlen_b_4 + 128 - 1) / 128;
                int num_chunks_b_4 = num_pairs_b_4 * 2;
                int active_chunks_b_1 = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_4 > 0)) != 0) ? 1 : num_chunks_b_4);
                if (active_chunks_b_1 > 0) {
                    if (kv_acc_release_pending == 1) {
                        mbarrier_wait(kv_state_drained_addr, _phase_kv_state_drained);
                        _phase_kv_state_drained ^= 1;
                    }
                    {
                    }
                    {
                        int chunk_offset_6 = batch_start_4;
                        int _mma_marker = batch_idx_4 + head_idx_4 + chunk_offset_6 + batch_end_4 + 512;
                        {
                            mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, _phase_load_k_full_1);
                        }
                        mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, _phase_load_q_full_1);
                        mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                        _phase_state_inp_ready_0 ^= 1;
                        mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                        _phase_cg1_shared_acc_empty_0 ^= 1;
                        int mma_b_lo = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(mma_b_lo), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                        elect_commit(cg1_shared_acc_full_addr);
                        mbarrier_wait(q_state_acc_empty_addr, _phase_q_state_acc_empty);
                        uint32_t _elected_0 = elect_sync();
                        int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x7FFF) + (q_stage_1) * 1024;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_q_state), "r"(_smem_desc_lo_1), "r"(tmem_tmem_state_inp), "r"(0), "r"(_elected_0) : "memory");
                        tcgen05_commit_leader(q_state_acc_full_addr, _elected_0);
                        tcgen05_commit_leader(q_smem_empty_addr + (q_stage_1) * 8, _elected_0);
                        _phase_q_state_acc_empty ^= 1;
                        mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
                        _phase_vks_ready_0 ^= 1;
                        mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, _phase_ainv_ready);
                        mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                        _phase_cg1_shared_acc_empty_0 ^= 1;
                        int mma_b_lo_1 = make_warp_uniform_local((((smem_ainv_addr) >> 4) & 0x7FFF) + (ainv_stage_mma) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(mma_b_lo_1), "r"(tmem_tmem_shared_inp + 0), "r"(0) : "memory");
                        elect_commit(cg1_shared_acc_full_addr);
                        elect_commit(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
                        ainv_stage_mma += 1;
                        if (ainv_stage_mma == 3) { ainv_stage_mma = 0; _phase_ainv_ready ^= 1; }
                        mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, _phase_qk_ready);
                        mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
                        _phase_nv_ready_0 ^= 1;
                        mbarrier_wait(q_state_acc_empty_addr, _phase_q_state_acc_empty);
                        int mma_b_lo_2 = make_warp_uniform_local((((smem_qk_addr) >> 4) & 0x7FFF) + (qk_stage_mma) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_q_state), "r"(mma_b_lo_2), "r"(tmem_tmem_shared_inp + 0), "r"(1) : "memory");
                        elect_commit(q_state_acc_full_addr);
                        _phase_q_state_acc_empty ^= 1;
                        elect_commit(qk_smem_empty_addr + (qk_stage_mma) * 8);
                        qk_stage_mma += 1;
                        if (qk_stage_mma == 2) { qk_stage_mma = 0; _phase_qk_ready ^= 1; }
                        {
                            mbarrier_wait(kv_acc_empty_addr, 0);
                            mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                            _phase_decay_v_ready_0 ^= 1;
                            int mma_b_lo_3 = make_warp_uniform_local(((((smem_k_trans_mma_addr) >> 4) & 0x7FFF) | 0x2000000) + (k_stage_1) * 1024);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_tmem_state), "r"(mma_b_lo_3), "r"(tmem_tmem_shared_inp + 32), "r"(1) : "memory");
                            elect_commit(kv_acc_full_addr);
                            elect_commit(load_k_empty_addr + (k_stage_1) * 8);
                        }
                        k_stage_1 += 1;
                        if (k_stage_1 == 4) { k_stage_1 = 0; _phase_load_k_full_1 ^= 1; }
                        q_stage_1 += 1;
                        if (q_stage_1 == 2) { q_stage_1 = 0; _phase_load_q_full_1 ^= 1; }
                        v_stage_mma += 1;
                        if (v_stage_mma == 3) { v_stage_mma = 0; _phase_load_v_full_1 ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_5 = 1; chunk_idx_5 < active_chunks_b_1; chunk_idx_5++) {
                    int chunk_offset_7 = batch_start_4 + chunk_idx_5 * 64;
                    int _mma_marker_1 = batch_idx_4 + head_idx_4 + chunk_offset_7 + batch_end_4 + 512;
                    {
                        mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, _phase_load_k_full_1);
                    }
                    mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, _phase_load_q_full_1);
                    mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                    _phase_state_inp_ready_0 ^= 1;
                    mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                    _phase_cg1_shared_acc_empty_0 ^= 1;
                    int mma_b_lo_4 = make_warp_uniform_local((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(mma_b_lo_4), "r"(tmem_tmem_state_inp), "r"(0) : "memory");
                    elect_commit(cg1_shared_acc_full_addr);
                    mbarrier_wait(q_state_acc_empty_addr, _phase_q_state_acc_empty);
                    uint32_t _elected_2 = elect_sync();
                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x7FFF) + (q_stage_1) * 1024;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_q_state), "r"(_smem_desc_lo_3), "r"(tmem_tmem_state_inp), "r"(0), "r"(_elected_2) : "memory");
                    tcgen05_commit_leader(q_state_acc_full_addr, _elected_2);
                    tcgen05_commit_leader(q_smem_empty_addr + (q_stage_1) * 8, _elected_2);
                    _phase_q_state_acc_empty ^= 1;
                    mbarrier_wait(vks_ready_addr, _phase_vks_ready_0);
                    _phase_vks_ready_0 ^= 1;
                    mbarrier_wait(ainv_ready_addr + (ainv_stage_mma) * 8, _phase_ainv_ready);
                    mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                    _phase_cg1_shared_acc_empty_0 ^= 1;
                    int mma_b_lo_1_1 = make_warp_uniform_local((((smem_ainv_addr) >> 4) & 0x7FFF) + (ainv_stage_mma) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_cg1_shared_acc), "r"(mma_b_lo_1_1), "r"(tmem_tmem_shared_inp + 0), "r"(0) : "memory");
                    elect_commit(cg1_shared_acc_full_addr);
                    elect_commit(ainv_smem_empty_addr + (ainv_stage_mma) * 8);
                    ainv_stage_mma += 1;
                    if (ainv_stage_mma == 3) { ainv_stage_mma = 0; _phase_ainv_ready ^= 1; }
                    mbarrier_wait(qk_ready_addr + (qk_stage_mma) * 8, _phase_qk_ready);
                    mbarrier_wait(nv_ready_addr, _phase_nv_ready_0);
                    _phase_nv_ready_0 ^= 1;
                    mbarrier_wait(q_state_acc_empty_addr, _phase_q_state_acc_empty);
                    int mma_b_lo_2_1 = make_warp_uniform_local((((smem_qk_addr) >> 4) & 0x7FFF) + (qk_stage_mma) * 512);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(tmem_tmem_q_state), "r"(mma_b_lo_2_1), "r"(tmem_tmem_shared_inp + 0), "r"(1) : "memory");
                    elect_commit(q_state_acc_full_addr);
                    _phase_q_state_acc_empty ^= 1;
                    elect_commit(qk_smem_empty_addr + (qk_stage_mma) * 8);
                    qk_stage_mma += 1;
                    if (qk_stage_mma == 2) { qk_stage_mma = 0; _phase_qk_ready ^= 1; }
                    {
                        int _parity_kv_acc_empty = (int)((unsigned int)chunk_idx_5 % 2);
                        mbarrier_wait(kv_acc_empty_addr, _parity_kv_acc_empty);
                        mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                        _phase_decay_v_ready_0 ^= 1;
                        int mma_b_lo_3_1 = make_warp_uniform_local(((((smem_k_trans_mma_addr) >> 4) & 0x7FFF) | 0x2000000) + (k_stage_1) * 1024);
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
                    "mov.b32 id, 136381584;\n\t"
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
                    :: "r"(tmem_tmem_state), "r"(mma_b_lo_3_1), "r"(tmem_tmem_shared_inp + 32), "r"(1) : "memory");
                        elect_commit(kv_acc_full_addr);
                        elect_commit(load_k_empty_addr + (k_stage_1) * 8);
                    }
                    k_stage_1 += 1;
                    if (k_stage_1 == 4) { k_stage_1 = 0; _phase_load_k_full_1 ^= 1; }
                    q_stage_1 += 1;
                    if (q_stage_1 == 2) { q_stage_1 = 0; _phase_load_q_full_1 ^= 1; }
                    v_stage_mma += 1;
                    if (v_stage_mma == 3) { v_stage_mma = 0; _phase_load_v_full_1 ^= 1; }
                }
                if (num_chunks_b_4 > 0) {
                    kv_acc_release_pending = ((STORE_FINAL_STATE != 0 || ENABLE_CHECKPOINTS != 0) ? 1 : 0);
                }
            }
        }
    }
    // ---- Role: output_epilogue ----
    if (warp == 11) {
        { // output_epilogue_main
            int cta_slot_base_epi = bid * 512;
            unsigned int gate_k_prod = 0;
            unsigned int o_epi_stage = 0;
            unsigned int o_release_stage = 0;
            if (lane == 0) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&O));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
            }
            unsigned int _phase_gate_cg0_empty = 1;
            unsigned int _phase_o_store_ready = 0;
            #pragma unroll 1
            for (unsigned int tile_5 = bid; tile_5 < total_tiles; tile_5 += num_bids) {
                int num_o_heads_5 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int batch_idx_5 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_5 >> (unsigned int)NUM_O_HEADS_LOG2 : tile_5 / (unsigned int)num_o_heads_5);
                int head_idx_5 = ((NUM_O_HEADS_LOG2 >= 0) ? tile_5 & (unsigned int)(num_o_heads_5 - 1) : tile_5 % (unsigned int)num_o_heads_5);
                int qk_head_idx_5 = ((IS_GQA != 0) ? head_idx_5 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_v_heads / num_q_heads)));
                int v_head_idx_5 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_q_heads / num_v_heads)) : head_idx_5);
                int raw_batch_start_5 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_5] : cu_seqlens[batch_idx_5]);
                int raw_batch_end_5 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_5 + 1)] : cu_seqlens[batch_idx_5 + 1]);
                int batch_start_floor_5 = ((raw_batch_start_5 > 0) ? raw_batch_start_5 : 0);
                int batch_start_5 = ((batch_start_floor_5 < total_tokens) ? batch_start_floor_5 : total_tokens);
                int batch_end_floor_5 = ((raw_batch_end_5 > batch_start_5) ? raw_batch_end_5 : batch_start_5);
                int batch_end_5 = ((batch_end_floor_5 < total_tokens) ? batch_end_floor_5 : total_tokens);
                int seqlen_b_5 = batch_end_5 - batch_start_5;
                int num_pairs_b_5 = (seqlen_b_5 + 128 - 1) / 128;
                int num_chunks_b_5 = num_pairs_b_5 * 2;
                if (lane == 0) {
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))), "r"((uint32_t)(batch_end_5)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                }
                int num_valid_chunks_b = (batch_end_5 - batch_start_5 + 64 - 1) / 64;
                int gate_chunks_b = ((UNIT_GATES != 0) ? 0 : num_chunks_b_5);
                int tile_ord = (int)tile_5 - bid;
                int gate_prologue_chunks = ((tile_ord == 0) ? gate_chunks_b : 0);
                int gate_next_issued = 0;
                int next_tile_i32 = (int)tile_5 + num_bids;
                int next_head_idx = 0;
                int next_batch_start = 0;
                int next_batch_end = 0;
                int next_gate_budget = 0;
                if (next_tile_i32 < total_tiles) {
                    int num_o_heads_0 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                    int batch_idx_0 = ((NUM_O_HEADS_LOG2 >= 0) ? (unsigned int)next_tile_i32 >> (unsigned int)NUM_O_HEADS_LOG2 : (unsigned int)next_tile_i32 / (unsigned int)num_o_heads_0);
                    int head_idx_0 = ((NUM_O_HEADS_LOG2 >= 0) ? (unsigned int)next_tile_i32 & (unsigned int)(num_o_heads_0 - 1) : (unsigned int)next_tile_i32 % (unsigned int)num_o_heads_0);
                    int qk_head_idx_0 = ((IS_GQA != 0) ? head_idx_0 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_0 >> HEAD_GROUP_LOG2 : head_idx_0 / (num_v_heads / num_q_heads)));
                    int v_head_idx_0 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_0 >> HEAD_GROUP_LOG2 : head_idx_0 / (num_q_heads / num_v_heads)) : head_idx_0);
                    int raw_batch_start_0 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)batch_idx_0] : cu_seqlens[batch_idx_0]);
                    int raw_batch_end_0 = (((metadata_i64_mask & 1) != 0) ? (int)reinterpret_cast<long long*>(cu_seqlens)[(int)(batch_idx_0 + 1)] : cu_seqlens[batch_idx_0 + 1]);
                    int batch_start_floor_0 = ((raw_batch_start_0 > 0) ? raw_batch_start_0 : 0);
                    int batch_start_0 = ((batch_start_floor_0 < total_tokens) ? batch_start_floor_0 : total_tokens);
                    int batch_end_floor_0 = ((raw_batch_end_0 > batch_start_0) ? raw_batch_end_0 : batch_start_0);
                    int batch_end_0 = ((batch_end_floor_0 < total_tokens) ? batch_end_floor_0 : total_tokens);
                    int seqlen_b_0 = batch_end_0 - batch_start_0;
                    int num_pairs_b_1_1 = (seqlen_b_0 + 128 - 1) / 128;
                    int num_chunks_b_0 = num_pairs_b_1_1 * 2;
                    next_head_idx = head_idx_0;
                    next_batch_start = batch_start_0;
                    next_batch_end = batch_end_0;
                    int la_gate_chunks = ((UNIT_GATES != 0) ? 0 : num_chunks_b_0);
                    next_gate_budget = ((la_gate_chunks < 4) ? la_gate_chunks : 4);
                }
                #pragma unroll
                for (int prefetch_idx = 0; prefetch_idx < 4; prefetch_idx++) {
                    if (gate_prologue_chunks > prefetch_idx) {
                        int prefetch_offset = batch_start_5 + prefetch_idx * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(beta_smem_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        int gb_lane = lane;
                        int gate_elem_base = gate_k_prod * 64;
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
                        float shfl_up = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 1, 32);
                        if (gb_lane >= 1) {
                            gate_log_0 = gate_log_0 + shfl_up;
                        }
                        float shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 1, 32);
                        if (gb_lane >= 1) {
                            gate_log_1 = gate_log_1 + shfl_up_1;
                        }
                        float shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 2, 32);
                        if (gb_lane >= 2) {
                            gate_log_0 = gate_log_0 + shfl_up_2;
                        }
                        float shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 2, 32);
                        if (gb_lane >= 2) {
                            gate_log_1 = gate_log_1 + shfl_up_3;
                        }
                        float shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 4, 32);
                        if (gb_lane >= 4) {
                            gate_log_0 = gate_log_0 + shfl_up_4;
                        }
                        float shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 4, 32);
                        if (gb_lane >= 4) {
                            gate_log_1 = gate_log_1 + shfl_up_5;
                        }
                        float shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 8, 32);
                        if (gb_lane >= 8) {
                            gate_log_0 = gate_log_0 + shfl_up_6;
                        }
                        float shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 8, 32);
                        if (gb_lane >= 8) {
                            gate_log_1 = gate_log_1 + shfl_up_7;
                        }
                        float shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, gate_log_0, 16, 32);
                        if (gb_lane >= 16) {
                            gate_log_0 = gate_log_0 + shfl_up_8;
                        }
                        float shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, gate_log_1, 16, 32);
                        if (gb_lane >= 16) {
                            gate_log_1 = gate_log_1 + shfl_up_9;
                        }
                        float _shfl_sync_2;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(_shfl_sync_2) : "f"(gate_log_0), "r"(31));
                        gate_log_1 = gate_log_1 + _shfl_sync_2;
                        smem_cumsumlog[gate_elem_base + gb_lane] = gate_log_0;
                        smem_cumsumlog[gate_elem_base + gb_lane + 32] = gate_log_1;
                        float _exp2_3 = approx_exp2(gate_log_0);
                        smem_cumprod[gate_elem_base + gb_lane] = _exp2_3;
                        float _exp2_4 = approx_exp2(gate_log_1);
                        smem_cumprod[gate_elem_base + gb_lane + 32] = _exp2_4;
                        mbarrier_arrive(load_gate_full_addr + (gate_k_prod) * 8);
                        int beta_stage_addr = smem_beta_addr + gate_k_prod * 256;
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
                            :: "r"(load_beta_full_addr + (gate_k_prod) * 8) : "memory");
                        gate_k_prod += 1;
                        if (gate_k_prod == 5) { gate_k_prod = 0; _phase_gate_cg0_empty ^= 1; }
                    }
                }
                int active_output_chunks_b = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_5 > 0)) != 0) ? 1 : num_chunks_b_5);
                #pragma unroll 1
                for (int chunk_idx_6 = 0; chunk_idx_6 < active_output_chunks_b; chunk_idx_6++) {
                    int chunk_offset_8 = batch_start_5 + chunk_idx_6 * 64;
                    int prefetch_idx_1 = chunk_idx_6 + 5 - 1;
                    if (prefetch_idx_1 < gate_chunks_b) {
                        int prefetch_offset_1 = batch_start_5 + prefetch_idx_1 * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(beta_smem_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        int gb_lane_1 = lane;
                        int gate_elem_base_1 = gate_k_prod * 64;
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
                        float _log2_5;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(gate_val_0_1 + 1e-10f));
                        gate_log_0_1 = _log2_5;
                        float _log2_6;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(gate_val_1_1 + 1e-10f));
                        gate_log_1_1 = _log2_6;
                        float shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 1, 32);
                        if (gb_lane_1 >= 1) {
                            gate_log_0_1 = gate_log_0_1 + shfl_up_10;
                        }
                        float shfl_up_1_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 1, 32);
                        if (gb_lane_1 >= 1) {
                            gate_log_1_1 = gate_log_1_1 + shfl_up_1_1;
                        }
                        float shfl_up_2_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 2, 32);
                        if (gb_lane_1 >= 2) {
                            gate_log_0_1 = gate_log_0_1 + shfl_up_2_1;
                        }
                        float shfl_up_3_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 2, 32);
                        if (gb_lane_1 >= 2) {
                            gate_log_1_1 = gate_log_1_1 + shfl_up_3_1;
                        }
                        float shfl_up_4_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 4, 32);
                        if (gb_lane_1 >= 4) {
                            gate_log_0_1 = gate_log_0_1 + shfl_up_4_1;
                        }
                        float shfl_up_5_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 4, 32);
                        if (gb_lane_1 >= 4) {
                            gate_log_1_1 = gate_log_1_1 + shfl_up_5_1;
                        }
                        float shfl_up_6_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 8, 32);
                        if (gb_lane_1 >= 8) {
                            gate_log_0_1 = gate_log_0_1 + shfl_up_6_1;
                        }
                        float shfl_up_7_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 8, 32);
                        if (gb_lane_1 >= 8) {
                            gate_log_1_1 = gate_log_1_1 + shfl_up_7_1;
                        }
                        float shfl_up_8_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_1, 16, 32);
                        if (gb_lane_1 >= 16) {
                            gate_log_0_1 = gate_log_0_1 + shfl_up_8_1;
                        }
                        float shfl_up_9_1 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_1, 16, 32);
                        if (gb_lane_1 >= 16) {
                            gate_log_1_1 = gate_log_1_1 + shfl_up_9_1;
                        }
                        float _shfl_sync_7;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(_shfl_sync_7) : "f"(gate_log_0_1), "r"(31));
                        gate_log_1_1 = gate_log_1_1 + _shfl_sync_7;
                        smem_cumsumlog[gate_elem_base_1 + gb_lane_1] = gate_log_0_1;
                        smem_cumsumlog[gate_elem_base_1 + gb_lane_1 + 32] = gate_log_1_1;
                        float _exp2_8 = approx_exp2(gate_log_0_1);
                        smem_cumprod[gate_elem_base_1 + gb_lane_1] = _exp2_8;
                        float _exp2_9 = approx_exp2(gate_log_1_1);
                        smem_cumprod[gate_elem_base_1 + gb_lane_1 + 32] = _exp2_9;
                        mbarrier_arrive(load_gate_full_addr + (gate_k_prod) * 8);
                        int beta_stage_addr_1 = smem_beta_addr + gate_k_prod * 256;
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
                            :: "r"(load_beta_full_addr + (gate_k_prod) * 8) : "memory");
                        gate_k_prod += 1;
                        if (gate_k_prod == 5) { gate_k_prod = 0; _phase_gate_cg0_empty ^= 1; }
                    }
                    int next_due = ((prefetch_idx_1 >= gate_chunks_b && gate_next_issued < next_gate_budget) ? 1 : 0);
                    if (next_due == 1) {
                        int next_valid_chunks = (next_batch_end - next_batch_start + 64 - 1) / 64;
                        int next_gate_offset = next_batch_start + gate_next_issued * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(beta_smem_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        int gb_lane_2 = lane;
                        int gate_elem_base_2 = gate_k_prod * 64;
                        float gate_log_0_2 = 0.0f;
                        float gate_log_1_2 = 0.0f;
                        int token_0_2 = next_gate_offset + gb_lane_2;
                        int token_1_2 = token_0_2 + 32;
                        int gate_idx_0_2 = token_0_2 * num_o_heads_5 + next_head_idx;
                        int gate_idx_1_2 = token_1_2 * num_o_heads_5 + next_head_idx;
                        int beta_idx_0_2 = gate_idx_0_2;
                        int beta_idx_1_2 = gate_idx_1_2;
                        float gate_val_0_2 = 1.0f;
                        float gate_val_1_2 = 1.0f;
                        if (gate_next_issued >= next_valid_chunks - 1) {
                            if (token_0_2 < next_batch_end) {
                                gate_val_0_2 = gate[gate_idx_0_2];
                            }
                            if (token_1_2 < next_batch_end) {
                                gate_val_1_2 = gate[gate_idx_1_2];
                            }
                        } else {
                            gate_val_0_2 = gate[gate_idx_0_2];
                            gate_val_1_2 = gate[gate_idx_1_2];
                        }
                        float _log2_10;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_10) : "f"(gate_val_0_2 + 1e-10f));
                        gate_log_0_2 = _log2_10;
                        float _log2_11;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_11) : "f"(gate_val_1_2 + 1e-10f));
                        gate_log_1_2 = _log2_11;
                        float shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 1, 32);
                        if (gb_lane_2 >= 1) {
                            gate_log_0_2 = gate_log_0_2 + shfl_up_11;
                        }
                        float shfl_up_1_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 1, 32);
                        if (gb_lane_2 >= 1) {
                            gate_log_1_2 = gate_log_1_2 + shfl_up_1_2;
                        }
                        float shfl_up_2_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 2, 32);
                        if (gb_lane_2 >= 2) {
                            gate_log_0_2 = gate_log_0_2 + shfl_up_2_2;
                        }
                        float shfl_up_3_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 2, 32);
                        if (gb_lane_2 >= 2) {
                            gate_log_1_2 = gate_log_1_2 + shfl_up_3_2;
                        }
                        float shfl_up_4_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 4, 32);
                        if (gb_lane_2 >= 4) {
                            gate_log_0_2 = gate_log_0_2 + shfl_up_4_2;
                        }
                        float shfl_up_5_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 4, 32);
                        if (gb_lane_2 >= 4) {
                            gate_log_1_2 = gate_log_1_2 + shfl_up_5_2;
                        }
                        float shfl_up_6_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 8, 32);
                        if (gb_lane_2 >= 8) {
                            gate_log_0_2 = gate_log_0_2 + shfl_up_6_2;
                        }
                        float shfl_up_7_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 8, 32);
                        if (gb_lane_2 >= 8) {
                            gate_log_1_2 = gate_log_1_2 + shfl_up_7_2;
                        }
                        float shfl_up_8_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_2, 16, 32);
                        if (gb_lane_2 >= 16) {
                            gate_log_0_2 = gate_log_0_2 + shfl_up_8_2;
                        }
                        float shfl_up_9_2 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_2, 16, 32);
                        if (gb_lane_2 >= 16) {
                            gate_log_1_2 = gate_log_1_2 + shfl_up_9_2;
                        }
                        float _shfl_sync_12;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(_shfl_sync_12) : "f"(gate_log_0_2), "r"(31));
                        gate_log_1_2 = gate_log_1_2 + _shfl_sync_12;
                        smem_cumsumlog[gate_elem_base_2 + gb_lane_2] = gate_log_0_2;
                        smem_cumsumlog[gate_elem_base_2 + gb_lane_2 + 32] = gate_log_1_2;
                        #if CAKE_GDN_STATE_VEC_ALIGNED
                        float _exp2_13_1 = approx_exp2(gate_log_0_2);
                        smem_cumprod[gate_elem_base_2 + gb_lane_2] = _exp2_13_1;
                        #else  // CAKE_GDN_STATE_VEC_ALIGNED
                        float _exp2_13 = approx_exp2(gate_log_0_2);
                        smem_cumprod[gate_elem_base_2 + gb_lane_2] = _exp2_13;
                        #endif  // CAKE_GDN_STATE_VEC_ALIGNED
                        float _exp2_14 = approx_exp2(gate_log_1_2);
                        smem_cumprod[gate_elem_base_2 + gb_lane_2 + 32] = _exp2_14;
                        mbarrier_arrive(load_gate_full_addr + (gate_k_prod) * 8);
                        int beta_stage_addr_2 = smem_beta_addr + gate_k_prod * 256;
                        int beta_dst_0_2 = beta_stage_addr_2 + gb_lane_2 * 4;
                        int beta_dst_1_2 = beta_stage_addr_2 + (gb_lane_2 + 32) * 4;
                        if (gate_next_issued >= next_valid_chunks - 1) {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_0_2), "l"(beta + beta_idx_0_2), "r"((token_0_2 < next_batch_end) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1_2), "l"(beta + beta_idx_1_2), "r"((token_1_2 < next_batch_end) ? 4 : 0));
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
                            :: "r"(load_beta_full_addr + (gate_k_prod) * 8) : "memory");
                        gate_k_prod += 1;
                        if (gate_k_prod == 5) { gate_k_prod = 0; _phase_gate_cg0_empty ^= 1; }
                        gate_next_issued = gate_next_issued + 1;
                    }
                    mbarrier_wait(o_store_ready_addr + (o_epi_stage) * 8, _phase_o_store_ready);
                    if (lane == 0) {
                        if (chunk_idx_6 == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        #pragma unroll
                        for (int dim_half_3 = 0; dim_half_3 < 2; dim_half_3++) {
                            tma_store_3d(tensormap_workspace + (cta_slot_base_epi + 384), dim_half_3 * 64, chunk_offset_8, head_idx_5, smem_o_addr + o_epi_stage * 16384 + (unsigned int)(dim_half_3 * 8192));
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    o_epi_stage += 1;
                    if (o_epi_stage == 2) { o_epi_stage = 0; _phase_o_store_ready ^= 1; }
                    if (chunk_idx_6 > 0) {
                        asm volatile("cp.async.bulk.wait_group 1;");
                        mbarrier_arrive(o_smem_empty_addr + (o_release_stage) * 8);
                        o_release_stage += 1;
                        if (o_release_stage == 2) { o_release_stage = 0; }
                    }
                }
                #pragma unroll
                for (int next_idx = 0; next_idx < 4; next_idx++) {
                    int tail_due = ((gate_next_issued <= next_idx && next_gate_budget > next_idx) ? 1 : 0);
                    if (tail_due == 1) {
                        int next_valid_chunks_1 = (next_batch_end - next_batch_start + 64 - 1) / 64;
                        int next_gate_offset_1 = next_batch_start + next_idx * 64;
                        mbarrier_wait(gate_cg0_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(gate_cg1_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        mbarrier_wait(beta_smem_empty_addr + (gate_k_prod) * 8, _phase_gate_cg0_empty);
                        int gb_lane_3 = lane;
                        int gate_elem_base_3 = gate_k_prod * 64;
                        float gate_log_0_3 = 0.0f;
                        float gate_log_1_3 = 0.0f;
                        int token_0_3 = next_gate_offset_1 + gb_lane_3;
                        int token_1_3 = token_0_3 + 32;
                        int gate_idx_0_3 = token_0_3 * num_o_heads_5 + next_head_idx;
                        int gate_idx_1_3 = token_1_3 * num_o_heads_5 + next_head_idx;
                        int beta_idx_0_3 = gate_idx_0_3;
                        int beta_idx_1_3 = gate_idx_1_3;
                        float gate_val_0_3 = 1.0f;
                        float gate_val_1_3 = 1.0f;
                        if (next_valid_chunks_1 - 1 <= next_idx) {
                            if (token_0_3 < next_batch_end) {
                                gate_val_0_3 = gate[gate_idx_0_3];
                            }
                            if (token_1_3 < next_batch_end) {
                                gate_val_1_3 = gate[gate_idx_1_3];
                            }
                        } else {
                            gate_val_0_3 = gate[gate_idx_0_3];
                            gate_val_1_3 = gate[gate_idx_1_3];
                        }
                        float _log2_15;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_15) : "f"(gate_val_0_3 + 1e-10f));
                        gate_log_0_3 = _log2_15;
                        float _log2_16;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_16) : "f"(gate_val_1_3 + 1e-10f));
                        gate_log_1_3 = _log2_16;
                        float shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_3, 1, 32);
                        if (gb_lane_3 >= 1) {
                            gate_log_0_3 = gate_log_0_3 + shfl_up_12;
                        }
                        float shfl_up_1_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_3, 1, 32);
                        if (gb_lane_3 >= 1) {
                            gate_log_1_3 = gate_log_1_3 + shfl_up_1_3;
                        }
                        float shfl_up_2_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_3, 2, 32);
                        if (gb_lane_3 >= 2) {
                            gate_log_0_3 = gate_log_0_3 + shfl_up_2_3;
                        }
                        float shfl_up_3_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_3, 2, 32);
                        if (gb_lane_3 >= 2) {
                            gate_log_1_3 = gate_log_1_3 + shfl_up_3_3;
                        }
                        float shfl_up_4_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_3, 4, 32);
                        if (gb_lane_3 >= 4) {
                            gate_log_0_3 = gate_log_0_3 + shfl_up_4_3;
                        }
                        float shfl_up_5_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_3, 4, 32);
                        if (gb_lane_3 >= 4) {
                            gate_log_1_3 = gate_log_1_3 + shfl_up_5_3;
                        }
                        float shfl_up_6_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_3, 8, 32);
                        if (gb_lane_3 >= 8) {
                            gate_log_0_3 = gate_log_0_3 + shfl_up_6_3;
                        }
                        float shfl_up_7_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_3, 8, 32);
                        if (gb_lane_3 >= 8) {
                            gate_log_1_3 = gate_log_1_3 + shfl_up_7_3;
                        }
                        float shfl_up_8_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_0_3, 16, 32);
                        if (gb_lane_3 >= 16) {
                            gate_log_0_3 = gate_log_0_3 + shfl_up_8_3;
                        }
                        float shfl_up_9_3 = __shfl_up_sync(0xFFFFFFFF, gate_log_1_3, 16, 32);
                        if (gb_lane_3 >= 16) {
                            gate_log_1_3 = gate_log_1_3 + shfl_up_9_3;
                        }
                        float _shfl_sync_17;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xFFFFFFFF;" : "=f"(_shfl_sync_17) : "f"(gate_log_0_3), "r"(31));
                        gate_log_1_3 = gate_log_1_3 + _shfl_sync_17;
                        smem_cumsumlog[gate_elem_base_3 + gb_lane_3] = gate_log_0_3;
                        smem_cumsumlog[gate_elem_base_3 + gb_lane_3 + 32] = gate_log_1_3;
                        float _exp2_18 = approx_exp2(gate_log_0_3);
                        smem_cumprod[gate_elem_base_3 + gb_lane_3] = _exp2_18;
                        float _exp2_19 = approx_exp2(gate_log_1_3);
                        smem_cumprod[gate_elem_base_3 + gb_lane_3 + 32] = _exp2_19;
                        mbarrier_arrive(load_gate_full_addr + (gate_k_prod) * 8);
                        int beta_stage_addr_3 = smem_beta_addr + gate_k_prod * 256;
                        int beta_dst_0_3 = beta_stage_addr_3 + gb_lane_3 * 4;
                        int beta_dst_1_3 = beta_stage_addr_3 + (gb_lane_3 + 32) * 4;
                        if (next_valid_chunks_1 - 1 <= next_idx) {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_0_3), "l"(beta + beta_idx_0_3), "r"((token_0_3 < next_batch_end) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1_3), "l"(beta + beta_idx_1_3), "r"((token_1_3 < next_batch_end) ? 4 : 0));
                        } else {
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_0_3), "l"(beta + beta_idx_0_3));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                                :: "r"(beta_dst_1_3), "l"(beta + beta_idx_1_3));
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(load_beta_full_addr + (gate_k_prod) * 8) : "memory");
                        gate_k_prod += 1;
                        if (gate_k_prod == 5) { gate_k_prod = 0; _phase_gate_cg0_empty ^= 1; }
                    }
                }
                if (active_output_chunks_b > 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    mbarrier_arrive(o_smem_empty_addr + (o_release_stage) * 8);
                    o_release_stage += 1;
                    if (o_release_stage == 2) { o_release_stage = 0; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
// clang-format on
