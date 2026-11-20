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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define NUM_O_HEADS_LOG2 -1
#define HEAD_GROUP_LOG2 -1
#define SINGLE_CHUNK_NO_STATE 0
#define UNIT_GATES 0
#define USE_INITIAL_STATE 1
#define STORE_FINAL_STATE 1
#define ENABLE_CHECKPOINTS 0
#define IS_GQA 0
#define USE_STATE_INDICES 0
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}





__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_affine_split_03829716e38373da30f9(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap O, float* __restrict__ gate, float* __restrict__ beta, int* __restrict__ cu_seqlens, int* __restrict__ state_indices, float* __restrict__ initial_state, float* __restrict__ output_state, float* __restrict__ checkpoint_state, int* __restrict__ cu_checkpoints, uint8_t* __restrict__ tensormap_workspace, long long initial_state_stride_slot, long long output_state_stride_slot, int checkpoint_every_n_tokens, float scale, int num_seqs, int total_tokens, int num_q_heads, int num_v_heads, int total_tiles, float* __restrict__ o_loc, __nv_bfloat16* __restrict__ c, float* __restrict__ bm, int p_cap, int c_min)
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
#if __CUDA_ARCH__ == 1070
    #define kv_state_drained_addr (mbar_base + 576)
#else
#endif

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

#if __CUDA_ARCH__ == 1070
    // Mbarrier init (80 barriers at smem_raw[0..640))
#else
    // Mbarrier init (72 barriers at smem_raw[0..576))
#endif

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
#if __CUDA_ARCH__ == 1070
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
#else
#endif
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
#if __CUDA_ARCH__ == 1070
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 640);
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 576);
#endif
    if (warp == 4) {
#if __CUDA_ARCH__ == 1070
        int _tmem_hold = smem + 640;
#else
        int _tmem_hold = smem + 576;
#endif
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
                int split_seq_tiles = num_o_heads * (4 * p_cap - 2);
                int batch_idx = (int)((unsigned int)tile / (unsigned int)split_seq_tiles);
                int split_rest = (int)tile - batch_idx * split_seq_tiles;
                int split_tail = split_rest - 2 * num_o_heads;
                int split_seg = ((split_tail < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail >> NUM_O_HEADS_LOG2 + 2 : split_tail / (4 * num_o_heads)) + 1);
                int split_pos = ((split_tail < 0) ? split_rest : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail & 4 * num_o_heads - 1 : split_tail % (4 * num_o_heads)));
                int split_kind = ((split_pos >= 2 * num_o_heads) ? 1 : 0);
                int split_half = split_pos - split_kind * 2 * num_o_heads;
                int head_idx = split_half >> 1;
                int value_split_idx = split_half & 1;
                int qk_head_idx = ((IS_GQA != 0) ? head_idx : ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_v_heads / num_q_heads)));
                int v_head_idx = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx >> HEAD_GROUP_LOG2 : head_idx / (num_q_heads / num_v_heads)) : head_idx);
                int batch_start_floor = ((cu_seqlens[batch_idx] > 0) ? cu_seqlens[batch_idx] : 0);
                int batch_start = ((batch_start_floor < total_tokens) ? batch_start_floor : total_tokens);
                int batch_end_floor = ((cu_seqlens[batch_idx + 1] > batch_start) ? cu_seqlens[batch_idx + 1] : batch_start);
                int batch_end = ((batch_end_floor < total_tokens) ? batch_end_floor : total_tokens);
                int seqlen_b = batch_end - batch_start;
                int num_pairs_b = (seqlen_b + 128 - 1) / 128;
                int num_chunks_b = num_pairs_b * 2;
                int split_len = batch_end - batch_start;
                int split_nchunks = split_len + 64 - 1 >> 6;
                int split_quota = (int)((unsigned int)split_nchunks / (unsigned int)c_min);
                int split_count = ((split_quota < 1) ? 1 : ((split_quota > p_cap) ? p_cap : split_quota));
                int split_lo = (int)((unsigned int)(split_seg * split_nchunks) / (unsigned int)split_count) * 64;
                int split_hi = (int)((unsigned int)((split_seg + 1) * split_nchunks) / (unsigned int)split_count) * 64;
                int batch_start_0 = ((split_seg < split_count) ? batch_start + split_lo : batch_end);
                int batch_end_0 = ((split_seg < split_count) ? batch_start + ((split_hi > split_len) ? split_len : split_hi) : batch_end);
                int num_chunks_b_0 = (batch_end_0 - batch_start_0 + 128 - 1 >> 7) * 2;
                #pragma unroll 1
                for (int chunk_idx = 0; chunk_idx < num_chunks_b_0; chunk_idx += 2) {
                    int chunk_offset = batch_start_0 + chunk_idx * 64;
                    int _cg0_marker = batch_idx + head_idx + chunk_offset + batch_end_0;
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
                        mbarrier_arrive(gate_cg0_empty_addr + (((gate_k_cg0 + 3 >= 5) ? gate_k_cg0 + 3 - 5 : gate_k_cg0 + 3)) * 8);
                        mbarrier_arrive(gate_cg0_empty_addr + (((gate_k_cg0 + 4 >= 5) ? gate_k_cg0 + 4 - 5 : gate_k_cg0 + 4)) * 8);
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int kk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[31]))
                        : "r"(kk_addr));
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg0_shared_acc_empty_addr);
                    mbarrier_wait(cg0_shared_acc_full_addr + 8, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int kk_addr_1 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[31]))
                        : "r"(kk_addr_1));
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    int inverse_group_cg0 = warp_id_in_role_cg0 / 2;
                    int inverse_local_warp_cg0 = warp_id_in_role_cg0 & 1;
                    unsigned int inverse_stage_cg0 = ((inverse_group_cg0 == 1) ? ainv1_stage : ainv0_stage);
                    int inverse_row_cg0 = inverse_local_warp_cg0 * 32 + lane;
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
                    mbarrier_arrive(ainv_ready_addr + (((ainv_cg0_stage + 1 < 3) ? ainv_cg0_stage + 1 : (unsigned int)0)) * 8);
                    {
                        mbarrier_arrive(beta_smem_empty_addr + (((beta_k_cg0 + 3 >= 5) ? beta_k_cg0 + 3 - 5 : beta_k_cg0 + 3)) * 8);
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
                    mbarrier_arrive(ainv_ready_addr + (((ainv_cg0_stage + 2 >= 3) ? ainv_cg0_stage + 2 - 3 : ainv_cg0_stage + 2)) * 8);
                    {
                        mbarrier_arrive(beta_smem_empty_addr + (((beta_k_cg0 + 4 >= 5) ? beta_k_cg0 + 4 - 5 : beta_k_cg0 + 4)) * 8);
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int qk_addr = taddr + 256 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[31]))
                        : "r"(qk_addr));
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg0_shared_acc_empty_addr);
                    mbarrier_arrive(qk_ready_addr + (qk_cg0_stage) * 8);
                    mbarrier_wait(cg0_shared_acc_full_addr + 8, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int qk_addr_0 = taddr + 256 + 64 + (unsigned int)qk_tmem_row_base_cg0;
                    float tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_2[31]))
                        : "r"(qk_addr_0));
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg0_shared_acc_empty_addr + 8);
                    mbarrier_arrive(qk_ready_addr + (((qk_cg0_stage + 1 < 2) ? qk_cg0_stage + 1 : (unsigned int)0)) * 8);
                    cg0_pairs += 1;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
            float seed_frag_0_cg1[32];
            float seed_frag_1_cg1[32];
            unsigned int _phase_load_gate_full_1 = 0;
            unsigned int _phase_load_v_full = 0;
            unsigned int _phase_cg1_shared_acc_full_0 = 0;
            unsigned int _phase_q_state_acc_full_0 = 0;
            unsigned int _phase_o_smem_empty = 1;
            unsigned int _phase_kv_acc_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                int num_o_heads_1 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                int split_seq_tiles_1 = num_o_heads_1 * (4 * p_cap - 2);
                int batch_idx_1 = (int)((unsigned int)tile_1 / (unsigned int)split_seq_tiles_1);
                int split_rest_1 = (int)tile_1 - batch_idx_1 * split_seq_tiles_1;
                int split_tail_1 = split_rest_1 - 2 * num_o_heads_1;
                int split_seg_1 = ((split_tail_1 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_1 >> NUM_O_HEADS_LOG2 + 2 : split_tail_1 / (4 * num_o_heads_1)) + 1);
                int split_pos_1 = ((split_tail_1 < 0) ? split_rest_1 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_1 & 4 * num_o_heads_1 - 1 : split_tail_1 % (4 * num_o_heads_1)));
                int split_kind_1 = ((split_pos_1 >= 2 * num_o_heads_1) ? 1 : 0);
                int split_half_1 = split_pos_1 - split_kind_1 * 2 * num_o_heads_1;
                int head_idx_1 = split_half_1 >> 1;
                int value_split_idx_1 = split_half_1 & 1;
                int qk_head_idx_1 = ((IS_GQA != 0) ? head_idx_1 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_v_heads / num_q_heads)));
                int v_head_idx_1 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_1 >> HEAD_GROUP_LOG2 : head_idx_1 / (num_q_heads / num_v_heads)) : head_idx_1);
                int batch_start_floor_1 = ((cu_seqlens[batch_idx_1] > 0) ? cu_seqlens[batch_idx_1] : 0);
                int batch_start_1 = ((batch_start_floor_1 < total_tokens) ? batch_start_floor_1 : total_tokens);
                int batch_end_floor_1 = ((cu_seqlens[batch_idx_1 + 1] > batch_start_1) ? cu_seqlens[batch_idx_1 + 1] : batch_start_1);
                int batch_end_1 = ((batch_end_floor_1 < total_tokens) ? batch_end_floor_1 : total_tokens);
                int seqlen_b_1 = batch_end_1 - batch_start_1;
                int num_pairs_b_1 = (seqlen_b_1 + 128 - 1) / 128;
                int num_chunks_b_1 = num_pairs_b_1 * 2;
                int split_len_1 = batch_end_1 - batch_start_1;
                int split_nchunks_1 = split_len_1 + 64 - 1 >> 6;
                int split_quota_1 = (int)((unsigned int)split_nchunks_1 / (unsigned int)c_min);
                int split_count_1 = ((split_quota_1 < 1) ? 1 : ((split_quota_1 > p_cap) ? p_cap : split_quota_1));
                int split_lo_1 = (int)((unsigned int)(split_seg_1 * split_nchunks_1) / (unsigned int)split_count_1) * 64;
                int split_hi_1 = (int)((unsigned int)((split_seg_1 + 1) * split_nchunks_1) / (unsigned int)split_count_1) * 64;
                int batch_start_1_1 = ((split_seg_1 < split_count_1) ? batch_start_1 + split_lo_1 : batch_end_1);
                int batch_end_1_1 = ((split_seg_1 < split_count_1) ? batch_start_1 + ((split_hi_1 > split_len_1) ? split_len_1 : split_hi_1) : batch_end_1);
                int num_chunks_b_1_1 = (batch_end_1_1 - batch_start_1_1 + 128 - 1 >> 7) * 2;
                int split_h0_chunks = ((split_seg_1 == 0) ? num_chunks_b_1_1 : 0);
                int split_empty_chunks = ((split_seg_1 < split_count_1) ? num_chunks_b_1_1 : 1);
                float* split_f32_out = o_loc;
                int split_f32_on = ((split_seg_1 > 0) ? 1 : 0);
                int split_bf16_on = split_f32_on * split_kind_1;
                split_f32_on = split_f32_on - split_bf16_on;
                int split_f32_dim = head_idx_1 * 128 + value_split_idx_1 * 64;
                long long split_f32_stride = (long long)(num_o_heads_1 * 128);
                int seed_tile_ord_cg1 = (int)tile_1 - bid;
                int seed_next_tile_cg1 = (int)tile_1 + num_bids;
                int seed_next_enabled_cg1 = 0;
                long long seed_next_base_cg1 = 0;
                int split_next_kind = 0;
                int split_next_half = 0;
                if (seed_next_tile_cg1 < total_tiles) {
                    int num_o_heads_0 = ((IS_GQA != 0) ? num_q_heads : num_v_heads);
                    int split_seq_tiles_0 = num_o_heads_0 * (4 * p_cap - 2);
                    int batch_idx_0 = (int)((unsigned int)(unsigned int)seed_next_tile_cg1 / (unsigned int)split_seq_tiles_0);
                    int split_rest_0 = (int)(unsigned int)seed_next_tile_cg1 - batch_idx_0 * split_seq_tiles_0;
                    int split_tail_0 = split_rest_0 - 2 * num_o_heads_0;
                    int split_seg_0 = ((split_tail_0 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_0 >> NUM_O_HEADS_LOG2 + 2 : split_tail_0 / (4 * num_o_heads_0)) + 1);
                    int split_pos_0 = ((split_tail_0 < 0) ? split_rest_0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_0 & 4 * num_o_heads_0 - 1 : split_tail_0 % (4 * num_o_heads_0)));
                    int split_kind_0 = ((split_pos_0 >= 2 * num_o_heads_0) ? 1 : 0);
                    int split_half_0 = split_pos_0 - split_kind_0 * 2 * num_o_heads_0;
                    int head_idx_0 = split_half_0 >> 1;
                    int value_split_idx_0 = split_half_0 & 1;
                    int qk_head_idx_0 = ((IS_GQA != 0) ? head_idx_0 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_0 >> HEAD_GROUP_LOG2 : head_idx_0 / (num_v_heads / num_q_heads)));
                    int v_head_idx_0 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_0 >> HEAD_GROUP_LOG2 : head_idx_0 / (num_q_heads / num_v_heads)) : head_idx_0);
                    int batch_start_floor_0 = ((cu_seqlens[batch_idx_0] > 0) ? cu_seqlens[batch_idx_0] : 0);
                    int batch_start_2 = ((batch_start_floor_0 < total_tokens) ? batch_start_floor_0 : total_tokens);
                    int batch_end_floor_0 = ((cu_seqlens[batch_idx_0 + 1] > batch_start_2) ? cu_seqlens[batch_idx_0 + 1] : batch_start_2);
                    int batch_end_2 = ((batch_end_floor_0 < total_tokens) ? batch_end_floor_0 : total_tokens);
                    int seqlen_b_0 = batch_end_2 - batch_start_2;
                    int num_pairs_b_0 = (seqlen_b_0 + 128 - 1) / 128;
                    int num_chunks_b_2 = num_pairs_b_0 * 2;
                    int split_len_0 = batch_end_2 - batch_start_2;
                    int split_nchunks_0 = split_len_0 + 64 - 1 >> 6;
                    int split_quota_0 = (int)((unsigned int)split_nchunks_0 / (unsigned int)c_min);
                    int split_count_0 = ((split_quota_0 < 1) ? 1 : ((split_quota_0 > p_cap) ? p_cap : split_quota_0));
                    int split_lo_0 = (int)((unsigned int)(split_seg_0 * split_nchunks_0) / (unsigned int)split_count_0) * 64;
                    int split_hi_0 = (int)((unsigned int)((split_seg_0 + 1) * split_nchunks_0) / (unsigned int)split_count_0) * 64;
                    int batch_start_3 = ((split_seg_0 < split_count_0) ? batch_start_2 + split_lo_0 : batch_end_2);
                    int batch_end_3 = ((split_seg_0 < split_count_0) ? batch_start_2 + ((split_hi_0 > split_len_0) ? split_len_0 : split_hi_0) : batch_end_2);
                    int num_chunks_b_3 = (batch_end_3 - batch_start_3 + 128 - 1 >> 7) * 2;
                    int num_chunks_b_1_0 = ((split_seg_0 == 0) ? num_chunks_b_3 : 0);
                    split_next_kind = split_kind_0;
                    split_next_half = value_split_idx_0;
                    int seed_next_slot_cg1 = batch_idx_0;
                    long long seed_next_head_offset_cg1 = (long long)head_idx_0 * 16384ll;
                    seed_next_base_cg1 = (long long)seed_next_slot_cg1 * initial_state_stride_slot + seed_next_head_offset_cg1 + (long long)(value_split_idx_0 * 64) * 128ll;
                    seed_next_enabled_cg1 = ((USE_INITIAL_STATE != 0 && num_chunks_b_1_0 > 0) ? 1 : 0);
                }
                int state_slot_cg1 = batch_idx_1;
                long long state_head_offset_cg1 = (long long)head_idx_1 * 16384ll;
                long long initial_state_head_base_cg1 = (long long)state_slot_cg1 * initial_state_stride_slot + state_head_offset_cg1;
                long long output_state_head_base_cg1 = (long long)state_slot_cg1 * output_state_stride_slot + state_head_offset_cg1;
                float* split_state_out = ((split_count_1 == 1) ? (output_state) : (bm));
                int split_state_slot = ((split_count_1 == 1) ? state_slot_cg1 : (batch_idx_1 * 2 + split_kind_1) * p_cap + split_seg_1);
                long long split_state_stride = ((split_count_1 == 1) ? output_state_stride_slot : (long long)num_o_heads_1 * 16384ll);
                output_state_head_base_cg1 = (long long)split_state_slot * split_state_stride + state_head_offset_cg1;
                if (USE_INITIAL_STATE != 0 && split_h0_chunks > 0) {
                    int warp_id_in_role_1 = warp - 4;
                    int state_warp_init = warp_id_in_role_1;
                    int state_row_top_init = state_warp_init * 16 + lane / 4;
                    int state_row_bot_init = state_row_top_init + 8;
                    long long state_base_top_init = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_top_init) * 128ll;
                    long long state_base_bot_init = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_bot_init) * 128ll;
                    if (state_warp_init < 4) {
                        if (seed_tile_ord_cg1 == 0) {
                            #pragma unroll
                            for (int state_col_group_init = 0; state_col_group_init < 8; state_col_group_init++) {
                                int state_col_pair_init = state_col_group_init * 8 + (lane & 3) * 2;
                                const int state_reg_base_init = state_col_group_init * 4;
                                seed_frag_0_cg1[state_reg_base_init] = initial_state[state_base_top_init + (long long)state_col_pair_init];
                                seed_frag_0_cg1[state_reg_base_init + 1] = initial_state[state_base_top_init + (long long)state_col_pair_init + 1ll];
                                seed_frag_0_cg1[state_reg_base_init + 2] = initial_state[state_base_bot_init + (long long)state_col_pair_init];
                                seed_frag_0_cg1[state_reg_base_init + 3] = initial_state[state_base_bot_init + (long long)state_col_pair_init + 1ll];
                            }
                            #pragma unroll
                            for (int state_col_group_init_1 = 0; state_col_group_init_1 < 8; state_col_group_init_1++) {
                                int state_col_pair_init_1 = 64 + state_col_group_init_1 * 8 + (lane & 3) * 2;
                                const int state_reg_base_init_1 = state_col_group_init_1 * 4;
                                seed_frag_1_cg1[state_reg_base_init_1] = initial_state[state_base_top_init + (long long)state_col_pair_init_1];
                                seed_frag_1_cg1[state_reg_base_init_1 + 1] = initial_state[state_base_top_init + (long long)state_col_pair_init_1 + 1ll];
                                seed_frag_1_cg1[state_reg_base_init_1 + 2] = initial_state[state_base_bot_init + (long long)state_col_pair_init_1];
                                seed_frag_1_cg1[state_reg_base_init_1 + 3] = initial_state[state_base_bot_init + (long long)state_col_pair_init_1 + 1ll];
                            }
                        }
                    }
                }
                if (seed_tile_ord_cg1 == 0 && split_seg_1 != 0) {
                    int warp_id_in_role_2 = warp - 4;
                    int split_seed_row = warp_id_in_role_2 * 16 + lane / 4;
                    int split_seed_col = (lane & 3) * 2;
                    seed_frag_0_cg1[0] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[1] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 1 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[2] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[3] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 1 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[4] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 8 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[5] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 9 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[6] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 8 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[7] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 9 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[8] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 16 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[9] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 17 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[10] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 16 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[11] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 17 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[12] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 24 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[13] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 25 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[14] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 24 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[15] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 25 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[16] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 32 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[17] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 33 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[18] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 32 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[19] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 33 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[20] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 40 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[21] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 41 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[22] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 40 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[23] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 41 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[24] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 48 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[25] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 49 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[26] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 48 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[27] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 49 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[28] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 56 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[29] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 57 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[30] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 56 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[31] = ((split_kind_1 != 0 && value_split_idx_1 == 0 && split_seed_col + 57 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[0] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[1] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 1 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[2] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[3] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 1 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[4] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 8 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[5] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 9 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[6] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 8 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[7] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 9 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[8] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 16 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[9] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 17 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[10] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 16 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[11] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 17 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[12] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 24 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[13] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 25 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[14] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 24 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[15] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 25 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[16] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 32 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[17] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 33 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[18] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 32 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[19] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 33 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[20] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 40 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[21] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 41 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[22] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 40 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[23] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 41 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[24] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 48 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[25] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 49 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[26] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 48 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[27] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 49 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[28] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 56 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[29] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 57 == split_seed_row) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[30] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 56 == split_seed_row + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[31] = ((split_kind_1 != 0 && value_split_idx_1 == 1 && split_seed_col + 57 == split_seed_row + 8) ? 1.0f : 0.0f);
                }
                int active_chunks_b = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_1_1 > 0)) != 0) ? 1 : num_chunks_b_1_1);
                if (active_chunks_b > 0) {
                    {
                        int chunk_offset_1 = batch_start_1_1;
                        int _cg1_marker = batch_idx_1 + head_idx_1 + chunk_offset_1 + batch_end_1_1 + checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE + ENABLE_CHECKPOINTS;
                        {
                            mbarrier_wait(load_gate_full_addr + (gate_k_cg1) * 8, _phase_load_gate_full_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        int gate_cg1_elem_base = gate_k_cg1 * 64;
                        int warp_in_wg = warp % 4;
                        int tmem_row_base_v = warp_in_wg * 32 << 16;
                        int lane_quad_cg1 = lane & 3;
                        float chunk_decay_cg1 = ((UNIT_GATES != 0) ? 1.0f : smem_cumprod[gate_cg1_elem_base + 64 - 1]);
                        int state_warp_cg1 = warp_in_wg;
                        if (state_warp_cg1 < 4) {
                            int state_addr_0 = taddr + (unsigned int)tmem_row_base_v;
                            int state_addr_1 = state_addr_0 + 64;
                            int state_inp_addr_0 = taddr + 192 + (unsigned int)tmem_row_base_v;
                            int state_inp_addr_1 = state_inp_addr_0 + 32;
                            unsigned int kk_scaled_cg0_bf16_5[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed_frag_0_cg1[_lp*2 + 0], seed_frag_0_cg1[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(state_inp_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_5[15])));
                            unsigned int kk_scaled_cg0_bf16_0[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed_frag_1_cg1[_lp*2 + 0], seed_frag_1_cg1[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(state_inp_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_0[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(state_inp_ready_addr);
                            const float2 _scale2_0 = {chunk_decay_cg1, chunk_decay_cg1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(seed_frag_0_cg1)[_ls], _scale2_0);
                            const float2 _scale2_1 = {chunk_decay_cg1, chunk_decay_cg1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(seed_frag_1_cg1)[_ls], _scale2_1);
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[15])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[16])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[17])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[18])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[19])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[20])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[21])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[22])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[23])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[24])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[25])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[26])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[27])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[28])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[29])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[30])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[31])));
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(state_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[15])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[16])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[17])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[18])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[19])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[20])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[21])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[22])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[23])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[24])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[25])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[26])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[27])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[28])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[29])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[30])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[31])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        if (state_warp_cg1 >= 4) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(state_inp_ready_addr);
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
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
                                float _exp2_2 = approx_exp2(last_cumsumlog_cg1 - cg1_scale_log);
                                cg1_decay_scale_frag[cg1_scale_j] = _exp2_2;
                            }
                        }
                        {
                            mbarrier_arrive(gate_cg1_empty_addr + (gate_k_cg1) * 8);
                            gate_k_cg1 += 1;
                            if (gate_k_cg1 == 5) { gate_k_cg1 = 0; _phase_load_gate_full_1 ^= 1; }
                        }
                        mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, _phase_load_v_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int v_stage_addr_cg1 = smem_v_mma_addr + v_cg1_stage * 16384;
                        unsigned int v_frag_cg1[16];
                        int v_warp_cg1 = warp_in_wg;
                        if (v_warp_cg1 < 4) {
                            unsigned int v_ld_bits_cg1[2];
                            #pragma unroll
                            for (int v_frag_repeat_cg1 = 0; v_frag_repeat_cg1 < 8; v_frag_repeat_cg1++) {
                                int v_ld_mtx_cg1 = lane / 8 & 1;
                                int v_ld_token_cg1 = v_frag_repeat_cg1 * 8 + (lane & 7);
                                int v_ld_dv_cg1 = v_warp_cg1 * 16 + v_ld_mtx_cg1 * 8;
                                int v_t0_cg1 = v_ld_token_cg1 & 15;
                                int v_t1_cg1 = v_ld_token_cg1 / 16;
                                int v_ld_linear_byte_off_cg1 = (v_ld_dv_cg1 + v_t0_cg1 * 64 + v_t1_cg1 * 1024) * 2;
                                int v_ld_byte_off_cg1 = v_ld_linear_byte_off_cg1 ^ (v_ld_linear_byte_off_cg1 >> 7 & 7) << 4;
                                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                    : "=r"(v_ld_bits_cg1[0]), "=r"(v_ld_bits_cg1[1])
                                    : "r"(v_stage_addr_cg1 + v_ld_byte_off_cg1)
                                    : "memory");
                                const int v_frag_j0_cg1 = v_frag_repeat_cg1 * 2;
                                v_frag_cg1[v_frag_j0_cg1] = v_ld_bits_cg1[0];
                                v_frag_cg1[v_frag_j0_cg1 + 1] = v_ld_bits_cg1[1];
                            }
                        }
                        if (elect_sync()) {
                            mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                        }
                        v_cg1_stage += 1;
                        if (v_cg1_stage == 3) { v_cg1_stage = 0; _phase_load_v_full ^= 1; }
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int ks_slice_warp_cg1 = warp_in_wg;
                        int ks_slice_dst_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
                        if (ks_slice_warp_cg1 < 4) {
                            int ks_slice_src_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
                            float ks_slice_cg1[4];
                            float ks_slice_v_delta_cg1[2];
                            unsigned int ks_slice_packed_delta_cg1[1];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 0))[_ls]);
                            const int ks_slice_pair_j_cg1_15_0 = 0;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_0)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_0)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_0] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_1 = 1;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_1)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_1)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_1] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 8));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 4))[_ls]);
                            const int ks_slice_pair_j_cg1_15_2 = 2;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_2)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_2)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_2] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_3 = 3;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_3)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_3)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_3] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 16));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 8))[_ls]);
                            const int ks_slice_pair_j_cg1_15_4 = 4;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_4)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_4)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_4] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_5 = 5;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_5)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_5)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_5] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 24));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 12))[_ls]);
                            const int ks_slice_pair_j_cg1_15_6 = 6;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_6)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_6)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_6] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_7 = 7;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_7)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_7)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_7] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 32));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 16))[_ls]);
                            const int ks_slice_pair_j_cg1_15_8 = 8;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_8)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_8)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_8] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_9 = 9;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_9)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_9)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_9] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 40));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 20))[_ls]);
                            const int ks_slice_pair_j_cg1_15_10 = 10;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_10)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_10)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_10] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_11 = 11;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_11)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_11)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_11] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 48));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 24))[_ls]);
                            const int ks_slice_pair_j_cg1_15_12 = 12;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_12)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_12)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_12] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_13 = 13;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_13)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_13)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_13] = ks_slice_packed_delta_cg1[0];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1[3]))
                                : "r"(ks_slice_src_cg1 + 56));
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag + 28))[_ls]);
                            const int ks_slice_pair_j_cg1_15_14 = 14;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_14)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_14)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 0))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_14] = ks_slice_packed_delta_cg1[0];
                            const int ks_slice_pair_j_cg1_15_15 = 15;
                            #pragma unroll
                            for (int _pair = 0; _pair < 1; _pair++) {
                                (&ks_slice_v_delta_cg1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_15)[_pair]) << 16);
                                (&ks_slice_v_delta_cg1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1 + ks_slice_pair_j_cg1_15_15)[_pair]) & 0xffff0000u);
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1 + 2))[_ls]);
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1[_lp*2 + 0], ks_slice_v_delta_cg1[_lp*2+1 + 0]));
                                ks_slice_packed_delta_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            v_frag_cg1[ks_slice_pair_j_cg1_15_15] = ks_slice_packed_delta_cg1[0];
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        if (ks_slice_warp_cg1 < 4) {
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(ks_slice_dst_cg1), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(vks_ready_addr);
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int qs_warp_cg1 = warp_in_wg;
                        if (qs_warp_cg1 < 4) {
                            int qs_addr = taddr + 128 + (unsigned int)tmem_row_base_v;
                            float _tmem_load_11[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[31]))
                                : "r"(qs_addr));
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag)[_ls]);
                            const float2 _scale2_3 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], _scale2_3);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                :: "r"(qs_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11[31])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(q_state_acc_empty_addr);
                        mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                        _phase_cg1_shared_acc_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float nv_frag_cg1[32];
                        unsigned int nv_packed_cg1[16];
                        int nv_load_warp_cg1 = warp_in_wg;
                        if (nv_load_warp_cg1 < 4) {
                            int nv_src_addr_cg1 = taddr + 384 + (unsigned int)tmem_row_base_v;
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1[31]))
                                : "r"(nv_src_addr_cg1));
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1[_lp*2 + 0], nv_frag_cg1[_lp*2+1 + 0]));
                                nv_packed_cg1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(cg1_shared_acc_empty_addr);
                        int nv_store_warp_cg1 = warp_in_wg;
                        if (nv_store_warp_cg1 < 4) {
                            int nv_dst_addr_cg1 = taddr + 448 + (unsigned int)tmem_row_base_v;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(nv_dst_addr_cg1), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(nv_ready_addr);
                            {
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(nv_frag_cg1)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag)[_ls]);
                                unsigned int kk_scaled_cg0_bf16_6[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1[_lp*2 + 0], nv_frag_cg1[_lp*2+1 + 0]));
                                    kk_scaled_cg0_bf16_6[_lp] = *(uint32_t*)&_bf2;
                                }
                                int decay_dst_addr_cg1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v;
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                    :: "r"(decay_dst_addr_cg1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_6[15])));
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(decay_v_ready_addr);
                            }
                        } else {
                            mbarrier_arrive(nv_ready_addr);
                            {
                                mbarrier_arrive(decay_v_ready_addr);
                            }
                        }
                        mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, _phase_o_smem_empty);
                        int o_stage_addr_cg1 = smem_o_addr + o_cg1_stage * 16384;
                        mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                        _phase_q_state_acc_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int output_warp_cg1 = warp_in_wg;
                        if (output_warp_cg1 < 4) {
                            int q_state_addr = taddr + 128 + (unsigned int)tmem_row_base_v;
                            float _tmem_load_5[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                                : "r"(q_state_addr));
                            unsigned int kk_scaled_cg0_bf16_7[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_7[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int token_group_cg1 = 0; token_group_cg1 < 4; token_group_cg1++) {
                                int o_mtx_idx_cg1 = lane / 8;
                                int o_row_addr_cg1 = lane & 7;
                                int o_dim_base_cg1 = output_warp_cg1 * 16 + (o_mtx_idx_cg1 & 1) * 8;
                                int o_token_base_cg1 = token_group_cg1 * 16 + o_mtx_idx_cg1 / 2 * 8;
                                int o_token_addr_cg1 = o_token_base_cg1 + o_row_addr_cg1;
                                int o_token_pair_cg1 = o_token_addr_cg1 / 2;
                                int o_token_parity_cg1 = o_token_addr_cg1 & 1;
                                int o_raw_row_cg1 = o_token_pair_cg1;
                                int o_raw_col_cg1 = (o_dim_base_cg1 & 63 ^ (o_token_pair_cg1 & 3) << 4 ^ o_token_parity_cg1 << 3) + o_token_parity_cg1 * 64;
                                int o_stsm_offset_cg1 = (o_raw_row_cg1 * 128 + o_raw_col_cg1) * 2;
                                const int o_pack_base_cg1 = token_group_cg1 * 4;
                                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1 + o_stsm_offset_cg1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[o_pack_base_cg1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[o_pack_base_cg1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[o_pack_base_cg1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_7[o_pack_base_cg1 + 3]))
                                    : "memory");
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (split_f32_on != 0) {
                                int split_o_dim = split_f32_dim + output_warp_cg1 * 16 + lane / 4;
                                int split_o_tok = chunk_offset_1 + (lane & 3) * 2;
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)split_o_tok * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(0) + 0]), "r"((unsigned int)(split_o_tok < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 1) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(1) + 0]), "r"((unsigned int)(split_o_tok + 1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)split_o_tok * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(2) + 0]), "r"((unsigned int)(split_o_tok < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 1) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(3) + 0]), "r"((unsigned int)(split_o_tok + 1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 8) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(4) + 0]), "r"((unsigned int)(split_o_tok + 8 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 9) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(5) + 0]), "r"((unsigned int)(split_o_tok + 9 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 8) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(6) + 0]), "r"((unsigned int)(split_o_tok + 8 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 9) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(7) + 0]), "r"((unsigned int)(split_o_tok + 9 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 16) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(8) + 0]), "r"((unsigned int)(split_o_tok + 16 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 17) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(9) + 0]), "r"((unsigned int)(split_o_tok + 17 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 16) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(10) + 0]), "r"((unsigned int)(split_o_tok + 16 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 17) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(11) + 0]), "r"((unsigned int)(split_o_tok + 17 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 24) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(12) + 0]), "r"((unsigned int)(split_o_tok + 24 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 25) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(13) + 0]), "r"((unsigned int)(split_o_tok + 25 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 24) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(14) + 0]), "r"((unsigned int)(split_o_tok + 24 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 25) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(15) + 0]), "r"((unsigned int)(split_o_tok + 25 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 32) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(16) + 0]), "r"((unsigned int)(split_o_tok + 32 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 33) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(17) + 0]), "r"((unsigned int)(split_o_tok + 33 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 32) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(18) + 0]), "r"((unsigned int)(split_o_tok + 32 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 33) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(19) + 0]), "r"((unsigned int)(split_o_tok + 33 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 40) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(20) + 0]), "r"((unsigned int)(split_o_tok + 40 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 41) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(21) + 0]), "r"((unsigned int)(split_o_tok + 41 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 40) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(22) + 0]), "r"((unsigned int)(split_o_tok + 40 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 41) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(23) + 0]), "r"((unsigned int)(split_o_tok + 41 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 48) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(24) + 0]), "r"((unsigned int)(split_o_tok + 48 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 49) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(25) + 0]), "r"((unsigned int)(split_o_tok + 49 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 48) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(26) + 0]), "r"((unsigned int)(split_o_tok + 48 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 49) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(27) + 0]), "r"((unsigned int)(split_o_tok + 49 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 56) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(28) + 0]), "r"((unsigned int)(split_o_tok + 56 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 57) * split_f32_stride + (long long)split_o_dim))), "f"(_tmem_load_5[(29) + 0]), "r"((unsigned int)(split_o_tok + 57 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 56) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(30) + 0]), "r"((unsigned int)(split_o_tok + 56 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok + 57) * split_f32_stride + (long long)(split_o_dim + 8)))), "f"(_tmem_load_5[(31) + 0]), "r"((unsigned int)(split_o_tok + 57 < batch_end_1_1)));
                            }
                            if (split_bf16_on != 0) {
                                int split_o_dim_1 = split_f32_dim + output_warp_cg1 * 16 + lane / 4;
                                int split_o_tok_1 = chunk_offset_1 + (lane & 3) * 2;
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)split_o_tok_1 * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[0])), "r"((unsigned int)(split_o_tok_1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 1) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[1])), "r"((unsigned int)(split_o_tok_1 + 1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)split_o_tok_1 * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[2])), "r"((unsigned int)(split_o_tok_1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 1) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[3])), "r"((unsigned int)(split_o_tok_1 + 1 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 8) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[4])), "r"((unsigned int)(split_o_tok_1 + 8 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 9) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[5])), "r"((unsigned int)(split_o_tok_1 + 9 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 8) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[6])), "r"((unsigned int)(split_o_tok_1 + 8 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 9) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[7])), "r"((unsigned int)(split_o_tok_1 + 9 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 16) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[8])), "r"((unsigned int)(split_o_tok_1 + 16 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 17) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[9])), "r"((unsigned int)(split_o_tok_1 + 17 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 16) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[10])), "r"((unsigned int)(split_o_tok_1 + 16 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 17) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[11])), "r"((unsigned int)(split_o_tok_1 + 17 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 24) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[12])), "r"((unsigned int)(split_o_tok_1 + 24 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 25) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[13])), "r"((unsigned int)(split_o_tok_1 + 25 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 24) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[14])), "r"((unsigned int)(split_o_tok_1 + 24 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 25) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[15])), "r"((unsigned int)(split_o_tok_1 + 25 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 32) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[16])), "r"((unsigned int)(split_o_tok_1 + 32 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 33) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[17])), "r"((unsigned int)(split_o_tok_1 + 33 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 32) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[18])), "r"((unsigned int)(split_o_tok_1 + 32 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 33) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[19])), "r"((unsigned int)(split_o_tok_1 + 33 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 40) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[20])), "r"((unsigned int)(split_o_tok_1 + 40 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 41) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[21])), "r"((unsigned int)(split_o_tok_1 + 41 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 40) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[22])), "r"((unsigned int)(split_o_tok_1 + 40 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 41) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[23])), "r"((unsigned int)(split_o_tok_1 + 41 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 48) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[24])), "r"((unsigned int)(split_o_tok_1 + 48 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 49) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[25])), "r"((unsigned int)(split_o_tok_1 + 49 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 48) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[26])), "r"((unsigned int)(split_o_tok_1 + 48 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 49) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[27])), "r"((unsigned int)(split_o_tok_1 + 49 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 56) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[28])), "r"((unsigned int)(split_o_tok_1 + 56 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 57) * split_f32_stride + (long long)split_o_dim_1))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[29])), "r"((unsigned int)(split_o_tok_1 + 57 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 56) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[30])), "r"((unsigned int)(split_o_tok_1 + 56 < batch_end_1_1)));
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_1 + 57) * split_f32_stride + (long long)(split_o_dim_1 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5[31])), "r"((unsigned int)(split_o_tok_1 + 57 < batch_end_1_1)));
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(q_state_acc_empty_addr);
                        mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                        o_cg1_stage += 1;
                        if (o_cg1_stage == 2) { o_cg1_stage = 0; _phase_o_smem_empty ^= 1; }
                    }
                }
                #pragma unroll 1
                for (int chunk_idx_1 = 1; chunk_idx_1 < active_chunks_b; chunk_idx_1++) {
                    int chunk_offset_2 = batch_start_1_1 + chunk_idx_1 * 64;
                    int _cg1_marker_1 = batch_idx_1 + head_idx_1 + chunk_offset_2 + batch_end_1_1 + checkpoint_every_n_tokens + USE_INITIAL_STATE + STORE_FINAL_STATE + ENABLE_CHECKPOINTS;
                    {
                        mbarrier_wait(load_gate_full_addr + (gate_k_cg1) * 8, _phase_load_gate_full_1);
                    }
                    int gate_cg1_elem_base_1 = gate_k_cg1 * 64;
                    int warp_in_wg_1 = warp % 4;
                    int tmem_row_base_v_1 = warp_in_wg_1 * 32 << 16;
                    int lane_quad_cg1_1 = lane & 3;
                    float chunk_decay_cg1_1 = ((UNIT_GATES != 0) ? 1.0f : smem_cumprod[gate_cg1_elem_base_1 + 64 - 1]);
                    mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                    _phase_kv_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int state_warp_cg1_1 = warp_in_wg_1;
                    if (state_warp_cg1_1 < 4) {
                        int state_addr_0_1 = taddr + (unsigned int)tmem_row_base_v_1;
                        int state_addr_1_1 = state_addr_0_1 + 64;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_0_cg1[31]))
                            : "r"(state_addr_0_1));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[0])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[1])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[2])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[3])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[4])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[5])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[6])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[7])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[8])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[9])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[10])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[11])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[12])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[13])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[14])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[15])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[16])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[17])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[18])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[19])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[20])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[21])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[22])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[23])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[24])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[25])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[26])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[27])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[28])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[29])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[30])), "=r"(*reinterpret_cast<uint32_t*>(&seed_frag_1_cg1[31]))
                            : "r"(state_addr_1_1));
                        int state_inp_addr_0_1 = taddr + 192 + (unsigned int)tmem_row_base_v_1;
                        int state_inp_addr_1_1 = state_inp_addr_0_1 + 32;
                        unsigned int kk_scaled_cg0_bf16_8[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed_frag_0_cg1[_lp*2 + 0], seed_frag_0_cg1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_0_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_8[15])));
                        unsigned int kk_scaled_cg0_bf16_1_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed_frag_1_cg1[_lp*2 + 0], seed_frag_1_cg1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(state_inp_addr_1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_1_1[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(state_inp_ready_addr);
                        const float2 _scale2_5 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(seed_frag_0_cg1)[_ls], _scale2_5);
                        const float2 _scale2_6 = {chunk_decay_cg1_1, chunk_decay_cg1_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(seed_frag_1_cg1)[_ls], _scale2_6);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(state_addr_0_1), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[15])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[16])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[17])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[18])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[19])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[20])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[21])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[22])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[23])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[24])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[25])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[26])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[27])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[28])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[29])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[30])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_0_cg1[31])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(state_addr_1_1), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[3])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[4])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[5])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[6])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[7])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[8])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[9])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[10])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[11])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[12])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[13])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[14])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[15])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[16])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[17])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[18])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[19])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[20])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[21])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[22])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[23])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[24])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[25])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[26])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[27])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[28])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[29])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[30])), "r"(*reinterpret_cast<const uint32_t*>(&seed_frag_1_cg1[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    if (state_warp_cg1_1 >= 4) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(state_inp_ready_addr);
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
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
                            float _exp2_7 = approx_exp2(last_cumsumlog_cg1_1 - cg1_scale_log_1);
                            cg1_decay_scale_frag_1[cg1_scale_j_1] = _exp2_7;
                        }
                    }
                    {
                        mbarrier_arrive(gate_cg1_empty_addr + (gate_k_cg1) * 8);
                        gate_k_cg1 += 1;
                        if (gate_k_cg1 == 5) { gate_k_cg1 = 0; _phase_load_gate_full_1 ^= 1; }
                    }
                    mbarrier_wait(load_v_full_addr + (v_cg1_stage) * 8, _phase_load_v_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int v_stage_addr_cg1_1 = smem_v_mma_addr + v_cg1_stage * 16384;
                    unsigned int v_frag_cg1_1[16];
                    int v_warp_cg1_1 = warp_in_wg_1;
                    if (v_warp_cg1_1 < 4) {
                        unsigned int v_ld_bits_cg1_1[2];
                        #pragma unroll
                        for (int v_frag_repeat_cg1_1 = 0; v_frag_repeat_cg1_1 < 8; v_frag_repeat_cg1_1++) {
                            int v_ld_mtx_cg1_1 = lane / 8 & 1;
                            int v_ld_token_cg1_1 = v_frag_repeat_cg1_1 * 8 + (lane & 7);
                            int v_ld_dv_cg1_1 = v_warp_cg1_1 * 16 + v_ld_mtx_cg1_1 * 8;
                            int v_t0_cg1_1 = v_ld_token_cg1_1 & 15;
                            int v_t1_cg1_1 = v_ld_token_cg1_1 / 16;
                            int v_ld_linear_byte_off_cg1_1 = (v_ld_dv_cg1_1 + v_t0_cg1_1 * 64 + v_t1_cg1_1 * 1024) * 2;
                            int v_ld_byte_off_cg1_1 = v_ld_linear_byte_off_cg1_1 ^ (v_ld_linear_byte_off_cg1_1 >> 7 & 7) << 4;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                : "=r"(v_ld_bits_cg1_1[0]), "=r"(v_ld_bits_cg1_1[1])
                                : "r"(v_stage_addr_cg1_1 + v_ld_byte_off_cg1_1)
                                : "memory");
                            const int v_frag_j0_cg1_1 = v_frag_repeat_cg1_1 * 2;
                            v_frag_cg1_1[v_frag_j0_cg1_1] = v_ld_bits_cg1_1[0];
                            v_frag_cg1_1[v_frag_j0_cg1_1 + 1] = v_ld_bits_cg1_1[1];
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(v_smem_empty_addr + (v_cg1_stage) * 8);
                    }
                    v_cg1_stage += 1;
                    if (v_cg1_stage == 3) { v_cg1_stage = 0; _phase_load_v_full ^= 1; }
                    mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                    _phase_cg1_shared_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int ks_slice_warp_cg1_1 = warp_in_wg_1;
                    int ks_slice_dst_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                    if (ks_slice_warp_cg1_1 < 4) {
                        int ks_slice_src_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                        float ks_slice_cg1_1[4];
                        float ks_slice_v_delta_cg1_1[2];
                        unsigned int ks_slice_packed_delta_cg1_1[1];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 0))[_ls]);
                        const int ks_slice_pair_j_cg1_15_16 = 0;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_16)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_16)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_16] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_17 = 1;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_17)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_17)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_17] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 8));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 4))[_ls]);
                        const int ks_slice_pair_j_cg1_15_18 = 2;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_18)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_18)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_18] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_19 = 3;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_19)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_19)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_19] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 16));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 8))[_ls]);
                        const int ks_slice_pair_j_cg1_15_20 = 4;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_20)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_20)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_20] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_21 = 5;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_21)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_21)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_21] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 24));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 12))[_ls]);
                        const int ks_slice_pair_j_cg1_15_22 = 6;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_22)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_22)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_22] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_23 = 7;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_23)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_23)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_23] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 32));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 16))[_ls]);
                        const int ks_slice_pair_j_cg1_15_24 = 8;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_24)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_24)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_24] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_25 = 9;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_25)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_25)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_25] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 40));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 20))[_ls]);
                        const int ks_slice_pair_j_cg1_15_26 = 10;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_26)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_26)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_26] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_27 = 11;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_27)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_27)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_27] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 48));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 24))[_ls]);
                        const int ks_slice_pair_j_cg1_15_28 = 12;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_28)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_28)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_28] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_29 = 13;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_29)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_29)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_29] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ks_slice_cg1_1[3]))
                            : "r"(ks_slice_src_cg1_1 + 56));
                        #pragma unroll
                        for (int _ls = 0; _ls < 2; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ks_slice_cg1_1)[_ls], reinterpret_cast<const float2*>((cg1_cumprod_frag_1 + 28))[_ls]);
                        const int ks_slice_pair_j_cg1_15_30 = 14;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_30)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_30)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 0))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_30] = ks_slice_packed_delta_cg1_1[0];
                        const int ks_slice_pair_j_cg1_15_31 = 15;
                        #pragma unroll
                        for (int _pair = 0; _pair < 1; _pair++) {
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_31)[_pair]) << 16);
                            (&ks_slice_v_delta_cg1_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>((v_frag_cg1_1 + ks_slice_pair_j_cg1_15_31)[_pair]) & 0xffff0000u);
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 1; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(ks_slice_v_delta_cg1_1)[_ls], reinterpret_cast<const float2*>((ks_slice_cg1_1 + 2))[_ls]);
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ks_slice_v_delta_cg1_1[_lp*2 + 0], ks_slice_v_delta_cg1_1[_lp*2+1 + 0]));
                            ks_slice_packed_delta_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        v_frag_cg1_1[ks_slice_pair_j_cg1_15_31] = ks_slice_packed_delta_cg1_1[0];
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg1_shared_acc_empty_addr);
                    if (ks_slice_warp_cg1_1 < 4) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(ks_slice_dst_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&v_frag_cg1_1[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(vks_ready_addr);
                    mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                    _phase_q_state_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int qs_warp_cg1_1 = warp_in_wg_1;
                    if (qs_warp_cg1_1 < 4) {
                        int qs_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1;
                        float _tmem_load_11_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11_1[31]))
                            : "r"(qs_addr_1));
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11_1)[_ls], reinterpret_cast<const float2*>(cg1_cumprod_frag_1)[_ls]);
                        const float2 _scale2_8 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11_1)[_ls], _scale2_8);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(qs_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_11_1[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(q_state_acc_empty_addr);
                    mbarrier_wait(cg1_shared_acc_full_addr, _phase_cg1_shared_acc_full_0);
                    _phase_cg1_shared_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float nv_frag_cg1_1[32];
                    unsigned int nv_packed_cg1_1[16];
                    int nv_load_warp_cg1_1 = warp_in_wg_1;
                    if (nv_load_warp_cg1_1 < 4) {
                        int nv_src_addr_cg1_1 = taddr + 384 + (unsigned int)tmem_row_base_v_1;
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&nv_frag_cg1_1[31]))
                            : "r"(nv_src_addr_cg1_1));
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1_1[_lp*2 + 0], nv_frag_cg1_1[_lp*2+1 + 0]));
                            nv_packed_cg1_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(cg1_shared_acc_empty_addr);
                    int nv_store_warp_cg1_1 = warp_in_wg_1;
                    if (nv_store_warp_cg1_1 < 4) {
                        int nv_dst_addr_cg1_1 = taddr + 448 + (unsigned int)tmem_row_base_v_1;
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(nv_dst_addr_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&nv_packed_cg1_1[15])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(nv_ready_addr);
                        {
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(nv_frag_cg1_1)[_ls], reinterpret_cast<const float2*>(cg1_decay_scale_frag_1)[_ls]);
                            unsigned int kk_scaled_cg0_bf16_9[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(nv_frag_cg1_1[_lp*2 + 0], nv_frag_cg1_1[_lp*2+1 + 0]));
                                kk_scaled_cg0_bf16_9[_lp] = *(uint32_t*)&_bf2;
                            }
                            int decay_dst_addr_cg1_1 = taddr + 448 + 32 + (unsigned int)tmem_row_base_v_1;
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x128b.x8.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(decay_dst_addr_cg1_1), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[3])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[4])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[5])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[6])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[7])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[8])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[9])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[10])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[11])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[12])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[13])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[14])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_9[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(decay_v_ready_addr);
                        }
                    } else {
                        mbarrier_arrive(nv_ready_addr);
                        {
                            mbarrier_arrive(decay_v_ready_addr);
                        }
                    }
                    mbarrier_wait(o_smem_empty_addr + (o_cg1_stage) * 8, _phase_o_smem_empty);
                    int o_stage_addr_cg1_1 = smem_o_addr + o_cg1_stage * 16384;
                    mbarrier_wait(q_state_acc_full_addr, _phase_q_state_acc_full_0);
                    _phase_q_state_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int output_warp_cg1_1 = warp_in_wg_1;
                    if (output_warp_cg1_1 < 4) {
                        int q_state_addr_1 = taddr + 128 + (unsigned int)tmem_row_base_v_1;
                        float _tmem_load_5_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5_1[31]))
                            : "r"(q_state_addr_1));
                        unsigned int kk_scaled_cg0_bf16_10[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5_1[_lp*2 + 0], _tmem_load_5_1[_lp*2+1 + 0]));
                            kk_scaled_cg0_bf16_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int token_group_cg1_1 = 0; token_group_cg1_1 < 4; token_group_cg1_1++) {
                            int o_mtx_idx_cg1_1 = lane / 8;
                            int o_row_addr_cg1_1 = lane & 7;
                            int o_dim_base_cg1_1 = output_warp_cg1_1 * 16 + (o_mtx_idx_cg1_1 & 1) * 8;
                            int o_token_base_cg1_1 = token_group_cg1_1 * 16 + o_mtx_idx_cg1_1 / 2 * 8;
                            int o_token_addr_cg1_1 = o_token_base_cg1_1 + o_row_addr_cg1_1;
                            int o_token_pair_cg1_1 = o_token_addr_cg1_1 / 2;
                            int o_token_parity_cg1_1 = o_token_addr_cg1_1 & 1;
                            int o_raw_row_cg1_1 = o_token_pair_cg1_1;
                            int o_raw_col_cg1_1 = (o_dim_base_cg1_1 & 63 ^ (o_token_pair_cg1_1 & 3) << 4 ^ o_token_parity_cg1_1 << 3) + o_token_parity_cg1_1 * 64;
                            int o_stsm_offset_cg1_1 = (o_raw_row_cg1_1 * 128 + o_raw_col_cg1_1) * 2;
                            const int o_pack_base_cg1_1 = token_group_cg1_1 * 4;
                            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>((unsigned long long)(o_stage_addr_cg1_1 + o_stsm_offset_cg1_1));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_10[o_pack_base_cg1_1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_10[o_pack_base_cg1_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_10[o_pack_base_cg1_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&kk_scaled_cg0_bf16_10[o_pack_base_cg1_1 + 3]))
                                : "memory");
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (split_f32_on != 0) {
                            int split_o_dim_2 = split_f32_dim + output_warp_cg1_1 * 16 + lane / 4;
                            int split_o_tok_2 = chunk_offset_2 + (lane & 3) * 2;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)split_o_tok_2 * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(0) + 0]), "r"((unsigned int)(split_o_tok_2 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 1) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(1) + 0]), "r"((unsigned int)(split_o_tok_2 + 1 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)split_o_tok_2 * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(2) + 0]), "r"((unsigned int)(split_o_tok_2 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 1) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(3) + 0]), "r"((unsigned int)(split_o_tok_2 + 1 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 8) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(4) + 0]), "r"((unsigned int)(split_o_tok_2 + 8 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 9) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(5) + 0]), "r"((unsigned int)(split_o_tok_2 + 9 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 8) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(6) + 0]), "r"((unsigned int)(split_o_tok_2 + 8 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 9) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(7) + 0]), "r"((unsigned int)(split_o_tok_2 + 9 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 16) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(8) + 0]), "r"((unsigned int)(split_o_tok_2 + 16 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 17) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(9) + 0]), "r"((unsigned int)(split_o_tok_2 + 17 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 16) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(10) + 0]), "r"((unsigned int)(split_o_tok_2 + 16 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 17) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(11) + 0]), "r"((unsigned int)(split_o_tok_2 + 17 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 24) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(12) + 0]), "r"((unsigned int)(split_o_tok_2 + 24 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 25) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(13) + 0]), "r"((unsigned int)(split_o_tok_2 + 25 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 24) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(14) + 0]), "r"((unsigned int)(split_o_tok_2 + 24 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 25) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(15) + 0]), "r"((unsigned int)(split_o_tok_2 + 25 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 32) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(16) + 0]), "r"((unsigned int)(split_o_tok_2 + 32 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 33) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(17) + 0]), "r"((unsigned int)(split_o_tok_2 + 33 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 32) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(18) + 0]), "r"((unsigned int)(split_o_tok_2 + 32 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 33) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(19) + 0]), "r"((unsigned int)(split_o_tok_2 + 33 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 40) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(20) + 0]), "r"((unsigned int)(split_o_tok_2 + 40 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 41) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(21) + 0]), "r"((unsigned int)(split_o_tok_2 + 41 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 40) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(22) + 0]), "r"((unsigned int)(split_o_tok_2 + 40 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 41) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(23) + 0]), "r"((unsigned int)(split_o_tok_2 + 41 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 48) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(24) + 0]), "r"((unsigned int)(split_o_tok_2 + 48 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 49) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(25) + 0]), "r"((unsigned int)(split_o_tok_2 + 49 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 48) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(26) + 0]), "r"((unsigned int)(split_o_tok_2 + 48 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 49) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(27) + 0]), "r"((unsigned int)(split_o_tok_2 + 49 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 56) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(28) + 0]), "r"((unsigned int)(split_o_tok_2 + 56 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 57) * split_f32_stride + (long long)split_o_dim_2))), "f"(_tmem_load_5_1[(29) + 0]), "r"((unsigned int)(split_o_tok_2 + 57 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 56) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(30) + 0]), "r"((unsigned int)(split_o_tok_2 + 56 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(split_f32_out + ((long long)(split_o_tok_2 + 57) * split_f32_stride + (long long)(split_o_dim_2 + 8)))), "f"(_tmem_load_5_1[(31) + 0]), "r"((unsigned int)(split_o_tok_2 + 57 < batch_end_1_1)));
                        }
                        if (split_bf16_on != 0) {
                            int split_o_dim_3 = split_f32_dim + output_warp_cg1_1 * 16 + lane / 4;
                            int split_o_tok_3 = chunk_offset_2 + (lane & 3) * 2;
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)split_o_tok_3 * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[0])), "r"((unsigned int)(split_o_tok_3 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 1) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[1])), "r"((unsigned int)(split_o_tok_3 + 1 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)split_o_tok_3 * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[2])), "r"((unsigned int)(split_o_tok_3 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 1) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[3])), "r"((unsigned int)(split_o_tok_3 + 1 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 8) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[4])), "r"((unsigned int)(split_o_tok_3 + 8 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 9) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[5])), "r"((unsigned int)(split_o_tok_3 + 9 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 8) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[6])), "r"((unsigned int)(split_o_tok_3 + 8 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 9) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[7])), "r"((unsigned int)(split_o_tok_3 + 9 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 16) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[8])), "r"((unsigned int)(split_o_tok_3 + 16 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 17) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[9])), "r"((unsigned int)(split_o_tok_3 + 17 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 16) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[10])), "r"((unsigned int)(split_o_tok_3 + 16 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 17) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[11])), "r"((unsigned int)(split_o_tok_3 + 17 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 24) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[12])), "r"((unsigned int)(split_o_tok_3 + 24 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 25) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[13])), "r"((unsigned int)(split_o_tok_3 + 25 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 24) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[14])), "r"((unsigned int)(split_o_tok_3 + 24 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 25) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[15])), "r"((unsigned int)(split_o_tok_3 + 25 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 32) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[16])), "r"((unsigned int)(split_o_tok_3 + 32 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 33) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[17])), "r"((unsigned int)(split_o_tok_3 + 33 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 32) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[18])), "r"((unsigned int)(split_o_tok_3 + 32 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 33) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[19])), "r"((unsigned int)(split_o_tok_3 + 33 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 40) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[20])), "r"((unsigned int)(split_o_tok_3 + 40 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 41) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[21])), "r"((unsigned int)(split_o_tok_3 + 41 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 40) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[22])), "r"((unsigned int)(split_o_tok_3 + 40 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 41) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[23])), "r"((unsigned int)(split_o_tok_3 + 41 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 48) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[24])), "r"((unsigned int)(split_o_tok_3 + 48 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 49) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[25])), "r"((unsigned int)(split_o_tok_3 + 49 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 48) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[26])), "r"((unsigned int)(split_o_tok_3 + 48 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 49) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[27])), "r"((unsigned int)(split_o_tok_3 + 49 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 56) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[28])), "r"((unsigned int)(split_o_tok_3 + 56 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 57) * split_f32_stride + (long long)split_o_dim_3))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[29])), "r"((unsigned int)(split_o_tok_3 + 57 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 56) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[30])), "r"((unsigned int)(split_o_tok_3 + 56 < batch_end_1_1)));
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(c + ((long long)(split_o_tok_3 + 57) * split_f32_stride + (long long)(split_o_dim_3 + 8)))), "h"(__bfloat16_as_ushort((__nv_bfloat16)_tmem_load_5_1[31])), "r"((unsigned int)(split_o_tok_3 + 57 < batch_end_1_1)));
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(q_state_acc_empty_addr);
                    mbarrier_arrive(o_store_ready_addr + (o_cg1_stage) * 8);
                    o_cg1_stage += 1;
                    if (o_cg1_stage == 2) { o_cg1_stage = 0; _phase_o_smem_empty ^= 1; }
                }
                if (seed_next_enabled_cg1 != 0) {
                    int warp_id_in_role_3 = warp - 4;
                    int state_warp_ahead = warp_id_in_role_3;
                    int state_row_top_ahead = state_warp_ahead * 16 + lane / 4;
                    int state_row_bot_ahead = state_row_top_ahead + 8;
                    long long state_base_top_ahead = seed_next_base_cg1 + (long long)state_row_top_ahead * 128ll;
                    long long state_base_bot_ahead = seed_next_base_cg1 + (long long)state_row_bot_ahead * 128ll;
                    #pragma unroll
                    for (int state_col_group_ahead = 0; state_col_group_ahead < 8; state_col_group_ahead++) {
                        int state_col_pair_ahead = state_col_group_ahead * 8 + (lane & 3) * 2;
                        const int state_reg_base_ahead = state_col_group_ahead * 4;
                        seed_frag_0_cg1[state_reg_base_ahead] = initial_state[state_base_top_ahead + (long long)state_col_pair_ahead];
                        seed_frag_0_cg1[state_reg_base_ahead + 1] = initial_state[state_base_top_ahead + (long long)state_col_pair_ahead + 1ll];
                        seed_frag_0_cg1[state_reg_base_ahead + 2] = initial_state[state_base_bot_ahead + (long long)state_col_pair_ahead];
                        seed_frag_0_cg1[state_reg_base_ahead + 3] = initial_state[state_base_bot_ahead + (long long)state_col_pair_ahead + 1ll];
                    }
                    #pragma unroll
                    for (int state_col_group_ahead_1 = 0; state_col_group_ahead_1 < 8; state_col_group_ahead_1++) {
                        int state_col_pair_ahead_1 = 64 + state_col_group_ahead_1 * 8 + (lane & 3) * 2;
                        const int state_reg_base_ahead_1 = state_col_group_ahead_1 * 4;
                        seed_frag_1_cg1[state_reg_base_ahead_1] = initial_state[state_base_top_ahead + (long long)state_col_pair_ahead_1];
                        seed_frag_1_cg1[state_reg_base_ahead_1 + 1] = initial_state[state_base_top_ahead + (long long)state_col_pair_ahead_1 + 1ll];
                        seed_frag_1_cg1[state_reg_base_ahead_1 + 2] = initial_state[state_base_bot_ahead + (long long)state_col_pair_ahead_1];
                        seed_frag_1_cg1[state_reg_base_ahead_1 + 3] = initial_state[state_base_bot_ahead + (long long)state_col_pair_ahead_1 + 1ll];
                    }
                } else {
                    seed_frag_0_cg1[0] = 0.0f;
                    seed_frag_0_cg1[1] = 0.0f;
                    seed_frag_0_cg1[2] = 0.0f;
                    seed_frag_0_cg1[3] = 0.0f;
                    seed_frag_0_cg1[4] = 0.0f;
                    seed_frag_0_cg1[5] = 0.0f;
                    seed_frag_0_cg1[6] = 0.0f;
                    seed_frag_0_cg1[7] = 0.0f;
                    seed_frag_0_cg1[8] = 0.0f;
                    seed_frag_0_cg1[9] = 0.0f;
                    seed_frag_0_cg1[10] = 0.0f;
                    seed_frag_0_cg1[11] = 0.0f;
                    seed_frag_0_cg1[12] = 0.0f;
                    seed_frag_0_cg1[13] = 0.0f;
                    seed_frag_0_cg1[14] = 0.0f;
                    seed_frag_0_cg1[15] = 0.0f;
                    seed_frag_0_cg1[16] = 0.0f;
                    seed_frag_0_cg1[17] = 0.0f;
                    seed_frag_0_cg1[18] = 0.0f;
                    seed_frag_0_cg1[19] = 0.0f;
                    seed_frag_0_cg1[20] = 0.0f;
                    seed_frag_0_cg1[21] = 0.0f;
                    seed_frag_0_cg1[22] = 0.0f;
                    seed_frag_0_cg1[23] = 0.0f;
                    seed_frag_0_cg1[24] = 0.0f;
                    seed_frag_0_cg1[25] = 0.0f;
                    seed_frag_0_cg1[26] = 0.0f;
                    seed_frag_0_cg1[27] = 0.0f;
                    seed_frag_0_cg1[28] = 0.0f;
                    seed_frag_0_cg1[29] = 0.0f;
                    seed_frag_0_cg1[30] = 0.0f;
                    seed_frag_0_cg1[31] = 0.0f;
                    seed_frag_1_cg1[0] = 0.0f;
                    seed_frag_1_cg1[1] = 0.0f;
                    seed_frag_1_cg1[2] = 0.0f;
                    seed_frag_1_cg1[3] = 0.0f;
                    seed_frag_1_cg1[4] = 0.0f;
                    seed_frag_1_cg1[5] = 0.0f;
                    seed_frag_1_cg1[6] = 0.0f;
                    seed_frag_1_cg1[7] = 0.0f;
                    seed_frag_1_cg1[8] = 0.0f;
                    seed_frag_1_cg1[9] = 0.0f;
                    seed_frag_1_cg1[10] = 0.0f;
                    seed_frag_1_cg1[11] = 0.0f;
                    seed_frag_1_cg1[12] = 0.0f;
                    seed_frag_1_cg1[13] = 0.0f;
                    seed_frag_1_cg1[14] = 0.0f;
                    seed_frag_1_cg1[15] = 0.0f;
                    seed_frag_1_cg1[16] = 0.0f;
                    seed_frag_1_cg1[17] = 0.0f;
                    seed_frag_1_cg1[18] = 0.0f;
                    seed_frag_1_cg1[19] = 0.0f;
                    seed_frag_1_cg1[20] = 0.0f;
                    seed_frag_1_cg1[21] = 0.0f;
                    seed_frag_1_cg1[22] = 0.0f;
                    seed_frag_1_cg1[23] = 0.0f;
                    seed_frag_1_cg1[24] = 0.0f;
                    seed_frag_1_cg1[25] = 0.0f;
                    seed_frag_1_cg1[26] = 0.0f;
                    seed_frag_1_cg1[27] = 0.0f;
                    seed_frag_1_cg1[28] = 0.0f;
                    seed_frag_1_cg1[29] = 0.0f;
                    seed_frag_1_cg1[30] = 0.0f;
                    seed_frag_1_cg1[31] = 0.0f;
                    int warp_id_in_role_4 = warp - 4;
                    int split_seed_row_1 = warp_id_in_role_4 * 16 + lane / 4;
                    int split_seed_col_1 = (lane & 3) * 2;
                    seed_frag_0_cg1[0] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[1] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 1 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[2] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[3] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 1 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[4] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 8 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[5] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 9 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[6] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 8 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[7] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 9 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[8] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 16 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[9] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 17 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[10] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 16 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[11] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 17 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[12] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 24 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[13] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 25 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[14] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 24 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[15] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 25 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[16] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 32 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[17] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 33 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[18] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 32 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[19] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 33 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[20] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 40 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[21] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 41 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[22] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 40 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[23] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 41 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[24] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 48 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[25] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 49 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[26] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 48 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[27] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 49 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[28] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 56 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[29] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 57 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[30] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 56 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_0_cg1[31] = ((split_next_kind != 0 && split_next_half == 0 && split_seed_col_1 + 57 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[0] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[1] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 1 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[2] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[3] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 1 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[4] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 8 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[5] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 9 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[6] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 8 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[7] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 9 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[8] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 16 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[9] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 17 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[10] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 16 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[11] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 17 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[12] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 24 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[13] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 25 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[14] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 24 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[15] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 25 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[16] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 32 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[17] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 33 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[18] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 32 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[19] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 33 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[20] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 40 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[21] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 41 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[22] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 40 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[23] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 41 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[24] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 48 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[25] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 49 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[26] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 48 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[27] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 49 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[28] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 56 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[29] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 57 == split_seed_row_1) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[30] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 56 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                    seed_frag_1_cg1[31] = ((split_next_kind != 0 && split_next_half == 1 && split_seed_col_1 + 57 == split_seed_row_1 + 8) ? 1.0f : 0.0f);
                }
                if ((STORE_FINAL_STATE != 0 || ENABLE_CHECKPOINTS != 0) && num_chunks_b_1_1 > 0) {
                    mbarrier_wait(kv_acc_full_addr, _phase_kv_acc_full_0);
                    _phase_kv_acc_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int warp_in_wg_2 = warp % 4;
                    int state_tmem_row_base_cg1 = warp_in_wg_2 * 32 << 16;
                    int warp_id_in_role_5 = warp - 4;
                    int state_warp_id_in_role_cg1 = warp_id_in_role_5;
                    int state_row_top_cg1 = state_warp_id_in_role_cg1 * 16 + lane / 4;
                    int state_row_bot_cg1 = state_row_top_cg1 + 8;
                    long long state_base_idx_top_cg1 = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_top_cg1) * 128ll;
                    long long state_base_idx_bot_cg1 = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + state_row_bot_cg1) * 128ll;
                    if (state_warp_id_in_role_cg1 < 4) {
                        #pragma unroll
                        for (int state_col_half_cg1 = 0; state_col_half_cg1 < 2; state_col_half_cg1++) {
                            int final_state_addr_cg1 = taddr + (unsigned int)state_tmem_row_base_cg1 + (unsigned int)(state_col_half_cg1 * 64);
                            float _tmem_load[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load[31]))
                                : "r"(final_state_addr_cg1));
                            #pragma unroll
                            for (int state_col_group_cg1 = 0; state_col_group_cg1 < 8; state_col_group_cg1++) {
                                int state_col_pair_cg1 = state_col_half_cg1 * 64 + state_col_group_cg1 * 8 + (lane & 3) * 2;
                                const int state_reg_base_cg1 = state_col_group_cg1 * 4;
                                {
                                    {
                                        auto _gst_10 = split_state_out + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1);
                                        __builtin_assume(__isGlobal(_gst_10));
                                        *(reinterpret_cast<float*>(_gst_10) + (0)) = _tmem_load[state_reg_base_cg1];
                                    }
                                    {
                                        auto _gst_11 = split_state_out + (state_base_idx_top_cg1 + (long long)state_col_pair_cg1 + 1ll);
                                        __builtin_assume(__isGlobal(_gst_11));
                                        *(reinterpret_cast<float*>(_gst_11) + (0)) = _tmem_load[state_reg_base_cg1 + 1];
                                    }
                                    {
                                        auto _gst_12 = split_state_out + (state_base_idx_bot_cg1 + (long long)state_col_pair_cg1);
                                        __builtin_assume(__isGlobal(_gst_12));
                                        *(reinterpret_cast<float*>(_gst_12) + (0)) = _tmem_load[state_reg_base_cg1 + 2];
                                    }
                                    {
                                        auto _gst_13 = split_state_out + (state_base_idx_bot_cg1 + (long long)state_col_pair_cg1 + 1ll);
                                        __builtin_assume(__isGlobal(_gst_13));
                                        *(reinterpret_cast<float*>(_gst_13) + (0)) = _tmem_load[state_reg_base_cg1 + 3];
                                    }
                                }
                            }
                        }
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    mbarrier_arrive(kv_state_drained_addr);
#else
#endif
                }
                if (STORE_FINAL_STATE != 0 && USE_INITIAL_STATE != 0 && split_empty_chunks == 0) {
                    int warp_id_in_role_6 = warp - 4;
                    int empty_state_warp = warp_id_in_role_6;
                    int empty_state_row_top = empty_state_warp * 16 + lane / 4;
                    int empty_state_row_bot = empty_state_row_top + 8;
                    int empty_state_col_lane = (lane & 3) * 4;
                    if (empty_state_warp < 4) {
                        #pragma unroll
                        for (int empty_col_group = 0; empty_col_group < 8; empty_col_group++) {
                            int empty_state_col = empty_col_group * 16 + empty_state_col_lane;
                            long long empty_initial_top_base = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_top) * 128ll;
                            long long empty_initial_bot_base = initial_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_bot) * 128ll;
                            long long empty_output_top_base = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_top) * 128ll;
                            long long empty_output_bot_base = output_state_head_base_cg1 + (long long)(value_split_idx_1 * 64 + empty_state_row_bot) * 128ll;
                            float _vec_load[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(initial_state + empty_initial_top_base + (long long)empty_state_col);
                                _vec_load[0 + 0] = _v4.x;
                                _vec_load[0 + 1] = _v4.y;
                                _vec_load[0 + 2] = _v4.z;
                                _vec_load[0 + 3] = _v4.w;
                            }
                            float _vec_load_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(initial_state + empty_initial_bot_base + (long long)empty_state_col);
                                _vec_load_1[0 + 0] = _v4.x;
                                _vec_load_1[0 + 1] = _v4.y;
                                _vec_load_1[0 + 2] = _v4.z;
                                _vec_load_1[0 + 3] = _v4.w;
                            }
                            {
                                float4 _v4 = make_float4(_vec_load[0 + 0], _vec_load[0 + 1], _vec_load[0 + 2], _vec_load[0 + 3]);
                                *reinterpret_cast<float4*>((output_state + (empty_output_top_base + (long long)empty_state_col)) + 0) = _v4;
                            }
                            {
                                float4 _v4 = make_float4(_vec_load_1[0 + 0], _vec_load_1[0 + 1], _vec_load_1[0 + 2], _vec_load_1[0 + 3]);
                                *reinterpret_cast<float4*>((output_state + (empty_output_bot_base + (long long)empty_state_col)) + 0) = _v4;
                            }
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                int split_seq_tiles_2 = num_o_heads_2 * (4 * p_cap - 2);
                int batch_idx_2 = (int)((unsigned int)tile_2 / (unsigned int)split_seq_tiles_2);
                int split_rest_2 = (int)tile_2 - batch_idx_2 * split_seq_tiles_2;
                int split_tail_2 = split_rest_2 - 2 * num_o_heads_2;
                int split_seg_2 = ((split_tail_2 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_2 >> NUM_O_HEADS_LOG2 + 2 : split_tail_2 / (4 * num_o_heads_2)) + 1);
                int split_pos_2 = ((split_tail_2 < 0) ? split_rest_2 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_2 & 4 * num_o_heads_2 - 1 : split_tail_2 % (4 * num_o_heads_2)));
                int split_kind_2 = ((split_pos_2 >= 2 * num_o_heads_2) ? 1 : 0);
                int split_half_2 = split_pos_2 - split_kind_2 * 2 * num_o_heads_2;
                int head_idx_2 = split_half_2 >> 1;
                int value_split_idx_2 = split_half_2 & 1;
                int qk_head_idx_2 = ((IS_GQA != 0) ? head_idx_2 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_v_heads / num_q_heads)));
                int v_head_idx_2 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_2 >> HEAD_GROUP_LOG2 : head_idx_2 / (num_q_heads / num_v_heads)) : head_idx_2);
                int batch_start_floor_2 = ((cu_seqlens[batch_idx_2] > 0) ? cu_seqlens[batch_idx_2] : 0);
                int batch_start_4 = ((batch_start_floor_2 < total_tokens) ? batch_start_floor_2 : total_tokens);
                int batch_end_floor_2 = ((cu_seqlens[batch_idx_2 + 1] > batch_start_4) ? cu_seqlens[batch_idx_2 + 1] : batch_start_4);
                int batch_end_4 = ((batch_end_floor_2 < total_tokens) ? batch_end_floor_2 : total_tokens);
                int seqlen_b_2 = batch_end_4 - batch_start_4;
                int num_pairs_b_2 = (seqlen_b_2 + 128 - 1) / 128;
                int num_chunks_b_4 = num_pairs_b_2 * 2;
                int split_len_2 = batch_end_4 - batch_start_4;
                int split_nchunks_2 = split_len_2 + 64 - 1 >> 6;
                int split_quota_2 = (int)((unsigned int)split_nchunks_2 / (unsigned int)c_min);
                int split_count_2 = ((split_quota_2 < 1) ? 1 : ((split_quota_2 > p_cap) ? p_cap : split_quota_2));
                int split_lo_2 = (int)((unsigned int)(split_seg_2 * split_nchunks_2) / (unsigned int)split_count_2) * 64;
                int split_hi_2 = (int)((unsigned int)((split_seg_2 + 1) * split_nchunks_2) / (unsigned int)split_count_2) * 64;
                int batch_start_4_1 = ((split_seg_2 < split_count_2) ? batch_start_4 + split_lo_2 : batch_end_4);
                int batch_end_4_1 = ((split_seg_2 < split_count_2) ? batch_start_4 + ((split_hi_2 > split_len_2) ? split_len_2 : split_hi_2) : batch_end_4);
                int num_chunks_b_4_1 = (batch_end_4_1 - batch_start_4_1 + 128 - 1 >> 7) * 2;
                int num_pairs_b_1_1 = num_chunks_b_4_1 / 2;
                #pragma unroll 1
                for (int _pair_idx = 0; _pair_idx < num_pairs_b_1_1; _pair_idx++) {
                    unsigned int stage_0 = k_cg0_stage;
                    mbarrier_wait(load_k_full_addr + (k_cg0_stage) * 8, _phase_load_k_full);
                    k_cg0_stage += 1;
                    if (k_cg0_stage == 4) { k_cg0_stage = 0; _phase_load_k_full ^= 1; }
                    mbarrier_wait(cg0_shared_acc_empty_addr, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
#else
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_0) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_0) * 1024);
#endif
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
#else
                    int _mma_a_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_1) * 1024);
#endif
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x7FFF) + (stage_2) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_0) * 1024);
#else
                    int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (stage_2) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_0) * 1024);
#endif
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x7FFF) + (stage_3) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (stage_1) * 1024);
#else
                    int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (stage_3) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (stage_1) * 1024);
#endif
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
                    elect_commit(load_k_empty_addr + (((k_cg0_stage + 2 >= 4) ? k_cg0_stage + 2 - 4 : k_cg0_stage + 2)) * 8);
                    elect_commit(load_k_empty_addr + (((k_cg0_stage + 3 >= 4) ? k_cg0_stage + 3 - 4 : k_cg0_stage + 3)) * 8);
                    elect_commit(q_smem_empty_addr + (q_cg0_stage) * 8);
                    elect_commit(q_smem_empty_addr + (((q_cg0_stage + 1 < 2) ? q_cg0_stage + 1 : (unsigned int)0)) * 8);
                    mma_cg0_pairs += 1;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                int split_seq_tiles_3 = num_o_heads_3 * (4 * p_cap - 2);
                int batch_idx_3 = (int)((unsigned int)tile_3 / (unsigned int)split_seq_tiles_3);
                int split_rest_3 = (int)tile_3 - batch_idx_3 * split_seq_tiles_3;
                int split_tail_3 = split_rest_3 - 2 * num_o_heads_3;
                int split_seg_3 = ((split_tail_3 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_3 >> NUM_O_HEADS_LOG2 + 2 : split_tail_3 / (4 * num_o_heads_3)) + 1);
                int split_pos_3 = ((split_tail_3 < 0) ? split_rest_3 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_3 & 4 * num_o_heads_3 - 1 : split_tail_3 % (4 * num_o_heads_3)));
                int split_kind_3 = ((split_pos_3 >= 2 * num_o_heads_3) ? 1 : 0);
                int split_half_3 = split_pos_3 - split_kind_3 * 2 * num_o_heads_3;
                int head_idx_3 = split_half_3 >> 1;
                int value_split_idx_3 = split_half_3 & 1;
                int qk_head_idx_3 = ((IS_GQA != 0) ? head_idx_3 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_v_heads / num_q_heads)));
                int v_head_idx_3 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_3 >> HEAD_GROUP_LOG2 : head_idx_3 / (num_q_heads / num_v_heads)) : head_idx_3);
                int batch_start_floor_3 = ((cu_seqlens[batch_idx_3] > 0) ? cu_seqlens[batch_idx_3] : 0);
                int batch_start_5 = ((batch_start_floor_3 < total_tokens) ? batch_start_floor_3 : total_tokens);
                int batch_end_floor_3 = ((cu_seqlens[batch_idx_3 + 1] > batch_start_5) ? cu_seqlens[batch_idx_3 + 1] : batch_start_5);
                int batch_end_5 = ((batch_end_floor_3 < total_tokens) ? batch_end_floor_3 : total_tokens);
                int seqlen_b_3 = batch_end_5 - batch_start_5;
                int num_pairs_b_3 = (seqlen_b_3 + 128 - 1) / 128;
                int num_chunks_b_5 = num_pairs_b_3 * 2;
                int split_len_3 = batch_end_5 - batch_start_5;
                int split_nchunks_3 = split_len_3 + 64 - 1 >> 6;
                int split_quota_3 = (int)((unsigned int)split_nchunks_3 / (unsigned int)c_min);
                int split_count_3 = ((split_quota_3 < 1) ? 1 : ((split_quota_3 > p_cap) ? p_cap : split_quota_3));
                int split_lo_3 = (int)((unsigned int)(split_seg_3 * split_nchunks_3) / (unsigned int)split_count_3) * 64;
                int split_hi_3 = (int)((unsigned int)((split_seg_3 + 1) * split_nchunks_3) / (unsigned int)split_count_3) * 64;
                int batch_start_6 = ((split_seg_3 < split_count_3) ? batch_start_5 + split_lo_3 : batch_end_5);
                int batch_end_6 = ((split_seg_3 < split_count_3) ? batch_start_5 + ((split_hi_3 > split_len_3) ? split_len_3 : split_hi_3) : batch_end_5);
                int num_chunks_b_6 = (batch_end_6 - batch_start_6 + 128 - 1 >> 7) * 2;
                int split_v_extent = ((split_kind_3 != 0) ? batch_start_6 : batch_end_6);
                int k_head_idx = ((num_q_heads >= num_v_heads) ? v_head_idx_3 : qk_head_idx_3);
                int value_split_offset = value_split_idx_3 * 64;
                if (elect_sync()) {
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base)), "r"((uint32_t)(batch_end_6)) : "memory");
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 128))), "r"((uint32_t)(batch_end_6)) : "memory");
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 256))), "r"((uint32_t)(split_v_extent)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 128))) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + cta_slot_base)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base + 256))) : "memory");
                }
                int num_pairs_tma = num_chunks_b_6 / 2;
                #pragma unroll 1
                for (int pair_idx_tma = 0; pair_idx_tma < num_pairs_tma; pair_idx_tma++) {
                    #pragma unroll
                    for (int chunk_in_pair = 0; chunk_in_pair < 2; chunk_in_pair++) {
                        int chunk_idx_2 = pair_idx_tma * 2 + chunk_in_pair;
                        int chunk_offset_3 = batch_start_6 + chunk_idx_2 * 64;
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
                        int chunk_offset_4 = batch_start_6 + chunk_idx_3 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(q_smem_empty_addr + (q_stage) * 8, _phase_q_smem_empty);
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
                        int chunk_offset_5 = batch_start_6 + chunk_idx_4 * 64;
                        if (elect_sync()) {
                            mbarrier_wait(v_smem_empty_addr + (v_stage) * 8, _phase_v_smem_empty);
                            mbarrier_arrive_expect_tx(load_v_full_addr + (v_stage) * 8, 8192);
                            tma_3d_gmem2smem(smem_v_mma_addr + v_stage * 16384, tensormap_workspace + (cta_slot_base + 256), value_split_offset, chunk_offset_5, v_head_idx_3, load_v_full_addr + (v_stage) * 8);
                        }
                        v_stage += 1;
                        if (v_stage == 3) { v_stage = 0; _phase_v_smem_empty ^= 1; }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
#if __CUDA_ARCH__ == 1070
#else
            unsigned int _phase_kv_acc_empty = 1;
            {
                _phase_kv_acc_empty ^= 1;
            }
#endif
            int kv_acc_release_pending = 1;
#if __CUDA_ARCH__ == 1070
            unsigned int _phase_kv_state_drained = 1;
#else
#endif
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
                int split_seq_tiles_4 = num_o_heads_4 * (4 * p_cap - 2);
                int batch_idx_4 = (int)((unsigned int)tile_4 / (unsigned int)split_seq_tiles_4);
                int split_rest_4 = (int)tile_4 - batch_idx_4 * split_seq_tiles_4;
                int split_tail_4 = split_rest_4 - 2 * num_o_heads_4;
                int split_seg_4 = ((split_tail_4 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_4 >> NUM_O_HEADS_LOG2 + 2 : split_tail_4 / (4 * num_o_heads_4)) + 1);
                int split_pos_4 = ((split_tail_4 < 0) ? split_rest_4 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_4 & 4 * num_o_heads_4 - 1 : split_tail_4 % (4 * num_o_heads_4)));
                int split_kind_4 = ((split_pos_4 >= 2 * num_o_heads_4) ? 1 : 0);
                int split_half_4 = split_pos_4 - split_kind_4 * 2 * num_o_heads_4;
                int head_idx_4 = split_half_4 >> 1;
                int value_split_idx_4 = split_half_4 & 1;
                int qk_head_idx_4 = ((IS_GQA != 0) ? head_idx_4 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_v_heads / num_q_heads)));
                int v_head_idx_4 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_4 >> HEAD_GROUP_LOG2 : head_idx_4 / (num_q_heads / num_v_heads)) : head_idx_4);
                int batch_start_floor_4 = ((cu_seqlens[batch_idx_4] > 0) ? cu_seqlens[batch_idx_4] : 0);
                int batch_start_7 = ((batch_start_floor_4 < total_tokens) ? batch_start_floor_4 : total_tokens);
                int batch_end_floor_4 = ((cu_seqlens[batch_idx_4 + 1] > batch_start_7) ? cu_seqlens[batch_idx_4 + 1] : batch_start_7);
                int batch_end_7 = ((batch_end_floor_4 < total_tokens) ? batch_end_floor_4 : total_tokens);
                int seqlen_b_4 = batch_end_7 - batch_start_7;
                int num_pairs_b_4 = (seqlen_b_4 + 128 - 1) / 128;
                int num_chunks_b_7 = num_pairs_b_4 * 2;
                int split_len_4 = batch_end_7 - batch_start_7;
                int split_nchunks_4 = split_len_4 + 64 - 1 >> 6;
                int split_quota_4 = (int)((unsigned int)split_nchunks_4 / (unsigned int)c_min);
                int split_count_4 = ((split_quota_4 < 1) ? 1 : ((split_quota_4 > p_cap) ? p_cap : split_quota_4));
                int split_lo_4 = (int)((unsigned int)(split_seg_4 * split_nchunks_4) / (unsigned int)split_count_4) * 64;
                int split_hi_4 = (int)((unsigned int)((split_seg_4 + 1) * split_nchunks_4) / (unsigned int)split_count_4) * 64;
                int batch_start_5_1 = ((split_seg_4 < split_count_4) ? batch_start_7 + split_lo_4 : batch_end_7);
                int batch_end_5_1 = ((split_seg_4 < split_count_4) ? batch_start_7 + ((split_hi_4 > split_len_4) ? split_len_4 : split_hi_4) : batch_end_7);
                int num_chunks_b_5_1 = (batch_end_5_1 - batch_start_5_1 + 128 - 1 >> 7) * 2;
                int active_chunks_b_1 = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_5_1 > 0)) != 0) ? 1 : num_chunks_b_5_1);
                if (active_chunks_b_1 > 0) {
#if __CUDA_ARCH__ == 1070
                    if (kv_acc_release_pending == 1) {
                        mbarrier_wait(kv_state_drained_addr, _phase_kv_state_drained);
                        _phase_kv_state_drained ^= 1;
#else
                    if (USE_INITIAL_STATE == 0 && kv_acc_release_pending == 1) {
                        mbarrier_wait(kv_acc_empty_addr, _phase_kv_acc_empty);
                        _phase_kv_acc_empty ^= 1;
#endif
                    }
                    {
                        int chunk_offset_6 = batch_start_5_1;
                        int _mma_marker = batch_idx_4 + head_idx_4 + chunk_offset_6 + batch_end_5_1 + 512;
                        {
                            mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, _phase_load_k_full_1);
                        }
                        mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, _phase_load_q_full_1);
                        mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                        _phase_state_inp_ready_0 ^= 1;
                        mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                        _phase_cg1_shared_acc_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int mma_b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
#else
                        int mma_b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_0 = elect_sync();
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x7FFF) + (q_stage_1) * 1024;
#else
                        int _smem_desc_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024;
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int mma_b_lo_1 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x7FFF) + (ainv_stage_mma) * 512);
#else
                        int mma_b_lo_1 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int mma_b_lo_2 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x7FFF) + (qk_stage_mma) * 512);
#else
                        int mma_b_lo_2 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
#if __CUDA_ARCH__ == 1070
                            mbarrier_wait(kv_acc_empty_addr, 0);
#else
                            mbarrier_wait(kv_acc_empty_addr, _phase_kv_acc_empty);
#endif
                            mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                            _phase_decay_v_ready_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                            int mma_b_lo_3 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x7FFF) | 0x2000000) + (k_stage_1) * 1024);
#else
                            int mma_b_lo_3 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (k_stage_1) * 1024);
#endif
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
                    "mov.b32 id, 69272720;\n\t"
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
#if __CUDA_ARCH__ == 1070
#else
                            _phase_kv_acc_empty ^= 1;
#endif
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
                    int chunk_offset_7 = batch_start_5_1 + chunk_idx_5 * 64;
                    int _mma_marker_1 = batch_idx_4 + head_idx_4 + chunk_offset_7 + batch_end_5_1 + 512;
                    {
                        mbarrier_wait(load_k_full_addr + (k_stage_1) * 8, _phase_load_k_full_1);
                    }
                    mbarrier_wait(load_q_full_addr + (q_stage_1) * 8, _phase_load_q_full_1);
                    mbarrier_wait(state_inp_ready_addr, _phase_state_inp_ready_0);
                    _phase_state_inp_ready_0 ^= 1;
                    mbarrier_wait(cg1_shared_acc_empty_addr, _phase_cg1_shared_acc_empty_0);
                    _phase_cg1_shared_acc_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int mma_b_lo_4 = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
#else
                    int mma_b_lo_4 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 1024);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_2 = elect_sync();
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x7FFF) + (q_stage_1) * 1024;
#else
                    int _smem_desc_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (q_stage_1) * 1024;
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int mma_b_lo_1_1 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x7FFF) + (ainv_stage_mma) * 512);
#else
                    int mma_b_lo_1_1 = make_warp_uniform((((smem_ainv_addr) >> 4) & 0x3FFF) + (ainv_stage_mma) * 512);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int mma_b_lo_2_1 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x7FFF) + (qk_stage_mma) * 512);
#else
                    int mma_b_lo_2_1 = make_warp_uniform((((smem_qk_addr) >> 4) & 0x3FFF) + (qk_stage_mma) * 512);
#endif
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
                    "mov.b32 id, 68158608;\n\t"
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
#if __CUDA_ARCH__ == 1070
                        int _parity_kv_acc_empty = (int)((unsigned int)chunk_idx_5 % 2);
                        mbarrier_wait(kv_acc_empty_addr, _parity_kv_acc_empty);
#else
                        mbarrier_wait(kv_acc_empty_addr, _phase_kv_acc_empty);
#endif
                        mbarrier_wait(decay_v_ready_addr, _phase_decay_v_ready_0);
                        _phase_decay_v_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int mma_b_lo_3_1 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x7FFF) | 0x2000000) + (k_stage_1) * 1024);
#else
                        int mma_b_lo_3_1 = make_warp_uniform(((((smem_k_trans_mma_addr) >> 4) & 0x3FFF) | 0x2000000) + (k_stage_1) * 1024);
#endif
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
                    "mov.b32 id, 69272720;\n\t"
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
#if __CUDA_ARCH__ == 1070
#else
                        _phase_kv_acc_empty ^= 1;
#endif
                    }
                    k_stage_1 += 1;
                    if (k_stage_1 == 4) { k_stage_1 = 0; _phase_load_k_full_1 ^= 1; }
                    q_stage_1 += 1;
                    if (q_stage_1 == 2) { q_stage_1 = 0; _phase_load_q_full_1 ^= 1; }
                    v_stage_mma += 1;
                    if (v_stage_mma == 3) { v_stage_mma = 0; _phase_load_v_full_1 ^= 1; }
                }
                if (num_chunks_b_5_1 > 0) {
                    kv_acc_release_pending = ((STORE_FINAL_STATE != 0 || ENABLE_CHECKPOINTS != 0) ? 1 : 0);
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
                int split_seq_tiles_5 = num_o_heads_5 * (4 * p_cap - 2);
                int batch_idx_5 = (int)((unsigned int)tile_5 / (unsigned int)split_seq_tiles_5);
                int split_rest_5 = (int)tile_5 - batch_idx_5 * split_seq_tiles_5;
                int split_tail_5 = split_rest_5 - 2 * num_o_heads_5;
                int split_seg_5 = ((split_tail_5 < 0) ? 0 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_5 >> NUM_O_HEADS_LOG2 + 2 : split_tail_5 / (4 * num_o_heads_5)) + 1);
                int split_pos_5 = ((split_tail_5 < 0) ? split_rest_5 : ((NUM_O_HEADS_LOG2 >= 0) ? split_tail_5 & 4 * num_o_heads_5 - 1 : split_tail_5 % (4 * num_o_heads_5)));
                int split_kind_5 = ((split_pos_5 >= 2 * num_o_heads_5) ? 1 : 0);
                int split_half_5 = split_pos_5 - split_kind_5 * 2 * num_o_heads_5;
                int head_idx_5 = split_half_5 >> 1;
                int value_split_idx_5 = split_half_5 & 1;
                int qk_head_idx_5 = ((IS_GQA != 0) ? head_idx_5 : ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_v_heads / num_q_heads)));
                int v_head_idx_5 = ((IS_GQA != 0) ? ((HEAD_GROUP_LOG2 >= 0) ? head_idx_5 >> HEAD_GROUP_LOG2 : head_idx_5 / (num_q_heads / num_v_heads)) : head_idx_5);
                int batch_start_floor_5 = ((cu_seqlens[batch_idx_5] > 0) ? cu_seqlens[batch_idx_5] : 0);
                int batch_start_8 = ((batch_start_floor_5 < total_tokens) ? batch_start_floor_5 : total_tokens);
                int batch_end_floor_5 = ((cu_seqlens[batch_idx_5 + 1] > batch_start_8) ? cu_seqlens[batch_idx_5 + 1] : batch_start_8);
                int batch_end_8 = ((batch_end_floor_5 < total_tokens) ? batch_end_floor_5 : total_tokens);
                int seqlen_b_5 = batch_end_8 - batch_start_8;
                int num_pairs_b_5 = (seqlen_b_5 + 128 - 1) / 128;
                int num_chunks_b_8 = num_pairs_b_5 * 2;
                int split_len_5 = batch_end_8 - batch_start_8;
                int split_nchunks_5 = split_len_5 + 64 - 1 >> 6;
                int split_quota_5 = (int)((unsigned int)split_nchunks_5 / (unsigned int)c_min);
                int split_count_5 = ((split_quota_5 < 1) ? 1 : ((split_quota_5 > p_cap) ? p_cap : split_quota_5));
                int split_lo_5 = (int)((unsigned int)(split_seg_5 * split_nchunks_5) / (unsigned int)split_count_5) * 64;
                int split_hi_5 = (int)((unsigned int)((split_seg_5 + 1) * split_nchunks_5) / (unsigned int)split_count_5) * 64;
                int batch_start_7_1 = ((split_seg_5 < split_count_5) ? batch_start_8 + split_lo_5 : batch_end_8);
                int batch_end_7_1 = ((split_seg_5 < split_count_5) ? batch_start_8 + ((split_hi_5 > split_len_5) ? split_len_5 : split_hi_5) : batch_end_8);
                int num_chunks_b_7_1 = (batch_end_7_1 - batch_start_7_1 + 128 - 1 >> 7) * 2;
                int split_o_extent = ((split_seg_5 == 0) ? batch_end_7_1 : batch_start_7_1);
                int value_split_offset_1 = value_split_idx_5 * 64;
                if (lane == 0) {
                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))), "r"((uint32_t)(split_o_extent)) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                }
                int num_valid_chunks_b = (batch_end_7_1 - batch_start_7_1 + 64 - 1) / 64;
                int gate_chunks_b = ((UNIT_GATES != 0) ? 0 : num_chunks_b_7_1);
                #pragma unroll
                for (int prefetch_idx = 0; prefetch_idx < 4; prefetch_idx++) {
                    if (gate_chunks_b > prefetch_idx) {
                        int prefetch_offset = batch_start_7_1 + prefetch_idx * 64;
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
                            if (token_0 < batch_end_7_1) {
                                gate_val_0 = gate[gate_idx_0];
                            }
                            if (token_1 < batch_end_7_1) {
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
                                :: "r"(beta_dst_0), "l"(beta + beta_idx_0), "r"((token_0 < batch_end_7_1) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1), "l"(beta + beta_idx_1), "r"((token_1 < batch_end_7_1) ? 4 : 0));
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
                int active_output_chunks_b = (((SINGLE_CHUNK_NO_STATE & (int)(num_chunks_b_7_1 > 0)) != 0) ? 1 : num_chunks_b_7_1);
                #pragma unroll 1
                for (int chunk_idx_6 = 0; chunk_idx_6 < active_output_chunks_b; chunk_idx_6++) {
                    int chunk_offset_8 = batch_start_7_1 + chunk_idx_6 * 64;
                    int prefetch_idx_1 = chunk_idx_6 + 5 - 1;
                    if (prefetch_idx_1 < gate_chunks_b) {
                        int prefetch_offset_1 = batch_start_7_1 + prefetch_idx_1 * 64;
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
                            if (token_0_1 < batch_end_7_1) {
                                gate_val_0_1 = gate[gate_idx_0_1];
                            }
                            if (token_1_1 < batch_end_7_1) {
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
                                :: "r"(beta_dst_0_1), "l"(beta + beta_idx_0_1), "r"((token_0_1 < batch_end_7_1) ? 4 : 0));
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(beta_dst_1_1), "l"(beta + beta_idx_1_1), "r"((token_1_1 < batch_end_7_1) ? 4 : 0));
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
                    mbarrier_wait(o_store_ready_addr + (o_epi_stage) * 8, _phase_o_store_ready);
                    if (lane == 0) {
                        if (chunk_idx_6 == 0) {
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_slot_base_epi + 384))) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        tma_store_3d(tensormap_workspace + (cta_slot_base_epi + 384), value_split_offset_1, chunk_offset_8, head_idx_5, smem_o_addr + o_epi_stage * 16384);
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
                if (active_output_chunks_b > 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                    mbarrier_arrive(o_smem_empty_addr + (o_release_stage) * 8);
                    o_release_stage += 1;
                    if (o_release_stage == 2) { o_release_stage = 0; }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
