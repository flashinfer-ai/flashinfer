/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_CB_TMEM_OFFSET 0
#define TMEM_Q_TMEM_OFFSET 256
#define TMEM_INTRA_TMEM_OFFSET 320
#define TMEM_STATE_DELTA_TMEM_OFFSET 384
#define TMEM_INTER_TMEM_OFFSET 448
#define NUM_PHASE1_PIPE_STAGES 2
#define NUM_PHASE3_PIPE_STAGES 2
#define NUM_STATE_PIPE_STAGES 2
#define NUM_INTRA1_ACC_PIPE_STAGES 2
#define SMEM_SMEM_B_OFF 1024
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_C_OFF 66560
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_X_TMA_OFF 132096
#define SMEM_SMEM_X_TMA_STAGE_BYTES 16384
#define SMEM_SMEM_X_TMA_STRIDE 16384
#define SMEM_SMEM_X_OFF 132096
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_SCALED_B_OFF 164864
#define SMEM_SMEM_SCALED_B_STAGE_BYTES 32768
#define SMEM_SMEM_SCALED_B_STRIDE 32768
#define SMEM_SMEM_STATE_OFF 164864
#define SMEM_SMEM_STATE_STAGE_BYTES 16384
#define SMEM_SMEM_STATE_STRIDE 16384
#define SMEM_SMEM_DELTA_ALL_OFF 230400
#define SMEM_SMEM_DELTA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_DELTA_ALL_STRIDE 1024
#define SMEM_SMEM_CUMSUM_ALL_OFF 231424
#define SMEM_SMEM_CUMSUM_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_CUMSUM_ALL_STRIDE 1024
#define SMEM_SMEM_Y_OFF 197632
#define SMEM_SMEM_Y_STAGE_BYTES 16384
#define SMEM_SMEM_Y_STRIDE 16384
#define SMEM_TOTAL 232448
#define THREADS 512
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


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_mamba_ssd_chunk_parallel_f16_batched(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap b_map, const __grid_constant__ CUtensorMap c_map, const __grid_constant__ CUtensorMap out_map, const __grid_constant__ CUtensorMap h_map, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, __half* __restrict__ delta_precomputed, float* __restrict__ cumsum_precomputed, float* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, __half* __restrict__ initial_states, __half* __restrict__ final_states, __half* __restrict__ checkpoint_states, int* __restrict__ checkpoint_token_indices, int* __restrict__ checkpoint_state_slots, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, int* __restrict__ seq_chunk_cumsum, __nv_bfloat16* __restrict__ out_native, float* __restrict__ s_work, unsigned int* __restrict__ h_words, unsigned int* __restrict__ grid_barrier, int nheads, int ngroups, int batch, int seqlen, int nchunks, int sequence_count, int num_logical_chunks, int mode_varlen, int D_mode, int has_z, int has_initial, int dt_softplus, float dt_min, float dt_max, int write_final_states, int checkpoint_state_count)
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
    #define b1_full_addr (mbar_base + 0)
    #define b1_empty_addr (mbar_base + 16)
    #define x1_full_addr (mbar_base + 32)
    #define x1_empty_addr (mbar_base + 48)
    #define aux1_full_addr (mbar_base + 64)
    #define aux1_empty_addr (mbar_base + 80)
    #define scaled_b_full_addr (mbar_base + 96)
    #define scaled_b_empty_addr (mbar_base + 104)
    #define state_delta_full_addr (mbar_base + 112)
    #define bc_full_addr (mbar_base + 120)
    #define bc_empty_addr (mbar_base + 136)
    #define x3_full_addr (mbar_base + 152)
    #define x3_empty_addr (mbar_base + 168)
    #define aux3_full_addr (mbar_base + 184)
    #define aux3_empty_addr (mbar_base + 200)
    #define state_full_addr (mbar_base + 216)
    #define state_empty_addr (mbar_base + 232)
    #define cb_full_addr (mbar_base + 248)
    #define cb_empty_addr (mbar_base + 264)
    #define q_full_addr (mbar_base + 280)
    #define q_empty_addr (mbar_base + 288)
    #define intra_full_addr (mbar_base + 296)
    #define inter_full_addr (mbar_base + 304)
    #define outputs_empty_addr (mbar_base + 312)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_C_OFF);
    const int smem_c_addr = smem + SMEM_SMEM_C_OFF;
    __nv_bfloat16* smem_x_tma = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_X_TMA_OFF);
    const int smem_x_tma_addr = smem + SMEM_SMEM_X_TMA_OFF;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_X_OFF);
    const int smem_x_addr = smem + SMEM_SMEM_X_OFF;
    __nv_bfloat16* smem_scaled_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_SCALED_B_OFF);
    const int smem_scaled_b_addr = smem + SMEM_SMEM_SCALED_B_OFF;
    __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_STATE_OFF);
    const int smem_state_addr = smem + SMEM_SMEM_STATE_OFF;
    float* smem_delta_all = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_DELTA_ALL_OFF);
    const int smem_delta_all_addr = smem + SMEM_SMEM_DELTA_ALL_OFF;
    float* smem_cumsum_all = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_CUMSUM_ALL_OFF);
    const int smem_cumsum_all_addr = smem + SMEM_SMEM_CUMSUM_ALL_OFF;
    __nv_bfloat16* smem_y = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Y_OFF);
    const int smem_y_addr = smem + SMEM_SMEM_Y_OFF;

    // Mbarrier init (24 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'phase1_pipe' ---
            // b1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // b1_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // x1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // x1_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // aux1_full: 2 barriers, init_count=32
            mbarrier_init(smem + 64, 32);
            mbarrier_init(smem + 72, 32);
            // aux1_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // scaled_b_full: 1 barriers, init_count=2
            mbarrier_init(smem + 96, 2);
            // scaled_b_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // state_delta_full: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // --- pipeline 'phase3_pipe' ---
            // bc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // bc_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 136, 2);
            mbarrier_init(smem + 144, 2);
            // x3_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // x3_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            // aux3_full: 2 barriers, init_count=32
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            // aux3_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // --- pipeline 'state_pipe' ---
            // state_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // state_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // --- pipeline 'intra1_acc_pipe' ---
            // cb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            // cb_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            // intra_full: 1 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            // inter_full: 1 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            // outputs_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 312, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 12) {
        int _tmem_hold = smem + 320;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_cb_tmem = taddr;
    const int tmem_q_tmem = taddr + 256;
    const int tmem_intra_tmem = taddr + 320;
    const int tmem_state_delta_tmem = taddr + 384;
    const int tmem_inter_tmem = taddr + 448;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: mma_inter ----
    if (warp == 0) {
        { // mma_inter_main
            int total_tiles = sequence_count * nchunks * nheads;
            int _uniform_3 = make_warp_uniform(total_tiles);
            total_tiles = _uniform_3;
            unsigned int stage1 = 0;
            unsigned int _phase_scaled_b_full_0 = 0;
            unsigned int _phase_x1_full = 0;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
                mbarrier_wait(scaled_b_full_addr, _phase_scaled_b_full_0);
                _phase_scaled_b_full_0 ^= 1;
                mbarrier_wait(x1_full_addr + (stage1) * 8, _phase_x1_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_2 = make_warp_uniform((((smem_scaled_b_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_2 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage1) * 1024);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                    }
                }
                elect_commit(state_delta_full_addr);
                elect_commit(scaled_b_empty_addr);
                elect_commit(x1_empty_addr + (stage1) * 8);
                stage1 += 1;
                if (stage1 == 2) { stage1 = 0; _phase_x1_full ^= 1; }
            }
            unsigned int _load_acquire_0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))) : "memory");
            unsigned int generation_seen = _load_acquire_0;
            unsigned int last_arrival = (unsigned int)(num_bids - 1);
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    __threadfence();
                    uint32_t _atomic_inc_old_0;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_inc_old_0) : "l"(&grid_barrier[0]), "r"(static_cast<uint32_t>(last_arrival)) : "memory");
                    unsigned int arrived = _atomic_inc_old_0;
                    if (arrived == last_arrival) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(generation_seen + 1)) >= static_cast<unsigned int>(2147483647));
                    }
                }
            }
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int _load_acquire_1;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_1) : "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))) : "memory");
            unsigned int generation_seen_0 = _load_acquire_1;
            unsigned int last_arrival_1 = (unsigned int)(num_bids - 1);
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    __threadfence();
                    uint32_t _atomic_inc_old_1;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_inc_old_1) : "l"(&grid_barrier[0]), "r"(static_cast<uint32_t>(last_arrival_1)) : "memory");
                    unsigned int arrived_1 = _atomic_inc_old_1;
                    if (arrived_1 == last_arrival_1) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))) : "memory");
                    } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(generation_seen_0 + 1)) >= static_cast<unsigned int>(2147483647));
                    }
                }
            }
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int stage3 = 0;
            unsigned int sstage = 0;
            unsigned int _phase_bc_full = 0;
            unsigned int _phase_state_full = 0;
            unsigned int _phase_outputs_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles; tile_1 += num_bids) {
                mbarrier_wait(bc_full_addr + (stage3) * 8, _phase_bc_full);
                mbarrier_wait(state_full_addr + (sstage) * 8, _phase_state_full);
                mbarrier_wait(outputs_empty_addr, _phase_outputs_empty_0);
                _phase_outputs_empty_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_3 = make_warp_uniform((((smem_c_addr) >> 4) & 0x3FFF) + (stage3) * 2048);
                int _mma_b_lo_3 = make_warp_uniform((((smem_state_addr) >> 4) & 0x3FFF) + (sstage) * 1024);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 506U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                    }
                }
                elect_commit(inter_full_addr);
                elect_commit(state_empty_addr + (sstage) * 8);
                elect_commit(bc_empty_addr + (stage3) * 8);
                stage3 += 1;
                if (stage3 == 2) { stage3 = 0; _phase_bc_full ^= 1; }
                sstage += 1;
                if (sstage == 2) { sstage = 0; _phase_state_full ^= 1; }
            }
        }
    }
    // ---- Role: mma_intra ----
    if (warp == 1) {
        { // mma_intra_main
            int total_tiles_1 = sequence_count * nchunks * nheads;
            int _uniform_2 = make_warp_uniform(total_tiles_1);
            total_tiles_1 = _uniform_2;
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int stage3_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_bc_full_1 = 0;
            unsigned int _phase_cb_empty = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_outputs_empty_0_1 = 1;
            unsigned int _phase_x3_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < total_tiles_1; tile_2 += num_bids) {
                mbarrier_wait(bc_full_addr + (stage3_1) * 8, _phase_bc_full_1);
                mbarrier_wait(cb_empty_addr + (acc_stage) * 8, _phase_cb_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_c_addr) >> 4) & 0x3FFF) + (stage3_1) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage3_1) * 2048);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_cb_tmem + (acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                }
                elect_commit(cb_full_addr + (acc_stage) * 8);
                elect_commit(bc_empty_addr + (stage3_1) * 8);
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(outputs_empty_addr, _phase_outputs_empty_0_1);
                _phase_outputs_empty_0_1 ^= 1;
                mbarrier_wait(x3_full_addr + (stage3_1) * 8, _phase_x3_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_1 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage3_1) * 1024);
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
                    "mov.b32 id, 135333008;\n\t"
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_intra_tmem), "r"(_mma_b_lo_1), "r"(tmem_q_tmem), "r"(0));
                elect_commit(intra_full_addr);
                elect_commit(q_empty_addr);
                elect_commit(x3_empty_addr + (stage3_1) * 8);
                stage3_1 += 1;
                if (stage3_1 == 2) { stage3_1 = 0; _phase_bc_full_1 ^= 1; _phase_x3_full ^= 1; }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_cb_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_bc ----
    if (warp == 2) {
        { // load_bc_main
            int total_tiles_2 = sequence_count * nchunks * nheads;
            int _uniform_0 = make_warp_uniform(total_tiles_2);
            total_tiles_2 = _uniform_0;
            unsigned int stage1_1 = 0;
            unsigned int _phase_b1_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_3 = bid; tile_3 < total_tiles_2; tile_3 += num_bids) {
                int logical = tile_3 / (unsigned int)nheads;
                int head = tile_3 % (unsigned int)nheads;
                int group = head * ngroups / nheads;
                int physical_chunk = 0;
                int physical_batch = 0;
                physical_batch = logical / nchunks;
                physical_chunk = logical - physical_batch * nchunks;
                int token_in_batch = physical_chunk * 128;
                mbarrier_wait(b1_empty_addr + (stage1_1) * 8, _phase_b1_empty);
                if (elect_sync()) {
                    tma_5d_gmem2smem(smem_b_addr + stage1_1 * 32768, (&b_map), 0, 0, group, token_in_batch, physical_batch, b1_full_addr + (stage1_1) * 8);
                    tma_5d_gmem2smem(smem_b_addr + stage1_1 * 32768 + 16384, (&b_map), 0, 1, group, token_in_batch, physical_batch, b1_full_addr + (stage1_1) * 8);
                    mbarrier_arrive_expect_tx(b1_full_addr + (stage1_1) * 8, 32768);
                }
                stage1_1 += 1;
                if (stage1_1 == 2) { stage1_1 = 0; _phase_b1_empty ^= 1; }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("fence.proxy.async;");
            unsigned int stage3_2 = 0;
            unsigned int sstage_1 = 0;
            unsigned int _phase_bc_empty = 1;
            unsigned int _phase_state_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_4 = bid; tile_4 < total_tiles_2; tile_4 += num_bids) {
                int logical_1 = tile_4 / (unsigned int)nheads;
                int head_1 = tile_4 % (unsigned int)nheads;
                int tile_row = logical_1 * nheads + head_1;
                int group_1 = head_1 * ngroups / nheads;
                int physical_chunk_1 = 0;
                int physical_batch_1 = 0;
                physical_batch_1 = logical_1 / nchunks;
                physical_chunk_1 = logical_1 - physical_batch_1 * nchunks;
                int token_in_batch_1 = physical_chunk_1 * 128;
                mbarrier_wait(bc_empty_addr + (stage3_2) * 8, _phase_bc_empty);
                if (elect_sync()) {
                    tma_5d_gmem2smem(smem_b_addr + stage3_2 * 32768, (&b_map), 0, 0, group_1, token_in_batch_1, physical_batch_1, bc_full_addr + (stage3_2) * 8);
                    tma_5d_gmem2smem(smem_b_addr + stage3_2 * 32768 + 16384, (&b_map), 0, 1, group_1, token_in_batch_1, physical_batch_1, bc_full_addr + (stage3_2) * 8);
                    tma_5d_gmem2smem(smem_c_addr + stage3_2 * 32768, (&c_map), 0, 0, group_1, token_in_batch_1, physical_batch_1, bc_full_addr + (stage3_2) * 8);
                    tma_5d_gmem2smem(smem_c_addr + stage3_2 * 32768 + 16384, (&c_map), 0, 1, group_1, token_in_batch_1, physical_batch_1, bc_full_addr + (stage3_2) * 8);
                    mbarrier_arrive_expect_tx(bc_full_addr + (stage3_2) * 8, 65536);
                }
                mbarrier_wait(state_empty_addr + (sstage_1) * 8, _phase_state_empty);
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_state_addr + sstage_1 * 16384, (&h_map), 0, 0, 0, tile_row, state_full_addr + (sstage_1) * 8);
                    tma_4d_gmem2smem(smem_state_addr + sstage_1 * 16384 + 8192, (&h_map), 0, 1, 0, tile_row, state_full_addr + (sstage_1) * 8);
                    mbarrier_arrive_expect_tx(state_full_addr + (sstage_1) * 8, 16384);
                }
                stage3_2 += 1;
                if (stage3_2 == 2) { stage3_2 = 0; _phase_bc_empty ^= 1; }
                sstage_1 += 1;
                if (sstage_1 == 2) { sstage_1 = 0; _phase_state_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_x_dt ----
    if (warp == 3) {
        { // load_x_dt_main
            int total_tiles_3 = sequence_count * nchunks * nheads;
            int _uniform_1 = make_warp_uniform(total_tiles_3);
            total_tiles_3 = _uniform_1;
            unsigned int stage1_2 = 0;
            unsigned int _phase_aux1_empty = 1;
            unsigned int _phase_x1_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_5 = bid; tile_5 < total_tiles_3; tile_5 += num_bids) {
                int logical_2 = tile_5 / (unsigned int)nheads;
                int head_2 = tile_5 % (unsigned int)nheads;
                int tile_row_1 = logical_2 * nheads + head_2;
                int physical_chunk_2 = 0;
                int physical_batch_2 = 0;
                int segment_offset = 0;
                int segment_limit = 128;
                physical_batch_2 = logical_2 / nchunks;
                physical_chunk_2 = logical_2 - physical_batch_2 * nchunks;
                int chunk_tokens = seqlen - physical_chunk_2 * 128;
                if (chunk_tokens > 128) {
                    chunk_tokens = 128;
                }
                mbarrier_wait(aux1_empty_addr + (stage1_2) * 8, _phase_aux1_empty);
                mbarrier_wait(x1_empty_addr + (stage1_2) * 8, _phase_x1_empty);
                int token_in_batch_2 = physical_chunk_2 * 128;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_x_tma_addr + stage1_2 * 16384, (&x_map), 0, head_2, token_in_batch_2, physical_batch_2, x1_full_addr + (stage1_2) * 8);
                    mbarrier_arrive_expect_tx(x1_full_addr + (stage1_2) * 8, 16384);
                }
                int group_start = lane * 4;
                #pragma unroll
                for (int local = 0; local < 4; local++) {
                    int physical_token = group_start + local;
                    int factor_token = physical_token - segment_offset;
                    float cumsum_value = 0.0f;
                    float delta_value = 0.0f;
                    int source_token = physical_token;
                    if (source_token >= chunk_tokens) {
                        source_token = chunk_tokens - 1;
                    }
                    cumsum_value = cumsum_precomputed[tile_row_1 * 128 + source_token];
                    if (physical_token < chunk_tokens) {
                        float _cvt_f32_0 = __half2float(delta_precomputed[tile_row_1 * 128 + physical_token]);
                        delta_value = _cvt_f32_0;
                    }
                    smem_cumsum_all[stage1_2 * 128 + (unsigned int)physical_token] = cumsum_value;
                    smem_delta_all[stage1_2 * 128 + (unsigned int)physical_token] = delta_value;
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                mbarrier_arrive(aux1_full_addr + (stage1_2) * 8);
                stage1_2 += 1;
                if (stage1_2 == 2) { stage1_2 = 0; _phase_aux1_empty ^= 1; _phase_x1_empty ^= 1; }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int stage3_3 = 0;
            unsigned int _phase_aux3_empty = 1;
            unsigned int _phase_x3_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_6 = bid; tile_6 < total_tiles_3; tile_6 += num_bids) {
                int logical_3 = tile_6 / (unsigned int)nheads;
                int head_3 = tile_6 % (unsigned int)nheads;
                int tile_row_2 = logical_3 * nheads + head_3;
                int physical_chunk_3 = 0;
                int physical_batch_3 = 0;
                int segment_offset_1 = 0;
                int segment_limit_1 = 128;
                physical_batch_3 = logical_3 / nchunks;
                physical_chunk_3 = logical_3 - physical_batch_3 * nchunks;
                int chunk_tokens_1 = seqlen - physical_chunk_3 * 128;
                if (chunk_tokens_1 > 128) {
                    chunk_tokens_1 = 128;
                }
                mbarrier_wait(aux3_empty_addr + (stage3_3) * 8, _phase_aux3_empty);
                mbarrier_wait(x3_empty_addr + (stage3_3) * 8, _phase_x3_empty);
                int token_in_batch_3 = physical_chunk_3 * 128;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_x_tma_addr + stage3_3 * 16384, (&x_map), 0, head_3, token_in_batch_3, physical_batch_3, x3_full_addr + (stage3_3) * 8);
                    mbarrier_arrive_expect_tx(x3_full_addr + (stage3_3) * 8, 16384);
                }
                int group_start_1 = lane * 4;
                #pragma unroll
                for (int local_1 = 0; local_1 < 4; local_1++) {
                    int physical_token_1 = group_start_1 + local_1;
                    int factor_token_1 = physical_token_1 - segment_offset_1;
                    float cumsum_value_1 = 0.0f;
                    float delta_value_1 = 0.0f;
                    int source_token_1 = physical_token_1;
                    if (source_token_1 >= chunk_tokens_1) {
                        source_token_1 = chunk_tokens_1 - 1;
                    }
                    cumsum_value_1 = cumsum_precomputed[tile_row_2 * 128 + source_token_1];
                    if (physical_token_1 < chunk_tokens_1) {
                        float _cvt_f32_1 = __half2float(delta_precomputed[tile_row_2 * 128 + physical_token_1]);
                        delta_value_1 = _cvt_f32_1;
                    }
                    smem_cumsum_all[stage3_3 * 128 + (unsigned int)physical_token_1] = cumsum_value_1;
                    smem_delta_all[stage3_3 * 128 + (unsigned int)physical_token_1] = delta_value_1;
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                mbarrier_arrive(aux3_full_addr + (stage3_3) * 8);
                stage3_3 += 1;
                if (stage3_3 == 2) { stage3_3 = 0; _phase_aux3_empty ^= 1; _phase_x3_empty ^= 1; }
            }
        }
    }
    // ---- Role: pre_inter ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // pre_inter_main
            int taddr_0 = taddr;
            unsigned int stage1_3 = 0;
            int total_tiles_4 = sequence_count * nchunks * nheads;
            int _uniform_8 = make_warp_uniform(total_tiles_4);
            total_tiles_4 = _uniform_8;
            int b_state_base = (warp % 4 * 4 + lane / 8) * 8;
            int b_col_lane = lane % 8;
            unsigned int _phase_aux1_full = 0;
            unsigned int _phase_b1_full = 0;
            unsigned int _phase_scaled_b_empty_0 = 1;
            unsigned int _phase_state_delta_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_7 = bid; tile_7 < total_tiles_4; tile_7 += num_bids) {
                int logical_4 = tile_7 / (unsigned int)nheads;
                int head_4 = tile_7 % (unsigned int)nheads;
                int tile_row_3 = logical_4 * nheads + head_4;
                int physical_chunk_4 = 0;
                int segment_offset_2 = 0;
                int segment_limit_2 = 128;
                int sequence = logical_4 / nchunks;
                physical_chunk_4 = logical_4 - sequence * nchunks;
                int chunk_tokens_2 = seqlen - physical_chunk_4 * 128;
                if (chunk_tokens_2 > 128) {
                    chunk_tokens_2 = 128;
                }
                mbarrier_wait(aux1_full_addr + (stage1_3) * 8, _phase_aux1_full);
                mbarrier_wait(b1_full_addr + (stage1_3) * 8, _phase_b1_full);
                float last_cumsum = smem_cumsum_all[stage1_3 * 128 + (unsigned int)segment_limit_2 - 1];
                mbarrier_wait(scaled_b_empty_addr, _phase_scaled_b_empty_0);
                _phase_scaled_b_empty_0 ^= 1;
                #pragma unroll 1
                for (int b_col_iter = 0; b_col_iter < 8; b_col_iter++) {
                    int col = b_col_iter * 8 + b_col_lane;
                    float scaled_b_values[8];
                    #pragma unroll
                    for (int b_local = 0; b_local < 8; b_local++) {
                        scaled_b_values[b_local] = 0.0f;
                    }
                    if (col >= segment_offset_2 && col < segment_limit_2) {
                        unsigned int b_packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&b_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 3]))
                            : "r"((smem_b_addr + stage1_3 * 32768 + (unsigned int)(b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 ^ (b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 >> 7 & 7) << 4))));
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&scaled_b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_packed[_pair]) << 16);
                            (&scaled_b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_packed[_pair]) & 0xffff0000u);
                        }
                        float _exp2_3 = approx_exp2((last_cumsum - smem_cumsum_all[stage1_3 * 128 + (unsigned int)col]) * 1.4426950408889634f);
                        float b_scale = _exp2_3;
                        b_scale *= smem_delta_all[stage1_3 * 128 + (unsigned int)col];
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {b_scale, b_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(scaled_b_values)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++) {
                            scaled_b_values[_ls] = scaled_b_values[_ls] * b_scale;
                        }
                        #endif
                    }
                    #pragma unroll
                    for (int b_local_1 = 0; b_local_1 < 8; b_local_1++) {
                        {
                            __nv_bfloat16 _bval_1 = __float2bfloat16_rn(scaled_b_values[b_local_1]);
                            uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                            uint32_t _addr_1 = static_cast<uint32_t>((smem_scaled_b_addr + (unsigned int)(col / 64 * 16384 + (b_state_base + b_local_1) * 128 + col % 64 * 2 ^ (col / 64 * 16384 + (b_state_base + b_local_1) * 128 + col % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive(scaled_b_full_addr);
                        mbarrier_arrive(b1_empty_addr + (stage1_3) * 8);
                        mbarrier_arrive(aux1_empty_addr + (stage1_3) * 8);
                    }
                }
                mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0);
                _phase_state_delta_full_0 ^= 1;
                int s_tile_base = tile_row_3 * 8192;
                #pragma unroll
                for (int row_half = 0; row_half < 2; row_half++) {
                    int state_row_origin = warp % 4 * 32 + row_half * 16;
                    float _tmem_load_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                        : "r"(taddr_0 + (state_row_origin << 16) + 384));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int state_local = 0; state_local < 16; state_local++) {
                        int state_reg = state_local % 4;
                        int state_repeat = state_local / 4;
                        int state_row = state_row_origin + lane / 4 + state_reg / 2 * 8;
                        int dim = state_repeat * 8 + lane % 4 * 2 + state_reg % 2;
                        s_work[s_tile_base + dim * 128 + state_row] = _tmem_load_2[state_local];
                    }
                }
                stage1_3 += 1;
                if (stage1_3 == 2) { stage1_3 = 0; _phase_aux1_full ^= 1; _phase_b1_full ^= 1; }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            int heavy_tid = (warp - 4) * 32 + lane;
            int total_pairs = sequence_count * nheads * 4096;
            #pragma unroll 1
            for (int pair = bid * 384 + heavy_tid; pair < total_pairs; pair += num_bids * 384) {
                int work = pair / 4096;
                int local_pair = pair - work * 4096;
                int sequence_1 = work / nheads;
                int head_5 = work - sequence_1 * nheads;
                int elem = local_pair * 2;
                int state_head_base = work * 8192;
                float h_pair[2];
                h_pair[0] = 0.0f;
                h_pair[1] = 0.0f;
                if (has_initial != 0) {
                    h_pair[0] = (float)initial_states[state_head_base + elem];
                    h_pair[1] = (float)initial_states[state_head_base + elem + 1];
                }
                int first_logical = sequence_1 * nchunks;
                int logical_end = first_logical + nchunks;
                int _uniform_9 = make_warp_uniform(first_logical);
                first_logical = _uniform_9;
                int _uniform_10 = make_warp_uniform(logical_end);
                logical_end = _uniform_10;
                int checkpoint_token = -1;
                if (checkpoint_state_count > 0) {
                    checkpoint_token = checkpoint_token_indices[sequence_1];
                }
                if (first_logical < logical_end) {
                    uint32_t h_pair_bf16[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair[_lp*2 + 0], h_pair[_lp*2+1 + 0]));
                        h_pair_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    h_words[(first_logical * nheads + head_5) * 4096 + local_pair] = h_pair_bf16[0];
                }
                #pragma unroll 1
                for (int logical_5 = first_logical; logical_5 < logical_end; logical_5 += 16) {
                    float s_pre[32];
                    float d_pre[16];
                    #pragma unroll
                    for (int k = 0; k < 16; k++) {
                        int _min_1 = ((logical_5 + k) < (logical_end - 1) ? (logical_5 + k) : (logical_end - 1));
                        int look = _min_1;
                        int look_tile = look * nheads + head_5;
                        int look_physical = 0;
                        int look_offset = 0;
                        int look_limit = 128;
                        look_physical = look - sequence_1 * nchunks;
                        int look_tokens = seqlen - look_physical * 128;
                        if (look_tokens > 128) {
                            look_tokens = 128;
                        }
                        int source_token_2 = look_tokens - 1;
                        float last_cumsum_1 = cumsum_precomputed[look_tile * 128 + source_token_2];
                        float segment_base = 0.0f;
                        float _exp2_4 = approx_exp2((last_cumsum_1 - segment_base) * 1.4426950408889634f);
                        d_pre[k] = _exp2_4;
                        {
                            float2 _v2_2 = *reinterpret_cast<const float2*>(s_work + look_tile * 8192 + elem);
                            s_pre[2 * k] = _v2_2.x;
                            s_pre[2 * k + 1] = _v2_2.y;
                        }
                    }
                    #pragma unroll
                    for (int k_1 = 0; k_1 < 16; k_1++) {
                        int current = logical_5 + k_1;
                        if (current < logical_end) {
                            float decay = d_pre[k_1];
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_3 = {decay, decay};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(h_pair)[_ls], _scale2_3);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++) {
                                h_pair[_ls] = h_pair[_ls] * decay;
                            }
                            #endif
                            h_pair[0] = h_pair[0] + s_pre[2 * k_1];
                            h_pair[1] = h_pair[1] + s_pre[2 * k_1 + 1];
                            if (logical_end > current + 1) {
                                uint32_t h_pair_bf16_1[1];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair[_lp*2 + 0], h_pair[_lp*2+1 + 0]));
                                    h_pair_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                h_words[((current + 1) * nheads + head_5) * 4096 + local_pair] = h_pair_bf16_1[0];
                            }
                            if (checkpoint_state_count > 0) {
                                int cur_physical = 0;
                                int cur_limit = 128;
                                cur_physical = current - sequence_1 * nchunks;
                                int cur_tokens = seqlen - cur_physical * 128;
                                if (cur_tokens > 128) {
                                    cur_tokens = 128;
                                }
                                int segment_end = cur_physical * 128 + cur_limit;
                                if (segment_end > seqlen) {
                                    segment_end = seqlen;
                                }
                                if (checkpoint_token == segment_end) {
                                    int checkpoint_slot = checkpoint_state_slots[sequence_1];
                                    if (checkpoint_slot >= 0 && checkpoint_slot < checkpoint_state_count) {
                                        int checkpoint_head_base = (checkpoint_slot * nheads + head_5) * 8192;
                                        checkpoint_states[checkpoint_head_base + elem] = h_pair[0];
                                        checkpoint_states[checkpoint_head_base + elem + 1] = h_pair[1];
                                    }
                                }
                            }
                        }
                    }
                }
                if (write_final_states != 0) {
                    final_states[state_head_base + elem] = h_pair[0];
                    final_states[state_head_base + elem + 1] = h_pair[1];
                }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
        }
    }
    // ---- Role: pre_intra ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // pre_intra_main
            int total_tiles_5 = sequence_count * nchunks * nheads;
            int _uniform_4 = make_warp_uniform(total_tiles_5);
            total_tiles_5 = _uniform_4;
            int taddr_0_1 = taddr;
            unsigned int stage1_4 = 0;
            int total_tiles_1_1 = sequence_count * nchunks * nheads;
            int _uniform_5 = make_warp_uniform(total_tiles_1_1);
            total_tiles_1_1 = _uniform_5;
            int b_state_base_1 = (warp % 4 * 4 + lane / 8) * 8;
            int b_col_lane_1 = lane % 8;
            unsigned int _phase_aux1_full_1 = 0;
            unsigned int _phase_b1_full_1 = 0;
            unsigned int _phase_scaled_b_empty_0_1 = 1;
            unsigned int _phase_state_delta_full_0_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_8 = bid; tile_8 < total_tiles_1_1; tile_8 += num_bids) {
                int logical_6 = tile_8 / (unsigned int)nheads;
                int head_6 = tile_8 % (unsigned int)nheads;
                int tile_row_4 = logical_6 * nheads + head_6;
                int physical_chunk_5 = 0;
                int segment_offset_3 = 0;
                int segment_limit_3 = 128;
                int sequence_2 = logical_6 / nchunks;
                physical_chunk_5 = logical_6 - sequence_2 * nchunks;
                int chunk_tokens_3 = seqlen - physical_chunk_5 * 128;
                if (chunk_tokens_3 > 128) {
                    chunk_tokens_3 = 128;
                }
                mbarrier_wait(aux1_full_addr + (stage1_4) * 8, _phase_aux1_full_1);
                mbarrier_wait(b1_full_addr + (stage1_4) * 8, _phase_b1_full_1);
                float last_cumsum_2 = smem_cumsum_all[stage1_4 * 128 + (unsigned int)segment_limit_3 - 1];
                mbarrier_wait(scaled_b_empty_addr, _phase_scaled_b_empty_0_1);
                _phase_scaled_b_empty_0_1 ^= 1;
                #pragma unroll 1
                for (int b_col_iter_1 = 8; b_col_iter_1 < 16; b_col_iter_1++) {
                    int col_1 = b_col_iter_1 * 8 + b_col_lane_1;
                    float scaled_b_values_1[8];
                    #pragma unroll
                    for (int b_local_2 = 0; b_local_2 < 8; b_local_2++) {
                        scaled_b_values_1[b_local_2] = 0.0f;
                    }
                    if (col_1 >= segment_offset_3 && col_1 < segment_limit_3) {
                        unsigned int b_packed_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&b_packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed_1[(0) + 3]))
                            : "r"((smem_b_addr + stage1_4 * 32768 + (unsigned int)(b_state_base_1 / 64 * 16384 + col_1 * 128 + b_state_base_1 % 64 * 2 ^ (b_state_base_1 / 64 * 16384 + col_1 * 128 + b_state_base_1 % 64 * 2 >> 7 & 7) << 4))));
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&scaled_b_values_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_packed_1[_pair]) << 16);
                            (&scaled_b_values_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_packed_1[_pair]) & 0xffff0000u);
                        }
                        float _exp2_0 = approx_exp2((last_cumsum_2 - smem_cumsum_all[stage1_4 * 128 + (unsigned int)col_1]) * 1.4426950408889634f);
                        float b_scale_1 = _exp2_0;
                        b_scale_1 *= smem_delta_all[stage1_4 * 128 + (unsigned int)col_1];
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_0 = {b_scale_1, b_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(scaled_b_values_1)[_ls], _scale2_0);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++) {
                            scaled_b_values_1[_ls] = scaled_b_values_1[_ls] * b_scale_1;
                        }
                        #endif
                    }
                    #pragma unroll
                    for (int b_local_3 = 0; b_local_3 < 8; b_local_3++) {
                        {
                            __nv_bfloat16 _bval_1 = __float2bfloat16_rn(scaled_b_values_1[b_local_3]);
                            uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                            uint32_t _addr_1 = static_cast<uint32_t>((smem_scaled_b_addr + (unsigned int)(col_1 / 64 * 16384 + (b_state_base_1 + b_local_3) * 128 + col_1 % 64 * 2 ^ (col_1 / 64 * 16384 + (b_state_base_1 + b_local_3) * 128 + col_1 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(scaled_b_full_addr);
                        mbarrier_arrive(b1_empty_addr + (stage1_4) * 8);
                        mbarrier_arrive(aux1_empty_addr + (stage1_4) * 8);
                    }
                }
                mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0_1);
                _phase_state_delta_full_0_1 ^= 1;
                int s_tile_base_1 = tile_row_4 * 8192;
                #pragma unroll
                for (int row_half_1 = 0; row_half_1 < 2; row_half_1++) {
                    int state_row_origin_1 = warp % 4 * 32 + row_half_1 * 16;
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr_0_1 + (state_row_origin_1 << 16) + 384 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int state_local_1 = 0; state_local_1 < 16; state_local_1++) {
                        int state_reg_1 = state_local_1 % 4;
                        int state_repeat_1 = state_local_1 / 4;
                        int state_row_1 = state_row_origin_1 + lane / 4 + state_reg_1 / 2 * 8;
                        int dim_1 = 32 + state_repeat_1 * 8 + lane % 4 * 2 + state_reg_1 % 2;
                        s_work[s_tile_base_1 + dim_1 * 128 + state_row_1] = _tmem_load_0[state_local_1];
                    }
                }
                stage1_4 += 1;
                if (stage1_4 == 2) { stage1_4 = 0; _phase_aux1_full_1 ^= 1; _phase_b1_full_1 ^= 1; }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            int heavy_tid_1 = (warp - 4) * 32 + lane;
            int total_pairs_1 = sequence_count * nheads * 4096;
            #pragma unroll 1
            for (int pair_1 = bid * 384 + heavy_tid_1; pair_1 < total_pairs_1; pair_1 += num_bids * 384) {
                int work_1 = pair_1 / 4096;
                int local_pair_1 = pair_1 - work_1 * 4096;
                int sequence_3 = work_1 / nheads;
                int head_7 = work_1 - sequence_3 * nheads;
                int elem_1 = local_pair_1 * 2;
                int state_head_base_1 = work_1 * 8192;
                float h_pair_1[2];
                h_pair_1[0] = 0.0f;
                h_pair_1[1] = 0.0f;
                if (has_initial != 0) {
                    h_pair_1[0] = (float)initial_states[state_head_base_1 + elem_1];
                    h_pair_1[1] = (float)initial_states[state_head_base_1 + elem_1 + 1];
                }
                int first_logical_1 = sequence_3 * nchunks;
                int logical_end_1 = first_logical_1 + nchunks;
                int _uniform_6 = make_warp_uniform(first_logical_1);
                first_logical_1 = _uniform_6;
                int _uniform_7 = make_warp_uniform(logical_end_1);
                logical_end_1 = _uniform_7;
                int checkpoint_token_1 = -1;
                if (checkpoint_state_count > 0) {
                    checkpoint_token_1 = checkpoint_token_indices[sequence_3];
                }
                if (first_logical_1 < logical_end_1) {
                    uint32_t h_pair_bf16_2[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair_1[_lp*2 + 0], h_pair_1[_lp*2+1 + 0]));
                        h_pair_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    h_words[(first_logical_1 * nheads + head_7) * 4096 + local_pair_1] = h_pair_bf16_2[0];
                }
                #pragma unroll 1
                for (int logical_7 = first_logical_1; logical_7 < logical_end_1; logical_7 += 16) {
                    float s_pre_1[32];
                    float d_pre_1[16];
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 16; k_2++) {
                        int _min_0 = ((logical_7 + k_2) < (logical_end_1 - 1) ? (logical_7 + k_2) : (logical_end_1 - 1));
                        int look_1 = _min_0;
                        int look_tile_1 = look_1 * nheads + head_7;
                        int look_physical_1 = 0;
                        int look_offset_1 = 0;
                        int look_limit_1 = 128;
                        look_physical_1 = look_1 - sequence_3 * nchunks;
                        int look_tokens_1 = seqlen - look_physical_1 * 128;
                        if (look_tokens_1 > 128) {
                            look_tokens_1 = 128;
                        }
                        int source_token_3 = look_tokens_1 - 1;
                        float last_cumsum_3 = cumsum_precomputed[look_tile_1 * 128 + source_token_3];
                        float segment_base_1 = 0.0f;
                        float _exp2_1 = approx_exp2((last_cumsum_3 - segment_base_1) * 1.4426950408889634f);
                        d_pre_1[k_2] = _exp2_1;
                        {
                            float2 _v2_2 = *reinterpret_cast<const float2*>(s_work + look_tile_1 * 8192 + elem_1);
                            s_pre_1[2 * k_2] = _v2_2.x;
                            s_pre_1[2 * k_2 + 1] = _v2_2.y;
                        }
                    }
                    #pragma unroll
                    for (int k_3 = 0; k_3 < 16; k_3++) {
                        int current_1 = logical_7 + k_3;
                        if (current_1 < logical_end_1) {
                            float decay_1 = d_pre_1[k_3];
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_3 = {decay_1, decay_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(h_pair_1)[_ls], _scale2_3);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++) {
                                h_pair_1[_ls] = h_pair_1[_ls] * decay_1;
                            }
                            #endif
                            h_pair_1[0] = h_pair_1[0] + s_pre_1[2 * k_3];
                            h_pair_1[1] = h_pair_1[1] + s_pre_1[2 * k_3 + 1];
                            if (logical_end_1 > current_1 + 1) {
                                uint32_t h_pair_bf16_3[1];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair_1[_lp*2 + 0], h_pair_1[_lp*2+1 + 0]));
                                    h_pair_bf16_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                h_words[((current_1 + 1) * nheads + head_7) * 4096 + local_pair_1] = h_pair_bf16_3[0];
                            }
                            if (checkpoint_state_count > 0) {
                                int cur_physical_1 = 0;
                                int cur_limit_1 = 128;
                                cur_physical_1 = current_1 - sequence_3 * nchunks;
                                int cur_tokens_1 = seqlen - cur_physical_1 * 128;
                                if (cur_tokens_1 > 128) {
                                    cur_tokens_1 = 128;
                                }
                                int segment_end_1 = cur_physical_1 * 128 + cur_limit_1;
                                if (segment_end_1 > seqlen) {
                                    segment_end_1 = seqlen;
                                }
                                if (checkpoint_token_1 == segment_end_1) {
                                    int checkpoint_slot_1 = checkpoint_state_slots[sequence_3];
                                    if (checkpoint_slot_1 >= 0 && checkpoint_slot_1 < checkpoint_state_count) {
                                        int checkpoint_head_base_1 = (checkpoint_slot_1 * nheads + head_7) * 8192;
                                        checkpoint_states[checkpoint_head_base_1 + elem_1] = h_pair_1[0];
                                        checkpoint_states[checkpoint_head_base_1 + elem_1 + 1] = h_pair_1[1];
                                    }
                                }
                            }
                        }
                    }
                }
                if (write_final_states != 0) {
                    final_states[state_head_base_1 + elem_1] = h_pair_1[0];
                    final_states[state_head_base_1 + elem_1 + 1] = h_pair_1[1];
                }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int stage3_4 = 0;
            unsigned int acc_stage_1 = 0;
            int row = warp % 4 * 32 + lane;
            int row_tmem_base = warp % 4 * 32 << 16;
            unsigned int _phase_cb_full = 0;
            unsigned int _phase_aux3_full = 0;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_9 = bid; tile_9 < total_tiles_5; tile_9 += num_bids) {
                int logical_8 = tile_9 / (unsigned int)nheads;
                int physical_chunk_6 = 0;
                int segment_offset_4 = 0;
                int segment_limit_4 = 128;
                int sequence_4 = logical_8 / nchunks;
                physical_chunk_6 = logical_8 - sequence_4 * nchunks;
                int chunk_tokens_4 = seqlen - physical_chunk_6 * 128;
                if (chunk_tokens_4 > 128) {
                    chunk_tokens_4 = 128;
                }
                mbarrier_wait(cb_full_addr + (acc_stage_1) * 8, _phase_cb_full);
                mbarrier_wait(aux3_full_addr + (stage3_4) * 8, _phase_aux3_full);
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                float row_cumsum = smem_cumsum_all[stage3_4 * 128 + (unsigned int)row];
                float segment_base_2 = 0.0f;
                if (segment_offset_4 > 0) {
                    segment_base_2 = smem_cumsum_all[stage3_4 * 128 + (unsigned int)(segment_offset_4 - 1)];
                }
                #pragma unroll 1
                for (int col_chunk = 0; col_chunk < 4; col_chunk++) {
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"((unsigned int)(taddr_0_1 + row_tmem_base) + acc_stage_1 * 128 + (unsigned int)(col_chunk * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    int chunk_elem = stage3_4 * 128 + (unsigned int)(col_chunk * 32);
                    float cumsum_values[32];
                    float delta_values[32];
                    #pragma unroll
                    for (int quad = 0; quad < 8; quad++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&cumsum_values[4 * quad])), "=r"(*reinterpret_cast<uint32_t*>(&cumsum_values[(4 * quad) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cumsum_values[(4 * quad) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cumsum_values[(4 * quad) + 3]))
                            : "r"(smem_cumsum_all_addr + (unsigned int)((chunk_elem + 4 * quad) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&delta_values[4 * quad])), "=r"(*reinterpret_cast<uint32_t*>(&delta_values[(4 * quad) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&delta_values[(4 * quad) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&delta_values[(4 * quad) + 3]))
                            : "r"(smem_delta_all_addr + (unsigned int)((chunk_elem + 4 * quad) * 4)));
                    }
                    float q_values[32];
                    #pragma unroll
                    for (int local_col = 0; local_col < 32; local_col++) {
                        int col_2 = col_chunk * 32 + local_col;
                        float q_value = 0.0f;
                        if (row >= segment_offset_4 && row < segment_limit_4 && col_2 >= segment_offset_4 && col_2 <= row) {
                            float _exp2_2 = approx_exp2((row_cumsum - cumsum_values[local_col]) * 1.4426950408889634f);
                            float decay_2 = _exp2_2;
                            q_value = decay_2 * delta_values[local_col];
                            q_value *= _tmem_load_1[local_col];
                        }
                        q_values[local_col] = q_value;
                    }
                    uint32_t q_values_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_values[_lp*2 + 0], q_values[_lp*2+1 + 0]));
                        q_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr_0_1 + row_tmem_base + 256 + col_chunk * 16), "r"(q_values_bf16[0]), "r"(q_values_bf16[1]), "r"(q_values_bf16[2]), "r"(q_values_bf16[3]), "r"(q_values_bf16[4]), "r"(q_values_bf16[5]), "r"(q_values_bf16[6]), "r"(q_values_bf16[7]), "r"(q_values_bf16[8]), "r"(q_values_bf16[9]), "r"(q_values_bf16[10]), "r"(q_values_bf16[11]), "r"(q_values_bf16[12]), "r"(q_values_bf16[13]), "r"(q_values_bf16[14]), "r"(q_values_bf16[15]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(q_full_addr);
                        mbarrier_arrive(cb_empty_addr + (acc_stage_1) * 8);
                    }
                }
                stage3_4 += 1;
                if (stage3_4 == 2) { stage3_4 = 0; _phase_aux3_full ^= 1; }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_cb_full ^= 1; }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // epilogue_main
            int total_tiles_6 = sequence_count * nchunks * nheads;
            int _uniform_11 = make_warp_uniform(total_tiles_6);
            total_tiles_6 = _uniform_11;
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            int heavy_tid_2 = (warp - 4) * 32 + lane;
            int total_pairs_2 = sequence_count * nheads * 4096;
            #pragma unroll 1
            for (int pair_2 = bid * 384 + heavy_tid_2; pair_2 < total_pairs_2; pair_2 += num_bids * 384) {
                int work_2 = pair_2 / 4096;
                int local_pair_2 = pair_2 - work_2 * 4096;
                int sequence_5 = work_2 / nheads;
                int head_8 = work_2 - sequence_5 * nheads;
                int elem_2 = local_pair_2 * 2;
                int state_head_base_2 = work_2 * 8192;
                float h_pair_2[2];
                h_pair_2[0] = 0.0f;
                h_pair_2[1] = 0.0f;
                if (has_initial != 0) {
                    h_pair_2[0] = (float)initial_states[state_head_base_2 + elem_2];
                    h_pair_2[1] = (float)initial_states[state_head_base_2 + elem_2 + 1];
                }
                int first_logical_2 = sequence_5 * nchunks;
                int logical_end_2 = first_logical_2 + nchunks;
                int _uniform_12 = make_warp_uniform(first_logical_2);
                first_logical_2 = _uniform_12;
                int _uniform_13 = make_warp_uniform(logical_end_2);
                logical_end_2 = _uniform_13;
                int checkpoint_token_2 = -1;
                if (checkpoint_state_count > 0) {
                    checkpoint_token_2 = checkpoint_token_indices[sequence_5];
                }
                if (first_logical_2 < logical_end_2) {
                    uint32_t h_pair_bf16_4[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair_2[_lp*2 + 0], h_pair_2[_lp*2+1 + 0]));
                        h_pair_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    h_words[(first_logical_2 * nheads + head_8) * 4096 + local_pair_2] = h_pair_bf16_4[0];
                }
                #pragma unroll 1
                for (int logical_9 = first_logical_2; logical_9 < logical_end_2; logical_9 += 16) {
                    float s_pre_2[32];
                    float d_pre_2[16];
                    #pragma unroll
                    for (int k_4 = 0; k_4 < 16; k_4++) {
                        int _min_2 = ((logical_9 + k_4) < (logical_end_2 - 1) ? (logical_9 + k_4) : (logical_end_2 - 1));
                        int look_2 = _min_2;
                        int look_tile_2 = look_2 * nheads + head_8;
                        int look_physical_2 = 0;
                        int look_offset_2 = 0;
                        int look_limit_2 = 128;
                        look_physical_2 = look_2 - sequence_5 * nchunks;
                        int look_tokens_2 = seqlen - look_physical_2 * 128;
                        if (look_tokens_2 > 128) {
                            look_tokens_2 = 128;
                        }
                        int source_token_4 = look_tokens_2 - 1;
                        float last_cumsum_4 = cumsum_precomputed[look_tile_2 * 128 + source_token_4];
                        float segment_base_3 = 0.0f;
                        float _exp2_5 = approx_exp2((last_cumsum_4 - segment_base_3) * 1.4426950408889634f);
                        d_pre_2[k_4] = _exp2_5;
                        {
                            float2 _v2_0 = *reinterpret_cast<const float2*>(s_work + look_tile_2 * 8192 + elem_2);
                            s_pre_2[2 * k_4] = _v2_0.x;
                            s_pre_2[2 * k_4 + 1] = _v2_0.y;
                        }
                    }
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 16; k_5++) {
                        int current_2 = logical_9 + k_5;
                        if (current_2 < logical_end_2) {
                            float decay_3 = d_pre_2[k_5];
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {decay_3, decay_3};
                            #pragma unroll
                            for (int _ls = 0; _ls < 1; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(h_pair_2)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 2; _ls++) {
                                h_pair_2[_ls] = h_pair_2[_ls] * decay_3;
                            }
                            #endif
                            h_pair_2[0] = h_pair_2[0] + s_pre_2[2 * k_5];
                            h_pair_2[1] = h_pair_2[1] + s_pre_2[2 * k_5 + 1];
                            if (logical_end_2 > current_2 + 1) {
                                uint32_t h_pair_bf16_5[1];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_pair_2[_lp*2 + 0], h_pair_2[_lp*2+1 + 0]));
                                    h_pair_bf16_5[_lp] = *(uint32_t*)&_bf2;
                                }
                                h_words[((current_2 + 1) * nheads + head_8) * 4096 + local_pair_2] = h_pair_bf16_5[0];
                            }
                            if (checkpoint_state_count > 0) {
                                int cur_physical_2 = 0;
                                int cur_limit_2 = 128;
                                cur_physical_2 = current_2 - sequence_5 * nchunks;
                                int cur_tokens_2 = seqlen - cur_physical_2 * 128;
                                if (cur_tokens_2 > 128) {
                                    cur_tokens_2 = 128;
                                }
                                int segment_end_2 = cur_physical_2 * 128 + cur_limit_2;
                                if (segment_end_2 > seqlen) {
                                    segment_end_2 = seqlen;
                                }
                                if (checkpoint_token_2 == segment_end_2) {
                                    int checkpoint_slot_2 = checkpoint_state_slots[sequence_5];
                                    if (checkpoint_slot_2 >= 0 && checkpoint_slot_2 < checkpoint_state_count) {
                                        int checkpoint_head_base_2 = (checkpoint_slot_2 * nheads + head_8) * 8192;
                                        checkpoint_states[checkpoint_head_base_2 + elem_2] = h_pair_2[0];
                                        checkpoint_states[checkpoint_head_base_2 + elem_2 + 1] = h_pair_2[1];
                                    }
                                }
                            }
                        }
                    }
                }
                if (write_final_states != 0) {
                    final_states[state_head_base_2 + elem_2] = h_pair_2[0];
                    final_states[state_head_base_2 + elem_2 + 1] = h_pair_2[1];
                }
            }
            __threadfence();
            asm volatile("fence.proxy.async;");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            asm volatile("barrier.sync 11, 512;" ::: "memory");
            unsigned int stage3_5 = 0;
            unsigned int output_stage = 0;
            int output_issued = 0;
            int row_1 = warp % 4 * 32 + lane;
            int row_tmem_base_1 = warp % 4 * 32 << 16;
            int taddr_0_2 = taddr;
            unsigned int _phase_intra_full_0 = 0;
            unsigned int _phase_inter_full_0 = 0;
            unsigned int _phase_aux3_full_1 = 0;
            unsigned int _phase_x3_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_10 = bid; tile_10 < total_tiles_6; tile_10 += num_bids) {
                int logical_10 = tile_10 / (unsigned int)nheads;
                int head_9 = tile_10 % (unsigned int)nheads;
                float d_head_value = 0.0f;
                if (D_mode == 1) {
                    d_head_value = (float)D[head_9];
                }
                int physical_chunk_7 = 0;
                int physical_batch_4 = 0;
                int segment_offset_5 = 0;
                int segment_limit_5 = 128;
                physical_batch_4 = logical_10 / nchunks;
                physical_chunk_7 = logical_10 - physical_batch_4 * nchunks;
                int chunk_tokens_5 = seqlen - physical_chunk_7 * 128;
                if (chunk_tokens_5 > 128) {
                    chunk_tokens_5 = 128;
                }
                mbarrier_wait(intra_full_addr, _phase_intra_full_0);
                _phase_intra_full_0 ^= 1;
                mbarrier_wait(inter_full_addr, _phase_inter_full_0);
                _phase_inter_full_0 ^= 1;
                mbarrier_wait(aux3_full_addr + (stage3_5) * 8, _phase_aux3_full_1);
                mbarrier_wait(x3_full_addr + (stage3_5) * 8, _phase_x3_full_1);
                float segment_base_4 = 0.0f;
                if (segment_offset_5 > 0) {
                    segment_base_4 = smem_cumsum_all[stage3_5 * 128 + (unsigned int)(segment_offset_5 - 1)];
                }
                int full_chunk = 1;
                if (full_chunk != 0 && output_issued >= 2) {
                    int warp_id_in_role = (warp - 12);
                    if (warp_id_in_role == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                }
                #pragma unroll
                for (int dim_chunk = 0; dim_chunk < 2; dim_chunk++) {
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(taddr_0_2 + row_tmem_base_1 + 320 + dim_chunk * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(taddr_0_2 + row_tmem_base_1 + 448 + dim_chunk * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (row_1 >= segment_offset_5 && row_1 < segment_limit_5) {
                        int token_in_batch_4 = physical_chunk_7 * 128 + row_1;
                        int token = physical_batch_4 * seqlen + token_in_batch_4;
                        float _exp2_6 = approx_exp2((smem_cumsum_all[stage3_5 * 128 + (unsigned int)row_1] - segment_base_4) * 1.4426950408889634f);
                        float decay_4 = _exp2_6;
                        int z_token = token;
                        if (row_1 >= chunk_tokens_5) {
                            z_token = physical_batch_4 * seqlen + physical_chunk_7 * 128 + chunk_tokens_5 - 1;
                        }
                        int z_row_base = z_token * nheads * 64 + head_9 * 64;
                        int out_row_base = (token * nheads + head_9) * 64;
                        #pragma unroll
                        for (int local_group = 0; local_group < 4; local_group++) {
                            int dim_base = dim_chunk * 32 + local_group * 8;
                            unsigned int x_packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + stage3_5 * 16384 + (unsigned int)(dim_base / 64 * 16384 + row_1 * 128 + dim_base % 64 * 2 ^ (dim_base / 64 * 16384 + row_1 * 128 + dim_base % 64 * 2 >> 7 & 7) << 4))));
                            float x_packed_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&x_packed_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&x_packed_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            float d_values[8];
                            #pragma unroll
                            for (int group_dim = 0; group_dim < 8; group_dim++) {
                                d_values[group_dim] = d_head_value;
                            }
                            if (D_mode == 2) {
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(D + head_9 * 64 + dim_base);
                                    uint4 _vld_2[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_2[_blk] = _vptr_2[_blk];
                                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&d_values[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                                            (&d_values[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                            }
                            float z_scale[8];
                            if (has_z != 0) {
                                {
                                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(z + z_row_base + dim_base);
                                    uint4 _vld_3[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_3[_blk] = _vptr_3[_blk];
                                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            (&z_scale[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                                            (&z_scale[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_1 = 0; group_dim_1 < 8; group_dim_1++) {
                                    float z_value = z_scale[group_dim_1];
                                    float _expf_0 = __expf(-z_value);
                                    float _rcp_0 = approx_rcp(1.0f + _expf_0);
                                    z_scale[group_dim_1] = z_value * _rcp_0;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_2 = 0; group_dim_2 < 8; group_dim_2++) {
                                    z_scale[group_dim_2] = 1.0f;
                                }
                            }
                            float group_values[8];
                            #pragma unroll
                            for (int group_dim_3 = 0; group_dim_3 < 8; group_dim_3++) {
                                int local_dim = local_group * 8 + group_dim_3;
                                float _fma_0 = __fmaf_rn(_tmem_load_4[local_dim], decay_4, _tmem_load_3[local_dim]);
                                float value = _fma_0;
                                float x_value = x_packed_f32[group_dim_3];
                                float d_value = d_values[group_dim_3];
                                if (D_mode != 0) {
                                    float _fma_1 = __fmaf_rn(x_value, d_value, value);
                                    value = _fma_1;
                                }
                                float z_mul = z_scale[group_dim_3];
                                value = value * z_mul;
                                group_values[group_dim_3] = value;
                            }
                            if (full_chunk != 0) {
                                uint32_t group_values_bf16[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + dim_base * 2 ^ (row_1 * 128 + dim_base * 2 >> 7 & 7) << 4))), "r"(group_values_bf16[0]), "r"(group_values_bf16[1]), "r"(group_values_bf16[2]), "r"(group_values_bf16[3]) : "memory");
                            } else {
                                #pragma unroll
                                for (int group_dim_4 = 0; group_dim_4 < 8; group_dim_4++) {
                                    out_native[out_row_base + dim_base + group_dim_4] = group_values[group_dim_4];
                                }
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                if (full_chunk != 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    int warp_id_in_role_1 = (warp - 12);
                    if (warp_id_in_role_1 == 0) {
                        if (elect_sync()) {
                            tma_store_4d((&out_map), 0, head_9, physical_chunk_7 * 128, physical_batch_4, smem_y_addr + output_stage * 16384);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    output_stage = output_stage ^ 1;
                    output_issued += 1;
                }
                if (warp == 12) {
                    if (elect_sync()) {
                        mbarrier_arrive(outputs_empty_addr);
                        mbarrier_arrive(aux3_empty_addr + (stage3_5) * 8);
                        mbarrier_arrive(x3_empty_addr + (stage3_5) * 8);
                    }
                }
                stage3_5 += 1;
                if (stage3_5 == 2) { stage3_5 = 0; _phase_aux3_full_1 ^= 1; _phase_x3_full_1 ^= 1; }
            }
            int warp_id_in_role_2 = (warp - 12);
            if (warp_id_in_role_2 == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 12) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

