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
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_CB_TMEM_OFFSET 0
#define TMEM_INTRA_TMEM_OFFSET 256
#define TMEM_STATE_DELTA_TMEM_OFFSET 384
#define TMEM_INTER_TMEM_OFFSET 448
#define NUM_INPUT_PIPE_STAGES 2
#define NUM_INTRA1_ACC_PIPE_STAGES 2
#define NUM_STATE_PIPE_STAGES 2
#define SMEM_SMEM_B_OFF 0
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_T_OFF 0
#define SMEM_SMEM_B_T_STAGE_BYTES 32768
#define SMEM_SMEM_B_T_STRIDE 32768
#define SMEM_SMEM_C_OFF 65536
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_X_TMA_OFF 131072
#define SMEM_SMEM_X_TMA_STAGE_BYTES 16384
#define SMEM_SMEM_X_TMA_STRIDE 16384
#define SMEM_SMEM_X_OFF 131072
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_STATE_OFF 163840
#define SMEM_SMEM_STATE_STAGE_BYTES 16384
#define SMEM_SMEM_STATE_STRIDE 16384
#define SMEM_SMEM_DELTA_ALL_OFF 229376
#define SMEM_SMEM_DELTA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_DELTA_ALL_STRIDE 1024
#define SMEM_SMEM_CUMSUM_ALL_OFF 230400
#define SMEM_SMEM_CUMSUM_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_CUMSUM_ALL_STRIDE 1024
#define SMEM_SMEM_D_ROW_OFF 231424
#define SMEM_SMEM_D_ROW_STAGE_BYTES 128
#define SMEM_SMEM_D_ROW_STRIDE 128
#define SMEM_SMEM_BSCALE_OFF 231552
#define SMEM_SMEM_BSCALE_STAGE_BYTES 512
#define SMEM_SMEM_BSCALE_STRIDE 512
#define SMEM_SMEM_GEOMETRY_OFF 232064
#define SMEM_SMEM_GEOMETRY_STAGE_BYTES 8
#define SMEM_SMEM_GEOMETRY_STRIDE 8
#define SMEM_SMEM_Y_OFF 196608
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


__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}





__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_mamba_ssd_q_tmem_alias_f16_varlen_precomputed_ibulk(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap b_map, const __grid_constant__ CUtensorMap c_map, const __grid_constant__ CUtensorMap out_map, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, __half* __restrict__ delta_precomputed, float* __restrict__ cumsum_precomputed, float* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, __half* __restrict__ initial_states, __half* __restrict__ final_states, __half* __restrict__ checkpoint_states, int* __restrict__ checkpoint_token_indices, int* __restrict__ checkpoint_state_slots, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, int* __restrict__ seq_chunk_cumsum, __nv_bfloat16* __restrict__ out_native, int nheads, int ngroups, int batch, int seqlen, int nchunks, int sequence_count, int num_logical_chunks, int mode_varlen, int D_mode, int has_z, int has_initial, int dt_softplus, float dt_min, float dt_max, int write_final_states, int checkpoint_state_count, int* __restrict__ cu_seqlens, int metadata_from_cu_seqlens, int* __restrict__ preprocess_status)
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

    const int mbar_base = smem + 232072;
    #define bc_full_addr (mbar_base + 0)
    #define bc_empty_addr (mbar_base + 16)
    #define x_full_addr (mbar_base + 32)
    #define x_empty_addr (mbar_base + 48)
    #define aux_full_addr (mbar_base + 64)
    #define aux_empty_addr (mbar_base + 80)
    #define cb_full_addr (mbar_base + 96)
    #define cb_empty_addr (mbar_base + 112)
    #define q_full_addr (mbar_base + 128)
    #define scaled_b_full_addr (mbar_base + 144)
    #define state_operand_full_addr (mbar_base + 152)
    #define state_read_released_addr (mbar_base + 168)
    #define state_delta_full_addr (mbar_base + 184)
    #define intra_full_addr (mbar_base + 192)
    #define intra_empty_addr (mbar_base + 208)
    #define inter_full_addr (mbar_base + 224)
    #define inter_empty_addr (mbar_base + 232)
    #define init_full_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_b_addr = smem + 0;
    __nv_bfloat16* smem_b_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_b_t_addr = smem + 0;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int smem_c_addr = smem + 65536;
    __nv_bfloat16* smem_x_tma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int smem_x_tma_addr = smem + 131072;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int smem_x_addr = smem + 131072;
    __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_state_addr = smem + 163840;
    float* smem_delta_all = reinterpret_cast<float*>(smem_raw + 229376);
    const int smem_delta_all_addr = smem + 229376;
    float* smem_cumsum_all = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_cumsum_all_addr = smem + 230400;
    __nv_bfloat16* smem_d_row = reinterpret_cast<__nv_bfloat16*>(smem_raw + 231424);
    const int smem_d_row_addr = smem + 231424;
    float* smem_bscale = reinterpret_cast<float*>(smem_raw + 231552);
    const int smem_bscale_addr = smem + 231552;
    int* smem_geometry = reinterpret_cast<int*>(smem_raw + 232064);
    const int smem_geometry_addr = smem + 232064;
    __nv_bfloat16* smem_y = reinterpret_cast<__nv_bfloat16*>(smem_raw + 196608);
    const int smem_y_addr = smem + 196608;

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 31 barriers)
    // Mbarriers at smem_raw[232072..232320)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'input_pipe' ---
            // bc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232072, 1);
            mbarrier_init(smem + 232080, 1);
            // bc_empty: 2 barriers, init_count=3
            mbarrier_init(smem + 232088, 3);
            mbarrier_init(smem + 232096, 3);
            // x_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232104, 1);
            mbarrier_init(smem + 232112, 1);
            // x_empty: 2 barriers, init_count=3
            mbarrier_init(smem + 232120, 3);
            mbarrier_init(smem + 232128, 3);
            // aux_full: 2 barriers, init_count=32
            mbarrier_init(smem + 232136, 32);
            mbarrier_init(smem + 232144, 32);
            // aux_empty: 2 barriers, init_count=384
            mbarrier_init(smem + 232152, 384);
            mbarrier_init(smem + 232160, 384);
            // --- pipeline 'intra1_acc_pipe' ---
            // cb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232168, 1);
            mbarrier_init(smem + 232176, 1);
            // cb_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 232184, 2);
            mbarrier_init(smem + 232192, 2);
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232200, 1);
            mbarrier_init(smem + 232208, 1);
            // scaled_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 232216, 1);
            // --- pipeline 'state_pipe' ---
            // state_operand_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232224, 1);
            mbarrier_init(smem + 232232, 1);
            // state_read_released: 2 barriers, init_count=1
            mbarrier_init(smem + 232240, 1);
            mbarrier_init(smem + 232248, 1);
            // state_delta_full: 1 barriers, init_count=1
            mbarrier_init(smem + 232256, 1);
            // --- pipeline 'intra1_acc_pipe' ---
            // intra_full: 2 barriers, init_count=1
            mbarrier_init(smem + 232264, 1);
            mbarrier_init(smem + 232272, 1);
            // intra_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 232280, 4);
            mbarrier_init(smem + 232288, 4);
            // inter_full: 1 barriers, init_count=1
            mbarrier_init(smem + 232296, 1);
            // inter_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 232304, 4);
            // init_full: 1 barriers, init_count=1
            mbarrier_init(smem + 232312, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232320);
    if (warp == 12) {
        int _tmem_hold = smem + 232320;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_cb_tmem = taddr;
    const int tmem_intra_tmem = taddr + 256;
    const int tmem_state_delta_tmem = taddr + 384;
    const int tmem_inter_tmem = taddr + 448;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: mma_inter ----
    if (warp == 0) {
        { // mma_inter_main
            unsigned int input_stage = 0;
            unsigned int state_stage = 0;
            unsigned int _phase_bc_full = 0;
            unsigned int _phase_state_operand_full = 0;
            unsigned int _phase_inter_empty_0 = 1;
            unsigned int _phase_scaled_b_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work = bid; work < sequence_count * nheads; work += num_bids) {
                int sequence = work / (unsigned int)nheads;
                int first_logical = sequence * nchunks;
                int logical_end = first_logical + nchunks;
                int item_lo = 0;
                int item_hi = 0;
                int item_checkpoint = -1;
                int checkpoint_rank = 1073741824;
                first_logical = seq_chunk_cumsum[sequence];
                logical_end = seq_chunk_cumsum[sequence + 1];
                int uniform = make_warp_uniform(first_logical);
                first_logical = uniform;
                int uniform_1 = make_warp_uniform(logical_end);
                logical_end = uniform_1;
                #pragma unroll 1
                for (int logical = first_logical; logical < logical_end; logical++) {
                    mbarrier_wait(bc_full_addr + (input_stage) * 8, _phase_bc_full);
                    mbarrier_wait(state_operand_full_addr + (state_stage) * 8, _phase_state_operand_full);
                    mbarrier_wait(inter_empty_addr, _phase_inter_empty_0);
                    _phase_inter_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = make_warp_uniform((((smem_c_addr) >> 4) & 0x7FFF) + (input_stage) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_state_addr) >> 4) & 0x7FFF) + (state_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_inter_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                    }
                    elect_commit(inter_full_addr);
                    elect_commit(state_read_released_addr + (state_stage) * 8);
                    mbarrier_wait(x_full_addr + (input_stage) * 8, _phase_bc_full);
                    mbarrier_wait(scaled_b_full_addr, _phase_scaled_b_full_0);
                    _phase_scaled_b_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform(((((smem_b_t_addr) >> 4) & 0x7FFF) | 0x4000000) + (input_stage) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x7FFF) | 0x4000000) + (input_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_state_delta_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135365776, 1);
                        }
                    }
                    elect_commit(state_delta_full_addr);
                    elect_commit(x_empty_addr + (input_stage) * 8);
                    elect_commit(bc_empty_addr + (input_stage) * 8);
                    input_stage += 1;
                    if (input_stage == 2) { input_stage = 0; _phase_bc_full ^= 1; }
                    state_stage += 1;
                    if (state_stage == 2) { state_stage = 0; _phase_state_operand_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma_intra ----
    if (warp == 1) {
        { // mma_intra_main
            unsigned int input_stage_1 = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_intra_empty = 1;
            unsigned int _phase_x_full = 0;
            #pragma unroll 1
            for (unsigned int work_1 = bid; work_1 < sequence_count * nheads; work_1 += num_bids) {
                int sequence_1 = work_1 / (unsigned int)nheads;
                int first_logical_1 = sequence_1 * nchunks;
                int logical_end_1 = first_logical_1 + nchunks;
                int item_lo_1 = 0;
                int item_hi_1 = 0;
                int item_checkpoint_1 = -1;
                int checkpoint_rank_1 = 1073741824;
                first_logical_1 = seq_chunk_cumsum[sequence_1];
                logical_end_1 = seq_chunk_cumsum[sequence_1 + 1];
                int uniform_2 = make_warp_uniform(first_logical_1);
                first_logical_1 = uniform_2;
                int uniform_1_1 = make_warp_uniform(logical_end_1);
                logical_end_1 = uniform_1_1;
                #pragma unroll 1
                for (int logical_1 = first_logical_1; logical_1 < logical_end_1; logical_1++) {
                    mbarrier_wait(q_full_addr + (acc_stage) * 8, _phase_q_full);
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(cb_empty_addr + (acc_stage) * 8);
                    }
                    mbarrier_wait(intra_empty_addr + (acc_stage) * 8, _phase_intra_empty);
                    mbarrier_wait(x_full_addr + (input_stage_1) * 8, _phase_x_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x7FFF) | 0x4000000) + (input_stage_1) * 1024);
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
                    :: "r"((tmem_intra_tmem + (acc_stage * 64))), "r"(_mma_b_lo_1), "r"((unsigned int)tmem_cb_tmem + acc_stage * 128), "r"(0) : "memory");
                    elect_commit(intra_full_addr + (acc_stage) * 8);
                    elect_commit(cb_empty_addr + (acc_stage) * 8);
                    elect_commit(x_empty_addr + (input_stage_1) * 8);
                    input_stage_1 += 1;
                    if (input_stage_1 == 2) { input_stage_1 = 0; _phase_x_full ^= 1; }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_q_full ^= 1; _phase_intra_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: load_bc ----
    if (warp == 2) {
        { // load_bc_main
            unsigned int input_stage_2 = 0;
            unsigned int cb_input_stage = 0;
            unsigned int cb_acc_stage = 0;
            unsigned int _phase_bc_empty = 1;
            unsigned int _phase_bc_full_1 = 0;
            unsigned int _phase_cb_empty = 1;
            #pragma unroll 1
            for (unsigned int work_2 = bid; work_2 < sequence_count * nheads; work_2 += num_bids) {
                int sequence_2 = work_2 / (unsigned int)nheads;
                int head = work_2 % (unsigned int)nheads;
                int group = head * ngroups / nheads;
                int first_logical_2 = sequence_2 * nchunks;
                int logical_end_2 = first_logical_2 + nchunks;
                int item_lo_2 = 0;
                int item_hi_2 = 0;
                int item_checkpoint_2 = -1;
                int checkpoint_rank_2 = 1073741824;
                first_logical_2 = seq_chunk_cumsum[sequence_2];
                logical_end_2 = seq_chunk_cumsum[sequence_2 + 1];
                int uniform_3 = make_warp_uniform(first_logical_2);
                first_logical_2 = uniform_3;
                int uniform_1_2 = make_warp_uniform(logical_end_2);
                logical_end_2 = uniform_1_2;
                int physical_batch = 0;
                int refill_first = 0;
                int next_work = work_2 + (unsigned int)num_bids;
                if (next_work < sequence_count * nheads) {
                    refill_first = 1;
                }
                int item_chunks = logical_end_2 - first_logical_2;
                if (item_chunks >= 32) {
                    refill_first = 1;
                }
                int _uniform_2 = make_warp_uniform(refill_first);
                refill_first = _uniform_2;
                int next_physical = 0;
                int next_token = 0;
                if (first_logical_2 < logical_end_2) {
                    int chunk = first_logical_2 - sequence_2 * nchunks;
                    chunk = chunk_indices[first_logical_2];
                    int first_token = chunk * 128;
                    mbarrier_wait(bc_empty_addr + (input_stage_2) * 8, _phase_bc_empty);
                    if (elect_sync()) {
                        tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768, (&b_map), 0, 0, group, first_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768 + 16384, (&b_map), 0, 1, group, first_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768, (&c_map), 0, 0, group, first_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768 + 16384, (&c_map), 0, 1, group, first_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive_expect_tx(bc_full_addr + (input_stage_2) * 8, 65536);
                    }
                    input_stage_2 += 1;
                    if (input_stage_2 == 2) { input_stage_2 = 0; _phase_bc_empty ^= 1; }
                }
                #pragma unroll 1
                for (int logical_2 = first_logical_2; logical_2 < logical_end_2; logical_2++) {
                    int refill_before_cb = refill_first;
                    if (logical_end_2 > logical_2 + 1) {
                        if (refill_before_cb != 0) {
                            int chunk_1 = logical_2 + 1 - sequence_2 * nchunks;
                            chunk_1 = chunk_indices[logical_2 + 1];
                            next_physical = chunk_1;
                            next_token = next_physical * 128;
                            mbarrier_wait(bc_empty_addr + (input_stage_2) * 8, _phase_bc_empty);
                            if (elect_sync()) {
                                tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768, (&b_map), 0, 0, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768 + 16384, (&b_map), 0, 1, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768, (&c_map), 0, 0, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768 + 16384, (&c_map), 0, 1, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive_expect_tx(bc_full_addr + (input_stage_2) * 8, 65536);
                            }
                            input_stage_2 += 1;
                            if (input_stage_2 == 2) { input_stage_2 = 0; _phase_bc_empty ^= 1; }
                        }
                    }
                    mbarrier_wait(bc_full_addr + (cb_input_stage) * 8, _phase_bc_full_1);
                    mbarrier_wait(cb_empty_addr + (cb_acc_stage) * 8, _phase_cb_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_c_addr) >> 4) & 0x7FFF) + (cb_input_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x7FFF) + (cb_input_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_cb_tmem + (cb_acc_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                        }
                    }
                    elect_commit(cb_full_addr + (cb_acc_stage) * 8);
                    elect_commit(bc_empty_addr + (cb_input_stage) * 8);
                    cb_input_stage += 1;
                    if (cb_input_stage == 2) { cb_input_stage = 0; _phase_bc_full_1 ^= 1; }
                    cb_acc_stage += 1;
                    if (cb_acc_stage == 2) { cb_acc_stage = 0; _phase_cb_empty ^= 1; }
                    if (logical_end_2 > logical_2 + 1) {
                        if (refill_before_cb == 0) {
                            int chunk_2 = logical_2 + 1 - sequence_2 * nchunks;
                            chunk_2 = chunk_indices[logical_2 + 1];
                            next_physical = chunk_2;
                            next_token = next_physical * 128;
                            mbarrier_wait(bc_empty_addr + (input_stage_2) * 8, _phase_bc_empty);
                            if (elect_sync()) {
                                tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768, (&b_map), 0, 0, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_b_addr + input_stage_2 * 32768 + 16384, (&b_map), 0, 1, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768, (&c_map), 0, 0, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                tma_5d_gmem2smem(smem_c_addr + input_stage_2 * 32768 + 16384, (&c_map), 0, 1, group, next_token, physical_batch, bc_full_addr + (input_stage_2) * 8);
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive_expect_tx(bc_full_addr + (input_stage_2) * 8, 65536);
                            }
                            input_stage_2 += 1;
                            if (input_stage_2 == 2) { input_stage_2 = 0; _phase_bc_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: load_x_dt ----
    if (warp == 3) {
        { // load_x_dt_main
            unsigned int input_stage_3 = 0;
            unsigned int _phase_aux_empty = 1;
            #pragma unroll 1
            for (unsigned int work_3 = bid; work_3 < sequence_count * nheads; work_3 += num_bids) {
                int sequence_3 = work_3 / (unsigned int)nheads;
                int head_1 = work_3 % (unsigned int)nheads;
                int first_logical_3 = sequence_3 * nchunks;
                int logical_end_3 = first_logical_3 + nchunks;
                int item_lo_3 = 0;
                int item_hi_3 = 0;
                int item_checkpoint_3 = -1;
                int checkpoint_rank_3 = 1073741824;
                first_logical_3 = seq_chunk_cumsum[sequence_3];
                logical_end_3 = seq_chunk_cumsum[sequence_3 + 1];
                int uniform_4 = make_warp_uniform(first_logical_3);
                first_logical_3 = uniform_4;
                int uniform_1_3 = make_warp_uniform(logical_end_3);
                logical_end_3 = uniform_1_3;
                #pragma unroll 1
                for (int logical_3 = first_logical_3; logical_3 < logical_end_3; logical_3++) {
                    int physical_chunk = logical_3 - sequence_3 * nchunks;
                    int segment_offset = 0;
                    int segment_limit = 128;
                    physical_chunk = chunk_indices[logical_3];
                    segment_offset = chunk_offsets[logical_3];
                    if (logical_3 + 1 < num_logical_chunks) {
                        int next_chunk = chunk_indices[logical_3 + 1];
                        if (next_chunk == physical_chunk) {
                            segment_limit = chunk_offsets[logical_3 + 1];
                        }
                    }
                    int chunk_tokens = seqlen - physical_chunk * 128;
                    if (chunk_tokens > 128) {
                        chunk_tokens = 128;
                    }
                    if (segment_limit > chunk_tokens) {
                        segment_limit = chunk_tokens;
                    }
                    int physical_batch_1 = 0;
                    int token_in_batch = physical_chunk * 128;
                    int physical_start = physical_batch_1 * seqlen + token_in_batch;
                    int geometry_word = segment_offset + segment_limit * 256 + chunk_tokens * 65536;
                    mbarrier_wait(aux_empty_addr + (input_stage_3) * 8, _phase_aux_empty);
                    mbarrier_wait(x_empty_addr + (input_stage_3) * 8, _phase_aux_empty);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_x_tma_addr + input_stage_3 * 16384, (&x_map), 0, head_1, token_in_batch, physical_batch_1, x_full_addr + (input_stage_3) * 8);
                        mbarrier_arrive_expect_tx(x_full_addr + (input_stage_3) * 8, 16384);
                        smem_geometry[input_stage_3] = geometry_word;
                    }
                    int tile = logical_3 * nheads + head_1;
                    int group_start = lane * 4;
                    #pragma unroll
                    for (int local = 0; local < 4; local++) {
                        int physical_token = group_start + local;
                        int factor_token = physical_token - segment_offset;
                        float cumsum_value = 0.0f;
                        float delta_value = 0.0f;
                        if (physical_token >= segment_offset && physical_token < segment_limit) {
                            cumsum_value = cumsum_precomputed[tile * 128 + factor_token];
                            delta_value = (float)delta_precomputed[tile * 128 + factor_token];
                        }
                        smem_cumsum_all[input_stage_3 * 128 + (unsigned int)physical_token] = cumsum_value;
                        smem_delta_all[input_stage_3 * 128 + (unsigned int)physical_token] = delta_value;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(aux_full_addr + (input_stage_3) * 8);
                    input_stage_3 += 1;
                    if (input_stage_3 == 2) { input_stage_3 = 0; _phase_aux_empty ^= 1; }
                }
                if (has_initial != 0) {
                    int pf_work = work_3 + (unsigned int)num_bids;
                    if (pf_work < sequence_count * nheads) {
                        int pf_state_base = pf_work * 64 * 128;
                        #pragma unroll
                        for (int pf_repeat = 0; pf_repeat < 4; pf_repeat++) {
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(initial_states + (pf_state_base + (pf_repeat * 32 + lane) * 64))));
                        }
                    }
                }
            }
        }
    }
    // ---- Role: pre_inter ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // pre_inter_main
            unsigned int input_stage_4 = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int state_stage_1 = 0;
            int taddr_0 = taddr;
            float state_values[64];
            int init_staged = 0;
            unsigned int _phase_state_read_released = 1;
            if (has_initial != 0) {
                int ib_first_work = bid;
                if (ib_first_work < sequence_count * nheads) {
                    if (warp == 4) {
                        mbarrier_wait(state_read_released_addr + (state_stage_1) * 8, _phase_state_read_released);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(init_full_addr, 16384);
                            cp_async_bulk_gmem2smem(smem_state_addr + state_stage_1 * 16384, initial_states + (ib_first_work * 64 * 128), 16384, init_full_addr);
                        }
                    }
                    init_staged = 1;
                }
            }
            unsigned int _phase_init_full_0 = 0;
            unsigned int _phase_aux_full = 0;
            unsigned int _phase_cb_full = 0;
            unsigned int _phase_state_delta_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_4 = bid; work_4 < sequence_count * nheads; work_4 += num_bids) {
                int sequence_4 = work_4 / (unsigned int)nheads;
                int head_2 = work_4 % (unsigned int)nheads;
                int state_head_base = (sequence_4 * nheads + head_2) * 64 * 128;
                if (init_staged != 0) {
                    mbarrier_wait(init_full_addr, _phase_init_full_0);
                    _phase_init_full_0 ^= 1;
                    #pragma unroll
                    for (int dim_chunk = 0; dim_chunk < 2; dim_chunk++) {
                        unsigned int initial_packed[16];
                        #pragma unroll
                        for (int initial_group = 0; initial_group < 4; initial_group++) {
                            int initial_matrix = lane / 8;
                            int initial_dim = dim_chunk * 32 + initial_group * 8 + lane % 8;
                            int initial_row = warp % 4 * 32 + initial_matrix / 2 * 16 + initial_matrix % 2 * 8;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(initial_packed[initial_group * 2]), "=r"(initial_packed[initial_group * 2 + 1]), "=r"(initial_packed[initial_group * 2 + 8]), "=r"(initial_packed[initial_group * 2 + 9])
                                : "r"(smem_state_addr + state_stage_1 * 16384 + (unsigned int)(initial_dim * 256 + initial_row * 2))
                                : "memory");
                        }
                        float initial_fragment[32];
                        #pragma unroll
                        for (int _pair = 0; _pair < 16; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&initial_fragment[_pair * 2]))
                                : "r"(initial_packed[_pair]));
                        }
                        #pragma unroll
                        for (int initial_local = 0; initial_local < 32; initial_local++) {
                            state_values[dim_chunk * 32 + initial_local] = initial_fragment[initial_local];
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    init_staged = 0;
                } else {
                    #pragma unroll
                    for (int dim_chunk_1 = 0; dim_chunk_1 < 2; dim_chunk_1++) {
                        #pragma unroll
                        for (int row_half = 0; row_half < 2; row_half++) {
                            int state_fragment_offset = (dim_chunk_1 * 2 + row_half) * 16;
                            int state_row_origin = warp % 4 * 32 + row_half * 16;
                            #pragma unroll
                            for (int state_local = 0; state_local < 16; state_local++) {
                                int state_reg = state_local % 4;
                                int state_repeat = state_local / 4;
                                int state_row = state_row_origin + lane / 4 + state_reg / 2 * 8;
                                int dim = dim_chunk_1 * 32 + state_repeat * 8 + lane % 4 * 2 + state_reg % 2;
                                float initial_value = 0.0f;
                                if (has_initial != 0) {
                                    initial_value = (float)initial_states[state_head_base + dim * 128 + state_row];
                                }
                                state_values[state_fragment_offset + state_local] = initial_value;
                            }
                        }
                    }
                }
                int first_logical_4 = sequence_4 * nchunks;
                int logical_end_4 = first_logical_4 + nchunks;
                int item_lo_4 = 0;
                int item_hi_4 = 0;
                int item_checkpoint_4 = -1;
                int checkpoint_rank_4 = 1073741824;
                first_logical_4 = seq_chunk_cumsum[sequence_4];
                logical_end_4 = seq_chunk_cumsum[sequence_4 + 1];
                int uniform_5 = make_warp_uniform(first_logical_4);
                first_logical_4 = uniform_5;
                int uniform_1_4 = make_warp_uniform(logical_end_4);
                logical_end_4 = uniform_1_4;
                #pragma unroll 1
                for (int logical_4 = first_logical_4; logical_4 < logical_end_4; logical_4++) {
                    int segment_offset_1 = 0;
                    int segment_limit_1 = 128;
                    mbarrier_wait(aux_full_addr + (input_stage_4) * 8, _phase_aux_full);
                    int geometry = smem_geometry[input_stage_4];
                    segment_offset_1 = geometry & 255;
                    segment_limit_1 = geometry >> 8 & 255;
                    float last_cumsum = smem_cumsum_all[input_stage_4 * 128 + (unsigned int)(segment_limit_1 - 1)];
                    int factor_token_1 = warp % 4 * 32 + lane;
                    float factor_value = 0.0f;
                    if (factor_token_1 >= segment_offset_1 && factor_token_1 < segment_limit_1) {
                        float _exp2_0 = approx_exp2((last_cumsum - smem_cumsum_all[input_stage_4 * 128 + (unsigned int)factor_token_1]) * 1.442695f);
                        factor_value = _exp2_0;
                        factor_value *= smem_delta_all[input_stage_4 * 128 + (unsigned int)factor_token_1];
                    }
                    smem_bscale[factor_token_1] = factor_value;
                    mbarrier_wait(state_read_released_addr + (state_stage_1) * 8, _phase_state_read_released);
                    #pragma unroll
                    for (int dim_chunk_2 = 0; dim_chunk_2 < 2; dim_chunk_2++) {
                        #pragma unroll
                        for (int row_half_1 = 0; row_half_1 < 2; row_half_1++) {
                            int state_fragment_offset_1 = (dim_chunk_2 * 2 + row_half_1) * 16;
                            float state_fragment[16];
                            #pragma unroll
                            for (int state_local_1 = 0; state_local_1 < 16; state_local_1++) {
                                state_fragment[state_local_1] = state_values[state_fragment_offset_1 + state_local_1];
                            }
                            unsigned int state_fragment_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_fragment[_lp*2 + 0], state_fragment[_lp*2+1 + 0]));
                                state_fragment_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int state_row_origin_1 = warp % 4 * 32 + row_half_1 * 16;
                            #pragma unroll
                            for (int stsm_repeat = 0; stsm_repeat < 2; stsm_repeat++) {
                                int stsm_matrix = lane / 8;
                                int stsm_row = dim_chunk_2 * 32 + stsm_repeat * 16 + stsm_matrix / 2 * 8 + lane % 8;
                                int stsm_col = state_row_origin_1 + stsm_matrix % 2 * 8;
                                int packed_base = stsm_repeat * 4;
                                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((smem_state_addr + state_stage_1 * 16384 + (unsigned int)(stsm_col / 64 * 8192 + stsm_row * 128 + stsm_col % 64 * 2 ^ (stsm_col / 64 * 8192 + stsm_row * 128 + stsm_col % 64 * 2 >> 7 & 7) << 4)));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&state_fragment_bf16[packed_base + 3]))
                                    : "memory");
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(state_operand_full_addr + (state_stage_1) * 8);
                        }
                    }
                    if (logical_4 == logical_end_4 - 1) {
                        if (has_initial != 0) {
                            int ib_next_work = work_4 + (unsigned int)num_bids;
                            if (ib_next_work < sequence_count * nheads) {
                                if (warp == 4) {
                                    mbarrier_wait(state_read_released_addr + (((state_stage_1 + 1 < 2) ? state_stage_1 + 1 : (unsigned int)0)) * 8, ((state_stage_1 + 1 < 2) ? _phase_state_read_released : _phase_state_read_released ^ 1));
                                    if (elect_sync()) {
                                        mbarrier_arrive_expect_tx(init_full_addr, 16384);
                                        cp_async_bulk_gmem2smem(smem_state_addr + ((state_stage_1 + 1 < 2) ? state_stage_1 + 1 : (unsigned int)0) * 16384, initial_states + (ib_next_work * 64 * 128), 16384, init_full_addr);
                                    }
                                }
                                init_staged = 1;
                            }
                        }
                    }
                    mbarrier_wait(bc_full_addr + (input_stage_4) * 8, _phase_aux_full);
                    mbarrier_wait(cb_full_addr + (acc_stage_1) * 8, _phase_cb_full);
                    int b_state_base = (warp % 4 * 4 + lane / 8) * 8;
                    int b_col_lane = lane % 8;
                    int b_plane_row = b_state_base / 64 * 128;
                    int b_plane_col_bytes = b_state_base % 64 * 2;
                    #pragma unroll 1
                    for (int b_col_iter = 0; b_col_iter < 16; b_col_iter++) {
                        int col = b_col_iter * 8 + b_col_lane;
                        float scaled_b_values[8];
                        #pragma unroll
                        for (int b_local = 0; b_local < 8; b_local++) {
                            scaled_b_values[b_local] = 0.0f;
                        }
                        unsigned int b_packed[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&b_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&b_packed[(0) + 3]))
                            : "r"((smem_b_addr + input_stage_4 * 32768 + (unsigned int)(b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 ^ (b_state_base / 64 * 16384 + col * 128 + b_state_base % 64 * 2 >> 7 & 7) << 4))));
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&scaled_b_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(b_packed[_pair]) << 16);
                            (&scaled_b_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(b_packed[_pair]) & 0xffff0000u);
                        }
                        float b_scale = smem_bscale[col];
                        const float2 _scale2_2 = {b_scale, b_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(scaled_b_values)[_ls], _scale2_2);
                        if (col < segment_offset_1 || col >= segment_limit_1) {
                            #pragma unroll
                            for (int b_local_1 = 0; b_local_1 < 8; b_local_1++) {
                                scaled_b_values[b_local_1] = 0.0f;
                            }
                        }
                        unsigned int scaled_b_values_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_b_values[_lp*2 + 0], scaled_b_values[_lp*2+1 + 0]));
                            scaled_b_values_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_b_addr + input_stage_4 * 32768 + (unsigned int)((b_plane_row + col) * 128 + b_plane_col_bytes ^ ((b_plane_row + col) * 128 + b_plane_col_bytes >> 7 & 7) << 4))), "r"(scaled_b_values_bf16[0]), "r"(scaled_b_values_bf16[1]), "r"(scaled_b_values_bf16[2]), "r"(scaled_b_values_bf16[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(scaled_b_full_addr);
                            mbarrier_arrive(bc_empty_addr + (input_stage_4) * 8);
                        }
                    }
                    mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0);
                    _phase_state_delta_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float segment_base = 0.0f;
                    if (segment_offset_1 > 0) {
                        segment_base = smem_cumsum_all[input_stage_4 * 128 + (unsigned int)(segment_offset_1 - 1)];
                    }
                    float last_decay = approx_exp2((last_cumsum - segment_base) * 1.442695f);
                    #pragma unroll
                    for (int dim_chunk_3 = 0; dim_chunk_3 < 2; dim_chunk_3++) {
                        #pragma unroll
                        for (int row_half_2 = 0; row_half_2 < 2; row_half_2++) {
                            int state_row_origin_2 = warp % 4 * 32 + row_half_2 * 16;
                            float _tmem_load_1[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                                : "r"(taddr_0 + (state_row_origin_2 << 16) + 384 + dim_chunk_3 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int state_fragment_offset_2 = (dim_chunk_3 * 2 + row_half_2) * 16;
                            #pragma unroll
                            for (int state_local_2 = 0; state_local_2 < 16; state_local_2++) {
                                int state_index = state_fragment_offset_2 + state_local_2;
                                float fused_state = __fmaf_rn(state_values[state_index], last_decay, _tmem_load_1[state_local_2]);
                                state_values[state_index] = fused_state;
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    mbarrier_arrive(aux_empty_addr + (input_stage_4) * 8);
                    input_stage_4 += 1;
                    if (input_stage_4 == 2) { input_stage_4 = 0; _phase_aux_full ^= 1; }
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_cb_full ^= 1; }
                    state_stage_1 += 1;
                    if (state_stage_1 == 2) { state_stage_1 = 0; _phase_state_read_released ^= 1; }
                    if (checkpoint_state_count > 0) {
                        int checkpoint_token = checkpoint_token_indices[sequence_4];
                        int chunk_3 = logical_4 - sequence_4 * nchunks;
                        chunk_3 = chunk_indices[logical_4];
                        int segment_end = chunk_3 * 128 + segment_limit_1;
                        if (segment_end > seqlen) {
                            segment_end = seqlen;
                        }
                        if (checkpoint_token == segment_end) {
                            int checkpoint_slot = checkpoint_state_slots[sequence_4];
                            if (checkpoint_slot >= 0 && checkpoint_slot < checkpoint_state_count) {
                                int checkpoint_head_base = (checkpoint_slot * nheads + head_2) * 64 * 128;
                                #pragma unroll
                                for (int dim_chunk_4 = 0; dim_chunk_4 < 2; dim_chunk_4++) {
                                    #pragma unroll
                                    for (int row_half_3 = 0; row_half_3 < 2; row_half_3++) {
                                        int state_fragment_offset_3 = (dim_chunk_4 * 2 + row_half_3) * 16;
                                        int state_row_origin_3 = warp % 4 * 32 + row_half_3 * 16;
                                        #pragma unroll
                                        for (int state_local_3 = 0; state_local_3 < 16; state_local_3++) {
                                            int state_reg_1 = state_local_3 % 4;
                                            int state_repeat_1 = state_local_3 / 4;
                                            int state_row_1 = state_row_origin_3 + lane / 4 + state_reg_1 / 2 * 8;
                                            int dim_1 = dim_chunk_4 * 32 + state_repeat_1 * 8 + lane % 4 * 2 + state_reg_1 % 2;
                                            checkpoint_states[checkpoint_head_base + dim_1 * 128 + state_row_1] = state_values[state_fragment_offset_3 + state_local_3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (write_final_states != 0) {
                    #pragma unroll
                    for (int dim_chunk_5 = 0; dim_chunk_5 < 2; dim_chunk_5++) {
                        #pragma unroll
                        for (int row_half_4 = 0; row_half_4 < 2; row_half_4++) {
                            int state_fragment_offset_4 = (dim_chunk_5 * 2 + row_half_4) * 16;
                            int state_row_origin_4 = warp % 4 * 32 + row_half_4 * 16;
                            #pragma unroll
                            for (int state_local_4 = 0; state_local_4 < 16; state_local_4++) {
                                int state_reg_2 = state_local_4 % 4;
                                int state_repeat_2 = state_local_4 / 4;
                                int state_row_2 = state_row_origin_4 + lane / 4 + state_reg_2 / 2 * 8;
                                int dim_2 = dim_chunk_5 * 32 + state_repeat_2 * 8 + lane % 4 * 2 + state_reg_2 % 2;
                                final_states[state_head_base + dim_2 * 128 + state_row_2] = state_values[state_fragment_offset_4 + state_local_4];
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: pre_intra ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // pre_intra_main
            unsigned int input_stage_5 = 0;
            unsigned int acc_stage_2 = 0;
            int row = warp % 4 * 32 + lane;
            int row_tmem_base = warp % 4 * 32 << 16;
            int taddr_0_1 = taddr;
            unsigned int _phase_cb_full_1 = 0;
            unsigned int _phase_aux_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_5 = bid; work_5 < sequence_count * nheads; work_5 += num_bids) {
                int sequence_5 = work_5 / (unsigned int)nheads;
                int first_logical_5 = sequence_5 * nchunks;
                int logical_end_5 = first_logical_5 + nchunks;
                int item_lo_5 = 0;
                int item_hi_5 = 0;
                int item_checkpoint_5 = -1;
                int checkpoint_rank_5 = 1073741824;
                first_logical_5 = seq_chunk_cumsum[sequence_5];
                logical_end_5 = seq_chunk_cumsum[sequence_5 + 1];
                int uniform_6 = make_warp_uniform(first_logical_5);
                first_logical_5 = uniform_6;
                int uniform_1_5 = make_warp_uniform(logical_end_5);
                logical_end_5 = uniform_1_5;
                #pragma unroll 1
                for (int logical_5 = first_logical_5; logical_5 < logical_end_5; logical_5++) {
                    int segment_offset_2 = 0;
                    int segment_limit_2 = 128;
                    mbarrier_wait(cb_full_addr + (acc_stage_2) * 8, _phase_cb_full_1);
                    mbarrier_wait(aux_full_addr + (input_stage_5) * 8, _phase_aux_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int geometry_1 = smem_geometry[input_stage_5];
                    segment_offset_2 = geometry_1 & 255;
                    segment_limit_2 = geometry_1 >> 8 & 255;
                    float row_cumsum = smem_cumsum_all[input_stage_5 * 128 + (unsigned int)row];
                    float segment_base_1 = 0.0f;
                    if (segment_offset_2 > 0) {
                        segment_base_1 = smem_cumsum_all[input_stage_5 * 128 + (unsigned int)(segment_offset_2 - 1)];
                    }
                    #pragma unroll 1
                    for (int col_chunk = 0; col_chunk < 4; col_chunk++) {
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"((unsigned int)(taddr_0_1 + row_tmem_base) + acc_stage_2 * 128 + (unsigned int)(col_chunk * 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        int chunk_elem = input_stage_5 * 128 + (unsigned int)(col_chunk * 32);
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
                            int col_1 = col_chunk * 32 + local_col;
                            float q_value = 0.0f;
                            if (row >= segment_offset_2 && row < segment_limit_2 && col_1 >= segment_offset_2 && col_1 <= row) {
                                float decay = approx_exp2((row_cumsum - cumsum_values[local_col]) * 1.442695f);
                                q_value = decay * delta_values[local_col];
                                q_value *= _tmem_load_0[local_col];
                            }
                            q_values[local_col] = q_value;
                        }
                        unsigned int q_values_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_values[_lp*2 + 0], q_values[_lp*2+1 + 0]));
                            q_values_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"((unsigned int)(taddr_0_1 + row_tmem_base) + acc_stage_2 * 128 + (unsigned int)(col_chunk * 16)), "r"(q_values_bf16[0]), "r"(q_values_bf16[1]), "r"(q_values_bf16[2]), "r"(q_values_bf16[3]), "r"(q_values_bf16[4]), "r"(q_values_bf16[5]), "r"(q_values_bf16[6]), "r"(q_values_bf16[7]), "r"(q_values_bf16[8]), "r"(q_values_bf16[9]), "r"(q_values_bf16[10]), "r"(q_values_bf16[11]), "r"(q_values_bf16[12]), "r"(q_values_bf16[13]), "r"(q_values_bf16[14]), "r"(q_values_bf16[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(aux_empty_addr + (input_stage_5) * 8);
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            mbarrier_arrive(q_full_addr + (acc_stage_2) * 8);
                        }
                    }
                    input_stage_5 += 1;
                    if (input_stage_5 == 2) { input_stage_5 = 0; _phase_aux_full_1 ^= 1; }
                    acc_stage_2 += 1;
                    if (acc_stage_2 == 2) { acc_stage_2 = 0; _phase_cb_full_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // epilogue_main
            unsigned int input_stage_6 = 0;
            unsigned int acc_stage_3 = 0;
            unsigned int output_stage = 0;
            int output_issued = 0;
            int row_1 = warp % 4 * 32 + lane;
            int row_tmem_base_1 = warp % 4 * 32 << 16;
            int taddr_0_2 = taddr;
            unsigned int _phase_intra_full = 0;
            unsigned int _phase_inter_full_0 = 0;
            unsigned int _phase_aux_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_6 = bid; work_6 < sequence_count * nheads; work_6 += num_bids) {
                int sequence_6 = work_6 / (unsigned int)nheads;
                int head_3 = work_6 % (unsigned int)nheads;
                float d_head_value = 0.0f;
                if (D_mode == 1) {
                    d_head_value = (float)D[head_3];
                }
                float d_row[8];
                #pragma unroll
                for (int d_slot = 0; d_slot < 8; d_slot++) {
                    d_row[d_slot] = d_head_value;
                }
                unsigned int d_row_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_row[_lp*2 + 0], d_row[_lp*2+1 + 0]));
                    d_row_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                int warp_id_in_role = warp - 12;
                if (warp_id_in_role == 0) {
                    if (lane < 8) {
                        if (D_mode == 2) {
                            unsigned int d_load[4];
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(D + head_3 * 64 + lane * 8);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&d_load[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_0[_blk] = _vptr_0[_blk];
                                }
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_d_row_addr + (unsigned int)(lane * 16)), "r"(d_load[0]), "r"(d_load[1]), "r"(d_load[2]), "r"(d_load[3]));
                        } else {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_d_row_addr + (unsigned int)(lane * 16)), "r"(d_row_bf16[0]), "r"(d_row_bf16[1]), "r"(d_row_bf16[2]), "r"(d_row_bf16[3]));
                        }
                    }
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                int first_logical_6 = sequence_6 * nchunks;
                int logical_end_6 = first_logical_6 + nchunks;
                int item_lo_6 = 0;
                int item_hi_6 = 0;
                int item_checkpoint_6 = -1;
                int checkpoint_rank_6 = 1073741824;
                first_logical_6 = seq_chunk_cumsum[sequence_6];
                logical_end_6 = seq_chunk_cumsum[sequence_6 + 1];
                int uniform_7 = make_warp_uniform(first_logical_6);
                first_logical_6 = uniform_7;
                int uniform_1_6 = make_warp_uniform(logical_end_6);
                logical_end_6 = uniform_1_6;
                #pragma unroll 1
                for (int logical_6 = first_logical_6; logical_6 < logical_end_6; logical_6++) {
                    int chunk_4 = logical_6 - sequence_6 * nchunks;
                    chunk_4 = chunk_indices[logical_6];
                    int segment_offset_3 = 0;
                    int segment_limit_3 = 128;
                    int chunk_tokens_1 = 128;
                    mbarrier_wait(intra_full_addr + (acc_stage_3) * 8, _phase_intra_full);
                    mbarrier_wait(inter_full_addr, _phase_inter_full_0);
                    _phase_inter_full_0 ^= 1;
                    mbarrier_wait(aux_full_addr + (input_stage_6) * 8, _phase_aux_full_2);
                    mbarrier_wait(x_full_addr + (input_stage_6) * 8, _phase_aux_full_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int geometry_2 = smem_geometry[input_stage_6];
                    segment_offset_3 = geometry_2 & 255;
                    segment_limit_3 = geometry_2 >> 8 & 255;
                    chunk_tokens_1 = geometry_2 >> 16;
                    int physical_batch_2 = 0;
                    float segment_base_2 = 0.0f;
                    if (segment_offset_3 > 0) {
                        segment_base_2 = smem_cumsum_all[input_stage_6 * 128 + (unsigned int)(segment_offset_3 - 1)];
                    }
                    int full_chunk = 1;
                    full_chunk = 0;
                    if (segment_offset_3 == 0 && segment_limit_3 == chunk_tokens_1) {
                        full_chunk = 1;
                    }
                    if (full_chunk != 0 && output_issued >= 2) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    float ring_intra_0[8];
                    float ring_inter_0[8];
                    unsigned int ring_z_0[4];
                    float ring_intra_1[8];
                    float ring_inter_1[8];
                    unsigned int ring_z_1[4];
                    float ring_intra_2[8];
                    float ring_inter_2[8];
                    unsigned int ring_z_2[4];
                    float ring_intra_3[8];
                    float ring_inter_3[8];
                    unsigned int ring_z_3[4];
                    unsigned int x_packed[4];
                    unsigned int d_packed[4];
                    float d_values[8];
                    float z_scale[8];
                    float group_values[8];
                    int token_in_batch_1 = chunk_4 * 128 + row_1;
                    int token = physical_batch_2 * seqlen + token_in_batch_1;
                    int z_token = token;
                    if (row_1 >= chunk_tokens_1) {
                        z_token = physical_batch_2 * seqlen + chunk_4 * 128 + chunk_tokens_1 - 1;
                    }
                    int z_row_base = z_token * nheads * 64 + head_3 * 64;
                    int out_row_base = (token * nheads + head_3) * 64;
                    tmem_ld_x8(&ring_intra_0[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64);
                    tmem_ld_x8(&ring_inter_0[0], taddr_0_2 + row_tmem_base_1 + 448);
                    if (has_z != 0) {
                        if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(z + z_row_base);
                                uint4* _vdst_1 = reinterpret_cast<uint4*>(&ring_z_0[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_1[_blk] = _vptr_1[_blk];
                                }
                            }
                        }
                    }
                    tmem_ld_x8(&ring_intra_1[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 8);
                    tmem_ld_x8(&ring_inter_1[0], taddr_0_2 + row_tmem_base_1 + 448 + 8);
                    if (has_z != 0) {
                        if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(z + z_row_base + 8);
                                uint4* _vdst_2 = reinterpret_cast<uint4*>(&ring_z_1[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_2[_blk] = _vptr_2[_blk];
                                }
                            }
                        }
                    }
                    tmem_ld_x8(&ring_intra_2[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 16);
                    tmem_ld_x8(&ring_inter_2[0], taddr_0_2 + row_tmem_base_1 + 448 + 16);
                    if (has_z != 0) {
                        if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(z + z_row_base + 16);
                                uint4* _vdst_3 = reinterpret_cast<uint4*>(&ring_z_2[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_3[_blk] = _vptr_3[_blk];
                                }
                            }
                        }
                    }
                    tmem_ld_x8(&ring_intra_3[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 24);
                    tmem_ld_x8(&ring_inter_3[0], taddr_0_2 + row_tmem_base_1 + 448 + 24);
                    if (has_z != 0) {
                        if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(z + z_row_base + 24);
                                uint4* _vdst_4 = reinterpret_cast<uint4*>(&ring_z_3[0]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_4[_blk] = _vptr_4[_blk];
                                }
                            }
                        }
                    }
                    float decay_1 = approx_exp2((smem_cumsum_all[input_stage_6 * 128 + (unsigned int)row_1] - segment_base_2) * 1.442695f);
                    if (has_z == 0) {
                        #pragma unroll 1
                        for (int dim_chunk_6 = 0; dim_chunk_6 < 2; dim_chunk_6++) {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (dim_chunk_6 == 1) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                if (elect_sync()) {
                                    mbarrier_arrive(intra_empty_addr + (acc_stage_3) * 8);
                                    mbarrier_arrive(inter_empty_addr);
                                }
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)(dim_chunk_6 * 4 * 8 / 64 * 16384 + row_1 * 128 + dim_chunk_6 * 4 * 8 % 64 * 2 ^ (dim_chunk_6 * 4 * 8 / 64 * 16384 + row_1 * 128 + dim_chunk_6 * 4 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)(dim_chunk_6 * 4 * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_0[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_0[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_6 * 4 < 4) {
                                    {
                                        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_6 * 4 + 4) * 8);
                                        uint4* _vdst_5 = reinterpret_cast<uint4*>(&ring_z_0[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_5[_blk] = _vptr_5[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim = 0; group_dim < 8; group_dim++) {
                                    float z_value = z_scale[group_dim];
                                    float _exp_7 = __expf(-z_value);
                                    float _rcp_6 = approx_rcp(1.0f + _exp_7);
                                    z_scale[group_dim] = z_value * _rcp_6;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_1 = 0; group_dim_1 < 8; group_dim_1++) {
                                    z_scale[group_dim_1] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_2 = 0; group_dim_2 < 8; group_dim_2++) {
                                float value = __fmaf_rn(ring_inter_0[group_dim_2], decay_1, ring_intra_0[group_dim_2]);
                                float x_value = unpack_bf16[group_dim_2];
                                float d_value = d_values[group_dim_2];
                                if (D_mode != 0) {
                                    value = __fmaf_rn(x_value, d_value, value);
                                }
                                float z_mul = z_scale[group_dim_2];
                                value = value * z_mul;
                                group_values[group_dim_2] = value;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + dim_chunk_6 * 4 * 8 * 2 ^ (row_1 * 128 + dim_chunk_6 * 4 * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16[0]), "r"(group_values_bf16[1]), "r"(group_values_bf16[2]), "r"(group_values_bf16[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + dim_chunk_6 * 4 * 8))[0] = reinterpret_cast<int4*>(group_values_pv)[0];
                            }
                            if (dim_chunk_6 == 0) {
                                tmem_ld_x8(&ring_intra_0[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 32);
                                tmem_ld_x8(&ring_inter_0[0], taddr_0_2 + row_tmem_base_1 + 448 + 32);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_6 * 4 + 1) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 1) * 8 % 64 * 2 ^ ((dim_chunk_6 * 4 + 1) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 1) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_0[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_6 * 4 + 1) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_1[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_1[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_6 * 4 + 1 < 4) {
                                    {
                                        const uint4* _vptr_8 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_6 * 4 + 1 + 4) * 8);
                                        uint4* _vdst_8 = reinterpret_cast<uint4*>(&ring_z_1[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_8[_blk] = _vptr_8[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_3 = 0; group_dim_3 < 8; group_dim_3++) {
                                    float z_value_1 = z_scale[group_dim_3];
                                    float _exp_10 = __expf(-z_value_1);
                                    float _rcp_9 = approx_rcp(1.0f + _exp_10);
                                    z_scale[group_dim_3] = z_value_1 * _rcp_9;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_4 = 0; group_dim_4 < 8; group_dim_4++) {
                                    z_scale[group_dim_4] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_5 = 0; group_dim_5 < 8; group_dim_5++) {
                                float value_1 = __fmaf_rn(ring_inter_1[group_dim_5], decay_1, ring_intra_1[group_dim_5]);
                                float x_value_1 = unpack_bf16_0[group_dim_5];
                                float d_value_1 = d_values[group_dim_5];
                                if (D_mode != 0) {
                                    value_1 = __fmaf_rn(x_value_1, d_value_1, value_1);
                                }
                                float z_mul_1 = z_scale[group_dim_5];
                                value_1 = value_1 * z_mul_1;
                                group_values[group_dim_5] = value_1;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_6 * 4 + 1) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_6 * 4 + 1) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_1[0]), "r"(group_values_bf16_1[1]), "r"(group_values_bf16_1[2]), "r"(group_values_bf16_1[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_1[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_6 * 4 + 1) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_1)[0];
                            }
                            if (dim_chunk_6 == 0) {
                                tmem_ld_x8(&ring_intra_1[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 40);
                                tmem_ld_x8(&ring_inter_1[0], taddr_0_2 + row_tmem_base_1 + 448 + 40);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_6 * 4 + 2) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 2) * 8 % 64 * 2 ^ ((dim_chunk_6 * 4 + 2) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 2) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_6 * 4 + 2) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_2[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_2[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_6 * 4 + 2 < 4) {
                                    {
                                        const uint4* _vptr_11 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_6 * 4 + 2 + 4) * 8);
                                        uint4* _vdst_11 = reinterpret_cast<uint4*>(&ring_z_2[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_11[_blk] = _vptr_11[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_6 = 0; group_dim_6 < 8; group_dim_6++) {
                                    float z_value_2 = z_scale[group_dim_6];
                                    float _exp_13 = __expf(-z_value_2);
                                    float _rcp_12 = approx_rcp(1.0f + _exp_13);
                                    z_scale[group_dim_6] = z_value_2 * _rcp_12;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_7 = 0; group_dim_7 < 8; group_dim_7++) {
                                    z_scale[group_dim_7] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_8 = 0; group_dim_8 < 8; group_dim_8++) {
                                float value_2 = __fmaf_rn(ring_inter_2[group_dim_8], decay_1, ring_intra_2[group_dim_8]);
                                float x_value_2 = unpack_bf16_1[group_dim_8];
                                float d_value_2 = d_values[group_dim_8];
                                if (D_mode != 0) {
                                    value_2 = __fmaf_rn(x_value_2, d_value_2, value_2);
                                }
                                float z_mul_2 = z_scale[group_dim_8];
                                value_2 = value_2 * z_mul_2;
                                group_values[group_dim_8] = value_2;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_2[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_6 * 4 + 2) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_6 * 4 + 2) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_2[0]), "r"(group_values_bf16_2[1]), "r"(group_values_bf16_2[2]), "r"(group_values_bf16_2[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_2[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_6 * 4 + 2) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_2)[0];
                            }
                            if (dim_chunk_6 == 0) {
                                tmem_ld_x8(&ring_intra_2[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 48);
                                tmem_ld_x8(&ring_inter_2[0], taddr_0_2 + row_tmem_base_1 + 448 + 48);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_6 * 4 + 3) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 3) * 8 % 64 * 2 ^ ((dim_chunk_6 * 4 + 3) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_6 * 4 + 3) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_6 * 4 + 3) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_3[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_3[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_6 * 4 + 3 < 4) {
                                    {
                                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_6 * 4 + 3 + 4) * 8);
                                        uint4* _vdst_14 = reinterpret_cast<uint4*>(&ring_z_3[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_14[_blk] = _vptr_14[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_9 = 0; group_dim_9 < 8; group_dim_9++) {
                                    float z_value_3 = z_scale[group_dim_9];
                                    float _exp_16 = __expf(-z_value_3);
                                    float _rcp_15 = approx_rcp(1.0f + _exp_16);
                                    z_scale[group_dim_9] = z_value_3 * _rcp_15;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_10 = 0; group_dim_10 < 8; group_dim_10++) {
                                    z_scale[group_dim_10] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_11 = 0; group_dim_11 < 8; group_dim_11++) {
                                float value_3 = __fmaf_rn(ring_inter_3[group_dim_11], decay_1, ring_intra_3[group_dim_11]);
                                float x_value_3 = unpack_bf16_2[group_dim_11];
                                float d_value_3 = d_values[group_dim_11];
                                if (D_mode != 0) {
                                    value_3 = __fmaf_rn(x_value_3, d_value_3, value_3);
                                }
                                float z_mul_3 = z_scale[group_dim_11];
                                value_3 = value_3 * z_mul_3;
                                group_values[group_dim_11] = value_3;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_3[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_6 * 4 + 3) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_6 * 4 + 3) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_3[0]), "r"(group_values_bf16_3[1]), "r"(group_values_bf16_3[2]), "r"(group_values_bf16_3[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_3[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_6 * 4 + 3) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_3)[0];
                            }
                            if (dim_chunk_6 == 0) {
                                tmem_ld_x8(&ring_intra_3[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 56);
                                tmem_ld_x8(&ring_inter_3[0], taddr_0_2 + row_tmem_base_1 + 448 + 56);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int dim_chunk_7 = 0; dim_chunk_7 < 2; dim_chunk_7++) {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            if (dim_chunk_7 == 1) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                if (elect_sync()) {
                                    mbarrier_arrive(intra_empty_addr + (acc_stage_3) * 8);
                                    mbarrier_arrive(inter_empty_addr);
                                }
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)(dim_chunk_7 * 4 * 8 / 64 * 16384 + row_1 * 128 + dim_chunk_7 * 4 * 8 % 64 * 2 ^ (dim_chunk_7 * 4 * 8 / 64 * 16384 + row_1 * 128 + dim_chunk_7 * 4 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)(dim_chunk_7 * 4 * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_0[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_0[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_7 * 4 < 4) {
                                    {
                                        const uint4* _vptr_17 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_7 * 4 + 4) * 8);
                                        uint4* _vdst_17 = reinterpret_cast<uint4*>(&ring_z_0[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_17[_blk] = _vptr_17[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_12 = 0; group_dim_12 < 8; group_dim_12++) {
                                    float z_value_4 = z_scale[group_dim_12];
                                    float _exp_19 = __expf(-z_value_4);
                                    float _rcp_18 = approx_rcp(1.0f + _exp_19);
                                    z_scale[group_dim_12] = z_value_4 * _rcp_18;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_13 = 0; group_dim_13 < 8; group_dim_13++) {
                                    z_scale[group_dim_13] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_14 = 0; group_dim_14 < 8; group_dim_14++) {
                                float value_4 = __fmaf_rn(ring_inter_0[group_dim_14], decay_1, ring_intra_0[group_dim_14]);
                                float x_value_4 = unpack_bf16_3[group_dim_14];
                                float d_value_4 = d_values[group_dim_14];
                                if (D_mode != 0) {
                                    value_4 = __fmaf_rn(x_value_4, d_value_4, value_4);
                                }
                                float z_mul_4 = z_scale[group_dim_14];
                                value_4 = value_4 * z_mul_4;
                                group_values[group_dim_14] = value_4;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_4[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_4[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + dim_chunk_7 * 4 * 8 * 2 ^ (row_1 * 128 + dim_chunk_7 * 4 * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_4[0]), "r"(group_values_bf16_4[1]), "r"(group_values_bf16_4[2]), "r"(group_values_bf16_4[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_4[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_4[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + dim_chunk_7 * 4 * 8))[0] = reinterpret_cast<int4*>(group_values_pv_4)[0];
                            }
                            if (dim_chunk_7 == 0) {
                                tmem_ld_x8(&ring_intra_0[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 32);
                                tmem_ld_x8(&ring_inter_0[0], taddr_0_2 + row_tmem_base_1 + 448 + 32);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_7 * 4 + 1) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 1) * 8 % 64 * 2 ^ ((dim_chunk_7 * 4 + 1) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 1) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_3_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_3_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_3_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_7 * 4 + 1) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_1[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_1[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_7 * 4 + 1 < 4) {
                                    {
                                        const uint4* _vptr_20 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_7 * 4 + 1 + 4) * 8);
                                        uint4* _vdst_20 = reinterpret_cast<uint4*>(&ring_z_1[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_20[_blk] = _vptr_20[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_15 = 0; group_dim_15 < 8; group_dim_15++) {
                                    float z_value_5 = z_scale[group_dim_15];
                                    float _exp_22 = __expf(-z_value_5);
                                    float _rcp_21 = approx_rcp(1.0f + _exp_22);
                                    z_scale[group_dim_15] = z_value_5 * _rcp_21;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_16 = 0; group_dim_16 < 8; group_dim_16++) {
                                    z_scale[group_dim_16] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_17 = 0; group_dim_17 < 8; group_dim_17++) {
                                float value_5 = __fmaf_rn(ring_inter_1[group_dim_17], decay_1, ring_intra_1[group_dim_17]);
                                float x_value_5 = unpack_bf16_3_1[group_dim_17];
                                float d_value_5 = d_values[group_dim_17];
                                if (D_mode != 0) {
                                    value_5 = __fmaf_rn(x_value_5, d_value_5, value_5);
                                }
                                float z_mul_5 = z_scale[group_dim_17];
                                value_5 = value_5 * z_mul_5;
                                group_values[group_dim_17] = value_5;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_5[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_5[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_7 * 4 + 1) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_7 * 4 + 1) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_5[0]), "r"(group_values_bf16_5[1]), "r"(group_values_bf16_5[2]), "r"(group_values_bf16_5[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_5[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_5[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_7 * 4 + 1) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_5)[0];
                            }
                            if (dim_chunk_7 == 0) {
                                tmem_ld_x8(&ring_intra_1[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 40);
                                tmem_ld_x8(&ring_inter_1[0], taddr_0_2 + row_tmem_base_1 + 448 + 40);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_7 * 4 + 2) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 2) * 8 % 64 * 2 ^ ((dim_chunk_7 * 4 + 2) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 2) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_4[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_7 * 4 + 2) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_2[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_2[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_7 * 4 + 2 < 4) {
                                    {
                                        const uint4* _vptr_23 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_7 * 4 + 2 + 4) * 8);
                                        uint4* _vdst_23 = reinterpret_cast<uint4*>(&ring_z_2[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_23[_blk] = _vptr_23[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_18 = 0; group_dim_18 < 8; group_dim_18++) {
                                    float z_value_6 = z_scale[group_dim_18];
                                    float _exp_25 = __expf(-z_value_6);
                                    float _rcp_24 = approx_rcp(1.0f + _exp_25);
                                    z_scale[group_dim_18] = z_value_6 * _rcp_24;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_19 = 0; group_dim_19 < 8; group_dim_19++) {
                                    z_scale[group_dim_19] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_20 = 0; group_dim_20 < 8; group_dim_20++) {
                                float value_6 = __fmaf_rn(ring_inter_2[group_dim_20], decay_1, ring_intra_2[group_dim_20]);
                                float x_value_6 = unpack_bf16_4[group_dim_20];
                                float d_value_6 = d_values[group_dim_20];
                                if (D_mode != 0) {
                                    value_6 = __fmaf_rn(x_value_6, d_value_6, value_6);
                                }
                                float z_mul_6 = z_scale[group_dim_20];
                                value_6 = value_6 * z_mul_6;
                                group_values[group_dim_20] = value_6;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_6[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_6[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_7 * 4 + 2) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_7 * 4 + 2) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_6[0]), "r"(group_values_bf16_6[1]), "r"(group_values_bf16_6[2]), "r"(group_values_bf16_6[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_6[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_6[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_7 * 4 + 2) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_6)[0];
                            }
                            if (dim_chunk_7 == 0) {
                                tmem_ld_x8(&ring_intra_2[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 48);
                                tmem_ld_x8(&ring_inter_2[0], taddr_0_2 + row_tmem_base_1 + 448 + 48);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                : "r"((smem_x_tma_addr + input_stage_6 * 16384 + (unsigned int)((dim_chunk_7 * 4 + 3) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 3) * 8 % 64 * 2 ^ ((dim_chunk_7 * 4 + 3) * 8 / 64 * 16384 + row_1 * 128 + (dim_chunk_7 * 4 + 3) * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float unpack_bf16_5[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                (&unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                            }
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&d_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d_packed[(0) + 3]))
                                : "r"(smem_d_row_addr + (unsigned int)((dim_chunk_7 * 4 + 3) * 8 * 2)));
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&d_values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) << 16);
                                (&d_values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d_packed[_pair]) & 0xffff0000u);
                            }
                            if (has_z != 0) {
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    (&z_scale[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(ring_z_3[_pair]) << 16);
                                    (&z_scale[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(ring_z_3[_pair]) & 0xffff0000u);
                                }
                                if (dim_chunk_7 * 4 + 3 < 4) {
                                    {
                                        const uint4* _vptr_26 = reinterpret_cast<const uint4*>(z + z_row_base + (dim_chunk_7 * 4 + 3 + 4) * 8);
                                        uint4* _vdst_26 = reinterpret_cast<uint4*>(&ring_z_3[0]);
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vdst_26[_blk] = _vptr_26[_blk];
                                        }
                                    }
                                }
                                #pragma unroll
                                for (int group_dim_21 = 0; group_dim_21 < 8; group_dim_21++) {
                                    float z_value_7 = z_scale[group_dim_21];
                                    float _exp_28 = __expf(-z_value_7);
                                    float _rcp_27 = approx_rcp(1.0f + _exp_28);
                                    z_scale[group_dim_21] = z_value_7 * _rcp_27;
                                }
                            } else {
                                #pragma unroll
                                for (int group_dim_22 = 0; group_dim_22 < 8; group_dim_22++) {
                                    z_scale[group_dim_22] = 1.0f;
                                }
                            }
                            #pragma unroll
                            for (int group_dim_23 = 0; group_dim_23 < 8; group_dim_23++) {
                                float value_7 = __fmaf_rn(ring_inter_3[group_dim_23], decay_1, ring_intra_3[group_dim_23]);
                                float x_value_7 = unpack_bf16_5[group_dim_23];
                                float d_value_7 = d_values[group_dim_23];
                                if (D_mode != 0) {
                                    value_7 = __fmaf_rn(x_value_7, d_value_7, value_7);
                                }
                                float z_mul_7 = z_scale[group_dim_23];
                                value_7 = value_7 * z_mul_7;
                                group_values[group_dim_23] = value_7;
                            }
                            if (full_chunk != 0) {
                                unsigned int group_values_bf16_7[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_bf16_7[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + (dim_chunk_7 * 4 + 3) * 8 * 2 ^ (row_1 * 128 + (dim_chunk_7 * 4 + 3) * 8 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16_7[0]), "r"(group_values_bf16_7[1]), "r"(group_values_bf16_7[2]), "r"(group_values_bf16_7[3]) : "memory");
                            } else if (row_1 >= segment_offset_3 && row_1 < segment_limit_3) {
                                unsigned int group_values_pv_7[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                    group_values_pv_7[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(out_native + (out_row_base + (dim_chunk_7 * 4 + 3) * 8))[0] = reinterpret_cast<int4*>(group_values_pv_7)[0];
                            }
                            if (dim_chunk_7 == 0) {
                                tmem_ld_x8(&ring_intra_3[0], (unsigned int)(taddr_0_2 + row_tmem_base_1 + 256) + acc_stage_3 * 64 + 56);
                                tmem_ld_x8(&ring_inter_3[0], taddr_0_2 + row_tmem_base_1 + 448 + 56);
                            }
                        }
                    }
                    if (full_chunk != 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        if (elect_sync()) {
                            tma_store_4d((&out_map), 0, head_3, chunk_4 * 128 + warp_id_in_role * 32, physical_batch_2, smem_y_addr + output_stage * 16384 + (unsigned int)(warp_id_in_role * 4096));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        output_stage = output_stage ^ 1;
                        output_issued += 1;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    mbarrier_arrive(aux_empty_addr + (input_stage_6) * 8);
                    if (warp == 12) {
                        if (elect_sync()) {
                            mbarrier_arrive(x_empty_addr + (input_stage_6) * 8);
                        }
                    }
                    input_stage_6 += 1;
                    if (input_stage_6 == 2) { input_stage_6 = 0; _phase_aux_full_2 ^= 1; }
                    acc_stage_3 += 1;
                    if (acc_stage_3 == 2) { acc_stage_3 = 0; _phase_intra_full ^= 1; }
                }
            }
            int warp_id_in_role_1 = warp - 12;
            if (warp_id_in_role_1 == 0) {
                input_stage_6 += 1;
                if (input_stage_6 == 2) { input_stage_6 = 0; _phase_aux_full_2 ^= 1; }
                mbarrier_wait(aux_empty_addr + (input_stage_6) * 8, _phase_aux_full_2 ^ (unsigned int)(input_stage_6 + 2 >= 2));
                __syncwarp();
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("barrier.sync 10, 128;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

