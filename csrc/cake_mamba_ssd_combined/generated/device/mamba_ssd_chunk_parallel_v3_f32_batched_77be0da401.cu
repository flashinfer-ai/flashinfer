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
#define TMEM_Q_TMEM_OFFSET 256
#define TMEM_INTRA_TMEM_OFFSET 320
#define TMEM_STATE_DELTA_TMEM_OFFSET 384
#define TMEM_INTER_TMEM_OFFSET 448
#define TMEM_S_TMEM_OFFSET 0
#define NUM_PHASE1_PIPE_STAGES 2
#define NUM_PHASE3_PIPE_STAGES 2
#define NUM_STATE_PIPE_STAGES 2
#define NUM_INTRA1_ACC_PIPE_STAGES 2
#define SMEM_SMEM_B_OFF 0
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_C_OFF 65536
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_X_TMA_OFF 131072
#define SMEM_SMEM_X_TMA_STAGE_BYTES 16384
#define SMEM_SMEM_X_TMA_STRIDE 16384
#define SMEM_SMEM_X_OFF 131072
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_SCALED_B_OFF 163840
#define SMEM_SMEM_SCALED_B_STAGE_BYTES 32768
#define SMEM_SMEM_SCALED_B_STRIDE 32768
#define SMEM_SMEM_STATE_OFF 163840
#define SMEM_SMEM_STATE_STAGE_BYTES 16384
#define SMEM_SMEM_STATE_STRIDE 16384
#define SMEM_SMEM_DELTA_ALL_OFF 229376
#define SMEM_SMEM_DELTA_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_DELTA_ALL_STRIDE 1024
#define SMEM_SMEM_CUMSUM_ALL_OFF 230400
#define SMEM_SMEM_CUMSUM_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_CUMSUM_ALL_STRIDE 1024
#define SMEM_SMEM_Y_OFF 196608
#define SMEM_SMEM_Y_STAGE_BYTES 16384
#define SMEM_SMEM_Y_STRIDE 16384
#define SMEM_CTRL_OFF 231424
#define SMEM_CTRL_STAGE_BYTES 16
#define SMEM_CTRL_STRIDE 16
#define SMEM_TOTAL 231808
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





__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_mamba_ssd_chunk_parallel_v3_f32_batched(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap b_map, const __grid_constant__ CUtensorMap c_map, const __grid_constant__ CUtensorMap out_map, const __grid_constant__ CUtensorMap h_map, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, __half* __restrict__ delta_precomputed, float* __restrict__ cumsum_precomputed, float* __restrict__ A, __nv_bfloat16* __restrict__ B_tensor, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ D, __nv_bfloat16* __restrict__ z, float* __restrict__ dt_bias, float* __restrict__ initial_states, float* __restrict__ final_states, float* __restrict__ checkpoint_states, int* __restrict__ checkpoint_token_indices, int* __restrict__ checkpoint_state_slots, int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64, int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets, int* __restrict__ seq_chunk_cumsum, __nv_bfloat16* __restrict__ out_native, float* __restrict__ s_work, float* __restrict__ agg_work, unsigned int* __restrict__ h_words, unsigned int* __restrict__ grid_barrier, int nheads, int ngroups, int batch, int seqlen, int nchunks, int sequence_count, int num_logical_chunks, int mode_varlen, int D_mode, int has_z, int has_initial, int dt_softplus, float dt_min, float dt_max, int write_final_states, int checkpoint_state_count, int vcount)
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

    const int mbar_base = smem + 231440;
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
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_b_addr = smem + 0;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int smem_c_addr = smem + 65536;
    __nv_bfloat16* smem_x_tma = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int smem_x_tma_addr = smem + 131072;
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int smem_x_addr = smem + 131072;
    __nv_bfloat16* smem_scaled_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_scaled_b_addr = smem + 163840;
    __nv_bfloat16* smem_state = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int smem_state_addr = smem + 163840;
    float* smem_delta_all = reinterpret_cast<float*>(smem_raw + 229376);
    const int smem_delta_all_addr = smem + 229376;
    float* smem_cumsum_all = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_cumsum_all_addr = smem + 230400;
    __nv_bfloat16* smem_y = reinterpret_cast<__nv_bfloat16*>(smem_raw + 196608);
    const int smem_y_addr = smem + 196608;
    unsigned int* ctrl = reinterpret_cast<unsigned int*>(smem_raw + 231424);
    const int ctrl_addr = smem + 231424;

    // Mbarrier init (24 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[231440..231760)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'phase1_pipe' ---
            // b1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231440, 1);
            mbarrier_init(smem + 231448, 1);
            // b1_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 231456, 2);
            mbarrier_init(smem + 231464, 2);
            // x1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231472, 1);
            mbarrier_init(smem + 231480, 1);
            // x1_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 231488, 1);
            mbarrier_init(smem + 231496, 1);
            // aux1_full: 2 barriers, init_count=32
            mbarrier_init(smem + 231504, 32);
            mbarrier_init(smem + 231512, 32);
            // aux1_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 231520, 256);
            mbarrier_init(smem + 231528, 256);
            // scaled_b_full: 1 barriers, init_count=2
            mbarrier_init(smem + 231536, 2);
            // scaled_b_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 231544, 1);
            // state_delta_full: 1 barriers, init_count=1
            mbarrier_init(smem + 231552, 1);
            // --- pipeline 'phase3_pipe' ---
            // bc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231560, 1);
            mbarrier_init(smem + 231568, 1);
            // bc_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 231576, 2);
            mbarrier_init(smem + 231584, 2);
            // x3_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231592, 1);
            mbarrier_init(smem + 231600, 1);
            // x3_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 231608, 2);
            mbarrier_init(smem + 231616, 2);
            // aux3_full: 2 barriers, init_count=32
            mbarrier_init(smem + 231624, 32);
            mbarrier_init(smem + 231632, 32);
            // aux3_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 231640, 256);
            mbarrier_init(smem + 231648, 256);
            // --- pipeline 'state_pipe' ---
            // state_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231656, 1);
            mbarrier_init(smem + 231664, 1);
            // state_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 231672, 1);
            mbarrier_init(smem + 231680, 1);
            // --- pipeline 'intra1_acc_pipe' ---
            // cb_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231688, 1);
            mbarrier_init(smem + 231696, 1);
            // cb_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 231704, 1);
            mbarrier_init(smem + 231712, 1);
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 231720, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 231728, 1);
            // intra_full: 1 barriers, init_count=1
            mbarrier_init(smem + 231736, 1);
            // inter_full: 1 barriers, init_count=1
            mbarrier_init(smem + 231744, 1);
            // outputs_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 231752, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 231760);
    if (warp == 12) {
        int _tmem_hold = smem + 231760;
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
    const int tmem_s_tmem = taddr;
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
            unsigned int stage1 = 0;
            unsigned int stage3 = 0;
            unsigned int sstage = 0;
            int tot[1];
            tot[0] = sequence_count * nchunks * nheads;
            int veff[1];
            int total = tot[0];
            int g = 1;
            int ranges = (total + g - 1) / g;
            if (ranges < 1) {
                ranges = 1;
            }
            if (ranges > vcount) {
                ranges = vcount;
            }
            veff[0] = ranges;
            int vcount_eff = veff[0];
            unsigned int total_u = (unsigned int)vcount_eff;
            unsigned int _load_acquire_0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"(reinterpret_cast<unsigned int*>(grid_barrier + 1)) : "memory");
            unsigned int epoch_seen = _load_acquire_0;
            unsigned int claim_limit = (unsigned int)(vcount_eff + num_bids - 1);
            unsigned int _phase_scaled_b_full_0 = 0;
            unsigned int _phase_x1_full = 0;
            unsigned int _phase_bc_full = 0;
            unsigned int _phase_state_full = 0;
            unsigned int _phase_outputs_empty_0 = 1;
            while (1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        uint32_t claimed;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(claimed) : "l"(&grid_barrier[0]), "r"(static_cast<uint32_t>(claim_limit)) : "memory");
                        if (claimed == claim_limit) {
                            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                        ctrl[0] = claimed;
                        ctrl[1] = epoch_seen;
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u = ctrl[0];
                if (ticket_u >= total_u) {
                    break;
                }
                int ticket = (int)ticket_u;
                int rb[2];
                int total_0 = tot[0];
                rb[0] = (ticket * total_0 + vcount_eff - 1) / vcount_eff;
                rb[1] = ((ticket + 1) * total_0 + vcount_eff - 1) / vcount_eff;
                int a_b = rb[0];
                int a_end = rb[1];
                int _uniform_21 = make_warp_uniform(a_b);
                a_b = _uniform_21;
                int _uniform_22 = make_warp_uniform(a_end);
                a_end = _uniform_22;
                int xa = a_b;
                while (xa < a_end) {
                    int loca[6];
                    int sequence = 0;
                    int s_base = 0;
                    int n_item = nchunks;
                    sequence = xa / (nchunks * nheads);
                    s_base = sequence * nchunks;
                    int rem = xa - s_base * nheads;
                    int head = rem / n_item;
                    int chunk = rem - head * n_item;
                    loca[0] = sequence;
                    loca[1] = head;
                    loca[2] = chunk;
                    loca[3] = n_item;
                    loca[4] = s_base + chunk;
                    loca[5] = xa - chunk;
                    int head_0 = loca[1];
                    int first = loca[4];
                    int seg_len = loca[3] - loca[2];
                    if (seg_len > a_end - xa) {
                        seg_len = a_end - xa;
                    }
                    int _uniform_23 = make_warp_uniform(head_0);
                    head_0 = _uniform_23;
                    int _uniform_24 = make_warp_uniform(first);
                    first = _uniform_24;
                    int _uniform_25 = make_warp_uniform(seg_len);
                    seg_len = _uniform_25;
                    #pragma unroll 1
                    for (int logical = first; logical < first + seg_len; logical++) {
                        mbarrier_wait(scaled_b_full_addr, _phase_scaled_b_full_0);
                        _phase_scaled_b_full_0 ^= 1;
                        mbarrier_wait(x1_full_addr + (stage1) * 8, _phase_x1_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int t_range = xa - a_b + (logical - first);
                        unsigned int s_slot = (unsigned int)(t_range % 8);
                        int _mma_a_lo_1 = make_warp_uniform((((smem_scaled_b_addr) >> 4) & 0x7FFF) + (0) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x7FFF) | 0x4000000) + (stage1) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_s_tmem + (s_slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135333008, 1);
                            }
                        }
                        elect_commit(state_delta_full_addr);
                        elect_commit(scaled_b_empty_addr);
                        elect_commit(x1_empty_addr + (stage1) * 8);
                        stage1 += 1;
                        if (stage1 == 2) { stage1 = 0; _phase_x1_full ^= 1; }
                    }
                    xa = xa + seg_len;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int xc = a_b;
                while (xc < a_end) {
                    int locc[6];
                    int sequence_1 = 0;
                    int s_base_1 = 0;
                    int n_item_1 = nchunks;
                    sequence_1 = xc / (nchunks * nheads);
                    s_base_1 = sequence_1 * nchunks;
                    int rem_1 = xc - s_base_1 * nheads;
                    int head_1 = rem_1 / n_item_1;
                    int chunk_1 = rem_1 - head_1 * n_item_1;
                    locc[0] = sequence_1;
                    locc[1] = head_1;
                    locc[2] = chunk_1;
                    locc[3] = n_item_1;
                    locc[4] = s_base_1 + chunk_1;
                    locc[5] = xc - chunk_1;
                    int head_0_1 = locc[1];
                    int first_1 = locc[4];
                    int seg_len_1 = locc[3] - locc[2];
                    if (seg_len_1 > a_end - xc) {
                        seg_len_1 = a_end - xc;
                    }
                    int _uniform_26 = make_warp_uniform(head_0_1);
                    head_0_1 = _uniform_26;
                    int _uniform_27 = make_warp_uniform(first_1);
                    first_1 = _uniform_27;
                    int _uniform_28 = make_warp_uniform(seg_len_1);
                    seg_len_1 = _uniform_28;
                    #pragma unroll 1
                    for (int logical_1 = first_1; logical_1 < first_1 + seg_len_1; logical_1++) {
                        mbarrier_wait(bc_full_addr + (stage3) * 8, _phase_bc_full);
                        mbarrier_wait(state_full_addr + (sstage) * 8, _phase_state_full);
                        mbarrier_wait(outputs_empty_addr, _phase_outputs_empty_0);
                        _phase_outputs_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = make_warp_uniform((((smem_c_addr) >> 4) & 0x7FFF) + (stage3) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform((((smem_state_addr) >> 4) & 0x7FFF) + (sstage) * 1024);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
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
                    xc = xc + seg_len_1;
                }
            }
        }
    }
    // ---- Role: mma_intra ----
    if (warp == 1) {
        { // mma_intra_main
            unsigned int stage3_1 = 0;
            unsigned int acc_stage = 0;
            int tot_1[1];
            tot_1[0] = sequence_count * nchunks * nheads;
            int veff_1[1];
            int total_1 = tot_1[0];
            int g_1 = 1;
            int ranges_1 = (total_1 + g_1 - 1) / g_1;
            if (ranges_1 < 1) {
                ranges_1 = 1;
            }
            if (ranges_1 > vcount) {
                ranges_1 = vcount;
            }
            veff_1[0] = ranges_1;
            int vcount_eff_1 = veff_1[0];
            unsigned int total_u_1 = (unsigned int)vcount_eff_1;
            unsigned int _phase_bc_full_1 = 0;
            unsigned int _phase_cb_empty = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_outputs_empty_0_1 = 1;
            while (1) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_1 = ctrl[0];
                if (ticket_u_1 >= total_u_1) {
                    break;
                }
                int ticket_1 = (int)ticket_u_1;
                int rb_1[2];
                int total_0_1 = tot_1[0];
                rb_1[0] = (ticket_1 * total_0_1 + vcount_eff_1 - 1) / vcount_eff_1;
                rb_1[1] = ((ticket_1 + 1) * total_0_1 + vcount_eff_1 - 1) / vcount_eff_1;
                int a_b_1 = rb_1[0];
                int a_end_1 = rb_1[1];
                int _uniform_16 = make_warp_uniform(a_b_1);
                a_b_1 = _uniform_16;
                int _uniform_17 = make_warp_uniform(a_end_1);
                a_end_1 = _uniform_17;
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int xa_1 = a_b_1;
                while (xa_1 < a_end_1) {
                    int loca_1[6];
                    int sequence_2 = 0;
                    int s_base_2 = 0;
                    int n_item_2 = nchunks;
                    sequence_2 = xa_1 / (nchunks * nheads);
                    s_base_2 = sequence_2 * nchunks;
                    int rem_2 = xa_1 - s_base_2 * nheads;
                    int head_2 = rem_2 / n_item_2;
                    int chunk_2 = rem_2 - head_2 * n_item_2;
                    loca_1[0] = sequence_2;
                    loca_1[1] = head_2;
                    loca_1[2] = chunk_2;
                    loca_1[3] = n_item_2;
                    loca_1[4] = s_base_2 + chunk_2;
                    loca_1[5] = xa_1 - chunk_2;
                    int head_0_2 = loca_1[1];
                    int first_2 = loca_1[4];
                    int seg_len_2 = loca_1[3] - loca_1[2];
                    if (seg_len_2 > a_end_1 - xa_1) {
                        seg_len_2 = a_end_1 - xa_1;
                    }
                    int _uniform_18 = make_warp_uniform(head_0_2);
                    head_0_2 = _uniform_18;
                    int _uniform_19 = make_warp_uniform(first_2);
                    first_2 = _uniform_19;
                    int _uniform_20 = make_warp_uniform(seg_len_2);
                    seg_len_2 = _uniform_20;
                    #pragma unroll 1
                    for (int logical_2 = first_2; logical_2 < first_2 + seg_len_2; logical_2++) {
                        mbarrier_wait(bc_full_addr + (stage3_1) * 8, _phase_bc_full_1);
                        mbarrier_wait(cb_empty_addr + (acc_stage) * 8, _phase_cb_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_c_addr) >> 4) & 0x7FFF) + (stage3_1) * 2048);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x7FFF) + (stage3_1) * 2048);
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
                        mbarrier_wait(x3_full_addr + (stage3_1) * 8, _phase_bc_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_1 = make_warp_uniform(((((smem_x_addr) >> 4) & 0x7FFF) | 0x4000000) + (stage3_1) * 1024);
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
                    :: "r"(tmem_intra_tmem), "r"(_mma_b_lo_1), "r"(tmem_q_tmem), "r"(0) : "memory");
                        elect_commit(intra_full_addr);
                        elect_commit(q_empty_addr);
                        elect_commit(x3_empty_addr + (stage3_1) * 8);
                        stage3_1 += 1;
                        if (stage3_1 == 2) { stage3_1 = 0; _phase_bc_full_1 ^= 1; }
                        acc_stage += 1;
                        if (acc_stage == 2) { acc_stage = 0; _phase_cb_empty ^= 1; }
                    }
                    xa_1 = xa_1 + seg_len_2;
                }
            }
        }
    }
    // ---- Role: load_bc ----
    if (warp == 2) {
        { // load_bc_main
            unsigned int stage1_1 = 0;
            unsigned int stage3_2 = 0;
            unsigned int sstage_1 = 0;
            int tot_2[1];
            tot_2[0] = sequence_count * nchunks * nheads;
            int veff_2[1];
            int total_2 = tot_2[0];
            int g_2 = 1;
            int ranges_2 = (total_2 + g_2 - 1) / g_2;
            if (ranges_2 < 1) {
                ranges_2 = 1;
            }
            if (ranges_2 > vcount) {
                ranges_2 = vcount;
            }
            veff_2[0] = ranges_2;
            int vcount_eff_2 = veff_2[0];
            unsigned int total_u_2 = (unsigned int)vcount_eff_2;
            unsigned int _phase_b1_empty = 1;
            unsigned int _phase_bc_empty = 1;
            unsigned int _phase_state_empty = 1;
            while (1) {
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_2 = ctrl[0];
                if (ticket_u_2 >= total_u_2) {
                    break;
                }
                int ticket_2 = (int)ticket_u_2;
                int rb_2[2];
                int total_0_2 = tot_2[0];
                rb_2[0] = (ticket_2 * total_0_2 + vcount_eff_2 - 1) / vcount_eff_2;
                rb_2[1] = ((ticket_2 + 1) * total_0_2 + vcount_eff_2 - 1) / vcount_eff_2;
                int a_b_2 = rb_2[0];
                int a_end_2 = rb_2[1];
                int _uniform_0 = make_warp_uniform(a_b_2);
                a_b_2 = _uniform_0;
                int _uniform_1 = make_warp_uniform(a_end_2);
                a_end_2 = _uniform_1;
                int xa_2 = a_b_2;
                while (xa_2 < a_end_2) {
                    int loca_2[6];
                    int sequence_3 = 0;
                    int s_base_3 = 0;
                    int n_item_3 = nchunks;
                    sequence_3 = xa_2 / (nchunks * nheads);
                    s_base_3 = sequence_3 * nchunks;
                    int rem_3 = xa_2 - s_base_3 * nheads;
                    int head_3 = rem_3 / n_item_3;
                    int chunk_3 = rem_3 - head_3 * n_item_3;
                    loca_2[0] = sequence_3;
                    loca_2[1] = head_3;
                    loca_2[2] = chunk_3;
                    loca_2[3] = n_item_3;
                    loca_2[4] = s_base_3 + chunk_3;
                    loca_2[5] = xa_2 - chunk_3;
                    int head_0_3 = loca_2[1];
                    int first_3 = loca_2[4];
                    int seg_len_3 = loca_2[3] - loca_2[2];
                    if (seg_len_3 > a_end_2 - xa_2) {
                        seg_len_3 = a_end_2 - xa_2;
                    }
                    int _uniform_2 = make_warp_uniform(head_0_3);
                    head_0_3 = _uniform_2;
                    int _uniform_3 = make_warp_uniform(first_3);
                    first_3 = _uniform_3;
                    int _uniform_4 = make_warp_uniform(seg_len_3);
                    seg_len_3 = _uniform_4;
                    #pragma unroll 1
                    for (int logical_3 = first_3; logical_3 < first_3 + seg_len_3; logical_3++) {
                        int group = head_0_3 * ngroups / nheads;
                        int physical_chunk = 0;
                        int physical_batch = 0;
                        physical_batch = logical_3 / nchunks;
                        physical_chunk = logical_3 - physical_batch * nchunks;
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
                    xa_2 = xa_2 + seg_len_3;
                }
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("fence.proxy.async;" ::: "memory");
                int xc_1 = a_b_2;
                while (xc_1 < a_end_2) {
                    int locc_1[6];
                    int sequence_4 = 0;
                    int s_base_4 = 0;
                    int n_item_4 = nchunks;
                    sequence_4 = xc_1 / (nchunks * nheads);
                    s_base_4 = sequence_4 * nchunks;
                    int rem_4 = xc_1 - s_base_4 * nheads;
                    int head_4 = rem_4 / n_item_4;
                    int chunk_4 = rem_4 - head_4 * n_item_4;
                    locc_1[0] = sequence_4;
                    locc_1[1] = head_4;
                    locc_1[2] = chunk_4;
                    locc_1[3] = n_item_4;
                    locc_1[4] = s_base_4 + chunk_4;
                    locc_1[5] = xc_1 - chunk_4;
                    int head_0_4 = locc_1[1];
                    int first_4 = locc_1[4];
                    int seg_len_4 = locc_1[3] - locc_1[2];
                    if (seg_len_4 > a_end_2 - xc_1) {
                        seg_len_4 = a_end_2 - xc_1;
                    }
                    int _uniform_5 = make_warp_uniform(head_0_4);
                    head_0_4 = _uniform_5;
                    int _uniform_6 = make_warp_uniform(first_4);
                    first_4 = _uniform_6;
                    int _uniform_7 = make_warp_uniform(seg_len_4);
                    seg_len_4 = _uniform_7;
                    #pragma unroll 1
                    for (int logical_4 = first_4; logical_4 < first_4 + seg_len_4; logical_4++) {
                        int tile_row = logical_4 * nheads + head_0_4;
                        int group_1 = head_0_4 * ngroups / nheads;
                        int physical_chunk_1 = 0;
                        int physical_batch_1 = 0;
                        physical_batch_1 = logical_4 / nchunks;
                        physical_chunk_1 = logical_4 - physical_batch_1 * nchunks;
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
                    xc_1 = xc_1 + seg_len_4;
                }
            }
        }
    }
    // ---- Role: load_x_dt ----
    if (warp == 3) {
        { // load_x_dt_main
            unsigned int stage1_2 = 0;
            unsigned int stage3_3 = 0;
            int tot_3[1];
            tot_3[0] = sequence_count * nchunks * nheads;
            int veff_3[1];
            int total_3 = tot_3[0];
            int g_3 = 1;
            int ranges_3 = (total_3 + g_3 - 1) / g_3;
            if (ranges_3 < 1) {
                ranges_3 = 1;
            }
            if (ranges_3 > vcount) {
                ranges_3 = vcount;
            }
            veff_3[0] = ranges_3;
            int vcount_eff_3 = veff_3[0];
            unsigned int total_u_3 = (unsigned int)vcount_eff_3;
            unsigned int _phase_aux1_empty = 1;
            unsigned int _phase_aux3_empty = 1;
            while (1) {
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_3 = ctrl[0];
                if (ticket_u_3 >= total_u_3) {
                    break;
                }
                int ticket_3 = (int)ticket_u_3;
                int rb_3[2];
                int total_0_3 = tot_3[0];
                rb_3[0] = (ticket_3 * total_0_3 + vcount_eff_3 - 1) / vcount_eff_3;
                rb_3[1] = ((ticket_3 + 1) * total_0_3 + vcount_eff_3 - 1) / vcount_eff_3;
                int a_b_3 = rb_3[0];
                int a_end_3 = rb_3[1];
                int _uniform_8 = make_warp_uniform(a_b_3);
                a_b_3 = _uniform_8;
                int _uniform_9 = make_warp_uniform(a_end_3);
                a_end_3 = _uniform_9;
                int xa_3 = a_b_3;
                while (xa_3 < a_end_3) {
                    int loca_3[6];
                    int sequence_5 = 0;
                    int s_base_5 = 0;
                    int n_item_5 = nchunks;
                    sequence_5 = xa_3 / (nchunks * nheads);
                    s_base_5 = sequence_5 * nchunks;
                    int rem_5 = xa_3 - s_base_5 * nheads;
                    int head_5 = rem_5 / n_item_5;
                    int chunk_5 = rem_5 - head_5 * n_item_5;
                    loca_3[0] = sequence_5;
                    loca_3[1] = head_5;
                    loca_3[2] = chunk_5;
                    loca_3[3] = n_item_5;
                    loca_3[4] = s_base_5 + chunk_5;
                    loca_3[5] = xa_3 - chunk_5;
                    int head_0_5 = loca_3[1];
                    int first_5 = loca_3[4];
                    int seg_len_5 = loca_3[3] - loca_3[2];
                    if (seg_len_5 > a_end_3 - xa_3) {
                        seg_len_5 = a_end_3 - xa_3;
                    }
                    int _uniform_10 = make_warp_uniform(head_0_5);
                    head_0_5 = _uniform_10;
                    int _uniform_11 = make_warp_uniform(first_5);
                    first_5 = _uniform_11;
                    int _uniform_12 = make_warp_uniform(seg_len_5);
                    seg_len_5 = _uniform_12;
                    #pragma unroll 1
                    for (int logical_5 = first_5; logical_5 < first_5 + seg_len_5; logical_5++) {
                        int tile_row_1 = logical_5 * nheads + head_0_5;
                        int physical_chunk_2 = 0;
                        int physical_batch_2 = 0;
                        int segment_offset = 0;
                        int segment_limit = 128;
                        physical_batch_2 = logical_5 / nchunks;
                        physical_chunk_2 = logical_5 - physical_batch_2 * nchunks;
                        int chunk_tokens = seqlen - physical_chunk_2 * 128;
                        if (chunk_tokens > 128) {
                            chunk_tokens = 128;
                        }
                        mbarrier_wait(aux1_empty_addr + (stage1_2) * 8, _phase_aux1_empty);
                        mbarrier_wait(x1_empty_addr + (stage1_2) * 8, _phase_aux1_empty);
                        int token_in_batch_2 = physical_chunk_2 * 128;
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_x_tma_addr + stage1_2 * 16384, (&x_map), 0, head_0_5, token_in_batch_2, physical_batch_2, x1_full_addr + (stage1_2) * 8);
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
                                delta_value = (float)delta_precomputed[tile_row_1 * 128 + physical_token];
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
                        if (stage1_2 == 2) { stage1_2 = 0; _phase_aux1_empty ^= 1; }
                    }
                    xa_3 = xa_3 + seg_len_5;
                }
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int xc_2 = a_b_3;
                while (xc_2 < a_end_3) {
                    int locc_2[6];
                    int sequence_6 = 0;
                    int s_base_6 = 0;
                    int n_item_6 = nchunks;
                    sequence_6 = xc_2 / (nchunks * nheads);
                    s_base_6 = sequence_6 * nchunks;
                    int rem_6 = xc_2 - s_base_6 * nheads;
                    int head_6 = rem_6 / n_item_6;
                    int chunk_6 = rem_6 - head_6 * n_item_6;
                    locc_2[0] = sequence_6;
                    locc_2[1] = head_6;
                    locc_2[2] = chunk_6;
                    locc_2[3] = n_item_6;
                    locc_2[4] = s_base_6 + chunk_6;
                    locc_2[5] = xc_2 - chunk_6;
                    int head_0_6 = locc_2[1];
                    int first_6 = locc_2[4];
                    int seg_len_6 = locc_2[3] - locc_2[2];
                    if (seg_len_6 > a_end_3 - xc_2) {
                        seg_len_6 = a_end_3 - xc_2;
                    }
                    int _uniform_13 = make_warp_uniform(head_0_6);
                    head_0_6 = _uniform_13;
                    int _uniform_14 = make_warp_uniform(first_6);
                    first_6 = _uniform_14;
                    int _uniform_15 = make_warp_uniform(seg_len_6);
                    seg_len_6 = _uniform_15;
                    #pragma unroll 1
                    for (int logical_6 = first_6; logical_6 < first_6 + seg_len_6; logical_6++) {
                        int tile_row_2 = logical_6 * nheads + head_0_6;
                        int physical_chunk_3 = 0;
                        int physical_batch_3 = 0;
                        int segment_offset_1 = 0;
                        int segment_limit_1 = 128;
                        physical_batch_3 = logical_6 / nchunks;
                        physical_chunk_3 = logical_6 - physical_batch_3 * nchunks;
                        int chunk_tokens_1 = seqlen - physical_chunk_3 * 128;
                        if (chunk_tokens_1 > 128) {
                            chunk_tokens_1 = 128;
                        }
                        mbarrier_wait(aux3_empty_addr + (stage3_3) * 8, _phase_aux3_empty);
                        mbarrier_wait(x3_empty_addr + (stage3_3) * 8, _phase_aux3_empty);
                        int token_in_batch_3 = physical_chunk_3 * 128;
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_x_tma_addr + stage3_3 * 16384, (&x_map), 0, head_0_6, token_in_batch_3, physical_batch_3, x3_full_addr + (stage3_3) * 8);
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
                                delta_value_1 = (float)delta_precomputed[tile_row_2 * 128 + physical_token_1];
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
                        if (stage3_3 == 2) { stage3_3 = 0; _phase_aux3_empty ^= 1; }
                    }
                    xc_2 = xc_2 + seg_len_6;
                }
            }
        }
    }
    // ---- Role: pre_inter ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // pre_inter_main
            int taddr_0 = taddr;
            unsigned int stage1_3 = 0;
            int chain_tid = (warp - 4) * 32 + lane;
            int tot_4[1];
            tot_4[0] = sequence_count * nchunks * nheads;
            int veff_4[1];
            int total_4 = tot_4[0];
            int g_4 = 1;
            int ranges_4 = (total_4 + g_4 - 1) / g_4;
            if (ranges_4 < 1) {
                ranges_4 = 1;
            }
            if (ranges_4 > vcount) {
                ranges_4 = vcount;
            }
            veff_4[0] = ranges_4;
            int vcount_eff_4 = veff_4[0];
            unsigned int total_u_4 = (unsigned int)vcount_eff_4;
            unsigned int _phase_aux1_full = 0;
            unsigned int _phase_scaled_b_empty_0 = 1;
            unsigned int _phase_state_delta_full_0 = 0;
            while (1) {
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_4 = ctrl[0];
                unsigned int epoch = ctrl[1];
                if (ticket_u_4 >= total_u_4) {
                    break;
                }
                int ticket_4 = (int)ticket_u_4;
                int rb_4[2];
                int total_0_4 = tot_4[0];
                rb_4[0] = (ticket_4 * total_0_4 + vcount_eff_4 - 1) / vcount_eff_4;
                rb_4[1] = ((ticket_4 + 1) * total_0_4 + vcount_eff_4 - 1) / vcount_eff_4;
                int a_b_4 = rb_4[0];
                int a_end_4 = rb_4[1];
                int _uniform_37 = make_warp_uniform(a_b_4);
                a_b_4 = _uniform_37;
                int _uniform_38 = make_warp_uniform(a_end_4);
                a_end_4 = _uniform_38;
                int xa_4 = a_b_4;
                while (xa_4 < a_end_4) {
                    int loca_4[6];
                    int sequence_7 = 0;
                    int s_base_7 = 0;
                    int n_item_7 = nchunks;
                    sequence_7 = xa_4 / (nchunks * nheads);
                    s_base_7 = sequence_7 * nchunks;
                    int rem_7 = xa_4 - s_base_7 * nheads;
                    int head_7 = rem_7 / n_item_7;
                    int chunk_7 = rem_7 - head_7 * n_item_7;
                    loca_4[0] = sequence_7;
                    loca_4[1] = head_7;
                    loca_4[2] = chunk_7;
                    loca_4[3] = n_item_7;
                    loca_4[4] = s_base_7 + chunk_7;
                    loca_4[5] = xa_4 - chunk_7;
                    int head_0_7 = loca_4[1];
                    int first_7 = loca_4[4];
                    int seg_len_7 = loca_4[3] - loca_4[2];
                    if (seg_len_7 > a_end_4 - xa_4) {
                        seg_len_7 = a_end_4 - xa_4;
                    }
                    int _uniform_39 = make_warp_uniform(head_0_7);
                    head_0_7 = _uniform_39;
                    int _uniform_40 = make_warp_uniform(first_7);
                    first_7 = _uniform_40;
                    int _uniform_41 = make_warp_uniform(seg_len_7);
                    seg_len_7 = _uniform_41;
                    int b_state_base = (warp % 4 * 4 + lane / 8) * 8;
                    int b_col_lane = lane % 8;
                    #pragma unroll 1
                    for (int logical_7 = first_7; logical_7 < first_7 + seg_len_7; logical_7++) {
                        int tile_row_3 = logical_7 * nheads + head_0_7;
                        int physical_chunk_4 = 0;
                        int segment_offset_2 = 0;
                        int segment_limit_2 = 128;
                        int sequence_0 = logical_7 / nchunks;
                        physical_chunk_4 = logical_7 - sequence_0 * nchunks;
                        int chunk_tokens_2 = seqlen - physical_chunk_4 * 128;
                        if (chunk_tokens_2 > 128) {
                            chunk_tokens_2 = 128;
                        }
                        mbarrier_wait(aux1_full_addr + (stage1_3) * 8, _phase_aux1_full);
                        mbarrier_wait(b1_full_addr + (stage1_3) * 8, _phase_aux1_full);
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
                                float b_scale = approx_exp2((last_cumsum - smem_cumsum_all[stage1_3 * 128 + (unsigned int)col]) * 1.442695f);
                                b_scale *= smem_delta_all[stage1_3 * 128 + (unsigned int)col];
                                const float2 _scale2_0 = {b_scale, b_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(scaled_b_values)[_ls], _scale2_0);
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
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        mbarrier_arrive(aux1_empty_addr + (stage1_3) * 8);
                        if (warp == 4) {
                            if (elect_sync()) {
                                mbarrier_arrive(scaled_b_full_addr);
                                mbarrier_arrive(b1_empty_addr + (stage1_3) * 8);
                            }
                        }
                        mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0);
                        _phase_state_delta_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int t_range_1 = xa_4 - a_b_4 + (logical_7 - first_7);
                        if (t_range_1 + 8 < a_end_4 - a_b_4) {
                            int s_tile_base = tile_row_3 * 8192;
                            int s_slot_col = t_range_1 % 8 * 64;
                            #pragma unroll
                            for (int row_half = 0; row_half < 2; row_half++) {
                                int state_row_origin = warp % 4 * 32 + row_half * 16;
                                float _tmem_load_2[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                    : "r"(taddr_0 + (state_row_origin << 16) + s_slot_col));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int state_pair = 0; state_pair < 8; state_pair++) {
                                    int state_row = state_row_origin + lane / 4 + state_pair % 2 * 8;
                                    int dim = state_pair / 2 * 8 + lane % 4 * 2;
                                    {
                                        float2 _v2 = make_float2(_tmem_load_2[2 * state_pair + 0], _tmem_load_2[2 * state_pair + 1]);
                                        *reinterpret_cast<float2*>(s_work + s_tile_base + state_row * 64 + dim) = _v2;
                                    }
                                }
                            }
                        }
                        stage1_3 += 1;
                        if (stage1_3 == 2) { stage1_3 = 0; _phase_aux1_full ^= 1; }
                    }
                    xa_4 = xa_4 + seg_len_7;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int total_1_1 = tot_4[0];
                asm volatile("tcgen05.fence::after_thread_sync;");
                int s_row = chain_tid % 128;
                int dim_base = chain_tid / 128 * 32;
                int elem_base = dim_base * 128 + s_row;
                int pair_base = (dim_base + s_row % 2) * 64 + s_row / 2;
                unsigned int h_half_shift = (unsigned int)((1 - s_row % 2) * 16);
                unsigned int h_mask_mine = 65535;
                if (s_row % 2 != 0) {
                    h_mask_mine = 4294901760u;
                }
                int s_tmem_row = taddr_0 + (warp % 4 * 32 << 16) + dim_base;
                int s_work_row = s_row * 64 + dim_base;
                int s_t_off = 0;
                int s_resident_from = a_end_4 - a_b_4 - 8;
                int handoff_lane = chain_tid * 4;
                float h[32];
                float s_a[32];
                float s_b[32];
                float d_ring[2];
                float lc_ring[2];
                d_ring[0] = 0.0f;
                d_ring[1] = 0.0f;
                lc_ring[0] = 0.0f;
                lc_ring[1] = 0.0f;
                #pragma unroll
                for (int k = 0; k < 32; k++) {
                    h[k] = 0.0f;
                }
                if (a_b_4 < a_end_4) {
                    int last_loc[6];
                    int sequence_8 = 0;
                    int s_base_8 = 0;
                    int n_item_8 = nchunks;
                    sequence_8 = (a_end_4 - 1) / (nchunks * nheads);
                    s_base_8 = sequence_8 * nchunks;
                    int rem_8 = a_end_4 - 1 - s_base_8 * nheads;
                    int head_8 = rem_8 / n_item_8;
                    int chunk_8 = rem_8 - head_8 * n_item_8;
                    last_loc[0] = sequence_8;
                    last_loc[1] = head_8;
                    last_loc[2] = chunk_8;
                    last_loc[3] = n_item_8;
                    last_loc[4] = s_base_8 + chunk_8;
                    last_loc[5] = a_end_4 - 1 - chunk_8;
                    int l_sequence = last_loc[0];
                    int l_head = last_loc[1];
                    int l_chunk = last_loc[2];
                    int l_n = last_loc[3];
                    int l_logical = last_loc[4];
                    int l_item_x = last_loc[5];
                    if (l_n > l_chunk + 1) {
                        int agg_first = l_logical - l_chunk;
                        if (a_b_4 > l_item_x) {
                            agg_first = agg_first + (a_b_4 - l_item_x);
                        }
                        int agg_last = l_logical;
                        float lsum = 0.0f;
                        s_t_off = l_item_x - a_b_4 - (l_logical - l_chunk);
                        int look_tile = agg_first * nheads + l_head;
                        int look_physical = 0;
                        int look_offset = 0;
                        int look_limit = 128;
                        look_physical = agg_first - l_sequence * nchunks;
                        int look_tokens = seqlen - look_physical * 128;
                        if (look_tokens > 128) {
                            look_tokens = 128;
                        }
                        int source_token_2 = look_tokens - 1;
                        float last_cumsum_1 = cumsum_precomputed[look_tile * 128 + source_token_2];
                        lc_ring[0] = last_cumsum_1;
                        float segment_base = 0.0f;
                        float _exp2_2 = approx_exp2((last_cumsum_1 - segment_base) * 1.442695f);
                        d_ring[0] = _exp2_2;
                        int look_t = agg_first + s_t_off;
                        if (look_t >= s_resident_from) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_a[0]), "=f"(s_a[1]), "=f"(s_a[2]), "=f"(s_a[3]), "=f"(s_a[4]), "=f"(s_a[5]), "=f"(s_a[6]), "=f"(s_a[7]), "=f"(s_a[8]), "=f"(s_a[9]), "=f"(s_a[10]), "=f"(s_a[11]), "=f"(s_a[12]), "=f"(s_a[13]), "=f"(s_a[14]), "=f"(s_a[15]), "=f"(s_a[16]), "=f"(s_a[17]), "=f"(s_a[18]), "=f"(s_a[19]), "=f"(s_a[20]), "=f"(s_a[21]), "=f"(s_a[22]), "=f"(s_a[23]), "=f"(s_a[24]), "=f"(s_a[25]), "=f"(s_a[26]), "=f"(s_a[27]), "=f"(s_a[28]), "=f"(s_a[29]), "=f"(s_a[30]), "=f"(s_a[31])
                                : "r"(s_tmem_row + look_t % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q = 0; q < 8; q++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile * 8192 + s_work_row + 4 * q);
                                    s_a[4 * q + 0] = _v4.x;
                                    s_a[4 * q + 1] = _v4.y;
                                    s_a[4 * q + 2] = _v4.z;
                                    s_a[4 * q + 3] = _v4.w;
                                }
                            }
                        }
                        int agg_look1 = ((agg_first + 1) < (agg_last) ? (agg_first + 1) : (agg_last));
                        int look_tile_6 = agg_look1 * nheads + l_head;
                        int look_physical_6 = 0;
                        int look_offset_6 = 0;
                        int look_limit_6 = 128;
                        look_physical_6 = agg_look1 - l_sequence * nchunks;
                        int look_tokens_6 = seqlen - look_physical_6 * 128;
                        if (look_tokens_6 > 128) {
                            look_tokens_6 = 128;
                        }
                        int source_token_6 = look_tokens_6 - 1;
                        float last_cumsum_6 = cumsum_precomputed[look_tile_6 * 128 + source_token_6];
                        lc_ring[1] = last_cumsum_6;
                        float segment_base_6 = 0.0f;
                        float _exp2_4 = approx_exp2((last_cumsum_6 - segment_base_6) * 1.442695f);
                        d_ring[1] = _exp2_4;
                        int look_t_6 = agg_look1 + s_t_off;
                        if (look_t_6 >= s_resident_from) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_b[0]), "=f"(s_b[1]), "=f"(s_b[2]), "=f"(s_b[3]), "=f"(s_b[4]), "=f"(s_b[5]), "=f"(s_b[6]), "=f"(s_b[7]), "=f"(s_b[8]), "=f"(s_b[9]), "=f"(s_b[10]), "=f"(s_b[11]), "=f"(s_b[12]), "=f"(s_b[13]), "=f"(s_b[14]), "=f"(s_b[15]), "=f"(s_b[16]), "=f"(s_b[17]), "=f"(s_b[18]), "=f"(s_b[19]), "=f"(s_b[20]), "=f"(s_b[21]), "=f"(s_b[22]), "=f"(s_b[23]), "=f"(s_b[24]), "=f"(s_b[25]), "=f"(s_b[26]), "=f"(s_b[27]), "=f"(s_b[28]), "=f"(s_b[29]), "=f"(s_b[30]), "=f"(s_b[31])
                                : "r"(s_tmem_row + look_t_6 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 8; q_1++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_6 * 8192 + s_work_row + 4 * q_1);
                                    s_b[4 * q_1 + 0] = _v4.x;
                                    s_b[4 * q_1 + 1] = _v4.y;
                                    s_b[4 * q_1 + 2] = _v4.z;
                                    s_b[4 * q_1 + 3] = _v4.w;
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int c = agg_first; c < agg_last + 1; c += 2) {
                            float agg_decay_a = d_ring[0];
                            lsum = lsum + lc_ring[0];
                            const float2 _scale2_6 = {agg_decay_a, agg_decay_a};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_6);
                            #pragma unroll
                            for (int k_1 = 0; k_1 < 16; k_1++) {
                                h[2 * k_1] = h[2 * k_1] + s_a[2 * k_1];
                                h[2 * k_1 + 1] = h[2 * k_1 + 1] + s_a[2 * k_1 + 1];
                            }
                            int agg_look_a = ((c + 2) < (agg_last) ? (c + 2) : (agg_last));
                            int look_tile_7 = agg_look_a * nheads + l_head;
                            int look_physical_7 = 0;
                            int look_offset_7 = 0;
                            int look_limit_7 = 128;
                            look_physical_7 = agg_look_a - l_sequence * nchunks;
                            int look_tokens_7 = seqlen - look_physical_7 * 128;
                            if (look_tokens_7 > 128) {
                                look_tokens_7 = 128;
                            }
                            int source_token_7 = look_tokens_7 - 1;
                            float last_cumsum_7 = cumsum_precomputed[look_tile_7 * 128 + source_token_7];
                            lc_ring[0] = last_cumsum_7;
                            float segment_base_7 = 0.0f;
                            float _exp2_7 = approx_exp2((last_cumsum_7 - segment_base_7) * 1.442695f);
                            d_ring[0] = _exp2_7;
                            int look_t_7 = agg_look_a + s_t_off;
                            if (look_t_7 >= s_resident_from) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(s_a[0]), "=f"(s_a[1]), "=f"(s_a[2]), "=f"(s_a[3]), "=f"(s_a[4]), "=f"(s_a[5]), "=f"(s_a[6]), "=f"(s_a[7]), "=f"(s_a[8]), "=f"(s_a[9]), "=f"(s_a[10]), "=f"(s_a[11]), "=f"(s_a[12]), "=f"(s_a[13]), "=f"(s_a[14]), "=f"(s_a[15]), "=f"(s_a[16]), "=f"(s_a[17]), "=f"(s_a[18]), "=f"(s_a[19]), "=f"(s_a[20]), "=f"(s_a[21]), "=f"(s_a[22]), "=f"(s_a[23]), "=f"(s_a[24]), "=f"(s_a[25]), "=f"(s_a[26]), "=f"(s_a[27]), "=f"(s_a[28]), "=f"(s_a[29]), "=f"(s_a[30]), "=f"(s_a[31])
                                    : "r"(s_tmem_row + look_t_7 % 8 * 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                            } else {
                                #pragma unroll
                                for (int q_2 = 0; q_2 < 8; q_2++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_7 * 8192 + s_work_row + 4 * q_2);
                                        s_a[4 * q_2 + 0] = _v4.x;
                                        s_a[4 * q_2 + 1] = _v4.y;
                                        s_a[4 * q_2 + 2] = _v4.z;
                                        s_a[4 * q_2 + 3] = _v4.w;
                                    }
                                }
                            }
                            int agg_current_b = c + 1;
                            if (agg_current_b <= agg_last) {
                                float agg_decay_b = d_ring[1];
                                lsum = lsum + lc_ring[1];
                                const float2 _scale2_9 = {agg_decay_b, agg_decay_b};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_9);
                                #pragma unroll
                                for (int k_2 = 0; k_2 < 16; k_2++) {
                                    h[2 * k_2] = h[2 * k_2] + s_b[2 * k_2];
                                    h[2 * k_2 + 1] = h[2 * k_2 + 1] + s_b[2 * k_2 + 1];
                                }
                                int agg_look_b = ((c + 3) < (agg_last) ? (c + 3) : (agg_last));
                                int look_tile_8 = agg_look_b * nheads + l_head;
                                int look_physical_8 = 0;
                                int look_offset_8 = 0;
                                int look_limit_8 = 128;
                                look_physical_8 = agg_look_b - l_sequence * nchunks;
                                int look_tokens_8 = seqlen - look_physical_8 * 128;
                                if (look_tokens_8 > 128) {
                                    look_tokens_8 = 128;
                                }
                                int source_token_8 = look_tokens_8 - 1;
                                float last_cumsum_8 = cumsum_precomputed[look_tile_8 * 128 + source_token_8];
                                lc_ring[1] = last_cumsum_8;
                                float segment_base_8 = 0.0f;
                                float _exp2_10 = approx_exp2((last_cumsum_8 - segment_base_8) * 1.442695f);
                                d_ring[1] = _exp2_10;
                                int look_t_8 = agg_look_b + s_t_off;
                                if (look_t_8 >= s_resident_from) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(s_b[0]), "=f"(s_b[1]), "=f"(s_b[2]), "=f"(s_b[3]), "=f"(s_b[4]), "=f"(s_b[5]), "=f"(s_b[6]), "=f"(s_b[7]), "=f"(s_b[8]), "=f"(s_b[9]), "=f"(s_b[10]), "=f"(s_b[11]), "=f"(s_b[12]), "=f"(s_b[13]), "=f"(s_b[14]), "=f"(s_b[15]), "=f"(s_b[16]), "=f"(s_b[17]), "=f"(s_b[18]), "=f"(s_b[19]), "=f"(s_b[20]), "=f"(s_b[21]), "=f"(s_b[22]), "=f"(s_b[23]), "=f"(s_b[24]), "=f"(s_b[25]), "=f"(s_b[26]), "=f"(s_b[27]), "=f"(s_b[28]), "=f"(s_b[29]), "=f"(s_b[30]), "=f"(s_b[31])
                                        : "r"(s_tmem_row + look_t_8 % 8 * 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int q_3 = 0; q_3 < 8; q_3++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_8 * 8192 + s_work_row + 4 * q_3);
                                            s_b[4 * q_3 + 0] = _v4.x;
                                            s_b[4 * q_3 + 1] = _v4.y;
                                            s_b[4 * q_3 + 2] = _v4.z;
                                            s_b[4 * q_3 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                        int agg_slot = ticket_4 * 8192;
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            {
                                float4 _v4 = make_float4(h[4 * j + 0], h[4 * j + 1], h[4 * j + 2], h[4 * j + 3]);
                                *reinterpret_cast<float4*>(agg_work + agg_slot + 1024 * j + handoff_lane) = _v4;
                            }
                        }
                        if (chain_tid == 0) {
                            agg_work[vcount_eff_4 * 8192 + ticket_4] = lsum;
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __threadfence();
                        asm volatile("barrier.sync 12, 256;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (4 + ticket_4))), "r"(static_cast<unsigned int>(epoch + 1)) : "memory");
                            }
                        }
                        #pragma unroll
                        for (int k_3 = 0; k_3 < 32; k_3++) {
                            h[k_3] = 0.0f;
                        }
                    }
                    int first_loc[6];
                    int sequence_3_1 = 0;
                    int s_base_2_1 = 0;
                    int n_item_2_1 = nchunks;
                    sequence_3_1 = a_b_4 / (nchunks * nheads);
                    s_base_2_1 = sequence_3_1 * nchunks;
                    int rem_2_1 = a_b_4 - s_base_2_1 * nheads;
                    int head_3_1 = rem_2_1 / n_item_2_1;
                    int chunk_2_1 = rem_2_1 - head_3_1 * n_item_2_1;
                    first_loc[0] = sequence_3_1;
                    first_loc[1] = head_3_1;
                    first_loc[2] = chunk_2_1;
                    first_loc[3] = n_item_2_1;
                    first_loc[4] = s_base_2_1 + chunk_2_1;
                    first_loc[5] = a_b_4 - chunk_2_1;
                    int f_sequence = first_loc[0];
                    int f_head = first_loc[1];
                    int f_chunk = first_loc[2];
                    int f_item_x = first_loc[5];
                    if (has_initial != 0) {
                        int f_state_base = (f_sequence * nheads + f_head) * 8192;
                        #pragma unroll
                        for (int k_4 = 0; k_4 < 32; k_4++) {
                            h[k_4] = initial_states[f_state_base + elem_base + k_4 * 128];
                        }
                    }
                    if (f_chunk > 0) {
                        int b_lo = f_item_x * vcount_eff_4 / total_1_1;
                        if (warp == 4) {
                            #pragma unroll 1
                            for (int bq0 = b_lo; bq0 < ticket_4; bq0 += 32) {
                                int bq = bq0 + lane;
                                int bq_q = ticket_4;
                                if (bq < ticket_4) {
                                    bq_q = bq;
                                }
                                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                                {
                                bool _warp_acquire_ready;
                                do {
                                _warp_acquire_ready = true;
                                if (bq < ticket_4) {
                                unsigned int _warp_acquire_observed;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_warp_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(grid_barrier) + (4 + bq_q))) : "memory");
                                _warp_acquire_ready = _warp_acquire_observed == static_cast<unsigned int>(epoch + 1);
                                }
                                } while (!__all_sync(static_cast<unsigned>(4294967295), _warp_acquire_ready));
                                }
                                #elif defined(__CUDA_ARCH__)
                                #error "GlobalWarpWaitAcquire requires SM70 or newer"
                                #endif
                            }
                        }
                        asm volatile("barrier.sync 12, 256;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float lsum_ring[2];
                        float lsum_bq = agg_work[vcount_eff_4 * 8192 + b_lo];
                        lsum_ring[0] = lsum_bq;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 8; j_1++) {
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(agg_work + b_lo * 8192 + 1024 * j_1 + handoff_lane);
                                s_a[4 * j_1 + 0] = _v4.x;
                                s_a[4 * j_1 + 1] = _v4.y;
                                s_a[4 * j_1 + 2] = _v4.z;
                                s_a[4 * j_1 + 3] = _v4.w;
                            }
                        }
                        int bq_pre = b_lo + 1;
                        if (bq_pre < ticket_4) {
                            float lsum_bq_3 = agg_work[vcount_eff_4 * 8192 + bq_pre];
                            lsum_ring[1] = lsum_bq_3;
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 8; j_2++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_pre * 8192 + 1024 * j_2 + handoff_lane);
                                    s_b[4 * j_2 + 0] = _v4.x;
                                    s_b[4 * j_2 + 1] = _v4.y;
                                    s_b[4 * j_2 + 2] = _v4.z;
                                    s_b[4 * j_2 + 3] = _v4.w;
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int bq_1 = b_lo; bq_1 < ticket_4; bq_1 += 2) {
                            float lsum_a = lsum_ring[0];
                            float y = lsum_a * 1.442695f;
                            float a_lo = 0.0f;
                            float a_hi = 1.0f;
                            if (y > -100.0f) {
                                float _exp2_14 = approx_exp2(y);
                                a_lo = _exp2_14;
                            } else {
                                float _exp2_15 = approx_exp2(y + 100.0f);
                                a_lo = _exp2_15;
                                a_hi = 7.888609e-31f;
                            }
                            const float2 _scale2_16 = {a_lo, a_lo};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_16);
                            const float2 _scale2_17 = {a_hi, a_hi};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_17);
                            #pragma unroll
                            for (int k_5 = 0; k_5 < 16; k_5++) {
                                h[2 * k_5] = h[2 * k_5] + s_a[2 * k_5];
                                h[2 * k_5 + 1] = h[2 * k_5 + 1] + s_a[2 * k_5 + 1];
                            }
                            int bq_a = bq_1 + 2;
                            if (bq_a < ticket_4) {
                                float lsum_bq_4 = agg_work[vcount_eff_4 * 8192 + bq_a];
                                lsum_ring[0] = lsum_bq_4;
                                #pragma unroll
                                for (int j_3 = 0; j_3 < 8; j_3++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_a * 8192 + 1024 * j_3 + handoff_lane);
                                        s_a[4 * j_3 + 0] = _v4.x;
                                        s_a[4 * j_3 + 1] = _v4.y;
                                        s_a[4 * j_3 + 2] = _v4.z;
                                        s_a[4 * j_3 + 3] = _v4.w;
                                    }
                                }
                            }
                            int bq_b = bq_1 + 1;
                            if (bq_b < ticket_4) {
                                float lsum_b = lsum_ring[1];
                                float y_1 = lsum_b * 1.442695f;
                                float a_lo_1 = 0.0f;
                                float a_hi_1 = 1.0f;
                                if (y_1 > -100.0f) {
                                    float _exp2_19 = approx_exp2(y_1);
                                    a_lo_1 = _exp2_19;
                                } else {
                                    float _exp2_20 = approx_exp2(y_1 + 100.0f);
                                    a_lo_1 = _exp2_20;
                                    a_hi_1 = 7.888609e-31f;
                                }
                                const float2 _scale2_21 = {a_lo_1, a_lo_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_21);
                                const float2 _scale2_22 = {a_hi_1, a_hi_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_22);
                                #pragma unroll
                                for (int k_6 = 0; k_6 < 16; k_6++) {
                                    h[2 * k_6] = h[2 * k_6] + s_b[2 * k_6];
                                    h[2 * k_6 + 1] = h[2 * k_6 + 1] + s_b[2 * k_6 + 1];
                                }
                                int bq_b2 = bq_1 + 3;
                                if (bq_b2 < ticket_4) {
                                    float lsum_bq_5 = agg_work[vcount_eff_4 * 8192 + bq_b2];
                                    lsum_ring[1] = lsum_bq_5;
                                    #pragma unroll
                                    for (int j_4 = 0; j_4 < 8; j_4++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_b2 * 8192 + 1024 * j_4 + handoff_lane);
                                            s_b[4 * j_4 + 0] = _v4.x;
                                            s_b[4 * j_4 + 1] = _v4.y;
                                            s_b[4 * j_4 + 2] = _v4.z;
                                            s_b[4 * j_4 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int x_6 = a_b_4;
                    while (x_6 < a_end_4) {
                        int loc[6];
                        int sequence_4_1 = 0;
                        int s_base_3_1 = 0;
                        int n_item_3_1 = nchunks;
                        sequence_4_1 = x_6 / (nchunks * nheads);
                        s_base_3_1 = sequence_4_1 * nchunks;
                        int rem_3_1 = x_6 - s_base_3_1 * nheads;
                        int head_4_1 = rem_3_1 / n_item_3_1;
                        int chunk_3_1 = rem_3_1 - head_4_1 * n_item_3_1;
                        loc[0] = sequence_4_1;
                        loc[1] = head_4_1;
                        loc[2] = chunk_3_1;
                        loc[3] = n_item_3_1;
                        loc[4] = s_base_3_1 + chunk_3_1;
                        loc[5] = x_6 - chunk_3_1;
                        int sequence_7_1 = loc[0];
                        int head_8_1 = loc[1];
                        int chunk0 = loc[2];
                        int n_item_9 = loc[3];
                        int first_8 = loc[4];
                        int seg_len_8 = n_item_9 - chunk0;
                        if (seg_len_8 > a_end_4 - x_6) {
                            seg_len_8 = a_end_4 - x_6;
                        }
                        int last = first_8 + seg_len_8 - 1;
                        int item_end = first_8 - chunk0 + n_item_9;
                        int state_head_base = (sequence_7_1 * nheads + head_8_1) * 8192;
                        int checkpoint_token = -1;
                        if (checkpoint_state_count > 0) {
                            checkpoint_token = checkpoint_token_indices[sequence_7_1];
                        }
                        s_t_off = x_6 - a_b_4 - first_8;
                        if (x_6 > a_b_4) {
                            #pragma unroll
                            for (int k_7 = 0; k_7 < 32; k_7++) {
                                h[k_7] = 0.0f;
                            }
                            if (has_initial != 0) {
                                #pragma unroll
                                for (int k_8 = 0; k_8 < 32; k_8++) {
                                    h[k_8] = initial_states[state_head_base + elem_base + k_8 * 128];
                                }
                            }
                        }
                        int look_tile_1 = first_8 * nheads + head_8_1;
                        int look_physical_1 = 0;
                        int look_offset_1 = 0;
                        int look_limit_1 = 128;
                        look_physical_1 = first_8 - sequence_7_1 * nchunks;
                        int look_tokens_1 = seqlen - look_physical_1 * 128;
                        if (look_tokens_1 > 128) {
                            look_tokens_1 = 128;
                        }
                        int source_token_3 = look_tokens_1 - 1;
                        float last_cumsum_2 = cumsum_precomputed[look_tile_1 * 128 + source_token_3];
                        lc_ring[0] = last_cumsum_2;
                        float segment_base_1 = 0.0f;
                        float _exp2_24 = approx_exp2((last_cumsum_2 - segment_base_1) * 1.442695f);
                        d_ring[0] = _exp2_24;
                        int look_t_1 = first_8 + s_t_off;
                        if (look_t_1 >= s_resident_from) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_a[0]), "=f"(s_a[1]), "=f"(s_a[2]), "=f"(s_a[3]), "=f"(s_a[4]), "=f"(s_a[5]), "=f"(s_a[6]), "=f"(s_a[7]), "=f"(s_a[8]), "=f"(s_a[9]), "=f"(s_a[10]), "=f"(s_a[11]), "=f"(s_a[12]), "=f"(s_a[13]), "=f"(s_a[14]), "=f"(s_a[15]), "=f"(s_a[16]), "=f"(s_a[17]), "=f"(s_a[18]), "=f"(s_a[19]), "=f"(s_a[20]), "=f"(s_a[21]), "=f"(s_a[22]), "=f"(s_a[23]), "=f"(s_a[24]), "=f"(s_a[25]), "=f"(s_a[26]), "=f"(s_a[27]), "=f"(s_a[28]), "=f"(s_a[29]), "=f"(s_a[30]), "=f"(s_a[31])
                                : "r"(s_tmem_row + look_t_1 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_4 = 0; q_4 < 8; q_4++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_1 * 8192 + s_work_row + 4 * q_4);
                                    s_a[4 * q_4 + 0] = _v4.x;
                                    s_a[4 * q_4 + 1] = _v4.y;
                                    s_a[4 * q_4 + 2] = _v4.z;
                                    s_a[4 * q_4 + 3] = _v4.w;
                                }
                            }
                        }
                        int look1 = ((first_8 + 1) < (last) ? (first_8 + 1) : (last));
                        int look_tile_9 = look1 * nheads + head_8_1;
                        int look_physical_9 = 0;
                        int look_offset_9 = 0;
                        int look_limit_9 = 128;
                        look_physical_9 = look1 - sequence_7_1 * nchunks;
                        int look_tokens_9 = seqlen - look_physical_9 * 128;
                        if (look_tokens_9 > 128) {
                            look_tokens_9 = 128;
                        }
                        int source_token_9 = look_tokens_9 - 1;
                        float last_cumsum_9 = cumsum_precomputed[look_tile_9 * 128 + source_token_9];
                        lc_ring[1] = last_cumsum_9;
                        float segment_base_9 = 0.0f;
                        float _exp2_26 = approx_exp2((last_cumsum_9 - segment_base_9) * 1.442695f);
                        d_ring[1] = _exp2_26;
                        int look_t_9 = look1 + s_t_off;
                        if (look_t_9 >= s_resident_from) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_b[0]), "=f"(s_b[1]), "=f"(s_b[2]), "=f"(s_b[3]), "=f"(s_b[4]), "=f"(s_b[5]), "=f"(s_b[6]), "=f"(s_b[7]), "=f"(s_b[8]), "=f"(s_b[9]), "=f"(s_b[10]), "=f"(s_b[11]), "=f"(s_b[12]), "=f"(s_b[13]), "=f"(s_b[14]), "=f"(s_b[15]), "=f"(s_b[16]), "=f"(s_b[17]), "=f"(s_b[18]), "=f"(s_b[19]), "=f"(s_b[20]), "=f"(s_b[21]), "=f"(s_b[22]), "=f"(s_b[23]), "=f"(s_b[24]), "=f"(s_b[25]), "=f"(s_b[26]), "=f"(s_b[27]), "=f"(s_b[28]), "=f"(s_b[29]), "=f"(s_b[30]), "=f"(s_b[31])
                                : "r"(s_tmem_row + look_t_9 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_5 = 0; q_5 < 8; q_5++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_9 * 8192 + s_work_row + 4 * q_5);
                                    s_b[4 * q_5 + 0] = _v4.x;
                                    s_b[4 * q_5 + 1] = _v4.y;
                                    s_b[4 * q_5 + 2] = _v4.z;
                                    s_b[4 * q_5 + 3] = _v4.w;
                                }
                            }
                        }
                        unsigned int h_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h[_lp*2 + 0], h[_lp*2+1 + 0]));
                            h_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            unsigned int sent = h_bf16[i] >> h_half_shift & 65535;
                            unsigned int _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, sent, 1);
                            h_words[(first_8 * nheads + head_8_1) * 4096 + pair_base + i * 128] = h_bf16[i] & h_mask_mine | _shfl_xor_sync_28 << h_half_shift;
                        }
                        #pragma unroll 1
                        for (int c_1 = first_8; c_1 < first_8 + seg_len_8; c_1 += 2) {
                            float decay_a = d_ring[0];
                            const float2 _scale2_29 = {decay_a, decay_a};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_29);
                            #pragma unroll
                            for (int k_9 = 0; k_9 < 16; k_9++) {
                                h[2 * k_9] = h[2 * k_9] + s_a[2 * k_9];
                                h[2 * k_9 + 1] = h[2 * k_9 + 1] + s_a[2 * k_9 + 1];
                            }
                            if (last > c_1) {
                                unsigned int h_bf16_2[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h[_lp*2 + 0], h[_lp*2+1 + 0]));
                                    h_bf16_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int i_1 = 0; i_1 < 16; i_1++) {
                                    unsigned int sent_1 = h_bf16_2[i_1] >> h_half_shift & 65535;
                                    unsigned int _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, sent_1, 1);
                                    h_words[((c_1 + 1) * nheads + head_8_1) * 4096 + pair_base + i_1 * 128] = h_bf16_2[i_1] & h_mask_mine | _shfl_xor_sync_30 << h_half_shift;
                                }
                            }
                            if (checkpoint_state_count > 0) {
                                int cur_physical = 0;
                                int cur_limit = 128;
                                cur_physical = c_1 - sequence_7_1 * nchunks;
                                int cur_tokens = seqlen - cur_physical * 128;
                                if (cur_tokens > 128) {
                                    cur_tokens = 128;
                                }
                                int segment_end = cur_physical * 128 + cur_limit;
                                if (segment_end > seqlen) {
                                    segment_end = seqlen;
                                }
                                if (checkpoint_token == segment_end) {
                                    int checkpoint_slot = checkpoint_state_slots[sequence_7_1];
                                    if (checkpoint_slot >= 0 && checkpoint_slot < checkpoint_state_count) {
                                        int checkpoint_head_base = (checkpoint_slot * nheads + head_8_1) * 8192;
                                        #pragma unroll
                                        for (int k_10 = 0; k_10 < 32; k_10++) {
                                            checkpoint_states[checkpoint_head_base + elem_base + k_10 * 128] = h[k_10];
                                        }
                                    }
                                }
                            }
                            int look_a = ((c_1 + 2) < (last) ? (c_1 + 2) : (last));
                            int look_tile_10 = look_a * nheads + head_8_1;
                            int look_physical_10 = 0;
                            int look_offset_10 = 0;
                            int look_limit_10 = 128;
                            look_physical_10 = look_a - sequence_7_1 * nchunks;
                            int look_tokens_10 = seqlen - look_physical_10 * 128;
                            if (look_tokens_10 > 128) {
                                look_tokens_10 = 128;
                            }
                            int source_token_10 = look_tokens_10 - 1;
                            float last_cumsum_10 = cumsum_precomputed[look_tile_10 * 128 + source_token_10];
                            lc_ring[0] = last_cumsum_10;
                            float segment_base_10 = 0.0f;
                            float _exp2_31 = approx_exp2((last_cumsum_10 - segment_base_10) * 1.442695f);
                            d_ring[0] = _exp2_31;
                            int look_t_10 = look_a + s_t_off;
                            if (look_t_10 >= s_resident_from) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(s_a[0]), "=f"(s_a[1]), "=f"(s_a[2]), "=f"(s_a[3]), "=f"(s_a[4]), "=f"(s_a[5]), "=f"(s_a[6]), "=f"(s_a[7]), "=f"(s_a[8]), "=f"(s_a[9]), "=f"(s_a[10]), "=f"(s_a[11]), "=f"(s_a[12]), "=f"(s_a[13]), "=f"(s_a[14]), "=f"(s_a[15]), "=f"(s_a[16]), "=f"(s_a[17]), "=f"(s_a[18]), "=f"(s_a[19]), "=f"(s_a[20]), "=f"(s_a[21]), "=f"(s_a[22]), "=f"(s_a[23]), "=f"(s_a[24]), "=f"(s_a[25]), "=f"(s_a[26]), "=f"(s_a[27]), "=f"(s_a[28]), "=f"(s_a[29]), "=f"(s_a[30]), "=f"(s_a[31])
                                    : "r"(s_tmem_row + look_t_10 % 8 * 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                            } else {
                                #pragma unroll
                                for (int q_6 = 0; q_6 < 8; q_6++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_10 * 8192 + s_work_row + 4 * q_6);
                                        s_a[4 * q_6 + 0] = _v4.x;
                                        s_a[4 * q_6 + 1] = _v4.y;
                                        s_a[4 * q_6 + 2] = _v4.z;
                                        s_a[4 * q_6 + 3] = _v4.w;
                                    }
                                }
                            }
                            int current_b = c_1 + 1;
                            if (current_b <= last) {
                                float decay_b = d_ring[1];
                                const float2 _scale2_33 = {decay_b, decay_b};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h)[_ls], _scale2_33);
                                #pragma unroll
                                for (int k_11 = 0; k_11 < 16; k_11++) {
                                    h[2 * k_11] = h[2 * k_11] + s_b[2 * k_11];
                                    h[2 * k_11 + 1] = h[2 * k_11 + 1] + s_b[2 * k_11 + 1];
                                }
                                if (current_b < last) {
                                    unsigned int h_bf16_3[16];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 16; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h[_lp*2 + 0], h[_lp*2+1 + 0]));
                                        h_bf16_3[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int i_2 = 0; i_2 < 16; i_2++) {
                                        unsigned int sent_2 = h_bf16_3[i_2] >> h_half_shift & 65535;
                                        unsigned int _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, sent_2, 1);
                                        h_words[((current_b + 1) * nheads + head_8_1) * 4096 + pair_base + i_2 * 128] = h_bf16_3[i_2] & h_mask_mine | _shfl_xor_sync_34 << h_half_shift;
                                    }
                                }
                                if (checkpoint_state_count > 0) {
                                    int cur_physical_1 = 0;
                                    int cur_limit_1 = 128;
                                    cur_physical_1 = current_b - sequence_7_1 * nchunks;
                                    int cur_tokens_1 = seqlen - cur_physical_1 * 128;
                                    if (cur_tokens_1 > 128) {
                                        cur_tokens_1 = 128;
                                    }
                                    int segment_end_1 = cur_physical_1 * 128 + cur_limit_1;
                                    if (segment_end_1 > seqlen) {
                                        segment_end_1 = seqlen;
                                    }
                                    if (checkpoint_token == segment_end_1) {
                                        int checkpoint_slot_1 = checkpoint_state_slots[sequence_7_1];
                                        if (checkpoint_slot_1 >= 0 && checkpoint_slot_1 < checkpoint_state_count) {
                                            int checkpoint_head_base_1 = (checkpoint_slot_1 * nheads + head_8_1) * 8192;
                                            #pragma unroll
                                            for (int k_12 = 0; k_12 < 32; k_12++) {
                                                checkpoint_states[checkpoint_head_base_1 + elem_base + k_12 * 128] = h[k_12];
                                            }
                                        }
                                    }
                                }
                                int look_b = ((c_1 + 3) < (last) ? (c_1 + 3) : (last));
                                int look_tile_11 = look_b * nheads + head_8_1;
                                int look_physical_11 = 0;
                                int look_offset_11 = 0;
                                int look_limit_11 = 128;
                                look_physical_11 = look_b - sequence_7_1 * nchunks;
                                int look_tokens_11 = seqlen - look_physical_11 * 128;
                                if (look_tokens_11 > 128) {
                                    look_tokens_11 = 128;
                                }
                                int source_token_11 = look_tokens_11 - 1;
                                float last_cumsum_11 = cumsum_precomputed[look_tile_11 * 128 + source_token_11];
                                lc_ring[1] = last_cumsum_11;
                                float segment_base_11 = 0.0f;
                                float _exp2_35 = approx_exp2((last_cumsum_11 - segment_base_11) * 1.442695f);
                                d_ring[1] = _exp2_35;
                                int look_t_11 = look_b + s_t_off;
                                if (look_t_11 >= s_resident_from) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(s_b[0]), "=f"(s_b[1]), "=f"(s_b[2]), "=f"(s_b[3]), "=f"(s_b[4]), "=f"(s_b[5]), "=f"(s_b[6]), "=f"(s_b[7]), "=f"(s_b[8]), "=f"(s_b[9]), "=f"(s_b[10]), "=f"(s_b[11]), "=f"(s_b[12]), "=f"(s_b[13]), "=f"(s_b[14]), "=f"(s_b[15]), "=f"(s_b[16]), "=f"(s_b[17]), "=f"(s_b[18]), "=f"(s_b[19]), "=f"(s_b[20]), "=f"(s_b[21]), "=f"(s_b[22]), "=f"(s_b[23]), "=f"(s_b[24]), "=f"(s_b[25]), "=f"(s_b[26]), "=f"(s_b[27]), "=f"(s_b[28]), "=f"(s_b[29]), "=f"(s_b[30]), "=f"(s_b[31])
                                        : "r"(s_tmem_row + look_t_11 % 8 * 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int q_7 = 0; q_7 < 8; q_7++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_11 * 8192 + s_work_row + 4 * q_7);
                                            s_b[4 * q_7 + 0] = _v4.x;
                                            s_b[4 * q_7 + 1] = _v4.y;
                                            s_b[4 * q_7 + 2] = _v4.z;
                                            s_b[4 * q_7 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                        if (first_8 + seg_len_8 == item_end) {
                            if (write_final_states != 0) {
                                #pragma unroll
                                for (int k_13 = 0; k_13 < 32; k_13++) {
                                    final_states[state_head_base + elem_base + k_13 * 128] = h[k_13];
                                }
                            }
                        }
                        x_6 = x_6 + seg_len_8;
                    }
                }
                __threadfence();
                asm volatile("fence.proxy.async;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
            }
        }
    }
    // ---- Role: pre_intra ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // pre_intra_main
            int taddr_0_1 = taddr;
            unsigned int stage1_4 = 0;
            unsigned int stage3_4 = 0;
            unsigned int acc_stage_1 = 0;
            int row = warp % 4 * 32 + lane;
            int row_tmem_base = warp % 4 * 32 << 16;
            int chain_tid_1 = (warp - 4) * 32 + lane;
            int tot_5[1];
            tot_5[0] = sequence_count * nchunks * nheads;
            int veff_5[1];
            int total_5 = tot_5[0];
            int g_5 = 1;
            int ranges_5 = (total_5 + g_5 - 1) / g_5;
            if (ranges_5 < 1) {
                ranges_5 = 1;
            }
            if (ranges_5 > vcount) {
                ranges_5 = vcount;
            }
            veff_5[0] = ranges_5;
            int vcount_eff_5 = veff_5[0];
            unsigned int total_u_5 = (unsigned int)vcount_eff_5;
            unsigned int _phase_aux1_full_1 = 0;
            unsigned int _phase_scaled_b_empty_0_1 = 1;
            unsigned int _phase_state_delta_full_0_1 = 0;
            unsigned int _phase_cb_full = 0;
            unsigned int _phase_aux3_full = 0;
            unsigned int _phase_q_empty_0 = 1;
            while (1) {
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_5 = ctrl[0];
                unsigned int epoch_1 = ctrl[1];
                if (ticket_u_5 >= total_u_5) {
                    break;
                }
                int ticket_5 = (int)ticket_u_5;
                int rb_5[2];
                int total_0_5 = tot_5[0];
                rb_5[0] = (ticket_5 * total_0_5 + vcount_eff_5 - 1) / vcount_eff_5;
                rb_5[1] = ((ticket_5 + 1) * total_0_5 + vcount_eff_5 - 1) / vcount_eff_5;
                int a_b_5 = rb_5[0];
                int a_end_5 = rb_5[1];
                int _uniform_29 = make_warp_uniform(a_b_5);
                a_b_5 = _uniform_29;
                int _uniform_30 = make_warp_uniform(a_end_5);
                a_end_5 = _uniform_30;
                int xa_5 = a_b_5;
                while (xa_5 < a_end_5) {
                    int loca_5[6];
                    int sequence_9 = 0;
                    int s_base_9 = 0;
                    int n_item_10 = nchunks;
                    sequence_9 = xa_5 / (nchunks * nheads);
                    s_base_9 = sequence_9 * nchunks;
                    int rem_9 = xa_5 - s_base_9 * nheads;
                    int head_9 = rem_9 / n_item_10;
                    int chunk_9 = rem_9 - head_9 * n_item_10;
                    loca_5[0] = sequence_9;
                    loca_5[1] = head_9;
                    loca_5[2] = chunk_9;
                    loca_5[3] = n_item_10;
                    loca_5[4] = s_base_9 + chunk_9;
                    loca_5[5] = xa_5 - chunk_9;
                    int head_0_8 = loca_5[1];
                    int first_9 = loca_5[4];
                    int seg_len_9 = loca_5[3] - loca_5[2];
                    if (seg_len_9 > a_end_5 - xa_5) {
                        seg_len_9 = a_end_5 - xa_5;
                    }
                    int _uniform_31 = make_warp_uniform(head_0_8);
                    head_0_8 = _uniform_31;
                    int _uniform_32 = make_warp_uniform(first_9);
                    first_9 = _uniform_32;
                    int _uniform_33 = make_warp_uniform(seg_len_9);
                    seg_len_9 = _uniform_33;
                    int b_state_base_1 = (warp % 4 * 4 + lane / 8) * 8;
                    int b_col_lane_1 = lane % 8;
                    #pragma unroll 1
                    for (int logical_8 = first_9; logical_8 < first_9 + seg_len_9; logical_8++) {
                        int tile_row_4 = logical_8 * nheads + head_0_8;
                        int physical_chunk_5 = 0;
                        int segment_offset_3 = 0;
                        int segment_limit_3 = 128;
                        int sequence_0_1 = logical_8 / nchunks;
                        physical_chunk_5 = logical_8 - sequence_0_1 * nchunks;
                        int chunk_tokens_3 = seqlen - physical_chunk_5 * 128;
                        if (chunk_tokens_3 > 128) {
                            chunk_tokens_3 = 128;
                        }
                        mbarrier_wait(aux1_full_addr + (stage1_4) * 8, _phase_aux1_full_1);
                        mbarrier_wait(b1_full_addr + (stage1_4) * 8, _phase_aux1_full_1);
                        float last_cumsum_3 = smem_cumsum_all[stage1_4 * 128 + (unsigned int)segment_limit_3 - 1];
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
                                float b_scale_1 = approx_exp2((last_cumsum_3 - smem_cumsum_all[stage1_4 * 128 + (unsigned int)col_1]) * 1.442695f);
                                b_scale_1 *= smem_delta_all[stage1_4 * 128 + (unsigned int)col_1];
                                const float2 _scale2_0 = {b_scale_1, b_scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(scaled_b_values_1)[_ls], _scale2_0);
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
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        mbarrier_arrive(aux1_empty_addr + (stage1_4) * 8);
                        if (warp == 8) {
                            if (elect_sync()) {
                                mbarrier_arrive(scaled_b_full_addr);
                                mbarrier_arrive(b1_empty_addr + (stage1_4) * 8);
                            }
                        }
                        mbarrier_wait(state_delta_full_addr, _phase_state_delta_full_0_1);
                        _phase_state_delta_full_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int t_range_2 = xa_5 - a_b_5 + (logical_8 - first_9);
                        if (t_range_2 + 8 < a_end_5 - a_b_5) {
                            int s_tile_base_1 = tile_row_4 * 8192;
                            int s_slot_col_1 = t_range_2 % 8 * 64;
                            #pragma unroll
                            for (int row_half_1 = 0; row_half_1 < 2; row_half_1++) {
                                int state_row_origin_1 = warp % 4 * 32 + row_half_1 * 16;
                                float _tmem_load_0[16];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                                    : "r"(taddr_0_1 + (state_row_origin_1 << 16) + s_slot_col_1 + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int state_pair_1 = 0; state_pair_1 < 8; state_pair_1++) {
                                    int state_row_1 = state_row_origin_1 + lane / 4 + state_pair_1 % 2 * 8;
                                    int dim_1 = 32 + state_pair_1 / 2 * 8 + lane % 4 * 2;
                                    {
                                        float2 _v2 = make_float2(_tmem_load_0[2 * state_pair_1 + 0], _tmem_load_0[2 * state_pair_1 + 1]);
                                        *reinterpret_cast<float2*>(s_work + s_tile_base_1 + state_row_1 * 64 + dim_1) = _v2;
                                    }
                                }
                            }
                        }
                        stage1_4 += 1;
                        if (stage1_4 == 2) { stage1_4 = 0; _phase_aux1_full_1 ^= 1; }
                    }
                    xa_5 = xa_5 + seg_len_9;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int total_1_2 = tot_5[0];
                asm volatile("tcgen05.fence::after_thread_sync;");
                int s_row_1 = chain_tid_1 % 128;
                int dim_base_1 = chain_tid_1 / 128 * 32;
                int elem_base_1 = dim_base_1 * 128 + s_row_1;
                int pair_base_1 = (dim_base_1 + s_row_1 % 2) * 64 + s_row_1 / 2;
                unsigned int h_half_shift_1 = (unsigned int)((1 - s_row_1 % 2) * 16);
                unsigned int h_mask_mine_1 = 65535;
                if (s_row_1 % 2 != 0) {
                    h_mask_mine_1 = 4294901760u;
                }
                int s_tmem_row_1 = taddr_0_1 + (warp % 4 * 32 << 16) + dim_base_1;
                int s_work_row_1 = s_row_1 * 64 + dim_base_1;
                int s_t_off_1 = 0;
                int s_resident_from_1 = a_end_5 - a_b_5 - 8;
                int handoff_lane_1 = chain_tid_1 * 4;
                float h_1[32];
                float s_a_1[32];
                float s_b_1[32];
                float d_ring_1[2];
                float lc_ring_1[2];
                d_ring_1[0] = 0.0f;
                d_ring_1[1] = 0.0f;
                lc_ring_1[0] = 0.0f;
                lc_ring_1[1] = 0.0f;
                #pragma unroll
                for (int k_14 = 0; k_14 < 32; k_14++) {
                    h_1[k_14] = 0.0f;
                }
                if (a_b_5 < a_end_5) {
                    int last_loc_1[6];
                    int sequence_10 = 0;
                    int s_base_10 = 0;
                    int n_item_11 = nchunks;
                    sequence_10 = (a_end_5 - 1) / (nchunks * nheads);
                    s_base_10 = sequence_10 * nchunks;
                    int rem_10 = a_end_5 - 1 - s_base_10 * nheads;
                    int head_10 = rem_10 / n_item_11;
                    int chunk_10 = rem_10 - head_10 * n_item_11;
                    last_loc_1[0] = sequence_10;
                    last_loc_1[1] = head_10;
                    last_loc_1[2] = chunk_10;
                    last_loc_1[3] = n_item_11;
                    last_loc_1[4] = s_base_10 + chunk_10;
                    last_loc_1[5] = a_end_5 - 1 - chunk_10;
                    int l_sequence_1 = last_loc_1[0];
                    int l_head_1 = last_loc_1[1];
                    int l_chunk_1 = last_loc_1[2];
                    int l_n_1 = last_loc_1[3];
                    int l_logical_1 = last_loc_1[4];
                    int l_item_x_1 = last_loc_1[5];
                    if (l_n_1 > l_chunk_1 + 1) {
                        int agg_first_1 = l_logical_1 - l_chunk_1;
                        if (a_b_5 > l_item_x_1) {
                            agg_first_1 = agg_first_1 + (a_b_5 - l_item_x_1);
                        }
                        int agg_last_1 = l_logical_1;
                        float lsum_1 = 0.0f;
                        s_t_off_1 = l_item_x_1 - a_b_5 - (l_logical_1 - l_chunk_1);
                        int look_tile_2 = agg_first_1 * nheads + l_head_1;
                        int look_physical_2 = 0;
                        int look_offset_2 = 0;
                        int look_limit_2 = 128;
                        look_physical_2 = agg_first_1 - l_sequence_1 * nchunks;
                        int look_tokens_2 = seqlen - look_physical_2 * 128;
                        if (look_tokens_2 > 128) {
                            look_tokens_2 = 128;
                        }
                        int source_token_4 = look_tokens_2 - 1;
                        float last_cumsum_4 = cumsum_precomputed[look_tile_2 * 128 + source_token_4];
                        lc_ring_1[0] = last_cumsum_4;
                        float segment_base_2 = 0.0f;
                        float _exp2_2_1 = approx_exp2((last_cumsum_4 - segment_base_2) * 1.442695f);
                        d_ring_1[0] = _exp2_2_1;
                        int look_t_2 = agg_first_1 + s_t_off_1;
                        if (look_t_2 >= s_resident_from_1) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_a_1[0]), "=f"(s_a_1[1]), "=f"(s_a_1[2]), "=f"(s_a_1[3]), "=f"(s_a_1[4]), "=f"(s_a_1[5]), "=f"(s_a_1[6]), "=f"(s_a_1[7]), "=f"(s_a_1[8]), "=f"(s_a_1[9]), "=f"(s_a_1[10]), "=f"(s_a_1[11]), "=f"(s_a_1[12]), "=f"(s_a_1[13]), "=f"(s_a_1[14]), "=f"(s_a_1[15]), "=f"(s_a_1[16]), "=f"(s_a_1[17]), "=f"(s_a_1[18]), "=f"(s_a_1[19]), "=f"(s_a_1[20]), "=f"(s_a_1[21]), "=f"(s_a_1[22]), "=f"(s_a_1[23]), "=f"(s_a_1[24]), "=f"(s_a_1[25]), "=f"(s_a_1[26]), "=f"(s_a_1[27]), "=f"(s_a_1[28]), "=f"(s_a_1[29]), "=f"(s_a_1[30]), "=f"(s_a_1[31])
                                : "r"(s_tmem_row_1 + look_t_2 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_8 = 0; q_8 < 8; q_8++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_2 * 8192 + s_work_row_1 + 4 * q_8);
                                    s_a_1[4 * q_8 + 0] = _v4.x;
                                    s_a_1[4 * q_8 + 1] = _v4.y;
                                    s_a_1[4 * q_8 + 2] = _v4.z;
                                    s_a_1[4 * q_8 + 3] = _v4.w;
                                }
                            }
                        }
                        int agg_look1_1 = ((agg_first_1 + 1) < (agg_last_1) ? (agg_first_1 + 1) : (agg_last_1));
                        int look_tile_0 = agg_look1_1 * nheads + l_head_1;
                        int look_physical_0 = 0;
                        int look_offset_0 = 0;
                        int look_limit_0 = 128;
                        look_physical_0 = agg_look1_1 - l_sequence_1 * nchunks;
                        int look_tokens_0 = seqlen - look_physical_0 * 128;
                        if (look_tokens_0 > 128) {
                            look_tokens_0 = 128;
                        }
                        int source_token_0 = look_tokens_0 - 1;
                        float last_cumsum_0 = cumsum_precomputed[look_tile_0 * 128 + source_token_0];
                        lc_ring_1[1] = last_cumsum_0;
                        float segment_base_0 = 0.0f;
                        float _exp2_4_1 = approx_exp2((last_cumsum_0 - segment_base_0) * 1.442695f);
                        d_ring_1[1] = _exp2_4_1;
                        int look_t_0 = agg_look1_1 + s_t_off_1;
                        if (look_t_0 >= s_resident_from_1) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_b_1[0]), "=f"(s_b_1[1]), "=f"(s_b_1[2]), "=f"(s_b_1[3]), "=f"(s_b_1[4]), "=f"(s_b_1[5]), "=f"(s_b_1[6]), "=f"(s_b_1[7]), "=f"(s_b_1[8]), "=f"(s_b_1[9]), "=f"(s_b_1[10]), "=f"(s_b_1[11]), "=f"(s_b_1[12]), "=f"(s_b_1[13]), "=f"(s_b_1[14]), "=f"(s_b_1[15]), "=f"(s_b_1[16]), "=f"(s_b_1[17]), "=f"(s_b_1[18]), "=f"(s_b_1[19]), "=f"(s_b_1[20]), "=f"(s_b_1[21]), "=f"(s_b_1[22]), "=f"(s_b_1[23]), "=f"(s_b_1[24]), "=f"(s_b_1[25]), "=f"(s_b_1[26]), "=f"(s_b_1[27]), "=f"(s_b_1[28]), "=f"(s_b_1[29]), "=f"(s_b_1[30]), "=f"(s_b_1[31])
                                : "r"(s_tmem_row_1 + look_t_0 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_9 = 0; q_9 < 8; q_9++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_0 * 8192 + s_work_row_1 + 4 * q_9);
                                    s_b_1[4 * q_9 + 0] = _v4.x;
                                    s_b_1[4 * q_9 + 1] = _v4.y;
                                    s_b_1[4 * q_9 + 2] = _v4.z;
                                    s_b_1[4 * q_9 + 3] = _v4.w;
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int c_2 = agg_first_1; c_2 < agg_last_1 + 1; c_2 += 2) {
                            float agg_decay_a_1 = d_ring_1[0];
                            lsum_1 = lsum_1 + lc_ring_1[0];
                            const float2 _scale2_6 = {agg_decay_a_1, agg_decay_a_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_6);
                            #pragma unroll
                            for (int k_15 = 0; k_15 < 16; k_15++) {
                                h_1[2 * k_15] = h_1[2 * k_15] + s_a_1[2 * k_15];
                                h_1[2 * k_15 + 1] = h_1[2 * k_15 + 1] + s_a_1[2 * k_15 + 1];
                            }
                            int agg_look_a_1 = ((c_2 + 2) < (agg_last_1) ? (c_2 + 2) : (agg_last_1));
                            int look_tile_1_1 = agg_look_a_1 * nheads + l_head_1;
                            int look_physical_1_1 = 0;
                            int look_offset_1_1 = 0;
                            int look_limit_1_1 = 128;
                            look_physical_1_1 = agg_look_a_1 - l_sequence_1 * nchunks;
                            int look_tokens_1_1 = seqlen - look_physical_1_1 * 128;
                            if (look_tokens_1_1 > 128) {
                                look_tokens_1_1 = 128;
                            }
                            int source_token_1_1 = look_tokens_1_1 - 1;
                            float last_cumsum_1_1 = cumsum_precomputed[look_tile_1_1 * 128 + source_token_1_1];
                            lc_ring_1[0] = last_cumsum_1_1;
                            float segment_base_1_1 = 0.0f;
                            float _exp2_7_1 = approx_exp2((last_cumsum_1_1 - segment_base_1_1) * 1.442695f);
                            d_ring_1[0] = _exp2_7_1;
                            int look_t_1_1 = agg_look_a_1 + s_t_off_1;
                            if (look_t_1_1 >= s_resident_from_1) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(s_a_1[0]), "=f"(s_a_1[1]), "=f"(s_a_1[2]), "=f"(s_a_1[3]), "=f"(s_a_1[4]), "=f"(s_a_1[5]), "=f"(s_a_1[6]), "=f"(s_a_1[7]), "=f"(s_a_1[8]), "=f"(s_a_1[9]), "=f"(s_a_1[10]), "=f"(s_a_1[11]), "=f"(s_a_1[12]), "=f"(s_a_1[13]), "=f"(s_a_1[14]), "=f"(s_a_1[15]), "=f"(s_a_1[16]), "=f"(s_a_1[17]), "=f"(s_a_1[18]), "=f"(s_a_1[19]), "=f"(s_a_1[20]), "=f"(s_a_1[21]), "=f"(s_a_1[22]), "=f"(s_a_1[23]), "=f"(s_a_1[24]), "=f"(s_a_1[25]), "=f"(s_a_1[26]), "=f"(s_a_1[27]), "=f"(s_a_1[28]), "=f"(s_a_1[29]), "=f"(s_a_1[30]), "=f"(s_a_1[31])
                                    : "r"(s_tmem_row_1 + look_t_1_1 % 8 * 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                            } else {
                                #pragma unroll
                                for (int q_10 = 0; q_10 < 8; q_10++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_1_1 * 8192 + s_work_row_1 + 4 * q_10);
                                        s_a_1[4 * q_10 + 0] = _v4.x;
                                        s_a_1[4 * q_10 + 1] = _v4.y;
                                        s_a_1[4 * q_10 + 2] = _v4.z;
                                        s_a_1[4 * q_10 + 3] = _v4.w;
                                    }
                                }
                            }
                            int agg_current_b_1 = c_2 + 1;
                            if (agg_current_b_1 <= agg_last_1) {
                                float agg_decay_b_1 = d_ring_1[1];
                                lsum_1 = lsum_1 + lc_ring_1[1];
                                const float2 _scale2_9 = {agg_decay_b_1, agg_decay_b_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_9);
                                #pragma unroll
                                for (int k_16 = 0; k_16 < 16; k_16++) {
                                    h_1[2 * k_16] = h_1[2 * k_16] + s_b_1[2 * k_16];
                                    h_1[2 * k_16 + 1] = h_1[2 * k_16 + 1] + s_b_1[2 * k_16 + 1];
                                }
                                int agg_look_b_1 = ((c_2 + 3) < (agg_last_1) ? (c_2 + 3) : (agg_last_1));
                                int look_tile_2_1 = agg_look_b_1 * nheads + l_head_1;
                                int look_physical_2_1 = 0;
                                int look_offset_2_1 = 0;
                                int look_limit_2_1 = 128;
                                look_physical_2_1 = agg_look_b_1 - l_sequence_1 * nchunks;
                                int look_tokens_2_1 = seqlen - look_physical_2_1 * 128;
                                if (look_tokens_2_1 > 128) {
                                    look_tokens_2_1 = 128;
                                }
                                int source_token_2_1 = look_tokens_2_1 - 1;
                                float last_cumsum_2_1 = cumsum_precomputed[look_tile_2_1 * 128 + source_token_2_1];
                                lc_ring_1[1] = last_cumsum_2_1;
                                float segment_base_2_1 = 0.0f;
                                float _exp2_10_1 = approx_exp2((last_cumsum_2_1 - segment_base_2_1) * 1.442695f);
                                d_ring_1[1] = _exp2_10_1;
                                int look_t_2_1 = agg_look_b_1 + s_t_off_1;
                                if (look_t_2_1 >= s_resident_from_1) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(s_b_1[0]), "=f"(s_b_1[1]), "=f"(s_b_1[2]), "=f"(s_b_1[3]), "=f"(s_b_1[4]), "=f"(s_b_1[5]), "=f"(s_b_1[6]), "=f"(s_b_1[7]), "=f"(s_b_1[8]), "=f"(s_b_1[9]), "=f"(s_b_1[10]), "=f"(s_b_1[11]), "=f"(s_b_1[12]), "=f"(s_b_1[13]), "=f"(s_b_1[14]), "=f"(s_b_1[15]), "=f"(s_b_1[16]), "=f"(s_b_1[17]), "=f"(s_b_1[18]), "=f"(s_b_1[19]), "=f"(s_b_1[20]), "=f"(s_b_1[21]), "=f"(s_b_1[22]), "=f"(s_b_1[23]), "=f"(s_b_1[24]), "=f"(s_b_1[25]), "=f"(s_b_1[26]), "=f"(s_b_1[27]), "=f"(s_b_1[28]), "=f"(s_b_1[29]), "=f"(s_b_1[30]), "=f"(s_b_1[31])
                                        : "r"(s_tmem_row_1 + look_t_2_1 % 8 * 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int q_11 = 0; q_11 < 8; q_11++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_2_1 * 8192 + s_work_row_1 + 4 * q_11);
                                            s_b_1[4 * q_11 + 0] = _v4.x;
                                            s_b_1[4 * q_11 + 1] = _v4.y;
                                            s_b_1[4 * q_11 + 2] = _v4.z;
                                            s_b_1[4 * q_11 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                        int agg_slot_1 = ticket_5 * 8192;
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 8; j_5++) {
                            {
                                float4 _v4 = make_float4(h_1[4 * j_5 + 0], h_1[4 * j_5 + 1], h_1[4 * j_5 + 2], h_1[4 * j_5 + 3]);
                                *reinterpret_cast<float4*>(agg_work + agg_slot_1 + 1024 * j_5 + handoff_lane_1) = _v4;
                            }
                        }
                        if (chain_tid_1 == 0) {
                            agg_work[vcount_eff_5 * 8192 + ticket_5] = lsum_1;
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __threadfence();
                        asm volatile("barrier.sync 12, 256;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int k_17 = 0; k_17 < 32; k_17++) {
                            h_1[k_17] = 0.0f;
                        }
                    }
                    int first_loc_1[6];
                    int sequence_1_1 = 0;
                    int s_base_0 = 0;
                    int n_item_0 = nchunks;
                    sequence_1_1 = a_b_5 / (nchunks * nheads);
                    s_base_0 = sequence_1_1 * nchunks;
                    int rem_0 = a_b_5 - s_base_0 * nheads;
                    int head_1_1 = rem_0 / n_item_0;
                    int chunk_0 = rem_0 - head_1_1 * n_item_0;
                    first_loc_1[0] = sequence_1_1;
                    first_loc_1[1] = head_1_1;
                    first_loc_1[2] = chunk_0;
                    first_loc_1[3] = n_item_0;
                    first_loc_1[4] = s_base_0 + chunk_0;
                    first_loc_1[5] = a_b_5 - chunk_0;
                    int f_sequence_1 = first_loc_1[0];
                    int f_head_1 = first_loc_1[1];
                    int f_chunk_1 = first_loc_1[2];
                    int f_item_x_1 = first_loc_1[5];
                    if (has_initial != 0) {
                        int f_state_base_1 = (f_sequence_1 * nheads + f_head_1) * 8192;
                        #pragma unroll
                        for (int k_18 = 0; k_18 < 32; k_18++) {
                            h_1[k_18] = initial_states[f_state_base_1 + elem_base_1 + k_18 * 128];
                        }
                    }
                    if (f_chunk_1 > 0) {
                        int b_lo_1 = f_item_x_1 * vcount_eff_5 / total_1_2;
                        asm volatile("barrier.sync 12, 256;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float lsum_ring_1[2];
                        float lsum_bq_1 = agg_work[vcount_eff_5 * 8192 + b_lo_1];
                        lsum_ring_1[0] = lsum_bq_1;
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(agg_work + b_lo_1 * 8192 + 1024 * j_6 + handoff_lane_1);
                                s_a_1[4 * j_6 + 0] = _v4.x;
                                s_a_1[4 * j_6 + 1] = _v4.y;
                                s_a_1[4 * j_6 + 2] = _v4.z;
                                s_a_1[4 * j_6 + 3] = _v4.w;
                            }
                        }
                        int bq_pre_1 = b_lo_1 + 1;
                        if (bq_pre_1 < ticket_5) {
                            float lsum_bq_0 = agg_work[vcount_eff_5 * 8192 + bq_pre_1];
                            lsum_ring_1[1] = lsum_bq_0;
                            #pragma unroll
                            for (int j_7 = 0; j_7 < 8; j_7++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_pre_1 * 8192 + 1024 * j_7 + handoff_lane_1);
                                    s_b_1[4 * j_7 + 0] = _v4.x;
                                    s_b_1[4 * j_7 + 1] = _v4.y;
                                    s_b_1[4 * j_7 + 2] = _v4.z;
                                    s_b_1[4 * j_7 + 3] = _v4.w;
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int bq_2 = b_lo_1; bq_2 < ticket_5; bq_2 += 2) {
                            float lsum_a_1 = lsum_ring_1[0];
                            float y_2 = lsum_a_1 * 1.442695f;
                            float a_lo_2 = 0.0f;
                            float a_hi_2 = 1.0f;
                            if (y_2 > -100.0f) {
                                float _exp2_14_1 = approx_exp2(y_2);
                                a_lo_2 = _exp2_14_1;
                            } else {
                                float _exp2_15_1 = approx_exp2(y_2 + 100.0f);
                                a_lo_2 = _exp2_15_1;
                                a_hi_2 = 7.888609e-31f;
                            }
                            const float2 _scale2_16 = {a_lo_2, a_lo_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_16);
                            const float2 _scale2_17 = {a_hi_2, a_hi_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_17);
                            #pragma unroll
                            for (int k_19 = 0; k_19 < 16; k_19++) {
                                h_1[2 * k_19] = h_1[2 * k_19] + s_a_1[2 * k_19];
                                h_1[2 * k_19 + 1] = h_1[2 * k_19 + 1] + s_a_1[2 * k_19 + 1];
                            }
                            int bq_a_1 = bq_2 + 2;
                            if (bq_a_1 < ticket_5) {
                                float lsum_bq_1_1 = agg_work[vcount_eff_5 * 8192 + bq_a_1];
                                lsum_ring_1[0] = lsum_bq_1_1;
                                #pragma unroll
                                for (int j_8 = 0; j_8 < 8; j_8++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_a_1 * 8192 + 1024 * j_8 + handoff_lane_1);
                                        s_a_1[4 * j_8 + 0] = _v4.x;
                                        s_a_1[4 * j_8 + 1] = _v4.y;
                                        s_a_1[4 * j_8 + 2] = _v4.z;
                                        s_a_1[4 * j_8 + 3] = _v4.w;
                                    }
                                }
                            }
                            int bq_b_1 = bq_2 + 1;
                            if (bq_b_1 < ticket_5) {
                                float lsum_b_1 = lsum_ring_1[1];
                                float y_0 = lsum_b_1 * 1.442695f;
                                float a_lo_0 = 0.0f;
                                float a_hi_0 = 1.0f;
                                if (y_0 > -100.0f) {
                                    float _exp2_19_1 = approx_exp2(y_0);
                                    a_lo_0 = _exp2_19_1;
                                } else {
                                    float _exp2_20_1 = approx_exp2(y_0 + 100.0f);
                                    a_lo_0 = _exp2_20_1;
                                    a_hi_0 = 7.888609e-31f;
                                }
                                const float2 _scale2_21 = {a_lo_0, a_lo_0};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_21);
                                const float2 _scale2_22 = {a_hi_0, a_hi_0};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_22);
                                #pragma unroll
                                for (int k_20 = 0; k_20 < 16; k_20++) {
                                    h_1[2 * k_20] = h_1[2 * k_20] + s_b_1[2 * k_20];
                                    h_1[2 * k_20 + 1] = h_1[2 * k_20 + 1] + s_b_1[2 * k_20 + 1];
                                }
                                int bq_b2_1 = bq_2 + 3;
                                if (bq_b2_1 < ticket_5) {
                                    float lsum_bq_2 = agg_work[vcount_eff_5 * 8192 + bq_b2_1];
                                    lsum_ring_1[1] = lsum_bq_2;
                                    #pragma unroll
                                    for (int j_9 = 0; j_9 < 8; j_9++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(agg_work + bq_b2_1 * 8192 + 1024 * j_9 + handoff_lane_1);
                                            s_b_1[4 * j_9 + 0] = _v4.x;
                                            s_b_1[4 * j_9 + 1] = _v4.y;
                                            s_b_1[4 * j_9 + 2] = _v4.z;
                                            s_b_1[4 * j_9 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int x_6_1 = a_b_5;
                    while (x_6_1 < a_end_5) {
                        int loc_1[6];
                        int sequence_2_1 = 0;
                        int s_base_1_1 = 0;
                        int n_item_1_1 = nchunks;
                        sequence_2_1 = x_6_1 / (nchunks * nheads);
                        s_base_1_1 = sequence_2_1 * nchunks;
                        int rem_1_1 = x_6_1 - s_base_1_1 * nheads;
                        int head_2_1 = rem_1_1 / n_item_1_1;
                        int chunk_1_1 = rem_1_1 - head_2_1 * n_item_1_1;
                        loc_1[0] = sequence_2_1;
                        loc_1[1] = head_2_1;
                        loc_1[2] = chunk_1_1;
                        loc_1[3] = n_item_1_1;
                        loc_1[4] = s_base_1_1 + chunk_1_1;
                        loc_1[5] = x_6_1 - chunk_1_1;
                        int sequence_7_2 = loc_1[0];
                        int head_8_2 = loc_1[1];
                        int chunk0_1 = loc_1[2];
                        int n_item_9_1 = loc_1[3];
                        int first_10 = loc_1[4];
                        int seg_len_10 = n_item_9_1 - chunk0_1;
                        if (seg_len_10 > a_end_5 - x_6_1) {
                            seg_len_10 = a_end_5 - x_6_1;
                        }
                        int last_1 = first_10 + seg_len_10 - 1;
                        int item_end_1 = first_10 - chunk0_1 + n_item_9_1;
                        int state_head_base_1 = (sequence_7_2 * nheads + head_8_2) * 8192;
                        int checkpoint_token_1 = -1;
                        if (checkpoint_state_count > 0) {
                            checkpoint_token_1 = checkpoint_token_indices[sequence_7_2];
                        }
                        s_t_off_1 = x_6_1 - a_b_5 - first_10;
                        if (x_6_1 > a_b_5) {
                            #pragma unroll
                            for (int k_21 = 0; k_21 < 32; k_21++) {
                                h_1[k_21] = 0.0f;
                            }
                            if (has_initial != 0) {
                                #pragma unroll
                                for (int k_22 = 0; k_22 < 32; k_22++) {
                                    h_1[k_22] = initial_states[state_head_base_1 + elem_base_1 + k_22 * 128];
                                }
                            }
                        }
                        int look_tile_3 = first_10 * nheads + head_8_2;
                        int look_physical_3 = 0;
                        int look_offset_3 = 0;
                        int look_limit_3 = 128;
                        look_physical_3 = first_10 - sequence_7_2 * nchunks;
                        int look_tokens_3 = seqlen - look_physical_3 * 128;
                        if (look_tokens_3 > 128) {
                            look_tokens_3 = 128;
                        }
                        int source_token_5 = look_tokens_3 - 1;
                        float last_cumsum_5 = cumsum_precomputed[look_tile_3 * 128 + source_token_5];
                        lc_ring_1[0] = last_cumsum_5;
                        float segment_base_3 = 0.0f;
                        float _exp2_24_1 = approx_exp2((last_cumsum_5 - segment_base_3) * 1.442695f);
                        d_ring_1[0] = _exp2_24_1;
                        int look_t_3 = first_10 + s_t_off_1;
                        if (look_t_3 >= s_resident_from_1) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_a_1[0]), "=f"(s_a_1[1]), "=f"(s_a_1[2]), "=f"(s_a_1[3]), "=f"(s_a_1[4]), "=f"(s_a_1[5]), "=f"(s_a_1[6]), "=f"(s_a_1[7]), "=f"(s_a_1[8]), "=f"(s_a_1[9]), "=f"(s_a_1[10]), "=f"(s_a_1[11]), "=f"(s_a_1[12]), "=f"(s_a_1[13]), "=f"(s_a_1[14]), "=f"(s_a_1[15]), "=f"(s_a_1[16]), "=f"(s_a_1[17]), "=f"(s_a_1[18]), "=f"(s_a_1[19]), "=f"(s_a_1[20]), "=f"(s_a_1[21]), "=f"(s_a_1[22]), "=f"(s_a_1[23]), "=f"(s_a_1[24]), "=f"(s_a_1[25]), "=f"(s_a_1[26]), "=f"(s_a_1[27]), "=f"(s_a_1[28]), "=f"(s_a_1[29]), "=f"(s_a_1[30]), "=f"(s_a_1[31])
                                : "r"(s_tmem_row_1 + look_t_3 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_12 = 0; q_12 < 8; q_12++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_3 * 8192 + s_work_row_1 + 4 * q_12);
                                    s_a_1[4 * q_12 + 0] = _v4.x;
                                    s_a_1[4 * q_12 + 1] = _v4.y;
                                    s_a_1[4 * q_12 + 2] = _v4.z;
                                    s_a_1[4 * q_12 + 3] = _v4.w;
                                }
                            }
                        }
                        int look1_1 = ((first_10 + 1) < (last_1) ? (first_10 + 1) : (last_1));
                        int look_tile_3_1 = look1_1 * nheads + head_8_2;
                        int look_physical_3_1 = 0;
                        int look_offset_3_1 = 0;
                        int look_limit_3_1 = 128;
                        look_physical_3_1 = look1_1 - sequence_7_2 * nchunks;
                        int look_tokens_3_1 = seqlen - look_physical_3_1 * 128;
                        if (look_tokens_3_1 > 128) {
                            look_tokens_3_1 = 128;
                        }
                        int source_token_3_1 = look_tokens_3_1 - 1;
                        float last_cumsum_3_1 = cumsum_precomputed[look_tile_3_1 * 128 + source_token_3_1];
                        lc_ring_1[1] = last_cumsum_3_1;
                        float segment_base_3_1 = 0.0f;
                        float _exp2_26_1 = approx_exp2((last_cumsum_3_1 - segment_base_3_1) * 1.442695f);
                        d_ring_1[1] = _exp2_26_1;
                        int look_t_3_1 = look1_1 + s_t_off_1;
                        if (look_t_3_1 >= s_resident_from_1) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(s_b_1[0]), "=f"(s_b_1[1]), "=f"(s_b_1[2]), "=f"(s_b_1[3]), "=f"(s_b_1[4]), "=f"(s_b_1[5]), "=f"(s_b_1[6]), "=f"(s_b_1[7]), "=f"(s_b_1[8]), "=f"(s_b_1[9]), "=f"(s_b_1[10]), "=f"(s_b_1[11]), "=f"(s_b_1[12]), "=f"(s_b_1[13]), "=f"(s_b_1[14]), "=f"(s_b_1[15]), "=f"(s_b_1[16]), "=f"(s_b_1[17]), "=f"(s_b_1[18]), "=f"(s_b_1[19]), "=f"(s_b_1[20]), "=f"(s_b_1[21]), "=f"(s_b_1[22]), "=f"(s_b_1[23]), "=f"(s_b_1[24]), "=f"(s_b_1[25]), "=f"(s_b_1[26]), "=f"(s_b_1[27]), "=f"(s_b_1[28]), "=f"(s_b_1[29]), "=f"(s_b_1[30]), "=f"(s_b_1[31])
                                : "r"(s_tmem_row_1 + look_t_3_1 % 8 * 64));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        } else {
                            #pragma unroll
                            for (int q_13 = 0; q_13 < 8; q_13++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_3_1 * 8192 + s_work_row_1 + 4 * q_13);
                                    s_b_1[4 * q_13 + 0] = _v4.x;
                                    s_b_1[4 * q_13 + 1] = _v4.y;
                                    s_b_1[4 * q_13 + 2] = _v4.z;
                                    s_b_1[4 * q_13 + 3] = _v4.w;
                                }
                            }
                        }
                        unsigned int h_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_1[_lp*2 + 0], h_1[_lp*2+1 + 0]));
                            h_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 16; i_3++) {
                            unsigned int sent_3 = h_bf16_1[i_3] >> h_half_shift_1 & 65535;
                            unsigned int _shfl_xor_sync_28_1 = __shfl_xor_sync(0xFFFFFFFF, sent_3, 1);
                            h_words[(first_10 * nheads + head_8_2) * 4096 + pair_base_1 + i_3 * 128] = h_bf16_1[i_3] & h_mask_mine_1 | _shfl_xor_sync_28_1 << h_half_shift_1;
                        }
                        #pragma unroll 1
                        for (int c_3 = first_10; c_3 < first_10 + seg_len_10; c_3 += 2) {
                            float decay_a_1 = d_ring_1[0];
                            const float2 _scale2_29 = {decay_a_1, decay_a_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_29);
                            #pragma unroll
                            for (int k_23 = 0; k_23 < 16; k_23++) {
                                h_1[2 * k_23] = h_1[2 * k_23] + s_a_1[2 * k_23];
                                h_1[2 * k_23 + 1] = h_1[2 * k_23 + 1] + s_a_1[2 * k_23 + 1];
                            }
                            if (last_1 > c_3) {
                                unsigned int h_bf16_0[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_1[_lp*2 + 0], h_1[_lp*2+1 + 0]));
                                    h_bf16_0[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int i_4 = 0; i_4 < 16; i_4++) {
                                    unsigned int sent_4 = h_bf16_0[i_4] >> h_half_shift_1 & 65535;
                                    unsigned int _shfl_xor_sync_30_1 = __shfl_xor_sync(0xFFFFFFFF, sent_4, 1);
                                    h_words[((c_3 + 1) * nheads + head_8_2) * 4096 + pair_base_1 + i_4 * 128] = h_bf16_0[i_4] & h_mask_mine_1 | _shfl_xor_sync_30_1 << h_half_shift_1;
                                }
                            }
                            if (checkpoint_state_count > 0) {
                                int cur_physical_2 = 0;
                                int cur_limit_2 = 128;
                                cur_physical_2 = c_3 - sequence_7_2 * nchunks;
                                int cur_tokens_2 = seqlen - cur_physical_2 * 128;
                                if (cur_tokens_2 > 128) {
                                    cur_tokens_2 = 128;
                                }
                                int segment_end_2 = cur_physical_2 * 128 + cur_limit_2;
                                if (segment_end_2 > seqlen) {
                                    segment_end_2 = seqlen;
                                }
                                if (checkpoint_token_1 == segment_end_2) {
                                    int checkpoint_slot_2 = checkpoint_state_slots[sequence_7_2];
                                    if (checkpoint_slot_2 >= 0 && checkpoint_slot_2 < checkpoint_state_count) {
                                        int checkpoint_head_base_2 = (checkpoint_slot_2 * nheads + head_8_2) * 8192;
                                        #pragma unroll
                                        for (int k_24 = 0; k_24 < 32; k_24++) {
                                            checkpoint_states[checkpoint_head_base_2 + elem_base_1 + k_24 * 128] = h_1[k_24];
                                        }
                                    }
                                }
                            }
                            int look_a_1 = ((c_3 + 2) < (last_1) ? (c_3 + 2) : (last_1));
                            int look_tile_4 = look_a_1 * nheads + head_8_2;
                            int look_physical_4 = 0;
                            int look_offset_4 = 0;
                            int look_limit_4 = 128;
                            look_physical_4 = look_a_1 - sequence_7_2 * nchunks;
                            int look_tokens_4 = seqlen - look_physical_4 * 128;
                            if (look_tokens_4 > 128) {
                                look_tokens_4 = 128;
                            }
                            int source_token_4_1 = look_tokens_4 - 1;
                            float last_cumsum_4_1 = cumsum_precomputed[look_tile_4 * 128 + source_token_4_1];
                            lc_ring_1[0] = last_cumsum_4_1;
                            float segment_base_4 = 0.0f;
                            float _exp2_31_1 = approx_exp2((last_cumsum_4_1 - segment_base_4) * 1.442695f);
                            d_ring_1[0] = _exp2_31_1;
                            int look_t_4 = look_a_1 + s_t_off_1;
                            if (look_t_4 >= s_resident_from_1) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(s_a_1[0]), "=f"(s_a_1[1]), "=f"(s_a_1[2]), "=f"(s_a_1[3]), "=f"(s_a_1[4]), "=f"(s_a_1[5]), "=f"(s_a_1[6]), "=f"(s_a_1[7]), "=f"(s_a_1[8]), "=f"(s_a_1[9]), "=f"(s_a_1[10]), "=f"(s_a_1[11]), "=f"(s_a_1[12]), "=f"(s_a_1[13]), "=f"(s_a_1[14]), "=f"(s_a_1[15]), "=f"(s_a_1[16]), "=f"(s_a_1[17]), "=f"(s_a_1[18]), "=f"(s_a_1[19]), "=f"(s_a_1[20]), "=f"(s_a_1[21]), "=f"(s_a_1[22]), "=f"(s_a_1[23]), "=f"(s_a_1[24]), "=f"(s_a_1[25]), "=f"(s_a_1[26]), "=f"(s_a_1[27]), "=f"(s_a_1[28]), "=f"(s_a_1[29]), "=f"(s_a_1[30]), "=f"(s_a_1[31])
                                    : "r"(s_tmem_row_1 + look_t_4 % 8 * 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                            } else {
                                #pragma unroll
                                for (int q_14 = 0; q_14 < 8; q_14++) {
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_4 * 8192 + s_work_row_1 + 4 * q_14);
                                        s_a_1[4 * q_14 + 0] = _v4.x;
                                        s_a_1[4 * q_14 + 1] = _v4.y;
                                        s_a_1[4 * q_14 + 2] = _v4.z;
                                        s_a_1[4 * q_14 + 3] = _v4.w;
                                    }
                                }
                            }
                            int current_b_1 = c_3 + 1;
                            if (current_b_1 <= last_1) {
                                float decay_b_1 = d_ring_1[1];
                                const float2 _scale2_33 = {decay_b_1, decay_b_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 16; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(h_1)[_ls], _scale2_33);
                                #pragma unroll
                                for (int k_25 = 0; k_25 < 16; k_25++) {
                                    h_1[2 * k_25] = h_1[2 * k_25] + s_b_1[2 * k_25];
                                    h_1[2 * k_25 + 1] = h_1[2 * k_25 + 1] + s_b_1[2 * k_25 + 1];
                                }
                                if (current_b_1 < last_1) {
                                    unsigned int h_bf16_1_1[16];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 16; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(h_1[_lp*2 + 0], h_1[_lp*2+1 + 0]));
                                        h_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int i_5 = 0; i_5 < 16; i_5++) {
                                        unsigned int sent_5 = h_bf16_1_1[i_5] >> h_half_shift_1 & 65535;
                                        unsigned int _shfl_xor_sync_34_1 = __shfl_xor_sync(0xFFFFFFFF, sent_5, 1);
                                        h_words[((current_b_1 + 1) * nheads + head_8_2) * 4096 + pair_base_1 + i_5 * 128] = h_bf16_1_1[i_5] & h_mask_mine_1 | _shfl_xor_sync_34_1 << h_half_shift_1;
                                    }
                                }
                                if (checkpoint_state_count > 0) {
                                    int cur_physical_3 = 0;
                                    int cur_limit_3 = 128;
                                    cur_physical_3 = current_b_1 - sequence_7_2 * nchunks;
                                    int cur_tokens_3 = seqlen - cur_physical_3 * 128;
                                    if (cur_tokens_3 > 128) {
                                        cur_tokens_3 = 128;
                                    }
                                    int segment_end_3 = cur_physical_3 * 128 + cur_limit_3;
                                    if (segment_end_3 > seqlen) {
                                        segment_end_3 = seqlen;
                                    }
                                    if (checkpoint_token_1 == segment_end_3) {
                                        int checkpoint_slot_3 = checkpoint_state_slots[sequence_7_2];
                                        if (checkpoint_slot_3 >= 0 && checkpoint_slot_3 < checkpoint_state_count) {
                                            int checkpoint_head_base_3 = (checkpoint_slot_3 * nheads + head_8_2) * 8192;
                                            #pragma unroll
                                            for (int k_26 = 0; k_26 < 32; k_26++) {
                                                checkpoint_states[checkpoint_head_base_3 + elem_base_1 + k_26 * 128] = h_1[k_26];
                                            }
                                        }
                                    }
                                }
                                int look_b_1 = ((c_3 + 3) < (last_1) ? (c_3 + 3) : (last_1));
                                int look_tile_5 = look_b_1 * nheads + head_8_2;
                                int look_physical_5 = 0;
                                int look_offset_5 = 0;
                                int look_limit_5 = 128;
                                look_physical_5 = look_b_1 - sequence_7_2 * nchunks;
                                int look_tokens_5 = seqlen - look_physical_5 * 128;
                                if (look_tokens_5 > 128) {
                                    look_tokens_5 = 128;
                                }
                                int source_token_5_1 = look_tokens_5 - 1;
                                float last_cumsum_5_1 = cumsum_precomputed[look_tile_5 * 128 + source_token_5_1];
                                lc_ring_1[1] = last_cumsum_5_1;
                                float segment_base_5 = 0.0f;
                                float _exp2_35_1 = approx_exp2((last_cumsum_5_1 - segment_base_5) * 1.442695f);
                                d_ring_1[1] = _exp2_35_1;
                                int look_t_5 = look_b_1 + s_t_off_1;
                                if (look_t_5 >= s_resident_from_1) {
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(s_b_1[0]), "=f"(s_b_1[1]), "=f"(s_b_1[2]), "=f"(s_b_1[3]), "=f"(s_b_1[4]), "=f"(s_b_1[5]), "=f"(s_b_1[6]), "=f"(s_b_1[7]), "=f"(s_b_1[8]), "=f"(s_b_1[9]), "=f"(s_b_1[10]), "=f"(s_b_1[11]), "=f"(s_b_1[12]), "=f"(s_b_1[13]), "=f"(s_b_1[14]), "=f"(s_b_1[15]), "=f"(s_b_1[16]), "=f"(s_b_1[17]), "=f"(s_b_1[18]), "=f"(s_b_1[19]), "=f"(s_b_1[20]), "=f"(s_b_1[21]), "=f"(s_b_1[22]), "=f"(s_b_1[23]), "=f"(s_b_1[24]), "=f"(s_b_1[25]), "=f"(s_b_1[26]), "=f"(s_b_1[27]), "=f"(s_b_1[28]), "=f"(s_b_1[29]), "=f"(s_b_1[30]), "=f"(s_b_1[31])
                                        : "r"(s_tmem_row_1 + look_t_5 % 8 * 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                } else {
                                    #pragma unroll
                                    for (int q_15 = 0; q_15 < 8; q_15++) {
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(s_work + look_tile_5 * 8192 + s_work_row_1 + 4 * q_15);
                                            s_b_1[4 * q_15 + 0] = _v4.x;
                                            s_b_1[4 * q_15 + 1] = _v4.y;
                                            s_b_1[4 * q_15 + 2] = _v4.z;
                                            s_b_1[4 * q_15 + 3] = _v4.w;
                                        }
                                    }
                                }
                            }
                        }
                        if (first_10 + seg_len_10 == item_end_1) {
                            if (write_final_states != 0) {
                                #pragma unroll
                                for (int k_27 = 0; k_27 < 32; k_27++) {
                                    final_states[state_head_base_1 + elem_base_1 + k_27 * 128] = h_1[k_27];
                                }
                            }
                        }
                        x_6_1 = x_6_1 + seg_len_10;
                    }
                }
                __threadfence();
                asm volatile("fence.proxy.async;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int xc_3 = a_b_5;
                while (xc_3 < a_end_5) {
                    int locc_3[6];
                    int sequence_11 = 0;
                    int s_base_11 = 0;
                    int n_item_12 = nchunks;
                    sequence_11 = xc_3 / (nchunks * nheads);
                    s_base_11 = sequence_11 * nchunks;
                    int rem_11 = xc_3 - s_base_11 * nheads;
                    int head_11 = rem_11 / n_item_12;
                    int chunk_11 = rem_11 - head_11 * n_item_12;
                    locc_3[0] = sequence_11;
                    locc_3[1] = head_11;
                    locc_3[2] = chunk_11;
                    locc_3[3] = n_item_12;
                    locc_3[4] = s_base_11 + chunk_11;
                    locc_3[5] = xc_3 - chunk_11;
                    int head_0_9 = locc_3[1];
                    int first_11 = locc_3[4];
                    int seg_len_11 = locc_3[3] - locc_3[2];
                    if (seg_len_11 > a_end_5 - xc_3) {
                        seg_len_11 = a_end_5 - xc_3;
                    }
                    int _uniform_34 = make_warp_uniform(head_0_9);
                    head_0_9 = _uniform_34;
                    int _uniform_35 = make_warp_uniform(first_11);
                    first_11 = _uniform_35;
                    int _uniform_36 = make_warp_uniform(seg_len_11);
                    seg_len_11 = _uniform_36;
                    #pragma unroll 1
                    for (int logical_9 = first_11; logical_9 < first_11 + seg_len_11; logical_9++) {
                        int physical_chunk_6 = 0;
                        int segment_offset_4 = 0;
                        int segment_limit_4 = 128;
                        int sequence_0_2 = logical_9 / nchunks;
                        physical_chunk_6 = logical_9 - sequence_0_2 * nchunks;
                        int chunk_tokens_4 = seqlen - physical_chunk_6 * 128;
                        if (chunk_tokens_4 > 128) {
                            chunk_tokens_4 = 128;
                        }
                        mbarrier_wait(cb_full_addr + (acc_stage_1) * 8, _phase_cb_full);
                        mbarrier_wait(aux3_full_addr + (stage3_4) * 8, _phase_aux3_full);
                        mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                        _phase_q_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float row_cumsum = smem_cumsum_all[stage3_4 * 128 + (unsigned int)row];
                        float segment_base_12 = 0.0f;
                        if (segment_offset_4 > 0) {
                            segment_base_12 = smem_cumsum_all[stage3_4 * 128 + (unsigned int)(segment_offset_4 - 1)];
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
                                    float decay = approx_exp2((row_cumsum - cumsum_values[local_col]) * 1.442695f);
                                    q_value = decay * delta_values[local_col];
                                    q_value *= _tmem_load_1[local_col];
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
                                :: "r"(taddr_0_1 + row_tmem_base + 256 + col_chunk * 16), "r"(q_values_bf16[0]), "r"(q_values_bf16[1]), "r"(q_values_bf16[2]), "r"(q_values_bf16[3]), "r"(q_values_bf16[4]), "r"(q_values_bf16[5]), "r"(q_values_bf16[6]), "r"(q_values_bf16[7]), "r"(q_values_bf16[8]), "r"(q_values_bf16[9]), "r"(q_values_bf16[10]), "r"(q_values_bf16[11]), "r"(q_values_bf16[12]), "r"(q_values_bf16[13]), "r"(q_values_bf16[14]), "r"(q_values_bf16[15]));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(aux3_empty_addr + (stage3_4) * 8);
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
                    xc_3 = xc_3 + seg_len_11;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // epilogue_main
            unsigned int stage3_5 = 0;
            unsigned int output_stage = 0;
            int output_issued = 0;
            int row_1 = warp % 4 * 32 + lane;
            int row_tmem_base_1 = warp % 4 * 32 << 16;
            int taddr_0_2 = taddr;
            int tot_6[1];
            tot_6[0] = sequence_count * nchunks * nheads;
            int veff_6[1];
            int total_6 = tot_6[0];
            int g_6 = 1;
            int ranges_6 = (total_6 + g_6 - 1) / g_6;
            if (ranges_6 < 1) {
                ranges_6 = 1;
            }
            if (ranges_6 > vcount) {
                ranges_6 = vcount;
            }
            veff_6[0] = ranges_6;
            int vcount_eff_6 = veff_6[0];
            unsigned int total_u_6 = (unsigned int)vcount_eff_6;
            unsigned int _phase_intra_full_0 = 0;
            unsigned int _phase_inter_full_0 = 0;
            unsigned int _phase_aux3_full_1 = 0;
            while (1) {
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                unsigned int ticket_u_6 = ctrl[0];
                if (ticket_u_6 >= total_u_6) {
                    break;
                }
                int ticket_6 = (int)ticket_u_6;
                int rb_6[2];
                int total_0_6 = tot_6[0];
                rb_6[0] = (ticket_6 * total_0_6 + vcount_eff_6 - 1) / vcount_eff_6;
                rb_6[1] = ((ticket_6 + 1) * total_0_6 + vcount_eff_6 - 1) / vcount_eff_6;
                int a_b_6 = rb_6[0];
                int a_end_6 = rb_6[1];
                int _uniform_42 = make_warp_uniform(a_b_6);
                a_b_6 = _uniform_42;
                int _uniform_43 = make_warp_uniform(a_end_6);
                a_end_6 = _uniform_43;
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                asm volatile("barrier.sync 11, 512;" ::: "memory");
                int xa_6 = a_b_6;
                while (xa_6 < a_end_6) {
                    int loca_6[6];
                    int sequence_12 = 0;
                    int s_base_12 = 0;
                    int n_item_13 = nchunks;
                    sequence_12 = xa_6 / (nchunks * nheads);
                    s_base_12 = sequence_12 * nchunks;
                    int rem_12 = xa_6 - s_base_12 * nheads;
                    int head_12 = rem_12 / n_item_13;
                    int chunk_12 = rem_12 - head_12 * n_item_13;
                    loca_6[0] = sequence_12;
                    loca_6[1] = head_12;
                    loca_6[2] = chunk_12;
                    loca_6[3] = n_item_13;
                    loca_6[4] = s_base_12 + chunk_12;
                    loca_6[5] = xa_6 - chunk_12;
                    int head_0_10 = loca_6[1];
                    int first_12 = loca_6[4];
                    int seg_len_12 = loca_6[3] - loca_6[2];
                    if (seg_len_12 > a_end_6 - xa_6) {
                        seg_len_12 = a_end_6 - xa_6;
                    }
                    int _uniform_44 = make_warp_uniform(head_0_10);
                    head_0_10 = _uniform_44;
                    int _uniform_45 = make_warp_uniform(first_12);
                    first_12 = _uniform_45;
                    int _uniform_46 = make_warp_uniform(seg_len_12);
                    seg_len_12 = _uniform_46;
                    #pragma unroll 1
                    for (int logical_10 = first_12; logical_10 < first_12 + seg_len_12; logical_10++) {
                        float d_head_value = 0.0f;
                        if (D_mode == 1) {
                            d_head_value = (float)D[head_0_10];
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
                        mbarrier_wait(x3_full_addr + (stage3_5) * 8, _phase_aux3_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float segment_base_13 = 0.0f;
                        if (segment_offset_5 > 0) {
                            segment_base_13 = smem_cumsum_all[stage3_5 * 128 + (unsigned int)(segment_offset_5 - 1)];
                        }
                        int full_chunk = 1;
                        if (full_chunk != 0 && output_issued >= 2) {
                            int warp_id_in_role = warp - 12;
                            if (warp_id_in_role == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 1;");
                            }
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                            asm volatile("tcgen05.fence::after_thread_sync;");
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
                            if (dim_chunk == 1) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                asm volatile("barrier.sync 10, 128;" ::: "memory");
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                if (warp == 12) {
                                    if (elect_sync()) {
                                        mbarrier_arrive(outputs_empty_addr);
                                    }
                                }
                            }
                            if (row_1 >= segment_offset_5 && row_1 < segment_limit_5) {
                                int token_in_batch_4 = physical_chunk_7 * 128 + row_1;
                                int token = physical_batch_4 * seqlen + token_in_batch_4;
                                float decay_1 = approx_exp2((smem_cumsum_all[stage3_5 * 128 + (unsigned int)row_1] - segment_base_13) * 1.442695f);
                                int z_token = token;
                                if (row_1 >= chunk_tokens_5) {
                                    z_token = physical_batch_4 * seqlen + physical_chunk_7 * 128 + chunk_tokens_5 - 1;
                                }
                                int z_row_base = z_token * nheads * 64 + head_0_10 * 64;
                                int out_row_base = (token * nheads + head_0_10) * 64;
                                #pragma unroll
                                for (int local_group = 0; local_group < 4; local_group++) {
                                    int dim_base_2 = dim_chunk * 32 + local_group * 8;
                                    unsigned int x_packed[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&x_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x_packed[(0) + 3]))
                                        : "r"((smem_x_tma_addr + stage3_5 * 16384 + (unsigned int)(dim_base_2 / 64 * 16384 + row_1 * 128 + dim_base_2 % 64 * 2 ^ (dim_base_2 / 64 * 16384 + row_1 * 128 + dim_base_2 % 64 * 2 >> 7 & 7) << 4))));
                                    float _unpack_bf16_0[8];
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) << 16);
                                        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_packed[_pair]) & 0xffff0000u);
                                    }
                                    float d_values[8];
                                    #pragma unroll
                                    for (int group_dim = 0; group_dim < 8; group_dim++) {
                                        d_values[group_dim] = d_head_value;
                                    }
                                    if (D_mode == 2) {
                                        {
                                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(D + head_0_10 * 64 + dim_base_2);
                                            uint4 _vld_0[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_0[_blk] = _vptr_0[_blk];
                                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&d_values[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                                                    (&d_values[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                    }
                                    float z_scale[8];
                                    if (has_z != 0) {
                                        {
                                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(z + z_row_base + dim_base_2);
                                            uint4 _vld_1[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_1[_blk] = _vptr_1[_blk];
                                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    (&z_scale[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                                                    (&z_scale[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int group_dim_1 = 0; group_dim_1 < 8; group_dim_1++) {
                                            float z_value = z_scale[group_dim_1];
                                            float _exp_3 = __expf(-z_value);
                                            float _rcp_2 = approx_rcp(1.0f + _exp_3);
                                            z_scale[group_dim_1] = z_value * _rcp_2;
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
                                        float value = __fmaf_rn(_tmem_load_4[local_dim], decay_1, _tmem_load_3[local_dim]);
                                        float x_value = _unpack_bf16_0[group_dim_3];
                                        float d_value = d_values[group_dim_3];
                                        if (D_mode != 0) {
                                            value = __fmaf_rn(x_value, d_value, value);
                                        }
                                        float z_mul = z_scale[group_dim_3];
                                        value = value * z_mul;
                                        group_values[group_dim_3] = value;
                                    }
                                    if (full_chunk != 0) {
                                        unsigned int group_values_bf16[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(group_values[_lp*2 + 0], group_values[_lp*2+1 + 0]));
                                            group_values_bf16[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_y_addr + output_stage * 16384 + (unsigned int)(row_1 * 128 + dim_base_2 * 2 ^ (row_1 * 128 + dim_base_2 * 2 >> 7 & 7) << 4))), "r"(group_values_bf16[0]), "r"(group_values_bf16[1]), "r"(group_values_bf16[2]), "r"(group_values_bf16[3]) : "memory");
                                    } else {
                                        #pragma unroll
                                        for (int group_dim_4 = 0; group_dim_4 < 8; group_dim_4++) {
                                            out_native[out_row_base + dim_base_2 + group_dim_4] = group_values[group_dim_4];
                                        }
                                    }
                                }
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (full_chunk != 0) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                            int warp_id_in_role_1 = warp - 12;
                            if (warp_id_in_role_1 == 0) {
                                if (elect_sync()) {
                                    tma_store_4d((&out_map), 0, head_0_10, physical_chunk_7 * 128, physical_batch_4, smem_y_addr + output_stage * 16384);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
                            output_stage = output_stage ^ 1;
                            output_issued += 1;
                        }
                        mbarrier_arrive(aux3_empty_addr + (stage3_5) * 8);
                        if (warp == 12) {
                            if (elect_sync()) {
                                mbarrier_arrive(x3_empty_addr + (stage3_5) * 8);
                            }
                        }
                        stage3_5 += 1;
                        if (stage3_5 == 2) { stage3_5 = 0; _phase_aux3_full_1 ^= 1; }
                    }
                    xa_6 = xa_6 + seg_len_12;
                }
            }
            int warp_id_in_role_2 = warp - 12;
            if (warp_id_in_role_2 == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 12) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

