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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 396
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 384
#define TMEM_SF_B_OFFSET 388
#define NUM_PAB_STAGES 6
#define NUM_PB_STAGES 6
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define NUM_PRELAY_STAGES 6
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 99328
#define SMEM_B_STAGE_BYTES 12288
#define SMEM_B_STRIDE 12288
#define SMEM_SFA_OFF 173056
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 176128
#define SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SFB_STRIDE 1024
#define SMEM_SINFO_OFF 182272
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 182496
#define SMEM_STOK_STAGE_BYTES 6144
#define SMEM_STOK_STRIDE 6144
#define SMEM_SSCALE_OFF 188640
#define SMEM_SSCALE_STAGE_BYTES 6176
#define SMEM_SSCALE_STRIDE 6176
#define SMEM_SEXCH_OFF 194816
#define SMEM_SEXCH_STAGE_BYTES 8448
#define SMEM_SEXCH_STRIDE 8448
#define SMEM_STAG_OFF 203264
#define SMEM_STAG_STAGE_BYTES 8704
#define SMEM_STAG_STRIDE 8704
#define SMEM_TOTAL 220672
#define THREADS 384

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2_elect(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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





__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}





extern "C" {

__global__ __launch_bounds__(384) __cluster_dims__(2,1,1) void
kernel_cake_mxfp4_situ_moe_01e722d0eb8ea01e871d(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, __nv_bfloat16* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 48)
    #define b_full_addr (mbar_base + 96)
    #define b_relay_full_addr (mbar_base + 144)
    #define b_free_addr (mbar_base + 192)
    #define acc_full_addr (mbar_base + 240)
    #define acc_free_addr (mbar_base + 256)
    #define tile_full_addr (mbar_base + 272)
    #define tile_free_addr (mbar_base + 336)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_addr = smem + 1024;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int b_addr = smem + 99328;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 173056);
    const int sfa_addr = smem + 173056;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 176128);
    const int sfb_addr = smem + 176128;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 182272);
    const int sinfo_addr = smem + 182272;
    int* stok = reinterpret_cast<int*>(smem_raw + 182496);
    const int stok_addr = smem + 182496;
    float* sscale = reinterpret_cast<float*>(smem_raw + 188640);
    const int sscale_addr = smem + 188640;
    float* sexch = reinterpret_cast<float*>(smem_raw + 194816);
    const int sexch_addr = smem + 194816;
    unsigned int* stag = reinterpret_cast<unsigned int*>(smem_raw + 203264);
    const int stag_addr = smem + 203264;
    int _mma_base_lo_0 = ((a_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((b_addr) >> 4) & 0x3FFF;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 50 barriers)
    // Mbarriers at smem_raw[0..400)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 6 barriers, init_count=1
        // ab_free: 6 barriers, init_count=1
        // --- pipeline 'pb' ---
        // b_full: 6 barriers, init_count=128
        // --- pipeline 'prelay' ---
        // b_relay_full: 6 barriers, init_count=64
        // --- pipeline 'pb' ---
        // b_free: 6 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=256
        // --- pipeline 'ptile' ---
        // tile_full: 8 barriers, init_count=32
        // tile_free: 8 barriers, init_count=352
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 352;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(10), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(2), "r"((uint32_t)(256)));
        if (lane < 18) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 396 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 400);
    if (warp == 0) {
        int _tmem_hold = smem + 400;
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
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 384;
    const int tmem_sf_b = taddr + 388;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int nrec_epilogue = 0;
            int nst_epilogue = 0;
            int epi_tidx = tid;
            int lane_0 = lane;
            int epi_warp = warp;
            unsigned int acc_stage = 0;
            unsigned int tile_stage = 0;
            int info[7];
            int cur_tok[1];
            float cur_scale[1];
            float meta_alpha = 0.0f;
            float wvals[32];
            float fin_scl[8];
            unsigned int fin_words[4];
            int fin_tok = epi_tidx / 16;
            int fin_chunk = epi_tidx % 16;
            int fin_row = epi_tidx / 4;
            int is_fin_issuer = (int)(epi_tidx % 4 == 0);
            int tok_pair = lane_0 % 4;
            int mat_row = lane_0 % 8;
            int mat_col_bytes = (epi_warp * 32 + 8 * (lane_0 / 8)) * 2;
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 7];
            info[1] = sinfo[tile_stage * 7 + 1];
            info[2] = sinfo[tile_stage * 7 + 2];
            info[3] = sinfo[tile_stage * 7 + 3];
            info[4] = sinfo[tile_stage * 7 + 4];
            info[5] = sinfo[tile_stage * 7 + 5];
            info[6] = sinfo[tile_stage * 7 + 6];
            meta_alpha = sscale[tile_stage * 193 + 192];
            nrec_epilogue = nrec_epilogue + 1;
            bool is_even_lane = lane_0 % 2 == 0;
            int is_gate_lane = (int)(epi_warp >= 2);
            int exch_row = epi_tidx & 63;
            float inv_fp8_max = 0.002232142857142857f;
            float zero_f32 = 0.0f;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < num_m_tiles * group_capacity + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = info[1] * 64;
                int mn_limit = info[4];
                int h0 = (info[0] * 2 + cta_rank) * 128;
                int h = h0 + epi_tidx;
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192));
                    float _tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_0[0] * fin_scl[0];
                    wvals[16] = _tmem_load_1[0] * fin_scl[0];
                    wvals[1] = _tmem_load_0[1] * fin_scl[1];
                    wvals[17] = _tmem_load_1[1] * fin_scl[1];
                    wvals[2] = _tmem_load_0[2] * fin_scl[0];
                    wvals[18] = _tmem_load_1[2] * fin_scl[0];
                    wvals[3] = _tmem_load_0[3] * fin_scl[1];
                    wvals[19] = _tmem_load_1[3] * fin_scl[1];
                    wvals[4] = _tmem_load_0[4] * fin_scl[2];
                    wvals[20] = _tmem_load_1[4] * fin_scl[2];
                    wvals[5] = _tmem_load_0[5] * fin_scl[3];
                    wvals[21] = _tmem_load_1[5] * fin_scl[3];
                    wvals[6] = _tmem_load_0[6] * fin_scl[2];
                    wvals[22] = _tmem_load_1[6] * fin_scl[2];
                    wvals[7] = _tmem_load_0[7] * fin_scl[3];
                    wvals[23] = _tmem_load_1[7] * fin_scl[3];
                    wvals[8] = _tmem_load_0[8] * fin_scl[4];
                    wvals[24] = _tmem_load_1[8] * fin_scl[4];
                    wvals[9] = _tmem_load_0[9] * fin_scl[5];
                    wvals[25] = _tmem_load_1[9] * fin_scl[5];
                    wvals[10] = _tmem_load_0[10] * fin_scl[4];
                    wvals[26] = _tmem_load_1[10] * fin_scl[4];
                    wvals[11] = _tmem_load_0[11] * fin_scl[5];
                    wvals[27] = _tmem_load_1[11] * fin_scl[5];
                    wvals[12] = _tmem_load_0[12] * fin_scl[6];
                    wvals[28] = _tmem_load_1[12] * fin_scl[6];
                    wvals[13] = _tmem_load_0[13] * fin_scl[7];
                    wvals[29] = _tmem_load_1[13] * fin_scl[7];
                    wvals[14] = _tmem_load_0[14] * fin_scl[6];
                    wvals[30] = _tmem_load_1[14] * fin_scl[6];
                    wvals[15] = _tmem_load_0[15] * fin_scl[7];
                    wvals[31] = _tmem_load_1[15] * fin_scl[7];
                    uint32_t wvals_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf = stag_addr;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(stag_buf + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(stag_buf + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(stag_buf + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(stag_buf + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s = fin_tok;
                    int prow_f = row_base + tok_s;
                    int ok_f = (int)(prow_f < mn_limit);
                    int tok_f = stok[tile_stage * 192 + (unsigned int)tok_s];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf + tok_s * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f)) : "memory");
                    int tok_s_0 = fin_tok + 8;
                    int prow_f_1 = row_base + tok_s_0;
                    int ok_f_2 = (int)(prow_f_1 < mn_limit);
                    int tok_f_3 = stok[tile_stage * 192 + (unsigned int)tok_s_0];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf + tok_s_0 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_3 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_2)) : "memory");
                    int tok_s_4 = fin_tok + 16;
                    int prow_f_5 = row_base + tok_s_4;
                    int ok_f_6 = (int)(prow_f_5 < mn_limit);
                    int tok_f_7 = stok[tile_stage * 192 + (unsigned int)tok_s_4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf + tok_s_4 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_7 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_6)) : "memory");
                    int tok_s_8 = fin_tok + 24;
                    int prow_f_9 = row_base + tok_s_8;
                    int ok_f_10 = (int)(prow_f_9 < mn_limit);
                    int tok_f_11 = stok[tile_stage * 192 + (unsigned int)tok_s_8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf + tok_s_8 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_11 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_10)) : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192 + 32));
                    float _tmem_load_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + 32 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + 32 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 32 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 32 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 32 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 32 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 32 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 32 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_2[0] * fin_scl[0];
                    wvals[16] = _tmem_load_3[0] * fin_scl[0];
                    wvals[1] = _tmem_load_2[1] * fin_scl[1];
                    wvals[17] = _tmem_load_3[1] * fin_scl[1];
                    wvals[2] = _tmem_load_2[2] * fin_scl[0];
                    wvals[18] = _tmem_load_3[2] * fin_scl[0];
                    wvals[3] = _tmem_load_2[3] * fin_scl[1];
                    wvals[19] = _tmem_load_3[3] * fin_scl[1];
                    wvals[4] = _tmem_load_2[4] * fin_scl[2];
                    wvals[20] = _tmem_load_3[4] * fin_scl[2];
                    wvals[5] = _tmem_load_2[5] * fin_scl[3];
                    wvals[21] = _tmem_load_3[5] * fin_scl[3];
                    wvals[6] = _tmem_load_2[6] * fin_scl[2];
                    wvals[22] = _tmem_load_3[6] * fin_scl[2];
                    wvals[7] = _tmem_load_2[7] * fin_scl[3];
                    wvals[23] = _tmem_load_3[7] * fin_scl[3];
                    wvals[8] = _tmem_load_2[8] * fin_scl[4];
                    wvals[24] = _tmem_load_3[8] * fin_scl[4];
                    wvals[9] = _tmem_load_2[9] * fin_scl[5];
                    wvals[25] = _tmem_load_3[9] * fin_scl[5];
                    wvals[10] = _tmem_load_2[10] * fin_scl[4];
                    wvals[26] = _tmem_load_3[10] * fin_scl[4];
                    wvals[11] = _tmem_load_2[11] * fin_scl[5];
                    wvals[27] = _tmem_load_3[11] * fin_scl[5];
                    wvals[12] = _tmem_load_2[12] * fin_scl[6];
                    wvals[28] = _tmem_load_3[12] * fin_scl[6];
                    wvals[13] = _tmem_load_2[13] * fin_scl[7];
                    wvals[29] = _tmem_load_3[13] * fin_scl[7];
                    wvals[14] = _tmem_load_2[14] * fin_scl[6];
                    wvals[30] = _tmem_load_3[14] * fin_scl[6];
                    wvals[15] = _tmem_load_2[15] * fin_scl[7];
                    wvals[31] = _tmem_load_3[15] * fin_scl[7];
                    uint32_t wvals_bf16_12[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16_12[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf_13 = stag_addr + 8704;
                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(stag_buf_13 + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(stag_buf_13 + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(stag_buf_13 + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(stag_buf_13 + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_12[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s_14 = fin_tok;
                    int prow_f_15 = row_base + 32 + tok_s_14;
                    int ok_f_16 = (int)(prow_f_15 < mn_limit);
                    int tok_f_17 = stok[tile_stage * 192 + 32 + (unsigned int)tok_s_14];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_13 + tok_s_14 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_17 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_16)) : "memory");
                    int tok_s_18 = fin_tok + 8;
                    int prow_f_19 = row_base + 32 + tok_s_18;
                    int ok_f_20 = (int)(prow_f_19 < mn_limit);
                    int tok_f_21 = stok[tile_stage * 192 + 32 + (unsigned int)tok_s_18];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_13 + tok_s_18 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_21 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_20)) : "memory");
                    int tok_s_22 = fin_tok + 16;
                    int prow_f_23 = row_base + 32 + tok_s_22;
                    int ok_f_24 = (int)(prow_f_23 < mn_limit);
                    int tok_f_25 = stok[tile_stage * 192 + 32 + (unsigned int)tok_s_22];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_13 + tok_s_22 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_25 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_24)) : "memory");
                    int tok_s_26 = fin_tok + 24;
                    int prow_f_27 = row_base + 32 + tok_s_26;
                    int ok_f_28 = (int)(prow_f_27 < mn_limit);
                    int tok_f_29 = stok[tile_stage * 192 + 32 + (unsigned int)tok_s_26];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_13 + tok_s_26 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_29 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_28)) : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_4[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192 + 64));
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + 64 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + 64 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 64 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 64 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 64 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 64 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 64 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 64 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_4[0] * fin_scl[0];
                    wvals[16] = _tmem_load_5[0] * fin_scl[0];
                    wvals[1] = _tmem_load_4[1] * fin_scl[1];
                    wvals[17] = _tmem_load_5[1] * fin_scl[1];
                    wvals[2] = _tmem_load_4[2] * fin_scl[0];
                    wvals[18] = _tmem_load_5[2] * fin_scl[0];
                    wvals[3] = _tmem_load_4[3] * fin_scl[1];
                    wvals[19] = _tmem_load_5[3] * fin_scl[1];
                    wvals[4] = _tmem_load_4[4] * fin_scl[2];
                    wvals[20] = _tmem_load_5[4] * fin_scl[2];
                    wvals[5] = _tmem_load_4[5] * fin_scl[3];
                    wvals[21] = _tmem_load_5[5] * fin_scl[3];
                    wvals[6] = _tmem_load_4[6] * fin_scl[2];
                    wvals[22] = _tmem_load_5[6] * fin_scl[2];
                    wvals[7] = _tmem_load_4[7] * fin_scl[3];
                    wvals[23] = _tmem_load_5[7] * fin_scl[3];
                    wvals[8] = _tmem_load_4[8] * fin_scl[4];
                    wvals[24] = _tmem_load_5[8] * fin_scl[4];
                    wvals[9] = _tmem_load_4[9] * fin_scl[5];
                    wvals[25] = _tmem_load_5[9] * fin_scl[5];
                    wvals[10] = _tmem_load_4[10] * fin_scl[4];
                    wvals[26] = _tmem_load_5[10] * fin_scl[4];
                    wvals[11] = _tmem_load_4[11] * fin_scl[5];
                    wvals[27] = _tmem_load_5[11] * fin_scl[5];
                    wvals[12] = _tmem_load_4[12] * fin_scl[6];
                    wvals[28] = _tmem_load_5[12] * fin_scl[6];
                    wvals[13] = _tmem_load_4[13] * fin_scl[7];
                    wvals[29] = _tmem_load_5[13] * fin_scl[7];
                    wvals[14] = _tmem_load_4[14] * fin_scl[6];
                    wvals[30] = _tmem_load_5[14] * fin_scl[6];
                    wvals[15] = _tmem_load_4[15] * fin_scl[7];
                    wvals[31] = _tmem_load_5[15] * fin_scl[7];
                    uint32_t wvals_bf16_30[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16_30[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf_31 = stag_addr;
                    uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(stag_buf_31 + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(stag_buf_31 + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(stag_buf_31 + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(stag_buf_31 + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_30[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s_32 = fin_tok;
                    int prow_f_33 = row_base + 64 + tok_s_32;
                    int ok_f_34 = (int)(prow_f_33 < mn_limit);
                    int tok_f_35 = stok[tile_stage * 192 + 64 + (unsigned int)tok_s_32];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_31 + tok_s_32 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_35 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_34)) : "memory");
                    int tok_s_36 = fin_tok + 8;
                    int prow_f_37 = row_base + 64 + tok_s_36;
                    int ok_f_38 = (int)(prow_f_37 < mn_limit);
                    int tok_f_39 = stok[tile_stage * 192 + 64 + (unsigned int)tok_s_36];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_31 + tok_s_36 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_39 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_38)) : "memory");
                    int tok_s_40 = fin_tok + 16;
                    int prow_f_41 = row_base + 64 + tok_s_40;
                    int ok_f_42 = (int)(prow_f_41 < mn_limit);
                    int tok_f_43 = stok[tile_stage * 192 + 64 + (unsigned int)tok_s_40];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_31 + tok_s_40 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_43 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_42)) : "memory");
                    int tok_s_44 = fin_tok + 24;
                    int prow_f_45 = row_base + 64 + tok_s_44;
                    int ok_f_46 = (int)(prow_f_45 < mn_limit);
                    int tok_f_47 = stok[tile_stage * 192 + 64 + (unsigned int)tok_s_44];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_31 + tok_s_44 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_47 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_46)) : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192 + 96));
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192 + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + 96 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + 96 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 96 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 96 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 96 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 96 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 96 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 96 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_6[0] * fin_scl[0];
                    wvals[16] = _tmem_load_7[0] * fin_scl[0];
                    wvals[1] = _tmem_load_6[1] * fin_scl[1];
                    wvals[17] = _tmem_load_7[1] * fin_scl[1];
                    wvals[2] = _tmem_load_6[2] * fin_scl[0];
                    wvals[18] = _tmem_load_7[2] * fin_scl[0];
                    wvals[3] = _tmem_load_6[3] * fin_scl[1];
                    wvals[19] = _tmem_load_7[3] * fin_scl[1];
                    wvals[4] = _tmem_load_6[4] * fin_scl[2];
                    wvals[20] = _tmem_load_7[4] * fin_scl[2];
                    wvals[5] = _tmem_load_6[5] * fin_scl[3];
                    wvals[21] = _tmem_load_7[5] * fin_scl[3];
                    wvals[6] = _tmem_load_6[6] * fin_scl[2];
                    wvals[22] = _tmem_load_7[6] * fin_scl[2];
                    wvals[7] = _tmem_load_6[7] * fin_scl[3];
                    wvals[23] = _tmem_load_7[7] * fin_scl[3];
                    wvals[8] = _tmem_load_6[8] * fin_scl[4];
                    wvals[24] = _tmem_load_7[8] * fin_scl[4];
                    wvals[9] = _tmem_load_6[9] * fin_scl[5];
                    wvals[25] = _tmem_load_7[9] * fin_scl[5];
                    wvals[10] = _tmem_load_6[10] * fin_scl[4];
                    wvals[26] = _tmem_load_7[10] * fin_scl[4];
                    wvals[11] = _tmem_load_6[11] * fin_scl[5];
                    wvals[27] = _tmem_load_7[11] * fin_scl[5];
                    wvals[12] = _tmem_load_6[12] * fin_scl[6];
                    wvals[28] = _tmem_load_7[12] * fin_scl[6];
                    wvals[13] = _tmem_load_6[13] * fin_scl[7];
                    wvals[29] = _tmem_load_7[13] * fin_scl[7];
                    wvals[14] = _tmem_load_6[14] * fin_scl[6];
                    wvals[30] = _tmem_load_7[14] * fin_scl[6];
                    wvals[15] = _tmem_load_6[15] * fin_scl[7];
                    wvals[31] = _tmem_load_7[15] * fin_scl[7];
                    uint32_t wvals_bf16_48[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16_48[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf_49 = stag_addr + 8704;
                    uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(stag_buf_49 + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(stag_buf_49 + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(stag_buf_49 + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(stag_buf_49 + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_48[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s_50 = fin_tok;
                    int prow_f_51 = row_base + 96 + tok_s_50;
                    int ok_f_52 = (int)(prow_f_51 < mn_limit);
                    int tok_f_53 = stok[tile_stage * 192 + 96 + (unsigned int)tok_s_50];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_49 + tok_s_50 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_53 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_52)) : "memory");
                    int tok_s_54 = fin_tok + 8;
                    int prow_f_55 = row_base + 96 + tok_s_54;
                    int ok_f_56 = (int)(prow_f_55 < mn_limit);
                    int tok_f_57 = stok[tile_stage * 192 + 96 + (unsigned int)tok_s_54];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_49 + tok_s_54 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_57 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_56)) : "memory");
                    int tok_s_58 = fin_tok + 16;
                    int prow_f_59 = row_base + 96 + tok_s_58;
                    int ok_f_60 = (int)(prow_f_59 < mn_limit);
                    int tok_f_61 = stok[tile_stage * 192 + 96 + (unsigned int)tok_s_58];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_49 + tok_s_58 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_61 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_60)) : "memory");
                    int tok_s_62 = fin_tok + 24;
                    int prow_f_63 = row_base + 96 + tok_s_62;
                    int ok_f_64 = (int)(prow_f_63 < mn_limit);
                    int tok_f_65 = stok[tile_stage * 192 + 96 + (unsigned int)tok_s_62];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_49 + tok_s_62 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_65 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_64)) : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_8[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192 + 128));
                    float _tmem_load_9[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192 + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + 128 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + 128 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 128 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 128 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 128 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 128 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 128 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 128 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_8[0] * fin_scl[0];
                    wvals[16] = _tmem_load_9[0] * fin_scl[0];
                    wvals[1] = _tmem_load_8[1] * fin_scl[1];
                    wvals[17] = _tmem_load_9[1] * fin_scl[1];
                    wvals[2] = _tmem_load_8[2] * fin_scl[0];
                    wvals[18] = _tmem_load_9[2] * fin_scl[0];
                    wvals[3] = _tmem_load_8[3] * fin_scl[1];
                    wvals[19] = _tmem_load_9[3] * fin_scl[1];
                    wvals[4] = _tmem_load_8[4] * fin_scl[2];
                    wvals[20] = _tmem_load_9[4] * fin_scl[2];
                    wvals[5] = _tmem_load_8[5] * fin_scl[3];
                    wvals[21] = _tmem_load_9[5] * fin_scl[3];
                    wvals[6] = _tmem_load_8[6] * fin_scl[2];
                    wvals[22] = _tmem_load_9[6] * fin_scl[2];
                    wvals[7] = _tmem_load_8[7] * fin_scl[3];
                    wvals[23] = _tmem_load_9[7] * fin_scl[3];
                    wvals[8] = _tmem_load_8[8] * fin_scl[4];
                    wvals[24] = _tmem_load_9[8] * fin_scl[4];
                    wvals[9] = _tmem_load_8[9] * fin_scl[5];
                    wvals[25] = _tmem_load_9[9] * fin_scl[5];
                    wvals[10] = _tmem_load_8[10] * fin_scl[4];
                    wvals[26] = _tmem_load_9[10] * fin_scl[4];
                    wvals[11] = _tmem_load_8[11] * fin_scl[5];
                    wvals[27] = _tmem_load_9[11] * fin_scl[5];
                    wvals[12] = _tmem_load_8[12] * fin_scl[6];
                    wvals[28] = _tmem_load_9[12] * fin_scl[6];
                    wvals[13] = _tmem_load_8[13] * fin_scl[7];
                    wvals[29] = _tmem_load_9[13] * fin_scl[7];
                    wvals[14] = _tmem_load_8[14] * fin_scl[6];
                    wvals[30] = _tmem_load_9[14] * fin_scl[6];
                    wvals[15] = _tmem_load_8[15] * fin_scl[7];
                    wvals[31] = _tmem_load_9[15] * fin_scl[7];
                    uint32_t wvals_bf16_66[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16_66[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf_67 = stag_addr;
                    uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(stag_buf_67 + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(stag_buf_67 + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(stag_buf_67 + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(stag_buf_67 + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_66[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s_68 = fin_tok;
                    int prow_f_69 = row_base + 128 + tok_s_68;
                    int ok_f_70 = (int)(prow_f_69 < mn_limit);
                    int tok_f_71 = stok[tile_stage * 192 + 128 + (unsigned int)tok_s_68];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_67 + tok_s_68 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_71 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_70)) : "memory");
                    int tok_s_72 = fin_tok + 8;
                    int prow_f_73 = row_base + 128 + tok_s_72;
                    int ok_f_74 = (int)(prow_f_73 < mn_limit);
                    int tok_f_75 = stok[tile_stage * 192 + 128 + (unsigned int)tok_s_72];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_67 + tok_s_72 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_75 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_74)) : "memory");
                    int tok_s_76 = fin_tok + 16;
                    int prow_f_77 = row_base + 128 + tok_s_76;
                    int ok_f_78 = (int)(prow_f_77 < mn_limit);
                    int tok_f_79 = stok[tile_stage * 192 + 128 + (unsigned int)tok_s_76];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_67 + tok_s_76 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_79 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_78)) : "memory");
                    int tok_s_80 = fin_tok + 24;
                    int prow_f_81 = row_base + 128 + tok_s_80;
                    int ok_f_82 = (int)(prow_f_81 < mn_limit);
                    int tok_f_83 = stok[tile_stage * 192 + 128 + (unsigned int)tok_s_80];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_67 + tok_s_80 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_83 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_82)) : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _tmem_load_10[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 192 + 160));
                    float _tmem_load_11[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                        : "r"(taddr + (unsigned int)(epi_warp * 32 + 16 << 16) + acc_stage * 192 + 160));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    fin_scl[0] = sscale[tile_stage * 193 + 160 + (unsigned int)(tok_pair * 2)];
                    fin_scl[1] = sscale[tile_stage * 193 + 160 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[2] = sscale[tile_stage * 193 + 160 + 8 + (unsigned int)(tok_pair * 2)];
                    fin_scl[3] = sscale[tile_stage * 193 + 160 + 8 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[4] = sscale[tile_stage * 193 + 160 + 16 + (unsigned int)(tok_pair * 2)];
                    fin_scl[5] = sscale[tile_stage * 193 + 160 + 16 + (unsigned int)(tok_pair * 2) + 1];
                    fin_scl[6] = sscale[tile_stage * 193 + 160 + 24 + (unsigned int)(tok_pair * 2)];
                    fin_scl[7] = sscale[tile_stage * 193 + 160 + 24 + (unsigned int)(tok_pair * 2) + 1];
                    wvals[0] = _tmem_load_10[0] * fin_scl[0];
                    wvals[16] = _tmem_load_11[0] * fin_scl[0];
                    wvals[1] = _tmem_load_10[1] * fin_scl[1];
                    wvals[17] = _tmem_load_11[1] * fin_scl[1];
                    wvals[2] = _tmem_load_10[2] * fin_scl[0];
                    wvals[18] = _tmem_load_11[2] * fin_scl[0];
                    wvals[3] = _tmem_load_10[3] * fin_scl[1];
                    wvals[19] = _tmem_load_11[3] * fin_scl[1];
                    wvals[4] = _tmem_load_10[4] * fin_scl[2];
                    wvals[20] = _tmem_load_11[4] * fin_scl[2];
                    wvals[5] = _tmem_load_10[5] * fin_scl[3];
                    wvals[21] = _tmem_load_11[5] * fin_scl[3];
                    wvals[6] = _tmem_load_10[6] * fin_scl[2];
                    wvals[22] = _tmem_load_11[6] * fin_scl[2];
                    wvals[7] = _tmem_load_10[7] * fin_scl[3];
                    wvals[23] = _tmem_load_11[7] * fin_scl[3];
                    wvals[8] = _tmem_load_10[8] * fin_scl[4];
                    wvals[24] = _tmem_load_11[8] * fin_scl[4];
                    wvals[9] = _tmem_load_10[9] * fin_scl[5];
                    wvals[25] = _tmem_load_11[9] * fin_scl[5];
                    wvals[10] = _tmem_load_10[10] * fin_scl[4];
                    wvals[26] = _tmem_load_11[10] * fin_scl[4];
                    wvals[11] = _tmem_load_10[11] * fin_scl[5];
                    wvals[27] = _tmem_load_11[11] * fin_scl[5];
                    wvals[12] = _tmem_load_10[12] * fin_scl[6];
                    wvals[28] = _tmem_load_11[12] * fin_scl[6];
                    wvals[13] = _tmem_load_10[13] * fin_scl[7];
                    wvals[29] = _tmem_load_11[13] * fin_scl[7];
                    wvals[14] = _tmem_load_10[14] * fin_scl[6];
                    wvals[30] = _tmem_load_11[14] * fin_scl[6];
                    wvals[15] = _tmem_load_10[15] * fin_scl[7];
                    wvals[31] = _tmem_load_11[15] * fin_scl[7];
                    uint32_t wvals_bf16_84[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wvals[_lp*2 + 0], wvals[_lp*2+1 + 0]));
                        wvals_bf16_84[_lp] = *(uint32_t*)&_bf2;
                    }
                    int stag_buf_85 = stag_addr + 8704;
                    uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(stag_buf_85 + mat_row * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[0])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[1])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[8])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[9]))
                        : "memory");
                    uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(stag_buf_85 + (8 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[2])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[3])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[10])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(stag_buf_85 + (16 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[4])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[5])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[12])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[13]))
                        : "memory");
                    uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(stag_buf_85 + (24 + mat_row) * 272 + mat_col_bytes);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[6])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[7])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[14])), "r"(*reinterpret_cast<const uint32_t*>(&wvals_bf16_84[15]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int tok_s_86 = fin_tok;
                    int prow_f_87 = row_base + 160 + tok_s_86;
                    int ok_f_88 = (int)(prow_f_87 < mn_limit);
                    int tok_f_89 = stok[tile_stage * 192 + 160 + (unsigned int)tok_s_86];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_85 + tok_s_86 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_89 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_88)) : "memory");
                    int tok_s_90 = fin_tok + 8;
                    int prow_f_91 = row_base + 160 + tok_s_90;
                    int ok_f_92 = (int)(prow_f_91 < mn_limit);
                    int tok_f_93 = stok[tile_stage * 192 + 160 + (unsigned int)tok_s_90];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_85 + tok_s_90 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_93 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_92)) : "memory");
                    int tok_s_94 = fin_tok + 16;
                    int prow_f_95 = row_base + 160 + tok_s_94;
                    int ok_f_96 = (int)(prow_f_95 < mn_limit);
                    int tok_f_97 = stok[tile_stage * 192 + 160 + (unsigned int)tok_s_94];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_85 + tok_s_94 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_97 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_96)) : "memory");
                    int tok_s_98 = fin_tok + 24;
                    int prow_f_99 = row_base + 160 + tok_s_98;
                    int ok_f_100 = (int)(prow_f_99 < mn_limit);
                    int tok_f_101 = stok[tile_stage * 192 + 160 + (unsigned int)tok_s_98];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&fin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fin_words[(0) + 3]))
                        : "r"(stag_buf_85 + tok_s_98 * 272 + fin_chunk * 16));
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4}; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_f_101 * out_cols + h0 + 8 * fin_chunk])), "r"((unsigned int)(fin_words[0])), "r"((unsigned int)(fin_words[1])), "r"((unsigned int)(fin_words[2])), "r"((unsigned int)(fin_words[3])), "r"((unsigned int)(ok_f_100)) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((acc_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                nst_epilogue = nst_epilogue + 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 7];
                info[1] = sinfo[tile_stage * 7 + 1];
                info[2] = sinfo[tile_stage * 7 + 2];
                info[3] = sinfo[tile_stage * 7 + 3];
                info[4] = sinfo[tile_stage * 7 + 4];
                info[5] = sinfo[tile_stage * 7 + 5];
                info[6] = sinfo[tile_stage * 7 + 6];
                meta_alpha = sscale[tile_stage * 193 + 192];
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int nrec_mma = 0;
            int nst_mma = 0;
            unsigned int sa = 0;
            unsigned int sb = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int tile_stage_1 = 0;
            unsigned int pha = 0;
            unsigned int ab_tok = 1;
            int info_1[7];
            unsigned int _phase_tile_full_1 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
            info_1[0] = sinfo[tile_stage_1 * 7];
            info_1[1] = sinfo[tile_stage_1 * 7 + 1];
            info_1[2] = sinfo[tile_stage_1 * 7 + 2];
            info_1[3] = sinfo[tile_stage_1 * 7 + 3];
            info_1[4] = sinfo[tile_stage_1 * 7 + 4];
            info_1[5] = sinfo[tile_stage_1 * 7 + 5];
            info_1[6] = sinfo[tile_stage_1 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
            tile_stage_1 += 1;
            if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            nrec_mma = nrec_mma + 1;
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_b_relay_full = 0;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                if (cta_rank == 0) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                    ab_tok = _mbar_token_0;
                    mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                    mbarrier_wait(b_relay_full_addr + (sb) * 8, _phase_b_relay_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64 + 32)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                        }
                        int _mma_a_lo_0 = (_mma_base_lo_0) + (sa) * 1024;
                        int _mma_b_lo_0 = (_mma_base_lo_1) + (sb) * 768;
                        {
                            uint32_t a_desc_lo = (uint32_t)_mma_a_lo_0;
                            uint32_t b_desc_lo = (uint32_t)_mma_b_lo_0;

                            tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                0x10b00280U, tmem_sf_a, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                            tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                0x30b00290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                0x50b002a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                0x70b002b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                    }
                    elect_commit_cg2_multicast(ab_free_addr + (sa) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(b_free_addr + (sb) * 8, (uint16_t)(3));
                    sa += 1;
                    if (sa == 6) { sa = 0; pha ^= 1; }
                    sb += 1;
                    if (sb == 6) { sb = 0; _phase_b_relay_full ^= 1; }
                    nst_mma = nst_mma + 1;
                    ab_tok = 1;
                    if (k_tiles > 1) {
                        uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                        ab_tok = _mbar_token_1;
                    }
                    #pragma unroll 1
                    for (int k = 1; k < k_tiles; k++) {
                        mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                        mbarrier_wait(b_relay_full_addr + (sb) * 8, _phase_b_relay_full);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64 + 32)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                            }
                            int _mma_a_lo_1 = (_mma_base_lo_0) + (sa) * 1024;
                            int _mma_b_lo_1 = (_mma_base_lo_1) + (sb) * 768;
                            {
                                uint32_t a_desc_lo = (uint32_t)_mma_a_lo_1;
                                uint32_t b_desc_lo = (uint32_t)_mma_b_lo_1;

                                tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                    0x10b00280U, tmem_sf_a, tmem_sf_b, 1);
                                tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                    0x30b00290U, tmem_sf_a, tmem_sf_b, 1);
                                tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                    0x50b002a0U, tmem_sf_a, tmem_sf_b, 1);
                                tcgen05_mma_mxf8_bs_cta2_elect((tmem_acc + (acc_stage_1 * 192)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                    0x70b002b0U, tmem_sf_a, tmem_sf_b, 1);
                            }
                        }
                        elect_commit_cg2_multicast(ab_free_addr + (sa) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(b_free_addr + (sb) * 8, (uint16_t)(3));
                        sa += 1;
                        if (sa == 6) { sa = 0; pha ^= 1; }
                        sb += 1;
                        if (sb == 6) { sb = 0; _phase_b_relay_full ^= 1; }
                        nst_mma = nst_mma + 1;
                        ab_tok = 1;
                        if (k + 1 < k_tiles) {
                            uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                            ab_tok = _mbar_token_2;
                        }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                    nst_mma = nst_mma + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
                info_1[0] = sinfo[tile_stage_1 * 7];
                info_1[1] = sinfo[tile_stage_1 * 7 + 1];
                info_1[2] = sinfo[tile_stage_1 * 7 + 2];
                info_1[3] = sinfo[tile_stage_1 * 7 + 3];
                info_1[4] = sinfo[tile_stage_1 * 7 + 4];
                info_1[5] = sinfo[tile_stage_1 * 7 + 5];
                info_1[6] = sinfo[tile_stage_1 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
                tile_stage_1 += 1;
                if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
                nrec_mma = nrec_mma + 1;
            }
        }
    }
    // ---- Role: tma ----
    if (warp == 5) {
        { // tma_main
            int nrec_tma = 0;
            int nst_tma = 0;
            unsigned int stage = 0;
            unsigned int tile_stage_2 = 0;
            int info_2[7];
            unsigned int _phase_tile_full_2 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
            info_2[0] = sinfo[tile_stage_2 * 7];
            info_2[1] = sinfo[tile_stage_2 * 7 + 1];
            info_2[2] = sinfo[tile_stage_2 * 7 + 2];
            info_2[3] = sinfo[tile_stage_2 * 7 + 3];
            info_2[4] = sinfo[tile_stage_2 * 7 + 4];
            info_2[5] = sinfo[tile_stage_2 * 7 + 5];
            info_2[6] = sinfo[tile_stage_2 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
            tile_stage_2 += 1;
            if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            nrec_tma = nrec_tma + 1;
            int batch[1];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                batch[0] = info_2[2] * num_m_tiles + (info_2[0] * 2 + cta_rank);
                int row_base_tma = info_2[1] * 64;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            if (cta_rank == 0) {
                                mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 17408);
                            }
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + stage * 16384), "l"((&A)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(((ab_full_addr + (stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + stage * 512), "l"((&SFA)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(((ab_full_addr + (stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_ab_free ^= 1; }
                    nst_tma = nst_tma + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
                info_2[0] = sinfo[tile_stage_2 * 7];
                info_2[1] = sinfo[tile_stage_2 * 7 + 1];
                info_2[2] = sinfo[tile_stage_2 * 7 + 2];
                info_2[3] = sinfo[tile_stage_2 * 7 + 3];
                info_2[4] = sinfo[tile_stage_2 * 7 + 4];
                info_2[5] = sinfo[tile_stage_2 * 7 + 5];
                info_2[6] = sinfo[tile_stage_2 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
                tile_stage_2 += 1;
                if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
                nrec_tma = nrec_tma + 1;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 6) {
        { // scheduler_main
            int nrec_scheduler = 0;
            int nst_scheduler = 0;
            unsigned int tile_stage_3 = 0;
            int num_valid = num_non_exiting_tiles[0];
            int m_chunks = (num_m_tiles + 1) / 2;
            int total_items = m_chunks * group_capacity;
            int sched_first = bid / 2;
            int sched_step = num_bids / 2;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int item = sched_first; item < total_items; item += sched_step) {
                int row_group = item / m_chunks;
                int m_tile = item - row_group * m_chunks;
                if (row_group >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                int sched_row_group = tile_idx_to_row_group[row_group];
                int lookup = sched_row_group * 64 / 128;
                int lookup_limit = (sched_row_group * 64 + 191) / 128;
                int expert = tile_idx_to_expert_idx[lookup];
                int mn_limit_1 = tile_idx_to_mn_limit[lookup_limit];
                if (lane == 0) {
                    sscale[tile_stage_3 * 193 + 192] = alpha[expert];
                }
                int meta_col = lane;
                if (meta_col < 192) {
                    int meta_prow = sched_row_group * 64 + meta_col;
                    int meta_valid = (int)(meta_prow < mn_limit_1);
                    int meta_expanded = permuted_idx_to_expanded_idx[meta_prow];
                    int _max_0 = ((meta_expanded) > (0) ? (meta_expanded) : (0));
                    int meta_safe = _max_0;
                    int meta_token = meta_safe / top_k;
                    int meta_topk = meta_safe - meta_token * top_k;
                    int meta_gather = meta_token * meta_valid;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col] = token_final_scales[meta_gather * top_k + meta_topk];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col] = meta_token;
                }
                int meta_col_0 = 32 + lane;
                if (meta_col_0 < 192) {
                    int meta_prow_1 = sched_row_group * 64 + meta_col_0;
                    int meta_valid_1 = (int)(meta_prow_1 < mn_limit_1);
                    int meta_expanded_1 = permuted_idx_to_expanded_idx[meta_prow_1];
                    int _max_1 = ((meta_expanded_1) > (0) ? (meta_expanded_1) : (0));
                    int meta_safe_1 = _max_1;
                    int meta_token_1 = meta_safe_1 / top_k;
                    int meta_topk_1 = meta_safe_1 - meta_token_1 * top_k;
                    int meta_gather_1 = meta_token_1 * meta_valid_1;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col_0] = token_final_scales[meta_gather_1 * top_k + meta_topk_1];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col_0] = meta_token_1;
                }
                int meta_col_1 = 64 + lane;
                if (meta_col_1 < 192) {
                    int meta_prow_2 = sched_row_group * 64 + meta_col_1;
                    int meta_valid_2 = (int)(meta_prow_2 < mn_limit_1);
                    int meta_expanded_2 = permuted_idx_to_expanded_idx[meta_prow_2];
                    int _max_2 = ((meta_expanded_2) > (0) ? (meta_expanded_2) : (0));
                    int meta_safe_2 = _max_2;
                    int meta_token_2 = meta_safe_2 / top_k;
                    int meta_topk_2 = meta_safe_2 - meta_token_2 * top_k;
                    int meta_gather_2 = meta_token_2 * meta_valid_2;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col_1] = token_final_scales[meta_gather_2 * top_k + meta_topk_2];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col_1] = meta_token_2;
                }
                int meta_col_2 = 96 + lane;
                if (meta_col_2 < 192) {
                    int meta_prow_3 = sched_row_group * 64 + meta_col_2;
                    int meta_valid_3 = (int)(meta_prow_3 < mn_limit_1);
                    int meta_expanded_3 = permuted_idx_to_expanded_idx[meta_prow_3];
                    int _max_3 = ((meta_expanded_3) > (0) ? (meta_expanded_3) : (0));
                    int meta_safe_3 = _max_3;
                    int meta_token_3 = meta_safe_3 / top_k;
                    int meta_topk_3 = meta_safe_3 - meta_token_3 * top_k;
                    int meta_gather_3 = meta_token_3 * meta_valid_3;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col_2] = token_final_scales[meta_gather_3 * top_k + meta_topk_3];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col_2] = meta_token_3;
                }
                int meta_col_3 = 128 + lane;
                if (meta_col_3 < 192) {
                    int meta_prow_4 = sched_row_group * 64 + meta_col_3;
                    int meta_valid_4 = (int)(meta_prow_4 < mn_limit_1);
                    int meta_expanded_4 = permuted_idx_to_expanded_idx[meta_prow_4];
                    int _max_4 = ((meta_expanded_4) > (0) ? (meta_expanded_4) : (0));
                    int meta_safe_4 = _max_4;
                    int meta_token_4 = meta_safe_4 / top_k;
                    int meta_topk_4 = meta_safe_4 - meta_token_4 * top_k;
                    int meta_gather_4 = meta_token_4 * meta_valid_4;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col_3] = token_final_scales[meta_gather_4 * top_k + meta_topk_4];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col_3] = meta_token_4;
                }
                int meta_col_4 = 160 + lane;
                if (meta_col_4 < 192) {
                    int meta_prow_5 = sched_row_group * 64 + meta_col_4;
                    int meta_valid_5 = (int)(meta_prow_5 < mn_limit_1);
                    int meta_expanded_5 = permuted_idx_to_expanded_idx[meta_prow_5];
                    int _max_5 = ((meta_expanded_5) > (0) ? (meta_expanded_5) : (0));
                    int meta_safe_5 = _max_5;
                    int meta_token_5 = meta_safe_5 / top_k;
                    int meta_topk_5 = meta_safe_5 - meta_token_5 * top_k;
                    int meta_gather_5 = meta_token_5 * meta_valid_5;
                    sscale[tile_stage_3 * 193 + (unsigned int)meta_col_4] = token_final_scales[meta_gather_5 * top_k + meta_topk_5];
                    stok[tile_stage_3 * 192 + (unsigned int)meta_col_4] = meta_token_5;
                }
                if (elect_sync()) {
                    sinfo[tile_stage_3 * 7] = m_tile;
                    sinfo[tile_stage_3 * 7 + 1] = sched_row_group;
                    sinfo[tile_stage_3 * 7 + 2] = expert;
                    sinfo[tile_stage_3 * 7 + 3] = 1;
                    sinfo[tile_stage_3 * 7 + 4] = mn_limit_1;
                    sinfo[tile_stage_3 * 7 + 5] = 0;
                    sinfo[tile_stage_3 * 7 + 6] = k_tiles;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
            }
            mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
            if (elect_sync()) {
                sinfo[tile_stage_3 * 7] = 0;
                sinfo[tile_stage_3 * 7 + 1] = 0;
                sinfo[tile_stage_3 * 7 + 2] = -1;
                sinfo[tile_stage_3 * 7 + 3] = 0;
                sinfo[tile_stage_3 * 7 + 4] = 0;
                sinfo[tile_stage_3 * 7 + 5] = 0;
                sinfo[tile_stage_3 * 7 + 6] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 4, 32;" ::: "memory");
            mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
            tile_stage_3 += 1;
            if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
        }
    }
    // ---- Role: gather ----
    if (warp >= 7 && warp <= 10) {
        { // gather_main
            int nrec_gather = 0;
            int nst_gather = 0;
            unsigned int stage_1 = 0;
            unsigned int tile_stage_4 = 0;
            int info_3[7];
            unsigned int _phase_tile_full_3 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
            info_3[0] = sinfo[tile_stage_4 * 7];
            info_3[1] = sinfo[tile_stage_4 * 7 + 1];
            info_3[2] = sinfo[tile_stage_4 * 7 + 2];
            info_3[3] = sinfo[tile_stage_4 * 7 + 3];
            info_3[4] = sinfo[tile_stage_4 * 7 + 4];
            info_3[5] = sinfo[tile_stage_4 * 7 + 5];
            info_3[6] = sinfo[tile_stage_4 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
            tile_stage_4 += 1;
            if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
            nrec_gather = nrec_gather + 1;
            int gather_sub = warp - 7;
            int lane_0_1 = lane;
            int chunk = lane_0_1 % 8;
            int row_in_pass = lane_0_1 / 8;
            int row_src[6];
            int row_ok[6];
            int sf_src[2];
            int sf_ok[2];
            int cta_row0 = cta_rank * 96;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 64;
                int mn_limit_2 = info_3[4];
                int row = gather_sub * 4 + row_in_pass;
                int prow = row_base_1 + cta_row0 + row;
                int ok = (int)(prow < mn_limit_2);
                row_src[0] = prow * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 4) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int ok_2 = (int)(prow_1 < mn_limit_2);
                row_src[1] = prow_1 * ok_2;
                row_ok[1] = ok_2;
                int row_3 = (gather_sub + 8) * 4 + row_in_pass;
                int prow_4 = row_base_1 + cta_row0 + row_3;
                int ok_5 = (int)(prow_4 < mn_limit_2);
                row_src[2] = prow_4 * ok_5;
                row_ok[2] = ok_5;
                int row_6 = (gather_sub + 12) * 4 + row_in_pass;
                int prow_7 = row_base_1 + cta_row0 + row_6;
                int ok_8 = (int)(prow_7 < mn_limit_2);
                row_src[3] = prow_7 * ok_8;
                row_ok[3] = ok_8;
                int row_9 = (gather_sub + 16) * 4 + row_in_pass;
                int prow_10 = row_base_1 + cta_row0 + row_9;
                int ok_11 = (int)(prow_10 < mn_limit_2);
                row_src[4] = prow_10 * ok_11;
                row_ok[4] = ok_11;
                int row_12 = (gather_sub + 20) * 4 + row_in_pass;
                int prow_13 = row_base_1 + cta_row0 + row_12;
                int ok_14 = (int)(prow_13 < mn_limit_2);
                row_src[5] = prow_13 * ok_14;
                row_ok[5] = ok_14;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int sok = (int)(sprow < mn_limit_2 && srow < 192);
                sf_src[0] = (sprow % 32 * 16 + sprow / 32 % 4 * 4 + sprow / 128 * (sf_cols * 128)) * sok;
                sf_ok[0] = sok;
                int srow_15 = (gather_sub + 4) * 32 + lane_0_1;
                int sprow_16 = row_base_1 + srow_15;
                int sok_17 = (int)(sprow_16 < mn_limit_2 && srow_15 < 192);
                sf_src[1] = (sprow_16 % 32 * 16 + sprow_16 / 32 % 4 * 4 + sprow_16 / 128 * (sf_cols * 128)) * sok_17;
                sf_ok[1] = sok_17;
                #pragma unroll 1
                for (int k_2 = 0; k_2 < k_tiles; k_2++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    int k0 = (info_3[5] + k_2) * 128;
                    int dst_off = (gather_sub * 4 + row_in_pass) * 128 + (chunk ^ (gather_sub * 4 + row_in_pass) % 8) * 16;
                    int src_off = row_src[0] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off), "l"(B + src_off), "r"((row_ok[0] != 0) ? 16 : 0));
                    }
                    int dst_off_0 = ((gather_sub + 4) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 4) * 4 + row_in_pass) % 8) * 16;
                    int src_off_1 = row_src[1] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off_0), "l"(B + src_off_1), "r"((row_ok[1] != 0) ? 16 : 0));
                    }
                    int dst_off_2 = ((gather_sub + 8) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 8) * 4 + row_in_pass) % 8) * 16;
                    int src_off_3 = row_src[2] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off_2), "l"(B + src_off_3), "r"((row_ok[2] != 0) ? 16 : 0));
                    }
                    int dst_off_4 = ((gather_sub + 12) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 12) * 4 + row_in_pass) % 8) * 16;
                    int src_off_5 = row_src[3] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off_4), "l"(B + src_off_5), "r"((row_ok[3] != 0) ? 16 : 0));
                    }
                    int dst_off_6 = ((gather_sub + 16) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 16) * 4 + row_in_pass) % 8) * 16;
                    int src_off_7 = row_src[4] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off_6), "l"(B + src_off_7), "r"((row_ok[4] != 0) ? 16 : 0));
                    }
                    int dst_off_8 = ((gather_sub + 20) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 20) * 4 + row_in_pass) % 8) * 16;
                    int src_off_9 = row_src[5] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 12288 + (unsigned int)dst_off_8), "l"(B + src_off_9), "r"((row_ok[5] != 0) ? 16 : 0));
                    }
                    int sf_src_off = sf_src[0] + (info_3[5] + k_2) * 512;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 1024 + (unsigned int)(gather_sub / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    int sf_src_off_10 = sf_src[1] + (info_3[5] + k_2) * 512;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 1024 + (unsigned int)((gather_sub + 4) / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)((gather_sub + 4) % 4 * 4)), "l"(SFB + sf_src_off_10), "r"((sf_ok[1] != 0) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                    stage_1 += 1;
                    if (stage_1 == 6) { stage_1 = 0; _phase_b_free ^= 1; }
                    nst_gather = nst_gather + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
                info_3[0] = sinfo[tile_stage_4 * 7];
                info_3[1] = sinfo[tile_stage_4 * 7 + 1];
                info_3[2] = sinfo[tile_stage_4 * 7 + 2];
                info_3[3] = sinfo[tile_stage_4 * 7 + 3];
                info_3[4] = sinfo[tile_stage_4 * 7 + 4];
                info_3[5] = sinfo[tile_stage_4 * 7 + 5];
                info_3[6] = sinfo[tile_stage_4 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
                tile_stage_4 += 1;
                if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
                nrec_gather = nrec_gather + 1;
            }
        }
    }
    // ---- Role: relay ----
    if (warp == 11) {
        { // relay_main
            unsigned int stage_2 = 0;
            unsigned int tile_stage_5 = 0;
            int info_4[7];
            unsigned int _phase_tile_full_4 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_5) * 8, _phase_tile_full_4);
            info_4[0] = sinfo[tile_stage_5 * 7];
            info_4[1] = sinfo[tile_stage_5 * 7 + 1];
            info_4[2] = sinfo[tile_stage_5 * 7 + 2];
            info_4[3] = sinfo[tile_stage_5 * 7 + 3];
            info_4[4] = sinfo[tile_stage_5 * 7 + 4];
            info_4[5] = sinfo[tile_stage_5 * 7 + 5];
            info_4[6] = sinfo[tile_stage_5 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_5) * 8);
            tile_stage_5 += 1;
            if (tile_stage_5 == 8) { tile_stage_5 = 0; _phase_tile_full_4 ^= 1; }
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int _tile_4 = 0; _tile_4 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_4++) {
                if (info_4[3] == 0) {
                    break;
                }
                #pragma unroll 1
                for (int k_3 = 0; k_3 < k_tiles; k_3++) {
                    mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((b_relay_full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 6) { stage_2 = 0; _phase_b_full ^= 1; }
                }
                mbarrier_wait(tile_full_addr + (tile_stage_5) * 8, _phase_tile_full_4);
                info_4[0] = sinfo[tile_stage_5 * 7];
                info_4[1] = sinfo[tile_stage_5 * 7 + 1];
                info_4[2] = sinfo[tile_stage_5 * 7 + 2];
                info_4[3] = sinfo[tile_stage_5 * 7 + 3];
                info_4[4] = sinfo[tile_stage_5 * 7 + 4];
                info_4[5] = sinfo[tile_stage_5 * 7 + 5];
                info_4[6] = sinfo[tile_stage_5 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_5) * 8);
                tile_stage_5 += 1;
                if (tile_stage_5 == 8) { tile_stage_5 = 0; _phase_tile_full_4 ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
