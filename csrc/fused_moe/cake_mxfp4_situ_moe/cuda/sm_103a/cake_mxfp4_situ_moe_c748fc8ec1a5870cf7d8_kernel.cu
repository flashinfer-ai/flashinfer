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
#define SMEM_SEXCHF_OFF 194832
#define SMEM_SEXCHF_STAGE_BYTES 8192
#define SMEM_SEXCHF_STRIDE 8192
#define SMEM_SACT_OFF 211328
#define SMEM_SACT_STAGE_BYTES 4608
#define SMEM_SACT_STRIDE 4608
#define SMEM_SCODE_OFF 220544
#define SMEM_SCODE_STAGE_BYTES 512
#define SMEM_SCODE_STRIDE 512
#define SMEM_TOTAL 221056
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
kernel_cake_mxfp4_situ_moe_c748fc8ec1a5870cf7d8(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, uint8_t* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
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
    float* sexchf = reinterpret_cast<float*>(smem_raw + 194832);
    const int sexchf_addr = smem + 194832;
    unsigned int* sact = reinterpret_cast<unsigned int*>(smem_raw + 211328);
    const int sact_addr = smem + 211328;
    unsigned int* scode = reinterpret_cast<unsigned int*>(smem_raw + 220544);
    const int scode_addr = smem + 220544;
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
            float frag[32];
            unsigned int gx[4];
            unsigned int w4[4];
            float famax[8];
            int t64 = epi_tidx % 64;
            int warp_in64 = t64 / 32;
            int st_tok = epi_tidx / 4;
            int st_chunk = epi_tidx % 4;
            int tok_pair = lane_0 % 4;
            int is_sf_writer = (int)(lane_0 < 4);
            float zero_pair[2];
            zero_pair[0] = 0.0f;
            zero_pair[1] = 0.0f;
            #pragma unroll 1
            for (int zi = bid * 128 + epi_tidx; zi < zero_words; zi += num_bids * 128) {
                {
                    float2 _v2 = make_float2(zero_pair[0 + 0], zero_pair[0 + 1]);
                    *reinterpret_cast<float2*>(zero_buf + (zi * 2) + 0) = _v2;
                }
            }
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
                int expert_e = info[2];
                float beta = situ_beta[expert_e];
                float _fdiv_rn_0 = __fdiv_rn(1.0f, beta);
                float inv_beta = _fdiv_rn_0;
                float linear_beta = situ_linear_beta[expert_e];
                float _fdiv_rn_1 = __fdiv_rn(1.0f, linear_beta);
                float inv_linear_beta = _fdiv_rn_1;
                int m_tile_out = info[0] * 2 + cta_rank;
                int j_col = m_tile_out * 64 + epi_tidx;
                int sf_kb = m_tile_out * 2 + epi_tidx / 32;
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    int j0 = m_tile_out * 64;
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
                    frag[0] = _tmem_load_0[0] * meta_alpha;
                    frag[16] = _tmem_load_1[0] * meta_alpha;
                    frag[1] = _tmem_load_0[1] * meta_alpha;
                    frag[17] = _tmem_load_1[1] * meta_alpha;
                    frag[2] = _tmem_load_0[2] * meta_alpha;
                    frag[18] = _tmem_load_1[2] * meta_alpha;
                    frag[3] = _tmem_load_0[3] * meta_alpha;
                    frag[19] = _tmem_load_1[3] * meta_alpha;
                    frag[4] = _tmem_load_0[4] * meta_alpha;
                    frag[20] = _tmem_load_1[4] * meta_alpha;
                    frag[5] = _tmem_load_0[5] * meta_alpha;
                    frag[21] = _tmem_load_1[5] * meta_alpha;
                    frag[6] = _tmem_load_0[6] * meta_alpha;
                    frag[22] = _tmem_load_1[6] * meta_alpha;
                    frag[7] = _tmem_load_0[7] * meta_alpha;
                    frag[23] = _tmem_load_1[7] * meta_alpha;
                    frag[8] = _tmem_load_0[8] * meta_alpha;
                    frag[24] = _tmem_load_1[8] * meta_alpha;
                    frag[9] = _tmem_load_0[9] * meta_alpha;
                    frag[25] = _tmem_load_1[9] * meta_alpha;
                    frag[10] = _tmem_load_0[10] * meta_alpha;
                    frag[26] = _tmem_load_1[10] * meta_alpha;
                    frag[11] = _tmem_load_0[11] * meta_alpha;
                    frag[27] = _tmem_load_1[11] * meta_alpha;
                    frag[12] = _tmem_load_0[12] * meta_alpha;
                    frag[28] = _tmem_load_1[12] * meta_alpha;
                    frag[13] = _tmem_load_0[13] * meta_alpha;
                    frag[29] = _tmem_load_1[13] * meta_alpha;
                    frag[14] = _tmem_load_0[14] * meta_alpha;
                    frag[30] = _tmem_load_1[14] * meta_alpha;
                    frag[15] = _tmem_load_0[15] * meta_alpha;
                    frag[31] = _tmem_load_1[15] * meta_alpha;
                    int exchf_buf = sexchf_addr;
                    int sact_buf = sact_addr;
                    if (is_gate_lane != 0) {
                        float x_g = frag[0];
                        float _exp2_0 = approx_exp2(x_g * -1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float sig_g = _rcp_0;
                        float _tanh_approx_0;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(x_g * inv_beta));
                        frag[0] = beta * _tanh_approx_0 * sig_g;
                        float x_g_0 = frag[1];
                        float _exp2_1 = approx_exp2(x_g_0 * -1.4426950408889634f);
                        float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                        float sig_g_1 = _rcp_1;
                        float _tanh_approx_1;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(x_g_0 * inv_beta));
                        frag[1] = beta * _tanh_approx_1 * sig_g_1;
                        float x_g_2 = frag[2];
                        float _exp2_2 = approx_exp2(x_g_2 * -1.4426950408889634f);
                        float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                        float sig_g_3 = _rcp_2;
                        float _tanh_approx_2;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(x_g_2 * inv_beta));
                        frag[2] = beta * _tanh_approx_2 * sig_g_3;
                        float x_g_4 = frag[3];
                        float _exp2_3 = approx_exp2(x_g_4 * -1.4426950408889634f);
                        float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                        float sig_g_5 = _rcp_3;
                        float _tanh_approx_3;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(x_g_4 * inv_beta));
                        frag[3] = beta * _tanh_approx_3 * sig_g_5;
                        float x_g_6 = frag[4];
                        float _exp2_4 = approx_exp2(x_g_6 * -1.4426950408889634f);
                        float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                        float sig_g_7 = _rcp_4;
                        float _tanh_approx_4;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(x_g_6 * inv_beta));
                        frag[4] = beta * _tanh_approx_4 * sig_g_7;
                        float x_g_8 = frag[5];
                        float _exp2_5 = approx_exp2(x_g_8 * -1.4426950408889634f);
                        float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                        float sig_g_9 = _rcp_5;
                        float _tanh_approx_5;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(x_g_8 * inv_beta));
                        frag[5] = beta * _tanh_approx_5 * sig_g_9;
                        float x_g_10 = frag[6];
                        float _exp2_6 = approx_exp2(x_g_10 * -1.4426950408889634f);
                        float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                        float sig_g_11 = _rcp_6;
                        float _tanh_approx_6;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(x_g_10 * inv_beta));
                        frag[6] = beta * _tanh_approx_6 * sig_g_11;
                        float x_g_12 = frag[7];
                        float _exp2_7 = approx_exp2(x_g_12 * -1.4426950408889634f);
                        float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                        float sig_g_13 = _rcp_7;
                        float _tanh_approx_7;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_7) : "f"(x_g_12 * inv_beta));
                        frag[7] = beta * _tanh_approx_7 * sig_g_13;
                        float x_g_14 = frag[8];
                        float _exp2_8 = approx_exp2(x_g_14 * -1.4426950408889634f);
                        float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                        float sig_g_15 = _rcp_8;
                        float _tanh_approx_8;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_8) : "f"(x_g_14 * inv_beta));
                        frag[8] = beta * _tanh_approx_8 * sig_g_15;
                        float x_g_16 = frag[9];
                        float _exp2_9 = approx_exp2(x_g_16 * -1.4426950408889634f);
                        float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                        float sig_g_17 = _rcp_9;
                        float _tanh_approx_9;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_9) : "f"(x_g_16 * inv_beta));
                        frag[9] = beta * _tanh_approx_9 * sig_g_17;
                        float x_g_18 = frag[10];
                        float _exp2_10 = approx_exp2(x_g_18 * -1.4426950408889634f);
                        float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                        float sig_g_19 = _rcp_10;
                        float _tanh_approx_10;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_10) : "f"(x_g_18 * inv_beta));
                        frag[10] = beta * _tanh_approx_10 * sig_g_19;
                        float x_g_20 = frag[11];
                        float _exp2_11 = approx_exp2(x_g_20 * -1.4426950408889634f);
                        float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                        float sig_g_21 = _rcp_11;
                        float _tanh_approx_11;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_11) : "f"(x_g_20 * inv_beta));
                        frag[11] = beta * _tanh_approx_11 * sig_g_21;
                        float x_g_22 = frag[12];
                        float _exp2_12 = approx_exp2(x_g_22 * -1.4426950408889634f);
                        float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                        float sig_g_23 = _rcp_12;
                        float _tanh_approx_12;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_12) : "f"(x_g_22 * inv_beta));
                        frag[12] = beta * _tanh_approx_12 * sig_g_23;
                        float x_g_24 = frag[13];
                        float _exp2_13 = approx_exp2(x_g_24 * -1.4426950408889634f);
                        float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                        float sig_g_25 = _rcp_13;
                        float _tanh_approx_13;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_13) : "f"(x_g_24 * inv_beta));
                        frag[13] = beta * _tanh_approx_13 * sig_g_25;
                        float x_g_26 = frag[14];
                        float _exp2_14 = approx_exp2(x_g_26 * -1.4426950408889634f);
                        float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                        float sig_g_27 = _rcp_14;
                        float _tanh_approx_14;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_14) : "f"(x_g_26 * inv_beta));
                        frag[14] = beta * _tanh_approx_14 * sig_g_27;
                        float x_g_28 = frag[15];
                        float _exp2_15 = approx_exp2(x_g_28 * -1.4426950408889634f);
                        float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                        float sig_g_29 = _rcp_15;
                        float _tanh_approx_15;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_15) : "f"(x_g_28 * inv_beta));
                        frag[15] = beta * _tanh_approx_15 * sig_g_29;
                        float x_g_30 = frag[16];
                        float _exp2_16 = approx_exp2(x_g_30 * -1.4426950408889634f);
                        float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                        float sig_g_31 = _rcp_16;
                        float _tanh_approx_16;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_16) : "f"(x_g_30 * inv_beta));
                        frag[16] = beta * _tanh_approx_16 * sig_g_31;
                        float x_g_32 = frag[17];
                        float _exp2_17 = approx_exp2(x_g_32 * -1.4426950408889634f);
                        float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                        float sig_g_33 = _rcp_17;
                        float _tanh_approx_17;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_17) : "f"(x_g_32 * inv_beta));
                        frag[17] = beta * _tanh_approx_17 * sig_g_33;
                        float x_g_34 = frag[18];
                        float _exp2_18 = approx_exp2(x_g_34 * -1.4426950408889634f);
                        float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                        float sig_g_35 = _rcp_18;
                        float _tanh_approx_18;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_18) : "f"(x_g_34 * inv_beta));
                        frag[18] = beta * _tanh_approx_18 * sig_g_35;
                        float x_g_36 = frag[19];
                        float _exp2_19 = approx_exp2(x_g_36 * -1.4426950408889634f);
                        float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                        float sig_g_37 = _rcp_19;
                        float _tanh_approx_19;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_19) : "f"(x_g_36 * inv_beta));
                        frag[19] = beta * _tanh_approx_19 * sig_g_37;
                        float x_g_38 = frag[20];
                        float _exp2_20 = approx_exp2(x_g_38 * -1.4426950408889634f);
                        float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                        float sig_g_39 = _rcp_20;
                        float _tanh_approx_20;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_20) : "f"(x_g_38 * inv_beta));
                        frag[20] = beta * _tanh_approx_20 * sig_g_39;
                        float x_g_40 = frag[21];
                        float _exp2_21 = approx_exp2(x_g_40 * -1.4426950408889634f);
                        float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                        float sig_g_41 = _rcp_21;
                        float _tanh_approx_21;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_21) : "f"(x_g_40 * inv_beta));
                        frag[21] = beta * _tanh_approx_21 * sig_g_41;
                        float x_g_42 = frag[22];
                        float _exp2_22 = approx_exp2(x_g_42 * -1.4426950408889634f);
                        float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                        float sig_g_43 = _rcp_22;
                        float _tanh_approx_22;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_22) : "f"(x_g_42 * inv_beta));
                        frag[22] = beta * _tanh_approx_22 * sig_g_43;
                        float x_g_44 = frag[23];
                        float _exp2_23 = approx_exp2(x_g_44 * -1.4426950408889634f);
                        float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                        float sig_g_45 = _rcp_23;
                        float _tanh_approx_23;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_23) : "f"(x_g_44 * inv_beta));
                        frag[23] = beta * _tanh_approx_23 * sig_g_45;
                        float x_g_46 = frag[24];
                        float _exp2_24 = approx_exp2(x_g_46 * -1.4426950408889634f);
                        float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                        float sig_g_47 = _rcp_24;
                        float _tanh_approx_24;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_24) : "f"(x_g_46 * inv_beta));
                        frag[24] = beta * _tanh_approx_24 * sig_g_47;
                        float x_g_48 = frag[25];
                        float _exp2_25 = approx_exp2(x_g_48 * -1.4426950408889634f);
                        float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                        float sig_g_49 = _rcp_25;
                        float _tanh_approx_25;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_25) : "f"(x_g_48 * inv_beta));
                        frag[25] = beta * _tanh_approx_25 * sig_g_49;
                        float x_g_50 = frag[26];
                        float _exp2_26 = approx_exp2(x_g_50 * -1.4426950408889634f);
                        float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                        float sig_g_51 = _rcp_26;
                        float _tanh_approx_26;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_26) : "f"(x_g_50 * inv_beta));
                        frag[26] = beta * _tanh_approx_26 * sig_g_51;
                        float x_g_52 = frag[27];
                        float _exp2_27 = approx_exp2(x_g_52 * -1.4426950408889634f);
                        float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                        float sig_g_53 = _rcp_27;
                        float _tanh_approx_27;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_27) : "f"(x_g_52 * inv_beta));
                        frag[27] = beta * _tanh_approx_27 * sig_g_53;
                        float x_g_54 = frag[28];
                        float _exp2_28 = approx_exp2(x_g_54 * -1.4426950408889634f);
                        float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                        float sig_g_55 = _rcp_28;
                        float _tanh_approx_28;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_28) : "f"(x_g_54 * inv_beta));
                        frag[28] = beta * _tanh_approx_28 * sig_g_55;
                        float x_g_56 = frag[29];
                        float _exp2_29 = approx_exp2(x_g_56 * -1.4426950408889634f);
                        float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                        float sig_g_57 = _rcp_29;
                        float _tanh_approx_29;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_29) : "f"(x_g_56 * inv_beta));
                        frag[29] = beta * _tanh_approx_29 * sig_g_57;
                        float x_g_58 = frag[30];
                        float _exp2_30 = approx_exp2(x_g_58 * -1.4426950408889634f);
                        float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                        float sig_g_59 = _rcp_30;
                        float _tanh_approx_30;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_30) : "f"(x_g_58 * inv_beta));
                        frag[30] = beta * _tanh_approx_30 * sig_g_59;
                        float x_g_60 = frag[31];
                        float _exp2_31 = approx_exp2(x_g_60 * -1.4426950408889634f);
                        float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                        float sig_g_61 = _rcp_31;
                        float _tanh_approx_31;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_31) : "f"(x_g_60 * inv_beta));
                        frag[31] = beta * _tanh_approx_31 * sig_g_61;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_32;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_32) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_32;
                        float _tanh_approx_33;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_33) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_33;
                        float _tanh_approx_34;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_34) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_34;
                        float _tanh_approx_35;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_35) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_35;
                        float _tanh_approx_36;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_36) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_36;
                        float _tanh_approx_37;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_37) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_37;
                        float _tanh_approx_38;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_38) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_38;
                        float _tanh_approx_39;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_39) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_39;
                        float _tanh_approx_40;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_40) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_40;
                        float _tanh_approx_41;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_41) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_41;
                        float _tanh_approx_42;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_42) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_42;
                        float _tanh_approx_43;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_43) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_43;
                        float _tanh_approx_44;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_44) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_44;
                        float _tanh_approx_45;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_45) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_45;
                        float _tanh_approx_46;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_46) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_46;
                        float _tanh_approx_47;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_47) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_47;
                        float _tanh_approx_48;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_48) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_48;
                        float _tanh_approx_49;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_49) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_49;
                        float _tanh_approx_50;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_50) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_50;
                        float _tanh_approx_51;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_51) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_51;
                        float _tanh_approx_52;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_52) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_52;
                        float _tanh_approx_53;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_53) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_53;
                        float _tanh_approx_54;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_54) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_54;
                        float _tanh_approx_55;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_55) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_55;
                        float _tanh_approx_56;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_56) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_56;
                        float _tanh_approx_57;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_57) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_57;
                        float _tanh_approx_58;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_58) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_58;
                        float _tanh_approx_59;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_59) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_59;
                        float _tanh_approx_60;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_60) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_60;
                        float _tanh_approx_61;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_61) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_61;
                        float _tanh_approx_62;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_62) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_62;
                        float _tanh_approx_63;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_63) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_63;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_0 = fmaxf(frag[0], -frag[0]);
                        float a_c = _fmax_0;
                        float _fmax_1 = fmaxf(frag[2], -frag[2]);
                        float _fmax_2 = fmaxf(a_c, _fmax_1);
                        a_c = _fmax_2;
                        float _fmax_3 = fmaxf(frag[16], -frag[16]);
                        float _fmax_4 = fmaxf(a_c, _fmax_3);
                        a_c = _fmax_4;
                        float _fmax_5 = fmaxf(frag[18], -frag[18]);
                        float _fmax_6 = fmaxf(a_c, _fmax_5);
                        a_c = _fmax_6;
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, a_c, 4);
                        float _fmax_7 = fmaxf(a_c, _shfl_xor_0);
                        a_c = _fmax_7;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a_c, 8);
                        float _fmax_8 = fmaxf(a_c, _shfl_xor_1);
                        a_c = _fmax_8;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a_c, 16);
                        float _fmax_9 = fmaxf(a_c, _shfl_xor_2);
                        a_c = _fmax_9;
                        uint16_t _ue8m0x2_f32_0;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(zero_f32), "f"(a_c * inv_fp8_max));
                        int code_full = (int)_ue8m0x2_f32_0;
                        int code = code_full & 255;
                        int _max_0 = ((254 - code) > (0) ? (254 - code) : (0));
                        unsigned int inv_bits = (unsigned int)(_max_0 << 23);
                        float inv_scale = __uint_as_float(inv_bits) * (float)(code != 0);
                        frag[0] = frag[0] * inv_scale;
                        frag[2] = frag[2] * inv_scale;
                        frag[16] = frag[16] * inv_scale;
                        frag[18] = frag[18] * inv_scale;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code;
                        }
                        float _fmax_10 = fmaxf(frag[1], -frag[1]);
                        float a_c_0 = _fmax_10;
                        float _fmax_11 = fmaxf(frag[3], -frag[3]);
                        float _fmax_12 = fmaxf(a_c_0, _fmax_11);
                        a_c_0 = _fmax_12;
                        float _fmax_13 = fmaxf(frag[17], -frag[17]);
                        float _fmax_14 = fmaxf(a_c_0, _fmax_13);
                        a_c_0 = _fmax_14;
                        float _fmax_15 = fmaxf(frag[19], -frag[19]);
                        float _fmax_16 = fmaxf(a_c_0, _fmax_15);
                        a_c_0 = _fmax_16;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a_c_0, 4);
                        float _fmax_17 = fmaxf(a_c_0, _shfl_xor_3);
                        a_c_0 = _fmax_17;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a_c_0, 8);
                        float _fmax_18 = fmaxf(a_c_0, _shfl_xor_4);
                        a_c_0 = _fmax_18;
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, a_c_0, 16);
                        float _fmax_19 = fmaxf(a_c_0, _shfl_xor_5);
                        a_c_0 = _fmax_19;
                        uint16_t _ue8m0x2_f32_1;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_1) : "f"(zero_f32), "f"(a_c_0 * inv_fp8_max));
                        int code_full_1 = (int)_ue8m0x2_f32_1;
                        int code_2 = code_full_1 & 255;
                        int _max_1 = ((254 - code_2) > (0) ? (254 - code_2) : (0));
                        unsigned int inv_bits_3 = (unsigned int)(_max_1 << 23);
                        float inv_scale_4 = __uint_as_float(inv_bits_3) * (float)(code_2 != 0);
                        frag[1] = frag[1] * inv_scale_4;
                        frag[3] = frag[3] * inv_scale_4;
                        frag[17] = frag[17] * inv_scale_4;
                        frag[19] = frag[19] * inv_scale_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2;
                        }
                        float _fmax_20 = fmaxf(frag[4], -frag[4]);
                        float a_c_5 = _fmax_20;
                        float _fmax_21 = fmaxf(frag[6], -frag[6]);
                        float _fmax_22 = fmaxf(a_c_5, _fmax_21);
                        a_c_5 = _fmax_22;
                        float _fmax_23 = fmaxf(frag[20], -frag[20]);
                        float _fmax_24 = fmaxf(a_c_5, _fmax_23);
                        a_c_5 = _fmax_24;
                        float _fmax_25 = fmaxf(frag[22], -frag[22]);
                        float _fmax_26 = fmaxf(a_c_5, _fmax_25);
                        a_c_5 = _fmax_26;
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, a_c_5, 4);
                        float _fmax_27 = fmaxf(a_c_5, _shfl_xor_6);
                        a_c_5 = _fmax_27;
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, a_c_5, 8);
                        float _fmax_28 = fmaxf(a_c_5, _shfl_xor_7);
                        a_c_5 = _fmax_28;
                        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, a_c_5, 16);
                        float _fmax_29 = fmaxf(a_c_5, _shfl_xor_8);
                        a_c_5 = _fmax_29;
                        uint16_t _ue8m0x2_f32_2;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_2) : "f"(zero_f32), "f"(a_c_5 * inv_fp8_max));
                        int code_full_6 = (int)_ue8m0x2_f32_2;
                        int code_7 = code_full_6 & 255;
                        int _max_2 = ((254 - code_7) > (0) ? (254 - code_7) : (0));
                        unsigned int inv_bits_8 = (unsigned int)(_max_2 << 23);
                        float inv_scale_9 = __uint_as_float(inv_bits_8) * (float)(code_7 != 0);
                        frag[4] = frag[4] * inv_scale_9;
                        frag[6] = frag[6] * inv_scale_9;
                        frag[20] = frag[20] * inv_scale_9;
                        frag[22] = frag[22] * inv_scale_9;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7;
                        }
                        float _fmax_30 = fmaxf(frag[5], -frag[5]);
                        float a_c_10 = _fmax_30;
                        float _fmax_31 = fmaxf(frag[7], -frag[7]);
                        float _fmax_32 = fmaxf(a_c_10, _fmax_31);
                        a_c_10 = _fmax_32;
                        float _fmax_33 = fmaxf(frag[21], -frag[21]);
                        float _fmax_34 = fmaxf(a_c_10, _fmax_33);
                        a_c_10 = _fmax_34;
                        float _fmax_35 = fmaxf(frag[23], -frag[23]);
                        float _fmax_36 = fmaxf(a_c_10, _fmax_35);
                        a_c_10 = _fmax_36;
                        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 4);
                        float _fmax_37 = fmaxf(a_c_10, _shfl_xor_9);
                        a_c_10 = _fmax_37;
                        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 8);
                        float _fmax_38 = fmaxf(a_c_10, _shfl_xor_10);
                        a_c_10 = _fmax_38;
                        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 16);
                        float _fmax_39 = fmaxf(a_c_10, _shfl_xor_11);
                        a_c_10 = _fmax_39;
                        uint16_t _ue8m0x2_f32_3;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_3) : "f"(zero_f32), "f"(a_c_10 * inv_fp8_max));
                        int code_full_11 = (int)_ue8m0x2_f32_3;
                        int code_12 = code_full_11 & 255;
                        int _max_3 = ((254 - code_12) > (0) ? (254 - code_12) : (0));
                        unsigned int inv_bits_13 = (unsigned int)(_max_3 << 23);
                        float inv_scale_14 = __uint_as_float(inv_bits_13) * (float)(code_12 != 0);
                        frag[5] = frag[5] * inv_scale_14;
                        frag[7] = frag[7] * inv_scale_14;
                        frag[21] = frag[21] * inv_scale_14;
                        frag[23] = frag[23] * inv_scale_14;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12;
                        }
                        float _fmax_40 = fmaxf(frag[8], -frag[8]);
                        float a_c_15 = _fmax_40;
                        float _fmax_41 = fmaxf(frag[10], -frag[10]);
                        float _fmax_42 = fmaxf(a_c_15, _fmax_41);
                        a_c_15 = _fmax_42;
                        float _fmax_43 = fmaxf(frag[24], -frag[24]);
                        float _fmax_44 = fmaxf(a_c_15, _fmax_43);
                        a_c_15 = _fmax_44;
                        float _fmax_45 = fmaxf(frag[26], -frag[26]);
                        float _fmax_46 = fmaxf(a_c_15, _fmax_45);
                        a_c_15 = _fmax_46;
                        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, a_c_15, 4);
                        float _fmax_47 = fmaxf(a_c_15, _shfl_xor_12);
                        a_c_15 = _fmax_47;
                        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, a_c_15, 8);
                        float _fmax_48 = fmaxf(a_c_15, _shfl_xor_13);
                        a_c_15 = _fmax_48;
                        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, a_c_15, 16);
                        float _fmax_49 = fmaxf(a_c_15, _shfl_xor_14);
                        a_c_15 = _fmax_49;
                        uint16_t _ue8m0x2_f32_4;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_4) : "f"(zero_f32), "f"(a_c_15 * inv_fp8_max));
                        int code_full_16 = (int)_ue8m0x2_f32_4;
                        int code_17 = code_full_16 & 255;
                        int _max_4 = ((254 - code_17) > (0) ? (254 - code_17) : (0));
                        unsigned int inv_bits_18 = (unsigned int)(_max_4 << 23);
                        float inv_scale_19 = __uint_as_float(inv_bits_18) * (float)(code_17 != 0);
                        frag[8] = frag[8] * inv_scale_19;
                        frag[10] = frag[10] * inv_scale_19;
                        frag[24] = frag[24] * inv_scale_19;
                        frag[26] = frag[26] * inv_scale_19;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17;
                        }
                        float _fmax_50 = fmaxf(frag[9], -frag[9]);
                        float a_c_20 = _fmax_50;
                        float _fmax_51 = fmaxf(frag[11], -frag[11]);
                        float _fmax_52 = fmaxf(a_c_20, _fmax_51);
                        a_c_20 = _fmax_52;
                        float _fmax_53 = fmaxf(frag[25], -frag[25]);
                        float _fmax_54 = fmaxf(a_c_20, _fmax_53);
                        a_c_20 = _fmax_54;
                        float _fmax_55 = fmaxf(frag[27], -frag[27]);
                        float _fmax_56 = fmaxf(a_c_20, _fmax_55);
                        a_c_20 = _fmax_56;
                        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, a_c_20, 4);
                        float _fmax_57 = fmaxf(a_c_20, _shfl_xor_15);
                        a_c_20 = _fmax_57;
                        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, a_c_20, 8);
                        float _fmax_58 = fmaxf(a_c_20, _shfl_xor_16);
                        a_c_20 = _fmax_58;
                        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, a_c_20, 16);
                        float _fmax_59 = fmaxf(a_c_20, _shfl_xor_17);
                        a_c_20 = _fmax_59;
                        uint16_t _ue8m0x2_f32_5;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_5) : "f"(zero_f32), "f"(a_c_20 * inv_fp8_max));
                        int code_full_21 = (int)_ue8m0x2_f32_5;
                        int code_22 = code_full_21 & 255;
                        int _max_5 = ((254 - code_22) > (0) ? (254 - code_22) : (0));
                        unsigned int inv_bits_23 = (unsigned int)(_max_5 << 23);
                        float inv_scale_24 = __uint_as_float(inv_bits_23) * (float)(code_22 != 0);
                        frag[9] = frag[9] * inv_scale_24;
                        frag[11] = frag[11] * inv_scale_24;
                        frag[25] = frag[25] * inv_scale_24;
                        frag[27] = frag[27] * inv_scale_24;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22;
                        }
                        float _fmax_60 = fmaxf(frag[12], -frag[12]);
                        float a_c_25 = _fmax_60;
                        float _fmax_61 = fmaxf(frag[14], -frag[14]);
                        float _fmax_62 = fmaxf(a_c_25, _fmax_61);
                        a_c_25 = _fmax_62;
                        float _fmax_63 = fmaxf(frag[28], -frag[28]);
                        float _fmax_64 = fmaxf(a_c_25, _fmax_63);
                        a_c_25 = _fmax_64;
                        float _fmax_65 = fmaxf(frag[30], -frag[30]);
                        float _fmax_66 = fmaxf(a_c_25, _fmax_65);
                        a_c_25 = _fmax_66;
                        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, a_c_25, 4);
                        float _fmax_67 = fmaxf(a_c_25, _shfl_xor_18);
                        a_c_25 = _fmax_67;
                        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, a_c_25, 8);
                        float _fmax_68 = fmaxf(a_c_25, _shfl_xor_19);
                        a_c_25 = _fmax_68;
                        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, a_c_25, 16);
                        float _fmax_69 = fmaxf(a_c_25, _shfl_xor_20);
                        a_c_25 = _fmax_69;
                        uint16_t _ue8m0x2_f32_6;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_6) : "f"(zero_f32), "f"(a_c_25 * inv_fp8_max));
                        int code_full_26 = (int)_ue8m0x2_f32_6;
                        int code_27 = code_full_26 & 255;
                        int _max_6 = ((254 - code_27) > (0) ? (254 - code_27) : (0));
                        unsigned int inv_bits_28 = (unsigned int)(_max_6 << 23);
                        float inv_scale_29 = __uint_as_float(inv_bits_28) * (float)(code_27 != 0);
                        frag[12] = frag[12] * inv_scale_29;
                        frag[14] = frag[14] * inv_scale_29;
                        frag[28] = frag[28] * inv_scale_29;
                        frag[30] = frag[30] * inv_scale_29;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27;
                        }
                        float _fmax_70 = fmaxf(frag[13], -frag[13]);
                        float a_c_30 = _fmax_70;
                        float _fmax_71 = fmaxf(frag[15], -frag[15]);
                        float _fmax_72 = fmaxf(a_c_30, _fmax_71);
                        a_c_30 = _fmax_72;
                        float _fmax_73 = fmaxf(frag[29], -frag[29]);
                        float _fmax_74 = fmaxf(a_c_30, _fmax_73);
                        a_c_30 = _fmax_74;
                        float _fmax_75 = fmaxf(frag[31], -frag[31]);
                        float _fmax_76 = fmaxf(a_c_30, _fmax_75);
                        a_c_30 = _fmax_76;
                        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, a_c_30, 4);
                        float _fmax_77 = fmaxf(a_c_30, _shfl_xor_21);
                        a_c_30 = _fmax_77;
                        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, a_c_30, 8);
                        float _fmax_78 = fmaxf(a_c_30, _shfl_xor_22);
                        a_c_30 = _fmax_78;
                        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, a_c_30, 16);
                        float _fmax_79 = fmaxf(a_c_30, _shfl_xor_23);
                        a_c_30 = _fmax_79;
                        uint16_t _ue8m0x2_f32_7;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_7) : "f"(zero_f32), "f"(a_c_30 * inv_fp8_max));
                        int code_full_31 = (int)_ue8m0x2_f32_7;
                        int code_32 = code_full_31 & 255;
                        int _max_7 = ((254 - code_32) > (0) ? (254 - code_32) : (0));
                        unsigned int inv_bits_33 = (unsigned int)(_max_7 << 23);
                        float inv_scale_34 = __uint_as_float(inv_bits_33) * (float)(code_32 != 0);
                        frag[13] = frag[13] * inv_scale_34;
                        frag[15] = frag[15] * inv_scale_34;
                        frag[29] = frag[29] * inv_scale_34;
                        frag[31] = frag[31] * inv_scale_34;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32;
                        }
                        uint32_t _fp8_0[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_0[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_0[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_0[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_0[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_0[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_0[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_0[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_0[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_0 = static_cast<uint32_t>(sact_buf + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_0), "r"(_fp8_0[0]), "r"(_fp8_0[1]), "r"(_fp8_0[2]), "r"(_fp8_0[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_1 = static_cast<uint32_t>(sact_buf + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_1), "r"(_fp8_0[4]), "r"(_fp8_0[5]), "r"(_fp8_0[6]), "r"(_fp8_0[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s = row_base + st_tok;
                    if (prow_s < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l = row_base + lane_0;
                        if (prow_l < mn_limit) {
                            unsigned int code_l = scode[warp_in64 * 32 + lane_0];
                            int sf_off_l = prow_l % 32 * 16 + prow_l / 32 % 4 * 4 + prow_l / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l) + (0)) = (unsigned char)(code_l);
                        }
                    }
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
                    frag[0] = _tmem_load_2[0] * meta_alpha;
                    frag[16] = _tmem_load_3[0] * meta_alpha;
                    frag[1] = _tmem_load_2[1] * meta_alpha;
                    frag[17] = _tmem_load_3[1] * meta_alpha;
                    frag[2] = _tmem_load_2[2] * meta_alpha;
                    frag[18] = _tmem_load_3[2] * meta_alpha;
                    frag[3] = _tmem_load_2[3] * meta_alpha;
                    frag[19] = _tmem_load_3[3] * meta_alpha;
                    frag[4] = _tmem_load_2[4] * meta_alpha;
                    frag[20] = _tmem_load_3[4] * meta_alpha;
                    frag[5] = _tmem_load_2[5] * meta_alpha;
                    frag[21] = _tmem_load_3[5] * meta_alpha;
                    frag[6] = _tmem_load_2[6] * meta_alpha;
                    frag[22] = _tmem_load_3[6] * meta_alpha;
                    frag[7] = _tmem_load_2[7] * meta_alpha;
                    frag[23] = _tmem_load_3[7] * meta_alpha;
                    frag[8] = _tmem_load_2[8] * meta_alpha;
                    frag[24] = _tmem_load_3[8] * meta_alpha;
                    frag[9] = _tmem_load_2[9] * meta_alpha;
                    frag[25] = _tmem_load_3[9] * meta_alpha;
                    frag[10] = _tmem_load_2[10] * meta_alpha;
                    frag[26] = _tmem_load_3[10] * meta_alpha;
                    frag[11] = _tmem_load_2[11] * meta_alpha;
                    frag[27] = _tmem_load_3[11] * meta_alpha;
                    frag[12] = _tmem_load_2[12] * meta_alpha;
                    frag[28] = _tmem_load_3[12] * meta_alpha;
                    frag[13] = _tmem_load_2[13] * meta_alpha;
                    frag[29] = _tmem_load_3[13] * meta_alpha;
                    frag[14] = _tmem_load_2[14] * meta_alpha;
                    frag[30] = _tmem_load_3[14] * meta_alpha;
                    frag[15] = _tmem_load_2[15] * meta_alpha;
                    frag[31] = _tmem_load_3[15] * meta_alpha;
                    int exchf_buf_0 = sexchf_addr + 8192;
                    int sact_buf_1 = sact_addr + 4608;
                    if (is_gate_lane != 0) {
                        float x_g_1 = frag[0];
                        float _exp2_32 = approx_exp2(x_g_1 * -1.4426950408889634f);
                        float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                        float sig_g_2 = _rcp_32;
                        float _tanh_approx_64;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_64) : "f"(x_g_1 * inv_beta));
                        frag[0] = beta * _tanh_approx_64 * sig_g_2;
                        float x_g_0_1 = frag[1];
                        float _exp2_33 = approx_exp2(x_g_0_1 * -1.4426950408889634f);
                        float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                        float sig_g_1_1 = _rcp_33;
                        float _tanh_approx_65;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_65) : "f"(x_g_0_1 * inv_beta));
                        frag[1] = beta * _tanh_approx_65 * sig_g_1_1;
                        float x_g_2_1 = frag[2];
                        float _exp2_34 = approx_exp2(x_g_2_1 * -1.4426950408889634f);
                        float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                        float sig_g_3_1 = _rcp_34;
                        float _tanh_approx_66;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_66) : "f"(x_g_2_1 * inv_beta));
                        frag[2] = beta * _tanh_approx_66 * sig_g_3_1;
                        float x_g_4_1 = frag[3];
                        float _exp2_35 = approx_exp2(x_g_4_1 * -1.4426950408889634f);
                        float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                        float sig_g_5_1 = _rcp_35;
                        float _tanh_approx_67;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_67) : "f"(x_g_4_1 * inv_beta));
                        frag[3] = beta * _tanh_approx_67 * sig_g_5_1;
                        float x_g_6_1 = frag[4];
                        float _exp2_36 = approx_exp2(x_g_6_1 * -1.4426950408889634f);
                        float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                        float sig_g_7_1 = _rcp_36;
                        float _tanh_approx_68;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_68) : "f"(x_g_6_1 * inv_beta));
                        frag[4] = beta * _tanh_approx_68 * sig_g_7_1;
                        float x_g_8_1 = frag[5];
                        float _exp2_37 = approx_exp2(x_g_8_1 * -1.4426950408889634f);
                        float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                        float sig_g_9_1 = _rcp_37;
                        float _tanh_approx_69;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_69) : "f"(x_g_8_1 * inv_beta));
                        frag[5] = beta * _tanh_approx_69 * sig_g_9_1;
                        float x_g_10_1 = frag[6];
                        float _exp2_38 = approx_exp2(x_g_10_1 * -1.4426950408889634f);
                        float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                        float sig_g_11_1 = _rcp_38;
                        float _tanh_approx_70;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_70) : "f"(x_g_10_1 * inv_beta));
                        frag[6] = beta * _tanh_approx_70 * sig_g_11_1;
                        float x_g_12_1 = frag[7];
                        float _exp2_39 = approx_exp2(x_g_12_1 * -1.4426950408889634f);
                        float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                        float sig_g_13_1 = _rcp_39;
                        float _tanh_approx_71;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_71) : "f"(x_g_12_1 * inv_beta));
                        frag[7] = beta * _tanh_approx_71 * sig_g_13_1;
                        float x_g_14_1 = frag[8];
                        float _exp2_40 = approx_exp2(x_g_14_1 * -1.4426950408889634f);
                        float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                        float sig_g_15_1 = _rcp_40;
                        float _tanh_approx_72;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_72) : "f"(x_g_14_1 * inv_beta));
                        frag[8] = beta * _tanh_approx_72 * sig_g_15_1;
                        float x_g_16_1 = frag[9];
                        float _exp2_41 = approx_exp2(x_g_16_1 * -1.4426950408889634f);
                        float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                        float sig_g_17_1 = _rcp_41;
                        float _tanh_approx_73;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_73) : "f"(x_g_16_1 * inv_beta));
                        frag[9] = beta * _tanh_approx_73 * sig_g_17_1;
                        float x_g_18_1 = frag[10];
                        float _exp2_42 = approx_exp2(x_g_18_1 * -1.4426950408889634f);
                        float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                        float sig_g_19_1 = _rcp_42;
                        float _tanh_approx_74;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_74) : "f"(x_g_18_1 * inv_beta));
                        frag[10] = beta * _tanh_approx_74 * sig_g_19_1;
                        float x_g_20_1 = frag[11];
                        float _exp2_43 = approx_exp2(x_g_20_1 * -1.4426950408889634f);
                        float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                        float sig_g_21_1 = _rcp_43;
                        float _tanh_approx_75;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_75) : "f"(x_g_20_1 * inv_beta));
                        frag[11] = beta * _tanh_approx_75 * sig_g_21_1;
                        float x_g_22_1 = frag[12];
                        float _exp2_44 = approx_exp2(x_g_22_1 * -1.4426950408889634f);
                        float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                        float sig_g_23_1 = _rcp_44;
                        float _tanh_approx_76;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_76) : "f"(x_g_22_1 * inv_beta));
                        frag[12] = beta * _tanh_approx_76 * sig_g_23_1;
                        float x_g_24_1 = frag[13];
                        float _exp2_45 = approx_exp2(x_g_24_1 * -1.4426950408889634f);
                        float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                        float sig_g_25_1 = _rcp_45;
                        float _tanh_approx_77;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_77) : "f"(x_g_24_1 * inv_beta));
                        frag[13] = beta * _tanh_approx_77 * sig_g_25_1;
                        float x_g_26_1 = frag[14];
                        float _exp2_46 = approx_exp2(x_g_26_1 * -1.4426950408889634f);
                        float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                        float sig_g_27_1 = _rcp_46;
                        float _tanh_approx_78;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_78) : "f"(x_g_26_1 * inv_beta));
                        frag[14] = beta * _tanh_approx_78 * sig_g_27_1;
                        float x_g_28_1 = frag[15];
                        float _exp2_47 = approx_exp2(x_g_28_1 * -1.4426950408889634f);
                        float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                        float sig_g_29_1 = _rcp_47;
                        float _tanh_approx_79;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_79) : "f"(x_g_28_1 * inv_beta));
                        frag[15] = beta * _tanh_approx_79 * sig_g_29_1;
                        float x_g_30_1 = frag[16];
                        float _exp2_48 = approx_exp2(x_g_30_1 * -1.4426950408889634f);
                        float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                        float sig_g_31_1 = _rcp_48;
                        float _tanh_approx_80;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_80) : "f"(x_g_30_1 * inv_beta));
                        frag[16] = beta * _tanh_approx_80 * sig_g_31_1;
                        float x_g_32_1 = frag[17];
                        float _exp2_49 = approx_exp2(x_g_32_1 * -1.4426950408889634f);
                        float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                        float sig_g_33_1 = _rcp_49;
                        float _tanh_approx_81;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_81) : "f"(x_g_32_1 * inv_beta));
                        frag[17] = beta * _tanh_approx_81 * sig_g_33_1;
                        float x_g_34_1 = frag[18];
                        float _exp2_50 = approx_exp2(x_g_34_1 * -1.4426950408889634f);
                        float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                        float sig_g_35_1 = _rcp_50;
                        float _tanh_approx_82;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_82) : "f"(x_g_34_1 * inv_beta));
                        frag[18] = beta * _tanh_approx_82 * sig_g_35_1;
                        float x_g_36_1 = frag[19];
                        float _exp2_51 = approx_exp2(x_g_36_1 * -1.4426950408889634f);
                        float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                        float sig_g_37_1 = _rcp_51;
                        float _tanh_approx_83;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_83) : "f"(x_g_36_1 * inv_beta));
                        frag[19] = beta * _tanh_approx_83 * sig_g_37_1;
                        float x_g_38_1 = frag[20];
                        float _exp2_52 = approx_exp2(x_g_38_1 * -1.4426950408889634f);
                        float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                        float sig_g_39_1 = _rcp_52;
                        float _tanh_approx_84;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_84) : "f"(x_g_38_1 * inv_beta));
                        frag[20] = beta * _tanh_approx_84 * sig_g_39_1;
                        float x_g_40_1 = frag[21];
                        float _exp2_53 = approx_exp2(x_g_40_1 * -1.4426950408889634f);
                        float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                        float sig_g_41_1 = _rcp_53;
                        float _tanh_approx_85;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_85) : "f"(x_g_40_1 * inv_beta));
                        frag[21] = beta * _tanh_approx_85 * sig_g_41_1;
                        float x_g_42_1 = frag[22];
                        float _exp2_54 = approx_exp2(x_g_42_1 * -1.4426950408889634f);
                        float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                        float sig_g_43_1 = _rcp_54;
                        float _tanh_approx_86;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_86) : "f"(x_g_42_1 * inv_beta));
                        frag[22] = beta * _tanh_approx_86 * sig_g_43_1;
                        float x_g_44_1 = frag[23];
                        float _exp2_55 = approx_exp2(x_g_44_1 * -1.4426950408889634f);
                        float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                        float sig_g_45_1 = _rcp_55;
                        float _tanh_approx_87;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_87) : "f"(x_g_44_1 * inv_beta));
                        frag[23] = beta * _tanh_approx_87 * sig_g_45_1;
                        float x_g_46_1 = frag[24];
                        float _exp2_56 = approx_exp2(x_g_46_1 * -1.4426950408889634f);
                        float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                        float sig_g_47_1 = _rcp_56;
                        float _tanh_approx_88;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_88) : "f"(x_g_46_1 * inv_beta));
                        frag[24] = beta * _tanh_approx_88 * sig_g_47_1;
                        float x_g_48_1 = frag[25];
                        float _exp2_57 = approx_exp2(x_g_48_1 * -1.4426950408889634f);
                        float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                        float sig_g_49_1 = _rcp_57;
                        float _tanh_approx_89;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_89) : "f"(x_g_48_1 * inv_beta));
                        frag[25] = beta * _tanh_approx_89 * sig_g_49_1;
                        float x_g_50_1 = frag[26];
                        float _exp2_58 = approx_exp2(x_g_50_1 * -1.4426950408889634f);
                        float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                        float sig_g_51_1 = _rcp_58;
                        float _tanh_approx_90;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_90) : "f"(x_g_50_1 * inv_beta));
                        frag[26] = beta * _tanh_approx_90 * sig_g_51_1;
                        float x_g_52_1 = frag[27];
                        float _exp2_59 = approx_exp2(x_g_52_1 * -1.4426950408889634f);
                        float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                        float sig_g_53_1 = _rcp_59;
                        float _tanh_approx_91;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_91) : "f"(x_g_52_1 * inv_beta));
                        frag[27] = beta * _tanh_approx_91 * sig_g_53_1;
                        float x_g_54_1 = frag[28];
                        float _exp2_60 = approx_exp2(x_g_54_1 * -1.4426950408889634f);
                        float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                        float sig_g_55_1 = _rcp_60;
                        float _tanh_approx_92;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_92) : "f"(x_g_54_1 * inv_beta));
                        frag[28] = beta * _tanh_approx_92 * sig_g_55_1;
                        float x_g_56_1 = frag[29];
                        float _exp2_61 = approx_exp2(x_g_56_1 * -1.4426950408889634f);
                        float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                        float sig_g_57_1 = _rcp_61;
                        float _tanh_approx_93;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_93) : "f"(x_g_56_1 * inv_beta));
                        frag[29] = beta * _tanh_approx_93 * sig_g_57_1;
                        float x_g_58_1 = frag[30];
                        float _exp2_62 = approx_exp2(x_g_58_1 * -1.4426950408889634f);
                        float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                        float sig_g_59_1 = _rcp_62;
                        float _tanh_approx_94;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_94) : "f"(x_g_58_1 * inv_beta));
                        frag[30] = beta * _tanh_approx_94 * sig_g_59_1;
                        float x_g_60_1 = frag[31];
                        float _exp2_63 = approx_exp2(x_g_60_1 * -1.4426950408889634f);
                        float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                        float sig_g_61_1 = _rcp_63;
                        float _tanh_approx_95;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_95) : "f"(x_g_60_1 * inv_beta));
                        frag[31] = beta * _tanh_approx_95 * sig_g_61_1;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_0 + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_96;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_96) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_96;
                        float _tanh_approx_97;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_97) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_97;
                        float _tanh_approx_98;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_98) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_98;
                        float _tanh_approx_99;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_99) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_99;
                        float _tanh_approx_100;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_100) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_100;
                        float _tanh_approx_101;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_101) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_101;
                        float _tanh_approx_102;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_102) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_102;
                        float _tanh_approx_103;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_103) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_103;
                        float _tanh_approx_104;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_104) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_104;
                        float _tanh_approx_105;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_105) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_105;
                        float _tanh_approx_106;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_106) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_106;
                        float _tanh_approx_107;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_107) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_107;
                        float _tanh_approx_108;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_108) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_108;
                        float _tanh_approx_109;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_109) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_109;
                        float _tanh_approx_110;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_110) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_110;
                        float _tanh_approx_111;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_111) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_111;
                        float _tanh_approx_112;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_112) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_112;
                        float _tanh_approx_113;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_113) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_113;
                        float _tanh_approx_114;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_114) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_114;
                        float _tanh_approx_115;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_115) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_115;
                        float _tanh_approx_116;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_116) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_116;
                        float _tanh_approx_117;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_117) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_117;
                        float _tanh_approx_118;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_118) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_118;
                        float _tanh_approx_119;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_119) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_119;
                        float _tanh_approx_120;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_120) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_120;
                        float _tanh_approx_121;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_121) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_121;
                        float _tanh_approx_122;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_122) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_122;
                        float _tanh_approx_123;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_123) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_123;
                        float _tanh_approx_124;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_124) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_124;
                        float _tanh_approx_125;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_125) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_125;
                        float _tanh_approx_126;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_126) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_126;
                        float _tanh_approx_127;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_127) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_127;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_0 + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_80 = fmaxf(frag[0], -frag[0]);
                        float a_c_1 = _fmax_80;
                        float _fmax_81 = fmaxf(frag[2], -frag[2]);
                        float _fmax_82 = fmaxf(a_c_1, _fmax_81);
                        a_c_1 = _fmax_82;
                        float _fmax_83 = fmaxf(frag[16], -frag[16]);
                        float _fmax_84 = fmaxf(a_c_1, _fmax_83);
                        a_c_1 = _fmax_84;
                        float _fmax_85 = fmaxf(frag[18], -frag[18]);
                        float _fmax_86 = fmaxf(a_c_1, _fmax_85);
                        a_c_1 = _fmax_86;
                        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, a_c_1, 4);
                        float _fmax_87 = fmaxf(a_c_1, _shfl_xor_24);
                        a_c_1 = _fmax_87;
                        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, a_c_1, 8);
                        float _fmax_88 = fmaxf(a_c_1, _shfl_xor_25);
                        a_c_1 = _fmax_88;
                        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, a_c_1, 16);
                        float _fmax_89 = fmaxf(a_c_1, _shfl_xor_26);
                        a_c_1 = _fmax_89;
                        uint16_t _ue8m0x2_f32_8;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_8) : "f"(zero_f32), "f"(a_c_1 * inv_fp8_max));
                        int code_full_2 = (int)_ue8m0x2_f32_8;
                        int code_1 = code_full_2 & 255;
                        int _max_8 = ((254 - code_1) > (0) ? (254 - code_1) : (0));
                        unsigned int inv_bits_1 = (unsigned int)(_max_8 << 23);
                        float inv_scale_1 = __uint_as_float(inv_bits_1) * (float)(code_1 != 0);
                        frag[0] = frag[0] * inv_scale_1;
                        frag[2] = frag[2] * inv_scale_1;
                        frag[16] = frag[16] * inv_scale_1;
                        frag[18] = frag[18] * inv_scale_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code_1;
                        }
                        float _fmax_90 = fmaxf(frag[1], -frag[1]);
                        float a_c_0_1 = _fmax_90;
                        float _fmax_91 = fmaxf(frag[3], -frag[3]);
                        float _fmax_92 = fmaxf(a_c_0_1, _fmax_91);
                        a_c_0_1 = _fmax_92;
                        float _fmax_93 = fmaxf(frag[17], -frag[17]);
                        float _fmax_94 = fmaxf(a_c_0_1, _fmax_93);
                        a_c_0_1 = _fmax_94;
                        float _fmax_95 = fmaxf(frag[19], -frag[19]);
                        float _fmax_96 = fmaxf(a_c_0_1, _fmax_95);
                        a_c_0_1 = _fmax_96;
                        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_1, 4);
                        float _fmax_97 = fmaxf(a_c_0_1, _shfl_xor_27);
                        a_c_0_1 = _fmax_97;
                        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_1, 8);
                        float _fmax_98 = fmaxf(a_c_0_1, _shfl_xor_28);
                        a_c_0_1 = _fmax_98;
                        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_1, 16);
                        float _fmax_99 = fmaxf(a_c_0_1, _shfl_xor_29);
                        a_c_0_1 = _fmax_99;
                        uint16_t _ue8m0x2_f32_9;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_9) : "f"(zero_f32), "f"(a_c_0_1 * inv_fp8_max));
                        int code_full_1_1 = (int)_ue8m0x2_f32_9;
                        int code_2_1 = code_full_1_1 & 255;
                        int _max_9 = ((254 - code_2_1) > (0) ? (254 - code_2_1) : (0));
                        unsigned int inv_bits_3_1 = (unsigned int)(_max_9 << 23);
                        float inv_scale_4_1 = __uint_as_float(inv_bits_3_1) * (float)(code_2_1 != 0);
                        frag[1] = frag[1] * inv_scale_4_1;
                        frag[3] = frag[3] * inv_scale_4_1;
                        frag[17] = frag[17] * inv_scale_4_1;
                        frag[19] = frag[19] * inv_scale_4_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2_1;
                        }
                        float _fmax_100 = fmaxf(frag[4], -frag[4]);
                        float a_c_5_1 = _fmax_100;
                        float _fmax_101 = fmaxf(frag[6], -frag[6]);
                        float _fmax_102 = fmaxf(a_c_5_1, _fmax_101);
                        a_c_5_1 = _fmax_102;
                        float _fmax_103 = fmaxf(frag[20], -frag[20]);
                        float _fmax_104 = fmaxf(a_c_5_1, _fmax_103);
                        a_c_5_1 = _fmax_104;
                        float _fmax_105 = fmaxf(frag[22], -frag[22]);
                        float _fmax_106 = fmaxf(a_c_5_1, _fmax_105);
                        a_c_5_1 = _fmax_106;
                        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_1, 4);
                        float _fmax_107 = fmaxf(a_c_5_1, _shfl_xor_30);
                        a_c_5_1 = _fmax_107;
                        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_1, 8);
                        float _fmax_108 = fmaxf(a_c_5_1, _shfl_xor_31);
                        a_c_5_1 = _fmax_108;
                        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_1, 16);
                        float _fmax_109 = fmaxf(a_c_5_1, _shfl_xor_32);
                        a_c_5_1 = _fmax_109;
                        uint16_t _ue8m0x2_f32_10;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_10) : "f"(zero_f32), "f"(a_c_5_1 * inv_fp8_max));
                        int code_full_6_1 = (int)_ue8m0x2_f32_10;
                        int code_7_1 = code_full_6_1 & 255;
                        int _max_10 = ((254 - code_7_1) > (0) ? (254 - code_7_1) : (0));
                        unsigned int inv_bits_8_1 = (unsigned int)(_max_10 << 23);
                        float inv_scale_9_1 = __uint_as_float(inv_bits_8_1) * (float)(code_7_1 != 0);
                        frag[4] = frag[4] * inv_scale_9_1;
                        frag[6] = frag[6] * inv_scale_9_1;
                        frag[20] = frag[20] * inv_scale_9_1;
                        frag[22] = frag[22] * inv_scale_9_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7_1;
                        }
                        float _fmax_110 = fmaxf(frag[5], -frag[5]);
                        float a_c_10_1 = _fmax_110;
                        float _fmax_111 = fmaxf(frag[7], -frag[7]);
                        float _fmax_112 = fmaxf(a_c_10_1, _fmax_111);
                        a_c_10_1 = _fmax_112;
                        float _fmax_113 = fmaxf(frag[21], -frag[21]);
                        float _fmax_114 = fmaxf(a_c_10_1, _fmax_113);
                        a_c_10_1 = _fmax_114;
                        float _fmax_115 = fmaxf(frag[23], -frag[23]);
                        float _fmax_116 = fmaxf(a_c_10_1, _fmax_115);
                        a_c_10_1 = _fmax_116;
                        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_1, 4);
                        float _fmax_117 = fmaxf(a_c_10_1, _shfl_xor_33);
                        a_c_10_1 = _fmax_117;
                        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_1, 8);
                        float _fmax_118 = fmaxf(a_c_10_1, _shfl_xor_34);
                        a_c_10_1 = _fmax_118;
                        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_1, 16);
                        float _fmax_119 = fmaxf(a_c_10_1, _shfl_xor_35);
                        a_c_10_1 = _fmax_119;
                        uint16_t _ue8m0x2_f32_11;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_11) : "f"(zero_f32), "f"(a_c_10_1 * inv_fp8_max));
                        int code_full_11_1 = (int)_ue8m0x2_f32_11;
                        int code_12_1 = code_full_11_1 & 255;
                        int _max_11 = ((254 - code_12_1) > (0) ? (254 - code_12_1) : (0));
                        unsigned int inv_bits_13_1 = (unsigned int)(_max_11 << 23);
                        float inv_scale_14_1 = __uint_as_float(inv_bits_13_1) * (float)(code_12_1 != 0);
                        frag[5] = frag[5] * inv_scale_14_1;
                        frag[7] = frag[7] * inv_scale_14_1;
                        frag[21] = frag[21] * inv_scale_14_1;
                        frag[23] = frag[23] * inv_scale_14_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12_1;
                        }
                        float _fmax_120 = fmaxf(frag[8], -frag[8]);
                        float a_c_15_1 = _fmax_120;
                        float _fmax_121 = fmaxf(frag[10], -frag[10]);
                        float _fmax_122 = fmaxf(a_c_15_1, _fmax_121);
                        a_c_15_1 = _fmax_122;
                        float _fmax_123 = fmaxf(frag[24], -frag[24]);
                        float _fmax_124 = fmaxf(a_c_15_1, _fmax_123);
                        a_c_15_1 = _fmax_124;
                        float _fmax_125 = fmaxf(frag[26], -frag[26]);
                        float _fmax_126 = fmaxf(a_c_15_1, _fmax_125);
                        a_c_15_1 = _fmax_126;
                        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_1, 4);
                        float _fmax_127 = fmaxf(a_c_15_1, _shfl_xor_36);
                        a_c_15_1 = _fmax_127;
                        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_1, 8);
                        float _fmax_128 = fmaxf(a_c_15_1, _shfl_xor_37);
                        a_c_15_1 = _fmax_128;
                        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_1, 16);
                        float _fmax_129 = fmaxf(a_c_15_1, _shfl_xor_38);
                        a_c_15_1 = _fmax_129;
                        uint16_t _ue8m0x2_f32_12;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_12) : "f"(zero_f32), "f"(a_c_15_1 * inv_fp8_max));
                        int code_full_16_1 = (int)_ue8m0x2_f32_12;
                        int code_17_1 = code_full_16_1 & 255;
                        int _max_12 = ((254 - code_17_1) > (0) ? (254 - code_17_1) : (0));
                        unsigned int inv_bits_18_1 = (unsigned int)(_max_12 << 23);
                        float inv_scale_19_1 = __uint_as_float(inv_bits_18_1) * (float)(code_17_1 != 0);
                        frag[8] = frag[8] * inv_scale_19_1;
                        frag[10] = frag[10] * inv_scale_19_1;
                        frag[24] = frag[24] * inv_scale_19_1;
                        frag[26] = frag[26] * inv_scale_19_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17_1;
                        }
                        float _fmax_130 = fmaxf(frag[9], -frag[9]);
                        float a_c_20_1 = _fmax_130;
                        float _fmax_131 = fmaxf(frag[11], -frag[11]);
                        float _fmax_132 = fmaxf(a_c_20_1, _fmax_131);
                        a_c_20_1 = _fmax_132;
                        float _fmax_133 = fmaxf(frag[25], -frag[25]);
                        float _fmax_134 = fmaxf(a_c_20_1, _fmax_133);
                        a_c_20_1 = _fmax_134;
                        float _fmax_135 = fmaxf(frag[27], -frag[27]);
                        float _fmax_136 = fmaxf(a_c_20_1, _fmax_135);
                        a_c_20_1 = _fmax_136;
                        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_1, 4);
                        float _fmax_137 = fmaxf(a_c_20_1, _shfl_xor_39);
                        a_c_20_1 = _fmax_137;
                        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_1, 8);
                        float _fmax_138 = fmaxf(a_c_20_1, _shfl_xor_40);
                        a_c_20_1 = _fmax_138;
                        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_1, 16);
                        float _fmax_139 = fmaxf(a_c_20_1, _shfl_xor_41);
                        a_c_20_1 = _fmax_139;
                        uint16_t _ue8m0x2_f32_13;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_13) : "f"(zero_f32), "f"(a_c_20_1 * inv_fp8_max));
                        int code_full_21_1 = (int)_ue8m0x2_f32_13;
                        int code_22_1 = code_full_21_1 & 255;
                        int _max_13 = ((254 - code_22_1) > (0) ? (254 - code_22_1) : (0));
                        unsigned int inv_bits_23_1 = (unsigned int)(_max_13 << 23);
                        float inv_scale_24_1 = __uint_as_float(inv_bits_23_1) * (float)(code_22_1 != 0);
                        frag[9] = frag[9] * inv_scale_24_1;
                        frag[11] = frag[11] * inv_scale_24_1;
                        frag[25] = frag[25] * inv_scale_24_1;
                        frag[27] = frag[27] * inv_scale_24_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22_1;
                        }
                        float _fmax_140 = fmaxf(frag[12], -frag[12]);
                        float a_c_25_1 = _fmax_140;
                        float _fmax_141 = fmaxf(frag[14], -frag[14]);
                        float _fmax_142 = fmaxf(a_c_25_1, _fmax_141);
                        a_c_25_1 = _fmax_142;
                        float _fmax_143 = fmaxf(frag[28], -frag[28]);
                        float _fmax_144 = fmaxf(a_c_25_1, _fmax_143);
                        a_c_25_1 = _fmax_144;
                        float _fmax_145 = fmaxf(frag[30], -frag[30]);
                        float _fmax_146 = fmaxf(a_c_25_1, _fmax_145);
                        a_c_25_1 = _fmax_146;
                        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_1, 4);
                        float _fmax_147 = fmaxf(a_c_25_1, _shfl_xor_42);
                        a_c_25_1 = _fmax_147;
                        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_1, 8);
                        float _fmax_148 = fmaxf(a_c_25_1, _shfl_xor_43);
                        a_c_25_1 = _fmax_148;
                        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_1, 16);
                        float _fmax_149 = fmaxf(a_c_25_1, _shfl_xor_44);
                        a_c_25_1 = _fmax_149;
                        uint16_t _ue8m0x2_f32_14;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_14) : "f"(zero_f32), "f"(a_c_25_1 * inv_fp8_max));
                        int code_full_26_1 = (int)_ue8m0x2_f32_14;
                        int code_27_1 = code_full_26_1 & 255;
                        int _max_14 = ((254 - code_27_1) > (0) ? (254 - code_27_1) : (0));
                        unsigned int inv_bits_28_1 = (unsigned int)(_max_14 << 23);
                        float inv_scale_29_1 = __uint_as_float(inv_bits_28_1) * (float)(code_27_1 != 0);
                        frag[12] = frag[12] * inv_scale_29_1;
                        frag[14] = frag[14] * inv_scale_29_1;
                        frag[28] = frag[28] * inv_scale_29_1;
                        frag[30] = frag[30] * inv_scale_29_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27_1;
                        }
                        float _fmax_150 = fmaxf(frag[13], -frag[13]);
                        float a_c_30_1 = _fmax_150;
                        float _fmax_151 = fmaxf(frag[15], -frag[15]);
                        float _fmax_152 = fmaxf(a_c_30_1, _fmax_151);
                        a_c_30_1 = _fmax_152;
                        float _fmax_153 = fmaxf(frag[29], -frag[29]);
                        float _fmax_154 = fmaxf(a_c_30_1, _fmax_153);
                        a_c_30_1 = _fmax_154;
                        float _fmax_155 = fmaxf(frag[31], -frag[31]);
                        float _fmax_156 = fmaxf(a_c_30_1, _fmax_155);
                        a_c_30_1 = _fmax_156;
                        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_1, 4);
                        float _fmax_157 = fmaxf(a_c_30_1, _shfl_xor_45);
                        a_c_30_1 = _fmax_157;
                        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_1, 8);
                        float _fmax_158 = fmaxf(a_c_30_1, _shfl_xor_46);
                        a_c_30_1 = _fmax_158;
                        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_1, 16);
                        float _fmax_159 = fmaxf(a_c_30_1, _shfl_xor_47);
                        a_c_30_1 = _fmax_159;
                        uint16_t _ue8m0x2_f32_15;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_15) : "f"(zero_f32), "f"(a_c_30_1 * inv_fp8_max));
                        int code_full_31_1 = (int)_ue8m0x2_f32_15;
                        int code_32_1 = code_full_31_1 & 255;
                        int _max_15 = ((254 - code_32_1) > (0) ? (254 - code_32_1) : (0));
                        unsigned int inv_bits_33_1 = (unsigned int)(_max_15 << 23);
                        float inv_scale_34_1 = __uint_as_float(inv_bits_33_1) * (float)(code_32_1 != 0);
                        frag[13] = frag[13] * inv_scale_34_1;
                        frag[15] = frag[15] * inv_scale_34_1;
                        frag[29] = frag[29] * inv_scale_34_1;
                        frag[31] = frag[31] * inv_scale_34_1;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32_1;
                        }
                        uint32_t _fp8_1[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_1[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_1[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_1[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_1[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_1[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_1[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_1[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_1[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_2 = static_cast<uint32_t>(sact_buf_1 + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_2), "r"(_fp8_1[0]), "r"(_fp8_1[1]), "r"(_fp8_1[2]), "r"(_fp8_1[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_3 = static_cast<uint32_t>(sact_buf_1 + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_3), "r"(_fp8_1[4]), "r"(_fp8_1[5]), "r"(_fp8_1[6]), "r"(_fp8_1[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s_2 = row_base + 32 + st_tok;
                    if (prow_s_2 < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf_1 + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s_2 * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l_1 = row_base + 32 + lane_0;
                        if (prow_l_1 < mn_limit) {
                            unsigned int code_l_1 = scode[64 + warp_in64 * 32 + lane_0];
                            int sf_off_l_1 = prow_l_1 % 32 * 16 + prow_l_1 / 32 % 4 * 4 + prow_l_1 / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l_1) + (0)) = (unsigned char)(code_l_1);
                        }
                    }
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
                    frag[0] = _tmem_load_4[0] * meta_alpha;
                    frag[16] = _tmem_load_5[0] * meta_alpha;
                    frag[1] = _tmem_load_4[1] * meta_alpha;
                    frag[17] = _tmem_load_5[1] * meta_alpha;
                    frag[2] = _tmem_load_4[2] * meta_alpha;
                    frag[18] = _tmem_load_5[2] * meta_alpha;
                    frag[3] = _tmem_load_4[3] * meta_alpha;
                    frag[19] = _tmem_load_5[3] * meta_alpha;
                    frag[4] = _tmem_load_4[4] * meta_alpha;
                    frag[20] = _tmem_load_5[4] * meta_alpha;
                    frag[5] = _tmem_load_4[5] * meta_alpha;
                    frag[21] = _tmem_load_5[5] * meta_alpha;
                    frag[6] = _tmem_load_4[6] * meta_alpha;
                    frag[22] = _tmem_load_5[6] * meta_alpha;
                    frag[7] = _tmem_load_4[7] * meta_alpha;
                    frag[23] = _tmem_load_5[7] * meta_alpha;
                    frag[8] = _tmem_load_4[8] * meta_alpha;
                    frag[24] = _tmem_load_5[8] * meta_alpha;
                    frag[9] = _tmem_load_4[9] * meta_alpha;
                    frag[25] = _tmem_load_5[9] * meta_alpha;
                    frag[10] = _tmem_load_4[10] * meta_alpha;
                    frag[26] = _tmem_load_5[10] * meta_alpha;
                    frag[11] = _tmem_load_4[11] * meta_alpha;
                    frag[27] = _tmem_load_5[11] * meta_alpha;
                    frag[12] = _tmem_load_4[12] * meta_alpha;
                    frag[28] = _tmem_load_5[12] * meta_alpha;
                    frag[13] = _tmem_load_4[13] * meta_alpha;
                    frag[29] = _tmem_load_5[13] * meta_alpha;
                    frag[14] = _tmem_load_4[14] * meta_alpha;
                    frag[30] = _tmem_load_5[14] * meta_alpha;
                    frag[15] = _tmem_load_4[15] * meta_alpha;
                    frag[31] = _tmem_load_5[15] * meta_alpha;
                    int exchf_buf_3 = sexchf_addr;
                    int sact_buf_4 = sact_addr;
                    if (is_gate_lane != 0) {
                        float x_g_3 = frag[0];
                        float _exp2_64 = approx_exp2(x_g_3 * -1.4426950408889634f);
                        float _rcp_64 = approx_rcp(1.0f + _exp2_64);
                        float sig_g_4 = _rcp_64;
                        float _tanh_approx_128;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_128) : "f"(x_g_3 * inv_beta));
                        frag[0] = beta * _tanh_approx_128 * sig_g_4;
                        float x_g_0_2 = frag[1];
                        float _exp2_65 = approx_exp2(x_g_0_2 * -1.4426950408889634f);
                        float _rcp_65 = approx_rcp(1.0f + _exp2_65);
                        float sig_g_1_2 = _rcp_65;
                        float _tanh_approx_129;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_129) : "f"(x_g_0_2 * inv_beta));
                        frag[1] = beta * _tanh_approx_129 * sig_g_1_2;
                        float x_g_2_2 = frag[2];
                        float _exp2_66 = approx_exp2(x_g_2_2 * -1.4426950408889634f);
                        float _rcp_66 = approx_rcp(1.0f + _exp2_66);
                        float sig_g_3_2 = _rcp_66;
                        float _tanh_approx_130;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_130) : "f"(x_g_2_2 * inv_beta));
                        frag[2] = beta * _tanh_approx_130 * sig_g_3_2;
                        float x_g_4_2 = frag[3];
                        float _exp2_67 = approx_exp2(x_g_4_2 * -1.4426950408889634f);
                        float _rcp_67 = approx_rcp(1.0f + _exp2_67);
                        float sig_g_5_2 = _rcp_67;
                        float _tanh_approx_131;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_131) : "f"(x_g_4_2 * inv_beta));
                        frag[3] = beta * _tanh_approx_131 * sig_g_5_2;
                        float x_g_6_2 = frag[4];
                        float _exp2_68 = approx_exp2(x_g_6_2 * -1.4426950408889634f);
                        float _rcp_68 = approx_rcp(1.0f + _exp2_68);
                        float sig_g_7_2 = _rcp_68;
                        float _tanh_approx_132;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_132) : "f"(x_g_6_2 * inv_beta));
                        frag[4] = beta * _tanh_approx_132 * sig_g_7_2;
                        float x_g_8_2 = frag[5];
                        float _exp2_69 = approx_exp2(x_g_8_2 * -1.4426950408889634f);
                        float _rcp_69 = approx_rcp(1.0f + _exp2_69);
                        float sig_g_9_2 = _rcp_69;
                        float _tanh_approx_133;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_133) : "f"(x_g_8_2 * inv_beta));
                        frag[5] = beta * _tanh_approx_133 * sig_g_9_2;
                        float x_g_10_2 = frag[6];
                        float _exp2_70 = approx_exp2(x_g_10_2 * -1.4426950408889634f);
                        float _rcp_70 = approx_rcp(1.0f + _exp2_70);
                        float sig_g_11_2 = _rcp_70;
                        float _tanh_approx_134;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_134) : "f"(x_g_10_2 * inv_beta));
                        frag[6] = beta * _tanh_approx_134 * sig_g_11_2;
                        float x_g_12_2 = frag[7];
                        float _exp2_71 = approx_exp2(x_g_12_2 * -1.4426950408889634f);
                        float _rcp_71 = approx_rcp(1.0f + _exp2_71);
                        float sig_g_13_2 = _rcp_71;
                        float _tanh_approx_135;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_135) : "f"(x_g_12_2 * inv_beta));
                        frag[7] = beta * _tanh_approx_135 * sig_g_13_2;
                        float x_g_14_2 = frag[8];
                        float _exp2_72 = approx_exp2(x_g_14_2 * -1.4426950408889634f);
                        float _rcp_72 = approx_rcp(1.0f + _exp2_72);
                        float sig_g_15_2 = _rcp_72;
                        float _tanh_approx_136;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_136) : "f"(x_g_14_2 * inv_beta));
                        frag[8] = beta * _tanh_approx_136 * sig_g_15_2;
                        float x_g_16_2 = frag[9];
                        float _exp2_73 = approx_exp2(x_g_16_2 * -1.4426950408889634f);
                        float _rcp_73 = approx_rcp(1.0f + _exp2_73);
                        float sig_g_17_2 = _rcp_73;
                        float _tanh_approx_137;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_137) : "f"(x_g_16_2 * inv_beta));
                        frag[9] = beta * _tanh_approx_137 * sig_g_17_2;
                        float x_g_18_2 = frag[10];
                        float _exp2_74 = approx_exp2(x_g_18_2 * -1.4426950408889634f);
                        float _rcp_74 = approx_rcp(1.0f + _exp2_74);
                        float sig_g_19_2 = _rcp_74;
                        float _tanh_approx_138;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_138) : "f"(x_g_18_2 * inv_beta));
                        frag[10] = beta * _tanh_approx_138 * sig_g_19_2;
                        float x_g_20_2 = frag[11];
                        float _exp2_75 = approx_exp2(x_g_20_2 * -1.4426950408889634f);
                        float _rcp_75 = approx_rcp(1.0f + _exp2_75);
                        float sig_g_21_2 = _rcp_75;
                        float _tanh_approx_139;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_139) : "f"(x_g_20_2 * inv_beta));
                        frag[11] = beta * _tanh_approx_139 * sig_g_21_2;
                        float x_g_22_2 = frag[12];
                        float _exp2_76 = approx_exp2(x_g_22_2 * -1.4426950408889634f);
                        float _rcp_76 = approx_rcp(1.0f + _exp2_76);
                        float sig_g_23_2 = _rcp_76;
                        float _tanh_approx_140;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_140) : "f"(x_g_22_2 * inv_beta));
                        frag[12] = beta * _tanh_approx_140 * sig_g_23_2;
                        float x_g_24_2 = frag[13];
                        float _exp2_77 = approx_exp2(x_g_24_2 * -1.4426950408889634f);
                        float _rcp_77 = approx_rcp(1.0f + _exp2_77);
                        float sig_g_25_2 = _rcp_77;
                        float _tanh_approx_141;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_141) : "f"(x_g_24_2 * inv_beta));
                        frag[13] = beta * _tanh_approx_141 * sig_g_25_2;
                        float x_g_26_2 = frag[14];
                        float _exp2_78 = approx_exp2(x_g_26_2 * -1.4426950408889634f);
                        float _rcp_78 = approx_rcp(1.0f + _exp2_78);
                        float sig_g_27_2 = _rcp_78;
                        float _tanh_approx_142;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_142) : "f"(x_g_26_2 * inv_beta));
                        frag[14] = beta * _tanh_approx_142 * sig_g_27_2;
                        float x_g_28_2 = frag[15];
                        float _exp2_79 = approx_exp2(x_g_28_2 * -1.4426950408889634f);
                        float _rcp_79 = approx_rcp(1.0f + _exp2_79);
                        float sig_g_29_2 = _rcp_79;
                        float _tanh_approx_143;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_143) : "f"(x_g_28_2 * inv_beta));
                        frag[15] = beta * _tanh_approx_143 * sig_g_29_2;
                        float x_g_30_2 = frag[16];
                        float _exp2_80 = approx_exp2(x_g_30_2 * -1.4426950408889634f);
                        float _rcp_80 = approx_rcp(1.0f + _exp2_80);
                        float sig_g_31_2 = _rcp_80;
                        float _tanh_approx_144;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_144) : "f"(x_g_30_2 * inv_beta));
                        frag[16] = beta * _tanh_approx_144 * sig_g_31_2;
                        float x_g_32_2 = frag[17];
                        float _exp2_81 = approx_exp2(x_g_32_2 * -1.4426950408889634f);
                        float _rcp_81 = approx_rcp(1.0f + _exp2_81);
                        float sig_g_33_2 = _rcp_81;
                        float _tanh_approx_145;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_145) : "f"(x_g_32_2 * inv_beta));
                        frag[17] = beta * _tanh_approx_145 * sig_g_33_2;
                        float x_g_34_2 = frag[18];
                        float _exp2_82 = approx_exp2(x_g_34_2 * -1.4426950408889634f);
                        float _rcp_82 = approx_rcp(1.0f + _exp2_82);
                        float sig_g_35_2 = _rcp_82;
                        float _tanh_approx_146;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_146) : "f"(x_g_34_2 * inv_beta));
                        frag[18] = beta * _tanh_approx_146 * sig_g_35_2;
                        float x_g_36_2 = frag[19];
                        float _exp2_83 = approx_exp2(x_g_36_2 * -1.4426950408889634f);
                        float _rcp_83 = approx_rcp(1.0f + _exp2_83);
                        float sig_g_37_2 = _rcp_83;
                        float _tanh_approx_147;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_147) : "f"(x_g_36_2 * inv_beta));
                        frag[19] = beta * _tanh_approx_147 * sig_g_37_2;
                        float x_g_38_2 = frag[20];
                        float _exp2_84 = approx_exp2(x_g_38_2 * -1.4426950408889634f);
                        float _rcp_84 = approx_rcp(1.0f + _exp2_84);
                        float sig_g_39_2 = _rcp_84;
                        float _tanh_approx_148;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_148) : "f"(x_g_38_2 * inv_beta));
                        frag[20] = beta * _tanh_approx_148 * sig_g_39_2;
                        float x_g_40_2 = frag[21];
                        float _exp2_85 = approx_exp2(x_g_40_2 * -1.4426950408889634f);
                        float _rcp_85 = approx_rcp(1.0f + _exp2_85);
                        float sig_g_41_2 = _rcp_85;
                        float _tanh_approx_149;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_149) : "f"(x_g_40_2 * inv_beta));
                        frag[21] = beta * _tanh_approx_149 * sig_g_41_2;
                        float x_g_42_2 = frag[22];
                        float _exp2_86 = approx_exp2(x_g_42_2 * -1.4426950408889634f);
                        float _rcp_86 = approx_rcp(1.0f + _exp2_86);
                        float sig_g_43_2 = _rcp_86;
                        float _tanh_approx_150;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_150) : "f"(x_g_42_2 * inv_beta));
                        frag[22] = beta * _tanh_approx_150 * sig_g_43_2;
                        float x_g_44_2 = frag[23];
                        float _exp2_87 = approx_exp2(x_g_44_2 * -1.4426950408889634f);
                        float _rcp_87 = approx_rcp(1.0f + _exp2_87);
                        float sig_g_45_2 = _rcp_87;
                        float _tanh_approx_151;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_151) : "f"(x_g_44_2 * inv_beta));
                        frag[23] = beta * _tanh_approx_151 * sig_g_45_2;
                        float x_g_46_2 = frag[24];
                        float _exp2_88 = approx_exp2(x_g_46_2 * -1.4426950408889634f);
                        float _rcp_88 = approx_rcp(1.0f + _exp2_88);
                        float sig_g_47_2 = _rcp_88;
                        float _tanh_approx_152;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_152) : "f"(x_g_46_2 * inv_beta));
                        frag[24] = beta * _tanh_approx_152 * sig_g_47_2;
                        float x_g_48_2 = frag[25];
                        float _exp2_89 = approx_exp2(x_g_48_2 * -1.4426950408889634f);
                        float _rcp_89 = approx_rcp(1.0f + _exp2_89);
                        float sig_g_49_2 = _rcp_89;
                        float _tanh_approx_153;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_153) : "f"(x_g_48_2 * inv_beta));
                        frag[25] = beta * _tanh_approx_153 * sig_g_49_2;
                        float x_g_50_2 = frag[26];
                        float _exp2_90 = approx_exp2(x_g_50_2 * -1.4426950408889634f);
                        float _rcp_90 = approx_rcp(1.0f + _exp2_90);
                        float sig_g_51_2 = _rcp_90;
                        float _tanh_approx_154;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_154) : "f"(x_g_50_2 * inv_beta));
                        frag[26] = beta * _tanh_approx_154 * sig_g_51_2;
                        float x_g_52_2 = frag[27];
                        float _exp2_91 = approx_exp2(x_g_52_2 * -1.4426950408889634f);
                        float _rcp_91 = approx_rcp(1.0f + _exp2_91);
                        float sig_g_53_2 = _rcp_91;
                        float _tanh_approx_155;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_155) : "f"(x_g_52_2 * inv_beta));
                        frag[27] = beta * _tanh_approx_155 * sig_g_53_2;
                        float x_g_54_2 = frag[28];
                        float _exp2_92 = approx_exp2(x_g_54_2 * -1.4426950408889634f);
                        float _rcp_92 = approx_rcp(1.0f + _exp2_92);
                        float sig_g_55_2 = _rcp_92;
                        float _tanh_approx_156;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_156) : "f"(x_g_54_2 * inv_beta));
                        frag[28] = beta * _tanh_approx_156 * sig_g_55_2;
                        float x_g_56_2 = frag[29];
                        float _exp2_93 = approx_exp2(x_g_56_2 * -1.4426950408889634f);
                        float _rcp_93 = approx_rcp(1.0f + _exp2_93);
                        float sig_g_57_2 = _rcp_93;
                        float _tanh_approx_157;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_157) : "f"(x_g_56_2 * inv_beta));
                        frag[29] = beta * _tanh_approx_157 * sig_g_57_2;
                        float x_g_58_2 = frag[30];
                        float _exp2_94 = approx_exp2(x_g_58_2 * -1.4426950408889634f);
                        float _rcp_94 = approx_rcp(1.0f + _exp2_94);
                        float sig_g_59_2 = _rcp_94;
                        float _tanh_approx_158;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_158) : "f"(x_g_58_2 * inv_beta));
                        frag[30] = beta * _tanh_approx_158 * sig_g_59_2;
                        float x_g_60_2 = frag[31];
                        float _exp2_95 = approx_exp2(x_g_60_2 * -1.4426950408889634f);
                        float _rcp_95 = approx_rcp(1.0f + _exp2_95);
                        float sig_g_61_2 = _rcp_95;
                        float _tanh_approx_159;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_159) : "f"(x_g_60_2 * inv_beta));
                        frag[31] = beta * _tanh_approx_159 * sig_g_61_2;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_3 + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_160;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_160) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_160;
                        float _tanh_approx_161;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_161) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_161;
                        float _tanh_approx_162;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_162) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_162;
                        float _tanh_approx_163;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_163) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_163;
                        float _tanh_approx_164;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_164) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_164;
                        float _tanh_approx_165;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_165) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_165;
                        float _tanh_approx_166;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_166) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_166;
                        float _tanh_approx_167;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_167) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_167;
                        float _tanh_approx_168;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_168) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_168;
                        float _tanh_approx_169;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_169) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_169;
                        float _tanh_approx_170;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_170) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_170;
                        float _tanh_approx_171;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_171) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_171;
                        float _tanh_approx_172;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_172) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_172;
                        float _tanh_approx_173;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_173) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_173;
                        float _tanh_approx_174;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_174) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_174;
                        float _tanh_approx_175;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_175) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_175;
                        float _tanh_approx_176;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_176) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_176;
                        float _tanh_approx_177;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_177) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_177;
                        float _tanh_approx_178;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_178) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_178;
                        float _tanh_approx_179;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_179) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_179;
                        float _tanh_approx_180;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_180) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_180;
                        float _tanh_approx_181;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_181) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_181;
                        float _tanh_approx_182;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_182) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_182;
                        float _tanh_approx_183;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_183) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_183;
                        float _tanh_approx_184;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_184) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_184;
                        float _tanh_approx_185;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_185) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_185;
                        float _tanh_approx_186;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_186) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_186;
                        float _tanh_approx_187;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_187) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_187;
                        float _tanh_approx_188;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_188) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_188;
                        float _tanh_approx_189;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_189) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_189;
                        float _tanh_approx_190;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_190) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_190;
                        float _tanh_approx_191;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_191) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_191;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_3 + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_160 = fmaxf(frag[0], -frag[0]);
                        float a_c_2 = _fmax_160;
                        float _fmax_161 = fmaxf(frag[2], -frag[2]);
                        float _fmax_162 = fmaxf(a_c_2, _fmax_161);
                        a_c_2 = _fmax_162;
                        float _fmax_163 = fmaxf(frag[16], -frag[16]);
                        float _fmax_164 = fmaxf(a_c_2, _fmax_163);
                        a_c_2 = _fmax_164;
                        float _fmax_165 = fmaxf(frag[18], -frag[18]);
                        float _fmax_166 = fmaxf(a_c_2, _fmax_165);
                        a_c_2 = _fmax_166;
                        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, a_c_2, 4);
                        float _fmax_167 = fmaxf(a_c_2, _shfl_xor_48);
                        a_c_2 = _fmax_167;
                        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, a_c_2, 8);
                        float _fmax_168 = fmaxf(a_c_2, _shfl_xor_49);
                        a_c_2 = _fmax_168;
                        float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, a_c_2, 16);
                        float _fmax_169 = fmaxf(a_c_2, _shfl_xor_50);
                        a_c_2 = _fmax_169;
                        uint16_t _ue8m0x2_f32_16;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_16) : "f"(zero_f32), "f"(a_c_2 * inv_fp8_max));
                        int code_full_3 = (int)_ue8m0x2_f32_16;
                        int code_3 = code_full_3 & 255;
                        int _max_16 = ((254 - code_3) > (0) ? (254 - code_3) : (0));
                        unsigned int inv_bits_2 = (unsigned int)(_max_16 << 23);
                        float inv_scale_2 = __uint_as_float(inv_bits_2) * (float)(code_3 != 0);
                        frag[0] = frag[0] * inv_scale_2;
                        frag[2] = frag[2] * inv_scale_2;
                        frag[16] = frag[16] * inv_scale_2;
                        frag[18] = frag[18] * inv_scale_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code_3;
                        }
                        float _fmax_170 = fmaxf(frag[1], -frag[1]);
                        float a_c_0_2 = _fmax_170;
                        float _fmax_171 = fmaxf(frag[3], -frag[3]);
                        float _fmax_172 = fmaxf(a_c_0_2, _fmax_171);
                        a_c_0_2 = _fmax_172;
                        float _fmax_173 = fmaxf(frag[17], -frag[17]);
                        float _fmax_174 = fmaxf(a_c_0_2, _fmax_173);
                        a_c_0_2 = _fmax_174;
                        float _fmax_175 = fmaxf(frag[19], -frag[19]);
                        float _fmax_176 = fmaxf(a_c_0_2, _fmax_175);
                        a_c_0_2 = _fmax_176;
                        float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_2, 4);
                        float _fmax_177 = fmaxf(a_c_0_2, _shfl_xor_51);
                        a_c_0_2 = _fmax_177;
                        float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_2, 8);
                        float _fmax_178 = fmaxf(a_c_0_2, _shfl_xor_52);
                        a_c_0_2 = _fmax_178;
                        float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_2, 16);
                        float _fmax_179 = fmaxf(a_c_0_2, _shfl_xor_53);
                        a_c_0_2 = _fmax_179;
                        uint16_t _ue8m0x2_f32_17;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_17) : "f"(zero_f32), "f"(a_c_0_2 * inv_fp8_max));
                        int code_full_1_2 = (int)_ue8m0x2_f32_17;
                        int code_2_2 = code_full_1_2 & 255;
                        int _max_17 = ((254 - code_2_2) > (0) ? (254 - code_2_2) : (0));
                        unsigned int inv_bits_3_2 = (unsigned int)(_max_17 << 23);
                        float inv_scale_4_2 = __uint_as_float(inv_bits_3_2) * (float)(code_2_2 != 0);
                        frag[1] = frag[1] * inv_scale_4_2;
                        frag[3] = frag[3] * inv_scale_4_2;
                        frag[17] = frag[17] * inv_scale_4_2;
                        frag[19] = frag[19] * inv_scale_4_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2_2;
                        }
                        float _fmax_180 = fmaxf(frag[4], -frag[4]);
                        float a_c_5_2 = _fmax_180;
                        float _fmax_181 = fmaxf(frag[6], -frag[6]);
                        float _fmax_182 = fmaxf(a_c_5_2, _fmax_181);
                        a_c_5_2 = _fmax_182;
                        float _fmax_183 = fmaxf(frag[20], -frag[20]);
                        float _fmax_184 = fmaxf(a_c_5_2, _fmax_183);
                        a_c_5_2 = _fmax_184;
                        float _fmax_185 = fmaxf(frag[22], -frag[22]);
                        float _fmax_186 = fmaxf(a_c_5_2, _fmax_185);
                        a_c_5_2 = _fmax_186;
                        float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_2, 4);
                        float _fmax_187 = fmaxf(a_c_5_2, _shfl_xor_54);
                        a_c_5_2 = _fmax_187;
                        float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_2, 8);
                        float _fmax_188 = fmaxf(a_c_5_2, _shfl_xor_55);
                        a_c_5_2 = _fmax_188;
                        float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_2, 16);
                        float _fmax_189 = fmaxf(a_c_5_2, _shfl_xor_56);
                        a_c_5_2 = _fmax_189;
                        uint16_t _ue8m0x2_f32_18;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_18) : "f"(zero_f32), "f"(a_c_5_2 * inv_fp8_max));
                        int code_full_6_2 = (int)_ue8m0x2_f32_18;
                        int code_7_2 = code_full_6_2 & 255;
                        int _max_18 = ((254 - code_7_2) > (0) ? (254 - code_7_2) : (0));
                        unsigned int inv_bits_8_2 = (unsigned int)(_max_18 << 23);
                        float inv_scale_9_2 = __uint_as_float(inv_bits_8_2) * (float)(code_7_2 != 0);
                        frag[4] = frag[4] * inv_scale_9_2;
                        frag[6] = frag[6] * inv_scale_9_2;
                        frag[20] = frag[20] * inv_scale_9_2;
                        frag[22] = frag[22] * inv_scale_9_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7_2;
                        }
                        float _fmax_190 = fmaxf(frag[5], -frag[5]);
                        float a_c_10_2 = _fmax_190;
                        float _fmax_191 = fmaxf(frag[7], -frag[7]);
                        float _fmax_192 = fmaxf(a_c_10_2, _fmax_191);
                        a_c_10_2 = _fmax_192;
                        float _fmax_193 = fmaxf(frag[21], -frag[21]);
                        float _fmax_194 = fmaxf(a_c_10_2, _fmax_193);
                        a_c_10_2 = _fmax_194;
                        float _fmax_195 = fmaxf(frag[23], -frag[23]);
                        float _fmax_196 = fmaxf(a_c_10_2, _fmax_195);
                        a_c_10_2 = _fmax_196;
                        float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_2, 4);
                        float _fmax_197 = fmaxf(a_c_10_2, _shfl_xor_57);
                        a_c_10_2 = _fmax_197;
                        float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_2, 8);
                        float _fmax_198 = fmaxf(a_c_10_2, _shfl_xor_58);
                        a_c_10_2 = _fmax_198;
                        float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_2, 16);
                        float _fmax_199 = fmaxf(a_c_10_2, _shfl_xor_59);
                        a_c_10_2 = _fmax_199;
                        uint16_t _ue8m0x2_f32_19;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_19) : "f"(zero_f32), "f"(a_c_10_2 * inv_fp8_max));
                        int code_full_11_2 = (int)_ue8m0x2_f32_19;
                        int code_12_2 = code_full_11_2 & 255;
                        int _max_19 = ((254 - code_12_2) > (0) ? (254 - code_12_2) : (0));
                        unsigned int inv_bits_13_2 = (unsigned int)(_max_19 << 23);
                        float inv_scale_14_2 = __uint_as_float(inv_bits_13_2) * (float)(code_12_2 != 0);
                        frag[5] = frag[5] * inv_scale_14_2;
                        frag[7] = frag[7] * inv_scale_14_2;
                        frag[21] = frag[21] * inv_scale_14_2;
                        frag[23] = frag[23] * inv_scale_14_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12_2;
                        }
                        float _fmax_200 = fmaxf(frag[8], -frag[8]);
                        float a_c_15_2 = _fmax_200;
                        float _fmax_201 = fmaxf(frag[10], -frag[10]);
                        float _fmax_202 = fmaxf(a_c_15_2, _fmax_201);
                        a_c_15_2 = _fmax_202;
                        float _fmax_203 = fmaxf(frag[24], -frag[24]);
                        float _fmax_204 = fmaxf(a_c_15_2, _fmax_203);
                        a_c_15_2 = _fmax_204;
                        float _fmax_205 = fmaxf(frag[26], -frag[26]);
                        float _fmax_206 = fmaxf(a_c_15_2, _fmax_205);
                        a_c_15_2 = _fmax_206;
                        float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_2, 4);
                        float _fmax_207 = fmaxf(a_c_15_2, _shfl_xor_60);
                        a_c_15_2 = _fmax_207;
                        float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_2, 8);
                        float _fmax_208 = fmaxf(a_c_15_2, _shfl_xor_61);
                        a_c_15_2 = _fmax_208;
                        float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_2, 16);
                        float _fmax_209 = fmaxf(a_c_15_2, _shfl_xor_62);
                        a_c_15_2 = _fmax_209;
                        uint16_t _ue8m0x2_f32_20;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_20) : "f"(zero_f32), "f"(a_c_15_2 * inv_fp8_max));
                        int code_full_16_2 = (int)_ue8m0x2_f32_20;
                        int code_17_2 = code_full_16_2 & 255;
                        int _max_20 = ((254 - code_17_2) > (0) ? (254 - code_17_2) : (0));
                        unsigned int inv_bits_18_2 = (unsigned int)(_max_20 << 23);
                        float inv_scale_19_2 = __uint_as_float(inv_bits_18_2) * (float)(code_17_2 != 0);
                        frag[8] = frag[8] * inv_scale_19_2;
                        frag[10] = frag[10] * inv_scale_19_2;
                        frag[24] = frag[24] * inv_scale_19_2;
                        frag[26] = frag[26] * inv_scale_19_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17_2;
                        }
                        float _fmax_210 = fmaxf(frag[9], -frag[9]);
                        float a_c_20_2 = _fmax_210;
                        float _fmax_211 = fmaxf(frag[11], -frag[11]);
                        float _fmax_212 = fmaxf(a_c_20_2, _fmax_211);
                        a_c_20_2 = _fmax_212;
                        float _fmax_213 = fmaxf(frag[25], -frag[25]);
                        float _fmax_214 = fmaxf(a_c_20_2, _fmax_213);
                        a_c_20_2 = _fmax_214;
                        float _fmax_215 = fmaxf(frag[27], -frag[27]);
                        float _fmax_216 = fmaxf(a_c_20_2, _fmax_215);
                        a_c_20_2 = _fmax_216;
                        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_2, 4);
                        float _fmax_217 = fmaxf(a_c_20_2, _shfl_xor_63);
                        a_c_20_2 = _fmax_217;
                        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_2, 8);
                        float _fmax_218 = fmaxf(a_c_20_2, _shfl_xor_64);
                        a_c_20_2 = _fmax_218;
                        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_2, 16);
                        float _fmax_219 = fmaxf(a_c_20_2, _shfl_xor_65);
                        a_c_20_2 = _fmax_219;
                        uint16_t _ue8m0x2_f32_21;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_21) : "f"(zero_f32), "f"(a_c_20_2 * inv_fp8_max));
                        int code_full_21_2 = (int)_ue8m0x2_f32_21;
                        int code_22_2 = code_full_21_2 & 255;
                        int _max_21 = ((254 - code_22_2) > (0) ? (254 - code_22_2) : (0));
                        unsigned int inv_bits_23_2 = (unsigned int)(_max_21 << 23);
                        float inv_scale_24_2 = __uint_as_float(inv_bits_23_2) * (float)(code_22_2 != 0);
                        frag[9] = frag[9] * inv_scale_24_2;
                        frag[11] = frag[11] * inv_scale_24_2;
                        frag[25] = frag[25] * inv_scale_24_2;
                        frag[27] = frag[27] * inv_scale_24_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22_2;
                        }
                        float _fmax_220 = fmaxf(frag[12], -frag[12]);
                        float a_c_25_2 = _fmax_220;
                        float _fmax_221 = fmaxf(frag[14], -frag[14]);
                        float _fmax_222 = fmaxf(a_c_25_2, _fmax_221);
                        a_c_25_2 = _fmax_222;
                        float _fmax_223 = fmaxf(frag[28], -frag[28]);
                        float _fmax_224 = fmaxf(a_c_25_2, _fmax_223);
                        a_c_25_2 = _fmax_224;
                        float _fmax_225 = fmaxf(frag[30], -frag[30]);
                        float _fmax_226 = fmaxf(a_c_25_2, _fmax_225);
                        a_c_25_2 = _fmax_226;
                        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_2, 4);
                        float _fmax_227 = fmaxf(a_c_25_2, _shfl_xor_66);
                        a_c_25_2 = _fmax_227;
                        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_2, 8);
                        float _fmax_228 = fmaxf(a_c_25_2, _shfl_xor_67);
                        a_c_25_2 = _fmax_228;
                        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_2, 16);
                        float _fmax_229 = fmaxf(a_c_25_2, _shfl_xor_68);
                        a_c_25_2 = _fmax_229;
                        uint16_t _ue8m0x2_f32_22;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_22) : "f"(zero_f32), "f"(a_c_25_2 * inv_fp8_max));
                        int code_full_26_2 = (int)_ue8m0x2_f32_22;
                        int code_27_2 = code_full_26_2 & 255;
                        int _max_22 = ((254 - code_27_2) > (0) ? (254 - code_27_2) : (0));
                        unsigned int inv_bits_28_2 = (unsigned int)(_max_22 << 23);
                        float inv_scale_29_2 = __uint_as_float(inv_bits_28_2) * (float)(code_27_2 != 0);
                        frag[12] = frag[12] * inv_scale_29_2;
                        frag[14] = frag[14] * inv_scale_29_2;
                        frag[28] = frag[28] * inv_scale_29_2;
                        frag[30] = frag[30] * inv_scale_29_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27_2;
                        }
                        float _fmax_230 = fmaxf(frag[13], -frag[13]);
                        float a_c_30_2 = _fmax_230;
                        float _fmax_231 = fmaxf(frag[15], -frag[15]);
                        float _fmax_232 = fmaxf(a_c_30_2, _fmax_231);
                        a_c_30_2 = _fmax_232;
                        float _fmax_233 = fmaxf(frag[29], -frag[29]);
                        float _fmax_234 = fmaxf(a_c_30_2, _fmax_233);
                        a_c_30_2 = _fmax_234;
                        float _fmax_235 = fmaxf(frag[31], -frag[31]);
                        float _fmax_236 = fmaxf(a_c_30_2, _fmax_235);
                        a_c_30_2 = _fmax_236;
                        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_2, 4);
                        float _fmax_237 = fmaxf(a_c_30_2, _shfl_xor_69);
                        a_c_30_2 = _fmax_237;
                        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_2, 8);
                        float _fmax_238 = fmaxf(a_c_30_2, _shfl_xor_70);
                        a_c_30_2 = _fmax_238;
                        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_2, 16);
                        float _fmax_239 = fmaxf(a_c_30_2, _shfl_xor_71);
                        a_c_30_2 = _fmax_239;
                        uint16_t _ue8m0x2_f32_23;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_23) : "f"(zero_f32), "f"(a_c_30_2 * inv_fp8_max));
                        int code_full_31_2 = (int)_ue8m0x2_f32_23;
                        int code_32_2 = code_full_31_2 & 255;
                        int _max_23 = ((254 - code_32_2) > (0) ? (254 - code_32_2) : (0));
                        unsigned int inv_bits_33_2 = (unsigned int)(_max_23 << 23);
                        float inv_scale_34_2 = __uint_as_float(inv_bits_33_2) * (float)(code_32_2 != 0);
                        frag[13] = frag[13] * inv_scale_34_2;
                        frag[15] = frag[15] * inv_scale_34_2;
                        frag[29] = frag[29] * inv_scale_34_2;
                        frag[31] = frag[31] * inv_scale_34_2;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32_2;
                        }
                        uint32_t _fp8_2[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_2[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_2[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_2[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_2[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_2[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_2[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_2[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_2[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_4 = static_cast<uint32_t>(sact_buf_4 + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_4), "r"(_fp8_2[0]), "r"(_fp8_2[1]), "r"(_fp8_2[2]), "r"(_fp8_2[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_5 = static_cast<uint32_t>(sact_buf_4 + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_5), "r"(_fp8_2[4]), "r"(_fp8_2[5]), "r"(_fp8_2[6]), "r"(_fp8_2[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s_5 = row_base + 64 + st_tok;
                    if (prow_s_5 < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf_4 + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s_5 * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l_2 = row_base + 64 + lane_0;
                        if (prow_l_2 < mn_limit) {
                            unsigned int code_l_2 = scode[warp_in64 * 32 + lane_0];
                            int sf_off_l_2 = prow_l_2 % 32 * 16 + prow_l_2 / 32 % 4 * 4 + prow_l_2 / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l_2) + (0)) = (unsigned char)(code_l_2);
                        }
                    }
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
                    frag[0] = _tmem_load_6[0] * meta_alpha;
                    frag[16] = _tmem_load_7[0] * meta_alpha;
                    frag[1] = _tmem_load_6[1] * meta_alpha;
                    frag[17] = _tmem_load_7[1] * meta_alpha;
                    frag[2] = _tmem_load_6[2] * meta_alpha;
                    frag[18] = _tmem_load_7[2] * meta_alpha;
                    frag[3] = _tmem_load_6[3] * meta_alpha;
                    frag[19] = _tmem_load_7[3] * meta_alpha;
                    frag[4] = _tmem_load_6[4] * meta_alpha;
                    frag[20] = _tmem_load_7[4] * meta_alpha;
                    frag[5] = _tmem_load_6[5] * meta_alpha;
                    frag[21] = _tmem_load_7[5] * meta_alpha;
                    frag[6] = _tmem_load_6[6] * meta_alpha;
                    frag[22] = _tmem_load_7[6] * meta_alpha;
                    frag[7] = _tmem_load_6[7] * meta_alpha;
                    frag[23] = _tmem_load_7[7] * meta_alpha;
                    frag[8] = _tmem_load_6[8] * meta_alpha;
                    frag[24] = _tmem_load_7[8] * meta_alpha;
                    frag[9] = _tmem_load_6[9] * meta_alpha;
                    frag[25] = _tmem_load_7[9] * meta_alpha;
                    frag[10] = _tmem_load_6[10] * meta_alpha;
                    frag[26] = _tmem_load_7[10] * meta_alpha;
                    frag[11] = _tmem_load_6[11] * meta_alpha;
                    frag[27] = _tmem_load_7[11] * meta_alpha;
                    frag[12] = _tmem_load_6[12] * meta_alpha;
                    frag[28] = _tmem_load_7[12] * meta_alpha;
                    frag[13] = _tmem_load_6[13] * meta_alpha;
                    frag[29] = _tmem_load_7[13] * meta_alpha;
                    frag[14] = _tmem_load_6[14] * meta_alpha;
                    frag[30] = _tmem_load_7[14] * meta_alpha;
                    frag[15] = _tmem_load_6[15] * meta_alpha;
                    frag[31] = _tmem_load_7[15] * meta_alpha;
                    int exchf_buf_6 = sexchf_addr + 8192;
                    int sact_buf_7 = sact_addr + 4608;
                    if (is_gate_lane != 0) {
                        float x_g_5 = frag[0];
                        float _exp2_96 = approx_exp2(x_g_5 * -1.4426950408889634f);
                        float _rcp_96 = approx_rcp(1.0f + _exp2_96);
                        float sig_g_6 = _rcp_96;
                        float _tanh_approx_192;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_192) : "f"(x_g_5 * inv_beta));
                        frag[0] = beta * _tanh_approx_192 * sig_g_6;
                        float x_g_0_3 = frag[1];
                        float _exp2_97 = approx_exp2(x_g_0_3 * -1.4426950408889634f);
                        float _rcp_97 = approx_rcp(1.0f + _exp2_97);
                        float sig_g_1_3 = _rcp_97;
                        float _tanh_approx_193;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_193) : "f"(x_g_0_3 * inv_beta));
                        frag[1] = beta * _tanh_approx_193 * sig_g_1_3;
                        float x_g_2_3 = frag[2];
                        float _exp2_98 = approx_exp2(x_g_2_3 * -1.4426950408889634f);
                        float _rcp_98 = approx_rcp(1.0f + _exp2_98);
                        float sig_g_3_3 = _rcp_98;
                        float _tanh_approx_194;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_194) : "f"(x_g_2_3 * inv_beta));
                        frag[2] = beta * _tanh_approx_194 * sig_g_3_3;
                        float x_g_4_3 = frag[3];
                        float _exp2_99 = approx_exp2(x_g_4_3 * -1.4426950408889634f);
                        float _rcp_99 = approx_rcp(1.0f + _exp2_99);
                        float sig_g_5_3 = _rcp_99;
                        float _tanh_approx_195;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_195) : "f"(x_g_4_3 * inv_beta));
                        frag[3] = beta * _tanh_approx_195 * sig_g_5_3;
                        float x_g_6_3 = frag[4];
                        float _exp2_100 = approx_exp2(x_g_6_3 * -1.4426950408889634f);
                        float _rcp_100 = approx_rcp(1.0f + _exp2_100);
                        float sig_g_7_3 = _rcp_100;
                        float _tanh_approx_196;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_196) : "f"(x_g_6_3 * inv_beta));
                        frag[4] = beta * _tanh_approx_196 * sig_g_7_3;
                        float x_g_8_3 = frag[5];
                        float _exp2_101 = approx_exp2(x_g_8_3 * -1.4426950408889634f);
                        float _rcp_101 = approx_rcp(1.0f + _exp2_101);
                        float sig_g_9_3 = _rcp_101;
                        float _tanh_approx_197;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_197) : "f"(x_g_8_3 * inv_beta));
                        frag[5] = beta * _tanh_approx_197 * sig_g_9_3;
                        float x_g_10_3 = frag[6];
                        float _exp2_102 = approx_exp2(x_g_10_3 * -1.4426950408889634f);
                        float _rcp_102 = approx_rcp(1.0f + _exp2_102);
                        float sig_g_11_3 = _rcp_102;
                        float _tanh_approx_198;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_198) : "f"(x_g_10_3 * inv_beta));
                        frag[6] = beta * _tanh_approx_198 * sig_g_11_3;
                        float x_g_12_3 = frag[7];
                        float _exp2_103 = approx_exp2(x_g_12_3 * -1.4426950408889634f);
                        float _rcp_103 = approx_rcp(1.0f + _exp2_103);
                        float sig_g_13_3 = _rcp_103;
                        float _tanh_approx_199;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_199) : "f"(x_g_12_3 * inv_beta));
                        frag[7] = beta * _tanh_approx_199 * sig_g_13_3;
                        float x_g_14_3 = frag[8];
                        float _exp2_104 = approx_exp2(x_g_14_3 * -1.4426950408889634f);
                        float _rcp_104 = approx_rcp(1.0f + _exp2_104);
                        float sig_g_15_3 = _rcp_104;
                        float _tanh_approx_200;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_200) : "f"(x_g_14_3 * inv_beta));
                        frag[8] = beta * _tanh_approx_200 * sig_g_15_3;
                        float x_g_16_3 = frag[9];
                        float _exp2_105 = approx_exp2(x_g_16_3 * -1.4426950408889634f);
                        float _rcp_105 = approx_rcp(1.0f + _exp2_105);
                        float sig_g_17_3 = _rcp_105;
                        float _tanh_approx_201;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_201) : "f"(x_g_16_3 * inv_beta));
                        frag[9] = beta * _tanh_approx_201 * sig_g_17_3;
                        float x_g_18_3 = frag[10];
                        float _exp2_106 = approx_exp2(x_g_18_3 * -1.4426950408889634f);
                        float _rcp_106 = approx_rcp(1.0f + _exp2_106);
                        float sig_g_19_3 = _rcp_106;
                        float _tanh_approx_202;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_202) : "f"(x_g_18_3 * inv_beta));
                        frag[10] = beta * _tanh_approx_202 * sig_g_19_3;
                        float x_g_20_3 = frag[11];
                        float _exp2_107 = approx_exp2(x_g_20_3 * -1.4426950408889634f);
                        float _rcp_107 = approx_rcp(1.0f + _exp2_107);
                        float sig_g_21_3 = _rcp_107;
                        float _tanh_approx_203;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_203) : "f"(x_g_20_3 * inv_beta));
                        frag[11] = beta * _tanh_approx_203 * sig_g_21_3;
                        float x_g_22_3 = frag[12];
                        float _exp2_108 = approx_exp2(x_g_22_3 * -1.4426950408889634f);
                        float _rcp_108 = approx_rcp(1.0f + _exp2_108);
                        float sig_g_23_3 = _rcp_108;
                        float _tanh_approx_204;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_204) : "f"(x_g_22_3 * inv_beta));
                        frag[12] = beta * _tanh_approx_204 * sig_g_23_3;
                        float x_g_24_3 = frag[13];
                        float _exp2_109 = approx_exp2(x_g_24_3 * -1.4426950408889634f);
                        float _rcp_109 = approx_rcp(1.0f + _exp2_109);
                        float sig_g_25_3 = _rcp_109;
                        float _tanh_approx_205;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_205) : "f"(x_g_24_3 * inv_beta));
                        frag[13] = beta * _tanh_approx_205 * sig_g_25_3;
                        float x_g_26_3 = frag[14];
                        float _exp2_110 = approx_exp2(x_g_26_3 * -1.4426950408889634f);
                        float _rcp_110 = approx_rcp(1.0f + _exp2_110);
                        float sig_g_27_3 = _rcp_110;
                        float _tanh_approx_206;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_206) : "f"(x_g_26_3 * inv_beta));
                        frag[14] = beta * _tanh_approx_206 * sig_g_27_3;
                        float x_g_28_3 = frag[15];
                        float _exp2_111 = approx_exp2(x_g_28_3 * -1.4426950408889634f);
                        float _rcp_111 = approx_rcp(1.0f + _exp2_111);
                        float sig_g_29_3 = _rcp_111;
                        float _tanh_approx_207;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_207) : "f"(x_g_28_3 * inv_beta));
                        frag[15] = beta * _tanh_approx_207 * sig_g_29_3;
                        float x_g_30_3 = frag[16];
                        float _exp2_112 = approx_exp2(x_g_30_3 * -1.4426950408889634f);
                        float _rcp_112 = approx_rcp(1.0f + _exp2_112);
                        float sig_g_31_3 = _rcp_112;
                        float _tanh_approx_208;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_208) : "f"(x_g_30_3 * inv_beta));
                        frag[16] = beta * _tanh_approx_208 * sig_g_31_3;
                        float x_g_32_3 = frag[17];
                        float _exp2_113 = approx_exp2(x_g_32_3 * -1.4426950408889634f);
                        float _rcp_113 = approx_rcp(1.0f + _exp2_113);
                        float sig_g_33_3 = _rcp_113;
                        float _tanh_approx_209;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_209) : "f"(x_g_32_3 * inv_beta));
                        frag[17] = beta * _tanh_approx_209 * sig_g_33_3;
                        float x_g_34_3 = frag[18];
                        float _exp2_114 = approx_exp2(x_g_34_3 * -1.4426950408889634f);
                        float _rcp_114 = approx_rcp(1.0f + _exp2_114);
                        float sig_g_35_3 = _rcp_114;
                        float _tanh_approx_210;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_210) : "f"(x_g_34_3 * inv_beta));
                        frag[18] = beta * _tanh_approx_210 * sig_g_35_3;
                        float x_g_36_3 = frag[19];
                        float _exp2_115 = approx_exp2(x_g_36_3 * -1.4426950408889634f);
                        float _rcp_115 = approx_rcp(1.0f + _exp2_115);
                        float sig_g_37_3 = _rcp_115;
                        float _tanh_approx_211;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_211) : "f"(x_g_36_3 * inv_beta));
                        frag[19] = beta * _tanh_approx_211 * sig_g_37_3;
                        float x_g_38_3 = frag[20];
                        float _exp2_116 = approx_exp2(x_g_38_3 * -1.4426950408889634f);
                        float _rcp_116 = approx_rcp(1.0f + _exp2_116);
                        float sig_g_39_3 = _rcp_116;
                        float _tanh_approx_212;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_212) : "f"(x_g_38_3 * inv_beta));
                        frag[20] = beta * _tanh_approx_212 * sig_g_39_3;
                        float x_g_40_3 = frag[21];
                        float _exp2_117 = approx_exp2(x_g_40_3 * -1.4426950408889634f);
                        float _rcp_117 = approx_rcp(1.0f + _exp2_117);
                        float sig_g_41_3 = _rcp_117;
                        float _tanh_approx_213;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_213) : "f"(x_g_40_3 * inv_beta));
                        frag[21] = beta * _tanh_approx_213 * sig_g_41_3;
                        float x_g_42_3 = frag[22];
                        float _exp2_118 = approx_exp2(x_g_42_3 * -1.4426950408889634f);
                        float _rcp_118 = approx_rcp(1.0f + _exp2_118);
                        float sig_g_43_3 = _rcp_118;
                        float _tanh_approx_214;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_214) : "f"(x_g_42_3 * inv_beta));
                        frag[22] = beta * _tanh_approx_214 * sig_g_43_3;
                        float x_g_44_3 = frag[23];
                        float _exp2_119 = approx_exp2(x_g_44_3 * -1.4426950408889634f);
                        float _rcp_119 = approx_rcp(1.0f + _exp2_119);
                        float sig_g_45_3 = _rcp_119;
                        float _tanh_approx_215;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_215) : "f"(x_g_44_3 * inv_beta));
                        frag[23] = beta * _tanh_approx_215 * sig_g_45_3;
                        float x_g_46_3 = frag[24];
                        float _exp2_120 = approx_exp2(x_g_46_3 * -1.4426950408889634f);
                        float _rcp_120 = approx_rcp(1.0f + _exp2_120);
                        float sig_g_47_3 = _rcp_120;
                        float _tanh_approx_216;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_216) : "f"(x_g_46_3 * inv_beta));
                        frag[24] = beta * _tanh_approx_216 * sig_g_47_3;
                        float x_g_48_3 = frag[25];
                        float _exp2_121 = approx_exp2(x_g_48_3 * -1.4426950408889634f);
                        float _rcp_121 = approx_rcp(1.0f + _exp2_121);
                        float sig_g_49_3 = _rcp_121;
                        float _tanh_approx_217;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_217) : "f"(x_g_48_3 * inv_beta));
                        frag[25] = beta * _tanh_approx_217 * sig_g_49_3;
                        float x_g_50_3 = frag[26];
                        float _exp2_122 = approx_exp2(x_g_50_3 * -1.4426950408889634f);
                        float _rcp_122 = approx_rcp(1.0f + _exp2_122);
                        float sig_g_51_3 = _rcp_122;
                        float _tanh_approx_218;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_218) : "f"(x_g_50_3 * inv_beta));
                        frag[26] = beta * _tanh_approx_218 * sig_g_51_3;
                        float x_g_52_3 = frag[27];
                        float _exp2_123 = approx_exp2(x_g_52_3 * -1.4426950408889634f);
                        float _rcp_123 = approx_rcp(1.0f + _exp2_123);
                        float sig_g_53_3 = _rcp_123;
                        float _tanh_approx_219;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_219) : "f"(x_g_52_3 * inv_beta));
                        frag[27] = beta * _tanh_approx_219 * sig_g_53_3;
                        float x_g_54_3 = frag[28];
                        float _exp2_124 = approx_exp2(x_g_54_3 * -1.4426950408889634f);
                        float _rcp_124 = approx_rcp(1.0f + _exp2_124);
                        float sig_g_55_3 = _rcp_124;
                        float _tanh_approx_220;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_220) : "f"(x_g_54_3 * inv_beta));
                        frag[28] = beta * _tanh_approx_220 * sig_g_55_3;
                        float x_g_56_3 = frag[29];
                        float _exp2_125 = approx_exp2(x_g_56_3 * -1.4426950408889634f);
                        float _rcp_125 = approx_rcp(1.0f + _exp2_125);
                        float sig_g_57_3 = _rcp_125;
                        float _tanh_approx_221;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_221) : "f"(x_g_56_3 * inv_beta));
                        frag[29] = beta * _tanh_approx_221 * sig_g_57_3;
                        float x_g_58_3 = frag[30];
                        float _exp2_126 = approx_exp2(x_g_58_3 * -1.4426950408889634f);
                        float _rcp_126 = approx_rcp(1.0f + _exp2_126);
                        float sig_g_59_3 = _rcp_126;
                        float _tanh_approx_222;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_222) : "f"(x_g_58_3 * inv_beta));
                        frag[30] = beta * _tanh_approx_222 * sig_g_59_3;
                        float x_g_60_3 = frag[31];
                        float _exp2_127 = approx_exp2(x_g_60_3 * -1.4426950408889634f);
                        float _rcp_127 = approx_rcp(1.0f + _exp2_127);
                        float sig_g_61_3 = _rcp_127;
                        float _tanh_approx_223;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_223) : "f"(x_g_60_3 * inv_beta));
                        frag[31] = beta * _tanh_approx_223 * sig_g_61_3;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_6 + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_224;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_224) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_224;
                        float _tanh_approx_225;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_225) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_225;
                        float _tanh_approx_226;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_226) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_226;
                        float _tanh_approx_227;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_227) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_227;
                        float _tanh_approx_228;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_228) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_228;
                        float _tanh_approx_229;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_229) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_229;
                        float _tanh_approx_230;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_230) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_230;
                        float _tanh_approx_231;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_231) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_231;
                        float _tanh_approx_232;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_232) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_232;
                        float _tanh_approx_233;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_233) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_233;
                        float _tanh_approx_234;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_234) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_234;
                        float _tanh_approx_235;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_235) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_235;
                        float _tanh_approx_236;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_236) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_236;
                        float _tanh_approx_237;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_237) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_237;
                        float _tanh_approx_238;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_238) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_238;
                        float _tanh_approx_239;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_239) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_239;
                        float _tanh_approx_240;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_240) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_240;
                        float _tanh_approx_241;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_241) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_241;
                        float _tanh_approx_242;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_242) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_242;
                        float _tanh_approx_243;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_243) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_243;
                        float _tanh_approx_244;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_244) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_244;
                        float _tanh_approx_245;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_245) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_245;
                        float _tanh_approx_246;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_246) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_246;
                        float _tanh_approx_247;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_247) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_247;
                        float _tanh_approx_248;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_248) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_248;
                        float _tanh_approx_249;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_249) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_249;
                        float _tanh_approx_250;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_250) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_250;
                        float _tanh_approx_251;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_251) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_251;
                        float _tanh_approx_252;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_252) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_252;
                        float _tanh_approx_253;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_253) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_253;
                        float _tanh_approx_254;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_254) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_254;
                        float _tanh_approx_255;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_255) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_255;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_6 + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_240 = fmaxf(frag[0], -frag[0]);
                        float a_c_3 = _fmax_240;
                        float _fmax_241 = fmaxf(frag[2], -frag[2]);
                        float _fmax_242 = fmaxf(a_c_3, _fmax_241);
                        a_c_3 = _fmax_242;
                        float _fmax_243 = fmaxf(frag[16], -frag[16]);
                        float _fmax_244 = fmaxf(a_c_3, _fmax_243);
                        a_c_3 = _fmax_244;
                        float _fmax_245 = fmaxf(frag[18], -frag[18]);
                        float _fmax_246 = fmaxf(a_c_3, _fmax_245);
                        a_c_3 = _fmax_246;
                        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, a_c_3, 4);
                        float _fmax_247 = fmaxf(a_c_3, _shfl_xor_72);
                        a_c_3 = _fmax_247;
                        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, a_c_3, 8);
                        float _fmax_248 = fmaxf(a_c_3, _shfl_xor_73);
                        a_c_3 = _fmax_248;
                        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, a_c_3, 16);
                        float _fmax_249 = fmaxf(a_c_3, _shfl_xor_74);
                        a_c_3 = _fmax_249;
                        uint16_t _ue8m0x2_f32_24;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_24) : "f"(zero_f32), "f"(a_c_3 * inv_fp8_max));
                        int code_full_4 = (int)_ue8m0x2_f32_24;
                        int code_4 = code_full_4 & 255;
                        int _max_24 = ((254 - code_4) > (0) ? (254 - code_4) : (0));
                        unsigned int inv_bits_4 = (unsigned int)(_max_24 << 23);
                        float inv_scale_3 = __uint_as_float(inv_bits_4) * (float)(code_4 != 0);
                        frag[0] = frag[0] * inv_scale_3;
                        frag[2] = frag[2] * inv_scale_3;
                        frag[16] = frag[16] * inv_scale_3;
                        frag[18] = frag[18] * inv_scale_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code_4;
                        }
                        float _fmax_250 = fmaxf(frag[1], -frag[1]);
                        float a_c_0_3 = _fmax_250;
                        float _fmax_251 = fmaxf(frag[3], -frag[3]);
                        float _fmax_252 = fmaxf(a_c_0_3, _fmax_251);
                        a_c_0_3 = _fmax_252;
                        float _fmax_253 = fmaxf(frag[17], -frag[17]);
                        float _fmax_254 = fmaxf(a_c_0_3, _fmax_253);
                        a_c_0_3 = _fmax_254;
                        float _fmax_255 = fmaxf(frag[19], -frag[19]);
                        float _fmax_256 = fmaxf(a_c_0_3, _fmax_255);
                        a_c_0_3 = _fmax_256;
                        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_3, 4);
                        float _fmax_257 = fmaxf(a_c_0_3, _shfl_xor_75);
                        a_c_0_3 = _fmax_257;
                        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_3, 8);
                        float _fmax_258 = fmaxf(a_c_0_3, _shfl_xor_76);
                        a_c_0_3 = _fmax_258;
                        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_3, 16);
                        float _fmax_259 = fmaxf(a_c_0_3, _shfl_xor_77);
                        a_c_0_3 = _fmax_259;
                        uint16_t _ue8m0x2_f32_25;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_25) : "f"(zero_f32), "f"(a_c_0_3 * inv_fp8_max));
                        int code_full_1_3 = (int)_ue8m0x2_f32_25;
                        int code_2_3 = code_full_1_3 & 255;
                        int _max_25 = ((254 - code_2_3) > (0) ? (254 - code_2_3) : (0));
                        unsigned int inv_bits_3_3 = (unsigned int)(_max_25 << 23);
                        float inv_scale_4_3 = __uint_as_float(inv_bits_3_3) * (float)(code_2_3 != 0);
                        frag[1] = frag[1] * inv_scale_4_3;
                        frag[3] = frag[3] * inv_scale_4_3;
                        frag[17] = frag[17] * inv_scale_4_3;
                        frag[19] = frag[19] * inv_scale_4_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2_3;
                        }
                        float _fmax_260 = fmaxf(frag[4], -frag[4]);
                        float a_c_5_3 = _fmax_260;
                        float _fmax_261 = fmaxf(frag[6], -frag[6]);
                        float _fmax_262 = fmaxf(a_c_5_3, _fmax_261);
                        a_c_5_3 = _fmax_262;
                        float _fmax_263 = fmaxf(frag[20], -frag[20]);
                        float _fmax_264 = fmaxf(a_c_5_3, _fmax_263);
                        a_c_5_3 = _fmax_264;
                        float _fmax_265 = fmaxf(frag[22], -frag[22]);
                        float _fmax_266 = fmaxf(a_c_5_3, _fmax_265);
                        a_c_5_3 = _fmax_266;
                        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_3, 4);
                        float _fmax_267 = fmaxf(a_c_5_3, _shfl_xor_78);
                        a_c_5_3 = _fmax_267;
                        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_3, 8);
                        float _fmax_268 = fmaxf(a_c_5_3, _shfl_xor_79);
                        a_c_5_3 = _fmax_268;
                        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_3, 16);
                        float _fmax_269 = fmaxf(a_c_5_3, _shfl_xor_80);
                        a_c_5_3 = _fmax_269;
                        uint16_t _ue8m0x2_f32_26;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_26) : "f"(zero_f32), "f"(a_c_5_3 * inv_fp8_max));
                        int code_full_6_3 = (int)_ue8m0x2_f32_26;
                        int code_7_3 = code_full_6_3 & 255;
                        int _max_26 = ((254 - code_7_3) > (0) ? (254 - code_7_3) : (0));
                        unsigned int inv_bits_8_3 = (unsigned int)(_max_26 << 23);
                        float inv_scale_9_3 = __uint_as_float(inv_bits_8_3) * (float)(code_7_3 != 0);
                        frag[4] = frag[4] * inv_scale_9_3;
                        frag[6] = frag[6] * inv_scale_9_3;
                        frag[20] = frag[20] * inv_scale_9_3;
                        frag[22] = frag[22] * inv_scale_9_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7_3;
                        }
                        float _fmax_270 = fmaxf(frag[5], -frag[5]);
                        float a_c_10_3 = _fmax_270;
                        float _fmax_271 = fmaxf(frag[7], -frag[7]);
                        float _fmax_272 = fmaxf(a_c_10_3, _fmax_271);
                        a_c_10_3 = _fmax_272;
                        float _fmax_273 = fmaxf(frag[21], -frag[21]);
                        float _fmax_274 = fmaxf(a_c_10_3, _fmax_273);
                        a_c_10_3 = _fmax_274;
                        float _fmax_275 = fmaxf(frag[23], -frag[23]);
                        float _fmax_276 = fmaxf(a_c_10_3, _fmax_275);
                        a_c_10_3 = _fmax_276;
                        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_3, 4);
                        float _fmax_277 = fmaxf(a_c_10_3, _shfl_xor_81);
                        a_c_10_3 = _fmax_277;
                        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_3, 8);
                        float _fmax_278 = fmaxf(a_c_10_3, _shfl_xor_82);
                        a_c_10_3 = _fmax_278;
                        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_3, 16);
                        float _fmax_279 = fmaxf(a_c_10_3, _shfl_xor_83);
                        a_c_10_3 = _fmax_279;
                        uint16_t _ue8m0x2_f32_27;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_27) : "f"(zero_f32), "f"(a_c_10_3 * inv_fp8_max));
                        int code_full_11_3 = (int)_ue8m0x2_f32_27;
                        int code_12_3 = code_full_11_3 & 255;
                        int _max_27 = ((254 - code_12_3) > (0) ? (254 - code_12_3) : (0));
                        unsigned int inv_bits_13_3 = (unsigned int)(_max_27 << 23);
                        float inv_scale_14_3 = __uint_as_float(inv_bits_13_3) * (float)(code_12_3 != 0);
                        frag[5] = frag[5] * inv_scale_14_3;
                        frag[7] = frag[7] * inv_scale_14_3;
                        frag[21] = frag[21] * inv_scale_14_3;
                        frag[23] = frag[23] * inv_scale_14_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12_3;
                        }
                        float _fmax_280 = fmaxf(frag[8], -frag[8]);
                        float a_c_15_3 = _fmax_280;
                        float _fmax_281 = fmaxf(frag[10], -frag[10]);
                        float _fmax_282 = fmaxf(a_c_15_3, _fmax_281);
                        a_c_15_3 = _fmax_282;
                        float _fmax_283 = fmaxf(frag[24], -frag[24]);
                        float _fmax_284 = fmaxf(a_c_15_3, _fmax_283);
                        a_c_15_3 = _fmax_284;
                        float _fmax_285 = fmaxf(frag[26], -frag[26]);
                        float _fmax_286 = fmaxf(a_c_15_3, _fmax_285);
                        a_c_15_3 = _fmax_286;
                        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_3, 4);
                        float _fmax_287 = fmaxf(a_c_15_3, _shfl_xor_84);
                        a_c_15_3 = _fmax_287;
                        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_3, 8);
                        float _fmax_288 = fmaxf(a_c_15_3, _shfl_xor_85);
                        a_c_15_3 = _fmax_288;
                        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_3, 16);
                        float _fmax_289 = fmaxf(a_c_15_3, _shfl_xor_86);
                        a_c_15_3 = _fmax_289;
                        uint16_t _ue8m0x2_f32_28;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_28) : "f"(zero_f32), "f"(a_c_15_3 * inv_fp8_max));
                        int code_full_16_3 = (int)_ue8m0x2_f32_28;
                        int code_17_3 = code_full_16_3 & 255;
                        int _max_28 = ((254 - code_17_3) > (0) ? (254 - code_17_3) : (0));
                        unsigned int inv_bits_18_3 = (unsigned int)(_max_28 << 23);
                        float inv_scale_19_3 = __uint_as_float(inv_bits_18_3) * (float)(code_17_3 != 0);
                        frag[8] = frag[8] * inv_scale_19_3;
                        frag[10] = frag[10] * inv_scale_19_3;
                        frag[24] = frag[24] * inv_scale_19_3;
                        frag[26] = frag[26] * inv_scale_19_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17_3;
                        }
                        float _fmax_290 = fmaxf(frag[9], -frag[9]);
                        float a_c_20_3 = _fmax_290;
                        float _fmax_291 = fmaxf(frag[11], -frag[11]);
                        float _fmax_292 = fmaxf(a_c_20_3, _fmax_291);
                        a_c_20_3 = _fmax_292;
                        float _fmax_293 = fmaxf(frag[25], -frag[25]);
                        float _fmax_294 = fmaxf(a_c_20_3, _fmax_293);
                        a_c_20_3 = _fmax_294;
                        float _fmax_295 = fmaxf(frag[27], -frag[27]);
                        float _fmax_296 = fmaxf(a_c_20_3, _fmax_295);
                        a_c_20_3 = _fmax_296;
                        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_3, 4);
                        float _fmax_297 = fmaxf(a_c_20_3, _shfl_xor_87);
                        a_c_20_3 = _fmax_297;
                        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_3, 8);
                        float _fmax_298 = fmaxf(a_c_20_3, _shfl_xor_88);
                        a_c_20_3 = _fmax_298;
                        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_3, 16);
                        float _fmax_299 = fmaxf(a_c_20_3, _shfl_xor_89);
                        a_c_20_3 = _fmax_299;
                        uint16_t _ue8m0x2_f32_29;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_29) : "f"(zero_f32), "f"(a_c_20_3 * inv_fp8_max));
                        int code_full_21_3 = (int)_ue8m0x2_f32_29;
                        int code_22_3 = code_full_21_3 & 255;
                        int _max_29 = ((254 - code_22_3) > (0) ? (254 - code_22_3) : (0));
                        unsigned int inv_bits_23_3 = (unsigned int)(_max_29 << 23);
                        float inv_scale_24_3 = __uint_as_float(inv_bits_23_3) * (float)(code_22_3 != 0);
                        frag[9] = frag[9] * inv_scale_24_3;
                        frag[11] = frag[11] * inv_scale_24_3;
                        frag[25] = frag[25] * inv_scale_24_3;
                        frag[27] = frag[27] * inv_scale_24_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22_3;
                        }
                        float _fmax_300 = fmaxf(frag[12], -frag[12]);
                        float a_c_25_3 = _fmax_300;
                        float _fmax_301 = fmaxf(frag[14], -frag[14]);
                        float _fmax_302 = fmaxf(a_c_25_3, _fmax_301);
                        a_c_25_3 = _fmax_302;
                        float _fmax_303 = fmaxf(frag[28], -frag[28]);
                        float _fmax_304 = fmaxf(a_c_25_3, _fmax_303);
                        a_c_25_3 = _fmax_304;
                        float _fmax_305 = fmaxf(frag[30], -frag[30]);
                        float _fmax_306 = fmaxf(a_c_25_3, _fmax_305);
                        a_c_25_3 = _fmax_306;
                        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_3, 4);
                        float _fmax_307 = fmaxf(a_c_25_3, _shfl_xor_90);
                        a_c_25_3 = _fmax_307;
                        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_3, 8);
                        float _fmax_308 = fmaxf(a_c_25_3, _shfl_xor_91);
                        a_c_25_3 = _fmax_308;
                        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_3, 16);
                        float _fmax_309 = fmaxf(a_c_25_3, _shfl_xor_92);
                        a_c_25_3 = _fmax_309;
                        uint16_t _ue8m0x2_f32_30;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_30) : "f"(zero_f32), "f"(a_c_25_3 * inv_fp8_max));
                        int code_full_26_3 = (int)_ue8m0x2_f32_30;
                        int code_27_3 = code_full_26_3 & 255;
                        int _max_30 = ((254 - code_27_3) > (0) ? (254 - code_27_3) : (0));
                        unsigned int inv_bits_28_3 = (unsigned int)(_max_30 << 23);
                        float inv_scale_29_3 = __uint_as_float(inv_bits_28_3) * (float)(code_27_3 != 0);
                        frag[12] = frag[12] * inv_scale_29_3;
                        frag[14] = frag[14] * inv_scale_29_3;
                        frag[28] = frag[28] * inv_scale_29_3;
                        frag[30] = frag[30] * inv_scale_29_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27_3;
                        }
                        float _fmax_310 = fmaxf(frag[13], -frag[13]);
                        float a_c_30_3 = _fmax_310;
                        float _fmax_311 = fmaxf(frag[15], -frag[15]);
                        float _fmax_312 = fmaxf(a_c_30_3, _fmax_311);
                        a_c_30_3 = _fmax_312;
                        float _fmax_313 = fmaxf(frag[29], -frag[29]);
                        float _fmax_314 = fmaxf(a_c_30_3, _fmax_313);
                        a_c_30_3 = _fmax_314;
                        float _fmax_315 = fmaxf(frag[31], -frag[31]);
                        float _fmax_316 = fmaxf(a_c_30_3, _fmax_315);
                        a_c_30_3 = _fmax_316;
                        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_3, 4);
                        float _fmax_317 = fmaxf(a_c_30_3, _shfl_xor_93);
                        a_c_30_3 = _fmax_317;
                        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_3, 8);
                        float _fmax_318 = fmaxf(a_c_30_3, _shfl_xor_94);
                        a_c_30_3 = _fmax_318;
                        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_3, 16);
                        float _fmax_319 = fmaxf(a_c_30_3, _shfl_xor_95);
                        a_c_30_3 = _fmax_319;
                        uint16_t _ue8m0x2_f32_31;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_31) : "f"(zero_f32), "f"(a_c_30_3 * inv_fp8_max));
                        int code_full_31_3 = (int)_ue8m0x2_f32_31;
                        int code_32_3 = code_full_31_3 & 255;
                        int _max_31 = ((254 - code_32_3) > (0) ? (254 - code_32_3) : (0));
                        unsigned int inv_bits_33_3 = (unsigned int)(_max_31 << 23);
                        float inv_scale_34_3 = __uint_as_float(inv_bits_33_3) * (float)(code_32_3 != 0);
                        frag[13] = frag[13] * inv_scale_34_3;
                        frag[15] = frag[15] * inv_scale_34_3;
                        frag[29] = frag[29] * inv_scale_34_3;
                        frag[31] = frag[31] * inv_scale_34_3;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32_3;
                        }
                        uint32_t _fp8_3[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_3[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_3[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_3[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_3[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_3[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_3[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_3[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_3[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_6 = static_cast<uint32_t>(sact_buf_7 + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_6), "r"(_fp8_3[0]), "r"(_fp8_3[1]), "r"(_fp8_3[2]), "r"(_fp8_3[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_7 = static_cast<uint32_t>(sact_buf_7 + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_7), "r"(_fp8_3[4]), "r"(_fp8_3[5]), "r"(_fp8_3[6]), "r"(_fp8_3[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s_8 = row_base + 96 + st_tok;
                    if (prow_s_8 < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf_7 + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s_8 * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l_3 = row_base + 96 + lane_0;
                        if (prow_l_3 < mn_limit) {
                            unsigned int code_l_3 = scode[64 + warp_in64 * 32 + lane_0];
                            int sf_off_l_3 = prow_l_3 % 32 * 16 + prow_l_3 / 32 % 4 * 4 + prow_l_3 / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l_3) + (0)) = (unsigned char)(code_l_3);
                        }
                    }
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
                    frag[0] = _tmem_load_8[0] * meta_alpha;
                    frag[16] = _tmem_load_9[0] * meta_alpha;
                    frag[1] = _tmem_load_8[1] * meta_alpha;
                    frag[17] = _tmem_load_9[1] * meta_alpha;
                    frag[2] = _tmem_load_8[2] * meta_alpha;
                    frag[18] = _tmem_load_9[2] * meta_alpha;
                    frag[3] = _tmem_load_8[3] * meta_alpha;
                    frag[19] = _tmem_load_9[3] * meta_alpha;
                    frag[4] = _tmem_load_8[4] * meta_alpha;
                    frag[20] = _tmem_load_9[4] * meta_alpha;
                    frag[5] = _tmem_load_8[5] * meta_alpha;
                    frag[21] = _tmem_load_9[5] * meta_alpha;
                    frag[6] = _tmem_load_8[6] * meta_alpha;
                    frag[22] = _tmem_load_9[6] * meta_alpha;
                    frag[7] = _tmem_load_8[7] * meta_alpha;
                    frag[23] = _tmem_load_9[7] * meta_alpha;
                    frag[8] = _tmem_load_8[8] * meta_alpha;
                    frag[24] = _tmem_load_9[8] * meta_alpha;
                    frag[9] = _tmem_load_8[9] * meta_alpha;
                    frag[25] = _tmem_load_9[9] * meta_alpha;
                    frag[10] = _tmem_load_8[10] * meta_alpha;
                    frag[26] = _tmem_load_9[10] * meta_alpha;
                    frag[11] = _tmem_load_8[11] * meta_alpha;
                    frag[27] = _tmem_load_9[11] * meta_alpha;
                    frag[12] = _tmem_load_8[12] * meta_alpha;
                    frag[28] = _tmem_load_9[12] * meta_alpha;
                    frag[13] = _tmem_load_8[13] * meta_alpha;
                    frag[29] = _tmem_load_9[13] * meta_alpha;
                    frag[14] = _tmem_load_8[14] * meta_alpha;
                    frag[30] = _tmem_load_9[14] * meta_alpha;
                    frag[15] = _tmem_load_8[15] * meta_alpha;
                    frag[31] = _tmem_load_9[15] * meta_alpha;
                    int exchf_buf_9 = sexchf_addr;
                    int sact_buf_10 = sact_addr;
                    if (is_gate_lane != 0) {
                        float x_g_7 = frag[0];
                        float _exp2_128 = approx_exp2(x_g_7 * -1.4426950408889634f);
                        float _rcp_128 = approx_rcp(1.0f + _exp2_128);
                        float sig_g_8 = _rcp_128;
                        float _tanh_approx_256;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_256) : "f"(x_g_7 * inv_beta));
                        frag[0] = beta * _tanh_approx_256 * sig_g_8;
                        float x_g_0_4 = frag[1];
                        float _exp2_129 = approx_exp2(x_g_0_4 * -1.4426950408889634f);
                        float _rcp_129 = approx_rcp(1.0f + _exp2_129);
                        float sig_g_1_4 = _rcp_129;
                        float _tanh_approx_257;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_257) : "f"(x_g_0_4 * inv_beta));
                        frag[1] = beta * _tanh_approx_257 * sig_g_1_4;
                        float x_g_2_4 = frag[2];
                        float _exp2_130 = approx_exp2(x_g_2_4 * -1.4426950408889634f);
                        float _rcp_130 = approx_rcp(1.0f + _exp2_130);
                        float sig_g_3_4 = _rcp_130;
                        float _tanh_approx_258;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_258) : "f"(x_g_2_4 * inv_beta));
                        frag[2] = beta * _tanh_approx_258 * sig_g_3_4;
                        float x_g_4_4 = frag[3];
                        float _exp2_131 = approx_exp2(x_g_4_4 * -1.4426950408889634f);
                        float _rcp_131 = approx_rcp(1.0f + _exp2_131);
                        float sig_g_5_4 = _rcp_131;
                        float _tanh_approx_259;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_259) : "f"(x_g_4_4 * inv_beta));
                        frag[3] = beta * _tanh_approx_259 * sig_g_5_4;
                        float x_g_6_4 = frag[4];
                        float _exp2_132 = approx_exp2(x_g_6_4 * -1.4426950408889634f);
                        float _rcp_132 = approx_rcp(1.0f + _exp2_132);
                        float sig_g_7_4 = _rcp_132;
                        float _tanh_approx_260;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_260) : "f"(x_g_6_4 * inv_beta));
                        frag[4] = beta * _tanh_approx_260 * sig_g_7_4;
                        float x_g_8_4 = frag[5];
                        float _exp2_133 = approx_exp2(x_g_8_4 * -1.4426950408889634f);
                        float _rcp_133 = approx_rcp(1.0f + _exp2_133);
                        float sig_g_9_4 = _rcp_133;
                        float _tanh_approx_261;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_261) : "f"(x_g_8_4 * inv_beta));
                        frag[5] = beta * _tanh_approx_261 * sig_g_9_4;
                        float x_g_10_4 = frag[6];
                        float _exp2_134 = approx_exp2(x_g_10_4 * -1.4426950408889634f);
                        float _rcp_134 = approx_rcp(1.0f + _exp2_134);
                        float sig_g_11_4 = _rcp_134;
                        float _tanh_approx_262;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_262) : "f"(x_g_10_4 * inv_beta));
                        frag[6] = beta * _tanh_approx_262 * sig_g_11_4;
                        float x_g_12_4 = frag[7];
                        float _exp2_135 = approx_exp2(x_g_12_4 * -1.4426950408889634f);
                        float _rcp_135 = approx_rcp(1.0f + _exp2_135);
                        float sig_g_13_4 = _rcp_135;
                        float _tanh_approx_263;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_263) : "f"(x_g_12_4 * inv_beta));
                        frag[7] = beta * _tanh_approx_263 * sig_g_13_4;
                        float x_g_14_4 = frag[8];
                        float _exp2_136 = approx_exp2(x_g_14_4 * -1.4426950408889634f);
                        float _rcp_136 = approx_rcp(1.0f + _exp2_136);
                        float sig_g_15_4 = _rcp_136;
                        float _tanh_approx_264;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_264) : "f"(x_g_14_4 * inv_beta));
                        frag[8] = beta * _tanh_approx_264 * sig_g_15_4;
                        float x_g_16_4 = frag[9];
                        float _exp2_137 = approx_exp2(x_g_16_4 * -1.4426950408889634f);
                        float _rcp_137 = approx_rcp(1.0f + _exp2_137);
                        float sig_g_17_4 = _rcp_137;
                        float _tanh_approx_265;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_265) : "f"(x_g_16_4 * inv_beta));
                        frag[9] = beta * _tanh_approx_265 * sig_g_17_4;
                        float x_g_18_4 = frag[10];
                        float _exp2_138 = approx_exp2(x_g_18_4 * -1.4426950408889634f);
                        float _rcp_138 = approx_rcp(1.0f + _exp2_138);
                        float sig_g_19_4 = _rcp_138;
                        float _tanh_approx_266;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_266) : "f"(x_g_18_4 * inv_beta));
                        frag[10] = beta * _tanh_approx_266 * sig_g_19_4;
                        float x_g_20_4 = frag[11];
                        float _exp2_139 = approx_exp2(x_g_20_4 * -1.4426950408889634f);
                        float _rcp_139 = approx_rcp(1.0f + _exp2_139);
                        float sig_g_21_4 = _rcp_139;
                        float _tanh_approx_267;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_267) : "f"(x_g_20_4 * inv_beta));
                        frag[11] = beta * _tanh_approx_267 * sig_g_21_4;
                        float x_g_22_4 = frag[12];
                        float _exp2_140 = approx_exp2(x_g_22_4 * -1.4426950408889634f);
                        float _rcp_140 = approx_rcp(1.0f + _exp2_140);
                        float sig_g_23_4 = _rcp_140;
                        float _tanh_approx_268;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_268) : "f"(x_g_22_4 * inv_beta));
                        frag[12] = beta * _tanh_approx_268 * sig_g_23_4;
                        float x_g_24_4 = frag[13];
                        float _exp2_141 = approx_exp2(x_g_24_4 * -1.4426950408889634f);
                        float _rcp_141 = approx_rcp(1.0f + _exp2_141);
                        float sig_g_25_4 = _rcp_141;
                        float _tanh_approx_269;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_269) : "f"(x_g_24_4 * inv_beta));
                        frag[13] = beta * _tanh_approx_269 * sig_g_25_4;
                        float x_g_26_4 = frag[14];
                        float _exp2_142 = approx_exp2(x_g_26_4 * -1.4426950408889634f);
                        float _rcp_142 = approx_rcp(1.0f + _exp2_142);
                        float sig_g_27_4 = _rcp_142;
                        float _tanh_approx_270;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_270) : "f"(x_g_26_4 * inv_beta));
                        frag[14] = beta * _tanh_approx_270 * sig_g_27_4;
                        float x_g_28_4 = frag[15];
                        float _exp2_143 = approx_exp2(x_g_28_4 * -1.4426950408889634f);
                        float _rcp_143 = approx_rcp(1.0f + _exp2_143);
                        float sig_g_29_4 = _rcp_143;
                        float _tanh_approx_271;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_271) : "f"(x_g_28_4 * inv_beta));
                        frag[15] = beta * _tanh_approx_271 * sig_g_29_4;
                        float x_g_30_4 = frag[16];
                        float _exp2_144 = approx_exp2(x_g_30_4 * -1.4426950408889634f);
                        float _rcp_144 = approx_rcp(1.0f + _exp2_144);
                        float sig_g_31_4 = _rcp_144;
                        float _tanh_approx_272;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_272) : "f"(x_g_30_4 * inv_beta));
                        frag[16] = beta * _tanh_approx_272 * sig_g_31_4;
                        float x_g_32_4 = frag[17];
                        float _exp2_145 = approx_exp2(x_g_32_4 * -1.4426950408889634f);
                        float _rcp_145 = approx_rcp(1.0f + _exp2_145);
                        float sig_g_33_4 = _rcp_145;
                        float _tanh_approx_273;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_273) : "f"(x_g_32_4 * inv_beta));
                        frag[17] = beta * _tanh_approx_273 * sig_g_33_4;
                        float x_g_34_4 = frag[18];
                        float _exp2_146 = approx_exp2(x_g_34_4 * -1.4426950408889634f);
                        float _rcp_146 = approx_rcp(1.0f + _exp2_146);
                        float sig_g_35_4 = _rcp_146;
                        float _tanh_approx_274;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_274) : "f"(x_g_34_4 * inv_beta));
                        frag[18] = beta * _tanh_approx_274 * sig_g_35_4;
                        float x_g_36_4 = frag[19];
                        float _exp2_147 = approx_exp2(x_g_36_4 * -1.4426950408889634f);
                        float _rcp_147 = approx_rcp(1.0f + _exp2_147);
                        float sig_g_37_4 = _rcp_147;
                        float _tanh_approx_275;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_275) : "f"(x_g_36_4 * inv_beta));
                        frag[19] = beta * _tanh_approx_275 * sig_g_37_4;
                        float x_g_38_4 = frag[20];
                        float _exp2_148 = approx_exp2(x_g_38_4 * -1.4426950408889634f);
                        float _rcp_148 = approx_rcp(1.0f + _exp2_148);
                        float sig_g_39_4 = _rcp_148;
                        float _tanh_approx_276;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_276) : "f"(x_g_38_4 * inv_beta));
                        frag[20] = beta * _tanh_approx_276 * sig_g_39_4;
                        float x_g_40_4 = frag[21];
                        float _exp2_149 = approx_exp2(x_g_40_4 * -1.4426950408889634f);
                        float _rcp_149 = approx_rcp(1.0f + _exp2_149);
                        float sig_g_41_4 = _rcp_149;
                        float _tanh_approx_277;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_277) : "f"(x_g_40_4 * inv_beta));
                        frag[21] = beta * _tanh_approx_277 * sig_g_41_4;
                        float x_g_42_4 = frag[22];
                        float _exp2_150 = approx_exp2(x_g_42_4 * -1.4426950408889634f);
                        float _rcp_150 = approx_rcp(1.0f + _exp2_150);
                        float sig_g_43_4 = _rcp_150;
                        float _tanh_approx_278;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_278) : "f"(x_g_42_4 * inv_beta));
                        frag[22] = beta * _tanh_approx_278 * sig_g_43_4;
                        float x_g_44_4 = frag[23];
                        float _exp2_151 = approx_exp2(x_g_44_4 * -1.4426950408889634f);
                        float _rcp_151 = approx_rcp(1.0f + _exp2_151);
                        float sig_g_45_4 = _rcp_151;
                        float _tanh_approx_279;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_279) : "f"(x_g_44_4 * inv_beta));
                        frag[23] = beta * _tanh_approx_279 * sig_g_45_4;
                        float x_g_46_4 = frag[24];
                        float _exp2_152 = approx_exp2(x_g_46_4 * -1.4426950408889634f);
                        float _rcp_152 = approx_rcp(1.0f + _exp2_152);
                        float sig_g_47_4 = _rcp_152;
                        float _tanh_approx_280;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_280) : "f"(x_g_46_4 * inv_beta));
                        frag[24] = beta * _tanh_approx_280 * sig_g_47_4;
                        float x_g_48_4 = frag[25];
                        float _exp2_153 = approx_exp2(x_g_48_4 * -1.4426950408889634f);
                        float _rcp_153 = approx_rcp(1.0f + _exp2_153);
                        float sig_g_49_4 = _rcp_153;
                        float _tanh_approx_281;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_281) : "f"(x_g_48_4 * inv_beta));
                        frag[25] = beta * _tanh_approx_281 * sig_g_49_4;
                        float x_g_50_4 = frag[26];
                        float _exp2_154 = approx_exp2(x_g_50_4 * -1.4426950408889634f);
                        float _rcp_154 = approx_rcp(1.0f + _exp2_154);
                        float sig_g_51_4 = _rcp_154;
                        float _tanh_approx_282;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_282) : "f"(x_g_50_4 * inv_beta));
                        frag[26] = beta * _tanh_approx_282 * sig_g_51_4;
                        float x_g_52_4 = frag[27];
                        float _exp2_155 = approx_exp2(x_g_52_4 * -1.4426950408889634f);
                        float _rcp_155 = approx_rcp(1.0f + _exp2_155);
                        float sig_g_53_4 = _rcp_155;
                        float _tanh_approx_283;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_283) : "f"(x_g_52_4 * inv_beta));
                        frag[27] = beta * _tanh_approx_283 * sig_g_53_4;
                        float x_g_54_4 = frag[28];
                        float _exp2_156 = approx_exp2(x_g_54_4 * -1.4426950408889634f);
                        float _rcp_156 = approx_rcp(1.0f + _exp2_156);
                        float sig_g_55_4 = _rcp_156;
                        float _tanh_approx_284;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_284) : "f"(x_g_54_4 * inv_beta));
                        frag[28] = beta * _tanh_approx_284 * sig_g_55_4;
                        float x_g_56_4 = frag[29];
                        float _exp2_157 = approx_exp2(x_g_56_4 * -1.4426950408889634f);
                        float _rcp_157 = approx_rcp(1.0f + _exp2_157);
                        float sig_g_57_4 = _rcp_157;
                        float _tanh_approx_285;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_285) : "f"(x_g_56_4 * inv_beta));
                        frag[29] = beta * _tanh_approx_285 * sig_g_57_4;
                        float x_g_58_4 = frag[30];
                        float _exp2_158 = approx_exp2(x_g_58_4 * -1.4426950408889634f);
                        float _rcp_158 = approx_rcp(1.0f + _exp2_158);
                        float sig_g_59_4 = _rcp_158;
                        float _tanh_approx_286;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_286) : "f"(x_g_58_4 * inv_beta));
                        frag[30] = beta * _tanh_approx_286 * sig_g_59_4;
                        float x_g_60_4 = frag[31];
                        float _exp2_159 = approx_exp2(x_g_60_4 * -1.4426950408889634f);
                        float _rcp_159 = approx_rcp(1.0f + _exp2_159);
                        float sig_g_61_4 = _rcp_159;
                        float _tanh_approx_287;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_287) : "f"(x_g_60_4 * inv_beta));
                        frag[31] = beta * _tanh_approx_287 * sig_g_61_4;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_9 + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_288;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_288) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_288;
                        float _tanh_approx_289;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_289) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_289;
                        float _tanh_approx_290;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_290) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_290;
                        float _tanh_approx_291;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_291) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_291;
                        float _tanh_approx_292;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_292) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_292;
                        float _tanh_approx_293;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_293) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_293;
                        float _tanh_approx_294;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_294) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_294;
                        float _tanh_approx_295;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_295) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_295;
                        float _tanh_approx_296;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_296) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_296;
                        float _tanh_approx_297;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_297) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_297;
                        float _tanh_approx_298;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_298) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_298;
                        float _tanh_approx_299;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_299) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_299;
                        float _tanh_approx_300;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_300) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_300;
                        float _tanh_approx_301;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_301) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_301;
                        float _tanh_approx_302;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_302) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_302;
                        float _tanh_approx_303;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_303) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_303;
                        float _tanh_approx_304;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_304) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_304;
                        float _tanh_approx_305;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_305) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_305;
                        float _tanh_approx_306;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_306) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_306;
                        float _tanh_approx_307;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_307) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_307;
                        float _tanh_approx_308;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_308) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_308;
                        float _tanh_approx_309;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_309) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_309;
                        float _tanh_approx_310;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_310) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_310;
                        float _tanh_approx_311;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_311) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_311;
                        float _tanh_approx_312;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_312) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_312;
                        float _tanh_approx_313;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_313) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_313;
                        float _tanh_approx_314;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_314) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_314;
                        float _tanh_approx_315;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_315) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_315;
                        float _tanh_approx_316;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_316) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_316;
                        float _tanh_approx_317;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_317) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_317;
                        float _tanh_approx_318;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_318) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_318;
                        float _tanh_approx_319;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_319) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_319;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_9 + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_320 = fmaxf(frag[0], -frag[0]);
                        float a_c_4 = _fmax_320;
                        float _fmax_321 = fmaxf(frag[2], -frag[2]);
                        float _fmax_322 = fmaxf(a_c_4, _fmax_321);
                        a_c_4 = _fmax_322;
                        float _fmax_323 = fmaxf(frag[16], -frag[16]);
                        float _fmax_324 = fmaxf(a_c_4, _fmax_323);
                        a_c_4 = _fmax_324;
                        float _fmax_325 = fmaxf(frag[18], -frag[18]);
                        float _fmax_326 = fmaxf(a_c_4, _fmax_325);
                        a_c_4 = _fmax_326;
                        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, a_c_4, 4);
                        float _fmax_327 = fmaxf(a_c_4, _shfl_xor_96);
                        a_c_4 = _fmax_327;
                        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, a_c_4, 8);
                        float _fmax_328 = fmaxf(a_c_4, _shfl_xor_97);
                        a_c_4 = _fmax_328;
                        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, a_c_4, 16);
                        float _fmax_329 = fmaxf(a_c_4, _shfl_xor_98);
                        a_c_4 = _fmax_329;
                        uint16_t _ue8m0x2_f32_32;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_32) : "f"(zero_f32), "f"(a_c_4 * inv_fp8_max));
                        int code_full_5 = (int)_ue8m0x2_f32_32;
                        int code_5 = code_full_5 & 255;
                        int _max_32 = ((254 - code_5) > (0) ? (254 - code_5) : (0));
                        unsigned int inv_bits_5 = (unsigned int)(_max_32 << 23);
                        float inv_scale_5 = __uint_as_float(inv_bits_5) * (float)(code_5 != 0);
                        frag[0] = frag[0] * inv_scale_5;
                        frag[2] = frag[2] * inv_scale_5;
                        frag[16] = frag[16] * inv_scale_5;
                        frag[18] = frag[18] * inv_scale_5;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code_5;
                        }
                        float _fmax_330 = fmaxf(frag[1], -frag[1]);
                        float a_c_0_4 = _fmax_330;
                        float _fmax_331 = fmaxf(frag[3], -frag[3]);
                        float _fmax_332 = fmaxf(a_c_0_4, _fmax_331);
                        a_c_0_4 = _fmax_332;
                        float _fmax_333 = fmaxf(frag[17], -frag[17]);
                        float _fmax_334 = fmaxf(a_c_0_4, _fmax_333);
                        a_c_0_4 = _fmax_334;
                        float _fmax_335 = fmaxf(frag[19], -frag[19]);
                        float _fmax_336 = fmaxf(a_c_0_4, _fmax_335);
                        a_c_0_4 = _fmax_336;
                        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_4, 4);
                        float _fmax_337 = fmaxf(a_c_0_4, _shfl_xor_99);
                        a_c_0_4 = _fmax_337;
                        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_4, 8);
                        float _fmax_338 = fmaxf(a_c_0_4, _shfl_xor_100);
                        a_c_0_4 = _fmax_338;
                        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_4, 16);
                        float _fmax_339 = fmaxf(a_c_0_4, _shfl_xor_101);
                        a_c_0_4 = _fmax_339;
                        uint16_t _ue8m0x2_f32_33;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_33) : "f"(zero_f32), "f"(a_c_0_4 * inv_fp8_max));
                        int code_full_1_4 = (int)_ue8m0x2_f32_33;
                        int code_2_4 = code_full_1_4 & 255;
                        int _max_33 = ((254 - code_2_4) > (0) ? (254 - code_2_4) : (0));
                        unsigned int inv_bits_3_4 = (unsigned int)(_max_33 << 23);
                        float inv_scale_4_4 = __uint_as_float(inv_bits_3_4) * (float)(code_2_4 != 0);
                        frag[1] = frag[1] * inv_scale_4_4;
                        frag[3] = frag[3] * inv_scale_4_4;
                        frag[17] = frag[17] * inv_scale_4_4;
                        frag[19] = frag[19] * inv_scale_4_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2_4;
                        }
                        float _fmax_340 = fmaxf(frag[4], -frag[4]);
                        float a_c_5_4 = _fmax_340;
                        float _fmax_341 = fmaxf(frag[6], -frag[6]);
                        float _fmax_342 = fmaxf(a_c_5_4, _fmax_341);
                        a_c_5_4 = _fmax_342;
                        float _fmax_343 = fmaxf(frag[20], -frag[20]);
                        float _fmax_344 = fmaxf(a_c_5_4, _fmax_343);
                        a_c_5_4 = _fmax_344;
                        float _fmax_345 = fmaxf(frag[22], -frag[22]);
                        float _fmax_346 = fmaxf(a_c_5_4, _fmax_345);
                        a_c_5_4 = _fmax_346;
                        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_4, 4);
                        float _fmax_347 = fmaxf(a_c_5_4, _shfl_xor_102);
                        a_c_5_4 = _fmax_347;
                        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_4, 8);
                        float _fmax_348 = fmaxf(a_c_5_4, _shfl_xor_103);
                        a_c_5_4 = _fmax_348;
                        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_4, 16);
                        float _fmax_349 = fmaxf(a_c_5_4, _shfl_xor_104);
                        a_c_5_4 = _fmax_349;
                        uint16_t _ue8m0x2_f32_34;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_34) : "f"(zero_f32), "f"(a_c_5_4 * inv_fp8_max));
                        int code_full_6_4 = (int)_ue8m0x2_f32_34;
                        int code_7_4 = code_full_6_4 & 255;
                        int _max_34 = ((254 - code_7_4) > (0) ? (254 - code_7_4) : (0));
                        unsigned int inv_bits_8_4 = (unsigned int)(_max_34 << 23);
                        float inv_scale_9_4 = __uint_as_float(inv_bits_8_4) * (float)(code_7_4 != 0);
                        frag[4] = frag[4] * inv_scale_9_4;
                        frag[6] = frag[6] * inv_scale_9_4;
                        frag[20] = frag[20] * inv_scale_9_4;
                        frag[22] = frag[22] * inv_scale_9_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7_4;
                        }
                        float _fmax_350 = fmaxf(frag[5], -frag[5]);
                        float a_c_10_4 = _fmax_350;
                        float _fmax_351 = fmaxf(frag[7], -frag[7]);
                        float _fmax_352 = fmaxf(a_c_10_4, _fmax_351);
                        a_c_10_4 = _fmax_352;
                        float _fmax_353 = fmaxf(frag[21], -frag[21]);
                        float _fmax_354 = fmaxf(a_c_10_4, _fmax_353);
                        a_c_10_4 = _fmax_354;
                        float _fmax_355 = fmaxf(frag[23], -frag[23]);
                        float _fmax_356 = fmaxf(a_c_10_4, _fmax_355);
                        a_c_10_4 = _fmax_356;
                        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_4, 4);
                        float _fmax_357 = fmaxf(a_c_10_4, _shfl_xor_105);
                        a_c_10_4 = _fmax_357;
                        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_4, 8);
                        float _fmax_358 = fmaxf(a_c_10_4, _shfl_xor_106);
                        a_c_10_4 = _fmax_358;
                        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_4, 16);
                        float _fmax_359 = fmaxf(a_c_10_4, _shfl_xor_107);
                        a_c_10_4 = _fmax_359;
                        uint16_t _ue8m0x2_f32_35;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_35) : "f"(zero_f32), "f"(a_c_10_4 * inv_fp8_max));
                        int code_full_11_4 = (int)_ue8m0x2_f32_35;
                        int code_12_4 = code_full_11_4 & 255;
                        int _max_35 = ((254 - code_12_4) > (0) ? (254 - code_12_4) : (0));
                        unsigned int inv_bits_13_4 = (unsigned int)(_max_35 << 23);
                        float inv_scale_14_4 = __uint_as_float(inv_bits_13_4) * (float)(code_12_4 != 0);
                        frag[5] = frag[5] * inv_scale_14_4;
                        frag[7] = frag[7] * inv_scale_14_4;
                        frag[21] = frag[21] * inv_scale_14_4;
                        frag[23] = frag[23] * inv_scale_14_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12_4;
                        }
                        float _fmax_360 = fmaxf(frag[8], -frag[8]);
                        float a_c_15_4 = _fmax_360;
                        float _fmax_361 = fmaxf(frag[10], -frag[10]);
                        float _fmax_362 = fmaxf(a_c_15_4, _fmax_361);
                        a_c_15_4 = _fmax_362;
                        float _fmax_363 = fmaxf(frag[24], -frag[24]);
                        float _fmax_364 = fmaxf(a_c_15_4, _fmax_363);
                        a_c_15_4 = _fmax_364;
                        float _fmax_365 = fmaxf(frag[26], -frag[26]);
                        float _fmax_366 = fmaxf(a_c_15_4, _fmax_365);
                        a_c_15_4 = _fmax_366;
                        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_4, 4);
                        float _fmax_367 = fmaxf(a_c_15_4, _shfl_xor_108);
                        a_c_15_4 = _fmax_367;
                        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_4, 8);
                        float _fmax_368 = fmaxf(a_c_15_4, _shfl_xor_109);
                        a_c_15_4 = _fmax_368;
                        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_4, 16);
                        float _fmax_369 = fmaxf(a_c_15_4, _shfl_xor_110);
                        a_c_15_4 = _fmax_369;
                        uint16_t _ue8m0x2_f32_36;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_36) : "f"(zero_f32), "f"(a_c_15_4 * inv_fp8_max));
                        int code_full_16_4 = (int)_ue8m0x2_f32_36;
                        int code_17_4 = code_full_16_4 & 255;
                        int _max_36 = ((254 - code_17_4) > (0) ? (254 - code_17_4) : (0));
                        unsigned int inv_bits_18_4 = (unsigned int)(_max_36 << 23);
                        float inv_scale_19_4 = __uint_as_float(inv_bits_18_4) * (float)(code_17_4 != 0);
                        frag[8] = frag[8] * inv_scale_19_4;
                        frag[10] = frag[10] * inv_scale_19_4;
                        frag[24] = frag[24] * inv_scale_19_4;
                        frag[26] = frag[26] * inv_scale_19_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17_4;
                        }
                        float _fmax_370 = fmaxf(frag[9], -frag[9]);
                        float a_c_20_4 = _fmax_370;
                        float _fmax_371 = fmaxf(frag[11], -frag[11]);
                        float _fmax_372 = fmaxf(a_c_20_4, _fmax_371);
                        a_c_20_4 = _fmax_372;
                        float _fmax_373 = fmaxf(frag[25], -frag[25]);
                        float _fmax_374 = fmaxf(a_c_20_4, _fmax_373);
                        a_c_20_4 = _fmax_374;
                        float _fmax_375 = fmaxf(frag[27], -frag[27]);
                        float _fmax_376 = fmaxf(a_c_20_4, _fmax_375);
                        a_c_20_4 = _fmax_376;
                        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_4, 4);
                        float _fmax_377 = fmaxf(a_c_20_4, _shfl_xor_111);
                        a_c_20_4 = _fmax_377;
                        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_4, 8);
                        float _fmax_378 = fmaxf(a_c_20_4, _shfl_xor_112);
                        a_c_20_4 = _fmax_378;
                        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_4, 16);
                        float _fmax_379 = fmaxf(a_c_20_4, _shfl_xor_113);
                        a_c_20_4 = _fmax_379;
                        uint16_t _ue8m0x2_f32_37;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_37) : "f"(zero_f32), "f"(a_c_20_4 * inv_fp8_max));
                        int code_full_21_4 = (int)_ue8m0x2_f32_37;
                        int code_22_4 = code_full_21_4 & 255;
                        int _max_37 = ((254 - code_22_4) > (0) ? (254 - code_22_4) : (0));
                        unsigned int inv_bits_23_4 = (unsigned int)(_max_37 << 23);
                        float inv_scale_24_4 = __uint_as_float(inv_bits_23_4) * (float)(code_22_4 != 0);
                        frag[9] = frag[9] * inv_scale_24_4;
                        frag[11] = frag[11] * inv_scale_24_4;
                        frag[25] = frag[25] * inv_scale_24_4;
                        frag[27] = frag[27] * inv_scale_24_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22_4;
                        }
                        float _fmax_380 = fmaxf(frag[12], -frag[12]);
                        float a_c_25_4 = _fmax_380;
                        float _fmax_381 = fmaxf(frag[14], -frag[14]);
                        float _fmax_382 = fmaxf(a_c_25_4, _fmax_381);
                        a_c_25_4 = _fmax_382;
                        float _fmax_383 = fmaxf(frag[28], -frag[28]);
                        float _fmax_384 = fmaxf(a_c_25_4, _fmax_383);
                        a_c_25_4 = _fmax_384;
                        float _fmax_385 = fmaxf(frag[30], -frag[30]);
                        float _fmax_386 = fmaxf(a_c_25_4, _fmax_385);
                        a_c_25_4 = _fmax_386;
                        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_4, 4);
                        float _fmax_387 = fmaxf(a_c_25_4, _shfl_xor_114);
                        a_c_25_4 = _fmax_387;
                        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_4, 8);
                        float _fmax_388 = fmaxf(a_c_25_4, _shfl_xor_115);
                        a_c_25_4 = _fmax_388;
                        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_4, 16);
                        float _fmax_389 = fmaxf(a_c_25_4, _shfl_xor_116);
                        a_c_25_4 = _fmax_389;
                        uint16_t _ue8m0x2_f32_38;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_38) : "f"(zero_f32), "f"(a_c_25_4 * inv_fp8_max));
                        int code_full_26_4 = (int)_ue8m0x2_f32_38;
                        int code_27_4 = code_full_26_4 & 255;
                        int _max_38 = ((254 - code_27_4) > (0) ? (254 - code_27_4) : (0));
                        unsigned int inv_bits_28_4 = (unsigned int)(_max_38 << 23);
                        float inv_scale_29_4 = __uint_as_float(inv_bits_28_4) * (float)(code_27_4 != 0);
                        frag[12] = frag[12] * inv_scale_29_4;
                        frag[14] = frag[14] * inv_scale_29_4;
                        frag[28] = frag[28] * inv_scale_29_4;
                        frag[30] = frag[30] * inv_scale_29_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27_4;
                        }
                        float _fmax_390 = fmaxf(frag[13], -frag[13]);
                        float a_c_30_4 = _fmax_390;
                        float _fmax_391 = fmaxf(frag[15], -frag[15]);
                        float _fmax_392 = fmaxf(a_c_30_4, _fmax_391);
                        a_c_30_4 = _fmax_392;
                        float _fmax_393 = fmaxf(frag[29], -frag[29]);
                        float _fmax_394 = fmaxf(a_c_30_4, _fmax_393);
                        a_c_30_4 = _fmax_394;
                        float _fmax_395 = fmaxf(frag[31], -frag[31]);
                        float _fmax_396 = fmaxf(a_c_30_4, _fmax_395);
                        a_c_30_4 = _fmax_396;
                        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_4, 4);
                        float _fmax_397 = fmaxf(a_c_30_4, _shfl_xor_117);
                        a_c_30_4 = _fmax_397;
                        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_4, 8);
                        float _fmax_398 = fmaxf(a_c_30_4, _shfl_xor_118);
                        a_c_30_4 = _fmax_398;
                        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_4, 16);
                        float _fmax_399 = fmaxf(a_c_30_4, _shfl_xor_119);
                        a_c_30_4 = _fmax_399;
                        uint16_t _ue8m0x2_f32_39;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_39) : "f"(zero_f32), "f"(a_c_30_4 * inv_fp8_max));
                        int code_full_31_4 = (int)_ue8m0x2_f32_39;
                        int code_32_4 = code_full_31_4 & 255;
                        int _max_39 = ((254 - code_32_4) > (0) ? (254 - code_32_4) : (0));
                        unsigned int inv_bits_33_4 = (unsigned int)(_max_39 << 23);
                        float inv_scale_34_4 = __uint_as_float(inv_bits_33_4) * (float)(code_32_4 != 0);
                        frag[13] = frag[13] * inv_scale_34_4;
                        frag[15] = frag[15] * inv_scale_34_4;
                        frag[29] = frag[29] * inv_scale_34_4;
                        frag[31] = frag[31] * inv_scale_34_4;
                        if (is_sf_writer != 0) {
                            scode[warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32_4;
                        }
                        uint32_t _fp8_4[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_4[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_4[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_4[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_4[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_4[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_4[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_4[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_4[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_8 = static_cast<uint32_t>(sact_buf_10 + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_8), "r"(_fp8_4[0]), "r"(_fp8_4[1]), "r"(_fp8_4[2]), "r"(_fp8_4[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_9 = static_cast<uint32_t>(sact_buf_10 + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_9), "r"(_fp8_4[4]), "r"(_fp8_4[5]), "r"(_fp8_4[6]), "r"(_fp8_4[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s_11 = row_base + 128 + st_tok;
                    if (prow_s_11 < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf_10 + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s_11 * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l_4 = row_base + 128 + lane_0;
                        if (prow_l_4 < mn_limit) {
                            unsigned int code_l_4 = scode[warp_in64 * 32 + lane_0];
                            int sf_off_l_4 = prow_l_4 % 32 * 16 + prow_l_4 / 32 % 4 * 4 + prow_l_4 / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l_4) + (0)) = (unsigned char)(code_l_4);
                        }
                    }
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
                    frag[0] = _tmem_load_10[0] * meta_alpha;
                    frag[16] = _tmem_load_11[0] * meta_alpha;
                    frag[1] = _tmem_load_10[1] * meta_alpha;
                    frag[17] = _tmem_load_11[1] * meta_alpha;
                    frag[2] = _tmem_load_10[2] * meta_alpha;
                    frag[18] = _tmem_load_11[2] * meta_alpha;
                    frag[3] = _tmem_load_10[3] * meta_alpha;
                    frag[19] = _tmem_load_11[3] * meta_alpha;
                    frag[4] = _tmem_load_10[4] * meta_alpha;
                    frag[20] = _tmem_load_11[4] * meta_alpha;
                    frag[5] = _tmem_load_10[5] * meta_alpha;
                    frag[21] = _tmem_load_11[5] * meta_alpha;
                    frag[6] = _tmem_load_10[6] * meta_alpha;
                    frag[22] = _tmem_load_11[6] * meta_alpha;
                    frag[7] = _tmem_load_10[7] * meta_alpha;
                    frag[23] = _tmem_load_11[7] * meta_alpha;
                    frag[8] = _tmem_load_10[8] * meta_alpha;
                    frag[24] = _tmem_load_11[8] * meta_alpha;
                    frag[9] = _tmem_load_10[9] * meta_alpha;
                    frag[25] = _tmem_load_11[9] * meta_alpha;
                    frag[10] = _tmem_load_10[10] * meta_alpha;
                    frag[26] = _tmem_load_11[10] * meta_alpha;
                    frag[11] = _tmem_load_10[11] * meta_alpha;
                    frag[27] = _tmem_load_11[11] * meta_alpha;
                    frag[12] = _tmem_load_10[12] * meta_alpha;
                    frag[28] = _tmem_load_11[12] * meta_alpha;
                    frag[13] = _tmem_load_10[13] * meta_alpha;
                    frag[29] = _tmem_load_11[13] * meta_alpha;
                    frag[14] = _tmem_load_10[14] * meta_alpha;
                    frag[30] = _tmem_load_11[14] * meta_alpha;
                    frag[15] = _tmem_load_10[15] * meta_alpha;
                    frag[31] = _tmem_load_11[15] * meta_alpha;
                    int exchf_buf_12 = sexchf_addr + 8192;
                    int sact_buf_13 = sact_addr + 4608;
                    if (is_gate_lane != 0) {
                        float x_g_9 = frag[0];
                        float _exp2_160 = approx_exp2(x_g_9 * -1.4426950408889634f);
                        float _rcp_160 = approx_rcp(1.0f + _exp2_160);
                        float sig_g_10 = _rcp_160;
                        float _tanh_approx_320;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_320) : "f"(x_g_9 * inv_beta));
                        frag[0] = beta * _tanh_approx_320 * sig_g_10;
                        float x_g_0_5 = frag[1];
                        float _exp2_161 = approx_exp2(x_g_0_5 * -1.4426950408889634f);
                        float _rcp_161 = approx_rcp(1.0f + _exp2_161);
                        float sig_g_1_5 = _rcp_161;
                        float _tanh_approx_321;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_321) : "f"(x_g_0_5 * inv_beta));
                        frag[1] = beta * _tanh_approx_321 * sig_g_1_5;
                        float x_g_2_5 = frag[2];
                        float _exp2_162 = approx_exp2(x_g_2_5 * -1.4426950408889634f);
                        float _rcp_162 = approx_rcp(1.0f + _exp2_162);
                        float sig_g_3_5 = _rcp_162;
                        float _tanh_approx_322;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_322) : "f"(x_g_2_5 * inv_beta));
                        frag[2] = beta * _tanh_approx_322 * sig_g_3_5;
                        float x_g_4_5 = frag[3];
                        float _exp2_163 = approx_exp2(x_g_4_5 * -1.4426950408889634f);
                        float _rcp_163 = approx_rcp(1.0f + _exp2_163);
                        float sig_g_5_5 = _rcp_163;
                        float _tanh_approx_323;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_323) : "f"(x_g_4_5 * inv_beta));
                        frag[3] = beta * _tanh_approx_323 * sig_g_5_5;
                        float x_g_6_5 = frag[4];
                        float _exp2_164 = approx_exp2(x_g_6_5 * -1.4426950408889634f);
                        float _rcp_164 = approx_rcp(1.0f + _exp2_164);
                        float sig_g_7_5 = _rcp_164;
                        float _tanh_approx_324;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_324) : "f"(x_g_6_5 * inv_beta));
                        frag[4] = beta * _tanh_approx_324 * sig_g_7_5;
                        float x_g_8_5 = frag[5];
                        float _exp2_165 = approx_exp2(x_g_8_5 * -1.4426950408889634f);
                        float _rcp_165 = approx_rcp(1.0f + _exp2_165);
                        float sig_g_9_5 = _rcp_165;
                        float _tanh_approx_325;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_325) : "f"(x_g_8_5 * inv_beta));
                        frag[5] = beta * _tanh_approx_325 * sig_g_9_5;
                        float x_g_10_5 = frag[6];
                        float _exp2_166 = approx_exp2(x_g_10_5 * -1.4426950408889634f);
                        float _rcp_166 = approx_rcp(1.0f + _exp2_166);
                        float sig_g_11_5 = _rcp_166;
                        float _tanh_approx_326;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_326) : "f"(x_g_10_5 * inv_beta));
                        frag[6] = beta * _tanh_approx_326 * sig_g_11_5;
                        float x_g_12_5 = frag[7];
                        float _exp2_167 = approx_exp2(x_g_12_5 * -1.4426950408889634f);
                        float _rcp_167 = approx_rcp(1.0f + _exp2_167);
                        float sig_g_13_5 = _rcp_167;
                        float _tanh_approx_327;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_327) : "f"(x_g_12_5 * inv_beta));
                        frag[7] = beta * _tanh_approx_327 * sig_g_13_5;
                        float x_g_14_5 = frag[8];
                        float _exp2_168 = approx_exp2(x_g_14_5 * -1.4426950408889634f);
                        float _rcp_168 = approx_rcp(1.0f + _exp2_168);
                        float sig_g_15_5 = _rcp_168;
                        float _tanh_approx_328;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_328) : "f"(x_g_14_5 * inv_beta));
                        frag[8] = beta * _tanh_approx_328 * sig_g_15_5;
                        float x_g_16_5 = frag[9];
                        float _exp2_169 = approx_exp2(x_g_16_5 * -1.4426950408889634f);
                        float _rcp_169 = approx_rcp(1.0f + _exp2_169);
                        float sig_g_17_5 = _rcp_169;
                        float _tanh_approx_329;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_329) : "f"(x_g_16_5 * inv_beta));
                        frag[9] = beta * _tanh_approx_329 * sig_g_17_5;
                        float x_g_18_5 = frag[10];
                        float _exp2_170 = approx_exp2(x_g_18_5 * -1.4426950408889634f);
                        float _rcp_170 = approx_rcp(1.0f + _exp2_170);
                        float sig_g_19_5 = _rcp_170;
                        float _tanh_approx_330;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_330) : "f"(x_g_18_5 * inv_beta));
                        frag[10] = beta * _tanh_approx_330 * sig_g_19_5;
                        float x_g_20_5 = frag[11];
                        float _exp2_171 = approx_exp2(x_g_20_5 * -1.4426950408889634f);
                        float _rcp_171 = approx_rcp(1.0f + _exp2_171);
                        float sig_g_21_5 = _rcp_171;
                        float _tanh_approx_331;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_331) : "f"(x_g_20_5 * inv_beta));
                        frag[11] = beta * _tanh_approx_331 * sig_g_21_5;
                        float x_g_22_5 = frag[12];
                        float _exp2_172 = approx_exp2(x_g_22_5 * -1.4426950408889634f);
                        float _rcp_172 = approx_rcp(1.0f + _exp2_172);
                        float sig_g_23_5 = _rcp_172;
                        float _tanh_approx_332;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_332) : "f"(x_g_22_5 * inv_beta));
                        frag[12] = beta * _tanh_approx_332 * sig_g_23_5;
                        float x_g_24_5 = frag[13];
                        float _exp2_173 = approx_exp2(x_g_24_5 * -1.4426950408889634f);
                        float _rcp_173 = approx_rcp(1.0f + _exp2_173);
                        float sig_g_25_5 = _rcp_173;
                        float _tanh_approx_333;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_333) : "f"(x_g_24_5 * inv_beta));
                        frag[13] = beta * _tanh_approx_333 * sig_g_25_5;
                        float x_g_26_5 = frag[14];
                        float _exp2_174 = approx_exp2(x_g_26_5 * -1.4426950408889634f);
                        float _rcp_174 = approx_rcp(1.0f + _exp2_174);
                        float sig_g_27_5 = _rcp_174;
                        float _tanh_approx_334;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_334) : "f"(x_g_26_5 * inv_beta));
                        frag[14] = beta * _tanh_approx_334 * sig_g_27_5;
                        float x_g_28_5 = frag[15];
                        float _exp2_175 = approx_exp2(x_g_28_5 * -1.4426950408889634f);
                        float _rcp_175 = approx_rcp(1.0f + _exp2_175);
                        float sig_g_29_5 = _rcp_175;
                        float _tanh_approx_335;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_335) : "f"(x_g_28_5 * inv_beta));
                        frag[15] = beta * _tanh_approx_335 * sig_g_29_5;
                        float x_g_30_5 = frag[16];
                        float _exp2_176 = approx_exp2(x_g_30_5 * -1.4426950408889634f);
                        float _rcp_176 = approx_rcp(1.0f + _exp2_176);
                        float sig_g_31_5 = _rcp_176;
                        float _tanh_approx_336;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_336) : "f"(x_g_30_5 * inv_beta));
                        frag[16] = beta * _tanh_approx_336 * sig_g_31_5;
                        float x_g_32_5 = frag[17];
                        float _exp2_177 = approx_exp2(x_g_32_5 * -1.4426950408889634f);
                        float _rcp_177 = approx_rcp(1.0f + _exp2_177);
                        float sig_g_33_5 = _rcp_177;
                        float _tanh_approx_337;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_337) : "f"(x_g_32_5 * inv_beta));
                        frag[17] = beta * _tanh_approx_337 * sig_g_33_5;
                        float x_g_34_5 = frag[18];
                        float _exp2_178 = approx_exp2(x_g_34_5 * -1.4426950408889634f);
                        float _rcp_178 = approx_rcp(1.0f + _exp2_178);
                        float sig_g_35_5 = _rcp_178;
                        float _tanh_approx_338;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_338) : "f"(x_g_34_5 * inv_beta));
                        frag[18] = beta * _tanh_approx_338 * sig_g_35_5;
                        float x_g_36_5 = frag[19];
                        float _exp2_179 = approx_exp2(x_g_36_5 * -1.4426950408889634f);
                        float _rcp_179 = approx_rcp(1.0f + _exp2_179);
                        float sig_g_37_5 = _rcp_179;
                        float _tanh_approx_339;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_339) : "f"(x_g_36_5 * inv_beta));
                        frag[19] = beta * _tanh_approx_339 * sig_g_37_5;
                        float x_g_38_5 = frag[20];
                        float _exp2_180 = approx_exp2(x_g_38_5 * -1.4426950408889634f);
                        float _rcp_180 = approx_rcp(1.0f + _exp2_180);
                        float sig_g_39_5 = _rcp_180;
                        float _tanh_approx_340;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_340) : "f"(x_g_38_5 * inv_beta));
                        frag[20] = beta * _tanh_approx_340 * sig_g_39_5;
                        float x_g_40_5 = frag[21];
                        float _exp2_181 = approx_exp2(x_g_40_5 * -1.4426950408889634f);
                        float _rcp_181 = approx_rcp(1.0f + _exp2_181);
                        float sig_g_41_5 = _rcp_181;
                        float _tanh_approx_341;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_341) : "f"(x_g_40_5 * inv_beta));
                        frag[21] = beta * _tanh_approx_341 * sig_g_41_5;
                        float x_g_42_5 = frag[22];
                        float _exp2_182 = approx_exp2(x_g_42_5 * -1.4426950408889634f);
                        float _rcp_182 = approx_rcp(1.0f + _exp2_182);
                        float sig_g_43_5 = _rcp_182;
                        float _tanh_approx_342;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_342) : "f"(x_g_42_5 * inv_beta));
                        frag[22] = beta * _tanh_approx_342 * sig_g_43_5;
                        float x_g_44_5 = frag[23];
                        float _exp2_183 = approx_exp2(x_g_44_5 * -1.4426950408889634f);
                        float _rcp_183 = approx_rcp(1.0f + _exp2_183);
                        float sig_g_45_5 = _rcp_183;
                        float _tanh_approx_343;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_343) : "f"(x_g_44_5 * inv_beta));
                        frag[23] = beta * _tanh_approx_343 * sig_g_45_5;
                        float x_g_46_5 = frag[24];
                        float _exp2_184 = approx_exp2(x_g_46_5 * -1.4426950408889634f);
                        float _rcp_184 = approx_rcp(1.0f + _exp2_184);
                        float sig_g_47_5 = _rcp_184;
                        float _tanh_approx_344;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_344) : "f"(x_g_46_5 * inv_beta));
                        frag[24] = beta * _tanh_approx_344 * sig_g_47_5;
                        float x_g_48_5 = frag[25];
                        float _exp2_185 = approx_exp2(x_g_48_5 * -1.4426950408889634f);
                        float _rcp_185 = approx_rcp(1.0f + _exp2_185);
                        float sig_g_49_5 = _rcp_185;
                        float _tanh_approx_345;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_345) : "f"(x_g_48_5 * inv_beta));
                        frag[25] = beta * _tanh_approx_345 * sig_g_49_5;
                        float x_g_50_5 = frag[26];
                        float _exp2_186 = approx_exp2(x_g_50_5 * -1.4426950408889634f);
                        float _rcp_186 = approx_rcp(1.0f + _exp2_186);
                        float sig_g_51_5 = _rcp_186;
                        float _tanh_approx_346;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_346) : "f"(x_g_50_5 * inv_beta));
                        frag[26] = beta * _tanh_approx_346 * sig_g_51_5;
                        float x_g_52_5 = frag[27];
                        float _exp2_187 = approx_exp2(x_g_52_5 * -1.4426950408889634f);
                        float _rcp_187 = approx_rcp(1.0f + _exp2_187);
                        float sig_g_53_5 = _rcp_187;
                        float _tanh_approx_347;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_347) : "f"(x_g_52_5 * inv_beta));
                        frag[27] = beta * _tanh_approx_347 * sig_g_53_5;
                        float x_g_54_5 = frag[28];
                        float _exp2_188 = approx_exp2(x_g_54_5 * -1.4426950408889634f);
                        float _rcp_188 = approx_rcp(1.0f + _exp2_188);
                        float sig_g_55_5 = _rcp_188;
                        float _tanh_approx_348;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_348) : "f"(x_g_54_5 * inv_beta));
                        frag[28] = beta * _tanh_approx_348 * sig_g_55_5;
                        float x_g_56_5 = frag[29];
                        float _exp2_189 = approx_exp2(x_g_56_5 * -1.4426950408889634f);
                        float _rcp_189 = approx_rcp(1.0f + _exp2_189);
                        float sig_g_57_5 = _rcp_189;
                        float _tanh_approx_349;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_349) : "f"(x_g_56_5 * inv_beta));
                        frag[29] = beta * _tanh_approx_349 * sig_g_57_5;
                        float x_g_58_5 = frag[30];
                        float _exp2_190 = approx_exp2(x_g_58_5 * -1.4426950408889634f);
                        float _rcp_190 = approx_rcp(1.0f + _exp2_190);
                        float sig_g_59_5 = _rcp_190;
                        float _tanh_approx_350;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_350) : "f"(x_g_58_5 * inv_beta));
                        frag[30] = beta * _tanh_approx_350 * sig_g_59_5;
                        float x_g_60_5 = frag[31];
                        float _exp2_191 = approx_exp2(x_g_60_5 * -1.4426950408889634f);
                        float _rcp_191 = approx_rcp(1.0f + _exp2_191);
                        float sig_g_61_5 = _rcp_191;
                        float _tanh_approx_351;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_351) : "f"(x_g_60_5 * inv_beta));
                        frag[31] = beta * _tanh_approx_351 * sig_g_61_5;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + 4 * t64 * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[0])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(0) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (256 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[4])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(4) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (512 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[8])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(8) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (768 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[12])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(12) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (1024 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[16])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(16) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (1280 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[20])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(20) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (1536 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[24])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(24) + 3])));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(exchf_buf_12 + (1792 + 4 * t64) * 4), "r"(*reinterpret_cast<uint32_t*>(&frag[28])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 1])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 2])), "r"(*reinterpret_cast<uint32_t*>(&frag[(28) + 3])));
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    } else {
                        float _tanh_approx_352;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_352) : "f"(frag[0] * inv_linear_beta));
                        frag[0] = linear_beta * _tanh_approx_352;
                        float _tanh_approx_353;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_353) : "f"(frag[1] * inv_linear_beta));
                        frag[1] = linear_beta * _tanh_approx_353;
                        float _tanh_approx_354;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_354) : "f"(frag[2] * inv_linear_beta));
                        frag[2] = linear_beta * _tanh_approx_354;
                        float _tanh_approx_355;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_355) : "f"(frag[3] * inv_linear_beta));
                        frag[3] = linear_beta * _tanh_approx_355;
                        float _tanh_approx_356;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_356) : "f"(frag[4] * inv_linear_beta));
                        frag[4] = linear_beta * _tanh_approx_356;
                        float _tanh_approx_357;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_357) : "f"(frag[5] * inv_linear_beta));
                        frag[5] = linear_beta * _tanh_approx_357;
                        float _tanh_approx_358;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_358) : "f"(frag[6] * inv_linear_beta));
                        frag[6] = linear_beta * _tanh_approx_358;
                        float _tanh_approx_359;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_359) : "f"(frag[7] * inv_linear_beta));
                        frag[7] = linear_beta * _tanh_approx_359;
                        float _tanh_approx_360;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_360) : "f"(frag[8] * inv_linear_beta));
                        frag[8] = linear_beta * _tanh_approx_360;
                        float _tanh_approx_361;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_361) : "f"(frag[9] * inv_linear_beta));
                        frag[9] = linear_beta * _tanh_approx_361;
                        float _tanh_approx_362;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_362) : "f"(frag[10] * inv_linear_beta));
                        frag[10] = linear_beta * _tanh_approx_362;
                        float _tanh_approx_363;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_363) : "f"(frag[11] * inv_linear_beta));
                        frag[11] = linear_beta * _tanh_approx_363;
                        float _tanh_approx_364;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_364) : "f"(frag[12] * inv_linear_beta));
                        frag[12] = linear_beta * _tanh_approx_364;
                        float _tanh_approx_365;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_365) : "f"(frag[13] * inv_linear_beta));
                        frag[13] = linear_beta * _tanh_approx_365;
                        float _tanh_approx_366;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_366) : "f"(frag[14] * inv_linear_beta));
                        frag[14] = linear_beta * _tanh_approx_366;
                        float _tanh_approx_367;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_367) : "f"(frag[15] * inv_linear_beta));
                        frag[15] = linear_beta * _tanh_approx_367;
                        float _tanh_approx_368;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_368) : "f"(frag[16] * inv_linear_beta));
                        frag[16] = linear_beta * _tanh_approx_368;
                        float _tanh_approx_369;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_369) : "f"(frag[17] * inv_linear_beta));
                        frag[17] = linear_beta * _tanh_approx_369;
                        float _tanh_approx_370;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_370) : "f"(frag[18] * inv_linear_beta));
                        frag[18] = linear_beta * _tanh_approx_370;
                        float _tanh_approx_371;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_371) : "f"(frag[19] * inv_linear_beta));
                        frag[19] = linear_beta * _tanh_approx_371;
                        float _tanh_approx_372;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_372) : "f"(frag[20] * inv_linear_beta));
                        frag[20] = linear_beta * _tanh_approx_372;
                        float _tanh_approx_373;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_373) : "f"(frag[21] * inv_linear_beta));
                        frag[21] = linear_beta * _tanh_approx_373;
                        float _tanh_approx_374;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_374) : "f"(frag[22] * inv_linear_beta));
                        frag[22] = linear_beta * _tanh_approx_374;
                        float _tanh_approx_375;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_375) : "f"(frag[23] * inv_linear_beta));
                        frag[23] = linear_beta * _tanh_approx_375;
                        float _tanh_approx_376;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_376) : "f"(frag[24] * inv_linear_beta));
                        frag[24] = linear_beta * _tanh_approx_376;
                        float _tanh_approx_377;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_377) : "f"(frag[25] * inv_linear_beta));
                        frag[25] = linear_beta * _tanh_approx_377;
                        float _tanh_approx_378;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_378) : "f"(frag[26] * inv_linear_beta));
                        frag[26] = linear_beta * _tanh_approx_378;
                        float _tanh_approx_379;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_379) : "f"(frag[27] * inv_linear_beta));
                        frag[27] = linear_beta * _tanh_approx_379;
                        float _tanh_approx_380;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_380) : "f"(frag[28] * inv_linear_beta));
                        frag[28] = linear_beta * _tanh_approx_380;
                        float _tanh_approx_381;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_381) : "f"(frag[29] * inv_linear_beta));
                        frag[29] = linear_beta * _tanh_approx_381;
                        float _tanh_approx_382;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_382) : "f"(frag[30] * inv_linear_beta));
                        frag[30] = linear_beta * _tanh_approx_382;
                        float _tanh_approx_383;
                        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_383) : "f"(frag[31] * inv_linear_beta));
                        frag[31] = linear_beta * _tanh_approx_383;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + 4 * t64 * 4));
                        frag[0] = frag[0] * __uint_as_float(gx[0]);
                        frag[1] = frag[1] * __uint_as_float(gx[1]);
                        frag[2] = frag[2] * __uint_as_float(gx[2]);
                        frag[3] = frag[3] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (256 + 4 * t64) * 4));
                        frag[4] = frag[4] * __uint_as_float(gx[0]);
                        frag[5] = frag[5] * __uint_as_float(gx[1]);
                        frag[6] = frag[6] * __uint_as_float(gx[2]);
                        frag[7] = frag[7] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (512 + 4 * t64) * 4));
                        frag[8] = frag[8] * __uint_as_float(gx[0]);
                        frag[9] = frag[9] * __uint_as_float(gx[1]);
                        frag[10] = frag[10] * __uint_as_float(gx[2]);
                        frag[11] = frag[11] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (768 + 4 * t64) * 4));
                        frag[12] = frag[12] * __uint_as_float(gx[0]);
                        frag[13] = frag[13] * __uint_as_float(gx[1]);
                        frag[14] = frag[14] * __uint_as_float(gx[2]);
                        frag[15] = frag[15] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (1024 + 4 * t64) * 4));
                        frag[16] = frag[16] * __uint_as_float(gx[0]);
                        frag[17] = frag[17] * __uint_as_float(gx[1]);
                        frag[18] = frag[18] * __uint_as_float(gx[2]);
                        frag[19] = frag[19] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (1280 + 4 * t64) * 4));
                        frag[20] = frag[20] * __uint_as_float(gx[0]);
                        frag[21] = frag[21] * __uint_as_float(gx[1]);
                        frag[22] = frag[22] * __uint_as_float(gx[2]);
                        frag[23] = frag[23] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (1536 + 4 * t64) * 4));
                        frag[24] = frag[24] * __uint_as_float(gx[0]);
                        frag[25] = frag[25] * __uint_as_float(gx[1]);
                        frag[26] = frag[26] * __uint_as_float(gx[2]);
                        frag[27] = frag[27] * __uint_as_float(gx[3]);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gx[0])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gx[(0) + 3]))
                            : "r"(exchf_buf_12 + (1792 + 4 * t64) * 4));
                        frag[28] = frag[28] * __uint_as_float(gx[0]);
                        frag[29] = frag[29] * __uint_as_float(gx[1]);
                        frag[30] = frag[30] * __uint_as_float(gx[2]);
                        frag[31] = frag[31] * __uint_as_float(gx[3]);
                        float _fmax_400 = fmaxf(frag[0], -frag[0]);
                        float a_c_6 = _fmax_400;
                        float _fmax_401 = fmaxf(frag[2], -frag[2]);
                        float _fmax_402 = fmaxf(a_c_6, _fmax_401);
                        a_c_6 = _fmax_402;
                        float _fmax_403 = fmaxf(frag[16], -frag[16]);
                        float _fmax_404 = fmaxf(a_c_6, _fmax_403);
                        a_c_6 = _fmax_404;
                        float _fmax_405 = fmaxf(frag[18], -frag[18]);
                        float _fmax_406 = fmaxf(a_c_6, _fmax_405);
                        a_c_6 = _fmax_406;
                        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, a_c_6, 4);
                        float _fmax_407 = fmaxf(a_c_6, _shfl_xor_120);
                        a_c_6 = _fmax_407;
                        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, a_c_6, 8);
                        float _fmax_408 = fmaxf(a_c_6, _shfl_xor_121);
                        a_c_6 = _fmax_408;
                        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, a_c_6, 16);
                        float _fmax_409 = fmaxf(a_c_6, _shfl_xor_122);
                        a_c_6 = _fmax_409;
                        uint16_t _ue8m0x2_f32_40;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_40) : "f"(zero_f32), "f"(a_c_6 * inv_fp8_max));
                        int code_full_7 = (int)_ue8m0x2_f32_40;
                        int code_6 = code_full_7 & 255;
                        int _max_40 = ((254 - code_6) > (0) ? (254 - code_6) : (0));
                        unsigned int inv_bits_6 = (unsigned int)(_max_40 << 23);
                        float inv_scale_6 = __uint_as_float(inv_bits_6) * (float)(code_6 != 0);
                        frag[0] = frag[0] * inv_scale_6;
                        frag[2] = frag[2] * inv_scale_6;
                        frag[16] = frag[16] * inv_scale_6;
                        frag[18] = frag[18] * inv_scale_6;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 2 * tok_pair] = (unsigned int)code_6;
                        }
                        float _fmax_410 = fmaxf(frag[1], -frag[1]);
                        float a_c_0_5 = _fmax_410;
                        float _fmax_411 = fmaxf(frag[3], -frag[3]);
                        float _fmax_412 = fmaxf(a_c_0_5, _fmax_411);
                        a_c_0_5 = _fmax_412;
                        float _fmax_413 = fmaxf(frag[17], -frag[17]);
                        float _fmax_414 = fmaxf(a_c_0_5, _fmax_413);
                        a_c_0_5 = _fmax_414;
                        float _fmax_415 = fmaxf(frag[19], -frag[19]);
                        float _fmax_416 = fmaxf(a_c_0_5, _fmax_415);
                        a_c_0_5 = _fmax_416;
                        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_5, 4);
                        float _fmax_417 = fmaxf(a_c_0_5, _shfl_xor_123);
                        a_c_0_5 = _fmax_417;
                        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_5, 8);
                        float _fmax_418 = fmaxf(a_c_0_5, _shfl_xor_124);
                        a_c_0_5 = _fmax_418;
                        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, a_c_0_5, 16);
                        float _fmax_419 = fmaxf(a_c_0_5, _shfl_xor_125);
                        a_c_0_5 = _fmax_419;
                        uint16_t _ue8m0x2_f32_41;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_41) : "f"(zero_f32), "f"(a_c_0_5 * inv_fp8_max));
                        int code_full_1_5 = (int)_ue8m0x2_f32_41;
                        int code_2_5 = code_full_1_5 & 255;
                        int _max_41 = ((254 - code_2_5) > (0) ? (254 - code_2_5) : (0));
                        unsigned int inv_bits_3_5 = (unsigned int)(_max_41 << 23);
                        float inv_scale_4_5 = __uint_as_float(inv_bits_3_5) * (float)(code_2_5 != 0);
                        frag[1] = frag[1] * inv_scale_4_5;
                        frag[3] = frag[3] * inv_scale_4_5;
                        frag[17] = frag[17] * inv_scale_4_5;
                        frag[19] = frag[19] * inv_scale_4_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 1 + 2 * tok_pair] = (unsigned int)code_2_5;
                        }
                        float _fmax_420 = fmaxf(frag[4], -frag[4]);
                        float a_c_5_5 = _fmax_420;
                        float _fmax_421 = fmaxf(frag[6], -frag[6]);
                        float _fmax_422 = fmaxf(a_c_5_5, _fmax_421);
                        a_c_5_5 = _fmax_422;
                        float _fmax_423 = fmaxf(frag[20], -frag[20]);
                        float _fmax_424 = fmaxf(a_c_5_5, _fmax_423);
                        a_c_5_5 = _fmax_424;
                        float _fmax_425 = fmaxf(frag[22], -frag[22]);
                        float _fmax_426 = fmaxf(a_c_5_5, _fmax_425);
                        a_c_5_5 = _fmax_426;
                        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_5, 4);
                        float _fmax_427 = fmaxf(a_c_5_5, _shfl_xor_126);
                        a_c_5_5 = _fmax_427;
                        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_5, 8);
                        float _fmax_428 = fmaxf(a_c_5_5, _shfl_xor_127);
                        a_c_5_5 = _fmax_428;
                        float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, a_c_5_5, 16);
                        float _fmax_429 = fmaxf(a_c_5_5, _shfl_xor_128);
                        a_c_5_5 = _fmax_429;
                        uint16_t _ue8m0x2_f32_42;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_42) : "f"(zero_f32), "f"(a_c_5_5 * inv_fp8_max));
                        int code_full_6_5 = (int)_ue8m0x2_f32_42;
                        int code_7_5 = code_full_6_5 & 255;
                        int _max_42 = ((254 - code_7_5) > (0) ? (254 - code_7_5) : (0));
                        unsigned int inv_bits_8_5 = (unsigned int)(_max_42 << 23);
                        float inv_scale_9_5 = __uint_as_float(inv_bits_8_5) * (float)(code_7_5 != 0);
                        frag[4] = frag[4] * inv_scale_9_5;
                        frag[6] = frag[6] * inv_scale_9_5;
                        frag[20] = frag[20] * inv_scale_9_5;
                        frag[22] = frag[22] * inv_scale_9_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 8 + 2 * tok_pair] = (unsigned int)code_7_5;
                        }
                        float _fmax_430 = fmaxf(frag[5], -frag[5]);
                        float a_c_10_5 = _fmax_430;
                        float _fmax_431 = fmaxf(frag[7], -frag[7]);
                        float _fmax_432 = fmaxf(a_c_10_5, _fmax_431);
                        a_c_10_5 = _fmax_432;
                        float _fmax_433 = fmaxf(frag[21], -frag[21]);
                        float _fmax_434 = fmaxf(a_c_10_5, _fmax_433);
                        a_c_10_5 = _fmax_434;
                        float _fmax_435 = fmaxf(frag[23], -frag[23]);
                        float _fmax_436 = fmaxf(a_c_10_5, _fmax_435);
                        a_c_10_5 = _fmax_436;
                        float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_5, 4);
                        float _fmax_437 = fmaxf(a_c_10_5, _shfl_xor_129);
                        a_c_10_5 = _fmax_437;
                        float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_5, 8);
                        float _fmax_438 = fmaxf(a_c_10_5, _shfl_xor_130);
                        a_c_10_5 = _fmax_438;
                        float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, a_c_10_5, 16);
                        float _fmax_439 = fmaxf(a_c_10_5, _shfl_xor_131);
                        a_c_10_5 = _fmax_439;
                        uint16_t _ue8m0x2_f32_43;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_43) : "f"(zero_f32), "f"(a_c_10_5 * inv_fp8_max));
                        int code_full_11_5 = (int)_ue8m0x2_f32_43;
                        int code_12_5 = code_full_11_5 & 255;
                        int _max_43 = ((254 - code_12_5) > (0) ? (254 - code_12_5) : (0));
                        unsigned int inv_bits_13_5 = (unsigned int)(_max_43 << 23);
                        float inv_scale_14_5 = __uint_as_float(inv_bits_13_5) * (float)(code_12_5 != 0);
                        frag[5] = frag[5] * inv_scale_14_5;
                        frag[7] = frag[7] * inv_scale_14_5;
                        frag[21] = frag[21] * inv_scale_14_5;
                        frag[23] = frag[23] * inv_scale_14_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 9 + 2 * tok_pair] = (unsigned int)code_12_5;
                        }
                        float _fmax_440 = fmaxf(frag[8], -frag[8]);
                        float a_c_15_5 = _fmax_440;
                        float _fmax_441 = fmaxf(frag[10], -frag[10]);
                        float _fmax_442 = fmaxf(a_c_15_5, _fmax_441);
                        a_c_15_5 = _fmax_442;
                        float _fmax_443 = fmaxf(frag[24], -frag[24]);
                        float _fmax_444 = fmaxf(a_c_15_5, _fmax_443);
                        a_c_15_5 = _fmax_444;
                        float _fmax_445 = fmaxf(frag[26], -frag[26]);
                        float _fmax_446 = fmaxf(a_c_15_5, _fmax_445);
                        a_c_15_5 = _fmax_446;
                        float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_5, 4);
                        float _fmax_447 = fmaxf(a_c_15_5, _shfl_xor_132);
                        a_c_15_5 = _fmax_447;
                        float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_5, 8);
                        float _fmax_448 = fmaxf(a_c_15_5, _shfl_xor_133);
                        a_c_15_5 = _fmax_448;
                        float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, a_c_15_5, 16);
                        float _fmax_449 = fmaxf(a_c_15_5, _shfl_xor_134);
                        a_c_15_5 = _fmax_449;
                        uint16_t _ue8m0x2_f32_44;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_44) : "f"(zero_f32), "f"(a_c_15_5 * inv_fp8_max));
                        int code_full_16_5 = (int)_ue8m0x2_f32_44;
                        int code_17_5 = code_full_16_5 & 255;
                        int _max_44 = ((254 - code_17_5) > (0) ? (254 - code_17_5) : (0));
                        unsigned int inv_bits_18_5 = (unsigned int)(_max_44 << 23);
                        float inv_scale_19_5 = __uint_as_float(inv_bits_18_5) * (float)(code_17_5 != 0);
                        frag[8] = frag[8] * inv_scale_19_5;
                        frag[10] = frag[10] * inv_scale_19_5;
                        frag[24] = frag[24] * inv_scale_19_5;
                        frag[26] = frag[26] * inv_scale_19_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 16 + 2 * tok_pair] = (unsigned int)code_17_5;
                        }
                        float _fmax_450 = fmaxf(frag[9], -frag[9]);
                        float a_c_20_5 = _fmax_450;
                        float _fmax_451 = fmaxf(frag[11], -frag[11]);
                        float _fmax_452 = fmaxf(a_c_20_5, _fmax_451);
                        a_c_20_5 = _fmax_452;
                        float _fmax_453 = fmaxf(frag[25], -frag[25]);
                        float _fmax_454 = fmaxf(a_c_20_5, _fmax_453);
                        a_c_20_5 = _fmax_454;
                        float _fmax_455 = fmaxf(frag[27], -frag[27]);
                        float _fmax_456 = fmaxf(a_c_20_5, _fmax_455);
                        a_c_20_5 = _fmax_456;
                        float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_5, 4);
                        float _fmax_457 = fmaxf(a_c_20_5, _shfl_xor_135);
                        a_c_20_5 = _fmax_457;
                        float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_5, 8);
                        float _fmax_458 = fmaxf(a_c_20_5, _shfl_xor_136);
                        a_c_20_5 = _fmax_458;
                        float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, a_c_20_5, 16);
                        float _fmax_459 = fmaxf(a_c_20_5, _shfl_xor_137);
                        a_c_20_5 = _fmax_459;
                        uint16_t _ue8m0x2_f32_45;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_45) : "f"(zero_f32), "f"(a_c_20_5 * inv_fp8_max));
                        int code_full_21_5 = (int)_ue8m0x2_f32_45;
                        int code_22_5 = code_full_21_5 & 255;
                        int _max_45 = ((254 - code_22_5) > (0) ? (254 - code_22_5) : (0));
                        unsigned int inv_bits_23_5 = (unsigned int)(_max_45 << 23);
                        float inv_scale_24_5 = __uint_as_float(inv_bits_23_5) * (float)(code_22_5 != 0);
                        frag[9] = frag[9] * inv_scale_24_5;
                        frag[11] = frag[11] * inv_scale_24_5;
                        frag[25] = frag[25] * inv_scale_24_5;
                        frag[27] = frag[27] * inv_scale_24_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 17 + 2 * tok_pair] = (unsigned int)code_22_5;
                        }
                        float _fmax_460 = fmaxf(frag[12], -frag[12]);
                        float a_c_25_5 = _fmax_460;
                        float _fmax_461 = fmaxf(frag[14], -frag[14]);
                        float _fmax_462 = fmaxf(a_c_25_5, _fmax_461);
                        a_c_25_5 = _fmax_462;
                        float _fmax_463 = fmaxf(frag[28], -frag[28]);
                        float _fmax_464 = fmaxf(a_c_25_5, _fmax_463);
                        a_c_25_5 = _fmax_464;
                        float _fmax_465 = fmaxf(frag[30], -frag[30]);
                        float _fmax_466 = fmaxf(a_c_25_5, _fmax_465);
                        a_c_25_5 = _fmax_466;
                        float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_5, 4);
                        float _fmax_467 = fmaxf(a_c_25_5, _shfl_xor_138);
                        a_c_25_5 = _fmax_467;
                        float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_5, 8);
                        float _fmax_468 = fmaxf(a_c_25_5, _shfl_xor_139);
                        a_c_25_5 = _fmax_468;
                        float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, a_c_25_5, 16);
                        float _fmax_469 = fmaxf(a_c_25_5, _shfl_xor_140);
                        a_c_25_5 = _fmax_469;
                        uint16_t _ue8m0x2_f32_46;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_46) : "f"(zero_f32), "f"(a_c_25_5 * inv_fp8_max));
                        int code_full_26_5 = (int)_ue8m0x2_f32_46;
                        int code_27_5 = code_full_26_5 & 255;
                        int _max_46 = ((254 - code_27_5) > (0) ? (254 - code_27_5) : (0));
                        unsigned int inv_bits_28_5 = (unsigned int)(_max_46 << 23);
                        float inv_scale_29_5 = __uint_as_float(inv_bits_28_5) * (float)(code_27_5 != 0);
                        frag[12] = frag[12] * inv_scale_29_5;
                        frag[14] = frag[14] * inv_scale_29_5;
                        frag[28] = frag[28] * inv_scale_29_5;
                        frag[30] = frag[30] * inv_scale_29_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 24 + 2 * tok_pair] = (unsigned int)code_27_5;
                        }
                        float _fmax_470 = fmaxf(frag[13], -frag[13]);
                        float a_c_30_5 = _fmax_470;
                        float _fmax_471 = fmaxf(frag[15], -frag[15]);
                        float _fmax_472 = fmaxf(a_c_30_5, _fmax_471);
                        a_c_30_5 = _fmax_472;
                        float _fmax_473 = fmaxf(frag[29], -frag[29]);
                        float _fmax_474 = fmaxf(a_c_30_5, _fmax_473);
                        a_c_30_5 = _fmax_474;
                        float _fmax_475 = fmaxf(frag[31], -frag[31]);
                        float _fmax_476 = fmaxf(a_c_30_5, _fmax_475);
                        a_c_30_5 = _fmax_476;
                        float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_5, 4);
                        float _fmax_477 = fmaxf(a_c_30_5, _shfl_xor_141);
                        a_c_30_5 = _fmax_477;
                        float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_5, 8);
                        float _fmax_478 = fmaxf(a_c_30_5, _shfl_xor_142);
                        a_c_30_5 = _fmax_478;
                        float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, a_c_30_5, 16);
                        float _fmax_479 = fmaxf(a_c_30_5, _shfl_xor_143);
                        a_c_30_5 = _fmax_479;
                        uint16_t _ue8m0x2_f32_47;
                        asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_47) : "f"(zero_f32), "f"(a_c_30_5 * inv_fp8_max));
                        int code_full_31_5 = (int)_ue8m0x2_f32_47;
                        int code_32_5 = code_full_31_5 & 255;
                        int _max_47 = ((254 - code_32_5) > (0) ? (254 - code_32_5) : (0));
                        unsigned int inv_bits_33_5 = (unsigned int)(_max_47 << 23);
                        float inv_scale_34_5 = __uint_as_float(inv_bits_33_5) * (float)(code_32_5 != 0);
                        frag[13] = frag[13] * inv_scale_34_5;
                        frag[15] = frag[15] * inv_scale_34_5;
                        frag[29] = frag[29] * inv_scale_34_5;
                        frag[31] = frag[31] * inv_scale_34_5;
                        if (is_sf_writer != 0) {
                            scode[64 + warp_in64 * 32 + 25 + 2 * tok_pair] = (unsigned int)code_32_5;
                        }
                        uint32_t _fp8_5[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[0]), "f"(frag[1]),
                                                   "f"(frag[2]), "f"(frag[3]));
                            _fp8_5[0] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[4]), "f"(frag[5]),
                                                   "f"(frag[6]), "f"(frag[7]));
                            _fp8_5[1] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[8]), "f"(frag[9]),
                                                   "f"(frag[10]), "f"(frag[11]));
                            _fp8_5[2] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[12]), "f"(frag[13]),
                                                   "f"(frag[14]), "f"(frag[15]));
                            _fp8_5[3] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[16]), "f"(frag[17]),
                                                   "f"(frag[18]), "f"(frag[19]));
                            _fp8_5[4] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[20]), "f"(frag[21]),
                                                   "f"(frag[22]), "f"(frag[23]));
                            _fp8_5[5] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[24]), "f"(frag[25]),
                                                   "f"(frag[26]), "f"(frag[27]));
                            _fp8_5[6] = _packed;
                        }
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(frag[28]), "f"(frag[29]),
                                                   "f"(frag[30]), "f"(frag[31]));
                            _fp8_5[7] = _packed;
                        }
                        uint32_t _stmatrix_b8_addr_10 = static_cast<uint32_t>(sact_buf_13 + lane_0 * 144 + warp_in64 * 32);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_10), "r"(_fp8_5[0]), "r"(_fp8_5[1]), "r"(_fp8_5[2]), "r"(_fp8_5[3])
                            : "memory");
                        uint32_t _stmatrix_b8_addr_11 = static_cast<uint32_t>(sact_buf_13 + lane_0 * 144 + warp_in64 * 32 + 16);
                        asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_b8_addr_11), "r"(_fp8_5[4]), "r"(_fp8_5[5]), "r"(_fp8_5[6]), "r"(_fp8_5[7])
                            : "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                    int prow_s_14 = row_base + 160 + st_tok;
                    if (prow_s_14 < mn_limit) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(sact_buf_13 + st_tok * 144 + st_chunk * 16));
                        reinterpret_cast<int4*>(out + (prow_s_14 * act_cols + j0 + 16 * st_chunk))[0] = reinterpret_cast<int4*>(w4)[0];
                    }
                    if (epi_tidx < 64) {
                        int prow_l_5 = row_base + 160 + lane_0;
                        if (prow_l_5 < mn_limit) {
                            unsigned int code_l_5 = scode[64 + warp_in64 * 32 + lane_0];
                            int sf_off_l_5 = prow_l_5 % 32 * 16 + prow_l_5 / 32 % 4 * 4 + prow_l_5 / 128 * (act_sf_cols * 128) + sf_kb / 4 * 512 + sf_kb % 4;
                            *(reinterpret_cast<unsigned char*>(act_sf + sf_off_l_5) + (0)) = (unsigned char)(code_l_5);
                        }
                    }
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
                int _min_0 = ((prow) < (row_base_1 + 191) ? (prow) : (row_base_1 + 191));
                int safe_row = _min_0;
                int expanded = permuted_idx_to_expanded_idx[safe_row];
                int tok_row = expanded / top_k;
                int ok = (int)(prow < mn_limit_2 && expanded >= 0 && tok_row < num_rows_b);
                row_src[0] = tok_row * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 4) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int _min_1 = ((prow_1) < (row_base_1 + 191) ? (prow_1) : (row_base_1 + 191));
                int safe_row_2 = _min_1;
                int expanded_3 = permuted_idx_to_expanded_idx[safe_row_2];
                int tok_row_4 = expanded_3 / top_k;
                int ok_5 = (int)(prow_1 < mn_limit_2 && expanded_3 >= 0 && tok_row_4 < num_rows_b);
                row_src[1] = tok_row_4 * ok_5;
                row_ok[1] = ok_5;
                int row_6 = (gather_sub + 8) * 4 + row_in_pass;
                int prow_7 = row_base_1 + cta_row0 + row_6;
                int _min_2 = ((prow_7) < (row_base_1 + 191) ? (prow_7) : (row_base_1 + 191));
                int safe_row_8 = _min_2;
                int expanded_9 = permuted_idx_to_expanded_idx[safe_row_8];
                int tok_row_10 = expanded_9 / top_k;
                int ok_11 = (int)(prow_7 < mn_limit_2 && expanded_9 >= 0 && tok_row_10 < num_rows_b);
                row_src[2] = tok_row_10 * ok_11;
                row_ok[2] = ok_11;
                int row_12 = (gather_sub + 12) * 4 + row_in_pass;
                int prow_13 = row_base_1 + cta_row0 + row_12;
                int _min_3 = ((prow_13) < (row_base_1 + 191) ? (prow_13) : (row_base_1 + 191));
                int safe_row_14 = _min_3;
                int expanded_15 = permuted_idx_to_expanded_idx[safe_row_14];
                int tok_row_16 = expanded_15 / top_k;
                int ok_17 = (int)(prow_13 < mn_limit_2 && expanded_15 >= 0 && tok_row_16 < num_rows_b);
                row_src[3] = tok_row_16 * ok_17;
                row_ok[3] = ok_17;
                int row_18 = (gather_sub + 16) * 4 + row_in_pass;
                int prow_19 = row_base_1 + cta_row0 + row_18;
                int _min_4 = ((prow_19) < (row_base_1 + 191) ? (prow_19) : (row_base_1 + 191));
                int safe_row_20 = _min_4;
                int expanded_21 = permuted_idx_to_expanded_idx[safe_row_20];
                int tok_row_22 = expanded_21 / top_k;
                int ok_23 = (int)(prow_19 < mn_limit_2 && expanded_21 >= 0 && tok_row_22 < num_rows_b);
                row_src[4] = tok_row_22 * ok_23;
                row_ok[4] = ok_23;
                int row_24 = (gather_sub + 20) * 4 + row_in_pass;
                int prow_25 = row_base_1 + cta_row0 + row_24;
                int _min_5 = ((prow_25) < (row_base_1 + 191) ? (prow_25) : (row_base_1 + 191));
                int safe_row_26 = _min_5;
                int expanded_27 = permuted_idx_to_expanded_idx[safe_row_26];
                int tok_row_28 = expanded_27 / top_k;
                int ok_29 = (int)(prow_25 < mn_limit_2 && expanded_27 >= 0 && tok_row_28 < num_rows_b);
                row_src[5] = tok_row_28 * ok_29;
                row_ok[5] = ok_29;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int _min_6 = ((sprow) < (row_base_1 + 191) ? (sprow) : (row_base_1 + 191));
                int safe_srow = _min_6;
                int sexpanded = permuted_idx_to_expanded_idx[safe_srow];
                int stok_row = sexpanded / top_k;
                int sok = (int)(sprow < mn_limit_2 && srow < 192 && sexpanded >= 0 && stok_row < num_rows_b);
                sf_src[0] = stok_row * sok;
                sf_ok[0] = sok;
                int srow_30 = (gather_sub + 4) * 32 + lane_0_1;
                int sprow_31 = row_base_1 + srow_30;
                int _min_7 = ((sprow_31) < (row_base_1 + 191) ? (sprow_31) : (row_base_1 + 191));
                int safe_srow_32 = _min_7;
                int sexpanded_33 = permuted_idx_to_expanded_idx[safe_srow_32];
                int stok_row_34 = sexpanded_33 / top_k;
                int sok_35 = (int)(sprow_31 < mn_limit_2 && srow_30 < 192 && sexpanded_33 >= 0 && stok_row_34 < num_rows_b);
                sf_src[1] = stok_row_34 * sok_35;
                sf_ok[1] = sok_35;
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
                    int sf_src_off = sf_src[0] * sf_cols + (info_3[5] + k_2) * 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 1024 + (unsigned int)(gather_sub / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    int sf_src_off_10 = sf_src[1] * sf_cols + (info_3[5] + k_2) * 4;
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
