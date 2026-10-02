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
#define TMEM_NCOLS 268
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 256
#define TMEM_ACC1_OFFSET 128
#define TMEM_SF_A1_OFFSET 260
#define TMEM_SF_B_OFFSET 264
#define NUM_PAB_STAGES 4
#define NUM_PB_STAGES 4
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 132096
#define SMEM_B_STAGE_BYTES 8192
#define SMEM_B_STRIDE 8192
#define SMEM_SFA_OFF 164864
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 168960
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_SINFO_OFF 171008
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 171232
#define SMEM_STOK_STAGE_BYTES 2048
#define SMEM_STOK_STRIDE 2048
#define SMEM_SSCALE_OFF 173280
#define SMEM_SSCALE_STAGE_BYTES 2080
#define SMEM_SSCALE_STRIDE 2080
#define SMEM_SEXCH_OFF 175360
#define SMEM_SEXCH_STAGE_BYTES 8448
#define SMEM_SEXCH_STRIDE 8448
#define SMEM_TOTAL 183808
#define THREADS 256

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



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_elect(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
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





__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}



__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_a82edeffb831f481725f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap B, uint8_t* __restrict__ SFB, __nv_bfloat16* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
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
    #define ab_free_addr (mbar_base + 32)
    #define b_full_addr (mbar_base + 64)
    #define b_free_addr (mbar_base + 96)
    #define acc_full_addr (mbar_base + 128)
    #define acc_free_addr (mbar_base + 144)
    #define tile_full_addr (mbar_base + 160)
    #define tile_free_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_addr = smem + 1024;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int b_addr = smem + 132096;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int sfa_addr = smem + 164864;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 168960);
    const int sfb_addr = smem + 168960;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 171008);
    const int sinfo_addr = smem + 171008;
    int* stok = reinterpret_cast<int*>(smem_raw + 171232);
    const int stok_addr = smem + 171232;
    float* sscale = reinterpret_cast<float*>(smem_raw + 173280);
    const int sscale_addr = smem + 173280;
    float* sexch = reinterpret_cast<float*>(smem_raw + 175360);
    const int sexch_addr = smem + 175360;
    int _mma_base_lo_0 = ((a_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((b_addr) >> 4) & 0x3FFF;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[0..288)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 4 barriers, init_count=1
        // ab_free: 4 barriers, init_count=1
        // --- pipeline 'pb' ---
        // b_full: 4 barriers, init_count=32
        // b_free: 4 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 8 barriers, init_count=32
        // tile_free: 8 barriers, init_count=224
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 224;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(28), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(20), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        if (lane < 4) {
            mbarrier_init(smem + 256 + lane * 8, 224);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 268 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 0) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_acc1 = taddr + 128;
    const int tmem_sf_a1 = taddr + 260;
    const int tmem_sf_b = taddr + 264;
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
            int rec_m_tile = 0;
            float vals[64];
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 7];
            info[1] = sinfo[tile_stage * 7 + 1];
            info[2] = sinfo[tile_stage * 7 + 2];
            info[3] = sinfo[tile_stage * 7 + 3];
            info[4] = sinfo[tile_stage * 7 + 4];
            info[5] = sinfo[tile_stage * 7 + 5];
            info[6] = sinfo[tile_stage * 7 + 6];
            rec_m_tile = sinfo[tile_stage * 7];
            meta_alpha = sscale[tile_stage * 65 + 64];
            nrec_epilogue = nrec_epilogue + 1;
            bool is_even_lane = lane_0 % 2 == 0;
            int is_gate_lane = (int)(epi_warp >= 2);
            int exch_row = epi_tidx & 63;
            float inv_fp8_max = 0.002232142857142857f;
            float zero_f32 = 0.0f;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = info[1] * 64;
                int mn_limit = info[4];
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int m_tile_e = rec_m_tile * 2;
                int h = m_tile_e * 128 + epi_tidx;
                if (m_tile_e < num_m_tiles) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_0[0] * meta_alpha;
                    vals[1] = _tmem_load_0[1] * meta_alpha;
                    vals[2] = _tmem_load_0[2] * meta_alpha;
                    vals[3] = _tmem_load_0[3] * meta_alpha;
                    vals[4] = _tmem_load_0[4] * meta_alpha;
                    vals[5] = _tmem_load_0[5] * meta_alpha;
                    vals[6] = _tmem_load_0[6] * meta_alpha;
                    vals[7] = _tmem_load_0[7] * meta_alpha;
                    vals[8] = _tmem_load_0[8] * meta_alpha;
                    vals[9] = _tmem_load_0[9] * meta_alpha;
                    vals[10] = _tmem_load_0[10] * meta_alpha;
                    vals[11] = _tmem_load_0[11] * meta_alpha;
                    vals[12] = _tmem_load_0[12] * meta_alpha;
                    vals[13] = _tmem_load_0[13] * meta_alpha;
                    vals[14] = _tmem_load_0[14] * meta_alpha;
                    vals[15] = _tmem_load_0[15] * meta_alpha;
                    vals[16] = _tmem_load_0[16] * meta_alpha;
                    vals[17] = _tmem_load_0[17] * meta_alpha;
                    vals[18] = _tmem_load_0[18] * meta_alpha;
                    vals[19] = _tmem_load_0[19] * meta_alpha;
                    vals[20] = _tmem_load_0[20] * meta_alpha;
                    vals[21] = _tmem_load_0[21] * meta_alpha;
                    vals[22] = _tmem_load_0[22] * meta_alpha;
                    vals[23] = _tmem_load_0[23] * meta_alpha;
                    vals[24] = _tmem_load_0[24] * meta_alpha;
                    vals[25] = _tmem_load_0[25] * meta_alpha;
                    vals[26] = _tmem_load_0[26] * meta_alpha;
                    vals[27] = _tmem_load_0[27] * meta_alpha;
                    vals[28] = _tmem_load_0[28] * meta_alpha;
                    vals[29] = _tmem_load_0[29] * meta_alpha;
                    vals[30] = _tmem_load_0[30] * meta_alpha;
                    vals[31] = _tmem_load_0[31] * meta_alpha;
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 64 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[32] = _tmem_load_1[0] * meta_alpha;
                    vals[33] = _tmem_load_1[1] * meta_alpha;
                    vals[34] = _tmem_load_1[2] * meta_alpha;
                    vals[35] = _tmem_load_1[3] * meta_alpha;
                    vals[36] = _tmem_load_1[4] * meta_alpha;
                    vals[37] = _tmem_load_1[5] * meta_alpha;
                    vals[38] = _tmem_load_1[6] * meta_alpha;
                    vals[39] = _tmem_load_1[7] * meta_alpha;
                    vals[40] = _tmem_load_1[8] * meta_alpha;
                    vals[41] = _tmem_load_1[9] * meta_alpha;
                    vals[42] = _tmem_load_1[10] * meta_alpha;
                    vals[43] = _tmem_load_1[11] * meta_alpha;
                    vals[44] = _tmem_load_1[12] * meta_alpha;
                    vals[45] = _tmem_load_1[13] * meta_alpha;
                    vals[46] = _tmem_load_1[14] * meta_alpha;
                    vals[47] = _tmem_load_1[15] * meta_alpha;
                    vals[48] = _tmem_load_1[16] * meta_alpha;
                    vals[49] = _tmem_load_1[17] * meta_alpha;
                    vals[50] = _tmem_load_1[18] * meta_alpha;
                    vals[51] = _tmem_load_1[19] * meta_alpha;
                    vals[52] = _tmem_load_1[20] * meta_alpha;
                    vals[53] = _tmem_load_1[21] * meta_alpha;
                    vals[54] = _tmem_load_1[22] * meta_alpha;
                    vals[55] = _tmem_load_1[23] * meta_alpha;
                    vals[56] = _tmem_load_1[24] * meta_alpha;
                    vals[57] = _tmem_load_1[25] * meta_alpha;
                    vals[58] = _tmem_load_1[26] * meta_alpha;
                    vals[59] = _tmem_load_1[27] * meta_alpha;
                    vals[60] = _tmem_load_1[28] * meta_alpha;
                    vals[61] = _tmem_load_1[29] * meta_alpha;
                    vals[62] = _tmem_load_1[30] * meta_alpha;
                    vals[63] = _tmem_load_1[31] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        float v = vals[0] * sscale[tile_stage * 65];
                        int tok = stok[tile_stage * 64];
                        bool col_ok = mn_limit > row_base;
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, v, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok * out_cols + h])), "f"(v), "f"(_shfl_xor_0), "r"((unsigned int)(col_ok && is_even_lane)) : "memory");
                        float v_0 = vals[1] * sscale[tile_stage * 65 + 1];
                        int tok_1 = stok[tile_stage * 64 + 1];
                        bool col_ok_2 = mn_limit > row_base + 1;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, v_0, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1 * out_cols + h])), "f"(v_0), "f"(_shfl_xor_1), "r"((unsigned int)(col_ok_2 && is_even_lane)) : "memory");
                        float v_3 = vals[2] * sscale[tile_stage * 65 + 2];
                        int tok_4 = stok[tile_stage * 64 + 2];
                        bool col_ok_5 = mn_limit > row_base + 2;
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, v_3, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4 * out_cols + h])), "f"(v_3), "f"(_shfl_xor_2), "r"((unsigned int)(col_ok_5 && is_even_lane)) : "memory");
                        float v_6 = vals[3] * sscale[tile_stage * 65 + 3];
                        int tok_7 = stok[tile_stage * 64 + 3];
                        bool col_ok_8 = mn_limit > row_base + 3;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, v_6, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7 * out_cols + h])), "f"(v_6), "f"(_shfl_xor_3), "r"((unsigned int)(col_ok_8 && is_even_lane)) : "memory");
                        float v_9 = vals[4] * sscale[tile_stage * 65 + 4];
                        int tok_10 = stok[tile_stage * 64 + 4];
                        bool col_ok_11 = mn_limit > row_base + 4;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, v_9, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10 * out_cols + h])), "f"(v_9), "f"(_shfl_xor_4), "r"((unsigned int)(col_ok_11 && is_even_lane)) : "memory");
                        float v_12 = vals[5] * sscale[tile_stage * 65 + 5];
                        int tok_13 = stok[tile_stage * 64 + 5];
                        bool col_ok_14 = mn_limit > row_base + 5;
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, v_12, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13 * out_cols + h])), "f"(v_12), "f"(_shfl_xor_5), "r"((unsigned int)(col_ok_14 && is_even_lane)) : "memory");
                        float v_15 = vals[6] * sscale[tile_stage * 65 + 6];
                        int tok_16 = stok[tile_stage * 64 + 6];
                        bool col_ok_17 = mn_limit > row_base + 6;
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, v_15, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16 * out_cols + h])), "f"(v_15), "f"(_shfl_xor_6), "r"((unsigned int)(col_ok_17 && is_even_lane)) : "memory");
                        float v_18 = vals[7] * sscale[tile_stage * 65 + 7];
                        int tok_19 = stok[tile_stage * 64 + 7];
                        bool col_ok_20 = mn_limit > row_base + 7;
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, v_18, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19 * out_cols + h])), "f"(v_18), "f"(_shfl_xor_7), "r"((unsigned int)(col_ok_20 && is_even_lane)) : "memory");
                        float v_21 = vals[8] * sscale[tile_stage * 65 + 8];
                        int tok_22 = stok[tile_stage * 64 + 8];
                        bool col_ok_23 = mn_limit > row_base + 8;
                        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, v_21, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22 * out_cols + h])), "f"(v_21), "f"(_shfl_xor_8), "r"((unsigned int)(col_ok_23 && is_even_lane)) : "memory");
                        float v_24 = vals[9] * sscale[tile_stage * 65 + 9];
                        int tok_25 = stok[tile_stage * 64 + 9];
                        bool col_ok_26 = mn_limit > row_base + 9;
                        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, v_24, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25 * out_cols + h])), "f"(v_24), "f"(_shfl_xor_9), "r"((unsigned int)(col_ok_26 && is_even_lane)) : "memory");
                        float v_27 = vals[10] * sscale[tile_stage * 65 + 10];
                        int tok_28 = stok[tile_stage * 64 + 10];
                        bool col_ok_29 = mn_limit > row_base + 10;
                        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, v_27, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28 * out_cols + h])), "f"(v_27), "f"(_shfl_xor_10), "r"((unsigned int)(col_ok_29 && is_even_lane)) : "memory");
                        float v_30 = vals[11] * sscale[tile_stage * 65 + 11];
                        int tok_31 = stok[tile_stage * 64 + 11];
                        bool col_ok_32 = mn_limit > row_base + 11;
                        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, v_30, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31 * out_cols + h])), "f"(v_30), "f"(_shfl_xor_11), "r"((unsigned int)(col_ok_32 && is_even_lane)) : "memory");
                        float v_33 = vals[12] * sscale[tile_stage * 65 + 12];
                        int tok_34 = stok[tile_stage * 64 + 12];
                        bool col_ok_35 = mn_limit > row_base + 12;
                        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, v_33, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34 * out_cols + h])), "f"(v_33), "f"(_shfl_xor_12), "r"((unsigned int)(col_ok_35 && is_even_lane)) : "memory");
                        float v_36 = vals[13] * sscale[tile_stage * 65 + 13];
                        int tok_37 = stok[tile_stage * 64 + 13];
                        bool col_ok_38 = mn_limit > row_base + 13;
                        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, v_36, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37 * out_cols + h])), "f"(v_36), "f"(_shfl_xor_13), "r"((unsigned int)(col_ok_38 && is_even_lane)) : "memory");
                        float v_39 = vals[14] * sscale[tile_stage * 65 + 14];
                        int tok_40 = stok[tile_stage * 64 + 14];
                        bool col_ok_41 = mn_limit > row_base + 14;
                        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, v_39, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40 * out_cols + h])), "f"(v_39), "f"(_shfl_xor_14), "r"((unsigned int)(col_ok_41 && is_even_lane)) : "memory");
                        float v_42 = vals[15] * sscale[tile_stage * 65 + 15];
                        int tok_43 = stok[tile_stage * 64 + 15];
                        bool col_ok_44 = mn_limit > row_base + 15;
                        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, v_42, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43 * out_cols + h])), "f"(v_42), "f"(_shfl_xor_15), "r"((unsigned int)(col_ok_44 && is_even_lane)) : "memory");
                        float v_45 = vals[16] * sscale[tile_stage * 65 + 16];
                        int tok_46 = stok[tile_stage * 64 + 16];
                        bool col_ok_47 = mn_limit > row_base + 16;
                        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, v_45, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_46 * out_cols + h])), "f"(v_45), "f"(_shfl_xor_16), "r"((unsigned int)(col_ok_47 && is_even_lane)) : "memory");
                        float v_48 = vals[17] * sscale[tile_stage * 65 + 17];
                        int tok_49 = stok[tile_stage * 64 + 17];
                        bool col_ok_50 = mn_limit > row_base + 17;
                        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, v_48, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_49 * out_cols + h])), "f"(v_48), "f"(_shfl_xor_17), "r"((unsigned int)(col_ok_50 && is_even_lane)) : "memory");
                        float v_51 = vals[18] * sscale[tile_stage * 65 + 18];
                        int tok_52 = stok[tile_stage * 64 + 18];
                        bool col_ok_53 = mn_limit > row_base + 18;
                        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, v_51, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_52 * out_cols + h])), "f"(v_51), "f"(_shfl_xor_18), "r"((unsigned int)(col_ok_53 && is_even_lane)) : "memory");
                        float v_54 = vals[19] * sscale[tile_stage * 65 + 19];
                        int tok_55 = stok[tile_stage * 64 + 19];
                        bool col_ok_56 = mn_limit > row_base + 19;
                        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, v_54, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_55 * out_cols + h])), "f"(v_54), "f"(_shfl_xor_19), "r"((unsigned int)(col_ok_56 && is_even_lane)) : "memory");
                        float v_57 = vals[20] * sscale[tile_stage * 65 + 20];
                        int tok_58 = stok[tile_stage * 64 + 20];
                        bool col_ok_59 = mn_limit > row_base + 20;
                        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, v_57, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_58 * out_cols + h])), "f"(v_57), "f"(_shfl_xor_20), "r"((unsigned int)(col_ok_59 && is_even_lane)) : "memory");
                        float v_60 = vals[21] * sscale[tile_stage * 65 + 21];
                        int tok_61 = stok[tile_stage * 64 + 21];
                        bool col_ok_62 = mn_limit > row_base + 21;
                        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, v_60, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_61 * out_cols + h])), "f"(v_60), "f"(_shfl_xor_21), "r"((unsigned int)(col_ok_62 && is_even_lane)) : "memory");
                        float v_63 = vals[22] * sscale[tile_stage * 65 + 22];
                        int tok_64 = stok[tile_stage * 64 + 22];
                        bool col_ok_65 = mn_limit > row_base + 22;
                        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, v_63, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_64 * out_cols + h])), "f"(v_63), "f"(_shfl_xor_22), "r"((unsigned int)(col_ok_65 && is_even_lane)) : "memory");
                        float v_66 = vals[23] * sscale[tile_stage * 65 + 23];
                        int tok_67 = stok[tile_stage * 64 + 23];
                        bool col_ok_68 = mn_limit > row_base + 23;
                        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, v_66, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_67 * out_cols + h])), "f"(v_66), "f"(_shfl_xor_23), "r"((unsigned int)(col_ok_68 && is_even_lane)) : "memory");
                        float v_69 = vals[24] * sscale[tile_stage * 65 + 24];
                        int tok_70 = stok[tile_stage * 64 + 24];
                        bool col_ok_71 = mn_limit > row_base + 24;
                        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, v_69, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_70 * out_cols + h])), "f"(v_69), "f"(_shfl_xor_24), "r"((unsigned int)(col_ok_71 && is_even_lane)) : "memory");
                        float v_72 = vals[25] * sscale[tile_stage * 65 + 25];
                        int tok_73 = stok[tile_stage * 64 + 25];
                        bool col_ok_74 = mn_limit > row_base + 25;
                        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, v_72, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_73 * out_cols + h])), "f"(v_72), "f"(_shfl_xor_25), "r"((unsigned int)(col_ok_74 && is_even_lane)) : "memory");
                        float v_75 = vals[26] * sscale[tile_stage * 65 + 26];
                        int tok_76 = stok[tile_stage * 64 + 26];
                        bool col_ok_77 = mn_limit > row_base + 26;
                        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, v_75, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_76 * out_cols + h])), "f"(v_75), "f"(_shfl_xor_26), "r"((unsigned int)(col_ok_77 && is_even_lane)) : "memory");
                        float v_78 = vals[27] * sscale[tile_stage * 65 + 27];
                        int tok_79 = stok[tile_stage * 64 + 27];
                        bool col_ok_80 = mn_limit > row_base + 27;
                        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, v_78, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_79 * out_cols + h])), "f"(v_78), "f"(_shfl_xor_27), "r"((unsigned int)(col_ok_80 && is_even_lane)) : "memory");
                        float v_81 = vals[28] * sscale[tile_stage * 65 + 28];
                        int tok_82 = stok[tile_stage * 64 + 28];
                        bool col_ok_83 = mn_limit > row_base + 28;
                        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, v_81, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_82 * out_cols + h])), "f"(v_81), "f"(_shfl_xor_28), "r"((unsigned int)(col_ok_83 && is_even_lane)) : "memory");
                        float v_84 = vals[29] * sscale[tile_stage * 65 + 29];
                        int tok_85 = stok[tile_stage * 64 + 29];
                        bool col_ok_86 = mn_limit > row_base + 29;
                        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, v_84, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_85 * out_cols + h])), "f"(v_84), "f"(_shfl_xor_29), "r"((unsigned int)(col_ok_86 && is_even_lane)) : "memory");
                        float v_87 = vals[30] * sscale[tile_stage * 65 + 30];
                        int tok_88 = stok[tile_stage * 64 + 30];
                        bool col_ok_89 = mn_limit > row_base + 30;
                        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, v_87, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_88 * out_cols + h])), "f"(v_87), "f"(_shfl_xor_30), "r"((unsigned int)(col_ok_89 && is_even_lane)) : "memory");
                        float v_90 = vals[31] * sscale[tile_stage * 65 + 31];
                        int tok_91 = stok[tile_stage * 64 + 31];
                        bool col_ok_92 = mn_limit > row_base + 31;
                        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, v_90, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_91 * out_cols + h])), "f"(v_90), "f"(_shfl_xor_31), "r"((unsigned int)(col_ok_92 && is_even_lane)) : "memory");
                        float v_93 = vals[32] * sscale[tile_stage * 65 + 32];
                        int tok_94 = stok[tile_stage * 64 + 32];
                        bool col_ok_95 = mn_limit > row_base + 32;
                        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, v_93, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_94 * out_cols + h])), "f"(v_93), "f"(_shfl_xor_32), "r"((unsigned int)(col_ok_95 && is_even_lane)) : "memory");
                        float v_96 = vals[33] * sscale[tile_stage * 65 + 33];
                        int tok_97 = stok[tile_stage * 64 + 33];
                        bool col_ok_98 = mn_limit > row_base + 33;
                        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, v_96, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_97 * out_cols + h])), "f"(v_96), "f"(_shfl_xor_33), "r"((unsigned int)(col_ok_98 && is_even_lane)) : "memory");
                        float v_99 = vals[34] * sscale[tile_stage * 65 + 34];
                        int tok_100 = stok[tile_stage * 64 + 34];
                        bool col_ok_101 = mn_limit > row_base + 34;
                        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, v_99, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_100 * out_cols + h])), "f"(v_99), "f"(_shfl_xor_34), "r"((unsigned int)(col_ok_101 && is_even_lane)) : "memory");
                        float v_102 = vals[35] * sscale[tile_stage * 65 + 35];
                        int tok_103 = stok[tile_stage * 64 + 35];
                        bool col_ok_104 = mn_limit > row_base + 35;
                        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, v_102, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_103 * out_cols + h])), "f"(v_102), "f"(_shfl_xor_35), "r"((unsigned int)(col_ok_104 && is_even_lane)) : "memory");
                        float v_105 = vals[36] * sscale[tile_stage * 65 + 36];
                        int tok_106 = stok[tile_stage * 64 + 36];
                        bool col_ok_107 = mn_limit > row_base + 36;
                        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, v_105, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_106 * out_cols + h])), "f"(v_105), "f"(_shfl_xor_36), "r"((unsigned int)(col_ok_107 && is_even_lane)) : "memory");
                        float v_108 = vals[37] * sscale[tile_stage * 65 + 37];
                        int tok_109 = stok[tile_stage * 64 + 37];
                        bool col_ok_110 = mn_limit > row_base + 37;
                        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, v_108, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_109 * out_cols + h])), "f"(v_108), "f"(_shfl_xor_37), "r"((unsigned int)(col_ok_110 && is_even_lane)) : "memory");
                        float v_111 = vals[38] * sscale[tile_stage * 65 + 38];
                        int tok_112 = stok[tile_stage * 64 + 38];
                        bool col_ok_113 = mn_limit > row_base + 38;
                        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, v_111, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_112 * out_cols + h])), "f"(v_111), "f"(_shfl_xor_38), "r"((unsigned int)(col_ok_113 && is_even_lane)) : "memory");
                        float v_114 = vals[39] * sscale[tile_stage * 65 + 39];
                        int tok_115 = stok[tile_stage * 64 + 39];
                        bool col_ok_116 = mn_limit > row_base + 39;
                        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, v_114, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_115 * out_cols + h])), "f"(v_114), "f"(_shfl_xor_39), "r"((unsigned int)(col_ok_116 && is_even_lane)) : "memory");
                        float v_117 = vals[40] * sscale[tile_stage * 65 + 40];
                        int tok_118 = stok[tile_stage * 64 + 40];
                        bool col_ok_119 = mn_limit > row_base + 40;
                        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, v_117, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_118 * out_cols + h])), "f"(v_117), "f"(_shfl_xor_40), "r"((unsigned int)(col_ok_119 && is_even_lane)) : "memory");
                        float v_120 = vals[41] * sscale[tile_stage * 65 + 41];
                        int tok_121 = stok[tile_stage * 64 + 41];
                        bool col_ok_122 = mn_limit > row_base + 41;
                        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, v_120, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_121 * out_cols + h])), "f"(v_120), "f"(_shfl_xor_41), "r"((unsigned int)(col_ok_122 && is_even_lane)) : "memory");
                        float v_123 = vals[42] * sscale[tile_stage * 65 + 42];
                        int tok_124 = stok[tile_stage * 64 + 42];
                        bool col_ok_125 = mn_limit > row_base + 42;
                        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, v_123, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_124 * out_cols + h])), "f"(v_123), "f"(_shfl_xor_42), "r"((unsigned int)(col_ok_125 && is_even_lane)) : "memory");
                        float v_126 = vals[43] * sscale[tile_stage * 65 + 43];
                        int tok_127 = stok[tile_stage * 64 + 43];
                        bool col_ok_128 = mn_limit > row_base + 43;
                        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, v_126, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_127 * out_cols + h])), "f"(v_126), "f"(_shfl_xor_43), "r"((unsigned int)(col_ok_128 && is_even_lane)) : "memory");
                        float v_129 = vals[44] * sscale[tile_stage * 65 + 44];
                        int tok_130 = stok[tile_stage * 64 + 44];
                        bool col_ok_131 = mn_limit > row_base + 44;
                        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, v_129, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_130 * out_cols + h])), "f"(v_129), "f"(_shfl_xor_44), "r"((unsigned int)(col_ok_131 && is_even_lane)) : "memory");
                        float v_132 = vals[45] * sscale[tile_stage * 65 + 45];
                        int tok_133 = stok[tile_stage * 64 + 45];
                        bool col_ok_134 = mn_limit > row_base + 45;
                        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, v_132, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_133 * out_cols + h])), "f"(v_132), "f"(_shfl_xor_45), "r"((unsigned int)(col_ok_134 && is_even_lane)) : "memory");
                        float v_135 = vals[46] * sscale[tile_stage * 65 + 46];
                        int tok_136 = stok[tile_stage * 64 + 46];
                        bool col_ok_137 = mn_limit > row_base + 46;
                        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, v_135, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_136 * out_cols + h])), "f"(v_135), "f"(_shfl_xor_46), "r"((unsigned int)(col_ok_137 && is_even_lane)) : "memory");
                        float v_138 = vals[47] * sscale[tile_stage * 65 + 47];
                        int tok_139 = stok[tile_stage * 64 + 47];
                        bool col_ok_140 = mn_limit > row_base + 47;
                        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, v_138, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_139 * out_cols + h])), "f"(v_138), "f"(_shfl_xor_47), "r"((unsigned int)(col_ok_140 && is_even_lane)) : "memory");
                        float v_141 = vals[48] * sscale[tile_stage * 65 + 48];
                        int tok_142 = stok[tile_stage * 64 + 48];
                        bool col_ok_143 = mn_limit > row_base + 48;
                        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, v_141, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_142 * out_cols + h])), "f"(v_141), "f"(_shfl_xor_48), "r"((unsigned int)(col_ok_143 && is_even_lane)) : "memory");
                        float v_144 = vals[49] * sscale[tile_stage * 65 + 49];
                        int tok_145 = stok[tile_stage * 64 + 49];
                        bool col_ok_146 = mn_limit > row_base + 49;
                        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, v_144, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_145 * out_cols + h])), "f"(v_144), "f"(_shfl_xor_49), "r"((unsigned int)(col_ok_146 && is_even_lane)) : "memory");
                        float v_147 = vals[50] * sscale[tile_stage * 65 + 50];
                        int tok_148 = stok[tile_stage * 64 + 50];
                        bool col_ok_149 = mn_limit > row_base + 50;
                        float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, v_147, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_148 * out_cols + h])), "f"(v_147), "f"(_shfl_xor_50), "r"((unsigned int)(col_ok_149 && is_even_lane)) : "memory");
                        float v_150 = vals[51] * sscale[tile_stage * 65 + 51];
                        int tok_151 = stok[tile_stage * 64 + 51];
                        bool col_ok_152 = mn_limit > row_base + 51;
                        float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, v_150, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_151 * out_cols + h])), "f"(v_150), "f"(_shfl_xor_51), "r"((unsigned int)(col_ok_152 && is_even_lane)) : "memory");
                        float v_153 = vals[52] * sscale[tile_stage * 65 + 52];
                        int tok_154 = stok[tile_stage * 64 + 52];
                        bool col_ok_155 = mn_limit > row_base + 52;
                        float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, v_153, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_154 * out_cols + h])), "f"(v_153), "f"(_shfl_xor_52), "r"((unsigned int)(col_ok_155 && is_even_lane)) : "memory");
                        float v_156 = vals[53] * sscale[tile_stage * 65 + 53];
                        int tok_157 = stok[tile_stage * 64 + 53];
                        bool col_ok_158 = mn_limit > row_base + 53;
                        float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, v_156, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_157 * out_cols + h])), "f"(v_156), "f"(_shfl_xor_53), "r"((unsigned int)(col_ok_158 && is_even_lane)) : "memory");
                        float v_159 = vals[54] * sscale[tile_stage * 65 + 54];
                        int tok_160 = stok[tile_stage * 64 + 54];
                        bool col_ok_161 = mn_limit > row_base + 54;
                        float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, v_159, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_160 * out_cols + h])), "f"(v_159), "f"(_shfl_xor_54), "r"((unsigned int)(col_ok_161 && is_even_lane)) : "memory");
                        float v_162 = vals[55] * sscale[tile_stage * 65 + 55];
                        int tok_163 = stok[tile_stage * 64 + 55];
                        bool col_ok_164 = mn_limit > row_base + 55;
                        float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, v_162, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_163 * out_cols + h])), "f"(v_162), "f"(_shfl_xor_55), "r"((unsigned int)(col_ok_164 && is_even_lane)) : "memory");
                        float v_165 = vals[56] * sscale[tile_stage * 65 + 56];
                        int tok_166 = stok[tile_stage * 64 + 56];
                        bool col_ok_167 = mn_limit > row_base + 56;
                        float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, v_165, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_166 * out_cols + h])), "f"(v_165), "f"(_shfl_xor_56), "r"((unsigned int)(col_ok_167 && is_even_lane)) : "memory");
                        float v_168 = vals[57] * sscale[tile_stage * 65 + 57];
                        int tok_169 = stok[tile_stage * 64 + 57];
                        bool col_ok_170 = mn_limit > row_base + 57;
                        float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, v_168, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_169 * out_cols + h])), "f"(v_168), "f"(_shfl_xor_57), "r"((unsigned int)(col_ok_170 && is_even_lane)) : "memory");
                        float v_171 = vals[58] * sscale[tile_stage * 65 + 58];
                        int tok_172 = stok[tile_stage * 64 + 58];
                        bool col_ok_173 = mn_limit > row_base + 58;
                        float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, v_171, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_172 * out_cols + h])), "f"(v_171), "f"(_shfl_xor_58), "r"((unsigned int)(col_ok_173 && is_even_lane)) : "memory");
                        float v_174 = vals[59] * sscale[tile_stage * 65 + 59];
                        int tok_175 = stok[tile_stage * 64 + 59];
                        bool col_ok_176 = mn_limit > row_base + 59;
                        float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, v_174, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_175 * out_cols + h])), "f"(v_174), "f"(_shfl_xor_59), "r"((unsigned int)(col_ok_176 && is_even_lane)) : "memory");
                        float v_177 = vals[60] * sscale[tile_stage * 65 + 60];
                        int tok_178 = stok[tile_stage * 64 + 60];
                        bool col_ok_179 = mn_limit > row_base + 60;
                        float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, v_177, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_178 * out_cols + h])), "f"(v_177), "f"(_shfl_xor_60), "r"((unsigned int)(col_ok_179 && is_even_lane)) : "memory");
                        float v_180 = vals[61] * sscale[tile_stage * 65 + 61];
                        int tok_181 = stok[tile_stage * 64 + 61];
                        bool col_ok_182 = mn_limit > row_base + 61;
                        float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, v_180, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_181 * out_cols + h])), "f"(v_180), "f"(_shfl_xor_61), "r"((unsigned int)(col_ok_182 && is_even_lane)) : "memory");
                        float v_183 = vals[62] * sscale[tile_stage * 65 + 62];
                        int tok_184 = stok[tile_stage * 64 + 62];
                        bool col_ok_185 = mn_limit > row_base + 62;
                        float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, v_183, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_184 * out_cols + h])), "f"(v_183), "f"(_shfl_xor_62), "r"((unsigned int)(col_ok_185 && is_even_lane)) : "memory");
                        float v_186 = vals[63] * sscale[tile_stage * 65 + 63];
                        int tok_187 = stok[tile_stage * 64 + 63];
                        bool col_ok_188 = mn_limit > row_base + 63;
                        float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, v_186, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_187 * out_cols + h])), "f"(v_186), "f"(_shfl_xor_63), "r"((unsigned int)(col_ok_188 && is_even_lane)) : "memory");
                    }
                }
                int m_tile_e_0 = rec_m_tile * 2 + 1;
                int h_1 = m_tile_e_0 * 128 + epi_tidx;
                if (m_tile_e_0 < num_m_tiles) {
                    float _tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(taddr + 128 + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_2[0] * meta_alpha;
                    vals[1] = _tmem_load_2[1] * meta_alpha;
                    vals[2] = _tmem_load_2[2] * meta_alpha;
                    vals[3] = _tmem_load_2[3] * meta_alpha;
                    vals[4] = _tmem_load_2[4] * meta_alpha;
                    vals[5] = _tmem_load_2[5] * meta_alpha;
                    vals[6] = _tmem_load_2[6] * meta_alpha;
                    vals[7] = _tmem_load_2[7] * meta_alpha;
                    vals[8] = _tmem_load_2[8] * meta_alpha;
                    vals[9] = _tmem_load_2[9] * meta_alpha;
                    vals[10] = _tmem_load_2[10] * meta_alpha;
                    vals[11] = _tmem_load_2[11] * meta_alpha;
                    vals[12] = _tmem_load_2[12] * meta_alpha;
                    vals[13] = _tmem_load_2[13] * meta_alpha;
                    vals[14] = _tmem_load_2[14] * meta_alpha;
                    vals[15] = _tmem_load_2[15] * meta_alpha;
                    vals[16] = _tmem_load_2[16] * meta_alpha;
                    vals[17] = _tmem_load_2[17] * meta_alpha;
                    vals[18] = _tmem_load_2[18] * meta_alpha;
                    vals[19] = _tmem_load_2[19] * meta_alpha;
                    vals[20] = _tmem_load_2[20] * meta_alpha;
                    vals[21] = _tmem_load_2[21] * meta_alpha;
                    vals[22] = _tmem_load_2[22] * meta_alpha;
                    vals[23] = _tmem_load_2[23] * meta_alpha;
                    vals[24] = _tmem_load_2[24] * meta_alpha;
                    vals[25] = _tmem_load_2[25] * meta_alpha;
                    vals[26] = _tmem_load_2[26] * meta_alpha;
                    vals[27] = _tmem_load_2[27] * meta_alpha;
                    vals[28] = _tmem_load_2[28] * meta_alpha;
                    vals[29] = _tmem_load_2[29] * meta_alpha;
                    vals[30] = _tmem_load_2[30] * meta_alpha;
                    vals[31] = _tmem_load_2[31] * meta_alpha;
                    float _tmem_load_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(taddr + 128 + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 64 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[32] = _tmem_load_3[0] * meta_alpha;
                    vals[33] = _tmem_load_3[1] * meta_alpha;
                    vals[34] = _tmem_load_3[2] * meta_alpha;
                    vals[35] = _tmem_load_3[3] * meta_alpha;
                    vals[36] = _tmem_load_3[4] * meta_alpha;
                    vals[37] = _tmem_load_3[5] * meta_alpha;
                    vals[38] = _tmem_load_3[6] * meta_alpha;
                    vals[39] = _tmem_load_3[7] * meta_alpha;
                    vals[40] = _tmem_load_3[8] * meta_alpha;
                    vals[41] = _tmem_load_3[9] * meta_alpha;
                    vals[42] = _tmem_load_3[10] * meta_alpha;
                    vals[43] = _tmem_load_3[11] * meta_alpha;
                    vals[44] = _tmem_load_3[12] * meta_alpha;
                    vals[45] = _tmem_load_3[13] * meta_alpha;
                    vals[46] = _tmem_load_3[14] * meta_alpha;
                    vals[47] = _tmem_load_3[15] * meta_alpha;
                    vals[48] = _tmem_load_3[16] * meta_alpha;
                    vals[49] = _tmem_load_3[17] * meta_alpha;
                    vals[50] = _tmem_load_3[18] * meta_alpha;
                    vals[51] = _tmem_load_3[19] * meta_alpha;
                    vals[52] = _tmem_load_3[20] * meta_alpha;
                    vals[53] = _tmem_load_3[21] * meta_alpha;
                    vals[54] = _tmem_load_3[22] * meta_alpha;
                    vals[55] = _tmem_load_3[23] * meta_alpha;
                    vals[56] = _tmem_load_3[24] * meta_alpha;
                    vals[57] = _tmem_load_3[25] * meta_alpha;
                    vals[58] = _tmem_load_3[26] * meta_alpha;
                    vals[59] = _tmem_load_3[27] * meta_alpha;
                    vals[60] = _tmem_load_3[28] * meta_alpha;
                    vals[61] = _tmem_load_3[29] * meta_alpha;
                    vals[62] = _tmem_load_3[30] * meta_alpha;
                    vals[63] = _tmem_load_3[31] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        float v_1 = vals[0] * sscale[tile_stage * 65];
                        int tok_2 = stok[tile_stage * 64];
                        bool col_ok_1 = mn_limit > row_base;
                        float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, v_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_2 * out_cols + h_1])), "f"(v_1), "f"(_shfl_xor_64), "r"((unsigned int)(col_ok_1 && is_even_lane)) : "memory");
                        float v_0_1 = vals[1] * sscale[tile_stage * 65 + 1];
                        int tok_1_1 = stok[tile_stage * 64 + 1];
                        bool col_ok_2_1 = mn_limit > row_base + 1;
                        float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, v_0_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1_1 * out_cols + h_1])), "f"(v_0_1), "f"(_shfl_xor_65), "r"((unsigned int)(col_ok_2_1 && is_even_lane)) : "memory");
                        float v_3_1 = vals[2] * sscale[tile_stage * 65 + 2];
                        int tok_4_1 = stok[tile_stage * 64 + 2];
                        bool col_ok_5_1 = mn_limit > row_base + 2;
                        float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, v_3_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4_1 * out_cols + h_1])), "f"(v_3_1), "f"(_shfl_xor_66), "r"((unsigned int)(col_ok_5_1 && is_even_lane)) : "memory");
                        float v_6_1 = vals[3] * sscale[tile_stage * 65 + 3];
                        int tok_7_1 = stok[tile_stage * 64 + 3];
                        bool col_ok_8_1 = mn_limit > row_base + 3;
                        float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, v_6_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7_1 * out_cols + h_1])), "f"(v_6_1), "f"(_shfl_xor_67), "r"((unsigned int)(col_ok_8_1 && is_even_lane)) : "memory");
                        float v_9_1 = vals[4] * sscale[tile_stage * 65 + 4];
                        int tok_10_1 = stok[tile_stage * 64 + 4];
                        bool col_ok_11_1 = mn_limit > row_base + 4;
                        float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, v_9_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10_1 * out_cols + h_1])), "f"(v_9_1), "f"(_shfl_xor_68), "r"((unsigned int)(col_ok_11_1 && is_even_lane)) : "memory");
                        float v_12_1 = vals[5] * sscale[tile_stage * 65 + 5];
                        int tok_13_1 = stok[tile_stage * 64 + 5];
                        bool col_ok_14_1 = mn_limit > row_base + 5;
                        float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, v_12_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13_1 * out_cols + h_1])), "f"(v_12_1), "f"(_shfl_xor_69), "r"((unsigned int)(col_ok_14_1 && is_even_lane)) : "memory");
                        float v_15_1 = vals[6] * sscale[tile_stage * 65 + 6];
                        int tok_16_1 = stok[tile_stage * 64 + 6];
                        bool col_ok_17_1 = mn_limit > row_base + 6;
                        float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, v_15_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16_1 * out_cols + h_1])), "f"(v_15_1), "f"(_shfl_xor_70), "r"((unsigned int)(col_ok_17_1 && is_even_lane)) : "memory");
                        float v_18_1 = vals[7] * sscale[tile_stage * 65 + 7];
                        int tok_19_1 = stok[tile_stage * 64 + 7];
                        bool col_ok_20_1 = mn_limit > row_base + 7;
                        float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, v_18_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19_1 * out_cols + h_1])), "f"(v_18_1), "f"(_shfl_xor_71), "r"((unsigned int)(col_ok_20_1 && is_even_lane)) : "memory");
                        float v_21_1 = vals[8] * sscale[tile_stage * 65 + 8];
                        int tok_22_1 = stok[tile_stage * 64 + 8];
                        bool col_ok_23_1 = mn_limit > row_base + 8;
                        float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, v_21_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22_1 * out_cols + h_1])), "f"(v_21_1), "f"(_shfl_xor_72), "r"((unsigned int)(col_ok_23_1 && is_even_lane)) : "memory");
                        float v_24_1 = vals[9] * sscale[tile_stage * 65 + 9];
                        int tok_25_1 = stok[tile_stage * 64 + 9];
                        bool col_ok_26_1 = mn_limit > row_base + 9;
                        float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, v_24_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25_1 * out_cols + h_1])), "f"(v_24_1), "f"(_shfl_xor_73), "r"((unsigned int)(col_ok_26_1 && is_even_lane)) : "memory");
                        float v_27_1 = vals[10] * sscale[tile_stage * 65 + 10];
                        int tok_28_1 = stok[tile_stage * 64 + 10];
                        bool col_ok_29_1 = mn_limit > row_base + 10;
                        float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, v_27_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28_1 * out_cols + h_1])), "f"(v_27_1), "f"(_shfl_xor_74), "r"((unsigned int)(col_ok_29_1 && is_even_lane)) : "memory");
                        float v_30_1 = vals[11] * sscale[tile_stage * 65 + 11];
                        int tok_31_1 = stok[tile_stage * 64 + 11];
                        bool col_ok_32_1 = mn_limit > row_base + 11;
                        float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, v_30_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31_1 * out_cols + h_1])), "f"(v_30_1), "f"(_shfl_xor_75), "r"((unsigned int)(col_ok_32_1 && is_even_lane)) : "memory");
                        float v_33_1 = vals[12] * sscale[tile_stage * 65 + 12];
                        int tok_34_1 = stok[tile_stage * 64 + 12];
                        bool col_ok_35_1 = mn_limit > row_base + 12;
                        float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, v_33_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34_1 * out_cols + h_1])), "f"(v_33_1), "f"(_shfl_xor_76), "r"((unsigned int)(col_ok_35_1 && is_even_lane)) : "memory");
                        float v_36_1 = vals[13] * sscale[tile_stage * 65 + 13];
                        int tok_37_1 = stok[tile_stage * 64 + 13];
                        bool col_ok_38_1 = mn_limit > row_base + 13;
                        float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, v_36_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37_1 * out_cols + h_1])), "f"(v_36_1), "f"(_shfl_xor_77), "r"((unsigned int)(col_ok_38_1 && is_even_lane)) : "memory");
                        float v_39_1 = vals[14] * sscale[tile_stage * 65 + 14];
                        int tok_40_1 = stok[tile_stage * 64 + 14];
                        bool col_ok_41_1 = mn_limit > row_base + 14;
                        float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, v_39_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40_1 * out_cols + h_1])), "f"(v_39_1), "f"(_shfl_xor_78), "r"((unsigned int)(col_ok_41_1 && is_even_lane)) : "memory");
                        float v_42_1 = vals[15] * sscale[tile_stage * 65 + 15];
                        int tok_43_1 = stok[tile_stage * 64 + 15];
                        bool col_ok_44_1 = mn_limit > row_base + 15;
                        float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, v_42_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43_1 * out_cols + h_1])), "f"(v_42_1), "f"(_shfl_xor_79), "r"((unsigned int)(col_ok_44_1 && is_even_lane)) : "memory");
                        float v_45_1 = vals[16] * sscale[tile_stage * 65 + 16];
                        int tok_46_1 = stok[tile_stage * 64 + 16];
                        bool col_ok_47_1 = mn_limit > row_base + 16;
                        float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, v_45_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_46_1 * out_cols + h_1])), "f"(v_45_1), "f"(_shfl_xor_80), "r"((unsigned int)(col_ok_47_1 && is_even_lane)) : "memory");
                        float v_48_1 = vals[17] * sscale[tile_stage * 65 + 17];
                        int tok_49_1 = stok[tile_stage * 64 + 17];
                        bool col_ok_50_1 = mn_limit > row_base + 17;
                        float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, v_48_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_49_1 * out_cols + h_1])), "f"(v_48_1), "f"(_shfl_xor_81), "r"((unsigned int)(col_ok_50_1 && is_even_lane)) : "memory");
                        float v_51_1 = vals[18] * sscale[tile_stage * 65 + 18];
                        int tok_52_1 = stok[tile_stage * 64 + 18];
                        bool col_ok_53_1 = mn_limit > row_base + 18;
                        float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, v_51_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_52_1 * out_cols + h_1])), "f"(v_51_1), "f"(_shfl_xor_82), "r"((unsigned int)(col_ok_53_1 && is_even_lane)) : "memory");
                        float v_54_1 = vals[19] * sscale[tile_stage * 65 + 19];
                        int tok_55_1 = stok[tile_stage * 64 + 19];
                        bool col_ok_56_1 = mn_limit > row_base + 19;
                        float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, v_54_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_55_1 * out_cols + h_1])), "f"(v_54_1), "f"(_shfl_xor_83), "r"((unsigned int)(col_ok_56_1 && is_even_lane)) : "memory");
                        float v_57_1 = vals[20] * sscale[tile_stage * 65 + 20];
                        int tok_58_1 = stok[tile_stage * 64 + 20];
                        bool col_ok_59_1 = mn_limit > row_base + 20;
                        float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, v_57_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_58_1 * out_cols + h_1])), "f"(v_57_1), "f"(_shfl_xor_84), "r"((unsigned int)(col_ok_59_1 && is_even_lane)) : "memory");
                        float v_60_1 = vals[21] * sscale[tile_stage * 65 + 21];
                        int tok_61_1 = stok[tile_stage * 64 + 21];
                        bool col_ok_62_1 = mn_limit > row_base + 21;
                        float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, v_60_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_61_1 * out_cols + h_1])), "f"(v_60_1), "f"(_shfl_xor_85), "r"((unsigned int)(col_ok_62_1 && is_even_lane)) : "memory");
                        float v_63_1 = vals[22] * sscale[tile_stage * 65 + 22];
                        int tok_64_1 = stok[tile_stage * 64 + 22];
                        bool col_ok_65_1 = mn_limit > row_base + 22;
                        float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, v_63_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_64_1 * out_cols + h_1])), "f"(v_63_1), "f"(_shfl_xor_86), "r"((unsigned int)(col_ok_65_1 && is_even_lane)) : "memory");
                        float v_66_1 = vals[23] * sscale[tile_stage * 65 + 23];
                        int tok_67_1 = stok[tile_stage * 64 + 23];
                        bool col_ok_68_1 = mn_limit > row_base + 23;
                        float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, v_66_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_67_1 * out_cols + h_1])), "f"(v_66_1), "f"(_shfl_xor_87), "r"((unsigned int)(col_ok_68_1 && is_even_lane)) : "memory");
                        float v_69_1 = vals[24] * sscale[tile_stage * 65 + 24];
                        int tok_70_1 = stok[tile_stage * 64 + 24];
                        bool col_ok_71_1 = mn_limit > row_base + 24;
                        float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, v_69_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_70_1 * out_cols + h_1])), "f"(v_69_1), "f"(_shfl_xor_88), "r"((unsigned int)(col_ok_71_1 && is_even_lane)) : "memory");
                        float v_72_1 = vals[25] * sscale[tile_stage * 65 + 25];
                        int tok_73_1 = stok[tile_stage * 64 + 25];
                        bool col_ok_74_1 = mn_limit > row_base + 25;
                        float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, v_72_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_73_1 * out_cols + h_1])), "f"(v_72_1), "f"(_shfl_xor_89), "r"((unsigned int)(col_ok_74_1 && is_even_lane)) : "memory");
                        float v_75_1 = vals[26] * sscale[tile_stage * 65 + 26];
                        int tok_76_1 = stok[tile_stage * 64 + 26];
                        bool col_ok_77_1 = mn_limit > row_base + 26;
                        float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, v_75_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_76_1 * out_cols + h_1])), "f"(v_75_1), "f"(_shfl_xor_90), "r"((unsigned int)(col_ok_77_1 && is_even_lane)) : "memory");
                        float v_78_1 = vals[27] * sscale[tile_stage * 65 + 27];
                        int tok_79_1 = stok[tile_stage * 64 + 27];
                        bool col_ok_80_1 = mn_limit > row_base + 27;
                        float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, v_78_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_79_1 * out_cols + h_1])), "f"(v_78_1), "f"(_shfl_xor_91), "r"((unsigned int)(col_ok_80_1 && is_even_lane)) : "memory");
                        float v_81_1 = vals[28] * sscale[tile_stage * 65 + 28];
                        int tok_82_1 = stok[tile_stage * 64 + 28];
                        bool col_ok_83_1 = mn_limit > row_base + 28;
                        float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, v_81_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_82_1 * out_cols + h_1])), "f"(v_81_1), "f"(_shfl_xor_92), "r"((unsigned int)(col_ok_83_1 && is_even_lane)) : "memory");
                        float v_84_1 = vals[29] * sscale[tile_stage * 65 + 29];
                        int tok_85_1 = stok[tile_stage * 64 + 29];
                        bool col_ok_86_1 = mn_limit > row_base + 29;
                        float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, v_84_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_85_1 * out_cols + h_1])), "f"(v_84_1), "f"(_shfl_xor_93), "r"((unsigned int)(col_ok_86_1 && is_even_lane)) : "memory");
                        float v_87_1 = vals[30] * sscale[tile_stage * 65 + 30];
                        int tok_88_1 = stok[tile_stage * 64 + 30];
                        bool col_ok_89_1 = mn_limit > row_base + 30;
                        float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, v_87_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_88_1 * out_cols + h_1])), "f"(v_87_1), "f"(_shfl_xor_94), "r"((unsigned int)(col_ok_89_1 && is_even_lane)) : "memory");
                        float v_90_1 = vals[31] * sscale[tile_stage * 65 + 31];
                        int tok_91_1 = stok[tile_stage * 64 + 31];
                        bool col_ok_92_1 = mn_limit > row_base + 31;
                        float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, v_90_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_91_1 * out_cols + h_1])), "f"(v_90_1), "f"(_shfl_xor_95), "r"((unsigned int)(col_ok_92_1 && is_even_lane)) : "memory");
                        float v_93_1 = vals[32] * sscale[tile_stage * 65 + 32];
                        int tok_94_1 = stok[tile_stage * 64 + 32];
                        bool col_ok_95_1 = mn_limit > row_base + 32;
                        float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, v_93_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_94_1 * out_cols + h_1])), "f"(v_93_1), "f"(_shfl_xor_96), "r"((unsigned int)(col_ok_95_1 && is_even_lane)) : "memory");
                        float v_96_1 = vals[33] * sscale[tile_stage * 65 + 33];
                        int tok_97_1 = stok[tile_stage * 64 + 33];
                        bool col_ok_98_1 = mn_limit > row_base + 33;
                        float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, v_96_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_97_1 * out_cols + h_1])), "f"(v_96_1), "f"(_shfl_xor_97), "r"((unsigned int)(col_ok_98_1 && is_even_lane)) : "memory");
                        float v_99_1 = vals[34] * sscale[tile_stage * 65 + 34];
                        int tok_100_1 = stok[tile_stage * 64 + 34];
                        bool col_ok_101_1 = mn_limit > row_base + 34;
                        float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, v_99_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_100_1 * out_cols + h_1])), "f"(v_99_1), "f"(_shfl_xor_98), "r"((unsigned int)(col_ok_101_1 && is_even_lane)) : "memory");
                        float v_102_1 = vals[35] * sscale[tile_stage * 65 + 35];
                        int tok_103_1 = stok[tile_stage * 64 + 35];
                        bool col_ok_104_1 = mn_limit > row_base + 35;
                        float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, v_102_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_103_1 * out_cols + h_1])), "f"(v_102_1), "f"(_shfl_xor_99), "r"((unsigned int)(col_ok_104_1 && is_even_lane)) : "memory");
                        float v_105_1 = vals[36] * sscale[tile_stage * 65 + 36];
                        int tok_106_1 = stok[tile_stage * 64 + 36];
                        bool col_ok_107_1 = mn_limit > row_base + 36;
                        float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, v_105_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_106_1 * out_cols + h_1])), "f"(v_105_1), "f"(_shfl_xor_100), "r"((unsigned int)(col_ok_107_1 && is_even_lane)) : "memory");
                        float v_108_1 = vals[37] * sscale[tile_stage * 65 + 37];
                        int tok_109_1 = stok[tile_stage * 64 + 37];
                        bool col_ok_110_1 = mn_limit > row_base + 37;
                        float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, v_108_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_109_1 * out_cols + h_1])), "f"(v_108_1), "f"(_shfl_xor_101), "r"((unsigned int)(col_ok_110_1 && is_even_lane)) : "memory");
                        float v_111_1 = vals[38] * sscale[tile_stage * 65 + 38];
                        int tok_112_1 = stok[tile_stage * 64 + 38];
                        bool col_ok_113_1 = mn_limit > row_base + 38;
                        float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, v_111_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_112_1 * out_cols + h_1])), "f"(v_111_1), "f"(_shfl_xor_102), "r"((unsigned int)(col_ok_113_1 && is_even_lane)) : "memory");
                        float v_114_1 = vals[39] * sscale[tile_stage * 65 + 39];
                        int tok_115_1 = stok[tile_stage * 64 + 39];
                        bool col_ok_116_1 = mn_limit > row_base + 39;
                        float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, v_114_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_115_1 * out_cols + h_1])), "f"(v_114_1), "f"(_shfl_xor_103), "r"((unsigned int)(col_ok_116_1 && is_even_lane)) : "memory");
                        float v_117_1 = vals[40] * sscale[tile_stage * 65 + 40];
                        int tok_118_1 = stok[tile_stage * 64 + 40];
                        bool col_ok_119_1 = mn_limit > row_base + 40;
                        float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, v_117_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_118_1 * out_cols + h_1])), "f"(v_117_1), "f"(_shfl_xor_104), "r"((unsigned int)(col_ok_119_1 && is_even_lane)) : "memory");
                        float v_120_1 = vals[41] * sscale[tile_stage * 65 + 41];
                        int tok_121_1 = stok[tile_stage * 64 + 41];
                        bool col_ok_122_1 = mn_limit > row_base + 41;
                        float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, v_120_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_121_1 * out_cols + h_1])), "f"(v_120_1), "f"(_shfl_xor_105), "r"((unsigned int)(col_ok_122_1 && is_even_lane)) : "memory");
                        float v_123_1 = vals[42] * sscale[tile_stage * 65 + 42];
                        int tok_124_1 = stok[tile_stage * 64 + 42];
                        bool col_ok_125_1 = mn_limit > row_base + 42;
                        float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, v_123_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_124_1 * out_cols + h_1])), "f"(v_123_1), "f"(_shfl_xor_106), "r"((unsigned int)(col_ok_125_1 && is_even_lane)) : "memory");
                        float v_126_1 = vals[43] * sscale[tile_stage * 65 + 43];
                        int tok_127_1 = stok[tile_stage * 64 + 43];
                        bool col_ok_128_1 = mn_limit > row_base + 43;
                        float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, v_126_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_127_1 * out_cols + h_1])), "f"(v_126_1), "f"(_shfl_xor_107), "r"((unsigned int)(col_ok_128_1 && is_even_lane)) : "memory");
                        float v_129_1 = vals[44] * sscale[tile_stage * 65 + 44];
                        int tok_130_1 = stok[tile_stage * 64 + 44];
                        bool col_ok_131_1 = mn_limit > row_base + 44;
                        float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, v_129_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_130_1 * out_cols + h_1])), "f"(v_129_1), "f"(_shfl_xor_108), "r"((unsigned int)(col_ok_131_1 && is_even_lane)) : "memory");
                        float v_132_1 = vals[45] * sscale[tile_stage * 65 + 45];
                        int tok_133_1 = stok[tile_stage * 64 + 45];
                        bool col_ok_134_1 = mn_limit > row_base + 45;
                        float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, v_132_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_133_1 * out_cols + h_1])), "f"(v_132_1), "f"(_shfl_xor_109), "r"((unsigned int)(col_ok_134_1 && is_even_lane)) : "memory");
                        float v_135_1 = vals[46] * sscale[tile_stage * 65 + 46];
                        int tok_136_1 = stok[tile_stage * 64 + 46];
                        bool col_ok_137_1 = mn_limit > row_base + 46;
                        float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, v_135_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_136_1 * out_cols + h_1])), "f"(v_135_1), "f"(_shfl_xor_110), "r"((unsigned int)(col_ok_137_1 && is_even_lane)) : "memory");
                        float v_138_1 = vals[47] * sscale[tile_stage * 65 + 47];
                        int tok_139_1 = stok[tile_stage * 64 + 47];
                        bool col_ok_140_1 = mn_limit > row_base + 47;
                        float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, v_138_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_139_1 * out_cols + h_1])), "f"(v_138_1), "f"(_shfl_xor_111), "r"((unsigned int)(col_ok_140_1 && is_even_lane)) : "memory");
                        float v_141_1 = vals[48] * sscale[tile_stage * 65 + 48];
                        int tok_142_1 = stok[tile_stage * 64 + 48];
                        bool col_ok_143_1 = mn_limit > row_base + 48;
                        float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, v_141_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_142_1 * out_cols + h_1])), "f"(v_141_1), "f"(_shfl_xor_112), "r"((unsigned int)(col_ok_143_1 && is_even_lane)) : "memory");
                        float v_144_1 = vals[49] * sscale[tile_stage * 65 + 49];
                        int tok_145_1 = stok[tile_stage * 64 + 49];
                        bool col_ok_146_1 = mn_limit > row_base + 49;
                        float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, v_144_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_145_1 * out_cols + h_1])), "f"(v_144_1), "f"(_shfl_xor_113), "r"((unsigned int)(col_ok_146_1 && is_even_lane)) : "memory");
                        float v_147_1 = vals[50] * sscale[tile_stage * 65 + 50];
                        int tok_148_1 = stok[tile_stage * 64 + 50];
                        bool col_ok_149_1 = mn_limit > row_base + 50;
                        float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, v_147_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_148_1 * out_cols + h_1])), "f"(v_147_1), "f"(_shfl_xor_114), "r"((unsigned int)(col_ok_149_1 && is_even_lane)) : "memory");
                        float v_150_1 = vals[51] * sscale[tile_stage * 65 + 51];
                        int tok_151_1 = stok[tile_stage * 64 + 51];
                        bool col_ok_152_1 = mn_limit > row_base + 51;
                        float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, v_150_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_151_1 * out_cols + h_1])), "f"(v_150_1), "f"(_shfl_xor_115), "r"((unsigned int)(col_ok_152_1 && is_even_lane)) : "memory");
                        float v_153_1 = vals[52] * sscale[tile_stage * 65 + 52];
                        int tok_154_1 = stok[tile_stage * 64 + 52];
                        bool col_ok_155_1 = mn_limit > row_base + 52;
                        float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, v_153_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_154_1 * out_cols + h_1])), "f"(v_153_1), "f"(_shfl_xor_116), "r"((unsigned int)(col_ok_155_1 && is_even_lane)) : "memory");
                        float v_156_1 = vals[53] * sscale[tile_stage * 65 + 53];
                        int tok_157_1 = stok[tile_stage * 64 + 53];
                        bool col_ok_158_1 = mn_limit > row_base + 53;
                        float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, v_156_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_157_1 * out_cols + h_1])), "f"(v_156_1), "f"(_shfl_xor_117), "r"((unsigned int)(col_ok_158_1 && is_even_lane)) : "memory");
                        float v_159_1 = vals[54] * sscale[tile_stage * 65 + 54];
                        int tok_160_1 = stok[tile_stage * 64 + 54];
                        bool col_ok_161_1 = mn_limit > row_base + 54;
                        float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, v_159_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_160_1 * out_cols + h_1])), "f"(v_159_1), "f"(_shfl_xor_118), "r"((unsigned int)(col_ok_161_1 && is_even_lane)) : "memory");
                        float v_162_1 = vals[55] * sscale[tile_stage * 65 + 55];
                        int tok_163_1 = stok[tile_stage * 64 + 55];
                        bool col_ok_164_1 = mn_limit > row_base + 55;
                        float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, v_162_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_163_1 * out_cols + h_1])), "f"(v_162_1), "f"(_shfl_xor_119), "r"((unsigned int)(col_ok_164_1 && is_even_lane)) : "memory");
                        float v_165_1 = vals[56] * sscale[tile_stage * 65 + 56];
                        int tok_166_1 = stok[tile_stage * 64 + 56];
                        bool col_ok_167_1 = mn_limit > row_base + 56;
                        float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, v_165_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_166_1 * out_cols + h_1])), "f"(v_165_1), "f"(_shfl_xor_120), "r"((unsigned int)(col_ok_167_1 && is_even_lane)) : "memory");
                        float v_168_1 = vals[57] * sscale[tile_stage * 65 + 57];
                        int tok_169_1 = stok[tile_stage * 64 + 57];
                        bool col_ok_170_1 = mn_limit > row_base + 57;
                        float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, v_168_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_169_1 * out_cols + h_1])), "f"(v_168_1), "f"(_shfl_xor_121), "r"((unsigned int)(col_ok_170_1 && is_even_lane)) : "memory");
                        float v_171_1 = vals[58] * sscale[tile_stage * 65 + 58];
                        int tok_172_1 = stok[tile_stage * 64 + 58];
                        bool col_ok_173_1 = mn_limit > row_base + 58;
                        float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, v_171_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_172_1 * out_cols + h_1])), "f"(v_171_1), "f"(_shfl_xor_122), "r"((unsigned int)(col_ok_173_1 && is_even_lane)) : "memory");
                        float v_174_1 = vals[59] * sscale[tile_stage * 65 + 59];
                        int tok_175_1 = stok[tile_stage * 64 + 59];
                        bool col_ok_176_1 = mn_limit > row_base + 59;
                        float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, v_174_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_175_1 * out_cols + h_1])), "f"(v_174_1), "f"(_shfl_xor_123), "r"((unsigned int)(col_ok_176_1 && is_even_lane)) : "memory");
                        float v_177_1 = vals[60] * sscale[tile_stage * 65 + 60];
                        int tok_178_1 = stok[tile_stage * 64 + 60];
                        bool col_ok_179_1 = mn_limit > row_base + 60;
                        float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, v_177_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_178_1 * out_cols + h_1])), "f"(v_177_1), "f"(_shfl_xor_124), "r"((unsigned int)(col_ok_179_1 && is_even_lane)) : "memory");
                        float v_180_1 = vals[61] * sscale[tile_stage * 65 + 61];
                        int tok_181_1 = stok[tile_stage * 64 + 61];
                        bool col_ok_182_1 = mn_limit > row_base + 61;
                        float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, v_180_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_181_1 * out_cols + h_1])), "f"(v_180_1), "f"(_shfl_xor_125), "r"((unsigned int)(col_ok_182_1 && is_even_lane)) : "memory");
                        float v_183_1 = vals[62] * sscale[tile_stage * 65 + 62];
                        int tok_184_1 = stok[tile_stage * 64 + 62];
                        bool col_ok_185_1 = mn_limit > row_base + 62;
                        float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, v_183_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_184_1 * out_cols + h_1])), "f"(v_183_1), "f"(_shfl_xor_126), "r"((unsigned int)(col_ok_185_1 && is_even_lane)) : "memory");
                        float v_186_1 = vals[63] * sscale[tile_stage * 65 + 63];
                        int tok_187_1 = stok[tile_stage * 64 + 63];
                        bool col_ok_188_1 = mn_limit > row_base + 63;
                        float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, v_186_1, 1);
                        asm volatile("{ .reg .pred p_; .reg .b32 pk_; setp.ne.b32 p_, %3, 0; cvt.rn.bf16x2.f32 pk_, %2, %1; @p_ red.global.add.noftz.bf16x2 [%0], pk_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_187_1 * out_cols + h_1])), "f"(v_186_1), "f"(_shfl_xor_127), "r"((unsigned int)(col_ok_188_1 && is_even_lane)) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
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
                rec_m_tile = sinfo[tile_stage * 7];
                meta_alpha = sscale[tile_stage * 65 + 64];
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
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                ab_tok = _mbar_token_0;
                mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                    }
                    int _mma_a_lo_0 = (_mma_base_lo_0) + (sa * 2) * 1024;
                    int _mma_b_lo_0 = (_mma_base_lo_1) + (sb) * 512;
                    {
                        uint32_t a_desc_lo = (uint32_t)_mma_a_lo_0;
                        uint32_t b_desc_lo = (uint32_t)_mma_b_lo_0;

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                            0x8900280U, tmem_sf_a, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                            0x28900290U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                            0x489002a0U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                            0x689002b0U, tmem_sf_a, tmem_sf_b, 1);
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                    }
                    int _mma_a_lo_1 = (_mma_base_lo_0) + (sa * 2 + 1) * 1024;
                    int _mma_b_lo_1 = (_mma_base_lo_1) + (sb) * 512;
                    {
                        uint32_t a_desc_lo = (uint32_t)_mma_a_lo_1;
                        uint32_t b_desc_lo = (uint32_t)_mma_b_lo_1;

                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                            0x8900280U, tmem_sf_a1, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                            0x28900290U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                            0x489002a0U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                            0x689002b0U, tmem_sf_a1, tmem_sf_b, 1);
                    }
                }
                elect_commit(ab_free_addr + (sa) * 8);
                elect_commit(b_free_addr + (sb) * 8);
                sa += 1;
                if (sa == 4) { sa = 0; pha ^= 1; }
                sb += 1;
                if (sb == 4) { sb = 0; _phase_b_full ^= 1; }
                nst_mma = nst_mma + 1;
                ab_tok = 1;
                if (k_tiles > 1) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                    ab_tok = _mbar_token_1;
                }
                #pragma unroll 1
                for (int k = 1; k < k_tiles; k++) {
                    mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                    mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                        }
                        int _mma_a_lo_2 = (_mma_base_lo_0) + (sa * 2) * 1024;
                        int _mma_b_lo_2 = (_mma_base_lo_1) + (sb) * 512;
                        {
                            uint32_t a_desc_lo = (uint32_t)_mma_a_lo_2;
                            uint32_t b_desc_lo = (uint32_t)_mma_b_lo_2;

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                0x8900280U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                0x28900290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                0x489002a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                0x689002b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                        }
                        int _mma_a_lo_3 = (_mma_base_lo_0) + (sa * 2 + 1) * 1024;
                        int _mma_b_lo_3 = (_mma_base_lo_1) + (sb) * 512;
                        {
                            uint32_t a_desc_lo = (uint32_t)_mma_a_lo_3;
                            uint32_t b_desc_lo = (uint32_t)_mma_b_lo_3;

                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 0u) | ((uint64_t)0x40004040 << 32)),
                                0x8900280U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x40004040 << 32)),
                                0x28900290U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 4u) | ((uint64_t)0x40004040 << 32)),
                                0x489002a0U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 64)), ((uint64_t)(a_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)), ((uint64_t)(b_desc_lo + 6u) | ((uint64_t)0x40004040 << 32)),
                                0x689002b0U, tmem_sf_a1, tmem_sf_b, 1);
                        }
                    }
                    elect_commit(ab_free_addr + (sa) * 8);
                    elect_commit(b_free_addr + (sb) * 8);
                    sa += 1;
                    if (sa == 4) { sa = 0; pha ^= 1; }
                    sb += 1;
                    if (sb == 4) { sb = 0; _phase_b_full ^= 1; }
                    nst_mma = nst_mma + 1;
                    ab_tok = 1;
                    if (k + 1 < k_tiles) {
                        uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                        ab_tok = _mbar_token_2;
                    }
                }
                elect_commit(acc_full_addr + (acc_stage_1) * 8);
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                nst_mma = nst_mma + 1;
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
            int batch[2];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                int _min_0 = ((info_2[0] * 2) < (num_m_tiles - 1) ? (info_2[0] * 2) : (num_m_tiles - 1));
                batch[0] = info_2[2] * num_m_tiles + _min_0;
                int _min_1 = ((info_2[0] * 2 + 1) < (num_m_tiles - 1) ? (info_2[0] * 2 + 1) : (num_m_tiles - 1));
                batch[1] = info_2[2] * num_m_tiles + _min_1;
                int row_base_tma = info_2[1] * 64;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 25600);
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + stage * 2 * 16384), "l"((&A)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + stage * 2 * 512), "l"((&SFA)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + (stage * 2 + 1) * 16384), "l"((&A)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + (stage * 2 + 1) * 512), "l"((&SFA)), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            tma_2d_gmem2smem(b_addr + stage * 8192, (&B), (info_2[5] + k_1) * 128, row_base_tma, ab_full_addr + (stage) * 8);
                        }
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_ab_free ^= 1; }
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
            int sched_first = bid;
            int sched_step = num_bids;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int item = sched_first; item < total_items; item += sched_step) {
                int row_group = item / m_chunks;
                int m_tile = item - row_group * m_chunks;
                if (row_group >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                int sched_row_group = row_group;
                int lookup = row_group;
                int lookup_limit = row_group;
                int expert = tile_idx_to_expert_idx[lookup];
                int mn_limit_1 = tile_idx_to_mn_limit[lookup_limit];
                if (lane == 0) {
                    sscale[tile_stage_3 * 65 + 64] = alpha[expert];
                }
                int meta_col = lane;
                if (meta_col < 64) {
                    int meta_prow = sched_row_group * 64 + meta_col;
                    int meta_valid = (int)(meta_prow < mn_limit_1);
                    int meta_expanded = permuted_idx_to_expanded_idx[meta_prow];
                    int _max_0 = ((meta_expanded) > (0) ? (meta_expanded) : (0));
                    int meta_safe = _max_0;
                    int meta_token = meta_safe / top_k;
                    int meta_topk = meta_safe - meta_token * top_k;
                    int meta_gather = meta_token * meta_valid;
                    sscale[tile_stage_3 * 65 + (unsigned int)meta_col] = token_final_scales[meta_gather * top_k + meta_topk];
                    stok[tile_stage_3 * 64 + (unsigned int)meta_col] = meta_token;
                }
                int meta_col_0 = 32 + lane;
                if (meta_col_0 < 64) {
                    int meta_prow_1 = sched_row_group * 64 + meta_col_0;
                    int meta_valid_1 = (int)(meta_prow_1 < mn_limit_1);
                    int meta_expanded_1 = permuted_idx_to_expanded_idx[meta_prow_1];
                    int _max_1 = ((meta_expanded_1) > (0) ? (meta_expanded_1) : (0));
                    int meta_safe_1 = _max_1;
                    int meta_token_1 = meta_safe_1 / top_k;
                    int meta_topk_1 = meta_safe_1 - meta_token_1 * top_k;
                    int meta_gather_1 = meta_token_1 * meta_valid_1;
                    sscale[tile_stage_3 * 65 + (unsigned int)meta_col_0] = token_final_scales[meta_gather_1 * top_k + meta_topk_1];
                    stok[tile_stage_3 * 64 + (unsigned int)meta_col_0] = meta_token_1;
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
    if (warp == 7) {
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
            int row_src[1];
            int row_ok[1];
            int sf_src[2];
            int sf_ok[2];
            int cta_row0 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 64;
                int mn_limit_2 = info_3[4];
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int sok = (int)(sprow < mn_limit_2 && srow < 64);
                sf_src[0] = sprow * sok;
                sf_ok[0] = sok;
                int srow_0 = (gather_sub + 1) * 32 + lane_0_1;
                int sprow_1 = row_base_1 + srow_0;
                int sok_2 = (int)(sprow_1 < mn_limit_2 && srow_0 < 64);
                sf_src[1] = sprow_1 * sok_2;
                sf_ok[1] = sok_2;
                #pragma unroll 1
                for (int k_2 = 0; k_2 < k_tiles; k_2++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    int k0 = (info_3[5] + k_2) * 128;
                    int sf_src_off = sf_src[0] * sf_cols + (info_3[5] + k_2) * 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 512 + (unsigned int)(gather_sub / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    int sf_src_off_0 = sf_src[1] * sf_cols + (info_3[5] + k_2) * 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 512 + (unsigned int)((gather_sub + 1) / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)((gather_sub + 1) % 4 * 4)), "l"(SFB + sf_src_off_0), "r"((sf_ok[1] != 0) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_b_free ^= 1; }
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

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
