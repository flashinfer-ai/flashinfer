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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 144
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 128
#define TMEM_TMEM_SFB_OFFSET 136
#define NUM_TMA_PIPE_STAGES 11
#define NUM_ACC_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 18432
#define SMEM_SMEM_B_OFF 9216
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 18432
#define SMEM_SMEM_SFA_OFF 17408
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 18432
#define SMEM_SMEM_SFB_OFF 18432
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 18432
#define SMEM_SMEM_D_OFF 1024
#define SMEM_SMEM_D_STAGE_BYTES 8192
#define SMEM_SMEM_D_STRIDE 18432
#define SMEM_SMEM_L1_OFF 9216
#define SMEM_SMEM_L1_STAGE_BYTES 8192
#define SMEM_SMEM_L1_STRIDE 18432
#define SMEM_SMEM_OUT_OFF 203776
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_BIAS_PAIRS_OFF 220160
#define SMEM_SMEM_BIAS_PAIRS_STAGE_BYTES 256
#define SMEM_SMEM_BIAS_PAIRS_STRIDE 256
#define SMEM_TOTAL 220416
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}





__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
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



__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_nvfp4_svdquant_gemm_ffc7cbc225f9c3e46796(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, const __grid_constant__ CUtensorMap L1, float* __restrict__ alpha, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int rank_tiles, int grid_n, int has_bias)
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
    #define tma_full_addr (mbar_base + 0)
    #define tma_empty_addr (mbar_base + 88)
    #define acc_full_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_b_addr = smem + 9216;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_sfa_addr = smem + 17408;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 18432);
    const int smem_sfb_addr = smem + 18432;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_d_addr = smem + 1024;
    __nv_bfloat16* smem_l1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_l1_addr = smem + 9216;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 203776);
    const int smem_out_addr = smem + 203776;
    unsigned int* smem_bias_pairs = reinterpret_cast<unsigned int*>(smem_raw + 220160);
    const int smem_bias_pairs_addr = smem + 220160;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 11 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // tma_empty: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 144 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 0) {
        int _tmem_hold = smem + 184;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 128;
    const int tmem_tmem_sfb = taddr + 136;

    // ---- Role: mma ----
    if (warp == 0) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int k_tile_mma = 0; k_tile_mma < K_tiles; k_tile_mma++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (mma_stage) * 1152)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (mma_stage) * 1152 + 8)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (mma_stage) * 1152)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (mma_stage) * 1152 + 8)));
                }
                int init_flag = ((k_tile_mma == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8200480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 4 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                    }
                }
                elect_commit(tma_empty_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 11) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            #pragma unroll 1
            for (unsigned int rank_tile_mma = 0; rank_tile_mma < rank_tiles; rank_tile_mma++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int _mma_a_lo_2 = make_warp_uniform((((smem_d_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                int _mma_b_lo_2 = make_warp_uniform((((smem_l1_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x80004020U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                    }
                }
                elect_commit(tma_empty_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 11) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(acc_full_addr);
        }
    }
    // ---- Role: mainloop_prefetch ----
    if (warp == 1) {
        { // mainloop_prefetch_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1))) : "memory");
            }
        }
    }
    // ---- Role: load ----
    if (warp == 2) {
        { // load_main
            unsigned int load_stage = 0;
            int bid_m = bid / grid_n;
            int bid_n = bid - bid_m * grid_n;
            int off_m = bid_m * 128;
            int off_n = bid_n * 128;
            unsigned int _phase_tma_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 18432);
                    tma_3d_gmem2smem(smem_a_addr + load_stage * 18432, (&A), 0, off_m, k_tile, tma_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(smem_b_addr + load_stage * 18432, (&B), 0, off_n, k_tile, tma_full_addr + (load_stage) * 8);
                    tma_5d_gmem2smem(smem_sfa_addr + load_stage * 18432, (&SFA), 0, 0, 2 * k_tile, 0, bid_m, tma_full_addr + (load_stage) * 8);
                    tma_5d_gmem2smem(smem_sfb_addr + load_stage * 18432, (&SFB), 0, 0, 2 * k_tile, 0, bid_n, tma_full_addr + (load_stage) * 8);
                    load_stage += 1;
                    if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
                #pragma unroll 1
                for (unsigned int rank_tile = 0; rank_tile < rank_tiles; rank_tile++) {
                    int rank_offset = rank_tile * 32;
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 18432);
                    tma_2d_gmem2smem(smem_d_addr + load_stage * 18432, (&D), rank_offset, off_m, tma_full_addr + (load_stage) * 8);
                    tma_2d_gmem2smem(smem_l1_addr + load_stage * 18432, (&L1), rank_offset, off_n, tma_full_addr + (load_stage) * 8);
                    tma_5d_gmem2smem(smem_sfa_addr + load_stage * 18432, (&SFA), 0, 0, 0, 0, bid_m, tma_full_addr + (load_stage) * 8);
                    tma_5d_gmem2smem(smem_sfb_addr + load_stage * 18432, (&SFB), 0, 0, 0, 0, bid_n, tma_full_addr + (load_stage) * 8);
                    load_stage += 1;
                    if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: output_prefetch ----
    if (warp == 3) {
        { // output_prefetch_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory");
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int bid_m_1 = bid / grid_n;
            int bid_n_1 = bid - bid_m_1 * grid_n;
            int off_m_1 = bid_m_1 * 128;
            int off_n_1 = bid_n_1 * 128;
            int epi_tid = epi_warp * 32 + lane;
            if (epi_tid < 64) {
                unsigned int staged_bias_pair = 0;
                if (has_bias != 0) {
                    unsigned int _vec_load_0[2];
                    {
                        uint32_t _bf16x2_bits_0;
                        _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(bias + off_n_1 + epi_tid * 2);
                        _vec_load_0[0] = _bf16x2_bits_0;
                    }
                    staged_bias_pair = _vec_load_0[0];
                }
                smem_bias_pairs[epi_tid] = staged_bias_pair;
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            unsigned int _phase_acc_full_0 = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full_0);
            _phase_acc_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float alpha_value = alpha[0];
            unsigned int store_stage = 0;
            #pragma unroll
            for (int subtile = 0; subtile < 4; subtile++) {
                int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(subtile * 32);
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_addr));
                unsigned int bias_pairs[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[0])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 3]))
                    : "r"(smem_bias_pairs_addr + (unsigned int)(subtile * 16 * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[4])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 3]))
                    : "r"(smem_bias_pairs_addr + (unsigned int)((subtile * 16 + 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[8])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 3]))
                    : "r"(smem_bias_pairs_addr + (unsigned int)((subtile * 16 + 8) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[12])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 3]))
                    : "r"(smem_bias_pairs_addr + (unsigned int)((subtile * 16 + 12) * 4)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float result[32];
                unsigned int bias_pair = 0;
                unsigned int bias_lo_bits = 0;
                float bias_lo = 0.0f;
                unsigned int bias_hi_bits = 0;
                float bias_hi = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    bias_pair = bias_pairs[j];
                    bias_lo_bits = bias_pair << 16;
                    bias_lo = reinterpret_cast<float*>(&bias_lo_bits)[0];
                    bias_hi_bits = bias_pair & 4294901760u;
                    bias_hi = reinterpret_cast<float*>(&bias_hi_bits)[0];
                    result[2 * j] = _tmem_load_0[2 * j] * alpha_value + bias_lo;
                    result[2 * j + 1] = _tmem_load_0[2 * j + 1] * alpha_value + bias_hi;
                }
                uint32_t result_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                    result_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (subtile > 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_2d((&out), off_n_1 + (subtile - 1) * 32, off_m_1, smem_out_addr + store_stage * 8192);
                        }
                    }
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    store_stage = store_stage ^ 1;
                }
                int out_stage_row = store_stage * 128 + (unsigned int)(epi_warp * 32) + (unsigned int)lane;
                unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                unsigned int out_swz = out_abs / 8 & 48;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(result_bf16[0]), "r"(result_bf16[1]), "r"(result_bf16[2]), "r"(result_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(result_bf16[4]), "r"(result_bf16[5]), "r"(result_bf16[6]), "r"(result_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(result_bf16[8]), "r"(result_bf16[9]), "r"(result_bf16[10]), "r"(result_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(result_bf16[12]), "r"(result_bf16[13]), "r"(result_bf16[14]), "r"(result_bf16[15]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    tma_store_2d((&out), off_n_1 + 96, off_m_1, smem_out_addr + store_stage * 8192);
                }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 1;");
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
