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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 336
#define TMEM_REM_ACC_OFFSET 416
#define TMEM_REM_SF_OFFSET 432
#define NUM_TMA_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 36864
#define SMEM_SMEM_V1_OFF 33792
#define SMEM_SMEM_V1_STAGE_BYTES 2048
#define SMEM_SMEM_V1_STRIDE 36864
#define SMEM_SMEM_V2_OFF 17408
#define SMEM_SMEM_V2_STAGE_BYTES 16384
#define SMEM_SMEM_V2_STRIDE 36864
#define SMEM_SMEM_V3_OFF 35840
#define SMEM_SMEM_V3_STAGE_BYTES 2048
#define SMEM_SMEM_V3_STRIDE 36864
#define SMEM_SMEM_OUT_OFF 185344
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_REM_B_OFF 218112
#define SMEM_REM_B_STAGE_BYTES 1024
#define SMEM_REM_B_STRIDE 1024
#define SMEM_REM_SF_OFF 223232
#define SMEM_REM_SF_STAGE_BYTES 640
#define SMEM_REM_SF_STRIDE 640
#define SMEM_TOTAL 226432
#define THREADS 256
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
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





// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_per_token_045ed4b8200220c8d8cd(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, const __grid_constant__ CUtensorMap A_rem, const __grid_constant__ CUtensorMap SFA_rem, const __grid_constant__ CUtensorMap out_rem, int M, int N, int K_tiles, int tok_tiles, int num_tiles)
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
    #define tma_empty_addr (mbar_base + 40)
    #define acc_full_addr (mbar_base + 80)
    #define acc_empty_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_v0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    uint8_t* smem_v1 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v1_addr = smem + 33792;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr = smem + 17408;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v3_addr = smem + 35840;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 185344);
    const int smem_out_addr = smem + 185344;
    uint8_t* rem_b = reinterpret_cast<uint8_t*>(smem_raw + 218112);
    const int rem_b_addr = smem + 218112;
    uint8_t* rem_sf = reinterpret_cast<uint8_t*>(smem_raw + 223232);
    const int rem_sf_addr = smem + 223232;
    int rem_row0 = tok_tiles * 128;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_rem))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA_rem))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out_rem))) : "memory"); }

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // tma_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 2) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 336;
    const int tmem_rem_acc = taddr + 416;
    const int tmem_rem_sf = taddr + 432;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int tiles_end = num_tiles;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_tma_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < tiles_end; tile += num_bids) {
                int tile_c = (int)tile;
                int w_idx = tile_c / tok_tiles;
                int tok_idx = tile_c - w_idx * tok_tiles;
                int a_rows = tok_idx * 128;
                int b_rows = w_idx * 128;
                int a_atom = a_rows / 128;
                int b_atom = b_rows / 128;
                int b_in_atom = b_rows - b_atom * 128;
                int sfb_g4 = b_in_atom % 32 / 8;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 38400);
                        tma_3d_gmem2smem(smem_v0_addr + load_stage * 36864, (&A), 0, a_rows, (int)k_tile, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_v1_addr + load_stage * 36864, (&SFA), 0, 4 * (int)k_tile, a_atom, tma_full_addr + (load_stage) * 8);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v2_addr + load_stage * 36864), "l"((&B)), "r"(0), "r"(b_rows), "r"((int)k_tile),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v3_addr + load_stage * 36864), "l"((&SFB)), "r"(0), "r"(4 * (int)k_tile), "r"(b_atom + b_in_atom / 128),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem(rem_b_addr + load_stage * 1024, (&A_rem), 0, rem_row0, (int)k_tile, tma_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(rem_sf_addr + load_stage * 640, (&SFA_rem), 0, 4 * (int)k_tile, rem_row0 % 32 / 8, rem_row0 / 128, tma_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
            #pragma unroll
            for (int _tail = 0; _tail < 5; _tail++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                load_stage += 1;
                if (load_stage == 5) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            int tiles_end_1 = num_tiles;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < tiles_end_1; tile_1 += num_bids) {
                int tile_c_1 = (int)tile_1;
                int w_idx_1 = tile_c_1 / tok_tiles;
                int tok_idx_1 = tile_c_1 - w_idx_1 * tok_tiles;
                int b_rows_1 = w_idx_1 * 128;
                int sfb_p = b_rows_1 % 128 / 32;
                int acc_base = (int)acc_stage * 128;
                mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_col = (int)mma_stage * 16;
                    int sfb_col = (int)mma_stage * 16;
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfa + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 2304)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 2304 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 2304 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 2304 + 96)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfb + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 2304)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 2304 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 2304 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 2304 + 96)));
                    }
                    uint32_t _elected_0 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(rem_sf_addr + mma_stage * 640 + 0)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_rem_sf + ((int)mma_stage * 16 + 0))), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(rem_sf_addr + mma_stage * 640 + 128)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_rem_sf + ((int)mma_stage * 16 + 4))), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(rem_sf_addr + mma_stage * 640 + 256)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_rem_sf + ((int)mma_stage * 16 + 8))), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(rem_sf_addr + mma_stage * 640 + 384)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(tmem_rem_sf + ((int)mma_stage * 16 + 12))), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                            : "memory");
                    }
                    int init_flag = ((k_tile_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_v0_addr) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_v2_addr) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + b_rows_1 % 128 / 32) + 0, ((init_flag) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_v0_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_v2_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 4 + b_rows_1 % 128 / 32) + 0, 1);
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_v0_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_v2_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 8 + b_rows_1 % 128 / 32) + 0, 1);
                        }
                    }
                    int _mma_a_lo_3 = make_warp_uniform((((smem_v0_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_v2_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2304);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8200480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 12 + b_rows_1 % 128 / 32) + 0, 1);
                        }
                    }
                    int _rem_b_lo_0 = make_warp_uniform((((rem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 64);
                    int _rem_b_lo_1 = make_warp_uniform((((rem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 64);
                    int _rem_b_lo_2 = make_warp_uniform((((rem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 64);
                    int _rem_b_lo_3 = make_warp_uniform((((rem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 64);
                    uint32_t _elected_5 = elect_sync();
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_rem_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_leader((tmem_rem_acc + ((int)acc_stage * 8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_tmem_sfb + (sfb_col + b_rows_1 % 128 / 32) + 0, tmem_rem_sf + ((int)mma_stage * 16 + 0) + 0, ((init_flag) ? 0 : 1), _elected_5);
                    }
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_rem_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_leader((tmem_rem_acc + ((int)acc_stage * 8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_tmem_sfb + (sfb_col + 4 + b_rows_1 % 128 / 32) + 0, tmem_rem_sf + ((int)mma_stage * 16 + 4) + 0, 1, _elected_5);
                    }
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_rem_b_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_leader((tmem_rem_acc + ((int)acc_stage * 8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_tmem_sfb + (sfb_col + 8 + b_rows_1 % 128 / 32) + 0, tmem_rem_sf + ((int)mma_stage * 16 + 8) + 0, 1, _elected_5);
                    }
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_rem_b_lo_3) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs_leader((tmem_rem_acc + ((int)acc_stage * 8)), a_desc + 0, b_desc + 0,
                            0x8020480U, tmem_tmem_sfb + (sfb_col + 12 + b_rows_1 % 128 / 32) + 0, tmem_rem_sf + ((int)mma_stage * 16 + 12) + 0, 1, _elected_5);
                    }
                    elect_commit(tma_empty_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(acc_full_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
                completed += 1;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            if (completed > 0) {
                unsigned int _stage_acc_empty = (completed + 1) % 2;
                unsigned int _parity_acc_empty = (completed + 1) / 2 % 2 ^ 1;
                mbarrier_wait(acc_empty_addr + (_stage_acc_empty) * 8, _parity_acc_empty);
            }
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: prefetch ----
    if (warp == 2) {
        { // prefetch_main
            if (elect_sync()) {
                int ep_tile = (int)bid;
                if (ep_tile < num_tiles) {
                    int ep_w = ep_tile / tok_tiles;
                    int ep_rows = ep_w * 128;
                    int ep_atom = ep_rows / 128;
                    if (K_tiles > 0) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(ep_rows)), "r"((int)(0)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&SFB))), "r"((int)(0)), "r"((int)(0)), "r"((int)(ep_atom + (ep_rows - ep_atom * 128) / 128)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (K_tiles > 1) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(ep_rows)), "r"((int)(1)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&SFB))), "r"((int)(0)), "r"((int)(4)), "r"((int)(ep_atom + (ep_rows - ep_atom * 128) / 128)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (K_tiles > 2) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(ep_rows)), "r"((int)(2)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&SFB))), "r"((int)(0)), "r"((int)(8)), "r"((int)(ep_atom + (ep_rows - ep_atom * 128) / 128)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (K_tiles > 3) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&B))), "r"((int)(0)), "r"((int)(ep_rows)), "r"((int)(3)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;" :: "l"((uint64_t)((&SFB))), "r"((int)(0)), "r"((int)(12)), "r"((int)(ep_atom + (ep_rows - ep_atom * 128) / 128)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp * 32 + lane;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            int tiles_end_2 = num_tiles;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < tiles_end_2; tile_2 += num_bids) {
                int tile_c_2 = (int)tile_2;
                int w_idx_2 = tile_c_2 / tok_tiles;
                int tok_idx_2 = tile_c_2 - w_idx_2 * tok_tiles;
                int off_tok = tok_idx_2 * 128;
                int off_w = w_idx_2 * 128;
                int acc_base_1 = (int)acc_stage_1 * 128;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_1;
                float alpha_row = 0.0f;
                float alphas[128];
                float frag_alpha[2];
                int n_col = off_w + epi_row;
                long long row_ptr = (long long)off_tok * (long long)N + (long long)n_col;
                long long n_stride = N;
                int n_rows = ((128) < (M - off_tok) ? (128) : (M - off_tok));
                int tok_row = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                alpha_row = alpha[tok_row];
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int rem_lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + 416 + acc_stage_1 * 8;
                float _rem_ld_0[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_0[3]))
                    : "r"(rem_lane_addr));
                float _rem_ld_1[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_rem_ld_1[3]))
                    : "r"(rem_lane_addr + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int subtile = 0; subtile < 4; subtile++) {
                    int tmem_addr = lane_addr + subtile * 32;
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (subtile == 3) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(acc_empty_addr + (acc_stage_1) * 8);
                    }
                    {
                        float2 _pair_scale_even2_0 = make_float2(alpha_row, alpha_row);
                        float2 _pair_scale_odd2_0 = make_float2(alpha_row, alpha_row);
                        float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&_tmem_load_0[0]);
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                    }
                    unsigned int _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (subtile > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                if ((unsigned int)tile_c_2 == tile_2) {
                                    tma_store_2d((&out), off_w + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192);
                                }
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 3;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        store_stage = store_stage + 1;
                        if (store_stage == 4) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_row;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                    unsigned int out_swz = out_abs / 8 & 48;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        if ((unsigned int)tile_c_2 == tile_2) {
                            tma_store_2d((&out), off_w + 96, off_tok, smem_out_addr + store_stage * 8192);
                        }
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 3;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 4) {
                    store_stage = 0;
                }
                int lane_pair = lane % 4 * 2;
                int tok_e = ((rem_row0 + lane_pair) < (M - 1) ? (rem_row0 + lane_pair) : (M - 1));
                int tok_o = ((rem_row0 + lane_pair + 1) < (M - 1) ? (rem_row0 + lane_pair + 1) : (M - 1));
                float rem_alpha_e = alpha[tok_e];
                float rem_alpha_o = alpha[tok_o];
                _rem_ld_0[0] = _rem_ld_0[0] * rem_alpha_e;
                _rem_ld_0[1] = _rem_ld_0[1] * rem_alpha_o;
                _rem_ld_0[2] = _rem_ld_0[2] * rem_alpha_e;
                _rem_ld_0[3] = _rem_ld_0[3] * rem_alpha_o;
                _rem_ld_1[0] = _rem_ld_1[0] * rem_alpha_e;
                _rem_ld_1[1] = _rem_ld_1[1] * rem_alpha_o;
                _rem_ld_1[2] = _rem_ld_1[2] * rem_alpha_e;
                _rem_ld_1[3] = _rem_ld_1[3] * rem_alpha_o;
                unsigned int _rem_pk_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_rem_ld_0[_lp*2 + 0], _rem_ld_0[_lp*2+1 + 0]));
                    _rem_pk_0[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int _rem_pk_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_rem_ld_1[_lp*2 + 0], _rem_ld_1[_lp*2+1 + 0]));
                    _rem_pk_1[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int st_row = lane % 8;
                unsigned int st_col = epi_warp % 2 * 4 + lane / 8;
                unsigned int write_base = smem_out_addr + store_stage * 8192 + (unsigned int)(epi_warp / 2 * 1024) + st_row * 128 + (st_col ^ st_row) * 16;
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(write_base);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&_rem_pk_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_rem_pk_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_rem_pk_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_rem_pk_1[1]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        if ((unsigned int)tile_c_2 == tile_2) {
                            tma_store_2d((&out_rem), off_w, rem_row0, smem_out_addr + store_stage * 8192);
                            if (off_w + 64 < N) {
                                tma_store_2d((&out_rem), off_w + 64, rem_row0, smem_out_addr + store_stage * 8192 + 1024);
                            }
                        }
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 3;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 4) {
                    store_stage = 0;
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
