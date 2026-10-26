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
#define TMEM_PARTIALS_OFFSET 0
#define NUM_AB_PIPE_STAGES 5
#define NUM_PARTIAL_PIPE_STAGES 4
#define NUM_SCALE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 82944
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_0_OFF 164864
#define SMEM_EPI_STAGING_0_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_0_STRIDE 8192
#define SMEM_EPI_STAGING_1_OFF 173056
#define SMEM_EPI_STAGING_1_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_1_STRIDE 8192
#define SMEM_EPI_STAGING_2_OFF 181248
#define SMEM_EPI_STAGING_2_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_2_STRIDE 8192
#define SMEM_EPI_STAGING_3_OFF 189440
#define SMEM_EPI_STAGING_3_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_3_STRIDE 8192
#define SMEM_SMEM_ASCALE_OFF 197632
#define SMEM_SMEM_ASCALE_STAGE_BYTES 1536
#define SMEM_SMEM_ASCALE_STRIDE 1536
#define SMEM_SMEM_BSCALE_OFF 199168
#define SMEM_SMEM_BSCALE_STAGE_BYTES 12
#define SMEM_SMEM_BSCALE_STRIDE 12
#define SMEM_EA_START_OFF 199184
#define SMEM_EA_START_STAGE_BYTES 1088
#define SMEM_EA_START_STRIDE 1088
#define SMEM_EA_FIRST_OFF 200272
#define SMEM_EA_FIRST_STAGE_BYTES 1024
#define SMEM_EA_FIRST_STRIDE 1024
#define SMEM_EA_SLOT_ROW0_OFF 201296
#define SMEM_EA_SLOT_ROW0_STAGE_BYTES 2048
#define SMEM_EA_SLOT_ROW0_STRIDE 2048
#define SMEM_EA_SLOT_ER_OFF 203344
#define SMEM_EA_SLOT_ER_STAGE_BYTES 2048
#define SMEM_EA_SLOT_ER_STRIDE 2048
#define SMEM_EA_META_OFF 205392
#define SMEM_EA_META_STAGE_BYTES 16
#define SMEM_EA_META_STRIDE 16
#define SMEM_TOTAL 205440
#define THREADS 384
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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



__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_grouped_fp8_gemm_ff32d0570617af1d5c2e(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ a_scale, float* __restrict__ b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
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
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 40)
    #define partial_full_addr (mbar_base + 80)
    #define partial_free_addr (mbar_base + 112)
    #define scale_full_addr (mbar_base + 144)
    #define scale_free_addr (mbar_base + 168)
    #define scan_done_addr (mbar_base + 192)
    #define table_full_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 82944);
    const int smem_b_addr = smem + 82944;
    __nv_bfloat16* epi_staging_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int epi_staging_0_addr = smem + 164864;
    __nv_bfloat16* epi_staging_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int epi_staging_1_addr = smem + 173056;
    __nv_bfloat16* epi_staging_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int epi_staging_2_addr = smem + 181248;
    __nv_bfloat16* epi_staging_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int epi_staging_3_addr = smem + 189440;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 197632);
    const int smem_ascale_addr = smem + 197632;
    float* smem_bscale = reinterpret_cast<float*>(smem_raw + 199168);
    const int smem_bscale_addr = smem + 199168;
    int* ea_start = reinterpret_cast<int*>(smem_raw + 199184);
    const int ea_start_addr = smem + 199184;
    int* ea_first = reinterpret_cast<int*>(smem_raw + 200272);
    const int ea_first_addr = smem + 200272;
    int* ea_slot_row0 = reinterpret_cast<int*>(smem_raw + 201296);
    const int ea_slot_row0_addr = smem + 201296;
    int* ea_slot_er = reinterpret_cast<int*>(smem_raw + 203344);
    const int ea_slot_er_addr = smem + 203344;
    int* ea_meta = reinterpret_cast<int*>(smem_raw + 205392);
    const int ea_meta_addr = smem + 205392;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    int taddr;
    int tmem_partials;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // ab_free: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'partial_pipe' ---
            // partial_full: 4 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // partial_free: 4 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 3 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            // scale_free: 3 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            // scan_done: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // table_full: 1 barriers, init_count=32
            mbarrier_init(smem + 200, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: acc_epi ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // acc_epi_main
            int lane_0 = (int)lane;
            int warp_1 = (int)warp;
            int tid_1 = warp_1 * 32 + lane_0;
            ea_start[tid_1] = M;
            if (tid_1 == 0) {
                ea_start[256] = M;
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            int scan_rounds = (M + 8192 - 1) / 8192;
            #pragma unroll 1
            for (int scan_round = 0; scan_round < scan_rounds; scan_round++) {
                int scan_r0 = ((scan_round * 8 + warp_1) * 32 + lane_0) * 32;
                if (scan_r0 < M) {
                    int scan_vals[32];
                    int scan_prev = -1;
                    if (scan_r0 > 0) {
                        scan_prev = m_indices[scan_r0 - 1];
                    }
                    if (scan_r0 + 32 <= M) {
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(m_indices + scan_r0 + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            scan_vals[0 + 0] = _ivld_0.x;
                            scan_vals[0 + 1] = _ivld_0.y;
                            scan_vals[0 + 2] = _ivld_0.z;
                            scan_vals[0 + 3] = _ivld_0.w;
                        }
                        {
                            const int4* _ivptr_1 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 4) + 0);
                            int4 _ivld_1;
                            _ivld_1 = *_ivptr_1;
                            scan_vals[4 + 0] = _ivld_1.x;
                            scan_vals[4 + 1] = _ivld_1.y;
                            scan_vals[4 + 2] = _ivld_1.z;
                            scan_vals[4 + 3] = _ivld_1.w;
                        }
                        {
                            const int4* _ivptr_2 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 8) + 0);
                            int4 _ivld_2;
                            _ivld_2 = *_ivptr_2;
                            scan_vals[8 + 0] = _ivld_2.x;
                            scan_vals[8 + 1] = _ivld_2.y;
                            scan_vals[8 + 2] = _ivld_2.z;
                            scan_vals[8 + 3] = _ivld_2.w;
                        }
                        {
                            const int4* _ivptr_3 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 12) + 0);
                            int4 _ivld_3;
                            _ivld_3 = *_ivptr_3;
                            scan_vals[12 + 0] = _ivld_3.x;
                            scan_vals[12 + 1] = _ivld_3.y;
                            scan_vals[12 + 2] = _ivld_3.z;
                            scan_vals[12 + 3] = _ivld_3.w;
                        }
                        {
                            const int4* _ivptr_4 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 16) + 0);
                            int4 _ivld_4;
                            _ivld_4 = *_ivptr_4;
                            scan_vals[16 + 0] = _ivld_4.x;
                            scan_vals[16 + 1] = _ivld_4.y;
                            scan_vals[16 + 2] = _ivld_4.z;
                            scan_vals[16 + 3] = _ivld_4.w;
                        }
                        {
                            const int4* _ivptr_5 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 20) + 0);
                            int4 _ivld_5;
                            _ivld_5 = *_ivptr_5;
                            scan_vals[20 + 0] = _ivld_5.x;
                            scan_vals[20 + 1] = _ivld_5.y;
                            scan_vals[20 + 2] = _ivld_5.z;
                            scan_vals[20 + 3] = _ivld_5.w;
                        }
                        {
                            const int4* _ivptr_6 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 24) + 0);
                            int4 _ivld_6;
                            _ivld_6 = *_ivptr_6;
                            scan_vals[24 + 0] = _ivld_6.x;
                            scan_vals[24 + 1] = _ivld_6.y;
                            scan_vals[24 + 2] = _ivld_6.z;
                            scan_vals[24 + 3] = _ivld_6.w;
                        }
                        {
                            const int4* _ivptr_7 = reinterpret_cast<const int4*>(m_indices + (scan_r0 + 28) + 0);
                            int4 _ivld_7;
                            _ivld_7 = *_ivptr_7;
                            scan_vals[28 + 0] = _ivld_7.x;
                            scan_vals[28 + 1] = _ivld_7.y;
                            scan_vals[28 + 2] = _ivld_7.z;
                            scan_vals[28 + 3] = _ivld_7.w;
                        }
                    } else {
                        scan_vals[0] = -1;
                        if (scan_r0 < M) {
                            scan_vals[0] = m_indices[scan_r0];
                        }
                        scan_vals[1] = -1;
                        if (scan_r0 + 1 < M) {
                            scan_vals[1] = m_indices[scan_r0 + 1];
                        }
                        scan_vals[2] = -1;
                        if (scan_r0 + 2 < M) {
                            scan_vals[2] = m_indices[scan_r0 + 2];
                        }
                        scan_vals[3] = -1;
                        if (scan_r0 + 3 < M) {
                            scan_vals[3] = m_indices[scan_r0 + 3];
                        }
                        scan_vals[4] = -1;
                        if (scan_r0 + 4 < M) {
                            scan_vals[4] = m_indices[scan_r0 + 4];
                        }
                        scan_vals[5] = -1;
                        if (scan_r0 + 5 < M) {
                            scan_vals[5] = m_indices[scan_r0 + 5];
                        }
                        scan_vals[6] = -1;
                        if (scan_r0 + 6 < M) {
                            scan_vals[6] = m_indices[scan_r0 + 6];
                        }
                        scan_vals[7] = -1;
                        if (scan_r0 + 7 < M) {
                            scan_vals[7] = m_indices[scan_r0 + 7];
                        }
                        scan_vals[8] = -1;
                        if (scan_r0 + 8 < M) {
                            scan_vals[8] = m_indices[scan_r0 + 8];
                        }
                        scan_vals[9] = -1;
                        if (scan_r0 + 9 < M) {
                            scan_vals[9] = m_indices[scan_r0 + 9];
                        }
                        scan_vals[10] = -1;
                        if (scan_r0 + 10 < M) {
                            scan_vals[10] = m_indices[scan_r0 + 10];
                        }
                        scan_vals[11] = -1;
                        if (scan_r0 + 11 < M) {
                            scan_vals[11] = m_indices[scan_r0 + 11];
                        }
                        scan_vals[12] = -1;
                        if (scan_r0 + 12 < M) {
                            scan_vals[12] = m_indices[scan_r0 + 12];
                        }
                        scan_vals[13] = -1;
                        if (scan_r0 + 13 < M) {
                            scan_vals[13] = m_indices[scan_r0 + 13];
                        }
                        scan_vals[14] = -1;
                        if (scan_r0 + 14 < M) {
                            scan_vals[14] = m_indices[scan_r0 + 14];
                        }
                        scan_vals[15] = -1;
                        if (scan_r0 + 15 < M) {
                            scan_vals[15] = m_indices[scan_r0 + 15];
                        }
                        scan_vals[16] = -1;
                        if (scan_r0 + 16 < M) {
                            scan_vals[16] = m_indices[scan_r0 + 16];
                        }
                        scan_vals[17] = -1;
                        if (scan_r0 + 17 < M) {
                            scan_vals[17] = m_indices[scan_r0 + 17];
                        }
                        scan_vals[18] = -1;
                        if (scan_r0 + 18 < M) {
                            scan_vals[18] = m_indices[scan_r0 + 18];
                        }
                        scan_vals[19] = -1;
                        if (scan_r0 + 19 < M) {
                            scan_vals[19] = m_indices[scan_r0 + 19];
                        }
                        scan_vals[20] = -1;
                        if (scan_r0 + 20 < M) {
                            scan_vals[20] = m_indices[scan_r0 + 20];
                        }
                        scan_vals[21] = -1;
                        if (scan_r0 + 21 < M) {
                            scan_vals[21] = m_indices[scan_r0 + 21];
                        }
                        scan_vals[22] = -1;
                        if (scan_r0 + 22 < M) {
                            scan_vals[22] = m_indices[scan_r0 + 22];
                        }
                        scan_vals[23] = -1;
                        if (scan_r0 + 23 < M) {
                            scan_vals[23] = m_indices[scan_r0 + 23];
                        }
                        scan_vals[24] = -1;
                        if (scan_r0 + 24 < M) {
                            scan_vals[24] = m_indices[scan_r0 + 24];
                        }
                        scan_vals[25] = -1;
                        if (scan_r0 + 25 < M) {
                            scan_vals[25] = m_indices[scan_r0 + 25];
                        }
                        scan_vals[26] = -1;
                        if (scan_r0 + 26 < M) {
                            scan_vals[26] = m_indices[scan_r0 + 26];
                        }
                        scan_vals[27] = -1;
                        if (scan_r0 + 27 < M) {
                            scan_vals[27] = m_indices[scan_r0 + 27];
                        }
                        scan_vals[28] = -1;
                        if (scan_r0 + 28 < M) {
                            scan_vals[28] = m_indices[scan_r0 + 28];
                        }
                        scan_vals[29] = -1;
                        if (scan_r0 + 29 < M) {
                            scan_vals[29] = m_indices[scan_r0 + 29];
                        }
                        scan_vals[30] = -1;
                        if (scan_r0 + 30 < M) {
                            scan_vals[30] = m_indices[scan_r0 + 30];
                        }
                        scan_vals[31] = -1;
                        if (scan_r0 + 31 < M) {
                            scan_vals[31] = m_indices[scan_r0 + 31];
                        }
                    }
                    int scan_cur = scan_vals[0];
                    if (scan_cur != scan_prev && (unsigned int)scan_cur < (unsigned int)G) {
                        ea_start[scan_cur] = scan_r0;
                    }
                    scan_prev = scan_cur;
                    int scan_cur_0 = scan_vals[1];
                    if (scan_cur_0 != scan_prev && (unsigned int)scan_cur_0 < (unsigned int)G) {
                        ea_start[scan_cur_0] = scan_r0 + 1;
                    }
                    scan_prev = scan_cur_0;
                    int scan_cur_1 = scan_vals[2];
                    if (scan_cur_1 != scan_prev && (unsigned int)scan_cur_1 < (unsigned int)G) {
                        ea_start[scan_cur_1] = scan_r0 + 2;
                    }
                    scan_prev = scan_cur_1;
                    int scan_cur_2 = scan_vals[3];
                    if (scan_cur_2 != scan_prev && (unsigned int)scan_cur_2 < (unsigned int)G) {
                        ea_start[scan_cur_2] = scan_r0 + 3;
                    }
                    scan_prev = scan_cur_2;
                    int scan_cur_3 = scan_vals[4];
                    if (scan_cur_3 != scan_prev && (unsigned int)scan_cur_3 < (unsigned int)G) {
                        ea_start[scan_cur_3] = scan_r0 + 4;
                    }
                    scan_prev = scan_cur_3;
                    int scan_cur_4 = scan_vals[5];
                    if (scan_cur_4 != scan_prev && (unsigned int)scan_cur_4 < (unsigned int)G) {
                        ea_start[scan_cur_4] = scan_r0 + 5;
                    }
                    scan_prev = scan_cur_4;
                    int scan_cur_5 = scan_vals[6];
                    if (scan_cur_5 != scan_prev && (unsigned int)scan_cur_5 < (unsigned int)G) {
                        ea_start[scan_cur_5] = scan_r0 + 6;
                    }
                    scan_prev = scan_cur_5;
                    int scan_cur_6 = scan_vals[7];
                    if (scan_cur_6 != scan_prev && (unsigned int)scan_cur_6 < (unsigned int)G) {
                        ea_start[scan_cur_6] = scan_r0 + 7;
                    }
                    scan_prev = scan_cur_6;
                    int scan_cur_7 = scan_vals[8];
                    if (scan_cur_7 != scan_prev && (unsigned int)scan_cur_7 < (unsigned int)G) {
                        ea_start[scan_cur_7] = scan_r0 + 8;
                    }
                    scan_prev = scan_cur_7;
                    int scan_cur_8 = scan_vals[9];
                    if (scan_cur_8 != scan_prev && (unsigned int)scan_cur_8 < (unsigned int)G) {
                        ea_start[scan_cur_8] = scan_r0 + 9;
                    }
                    scan_prev = scan_cur_8;
                    int scan_cur_9 = scan_vals[10];
                    if (scan_cur_9 != scan_prev && (unsigned int)scan_cur_9 < (unsigned int)G) {
                        ea_start[scan_cur_9] = scan_r0 + 10;
                    }
                    scan_prev = scan_cur_9;
                    int scan_cur_10 = scan_vals[11];
                    if (scan_cur_10 != scan_prev && (unsigned int)scan_cur_10 < (unsigned int)G) {
                        ea_start[scan_cur_10] = scan_r0 + 11;
                    }
                    scan_prev = scan_cur_10;
                    int scan_cur_11 = scan_vals[12];
                    if (scan_cur_11 != scan_prev && (unsigned int)scan_cur_11 < (unsigned int)G) {
                        ea_start[scan_cur_11] = scan_r0 + 12;
                    }
                    scan_prev = scan_cur_11;
                    int scan_cur_12 = scan_vals[13];
                    if (scan_cur_12 != scan_prev && (unsigned int)scan_cur_12 < (unsigned int)G) {
                        ea_start[scan_cur_12] = scan_r0 + 13;
                    }
                    scan_prev = scan_cur_12;
                    int scan_cur_13 = scan_vals[14];
                    if (scan_cur_13 != scan_prev && (unsigned int)scan_cur_13 < (unsigned int)G) {
                        ea_start[scan_cur_13] = scan_r0 + 14;
                    }
                    scan_prev = scan_cur_13;
                    int scan_cur_14 = scan_vals[15];
                    if (scan_cur_14 != scan_prev && (unsigned int)scan_cur_14 < (unsigned int)G) {
                        ea_start[scan_cur_14] = scan_r0 + 15;
                    }
                    scan_prev = scan_cur_14;
                    int scan_cur_15 = scan_vals[16];
                    if (scan_cur_15 != scan_prev && (unsigned int)scan_cur_15 < (unsigned int)G) {
                        ea_start[scan_cur_15] = scan_r0 + 16;
                    }
                    scan_prev = scan_cur_15;
                    int scan_cur_16 = scan_vals[17];
                    if (scan_cur_16 != scan_prev && (unsigned int)scan_cur_16 < (unsigned int)G) {
                        ea_start[scan_cur_16] = scan_r0 + 17;
                    }
                    scan_prev = scan_cur_16;
                    int scan_cur_17 = scan_vals[18];
                    if (scan_cur_17 != scan_prev && (unsigned int)scan_cur_17 < (unsigned int)G) {
                        ea_start[scan_cur_17] = scan_r0 + 18;
                    }
                    scan_prev = scan_cur_17;
                    int scan_cur_18 = scan_vals[19];
                    if (scan_cur_18 != scan_prev && (unsigned int)scan_cur_18 < (unsigned int)G) {
                        ea_start[scan_cur_18] = scan_r0 + 19;
                    }
                    scan_prev = scan_cur_18;
                    int scan_cur_19 = scan_vals[20];
                    if (scan_cur_19 != scan_prev && (unsigned int)scan_cur_19 < (unsigned int)G) {
                        ea_start[scan_cur_19] = scan_r0 + 20;
                    }
                    scan_prev = scan_cur_19;
                    int scan_cur_20 = scan_vals[21];
                    if (scan_cur_20 != scan_prev && (unsigned int)scan_cur_20 < (unsigned int)G) {
                        ea_start[scan_cur_20] = scan_r0 + 21;
                    }
                    scan_prev = scan_cur_20;
                    int scan_cur_21 = scan_vals[22];
                    if (scan_cur_21 != scan_prev && (unsigned int)scan_cur_21 < (unsigned int)G) {
                        ea_start[scan_cur_21] = scan_r0 + 22;
                    }
                    scan_prev = scan_cur_21;
                    int scan_cur_22 = scan_vals[23];
                    if (scan_cur_22 != scan_prev && (unsigned int)scan_cur_22 < (unsigned int)G) {
                        ea_start[scan_cur_22] = scan_r0 + 23;
                    }
                    scan_prev = scan_cur_22;
                    int scan_cur_23 = scan_vals[24];
                    if (scan_cur_23 != scan_prev && (unsigned int)scan_cur_23 < (unsigned int)G) {
                        ea_start[scan_cur_23] = scan_r0 + 24;
                    }
                    scan_prev = scan_cur_23;
                    int scan_cur_24 = scan_vals[25];
                    if (scan_cur_24 != scan_prev && (unsigned int)scan_cur_24 < (unsigned int)G) {
                        ea_start[scan_cur_24] = scan_r0 + 25;
                    }
                    scan_prev = scan_cur_24;
                    int scan_cur_25 = scan_vals[26];
                    if (scan_cur_25 != scan_prev && (unsigned int)scan_cur_25 < (unsigned int)G) {
                        ea_start[scan_cur_25] = scan_r0 + 26;
                    }
                    scan_prev = scan_cur_25;
                    int scan_cur_26 = scan_vals[27];
                    if (scan_cur_26 != scan_prev && (unsigned int)scan_cur_26 < (unsigned int)G) {
                        ea_start[scan_cur_26] = scan_r0 + 27;
                    }
                    scan_prev = scan_cur_26;
                    int scan_cur_27 = scan_vals[28];
                    if (scan_cur_27 != scan_prev && (unsigned int)scan_cur_27 < (unsigned int)G) {
                        ea_start[scan_cur_27] = scan_r0 + 28;
                    }
                    scan_prev = scan_cur_27;
                    int scan_cur_28 = scan_vals[29];
                    if (scan_cur_28 != scan_prev && (unsigned int)scan_cur_28 < (unsigned int)G) {
                        ea_start[scan_cur_28] = scan_r0 + 29;
                    }
                    scan_prev = scan_cur_28;
                    int scan_cur_29 = scan_vals[30];
                    if (scan_cur_29 != scan_prev && (unsigned int)scan_cur_29 < (unsigned int)G) {
                        ea_start[scan_cur_29] = scan_r0 + 30;
                    }
                    scan_prev = scan_cur_29;
                    int scan_cur_30 = scan_vals[31];
                    if (scan_cur_30 != scan_prev && (unsigned int)scan_cur_30 < (unsigned int)G) {
                        ea_start[scan_cur_30] = scan_r0 + 31;
                    }
                    scan_prev = scan_cur_30;
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            if (warp_1 == 0) {
                int starts[8];
                starts[0] = ea_start[lane_0 * 8];
                starts[1] = ea_start[lane_0 * 8 + 1];
                starts[2] = ea_start[lane_0 * 8 + 2];
                starts[3] = ea_start[lane_0 * 8 + 3];
                starts[4] = ea_start[lane_0 * 8 + 4];
                starts[5] = ea_start[lane_0 * 8 + 5];
                starts[6] = ea_start[lane_0 * 8 + 6];
                starts[7] = ea_start[lane_0 * 8 + 7];
                if (starts[7] < starts[6]) {
                    starts[6] = starts[7];
                }
                if (starts[6] < starts[5]) {
                    starts[5] = starts[6];
                }
                if (starts[5] < starts[4]) {
                    starts[4] = starts[5];
                }
                if (starts[4] < starts[3]) {
                    starts[3] = starts[4];
                }
                if (starts[3] < starts[2]) {
                    starts[2] = starts[3];
                }
                if (starts[2] < starts[1]) {
                    starts[1] = starts[2];
                }
                if (starts[1] < starts[0]) {
                    starts[0] = starts[1];
                }
                int suffix = starts[0];
                int other = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                if (lane_0 + 1 < 32) {
                    if (other < suffix) {
                        suffix = other;
                    }
                }
                int other_0 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
                if (lane_0 + 2 < 32) {
                    if (other_0 < suffix) {
                        suffix = other_0;
                    }
                }
                int other_1 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
                if (lane_0 + 4 < 32) {
                    if (other_1 < suffix) {
                        suffix = other_1;
                    }
                }
                int other_2 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
                if (lane_0 + 8 < 32) {
                    if (other_2 < suffix) {
                        suffix = other_2;
                    }
                }
                int other_3 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
                if (lane_0 + 16 < 32) {
                    if (other_3 < suffix) {
                        suffix = other_3;
                    }
                }
                int after = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                if (lane_0 == 31) {
                    after = M;
                }
                if (after < starts[0]) {
                    starts[0] = after;
                }
                ea_start[lane_0 * 8] = starts[0];
                if (after < starts[1]) {
                    starts[1] = after;
                }
                ea_start[lane_0 * 8 + 1] = starts[1];
                if (after < starts[2]) {
                    starts[2] = after;
                }
                ea_start[lane_0 * 8 + 2] = starts[2];
                if (after < starts[3]) {
                    starts[3] = after;
                }
                ea_start[lane_0 * 8 + 3] = starts[3];
                if (after < starts[4]) {
                    starts[4] = after;
                }
                ea_start[lane_0 * 8 + 4] = starts[4];
                if (after < starts[5]) {
                    starts[5] = after;
                }
                ea_start[lane_0 * 8 + 5] = starts[5];
                if (after < starts[6]) {
                    starts[6] = after;
                }
                ea_start[lane_0 * 8 + 6] = starts[6];
                if (after < starts[7]) {
                    starts[7] = after;
                }
                ea_start[lane_0 * 8 + 7] = starts[7];
                int next_start = __shfl_down_sync(0xFFFFFFFF, starts[0], 1, 32);
                if (lane_0 == 31) {
                    next_start = M;
                }
                int tiles[8];
                int lane_tiles = 0;
                tiles[0] = 0;
                if (lane_0 * 8 < G) {
                    tiles[0] = (starts[1] - starts[0] + 128 - 1) / 128;
                }
                lane_tiles += tiles[0];
                tiles[1] = 0;
                if (lane_0 * 8 + 1 < G) {
                    tiles[1] = (starts[2] - starts[1] + 128 - 1) / 128;
                }
                lane_tiles += tiles[1];
                tiles[2] = 0;
                if (lane_0 * 8 + 2 < G) {
                    tiles[2] = (starts[3] - starts[2] + 128 - 1) / 128;
                }
                lane_tiles += tiles[2];
                tiles[3] = 0;
                if (lane_0 * 8 + 3 < G) {
                    tiles[3] = (starts[4] - starts[3] + 128 - 1) / 128;
                }
                lane_tiles += tiles[3];
                tiles[4] = 0;
                if (lane_0 * 8 + 4 < G) {
                    tiles[4] = (starts[5] - starts[4] + 128 - 1) / 128;
                }
                lane_tiles += tiles[4];
                tiles[5] = 0;
                if (lane_0 * 8 + 5 < G) {
                    tiles[5] = (starts[6] - starts[5] + 128 - 1) / 128;
                }
                lane_tiles += tiles[5];
                tiles[6] = 0;
                if (lane_0 * 8 + 6 < G) {
                    tiles[6] = (starts[7] - starts[6] + 128 - 1) / 128;
                }
                lane_tiles += tiles[6];
                tiles[7] = 0;
                if (lane_0 * 8 + 7 < G) {
                    tiles[7] = (next_start - starts[7] + 128 - 1) / 128;
                }
                lane_tiles += tiles[7];
                uint32_t _warp_scan_sum_8 = (unsigned int)lane_tiles;
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_8, 1, 32); if (lane >= 1) { _warp_scan_sum_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_8, 2, 32); if (lane >= 2) { _warp_scan_sum_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_8, 4, 32); if (lane >= 4) { _warp_scan_sum_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_8, 8, 32); if (lane >= 8) { _warp_scan_sum_8 += peer; } }
                { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_8, 16, 32); if (lane >= 16) { _warp_scan_sum_8 += peer; } }
                int inclusive = (int)_warp_scan_sum_8;
                int first = inclusive - lane_tiles;
                if (lane_0 * 8 <= G) {
                    ea_first[lane_0 * 8] = first;
                }
                if (lane_0 * 8 > G) {
                    ea_first[lane_0 * 8] = 2147483647;
                }
                first += tiles[0];
                if (lane_0 * 8 + 1 <= G) {
                    ea_first[lane_0 * 8 + 1] = first;
                }
                if (lane_0 * 8 + 1 > G) {
                    ea_first[lane_0 * 8 + 1] = 2147483647;
                }
                first += tiles[1];
                if (lane_0 * 8 + 2 <= G) {
                    ea_first[lane_0 * 8 + 2] = first;
                }
                if (lane_0 * 8 + 2 > G) {
                    ea_first[lane_0 * 8 + 2] = 2147483647;
                }
                first += tiles[2];
                if (lane_0 * 8 + 3 <= G) {
                    ea_first[lane_0 * 8 + 3] = first;
                }
                if (lane_0 * 8 + 3 > G) {
                    ea_first[lane_0 * 8 + 3] = 2147483647;
                }
                first += tiles[3];
                if (lane_0 * 8 + 4 <= G) {
                    ea_first[lane_0 * 8 + 4] = first;
                }
                if (lane_0 * 8 + 4 > G) {
                    ea_first[lane_0 * 8 + 4] = 2147483647;
                }
                first += tiles[4];
                if (lane_0 * 8 + 5 <= G) {
                    ea_first[lane_0 * 8 + 5] = first;
                }
                if (lane_0 * 8 + 5 > G) {
                    ea_first[lane_0 * 8 + 5] = 2147483647;
                }
                first += tiles[5];
                if (lane_0 * 8 + 6 <= G) {
                    ea_first[lane_0 * 8 + 6] = first;
                }
                if (lane_0 * 8 + 6 > G) {
                    ea_first[lane_0 * 8 + 6] = 2147483647;
                }
                first += tiles[6];
                if (lane_0 * 8 + 7 <= G) {
                    ea_first[lane_0 * 8 + 7] = first;
                }
                if (lane_0 * 8 + 7 > G) {
                    ea_first[lane_0 * 8 + 7] = 2147483647;
                }
                first += tiles[7];
                int total = __shfl_sync(0xFFFFFFFF, inclusive, 31);
                if (lane_0 == 0) {
                    ea_meta[0] = total;
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            int slot_limit = ((ea_meta[0]) < (512) ? (ea_meta[0]) : (512));
            int slot = tid_1;
            if (slot < slot_limit) {
                int expert = 0;
                if (slot >= ea_first[expert + 128]) {
                    expert += 128;
                }
                if (slot >= ea_first[expert + 64]) {
                    expert += 64;
                }
                if (slot >= ea_first[expert + 32]) {
                    expert += 32;
                }
                if (slot >= ea_first[expert + 16]) {
                    expert += 16;
                }
                if (slot >= ea_first[expert + 8]) {
                    expert += 8;
                }
                if (slot >= ea_first[expert + 4]) {
                    expert += 4;
                }
                if (slot >= ea_first[expert + 2]) {
                    expert += 2;
                }
                if (slot >= ea_first[expert + 1]) {
                    expert += 1;
                }
                int slot_row0 = ea_start[expert] + (slot - ea_first[expert]) * 128;
                int slot_rows = ea_start[expert + 1] - slot_row0;
                if (slot_rows > 128) {
                    slot_rows = 128;
                }
                ea_slot_row0[slot] = slot_row0;
                ea_slot_er[slot] = expert << 8 | slot_rows;
            }
            int slot_2 = 256 + tid_1;
            if (slot_2 < slot_limit) {
                int expert_1 = 0;
                if (slot_2 >= ea_first[expert_1 + 128]) {
                    expert_1 += 128;
                }
                if (slot_2 >= ea_first[expert_1 + 64]) {
                    expert_1 += 64;
                }
                if (slot_2 >= ea_first[expert_1 + 32]) {
                    expert_1 += 32;
                }
                if (slot_2 >= ea_first[expert_1 + 16]) {
                    expert_1 += 16;
                }
                if (slot_2 >= ea_first[expert_1 + 8]) {
                    expert_1 += 8;
                }
                if (slot_2 >= ea_first[expert_1 + 4]) {
                    expert_1 += 4;
                }
                if (slot_2 >= ea_first[expert_1 + 2]) {
                    expert_1 += 2;
                }
                if (slot_2 >= ea_first[expert_1 + 1]) {
                    expert_1 += 1;
                }
                int slot_row0_1 = ea_start[expert_1] + (slot_2 - ea_first[expert_1]) * 128;
                int slot_rows_1 = ea_start[expert_1 + 1] - slot_row0_1;
                if (slot_rows_1 > 128) {
                    slot_rows_1 = 128;
                }
                ea_slot_row0[slot_2] = slot_row0_1;
                ea_slot_er[slot_2] = expert_1 << 8 | slot_rows_1;
            }
            mbarrier_arrive(scan_done_addr);
            if (warp == 0) {
                int _tmem_hold_9 = smem + 208;
                asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_9), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
            }
            unsigned int partial_stage = 0;
            unsigned int scale_stage = 0;
            const int acc_wg = warp / 4;
            const int acc_warp_in_wg = warp % 4;
            int wg_col_base = acc_wg * 64;
            int row_base = acc_warp_in_wg * 32;
            int row = row_base + lane;
            unsigned int _phase_table_full_0 = 0;
            mbarrier_wait(table_full_addr, _phase_table_full_0);
            _phase_table_full_0 ^= 1;
            int m_tiles = ea_meta[0];
            int n_tiles = N / 128;
            int total_tiles = m_tiles * n_tiles;
            int m_tile = bid / n_tiles;
            int n_tile = bid % n_tiles;
            int dm_tile = num_bids / n_tiles;
            int dn_tile = num_bids % n_tiles;
            unsigned int acc_zero_bits = ((K == 128) ? 2147483648u : 0);
            float acc_zero = __uint_as_float(acc_zero_bits);
            unsigned int _phase_scale_full = 0;
            unsigned int _phase_partial_full = 0;
            #pragma unroll 1
            for (int tile_id = bid; tile_id < total_tiles; tile_id += num_bids) {
                int expert_2 = 0;
                int row0 = 0;
                int tile_rows = 0;
                if (m_tile < 512) {
                    int entry = ea_slot_er[m_tile];
                    row0 = ea_slot_row0[m_tile];
                    expert_2 = entry >> 8;
                    tile_rows = entry & 255;
                } else {
                    int expert_0 = 0;
                    if (m_tile >= ea_first[expert_0 + 128]) {
                        expert_0 += 128;
                    }
                    if (m_tile >= ea_first[expert_0 + 64]) {
                        expert_0 += 64;
                    }
                    if (m_tile >= ea_first[expert_0 + 32]) {
                        expert_0 += 32;
                    }
                    if (m_tile >= ea_first[expert_0 + 16]) {
                        expert_0 += 16;
                    }
                    if (m_tile >= ea_first[expert_0 + 8]) {
                        expert_0 += 8;
                    }
                    if (m_tile >= ea_first[expert_0 + 4]) {
                        expert_0 += 4;
                    }
                    if (m_tile >= ea_first[expert_0 + 2]) {
                        expert_0 += 2;
                    }
                    if (m_tile >= ea_first[expert_0 + 1]) {
                        expert_0 += 1;
                    }
                    expert_2 = expert_0;
                    row0 = ea_start[expert_2] + (m_tile - ea_first[expert_2]) * 128;
                    tile_rows = ea_start[expert_2 + 1] - row0;
                    if (tile_rows > 128) {
                        tile_rows = 128;
                    }
                }
                float acc0[32];
                float acc1[32];
                acc0[0] = acc_zero;
                acc0[1] = acc_zero;
                acc0[2] = acc_zero;
                acc0[3] = acc_zero;
                acc0[4] = acc_zero;
                acc0[5] = acc_zero;
                acc0[6] = acc_zero;
                acc0[7] = acc_zero;
                acc0[8] = acc_zero;
                acc0[9] = acc_zero;
                acc0[10] = acc_zero;
                acc0[11] = acc_zero;
                acc0[12] = acc_zero;
                acc0[13] = acc_zero;
                acc0[14] = acc_zero;
                acc0[15] = acc_zero;
                acc0[16] = acc_zero;
                acc0[17] = acc_zero;
                acc0[18] = acc_zero;
                acc0[19] = acc_zero;
                acc0[20] = acc_zero;
                acc0[21] = acc_zero;
                acc0[22] = acc_zero;
                acc0[23] = acc_zero;
                acc0[24] = acc_zero;
                acc0[25] = acc_zero;
                acc0[26] = acc_zero;
                acc0[27] = acc_zero;
                acc0[28] = acc_zero;
                acc0[29] = acc_zero;
                acc0[30] = acc_zero;
                acc0[31] = acc_zero;
                acc1[0] = acc_zero;
                acc1[1] = acc_zero;
                acc1[2] = acc_zero;
                acc1[3] = acc_zero;
                acc1[4] = acc_zero;
                acc1[5] = acc_zero;
                acc1[6] = acc_zero;
                acc1[7] = acc_zero;
                acc1[8] = acc_zero;
                acc1[9] = acc_zero;
                acc1[10] = acc_zero;
                acc1[11] = acc_zero;
                acc1[12] = acc_zero;
                acc1[13] = acc_zero;
                acc1[14] = acc_zero;
                acc1[15] = acc_zero;
                acc1[16] = acc_zero;
                acc1[17] = acc_zero;
                acc1[18] = acc_zero;
                acc1[19] = acc_zero;
                acc1[20] = acc_zero;
                acc1[21] = acc_zero;
                acc1[22] = acc_zero;
                acc1[23] = acc_zero;
                acc1[24] = acc_zero;
                acc1[25] = acc_zero;
                acc1[26] = acc_zero;
                acc1[27] = acc_zero;
                acc1[28] = acc_zero;
                acc1[29] = acc_zero;
                acc1[30] = acc_zero;
                acc1[31] = acc_zero;
                float partial_fragment[16];
                int k_blocks = K / 128;
                int scale_groups = k_blocks;
                #pragma unroll 1
                for (int scale_group = 0; scale_group < scale_groups; scale_group++) {
                    mbarrier_wait(scale_full_addr + (scale_stage) * 8, _phase_scale_full);
                    int scale_base = scale_stage * 128 + (unsigned int)row;
                    #pragma unroll
                    for (int k_inner = 0; k_inner < 1; k_inner++) {
                        mbarrier_wait(partial_full_addr + (partial_stage) * 8, _phase_partial_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float a_s = smem_ascale[scale_base + k_inner];
                        float b_s = smem_bscale[scale_stage + (unsigned int)k_inner];
                        float combined = a_s * b_s;
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)wg_col_base) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        {
                            unsigned long long _fma_acc_scale2_10;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_10));
                        }
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)wg_col_base + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        {
                            unsigned long long _fma_acc_scale2_11;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_11));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_11));
                        }
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)wg_col_base + 32) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        {
                            unsigned long long _fma_acc_scale2_12;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_12));
                        }
                        tmem_ld_x16(&partial_fragment[0], (unsigned int)tmem_partials + (partial_stage * 128 + (unsigned int)wg_col_base + 32 + 16) + (unsigned int)(row_base << 16));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        {
                            unsigned long long _fma_acc_scale2_13;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_13));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_13));
                        }
                        if (elect_sync()) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(partial_free_addr + (partial_stage) * 8);
                        }
                        partial_stage += 1;
                        if (partial_stage == 4) { partial_stage = 0; _phase_partial_full ^= 1; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(scale_free_addr + (scale_stage) * 8);
                    scale_stage += 1;
                    if (scale_stage == 3) { scale_stage = 0; _phase_scale_full ^= 1; }
                }
                int full_tile = 0;
                if (tile_rows == 128) {
                    full_tile = 1;
                }
                if (row0 + tile_rows == M) {
                    full_tile = 1;
                }
                if (full_tile != 0) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    unsigned int epi_stage = 0;
                    int c_stage_row = epi_stage * 128 + (unsigned int)row;
                    unsigned int acc0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc0[_lp*2 + 0], acc0[_lp*2+1 + 0]));
                        acc0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int acc1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc1[_lp*2 + 0], acc1[_lp*2+1 + 0]));
                        acc1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (acc_wg == 0) {
                        #pragma unroll
                        for (int chunk = 0; chunk < 4; chunk++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_0_addr + (unsigned int)(c_stage_row * 64 + chunk * 16 ^ (c_stage_row * 64 + chunk * 16 >> 7 & 3) << 4))), "r"(acc0_bf16[chunk * 4]), "r"(acc0_bf16[chunk * 4 + 1]), "r"(acc0_bf16[chunk * 4 + 2]), "r"(acc0_bf16[chunk * 4 + 3]) : "memory");
                        }
                        #pragma unroll
                        for (int chunk_1 = 0; chunk_1 < 4; chunk_1++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_1_addr + (unsigned int)(c_stage_row * 64 + chunk_1 * 16 ^ (c_stage_row * 64 + chunk_1 * 16 >> 7 & 3) << 4))), "r"(acc1_bf16[chunk_1 * 4]), "r"(acc1_bf16[chunk_1 * 4 + 1]), "r"(acc1_bf16[chunk_1 * 4 + 2]), "r"(acc1_bf16[chunk_1 * 4 + 3]) : "memory");
                        }
                    }
                    if (acc_wg == 1) {
                        #pragma unroll
                        for (int chunk_2 = 0; chunk_2 < 4; chunk_2++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_2_addr + (unsigned int)(c_stage_row * 64 + chunk_2 * 16 ^ (c_stage_row * 64 + chunk_2 * 16 >> 7 & 3) << 4))), "r"(acc0_bf16[chunk_2 * 4]), "r"(acc0_bf16[chunk_2 * 4 + 1]), "r"(acc0_bf16[chunk_2 * 4 + 2]), "r"(acc0_bf16[chunk_2 * 4 + 3]) : "memory");
                        }
                        #pragma unroll
                        for (int chunk_3 = 0; chunk_3 < 4; chunk_3++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_3_addr + (unsigned int)(c_stage_row * 64 + chunk_3 * 16 ^ (c_stage_row * 64 + chunk_3 * 16 >> 7 & 3) << 4))), "r"(acc1_bf16[chunk_3 * 4]), "r"(acc1_bf16[chunk_3 * 4 + 1]), "r"(acc1_bf16[chunk_3 * 4 + 2]), "r"(acc1_bf16[chunk_3 * 4 + 3]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 256;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            int staging_offset = epi_stage * 8192;
                            tma_store_2d((&C_tma), n_tile * 128, row0, epi_staging_0_addr + (unsigned int)staging_offset);
                            tma_store_2d((&C_tma), n_tile * 128 + 32, row0, epi_staging_1_addr + (unsigned int)staging_offset);
                            tma_store_2d((&C_tma), n_tile * 128 + 64, row0, epi_staging_2_addr + (unsigned int)staging_offset);
                            tma_store_2d((&C_tma), n_tile * 128 + 96, row0, epi_staging_3_addr + (unsigned int)staging_offset);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row < tile_rows) {
                    long long output_row = row0 + row;
                    #pragma unroll
                    for (int vec = 0; vec < 32; vec += 8) {
                        unsigned int _packed_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc0 + vec)[_lp*2 + 0], (acc0 + vec)[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc0 + vec)[_lp*2 + 2], (acc0 + vec)[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc0 + vec)[_lp*2 + 4], (acc0 + vec)[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc0 + vec)[_lp*2 + 6], (acc0 + vec)[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(C + (output_row * (long long)N + (long long)(n_tile * 128) + (long long)wg_col_base + (long long)vec))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                        unsigned int _packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc1 + vec)[_lp*2 + 0], (acc1 + vec)[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc1 + vec)[_lp*2 + 2], (acc1 + vec)[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc1 + vec)[_lp*2 + 4], (acc1 + vec)[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((acc1 + vec)[_lp*2 + 6], (acc1 + vec)[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(C + (output_row * (long long)N + (long long)(n_tile * 128) + (long long)wg_col_base + 32 + (long long)vec))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                    }
                }
                m_tile += dm_tile;
                n_tile += dn_tile;
                if (n_tile >= n_tiles) {
                    n_tile -= n_tiles;
                    m_tile += 1;
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: mma_role ----
    if (warp == 8) {
        { // mma_role_main
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            unsigned int ab_stage = 0;
            unsigned int partial_stage_1 = 0;
            unsigned int _phase_table_full_0_1 = 0;
            mbarrier_wait(table_full_addr, _phase_table_full_0_1);
            _phase_table_full_0_1 ^= 1;
            int m_tiles_1 = ea_meta[0];
            int n_tiles_1 = N / 128;
            int total_tiles_1 = m_tiles_1 * n_tiles_1;
            unsigned int _phase_partial_free = 1;
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int _tile_id = bid; _tile_id < total_tiles_1; _tile_id += num_bids) {
                int k_blocks_1 = K / 128;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < k_blocks_1; iter_k++) {
                    mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                    mbarrier_wait(ab_full_addr + (ab_stage) * 8, _phase_ab_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_partials + (partial_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_partials + (partial_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_partials + (partial_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_partials + (partial_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                        }
                    }
                    elect_commit(ab_free_addr + (ab_stage) * 8);
                    elect_commit(partial_full_addr + (partial_stage_1) * 8);
                    ab_stage += 1;
                    if (ab_stage == 5) { ab_stage = 0; _phase_ab_full ^= 1; }
                    partial_stage_1 += 1;
                    if (partial_stage_1 == 4) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            if (lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
            if (lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory"); }
            if (lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
            unsigned int ab_stage_1 = 0;
            unsigned int _phase_table_full_0_2 = 0;
            mbarrier_wait(table_full_addr, _phase_table_full_0_2);
            _phase_table_full_0_2 ^= 1;
            int m_tiles_2 = ea_meta[0];
            int n_tiles_2 = N / 128;
            int total_tiles_2 = m_tiles_2 * n_tiles_2;
            int m_tile_1 = bid / n_tiles_2;
            int n_tile_1 = bid % n_tiles_2;
            int dm_tile_1 = num_bids / n_tiles_2;
            int dn_tile_1 = num_bids % n_tiles_2;
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int tile_id_1 = bid; tile_id_1 < total_tiles_2; tile_id_1 += num_bids) {
                int expert_3 = 0;
                int row0_1 = 0;
                int tile_rows_1 = 0;
                if (m_tile_1 < 512) {
                    int entry_1 = ea_slot_er[m_tile_1];
                    row0_1 = ea_slot_row0[m_tile_1];
                    expert_3 = entry_1 >> 8;
                    tile_rows_1 = entry_1 & 255;
                } else {
                    int expert_0_1 = 0;
                    if (m_tile_1 >= ea_first[expert_0_1 + 128]) {
                        expert_0_1 += 128;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 64]) {
                        expert_0_1 += 64;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 32]) {
                        expert_0_1 += 32;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 16]) {
                        expert_0_1 += 16;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 8]) {
                        expert_0_1 += 8;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 4]) {
                        expert_0_1 += 4;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 2]) {
                        expert_0_1 += 2;
                    }
                    if (m_tile_1 >= ea_first[expert_0_1 + 1]) {
                        expert_0_1 += 1;
                    }
                    expert_3 = expert_0_1;
                    row0_1 = ea_start[expert_3] + (m_tile_1 - ea_first[expert_3]) * 128;
                    tile_rows_1 = ea_start[expert_3 + 1] - row0_1;
                    if (tile_rows_1 > 128) {
                        tile_rows_1 = 128;
                    }
                }
                int k_blocks_2 = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_blocks_2; iter_k_1++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                        tma_3d_gmem2smem(smem_a_addr + ab_stage_1 * 16384, (&A), 0, row0_1, iter_k_1, ab_full_addr + (ab_stage_1) * 8);
                        tma_4d_gmem2smem(smem_b_addr + ab_stage_1 * 16384, (&B), 0, n_tile_1 * 128, iter_k_1, expert_3, ab_full_addr + (ab_stage_1) * 8);
                        mbarrier_arrive_expect_tx(ab_full_addr + (ab_stage_1) * 8, 32768);
                        ab_stage_1 += 1;
                        if (ab_stage_1 == 5) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                    }
                }
                m_tile_1 += dm_tile_1;
                n_tile_1 += dn_tile_1;
                if (n_tile_1 >= n_tiles_2) {
                    n_tile_1 -= n_tiles_2;
                    m_tile_1 += 1;
                }
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp == 10) {
        { // scale_role_main
            unsigned int scale_stage_1 = 0;
            unsigned int _phase_table_full_0_3 = 0;
            mbarrier_wait(table_full_addr, _phase_table_full_0_3);
            _phase_table_full_0_3 ^= 1;
            int m_tiles_3 = ea_meta[0];
            int n_tiles_3 = N / 128;
            int total_tiles_3 = m_tiles_3 * n_tiles_3;
            int m_tile_2 = bid / n_tiles_3;
            int n_tile_2 = bid % n_tiles_3;
            int dm_tile_2 = num_bids / n_tiles_3;
            int dn_tile_2 = num_bids % n_tiles_3;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int tile_id_2 = bid; tile_id_2 < total_tiles_3; tile_id_2 += num_bids) {
                int expert_4 = 0;
                int row0_2 = 0;
                int tile_rows_2 = 0;
                if (m_tile_2 < 512) {
                    int entry_2 = ea_slot_er[m_tile_2];
                    row0_2 = ea_slot_row0[m_tile_2];
                    expert_4 = entry_2 >> 8;
                    tile_rows_2 = entry_2 & 255;
                } else {
                    int expert_0_2 = 0;
                    if (m_tile_2 >= ea_first[expert_0_2 + 128]) {
                        expert_0_2 += 128;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 64]) {
                        expert_0_2 += 64;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 32]) {
                        expert_0_2 += 32;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 16]) {
                        expert_0_2 += 16;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 8]) {
                        expert_0_2 += 8;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 4]) {
                        expert_0_2 += 4;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 2]) {
                        expert_0_2 += 2;
                    }
                    if (m_tile_2 >= ea_first[expert_0_2 + 1]) {
                        expert_0_2 += 1;
                    }
                    expert_4 = expert_0_2;
                    row0_2 = ea_start[expert_4] + (m_tile_2 - ea_first[expert_4]) * 128;
                    tile_rows_2 = ea_start[expert_4 + 1] - row0_2;
                    if (tile_rows_2 > 128) {
                        tile_rows_2 = 128;
                    }
                }
                int k_blocks_3 = K / 128;
                int scale_groups_1 = k_blocks_3;
                int n_blocks = N / 128;
                #pragma unroll 1
                for (int scale_group_1 = 0; scale_group_1 < scale_groups_1; scale_group_1++) {
                    mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                    int stage_base = scale_stage_1 * 128;
                    #pragma unroll
                    for (int chunk_4 = 0; chunk_4 < 4; chunk_4++) {
                        int row_1 = chunk_4 * 32 + lane;
                        int g_row = row0_2 + row_1;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_ascale_addr + (unsigned int)((stage_base + row_1) * 4)), "l"(a_scale + (g_row * k_blocks_3 + scale_group_1)), "r"((g_row < M) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((lane == 0) ? 1 : 0), "r"(smem_bscale_addr + scale_stage_1 * 4), "l"(b_scale + ((expert_4 * n_blocks + n_tile_2) * k_blocks_3 + scale_group_1)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(scale_full_addr + (scale_stage_1) * 8) : "memory");
                    scale_stage_1 += 1;
                    if (scale_stage_1 == 3) { scale_stage_1 = 0; _phase_scale_free ^= 1; }
                }
                m_tile_2 += dm_tile_2;
                n_tile_2 += dn_tile_2;
                if (n_tile_2 >= n_tiles_3) {
                    n_tile_2 -= n_tiles_3;
                    m_tile_2 += 1;
                }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 3; _drain++) {
                mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                scale_stage_1 += 1;
                if (scale_stage_1 == 3) { scale_stage_1 = 0; _phase_scale_free ^= 1; }
            }
        }
    }
    // ---- Role: tma_b_role ----
    if (warp == 11) {
        { // tma_b_role_main
            unsigned int _phase_scan_done_0 = 0;
            mbarrier_wait(scan_done_addr, _phase_scan_done_0);
            _phase_scan_done_0 ^= 1;
            mbarrier_arrive(table_full_addr);
        }
    }

    // Cleanup
}

} // extern "C"
