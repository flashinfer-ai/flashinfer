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
#define TMEM_NCOLS 192
#define TMEM_ACC_BIG_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_CHUNK_PIPE_STAGES 2
#define SMEM_A_S_OFF 1024
#define SMEM_A_S_STAGE_BYTES 16384
#define SMEM_A_S_STRIDE 28672
#define SMEM_B1_S_OFF 17408
#define SMEM_B1_S_STAGE_BYTES 4096
#define SMEM_B1_S_STRIDE 28672
#define SMEM_B2_S_OFF 21504
#define SMEM_B2_S_STAGE_BYTES 4096
#define SMEM_B2_S_STRIDE 28672
#define SMEM_B3_S_OFF 25600
#define SMEM_B3_S_STAGE_BYTES 4096
#define SMEM_B3_S_STRIDE 28672
#define SMEM_B123_S_OFF 17408
#define SMEM_B123_S_STAGE_BYTES 12288
#define SMEM_B123_S_STRIDE 28672
#define SMEM_TOTAL 115712
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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


__device__ __forceinline__ void mbarrier_wait_plain_nonuniform(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_PLAIN:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra DONE_PLAIN;\n\t"
        "bra LAB_WAIT_PLAIN;\n\t"
        "DONE_PLAIN:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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



__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_fwd_map__ns96_g2_ol2l(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap FNP, float* __restrict__ spart, float* __restrict__ qpart, int T, int m_tiles, int log2_pk, unsigned int* __restrict__ ticket)
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
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 32)
    #define big_full_addr (mbar_base + 64)
    #define big_empty_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* a_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int a_s_addr = smem + 1024;
    __nv_bfloat16* b1_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int b1_s_addr = smem + 17408;
    __nv_bfloat16* b2_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int b2_s_addr = smem + 21504;
    __nv_bfloat16* b3_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int b3_s_addr = smem + 25600;
    __nv_bfloat16* b123_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int b123_s_addr = smem + 17408;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // a_empty: 4 barriers, init_count=5
            mbarrier_init(smem + 32, 5);
            mbarrier_init(smem + 40, 5);
            mbarrier_init(smem + 48, 5);
            mbarrier_init(smem + 56, 5);
            // --- pipeline 'chunk_pipe' ---
            // big_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // big_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_big = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int l_stage = 0;
            unsigned int lk_stage = 0;
            int units = m_tiles << log2_pk;
            int all_stages = 256;
            int unit_stages = all_stages >> log2_pk;
            int k_bits = 14;
            int k_shift = k_bits - log2_pk;
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int w = bid; w < units; w += num_bids) {
                    int m = w >> log2_pk;
                    int kappa = w - (m << log2_pk);
                    int t0 = m * 128;
                    int k_base = kappa << k_shift;
                    #pragma unroll 1
                    for (int st = 0; st < unit_stages; st += 2) {
                        mbarrier_wait(a_empty_addr + (l_stage) * 8, _phase_a_empty);
                        unsigned int slot_j = l_stage;
                        l_stage += 1;
                        if (l_stage == 4) { l_stage = 0; _phase_a_empty ^= 1; }
                        mbarrier_wait(a_empty_addr + (l_stage) * 8, _phase_a_empty);
                        unsigned int slot_j_0 = l_stage;
                        l_stage += 1;
                        if (l_stage == 4) { l_stage = 0; _phase_a_empty ^= 1; }
                        int k0 = k_base + st * 64;
                        mbarrier_arrive_expect_tx(a_full_addr + (((l_stage + 2 >= 4) ? l_stage + 2 - 4 : l_stage + 2)) * 8, 28672);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(a_s_addr + slot_j * 28672), "l"((&X)), "r"(k0), "r"(t0),
                               "r"(a_full_addr + (((l_stage + 2 >= 4) ? l_stage + 2 - 4 : l_stage + 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem(b1_s_addr + slot_j * 28672, (&FNP), k0, 0, 0, a_full_addr + (((l_stage + 2 >= 4) ? l_stage + 2 - 4 : l_stage + 2)) * 8);
                        tma_3d_gmem2smem(b2_s_addr + slot_j * 28672, (&FNP), k0, 0, 1, a_full_addr + (((l_stage + 2 >= 4) ? l_stage + 2 - 4 : l_stage + 2)) * 8);
                        tma_3d_gmem2smem(b3_s_addr + slot_j * 28672, (&FNP), k0, 0, 2, a_full_addr + (((l_stage + 2 >= 4) ? l_stage + 2 - 4 : l_stage + 2)) * 8);
                        int k0_0 = k_base + (st + 1) * 64;
                        mbarrier_arrive_expect_tx(a_full_addr + (((l_stage + 3 >= 4) ? l_stage + 3 - 4 : l_stage + 3)) * 8, 28672);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(a_s_addr + slot_j_0 * 28672), "l"((&X)), "r"(k0_0), "r"(t0),
                               "r"(a_full_addr + (((l_stage + 3 >= 4) ? l_stage + 3 - 4 : l_stage + 3)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem(b1_s_addr + slot_j_0 * 28672, (&FNP), k0_0, 0, 0, a_full_addr + (((l_stage + 3 >= 4) ? l_stage + 3 - 4 : l_stage + 3)) * 8);
                        tma_3d_gmem2smem(b2_s_addr + slot_j_0 * 28672, (&FNP), k0_0, 0, 1, a_full_addr + (((l_stage + 3 >= 4) ? l_stage + 3 - 4 : l_stage + 3)) * 8);
                        tma_3d_gmem2smem(b3_s_addr + slot_j_0 * 28672, (&FNP), k0_0, 0, 2, a_full_addr + (((l_stage + 3 >= 4) ? l_stage + 3 - 4 : l_stage + 3)) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int m_stage = 0;
            unsigned int c_stage = 0;
            unsigned int u_stage = 0;
            unsigned int mk_stage = 0;
            int units_1 = m_tiles << log2_pk;
            int all_stages_1 = 256;
            int unit_stages_1 = all_stages_1 >> log2_pk;
            int k_bits_1 = 14;
            int k_shift_1 = k_bits_1 - log2_pk;
            unsigned int _phase_big_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int _w = bid; _w < units_1; _w += num_bids) {
                    #pragma unroll 1
                    for (int st_1 = 0; st_1 < unit_stages_1; st_1++) {
                        int kc_pos = st_1 % 2;
                        if (kc_pos == 0) {
                            mbarrier_wait(big_empty_addr + (c_stage) * 8, _phase_big_empty);
                        }
                        mbarrier_wait(a_full_addr + (m_stage) * 8, _phase_a_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int big_init = ((kc_pos == 0) ? 1 : 0);
                        int small_init = ((st_1 == 0) ? 1 : 0);
                        int _smem_desc_lo_0 = (((a_s_addr) >> 4) & 0x3FFF) + (m_stage) * 1792;
                        int _smem_desc_lo_1 = (((b123_s_addr) >> 4) & 0x3FFF) + (m_stage) * 1792;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135791760;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_acc_big + (c_stage * 96))), "r"(((big_init) ? 0 : 1)) : "memory");
                        tcgen05_commit(a_empty_addr + (m_stage) * 8);
                        m_stage += 1;
                        if (m_stage == 4) { m_stage = 0; _phase_a_full ^= 1; }
                        if (kc_pos == 1) {
                            tcgen05_commit(big_full_addr + (c_stage) * 8);
                            c_stage += 1;
                            if (c_stage == 2) { c_stage = 0; _phase_big_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: rows ----
    if (warp >= 2 && warp <= 5) {
        { // rows_main
            unsigned int r_stage = 0;
            unsigned int rc_stage = 0;
            unsigned int ru_stage = 0;
            unsigned int rk_stage = 0;
            const int quarter = warp % 4;
            const int row = quarter * 32 + lane;
            unsigned int row_base = (unsigned int)(quarter * 32 << 16);
            long long T_pad = (long long)m_tiles * 128;
            int units_2 = m_tiles << log2_pk;
            int all_stages_2 = 256;
            int unit_stages_2 = all_stages_2 >> log2_pk;
            int k_bits_2 = 14;
            int k_shift_2 = k_bits_2 - log2_pk;
            unsigned int _phase_a_full_1 = 0;
            unsigned int _phase_big_full = 0;
            #pragma unroll 1
            for (int w_1 = bid; w_1 < units_2; w_1 += num_bids) {
                int m_1 = w_1 >> log2_pk;
                int kappa_1 = w_1 - (m_1 << log2_pk);
                float acc[24];
                float comp[24];
                float qsum[1];
                float qcomp[1];
                #pragma unroll
                for (int n = 0; n < 24; n++) {
                    acc[n] = 0.0f;
                    comp[n] = 0.0f;
                }
                qsum[0] = 0.0f;
                qcomp[0] = 0.0f;
                long long t_row = (long long)kappa_1 * T_pad + (long long)(m_1 * 128 + row);
                long long s_off = t_row * 24;
                #pragma unroll 1
                for (int st_2 = 0; st_2 < unit_stages_2; st_2++) {
                    mbarrier_wait_plain_nonuniform(a_full_addr + (r_stage) * 8, _phase_a_full_1);
                    __syncwarp();
                    unsigned int a_base = a_s_addr + r_stage * 28672;
                    unsigned int words[32];
                    #pragma unroll
                    for (int c = 0; c < 8; c++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&words[4 * c])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4 * c) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4 * c) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4 * c) + 3]))
                            : "r"((a_base + (unsigned int)(row * 128 + c * 16 ^ (row * 128 + c * 16 >> 7 & 7) << 4))) : "memory");
                    }
                    float part[2];
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        part[h] = 0.0f;
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            float _dot_0;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                "}\n"
                                : "=f"(_dot_0) : "r"(words[h * 16 + i]), "r"(words[h * 16 + i]), "f"(part[h]));
                            part[h] = _dot_0;
                        }
                        float sub;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(part[h]), "f"(qcomp[0]));
                        float add;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(qsum[0]), "f"(sub));
                        float _sub_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(add), "f"(qsum[0]));
                        float _sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(_sub_2), "f"(sub));
                        qcomp[0] = _sub_1;
                        qsum[0] = add;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(a_empty_addr + (r_stage) * 8);
                    }
                    r_stage += 1;
                    if (r_stage == 4) { r_stage = 0; _phase_a_full_1 ^= 1; }
                    if (st_2 % 2 == 1) {
                        mbarrier_wait_plain_nonuniform(big_full_addr + (rc_stage) * 8, _phase_big_full);
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(taddr + row_base + rc_stage * 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int n_1 = 0; n_1 < 24; n_1++) {
                            float sub_1;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(_tmem_load_0[n_1]), "f"(comp[n_1]));
                            float add_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(acc[n_1]), "f"(sub_1));
                            float _sub_4;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(add_1), "f"(acc[n_1]));
                            float _sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_sub_4), "f"(sub_1));
                            comp[n_1] = _sub_3;
                            acc[n_1] = add_1;
                        }
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                            : "r"(taddr + row_base + (rc_stage * 96 + 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int n_2 = 0; n_2 < 24; n_2++) {
                            float sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_2) : "f"(_tmem_load_1[n_2]), "f"(comp[n_2]));
                            float add_2;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_2) : "f"(acc[n_2]), "f"(sub_2));
                            float _sub_6;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(add_2), "f"(acc[n_2]));
                            float _sub_5;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_sub_6), "f"(sub_2));
                            comp[n_2] = _sub_5;
                            acc[n_2] = add_2;
                        }
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(taddr + row_base + (rc_stage * 96 + 64)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            mbarrier_arrive(big_empty_addr + (rc_stage) * 8);
                        }
                        rc_stage += 1;
                        if (rc_stage == 2) { rc_stage = 0; _phase_big_full ^= 1; }
                        #pragma unroll
                        for (int n_3 = 0; n_3 < 24; n_3++) {
                            float sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_3) : "f"(_tmem_load_2[n_3]), "f"(comp[n_3]));
                            float add_3;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_3) : "f"(acc[n_3]), "f"(sub_3));
                            float _sub_8;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(add_3), "f"(acc[n_3]));
                            float _sub_7;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(_sub_8), "f"(sub_3));
                            comp[n_3] = _sub_7;
                            acc[n_3] = add_3;
                        }
                    }
                }
                float out[24];
                #pragma unroll
                for (int n_4 = 0; n_4 < 24; n_4++) {
                    float _sub_9;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(acc[n_4]), "f"(comp[n_4]));
                    out[n_4] = _sub_9;
                }
                float qout;
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(qout) : "f"(qsum[0]), "f"(qcomp[0]));
                #pragma unroll
                for (int qq = 0; qq < 6; qq++) {
                    asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(spart + (s_off + (long long)(4 * qq))), "f"(out[(4 * qq) + 0]), "f"(out[(4 * qq) + 1]), "f"(out[(4 * qq) + 2]), "f"(out[(4 * qq) + 3]), "l"(0x14F0000000000000ULL) : "memory");
                }
                asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;" :: "l"(qpart + t_row), "f"(qout), "l"(0x14F0000000000000ULL) : "memory");
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
