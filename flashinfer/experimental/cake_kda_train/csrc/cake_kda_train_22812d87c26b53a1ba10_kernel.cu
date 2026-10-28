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
#define TMEM_NCOLS 512
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 128
#define TMEM_ACC_QH_OFFSET 256
#define TMEM_ACC_AV_OFFSET 384
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_W_PANEL_OFF 1024
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_KG_PANEL_OFF 33792
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_W_OP_OFF 1024
#define SMEM_W_OP_STAGE_BYTES 16384
#define SMEM_W_OP_STRIDE 16384
#define SMEM_KG_MN_OFF 33792
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_HS_OFF 66560
#define SMEM_HS_STAGE_BYTES 32768
#define SMEM_HS_STRIDE 32768
#define SMEM_HS_MN_OFF 66560
#define SMEM_HS_MN_STAGE_BYTES 32768
#define SMEM_HS_MN_STRIDE 32768
#define SMEM_BV_OFF 99328
#define SMEM_BV_STAGE_BYTES 16384
#define SMEM_BV_STRIDE 16384
#define SMEM_BV_MN_OFF 99328
#define SMEM_BV_MN_STAGE_BYTES 16384
#define SMEM_BV_MN_STRIDE 16384
#define SMEM_QG_PANEL_OFF 115712
#define SMEM_QG_PANEL_STAGE_BYTES 8192
#define SMEM_QG_PANEL_STRIDE 8192
#define SMEM_QG_OP_OFF 115712
#define SMEM_QG_OP_STAGE_BYTES 16384
#define SMEM_QG_OP_STRIDE 16384
#define SMEM_AQK_OP_OFF 132096
#define SMEM_AQK_OP_STAGE_BYTES 8192
#define SMEM_AQK_OP_STRIDE 8192
#define SMEM_TOTAL 140288
#define THREADS 320
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_22812d87c26b53a1ba10(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int segments, float* h0, unsigned int* flags, float* __restrict__ ht_out, int use_h0, int write_tapes, int write_ht, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap aqk_tma, __nv_bfloat16* __restrict__ o_out, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale, int epoch)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(h0));
    __builtin_assume(__isGlobal(h0));

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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define dec_done_addr (mbar_base + 56)
    #define h_ready_addr (mbar_base + 64)
    #define h_tape_read_addr (mbar_base + 72)
    #define bv_tape_read_addr (mbar_base + 80)
    #define qg_full_addr (mbar_base + 88)
    #define aqk_full_addr (mbar_base + 96)
    #define qg_empty_addr (mbar_base + 104)
    #define aqk_empty_addr (mbar_base + 112)
    #define o_done_addr (mbar_base + 120)
    #define o_drained_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_panel_addr = smem + 1024;
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_panel_addr = smem + 33792;
    __nv_bfloat16* w_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int w_op_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int kg_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_addr = smem + 66560;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int hs_mn_addr = smem + 66560;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_addr = smem + 99328;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bv_mn_addr = smem + 99328;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int qg_panel_addr = smem + 115712;
    __nv_bfloat16* qg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int qg_op_addr = smem + 115712;
    __nv_bfloat16* aqk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int aqk_op_addr = smem + 132096;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // vp_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dec_done: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // h_ready: 1 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            // h_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // bv_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // qg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // qg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // aqk_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // o_drained: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 128;
    const int tmem_acc_qh = taddr + 256;
    const int tmem_acc_av = taddr + 384;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seq = blockIdx.x / num_heads % num_seqs;
            int walk = blockIdx.x / (num_heads * num_seqs);
            int chain = 0;
            int nchain = 0;
            if (segments > 0) {
                nchain = num_seqs / segments;
                chain = seq / segments;
            } else {
                nchain = -segments;
                for (int j = 1; j < nchain; j++) {
                    if (seq >= seq_chunk_start[j]) {
                        chain = chain + 1;
                    }
                }
            }
            int tblk = nchain + 1 + seq + chain;
            int c0 = seq_chunk_start[tblk];
            int nc = seq_chunk_start[tblk + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
            int tapes_on = 0;
            if (walk == 0) {
                if (write_tapes == 1) {
                    tapes_on = 1;
                }
            }
            float init_row[16];
            if (walk == 1) {
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    init_row[j_1] = ((k_row == j_1) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    init_row[j_2] = ((k_row == 16 + j_2) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    init_row[j_3] = ((k_row == 32 + j_3) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    init_row[j_4] = ((k_row == 48 + j_4) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    init_row[j_5] = ((k_row == 64 + j_5) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    init_row[j_6] = ((k_row == 80 + j_6) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    init_row[j_7] = ((k_row == 96 + j_7) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    init_row[j_8] = ((k_row == 112 + j_8) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            } else if (use_h0 == 1) {
                int fbase = (seq * num_heads + head) * 2;
                unsigned int want = (unsigned int)epoch;
                if (elect_sync()) {
                    #pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (fbase + hf))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(want)) >= static_cast<unsigned int>(1));
                        }
                    }
                }
                __syncwarp();
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long h0_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    {
                        unsigned _v4_0_0;
                        unsigned _v4_0_1;
                        unsigned _v4_0_2;
                        unsigned _v4_0_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(h0 + (h0_base + (long long)(q * 4)))) : "memory");
                        init_row[q * 4 + 0] = __uint_as_float(_v4_0_0);
                        init_row[q * 4 + 1] = __uint_as_float(_v4_0_1);
                        init_row[q * 4 + 2] = __uint_as_float(_v4_0_2);
                        init_row[q * 4 + 3] = __uint_as_float(_v4_0_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        unsigned _v4_1_0;
                        unsigned _v4_1_1;
                        unsigned _v4_1_2;
                        unsigned _v4_1_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(h0 + (h0_base + (long long)(16 + q_1 * 4)))) : "memory");
                        init_row[q_1 * 4 + 0] = __uint_as_float(_v4_1_0);
                        init_row[q_1 * 4 + 1] = __uint_as_float(_v4_1_1);
                        init_row[q_1 * 4 + 2] = __uint_as_float(_v4_1_2);
                        init_row[q_1 * 4 + 3] = __uint_as_float(_v4_1_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        unsigned _v4_2_0;
                        unsigned _v4_2_1;
                        unsigned _v4_2_2;
                        unsigned _v4_2_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(h0 + (h0_base + (long long)(32 + q_2 * 4)))) : "memory");
                        init_row[q_2 * 4 + 0] = __uint_as_float(_v4_2_0);
                        init_row[q_2 * 4 + 1] = __uint_as_float(_v4_2_1);
                        init_row[q_2 * 4 + 2] = __uint_as_float(_v4_2_2);
                        init_row[q_2 * 4 + 3] = __uint_as_float(_v4_2_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int q_3 = 0; q_3 < 4; q_3++) {
                    {
                        unsigned _v4_3_0;
                        unsigned _v4_3_1;
                        unsigned _v4_3_2;
                        unsigned _v4_3_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(h0 + (h0_base + (long long)(48 + q_3 * 4)))) : "memory");
                        init_row[q_3 * 4 + 0] = __uint_as_float(_v4_3_0);
                        init_row[q_3 * 4 + 1] = __uint_as_float(_v4_3_1);
                        init_row[q_3 * 4 + 2] = __uint_as_float(_v4_3_2);
                        init_row[q_3 * 4 + 3] = __uint_as_float(_v4_3_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int q_4 = 0; q_4 < 4; q_4++) {
                    {
                        unsigned _v4_4_0;
                        unsigned _v4_4_1;
                        unsigned _v4_4_2;
                        unsigned _v4_4_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_4_0), "=r"(_v4_4_1), "=r"(_v4_4_2), "=r"(_v4_4_3) : "l"((const void*)(h0 + (h0_base + (long long)(64 + q_4 * 4)))) : "memory");
                        init_row[q_4 * 4 + 0] = __uint_as_float(_v4_4_0);
                        init_row[q_4 * 4 + 1] = __uint_as_float(_v4_4_1);
                        init_row[q_4 * 4 + 2] = __uint_as_float(_v4_4_2);
                        init_row[q_4 * 4 + 3] = __uint_as_float(_v4_4_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int q_5 = 0; q_5 < 4; q_5++) {
                    {
                        unsigned _v4_5_0;
                        unsigned _v4_5_1;
                        unsigned _v4_5_2;
                        unsigned _v4_5_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_5_0), "=r"(_v4_5_1), "=r"(_v4_5_2), "=r"(_v4_5_3) : "l"((const void*)(h0 + (h0_base + (long long)(80 + q_5 * 4)))) : "memory");
                        init_row[q_5 * 4 + 0] = __uint_as_float(_v4_5_0);
                        init_row[q_5 * 4 + 1] = __uint_as_float(_v4_5_1);
                        init_row[q_5 * 4 + 2] = __uint_as_float(_v4_5_2);
                        init_row[q_5 * 4 + 3] = __uint_as_float(_v4_5_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int q_6 = 0; q_6 < 4; q_6++) {
                    {
                        unsigned _v4_6_0;
                        unsigned _v4_6_1;
                        unsigned _v4_6_2;
                        unsigned _v4_6_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_6_0), "=r"(_v4_6_1), "=r"(_v4_6_2), "=r"(_v4_6_3) : "l"((const void*)(h0 + (h0_base + (long long)(96 + q_6 * 4)))) : "memory");
                        init_row[q_6 * 4 + 0] = __uint_as_float(_v4_6_0);
                        init_row[q_6 * 4 + 1] = __uint_as_float(_v4_6_1);
                        init_row[q_6 * 4 + 2] = __uint_as_float(_v4_6_2);
                        init_row[q_6 * 4 + 3] = __uint_as_float(_v4_6_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int q_7 = 0; q_7 < 4; q_7++) {
                    {
                        unsigned _v4_7_0;
                        unsigned _v4_7_1;
                        unsigned _v4_7_2;
                        unsigned _v4_7_3;
                        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_7_0), "=r"(_v4_7_1), "=r"(_v4_7_2), "=r"(_v4_7_3) : "l"((const void*)(h0 + (h0_base + (long long)(112 + q_7 * 4)))) : "memory");
                        init_row[q_7 * 4 + 0] = __uint_as_float(_v4_7_0);
                        init_row[q_7 * 4 + 1] = __uint_as_float(_v4_7_1);
                        init_row[q_7 * 4 + 2] = __uint_as_float(_v4_7_2);
                        init_row[q_7 * 4 + 3] = __uint_as_float(_v4_7_3);
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            } else {
                init_row[0] = 0.0f;
                init_row[1] = 0.0f;
                init_row[2] = 0.0f;
                init_row[3] = 0.0f;
                init_row[4] = 0.0f;
                init_row[5] = 0.0f;
                init_row[6] = 0.0f;
                init_row[7] = 0.0f;
                init_row[8] = 0.0f;
                init_row[9] = 0.0f;
                init_row[10] = 0.0f;
                init_row[11] = 0.0f;
                init_row[12] = 0.0f;
                init_row[13] = 0.0f;
                init_row[14] = 0.0f;
                init_row[15] = 0.0f;
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int u_pk[32];
            u_pk[0] = 0;
            u_pk[1] = 0;
            u_pk[2] = 0;
            u_pk[3] = 0;
            u_pk[4] = 0;
            u_pk[5] = 0;
            u_pk[6] = 0;
            u_pk[7] = 0;
            u_pk[8] = 0;
            u_pk[9] = 0;
            u_pk[10] = 0;
            u_pk[11] = 0;
            u_pk[12] = 0;
            u_pk[13] = 0;
            u_pk[14] = 0;
            u_pk[15] = 0;
            u_pk[16] = 0;
            u_pk[17] = 0;
            u_pk[18] = 0;
            u_pk[19] = 0;
            u_pk[20] = 0;
            u_pk[21] = 0;
            u_pk[22] = 0;
            u_pk[23] = 0;
            u_pk[24] = 0;
            u_pk[25] = 0;
            u_pk[26] = 0;
            u_pk[27] = 0;
            u_pk[28] = 0;
            u_pk[29] = 0;
            u_pk[30] = 0;
            u_pk[31] = 0;
            if (nc > 0) {
                if (walk == 0) {
                    long long u_next = ((long long)(row0 + t_row) * (long long)num_heads + (long long)head) * 128;
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 2; q_8++) {
                        {
                            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(lane_half * 16 + q_8 * 8));
                            u_pk[q_8 * 4 + 0] = _uv4_8.x;
                            u_pk[q_8 * 4 + 1] = _uv4_8.y;
                            u_pk[q_8 * 4 + 2] = _uv4_8.z;
                            u_pk[q_8 * 4 + 3] = _uv4_8.w;
                        }
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 2; q_9++) {
                        {
                            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(32 + lane_half * 16 + q_9 * 8));
                            u_pk[8 + q_9 * 4 + 0] = _uv4_9.x;
                            u_pk[8 + q_9 * 4 + 1] = _uv4_9.y;
                            u_pk[8 + q_9 * 4 + 2] = _uv4_9.z;
                            u_pk[8 + q_9 * 4 + 3] = _uv4_9.w;
                        }
                    }
                    #pragma unroll
                    for (int q_10 = 0; q_10 < 2; q_10++) {
                        {
                            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(64 + lane_half * 16 + q_10 * 8));
                            u_pk[16 + q_10 * 4 + 0] = _uv4_10.x;
                            u_pk[16 + q_10 * 4 + 1] = _uv4_10.y;
                            u_pk[16 + q_10 * 4 + 2] = _uv4_10.z;
                            u_pk[16 + q_10 * 4 + 3] = _uv4_10.w;
                        }
                    }
                    #pragma unroll
                    for (int q_11 = 0; q_11 < 2; q_11++) {
                        {
                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(96 + lane_half * 16 + q_11 * 8));
                            u_pk[24 + q_11 * 4 + 0] = _uv4_11.x;
                            u_pk[24 + q_11 * 4 + 1] = _uv4_11.y;
                            u_pk[24 + q_11 * 4 + 2] = _uv4_11.z;
                            u_pk[24 + q_11 * 4 + 3] = _uv4_11.w;
                        }
                    }
                }
            }
            unsigned int _phase_bv_tape_read_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_tape_read_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float _tmem_load_16[16];
                tmem_ld_x16(&_tmem_load_16[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 0], _tmem_load_16[_lp*2+1 + 0]));
                    _tmem_load_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_12 * 8 / 64 * 16384 + k_row * 128 + q_12 * 8 % 64 * 2 ^ (q_12 * 8 / 64 * 16384 + k_row * 128 + q_12 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_16_bf16[q_12 * 4]), "r"(_tmem_load_16_bf16[q_12 * 4 + 1]), "r"(_tmem_load_16_bf16[q_12 * 4 + 2]), "r"(_tmem_load_16_bf16[q_12 * 4 + 3]) : "memory");
                }
                float _tmem_load_17[16];
                tmem_ld_x16(&_tmem_load_17[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_17_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 0], _tmem_load_17[_lp*2+1 + 0]));
                    _tmem_load_17_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_13 * 8) / 64 * 16384 + k_row * 128 + (16 + q_13 * 8) % 64 * 2 ^ ((16 + q_13 * 8) / 64 * 16384 + k_row * 128 + (16 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_17_bf16[q_13 * 4]), "r"(_tmem_load_17_bf16[q_13 * 4 + 1]), "r"(_tmem_load_17_bf16[q_13 * 4 + 2]), "r"(_tmem_load_17_bf16[q_13 * 4 + 3]) : "memory");
                }
                float _tmem_load_18[16];
                tmem_ld_x16(&_tmem_load_18[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_18_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                    _tmem_load_18_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_14 * 8) / 64 * 16384 + k_row * 128 + (32 + q_14 * 8) % 64 * 2 ^ ((32 + q_14 * 8) / 64 * 16384 + k_row * 128 + (32 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[q_14 * 4]), "r"(_tmem_load_18_bf16[q_14 * 4 + 1]), "r"(_tmem_load_18_bf16[q_14 * 4 + 2]), "r"(_tmem_load_18_bf16[q_14 * 4 + 3]) : "memory");
                }
                float _tmem_load_19[16];
                tmem_ld_x16(&_tmem_load_19[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_19_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                    _tmem_load_19_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_15 * 8) / 64 * 16384 + k_row * 128 + (48 + q_15 * 8) % 64 * 2 ^ ((48 + q_15 * 8) / 64 * 16384 + k_row * 128 + (48 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[q_15 * 4]), "r"(_tmem_load_19_bf16[q_15 * 4 + 1]), "r"(_tmem_load_19_bf16[q_15 * 4 + 2]), "r"(_tmem_load_19_bf16[q_15 * 4 + 3]) : "memory");
                }
                float _tmem_load_20[16];
                tmem_ld_x16(&_tmem_load_20[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_20_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_20[_lp*2 + 0], _tmem_load_20[_lp*2+1 + 0]));
                    _tmem_load_20_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_16 = 0; q_16 < 2; q_16++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((64 + q_16 * 8) / 64 * 16384 + k_row * 128 + (64 + q_16 * 8) % 64 * 2 ^ ((64 + q_16 * 8) / 64 * 16384 + k_row * 128 + (64 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_20_bf16[q_16 * 4]), "r"(_tmem_load_20_bf16[q_16 * 4 + 1]), "r"(_tmem_load_20_bf16[q_16 * 4 + 2]), "r"(_tmem_load_20_bf16[q_16 * 4 + 3]) : "memory");
                }
                float _tmem_load_21[16];
                tmem_ld_x16(&_tmem_load_21[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_21_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_21[_lp*2 + 0], _tmem_load_21[_lp*2+1 + 0]));
                    _tmem_load_21_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_17 = 0; q_17 < 2; q_17++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((80 + q_17 * 8) / 64 * 16384 + k_row * 128 + (80 + q_17 * 8) % 64 * 2 ^ ((80 + q_17 * 8) / 64 * 16384 + k_row * 128 + (80 + q_17 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_21_bf16[q_17 * 4]), "r"(_tmem_load_21_bf16[q_17 * 4 + 1]), "r"(_tmem_load_21_bf16[q_17 * 4 + 2]), "r"(_tmem_load_21_bf16[q_17 * 4 + 3]) : "memory");
                }
                float _tmem_load_22[16];
                tmem_ld_x16(&_tmem_load_22[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_22_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_22[_lp*2 + 0], _tmem_load_22[_lp*2+1 + 0]));
                    _tmem_load_22_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_18 = 0; q_18 < 2; q_18++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((96 + q_18 * 8) / 64 * 16384 + k_row * 128 + (96 + q_18 * 8) % 64 * 2 ^ ((96 + q_18 * 8) / 64 * 16384 + k_row * 128 + (96 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_22_bf16[q_18 * 4]), "r"(_tmem_load_22_bf16[q_18 * 4 + 1]), "r"(_tmem_load_22_bf16[q_18 * 4 + 2]), "r"(_tmem_load_22_bf16[q_18 * 4 + 3]) : "memory");
                }
                float _tmem_load_23[16];
                tmem_ld_x16(&_tmem_load_23[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_23_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_23[_lp*2 + 0], _tmem_load_23[_lp*2+1 + 0]));
                    _tmem_load_23_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_19 = 0; q_19 < 2; q_19++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((112 + q_19 * 8) / 64 * 16384 + k_row * 128 + (112 + q_19 * 8) % 64 * 2 ^ ((112 + q_19 * 8) / 64 * 16384 + k_row * 128 + (112 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_23_bf16[q_19 * 4]), "r"(_tmem_load_23_bf16[q_19 * 4 + 1]), "r"(_tmem_load_23_bf16[q_19 * 4 + 2]), "r"(_tmem_load_23_bf16[q_19 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                if (tapes_on == 1) {
                    mbarrier_wait(bv_tape_read_addr, _phase_bv_tape_read_0);
                    _phase_bv_tape_read_0 ^= 1;
                }
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long u_base = ((long long)trow * (long long)num_heads + (long long)head) * 128;
                float _tmem_load_24[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_24[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                float _unpack_bf16_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[_pair]) & 0xffff0000u);
                }
                float bvv[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    bvv[j_9] = _unpack_bf16_0[j_9] - _tmem_load_24[j_9];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_20 = 0; q_20 < 2; q_20++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_20 * 8) / 64 * 8192 + t_row * 128 + (col + q_20 * 8) % 64 * 2 ^ ((col + q_20 * 8) / 64 * 8192 + t_row * 128 + (col + q_20 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_20 * 4]), "r"(bvv_bf16[q_20 * 4 + 1]), "r"(bvv_bf16[q_20 * 4 + 2]), "r"(bvv_bf16[q_20 * 4 + 3]) : "memory");
                }
                float _tmem_load_25[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_25[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_0 = 32 + lane_half * 16;
                float _unpack_bf16_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_1[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    bvv_1[j_10] = _unpack_bf16_1[j_10] - _tmem_load_25[j_10];
                }
                unsigned int bvv_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_1[_lp*2 + 0], bvv_1[_lp*2+1 + 0]));
                    bvv_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_21 = 0; q_21 < 2; q_21++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_0 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_21 * 8) % 64 * 2 ^ ((col_0 + q_21 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_21 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_1_bf16[q_21 * 4]), "r"(bvv_1_bf16[q_21 * 4 + 1]), "r"(bvv_1_bf16[q_21 * 4 + 2]), "r"(bvv_1_bf16[q_21 * 4 + 3]) : "memory");
                }
                float _tmem_load_26[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_26[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_2 = 64 + lane_half * 16;
                float _unpack_bf16_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) & 0xffff0000u);
                }
                float bvv_3[16];
                #pragma unroll
                for (int j_11 = 0; j_11 < 16; j_11++) {
                    bvv_3[j_11] = _unpack_bf16_2[j_11] - _tmem_load_26[j_11];
                }
                unsigned int bvv_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_3[_lp*2 + 0], bvv_3[_lp*2+1 + 0]));
                    bvv_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_22 = 0; q_22 < 2; q_22++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_2 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_22 * 8) % 64 * 2 ^ ((col_2 + q_22 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_22 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_3_bf16[q_22 * 4]), "r"(bvv_3_bf16[q_22 * 4 + 1]), "r"(bvv_3_bf16[q_22 * 4 + 2]), "r"(bvv_3_bf16[q_22 * 4 + 3]) : "memory");
                }
                float _tmem_load_27[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_27[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_4 = 96 + lane_half * 16;
                float _unpack_bf16_3[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) & 0xffff0000u);
                }
                float bvv_5[16];
                #pragma unroll
                for (int j_12 = 0; j_12 < 16; j_12++) {
                    bvv_5[j_12] = _unpack_bf16_3[j_12] - _tmem_load_27[j_12];
                }
                unsigned int bvv_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_5[_lp*2 + 0], bvv_5[_lp*2+1 + 0]));
                    bvv_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_23 = 0; q_23 < 2; q_23++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_4 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_23 * 8) % 64 * 2 ^ ((col_4 + q_23 * 8) / 64 * 8192 + t_row * 128 + (col_4 + q_23 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_5_bf16[q_23 * 4]), "r"(bvv_5_bf16[q_23 * 4 + 1]), "r"(bvv_5_bf16[q_23 * 4 + 2]), "r"(bvv_5_bf16[q_23 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    if (walk == 0) {
                        long long u_next_1 = u_base + (long long)(64 * num_heads * 128);
                        #pragma unroll
                        for (int q_24 = 0; q_24 < 2; q_24++) {
                            {
                                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(lane_half * 16 + q_24 * 8));
                                u_pk[q_24 * 4 + 0] = _uv4_12.x;
                                u_pk[q_24 * 4 + 1] = _uv4_12.y;
                                u_pk[q_24 * 4 + 2] = _uv4_12.z;
                                u_pk[q_24 * 4 + 3] = _uv4_12.w;
                            }
                        }
                        #pragma unroll
                        for (int q_25 = 0; q_25 < 2; q_25++) {
                            {
                                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(32 + lane_half * 16 + q_25 * 8));
                                u_pk[8 + q_25 * 4 + 0] = _uv4_13.x;
                                u_pk[8 + q_25 * 4 + 1] = _uv4_13.y;
                                u_pk[8 + q_25 * 4 + 2] = _uv4_13.z;
                                u_pk[8 + q_25 * 4 + 3] = _uv4_13.w;
                            }
                        }
                        #pragma unroll
                        for (int q_26 = 0; q_26 < 2; q_26++) {
                            {
                                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(64 + lane_half * 16 + q_26 * 8));
                                u_pk[16 + q_26 * 4 + 0] = _uv4_14.x;
                                u_pk[16 + q_26 * 4 + 1] = _uv4_14.y;
                                u_pk[16 + q_26 * 4 + 2] = _uv4_14.z;
                                u_pk[16 + q_26 * 4 + 3] = _uv4_14.w;
                            }
                        }
                        #pragma unroll
                        for (int q_27 = 0; q_27 < 2; q_27++) {
                            {
                                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(96 + lane_half * 16 + q_27 * 8));
                                u_pk[24 + q_27 * 4 + 0] = _uv4_15.x;
                                u_pk[24 + q_27 * 4 + 1] = _uv4_15.y;
                                u_pk[24 + q_27 * 4 + 2] = _uv4_15.z;
                                u_pk[24 + q_27 * 4 + 3] = _uv4_15.w;
                            }
                        }
                    }
                }
                if (tapes_on == 1) {
                    mbarrier_wait(h_tape_read_addr, _phase_h_tape_read_0);
                    _phase_h_tape_read_0 ^= 1;
                }
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (write_ht == 1) {
                long long ht_base = (((long long)walk * (long long)num_seqs + (long long)seq) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                float _tmem_load_28[16];
                tmem_ld_x16(&_tmem_load_28[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_28 = 0; q_28 < 4; q_28++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_28[q_28 * 4 + 0], _tmem_load_28[q_28 * 4 + 1], _tmem_load_28[q_28 * 4 + 2], _tmem_load_28[q_28 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(q_28 * 4)) = _v4;
                    }
                }
                float _tmem_load_29[16];
                tmem_ld_x16(&_tmem_load_29[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_29 = 0; q_29 < 4; q_29++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_29[q_29 * 4 + 0], _tmem_load_29[q_29 * 4 + 1], _tmem_load_29[q_29 * 4 + 2], _tmem_load_29[q_29 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(16 + q_29 * 4)) = _v4;
                    }
                }
                float _tmem_load_30[16];
                tmem_ld_x16(&_tmem_load_30[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_30 = 0; q_30 < 4; q_30++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_30[q_30 * 4 + 0], _tmem_load_30[q_30 * 4 + 1], _tmem_load_30[q_30 * 4 + 2], _tmem_load_30[q_30 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(32 + q_30 * 4)) = _v4;
                    }
                }
                float _tmem_load_31[16];
                tmem_ld_x16(&_tmem_load_31[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_31 = 0; q_31 < 4; q_31++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_31[q_31 * 4 + 0], _tmem_load_31[q_31 * 4 + 1], _tmem_load_31[q_31 * 4 + 2], _tmem_load_31[q_31 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(48 + q_31 * 4)) = _v4;
                    }
                }
                float _tmem_load_32[16];
                tmem_ld_x16(&_tmem_load_32[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_32 = 0; q_32 < 4; q_32++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_32[q_32 * 4 + 0], _tmem_load_32[q_32 * 4 + 1], _tmem_load_32[q_32 * 4 + 2], _tmem_load_32[q_32 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(64 + q_32 * 4)) = _v4;
                    }
                }
                float _tmem_load_33[16];
                tmem_ld_x16(&_tmem_load_33[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_33 = 0; q_33 < 4; q_33++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_33[q_33 * 4 + 0], _tmem_load_33[q_33 * 4 + 1], _tmem_load_33[q_33 * 4 + 2], _tmem_load_33[q_33 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(80 + q_33 * 4)) = _v4;
                    }
                }
                float _tmem_load_34[16];
                tmem_ld_x16(&_tmem_load_34[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_34 = 0; q_34 < 4; q_34++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_34[q_34 * 4 + 0], _tmem_load_34[q_34 * 4 + 1], _tmem_load_34[q_34 * 4 + 2], _tmem_load_34[q_34 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(96 + q_34 * 4)) = _v4;
                    }
                }
                float _tmem_load_35[16];
                tmem_ld_x16(&_tmem_load_35[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_35 = 0; q_35 < 4; q_35++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_35[q_35 * 4 + 0], _tmem_load_35[q_35 * 4 + 1], _tmem_load_35[q_35 * 4 + 2], _tmem_load_35[q_35 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(112 + q_35 * 4)) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
            }
        }
    }
    // ---- Role: decayw ----
    if (warp >= 4 && warp <= 7) {
        { // decayw_main
            int head_1 = blockIdx.x % num_heads;
            int seq_1 = blockIdx.x / num_heads % num_seqs;
            int walk_1 = blockIdx.x / (num_heads * num_seqs);
            int chain_1 = 0;
            int nchain_1 = 0;
            if (segments > 0) {
                nchain_1 = num_seqs / segments;
                chain_1 = seq_1 / segments;
            } else {
                nchain_1 = -segments;
                for (int j_13 = 1; j_13 < nchain_1; j_13++) {
                    if (seq_1 >= seq_chunk_start[j_13]) {
                        chain_1 = chain_1 + 1;
                    }
                }
            }
            int tblk_1 = nchain_1 + 1 + seq_1 + chain_1;
            int c0_1 = seq_chunk_start[tblk_1];
            int nc_1 = seq_chunk_start[tblk_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            int lane_half_1 = lane >> 4;
            int t_row_1 = warp_in_wg_1 * 16 + (lane & 15);
            int o_row_base = warp_in_wg_1 * 32 << 16;
            int ext_next = 0;
            int clen_next = 0;
            if (nc_1 > 0) {
                ext_next = chunk_bos[c0_1];
                clen_next = chunk_len[c0_1];
            }
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_o_done_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2[16];
                #pragma unroll
                for (int j_14 = 0; j_14 < 16; j_14++) {
                    hs2[j_14] = _tmem_load_0[j_14] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_0[16];
                #pragma unroll
                for (int j_15 = 0; j_15 < 16; j_15++) {
                    hs2_0[j_15] = _tmem_load_1[j_15] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_1[16];
                #pragma unroll
                for (int j_16 = 0; j_16 < 16; j_16++) {
                    hs2_1[j_16] = _tmem_load_2[j_16] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_2[16];
                #pragma unroll
                for (int j_17 = 0; j_17 < 16; j_17++) {
                    hs2_2[j_17] = _tmem_load_3[j_17] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_2);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base_1 + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3[16];
                #pragma unroll
                for (int j_18 = 0; j_18 < 16; j_18++) {
                    hs2_3[j_18] = _tmem_load_4[j_18] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 64, hs2_3);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)h_row_base_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_4[16];
                #pragma unroll
                for (int j_19 = 0; j_19 < 16; j_19++) {
                    hs2_4[j_19] = _tmem_load_5[j_19] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 80, hs2_4);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_5[16];
                #pragma unroll
                for (int j_20 = 0; j_20 < 16; j_20++) {
                    hs2_5[j_20] = _tmem_load_6[j_20] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 96, hs2_5);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_6[16];
                #pragma unroll
                for (int j_21 = 0; j_21 < 16; j_21++) {
                    hs2_6[j_21] = _tmem_load_7[j_21] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 112, hs2_6);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
                int ext0 = ext_next;
                int clen = clen_next;
                if (nc_1 > c_1 + 1) {
                    ext_next = chunk_bos[c0_1 + c_1 + 1];
                    clen_next = chunk_len[c0_1 + c_1 + 1];
                }
                long long obase = ((long long)(ext0 + t_row_1) * (long long)num_heads + (long long)head_1) * 128;
                mbarrier_wait(o_done_addr, _phase_o_done_0);
                _phase_o_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int col_1 = lane_half_1 * 16;
                float _tmem_load_8[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                    : "r"(taddr + 256 + (unsigned int)o_row_base));
                float _tmem_load_9[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                    : "r"(taddr + 384 + (unsigned int)o_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov[16];
                #pragma unroll
                for (int j_22 = 0; j_22 < 16; j_22++) {
                    ov[j_22] = __fmaf_rn(_tmem_load_8[j_22], scale, _tmem_load_9[j_22]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 0], ov[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 2], ov[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 4], ov[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 6], ov[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 8], ov[_lp*2+1 + 8]));
                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 10], ov[_lp*2+1 + 10]));
                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 12], ov[_lp*2+1 + 12]));
                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov[_lp*2 + 14], ov[_lp*2+1 + 14]));
                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_1))), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                    }
                }
                int col_7 = 32 + lane_half_1 * 16;
                float _tmem_load_10[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                    : "r"(taddr + 256 + (unsigned int)o_row_base + 32));
                float _tmem_load_11[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                    : "r"(taddr + 384 + (unsigned int)o_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_8[16];
                #pragma unroll
                for (int j_23 = 0; j_23 < 16; j_23++) {
                    ov_8[j_23] = __fmaf_rn(_tmem_load_10[j_23], scale, _tmem_load_11[j_23]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 0], ov_8[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 2], ov_8[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 4], ov_8[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 6], ov_8[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 8], ov_8[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 10], ov_8[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 12], ov_8[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_8[_lp*2 + 14], ov_8[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_7))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                    }
                }
                int col_9 = 64 + lane_half_1 * 16;
                float _tmem_load_12[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                    : "r"(taddr + 256 + (unsigned int)o_row_base + 64));
                float _tmem_load_13[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                    : "r"(taddr + 384 + (unsigned int)o_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_10[16];
                #pragma unroll
                for (int j_24 = 0; j_24 < 16; j_24++) {
                    ov_10[j_24] = __fmaf_rn(_tmem_load_12[j_24], scale, _tmem_load_13[j_24]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 0], ov_10[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 2], ov_10[_lp*2+1 + 2]));
                        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 4], ov_10[_lp*2+1 + 4]));
                        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 6], ov_10[_lp*2+1 + 6]));
                        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 8], ov_10[_lp*2+1 + 8]));
                        _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 10], ov_10[_lp*2+1 + 10]));
                        _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 12], ov_10[_lp*2+1 + 12]));
                        _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_10[_lp*2 + 14], ov_10[_lp*2+1 + 14]));
                        _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(_packed_2);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_9))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                    }
                }
                int col_11 = 96 + lane_half_1 * 16;
                float _tmem_load_14[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                    : "r"(taddr + 256 + (unsigned int)o_row_base + 96));
                float _tmem_load_15[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                    : "r"(taddr + 384 + (unsigned int)o_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ov_12[16];
                #pragma unroll
                for (int j_25 = 0; j_25 < 16; j_25++) {
                    ov_12[j_25] = __fmaf_rn(_tmem_load_14[j_25], scale, _tmem_load_15[j_25]);
                }
                if (t_row_1 < clen) {
                    unsigned int _packed_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 0], ov_12[_lp*2+1 + 0]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 2], ov_12[_lp*2+1 + 2]));
                        _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 4], ov_12[_lp*2+1 + 4]));
                        _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 6], ov_12[_lp*2+1 + 6]));
                        _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 8], ov_12[_lp*2+1 + 8]));
                        _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 10], ov_12[_lp*2+1 + 10]));
                        _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 12], ov_12[_lp*2+1 + 12]));
                        _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ov_12[_lp*2 + 14], ov_12[_lp*2+1 + 14]));
                        _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_3);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(o_out + (obase + (long long)col_11))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_drained_addr);
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int head_2 = blockIdx.x % num_heads;
            int seq_2 = blockIdx.x / num_heads % num_seqs;
            int walk_2 = blockIdx.x / (num_heads * num_seqs);
            int chain_2 = 0;
            int nchain_2 = 0;
            if (segments > 0) {
                nchain_2 = num_seqs / segments;
                chain_2 = seq_2 / segments;
            } else {
                nchain_2 = -segments;
                for (int j_26 = 1; j_26 < nchain_2; j_26++) {
                    if (seq_2 >= seq_chunk_start[j_26]) {
                        chain_2 = chain_2 + 1;
                    }
                }
            }
            int tblk_2 = nchain_2 + 1 + seq_2 + chain_2;
            int c0_2 = seq_chunk_start[tblk_2];
            int nc_2 = seq_chunk_start[tblk_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int tapes_on_1 = 0;
            if (walk_2 == 0) {
                if (write_tapes == 1) {
                    tapes_on_1 = 1;
                }
            }
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_qg_empty_0 = 1;
            unsigned int _phase_aqk_empty_0 = 1;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                if (nc_2 > 0) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                    int trow_1 = row0_2;
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&w_tma), seg * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&kg_tma), seg * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                    _phase_qg_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                    int trow_0 = row0_2;
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)seg_1 * 8192, (&qg_tma), seg_1 * 64, head_2, trow_0, qg_full_addr);
                    }
                    if (nc_2 > 1) {
                        int trow2 = row0_2 + 64;
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(seg_2 * 64)), "r"((int)(head_2)), "r"((int)(trow2)) : "memory");
                        }
                    }
                    mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                    _phase_aqk_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    int trow_1_1 = row0_2;
                    tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, trow_1_1, aqk_full_addr);
                    if (nc_2 > 1) {
                        int trow2_1 = row0_2 + 64;
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(trow2_1)) : "memory");
                    }
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        int trow_2 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg_3) * 8192, (&w_tma), seg_3 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_3) * 8192, (&kg_tma), seg_3 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (tapes_on_1 == 1) {
                        int chunk = (c0_2 + c_2) * num_heads + head_2;
                        mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                        _phase_hs_ready_0_1 ^= 1;
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            tma_store_3d((&h_out), p * 64, 0, chunk, hs_addr + (unsigned int)(p * 16384));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(qg_empty_addr, _phase_qg_empty_0);
                        _phase_qg_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                        int trow_3 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                            tma_3d_gmem2smem(qg_panel_addr + (unsigned int)seg_4 * 8192, (&qg_tma), seg_4 * 64, head_2, trow_3, qg_full_addr);
                        }
                        if (nc_2 > c_2 + 1 + 1) {
                            int trow2_2 = row0_2 + (c_2 + 1 + 1) * 64;
                            #pragma unroll
                            for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&qg_tma))), "r"((int)(seg_5 * 64)), "r"((int)(head_2)), "r"((int)(trow2_2)) : "memory");
                            }
                        }
                    }
                    if (tapes_on_1 == 1) {
                        int trow_4 = row0_2 + c_2 * 64;
                        mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                        _phase_bv_ready_0 ^= 1;
                        #pragma unroll
                        for (int p_1 = 0; p_1 < 2; p_1++) {
                            tma_store_3d((&v_new), p_1 * 64, head_2, trow_4, bv_addr + (unsigned int)(p_1 * 8192));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        mbarrier_arrive(h_tape_read_addr);
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(bv_tape_read_addr);
                    }
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(aqk_empty_addr, _phase_aqk_empty_0);
                        _phase_aqk_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                        int trow_5 = row0_2 + (c_2 + 1) * 64;
                        tma_3d_gmem2smem(aqk_op_addr, (&aqk_tma), 0, head_2, trow_5, aqk_full_addr);
                        if (nc_2 > c_2 + 1 + 1) {
                            int trow2_3 = row0_2 + (c_2 + 1 + 1) * 64;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&aqk_tma))), "r"((int)(0)), "r"((int)(head_2)), "r"((int)(trow2_3)) : "memory");
                        }
                    }
                }
                if (tapes_on_1 == 1) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int head_3 = blockIdx.x % num_heads;
            int seq_3 = blockIdx.x / num_heads % num_seqs;
            int walk_3 = blockIdx.x / (num_heads * num_seqs);
            int chain_3 = 0;
            int nchain_3 = 0;
            if (segments > 0) {
                nchain_3 = num_seqs / segments;
                chain_3 = seq_3 / segments;
            } else {
                nchain_3 = -segments;
                for (int j_27 = 1; j_27 < nchain_3; j_27++) {
                    if (seq_3 >= seq_chunk_start[j_27]) {
                        chain_3 = chain_3 + 1;
                    }
                }
            }
            int tblk_3 = nchain_3 + 1 + seq_3 + chain_3;
            int c0_3 = seq_chunk_start[tblk_3];
            int nc_3 = seq_chunk_start[tblk_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hs_ready_0_2 = 0;
            unsigned int _phase_qg_full_0 = 0;
            unsigned int _phase_o_drained_0 = 1;
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0 = 0;
            unsigned int _phase_aqk_full_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _mma_b_lo_0 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _mma_a_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _mma_b_lo_0 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(qg_full_addr, _phase_qg_full_0);
                    _phase_qg_full_0 ^= 1;
                    mbarrier_wait(o_drained_addr, _phase_o_drained_0);
                    _phase_o_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((qg_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_0 = ((qg_op_addr) >> 4) & 0x3FFF;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_qh), "r"(0) : "memory");
                    tcgen05_commit(qg_empty_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_1 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_1 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _mma_a_lo_1 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_1 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
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
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = ((aqk_op_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_1 = ((aqk_op_addr) >> 4) & 0x3FFF;
#endif
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
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_av), "r"(0) : "memory");
                    tcgen05_commit(aqk_empty_addr);
                    tcgen05_commit(o_done_addr);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
