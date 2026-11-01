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
#define TMEM_NCOLS 256
#define TMEM_ACC_H_OFFSET 0
#define TMEM_ACC_VP_OFFSET 128
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
#define SMEM_TOTAL 115712
#define THREADS 320
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


extern "C" {

__global__ __launch_bounds__(320, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_0e5fa2fe9ae292c3e3aa(const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap kg_tma, __nv_bfloat16* __restrict__ u, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int segments, float* __restrict__ h0, float* __restrict__ eye, float* __restrict__ ht_out, int use_h0, int write_tapes, int write_ht)
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
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hs_ready_addr (mbar_base + 32)
    #define vp_ready_addr (mbar_base + 40)
    #define bv_ready_addr (mbar_base + 48)
    #define dec_done_addr (mbar_base + 56)
    #define h_ready_addr (mbar_base + 64)
    #define h_tape_read_addr (mbar_base + 72)
    #define bv_tape_read_addr (mbar_base + 80)

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

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seq = blockIdx.x / num_heads % num_seqs;
            int walk = blockIdx.x / (num_heads * num_seqs);
            int chain = 0;
            int nchain = 0;
            if (segments > 0) {
                chain = seq / segments;
                nchain = num_seqs / segments;
            } else {
                nchain = -segments;
                chain = 0;
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
                for (int j_0 = 0; j_0 < 16; j_0++) {
                    init_row[j_0] = ((k_row == 16 + j_0) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    init_row[j_2] = ((k_row == 32 + j_2) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    init_row[j_3] = ((k_row == 48 + j_3) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    init_row[j_4] = ((k_row == 64 + j_4) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    init_row[j_5] = ((k_row == 80 + j_5) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    init_row[j_6] = ((k_row == 96 + j_6) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    init_row[j_7] = ((k_row == 112 + j_7) ? 1.0f : 0.0f);
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 112, init_row);
            } else if (use_h0 == 1) {
                long long h0_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(q * 4));
                        init_row[q * 4 + 0] = _v4.x;
                        init_row[q * 4 + 1] = _v4.y;
                        init_row[q * 4 + 2] = _v4.z;
                        init_row[q * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base, init_row);
                #pragma unroll
                for (int q_0 = 0; q_0 < 4; q_0++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(16 + q_0 * 4));
                        init_row[q_0 * 4 + 0] = _v4.x;
                        init_row[q_0 * 4 + 1] = _v4.y;
                        init_row[q_0 * 4 + 2] = _v4.z;
                        init_row[q_0 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(32 + q_1 * 4));
                        init_row[q_1 * 4 + 0] = _v4.x;
                        init_row[q_1 * 4 + 1] = _v4.y;
                        init_row[q_1 * 4 + 2] = _v4.z;
                        init_row[q_1 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(48 + q_2 * 4));
                        init_row[q_2 * 4 + 0] = _v4.x;
                        init_row[q_2 * 4 + 1] = _v4.y;
                        init_row[q_2 * 4 + 2] = _v4.z;
                        init_row[q_2 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, init_row);
                #pragma unroll
                for (int q_3 = 0; q_3 < 4; q_3++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(64 + q_3 * 4));
                        init_row[q_3 * 4 + 0] = _v4.x;
                        init_row[q_3 * 4 + 1] = _v4.y;
                        init_row[q_3 * 4 + 2] = _v4.z;
                        init_row[q_3 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 64, init_row);
                #pragma unroll
                for (int q_4 = 0; q_4 < 4; q_4++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(80 + q_4 * 4));
                        init_row[q_4 * 4 + 0] = _v4.x;
                        init_row[q_4 * 4 + 1] = _v4.y;
                        init_row[q_4 * 4 + 2] = _v4.z;
                        init_row[q_4 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 80, init_row);
                #pragma unroll
                for (int q_5 = 0; q_5 < 4; q_5++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(96 + q_5 * 4));
                        init_row[q_5 * 4 + 0] = _v4.x;
                        init_row[q_5 * 4 + 1] = _v4.y;
                        init_row[q_5 * 4 + 2] = _v4.z;
                        init_row[q_5 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 96, init_row);
                #pragma unroll
                for (int q_6 = 0; q_6 < 4; q_6++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(h0 + h0_base + (long long)(112 + q_6 * 4));
                        init_row[q_6 * 4 + 0] = _v4.x;
                        init_row[q_6 * 4 + 1] = _v4.y;
                        init_row[q_6 * 4 + 2] = _v4.z;
                        init_row[q_6 * 4 + 3] = _v4.w;
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
                    for (int q_7 = 0; q_7 < 2; q_7++) {
                        {
                            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(lane_half * 16 + q_7 * 8));
                            u_pk[q_7 * 4 + 0] = _uv4_8.x;
                            u_pk[q_7 * 4 + 1] = _uv4_8.y;
                            u_pk[q_7 * 4 + 2] = _uv4_8.z;
                            u_pk[q_7 * 4 + 3] = _uv4_8.w;
                        }
                    }
                    #pragma unroll
                    for (int q_7_1 = 0; q_7_1 < 2; q_7_1++) {
                        {
                            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(32 + lane_half * 16 + q_7_1 * 8));
                            u_pk[8 + q_7_1 * 4 + 0] = _uv4_9.x;
                            u_pk[8 + q_7_1 * 4 + 1] = _uv4_9.y;
                            u_pk[8 + q_7_1 * 4 + 2] = _uv4_9.z;
                            u_pk[8 + q_7_1 * 4 + 3] = _uv4_9.w;
                        }
                    }
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 2; q_8++) {
                        {
                            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(64 + lane_half * 16 + q_8 * 8));
                            u_pk[16 + q_8 * 4 + 0] = _uv4_10.x;
                            u_pk[16 + q_8 * 4 + 1] = _uv4_10.y;
                            u_pk[16 + q_8 * 4 + 2] = _uv4_10.z;
                            u_pk[16 + q_8 * 4 + 3] = _uv4_10.w;
                        }
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 2; q_9++) {
                        {
                            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(u + u_next + (long long)(96 + lane_half * 16 + q_9 * 8));
                            u_pk[24 + q_9 * 4 + 0] = _uv4_11.x;
                            u_pk[24 + q_9 * 4 + 1] = _uv4_11.y;
                            u_pk[24 + q_9 * 4 + 2] = _uv4_11.z;
                            u_pk[24 + q_9 * 4 + 3] = _uv4_11.w;
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
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                    _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q_10 * 8 / 64 * 16384 + k_row * 128 + q_10 * 8 % 64 * 2 ^ (q_10 * 8 / 64 * 16384 + k_row * 128 + q_10 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[q_10 * 4]), "r"(_tmem_load_8_bf16[q_10 * 4 + 1]), "r"(_tmem_load_8_bf16[q_10 * 4 + 2]), "r"(_tmem_load_8_bf16[q_10 * 4 + 3]) : "memory");
                }
                float tmem_load_0[16];
                tmem_ld_x16(&tmem_load_0[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_0[_lp*2 + 0], tmem_load_0[_lp*2+1 + 0]));
                    tmem_load_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10_1 = 0; q_10_1 < 2; q_10_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_10_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_10_1 * 8) % 64 * 2 ^ ((16 + q_10_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_10_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_0[q_10_1 * 4]), "r"(tmem_load_bf16_0[q_10_1 * 4 + 1]), "r"(tmem_load_bf16_0[q_10_1 * 4 + 2]), "r"(tmem_load_bf16_0[q_10_1 * 4 + 3]) : "memory");
                }
                float tmem_load_1[16];
                tmem_ld_x16(&tmem_load_1[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_1[_lp*2 + 0], tmem_load_1[_lp*2+1 + 0]));
                    tmem_load_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_11 * 8) / 64 * 16384 + k_row * 128 + (32 + q_11 * 8) % 64 * 2 ^ ((32 + q_11 * 8) / 64 * 16384 + k_row * 128 + (32 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_1[q_11 * 4]), "r"(tmem_load_bf16_1[q_11 * 4 + 1]), "r"(tmem_load_bf16_1[q_11 * 4 + 2]), "r"(tmem_load_bf16_1[q_11 * 4 + 3]) : "memory");
                }
                float tmem_load_2[16];
                tmem_ld_x16(&tmem_load_2[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_2[_lp*2 + 0], tmem_load_2[_lp*2+1 + 0]));
                    tmem_load_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_12 * 8) / 64 * 16384 + k_row * 128 + (48 + q_12 * 8) % 64 * 2 ^ ((48 + q_12 * 8) / 64 * 16384 + k_row * 128 + (48 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_2[q_12 * 4]), "r"(tmem_load_bf16_2[q_12 * 4 + 1]), "r"(tmem_load_bf16_2[q_12 * 4 + 2]), "r"(tmem_load_bf16_2[q_12 * 4 + 3]) : "memory");
                }
                float tmem_load_3[16];
                tmem_ld_x16(&tmem_load_3[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_3[_lp*2 + 0], tmem_load_3[_lp*2+1 + 0]));
                    tmem_load_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((64 + q_13 * 8) / 64 * 16384 + k_row * 128 + (64 + q_13 * 8) % 64 * 2 ^ ((64 + q_13 * 8) / 64 * 16384 + k_row * 128 + (64 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_3[q_13 * 4]), "r"(tmem_load_bf16_3[q_13 * 4 + 1]), "r"(tmem_load_bf16_3[q_13 * 4 + 2]), "r"(tmem_load_bf16_3[q_13 * 4 + 3]) : "memory");
                }
                float tmem_load_4[16];
                tmem_ld_x16(&tmem_load_4[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_4[_lp*2 + 0], tmem_load_4[_lp*2+1 + 0]));
                    tmem_load_bf16_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((80 + q_14 * 8) / 64 * 16384 + k_row * 128 + (80 + q_14 * 8) % 64 * 2 ^ ((80 + q_14 * 8) / 64 * 16384 + k_row * 128 + (80 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_4[q_14 * 4]), "r"(tmem_load_bf16_4[q_14 * 4 + 1]), "r"(tmem_load_bf16_4[q_14 * 4 + 2]), "r"(tmem_load_bf16_4[q_14 * 4 + 3]) : "memory");
                }
                float tmem_load_5[16];
                tmem_ld_x16(&tmem_load_5[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_5[_lp*2 + 0], tmem_load_5[_lp*2+1 + 0]));
                    tmem_load_bf16_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((96 + q_15 * 8) / 64 * 16384 + k_row * 128 + (96 + q_15 * 8) % 64 * 2 ^ ((96 + q_15 * 8) / 64 * 16384 + k_row * 128 + (96 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_5[q_15 * 4]), "r"(tmem_load_bf16_5[q_15 * 4 + 1]), "r"(tmem_load_bf16_5[q_15 * 4 + 2]), "r"(tmem_load_bf16_5[q_15 * 4 + 3]) : "memory");
                }
                float tmem_load_6[16];
                tmem_ld_x16(&tmem_load_6[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_6[_lp*2 + 0], tmem_load_6[_lp*2+1 + 0]));
                    tmem_load_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_16 = 0; q_16 < 2; q_16++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((112 + q_16 * 8) / 64 * 16384 + k_row * 128 + (112 + q_16 * 8) % 64 * 2 ^ ((112 + q_16 * 8) / 64 * 16384 + k_row * 128 + (112 + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_6[q_16 * 4]), "r"(tmem_load_bf16_6[q_16 * 4 + 1]), "r"(tmem_load_bf16_6[q_16 * 4 + 2]), "r"(tmem_load_bf16_6[q_16 * 4 + 3]) : "memory");
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
                float _tmem_load_16[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
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
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    bvv[j_8] = _unpack_bf16_0[j_8] - _tmem_load_16[j_8];
                }
                unsigned int bvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv[_lp*2 + 0], bvv[_lp*2+1 + 0]));
                    bvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_17 = 0; q_17 < 2; q_17++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col + q_17 * 8) / 64 * 8192 + t_row * 128 + (col + q_17 * 8) % 64 * 2 ^ ((col + q_17 * 8) / 64 * 8192 + t_row * 128 + (col + q_17 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16[q_17 * 4]), "r"(bvv_bf16[q_17 * 4 + 1]), "r"(bvv_bf16[q_17 * 4 + 2]), "r"(bvv_bf16[q_17 * 4 + 3]) : "memory");
                }
                float tmem_load_r_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_0[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_r_0 = 32 + lane_half * 16;
                float unpack_bf16_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) << 16);
                    (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[8 + _pair]) & 0xffff0000u);
                }
                float bvv_r_0[16];
                #pragma unroll
                for (int j_8_1 = 0; j_8_1 < 16; j_8_1++) {
                    bvv_r_0[j_8_1] = unpack_bf16_0[j_8_1] - tmem_load_r_0[j_8_1];
                }
                unsigned int bvv_bf16_r_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_r_0[_lp*2 + 0], bvv_r_0[_lp*2+1 + 0]));
                    bvv_bf16_r_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_17_1 = 0; q_17_1 < 2; q_17_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_r_0 + q_17_1 * 8) / 64 * 8192 + t_row * 128 + (col_r_0 + q_17_1 * 8) % 64 * 2 ^ ((col_r_0 + q_17_1 * 8) / 64 * 8192 + t_row * 128 + (col_r_0 + q_17_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_r_0[q_17_1 * 4]), "r"(bvv_bf16_r_0[q_17_1 * 4 + 1]), "r"(bvv_bf16_r_0[q_17_1 * 4 + 2]), "r"(bvv_bf16_r_0[q_17_1 * 4 + 3]) : "memory");
                }
                float tmem_load_r_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_1[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_r_1 = 64 + lane_half * 16;
                float unpack_bf16_1[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) << 16);
                    (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[16 + _pair]) & 0xffff0000u);
                }
                float bvv_r_1[16];
                #pragma unroll
                for (int j_9 = 0; j_9 < 16; j_9++) {
                    bvv_r_1[j_9] = unpack_bf16_1[j_9] - tmem_load_r_1[j_9];
                }
                unsigned int bvv_bf16_r_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_r_1[_lp*2 + 0], bvv_r_1[_lp*2+1 + 0]));
                    bvv_bf16_r_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_18 = 0; q_18 < 2; q_18++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_r_1 + q_18 * 8) / 64 * 8192 + t_row * 128 + (col_r_1 + q_18 * 8) % 64 * 2 ^ ((col_r_1 + q_18 * 8) / 64 * 8192 + t_row * 128 + (col_r_1 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_r_1[q_18 * 4]), "r"(bvv_bf16_r_1[q_18 * 4 + 1]), "r"(bvv_bf16_r_1[q_18 * 4 + 2]), "r"(bvv_bf16_r_1[q_18 * 4 + 3]) : "memory");
                }
                float tmem_load_r_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_r_2[15]))
                    : "r"(taddr + 128 + (unsigned int)vp_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_r_2 = 96 + lane_half * 16;
                float unpack_bf16_2[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) << 16);
                    (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(u_pk[24 + _pair]) & 0xffff0000u);
                }
                float bvv_r_2[16];
                #pragma unroll
                for (int j_10 = 0; j_10 < 16; j_10++) {
                    bvv_r_2[j_10] = unpack_bf16_2[j_10] - tmem_load_r_2[j_10];
                }
                unsigned int bvv_bf16_r_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_r_2[_lp*2 + 0], bvv_r_2[_lp*2+1 + 0]));
                    bvv_bf16_r_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_19 = 0; q_19 < 2; q_19++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_r_2 + q_19 * 8) / 64 * 8192 + t_row * 128 + (col_r_2 + q_19 * 8) % 64 * 2 ^ ((col_r_2 + q_19 * 8) / 64 * 8192 + t_row * 128 + (col_r_2 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_r_2[q_19 * 4]), "r"(bvv_bf16_r_2[q_19 * 4 + 1]), "r"(bvv_bf16_r_2[q_19 * 4 + 2]), "r"(bvv_bf16_r_2[q_19 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                if (nc > c + 1) {
                    if (walk == 0) {
                        long long u_next_1 = u_base + (long long)(64 * num_heads * 128);
                        #pragma unroll
                        for (int q_20 = 0; q_20 < 2; q_20++) {
                            {
                                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(lane_half * 16 + q_20 * 8));
                                u_pk[q_20 * 4 + 0] = _uv4_12.x;
                                u_pk[q_20 * 4 + 1] = _uv4_12.y;
                                u_pk[q_20 * 4 + 2] = _uv4_12.z;
                                u_pk[q_20 * 4 + 3] = _uv4_12.w;
                            }
                        }
                        #pragma unroll
                        for (int q_20_1 = 0; q_20_1 < 2; q_20_1++) {
                            {
                                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(32 + lane_half * 16 + q_20_1 * 8));
                                u_pk[8 + q_20_1 * 4 + 0] = _uv4_13.x;
                                u_pk[8 + q_20_1 * 4 + 1] = _uv4_13.y;
                                u_pk[8 + q_20_1 * 4 + 2] = _uv4_13.z;
                                u_pk[8 + q_20_1 * 4 + 3] = _uv4_13.w;
                            }
                        }
                        #pragma unroll
                        for (int q_21 = 0; q_21 < 2; q_21++) {
                            {
                                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(64 + lane_half * 16 + q_21 * 8));
                                u_pk[16 + q_21 * 4 + 0] = _uv4_14.x;
                                u_pk[16 + q_21 * 4 + 1] = _uv4_14.y;
                                u_pk[16 + q_21 * 4 + 2] = _uv4_14.z;
                                u_pk[16 + q_21 * 4 + 3] = _uv4_14.w;
                            }
                        }
                        #pragma unroll
                        for (int q_22 = 0; q_22 < 2; q_22++) {
                            {
                                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(u + u_next_1 + (long long)(96 + lane_half * 16 + q_22 * 8));
                                u_pk[24 + q_22 * 4 + 0] = _uv4_15.x;
                                u_pk[24 + q_22 * 4 + 1] = _uv4_15.y;
                                u_pk[24 + q_22 * 4 + 2] = _uv4_15.z;
                                u_pk[24 + q_22 * 4 + 3] = _uv4_15.w;
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
                float _tmem_load_20[16];
                tmem_ld_x16(&_tmem_load_20[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_23 = 0; q_23 < 4; q_23++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_20[q_23 * 4 + 0], _tmem_load_20[q_23 * 4 + 1], _tmem_load_20[q_23 * 4 + 2], _tmem_load_20[q_23 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(q_23 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_0[16];
                tmem_ld_x16(&tmem_load_r_r_0[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_23_1 = 0; q_23_1 < 4; q_23_1++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_0[q_23_1 * 4 + 0], tmem_load_r_r_0[q_23_1 * 4 + 1], tmem_load_r_r_0[q_23_1 * 4 + 2], tmem_load_r_r_0[q_23_1 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(16 + q_23_1 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_1[16];
                tmem_ld_x16(&tmem_load_r_r_1[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_24 = 0; q_24 < 4; q_24++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_1[q_24 * 4 + 0], tmem_load_r_r_1[q_24 * 4 + 1], tmem_load_r_r_1[q_24 * 4 + 2], tmem_load_r_r_1[q_24 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(32 + q_24 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_2[16];
                tmem_ld_x16(&tmem_load_r_r_2[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_25 = 0; q_25 < 4; q_25++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_2[q_25 * 4 + 0], tmem_load_r_r_2[q_25 * 4 + 1], tmem_load_r_r_2[q_25 * 4 + 2], tmem_load_r_r_2[q_25 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(48 + q_25 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_3[16];
                tmem_ld_x16(&tmem_load_r_r_3[0], taddr + (unsigned int)h_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_26 = 0; q_26 < 4; q_26++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_3[q_26 * 4 + 0], tmem_load_r_r_3[q_26 * 4 + 1], tmem_load_r_r_3[q_26 * 4 + 2], tmem_load_r_r_3[q_26 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(64 + q_26 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_4[16];
                tmem_ld_x16(&tmem_load_r_r_4[0], taddr + (unsigned int)h_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_27 = 0; q_27 < 4; q_27++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_4[q_27 * 4 + 0], tmem_load_r_r_4[q_27 * 4 + 1], tmem_load_r_r_4[q_27 * 4 + 2], tmem_load_r_r_4[q_27 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(80 + q_27 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_5[16];
                tmem_ld_x16(&tmem_load_r_r_5[0], taddr + (unsigned int)h_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_28 = 0; q_28 < 4; q_28++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_5[q_28 * 4 + 0], tmem_load_r_r_5[q_28 * 4 + 1], tmem_load_r_r_5[q_28 * 4 + 2], tmem_load_r_r_5[q_28 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(96 + q_28 * 4)) = _v4;
                    }
                }
                float tmem_load_r_r_6[16];
                tmem_ld_x16(&tmem_load_r_r_6[0], taddr + (unsigned int)h_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_29 = 0; q_29 < 4; q_29++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_r_6[q_29 * 4 + 0], tmem_load_r_r_6[q_29 * 4 + 1], tmem_load_r_r_6[q_29 * 4 + 2], tmem_load_r_r_6[q_29 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(112 + q_29 * 4)) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
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
                chain_1 = seq_1 / segments;
                nchain_1 = num_seqs / segments;
            } else {
                nchain_1 = -segments;
                chain_1 = 0;
                for (int j_11 = 1; j_11 < nchain_1; j_11++) {
                    if (seq_1 >= seq_chunk_start[j_11]) {
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
            int tapes_on_1 = 0;
            if (walk_1 == 0) {
                if (write_tapes == 1) {
                    tapes_on_1 = 1;
                }
            }
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_hs_ready_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long h_base = ((long long)(c0_1 + c_1) * (long long)num_heads + (long long)head_1) * 16384 + (long long)(k_row_1 * 128);
                float decay = approx_exp2(gk_pre);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_0[16];
                #pragma unroll
                for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                    hs2_7_0[j_1_0] = _tmem_load_0[j_1_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2_7_0);
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_1[16];
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                    hs2_7_1[j_1_1] = _tmem_load_1[j_1_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_7_1);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_2[16];
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                    hs2_7_2[j_1_2] = _tmem_load_2[j_1_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_7_2);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_3[16];
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                    hs2_7_3[j_1_3] = _tmem_load_3[j_1_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_7_3);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base_1 + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_4[16];
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                    hs2_7_4[j_1_4] = _tmem_load_4[j_1_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 64, hs2_7_4);
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)h_row_base_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_5[16];
                #pragma unroll
                for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                    hs2_7_5[j_1_5] = _tmem_load_5[j_1_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 80, hs2_7_5);
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)h_row_base_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_6[16];
                #pragma unroll
                for (int j_1_6 = 0; j_1_6 < 16; j_1_6++) {
                    hs2_7_6[j_1_6] = _tmem_load_6[j_1_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 96, hs2_7_6);
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)h_row_base_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_7_7[16];
                #pragma unroll
                for (int j_1_7 = 0; j_1_7 < 16; j_1_7++) {
                    hs2_7_7[j_1_7] = _tmem_load_7[j_1_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 112, hs2_7_7);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
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
                chain_2 = seq_2 / segments;
                nchain_2 = num_seqs / segments;
            } else {
                nchain_2 = -segments;
                chain_2 = 0;
                for (int j_12 = 1; j_12 < nchain_2; j_12++) {
                    if (seq_2 >= seq_chunk_start[j_12]) {
                        chain_2 = chain_2 + 1;
                    }
                }
            }
            int tblk_2 = nchain_2 + 1 + seq_2 + chain_2;
            int c0_2 = seq_chunk_start[tblk_2];
            int nc_2 = seq_chunk_start[tblk_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int tapes_on_2 = 0;
            if (walk_2 == 0) {
                if (write_tapes == 1) {
                    tapes_on_2 = 1;
                }
            }
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
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
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 32768);
                        int trow_2 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&w_tma), seg_1 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&kg_tma), seg_1 * 64, head_2, trow_2, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    if (tapes_on_2 == 1) {
                        int chunk = (c0_2 + c_2) * num_heads + head_2;
                        mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                        _phase_hs_ready_0_1 ^= 1;
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            tma_store_3d((&h_out), p * 64, 0, chunk, hs_addr + (unsigned int)(p * 16384));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        int trow_3 = row0_2 + c_2 * 64;
                        mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                        _phase_bv_ready_0 ^= 1;
                        #pragma unroll
                        for (int p_1 = 0; p_1 < 2; p_1++) {
                            tma_store_3d((&v_new), p_1 * 64, head_2, trow_3, bv_addr + (unsigned int)(p_1 * 8192));
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        mbarrier_arrive(h_tape_read_addr);
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(bv_tape_read_addr);
                    }
                }
                if (tapes_on_2 == 1) {
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
                chain_3 = seq_3 / segments;
                nchain_3 = num_seqs / segments;
            } else {
                nchain_3 = -segments;
                chain_3 = 0;
                for (int j_13 = 1; j_13 < nchain_3; j_13++) {
                    if (seq_3 >= seq_chunk_start[j_13]) {
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
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x7FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = (((w_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _smem_desc_lo_1 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_3 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
