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
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 128
#define TMEM_ACC_P_OFFSET 128
#define NUM_KG_PIPE_STAGES 1
#define NUM_QD_PIPE_STAGES 1
#define NUM_W_PIPE_STAGES 1
#define NUM_ONE_STAGES 1
#define SMEM_KG_PANEL_OFF 1024
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_QG_PANEL_OFF 17408
#define SMEM_QG_PANEL_STAGE_BYTES 8192
#define SMEM_QG_PANEL_STRIDE 8192
#define SMEM_W_PANEL_OFF 33792
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_DO_PANEL_OFF 50176
#define SMEM_DO_PANEL_STAGE_BYTES 8192
#define SMEM_DO_PANEL_STRIDE 8192
#define SMEM_KG_OP_OFF 1024
#define SMEM_KG_OP_STAGE_BYTES 16384
#define SMEM_KG_OP_STRIDE 16384
#define SMEM_QG_MN_OFF 17408
#define SMEM_QG_MN_STAGE_BYTES 16384
#define SMEM_QG_MN_STRIDE 16384
#define SMEM_W_MN_OFF 33792
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_DO_MN_OFF 50176
#define SMEM_DO_MN_STAGE_BYTES 16384
#define SMEM_DO_MN_STRIDE 16384
#define SMEM_DHS_OFF 66560
#define SMEM_DHS_STAGE_BYTES 32768
#define SMEM_DHS_STRIDE 32768
#define SMEM_DHS_MN_OFF 66560
#define SMEM_DHS_MN_STAGE_BYTES 32768
#define SMEM_DHS_MN_STRIDE 32768
#define SMEM_BDV_OFF 99328
#define SMEM_BDV_STAGE_BYTES 16384
#define SMEM_BDV_STRIDE 16384
#define SMEM_BDV_MN_OFF 99328
#define SMEM_BDV_MN_STAGE_BYTES 16384
#define SMEM_BDV_MN_STRIDE 16384
#define SMEM_AQK_DST_OFF 99328
#define SMEM_AQK_DST_STAGE_BYTES 8192
#define SMEM_AQK_DST_STRIDE 8192
#define SMEM_AQK_MN_OFF 99328
#define SMEM_AQK_MN_STAGE_BYTES 8192
#define SMEM_AQK_MN_STRIDE 8192
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_909a0aa57efb3f353972(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ gk, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int segments, float scale, float* __restrict__ dht, float* __restrict__ ht_out, int use_dht, int write_tapes, int write_ht, int* __restrict__ chunk_bos, int* __restrict__ chunk_len)
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
    #define kg_full_addr (mbar_base + 0)
    #define kg_empty_addr (mbar_base + 8)
    #define qd_full_addr (mbar_base + 16)
    #define qd_empty_addr (mbar_base + 24)
    #define w_full_addr (mbar_base + 32)
    #define w_empty_addr (mbar_base + 40)
    #define dhs_ready_addr (mbar_base + 48)
    #define dv_ready_addr (mbar_base + 56)
    #define dv_read_addr (mbar_base + 64)
    #define bdv_ready_addr (mbar_base + 72)
    #define dh_ready_addr (mbar_base + 80)
    #define dh_tape_read_addr (mbar_base + 88)
    #define dv2_tape_read_addr (mbar_base + 96)
    #define aqk_full_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_panel_addr = smem + 1024;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_panel_addr = smem + 17408;
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_panel_addr = smem + 33792;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_panel_addr = smem + 50176;
    __nv_bfloat16* kg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_op_addr = smem + 1024;
    __nv_bfloat16* qg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_mn_addr = smem + 17408;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int do_mn_addr = smem + 50176;
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_addr = smem + 66560;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dhs_mn_addr = smem + 66560;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_addr = smem + 99328;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_mn_addr = smem + 99328;
    __nv_bfloat16* aqk_dst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int aqk_dst_addr = smem + 99328;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int aqk_mn_addr = smem + 99328;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kg_pipe' ---
            // kg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'qd_pipe' ---
            // qd_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // qd_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'w_pipe' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // w_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // dv_read: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // dh_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // dv2_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // aqk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 128;
    const int tmem_acc_p = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int head = blockIdx.x % num_heads;
            int seq = blockIdx.x / num_heads;
            int chain = 0;
            int nchain = 0;
            if (segments > 0) {
                chain = seq / segments;
                nchain = num_seqs / segments;
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
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            if (use_dht == 1) {
                long long dht_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(q * 4));
                        init_row[q * 4 + 0] = _v4.x;
                        init_row[q * 4 + 1] = _v4.y;
                        init_row[q * 4 + 2] = _v4.z;
                        init_row[q * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(16 + q_1 * 4));
                        init_row[q_1 * 4 + 0] = _v4.x;
                        init_row[q_1 * 4 + 1] = _v4.y;
                        init_row[q_1 * 4 + 2] = _v4.z;
                        init_row[q_1 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(32 + q_2 * 4));
                        init_row[q_2 * 4 + 0] = _v4.x;
                        init_row[q_2 * 4 + 1] = _v4.y;
                        init_row[q_2 * 4 + 2] = _v4.z;
                        init_row[q_2 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, init_row);
                #pragma unroll
                for (int q_3 = 0; q_3 < 4; q_3++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(48 + q_3 * 4));
                        init_row[q_3 * 4 + 0] = _v4.x;
                        init_row[q_3 * 4 + 1] = _v4.y;
                        init_row[q_3 * 4 + 2] = _v4.z;
                        init_row[q_3 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, init_row);
                #pragma unroll
                for (int q_4 = 0; q_4 < 4; q_4++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(64 + q_4 * 4));
                        init_row[q_4 * 4 + 0] = _v4.x;
                        init_row[q_4 * 4 + 1] = _v4.y;
                        init_row[q_4 * 4 + 2] = _v4.z;
                        init_row[q_4 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, init_row);
                #pragma unroll
                for (int q_5 = 0; q_5 < 4; q_5++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(80 + q_5 * 4));
                        init_row[q_5 * 4 + 0] = _v4.x;
                        init_row[q_5 * 4 + 1] = _v4.y;
                        init_row[q_5 * 4 + 2] = _v4.z;
                        init_row[q_5 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, init_row);
                #pragma unroll
                for (int q_6 = 0; q_6 < 4; q_6++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(96 + q_6 * 4));
                        init_row[q_6 * 4 + 0] = _v4.x;
                        init_row[q_6 * 4 + 1] = _v4.y;
                        init_row[q_6 * 4 + 2] = _v4.z;
                        init_row[q_6 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, init_row);
                #pragma unroll
                for (int q_7 = 0; q_7 < 4; q_7++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(112 + q_7 * 4));
                        init_row[q_7 * 4 + 0] = _v4.x;
                        init_row[q_7 * 4 + 1] = _v4.y;
                        init_row[q_7 * 4 + 2] = _v4.z;
                        init_row[q_7 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, init_row);
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
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, init_row);
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, init_row);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float gk_pre = 0.0f;
            float decay = 1.0f;
            if (nc > 0) {
                long long gn_next = ((long long)(row0 + nc * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                gk_pre = gk[gn_next];
                long long dh_base = ((long long)(c0 + nc - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_8 * 8 / 64 * 16384 + k_row * 128 + q_8 * 8 % 64 * 2 ^ (q_8 * 8 / 64 * 16384 + k_row * 128 + q_8 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[q_8 * 4]), "r"(_tmem_load_0_bf16[q_8 * 4 + 1]), "r"(_tmem_load_0_bf16[q_8 * 4 + 2]), "r"(_tmem_load_0_bf16[q_8 * 4 + 3]) : "memory");
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_9 * 8) / 64 * 16384 + k_row * 128 + (16 + q_9 * 8) % 64 * 2 ^ ((16 + q_9 * 8) / 64 * 16384 + k_row * 128 + (16 + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[q_9 * 4]), "r"(_tmem_load_1_bf16[q_9 * 4 + 1]), "r"(_tmem_load_1_bf16[q_9 * 4 + 2]), "r"(_tmem_load_1_bf16[q_9 * 4 + 3]) : "memory");
                }
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_10 = 0; q_10 < 2; q_10++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_10 * 8) / 64 * 16384 + k_row * 128 + (32 + q_10 * 8) % 64 * 2 ^ ((32 + q_10 * 8) / 64 * 16384 + k_row * 128 + (32 + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[q_10 * 4]), "r"(_tmem_load_2_bf16[q_10 * 4 + 1]), "r"(_tmem_load_2_bf16[q_10 * 4 + 2]), "r"(_tmem_load_2_bf16[q_10 * 4 + 3]) : "memory");
                }
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_11 = 0; q_11 < 2; q_11++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_11 * 8) / 64 * 16384 + k_row * 128 + (48 + q_11 * 8) % 64 * 2 ^ ((48 + q_11 * 8) / 64 * 16384 + k_row * 128 + (48 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[q_11 * 4]), "r"(_tmem_load_3_bf16[q_11 * 4 + 1]), "r"(_tmem_load_3_bf16[q_11 * 4 + 2]), "r"(_tmem_load_3_bf16[q_11 * 4 + 3]) : "memory");
                }
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_12 = 0; q_12 < 2; q_12++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((64 + q_12 * 8) / 64 * 16384 + k_row * 128 + (64 + q_12 * 8) % 64 * 2 ^ ((64 + q_12 * 8) / 64 * 16384 + k_row * 128 + (64 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[q_12 * 4]), "r"(_tmem_load_4_bf16[q_12 * 4 + 1]), "r"(_tmem_load_4_bf16[q_12 * 4 + 2]), "r"(_tmem_load_4_bf16[q_12 * 4 + 3]) : "memory");
                }
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_13 = 0; q_13 < 2; q_13++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((80 + q_13 * 8) / 64 * 16384 + k_row * 128 + (80 + q_13 * 8) % 64 * 2 ^ ((80 + q_13 * 8) / 64 * 16384 + k_row * 128 + (80 + q_13 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[q_13 * 4]), "r"(_tmem_load_5_bf16[q_13 * 4 + 1]), "r"(_tmem_load_5_bf16[q_13 * 4 + 2]), "r"(_tmem_load_5_bf16[q_13 * 4 + 3]) : "memory");
                }
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_14 = 0; q_14 < 2; q_14++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((96 + q_14 * 8) / 64 * 16384 + k_row * 128 + (96 + q_14 * 8) % 64 * 2 ^ ((96 + q_14 * 8) / 64 * 16384 + k_row * 128 + (96 + q_14 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_14 * 4]), "r"(_tmem_load_6_bf16[q_14 * 4 + 1]), "r"(_tmem_load_6_bf16[q_14 * 4 + 2]), "r"(_tmem_load_6_bf16[q_14 * 4 + 3]) : "memory");
                }
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_15 = 0; q_15 < 2; q_15++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((112 + q_15 * 8) / 64 * 16384 + k_row * 128 + (112 + q_15 * 8) % 64 * 2 ^ ((112 + q_15 * 8) / 64 * 16384 + k_row * 128 + (112 + q_15 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[q_15 * 4]), "r"(_tmem_load_7_bf16[q_15 * 4 + 1]), "r"(_tmem_load_7_bf16[q_15 * 4 + 2]), "r"(_tmem_load_7_bf16[q_15 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                float _exp2_8 = approx_exp2(gk_pre);
                decay = _exp2_8;
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 2], _tmem_load_8[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 4], _tmem_load_8[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 6], _tmem_load_8[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 8], _tmem_load_8[_lp*2+1 + 8]));
                        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 10], _tmem_load_8[_lp*2+1 + 10]));
                        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 12], _tmem_load_8[_lp*2+1 + 12]));
                        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 14], _tmem_load_8[_lp*2+1 + 14]));
                        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + dh_base)), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                    }
                }
                float sc[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    sc[j_1] = _tmem_load_8[j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc);
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                        _packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 2], _tmem_load_9[_lp*2+1 + 2]));
                        _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 4], _tmem_load_9[_lp*2+1 + 4]));
                        _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 6], _tmem_load_9[_lp*2+1 + 6]));
                        _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 8], _tmem_load_9[_lp*2+1 + 8]));
                        _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 10], _tmem_load_9[_lp*2+1 + 10]));
                        _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 12], _tmem_load_9[_lp*2+1 + 12]));
                        _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 14], _tmem_load_9[_lp*2+1 + 14]));
                        _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_10 = reinterpret_cast<const unsigned*>(_packed_1);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 16))), "r"(_raw_stv8_10[0]), "r"(_raw_stv8_10[1]), "r"(_raw_stv8_10[2]), "r"(_raw_stv8_10[3]), "r"(_raw_stv8_10[4]), "r"(_raw_stv8_10[5]), "r"(_raw_stv8_10[6]), "r"(_raw_stv8_10[7]) : "memory");
                    }
                }
                float sc_0[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    sc_0[j_2] = _tmem_load_9[j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_0);
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                        _packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 2], _tmem_load_10[_lp*2+1 + 2]));
                        _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 4], _tmem_load_10[_lp*2+1 + 4]));
                        _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 6], _tmem_load_10[_lp*2+1 + 6]));
                        _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 8], _tmem_load_10[_lp*2+1 + 8]));
                        _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 10], _tmem_load_10[_lp*2+1 + 10]));
                        _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 12], _tmem_load_10[_lp*2+1 + 12]));
                        _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 14], _tmem_load_10[_lp*2+1 + 14]));
                        _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_11 = reinterpret_cast<const unsigned*>(_packed_2);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 32))), "r"(_raw_stv8_11[0]), "r"(_raw_stv8_11[1]), "r"(_raw_stv8_11[2]), "r"(_raw_stv8_11[3]), "r"(_raw_stv8_11[4]), "r"(_raw_stv8_11[5]), "r"(_raw_stv8_11[6]), "r"(_raw_stv8_11[7]) : "memory");
                    }
                }
                float sc_1[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    sc_1[j_3] = _tmem_load_10[j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_1);
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                        _packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 2], _tmem_load_11[_lp*2+1 + 2]));
                        _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 4], _tmem_load_11[_lp*2+1 + 4]));
                        _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 6], _tmem_load_11[_lp*2+1 + 6]));
                        _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 8], _tmem_load_11[_lp*2+1 + 8]));
                        _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 10], _tmem_load_11[_lp*2+1 + 10]));
                        _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 12], _tmem_load_11[_lp*2+1 + 12]));
                        _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 14], _tmem_load_11[_lp*2+1 + 14]));
                        _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_12 = reinterpret_cast<const unsigned*>(_packed_3);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 48))), "r"(_raw_stv8_12[0]), "r"(_raw_stv8_12[1]), "r"(_raw_stv8_12[2]), "r"(_raw_stv8_12[3]), "r"(_raw_stv8_12[4]), "r"(_raw_stv8_12[5]), "r"(_raw_stv8_12[6]), "r"(_raw_stv8_12[7]) : "memory");
                    }
                }
                float sc_2[16];
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    sc_2[j_4] = _tmem_load_11[j_4] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_2);
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                        _packed_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 2], _tmem_load_12[_lp*2+1 + 2]));
                        _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 4], _tmem_load_12[_lp*2+1 + 4]));
                        _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 6], _tmem_load_12[_lp*2+1 + 6]));
                        _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 8], _tmem_load_12[_lp*2+1 + 8]));
                        _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 10], _tmem_load_12[_lp*2+1 + 10]));
                        _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 12], _tmem_load_12[_lp*2+1 + 12]));
                        _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 14], _tmem_load_12[_lp*2+1 + 14]));
                        _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_13 = reinterpret_cast<const unsigned*>(_packed_4);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 64))), "r"(_raw_stv8_13[0]), "r"(_raw_stv8_13[1]), "r"(_raw_stv8_13[2]), "r"(_raw_stv8_13[3]), "r"(_raw_stv8_13[4]), "r"(_raw_stv8_13[5]), "r"(_raw_stv8_13[6]), "r"(_raw_stv8_13[7]) : "memory");
                    }
                }
                float sc_3[16];
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    sc_3[j_5] = _tmem_load_12[j_5] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, sc_3);
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], taddr + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                        _packed_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 2], _tmem_load_13[_lp*2+1 + 2]));
                        _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 4], _tmem_load_13[_lp*2+1 + 4]));
                        _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 6], _tmem_load_13[_lp*2+1 + 6]));
                        _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 8], _tmem_load_13[_lp*2+1 + 8]));
                        _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 10], _tmem_load_13[_lp*2+1 + 10]));
                        _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 12], _tmem_load_13[_lp*2+1 + 12]));
                        _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 14], _tmem_load_13[_lp*2+1 + 14]));
                        _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_14 = reinterpret_cast<const unsigned*>(_packed_5);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 80))), "r"(_raw_stv8_14[0]), "r"(_raw_stv8_14[1]), "r"(_raw_stv8_14[2]), "r"(_raw_stv8_14[3]), "r"(_raw_stv8_14[4]), "r"(_raw_stv8_14[5]), "r"(_raw_stv8_14[6]), "r"(_raw_stv8_14[7]) : "memory");
                    }
                }
                float sc_4[16];
                #pragma unroll
                for (int j_6 = 0; j_6 < 16; j_6++) {
                    sc_4[j_6] = _tmem_load_13[j_6] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, sc_4);
                float _tmem_load_14[16];
                tmem_ld_x16(&_tmem_load_14[0], taddr + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                        _packed_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 2], _tmem_load_14[_lp*2+1 + 2]));
                        _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 4], _tmem_load_14[_lp*2+1 + 4]));
                        _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 6], _tmem_load_14[_lp*2+1 + 6]));
                        _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 8], _tmem_load_14[_lp*2+1 + 8]));
                        _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 10], _tmem_load_14[_lp*2+1 + 10]));
                        _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 12], _tmem_load_14[_lp*2+1 + 12]));
                        _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 14], _tmem_load_14[_lp*2+1 + 14]));
                        _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_15 = reinterpret_cast<const unsigned*>(_packed_6);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 96))), "r"(_raw_stv8_15[0]), "r"(_raw_stv8_15[1]), "r"(_raw_stv8_15[2]), "r"(_raw_stv8_15[3]), "r"(_raw_stv8_15[4]), "r"(_raw_stv8_15[5]), "r"(_raw_stv8_15[6]), "r"(_raw_stv8_15[7]) : "memory");
                    }
                }
                float sc_5[16];
                #pragma unroll
                for (int j_7 = 0; j_7 < 16; j_7++) {
                    sc_5[j_7] = _tmem_load_14[j_7] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, sc_5);
                float _tmem_load_15[16];
                tmem_ld_x16(&_tmem_load_15[0], taddr + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (write_tapes == 1) {
                    unsigned int _packed_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 0], _tmem_load_15[_lp*2+1 + 0]));
                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 2], _tmem_load_15[_lp*2+1 + 2]));
                        _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 4], _tmem_load_15[_lp*2+1 + 4]));
                        _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 6], _tmem_load_15[_lp*2+1 + 6]));
                        _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 8], _tmem_load_15[_lp*2+1 + 8]));
                        _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 10], _tmem_load_15[_lp*2+1 + 10]));
                        _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 12], _tmem_load_15[_lp*2+1 + 12]));
                        _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 14], _tmem_load_15[_lp*2+1 + 14]));
                        _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_16 = reinterpret_cast<const unsigned*>(_packed_7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base + 112))), "r"(_raw_stv8_16[0]), "r"(_raw_stv8_16[1]), "r"(_raw_stv8_16[2]), "r"(_raw_stv8_16[3]), "r"(_raw_stv8_16[4]), "r"(_raw_stv8_16[5]), "r"(_raw_stv8_16[6]), "r"(_raw_stv8_16[7]) : "memory");
                    }
                }
                float sc_6[16];
                #pragma unroll
                for (int j_8 = 0; j_8 < 16; j_8++) {
                    sc_6[j_8] = _tmem_load_15[j_8] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, sc_6);
            }
            unsigned int _phase_dv_ready_0 = 0;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int rc = 0; rc < nc; rc++) {
                int c = nc - 1 - rc;
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int trow = row0 + c * 64 + t_row;
                long long t_base = ((long long)trow * (long long)num_heads + (long long)head) * 128;
                float _tmem_load_16[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_16[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base));
                float _tmem_load_17[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_17[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int col = lane_half * 16;
                if (write_tapes == 1) {
                    unsigned int _packed_8[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 0], _tmem_load_16[_lp*2+1 + 0]));
                        _packed_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 2], _tmem_load_16[_lp*2+1 + 2]));
                        _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 4], _tmem_load_16[_lp*2+1 + 4]));
                        _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 6], _tmem_load_16[_lp*2+1 + 6]));
                        _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 8], _tmem_load_16[_lp*2+1 + 8]));
                        _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 10], _tmem_load_16[_lp*2+1 + 10]));
                        _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 12], _tmem_load_16[_lp*2+1 + 12]));
                        _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 14], _tmem_load_16[_lp*2+1 + 14]));
                        _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_17 = reinterpret_cast<const unsigned*>(_packed_8);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dv2 + (t_base + (long long)col))), "r"(_raw_stv8_17[0]), "r"(_raw_stv8_17[1]), "r"(_raw_stv8_17[2]), "r"(_raw_stv8_17[3]), "r"(_raw_stv8_17[4]), "r"(_raw_stv8_17[5]), "r"(_raw_stv8_17[6]), "r"(_raw_stv8_17[7]) : "memory");
                    }
                }
                unsigned int _tmem_load_16_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_16[_lp*2 + 0], _tmem_load_16[_lp*2+1 + 0]));
                    _tmem_load_16_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_16 = 0; q_16 < 2; q_16++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col + q_16 * 8) / 64 * 8192 + t_row * 128 + (col + q_16 * 8) % 64 * 2 ^ ((col + q_16 * 8) / 64 * 8192 + t_row * 128 + (col + q_16 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_16_bf16[q_16 * 4]), "r"(_tmem_load_16_bf16[q_16 * 4 + 1]), "r"(_tmem_load_16_bf16[q_16 * 4 + 2]), "r"(_tmem_load_16_bf16[q_16 * 4 + 3]) : "memory");
                }
                int col_0 = 32 + lane_half * 16;
                if (write_tapes == 1) {
                    unsigned int _packed_9[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 0], _tmem_load_17[_lp*2+1 + 0]));
                        _packed_9[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 2], _tmem_load_17[_lp*2+1 + 2]));
                        _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 4], _tmem_load_17[_lp*2+1 + 4]));
                        _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 6], _tmem_load_17[_lp*2+1 + 6]));
                        _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 8], _tmem_load_17[_lp*2+1 + 8]));
                        _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 10], _tmem_load_17[_lp*2+1 + 10]));
                        _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 12], _tmem_load_17[_lp*2+1 + 12]));
                        _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 14], _tmem_load_17[_lp*2+1 + 14]));
                        _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_18 = reinterpret_cast<const unsigned*>(_packed_9);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dv2 + (t_base + (long long)col_0))), "r"(_raw_stv8_18[0]), "r"(_raw_stv8_18[1]), "r"(_raw_stv8_18[2]), "r"(_raw_stv8_18[3]), "r"(_raw_stv8_18[4]), "r"(_raw_stv8_18[5]), "r"(_raw_stv8_18[6]), "r"(_raw_stv8_18[7]) : "memory");
                    }
                }
                unsigned int _tmem_load_17_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_17[_lp*2 + 0], _tmem_load_17[_lp*2+1 + 0]));
                    _tmem_load_17_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_17 = 0; q_17 < 2; q_17++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_0 + q_17 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_17 * 8) % 64 * 2 ^ ((col_0 + q_17 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_17 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_17_bf16[q_17 * 4]), "r"(_tmem_load_17_bf16[q_17 * 4 + 1]), "r"(_tmem_load_17_bf16[q_17 * 4 + 2]), "r"(_tmem_load_17_bf16[q_17 * 4 + 3]) : "memory");
                }
                float _tmem_load_18[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_18[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 64));
                float _tmem_load_19[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_19[15]))
                    : "r"(taddr + 128 + (unsigned int)m64_row_base + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dv_read_addr);
                int col_1 = 64 + lane_half * 16;
                if (write_tapes == 1) {
                    unsigned int _packed_10[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                        _packed_10[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 2], _tmem_load_18[_lp*2+1 + 2]));
                        _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 4], _tmem_load_18[_lp*2+1 + 4]));
                        _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 6], _tmem_load_18[_lp*2+1 + 6]));
                        _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 8], _tmem_load_18[_lp*2+1 + 8]));
                        _packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 10], _tmem_load_18[_lp*2+1 + 10]));
                        _packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 12], _tmem_load_18[_lp*2+1 + 12]));
                        _packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 14], _tmem_load_18[_lp*2+1 + 14]));
                        _packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_19 = reinterpret_cast<const unsigned*>(_packed_10);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dv2 + (t_base + (long long)col_1))), "r"(_raw_stv8_19[0]), "r"(_raw_stv8_19[1]), "r"(_raw_stv8_19[2]), "r"(_raw_stv8_19[3]), "r"(_raw_stv8_19[4]), "r"(_raw_stv8_19[5]), "r"(_raw_stv8_19[6]), "r"(_raw_stv8_19[7]) : "memory");
                    }
                }
                unsigned int _tmem_load_18_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                    _tmem_load_18_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_18 = 0; q_18 < 2; q_18++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_1 + q_18 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_18 * 8) % 64 * 2 ^ ((col_1 + q_18 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_18 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[q_18 * 4]), "r"(_tmem_load_18_bf16[q_18 * 4 + 1]), "r"(_tmem_load_18_bf16[q_18 * 4 + 2]), "r"(_tmem_load_18_bf16[q_18 * 4 + 3]) : "memory");
                }
                int col_2 = 96 + lane_half * 16;
                if (write_tapes == 1) {
                    unsigned int _packed_11[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                        _packed_11[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 2], _tmem_load_19[_lp*2+1 + 2]));
                        _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 4], _tmem_load_19[_lp*2+1 + 4]));
                        _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 6], _tmem_load_19[_lp*2+1 + 6]));
                        _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 8], _tmem_load_19[_lp*2+1 + 8]));
                        _packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 10], _tmem_load_19[_lp*2+1 + 10]));
                        _packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 12], _tmem_load_19[_lp*2+1 + 12]));
                        _packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 14], _tmem_load_19[_lp*2+1 + 14]));
                        _packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_20 = reinterpret_cast<const unsigned*>(_packed_11);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dv2 + (t_base + (long long)col_2))), "r"(_raw_stv8_20[0]), "r"(_raw_stv8_20[1]), "r"(_raw_stv8_20[2]), "r"(_raw_stv8_20[3]), "r"(_raw_stv8_20[4]), "r"(_raw_stv8_20[5]), "r"(_raw_stv8_20[6]), "r"(_raw_stv8_20[7]) : "memory");
                    }
                }
                unsigned int _tmem_load_19_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                    _tmem_load_19_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_19 = 0; q_19 < 2; q_19++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_2 + q_19 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_19 * 8) % 64 * 2 ^ ((col_2 + q_19 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_19 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[q_19 * 4]), "r"(_tmem_load_19_bf16[q_19 * 4 + 1]), "r"(_tmem_load_19_bf16[q_19 * 4 + 2]), "r"(_tmem_load_19_bf16[q_19 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                if (nc > rc + 1) {
                    long long gn_next_1 = ((long long)(row0 + c * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                    gk_pre = gk[gn_next_1];
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (nc > rc + 1) {
                    float _exp2_21 = approx_exp2(gk_pre);
                    decay = _exp2_21;
                    long long dh_base_1 = ((long long)(c0 + c - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                    float _tmem_load_20[16];
                    tmem_ld_x16(&_tmem_load_20[0], taddr + (unsigned int)m128_row_base);
                    float _tmem_load_21[16];
                    tmem_ld_x16(&_tmem_load_21[0], taddr + 128 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd[16];
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 16; j_9++) {
                        upd[j_9] = __fmaf_rn(_tmem_load_21[j_9], scale, _tmem_load_20[j_9]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, upd);
                    unsigned int upd_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        upd_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_20 = 0; q_20 < 2; q_20++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_20 * 8 / 64 * 16384 + k_row * 128 + q_20 * 8 % 64 * 2 ^ (q_20 * 8 / 64 * 16384 + k_row * 128 + q_20 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16[q_20 * 4]), "r"(upd_bf16[q_20 * 4 + 1]), "r"(upd_bf16[q_20 * 4 + 2]), "r"(upd_bf16[q_20 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_22[16];
                    tmem_ld_x16(&_tmem_load_22[0], taddr + (unsigned int)m128_row_base + 16);
                    float _tmem_load_23[16];
                    tmem_ld_x16(&_tmem_load_23[0], taddr + 128 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_0[16];
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 16; j_10++) {
                        upd_0[j_10] = __fmaf_rn(_tmem_load_23[j_10], scale, _tmem_load_22[j_10]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, upd_0);
                    unsigned int upd_0_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_0[_lp*2 + 0], upd_0[_lp*2+1 + 0]));
                        upd_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_21 = 0; q_21 < 2; q_21++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_21 * 8) / 64 * 16384 + k_row * 128 + (16 + q_21 * 8) % 64 * 2 ^ ((16 + q_21 * 8) / 64 * 16384 + k_row * 128 + (16 + q_21 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_0_bf16[q_21 * 4]), "r"(upd_0_bf16[q_21 * 4 + 1]), "r"(upd_0_bf16[q_21 * 4 + 2]), "r"(upd_0_bf16[q_21 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_24[16];
                    tmem_ld_x16(&_tmem_load_24[0], taddr + (unsigned int)m128_row_base + 32);
                    float _tmem_load_25[16];
                    tmem_ld_x16(&_tmem_load_25[0], taddr + 128 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_1[16];
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 16; j_11++) {
                        upd_1[j_11] = __fmaf_rn(_tmem_load_25[j_11], scale, _tmem_load_24[j_11]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, upd_1);
                    unsigned int upd_1_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_1[_lp*2 + 0], upd_1[_lp*2+1 + 0]));
                        upd_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_22 = 0; q_22 < 2; q_22++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_22 * 8) / 64 * 16384 + k_row * 128 + (32 + q_22 * 8) % 64 * 2 ^ ((32 + q_22 * 8) / 64 * 16384 + k_row * 128 + (32 + q_22 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_1_bf16[q_22 * 4]), "r"(upd_1_bf16[q_22 * 4 + 1]), "r"(upd_1_bf16[q_22 * 4 + 2]), "r"(upd_1_bf16[q_22 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_26[16];
                    tmem_ld_x16(&_tmem_load_26[0], taddr + (unsigned int)m128_row_base + 48);
                    float _tmem_load_27[16];
                    tmem_ld_x16(&_tmem_load_27[0], taddr + 128 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_2[16];
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 16; j_12++) {
                        upd_2[j_12] = __fmaf_rn(_tmem_load_27[j_12], scale, _tmem_load_26[j_12]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, upd_2);
                    unsigned int upd_2_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_2[_lp*2 + 0], upd_2[_lp*2+1 + 0]));
                        upd_2_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_23 = 0; q_23 < 2; q_23++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_23 * 8) / 64 * 16384 + k_row * 128 + (48 + q_23 * 8) % 64 * 2 ^ ((48 + q_23 * 8) / 64 * 16384 + k_row * 128 + (48 + q_23 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_2_bf16[q_23 * 4]), "r"(upd_2_bf16[q_23 * 4 + 1]), "r"(upd_2_bf16[q_23 * 4 + 2]), "r"(upd_2_bf16[q_23 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_28[16];
                    tmem_ld_x16(&_tmem_load_28[0], taddr + (unsigned int)m128_row_base + 64);
                    float _tmem_load_29[16];
                    tmem_ld_x16(&_tmem_load_29[0], taddr + 128 + (unsigned int)m128_row_base + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_3[16];
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 16; j_13++) {
                        upd_3[j_13] = __fmaf_rn(_tmem_load_29[j_13], scale, _tmem_load_28[j_13]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, upd_3);
                    unsigned int upd_3_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_3[_lp*2 + 0], upd_3[_lp*2+1 + 0]));
                        upd_3_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_24 = 0; q_24 < 2; q_24++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((64 + q_24 * 8) / 64 * 16384 + k_row * 128 + (64 + q_24 * 8) % 64 * 2 ^ ((64 + q_24 * 8) / 64 * 16384 + k_row * 128 + (64 + q_24 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_3_bf16[q_24 * 4]), "r"(upd_3_bf16[q_24 * 4 + 1]), "r"(upd_3_bf16[q_24 * 4 + 2]), "r"(upd_3_bf16[q_24 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_30[16];
                    tmem_ld_x16(&_tmem_load_30[0], taddr + (unsigned int)m128_row_base + 80);
                    float _tmem_load_31[16];
                    tmem_ld_x16(&_tmem_load_31[0], taddr + 128 + (unsigned int)m128_row_base + 80);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_4[16];
                    #pragma unroll
                    for (int j_14 = 0; j_14 < 16; j_14++) {
                        upd_4[j_14] = __fmaf_rn(_tmem_load_31[j_14], scale, _tmem_load_30[j_14]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, upd_4);
                    unsigned int upd_4_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_4[_lp*2 + 0], upd_4[_lp*2+1 + 0]));
                        upd_4_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_25 = 0; q_25 < 2; q_25++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((80 + q_25 * 8) / 64 * 16384 + k_row * 128 + (80 + q_25 * 8) % 64 * 2 ^ ((80 + q_25 * 8) / 64 * 16384 + k_row * 128 + (80 + q_25 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_4_bf16[q_25 * 4]), "r"(upd_4_bf16[q_25 * 4 + 1]), "r"(upd_4_bf16[q_25 * 4 + 2]), "r"(upd_4_bf16[q_25 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_32[16];
                    tmem_ld_x16(&_tmem_load_32[0], taddr + (unsigned int)m128_row_base + 96);
                    float _tmem_load_33[16];
                    tmem_ld_x16(&_tmem_load_33[0], taddr + 128 + (unsigned int)m128_row_base + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_5[16];
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 16; j_15++) {
                        upd_5[j_15] = __fmaf_rn(_tmem_load_33[j_15], scale, _tmem_load_32[j_15]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, upd_5);
                    unsigned int upd_5_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_5[_lp*2 + 0], upd_5[_lp*2+1 + 0]));
                        upd_5_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_26 = 0; q_26 < 2; q_26++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((96 + q_26 * 8) / 64 * 16384 + k_row * 128 + (96 + q_26 * 8) % 64 * 2 ^ ((96 + q_26 * 8) / 64 * 16384 + k_row * 128 + (96 + q_26 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_5_bf16[q_26 * 4]), "r"(upd_5_bf16[q_26 * 4 + 1]), "r"(upd_5_bf16[q_26 * 4 + 2]), "r"(upd_5_bf16[q_26 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_34[16];
                    tmem_ld_x16(&_tmem_load_34[0], taddr + (unsigned int)m128_row_base + 112);
                    float _tmem_load_35[16];
                    tmem_ld_x16(&_tmem_load_35[0], taddr + 128 + (unsigned int)m128_row_base + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_6[16];
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 16; j_16++) {
                        upd_6[j_16] = __fmaf_rn(_tmem_load_35[j_16], scale, _tmem_load_34[j_16]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, upd_6);
                    unsigned int upd_6_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_6[_lp*2 + 0], upd_6[_lp*2+1 + 0]));
                        upd_6_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_27 = 0; q_27 < 2; q_27++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((112 + q_27 * 8) / 64 * 16384 + k_row * 128 + (112 + q_27 * 8) % 64 * 2 ^ ((112 + q_27 * 8) / 64 * 16384 + k_row * 128 + (112 + q_27 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_6_bf16[q_27 * 4]), "r"(upd_6_bf16[q_27 * 4 + 1]), "r"(upd_6_bf16[q_27 * 4 + 2]), "r"(upd_6_bf16[q_27 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dhs_ready_addr);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    float _tmem_load_36[16];
                    tmem_ld_x16(&_tmem_load_36[0], taddr + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_12[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 0], _tmem_load_36[_lp*2+1 + 0]));
                            _packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 2], _tmem_load_36[_lp*2+1 + 2]));
                            _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 4], _tmem_load_36[_lp*2+1 + 4]));
                            _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 6], _tmem_load_36[_lp*2+1 + 6]));
                            _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 8], _tmem_load_36[_lp*2+1 + 8]));
                            _packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 10], _tmem_load_36[_lp*2+1 + 10]));
                            _packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 12], _tmem_load_36[_lp*2+1 + 12]));
                            _packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_36[_lp*2 + 14], _tmem_load_36[_lp*2+1 + 14]));
                            _packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_22 = reinterpret_cast<const unsigned*>(_packed_12);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + dh_base_1)), "r"(_raw_stv8_22[0]), "r"(_raw_stv8_22[1]), "r"(_raw_stv8_22[2]), "r"(_raw_stv8_22[3]), "r"(_raw_stv8_22[4]), "r"(_raw_stv8_22[5]), "r"(_raw_stv8_22[6]), "r"(_raw_stv8_22[7]) : "memory");
                        }
                    }
                    float sc_7[16];
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 16; j_17++) {
                        sc_7[j_17] = _tmem_load_36[j_17] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc_7);
                    float _tmem_load_37[16];
                    tmem_ld_x16(&_tmem_load_37[0], taddr + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_13[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 0], _tmem_load_37[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 2], _tmem_load_37[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 4], _tmem_load_37[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 6], _tmem_load_37[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 8], _tmem_load_37[_lp*2+1 + 8]));
                            _packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 10], _tmem_load_37[_lp*2+1 + 10]));
                            _packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 12], _tmem_load_37[_lp*2+1 + 12]));
                            _packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_37[_lp*2 + 14], _tmem_load_37[_lp*2+1 + 14]));
                            _packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_23 = reinterpret_cast<const unsigned*>(_packed_13);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 16))), "r"(_raw_stv8_23[0]), "r"(_raw_stv8_23[1]), "r"(_raw_stv8_23[2]), "r"(_raw_stv8_23[3]), "r"(_raw_stv8_23[4]), "r"(_raw_stv8_23[5]), "r"(_raw_stv8_23[6]), "r"(_raw_stv8_23[7]) : "memory");
                        }
                    }
                    float sc_7_1[16];
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 16; j_18++) {
                        sc_7_1[j_18] = _tmem_load_37[j_18] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_7_1);
                    float _tmem_load_38[16];
                    tmem_ld_x16(&_tmem_load_38[0], taddr + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_14[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 0], _tmem_load_38[_lp*2+1 + 0]));
                            _packed_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 2], _tmem_load_38[_lp*2+1 + 2]));
                            _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 4], _tmem_load_38[_lp*2+1 + 4]));
                            _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 6], _tmem_load_38[_lp*2+1 + 6]));
                            _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 8], _tmem_load_38[_lp*2+1 + 8]));
                            _packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 10], _tmem_load_38[_lp*2+1 + 10]));
                            _packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 12], _tmem_load_38[_lp*2+1 + 12]));
                            _packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_38[_lp*2 + 14], _tmem_load_38[_lp*2+1 + 14]));
                            _packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_24 = reinterpret_cast<const unsigned*>(_packed_14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 32))), "r"(_raw_stv8_24[0]), "r"(_raw_stv8_24[1]), "r"(_raw_stv8_24[2]), "r"(_raw_stv8_24[3]), "r"(_raw_stv8_24[4]), "r"(_raw_stv8_24[5]), "r"(_raw_stv8_24[6]), "r"(_raw_stv8_24[7]) : "memory");
                        }
                    }
                    float sc_8[16];
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 16; j_19++) {
                        sc_8[j_19] = _tmem_load_38[j_19] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_8);
                    float _tmem_load_39[16];
                    tmem_ld_x16(&_tmem_load_39[0], taddr + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_15[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 0], _tmem_load_39[_lp*2+1 + 0]));
                            _packed_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 2], _tmem_load_39[_lp*2+1 + 2]));
                            _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 4], _tmem_load_39[_lp*2+1 + 4]));
                            _packed_15[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 6], _tmem_load_39[_lp*2+1 + 6]));
                            _packed_15[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 8], _tmem_load_39[_lp*2+1 + 8]));
                            _packed_15[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 10], _tmem_load_39[_lp*2+1 + 10]));
                            _packed_15[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 12], _tmem_load_39[_lp*2+1 + 12]));
                            _packed_15[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_39[_lp*2 + 14], _tmem_load_39[_lp*2+1 + 14]));
                            _packed_15[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_25 = reinterpret_cast<const unsigned*>(_packed_15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 48))), "r"(_raw_stv8_25[0]), "r"(_raw_stv8_25[1]), "r"(_raw_stv8_25[2]), "r"(_raw_stv8_25[3]), "r"(_raw_stv8_25[4]), "r"(_raw_stv8_25[5]), "r"(_raw_stv8_25[6]), "r"(_raw_stv8_25[7]) : "memory");
                        }
                    }
                    float sc_9[16];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 16; j_20++) {
                        sc_9[j_20] = _tmem_load_39[j_20] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_9);
                    float _tmem_load_40[16];
                    tmem_ld_x16(&_tmem_load_40[0], taddr + (unsigned int)m128_row_base + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 0], _tmem_load_40[_lp*2+1 + 0]));
                            _packed_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 2], _tmem_load_40[_lp*2+1 + 2]));
                            _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 4], _tmem_load_40[_lp*2+1 + 4]));
                            _packed_16[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 6], _tmem_load_40[_lp*2+1 + 6]));
                            _packed_16[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 8], _tmem_load_40[_lp*2+1 + 8]));
                            _packed_16[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 10], _tmem_load_40[_lp*2+1 + 10]));
                            _packed_16[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 12], _tmem_load_40[_lp*2+1 + 12]));
                            _packed_16[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_40[_lp*2 + 14], _tmem_load_40[_lp*2+1 + 14]));
                            _packed_16[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_26 = reinterpret_cast<const unsigned*>(_packed_16);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 64))), "r"(_raw_stv8_26[0]), "r"(_raw_stv8_26[1]), "r"(_raw_stv8_26[2]), "r"(_raw_stv8_26[3]), "r"(_raw_stv8_26[4]), "r"(_raw_stv8_26[5]), "r"(_raw_stv8_26[6]), "r"(_raw_stv8_26[7]) : "memory");
                        }
                    }
                    float sc_10[16];
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 16; j_21++) {
                        sc_10[j_21] = _tmem_load_40[j_21] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, sc_10);
                    float _tmem_load_41[16];
                    tmem_ld_x16(&_tmem_load_41[0], taddr + (unsigned int)m128_row_base + 80);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_17[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 0], _tmem_load_41[_lp*2+1 + 0]));
                            _packed_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 2], _tmem_load_41[_lp*2+1 + 2]));
                            _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 4], _tmem_load_41[_lp*2+1 + 4]));
                            _packed_17[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 6], _tmem_load_41[_lp*2+1 + 6]));
                            _packed_17[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 8], _tmem_load_41[_lp*2+1 + 8]));
                            _packed_17[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 10], _tmem_load_41[_lp*2+1 + 10]));
                            _packed_17[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 12], _tmem_load_41[_lp*2+1 + 12]));
                            _packed_17[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_41[_lp*2 + 14], _tmem_load_41[_lp*2+1 + 14]));
                            _packed_17[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_27 = reinterpret_cast<const unsigned*>(_packed_17);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 80))), "r"(_raw_stv8_27[0]), "r"(_raw_stv8_27[1]), "r"(_raw_stv8_27[2]), "r"(_raw_stv8_27[3]), "r"(_raw_stv8_27[4]), "r"(_raw_stv8_27[5]), "r"(_raw_stv8_27[6]), "r"(_raw_stv8_27[7]) : "memory");
                        }
                    }
                    float sc_11[16];
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 16; j_22++) {
                        sc_11[j_22] = _tmem_load_41[j_22] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, sc_11);
                    float _tmem_load_42[16];
                    tmem_ld_x16(&_tmem_load_42[0], taddr + (unsigned int)m128_row_base + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_18[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 0], _tmem_load_42[_lp*2+1 + 0]));
                            _packed_18[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 2], _tmem_load_42[_lp*2+1 + 2]));
                            _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 4], _tmem_load_42[_lp*2+1 + 4]));
                            _packed_18[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 6], _tmem_load_42[_lp*2+1 + 6]));
                            _packed_18[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 8], _tmem_load_42[_lp*2+1 + 8]));
                            _packed_18[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 10], _tmem_load_42[_lp*2+1 + 10]));
                            _packed_18[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 12], _tmem_load_42[_lp*2+1 + 12]));
                            _packed_18[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_42[_lp*2 + 14], _tmem_load_42[_lp*2+1 + 14]));
                            _packed_18[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_28 = reinterpret_cast<const unsigned*>(_packed_18);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 96))), "r"(_raw_stv8_28[0]), "r"(_raw_stv8_28[1]), "r"(_raw_stv8_28[2]), "r"(_raw_stv8_28[3]), "r"(_raw_stv8_28[4]), "r"(_raw_stv8_28[5]), "r"(_raw_stv8_28[6]), "r"(_raw_stv8_28[7]) : "memory");
                        }
                    }
                    float sc_12[16];
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 16; j_23++) {
                        sc_12[j_23] = _tmem_load_42[j_23] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, sc_12);
                    float _tmem_load_43[16];
                    tmem_ld_x16(&_tmem_load_43[0], taddr + (unsigned int)m128_row_base + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (write_tapes == 1) {
                        unsigned int _packed_19[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 0], _tmem_load_43[_lp*2+1 + 0]));
                            _packed_19[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 2], _tmem_load_43[_lp*2+1 + 2]));
                            _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 4], _tmem_load_43[_lp*2+1 + 4]));
                            _packed_19[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 6], _tmem_load_43[_lp*2+1 + 6]));
                            _packed_19[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 8], _tmem_load_43[_lp*2+1 + 8]));
                            _packed_19[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 10], _tmem_load_43[_lp*2+1 + 10]));
                            _packed_19[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 12], _tmem_load_43[_lp*2+1 + 12]));
                            _packed_19[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 14], _tmem_load_43[_lp*2+1 + 14]));
                            _packed_19[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_29 = reinterpret_cast<const unsigned*>(_packed_19);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(dh_out + (dh_base_1 + 112))), "r"(_raw_stv8_29[0]), "r"(_raw_stv8_29[1]), "r"(_raw_stv8_29[2]), "r"(_raw_stv8_29[3]), "r"(_raw_stv8_29[4]), "r"(_raw_stv8_29[5]), "r"(_raw_stv8_29[6]), "r"(_raw_stv8_29[7]) : "memory");
                        }
                    }
                    float sc_13[16];
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 16; j_24++) {
                        sc_13[j_24] = _tmem_load_43[j_24] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, sc_13);
                } else {
                    float _tmem_load_44[16];
                    tmem_ld_x16(&_tmem_load_44[0], taddr + (unsigned int)m128_row_base);
                    float _tmem_load_45[16];
                    tmem_ld_x16(&_tmem_load_45[0], taddr + 128 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_7[16];
                    #pragma unroll
                    for (int j_25 = 0; j_25 < 16; j_25++) {
                        upd_7[j_25] = __fmaf_rn(_tmem_load_45[j_25], scale, _tmem_load_44[j_25]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, upd_7);
                    float _tmem_load_46[16];
                    tmem_ld_x16(&_tmem_load_46[0], taddr + (unsigned int)m128_row_base + 16);
                    float _tmem_load_47[16];
                    tmem_ld_x16(&_tmem_load_47[0], taddr + 128 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_0_1[16];
                    #pragma unroll
                    for (int j_26 = 0; j_26 < 16; j_26++) {
                        upd_0_1[j_26] = __fmaf_rn(_tmem_load_47[j_26], scale, _tmem_load_46[j_26]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, upd_0_1);
                    float _tmem_load_48[16];
                    tmem_ld_x16(&_tmem_load_48[0], taddr + (unsigned int)m128_row_base + 32);
                    float _tmem_load_49[16];
                    tmem_ld_x16(&_tmem_load_49[0], taddr + 128 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_1_1[16];
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 16; j_27++) {
                        upd_1_1[j_27] = __fmaf_rn(_tmem_load_49[j_27], scale, _tmem_load_48[j_27]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, upd_1_1);
                    float _tmem_load_50[16];
                    tmem_ld_x16(&_tmem_load_50[0], taddr + (unsigned int)m128_row_base + 48);
                    float _tmem_load_51[16];
                    tmem_ld_x16(&_tmem_load_51[0], taddr + 128 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_2_1[16];
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 16; j_28++) {
                        upd_2_1[j_28] = __fmaf_rn(_tmem_load_51[j_28], scale, _tmem_load_50[j_28]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, upd_2_1);
                    float _tmem_load_52[16];
                    tmem_ld_x16(&_tmem_load_52[0], taddr + (unsigned int)m128_row_base + 64);
                    float _tmem_load_53[16];
                    tmem_ld_x16(&_tmem_load_53[0], taddr + 128 + (unsigned int)m128_row_base + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_3_1[16];
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 16; j_29++) {
                        upd_3_1[j_29] = __fmaf_rn(_tmem_load_53[j_29], scale, _tmem_load_52[j_29]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 64, upd_3_1);
                    float _tmem_load_54[16];
                    tmem_ld_x16(&_tmem_load_54[0], taddr + (unsigned int)m128_row_base + 80);
                    float _tmem_load_55[16];
                    tmem_ld_x16(&_tmem_load_55[0], taddr + 128 + (unsigned int)m128_row_base + 80);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_4_1[16];
                    #pragma unroll
                    for (int j_30 = 0; j_30 < 16; j_30++) {
                        upd_4_1[j_30] = __fmaf_rn(_tmem_load_55[j_30], scale, _tmem_load_54[j_30]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 80, upd_4_1);
                    float _tmem_load_56[16];
                    tmem_ld_x16(&_tmem_load_56[0], taddr + (unsigned int)m128_row_base + 96);
                    float _tmem_load_57[16];
                    tmem_ld_x16(&_tmem_load_57[0], taddr + 128 + (unsigned int)m128_row_base + 96);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_5_1[16];
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 16; j_31++) {
                        upd_5_1[j_31] = __fmaf_rn(_tmem_load_57[j_31], scale, _tmem_load_56[j_31]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 96, upd_5_1);
                    float _tmem_load_58[16];
                    tmem_ld_x16(&_tmem_load_58[0], taddr + (unsigned int)m128_row_base + 112);
                    float _tmem_load_59[16];
                    tmem_ld_x16(&_tmem_load_59[0], taddr + 128 + (unsigned int)m128_row_base + 112);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_6_1[16];
                    #pragma unroll
                    for (int j_32 = 0; j_32 < 16; j_32++) {
                        upd_6_1[j_32] = __fmaf_rn(_tmem_load_59[j_32], scale, _tmem_load_58[j_32]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 112, upd_6_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
            }
            if (write_ht == 1) {
                long long ht_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128);
                float _tmem_load_60[16];
                tmem_ld_x16(&_tmem_load_60[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_28 = 0; q_28 < 4; q_28++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_60[q_28 * 4 + 0], _tmem_load_60[q_28 * 4 + 1], _tmem_load_60[q_28 * 4 + 2], _tmem_load_60[q_28 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(q_28 * 4)) = _v4;
                    }
                }
                float _tmem_load_61[16];
                tmem_ld_x16(&_tmem_load_61[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_29 = 0; q_29 < 4; q_29++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_61[q_29 * 4 + 0], _tmem_load_61[q_29 * 4 + 1], _tmem_load_61[q_29 * 4 + 2], _tmem_load_61[q_29 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(16 + q_29 * 4)) = _v4;
                    }
                }
                float _tmem_load_62[16];
                tmem_ld_x16(&_tmem_load_62[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_30 = 0; q_30 < 4; q_30++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_62[q_30 * 4 + 0], _tmem_load_62[q_30 * 4 + 1], _tmem_load_62[q_30 * 4 + 2], _tmem_load_62[q_30 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(32 + q_30 * 4)) = _v4;
                    }
                }
                float _tmem_load_63[16];
                tmem_ld_x16(&_tmem_load_63[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_31 = 0; q_31 < 4; q_31++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_63[q_31 * 4 + 0], _tmem_load_63[q_31 * 4 + 1], _tmem_load_63[q_31 * 4 + 2], _tmem_load_63[q_31 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(48 + q_31 * 4)) = _v4;
                    }
                }
                float _tmem_load_64[16];
                tmem_ld_x16(&_tmem_load_64[0], taddr + (unsigned int)m128_row_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_32 = 0; q_32 < 4; q_32++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_64[q_32 * 4 + 0], _tmem_load_64[q_32 * 4 + 1], _tmem_load_64[q_32 * 4 + 2], _tmem_load_64[q_32 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(64 + q_32 * 4)) = _v4;
                    }
                }
                float _tmem_load_65[16];
                tmem_ld_x16(&_tmem_load_65[0], taddr + (unsigned int)m128_row_base + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_33 = 0; q_33 < 4; q_33++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_65[q_33 * 4 + 0], _tmem_load_65[q_33 * 4 + 1], _tmem_load_65[q_33 * 4 + 2], _tmem_load_65[q_33 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(80 + q_33 * 4)) = _v4;
                    }
                }
                float _tmem_load_66[16];
                tmem_ld_x16(&_tmem_load_66[0], taddr + (unsigned int)m128_row_base + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_34 = 0; q_34 < 4; q_34++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_66[q_34 * 4 + 0], _tmem_load_66[q_34 * 4 + 1], _tmem_load_66[q_34 * 4 + 2], _tmem_load_66[q_34 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(96 + q_34 * 4)) = _v4;
                    }
                }
                float _tmem_load_67[16];
                tmem_ld_x16(&_tmem_load_67[0], taddr + (unsigned int)m128_row_base + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_35 = 0; q_35 < 4; q_35++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_67[q_35 * 4 + 0], _tmem_load_67[q_35 * 4 + 1], _tmem_load_67[q_35 * 4 + 2], _tmem_load_67[q_35 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(112 + q_35 * 4)) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int head_1 = blockIdx.x % num_heads;
            int seq_1 = blockIdx.x / num_heads;
            int chain_1 = 0;
            int nchain_1 = 0;
            if (segments > 0) {
                chain_1 = seq_1 / segments;
                nchain_1 = num_seqs / segments;
            } else {
                nchain_1 = -segments;
                for (int j_33 = 1; j_33 < nchain_1; j_33++) {
                    if (seq_1 >= seq_chunk_start[j_33]) {
                        chain_1 = chain_1 + 1;
                    }
                }
            }
            int tblk_1 = nchain_1 + 1 + seq_1 + chain_1;
            int c0_1 = seq_chunk_start[tblk_1];
            int nc_1 = seq_chunk_start[tblk_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int _phase_kg_empty = 1;
            unsigned int _phase_qd_empty = 1;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                if (nc_1 > 0) {
                    int trow0 = row0_1 + (nc_1 - 1) * 64;
                    mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg * 8192), (&kg_tma), seg * 64, head_1, trow0, kg_full_addr);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qd_empty_addr, _phase_qd_empty);
                    mbarrier_arrive_expect_tx(qd_full_addr, 32768);
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_1 * 8192), (&qg_tma), seg_1 * 64, head_1, trow0, qd_full_addr);
                        tma_3d_gmem2smem(do_panel_addr + (unsigned int)(seg_1 * 8192), (&do_tma), seg_1 * 64, head_1, trow0, qd_full_addr);
                    }
                    _phase_qd_empty ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    tma_3d_gmem2smem(aqk_dst_addr, (&aqk_tma), 0, head_1, trow0, aqk_full_addr);
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    #pragma unroll
                    for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)(seg_2 * 8192), (&w_tma), seg_2 * 64, head_1, trow0, w_full_addr);
                    }
                    _phase_w_empty ^= 1;
                }
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 1; rc_1 < nc_1; rc_1++) {
                    int c_1 = nc_1 - 1 - rc_1;
                    int trow_1 = row0_1 + c_1 * 64;
                    mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg_3 * 8192), (&kg_tma), seg_3 * 64, head_1, trow_1, kg_full_addr);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qd_empty_addr, _phase_qd_empty);
                    mbarrier_arrive_expect_tx(qd_full_addr, 32768);
                    #pragma unroll
                    for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_4 * 8192), (&qg_tma), seg_4 * 64, head_1, trow_1, qd_full_addr);
                        tma_3d_gmem2smem(do_panel_addr + (unsigned int)(seg_4 * 8192), (&do_tma), seg_4 * 64, head_1, trow_1, qd_full_addr);
                    }
                    _phase_qd_empty ^= 1;
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    #pragma unroll
                    for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)(seg_5 * 8192), (&w_tma), seg_5 * 64, head_1, trow_1, w_full_addr);
                    }
                    _phase_w_empty ^= 1;
                    mbarrier_arrive_expect_tx(aqk_full_addr, 8192);
                    tma_3d_gmem2smem(aqk_dst_addr, (&aqk_tma), 0, head_1, trow_1, aqk_full_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int head_2 = blockIdx.x % num_heads;
            int seq_2 = blockIdx.x / num_heads;
            int chain_2 = 0;
            int nchain_2 = 0;
            if (segments > 0) {
                chain_2 = seq_2 / segments;
                nchain_2 = num_seqs / segments;
            } else {
                nchain_2 = -segments;
                for (int j_34 = 1; j_34 < nchain_2; j_34++) {
                    if (seq_2 >= seq_chunk_start[j_34]) {
                        chain_2 = chain_2 + 1;
                    }
                }
            }
            int tblk_2 = nchain_2 + 1 + seq_2 + chain_2;
            int c0_2 = seq_chunk_start[tblk_2];
            int nc_2 = seq_chunk_start[tblk_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int _phase_kg_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_qd_full = 0;
            unsigned int _phase_aqk_full_0 = 0;
            unsigned int _phase_dv_read_0 = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            unsigned int _phase_w_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(kg_full_addr, _phase_kg_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    mbarrier_wait(qd_full_addr, _phase_qd_full);
                    mbarrier_wait(aqk_full_addr, _phase_aqk_full_0);
                    _phase_aqk_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = (((aqk_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _mma_b_lo_0 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _mma_a_lo_0 = (((aqk_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _mma_b_lo_0 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 69305488;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_dv), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((kg_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_1 = (((dhs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_0 = ((kg_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_1 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_dv), "r"(1) : "memory");
                    tcgen05_commit(dv_ready_addr);
                    tcgen05_commit(kg_empty_addr);
                    _phase_kg_full ^= 1;
                    mbarrier_wait(dv_read_addr, _phase_dv_read_0);
                    _phase_dv_read_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((qg_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = (((qg_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_mma_b_lo_0), "r"(tmem_acc_p), "r"(0) : "memory");
                    tcgen05_commit(qd_empty_addr);
                    _phase_qd_full ^= 1;
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    mbarrier_wait(w_full_addr, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_4 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_3 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_4 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(_smem_desc_lo_3), "r"(_smem_desc_lo_4), "r"(tmem_acc_dh), "r"(1), "r"(((uint32_t)(136414352) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(w_empty_addr);
                    _phase_w_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
