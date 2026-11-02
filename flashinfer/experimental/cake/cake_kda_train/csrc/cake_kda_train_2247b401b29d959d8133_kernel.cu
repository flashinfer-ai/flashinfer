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
#define TMEM_NCOLS 128
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 64
#define TMEM_ACC_P_OFFSET 64
#define NUM_KG_PIPE_STAGES 1
#define NUM_QG_PIPE_STAGES 1
#define NUM_AD_PIPE_STAGES 2
#define NUM_W_PIPE_STAGES 1
#define NUM_ONE_STAGES 1
#define SMEM_KG_PANEL_OFF 1024
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_QG_PANEL_OFF 17408
#define SMEM_QG_PANEL_STAGE_BYTES 8192
#define SMEM_QG_PANEL_STRIDE 8192
#define SMEM_AQK_PANEL_OFF 50176
#define SMEM_AQK_PANEL_STAGE_BYTES 8192
#define SMEM_AQK_PANEL_STRIDE 16384
#define SMEM_DO_PANEL_OFF 58368
#define SMEM_DO_PANEL_STAGE_BYTES 8192
#define SMEM_DO_PANEL_STRIDE 16384
#define SMEM_KG_OP_OFF 1024
#define SMEM_KG_OP_STAGE_BYTES 16384
#define SMEM_KG_OP_STRIDE 16384
#define SMEM_QG_MN_OFF 17408
#define SMEM_QG_MN_STAGE_BYTES 16384
#define SMEM_QG_MN_STRIDE 16384
#define SMEM_W_K_OFF 33792
#define SMEM_W_K_STAGE_BYTES 16384
#define SMEM_W_K_STRIDE 16384
#define SMEM_AQK_MN_OFF 50176
#define SMEM_AQK_MN_STAGE_BYTES 8192
#define SMEM_AQK_MN_STRIDE 16384
#define SMEM_DO_MN_OFF 58368
#define SMEM_DO_MN_STAGE_BYTES 8192
#define SMEM_DO_MN_STRIDE 16384
#define SMEM_DHS_OFF 82944
#define SMEM_DHS_STAGE_BYTES 16384
#define SMEM_DHS_STRIDE 16384
#define SMEM_DHS_MN_OFF 82944
#define SMEM_DHS_MN_STAGE_BYTES 16384
#define SMEM_DHS_MN_STRIDE 16384
#define SMEM_BDV_OFF 99328
#define SMEM_BDV_STAGE_BYTES 8192
#define SMEM_BDV_STRIDE 8192
#define SMEM_BDV_MN_OFF 99328
#define SMEM_BDV_MN_STAGE_BYTES 8192
#define SMEM_BDV_MN_STRIDE 8192
#define SMEM_TOTAL 107520
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_2247b401b29d959d8133(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ gk, const __grid_constant__ CUtensorMap dh_out, const __grid_constant__ CUtensorMap dv2, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int segments, float scale, float* __restrict__ dht, float* __restrict__ ht_out, int use_dht, int write_tapes, int write_ht, int* __restrict__ chunk_bos, int* __restrict__ chunk_len)
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
    #define qg_full_addr (mbar_base + 16)
    #define qg_empty_addr (mbar_base + 24)
    #define ad_full_addr (mbar_base + 32)
    #define ad_empty_addr (mbar_base + 48)
    #define w_full_addr (mbar_base + 64)
    #define w_empty_addr (mbar_base + 72)
    #define dhs_ready_addr (mbar_base + 80)
    #define dv_ready_addr (mbar_base + 88)
    #define dv_read_addr (mbar_base + 96)
    #define bdv_ready_addr (mbar_base + 104)
    #define dh_ready_addr (mbar_base + 112)
    #define dh_tape_read_addr (mbar_base + 120)
    #define dv2_tape_read_addr (mbar_base + 128)
    #define do_ready_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_panel_addr = smem + 1024;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_panel_addr = smem + 17408;
    __nv_bfloat16* aqk_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int aqk_panel_addr = smem + 50176;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int do_panel_addr = smem + 58368;
    __nv_bfloat16* kg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_op_addr = smem + 1024;
    __nv_bfloat16* qg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_mn_addr = smem + 17408;
    __nv_bfloat16* w_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_k_addr = smem + 33792;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int aqk_mn_addr = smem + 50176;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int do_mn_addr = smem + 58368;
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int dhs_addr = smem + 82944;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int dhs_mn_addr = smem + 82944;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_addr = smem + 99328;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int bdv_mn_addr = smem + 99328;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kg_pipe' ---
            // kg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'qg_pipe' ---
            // qg_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // qg_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'ad_pipe' ---
            // ad_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // ad_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'w_pipe' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // w_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'one' ---
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // dv_read: 1 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // dh_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // dv2_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // do_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 136, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 64;
    const int tmem_acc_p = taddr + 64;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int seq = blockIdx.x / (2 * num_heads);
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
            int vbase = half * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            float init_row[16];
            if (use_dht == 1) {
                long long dht_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
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
                for (int q_0 = 0; q_0 < 4; q_0++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(16 + q_0 * 4));
                        init_row[q_0 * 4 + 0] = _v4.x;
                        init_row[q_0 * 4 + 1] = _v4.y;
                        init_row[q_0 * 4 + 2] = _v4.z;
                        init_row[q_0 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, init_row);
                #pragma unroll
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(32 + q_1 * 4));
                        init_row[q_1 * 4 + 0] = _v4.x;
                        init_row[q_1 * 4 + 1] = _v4.y;
                        init_row[q_1 * 4 + 2] = _v4.z;
                        init_row[q_1 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, init_row);
                #pragma unroll
                for (int q_2 = 0; q_2 < 4; q_2++) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(dht + dht_base + (long long)(48 + q_2 * 4));
                        init_row[q_2 * 4 + 0] = _v4.x;
                        init_row[q_2 * 4 + 1] = _v4.y;
                        init_row[q_2 * 4 + 2] = _v4.z;
                        init_row[q_2 * 4 + 3] = _v4.w;
                    }
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, init_row);
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
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float gk_pre = 0.0f;
            float decay = 1.0f;
            if (nc > 0) {
                long long gn_next = ((long long)(row0 + nc * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                gk_pre = gk[gn_next];
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
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_3 * 8 / 64 * 16384 + k_row * 128 + q_3 * 8 % 64 * 2 ^ (q_3 * 8 / 64 * 16384 + k_row * 128 + q_3 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[q_3 * 4]), "r"(_tmem_load_0_bf16[q_3 * 4 + 1]), "r"(_tmem_load_0_bf16[q_3 * 4 + 2]), "r"(_tmem_load_0_bf16[q_3 * 4 + 3]) : "memory");
                }
                float _tmem_load_r_0[16];
                tmem_ld_x16(&_tmem_load_r_0[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_r_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_r_0[_lp*2 + 0], _tmem_load_r_0[_lp*2+1 + 0]));
                    _tmem_load_r_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_4 * 8) / 64 * 16384 + k_row * 128 + (16 + q_4 * 8) % 64 * 2 ^ ((16 + q_4 * 8) / 64 * 16384 + k_row * 128 + (16 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_r_bf16_0[q_4 * 4]), "r"(_tmem_load_r_bf16_0[q_4 * 4 + 1]), "r"(_tmem_load_r_bf16_0[q_4 * 4 + 2]), "r"(_tmem_load_r_bf16_0[q_4 * 4 + 3]) : "memory");
                }
                float _tmem_load_r_1[16];
                tmem_ld_x16(&_tmem_load_r_1[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_r_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_r_1[_lp*2 + 0], _tmem_load_r_1[_lp*2+1 + 0]));
                    _tmem_load_r_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_5 * 8) / 64 * 16384 + k_row * 128 + (32 + q_5 * 8) % 64 * 2 ^ ((32 + q_5 * 8) / 64 * 16384 + k_row * 128 + (32 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_r_bf16_1[q_5 * 4]), "r"(_tmem_load_r_bf16_1[q_5 * 4 + 1]), "r"(_tmem_load_r_bf16_1[q_5 * 4 + 2]), "r"(_tmem_load_r_bf16_1[q_5 * 4 + 3]) : "memory");
                }
                float _tmem_load_r_2[16];
                tmem_ld_x16(&_tmem_load_r_2[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _tmem_load_r_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_r_2[_lp*2 + 0], _tmem_load_r_2[_lp*2+1 + 0]));
                    _tmem_load_r_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_6 = 0; q_6 < 2; q_6++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_6 * 8) / 64 * 16384 + k_row * 128 + (48 + q_6 * 8) % 64 * 2 ^ ((48 + q_6 * 8) / 64 * 16384 + k_row * 128 + (48 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_r_bf16_2[q_6 * 4]), "r"(_tmem_load_r_bf16_2[q_6 * 4 + 1]), "r"(_tmem_load_r_bf16_2[q_6 * 4 + 2]), "r"(_tmem_load_r_bf16_2[q_6 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                float _exp2_4 = approx_exp2(gk_pre);
                decay = _exp2_4;
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_0[16];
                #pragma unroll
                for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                    sc_0[j_1_0] = _tmem_load_1[j_1_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_1[16];
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                    sc_1[j_1_1] = _tmem_load_2[j_1_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_2[16];
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                    sc_2[j_1_2] = _tmem_load_3[j_1_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_2);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float sc_3[16];
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                    sc_3[j_1_3] = _tmem_load_4[j_1_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_3);
            }
            unsigned int _phase_dv2_tape_read_0 = 0;
            unsigned int _phase_dv_ready_0 = 0;
            unsigned int _phase_dh_tape_read_0 = 0;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int rc = 0; rc < nc; rc++) {
                int c = nc - 1 - rc;
                if (write_tapes == 1) {
                    if (rc >= 1) {
                        mbarrier_wait(dv2_tape_read_addr, _phase_dv2_tape_read_0);
                        _phase_dv2_tape_read_0 ^= 1;
                    }
                }
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float tmem_load[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15]))
                    : "r"(taddr + 64 + (unsigned int)m64_row_base));
                float tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15]))
                    : "r"(taddr + 64 + (unsigned int)m64_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dv_read_addr);
                int col = lane_half * 16;
                unsigned int tmem_load_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load[_lp*2 + 0], tmem_load[_lp*2+1 + 0]));
                    tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col + q_7 * 8) / 64 * 8192 + t_row * 128 + (col + q_7 * 8) % 64 * 2 ^ ((col + q_7 * 8) / 64 * 8192 + t_row * 128 + (col + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_8_bf16[q_7 * 4]), "r"(tmem_load_8_bf16[q_7 * 4 + 1]), "r"(tmem_load_8_bf16[q_7 * 4 + 2]), "r"(tmem_load_8_bf16[q_7 * 4 + 3]) : "memory");
                }
                int col_0 = 32 + lane_half * 16;
                unsigned int tmem_load_8_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_0[_lp*2 + 0], tmem_load_0[_lp*2+1 + 0]));
                    tmem_load_8_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col_0 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_8 * 8) % 64 * 2 ^ ((col_0 + q_8 * 8) / 64 * 8192 + t_row * 128 + (col_0 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_8_bf16_0[q_8 * 4]), "r"(tmem_load_8_bf16_0[q_8 * 4 + 1]), "r"(tmem_load_8_bf16_0[q_8 * 4 + 2]), "r"(tmem_load_8_bf16_0[q_8 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                if (nc > rc + 1) {
                    long long gn_next_1 = ((long long)(row0 + c * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                    gk_pre = gk[gn_next_1];
                }
                if (write_tapes == 1) {
                    mbarrier_wait(dh_tape_read_addr, _phase_dh_tape_read_0);
                    _phase_dh_tape_read_0 ^= 1;
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (nc > rc + 1) {
                    float _exp2_5 = approx_exp2(gk_pre);
                    decay = _exp2_5;
                    float _tmem_load_10[16];
                    tmem_ld_x16(&_tmem_load_10[0], taddr + (unsigned int)m128_row_base);
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 64 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd[16];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        upd[j_1] = __fmaf_rn(_tmem_load_11[j_1], scale, _tmem_load_10[j_1]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, upd);
                    unsigned int upd_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        upd_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 2; q_9++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_9 * 8 / 64 * 16384 + k_row * 128 + q_9 * 8 % 64 * 2 ^ (q_9 * 8 / 64 * 16384 + k_row * 128 + q_9 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16[q_9 * 4]), "r"(upd_bf16[q_9 * 4 + 1]), "r"(upd_bf16[q_9 * 4 + 2]), "r"(upd_bf16[q_9 * 4 + 3]) : "memory");
                    }
                    float tmem_load_1[16];
                    tmem_ld_x16(&tmem_load_1[0], taddr + (unsigned int)m128_row_base + 16);
                    float tmem_load_1_0[16];
                    tmem_ld_x16(&tmem_load_1_0[0], taddr + 64 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_0[16];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2++) {
                        upd_0[j_2] = __fmaf_rn(tmem_load_1_0[j_2], scale, tmem_load_1[j_2]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, upd_0);
                    unsigned int upd_bf16_r_0[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_0[_lp*2 + 0], upd_0[_lp*2+1 + 0]));
                        upd_bf16_r_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_10 = 0; q_10 < 2; q_10++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_10 * 8) / 64 * 16384 + k_row * 128 + (16 + q_10 * 8) % 64 * 2 ^ ((16 + q_10 * 8) / 64 * 16384 + k_row * 128 + (16 + q_10 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_r_0[q_10 * 4]), "r"(upd_bf16_r_0[q_10 * 4 + 1]), "r"(upd_bf16_r_0[q_10 * 4 + 2]), "r"(upd_bf16_r_0[q_10 * 4 + 3]) : "memory");
                    }
                    float tmem_load_2[16];
                    tmem_ld_x16(&tmem_load_2[0], taddr + (unsigned int)m128_row_base + 32);
                    float tmem_load_1_1[16];
                    tmem_ld_x16(&tmem_load_1_1[0], taddr + 64 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_1[16];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 16; j_3++) {
                        upd_1[j_3] = __fmaf_rn(tmem_load_1_1[j_3], scale, tmem_load_2[j_3]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, upd_1);
                    unsigned int upd_bf16_r_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_1[_lp*2 + 0], upd_1[_lp*2+1 + 0]));
                        upd_bf16_r_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_11 = 0; q_11 < 2; q_11++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_11 * 8) / 64 * 16384 + k_row * 128 + (32 + q_11 * 8) % 64 * 2 ^ ((32 + q_11 * 8) / 64 * 16384 + k_row * 128 + (32 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_r_1[q_11 * 4]), "r"(upd_bf16_r_1[q_11 * 4 + 1]), "r"(upd_bf16_r_1[q_11 * 4 + 2]), "r"(upd_bf16_r_1[q_11 * 4 + 3]) : "memory");
                    }
                    float tmem_load_3[16];
                    tmem_ld_x16(&tmem_load_3[0], taddr + (unsigned int)m128_row_base + 48);
                    float tmem_load_1_2[16];
                    tmem_ld_x16(&tmem_load_1_2[0], taddr + 64 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_2[16];
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 16; j_4++) {
                        upd_2[j_4] = __fmaf_rn(tmem_load_1_2[j_4], scale, tmem_load_3[j_4]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, upd_2);
                    unsigned int upd_bf16_r_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd_2[_lp*2 + 0], upd_2[_lp*2+1 + 0]));
                        upd_bf16_r_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_12 = 0; q_12 < 2; q_12++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_12 * 8) / 64 * 16384 + k_row * 128 + (48 + q_12 * 8) % 64 * 2 ^ ((48 + q_12 * 8) / 64 * 16384 + k_row * 128 + (48 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_r_2[q_12 * 4]), "r"(upd_bf16_r_2[q_12 * 4 + 1]), "r"(upd_bf16_r_2[q_12 * 4 + 2]), "r"(upd_bf16_r_2[q_12 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dhs_ready_addr);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sc_4[16];
                    #pragma unroll
                    for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                        sc_4[j_1_4] = _tmem_load_5[j_1_4] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc_4);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sc_5[16];
                    #pragma unroll
                    for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                        sc_5[j_1_5] = _tmem_load_6[j_1_5] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_5);
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sc_6[16];
                    #pragma unroll
                    for (int j_1_6 = 0; j_1_6 < 16; j_1_6++) {
                        sc_6[j_1_6] = _tmem_load_7[j_1_6] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_6);
                    float _tmem_load_12[16];
                    tmem_ld_x16(&_tmem_load_12[0], taddr + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sc_7[16];
                    #pragma unroll
                    for (int j_1_7 = 0; j_1_7 < 16; j_1_7++) {
                        sc_7[j_1_7] = _tmem_load_12[j_1_7] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_7);
                } else {
                    float tmem_load_4[16];
                    tmem_ld_x16(&tmem_load_4[0], taddr + (unsigned int)m128_row_base);
                    float tmem_load_1_3[16];
                    tmem_ld_x16(&tmem_load_1_3[0], taddr + 64 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_3[16];
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 16; j_5++) {
                        upd_3[j_5] = __fmaf_rn(tmem_load_1_3[j_5], scale, tmem_load_4[j_5]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, upd_3);
                    float tmem_load_5[16];
                    tmem_ld_x16(&tmem_load_5[0], taddr + (unsigned int)m128_row_base + 16);
                    float tmem_load_1_3_1[16];
                    tmem_ld_x16(&tmem_load_1_3_1[0], taddr + 64 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_3_1[16];
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 16; j_6++) {
                        upd_3_1[j_6] = __fmaf_rn(tmem_load_1_3_1[j_6], scale, tmem_load_5[j_6]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, upd_3_1);
                    float tmem_load_6[16];
                    tmem_ld_x16(&tmem_load_6[0], taddr + (unsigned int)m128_row_base + 32);
                    float tmem_load_1_4[16];
                    tmem_ld_x16(&tmem_load_1_4[0], taddr + 64 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_4[16];
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 16; j_7++) {
                        upd_4[j_7] = __fmaf_rn(tmem_load_1_4[j_7], scale, tmem_load_6[j_7]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, upd_4);
                    float tmem_load_7[16];
                    tmem_ld_x16(&tmem_load_7[0], taddr + (unsigned int)m128_row_base + 48);
                    float tmem_load_1_5[16];
                    tmem_ld_x16(&tmem_load_1_5[0], taddr + 64 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float upd_5[16];
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 16; j_8++) {
                        upd_5[j_8] = __fmaf_rn(tmem_load_1_5[j_8], scale, tmem_load_7[j_8]);
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, upd_5);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
            }
            if (write_ht == 1) {
                long long ht_base = ((long long)seq * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
                float _tmem_load_30[16];
                tmem_ld_x16(&_tmem_load_30[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_13 = 0; q_13 < 4; q_13++) {
                    {
                        float4 _v4 = make_float4(_tmem_load_30[q_13 * 4 + 0], _tmem_load_30[q_13 * 4 + 1], _tmem_load_30[q_13 * 4 + 2], _tmem_load_30[q_13 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(q_13 * 4)) = _v4;
                    }
                }
                float tmem_load_r_0[16];
                tmem_ld_x16(&tmem_load_r_0[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_3_1 = 0; q_3_1 < 4; q_3_1++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_0[q_3_1 * 4 + 0], tmem_load_r_0[q_3_1 * 4 + 1], tmem_load_r_0[q_3_1 * 4 + 2], tmem_load_r_0[q_3_1 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(16 + q_3_1 * 4)) = _v4;
                    }
                }
                float tmem_load_r_1[16];
                tmem_ld_x16(&tmem_load_r_1[0], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_4_1 = 0; q_4_1 < 4; q_4_1++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_1[q_4_1 * 4 + 0], tmem_load_r_1[q_4_1 * 4 + 1], tmem_load_r_1[q_4_1 * 4 + 2], tmem_load_r_1[q_4_1 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(32 + q_4_1 * 4)) = _v4;
                    }
                }
                float tmem_load_r_2[16];
                tmem_ld_x16(&tmem_load_r_2[0], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int q_5_1 = 0; q_5_1 < 4; q_5_1++) {
                    {
                        float4 _v4 = make_float4(tmem_load_r_2[q_5_1 * 4 + 0], tmem_load_r_2[q_5_1 * 4 + 1], tmem_load_r_2[q_5_1 * 4 + 2], tmem_load_r_2[q_5_1 * 4 + 3]);
                        *reinterpret_cast<float4*>(ht_out + ht_base + (long long)(48 + q_5_1 * 4)) = _v4;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int half_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int seq_1 = blockIdx.x / (2 * num_heads);
            int chain_1 = 0;
            int nchain_1 = 0;
            if (segments > 0) {
                chain_1 = seq_1 / segments;
                nchain_1 = num_seqs / segments;
            } else {
                nchain_1 = -segments;
                chain_1 = 0;
                for (int j_9 = 1; j_9 < nchain_1; j_9++) {
                    if (seq_1 >= seq_chunk_start[j_9]) {
                        chain_1 = chain_1 + 1;
                    }
                }
            }
            int tblk_1 = nchain_1 + 1 + seq_1 + chain_1;
            int c0_1 = seq_chunk_start[tblk_1];
            int nc_1 = seq_chunk_start[tblk_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int vbase_1 = half_1 * 64;
            unsigned int ad = 0;
            int tbos0 = 0;
            int clen_last = 64;
            if (nc_1 > 0) {
                tbos0 = chunk_bos[c0_1];
                clen_last = chunk_len[c0_1 + nc_1 - 1];
            }
            unsigned int _phase_ad_empty = 1;
            unsigned int _phase_kg_empty = 1;
            unsigned int _phase_qg_empty = 1;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                if (nc_1 > 0) {
                    int trow0 = row0_1 + (nc_1 - 1) * 64;
                    int tok0 = tbos0 + (nc_1 - 1) * 64;
                    mbarrier_wait(ad_empty_addr + (ad) * 8, _phase_ad_empty);
                    mbarrier_arrive_expect_tx(ad_full_addr + (ad) * 8, 16384);
                    tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok0, ad_full_addr + (ad) * 8);
                    tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok0, ad_full_addr + (ad) * 8);
                    ad += 1;
                    if (ad == 2) { ad = 0; _phase_ad_empty ^= 1; }
                    mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg * 8192), (&kg_tma), seg * 64, head_1, trow0, kg_full_addr);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qg_empty_addr, _phase_qg_empty);
                    mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_1 * 8192), (&qg_tma), seg_1 * 64, head_1, trow0, qg_full_addr);
                    }
                    _phase_qg_empty ^= 1;
                    if (nc_1 > 1) {
                        mbarrier_wait(ad_empty_addr + (ad) * 8, _phase_ad_empty);
                        mbarrier_arrive_expect_tx(ad_full_addr + (ad) * 8, 16384);
                        tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok0 - 64, ad_full_addr + (ad) * 8);
                        tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok0 - 64, ad_full_addr + (ad) * 8);
                        ad += 1;
                        if (ad == 2) { ad = 0; _phase_ad_empty ^= 1; }
                    }
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    tma_3d_gmem2smem(w_k_addr, (&w_tma), 0, head_1, trow0 * 2, w_full_addr);
                    _phase_w_empty ^= 1;
                }
            }
            if (clen_last < 64) {
                mbarrier_wait(ad_full_addr, 0);
                unsigned int zeros[4];
                zeros[0] = 0;
                zeros[1] = 0;
                zeros[2] = 0;
                zeros[3] = 0;
                int zrow = lane >> 3;
                unsigned int zcol = (unsigned int)((lane & 7) * 16);
                #pragma unroll
                for (int grp = 0; grp < 16; grp++) {
                    if (clen_last < (grp + 1) * 4) {
                        int trow_z = grp * 4 + zrow;
                        if (trow_z >= clen_last) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(do_panel_addr + (unsigned int)(trow_z * 128) + zcol), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(do_ready_addr);
            }
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 0; rc_1 < nc_1; rc_1++) {
                    int c_1 = nc_1 - 1 - rc_1;
                    int trow = row0_1 + c_1 * 64;
                    int tok = tbos0 + c_1 * 64;
                    if (write_tapes == 1) {
                        int chunk = (c0_1 + c_1) * num_heads + head_1;
                        mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                        _phase_dhs_ready_0 ^= 1;
                        tma_store_3d((&dh_out), vbase_1, 0, chunk, dhs_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (nc_1 > rc_1 + 1) {
                        mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                        mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg_2 * 8192), (&kg_tma), seg_2 * 64, head_1, trow - 64, kg_full_addr);
                        }
                        _phase_kg_empty ^= 1;
                    }
                    if (write_tapes == 1) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(dh_tape_read_addr);
                    }
                    if (nc_1 > rc_1 + 1) {
                        mbarrier_wait(qg_empty_addr, _phase_qg_empty);
                        mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                        #pragma unroll
                        for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                            tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_3 * 8192), (&qg_tma), seg_3 * 64, head_1, trow - 64, qg_full_addr);
                        }
                        _phase_qg_empty ^= 1;
                    }
                    if (nc_1 > rc_1 + 2) {
                        mbarrier_wait(ad_empty_addr + (ad) * 8, _phase_ad_empty);
                        mbarrier_arrive_expect_tx(ad_full_addr + (ad) * 8, 16384);
                        tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok - 128, ad_full_addr + (ad) * 8);
                        tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok - 128, ad_full_addr + (ad) * 8);
                        ad += 1;
                        if (ad == 2) { ad = 0; _phase_ad_empty ^= 1; }
                    }
                    if (write_tapes == 1) {
                        mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                        _phase_bdv_ready_0 ^= 1;
                        tma_store_3d((&dv2), vbase_1, head_1, trow, bdv_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (nc_1 > rc_1 + 1) {
                        mbarrier_wait(w_empty_addr, _phase_w_empty);
                        mbarrier_arrive_expect_tx(w_full_addr, 16384);
                        tma_3d_gmem2smem(w_k_addr, (&w_tma), 0, head_1, (trow - 64) * 2, w_full_addr);
                        _phase_w_empty ^= 1;
                    }
                    if (write_tapes == 1) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(dv2_tape_read_addr);
                    }
                }
                if (write_tapes == 1) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int half_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int seq_2 = blockIdx.x / (2 * num_heads);
            int chain_2 = 0;
            int nchain_2 = 0;
            if (segments > 0) {
                chain_2 = seq_2 / segments;
                nchain_2 = num_seqs / segments;
            } else {
                nchain_2 = -segments;
                chain_2 = 0;
                for (int j_10 = 1; j_10 < nchain_2; j_10++) {
                    if (seq_2 >= seq_chunk_start[j_10]) {
                        chain_2 = chain_2 + 1;
                    }
                }
            }
            int tblk_2 = nchain_2 + 1 + seq_2 + chain_2;
            int c0_2 = seq_chunk_start[tblk_2];
            int nc_2 = seq_chunk_start[tblk_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int ams = 0;
            int clen_last_m = 64;
            if (nc_2 > 0) {
                clen_last_m = chunk_len[c0_2 + nc_2 - 1];
            }
            const int lane_m = lane;
            if (clen_last_m < 64) {
                mbarrier_wait(ad_full_addr, 0);
                unsigned int zeros_1[4];
                zeros_1[0] = 0;
                zeros_1[1] = 0;
                zeros_1[2] = 0;
                zeros_1[3] = 0;
                int npad = (64 - clen_last_m) * 8;
                #pragma unroll 1
                for (int idx = lane_m; idx < npad; idx += 32) {
                    int off = (clen_last_m + (idx >> 3)) * 128 + (idx & 7) * 16;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(aqk_panel_addr + (unsigned int)off), "r"(zeros_1[0]), "r"(zeros_1[1]), "r"(zeros_1[2]), "r"(zeros_1[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
            }
            unsigned int _phase_kg_full = 0;
            unsigned int _phase_dhs_ready_0_1 = 0;
            unsigned int _phase_ad_full = 0;
            unsigned int _phase_dv_read_0 = 0;
            unsigned int _phase_qg_full = 0;
            unsigned int _phase_bdv_ready_0_1 = 0;
            unsigned int _phase_w_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(kg_full_addr, _phase_kg_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0_1);
                    _phase_dhs_ready_0_1 ^= 1;
                    mbarrier_wait(ad_full_addr + (ams) * 8, _phase_ad_full);
                    if (rc_2 == 0) {
                        if (clen_last_m < 64) {
                            mbarrier_wait(do_ready_addr, 0);
                        }
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((((aqk_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (ams) * 1024;
                    int _smem_desc_lo_1 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (ams) * 1024;
#else
                    int _smem_desc_lo_0 = ((((aqk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (ams) * 1024;
                    int _smem_desc_lo_1 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (ams) * 1024;
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
                    "mov.b32 id, 68256912;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_dv), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((kg_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((dhs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_2 = ((kg_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "mov.b32 id, 68224144;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_dv), "r"(1) : "memory");
                    tcgen05_commit(dv_ready_addr);
                    tcgen05_commit(kg_empty_addr);
                    _phase_kg_full ^= 1;
                    mbarrier_wait(dv_read_addr, _phase_dv_read_0);
                    _phase_dv_read_0 ^= 1;
                    mbarrier_wait(qg_full_addr, _phase_qg_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((qg_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_5 = ((((do_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (ams) * 1024;
#else
                    int _smem_desc_lo_4 = (((qg_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_5 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (ams) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_p), "r"(0) : "memory");
                    tcgen05_commit(qg_empty_addr);
                    _phase_qg_full ^= 1;
                    tcgen05_commit(ad_empty_addr + (ams) * 8);
                    ams += 1;
                    if (ams == 2) { ams = 0; _phase_ad_full ^= 1; }
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0_1);
                    _phase_bdv_ready_0_1 ^= 1;
                    mbarrier_wait(w_full_addr, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((w_k_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_7 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = ((w_k_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_7 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_dh), "r"(1), "r"((((uint32_t)(135333008)) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
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
