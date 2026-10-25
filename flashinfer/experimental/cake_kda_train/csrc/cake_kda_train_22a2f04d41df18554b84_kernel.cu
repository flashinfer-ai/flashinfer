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
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 64
#define TMEM_ACC_P_OFFSET 128
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
#define SMEM_W_PANEL_OFF 33792
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
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
#define SMEM_W_MN_OFF 33792
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_W_KOP_OFF 33792
#define SMEM_W_KOP_STAGE_BYTES 16384
#define SMEM_W_KOP_STRIDE 16384
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_22a2f04d41df18554b84(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ gk, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, int num_heads, int* __restrict__ seq_chunk_start, float scale, int* __restrict__ chunk_bos, int* __restrict__ chunk_len)
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
    #define bdv_ready_addr (mbar_base + 96)
    #define dh_ready_addr (mbar_base + 104)
    #define do_ready_addr (mbar_base + 112)

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
    __nv_bfloat16* aqk_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int aqk_panel_addr = smem + 50176;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int do_panel_addr = smem + 58368;
    __nv_bfloat16* kg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_op_addr = smem + 1024;
    __nv_bfloat16* qg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int qg_mn_addr = smem + 17408;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* w_kop = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_kop_addr = smem + 33792;
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

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 15 barriers)
    // Mbarriers at smem_raw[0..120)

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
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // do_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 112, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 120);
    if (warp == 0) {
        int _tmem_hold = smem + 120;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 64;
    const int tmem_acc_p = taddr + 128;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int rank = blockIdx.x / (2 * num_heads);
            int seq = rank;
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int vbase = half * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            float zeros16[16];
            zeros16[0] = 0.0f;
            zeros16[1] = 0.0f;
            zeros16[2] = 0.0f;
            zeros16[3] = 0.0f;
            zeros16[4] = 0.0f;
            zeros16[5] = 0.0f;
            zeros16[6] = 0.0f;
            zeros16[7] = 0.0f;
            zeros16[8] = 0.0f;
            zeros16[9] = 0.0f;
            zeros16[10] = 0.0f;
            zeros16[11] = 0.0f;
            zeros16[12] = 0.0f;
            zeros16[13] = 0.0f;
            zeros16[14] = 0.0f;
            zeros16[15] = 0.0f;
            tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, zeros16);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float gk_pre = 0.0f;
            float upd[64];
            float decay = 1.0f;
            if (nc > 0) {
                long long gn_next = ((long long)(row0 + nc * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                gk_pre = gk[gn_next];
                long long dh_base = ((long long)(c0 + nc - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
                tmem_ld_x16(&upd[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 2], upd[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 4], upd[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 6], upd[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 8], upd[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 10], upd[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 12], upd[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 14], upd[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_0 = reinterpret_cast<const unsigned*>(_packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dh_out + dh_base)), "r"(_raw_stv8_0[0]), "r"(_raw_stv8_0[1]), "r"(_raw_stv8_0[2]), "r"(_raw_stv8_0[3]), "r"(_raw_stv8_0[4]), "r"(_raw_stv8_0[5]), "r"(_raw_stv8_0[6]), "r"(_raw_stv8_0[7]) : "memory");
                }
                unsigned int upd_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                    upd_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 ^ (q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16[q * 4]), "r"(upd_bf16[q * 4 + 1]), "r"(upd_bf16[q * 4 + 2]), "r"(upd_bf16[q * 4 + 3]) : "memory");
                }
                tmem_ld_x16(&upd[16], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _packed_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 18], upd[_lp*2+1 + 18]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 20], upd[_lp*2+1 + 20]));
                    _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 22], upd[_lp*2+1 + 22]));
                    _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 24], upd[_lp*2+1 + 24]));
                    _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 26], upd[_lp*2+1 + 26]));
                    _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 28], upd[_lp*2+1 + 28]));
                    _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 30], upd[_lp*2+1 + 30]));
                    _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_1);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dh_out + (dh_base + 16))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                }
                unsigned int upd_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                    upd_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_1 * 8) % 64 * 2 ^ ((16 + q_1 * 8) / 64 * 16384 + k_row * 128 + (16 + q_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_0[q_1 * 4]), "r"(upd_bf16_0[q_1 * 4 + 1]), "r"(upd_bf16_0[q_1 * 4 + 2]), "r"(upd_bf16_0[q_1 * 4 + 3]) : "memory");
                }
                tmem_ld_x16(&upd[32], taddr + (unsigned int)m128_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _packed_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 32], upd[_lp*2+1 + 32]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 34], upd[_lp*2+1 + 34]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 36], upd[_lp*2+1 + 36]));
                    _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 38], upd[_lp*2+1 + 38]));
                    _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 40], upd[_lp*2+1 + 40]));
                    _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 42], upd[_lp*2+1 + 42]));
                    _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 44], upd[_lp*2+1 + 44]));
                    _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 46], upd[_lp*2+1 + 46]));
                    _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(_packed_2);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dh_out + (dh_base + 32))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                }
                unsigned int upd_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 32], upd[_lp*2+1 + 32]));
                    upd_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_2 * 8) / 64 * 16384 + k_row * 128 + (32 + q_2 * 8) % 64 * 2 ^ ((32 + q_2 * 8) / 64 * 16384 + k_row * 128 + (32 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_1[q_2 * 4]), "r"(upd_bf16_1[q_2 * 4 + 1]), "r"(upd_bf16_1[q_2 * 4 + 2]), "r"(upd_bf16_1[q_2 * 4 + 3]) : "memory");
                }
                tmem_ld_x16(&upd[48], taddr + (unsigned int)m128_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _packed_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 48], upd[_lp*2+1 + 48]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 50], upd[_lp*2+1 + 50]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 52], upd[_lp*2+1 + 52]));
                    _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 54], upd[_lp*2+1 + 54]));
                    _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 56], upd[_lp*2+1 + 56]));
                    _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 58], upd[_lp*2+1 + 58]));
                    _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 60], upd[_lp*2+1 + 60]));
                    _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 62], upd[_lp*2+1 + 62]));
                    _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_3);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dh_out + (dh_base + 48))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                }
                unsigned int upd_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 48], upd[_lp*2+1 + 48]));
                    upd_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_3 * 8) / 64 * 16384 + k_row * 128 + (48 + q_3 * 8) % 64 * 2 ^ ((48 + q_3 * 8) / 64 * 16384 + k_row * 128 + (48 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_2[q_3 * 4]), "r"(upd_bf16_2[q_3 * 4 + 1]), "r"(upd_bf16_2[q_3 * 4 + 2]), "r"(upd_bf16_2[q_3 * 4 + 3]) : "memory");
                }
                float _exp2_4 = approx_exp2(gk_pre);
                decay = _exp2_4;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                float sc[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    sc[j] = upd[j] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc);
                float sc_3[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    sc_3[j_1] = upd[16 + j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_3);
                float sc_4[16];
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    sc_4[j_2] = upd[32 + j_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_4);
                float sc_5[16];
                #pragma unroll
                for (int j_3 = 0; j_3 < 16; j_3++) {
                    sc_5[j_3] = upd[48 + j_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_5);
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
                long long t_base = ((long long)trow * (long long)num_heads + (long long)head) * 128 + (long long)vbase;
                float bdvv[32];
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + 64 + (unsigned int)m64_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_4 = 0; j_4 < 16; j_4++) {
                    bdvv[j_4] = _tmem_load_0[j_4];
                }
                unsigned int bdvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 0], bdvv[_lp*2+1 + 0]));
                    bdvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((lane_half * 16 + q_4 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_4 * 8) % 64 * 2 ^ ((lane_half * 16 + q_4 * 8) / 64 * 8192 + t_row * 128 + (lane_half * 16 + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16[q_4 * 4]), "r"(bdvv_bf16[q_4 * 4 + 1]), "r"(bdvv_bf16[q_4 * 4 + 2]), "r"(bdvv_bf16[q_4 * 4 + 3]) : "memory");
                }
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 64 + (unsigned int)m64_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_5 = 0; j_5 < 16; j_5++) {
                    bdvv[16 + j_5] = _tmem_load_1[j_5];
                }
                unsigned int bdvv_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 16], bdvv[_lp*2+1 + 16]));
                    bdvv_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_5 = 0; q_5 < 2; q_5++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((32 + lane_half * 16 + q_5 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_5 * 8) % 64 * 2 ^ ((32 + lane_half * 16 + q_5 * 8) / 64 * 8192 + t_row * 128 + (32 + lane_half * 16 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bdvv_bf16_0[q_5 * 4]), "r"(bdvv_bf16_0[q_5 * 4 + 1]), "r"(bdvv_bf16_0[q_5 * 4 + 2]), "r"(bdvv_bf16_0[q_5 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                unsigned int _packed_4[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 0], bdvv[_lp*2+1 + 0]));
                    _packed_4[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 2], bdvv[_lp*2+1 + 2]));
                    _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 4], bdvv[_lp*2+1 + 4]));
                    _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 6], bdvv[_lp*2+1 + 6]));
                    _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 8], bdvv[_lp*2+1 + 8]));
                    _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 10], bdvv[_lp*2+1 + 10]));
                    _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 12], bdvv[_lp*2+1 + 12]));
                    _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 14], bdvv[_lp*2+1 + 14]));
                    _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(_packed_4);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dv2 + (t_base + (long long)(lane_half * 16)))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                }
                unsigned int _packed_5[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 16], bdvv[_lp*2+1 + 16]));
                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 18], bdvv[_lp*2+1 + 18]));
                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 20], bdvv[_lp*2+1 + 20]));
                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 22], bdvv[_lp*2+1 + 22]));
                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 24], bdvv[_lp*2+1 + 24]));
                    _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 26], bdvv[_lp*2+1 + 26]));
                    _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 28], bdvv[_lp*2+1 + 28]));
                    _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 30], bdvv[_lp*2+1 + 30]));
                    _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_6 = reinterpret_cast<const unsigned*>(_packed_5);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dv2 + (t_base + (long long)(32 + lane_half * 16)))), "r"(_raw_stv8_6[0]), "r"(_raw_stv8_6[1]), "r"(_raw_stv8_6[2]), "r"(_raw_stv8_6[3]), "r"(_raw_stv8_6[4]), "r"(_raw_stv8_6[5]), "r"(_raw_stv8_6[6]), "r"(_raw_stv8_6[7]) : "memory");
                }
                if (nc > rc + 1) {
                    long long gn_next_1 = ((long long)(row0 + c * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                    gk_pre = gk[gn_next_1];
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (nc > rc + 1) {
                    float _exp2_7 = approx_exp2(gk_pre);
                    decay = _exp2_7;
                    long long dh_base_1 = ((long long)(c0 + c - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 128 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 16; j_6++) {
                        upd[j_6] = __fmaf_rn(_tmem_load_3[j_6], scale, _tmem_load_2[j_6]);
                    }
                    unsigned int _packed_6[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        _packed_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 2], upd[_lp*2+1 + 2]));
                        _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 4], upd[_lp*2+1 + 4]));
                        _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 6], upd[_lp*2+1 + 6]));
                        _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 8], upd[_lp*2+1 + 8]));
                        _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 10], upd[_lp*2+1 + 10]));
                        _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 12], upd[_lp*2+1 + 12]));
                        _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 14], upd[_lp*2+1 + 14]));
                        _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_8 = reinterpret_cast<const unsigned*>(_packed_6);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + dh_base_1)), "r"(_raw_stv8_8[0]), "r"(_raw_stv8_8[1]), "r"(_raw_stv8_8[2]), "r"(_raw_stv8_8[3]), "r"(_raw_stv8_8[4]), "r"(_raw_stv8_8[5]), "r"(_raw_stv8_8[6]), "r"(_raw_stv8_8[7]) : "memory");
                    }
                    unsigned int upd_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        upd_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 2; q_6++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_6 * 8 / 64 * 16384 + k_row * 128 + q_6 * 8 % 64 * 2 ^ (q_6 * 8 / 64 * 16384 + k_row * 128 + q_6 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_3[q_6 * 4]), "r"(upd_bf16_3[q_6 * 4 + 1]), "r"(upd_bf16_3[q_6 * 4 + 2]), "r"(upd_bf16_3[q_6 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 128 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 16; j_7++) {
                        upd[16 + j_7] = __fmaf_rn(_tmem_load_5[j_7], scale, _tmem_load_4[j_7]);
                    }
                    unsigned int _packed_7[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                        _packed_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 18], upd[_lp*2+1 + 18]));
                        _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 20], upd[_lp*2+1 + 20]));
                        _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 22], upd[_lp*2+1 + 22]));
                        _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 24], upd[_lp*2+1 + 24]));
                        _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 26], upd[_lp*2+1 + 26]));
                        _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 28], upd[_lp*2+1 + 28]));
                        _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 30], upd[_lp*2+1 + 30]));
                        _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_9 = reinterpret_cast<const unsigned*>(_packed_7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base_1 + 16))), "r"(_raw_stv8_9[0]), "r"(_raw_stv8_9[1]), "r"(_raw_stv8_9[2]), "r"(_raw_stv8_9[3]), "r"(_raw_stv8_9[4]), "r"(_raw_stv8_9[5]), "r"(_raw_stv8_9[6]), "r"(_raw_stv8_9[7]) : "memory");
                    }
                    unsigned int upd_bf16_0_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                        upd_bf16_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_7 = 0; q_7 < 2; q_7++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_7 * 8) / 64 * 16384 + k_row * 128 + (16 + q_7 * 8) % 64 * 2 ^ ((16 + q_7 * 8) / 64 * 16384 + k_row * 128 + (16 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_0_1[q_7 * 4]), "r"(upd_bf16_0_1[q_7 * 4 + 1]), "r"(upd_bf16_0_1[q_7 * 4 + 2]), "r"(upd_bf16_0_1[q_7 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 128 + (unsigned int)m128_row_base + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 16; j_8++) {
                        upd[32 + j_8] = __fmaf_rn(_tmem_load_7[j_8], scale, _tmem_load_6[j_8]);
                    }
                    unsigned int _packed_8[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 32], upd[_lp*2+1 + 32]));
                        _packed_8[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 34], upd[_lp*2+1 + 34]));
                        _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 36], upd[_lp*2+1 + 36]));
                        _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 38], upd[_lp*2+1 + 38]));
                        _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 40], upd[_lp*2+1 + 40]));
                        _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 42], upd[_lp*2+1 + 42]));
                        _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 44], upd[_lp*2+1 + 44]));
                        _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 46], upd[_lp*2+1 + 46]));
                        _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_10 = reinterpret_cast<const unsigned*>(_packed_8);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base_1 + 32))), "r"(_raw_stv8_10[0]), "r"(_raw_stv8_10[1]), "r"(_raw_stv8_10[2]), "r"(_raw_stv8_10[3]), "r"(_raw_stv8_10[4]), "r"(_raw_stv8_10[5]), "r"(_raw_stv8_10[6]), "r"(_raw_stv8_10[7]) : "memory");
                    }
                    unsigned int upd_bf16_1_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 32], upd[_lp*2+1 + 32]));
                        upd_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_8 = 0; q_8 < 2; q_8++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((32 + q_8 * 8) / 64 * 16384 + k_row * 128 + (32 + q_8 * 8) % 64 * 2 ^ ((32 + q_8 * 8) / 64 * 16384 + k_row * 128 + (32 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_1_1[q_8 * 4]), "r"(upd_bf16_1_1[q_8 * 4 + 1]), "r"(upd_bf16_1_1[q_8 * 4 + 2]), "r"(upd_bf16_1_1[q_8 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 128 + (unsigned int)m128_row_base + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 16; j_9++) {
                        upd[48 + j_9] = __fmaf_rn(_tmem_load_9[j_9], scale, _tmem_load_8[j_9]);
                    }
                    unsigned int _packed_9[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 48], upd[_lp*2+1 + 48]));
                        _packed_9[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 50], upd[_lp*2+1 + 50]));
                        _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 52], upd[_lp*2+1 + 52]));
                        _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 54], upd[_lp*2+1 + 54]));
                        _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 56], upd[_lp*2+1 + 56]));
                        _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 58], upd[_lp*2+1 + 58]));
                        _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 60], upd[_lp*2+1 + 60]));
                        _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 62], upd[_lp*2+1 + 62]));
                        _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_11 = reinterpret_cast<const unsigned*>(_packed_9);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dh_out + (dh_base_1 + 48))), "r"(_raw_stv8_11[0]), "r"(_raw_stv8_11[1]), "r"(_raw_stv8_11[2]), "r"(_raw_stv8_11[3]), "r"(_raw_stv8_11[4]), "r"(_raw_stv8_11[5]), "r"(_raw_stv8_11[6]), "r"(_raw_stv8_11[7]) : "memory");
                    }
                    unsigned int upd_bf16_2_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 48], upd[_lp*2+1 + 48]));
                        upd_bf16_2_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_9 = 0; q_9 < 2; q_9++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((48 + q_9 * 8) / 64 * 16384 + k_row * 128 + (48 + q_9 * 8) % 64 * 2 ^ ((48 + q_9 * 8) / 64 * 16384 + k_row * 128 + (48 + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(upd_bf16_2_1[q_9 * 4]), "r"(upd_bf16_2_1[q_9 * 4 + 1]), "r"(upd_bf16_2_1[q_9 * 4 + 2]), "r"(upd_bf16_2_1[q_9 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dhs_ready_addr);
                    float sc_1[16];
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 16; j_10++) {
                        sc_1[j_10] = upd[j_10] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc_1);
                    float sc_3_1[16];
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 16; j_11++) {
                        sc_3_1[j_11] = upd[16 + j_11] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_3_1);
                    float sc_4_1[16];
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 16; j_12++) {
                        sc_4_1[j_12] = upd[32 + j_12] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 32, sc_4_1);
                    float sc_5_1[16];
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 16; j_13++) {
                        sc_5_1[j_13] = upd[48 + j_13] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 48, sc_5_1);
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
            int half_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int rank_1 = blockIdx.x / (2 * num_heads);
            int seq_1 = rank_1;
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int vbase_1 = half_1 * 64;
            unsigned int ad = 0;
            int tok0 = chunk_bos[((nc_1 > 0) ? c0_1 : 0)];
            int clen_last = chunk_len[((nc_1 > 0) ? c0_1 + nc_1 - 1 : 0)];
            unsigned int _phase_ad_empty = 1;
            unsigned int _phase_kg_empty = 1;
            unsigned int _phase_qg_empty = 1;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                if (nc_1 > 0) {
                    int trow0 = row0_1 + (nc_1 - 1) * 64;
                    mbarrier_wait(ad_empty_addr + (ad) * 8, _phase_ad_empty);
                    mbarrier_arrive_expect_tx(ad_full_addr + (ad) * 8, 16384);
                    tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok0 + (nc_1 - 1) * 64, ad_full_addr + (ad) * 8);
                    tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok0 + (nc_1 - 1) * 64, ad_full_addr + (ad) * 8);
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
                        tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok0 + (nc_1 - 2) * 64, ad_full_addr + (ad) * 8);
                        tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok0 + (nc_1 - 2) * 64, ad_full_addr + (ad) * 8);
                        ad += 1;
                        if (ad == 2) { ad = 0; _phase_ad_empty ^= 1; }
                    }
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    tma_3d_gmem2smem(w_kop_addr, (&w_tma), 0, 0, (c0_1 + nc_1 - 1) * num_heads + head_1, w_full_addr);
                    _phase_w_empty ^= 1;
                }
            }
            if (nc_1 > 0 && clen_last < 64) {
                mbarrier_wait(ad_full_addr, 0);
                unsigned int zeros4[4];
                zeros4[0] = 0;
                zeros4[1] = 0;
                zeros4[2] = 0;
                zeros4[3] = 0;
                int t0 = clen_last + lane;
                const unsigned int do_base0 = do_panel_addr;
                #pragma unroll
                for (int j_14 = 0; j_14 < 2; j_14++) {
                    if (t0 + 32 * j_14 < 64) {
                        #pragma unroll
                        for (int q_10 = 0; q_10 < 8; q_10++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(do_base0 + (unsigned int)((t0 + 32 * j_14) * 128 + q_10 * 16)), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(do_ready_addr);
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 1; rc_1 < nc_1; rc_1++) {
                    int trow_1 = row0_1 + (nc_1 - 1 - rc_1) * 64;
                    int tok = tok0 + (nc_1 - 1 - rc_1) * 64;
                    mbarrier_wait(kg_empty_addr, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr, 16384);
                    #pragma unroll
                    for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg_2 * 8192), (&kg_tma), seg_2 * 64, head_1, trow_1, kg_full_addr);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qg_empty_addr, _phase_qg_empty);
                    mbarrier_arrive_expect_tx(qg_full_addr, 16384);
                    #pragma unroll
                    for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_3 * 8192), (&qg_tma), seg_3 * 64, head_1, trow_1, qg_full_addr);
                    }
                    _phase_qg_empty ^= 1;
                    if (nc_1 - 1 - rc_1 >= 1) {
                        mbarrier_wait(ad_empty_addr + (ad) * 8, _phase_ad_empty);
                        mbarrier_arrive_expect_tx(ad_full_addr + (ad) * 8, 16384);
                        tma_3d_gmem2smem(do_panel_addr + ad * 16384, (&do_tma), vbase_1, head_1, tok - 64, ad_full_addr + (ad) * 8);
                        tma_3d_gmem2smem(aqk_panel_addr + ad * 16384, (&aqk_tma), 0, head_1, tok - 64, ad_full_addr + (ad) * 8);
                        ad += 1;
                        if (ad == 2) { ad = 0; _phase_ad_empty ^= 1; }
                    }
                    mbarrier_wait(w_empty_addr, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr, 16384);
                    tma_3d_gmem2smem(w_kop_addr, (&w_tma), 0, 0, (c0_1 + (nc_1 - 1 - rc_1)) * num_heads + head_1, w_full_addr);
                    _phase_w_empty ^= 1;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int half_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int rank_2 = blockIdx.x / (2 * num_heads);
            int seq_2 = rank_2;
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int mclen_last = chunk_len[((nc_2 > 0) ? c0_2 + nc_2 - 1 : 0)];
            unsigned int ams = 0;
            const int lane_m = lane;
            if (nc_2 > 0 && mclen_last < 64) {
                mbarrier_wait(ad_full_addr, 0);
                unsigned int zeros[4];
                zeros[0] = 0;
                zeros[1] = 0;
                zeros[2] = 0;
                zeros[3] = 0;
                int npad = (64 - mclen_last) * 8;
                #pragma unroll 1
                for (int idx = lane_m; idx < npad; idx += 32) {
                    int off = (mclen_last + (idx >> 3)) * 128 + (idx & 7) * 16;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(aqk_panel_addr + (unsigned int)off), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
            }
            unsigned int _phase_ad_full = 0;
            unsigned int _phase_kg_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_qg_full = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            unsigned int _phase_w_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(ad_full_addr + (ams) * 8, _phase_ad_full);
                    if (rc_2 == 0 && mclen_last < 64) {
                        mbarrier_wait(do_ready_addr, 0);
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
                    mbarrier_wait(kg_full_addr, _phase_kg_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    mbarrier_wait(w_full_addr, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((w_kop_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_7 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_6 = ((w_kop_addr) >> 4) & 0x3FFF;
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_dh), "r"(1), "r"(((uint32_t)(135333008) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))) : "memory");
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
