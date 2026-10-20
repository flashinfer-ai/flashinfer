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
#define TMEM_NCOLS 96
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 32
#define TMEM_ACC_P_OFFSET 64
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
#define SMEM_DO_PANEL_STAGE_BYTES 4096
#define SMEM_DO_PANEL_STRIDE 4096
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
#define SMEM_DO_MN_STAGE_BYTES 4096
#define SMEM_DO_MN_STRIDE 4096
#define SMEM_DHS_OFF 54272
#define SMEM_DHS_STAGE_BYTES 8192
#define SMEM_DHS_STRIDE 8192
#define SMEM_DHS_MN_OFF 54272
#define SMEM_DHS_MN_STAGE_BYTES 8192
#define SMEM_DHS_MN_STRIDE 8192
#define SMEM_BDV_OFF 62464
#define SMEM_BDV_STAGE_BYTES 4096
#define SMEM_BDV_STRIDE 4096
#define SMEM_BDV_MN_OFF 62464
#define SMEM_BDV_MN_STAGE_BYTES 4096
#define SMEM_BDV_MN_STRIDE 4096
#define SMEM_TOTAL 66560
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
kernel_cake_kda_train_0ea173162573b1521e36(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, __nv_bfloat16* __restrict__ dv1, float* __restrict__ gk, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, int num_heads, int* __restrict__ seq_chunk_start, float scale)
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
    #define bdv_ready_addr (mbar_base + 64)
    #define dh_ready_addr (mbar_base + 72)

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
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int dhs_addr = smem + 54272;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int dhs_mn_addr = smem + 54272;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 62464);
    const int bdv_addr = smem + 62464;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 62464);
    const int bdv_mn_addr = smem + 62464;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

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
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 32;
    const int tmem_acc_p = taddr + 64;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 4;
            int head = blockIdx.x / 4 % num_heads;
            int seq = blockIdx.x / (4 * num_heads);
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int vbase = half * 32;
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
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float gk_pre = 0.0f;
            float d1_pre[16];
            unsigned int d1_pk[8];
            d1_pk[0] = 0;
            d1_pk[1] = 0;
            d1_pk[2] = 0;
            d1_pk[3] = 0;
            d1_pk[4] = 0;
            d1_pk[5] = 0;
            d1_pk[6] = 0;
            d1_pk[7] = 0;
            float upd[32];
            float decay = 1.0f;
            if (nc > 0) {
                long long gn_next = ((long long)(row0 + nc * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                gk_pre = gk[gn_next];
                long long d_next = ((long long)(row0 + (nc - 1) * 64 + t_row) * (long long)num_heads + (long long)head) * 128 + (long long)vbase;
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    {
                        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(dv1 + d_next + (long long)(lane_half * 16 + q * 8));
                        d1_pk[q * 4 + 0] = _uv4_0.x;
                        d1_pk[q * 4 + 1] = _uv4_0.y;
                        d1_pk[q * 4 + 2] = _uv4_0.z;
                        d1_pk[q * 4 + 3] = _uv4_0.w;
                    }
                }
                long long dh_base = ((long long)(c0 + nc - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
                tmem_ld_x16(&upd[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int upd_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                    upd_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_1 * 8 / 32 * 8192 + k_row * 64 + q_1 * 8 % 32 * 2 ^ (q_1 * 8 / 32 * 8192 + k_row * 64 + q_1 * 8 % 32 * 2 >> 7 & 3) << 4))), "r"(upd_bf16[q_1 * 4]), "r"(upd_bf16[q_1 * 4 + 1]), "r"(upd_bf16[q_1 * 4 + 2]), "r"(upd_bf16[q_1 * 4 + 3]) : "memory");
                }
                tmem_ld_x16(&upd[16], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int upd_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                    upd_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_2 * 8) / 32 * 8192 + k_row * 64 + (16 + q_2 * 8) % 32 * 2 ^ ((16 + q_2 * 8) / 32 * 8192 + k_row * 64 + (16 + q_2 * 8) % 32 * 2 >> 7 & 3) << 4))), "r"(upd_bf16_0[q_2 * 4]), "r"(upd_bf16_0[q_2 * 4 + 1]), "r"(upd_bf16_0[q_2 * 4 + 2]), "r"(upd_bf16_0[q_2 * 4 + 3]) : "memory");
                }
                float _exp2_0 = approx_exp2(gk_pre);
                decay = _exp2_0;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dhs_ready_addr);
                float sc[16];
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    sc[j] = upd[j] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc);
                float sc_1[16];
                #pragma unroll
                for (int j_1 = 0; j_1 < 16; j_1++) {
                    sc_1[j_1] = upd[16 + j_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_1);
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(upd[0 + 0], upd[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(upd[0 + 2], upd[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(upd[0 + 4], upd[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(upd[0 + 6], upd[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(upd[0 + 8], upd[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(upd[0 + 10], upd[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(upd[0 + 12], upd[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(upd[0 + 14], upd[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dh_out + dh_base))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(upd[16 + 0], upd[16 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(upd[16 + 2], upd[16 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(upd[16 + 4], upd[16 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(upd[16 + 6], upd[16 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(upd[16 + 8], upd[16 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(upd[16 + 10], upd[16 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(upd[16 + 12], upd[16 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(upd[16 + 14], upd[16 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dh_out + (dh_base + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
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
                float bdvv[16];
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + 32 + (unsigned int)m64_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _unpack_bf16_0[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d1_pk[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d1_pk[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 16; j_2++) {
                    bdvv[j_2] = _tmem_load_0[j_2] + _unpack_bf16_0[j_2];
                }
                unsigned int bdvv_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdvv[_lp*2 + 0], bdvv[_lp*2+1 + 0]));
                    bdvv_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((lane_half * 16 + q_3 * 8) / 32 * 4096 + t_row * 64 + (lane_half * 16 + q_3 * 8) % 32 * 2 ^ ((lane_half * 16 + q_3 * 8) / 32 * 4096 + t_row * 64 + (lane_half * 16 + q_3 * 8) % 32 * 2 >> 7 & 3) << 4))), "r"(bdvv_bf16[q_3 * 4]), "r"(bdvv_bf16[q_3 * 4 + 1]), "r"(bdvv_bf16[q_3 * 4 + 2]), "r"(bdvv_bf16[q_3 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(bdvv[0 + 0], bdvv[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(bdvv[0 + 2], bdvv[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(bdvv[0 + 4], bdvv[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(bdvv[0 + 6], bdvv[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(bdvv[0 + 8], bdvv[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(bdvv[0 + 10], bdvv[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(bdvv[0 + 12], bdvv[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(bdvv[0 + 14], bdvv[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dv2 + (t_base + (long long)(lane_half * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                if (nc > rc + 1) {
                    long long gn_next_1 = ((long long)(row0 + c * 64 - 1) * (long long)num_heads + (long long)head) * 128 + (long long)k_row;
                    gk_pre = gk[gn_next_1];
                    long long d_next_1 = t_base - (long long)(64 * num_heads * 128);
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 2; q_4++) {
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(dv1 + d_next_1 + (long long)(lane_half * 16 + q_4 * 8));
                            d1_pk[q_4 * 4 + 0] = _uv4_1.x;
                            d1_pk[q_4 * 4 + 1] = _uv4_1.y;
                            d1_pk[q_4 * 4 + 2] = _uv4_1.z;
                            d1_pk[q_4 * 4 + 3] = _uv4_1.w;
                        }
                    }
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (nc > rc + 1) {
                    float _exp2_1 = approx_exp2(gk_pre);
                    decay = _exp2_1;
                    long long dh_base_1 = ((long long)(c0 + c - 1) * (long long)num_heads + (long long)head) * 16384 + (long long)(k_row * 128 + vbase);
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 64 + (unsigned int)m128_row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 16; j_3++) {
                        float _fma_0 = __fmaf_rn(_tmem_load_2[j_3], scale, _tmem_load_1[j_3]);
                        upd[j_3] = _fma_0;
                    }
                    unsigned int upd_bf16_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 0], upd[_lp*2+1 + 0]));
                        upd_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 2; q_5++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_5 * 8 / 32 * 8192 + k_row * 64 + q_5 * 8 % 32 * 2 ^ (q_5 * 8 / 32 * 8192 + k_row * 64 + q_5 * 8 % 32 * 2 >> 7 & 3) << 4))), "r"(upd_bf16_1[q_5 * 4]), "r"(upd_bf16_1[q_5 * 4 + 1]), "r"(upd_bf16_1[q_5 * 4 + 2]), "r"(upd_bf16_1[q_5 * 4 + 3]) : "memory");
                    }
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 64 + (unsigned int)m128_row_base + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 16; j_4++) {
                        float _fma_1 = __fmaf_rn(_tmem_load_4[j_4], scale, _tmem_load_3[j_4]);
                        upd[16 + j_4] = _fma_1;
                    }
                    unsigned int upd_bf16_0_1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(upd[_lp*2 + 16], upd[_lp*2+1 + 16]));
                        upd_bf16_0_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 2; q_6++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_6 * 8) / 32 * 8192 + k_row * 64 + (16 + q_6 * 8) % 32 * 2 ^ ((16 + q_6 * 8) / 32 * 8192 + k_row * 64 + (16 + q_6 * 8) % 32 * 2 >> 7 & 3) << 4))), "r"(upd_bf16_0_1[q_6 * 4]), "r"(upd_bf16_0_1[q_6 * 4 + 1]), "r"(upd_bf16_0_1[q_6 * 4 + 2]), "r"(upd_bf16_0_1[q_6 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dhs_ready_addr);
                    float sc_2[16];
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 16; j_5++) {
                        sc_2[j_5] = upd[j_5] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base, sc_2);
                    float sc_1_1[16];
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 16; j_6++) {
                        sc_1_1[j_6] = upd[16 + j_6] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + 16, sc_1_1);
                    {
                        {
                            __nv_bfloat162 _pk0 = __floats2bfloat162_rn(upd[0 + 0], upd[0 + 1]);
                            unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                            __nv_bfloat162 _pk1 = __floats2bfloat162_rn(upd[0 + 2], upd[0 + 3]);
                            unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                            __nv_bfloat162 _pk2 = __floats2bfloat162_rn(upd[0 + 4], upd[0 + 5]);
                            unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                            __nv_bfloat162 _pk3 = __floats2bfloat162_rn(upd[0 + 6], upd[0 + 7]);
                            unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                            __nv_bfloat162 _pk4 = __floats2bfloat162_rn(upd[0 + 8], upd[0 + 9]);
                            unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                            __nv_bfloat162 _pk5 = __floats2bfloat162_rn(upd[0 + 10], upd[0 + 11]);
                            unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                            __nv_bfloat162 _pk6 = __floats2bfloat162_rn(upd[0 + 12], upd[0 + 13]);
                            unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                            __nv_bfloat162 _pk7 = __floats2bfloat162_rn(upd[0 + 14], upd[0 + 15]);
                            unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(&((__nv_bfloat16*)(dh_out + dh_base_1))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                        }
                    }
                    {
                        {
                            __nv_bfloat162 _pk0 = __floats2bfloat162_rn(upd[16 + 0], upd[16 + 1]);
                            unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                            __nv_bfloat162 _pk1 = __floats2bfloat162_rn(upd[16 + 2], upd[16 + 3]);
                            unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                            __nv_bfloat162 _pk2 = __floats2bfloat162_rn(upd[16 + 4], upd[16 + 5]);
                            unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                            __nv_bfloat162 _pk3 = __floats2bfloat162_rn(upd[16 + 6], upd[16 + 7]);
                            unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                            __nv_bfloat162 _pk4 = __floats2bfloat162_rn(upd[16 + 8], upd[16 + 9]);
                            unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                            __nv_bfloat162 _pk5 = __floats2bfloat162_rn(upd[16 + 10], upd[16 + 11]);
                            unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                            __nv_bfloat162 _pk6 = __floats2bfloat162_rn(upd[16 + 12], upd[16 + 13]);
                            unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                            __nv_bfloat162 _pk7 = __floats2bfloat162_rn(upd[16 + 14], upd[16 + 15]);
                            unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(&((__nv_bfloat16*)(dh_out + (dh_base_1 + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                        }
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int half_1 = blockIdx.x % 4;
            int head_1 = blockIdx.x / 4 % num_heads;
            int seq_1 = blockIdx.x / (4 * num_heads);
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            unsigned int ks = 0;
            unsigned int qs = 0;
            unsigned int ws = 0;
            unsigned int _phase_kg_empty = 1;
            unsigned int _phase_qd_empty = 1;
            unsigned int _phase_w_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 0; rc_1 < nc_1; rc_1++) {
                    int c_1 = nc_1 - 1 - rc_1;
                    int trow_1 = row0_1 + c_1 * 64;
                    mbarrier_wait(kg_empty_addr + (ks) * 8, _phase_kg_empty);
                    mbarrier_arrive_expect_tx(kg_full_addr + (ks) * 8, 16384);
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (unsigned int)(seg * 8192), (&kg_tma), seg * 64, head_1, trow_1, kg_full_addr + (ks) * 8);
                    }
                    _phase_kg_empty ^= 1;
                    mbarrier_wait(qd_empty_addr + (qs) * 8, _phase_qd_empty);
                    mbarrier_arrive_expect_tx(qd_full_addr + (qs) * 8, 20480);
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        tma_3d_gmem2smem(qg_panel_addr + (unsigned int)(seg_1 * 8192), (&qg_tma), seg_1 * 64, head_1, trow_1, qd_full_addr + (qs) * 8);
                    }
                    tma_3d_gmem2smem(do_panel_addr, (&do_tma), half_1 * 32, head_1, trow_1, qd_full_addr + (qs) * 8);
                    _phase_qd_empty ^= 1;
                    mbarrier_wait(w_empty_addr + (ws) * 8, _phase_w_empty);
                    mbarrier_arrive_expect_tx(w_full_addr + (ws) * 8, 16384);
                    #pragma unroll
                    for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                        tma_3d_gmem2smem(w_panel_addr + (unsigned int)(seg_2 * 8192), (&w_tma), seg_2 * 64, head_1, trow_1, w_full_addr + (ws) * 8);
                    }
                    _phase_w_empty ^= 1;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int half_2 = blockIdx.x % 4;
            int head_2 = blockIdx.x / 4 % num_heads;
            int seq_2 = blockIdx.x / (4 * num_heads);
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int ks_1 = 0;
            unsigned int qs_1 = 0;
            unsigned int ws_1 = 0;
            unsigned int _phase_kg_full = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_qd_full = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            unsigned int _phase_w_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(kg_full_addr + (ks_1) * 8, _phase_kg_full);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = ((kg_op_addr) >> 4) & 0x7FFF;
                    int _mma_b_lo_0 = (((dhs_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _mma_a_lo_0 = ((kg_op_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 67699856;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_dv), "r"(0));
                    tcgen05_commit(dv_ready_addr);
                    tcgen05_commit(kg_empty_addr + (ks_1) * 8);
                    _phase_kg_full ^= 1;
                    mbarrier_wait(qd_full_addr + (qs_1) * 8, _phase_qd_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_1 = (((qg_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _mma_b_lo_1 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x1000000;
#else
                    int _mma_a_lo_1 = (((qg_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _mma_b_lo_1 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x1000000;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_p), "r"(0));
                    tcgen05_commit(qd_empty_addr + (qs_1) * 8);
                    _phase_qd_full ^= 1;
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    mbarrier_wait(w_full_addr + (ws_1) * 8, _phase_w_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_2 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _mma_b_lo_2 = (((bdv_mn_addr) >> 4) & 0x7FFF) | 0x1000000;
#else
                    int _mma_a_lo_2 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _mma_b_lo_2 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x1000000;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, %4;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc_dh), "r"(1), "r"(((uint32_t)(134841488) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))));
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(w_empty_addr + (ws_1) * 8);
                    _phase_w_full ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

