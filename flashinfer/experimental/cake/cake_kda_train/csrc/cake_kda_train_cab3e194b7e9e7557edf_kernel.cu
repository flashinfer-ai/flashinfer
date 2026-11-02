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
#define TMEM_ACC_VP_OFFSET 64
#define TMEM_ACC_WT_OFFSET 128
#define TMEM_ACC_U_OFFSET 192
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_KG_PANEL_OFF 1024
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_KG_MN_OFF 1024
#define SMEM_KG_MN_STAGE_BYTES 16384
#define SMEM_KG_MN_STRIDE 16384
#define SMEM_W_PHYS_OFF 33792
#define SMEM_W_PHYS_STAGE_BYTES 16384
#define SMEM_W_PHYS_STRIDE 16384
#define SMEM_W_MN_OFF 33792
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_HS_OFF 50176
#define SMEM_HS_STAGE_BYTES 16384
#define SMEM_HS_STRIDE 16384
#define SMEM_HS_MN_OFF 50176
#define SMEM_HS_MN_STAGE_BYTES 16384
#define SMEM_HS_MN_STRIDE 16384
#define SMEM_BV_OFF 66560
#define SMEM_BV_STAGE_BYTES 8192
#define SMEM_BV_STRIDE 8192
#define SMEM_BV_MN_OFF 66560
#define SMEM_BV_MN_STAGE_BYTES 8192
#define SMEM_BV_MN_STRIDE 8192
#define SMEM_AKK_OP_OFF 74752
#define SMEM_AKK_OP_STAGE_BYTES 8192
#define SMEM_AKK_OP_STRIDE 8192
#define SMEM_KB_PANEL_OFF 82944
#define SMEM_KB_PANEL_STAGE_BYTES 8192
#define SMEM_KB_PANEL_STRIDE 8192
#define SMEM_KB_MN_OFF 82944
#define SMEM_KB_MN_STAGE_BYTES 16384
#define SMEM_KB_MN_STRIDE 16384
#define SMEM_VB_RAW_OFF 99328
#define SMEM_VB_RAW_STAGE_BYTES 8192
#define SMEM_VB_RAW_STRIDE 8192
#define SMEM_VB_MN_OFF 99328
#define SMEM_VB_MN_STAGE_BYTES 8192
#define SMEM_VB_MN_STRIDE 8192
#define SMEM_TOTAL 107520
#define THREADS 352
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

__global__ __launch_bounds__(352, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_cab3e194b7e9e7557edf(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap kb_tma, float* __restrict__ gk, const __grid_constant__ CUtensorMap h_out, const __grid_constant__ CUtensorMap v_new, int num_heads, int* __restrict__ seq_chunk_start, int* __restrict__ seq_order, const __grid_constant__ CUtensorMap v_tma, __nv_bfloat16* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, const __grid_constant__ CUtensorMap w_out)
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
    #define wy_full_addr (mbar_base + 88)
    #define akk_ready_addr (mbar_base + 96)
    #define vb_ready_addr (mbar_base + 104)
    #define wy_empty_addr (mbar_base + 112)
    #define w_full_addr (mbar_base + 120)
    #define w_ready_addr (mbar_base + 128)
    #define w_tape_read_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_panel_addr = smem + 1024;
    __nv_bfloat16* kg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kg_mn_addr = smem + 1024;
    __nv_bfloat16* w_phys = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_phys_addr = smem + 33792;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int w_mn_addr = smem + 33792;
    __nv_bfloat16* hs = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs_addr = smem + 50176;
    __nv_bfloat16* hs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int hs_mn_addr = smem + 50176;
    __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int bv_addr = smem + 66560;
    __nv_bfloat16* bv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int bv_mn_addr = smem + 66560;
    __nv_bfloat16* akk_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int akk_op_addr = smem + 74752;
    __nv_bfloat16* kb_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int kb_panel_addr = smem + 82944;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int kb_mn_addr = smem + 82944;
    __nv_bfloat16* vb_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int vb_raw_addr = smem + 99328;
    __nv_bfloat16* vb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int vb_mn_addr = smem + 99328;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

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
            // wy_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // akk_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 96, 32);
            // vb_ready: 1 barriers, init_count=32
            mbarrier_init(smem + 104, 32);
            // wy_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // w_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // w_tape_read: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_h = taddr;
    const int tmem_acc_vp = taddr + 64;
    const int tmem_acc_wt = taddr + 128;
    const int tmem_acc_u = taddr + 192;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 2;
            int head = blockIdx.x / 2 % num_heads;
            int rank = blockIdx.x / (2 * num_heads);
            int seq = seq_order[rank];
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int h_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int vp_row_base = warp_in_wg * 32 << 16;
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
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 16, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 32, zeros16);
            tmem_st_x16_f32(taddr + (unsigned int)h_row_base + 48, zeros16);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            unsigned int _phase_bv_tape_read_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_h_tape_read_0 = 0;
            unsigned int _phase_h_ready_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                float tmem_load[16];
                tmem_ld_x16(&tmem_load[0], taddr + (unsigned int)h_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load[_lp*2 + 0], tmem_load[_lp*2+1 + 0]));
                    tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)(q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 ^ (q * 8 / 64 * 16384 + k_row * 128 + q * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16[q * 4]), "r"(tmem_load_0_bf16[q * 4 + 1]), "r"(tmem_load_0_bf16[q * 4 + 2]), "r"(tmem_load_0_bf16[q * 4 + 3]) : "memory");
                }
                float tmem_load_r_6[16];
                tmem_ld_x16(&tmem_load_r_6[0], taddr + (unsigned int)h_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_6[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_6[_lp*2 + 0], tmem_load_r_6[_lp*2+1 + 0]));
                    tmem_load_bf16_6[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_7 = 0; q_7 < 2; q_7++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((16 + q_7 * 8) / 64 * 16384 + k_row * 128 + (16 + q_7 * 8) % 64 * 2 ^ ((16 + q_7 * 8) / 64 * 16384 + k_row * 128 + (16 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_6[q_7 * 4]), "r"(tmem_load_bf16_6[q_7 * 4 + 1]), "r"(tmem_load_bf16_6[q_7 * 4 + 2]), "r"(tmem_load_bf16_6[q_7 * 4 + 3]) : "memory");
                }
                float tmem_load_r_7[16];
                tmem_ld_x16(&tmem_load_r_7[0], taddr + (unsigned int)h_row_base + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_7[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_7[_lp*2 + 0], tmem_load_r_7[_lp*2+1 + 0]));
                    tmem_load_bf16_7[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_8 = 0; q_8 < 2; q_8++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((32 + q_8 * 8) / 64 * 16384 + k_row * 128 + (32 + q_8 * 8) % 64 * 2 ^ ((32 + q_8 * 8) / 64 * 16384 + k_row * 128 + (32 + q_8 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_7[q_8 * 4]), "r"(tmem_load_bf16_7[q_8 * 4 + 1]), "r"(tmem_load_bf16_7[q_8 * 4 + 2]), "r"(tmem_load_bf16_7[q_8 * 4 + 3]) : "memory");
                }
                float tmem_load_r_8[16];
                tmem_ld_x16(&tmem_load_r_8[0], taddr + (unsigned int)h_row_base + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_8[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_8[_lp*2 + 0], tmem_load_r_8[_lp*2+1 + 0]));
                    tmem_load_bf16_8[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_9 = 0; q_9 < 2; q_9++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((hs_addr + (unsigned int)((48 + q_9 * 8) / 64 * 16384 + k_row * 128 + (48 + q_9 * 8) % 64 * 2 ^ ((48 + q_9 * 8) / 64 * 16384 + k_row * 128 + (48 + q_9 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_8[q_9 * 4]), "r"(tmem_load_bf16_8[q_9 * 4 + 1]), "r"(tmem_load_bf16_8[q_9 * 4 + 2]), "r"(tmem_load_bf16_8[q_9 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(bv_tape_read_addr, _phase_bv_tape_read_0);
                _phase_bv_tape_read_0 ^= 1;
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_5[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_0[15]))
                    : "r"(taddr + 192 + (unsigned int)vp_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_0 = lane_half * 16;
                unsigned int _tmem_load_17_bf16_1_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1_0[_lp*2 + 0], _tmem_load_1_0[_lp*2+1 + 0]));
                    _tmem_load_17_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_16[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_tmem_load_17_bf16_1_0[_pair]) << 16);
                    (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_tmem_load_17_bf16_1_0[_pair]) & 0xffff0000u);
                }
                float bvv_2_0[16];
                #pragma unroll
                for (int j_1_4 = 0; j_1_4 < 16; j_1_4++) {
                    bvv_2_0[j_1_4] = _unpack_bf16_16[j_1_4] - _tmem_load_5[j_1_4];
                }
                unsigned int bvv_bf16_1_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_2_0[_lp*2 + 0], bvv_2_0[_lp*2+1 + 0]));
                    bvv_bf16_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1_0 = 0; q_1_0 < 2; q_1_0++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_1_0 + q_1_0 * 8) / 64 * 8192 + t_row * 128 + (col_1_0 + q_1_0 * 8) % 64 * 2 ^ ((col_1_0 + q_1_0 * 8) / 64 * 8192 + t_row * 128 + (col_1_0 + q_1_0 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_1_0[q_1_0 * 4]), "r"(bvv_bf16_1_0[q_1_0 * 4 + 1]), "r"(bvv_bf16_1_0[q_1_0 * 4 + 2]), "r"(bvv_bf16_1_0[q_1_0 * 4 + 3]) : "memory");
                }
                float _tmem_load_6[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                    : "r"(taddr + 64 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1_1[15]))
                    : "r"(taddr + 192 + (unsigned int)vp_row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_1_1 = 32 + lane_half * 16;
                unsigned int _tmem_load_17_bf16_1_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1_1[_lp*2 + 0], _tmem_load_1_1[_lp*2+1 + 0]));
                    _tmem_load_17_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                float _unpack_bf16_17[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_tmem_load_17_bf16_1_1[_pair]) << 16);
                    (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_tmem_load_17_bf16_1_1[_pair]) & 0xffff0000u);
                }
                float bvv_2_1[16];
                #pragma unroll
                for (int j_1_5 = 0; j_1_5 < 16; j_1_5++) {
                    bvv_2_1[j_1_5] = _unpack_bf16_17[j_1_5] - _tmem_load_6[j_1_5];
                }
                unsigned int bvv_bf16_1_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bvv_2_1[_lp*2 + 0], bvv_2_1[_lp*2+1 + 0]));
                    bvv_bf16_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1_1 = 0; q_1_1 < 2; q_1_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bv_addr + (unsigned int)((col_1_1 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (col_1_1 + q_1_1 * 8) % 64 * 2 ^ ((col_1_1 + q_1_1 * 8) / 64 * 8192 + t_row * 128 + (col_1_1 + q_1_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(bvv_bf16_1_1[q_1_1 * 4]), "r"(bvv_bf16_1_1[q_1_1 * 4 + 1]), "r"(bvv_bf16_1_1[q_1_1 * 4 + 2]), "r"(bvv_bf16_1_1[q_1_1 * 4 + 3]) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bv_ready_addr);
                mbarrier_wait(h_tape_read_addr, _phase_h_tape_read_0);
                _phase_h_tape_read_0 ^= 1;
                mbarrier_wait(h_ready_addr, _phase_h_ready_0);
                _phase_h_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
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
            int half_1 = blockIdx.x % 2;
            int head_1 = blockIdx.x / 2 % num_heads;
            int rank_1 = blockIdx.x / (2 * num_heads);
            int seq_1 = seq_order[rank_1];
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int warp_in_wg_1 = warp % 4;
            int k_row_1 = warp_in_wg_1 * 32 + lane;
            int h_row_base_1 = warp_in_wg_1 * 32 << 16;
            float gk_pre = 0.0f;
            if (nc_1 > 0) {
                long long gn_next = ((long long)(row0_1 + 64 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                gk_pre = gk[gn_next];
            }
            unsigned int _phase_w_tape_read_0 = 1;
            unsigned int _phase_w_full_0 = 0;
            if (nc_1 > 0) {
                mbarrier_wait(w_tape_read_addr, _phase_w_tape_read_0);
                _phase_w_tape_read_0 ^= 1;
                mbarrier_wait(w_full_addr, _phase_w_full_0);
                _phase_w_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float tmem_load_1[16];
                tmem_ld_x16(&tmem_load_1[0], taddr + 128 + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_0_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_1[_lp*2 + 0], tmem_load_1[_lp*2+1 + 0]));
                    tmem_load_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)(q_1 * 8 / 64 * 16384 + k_row_1 * 128 + q_1 * 8 % 64 * 2 ^ (q_1 * 8 / 64 * 16384 + k_row_1 * 128 + q_1 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_1[q_1 * 4]), "r"(tmem_load_0_bf16_1[q_1 * 4 + 1]), "r"(tmem_load_0_bf16_1[q_1 * 4 + 2]), "r"(tmem_load_0_bf16_1[q_1 * 4 + 3]) : "memory");
                }
                float tmem_load_r_0[16];
                tmem_ld_x16(&tmem_load_r_0[0], taddr + 128 + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_0[_lp*2 + 0], tmem_load_r_0[_lp*2+1 + 0]));
                    tmem_load_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_0 = 0; q_0 < 2; q_0++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((16 + q_0 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_0 * 8) % 64 * 2 ^ ((16 + q_0 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_0 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_0[q_0 * 4]), "r"(tmem_load_bf16_0[q_0 * 4 + 1]), "r"(tmem_load_bf16_0[q_0 * 4 + 2]), "r"(tmem_load_bf16_0[q_0 * 4 + 3]) : "memory");
                }
                float tmem_load_r_1[16];
                tmem_ld_x16(&tmem_load_r_1[0], taddr + 128 + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_1[_lp*2 + 0], tmem_load_r_1[_lp*2+1 + 0]));
                    tmem_load_bf16_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((32 + q_2 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_2 * 8) % 64 * 2 ^ ((32 + q_2 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_1[q_2 * 4]), "r"(tmem_load_bf16_1[q_2 * 4 + 1]), "r"(tmem_load_bf16_1[q_2 * 4 + 2]), "r"(tmem_load_bf16_1[q_2 * 4 + 3]) : "memory");
                }
                float tmem_load_r_2[16];
                tmem_ld_x16(&tmem_load_r_2[0], taddr + 128 + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int tmem_load_bf16_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_2[_lp*2 + 0], tmem_load_r_2[_lp*2+1 + 0]));
                    tmem_load_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((48 + q_3 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_3 * 8) % 64 * 2 ^ ((48 + q_3 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_2[q_3 * 4]), "r"(tmem_load_bf16_2[q_3 * 4 + 1]), "r"(tmem_load_bf16_2[q_3 * 4 + 2]), "r"(tmem_load_bf16_2[q_3 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(w_ready_addr);
            }
            unsigned int _phase_hs_ready_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float decay = approx_exp2(gk_pre);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)h_row_base_1);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3_0[16];
                #pragma unroll
                for (int j_1_0 = 0; j_1_0 < 16; j_1_0++) {
                    hs2_3_0[j_1_0] = _tmem_load_0[j_1_0] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1, hs2_3_0);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)h_row_base_1 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3_1[16];
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 16; j_1_1++) {
                    hs2_3_1[j_1_1] = _tmem_load_2[j_1_1] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 16, hs2_3_1);
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + (unsigned int)h_row_base_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3_2[16];
                #pragma unroll
                for (int j_1_2 = 0; j_1_2 < 16; j_1_2++) {
                    hs2_3_2[j_1_2] = _tmem_load_3[j_1_2] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 32, hs2_3_2);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)h_row_base_1 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float hs2_3_3[16];
                #pragma unroll
                for (int j_1_3 = 0; j_1_3 < 16; j_1_3++) {
                    hs2_3_3[j_1_3] = _tmem_load_4[j_1_3] * decay;
                }
                tmem_st_x16_f32(taddr + (unsigned int)h_row_base_1 + 48, hs2_3_3);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dec_done_addr);
                if (nc_1 > c_1 + 1) {
                    long long gn_next_1 = ((long long)(row0_1 + c_1 * 64 + 128 - 1) * (long long)num_heads + (long long)head_1) * 128 + (long long)k_row_1;
                    gk_pre = gk[gn_next_1];
                }
                if (nc_1 > c_1 + 1) {
                    mbarrier_wait(w_tape_read_addr, _phase_w_tape_read_0);
                    _phase_w_tape_read_0 ^= 1;
                    mbarrier_wait(w_full_addr, _phase_w_full_0);
                    _phase_w_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float tmem_load_2[16];
                    tmem_ld_x16(&tmem_load_2[0], taddr + 128 + (unsigned int)h_row_base_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int tmem_load_0_bf16_2[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_2[_lp*2 + 0], tmem_load_2[_lp*2+1 + 0]));
                        tmem_load_0_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_4 = 0; q_4 < 2; q_4++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)(q_4 * 8 / 64 * 16384 + k_row_1 * 128 + q_4 * 8 % 64 * 2 ^ (q_4 * 8 / 64 * 16384 + k_row_1 * 128 + q_4 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_0_bf16_2[q_4 * 4]), "r"(tmem_load_0_bf16_2[q_4 * 4 + 1]), "r"(tmem_load_0_bf16_2[q_4 * 4 + 2]), "r"(tmem_load_0_bf16_2[q_4 * 4 + 3]) : "memory");
                    }
                    float tmem_load_r_3[16];
                    tmem_ld_x16(&tmem_load_r_3[0], taddr + 128 + (unsigned int)h_row_base_1 + 16);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int tmem_load_bf16_3[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_3[_lp*2 + 0], tmem_load_r_3[_lp*2+1 + 0]));
                        tmem_load_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_4_1 = 0; q_4_1 < 2; q_4_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((16 + q_4_1 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_4_1 * 8) % 64 * 2 ^ ((16 + q_4_1 * 8) / 64 * 16384 + k_row_1 * 128 + (16 + q_4_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_3[q_4_1 * 4]), "r"(tmem_load_bf16_3[q_4_1 * 4 + 1]), "r"(tmem_load_bf16_3[q_4_1 * 4 + 2]), "r"(tmem_load_bf16_3[q_4_1 * 4 + 3]) : "memory");
                    }
                    float tmem_load_r_4[16];
                    tmem_ld_x16(&tmem_load_r_4[0], taddr + 128 + (unsigned int)h_row_base_1 + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int tmem_load_bf16_4[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_4[_lp*2 + 0], tmem_load_r_4[_lp*2+1 + 0]));
                        tmem_load_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 2; q_5++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((32 + q_5 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_5 * 8) % 64 * 2 ^ ((32 + q_5 * 8) / 64 * 16384 + k_row_1 * 128 + (32 + q_5 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_4[q_5 * 4]), "r"(tmem_load_bf16_4[q_5 * 4 + 1]), "r"(tmem_load_bf16_4[q_5 * 4 + 2]), "r"(tmem_load_bf16_4[q_5 * 4 + 3]) : "memory");
                    }
                    float tmem_load_r_5[16];
                    tmem_ld_x16(&tmem_load_r_5[0], taddr + 128 + (unsigned int)h_row_base_1 + 48);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int tmem_load_bf16_5[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tmem_load_r_5[_lp*2 + 0], tmem_load_r_5[_lp*2+1 + 0]));
                        tmem_load_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_6 = 0; q_6 < 2; q_6++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((w_phys_addr + (unsigned int)((48 + q_6 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_6 * 8) % 64 * 2 ^ ((48 + q_6 * 8) / 64 * 16384 + k_row_1 * 128 + (48 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(tmem_load_bf16_5[q_6 * 4]), "r"(tmem_load_bf16_5[q_6 * 4 + 1]), "r"(tmem_load_bf16_5[q_6 * 4 + 2]), "r"(tmem_load_bf16_5[q_6 * 4 + 3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(w_ready_addr);
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int half_2 = blockIdx.x % 2;
            int head_2 = blockIdx.x / 2 % num_heads;
            int rank_2 = blockIdx.x / (2 * num_heads);
            int seq_2 = seq_order[rank_2];
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            int vbase = half_2 * 64;
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_w_ready_0 = 0;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_bv_ready_0 = 0;
            if (elect_sync()) {
                if (nc_2 > 0) {
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 16384);
                    int trow = row0_2;
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&kg_tma), seg * 64, head_2, trow, tiles_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    mbarrier_wait(w_ready_addr, _phase_w_ready_0);
                    _phase_w_ready_0 ^= 1;
                    if (half_2 == 0) {
                        int chunk_j = c0_2 * num_heads + head_2;
                        tma_store_3d((&w_out), 0, 0, chunk_j, w_phys_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    mbarrier_arrive(w_tape_read_addr);
                }
                #pragma unroll 1
                for (int c_2 = 0; c_2 < nc_2; c_2++) {
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 16384);
                        int trow_1 = row0_2 + (c_2 + 1) * 64;
                        #pragma unroll
                        for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                            tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg_1) * 8192, (&kg_tma), seg_1 * 64, head_2, trow_1, tiles_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                    }
                    int chunk = (c0_2 + c_2) * num_heads + head_2;
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
                    tma_store_3d((&h_out), vbase, 0, chunk, hs_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                    int trow_2 = row0_2 + c_2 * 64;
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0);
                    _phase_bv_ready_0 ^= 1;
                    tma_store_3d((&v_new), vbase, head_2, trow_2, bv_addr);
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                    mbarrier_arrive(h_tape_read_addr);
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(bv_tape_read_addr);
                    if (nc_2 > c_2 + 1) {
                        mbarrier_wait(w_ready_addr, _phase_w_ready_0);
                        _phase_w_ready_0 ^= 1;
                        if (half_2 == 0) {
                            int chunk_j_1 = (c0_2 + (c_2 + 1)) * num_heads + head_2;
                            tma_store_3d((&w_out), 0, 0, chunk_j_1, w_phys_addr);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        mbarrier_arrive(w_tape_read_addr);
                    }
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int half_3 = blockIdx.x % 2;
            int head_3 = blockIdx.x / 2 % num_heads;
            int rank_3 = blockIdx.x / (2 * num_heads);
            int seq_3 = seq_order[rank_3];
            int c0_3 = seq_chunk_start[seq_3];
            int nc_3 = seq_chunk_start[seq_3 + 1] - c0_3;
            int row0_3 = c0_3 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_wy_full_0 = 0;
            unsigned int _phase_akk_ready_0 = 0;
            unsigned int _phase_vb_ready_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_w_ready_0_1 = 0;
            unsigned int _phase_hs_ready_0_2 = 0;
            unsigned int _phase_bv_ready_0_1 = 0;
            unsigned int _phase_dec_done_0 = 0;
            if (elect_sync()) {
                if (nc_3 > 0) {
                    mbarrier_wait(wy_full_addr, _phase_wy_full_0);
                    _phase_wy_full_0 ^= 1;
                    mbarrier_wait(akk_ready_addr, _phase_akk_ready_0);
                    _phase_akk_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    int _smem_desc_lo_1 = (((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc_wt), "r"(0) : "memory");
                    tcgen05_commit(w_full_addr);
                    mbarrier_wait(vb_ready_addr, _phase_vb_ready_0);
                    _phase_vb_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = ((akk_op_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_3 = (((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_2 = ((akk_op_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_3 = (((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc_u), "r"(0) : "memory");
                    tcgen05_commit(wy_empty_addr);
                }
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(w_ready_addr, _phase_w_ready_0_1);
                    _phase_w_ready_0_1 ^= 1;
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_2);
                    _phase_hs_ready_0_2 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
                    int _smem_desc_lo_5 = (((hs_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                    int _smem_desc_lo_4 = (((w_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
                    int _smem_desc_lo_5 = (((hs_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc_vp), "r"(0) : "memory");
                    tcgen05_commit(vp_ready_addr);
                    if (nc_3 > c_3 + 1) {
                        mbarrier_wait(wy_full_addr, _phase_wy_full_0);
                        _phase_wy_full_0 ^= 1;
                        mbarrier_wait(akk_ready_addr, _phase_akk_ready_0);
                        _phase_akk_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((akk_op_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_6 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_7 = (((akk_op_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc_wt), "r"(0) : "memory");
                        tcgen05_commit(w_full_addr);
                    }
                    mbarrier_wait(bv_ready_addr, _phase_bv_ready_0_1);
                    _phase_bv_ready_0_1 ^= 1;
                    mbarrier_wait(dec_done_addr, _phase_dec_done_0);
                    _phase_dec_done_0 ^= 1;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(h_ready_addr);
                    if (nc_3 > c_3 + 1) {
                        mbarrier_wait(vb_ready_addr, _phase_vb_ready_0);
                        _phase_vb_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = ((akk_op_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_9 = (((vb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_8 = ((akk_op_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_9 = (((vb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc_u), "r"(0) : "memory");
                        tcgen05_commit(wy_empty_addr);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_10 = ((((kg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _smem_desc_lo_11 = (((bv_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc_h), "r"(1) : "memory");
                    tcgen05_commit(h_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: cvt ----
    if (warp == 10) {
        { // cvt_main
            int half_4 = blockIdx.x % 2;
            int head_4 = blockIdx.x / 2 % num_heads;
            int rank_4 = blockIdx.x / (2 * num_heads);
            int seq_4 = seq_order[rank_4];
            int c0_4 = seq_chunk_start[seq_4];
            int nc_4 = seq_chunk_start[seq_4 + 1] - c0_4;
            int row0_4 = c0_4 * 64;
            const int cvt_tid = (warp - 10) * 32 + lane;
            int vbase_1 = half_4 * 64;
            int bos_a = 0;
            int bos_b = 0;
            if (nc_4 > 0) {
                bos_a = chunk_bos[c0_4];
            }
            if (nc_4 > 1) {
                bos_b = chunk_bos[c0_4 + 1];
            }
            unsigned int _phase_wy_empty_0 = 1;
            if (warp == 10) {
                if (elect_sync()) {
                    if (nc_4 > 0) {
                        mbarrier_wait(wy_empty_addr, _phase_wy_empty_0);
                        _phase_wy_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(wy_full_addr, 32768);
                        int trow_3 = row0_4;
                        tma_3d_gmem2smem(akk_op_addr, (&akk_tma), 0, head_4, bos_a, wy_full_addr);
                        #pragma unroll
                        for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                            tma_3d_gmem2smem(kb_panel_addr + (unsigned int)seg_2 * 8192, (&kb_tma), seg_2 * 64, head_4, trow_3, wy_full_addr);
                        }
                        tma_3d_gmem2smem(vb_raw_addr, (&v_tma), vbase_1, head_4, bos_a, wy_full_addr);
                        if (nc_4 > 1) {
                            int trow2 = row0_4 + 64;
                            #pragma unroll
                            for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_3 * 64)), "r"((int)(head_4)), "r"((int)(trow2)) : "memory");
                            }
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(vbase_1)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                        }
                        bos_a = bos_b;
                        if (nc_4 > 2) {
                            bos_b = chunk_bos[c0_4 + 2];
                        }
                    }
                }
            }
            int bos_n = 0;
            int clen_n = 0;
            if (nc_4 > 0) {
                bos_n = chunk_bos[c0_4];
                clen_n = chunk_len[c0_4];
            }
            float bt_n[2];
            bt_n[0] = 0.0f;
            bt_n[1] = 0.0f;
            int t_r_5_0 = cvt_tid;
            int clast_3_0 = clen_n - 1;
            if (clast_3_0 < 0) {
                clast_3_0 = 0;
            }
            int t_c_1_0 = t_r_5_0;
            if (t_c_1_0 > clast_3_0) {
                t_c_1_0 = clast_3_0;
            }
            if (nc_4 > 0) {
                long long bidx_1_0 = (long long)(bos_n + t_c_1_0) * (long long)num_heads + (long long)head_4;
                bt_n[0] = (float)beta[bidx_1_0];
            }
            int t_r_5_1 = cvt_tid + 32;
            int clast_3_1 = clen_n - 1;
            if (clast_3_1 < 0) {
                clast_3_1 = 0;
            }
            int t_c_1_1 = t_r_5_1;
            if (t_c_1_1 > clast_3_1) {
                t_c_1_1 = clast_3_1;
            }
            if (nc_4 > 0) {
                long long bidx_1_1 = (long long)(bos_n + t_c_1_1) * (long long)num_heads + (long long)head_4;
                bt_n[1] = (float)beta[bidx_1_1];
            }
            unsigned int zeros4[4];
            zeros4[0] = 0;
            zeros4[1] = 0;
            zeros4[2] = 0;
            zeros4[3] = 0;
            unsigned int _phase_wy_full_0_1 = 0;
            #pragma unroll 1
            for (int j = 0; j < nc_4; j++) {
                int clen_j = clen_n;
                float bt_j[2];
                bt_j[0] = bt_n[0];
                bt_j[1] = bt_n[1];
                if (nc_4 > j + 1) {
                    bos_n = chunk_bos[c0_4 + j + 1];
                    clen_n = chunk_len[c0_4 + j + 1];
                    int t_r_5_2 = cvt_tid;
                    int clast_3_2 = clen_n - 1;
                    if (clast_3_2 < 0) {
                        clast_3_2 = 0;
                    }
                    int t_c_1_2 = t_r_5_2;
                    if (t_c_1_2 > clast_3_2) {
                        t_c_1_2 = clast_3_2;
                    }
                    long long bidx_1_2 = (long long)(bos_n + t_c_1_2) * (long long)num_heads + (long long)head_4;
                    bt_n[0] = (float)beta[bidx_1_2];
                    int t_r_5_3 = cvt_tid + 32;
                    int clast_3_3 = clen_n - 1;
                    if (clast_3_3 < 0) {
                        clast_3_3 = 0;
                    }
                    int t_c_1_3 = t_r_5_3;
                    if (t_c_1_3 > clast_3_3) {
                        t_c_1_3 = clast_3_3;
                    }
                    long long bidx_1_3 = (long long)(bos_n + t_c_1_3) * (long long)num_heads + (long long)head_4;
                    bt_n[1] = (float)beta[bidx_1_3];
                }
                mbarrier_wait(wy_full_addr, _phase_wy_full_0_1);
                _phase_wy_full_0_1 ^= 1;
                int t_r_5_4 = cvt_tid;
                if (t_r_5_4 >= clen_j) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 ^ (t_r_5_4 * 128 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 16 ^ (t_r_5_4 * 128 + 16 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 32 ^ (t_r_5_4 * 128 + 32 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 48 ^ (t_r_5_4 * 128 + 48 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 64 ^ (t_r_5_4 * 128 + 64 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 80 ^ (t_r_5_4 * 128 + 80 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 96 ^ (t_r_5_4 * 128 + 96 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_4 * 128 + 112 ^ (t_r_5_4 * 128 + 112 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                }
                int t_r_5_5 = cvt_tid + 32;
                if (t_r_5_5 >= clen_j) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 ^ (t_r_5_5 * 128 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 16 ^ (t_r_5_5 * 128 + 16 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 32 ^ (t_r_5_5 * 128 + 32 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 48 ^ (t_r_5_5 * 128 + 48 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 64 ^ (t_r_5_5 * 128 + 64 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 80 ^ (t_r_5_5 * 128 + 80 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 96 ^ (t_r_5_5 * 128 + 96 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((akk_op_addr + (unsigned int)(t_r_5_5 * 128 + 112 ^ (t_r_5_5 * 128 + 112 >> 7 & 7) << 4))), "r"(zeros4[0]), "r"(zeros4[1]), "r"(zeros4[2]), "r"(zeros4[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(akk_ready_addr);
                int t_r_5_6 = cvt_tid;
                int valid_1_0 = 0;
                if (t_r_5_6 < clen_j) {
                    valid_1_0 = 1;
                }
                float bt_1_0 = 0.0f;
                if (valid_1_0 == 1) {
                    bt_1_0 = bt_j[0];
                }
                unsigned int mask_1_0 = (unsigned int)(-valid_1_0);
                unsigned int addr_1_0 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 ^ (t_r_5_6 * 128 >> 7 & 7) << 4));
                unsigned int raw_1_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_0[(0) + 3]))
                    : "r"(addr_1_0));
                unsigned int sel_1_0[4];
                #pragma unroll
                for (int i_1_0 = 0; i_1_0 < 4; i_1_0++) {
                    sel_1_0[i_1_0] = raw_1_0[i_1_0] & mask_1_0;
                }
                float _unpack_bf16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_0[_pair]) << 16);
                    (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_0[_pair]) & 0xffff0000u);
                }
                float prod_1_0[8];
                #pragma unroll
                for (int e_1_0 = 0; e_1_0 < 8; e_1_0++) {
                    prod_1_0[e_1_0] = _unpack_bf16_0[e_1_0] * bt_1_0;
                }
                unsigned int packed_1_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_0[_lp*2 + 0], prod_1_0[_lp*2+1 + 0]));
                    packed_1_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_0), "r"(packed_1_0[0]), "r"(packed_1_0[1]), "r"(packed_1_0[2]), "r"(packed_1_0[3]) : "memory");
                unsigned int addr_1_1 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 16 ^ (t_r_5_6 * 128 + 16 >> 7 & 7) << 4));
                unsigned int raw_1_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_1[(0) + 3]))
                    : "r"(addr_1_1));
                unsigned int sel_1_1[4];
                #pragma unroll
                for (int i_1_1 = 0; i_1_1 < 4; i_1_1++) {
                    sel_1_1[i_1_1] = raw_1_1[i_1_1] & mask_1_0;
                }
                float _unpack_bf16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_1[_pair]) << 16);
                    (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_1[_pair]) & 0xffff0000u);
                }
                float prod_1_1[8];
                #pragma unroll
                for (int e_1_1 = 0; e_1_1 < 8; e_1_1++) {
                    prod_1_1[e_1_1] = _unpack_bf16_1[e_1_1] * bt_1_0;
                }
                unsigned int packed_1_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_1[_lp*2 + 0], prod_1_1[_lp*2+1 + 0]));
                    packed_1_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_1), "r"(packed_1_1[0]), "r"(packed_1_1[1]), "r"(packed_1_1[2]), "r"(packed_1_1[3]) : "memory");
                unsigned int addr_1_2 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 32 ^ (t_r_5_6 * 128 + 32 >> 7 & 7) << 4));
                unsigned int raw_1_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_2[(0) + 3]))
                    : "r"(addr_1_2));
                unsigned int sel_1_2[4];
                #pragma unroll
                for (int i_1_2 = 0; i_1_2 < 4; i_1_2++) {
                    sel_1_2[i_1_2] = raw_1_2[i_1_2] & mask_1_0;
                }
                float _unpack_bf16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_2[_pair]) << 16);
                    (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_2[_pair]) & 0xffff0000u);
                }
                float prod_1_2[8];
                #pragma unroll
                for (int e_1_2 = 0; e_1_2 < 8; e_1_2++) {
                    prod_1_2[e_1_2] = _unpack_bf16_2[e_1_2] * bt_1_0;
                }
                unsigned int packed_1_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_2[_lp*2 + 0], prod_1_2[_lp*2+1 + 0]));
                    packed_1_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_2), "r"(packed_1_2[0]), "r"(packed_1_2[1]), "r"(packed_1_2[2]), "r"(packed_1_2[3]) : "memory");
                unsigned int addr_1_3 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 48 ^ (t_r_5_6 * 128 + 48 >> 7 & 7) << 4));
                unsigned int raw_1_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_3[(0) + 3]))
                    : "r"(addr_1_3));
                unsigned int sel_1_3[4];
                #pragma unroll
                for (int i_1_3 = 0; i_1_3 < 4; i_1_3++) {
                    sel_1_3[i_1_3] = raw_1_3[i_1_3] & mask_1_0;
                }
                float _unpack_bf16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_3[_pair]) << 16);
                    (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_3[_pair]) & 0xffff0000u);
                }
                float prod_1_3[8];
                #pragma unroll
                for (int e_1_3 = 0; e_1_3 < 8; e_1_3++) {
                    prod_1_3[e_1_3] = _unpack_bf16_3[e_1_3] * bt_1_0;
                }
                unsigned int packed_1_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_3[_lp*2 + 0], prod_1_3[_lp*2+1 + 0]));
                    packed_1_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_3), "r"(packed_1_3[0]), "r"(packed_1_3[1]), "r"(packed_1_3[2]), "r"(packed_1_3[3]) : "memory");
                unsigned int addr_1_4 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 64 ^ (t_r_5_6 * 128 + 64 >> 7 & 7) << 4));
                unsigned int raw_1_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_4[(0) + 3]))
                    : "r"(addr_1_4));
                unsigned int sel_1_4[4];
                #pragma unroll
                for (int i_1_4 = 0; i_1_4 < 4; i_1_4++) {
                    sel_1_4[i_1_4] = raw_1_4[i_1_4] & mask_1_0;
                }
                float _unpack_bf16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_4[_pair]) << 16);
                    (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_4[_pair]) & 0xffff0000u);
                }
                float prod_1_4[8];
                #pragma unroll
                for (int e_1_4 = 0; e_1_4 < 8; e_1_4++) {
                    prod_1_4[e_1_4] = _unpack_bf16_4[e_1_4] * bt_1_0;
                }
                unsigned int packed_1_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_4[_lp*2 + 0], prod_1_4[_lp*2+1 + 0]));
                    packed_1_4[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_4), "r"(packed_1_4[0]), "r"(packed_1_4[1]), "r"(packed_1_4[2]), "r"(packed_1_4[3]) : "memory");
                unsigned int addr_1_5 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 80 ^ (t_r_5_6 * 128 + 80 >> 7 & 7) << 4));
                unsigned int raw_1_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_5[(0) + 3]))
                    : "r"(addr_1_5));
                unsigned int sel_1_5[4];
                #pragma unroll
                for (int i_1_5 = 0; i_1_5 < 4; i_1_5++) {
                    sel_1_5[i_1_5] = raw_1_5[i_1_5] & mask_1_0;
                }
                float _unpack_bf16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_5[_pair]) << 16);
                    (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_5[_pair]) & 0xffff0000u);
                }
                float prod_1_5[8];
                #pragma unroll
                for (int e_1_5 = 0; e_1_5 < 8; e_1_5++) {
                    prod_1_5[e_1_5] = _unpack_bf16_5[e_1_5] * bt_1_0;
                }
                unsigned int packed_1_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_5[_lp*2 + 0], prod_1_5[_lp*2+1 + 0]));
                    packed_1_5[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_5), "r"(packed_1_5[0]), "r"(packed_1_5[1]), "r"(packed_1_5[2]), "r"(packed_1_5[3]) : "memory");
                unsigned int addr_1_6 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 96 ^ (t_r_5_6 * 128 + 96 >> 7 & 7) << 4));
                unsigned int raw_1_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_6[(0) + 3]))
                    : "r"(addr_1_6));
                unsigned int sel_1_6[4];
                #pragma unroll
                for (int i_1_6 = 0; i_1_6 < 4; i_1_6++) {
                    sel_1_6[i_1_6] = raw_1_6[i_1_6] & mask_1_0;
                }
                float _unpack_bf16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_6[_pair]) << 16);
                    (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_6[_pair]) & 0xffff0000u);
                }
                float prod_1_6[8];
                #pragma unroll
                for (int e_1_6 = 0; e_1_6 < 8; e_1_6++) {
                    prod_1_6[e_1_6] = _unpack_bf16_6[e_1_6] * bt_1_0;
                }
                unsigned int packed_1_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_6[_lp*2 + 0], prod_1_6[_lp*2+1 + 0]));
                    packed_1_6[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_6), "r"(packed_1_6[0]), "r"(packed_1_6[1]), "r"(packed_1_6[2]), "r"(packed_1_6[3]) : "memory");
                unsigned int addr_1_7 = (vb_raw_addr + (unsigned int)(t_r_5_6 * 128 + 112 ^ (t_r_5_6 * 128 + 112 >> 7 & 7) << 4));
                unsigned int raw_1_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_7[(0) + 3]))
                    : "r"(addr_1_7));
                unsigned int sel_1_7[4];
                #pragma unroll
                for (int i_1_7 = 0; i_1_7 < 4; i_1_7++) {
                    sel_1_7[i_1_7] = raw_1_7[i_1_7] & mask_1_0;
                }
                float _unpack_bf16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_7[_pair]) << 16);
                    (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_7[_pair]) & 0xffff0000u);
                }
                float prod_1_7[8];
                #pragma unroll
                for (int e_1_7 = 0; e_1_7 < 8; e_1_7++) {
                    prod_1_7[e_1_7] = _unpack_bf16_7[e_1_7] * bt_1_0;
                }
                unsigned int packed_1_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_7[_lp*2 + 0], prod_1_7[_lp*2+1 + 0]));
                    packed_1_7[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_7), "r"(packed_1_7[0]), "r"(packed_1_7[1]), "r"(packed_1_7[2]), "r"(packed_1_7[3]) : "memory");
                int t_r_5_7 = cvt_tid + 32;
                int valid_1_1 = 0;
                if (t_r_5_7 < clen_j) {
                    valid_1_1 = 1;
                }
                float bt_1_1 = 0.0f;
                if (valid_1_1 == 1) {
                    bt_1_1 = bt_j[1];
                }
                unsigned int mask_1_1 = (unsigned int)(-valid_1_1);
                unsigned int addr_1_8 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 ^ (t_r_5_7 * 128 >> 7 & 7) << 4));
                unsigned int raw_1_8[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_8[(0) + 3]))
                    : "r"(addr_1_8));
                unsigned int sel_1_8[4];
                #pragma unroll
                for (int i_1_8 = 0; i_1_8 < 4; i_1_8++) {
                    sel_1_8[i_1_8] = raw_1_8[i_1_8] & mask_1_1;
                }
                float _unpack_bf16_8[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_8[_pair]) << 16);
                    (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_8[_pair]) & 0xffff0000u);
                }
                float prod_1_8[8];
                #pragma unroll
                for (int e_1_8 = 0; e_1_8 < 8; e_1_8++) {
                    prod_1_8[e_1_8] = _unpack_bf16_8[e_1_8] * bt_1_1;
                }
                unsigned int packed_1_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_8[_lp*2 + 0], prod_1_8[_lp*2+1 + 0]));
                    packed_1_8[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_8), "r"(packed_1_8[0]), "r"(packed_1_8[1]), "r"(packed_1_8[2]), "r"(packed_1_8[3]) : "memory");
                unsigned int addr_1_9 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 16 ^ (t_r_5_7 * 128 + 16 >> 7 & 7) << 4));
                unsigned int raw_1_9[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_9[(0) + 3]))
                    : "r"(addr_1_9));
                unsigned int sel_1_9[4];
                #pragma unroll
                for (int i_1_9 = 0; i_1_9 < 4; i_1_9++) {
                    sel_1_9[i_1_9] = raw_1_9[i_1_9] & mask_1_1;
                }
                float _unpack_bf16_9[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_9[_pair]) << 16);
                    (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_9[_pair]) & 0xffff0000u);
                }
                float prod_1_9[8];
                #pragma unroll
                for (int e_1_9 = 0; e_1_9 < 8; e_1_9++) {
                    prod_1_9[e_1_9] = _unpack_bf16_9[e_1_9] * bt_1_1;
                }
                unsigned int packed_1_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_9[_lp*2 + 0], prod_1_9[_lp*2+1 + 0]));
                    packed_1_9[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_9), "r"(packed_1_9[0]), "r"(packed_1_9[1]), "r"(packed_1_9[2]), "r"(packed_1_9[3]) : "memory");
                unsigned int addr_1_10 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 32 ^ (t_r_5_7 * 128 + 32 >> 7 & 7) << 4));
                unsigned int raw_1_10[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_10[(0) + 3]))
                    : "r"(addr_1_10));
                unsigned int sel_1_10[4];
                #pragma unroll
                for (int i_1_10 = 0; i_1_10 < 4; i_1_10++) {
                    sel_1_10[i_1_10] = raw_1_10[i_1_10] & mask_1_1;
                }
                float _unpack_bf16_10[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_10[_pair]) << 16);
                    (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_10[_pair]) & 0xffff0000u);
                }
                float prod_1_10[8];
                #pragma unroll
                for (int e_1_10 = 0; e_1_10 < 8; e_1_10++) {
                    prod_1_10[e_1_10] = _unpack_bf16_10[e_1_10] * bt_1_1;
                }
                unsigned int packed_1_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_10[_lp*2 + 0], prod_1_10[_lp*2+1 + 0]));
                    packed_1_10[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_10), "r"(packed_1_10[0]), "r"(packed_1_10[1]), "r"(packed_1_10[2]), "r"(packed_1_10[3]) : "memory");
                unsigned int addr_1_11 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 48 ^ (t_r_5_7 * 128 + 48 >> 7 & 7) << 4));
                unsigned int raw_1_11[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_11[(0) + 3]))
                    : "r"(addr_1_11));
                unsigned int sel_1_11[4];
                #pragma unroll
                for (int i_1_11 = 0; i_1_11 < 4; i_1_11++) {
                    sel_1_11[i_1_11] = raw_1_11[i_1_11] & mask_1_1;
                }
                float _unpack_bf16_11[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_11[_pair]) << 16);
                    (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_11[_pair]) & 0xffff0000u);
                }
                float prod_1_11[8];
                #pragma unroll
                for (int e_1_11 = 0; e_1_11 < 8; e_1_11++) {
                    prod_1_11[e_1_11] = _unpack_bf16_11[e_1_11] * bt_1_1;
                }
                unsigned int packed_1_11[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_11[_lp*2 + 0], prod_1_11[_lp*2+1 + 0]));
                    packed_1_11[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_11), "r"(packed_1_11[0]), "r"(packed_1_11[1]), "r"(packed_1_11[2]), "r"(packed_1_11[3]) : "memory");
                unsigned int addr_1_12 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 64 ^ (t_r_5_7 * 128 + 64 >> 7 & 7) << 4));
                unsigned int raw_1_12[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_12[(0) + 3]))
                    : "r"(addr_1_12));
                unsigned int sel_1_12[4];
                #pragma unroll
                for (int i_1_12 = 0; i_1_12 < 4; i_1_12++) {
                    sel_1_12[i_1_12] = raw_1_12[i_1_12] & mask_1_1;
                }
                float _unpack_bf16_12[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_12[_pair]) << 16);
                    (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_12[_pair]) & 0xffff0000u);
                }
                float prod_1_12[8];
                #pragma unroll
                for (int e_1_12 = 0; e_1_12 < 8; e_1_12++) {
                    prod_1_12[e_1_12] = _unpack_bf16_12[e_1_12] * bt_1_1;
                }
                unsigned int packed_1_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_12[_lp*2 + 0], prod_1_12[_lp*2+1 + 0]));
                    packed_1_12[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_12), "r"(packed_1_12[0]), "r"(packed_1_12[1]), "r"(packed_1_12[2]), "r"(packed_1_12[3]) : "memory");
                unsigned int addr_1_13 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 80 ^ (t_r_5_7 * 128 + 80 >> 7 & 7) << 4));
                unsigned int raw_1_13[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_13[(0) + 3]))
                    : "r"(addr_1_13));
                unsigned int sel_1_13[4];
                #pragma unroll
                for (int i_1_13 = 0; i_1_13 < 4; i_1_13++) {
                    sel_1_13[i_1_13] = raw_1_13[i_1_13] & mask_1_1;
                }
                float _unpack_bf16_13[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_13[_pair]) << 16);
                    (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_13[_pair]) & 0xffff0000u);
                }
                float prod_1_13[8];
                #pragma unroll
                for (int e_1_13 = 0; e_1_13 < 8; e_1_13++) {
                    prod_1_13[e_1_13] = _unpack_bf16_13[e_1_13] * bt_1_1;
                }
                unsigned int packed_1_13[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_13[_lp*2 + 0], prod_1_13[_lp*2+1 + 0]));
                    packed_1_13[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_13), "r"(packed_1_13[0]), "r"(packed_1_13[1]), "r"(packed_1_13[2]), "r"(packed_1_13[3]) : "memory");
                unsigned int addr_1_14 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 96 ^ (t_r_5_7 * 128 + 96 >> 7 & 7) << 4));
                unsigned int raw_1_14[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_14[(0) + 3]))
                    : "r"(addr_1_14));
                unsigned int sel_1_14[4];
                #pragma unroll
                for (int i_1_14 = 0; i_1_14 < 4; i_1_14++) {
                    sel_1_14[i_1_14] = raw_1_14[i_1_14] & mask_1_1;
                }
                float _unpack_bf16_14[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_14[_pair]) << 16);
                    (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_14[_pair]) & 0xffff0000u);
                }
                float prod_1_14[8];
                #pragma unroll
                for (int e_1_14 = 0; e_1_14 < 8; e_1_14++) {
                    prod_1_14[e_1_14] = _unpack_bf16_14[e_1_14] * bt_1_1;
                }
                unsigned int packed_1_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_14[_lp*2 + 0], prod_1_14[_lp*2+1 + 0]));
                    packed_1_14[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_14), "r"(packed_1_14[0]), "r"(packed_1_14[1]), "r"(packed_1_14[2]), "r"(packed_1_14[3]) : "memory");
                unsigned int addr_1_15 = (vb_raw_addr + (unsigned int)(t_r_5_7 * 128 + 112 ^ (t_r_5_7 * 128 + 112 >> 7 & 7) << 4));
                unsigned int raw_1_15[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1_15[(0) + 3]))
                    : "r"(addr_1_15));
                unsigned int sel_1_15[4];
                #pragma unroll
                for (int i_1_15 = 0; i_1_15 < 4; i_1_15++) {
                    sel_1_15[i_1_15] = raw_1_15[i_1_15] & mask_1_1;
                }
                float _unpack_bf16_15[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(sel_1_15[_pair]) << 16);
                    (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(sel_1_15[_pair]) & 0xffff0000u);
                }
                float prod_1_15[8];
                #pragma unroll
                for (int e_1_15 = 0; e_1_15 < 8; e_1_15++) {
                    prod_1_15[e_1_15] = _unpack_bf16_15[e_1_15] * bt_1_1;
                }
                unsigned int packed_1_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(prod_1_15[_lp*2 + 0], prod_1_15[_lp*2+1 + 0]));
                    packed_1_15[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_1_15), "r"(packed_1_15[0]), "r"(packed_1_15[1]), "r"(packed_1_15[2]), "r"(packed_1_15[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vb_ready_addr);
                __syncwarp();
                if (warp == 10) {
                    if (elect_sync()) {
                        if (nc_4 > j + 1) {
                            mbarrier_wait(wy_empty_addr, _phase_wy_empty_0);
                            _phase_wy_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(wy_full_addr, 32768);
                            int trow_4 = row0_4 + (j + 1) * 64;
                            tma_3d_gmem2smem(akk_op_addr, (&akk_tma), 0, head_4, bos_a, wy_full_addr);
                            #pragma unroll
                            for (int seg_4 = 0; seg_4 < 2; seg_4++) {
                                tma_3d_gmem2smem(kb_panel_addr + (unsigned int)seg_4 * 8192, (&kb_tma), seg_4 * 64, head_4, trow_4, wy_full_addr);
                            }
                            tma_3d_gmem2smem(vb_raw_addr, (&v_tma), vbase_1, head_4, bos_a, wy_full_addr);
                            if (nc_4 > j + 1 + 1) {
                                int trow2_1 = row0_4 + (j + 1 + 1) * 64;
                                #pragma unroll
                                for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&kb_tma))), "r"((int)(seg_5 * 64)), "r"((int)(head_4)), "r"((int)(trow2_1)) : "memory");
                                }
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&akk_tma))), "r"((int)(0)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(vbase_1)), "r"((int)(head_4)), "r"((int)(bos_b)) : "memory");
                            }
                            bos_a = bos_b;
                            if (nc_4 > j + 1 + 2) {
                                bos_b = chunk_bos[c0_4 + (j + 1) + 2];
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
